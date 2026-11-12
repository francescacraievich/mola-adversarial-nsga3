"""Per-window and per-candidate records.

Pure or read-only functions, testable without ROS: distance from the baseline
at window start and end, added deviation and signed lateral offset, per-metre
and end-of-run damage (--danno-metro); scan counts of a window from the MOLA log
with MOLA_LO_DEBUG_ICP_QUALITY=1 (read with src/analysis/scarti_mola.leggi_printf);
first twin ICP and anomalous-twin flags; DDS transport state from the
environment; random genome sampling; chain-fault checks and the classification
of candidates that do not complete the metre.
"""

import math
import os
import re
import time

import numpy as np

from src.analysis.baseline_distance import dist_to_polyline
from src.analysis.scarti_mola import _stamp, _stamp_da_data, leggi_printf

PERIODO_S = 0.1            # LiDAR at 10 Hz
TOL_STAMP_S = 0.05         # half period: printf stamps are on 0.1 s steps

# am2: twin whose first ICP comes within 0.25 s of the previous twin's last
# scan. al: anomalous first ICP (MaxIterations or q < 0.9) within the twin's
# first 0.2 s.
AM2_GAP_S = 0.25
AL_Q_MIN = 0.9
AL_SPAN_S = 0.2
TWIN_FILTERS = ("none", "am2", "al", "am2_al")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def signed_lateral(points, poly):
    """Signed lateral offset (m) of each point from the polyline.

    The nearest segment is chosen (same clipped projection as
    dist_to_polyline); the value is the signed distance from that segment's
    line, positive to the left of the polyline direction. Past the ends the
    end segment's line applies (extended polyline), so overshooting the target
    does not enter the lateral offset.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64)).reshape(-1, 2)
    poly = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    if pts.shape[0] == 0:
        return np.zeros(0)
    if poly.shape[0] < 2:
        raise ValueError("servono almeno due vertici per la laterale")
    a = poly[:-1]
    d = poly[1:] - a
    len2 = np.einsum("ij,ij->i", d, d)
    if not np.any(len2 > 0):
        raise ValueError("polilinea senza segmenti di lunghezza positiva")
    ap = pts[:, None, :] - a[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(len2 > 0, np.einsum("nmj,mj->nm", ap, d) / len2, 0.0)
    closest = a[None, :, :] + np.clip(t, 0.0, 1.0)[:, :, None] * d[None, :, :]
    dist = np.linalg.norm(pts[:, None, :] - closest, axis=2)
    dist = np.where(len2[None, :] > 0, dist, np.inf)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(pts))
    dj = d[j]
    cross = dj[:, 0] * ap[rows, j, 1] - dj[:, 1] * ap[rows, j, 0]
    return cross / np.sqrt(len2[j])


def baseline_window_fields(path_xy, poly):
    """Distance from the baseline along a window's path (cm).

    path_xy: true poses of the path in order (Nx2, odom frame), first row =
    window start. Fields: start/end/mean of the unsigned distance (as
    baseline_damage), added deviation added_end = end - start and
    added_mean = mean - start, and the same five values for the signed lateral
    offset (lat_*). None without finite poses or without the polyline.
    """
    if poly is None:
        return None
    pts = np.atleast_2d(np.asarray(path_xy, dtype=np.float64)).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return None
    dist = dist_to_polyline(pts, poly) * 100.0
    out = {"n_points": int(len(pts))}
    out.update(_tre(dist, ""))
    try:
        out.update(_tre(signed_lateral(pts, poly) * 100.0, "lat_"))
    except ValueError:
        pass
    return out


def _tre(v, pre):
    s, e, m = float(v[0]), float(v[-1]), float(np.mean(v))
    return {f"{pre}start_cm": round(s, 3), f"{pre}end_cm": round(e, 3),
            f"{pre}mean_cm": round(m, 3),
            f"{pre}added_end_cm": round(e - s, 3), f"{pre}added_mean_cm": round(m - s, 3)}


def _closest_on_polyline(pt, poly):
    """Point of the polyline nearest to pt (same clipped projection as
    dist_to_polyline; with one vertex, the vertex)."""
    pt = np.asarray(pt, dtype=np.float64)
    poly = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    if poly.shape[0] == 1:
        return poly[0].copy()
    a = poly[:-1]
    d = poly[1:] - a
    len2 = np.einsum("ij,ij->i", d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(len2 > 0, np.einsum("mj,mj->m", pt[None, :] - a, d) / len2, 0.0)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * d
    return closest[int(np.argmin(np.linalg.norm(closest - pt, axis=1)))]


# Per-metre damage definitions (--danno-metro), copied into the history.
# Unsigned distances (cm) from the nearest baseline point, over the distinct
# true poses of the path (true_path).
DANNO_METRO_DEFINITIONS = {
    "mean_cm": "(i) distanza media dalla baseline lungo il metro; per i candidati e' la fitness "
               "(--damage mean con --damage-ref baseline). Comprende la deviazione accumulata nei "
               "metri precedenti",
    "start_cm": "distanza dalla baseline della prima posa del tratto (deviazione d'ingresso, ereditata)",
    "end_cm": "distanza dalla baseline dell'ultima posa del tratto",
    "end_minus_start_cm": "(ii) end_cm - start_cm: variazione dello scostamento agli estremi del "
                          "metro, non la massima deviazione prodotta al suo interno",
    "max_cm": "(iii) escursione massima: distanza massima dalla baseline nel metro",
    "max_minus_start_cm": "max_cm - start_cm",
    "added_cm": "(iv) deviazione aggiuntiva del metro: mean_cm - start_cm. La fitness si scompone "
                "in start_cm (ereditato, uguale per i candidati che partono dallo stesso stato) piu' "
                "added_cm (effetto del genoma corrente); negativa se il genoma riavvicina il robot "
                "alla baseline. Non toglie l'errore di prua ereditato",
    "added_shift_cm": "(iv, variante) distanza media dalla baseline traslata del vettore di "
                      "deviazione d'ingresso (prima posa meno il suo punto piu' vicino della "
                      "baseline): 0 per un tratto che conserva lo scostamento d'ingresso, sempre >= 0",
}


def danno_metro(path_xy, poly):
    """Damage values of one metre w.r.t. the baseline (cm), as defined in
    DANNO_METRO_DEFINITIONS. path_xy: true poses of the path in order (Nx2,
    odom frame), first row = metre start. None without finite poses or without
    the polyline."""
    if poly is None:
        return None
    pts = np.atleast_2d(np.asarray(path_xy, dtype=np.float64)).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return None
    poly = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    dist = dist_to_polyline(pts, poly) * 100.0
    v = pts[0] - _closest_on_polyline(pts[0], poly)
    shift = dist_to_polyline(pts, poly + v) * 100.0
    s, e, m, mx = float(dist[0]), float(dist[-1]), float(np.mean(dist)), float(np.max(dist))

    def r(x):
        return round(x, 3)
    return {"n_points": int(len(pts)), "mean_cm": r(m), "start_cm": r(s), "end_cm": r(e),
            "end_minus_start_cm": r(e - s), "max_cm": r(mx), "max_minus_start_cm": r(mx - s),
            "added_cm": r(m - s), "added_shift_cm": r(float(np.mean(shift)))}


# End-of-run summary definitions (--danno-metro).
DANNO_RUN_DEFINITIONS = {
    "victim_end_cm": "(v) danno totale a fine run: distanza dalla baseline dell'ultima posa vera della "
                     "vittima nell'ultimo metro applicato con danno (end_cm di quel metro)",
    "victim_mean_cm": "distanza media dalla baseline su tutte le pose della vittima nei metri con danno "
                      "(media dei mean_cm pesata con n_points)",
    "victim_max_cm": "massimo dei max_cm della vittima",
    "victim_added_sum_cm": "somma degli added_cm della vittima, metro per metro",
    "winner_mean_cm": "media dei mean_cm dei vincitori nella ricerca (danno cercato, gemello)",
    "n_windows": "metri con danno della vittima (esclusi applicazioni fallite e attacco scoperto)",
}


def danno_run(history, victim_key="danno_metro_vittima", winner_key="danno_metro_vincitore"):
    """End-of-run summary from the per-metre history entries
    (DANNO_RUN_DEFINITIONS). None if no window has the victim entry."""
    vic = [(h.get("window"), h[victim_key]) for h in history
           if isinstance(h, dict) and isinstance(h.get(victim_key), dict)]
    win = [h[winner_key]["mean_cm"] for h in history
           if isinstance(h, dict) and isinstance(h.get(winner_key), dict)]
    if not vic:
        return None
    n = np.array([d["n_points"] for _, d in vic], dtype=float)
    m = np.array([d["mean_cm"] for _, d in vic], dtype=float)

    def r(x):
        return round(float(x), 3)
    return {"victim_end_cm": r(vic[-1][1]["end_cm"]), "victim_end_window": vic[-1][0],
            "victim_mean_cm": r(np.sum(n * m) / np.sum(n)),
            "victim_max_cm": r(max(d["max_cm"] for _, d in vic)),
            "victim_added_sum_cm": r(sum(d["added_cm"] for _, d in vic)),
            "winner_mean_cm": r(np.mean(win)) if win else None,
            "n_windows": len(vic), "n_windows_winner": len(win),
            "definitions": DANNO_RUN_DEFINITIONS}


def winner_index(eval_log, genome, damage, chosen_idx=None):
    """Index in eval_log of the winning candidate: chosen_idx if given
    (pareto_maxdamage), otherwise the first with the same genome and search
    damage (a genome can appear more than once with warm start). None if not
    found."""
    if chosen_idx is not None and 0 <= chosen_idx < len(eval_log):
        return int(chosen_idx)
    if genome is None or damage is None:
        return None
    g = np.asarray(genome, dtype=float)
    for i, e in enumerate(eval_log):
        eg, ed = e.get("genome"), e.get("damage")
        if eg is None or ed is None or len(eg) != len(g):
            continue
        if np.array_equal(np.asarray(eg, dtype=float), g) and math.isclose(float(ed), float(damage),
                                                                           rel_tol=1e-9, abs_tol=1e-12):
            return i
    return None


# ---------------------------------------------------------------------------
# Scans from the MOLA log
# ---------------------------------------------------------------------------

def read_printf(log_path):
    """ICP printf lines from the MOLA log; empty list if the file is missing."""
    if not log_path:
        return []
    try:
        return leggi_printf(str(log_path))
    except OSError:
        return []


# Drops without ICP visible in the log (LidarOdometry_ProcessScan.cpp, MOLA 3.2.0):
# invalid observation (WARN with stamp, :697-701) and scan closer than
# min_time_between_scans to the previous one (DEBUG without stamp, :597-602; not
# written at INFO verbosity).
RX_INVALID = re.compile(r"Observation discarded as non-valid for pathStep=\d+, timestamp=(\S+) UTC")
RX_MIN_TIME = re.compile(r"dropping observation, for [-+.\deE]+< `min_time_between_scans`")

# Count definitions, copied into the history next to the values.
SCAN_DEFINITIONS = {
    "interval": "[t0, t1] = stamp sull'orologio della MOLA vittima, arrotondati a 0,1 s. "
                "scans_drive: t0 = stamp dell'ultima posa pubblicata prima del tratto (se "
                "l'ultima ICP prima della partenza e' stata rifiutata e' piu' vecchio e "
                "comprende slot dell'assestamento), t1 = ultima posa del tratto. "
                "scans_open: t0 = prima posa dopo la riapertura del flusso",
    "expected": "slot a 10 Hz fra t0 e t1, estremi inclusi: comprende lo scan di t0, "
                "gia' elaborato prima del tratto",
    "processed": "righe printf dell'ICP con stamp distinti nell'intervallo",
    "rejected": "processed con isGood=0 (ICP rifiutata, posa non pubblicata)",
    "lost": "max(0, expected - processed): slot senza ICP nel log, cioe' scan non arrivati, "
            "scartati per min_time_between_scans, osservazioni non valide o scan prima "
            "della localizzazione iniziale",
    "excess": "max(0, processed - expected): righe oltre gli slot, che lost non mostra",
    "discarded_invalid": "righe WARN 'Observation discarded as non-valid' con stamp "
                         "nell'intervallo (parte di lost)",
    "dropped_min_time": "righe DEBUG 'dropping observation ... min_time_between_scans' fra la "
                        "prima e l'ultima riga printf dell'intervallo, per posizione nel file "
                        "(la riga non ha stamp; parte di lost). None se il log non ne contiene "
                        "nessuna: a verbosita' INFO non vengono scritte",
    "lost_unexplained": "lost - discarded_invalid - dropped_min_time (None contato 0)",
}


def read_discards(log_path):
    """Drops without ICP from the MOLA log: stamps of invalid observations and
    line numbers of min_time_between_scans drops."""
    out = {"invalid_stamps": [], "min_time_lines": []}
    if not log_path:
        return out
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                m = RX_INVALID.search(line)
                if m:
                    try:
                        out["invalid_stamps"].append(_stamp(_stamp_da_data(m.group(1))))
                    except ValueError:
                        pass
                elif RX_MIN_TIME.search(line):
                    out["min_time_lines"].append(i)
    except OSError:
        pass
    return out


def scan_counts(records, t0, t1, period=PERIODO_S, discards=None):
    """Scans of a stamp interval [t0, t1] (s, log clock).

    Field definitions in SCAN_DEFINITIONS. q_median, q_min: goodness of the
    processed scans. log_complete: the log has lines up to t1 (MOLA writes stdout
    in blocks, so while it runs the last lines may not be in the file yet). With
    discards (output of read_discards) also excess, discarded_invalid,
    dropped_min_time and lost_unexplained. None if the interval is undefined.
    """
    if t0 is None or t1 is None or not (math.isfinite(t0) and math.isfinite(t1)) or t1 < t0:
        return None
    a, b = round(t0, 1), round(t1, 1)
    attese = int(round((b - a) / period)) + 1
    per_stamp = {}
    for r in records:
        if a - TOL_STAMP_S <= r["stamp_s"] <= b + TOL_STAMP_S:
            per_stamp[r["stamp_s"]] = r
    sel = list(per_stamp.values())
    q = [r["goodness"] for r in sel]
    last = max((r["stamp_s"] for r in records), default=None)
    out = {"t0": a, "t1": b, "expected": attese, "processed": len(sel),
           "rejected": sum(1 for r in sel if r["is_good"] == 0),
           "lost": max(0, attese - len(sel)),
           "n_maxiter": sum(1 for r in sel if r["term_reason"] == "MaxIterations"),
           "q_median": round(float(np.median(q)), 4) if q else None,
           "q_min": round(float(min(q)), 4) if q else None,
           "log_last_stamp": last,
           "log_complete": bool(last is not None and last >= b - TOL_STAMP_S)}
    if discards is not None:
        inv = sum(1 for st in discards.get("invalid_stamps", [])
                  if a - TOL_STAMP_S <= st <= b + TOL_STAMP_S)
        lines = discards.get("min_time_lines", [])
        drop = None
        if lines:
            rows = [r["riga"] for r in sel if r.get("riga") is not None]
            drop = sum(1 for ln in lines if rows and min(rows) <= ln <= max(rows))
        out.update({"excess": max(0, len(sel) - attese), "discarded_invalid": inv,
                    "dropped_min_time": drop,
                    "lost_unexplained": out["lost"] - inv - (drop or 0)})
    return out


# ---------------------------------------------------------------------------
# Anomalous twins
# ---------------------------------------------------------------------------

def twin_icp_flags(records, prev_last_stamp, gap_max=AM2_GAP_S, q_min=AL_Q_MIN, span=AL_SPAN_S):
    """First ICP of a twin and the am2 / al flags.

    records: printf lines of the twin log; prev_last_stamp: stamp of the last
    printf line of the previous twin (None if unknown). twin_am2: first ICP
    within gap_max s of the previous twin's last scan; twin_al: at least one
    ICP with MaxIterations or q < q_min in the first span s. None when it cannot
    be determined (no printf lines, previous unknown).
    """
    out = {"twin_icp_first_stamp": None, "twin_icp_last_stamp": None, "twin_icp_gap_s": None,
           "twin_icp_first_q": None, "twin_icp_first_iters": None, "twin_icp_first_term": None,
           "twin_am2": None, "twin_al": None}
    if not records:
        return out
    first = records[0]
    out.update(twin_icp_first_stamp=first["stamp_s"], twin_icp_last_stamp=records[-1]["stamp_s"],
               twin_icp_first_q=first["goodness"], twin_icp_first_iters=first["iters"],
               twin_icp_first_term=first["term_reason"])
    primi = [r for r in records if r["stamp_s"] <= first["stamp_s"] + span + 1e-6]
    out["twin_al"] = any(r["goodness"] < q_min or r["term_reason"] == "MaxIterations" for r in primi)
    if prev_last_stamp is not None:
        gap = round(first["stamp_s"] - prev_last_stamp, 3)
        out["twin_icp_gap_s"] = gap
        out["twin_am2"] = gap < gap_max
    return out


def twin_filtered(mode, flags):
    """True if filter `mode` excludes the twin; an unknown flag does not exclude."""
    if mode not in TWIN_FILTERS:
        raise ValueError(f"filtro sconosciuto: {mode}")
    if mode == "none" or not flags:
        return False
    am2 = flags.get("twin_am2") is True
    al = flags.get("twin_al") is True
    if mode == "am2":
        return am2
    if mode == "al":
        return al
    return am2 or al


# First ICP of the metre (victim when the stream reopens, twin after the state
# load): q <= threshold marks the metre, without excluding it. Threshold from
# the estimate-divergence diagnosis: a persistent offset appeared only with
# q <= 0.34.
PRIMA_ICP_Q_DEBOLE = 0.34


def prima_icp(records, stamp, soglia=PRIMA_ICP_Q_DEBOLE):
    """printf line of the first ICP at the given stamp (within half a period)
    and the prima_icp_debole flag; fields None if the stamp or line is missing."""
    out = {"stamp": None, "q": None, "iters": None, "term": None,
           "prima_icp_debole": None, "soglia_q": soglia}
    if stamp is None or not records:
        return out
    sel = [r for r in records if abs(r["stamp_s"] - stamp) <= TOL_STAMP_S]
    if not sel:
        return out
    r = sel[0]
    out.update(stamp=r["stamp_s"], q=r["goodness"], iters=r["iters"], term=r["term_reason"],
               prima_icp_debole=bool(r["goodness"] <= soglia))
    return out


def prima_icp_gemello(ev, soglia=PRIMA_ICP_Q_DEBOLE):
    """prima_icp for a twin, from the twin_icp_first_* fields of its evaluation."""
    ev = ev or {}
    q = ev.get("twin_icp_first_q")
    return {"stamp": ev.get("twin_icp_first_stamp"), "q": q, "iters": ev.get("twin_icp_first_iters"),
            "term": ev.get("twin_icp_first_term"),
            "prima_icp_debole": None if q is None else bool(q <= soglia), "soglia_q": soglia}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

TRANSPORT_ENV = ("RMW_IMPLEMENTATION", "ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY",
                 "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_STATIC_PEERS",
                 "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE",
                 "FASTDDS_BUILTIN_TRANSPORTS", "CYCLONEDDS_URI",
                 "MOLA_LO_DEBUG_ICP_QUALITY", "MOLA_LIDAR_TOPIC_TYPE",
                 "MOLA_VERBOSITY_MOLA_LO", "SERIEB_TRANSPORT")


def lidar_qos_from_env(value):
    """QoS of the LiDAR entry injected through MOLA_LIDAR_TOPIC_TYPE (YAML text
    with newlines after the type, see scripts/run_campaign.sh
    TRANSPORT=reliable). None if the variable is missing or has no qos: block
    (bridge QoS, best effort). Text only: the negotiated QoS is not measured."""
    if not value or "qos:" not in value:
        return None
    out = {}
    for line in value.split("qos:", 1)[1].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out or None


def transport_state(env=None):
    """Environment variables that set the DDS transport (None if absent).

    QoS are not in the environment and are not measured here: only what can be
    read is logged, including the injected LiDAR QoS (lidar_qos_injected).
    MolaRunner.spawn removes FASTRTPS_DEFAULT_PROFILES_FILE from the MOLA
    environment (mola_env_removed).
    """
    env = os.environ if env is None else env
    d = {k: env.get(k) for k in TRANSPORT_ENV}
    d["lidar_qos_injected"] = lidar_qos_from_env(env.get("MOLA_LIDAR_TOPIC_TYPE"))
    d["mola_env_removed"] = ["FASTRTPS_DEFAULT_PROFILES_FILE"]
    return d


# ---------------------------------------------------------------------------
# Random genomes
# ---------------------------------------------------------------------------

SAMPLING_MODES = ("legacy", "uniform", "sparse")


def sample_genomes(rng, n, n_search, mode, off=None):
    """n random genomes in the search space [-1, 1]^n_search.

    uniform: every gene uniform within bounds (as pymoo FloatRandomSampling, the
    NSGA-III initial population without warm start). sparse: a random number of
    active genes (1..n_search) with uniform values, the others at the off value
    `off` (-1 intensities, 0 directions); same draw sequence as --random-sparse.
    """
    if mode == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n, n_search))
    if mode == "sparse":
        off = np.full(n_search, -1.0) if off is None else np.asarray(off, dtype=float)
        X = np.tile(off, (n, 1))
        for i in range(n):
            kk = int(rng.integers(1, n_search + 1))
            idx = rng.choice(n_search, size=kk, replace=False)
            X[i, idx] = rng.uniform(-1.0, 1.0, size=kk)
        return X
    raise ValueError(f"campionamento sconosciuto: {mode}")


def first_admissible(draw, evaluate, is_valid, k_max):
    """Single random search with a budget: draws and evaluates one genome at a
    time, stops at the first for which is_valid() is true, at most k_max tries.
    Returns (genomes, objectives, found); if found, the valid one is last."""
    xs, fs = [], []
    for _ in range(int(k_max)):
        x = draw()
        xs.append(x)
        fs.append(evaluate(x))
        if is_valid():
            return xs, fs, True
    return xs, fs, False


def first_valid(records, is_valid):
    """Index of the first valid record in evaluation order, None if none."""
    for i, r in enumerate(records):
        if is_valid(r):
            return i
    return None


# ---------------------------------------------------------------------------
# NSGA-III warm start
# ---------------------------------------------------------------------------

def warm_start_counts(front_rows, mutated_rows, front_genomes):
    """Composition of the warm half of the initial population.

    front_rows: rows copied from the previous window's front; mutated_rows:
    rows obtained by perturbing front members; front_genomes: front genomes
    (search space). A perturbation can equal a front member when the noise hits
    genes already at a bound and is clipped: counted separately.
    """
    fg = [np.asarray(g, dtype=float) for g in front_genomes]
    same = sum(1 for r in mutated_rows
               if any(np.array_equal(np.asarray(r, dtype=float), g) for g in fg))
    return {"warm_start_from_prev": len(front_rows) + len(mutated_rows),
            "warm_front_n": len(front_rows), "warm_mut_n": len(mutated_rows),
            "warm_mut_identical_n": same}


# ---------------------------------------------------------------------------
# Origin of the victim and attacker frames
# ---------------------------------------------------------------------------

def compose_pose(base, local):
    """Pose `local` (in the frame with origin `base`) in the base frame (same
    formula as AttackNode.compose)."""
    bx, by, byaw = base
    lx, ly, lyaw = local
    c, s_ = math.cos(byaw), math.sin(byaw)
    return (bx + c * lx - s_ * ly, by + s_ * lx + c * ly, byaw + lyaw)


def pose_inverse(p):
    """Inverse of a planar pose (x, y, yaw)."""
    x, y, yaw = p
    c, s_ = math.cos(yaw), math.sin(yaw)
    return (-c * x - s_ * y, s_ * x - c * y, -yaw)


def origin_from_estimate(true_pose, est_raw):
    """Origin of the MOLA frame in the working frame: true pose composed with
    the inverse of the estimate at the same instant, so that
    compose(origin, estimate) = true pose."""
    return compose_pose(true_pose, pose_inverse(est_raw))


def origin_record(mode, true_pose, est_raw, origin):
    """Origin record at read time: offset (cm) of compose(origin, estimate) from
    the true pose with the origin in use (offset) and with the other rule's
    origin (offset_other_mode; legacy: origin = true pose)."""
    def off(o):
        if o is None or est_raw is None or true_pose is None:
            return None
        w = compose_pose(o, est_raw)
        return {"dx_cm": round(100 * (w[0] - true_pose[0]), 3),
                "dy_cm": round(100 * (w[1] - true_pose[1]), 3),
                "d_cm": round(100 * math.hypot(w[0] - true_pose[0], w[1] - true_pose[1]), 3)}
    alt = None
    if true_pose is not None and est_raw is not None:
        alt = origin_from_estimate(true_pose, est_raw) if mode == "true" else tuple(true_pose)
    return {"mode": mode,
            "true_pose": list(map(float, true_pose)) if true_pose is not None else None,
            "est_raw": list(map(float, est_raw)) if est_raw is not None else None,
            "origin": list(map(float, origin)) if origin is not None else None,
            "offset": off(origin),
            "offset_other_mode": off(alt)}


# ---------------------------------------------------------------------------
# Victim pose loss
# ---------------------------------------------------------------------------

def mark_victim_lost(rec, status, window, state):
    """Mark the window if the victim rollout ended without a pose
    (victim_no_pose, victim_crash): victim_lost true and the window where the
    loss began (first of a consecutive series). state: dict shared across
    windows ("since", "any"). status None (no victim) changes nothing."""
    if status is None:
        return rec
    if str(status).startswith("victim_"):
        if state.get("since") is None:
            state["since"] = window
        state["any"] = True
        rec["victim_lost"] = True
        rec["victim_lost_since_window"] = state["since"]
        rec["victim_lost_status"] = str(status)
    else:
        state["since"] = None
        rec.setdefault("victim_lost", False)
    return rec


# ---------------------------------------------------------------------------
# "MOLA notices" and "attack detected"
# ---------------------------------------------------------------------------

ACCORGE_ICP_RIFIUTATE = 10     # consecutive rejected ICPs: logged only
ACCORGE_SENZA_POSA_S = 1.0
# Victim rollout outcomes without a pose: victim_no_pose (no pose within 15 s of
# the stream reopening), victim_crash, pose_stale (no new pose for more than 3 s
# while driving); no_pose and mola_crash are the same outcomes seen from the
# victim drive (no estimate within 15 s, MOLA dead).
STATI_VITTIMA_PERSA = ("victim_no_pose", "victim_crash", "pose_stale", "no_pose", "mola_crash")


def _per_stamp(records):
    """printf lines with distinct stamps in stamp order (the last one wins on
    ties, as in scan_counts)."""
    per = {}
    for r in records or []:
        per[r["stamp_s"]] = r
    return [per[s] for s in sorted(per)]


def mola_accorge(records, n_rifiutate=ACCORGE_ICP_RIFIUTATE, senza_posa_s=ACCORGE_SENZA_POSA_S,
                 period=PERIODO_S, crash=False):
    """"MOLA notices": the candidate's twin MOLA crashes or publishes no new pose
    for more than senza_posa_s. Uses the printf lines of the whole twin log,
    also before the first pose.

    senza_posa_max_s: longest stamp interval between two good ICPs; MOLA
    publishes a pose only with isGood=1 (LidarOdometry_ProcessScan.cpp:1592-1596).
    The ends use the first stamp minus one period and the last plus one period,
    so k consecutive rejections give k+1 periods also at head or tail. Slots
    without a line (lost scans) are included; senza_righe_max_s, the longest
    gap between consecutive lines, tells them apart. rifiutate_max: longest
    run of consecutive isGood=0, logged only (rifiutate_oltre_soglia:
    rifiutate_max >= n_rifiutate), never excluding. crash: twin MOLA died
    during the rollout (mola_crash). accorge: crash or
    senza_posa_max_s > senza_posa_s; None without crash and without printf lines.
    """
    rows = _per_stamp(records)
    out = {"n_icp": len(rows), "n_rifiutate": sum(1 for r in rows if r["is_good"] == 0),
           "rifiutate_max": None, "rifiutate_oltre_soglia": None,
           "senza_posa_max_s": None, "senza_righe_max_s": None,
           "soglia_rifiutate": int(n_rifiutate), "soglia_senza_posa_s": float(senza_posa_s),
           "crash": bool(crash),
           "stamp_primo": rows[0]["stamp_s"] if rows else None,
           "stamp_ultimo": rows[-1]["stamp_s"] if rows else None,
           "accorge": None, "motivo": None}
    motivi = ["MOLA del gemello terminata (mola_crash)"] if crash else []
    if rows:
        serie = best = 0
        for r in rows:
            serie = serie + 1 if r["is_good"] == 0 else 0
            best = max(best, serie)
        pose = ([rows[0]["stamp_s"] - period] + [r["stamp_s"] for r in rows if r["is_good"] == 1]
                + [rows[-1]["stamp_s"] + period])
        gap = max(round(b - a, 3) for a, b in zip(pose, pose[1:]))
        st = [r["stamp_s"] for r in rows]
        out["rifiutate_max"] = best
        out["rifiutate_oltre_soglia"] = best >= n_rifiutate
        out["senza_posa_max_s"] = gap
        out["senza_righe_max_s"] = max((round(b - a, 3) for a, b in zip(st, st[1:])), default=0.0)
        # stamps on 0.1 s steps: tolerance for representation error
        if gap > senza_posa_s + 1e-6:
            motivi.append(f"{gap:.1f} s senza posa (soglia {senza_posa_s:.1f} s)")
    elif not crash:
        return out
    out["accorge"] = bool(motivi)
    out["motivo"] = "; ".join(motivi) or None
    return out


def vittima_persa(status):
    """True if the victim rollout outcome is pose-less (STATI_VITTIMA_PERSA)."""
    return status is not None and str(status).split("(")[0].strip() in STATI_VITTIMA_PERSA


# Distinct markers for a victim without a pose.
ESITI_VITTIMA_PERSA = {
    "crash_processo": "crash di processo (mola-cli della vittima terminato: MOLA o il bridge ROS 2, stesso processo)",
    "rifiuto_da_attacco": "rifiuto da attacco (processo vivo, ICP rifiutate dopo l'ultima buona)",
    "senza_posa_da_confermare": "senza posa, processo vivo, nessuna ICP rifiutata nelle righe lette in linea "
                                "(a processo vivo lo stdout arriva a blocchi: lo decide la fine della run)",
    "senza_posa_senza_rifiuto": "senza posa, processo vivo, nessuna ICP rifiutata nel log completo",
    "processo_non_trovato": "processo della vittima non trovato (non avviato o gia' chiuso): non determinato",
}


def segnali_rifiuto(records):
    """Signs of attack-induced rejection in the victim printf lines: ICPs rejected
    (isGood=0: goodness below MOLA's threshold, pose not published) after the
    last good ICP, their minimum goodness, rejections over the whole log.
    rifiuto: at least one rejection after the last good ICP; None without lines.
    While MOLA runs its stdout arrives in blocks, so live reads may miss the
    last lines."""
    rows = _per_stamp(records)
    buone = [r["stamp_s"] for r in rows if r["is_good"] == 1]
    ultima = buone[-1] if buone else None
    dopo = [r for r in rows if r["is_good"] == 0 and (ultima is None or r["stamp_s"] > ultima)]
    return {"n_icp": len(rows), "n_rifiutate": sum(1 for r in rows if r["is_good"] == 0),
            "stamp_ultima_buona": ultima, "rifiutate_dopo_ultima_buona": len(dopo),
            "goodness_min_dopo_ultima_buona": round(min(r["goodness"] for r in dopo), 4) if dopo else None,
            "soglia": rows[-1].get("min_required") if rows else None,
            "rifiuto": bool(dopo) if rows else None}


def esito_vittima_persa(processo, segnali, log_completo=False):
    """Marker (key of ESITI_VITTIMA_PERSA) for a victim without a pose, from the
    process state (orchestrator processo_mola_vittima: alive, terminated or
    non_trovato) and the rejection signs (segnali_rifiuto). log_completo: lines
    read after MOLA exited (end of run), so no rejection means none to confirm."""
    processo = processo or {}
    if processo.get("esito") == "non_trovato":
        return "processo_non_trovato"
    if processo.get("vivo") is False:
        return "crash_processo"
    if (segnali or {}).get("rifiuto"):
        return "rifiuto_da_attacco"
    return "senza_posa_senza_rifiuto" if log_completo else "senza_posa_da_confermare"


def attacco_scoperto_record(window, status, records=None, last_pose=None, true_pose=None,
                            sim_time_s=None, wall=None, driven=None, held=None):
    """run_end entry of a run closed as "attack detected": the victim lost its
    pose in window `window` (outcome `status`).

    records: printf lines of the victim log (the last may be missing while MOLA
    runs). Instant: last stamp with a good ICP (last pose) and first rejected
    stamp after it, on the victim MOLA clock; sim time from the last
    /chassis/odom; wall time. last_pose: last pose published by the victim (its
    MOLA frame); true_pose: last true pose received (odom frame).
    """
    rows = _per_stamp(records)
    buone = [r["stamp_s"] for r in rows if r["is_good"] == 1]
    ultima = buone[-1] if buone else None
    dopo = [r["stamp_s"] for r in rows if r["is_good"] == 0 and (ultima is None or r["stamp_s"] > ultima)]

    def fl(v):
        return [float(x) for x in v] if v is not None else None

    rec = {"reason": "attacco_scoperto", "window": int(window), "status": str(status),
           "stamp_ultima_posa": ultima,
           "stamp_prima_icp_rifiutata": dopo[0] if dopo else None,
           "stamp_ultima_riga_log": rows[-1]["stamp_s"] if rows else None,
           "sim_time_s": sim_time_s, "wall": wall,
           "wall_iso": (time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(wall))
                        if wall is not None else None),
           "ultima_posa": fl(last_pose), "posa_vera": fl(true_pose),
           "separato_dai_successi": True}
    if driven is not None:
        rec["windows_driven"] = int(driven)
    if held is not None:
        rec["windows_held"] = int(held)
    return rec


# ---------------------------------------------------------------------------
# Chain fault checked before declaring "attack detected"
# ---------------------------------------------------------------------------

# Thresholds shared with the other checks:
CATENA_BUCO_S = ACCORGE_SENZA_POSA_S   # gap in the clouds delivered to the victim (1 s)
T_ON_P99_MS = 100.0                    # p99 of time_onLidar (ms)
CATENA_T_ON_MS = 2 * T_ON_P99_MS       # victim MOLA times: fault above 2x the threshold
ADATTATORE_P99_MS = 100.0              # adapter check clean if p99 <= 100 ms (same value as
                                       # the node latency p99)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def leggi_csv(path):
    """CSV rows as dicts; None if the file is missing or unreadable."""
    import csv
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None


def _vero(v):
    return str(v).strip() in ("1", "True", "true")


def buchi_consegna_vittima(rows, t0_wall, t1_wall=None, soglia_s=CATENA_BUCO_S, never_clean=False,
                           period=PERIODO_S):
    """Gaps in the clouds delivered to the victim, from the node per-frame log
    (--frame-log), for arrivals between t0_wall and t1_wall.

    Expected rows: victim stream open (victim_stream), cloud not dropped by the
    post-restore filter (fresh_gate) and, with --victim-never-clean, attack on
    (clean clouds do not go to the victim). Delivered: stamp_victim and
    t_pub_victim_end_mono present; the node status is ignored (a 10 Hz timer).
    In each run of consecutive expected rows the gap is the stamp difference
    between consecutive deliveries; the ends use the first expected arrival
    minus one period and the last plus one period (as in mola_accorge), so k
    consecutive undelivered clouds count as k+1 periods also at head, tail or
    in a run without deliveries. Stamps are the input ones (stamp_in), on one
    clock within a run; buco_da_stamp is on the victim clock (stamp_victim of
    the last delivery before the gap, None for a head gap). An Isaac pause does
    not advance the stamps and does not count.
    """
    out = {"soglia_s": soglia_s, "n_righe": 0, "n_aperte": 0, "n_attese": 0, "n_consegnate": 0,
           "buco_max_s": None, "buco_da_stamp": None, "buco_in_coda_s": None,
           "esito": "non_determinato", "motivo": None}
    if rows is None:
        out["motivo"] = "registro per frame assente"
        return out
    sel = [r for r in rows if (_num(r.get("t_arr_wall")) is not None and _num(r["t_arr_wall"]) >= t0_wall
                               and (t1_wall is None or _num(r["t_arr_wall"]) <= t1_wall))]
    out["n_righe"] = len(sel)
    tratti, cur = [], None
    for r in sel:
        aperta = _vero(r.get("victim_stream"))
        out["n_aperte"] += int(aperta)
        attesa = (aperta and str(r.get("fresh_gate") or "").strip() != "scartata_ripristino"
                  and (not never_clean or _vero(r.get("victim_attack"))))
        if not attesa or _num(r.get("stamp_in")) is None:
            cur = None             # stream closed or cloud not meant for the victim: the segment ends
            continue
        if cur is None:
            cur = []
            tratti.append(cur)
        cons = _num(r.get("stamp_victim")) is not None and _num(r.get("t_pub_victim_end_mono")) is not None
        cur.append((_num(r["stamp_in"]), cons, _num(r.get("stamp_victim"))))
    if not tratti:
        out["motivo"] = "nessuna nuvola attesa dalla vittima nella finestra"
        return out
    best, da = None, None
    for t in tratti:
        out["n_attese"] += len(t)
        out["n_consegnate"] += sum(1 for _, c, _ in t if c)
        punti = ([(t[0][0] - period, None)] + [(s, sv) for s, c, sv in t if c]
                 + [(t[-1][0] + period, None)])
        for (a, va), (b, _) in zip(punti, punti[1:]):
            g = round(b - a, 3)
            if best is None or g > best:
                best, da = g, va
        out["buco_in_coda_s"] = round(punti[-1][0] - punti[-2][0], 3)   # last segment
    out["buco_max_s"], out["buco_da_stamp"] = best, da
    if best > soglia_s + 1e-6:
        out["esito"] = "guasto"
        dove = f"dallo stamp {da}" if da is not None else "nessuna consegna prima, nel tratto"
        out["motivo"] = f"buco di {best:.2f} s nelle nuvole consegnate alla vittima ({dove})"
    else:
        out["esito"] = "ok"
    return out


def ultimo_stamp_tracce(rows):
    """Highest stamp in the MOLA CSV traces (None if missing)."""
    st = [_num(r.get("timestamp")) for r in rows or []]
    st = [s for s in st if s is not None]
    return max(st) if st else None


def tempi_mola_vittima(rows, stamp_da=None, soglia_ms=CATENA_T_ON_MS):
    """Victim time_onLidar p99 (ms) over trace rows with stamp after stamp_da
    (the window); 'nearest' p99 as in src/analysis/scarti_mola.py. The initial
    row (zero iterations and quality) is excluded. Traces are written in
    blocks: while MOLA runs the last rows are missing."""
    out = {"soglia_ms": soglia_ms, "n": 0, "p99_ms": None, "max_ms": None,
           "esito": "non_determinato", "motivo": None}
    if rows is None:
        out["motivo"] = "tracce della vittima assenti"
        return out
    v = []
    for r in rows:
        ts, ton = _num(r.get("timestamp")), _num(r.get("time_onLidar"))
        if ts is None or ton is None or (stamp_da is not None and ts <= stamp_da):
            continue
        if _num(r.get("icp_iterations")) == 0 and _num(r.get("icp_quality")) == 0:
            continue
        v.append(1000.0 * ton)
    if not v:
        out["motivo"] = "nessuna riga della finestra nelle tracce"
        return out
    v.sort()
    out.update(n=len(v), p99_ms=round(v[int(round(0.99 * (len(v) - 1)))], 2), max_ms=round(v[-1], 2))
    if out["p99_ms"] > soglia_ms:
        out["esito"] = "guasto"
        out["motivo"] = f"time_onLidar p99 {out['p99_ms']:.0f} ms oltre {soglia_ms:.0f} ms (n {len(v)})"
    else:
        out["esito"] = "ok"
    return out


# Victim MOLA computation time treated as an attack outcome.
RX_OSS_RAW = re.compile(r"Generator \(raw\)\] Processing observation .* stamp=([\d.]+)")
RIPARTENZA_ICP = "ICP hook dt="
# tf2 warning written to the same file, possibly in the middle of a MOLA line
RX_WARN_TF = re.compile(r"\[WARN\] \[[\d.]+\] \[\]: TF_OLD_DATA[^\n]*\nPossible reasons are listed at "
                        r"http://wiki\.ros\.org/tf/Errors%20explained\n")
TEMPI_MOLA_CONVENZIONE = ("c = 1000 * time_onLidar della riga successiva delle tracce (durata di onLidar "
                          "della scansione della riga), percentili lineari; il p99 di "
                          "controllo_catena.mola_vittima usa la riga stessa e il percentile nearest")


def tempi_c(rows):
    """(stamp, c) per scan from the MOLA CSV traces: c = 1000 * time_onLidar of
    the next row (a scan's row is written inside its onLidar and reports the
    previous duration), only between consecutive rows with all fields numeric.
    Stamp rounded to 0.1 s."""
    val = []
    for r in rows or []:
        d = {k: _num(v) for k, v in r.items() if k}
        val.append(d if d and None not in d.values() else None)
    return [(round(a["timestamp"], 1), 1000.0 * b["time_onLidar"]) for a, b in zip(val[:-1], val[1:])
            if a is not None and b is not None and "timestamp" in a and "time_onLidar" in b]


def ripartenze_icp(log_path):
    """ICP restarts for the twist per scan (stamp at 0.1 s -> count) from a
    DEBUG-level MOLA log: "ICP hook dt=" lines between one "Generator (raw)]
    Processing observation" line and the next; a repeated stamp counts once.
    TF_OLD_DATA warnings, which split MOLA lines, are removed first. None if the
    log is missing or has no DEBUG observation lines."""
    if not log_path or not os.path.exists(log_path):
        return None
    out, cur = {}, None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            testo = RX_WARN_TF.sub("", f.read())
    except OSError:
        return None
    for line in testo.split("\n"):
        m = RX_OSS_RAW.search(line)
        if m:
            s = round(float(m.group(1)), 1)
            cur = None if s in out else s
            if cur is not None:
                out[s] = 0
        elif cur is not None and RIPARTENZA_ICP in line:
            out[cur] += 1
    return out or None


def confini_metri(history):
    """Metre boundaries in victim stamps: metre k runs from the scan after the
    stream reopens up to and including the next window's reopening, i.e.
    stamps in (open_stamp_k, open_stamp_k+1]. The first metre also holds the
    start-up scans, the last one runs to the end. List of {"finestra", "da_s",
    "a_s", "fine_guida_s"} (None = open) in order; da_s None after the first
    metre or a_s None before the last means the boundary is unavailable.
    fine_guida_s: last stamp of the driven part (drive_stamps[1]); the last
    metre's value marks the start of shutdown."""
    con_vw = [h for h in history if isinstance(h, dict) and h.get("window") is not None
              and isinstance(h.get("victim_window"), dict)]
    aperture = [h["victim_window"].get("open_stamp") for h in con_vw]
    out = []
    for i, h in enumerate(con_vw):
        out.append({"finestra": h["window"],
                    "da_s": None if i == 0 else aperture[i],
                    "a_s": aperture[i + 1] if i + 1 < len(con_vw) else None,
                    "fine_guida_s": (h["victim_window"].get("drive_stamps") or [None, None])[-1]})
    return out


def tempi_mola_finestra(rows, t0=None, t1=None, ripartenze=None):
    """p50, p99 and max of c (tempi_c, ms) over scans with stamp in [t0, t1]
    (None = open end), numpy linear percentiles; ICP restarts in the same
    interval from the ripartenze_icp map (None: log without DEBUG lines), over
    all log scans and over those with c (ripartenze_su_c). completo: traces (and
    log) reach past t1, None without t1; while MOLA runs the last lines are
    missing."""
    out = {"t0": t0, "t1": t1, "n": 0, "p50_ms": None, "p99_ms": None, "max_ms": None,
           "ripartenze": None, "scansioni_con_ripartenze": None, "ripartenze_max": None,
           "n_scansioni_debug": None, "ripartenze_su_c": None, "completo": False,
           "convenzione": TEMPI_MOLA_CONVENZIONE}
    lo = -math.inf if t0 is None else t0 - TOL_STAMP_S
    hi = math.inf if t1 is None else t1 + TOL_STAMP_S
    pc = tempi_c(rows)
    sc = [(s, c) for s, c in pc if lo <= s <= hi]
    v = sorted(c for _, c in sc)
    if v:
        out.update(n=len(v), p50_ms=round(float(np.percentile(v, 50)), 2),
                   p99_ms=round(float(np.percentile(v, 99)), 2), max_ms=round(v[-1], 2))
    fino = [max((s for s, _ in pc), default=None)]
    if ripartenze is not None:
        h = [n for s, n in ripartenze.items() if lo <= s <= hi]
        con_c = {s for s, _ in sc}
        out.update(ripartenze=sum(h), scansioni_con_ripartenze=sum(1 for n in h if n > 0),
                   ripartenze_max=max(h, default=0), n_scansioni_debug=len(h),
                   ripartenze_su_c=sum(n for s, n in ripartenze.items() if s in con_c))
        fino.append(max(ripartenze))
    out["completo"] = None if t1 is None else all(x is not None and x >= t1 - TOL_STAMP_S for x in fino)
    return out


def eventi_adattatore(events, t0_wall, t1_wall, soglia_ms=ADATTATORE_P99_MS):
    """Adapter restarts and unclean checks in the window, from the
    --adapter-events file: a restart whose interval [t_prev_clean, t_restart]
    touches the window (the same interval that triggers candidate
    re-evaluation, reevaluate_after_adapter_restart), or a check with p99 above
    threshold."""
    out = {"soglia_ms": soglia_ms, "riavvii": [], "controlli_non_puliti": [],
           "n_controlli": 0, "sorveglianza_attiva": False, "ultimo_evento_wall": None,
           "esito": "non_determinato", "motivo": None}
    if events is None:
        out["motivo"] = "sorveglianza dell'adattatore assente: file degli eventi assente"
        return out
    tempi = [t for ev in events for t in (_num(ev.get("t_check")), _num(ev.get("t_restart"))) if t is not None]
    out["ultimo_evento_wall"] = max(tempi) if tempi else None
    for ev in events:
        tr = _num(ev.get("t_restart"))
        if tr is not None:
            lo = _num(ev.get("t_prev_clean"))
            lo = lo if lo is not None else tr
            if lo <= t1_wall and tr >= t0_wall:
                out["riavvii"].append(ev)
            continue
        tc, p99 = _num(ev.get("t_check")), _num(ev.get("p99"))
        if tc is not None and t0_wall <= tc <= t1_wall:
            out["n_controlli"] += 1
            if p99 is not None and p99 > soglia_ms:
                out["controlli_non_puliti"].append(ev)
    motivi = []
    if out["riavvii"]:
        motivi.append(f"{len(out['riavvii'])} riavvii dell'adattatore nella finestra")
    if out["controlli_non_puliti"]:
        motivi.append(f"{len(out['controlli_non_puliti'])} controlli con p99 oltre {soglia_ms:.0f} ms")
    # Monitoring active = at least one check in the window. Without checks the
    # outcome stays undetermined and the record says so.
    out["sorveglianza_attiva"] = out["n_controlli"] > 0 or bool(out["riavvii"])
    if motivi:
        out["esito"], out["motivo"] = "guasto", "; ".join(motivi)
    elif out["n_controlli"]:
        out["esito"] = "ok"
    else:
        ult = out["ultimo_evento_wall"]
        ult = (time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ult)) if ult is not None else "nessuno")
        out["motivo"] = ("sorveglianza dell'adattatore assente: nessun controllo nella finestra "
                         f"(ultimo evento nel file: {ult})")
    return out


def soglia_icp_buona(environ=None, config=None):
    """min_icp_goodness of the MOLA pipeline (config/mola, ${VAR|default}
    resolved from the environment): (value, source), or (None, reason)."""
    from src.analysis.scarti_mola import soglia_min_icp_goodness
    config = config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "config", "mola", "lidar3d-gicp-optimize-twist-local.yaml")
    try:
        return soglia_min_icp_goodness(config, dict(os.environ if environ is None else environ))
    except (OSError, ValueError) as e:
        return None, str(e)


def pose_non_ricevute(records, stamp_ricevuta, tracce=None, q_min=None, soglia_s=CATENA_BUCO_S,
                      period=PERIODO_S):
    """Poses the victim MOLA published but the orchestrator did not receive
    (DDS between MOLA and orchestrator).

    MOLA publishes a pose only with a good ICP (isGood=1,
    LidarOdometry_ProcessScan.cpp:1594-1596). Source: victim printf lines
    (records); without printf, CSV traces with icp_quality >= q_min
    (min_icp_goodness), initial row excluded. stamp_ricevuta: stamp of the last
    pose received from the victim, same clock (None if none). Fault if the good
    ICPs after that stamp span more than soglia_s of sensor time (same threshold
    as the gaps). printf and traces are written in blocks, so while MOLA runs
    the last lines may be missing and the check stays conservative.
    """
    out = {"soglia_s": soglia_s, "fonte": None, "stamp_ultima_ricevuta": stamp_ricevuta,
           "n_buone": 0, "n_buone_dopo": 0, "stamp_ultima_buona": None, "non_ricevute_s": None,
           "q_min": q_min, "esito": "non_determinato", "motivo": None}
    rows = _per_stamp(records)
    if rows:
        out["fonte"] = "printf"
        buone = [r["stamp_s"] for r in rows if r["is_good"] == 1]
    elif tracce and q_min is not None:
        out["fonte"] = "tracce"
        buone = sorted(s for s, q, it in ((_num(r.get("timestamp")), _num(r.get("icp_quality")),
                                           _num(r.get("icp_iterations"))) for r in tracce)
                       if s is not None and q is not None and not (it == 0 and q == 0) and q >= q_min)
    else:
        out["motivo"] = ("ne' righe printf ne' tracce della vittima" if not tracce
                         else "tracce senza soglia di qualita' ICP")
        return out
    out["n_buone"] = len(buone)
    dopo = [s for s in buone if stamp_ricevuta is None or s > stamp_ricevuta + TOL_STAMP_S]
    out["n_buone_dopo"] = len(dopo)
    if not dopo:
        out.update(esito="ok", non_ricevute_s=0.0, stamp_ultima_buona=buone[-1] if buone else None)
        return out
    out["stamp_ultima_buona"] = dopo[-1]
    da = stamp_ricevuta if stamp_ricevuta is not None else dopo[0] - period
    out["non_ricevute_s"] = round(dopo[-1] - da, 3)
    if out["non_ricevute_s"] > soglia_s + 1e-6:
        out["esito"] = "guasto"
        out["motivo"] = (f"{len(dopo)} pose con ICP buona ({out['fonte']}) fino allo stamp {dopo[-1]} "
                         f"non ricevute dall'orchestratore (ultima ricevuta: {stamp_ricevuta})")
    else:
        out["esito"] = "ok"
    return out


RALLENTAMENTO_MOLA = ("guasto", "scoperto", "risultato")
SANI_PER_RALLENTAMENTO = ("isaac", "nuvole_vittima", "perturbation")


def guasto_catena(controlli, rallentamento="guasto"):
    """Overall outcome of the chain checks (name -> dict with outcome ok, guasto
    or non_determinato). An undetermined check is not enough to declare a
    fault: it stays in the record.

    rallentamento_mola: the only failing check is mola_vittima (time_onLidar
    p99 above threshold) and Isaac, victim clouds and node are ok (not merely
    not failing). With rallentamento "guasto" the run remains a chain fault;
    with "scoperto" it does not, and the attack-detected rule decides
    (--rallentamento-mola). With "risultato" mola_vittima is never a fault,
    whatever the other checks say: it stays among the signals and in the
    record."""
    if rallentamento not in RALLENTAMENTO_MOLA:
        raise ValueError(f"rallentamento sconosciuto: {rallentamento}")
    guasti = [n for n, c in controlli.items() if c.get("esito") == "guasto"]
    motivi = [f"{n}: {controlli[n].get('motivo')}" for n in guasti]
    rall = (guasti == ["mola_vittima"]
            and all(controlli.get(n, {}).get("esito") == "ok" for n in SANI_PER_RALLENTAMENTO))
    segnali = list(motivi)
    if rall and rallentamento == "scoperto":
        motivi = []
    elif rallentamento == "risultato":
        motivi = [m for n, m in zip(guasti, motivi) if n != "mola_vittima"]
    return {"guasto": bool(motivi), "motivi": motivi, "segnali": segnali,
            "rallentamento_mola": rall, "regola_rallentamento": rallentamento,
            "non_determinati": [n for n, c in controlli.items() if c.get("esito") == "non_determinato"],
            "sorveglianza_adattatore": (None if "adattatore" not in controlli else
                                        "attiva" if controlli["adattatore"].get("sorveglianza_attiva")
                                        else "assente"),
            "controlli": controlli}


def guasto_catena_record(window, status, catena, t0_wall=None, sim_time_s=None, wall=None,
                         seed=None, driven=None, held=None):
    """run_end entry of a run closed for a chain fault instead of "attack
    detected": the victim lost its pose in window `window` but a chain check
    failed. The run must be redone with the same seed and is kept out of the
    results."""
    rec = {"reason": "guasto_catena", "window": int(window), "status": str(status),
           "motivi": list(catena.get("motivi", [])), "controlli": catena.get("controlli"),
           "non_determinati": catena.get("non_determinati", []),
           "rallentamento_mola": bool(catena.get("rallentamento_mola")),
           "regola_rallentamento": catena.get("regola_rallentamento"),
           "sorveglianza_adattatore": catena.get("sorveglianza_adattatore"),
           "finestra_da_wall": t0_wall, "sim_time_s": sim_time_s, "wall": wall,
           "wall_iso": (time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(wall))
                        if wall is not None else None),
           "da_rifare": True, "seme": seed, "separato_dai_successi": True}
    if driven is not None:
        rec["windows_driven"] = int(driven)
    if held is not None:
        rec["windows_held"] = int(held)
    return rec



# ---------------------------------------------------------------------------
# Candidates that do not complete the metre (--incompleti-regola)
# ---------------------------------------------------------------------------

STATI_INCOMPLETI = ("timeout", "stopped_early")
CLASSI_INCOMPLETO = {
    "guasto_catena": "(a) segnale della catena indipendente dal genoma (stallo di Isaac, eventi "
                     "dell'adattatore, buco >= 1 s nelle nuvole in ingresso al nodo): candidato "
                     "rifatto una volta; il blocco [4] e' un controllo di fine run (rit4 = run rifatta)",
    "mola_perde_posa": "(b) MOLA del gemello crasha o resta senza posa oltre la soglia (regola 8): "
                       "non ammissibile",
    "calcolo": "calcolo del nodo oltre il periodo di scansione (calc_ok falso): non ammissibile, "
               "non rifatto",
    "attacco_valido": "(c) catena sana, MOLA localizza e pubblica: valido, danno = distanza della "
                      "posa vera a fine rollout dalla posa della baseline a fine finestra",
}


def stato_incompleto(status):
    """True if the rollout outcome is timeout or stopped_early (also with a suffix)."""
    return status is not None and str(status).split("(")[0].strip() in STATI_INCOMPLETI


def stalli_isaac_rollout(events, t0_wall, t1_wall, eta_posa_s=None, stall_sec=None):
    """Isaac stalls in the rollout [t0_wall, t1_wall] (STALL_POLICY events).

    A stall with outcome "retry" already caused the rollout to be redone: it
    counts only if it falls after the start of the last attempt (t_wall +
    recovered_after_s of the last retry). eta_posa_s: age of the Isaac loop pose
    at rollout end; above stall_sec it is a fault."""
    ev = [e for e in events or [] if _num(e.get("t_wall")) is not None
          and t0_wall <= _num(e["t_wall"]) <= t1_wall]
    inizio = t0_wall
    for e in ev:
        if e.get("outcome") == "retry":
            inizio = max(inizio, _num(e["t_wall"]) + (_num(e.get("recovered_after_s")) or 0.0))
    dopo = [e for e in ev if _num(e["t_wall"]) >= inizio]
    out = {"stalli": ev, "stalli_ultimo_tentativo": dopo, "rollout_rifatto_per_stallo": inizio > t0_wall,
           "eta_posa_s": None if eta_posa_s is None else round(eta_posa_s, 2),
           "esito": "ok", "motivo": None}
    mot = []
    if dopo:
        mot.append(f"{len(dopo)} stalli di Isaac nell'ultimo tentativo del rollout")
    if eta_posa_s is not None and stall_sec is not None and eta_posa_s > stall_sec:
        mot.append(f"posa del ciclo di Isaac ferma da {eta_posa_s:.1f} s (soglia {stall_sec:.0f} s)")
    if mot:
        out["esito"], out["motivo"] = "guasto", "; ".join(mot)
    elif eta_posa_s is None:
        out["esito"], out["motivo"] = "non_determinato", "eta' della posa di Isaac non leggibile"
    return out


def buchi_arrivi_nodo(rows, t0_wall, t1_wall, soglia_s=CATENA_BUCO_S):
    """Gaps in the clouds entering perturbation_node (per-frame log,
    --frame-log) for arrivals between t0_wall and t1_wall: stamp_in difference
    between consecutive arrivals. The input stamp comes from the adapter and does
    not depend on the genome. A backward jump (Isaac restore) does not count; an
    Isaac pause does not advance the stamps and does not count."""
    out = {"soglia_s": soglia_s, "n_righe": 0, "buco_max_s": None, "buco_da_stamp": None,
           "esito": "non_determinato", "motivo": None}
    if rows is None:
        out["motivo"] = "registro per frame assente"
        return out
    sel = []
    for r in rows:
        ta, si = _num(r.get("t_arr_wall")), _num(r.get("stamp_in"))
        if ta is not None and si is not None and t0_wall <= ta <= t1_wall:
            sel.append((ta, si))
    sel.sort()
    out["n_righe"] = len(sel)
    if len(sel) < 2:
        out["motivo"] = "meno di due nuvole in ingresso nel rollout"
        return out
    best, da = 0.0, None
    for (_, a), (_, b) in zip(sel, sel[1:]):
        g = round(b - a, 3)
        if g > best:
            best, da = g, a
    out["buco_max_s"], out["buco_da_stamp"] = best, da
    # gap >= 1 s; stamps on 0.1 s steps
    if best >= soglia_s - 1e-6:
        out["esito"] = "guasto"
        out["motivo"] = f"buco di {best:.2f} s nelle nuvole in ingresso al nodo (dallo stamp {da})"
    else:
        out["esito"] = "ok"
    return out


def classe_incompleto(status, controlli, accorge=None, calc_ok=True):
    """Class of a candidate with outcome timeout or stopped_early (None for other
    outcomes). controlli: name -> dict with outcome ok, guasto or
    non_determinato (genome-independent signals only). accorge: mola_accorge
    result on the twin log. Order: (a) guasto_catena, (b) mola_perde_posa,
    calcolo, (c) attacco_valido. An undetermined check is not enough for (a):
    it stays in the record."""
    if not stato_incompleto(status):
        return None
    controlli = controlli or {}
    guasti = [n for n, c in controlli.items() if c.get("esito") == "guasto"]
    nd = [n for n, c in controlli.items() if c.get("esito") == "non_determinato"]
    if guasti:
        classe = "guasto_catena"
        motivo = "; ".join(f"{n}: {controlli[n].get('motivo')}" for n in guasti)
    elif accorge is not None and accorge.get("accorge") is True:
        classe, motivo = "mola_perde_posa", accorge.get("motivo")
    elif calc_ok is False:
        classe, motivo = "calcolo", "calcolo del nodo oltre il periodo di scansione"
    else:
        classe = "attacco_valido"
        motivo = f"{str(status).split('(')[0].strip()} con catena sana e MOLA che pubblica"
    return {"classe": classe, "motivo": motivo, "status": str(status),
            "guasti": guasti, "non_determinati": nd, "controlli": controlli}


def punti_fine_finestra(baseline_csv):
    """Baseline pose at window end, from the JSON next to the baseline CSV
    (field punti_fine_finestra_medi: finestra, x_m, y_m). Returns
    {window: (x, y)}; ValueError if the file or field is missing."""
    import json
    p = os.path.splitext(str(baseline_csv))[0] + ".json"
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"punti di fine finestra della baseline non leggibili ({p}): {e}") from e
    righe = d.get("punti_fine_finestra_medi")
    if not isinstance(righe, list) or not righe:
        raise ValueError(f"{p}: campo punti_fine_finestra_medi assente o vuoto")
    out = {}
    for r in righe:
        k, x, y = r.get("finestra"), _num(r.get("x_m")), _num(r.get("y_m"))
        if k is None or x is None or y is None:
            raise ValueError(f"{p}: riga di punti_fine_finestra_medi incompleta: {r}")
        out[int(k)] = (x, y)
    return out


def danno_fine_finestra(end, punto):
    """Distance (m) between the true pose at rollout end and the baseline pose
    at window end; None if either is missing."""
    if end is None or punto is None:
        return None
    x, y = _num(end[0]), _num(end[1])
    if x is None or y is None:
        return None
    return float(math.hypot(x - punto[0], y - punto[1]))
