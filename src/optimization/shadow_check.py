"""Comparison between the victim MOLA and the shadow MOLA (--shadow-check).

The shadow (tv.SHADOW_NS) is a MOLA with the victim's configuration fed with
the same clouds, and the two are required to be identical. This module records,
scan by scan, how far they are from it and assigns a level per scan (A: bit
identical; B: within 1e-9 m / 1e-9 rad, CSV identical or one isolated unit in
the sixth digit; B': within the B' thresholds; C: above B' or a discrete field
differs; C+: above the C+ thresholds) and the episodes of scans processed by one
side only.

A lab measurement tool outside the decision loop: it reads only what the two
MOLA publish (pose and covariance on <ns>/lidar_odometry/pose, logged with 17
significant digits), the MOLA_LO_DEBUG_ICP_QUALITY printf lines in their logs,
the CSV traces (--trace), the updateLocalMap lines (only with
MOLA_VERBOSITY_MOLA_LO=DEBUG) and the node's crc log. Nothing computed here
reaches the twin, the fitness or the controller: the threat model excludes the
attacker reading the victim state, and here only the lab reads it.

Matching key: stamp in tenths of a second (10 Hz scans on multiples of 0.1 s;
leggi_printf rounds to the same step).

One-sided scan: stamp present in the CSV trace (or, without traces, in the
printf) of only one side. Classes: avvio (before the other's first stamp),
coda (after the last stamp common to both, not comparable) and in_corsa (the
rest: these are the episodes, scans within episodio_s count as one). With the
crc log a stamp not published to both is not an episode.
"""

import csv
import json
import math
import os
import re

# Default thresholds, fixed before the data. All can be overridden with
# --shadow-check-tol.
DEFAULT_TOL = {
    "b_xy_m": 1e-9,         # B: |dx|, |dy|
    "b_yaw_rad": 1e-9,      # B: |dyaw|
    "csv_ulp": 1e-6,        # B: one unit of the 6th digit, one column, isolated scan
    "bp_xy_m": 1e-4,        # B': |dxy|
    "bp_yaw_deg": 1e-3,     # B': |dyaw|
    "bp_sigma": 1e-5,       # B': |dsigma| (m)
    "bp_v": 1e-3,           # B': |dvx|, |dvy| (m/s)
    "bp_wz": 1.75e-4,       # B': |dwz| (rad/s)
    "bp_q": 4e-4,           # B': |dq| <= 4e-4
    "bp_radius": None,      # radii: no B' threshold, above one unit of the 6th digit is C
    "cp_xy_m": 4e-4,        # C+: |dxy| above
    "cp_yaw_deg": 1e-2,     # C+: |dyaw| above
    "cp_sigma": 2.5e-3,     # C+: |dsigma| above
    "episodio_s": 1.0,      # one-sided scans within 1 s: one episode; WARN within 1 s: worker
}
TOL_KEYS = tuple(DEFAULT_TOL)
LEVELS = ("A", "B", "B'", "C", "C+")
# The 25 columns of the trace CSV (6 decimals), all classified.
# Continuous state values with their B' threshold class (DEFAULT_TOL): sigma, v,
# wz, q, xy (m), yaw (rad, compared in deg); "altro": no B' threshold, above one
# unit of the 6th digit is C (as for the radii).
CSV_CONT = {"ADAPTIVE_THRESHOLD_SIGMA": "sigma", "vx": "v", "vy": "v", "wz": "wz", "icp_quality": "q",
            "ESTIMATED_OBSERVATION_RADIUS": "radius", "INSTANTANEOUS_OBSERVATION_RADIUS": "radius",
            "ESTIMATED_SENSOR_MAX_RANGE": "radius", "INSTANTANEOUS_SENSOR_MAX_RANGE": "radius",
            "robot_x": "xy", "robot_y": "xy", "robot_yaw": "yaw",
            "robot_z": "altro", "robot_pitch": "altro", "robot_roll": "altro",
            "vz": "altro", "wx": "altro", "wy": "altro", "SENSOR_TIME_OFFSET": "altro"}
# Discrete values: must be identical (otherwise C).
CSV_DISC = ("icp_iterations", "twistCorrectionCount", "ICP_ITERATION")
# Excluded columns, with the reason.
CSV_EXCLUDED = {
    "time_onLidar": "tempo di calcolo di onLidar (profiler_, LidarOdometry_Main.cpp:744): diverso fra due "
                    "processi a ogni scan; escluso come in G1 (G1_stato_completo/relazione.md:94)",
    "timestamp": "stamp della scansione: e' la chiave di abbinamento",
    "current_relative_timestamp": "stamp relativo al primo scan di ciascuna MOLA: il primo stamp e' "
                                  "confrontato a parte (primo_stamp)",
}
# B' threshold key per class; "radius" and "altro" use bp_radius (None by default).
CLASS_TOL = {"sigma": "bp_sigma", "v": "bp_v", "wz": "bp_wz", "q": "bp_q", "radius": "bp_radius",
             "altro": "bp_radius", "xy": "bp_xy_m", "yaw": "bp_yaw_deg"}
RX_KF = re.compile(r"updateLocalMap=(YES|NO)")
# Worker warning in the MRPT logger format, local time of day
# ([10:59:21.5424|WARN |LidarOdometry], checked against the ROS lines of the same
# log). Throttled to one per second (MRPT_LOG_THROTTLE_WARN_FMT(1.0, ...),
# LidarOdometry_SensorCallbacks.cpp:268-270).
RX_DROP = re.compile(r"\[(\d\d):(\d\d):(\d\d(?:\.\d+)?)\|WARN \|LidarOdometry\] Dropping LiDAR scan")
TWIST_COLS = ("vx", "vy", "vz", "wx", "wy", "wz")
POSE_COLS = ["who", "stamp_ns", "t_recv", "x", "y", "z", "qx", "qy", "qz", "qw"] + [f"c{i}" for i in range(36)]


def parse_tol(spec):
    """"bp_xy_m=1e-4,cp_sigma=0.0025" -> updated DEFAULT_TOL; "none" removes
    a threshold. Unknown keys or negative values raise ValueError."""
    tol = dict(DEFAULT_TOL)
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"soglia senza '=': {part!r}")
        k, v = (s.strip() for s in part.split("=", 1))
        if k not in DEFAULT_TOL:
            raise ValueError(f"soglia sconosciuta {k!r} (valide: {', '.join(TOL_KEYS)})")
        if v.lower() == "none":
            tol[k] = None
            continue
        v = float(v)
        if not v >= 0.0:
            raise ValueError(f"soglia {k} negativa o non numerica: {v}")
        tol[k] = v
    return tol


def key10(stamp_s):
    """Matching key: stamp in tenths of a second."""
    return int(round(float(stamp_s) * 10.0))


# ----------------------------------------------------------------------------- pose
def pose_record(msg, t_recv):
    """Pose record from nav_msgs/Odometry (pose and covariance; MOLA does not
    fill the message twist, BridgeROS2.cpp:2024-2038)."""
    p = msg.pose.pose
    q = p.orientation
    st = msg.header.stamp
    return {"stamp_ns": int(st.sec) * 1_000_000_000 + int(st.nanosec), "t_recv": float(t_recv),
            "x": float(p.position.x), "y": float(p.position.y), "z": float(p.position.z),
            "qx": float(q.x), "qy": float(q.y), "qz": float(q.z), "qw": float(q.w),
            "cov": [float(c) for c in msg.pose.covariance]}


def _g17(v):
    # 17 significant digits: the double reads back bit-identical
    return f"{float(v):.17g}"


def pose_row(who, r):
    return ([who, r["stamp_ns"], _g17(r["t_recv"])] + [_g17(r[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            + [_g17(c) for c in r["cov"]])


def read_pose_csv(path):
    """{'victim': [...], 'shadow': [...]} from shadow_check_poses.csv."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                r = {"stamp_ns": int(row["stamp_ns"]), "t_recv": float(row["t_recv"]),
                     **{k: float(row[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")},
                     "cov": [float(row[f"c{i}"]) for i in range(36)]}
            except (KeyError, TypeError, ValueError):
                continue        # truncated row
            out.setdefault(row["who"], []).append(r)
    return out


def yaw_of(r):
    return math.atan2(2.0 * (r["qw"] * r["qz"] + r["qx"] * r["qy"]), 1.0 - 2.0 * (r["qy"] ** 2 + r["qz"] ** 2))


def pose_diff(a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    dyaw = yaw_of(b) - yaw_of(a)
    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
    return {"dx_m": abs(dx), "dy_m": abs(dy), "dxy_m": math.hypot(dx, dy), "dyaw_rad": abs(dyaw),
            "dz_m": abs(b["z"] - a["z"]),
            "dcov": max((abs(u - v) for u, v in zip(a["cov"], b["cov"])), default=0.0),
            "dstamp_ns": abs(b["stamp_ns"] - a["stamp_ns"]),
            "bit": all(a[k] == b[k] for k in ("x", "y", "z", "qx", "qy", "qz", "qw"))}


# ----------------------------------------------------------------------------- logs and traces
def read_traces(path):
    """Complete rows of the MOLA CSV traces, first scan excluded
    (current_relative_timestamp = 0); raw values in "raw"; [] if missing.

    The CSV is an std::ofstream without flush (LidarOdometry_Main.cpp:752-755):
    even after MOLA exits the last row can be truncated. A row is kept only if
    it ends with a newline and has every header field."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    if len(lines) < 2:
        return []
    body = lines[1:-1]          # last piece is "" if the file ends with a newline, else truncated
    hdr_raw = next(csv.reader([lines[0]]))
    hdr = [h for h in hdr_raw if h != ""]
    out = []
    for r in csv.reader(body):
        if len(r) != len(hdr_raw) or any(v == "" for v in r[:len(hdr)]):
            continue
        d = dict(zip(hdr, r[:len(hdr)]))
        try:
            if float(d.get("current_relative_timestamp", "0")) == 0.0:
                continue
            out.append({"stamp_s": float(d["timestamp"]), "raw": d})
        except (KeyError, ValueError):
            continue
    return out


def read_printf(path):
    """MOLA_LO_DEBUG_ICP_QUALITY printf lines from the log; [] if missing."""
    if not path or not os.path.isfile(path):
        return []
    from src.analysis.scarti_mola import leggi_printf
    return leggi_printf(str(path))


def read_kf(path):
    """(YES, total) of the updateLocalMap lines in the log; None without lines
    (INFO verbosity: not measured, which is not a pass)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        v = RX_KF.findall(f.read())
    if not v:
        return None
    return (sum(1 for s in v if s == "YES"), len(v))


def read_drop_warn(path):
    """Local time of day (s since midnight) of the MRPT "Dropping LiDAR scan"
    warnings."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in RX_DROP.findall(f.read())]


def seconds_of_day(t_wall):
    """Local time of day of a time.time() value, as in the MRPT warnings."""
    import time as _t
    lt = _t.localtime(t_wall)
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec + (t_wall % 1.0)


def _tod_gap(a, b):
    d = abs(a - b) % 86400.0
    return min(d, 86400.0 - d)


def read_crc_log(path):
    """Node crc log (--shadow-crc-log): {key: row}."""
    if not path or not os.path.isfile(path):
        return None
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[key10(float(row["stamp"]))] = row
            except (KeyError, ValueError):
                continue
    return out


# ----------------------------------------------------------------------------- matching
def index_by_key(rows, stamp_of):
    """dict key -> first record, and number of repeated keys."""
    idx, dup = {}, 0
    for r in rows:
        k = stamp_of(r)
        if k in idx:
            dup += 1
            continue
        idx[k] = r
    return idx, dup


def pair_keys(a, b, final=False):
    """Paired keys and one-sided keys: avvio (before the other's first stamp),
    coda (after the last stamp present in BOTH: the CSV tail may still be
    buffered or lost at exit, so it is not comparable) and in_corsa. final does
    not change the rule (kept for the callers)."""
    ka, kb = set(a), set(b)
    lim = min(max(ka), max(kb)) if ka and kb else None

    def classify(only, other):
        c = {"avvio": [], "coda": [], "in_corsa": []}
        lo = min(other) if other else None
        for k in sorted(only):
            if lo is None or k < lo:
                c["avvio"].append(k)
            elif lim is None or k > lim:
                c["coda"].append(k)
            else:
                c["in_corsa"].append(k)
        return c
    return {"paired": sorted(ka & kb), "only_a": classify(ka - kb, kb), "only_b": classify(kb - ka, ka),
            "ultimo_comune": lim}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def csv_diff(ra, rb, tol):
    """Compare two trace rows on the state columns (CSV_EXCLUDED left out):
    identical as strings, differing columns, single unit of the 6th digit,
    differing discrete columns, max difference per class (yaw in deg)."""
    cols = sorted((set(ra) | set(rb)) - set(CSV_EXCLUDED))
    diff = [c for c in cols if ra.get(c) != rb.get(c)]
    out = {"identico": not diff, "colonne": diff, "discrete": [c for c in diff if c in CSV_DISC], "max": {},
           "non_classificate": [c for c in cols if c not in CSV_CONT and c not in CSV_DISC]}
    ulp = tol.get("csv_ulp")
    one = False
    if len(diff) == 1 and ulp is not None and diff[0] not in CSV_DISC:
        a, b = _num(ra.get(diff[0])), _num(rb.get(diff[0]))
        one = a is not None and b is not None and abs(abs(a - b) - ulp) <= ulp * 1e-3
    out["una_unita"] = one
    for c in cols:
        if c in CSV_DISC:
            continue
        cls = CSV_CONT.get(c, "altro")      # unknown column: treated as "altro"
        a, b = _num(ra.get(c)), _num(rb.get(c))
        if a is None or b is None:
            if ra.get(c) != rb.get(c):
                out["discrete"].append(c)
            continue
        d = abs(a - b)
        if cls == "yaw":
            d = abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))
        out["max"][cls] = max(out["max"].get(cls, 0.0), d)
    return out


def level_of(pd, cd, pr_disc_diff, tol, csv_isolated=True):
    """Level of a paired scan. pd: pose_diff or None, cd: csv_diff or None,
    pr_disc_diff: isGood differs (None if not measured)."""
    disc = bool(pr_disc_diff) or bool(cd and cd["discrete"])
    m = cd["max"] if cd else {}
    dxy = _mx_list([pd["dxy_m"] if pd else None, m.get("xy")])
    dyaw_deg = _mx_list([math.degrees(pd["dyaw_rad"]) if pd else None, m.get("yaw")])
    dsig = m.get("sigma")
    over_bp = False
    checks = [(dxy, "bp_xy_m"), (dyaw_deg, "bp_yaw_deg")] + [(v, CLASS_TOL[c]) for c, v in m.items()
                                                             if c not in ("xy", "yaw")]
    for v, k in checks:
        if v is None:
            continue
        t = tol.get(k)
        if t is None:
            # no B' threshold: above one unit of the 6th digit is C
            t = (tol.get("csv_ulp") or 0.0) * (1 + 1e-3)
        if v > t:
            over_bp = True
    if disc or over_bp:
        cp = ((dxy is not None and tol.get("cp_xy_m") is not None and dxy > tol["cp_xy_m"])
              or (dyaw_deg is not None and tol.get("cp_yaw_deg") is not None and dyaw_deg > tol["cp_yaw_deg"])
              or (dsig is not None and tol.get("cp_sigma") is not None and dsig > tol["cp_sigma"]))
        return "C+" if cp else "C"
    pose_b = pd is None or (pd["dx_m"] <= tol["b_xy_m"] and pd["dy_m"] <= tol["b_xy_m"]
                            and pd["dyaw_rad"] <= tol["b_yaw_rad"])
    csv_b = cd is None or cd["identico"] or (cd["una_unita"] and csv_isolated)
    if pose_b and csv_b:
        if (pd is None or pd["bit"]) and (cd is None or cd["identico"]):
            return "A"
        return "B"
    return "B'"


def _mx_list(vals):
    v = [x for x in vals if x is not None]
    return max(v) if v else None


def worst(levels):
    lv = [x for x in levels if x in LEVELS]
    return max(lv, key=LEVELS.index) if lv else None


def episodes(keys, gap_s):
    """Group keys (tenths of s) into episodes: scans within gap_s count as one."""
    eps = []
    for k in sorted(keys):
        if eps and (k - eps[-1][-1]) / 10.0 <= gap_s:
            eps[-1].append(k)
        else:
            eps.append([k])
    return eps


def compare(vp, sp, vpr, spr, vtr, str_, vkf, skf, tol=None, final=False, crc=None,
            drop_v=(), drop_s=(), imu_assente=False):
    """Full comparison: per-scan rows with level, one-sided scans, episodes.
    Returns (scans, info).

    Episode cause (needs the crc log): "worker" if the MOLA that lost the scan
    has a "Dropping LiDAR scan" WARN within episodio_s of publication; without a
    WARN, "trasporto" only with imu_assente (the silent drop,
    LidarOdometry_SensorCallbacks.cpp:248-251, exists only with an IMU),
    otherwise "non determinata". The one-WARN-per-second throttle does not hide
    a drop: the covering WARN is within the previous 1 s."""
    tol = DEFAULT_TOL if tol is None else tol
    P = [index_by_key(vp, lambda r: key10(r["stamp_ns"] * 1e-9))[0],
         index_by_key(sp, lambda r: key10(r["stamp_ns"] * 1e-9))[0]]
    R = [index_by_key(vpr, lambda r: key10(r["stamp_s"]))[0], index_by_key(spr, lambda r: key10(r["stamp_s"]))[0]]
    T = [index_by_key(vtr, lambda r: key10(r["stamp_s"]))[0], index_by_key(str_, lambda r: key10(r["stamp_s"]))[0]]
    # processed scans: from the traces if present, else from the printf
    src = "tracce" if (T[0] or T[1]) else ("printf" if (R[0] or R[1]) else "pose")
    E = {"tracce": T, "printf": R, "pose": P}[src]
    pk = pair_keys(E[0], E[1], final)
    common = set(pk["paired"])
    scans = {}
    for k in sorted(set(E[0]) | set(E[1]) | set(P[0]) | set(P[1])):
        s = {"key": k, "stamp_s": k / 10.0,
             "elaborata": ("entrambe" if k in common else "solo_vittima" if k in E[0] else
                           "solo_ombra" if k in E[1] else ""),
             "posa": ("entrambe" if k in P[0] and k in P[1] else "solo_vittima" if k in P[0] else
                      "solo_ombra" if k in P[1] else "")}
        if k in P[0] and k in P[1]:
            s["pd"] = pose_diff(P[0][k], P[1][k])
        if k in T[0] and k in T[1]:
            s["cd"] = csv_diff(T[0][k]["raw"], T[1][k]["raw"], tol)
        if k in R[0] and k in R[1]:
            s["isgood_diverso"] = R[0][k]["is_good"] != R[1][k]["is_good"]
            s["dsigma_printf"] = abs(R[0][k]["sigma"] - R[1][k]["sigma"])
        if crc is not None:
            c = crc.get(k)
            s["crc"] = (None if c is None else ("uguali" if c.get("crc_victim") == c.get("crc_shadow")
                                                else "omesso" if c.get("shadow_omitted") == "1" else "diversi"))
        if k in pk["only_a"]["coda"] or k in pk["only_b"]["coda"]:
            s["coda_non_confrontabile"] = True
        if "pd" in s:
            # pose-only level (live): available before the trace row is written
            s["livello_posa"] = level_of(s["pd"], None, None, tol)
        scans[k] = s
    ks = sorted(scans)
    for i, k in enumerate(ks):
        s = scans[k]
        if s["elaborata"] != "entrambe":
            continue
        cd = s.get("cd")
        iso = True
        if cd and not cd["identico"]:
            for j in (i - 1, i + 1):
                if 0 <= j < len(ks) and scans[ks[j]].get("cd") and not scans[ks[j]]["cd"]["identico"]:
                    iso = False
        s["livello"] = level_of(s.get("pd"), cd, s.get("isgood_diverso"), tol, iso)
    # episodes: one-sided in_corsa scans that were published to both
    ep_keys = []
    for side, other_drop in (("only_a", drop_s), ("only_b", drop_v)):
        for k in pk[side]["in_corsa"]:
            if crc is not None:
                c = crc.get(k)
                if c is None or c.get("shadow_omitted") == "1":
                    # not published to both: not an episode (deliberate omission kept apart)
                    scans[k]["omesso_o_non_pubblicato"] = True
                    continue
            ep_keys.append(k)
    eps = []
    gap = tol.get("episodio_s") or 1.0
    for grp in episodes(ep_keys, gap):
        who = sorted({scans[k]["elaborata"] for k in grp})
        cause = "non determinata"
        t = [_num(crc[k].get("t_pub_wall")) for k in grp if crc is not None and k in crc]
        t = [seconds_of_day(x) for x in t if x is not None]
        # the shadow lost the scan if only the victim processed it, and vice versa
        drops = (list(drop_s) if "solo_vittima" in who else []) + (list(drop_v) if "solo_ombra" in who else [])
        if t:
            if any(_tod_gap(d, x) <= gap for d in drops for x in t):
                cause = "worker"
            elif imu_assente:
                cause = "trasporto"
        for k in grp:
            scans[k]["episodio"] = True
        eps.append({"stamp_s": [k / 10 for k in grp], "chi": who, "causa": cause})
    info = {"sorgente_elaborate": src,
            "n_vittima": len(E[0]), "n_ombra": len(E[1]), "n_comuni": len(common),
            "solo_vittima": {c: len(v) for c, v in pk["only_a"].items()},
            "solo_ombra": {c: len(v) for c, v in pk["only_b"].items()},
            "primo_stamp_vittima_s": min(E[0]) / 10 if E[0] else None,
            "primo_stamp_ombra_s": min(E[1]) / 10 if E[1] else None,
            "ultimo_stamp_confrontabile_s": pk["ultimo_comune"] / 10 if pk["ultimo_comune"] is not None else None,
            "episodi": eps,
            "crc_mancanti": (None if crc is None else sum(1 for k in common if k not in crc)),
            "keyframe": (None if vkf is None or skf is None else
                         {"yes_vittima": vkf[0], "yes_ombra": skf[0], "uguali": vkf[0] == skf[0]}),
            "keyframe_stato": ("misurato" if vkf is not None and skf is not None else "non misurato"),
            "keyframe_ombra": None if skf is None else skf[0],
            "crc": (None if crc is None else
                    {"slot": len(crc), "uguali": sum(1 for c in crc.values() if c.get("crc_victim") == c.get("crc_shadow")),
                     "omessi": sum(1 for c in crc.values() if c.get("shadow_omitted") == "1")})}
    return scans, info


def window_summary(scans, info, lo_s, hi_s, entry_marked):
    """Summary of scans with lo_s < stamp <= hi_s (lo_s None = from start): worst
    level, level at the last common trace scan and last common pose, maxima,
    counts. Marked if C/C+, if it holds an episode, or if it follows an episode
    of the run (ingresso)."""
    sel = [s for k, s in sorted(scans.items()) if (lo_s is None or s["stamp_s"] > lo_s + 1e-9)
           and (hi_s is None or s["stamp_s"] <= hi_s + 1e-9)]
    com = [s for s in sel if s.get("livello")]

    def mx(f):
        v = [f(s) for s in com]
        v = [x for x in v if x is not None]
        return max(v) if v else None
    cnt = {lv: sum(1 for s in com if s["livello"] == lv) for lv in LEVELS}
    posa = [s for s in sel if s.get("livello_posa")]
    ep_here = sum(1 for s in sel if s.get("episodio"))
    kf = info.get("keyframe")
    lev = worst(s["livello"] for s in com)
    if kf is not None and not kf["uguali"]:
        lev = worst([lev, "C"])
    return {
        "scan_elaborati_vittima": sum(1 for s in sel if s["elaborata"] in ("entrambe", "solo_vittima")),
        "scan_elaborati_ombra": sum(1 for s in sel if s["elaborata"] in ("entrambe", "solo_ombra")),
        "scan_comuni": len(com),
        "solo_vittima_stamp_s": [s["stamp_s"] for s in sel if s["elaborata"] == "solo_vittima"][:20],
        "solo_ombra_stamp_s": [s["stamp_s"] for s in sel if s["elaborata"] == "solo_ombra"][:20],
        "livelli": cnt, "livello_peggiore": lev,
        "livello_ultimo_scan": com[-1]["livello"] if com else None,
        "stamp_ultimo_scan": com[-1]["stamp_s"] if com else None,
        "livello_ultima_posa": posa[-1]["livello_posa"] if posa else None,
        "stamp_ultima_posa": posa[-1]["stamp_s"] if posa else None,
        "episodi_nella_finestra": ep_here,
        "coda_non_confrontabile": sum(1 for s in sel if s.get("coda_non_confrontabile")),
        "max_dxy_m": mx(lambda s: s.get("pd", {}).get("dxy_m")),
        "max_dyaw_rad": mx(lambda s: s.get("pd", {}).get("dyaw_rad")),
        "max_dsigma": mx(lambda s: (s.get("cd") or {}).get("max", {}).get("sigma", s.get("dsigma_printf"))),
        "max_dv": mx(lambda s: (s.get("cd") or {}).get("max", {}).get("v")),
        "max_dwz": mx(lambda s: (s.get("cd") or {}).get("max", {}).get("wz")),
        "max_dq": mx(lambda s: (s.get("cd") or {}).get("max", {}).get("q")),
        "max_dcov": mx(lambda s: s.get("pd", {}).get("dcov")),
        "isgood_diversi": sum(1 for s in com if s.get("isgood_diverso")),
        "crc_diversi": sum(1 for s in sel if s.get("crc") == "diversi"),
        "keyframe": kf, "keyframe_stato": info.get("keyframe_stato"), "keyframe_ombra": info.get("keyframe_ombra"),
        "marcata": bool(lev in ("C", "C+") or entry_marked or ep_here),
        "ingresso": bool(entry_marked),
    }


def run_verdict(info):
    n = len(info["episodi"])
    return {"episodi": n, "esito": ("nessun episodio" if n == 0 else "1 episodio: da dichiarare" if n == 1
                                    else "ombra non affidabile: da segnalare")}


def missing_subscribers(infos, needed):
    """Namespaces in `needed` without a subscriber among the `infos` endpoints
    (TopicEndpointInfo from get_subscriptions_info_by_topic, node_namespace)."""
    have = {getattr(i, "node_namespace", None) for i in infos}
    return [ns for ns in needed if ns not in have]


def env_config_diff(victim_env, shadow_env):
    """Extra environment variables given to the two MOLA (the rest is inherited
    identically), with ROS_ARGS compared without the namespace."""
    def strip(env):
        e = dict(env or {})
        ra = e.pop("ROS_ARGS", None)
        if ra:
            try:
                a = json.loads(ra)
                keep = []
                for i, x in enumerate(a):
                    if str(x).startswith("__ns:=") or (x == "-r" and i + 1 < len(a)
                                                       and str(a[i + 1]).startswith("__ns:=")):
                        continue
                    keep.append(x)
                e["ROS_ARGS(senza __ns)"] = json.dumps(keep)
            except ValueError:
                e["ROS_ARGS(senza __ns)"] = ra
        return e
    v, s = strip(victim_env), strip(shadow_env)
    diff = sorted(k for k in set(v) | set(s) if v.get(k) != s.get(k))
    return {"vittima": v, "ombra": s, "diverse": diff, "identica": not diff}


SCAN_COLS = ["stamp_s", "elaborata", "posa", "livello", "crc", "dx_m", "dy_m", "dxy_m", "dyaw_rad", "dz_m",
             "dcov", "dstamp_ns", "posa_bit", "csv_identico", "csv_colonne", "csv_una_unita", "dsigma", "dv",
             "dwz", "dq", "dradius", "isgood_diverso"]


def scan_csv_rows(scans):
    out = []
    for k in sorted(scans):
        s = scans[k]
        pd, cd = s.get("pd") or {}, s.get("cd")
        m = (cd or {}).get("max", {})
        out.append({"stamp_s": s["stamp_s"], "elaborata": s["elaborata"], "posa": s["posa"],
                    "livello": s.get("livello", ""), "crc": s.get("crc", ""),
                    **{c: (_g17(pd[c]) if c in pd else "") for c in ("dx_m", "dy_m", "dxy_m", "dyaw_rad", "dz_m",
                                                                     "dcov", "dstamp_ns")},
                    "posa_bit": pd.get("bit", ""), "csv_identico": "" if cd is None else cd["identico"],
                    "csv_colonne": "" if cd is None else ";".join(cd["colonne"]),
                    "csv_una_unita": "" if cd is None else cd["una_unita"],
                    "dsigma": m.get("sigma", s.get("dsigma_printf", "")), "dv": m.get("v", ""),
                    "dwz": m.get("wz", ""), "dq": m.get("q", ""), "dradius": m.get("radius", ""),
                    "isgood_diverso": s.get("isgood_diverso", "")})
    return out


def write_scan_csv(path, scans):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCAN_COLS)
        w.writeheader()
        for r in scan_csv_rows(scans):
            w.writerow(r)


# ----------------------------------------------------------------------------- identity per metre
# In every run and every metre victim and shadow must process the same set of
# stamps with identical poses, otherwise the metre is "ombra_divergente" and
# invalid. Position threshold 1e-6 m; angle threshold the B level of
# DEFAULT_TOL (b_yaw_rad, 1e-9 rad), the only one in rad.
IDENT_XY_M = 1e-6
IDENT_YAW_RAD = DEFAULT_TOL["b_yaw_rad"]
ESITI_IDENTITA = ("identica", "ombra_divergente", "non_misurata")


def _lati(r):
    """(victim, shadow): whether each side processed the stamp, by CSV trace or
    published pose (the trace tail is lost at exit, the pose is not; the first
    scan has no trace row)."""
    el, po = r.get("elaborata") or "", r.get("posa") or ""
    return (el in ("entrambe", "solo_vittima") or po in ("entrambe", "solo_vittima"),
            el in ("entrambe", "solo_ombra") or po in ("entrambe", "solo_ombra"))


def identita_metri(rows, confini, xy_m=IDENT_XY_M, yaw_rad=IDENT_YAW_RAD):
    """Per-metre outcome from the rows of shadow_check_scans.csv (or scan_csv_rows).

    confini: window_record.confini_metri (metre k = stamps in (da_s, a_s], None =
    open). Per stamp: one-sided -> divergent; |dxy| > xy_m or |dyaw| > yaw_rad ->
    divergent; processed by both but without both poses (the lab log may miss
    one) the trace row decides: identical as strings (6 decimals, all state
    columns) confirms, otherwise divergent (pose not comparable). The effective
    threshold is then the trace one (|dx|, |dy| < 1e-6 m, so dxy up to
    1.41e-6 m, angle 1e-6 rad).

    End of run: only after the driven part of the last metre (fine_guida_s of
    the last boundary) a one-sided stamp counts as coda_non_confrontabile and
    does not decide (the MOLA instances stop one after the other). Before that,
    a MOLA that stops processing leaves one-sided stamps: divergent. Without
    fine_guida_s there is no exemption."""
    S = []
    for r in rows:
        try:
            k = key10(r["stamp_s"])
        except (KeyError, TypeError, ValueError):
            continue
        v, o = _lati(r)
        if v or o:
            S.append((k, v, o, r))
    S.sort(key=lambda x: x[0])
    fg = confini[-1].get("fine_guida_s") if confini else None
    chiusura = key10(fg) if fg is not None else None
    out = []
    n = len(confini)
    for i, c in enumerate(confini):
        lo = key10(c["da_s"]) if c.get("da_s") is not None else None
        hi = key10(c["a_s"]) if c.get("a_s") is not None else None
        m = {"finestra": c.get("finestra"), "da_s": c.get("da_s"), "a_s": c.get("a_s"),
             "esito": "non_misurata", "n_comuni": 0, "n_solo_vittima": 0, "n_solo_ombra": 0,
             "n_posa_mancante": 0, "n_solo_traccia": 0, "n_oltre_soglia": 0, "max_dxy_m": None, "max_dyaw_rad": None,
             "prima_stamp_divergente_s": None, "motivo_prima": None,
             "solo_vittima_stamp_s": [], "solo_ombra_stamp_s": [], "coda_non_confrontabile": 0}
        if (i > 0 and lo is None) or (i < n - 1 and hi is None):
            m["motivo"] = "confini del metro non disponibili (open_stamp assente nella history)"
            out.append(m)
            continue
        for k, v, o, r in S:
            if (lo is not None and k <= lo) or (hi is not None and k > hi):
                continue
            if chiusura is not None and k > chiusura and not (v and o):
                m["coda_non_confrontabile"] += 1
                continue
            motivo = None
            if v and o:
                m["n_comuni"] += 1
                dxy, dyaw = _num(r.get("dxy_m")), _num(r.get("dyaw_rad"))
                if r.get("posa") != "entrambe" or dxy is None or dyaw is None:
                    if str(r.get("csv_identico")) == "True":
                        m["n_solo_traccia"] += 1
                    else:
                        m["n_posa_mancante"] += 1
                        motivo = "posa non confrontabile"
                else:
                    m["max_dxy_m"] = dxy if m["max_dxy_m"] is None else max(m["max_dxy_m"], dxy)
                    m["max_dyaw_rad"] = dyaw if m["max_dyaw_rad"] is None else max(m["max_dyaw_rad"], dyaw)
                    if dxy > xy_m or dyaw > yaw_rad:
                        m["n_oltre_soglia"] += 1
                        motivo = "posa diversa"
            elif v:
                m["n_solo_vittima"] += 1
                m["solo_vittima_stamp_s"].append(k / 10)
                motivo = "solo vittima"
            else:
                m["n_solo_ombra"] += 1
                m["solo_ombra_stamp_s"].append(k / 10)
                motivo = "solo ombra"
            if motivo and m["prima_stamp_divergente_s"] is None:
                m["prima_stamp_divergente_s"], m["motivo_prima"] = k / 10, motivo
        m["solo_vittima_stamp_s"] = m["solo_vittima_stamp_s"][:20]
        m["solo_ombra_stamp_s"] = m["solo_ombra_stamp_s"][:20]
        if m["prima_stamp_divergente_s"] is not None:
            m["esito"] = "ombra_divergente"
        elif m["n_comuni"]:
            m["esito"] = "identica"
        else:
            m["motivo"] = "nessuno stamp confrontabile nel metro"
        out.append(m)
    return out


def _g(x):
    return "-" if x is None else f"{x:.3g}"


def esito_identita(metri, xy_m=IDENT_XY_M, yaw_rad=IDENT_YAW_RAD):
    """Run outcome (machine readable) and text lines for the summary."""
    nome = lambda m: f"w{m['finestra']}"
    div = [m for m in metri if m["esito"] == "ombra_divergente"]
    nm = [m for m in metri if m["esito"] == "non_misurata"]
    ok = [m for m in metri if m["esito"] == "identica"]
    if not metri:
        esito, testo = "non_misurata", "OMBRA NON MISURATA (nessun metro nella history): run non valida"
    elif div:
        esito = "ombra_divergente"
        testo = (f"OMBRA DIVERGENTE nei metri {', '.join(nome(m) for m in div)} ({len(div)} su {len(metri)}; "
                 f"identica in {len(ok)}" + (f", non misurata in {', '.join(nome(m) for m in nm)}" if nm else "") + ")")
    elif nm:
        esito = "non_misurata"
        testo = (f"OMBRA NON MISURATA nei metri {', '.join(nome(m) for m in nm)} (identica in {len(ok)}/{len(metri)}): "
                 "run non valida")
    else:
        esito, testo = "identica", f"ombra: identica in {len(ok)}/{len(metri)} metri"
    righe = [f"{nome(m)} ({m['da_s']}, {m['a_s']}]: solo vittima {m['n_solo_vittima']}, solo ombra "
             f"{m['n_solo_ombra']}, posa mancante {m['n_posa_mancante']}, oltre soglia {m['n_oltre_soglia']} "
             f"su {m['n_comuni']} comuni, max dxy {_g(m['max_dxy_m'])} m, max dyaw {_g(m['max_dyaw_rad'])} rad, "
             f"prima stamp divergente {m['prima_stamp_divergente_s']} ({m['motivo_prima']})" for m in div]
    return {"esito": esito, "testo": testo, "righe": righe,
            "metri_divergenti": [m["finestra"] for m in div], "metri_non_misurati": [m["finestra"] for m in nm],
            "n_metri": len(metri), "soglie": {"xy_m": xy_m, "yaw_rad": yaw_rad}}


def identita_da_rep(rep, xy_m=IDENT_XY_M, yaw_rad=IDENT_YAW_RAD):
    """Offline check of a finished rep_NN (read only): shadow_check_scans.csv and
    history.json. Returns esito_identita with the metres."""
    from src.optimization import window_record as wrec
    rep = str(rep or "")
    sc = os.path.join(rep, "shadow_check_scans.csv")
    if not rep or not os.path.isfile(sc):
        motivo = ("shadow_check_scans.csv assente (confronto finale non scritto)"
                  if rep and os.path.isfile(os.path.join(rep, "shadow_check.json"))
                  else "nessun confronto ombra-vittima nella run")
        return {"esito": "non_misurata", "testo": f"OMBRA NON MISURATA ({motivo}): run non valida", "righe": [],
                "metri": [],
                "metri_divergenti": [], "metri_non_misurati": [], "n_metri": 0, "motivo": motivo,
                "soglie": {"xy_m": xy_m, "yaw_rad": yaw_rad}}
    with open(sc, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        with open(os.path.join(rep, "history.json"), encoding="utf-8") as f:
            hist = json.load(f)
    except (OSError, ValueError) as e:
        motivo = f"history.json illeggibile: {type(e).__name__}"
        return {"esito": "non_misurata", "testo": f"OMBRA NON MISURATA ({motivo}): run non valida", "righe": [],
                "metri": [], "metri_divergenti": [], "metri_non_misurati": [], "n_metri": 0, "motivo": motivo,
                "soglie": {"xy_m": xy_m, "yaw_rad": yaw_rad}}
    confini = wrec.confini_metri(hist if isinstance(hist, list) else [])
    metri = identita_metri(rows, confini, xy_m, yaw_rad)
    return {**esito_identita(metri, xy_m, yaw_rad), "metri": metri,
            "fonte": "shadow_check_scans.csv, history.json (victim_window.open_stamp)"}


class ShadowCheck:
    """Live comparison log. Poses come from the lab subscriptions (add_pose);
    logs, traces and crc are read at every window start from the paths of the
    two runners and of the node."""

    def __init__(self, out_dir, victim_runner, shadow_runner, tol=None, config=None, crc_log=None):
        self.out_dir = str(out_dir)
        self.victim, self.shadow = victim_runner, shadow_runner
        self.tol = dict(DEFAULT_TOL if tol is None else tol)
        self.config = config
        self.crc_log = crc_log
        self.poses = {"victim": [], "shadow": []}
        self.by_window = {}
        self.final = None
        self.last_hi = None
        self.final_rows = None     # rows of shadow_check_scans.csv from the final comparison
        self.identita = None       # per-metre outcome (identita_metri), at end of run
        self.entry = False         # an episode occurred: later windows are marked "ingresso"
        self.pose_csv = os.path.join(self.out_dir, "shadow_check_poses.csv")
        self._f = open(self.pose_csv, "w", newline="", encoding="utf-8", buffering=1)
        self._w = csv.writer(self._f)
        self._w.writerow(POSE_COLS)

    def add_pose(self, who, rec):
        self.poses[who].append(rec)
        if self._f is not None:
            self._w.writerow(pose_row(who, rec))

    def _compare(self, final):
        def p(r, a):
            v = getattr(r, a, None)
            return str(v) if v else None
        vl, sl = p(self.victim, "log_path"), p(self.shadow, "log_path")
        return compare(self.poses["victim"], self.poses["shadow"], read_printf(vl), read_printf(sl),
                       read_traces(p(self.victim, "traces_path")), read_traces(p(self.shadow, "traces_path")),
                       read_kf(vl), read_kf(sl), tol=self.tol, final=final, crc=read_crc_log(self.crc_log),
                       drop_v=read_drop_warn(vl), drop_s=read_drop_warn(sl))

    def snapshot(self, window=None, final=False):
        """Comparison at window start (window) or final. Read errors go into the
        record and do not stop the run."""
        rec = {"window": window, "finale": bool(final)}
        try:
            scans, info = self._compare(final)
            hi = max((s["stamp_s"] for s in scans.values() if s.get("livello")), default=None)
            entry_before = self.entry
            rec["finestra_precedente"] = window_summary(scans, info, self.last_hi, None, entry_before)
            rec["dall_avvio"] = window_summary(scans, info, None, None, False)
            rec["episodi"] = info["episodi"]
            rec["run"] = run_verdict(info)
            rec["primo_stamp"] = {"vittima": info["primo_stamp_vittima_s"], "ombra": info["primo_stamp_ombra_s"],
                                  "uguale": info["primo_stamp_vittima_s"] == info["primo_stamp_ombra_s"]}
            rec["crc"] = info["crc"]
            rec["sorgente_elaborate"] = info["sorgente_elaborate"]
            if info["episodi"]:
                self.entry = True
            if hi is not None:
                self.last_hi = hi
            if final:
                write_scan_csv(os.path.join(self.out_dir, "shadow_check_scans.csv"), scans)
                self.final_rows = scan_csv_rows(scans)
        except Exception as e:   # measurement tool: must not stop the run
            rec["errore"] = f"{type(e).__name__}: {e}"
        if final:
            self.final = rec
        else:
            self.by_window[window] = rec
        self.save()
        return rec

    def save(self):
        with open(os.path.join(self.out_dir, "shadow_check.json"), "w", encoding="utf-8") as f:
            json.dump({"config": self.config, "soglie": self.tol,
                       "finestre": {str(k): v for k, v in self.by_window.items()},
                       "fine": self.final, "identita_per_metro": self.identita}, f, indent=1)

    def fill_history(self, history):
        """Add shadow_check to the compared window records and the final
        comparison to the last record (shadow_check_fine)."""
        for r in history:
            k = r.get("window")
            if k in self.by_window and "shadow_check" not in r:
                r["shadow_check"] = compact(self.by_window[k])
        if self.final is not None and history:
            history[-1]["shadow_check_fine"] = compact(self.final)
            if self.final_rows is not None:
                # identity per metre, with boundaries from the history (victim_window)
                try:
                    from src.optimization import window_record as wrec
                    metri = identita_metri(self.final_rows, wrec.confini_metri(history))
                    e = esito_identita(metri)
                    self.identita = {**e, "metri": {str(m["finestra"]): m for m in metri}}
                    history[-1]["shadow_check_fine"]["identita_per_metro"] = {
                        k: e[k] for k in ("esito", "testo", "metri_divergenti", "metri_non_misurati", "n_metri")}
                except Exception as ex:   # lab measurement: does not stop the run
                    self.identita = {"esito": "non_misurata", "errore": f"{type(ex).__name__}: {ex}"}
                self.save()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def compact(rec):
    """History field: previous window, episodes, run outcome."""
    if "errore" in rec:
        return {"errore": rec["errore"]}
    return {"finestra_precedente": rec["finestra_precedente"], "episodi": len(rec["episodi"]),
            "run": rec["run"], "primo_stamp": rec["primo_stamp"], "crc": rec["crc"],
            "sorgente_elaborate": rec["sorgente_elaborate"]}


def main(argv=None):
    """Offline check: python3 src/optimization/shadow_check.py REP [--json F].
    Exit code 0 identical, 3 shadow divergent, 5 not measured."""
    import argparse
    ap = argparse.ArgumentParser(description="identita' ombra-vittima per metro su una rep_NN gia' fatta")
    ap.add_argument("rep")
    ap.add_argument("--json", help="scrive il risultato completo in questo file (mai nella rep)")
    ap.add_argument("--xy-m", type=float, default=IDENT_XY_M)
    ap.add_argument("--yaw-rad", type=float, default=IDENT_YAW_RAD)
    a = ap.parse_args(argv)
    r = identita_da_rep(a.rep, a.xy_m, a.yaw_rad)
    print(r["testo"])
    for ln in r["righe"]:
        print("  " + ln)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return {"identica": 0, "ombra_divergente": 3}.get(r["esito"], 5)


if __name__ == "__main__":
    import sys
    _radice = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _radice not in sys.path:
        sys.path.insert(0, _radice)
    sys.exit(main())
