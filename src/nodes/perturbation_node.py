#!/usr/bin/env python3
"""
Inline adversarial perturbation node on the LiDAR stream.

Sits between the sensor and MOLA: every cloud is perturbed with the current
genome before it reaches the SLAM.

    Isaac Sim -> add_intensity -> [this node] -> MOLA -> waypoint_follower

The genome is hot-swapped on /attack/genome without restarting the node or
interrupting the stream to MOLA; the attack is toggled on /attack/enabled and
the state is published on /attack/status. The per-cloud seed is hash(genome) +
frame index, with the index reset on every genome change, so the sequence of
perturbed clouds is reproducible for the same genome and input. Perceptibility
(Chamfer) is sampled every N frames. Cloud decoding and encoding are vectorised
to stay within the 100 ms budget at 10 Hz.

Usage:
    python3 src/nodes/perturbation_node.py                # attack active
    python3 src/nodes/perturbation_node.py --passthrough  # baseline, clouds unchanged

Topics
------
  in      /carter/lidar_with_intensity   (configurable)
  out     /carter/lidar_perturbed        (configurable)
  genome  /attack/genome        Float32MultiArray, 17 values in [-1, 1]
  on/off  /attack/enabled       Bool
  status  /attack/status        String (JSON), for the orchestrator

Continuous victim stream (second output, /carter/lidar_victim): a MOLA instance
that is never restarted listens to it for the whole applied path. It is active
only while /attack/victim_stream is true (during the search the victim receives
nothing and stays still); /attack/victim_attack enables the perturbation on that
stream with the genome from /attack/victim_genome, with its own generator and
drift. While the victim stream is active the main output is in passthrough (one
perturbation per cloud, within the 100 ms budget) and the clean cloud also goes
to /carter/lidar_attacker, for the attacker's continuous MOLA.

Operator mode (--operator-mode): "random" (default) uses the per-cloud seed
described above; with "insieme" the operators that choose where to strike
(genes 5, 6, 10, 13, 16) perturb a fixed set of zones, chosen on the first cloud
of the metre from the genome seed and the metre index (/attack/metro, from the
orchestrator) and tracked on the following clouds using only the clean input
clouds (src/perturbations/structures.py). No ground-truth pose and no victim
state enter the choice.

With --zone-attaccante (only with insieme, off by default) the streams choose
nothing: the orchestrator (the attacker) asks on /attack/fotografia to save its
last clean cloud, computes the zones of every genome on that snapshot and sends
them on /attack/zone (path, sha256, hash); the node installs them on the first
cloud after the genome and tracks them. Missing or invalid zones: no zones (the
localised operators leave the cloud untouched), reported in the zone log and in
the status (src/perturbations/zone_attaccante.py).

With --rampa-genoma R > 1 (only with --zone-attaccante, off by default) a
stream's genome change is not a step: before the new genome N the orchestrator
sends on /attack/rampa the stream's previous genome P and its zones on the
metre snapshot; on the R - 1 mixed clouds of the metre (k = 0..R-2) the stream
applies P and N together with weights 1 - alpha_k and alpha_k,
alpha_k = (k+1)/R (PerturbationGenerator._coherent_pesato); from k = R-1
(alpha = 1) only N. Victim and twin use the same function on the same data.
Without P's data for that new genome (also after a cancel,
{"annulla": true} on /attack/rampa) only N is applied and the reason goes into
the status (rampa).

With --genoma-in-moto S (only with --zone-attaccante, off by default) the
stream's previous genome P arrives on /attack/rampa as for the ramp; from k = 0
the stream applies only P (step ramp), and N enters on the first cloud with
k >= --genoma-in-moto-kmin where the attacker's tracker has moved at least S cm
from the reference pose, or at k = --genoma-in-moto-kmax
(PerturbationGenerator.imposta_gradino). The reference is reset after a
generator-state load and when the victim stream reopens. On metre 0 only N.
Per-stream state and per-phase Chamfer (P, N) go into the status
(genoma_in_moto), and gradino_* columns into the zone log.

With --stato-generatore (off by default) the orchestrator saves on
/attack/stato_generatore the generator state of a stream at the end of the
settling phase and loads it into the realigned twin: with the same genome the
twin resumes cloud index, drift, tracked zones and ramp of the victim, and from
the next cloud on the attack is the one the victim receives.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import queue
import threading
from collections import deque

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Empty, Float32MultiArray, String

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import (  # noqa: E402
    FAMIGLIE_LIMITI,
    RAMPA_PREFISSO,
    PerturbationGenerator,
    impronta_stato,
    moltiplicatori_da_opzioni,
)
from src.perturbations import structures as strutture  # noqa: E402
from src.perturbations import zone_attaccante as za  # noqa: E402

# All four fields are contiguous float32: the message is packed with a single
# tobytes() instead of going through create_cloud().
FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]
POINT_STEP = 16


def _csv_val(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v) if v == v else ""
    return str(v)


class FrameLog:
    """Per-frame node log (--frame-log): one CSV row per received cloud.

    *_mono times are time.monotonic(), *_wall time.time(); stamps are in s.
    add() queues the row in memory (no I/O in the cloud threads); a thread
    writes to file every flush_s seconds and close() writes the rest. Exiting
    without close() loses at most flush_s seconds of rows."""

    COLUMNS = (
        "t_arr_wall", "t_arr_mono", "stamp_in",
        "frame_idx", "victim_frame_idx", "enabled", "victim_stream", "victim_attack",
        "n_points", "proc_ms", "t_release_target_mono", "stamp_victim", "stamp_main_out",
        "t_release_mono", "replay_active", "t_pub_main_end_mono",
        "t_pub_victim_end_mono", "t_pub_attacker_end_mono", "t_pub_end_wall",
    )

    def __init__(self, path, flush_s=1.0, extra_columns=()):
        self.path = self.libero(Path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_s = float(flush_s)
        self.n_written = 0
        # trailing columns only with the options that use them; header unchanged otherwise
        self.columns = tuple(self.COLUMNS) + tuple(extra_columns)
        self._q = deque()
        self._f = open(self.path, "w", buffering=1 << 16)
        self._f.write(",".join(self.columns) + "\n")
        self._f.flush()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    @staticmethod
    def libero(path):
        """Unused path: if the file exists (node restarted on the same output),
        _1, _2, ... is appended before the extension."""
        if not path.exists():
            return path
        i = 1
        while path.with_name(f"{path.stem}_{i}{path.suffix}").exists():
            i += 1
        return path.with_name(f"{path.stem}_{i}{path.suffix}")

    def add(self, row):
        # rows always have len(columns) fields (missing ones left empty)
        if len(row) < len(self.columns):
            row = list(row) + [None] * (len(self.columns) - len(row))
        self._q.append(row)

    def _drain(self):
        lines = []
        while True:
            try:
                row = self._q.popleft()
            except IndexError:
                break
            lines.append(",".join(_csv_val(v) for v in row))
        if lines:
            self._f.write("\n".join(lines) + "\n")
            self._f.flush()
            self.n_written += len(lines)

    def _loop(self):
        while not self._stop.wait(self.flush_s):
            self._drain()

    def close(self):
        self._stop.set()
        self._t.join(timeout=5.0)
        self._drain()
        self._f.close()


# Effective per-cloud delay (--frame-log-delay, trailing columns of the frame
# log): D_min = planned release - arrival; delta_X = end of publish on output X
# - arrival - D_min; DDS receive and source timestamps come from the
# subscription MessageInfo (ns -> s, rmw clock).
DELAY_COLUMNS = ("d_min_ms", "delta_main_ms", "delta_victim_ms", "delta_attacker_ms",
                 "t_pub_shadow_end_mono", "delta_shadow_ms", "t_dds_recv", "t_dds_src")


class _FlogRow(list):
    """Frame-log row carrying the DDS timestamps of the cloud and the end of the
    publish to the shadow (only with --frame-log-delay)."""
    __slots__ = ("dds", "t_shadow_end")


class ShadowCrcLog(FrameLog):
    """Per-slot crc log (--shadow-crc-log): one row per cloud published to the
    victim, with the crc of the message sent to the victim and of the one sent
    to the shadow (empty if omitted). Without --shadow-topic the shadow listens
    to the victim topic: a single message, shadow_topic empty."""

    COLUMNS = ("slot", "stamp", "victim_frame_idx", "n_points", "crc_victim", "crc_shadow",
               "shadow_topic", "shadow_omitted", "t_pub_wall")

    # If the file exists (node restarted on the next attempt after an Isaac
    # stall, same output folder) <name>_1, _2, ... is used as for the frame log
    # (FrameLog.libero); the orchestrator reads the latest one
    # (attack_orchestrator.registri_nodo_correnti).


class StructLog(FrameLog):
    """Per-cloud zone log of the insieme mode (--struct-log): seed, metre,
    chosen zones and points hit per operator, per stream."""

    COLUMNS = (
        "t_arr_mono", "stream", "genome_id", "frame_idx", "stamp", "seme", "metro",
        "per_op", "k", "pose_x", "pose_y", "pose_yaw_deg", "icp_pairs", "icp_rms_m",
        "n_hit_g5", "n_hit_g6", "n_hit_g10", "n_hit_g13", "n_hit_g16", "anchors",
    )


# Trailing zone-log columns with --rampa-genoma: ramp state and weight of the
# new genome on this cloud (k is the k column), sha of the previous genome,
# points hit per operator of the previous genome ("g6:806 g13:500").
RAMPA_COLONNE = ("rampa_stato", "rampa_alfa", "rampa_prec_sha", "n_hit_prec")


# Trailing zone-log columns with --genoma-in-moto: phase of the cloud (P,
# ingresso, N, metro0, empty without step), entry k of N, tracker displacement
# from the reference at this cloud and at the previous one of the segment (cm),
# stamp gap from the previous cloud (s), k of the reference cloud.
GRADINO_COLONNE = ("gradino_fase", "gradino_k_ingresso", "gradino_spost_cm", "gradino_spost_prima_cm",
                   "gradino_buco_stamp_s", "gradino_rif_k")


# Trailing zone-log columns with --moltiplicatore-limiti: input points, removed
# points (input minus those left after dropout) and added points of the cloud,
# measured on the stream's clouds, and removed fraction n_tolti / n_in.
LIMITI_COLONNE = ("n_in", "n_tolti", "n_aggiunti", "frac_tolti")


def moltiplicatori_uguali(z, gen):
    """True if the zones were computed with the same limit multiplier as the
    stream's generator (None = off on both sides)."""
    return (getattr(z, "moltiplicatori", None)
            == getattr(gen, "moltiplicatori_attivi", lambda: None)())


def limiti_riga(c):
    """LIMITI_COLONNE values for the cloud just processed by the stream."""
    c = c or {}
    return [c.get("n_in"), c.get("n_removed"), c.get("n_added"), c.get("frac_removed")]


def gradino_riga(sg):
    """GRADINO_COLONNE values for the cloud just processed by the stream."""
    sg = sg or {}
    return [sg.get("fase"), sg.get("k_ingresso"), sg.get("spost_cm"), sg.get("spost_prec_cm"),
            sg.get("buco_s"), sg.get("rif_k")]


def rampa_riga(info_rampa, n_hit):
    """RAMPA_COLONNE values for the cloud just processed by the stream."""
    ri = info_rampa or {}
    pre = RAMPA_PREFISSO
    prec = " ".join(f"{k[len(pre):]}:{v}" for k, v in sorted((n_hit or {}).items()) if k.startswith(pre))
    return [ri.get("stato"), ri.get("alpha"), ri.get("genoma_prec_sha"), prec]


def impronta_posto(p):
    """Fingerprint of a generator-state slot (--stato-generatore): cloud index,
    metre, genome, seed and generator state, not the source stream; equal to the
    twin's after loading if the copy is exact."""
    return impronta_stato({k: p[k] for k in ("frame_idx", "metro", "genoma_sha", "seme", "gen")})


def salva_nuvole_scelta(victim_stream, victim_perturbata, enabled, dropped, replay_active,
                        frame_idx, victim_frame_idx, k_max):
    """--salva-nuvole: (stream, k) of the cloud to save, or None.

    Only perturbed and published clouds are saved: for the victim those of the
    attack branch (not the clean ones nor those dropped by --victim-never-clean),
    for the twin those with the attack on, outside the victim stream and outside
    replay (during replay the main output does not publish). k is the cloud
    index since the stream's current genome, read at the start of processing."""
    if victim_stream:
        if not victim_perturbata or dropped:
            return None
        flusso, k = "vittima", victim_frame_idx
    elif enabled and not replay_active:
        flusso, k = "gemello", frame_idx
    else:
        return None
    if k is None or k < 0 or k >= int(k_max):
        return None
    return flusso, int(k)


def salva_nuvole_tag(tag):
    """Tag used as file-name prefix: empty -> "senza_tag", characters outside
    [A-Za-z0-9._-] replaced with "_"."""
    t = "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "_" for c in (tag or "").strip())
    return t or "senza_tag"


def salva_nuvole_nome(tag, flusso, genome_id, k, stamp_sec, stamp_nanosec):
    """<tag>_<stream>_g<genome_id>_k<kk>_<sec>.<nanosec>.npz (input stamp)."""
    return (f"{salva_nuvole_tag(tag)}_{flusso}_g{int(genome_id)}_k{int(k):02d}_"
            f"{int(stamp_sec)}.{int(stamp_nanosec):09d}.npz")


def _come_pubblicata(c):
    """The cloud as _publish_cloud writes it: x, y, z, intensity in float32
    (intensity 100 if missing), same point order."""
    if c.shape[1] == 3:
        return np.hstack([np.asarray(c, dtype=np.float32), np.full((len(c), 1), 100.0, dtype=np.float32)])
    return np.array(c[:, :4], dtype=np.float32, order="C")


class SalvaNuvole:
    """Input and output clouds of the first k_max clouds of every genome, per
    stream (--salva-nuvole DIR, --salva-nuvole-k N).

    accoda() is called from the cloud callback and never blocks: with a full
    queue the record is dropped and counted. A low-priority thread writes one
    uncompressed .npz per record (np.savez: ~0.4 ms for two 45000-point clouds
    against ~32 ms compressed, for 20 % less space) and one row of indice.csv.
    No existing file is overwritten."""

    COLONNE = ("file", "tag", "flusso", "genome_id", "seme", "metro", "k",
               "stamp_in_sec", "stamp_in_nanosec", "stamp_out_sec", "stamp_out_nanosec",
               "t_arr_mono", "n_msg", "n_in", "n_out", "t_scritto_wall")

    def __init__(self, cartella, k_max=20, coda=64):
        self.cartella = Path(cartella)
        self.cartella.mkdir(parents=True, exist_ok=True)
        self.k_max = int(k_max)
        self.tag = ""
        self.salvati = 0
        self.scartati_coda = 0
        self.errori = 0
        self._q = queue.Queue(maxsize=int(coda))
        self._indice = self.cartella / "indice.csv"
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def record(self, flusso, genome_id, seme, metro, k, h_in, h_out, t_arr_mono, n_msg,
               nuvola_in, nuvola_out):
        """Cloud record (float32 copies: the writer thread never reads node arrays)."""
        si, ni = int(h_in.stamp.sec), int(h_in.stamp.nanosec)
        so, no = int(h_out.stamp.sec), int(h_out.stamp.nanosec)
        diversa = (so, no) != (si, ni)
        return {"tag": self.tag, "flusso": flusso, "genome_id": int(genome_id), "seme": int(seme),
                "metro": metro, "k": int(k), "stamp_in": (si, ni),
                "stamp_out": (so, no) if diversa else None, "t_arr_mono": float(t_arr_mono),
                "n_msg": int(n_msg), "ingresso": _come_pubblicata(nuvola_in),
                "uscita": _come_pubblicata(nuvola_out)}

    def accoda(self, rec):
        try:
            self._q.put_nowait(rec)
            return True
        except queue.Full:
            self.scartati_coda += 1
            return False

    def stato(self):
        return {"dir": str(self.cartella), "k": self.k_max, "tag": self.tag,
                "salvati": self.salvati, "scartati_coda": self.scartati_coda, "errori": self.errori}

    def _scrivi(self, rec):
        nome = salva_nuvole_nome(rec["tag"], rec["flusso"], rec["genome_id"], rec["k"], *rec["stamp_in"])
        p = FrameLog.libero(self.cartella / nome)
        so = rec["stamp_out"] or (-1, -1)
        with open(p, "wb") as f:
            np.savez(f, ingresso=rec["ingresso"], uscita=rec["uscita"],
                     flusso=np.array(rec["flusso"]), tag=np.array(salva_nuvole_tag(rec["tag"])),
                     genome_id=np.int64(rec["genome_id"]), seme=np.int64(rec["seme"]),
                     metro=np.int64(-1 if rec["metro"] is None else rec["metro"]), k=np.int64(rec["k"]),
                     stamp_in=np.array(rec["stamp_in"], dtype=np.int64),
                     stamp_out=np.array(so, dtype=np.int64),
                     t_arr_mono=np.float64(rec["t_arr_mono"]), n_msg=np.int64(rec["n_msg"]),
                     n_in=np.int64(len(rec["ingresso"])), n_out=np.int64(len(rec["uscita"])))
        nuovo = not self._indice.exists()
        riga = [p.name, salva_nuvole_tag(rec["tag"]), rec["flusso"], rec["genome_id"], rec["seme"],
                rec["metro"], rec["k"], rec["stamp_in"][0], rec["stamp_in"][1],
                None if rec["stamp_out"] is None else rec["stamp_out"][0],
                None if rec["stamp_out"] is None else rec["stamp_out"][1],
                rec["t_arr_mono"], rec["n_msg"], len(rec["ingresso"]), len(rec["uscita"]), time.time()]
        with open(self._indice, "a") as f:
            if nuovo:
                f.write(",".join(self.COLONNE) + "\n")
            f.write(",".join(_csv_val(v) for v in riga) + "\n")

    def _loop(self):
        # lowest priority for the writer thread only (Linux: per-thread nice)
        try:
            import os
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass
        while True:
            try:
                rec = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self._scrivi(rec)
                self.salvati += 1
            except Exception:
                self.errori += 1
            finally:
                self._q.task_done()

    def chiudi(self, timeout=10.0):
        """Writes the queued records and exits."""
        self._stop.set()
        self._t.join(timeout=timeout)


def make_generator(args) -> PerturbationGenerator:
    """Generator built from the command-line options, identical for the main
    output (twin) and for the victim. With --operator-mode insieme: coherent
    mode with scan tracker, case rule, floor excluded for g10, fallback, g13 by
    score and fixed zone set."""
    kw = {}
    if getattr(args, "operator_mode", "random") == "insieme":
        kw = {"operator_mode": "coherent", "struct_track": "scan", "struct_rule": "caso",
              "struct_g10_pavimento": True, "struct_ripiego": True,
              "struct_g13_punteggio": True, "struct_insieme": True}
        if getattr(args, "zone_attaccante", False):
            kw["zone_attaccante"] = True
    # limit multiplier (--moltiplicatore-limiti, off by default): same dict for
    # twin and victim
    mol = moltiplicatori_da_opzioni(getattr(args, "moltiplicatore_limiti", 1.0),
                                    getattr(args, "famiglie_limiti", None))
    if mol:
        kw["moltiplicatori"] = mol
    return PerturbationGenerator(
        max_point_shift=args.max_point_shift,
        noise_std=args.noise_std,
        max_dropout_rate=args.dropout_rate,
        max_drift_lag_s=args.max_drift_lag_s,
        structural=getattr(args, "structural", "net"),
        fast_ops=bool(getattr(args, "fast_ops", False)),
        **kw,
    )


def cloud_crc(msg):
    """crc32 of the message: stamp, frame_id, shape and point bytes."""
    import zlib
    h = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}|{msg.header.frame_id}|" \
        f"{msg.width}|{msg.point_step}|".encode()
    return zlib.crc32(bytes(msg.data), zlib.crc32(h)) & 0xFFFFFFFF


def _stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class FreshCloudGate:
    """Filter for clouds acquired across an Isaac restore
    (--fresh-cloud-after-restore).

    After pause and restore, the first scan published after play was partly
    assembled before the pause, at the pose preceding the restore. arm(base) is
    called with the simulation stopped, base = simulated time of the pause (last
    /clock) or, if no /clock arrived, the last input stamp. While armed,
    check(stamp) drops clouds with stamp <= base + margin_s; the first one past
    the threshold disarms it. After max_drop consecutive drops the filter
    disarms anyway (clock restarted, e.g. Isaac relaunched)."""

    EPS = 1e-3      # stamps and /clock lie on the physics step grid (1/60 s)
    # trailing frame-log columns: event ("armato", "non_armato",
    # "scartata_ripristino"), threshold, arm base, its source, sequence
    COLUMNS = ("fresh_gate", "fresh_threshold", "fresh_base", "fresh_base_src", "fresh_seq")

    def __init__(self, margin_s=0.1, max_drop=5):
        self.margin_s = float(margin_s)
        self.max_drop = int(max_drop)
        self.threshold = None       # None = disarmed
        self.base = None
        self.base_src = None
        self.dropped = 0            # drops since the last arm
        self.dropped_total = 0
        self.n_arm = 0
        self.n_forced = 0           # disarms due to max_drop

    @property
    def armed(self):
        return self.threshold is not None

    def arm(self, base, src):
        self.base = float(base)
        self.base_src = src
        self.threshold = self.base + self.margin_s
        self.dropped = 0
        self.n_arm += 1

    def check(self, stamp):
        """(forward, event): event is None when disarmed, else "scartata",
        "prima_fresca" (the cloud that disarms) or "tetto_scarti"."""
        if self.threshold is None:
            return True, None
        if stamp > self.threshold + self.EPS:
            self.threshold = None
            return True, "prima_fresca"
        if self.dropped >= self.max_drop:
            self.threshold = None
            self.n_forced += 1
            return True, "tetto_scarti"
        self.dropped += 1
        self.dropped_total += 1
        return False, "scartata"


def genome_seed(genome: np.ndarray) -> int:
    """Deterministic seed derived from the genome.

    The same genome must produce the same perturbation, otherwise NSGA-III
    selection compares noisy fitness values.
    """
    h = hashlib.sha1(np.asarray(genome, dtype=np.float64).tobytes()).digest()
    return int.from_bytes(h[:4], "little")


# Perceptibility terms with the attack off (--perc-terms).
ZERO_TERMS = {"chamfer_m2": 0.0, "chamfer_cm": 0.0, "point_ratio": 0.0, "structural_cm": 0.0,
              "magnitude_cm": 0.0, "n_orig": 0, "n_pert": 0,
              "point_ratio_gross": 0.0, "n_removed": 0, "n_added": 0}
TC_KEYS = ("tc_frac8", "tc_frac16", "tc_med_cm", "tc_p90_cm", "tc_rot_deg", "tc_trans_cm", "tc_clean_frac8")


def apply_test_struct(out, frame_idx, sector_deg, ghost_wall, slow_drift, drift_until):
    """Applies the structured probes (V4/V5) in sequence to an already copied
    cloud and returns it. The probes are coherent in time (fixed sector and
    wall, slowly growing drift)."""
    if sector_deg and sector_deg > 0:
        import numpy as _np
        ang = _np.degrees(_np.arctan2(out[:, 1], out[:, 0]))
        keep = _np.abs(ang) > (sector_deg / 2.0)
        out = out[keep]
    if ghost_wall and ghost_wall > 0:
        import numpy as _np
        rng = _np.random.default_rng(12345 + frame_idx)
        n = 3000
        g = _np.zeros((n, out.shape[1]), dtype=out.dtype)
        g[:, 0] = ghost_wall
        g[:, 1] = rng.uniform(-6, 6, n)
        g[:, 2] = rng.uniform(-0.5, 3.0, n)
        if out.shape[1] >= 4:
            g[:, 3] = 100.0
        out = _np.vstack([out, g])
    if slow_drift and slow_drift != 0.0:
        # Slow coherent drift y += RATE*frame while the attack is on; past
        # drift_until the attack is removed (clean cloud, zero drift): for V5 this
        # shows whether MOLA recovers or stays poisoned by the corrupted map.
        if drift_until <= 0 or frame_idx < drift_until:
            out = out.copy()
            out[:, 1] += slow_drift * frame_idx
    return out


def temporal_coherence(prev_xyz, cur_xyz, thr=0.08, n_sub=4000, iters=5, seed=0):
    """Temporal coherence between two consecutive output scans.

    Aligns the current scan to the previous one with a rigid transform (mini
    ICP: nearest neighbour and Umeyama without scale, 5 iterations, pairs beyond
    3 medians rejected) and measures the residuals. A perturbation coherent in
    time (constant rotation, drift, scale) is absorbed as motion, as MOLA's ICP
    does; noise, ghosts and distortions that change every scan stay in the
    residuals. Returns the fraction beyond thr and 2 thr, median and p90 of the
    residual (cm), rotation (deg) and translation (cm) of the transform."""
    from scipy.spatial import cKDTree
    if prev_xyz is None or cur_xyz is None or len(prev_xyz) < 50 or len(cur_xyz) < 50:
        return None
    rng = np.random.default_rng(seed)
    a = prev_xyz[:, :3].astype(np.float64)
    b = cur_xyz[:, :3].astype(np.float64)
    if len(b) > n_sub:
        b = b[rng.choice(len(b), n_sub, replace=False)]
    tree = cKDTree(a)
    R = np.eye(3); t = np.zeros(3)
    bb = b
    for _ in range(iters):
        d, idx = tree.query(bb, k=1, workers=-1)
        lim = max(0.3, 3.0 * float(np.median(d)))
        ok = d < lim
        if ok.sum() < 20:
            break
        P = bb[ok]; Q = a[idx[ok]]
        mp = P.mean(axis=0); mq = Q.mean(axis=0)
        H = (P - mp).T @ (Q - mq)
        U, _, Vt = np.linalg.svd(H)
        Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        Ri = Vt.T @ Dm @ U.T
        ti = mq - Ri @ mp
        bb = (Ri @ bb.T).T + ti
        R = Ri @ R; t = Ri @ t + ti
    d, _ = tree.query(bb, k=1, workers=-1)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return {"tc_frac8": float(np.mean(d > thr)), "tc_frac16": float(np.mean(d > 2.0 * thr)),
            "tc_med_cm": float(np.median(d) * 100.0), "tc_p90_cm": float(np.percentile(d, 90) * 100.0),
            "tc_rot_deg": ang, "tc_trans_cm": float(np.linalg.norm(t) * 100.0)}


class PerturbationNode(Node):
    # Class-level defaults of the measurement options (--measures, --shared-nn,
    # --frame-log-delay), also for nodes built without __init__ in tests.
    _measures = "inline"
    _shared_nn = False
    _meas_q = None
    _meas_lock = None
    _delay_log = False
    # --victim-never-clean (off by default): clouds that would go out clean to
    # victim and shadow, dropped and delivered (always 0 with the option)
    _never_clean = False
    victim_clean_dropped = 0
    victim_clean_out = 0
    # --banco-twin-drift (off by default): initial drift of the twin
    _banco_twin_drift_on = False
    _twin_drift_init = None
    # --strumenti-ritardo (off by default): delay instrumentation and current cloud
    _strum = None
    _strum_rec = None
    # --rampa-genoma (1 = off): ramp length
    _rampa_R = 1
    # --stato-generatore (off by default): generator-state slots
    _stato_gen_on = False
    # --genoma-in-moto (off by default): step threshold and limits
    _gim = None
    # --chamfer-per-fase (off by default, on with --genoma-in-moto): Chamfer per stream and phase
    _cpf = False
    # --moltiplicatore-limiti (1 = off): removed-point columns in the zone log
    _mol_on = False

    def __init__(self, args):
        super().__init__("perturbation_node")

        self.generator = make_generator(args)
        self._max_drift_lag_s = args.max_drift_lag_s
        # Continuous victim stream: separate generator (own drift and zone
        # state), own genome, own frame counter.
        self.victim_generator = make_generator(args)
        # Limit multiplier (--moltiplicatore-limiti, 1 = off): the same in both
        # generators (make_generator); removed points per cloud in the zone log
        # (LIMITI_COLONNE).
        self._mol_on = self.generator.moltiplicatori_attivi() is not None
        # Output-preserving generator optimisations (PERT_OTTIMIZZAZIONI, off by
        # default): the thread pool is created here, not on the first cloud.
        strutture.avvia_pool()
        self.get_logger().info(
            "ottimizzazioni del generatore: "
            + (",".join(sorted(strutture.OTT)) + f" ({strutture.WORKERS} thread)" if strutture.OTT
               else "spente"))
        # --operator-mode insieme: current metre per stream, sent by the
        # orchestrator on /attack/metro before the genome; None = never received
        # (the generator then chooses with rule R1, without seed).
        self._operator_mode = getattr(args, "operator_mode", "random")
        self._metro = {"gemello": None, "vittima": None}
        # Attacker zones (--zone-attaccante, off by default): per-stream state.
        self._zone_on = bool(getattr(args, "zone_attaccante", False))
        if self._zone_on and self._operator_mode != "insieme":
            raise ValueError("--zone-attaccante richiede --operator-mode insieme")
        # Ramp into the new genome (--rampa-genoma R, 1 = off): previous genome and
        # its zones per stream, from /attack/rampa before the new genome.
        self._rampa_R = int(getattr(args, "rampa_genoma", 1))
        if self._rampa_R < 1:
            raise ValueError("--rampa-genoma: R >= 1 (1 = spenta)")
        if self._rampa_R > 1 and not self._zone_on:
            raise ValueError("--rampa-genoma R > 1 richiede --zone-attaccante")
        self._rampa_ric = {"gemello": None, "vittima": None}
        self._rampa_malformato = None      # last unreadable stream message
        # ramp messages received per stream: only the latest one counts
        self._rampa_arrivi = {"gemello": 0, "vittima": 0}
        self._rampa_arrivi_lock = threading.Lock()
        # Genome switch in motion (--genoma-in-moto S, 0 = off): same channel as the
        # ramp for the previous genome; Chamfer sum and count per stream and phase.
        S = float(getattr(args, "genoma_in_moto", 0.0) or 0.0)
        if S < 0:
            raise ValueError("--genoma-in-moto: S >= 0 (0 = spento)")
        if S > 0:
            kmin = int(getattr(args, "genoma_in_moto_kmin", 20))
            kmax = int(getattr(args, "genoma_in_moto_kmax", 40))
            if not self._zone_on:
                raise ValueError("--genoma-in-moto richiede --zone-attaccante")
            if self._rampa_R > 1:
                raise ValueError("--genoma-in-moto e --rampa-genoma R > 1 insieme")
            if not 0 <= kmin <= kmax:
                raise ValueError(f"--genoma-in-moto: serve 0 <= kmin <= kmax (kmin {kmin}, kmax {kmax})")
            self._gim = {"S_cm": S, "kmin": kmin, "kmax": kmax}
        # Chamfer per stream and phase P/N (--chamfer-per-fase, always with the
        # step): sum and count of computed values for the stream's current genome;
        # without step every cloud is phase N.
        self._cpf = bool(getattr(args, "chamfer_per_fase", False)) or self._gim is not None
        if self._cpf:
            self._gim_acc = {f: self._gim_acc_vuoto(None) for f in ("gemello", "vittima")}
        # --stato-generatore: slots holding a stream's generator state, driven by
        # /attack/stato_generatore ("salva", "carica"); the lock keeps save and
        # load between two clouds, never in the middle of one.
        self._stato_gen_on = bool(getattr(args, "stato_generatore", False))
        # Generator lock, always active: a whole cloud and the callbacks that change
        # genome or generator state (genomes, drift, metre, zones, ramp) never
        # overlap. Without it, a genome arriving mid-cloud would reset frame_idx and
        # the += 1 at the end of the cloud would bring it to 1. Reentrant:
        # _set_genome loads the generator state under the same lock.
        self._gen_lock = threading.RLock()
        if self._stato_gen_on:
            self._stato_gen = {"seq": None, "posti": {}, "armato": None, "armato_seq": None,
                               "ultimo_caricato": None, "ultimo_errore": None}
            self._stato_gen_sha_vittima = None
        self._struct_log = (StructLog(args.struct_log,
                                      extra_columns=((("zone_stato", "zone_hash", "zone_attesi", "zone_scarto")
                                                      if self._zone_on else ())
                                                     + (RAMPA_COLONNE if self._rampa_R > 1 else ())
                                                     + (GRADINO_COLONNE if self._gim else ())
                                                     + (LIMITI_COLONNE if self._mol_on else ())))
                            if self._operator_mode == "insieme" and getattr(args, "struct_log", None)
                            else None)
        if self._zone_on:
            self._genoma_sha = {"gemello": None, "vittima": None}
            self._zone_spec = {"gemello": None, "vittima": None}
            self._zone_ric = {"gemello": None, "vittima": None}
            self._zone_app = {f: {"hash": None, "stato": None, "n_ok": 0, "n_err": 0}
                              for f in ("gemello", "vittima")}
            self._foto_cache = {}
            # the snapshot cache is cleared and reloaded in carica_zone: one load at
            # a time (zones are loaded outside the generator lock)
            self._foto_lock = threading.Lock()
            # zone messages received per stream: only the latest is installed
            self._zone_arrivi = {"gemello": 0, "vittima": 0}
            self._zone_arrivi_lock = threading.Lock()
            self._foto_stato = None
            # last clean clouds sent to the attacker's MOLA (stamp, cloud)
            self._att_buf = deque(maxlen=20)
            self.generator.zone_fornitore = (
                lambda s_, m_, p_: self._zone_fornitore("gemello", s_, m_, p_))
            self.victim_generator.zone_fornitore = (
                lambda s_, m_, p_: self._zone_fornitore("vittima", s_, m_, p_))
        self.victim_stream = False
        self.victim_attack = False
        self.victim_params = None
        self.victim_seed = 0
        self.victim_genome_id = 0
        self.victim_frame_idx = 0
        self.victim_frames = 0
        # Continuous victim clock: during the search the stream is closed and
        # simulation time advances by minutes; on reopening MOLA would see a stamp
        # jump, discard its velocity model and restart without prior at the
        # steady-state threshold (tracking lost on restart). Victim (and attacker)
        # clouds are re-stamped with a clock that does not advance during pauses:
        # for them the pause does not exist, as on the real robot. The offset from
        # simulation time is in the status, to realign the estimates to ground truth.
        self._victim_last_sim = None
        self._victim_last_clock = None
        self._victim_offset = 0.0
        # Recording of the attacked clouds sent to the victim (stamp on the
        # continuous clock, frame, points) and replay to the evaluation twin:
        # before an evaluation the twin receives the victim's history, so it starts
        # from its state (map and estimate) and not from scratch. During and after
        # replay the main output is stamped on a twin clock that continues from the
        # last replayed stamp.
        self._victim_rec = deque(maxlen=args.victim_record_max)
        # Total recorded clouds (including those evicted from the full queue) and
        # first stamp of the last replay: they tell whether replay starts from the
        # first cloud sent to the victim.
        self._victim_rec_total = 0
        self._replay_first_stamp = None
        self._replay_seq = 0
        self._replay_active = False
        self._replay_n = 0
        self._replay_total = 0
        self._replay_last_stamp = None
        # Kept drift (orchestrator --drift-reset-on keep, command
        # /attack/drift_keep): no reset on genome change.
        self._drift_keep = False
        self._twin_clock_on = False
        self._twin_last_sim = None
        self._twin_last_clock = None
        self._twin_offset = 0.0
        # --twin-clock-base (off by default): first twin header taken from the
        # stamp saved by the shadow.
        self._twin_clock_base_on = bool(getattr(args, "twin_clock_base", False))
        self._twin_clock_seq = None
        self._twin_first_ns = None
        self._twin_clock_first_out_ns = None
        # --banco-twin-drift (off by default, no subscription): drift the twin
        # starts from at the next genome, set on /attack/twin_drift_init.
        self._banco_twin_drift_on = bool(getattr(args, "banco_twin_drift", False))
        self._twin_drift_init = None

        # Fresh clouds after a restore (--fresh-cloud-after-restore, off by
        # default). The orchestrator arms the filter with the simulation stopped,
        # after the restore (/attack/fresh_after_restore, sequence number); clouds
        # stamped up to the pause time + margin go to no output.
        self._fresh_on = bool(getattr(args, "fresh_cloud_after_restore", False))
        self._fresh_gate = FreshCloudGate(getattr(args, "fresh_cloud_margin_s", 0.1),
                                          getattr(args, "fresh_cloud_max_drop", 5))
        self._fresh_lock = threading.Lock()
        self._fresh_seq = None
        self._fresh_last_clock = None
        self._fresh_last_stamp_in = None

        # Per-frame log (--frame-log), off by default.
        self._frame_log = (FrameLog(args.frame_log,
                                    extra_columns=((FreshCloudGate.COLUMNS if self._fresh_on else ())
                                                   + (DELAY_COLUMNS if getattr(args, "frame_log_delay", False)
                                                      else ())))
                           if getattr(args, "frame_log", None) else None)
        if self._frame_log is not None:
            self.get_logger().info(f"  registro per frame: {self._frame_log.path}")
        # Delay instrumentation (--strumenti-ritardo, off by default): no import
        # and no thread without the option (src/nodes/strumenti_ritardo.py).
        self._strum = None
        if getattr(args, "strumenti_ritardo", None):
            from src.nodes.strumenti_ritardo import StrumentiRitardo
            self._strum = StrumentiRitardo(args.strumenti_ritardo)
            self.get_logger().info(f"  strumenti del ritardo: {self._strum.path}")

        self.enabled = not (args.start_disabled or args.passthrough)
        self.frame_idx = 0
        self.genome_id = 0
        self.n_in = 0
        self.n_out = 0
        self.proc_times = []       # perturbation compute time
        self.lat_times = []        # receive to publish, waits included
        # Perceptibility of the last measured cloud: second NSGA-III objective,
        # read by the orchestrator on /attack/status.
        self._chamfer_every = args.chamfer_every
        self._passthrough_delay = args.passthrough_delay_ms / 1000.0
        self._min_latency_ms = args.min_latency_ms
        self._min_spacing = args.min_spacing_ms / 1000.0
        self._gaussian_sigma = args.gaussian_sigma
        self._chamfer = float("nan")
        # Separate perceptibility terms (--perc-terms): last sample, always
        # updated together with _chamfer.
        self._perc_terms_on = bool(getattr(args, "perc_terms", False))
        self._perc_terms = None
        self._tc = None                   # temporal coherence of the last output scan
        self._tc_prev = {"main": None, "victim": None}   # subsample of the previous scan per stream
        self._tc_last_idx = {"main": None, "victim": None}
        # Test modes for checking MOLA's guards (explicit, testing only): rigid
        # step translation, cloud reduced to N points, doubled scan.
        self._test_offset = (np.array([float(v) for v in args.test_offset.split(",")]) if args.test_offset else None)
        self._test_offset_after = int(args.test_offset_after)
        self._test_keep_points = int(args.test_keep_points)
        self._test_double_scan = int(args.test_double_scan)
        self._test_jitter = float(args.test_jitter)
        self._test_seed = int(args.test_seed)
        self._test_sector = float(args.test_sector)
        self._test_ghost_wall = float(args.test_ghost_wall)
        self._test_slow_drift = float(args.test_slow_drift)
        self._test_drift_until = int(args.test_drift_until)
        self._frames_since_enable = 0
        self._double_pending = False
        # Geometric perceptibility metrics of the last measured cloud.
        self._perc = None
        # Perceptibility measures (Chamfer, metrics, temporal coherence) off the
        # cloud path (--measures thread): computed after enqueueing, in a dedicated
        # thread and in cloud order, with the same values. --shared-nn: Chamfer and
        # metrics share the same two nearest-neighbour queries. Default: inline and
        # separate queries.
        self._measures = getattr(args, "measures", "inline")
        self._shared_nn = bool(getattr(args, "shared_nn", False))
        self._meas_q = None
        self._meas_ms = deque(maxlen=200)
        if self._measures == "thread":
            self._meas_lock = threading.Lock()
            self._meas_q = queue.Queue()
            threading.Thread(target=self._meas_loop, daemon=True).start()
        self._delay_log = bool(getattr(args, "frame_log_delay", False)) and bool(getattr(args, "frame_log", None))
        # Victim and shadow never receive a clean cloud: with the stream open but
        # no attack or no genome, the cloud is dropped (and counted) instead of
        # passing through unchanged; the attacker still receives it.
        self._never_clean = bool(getattr(args, "victim_never_clean", False))
        # Input and output of the first N clouds of every genome (--salva-nuvole):
        # off by default, no thread and no subscription.
        self._salva = (SalvaNuvole(args.salva_nuvole, getattr(args, "salva_nuvole_k", 20))
                       if getattr(args, "salva_nuvole", None) else None)

        if args.genome_file:
            g = np.load(args.genome_file)
            self.get_logger().info(f"Genoma caricato da {args.genome_file}")
        else:
            # An all-zero genome is not "no perturbation": each gene maps [-1, 1]
            # onto its parameter range, so zero is mid-scale, a medium-strength
            # attack. Unchanged clouds need --passthrough, which bypasses
            # apply_perturbation.
            g = self.generator.random_genome() * args.perturbation_level
        self._set_genome(g, source="avvio")

        # Clouds must be processed one at a time and in order (per-frame seeds,
        # temporal drift), but the input queue must absorb bursts: with depth 10
        # and a callback sleeping for the minimum latency, upstream delay lost
        # scans. Status and commands live in a reentrant group so they never wait
        # behind a cloud.
        self._cloud_group = MutuallyExclusiveCallbackGroup()
        self._misc_group = ReentrantCallbackGroup()
        # with two arguments rclpy also passes the MessageInfo (DDS timestamps)
        cloud_cb = ((lambda msg, info: self._cloud_cb(msg, info)) if self._delay_log
                    else self._cloud_cb)
        if self._strum is not None:
            cloud_cb = lambda msg, info: self._cloud_cb_strumenti(msg, info)
        # whole cloud under the generator lock, same signature as the callback
        # (one or two arguments)
        interna = cloud_cb
        if self._delay_log or self._strum is not None:
            cloud_cb = lambda msg, info: self._con_lock_generatore(interna, msg, info)
        else:
            cloud_cb = lambda msg: self._con_lock_generatore(interna, msg)
        self.sub = self.create_subscription(
            PointCloud2, args.input_topic, cloud_cb, args.queue_depth,
            callback_group=self._cloud_group)
        self.pub = self.create_publisher(
            PointCloud2, args.output_topic, 10)
        self.victim_pub = self.create_publisher(
            PointCloud2, args.victim_topic, 10)
        # Clean cloud for the attacker's MOLA (continuous like the victim, same
        # gating: only on applied segments).
        self.attacker_pub = self.create_publisher(
            PointCloud2, args.attacker_topic, 10)
        # Shadow (--shadow-topic, off by default): the same cloud as the victim on
        # its own topic; --shadow-omit-index omits the given slot (positive
        # control); --shadow-crc-log logs the crcs.
        self.shadow_topic = getattr(args, "shadow_topic", None)
        self.shadow_pub = (self.create_publisher(PointCloud2, self.shadow_topic, 10)
                           if self.shadow_topic else None)
        self._shadow_omit = int(getattr(args, "shadow_omit_index", -1))
        self._shadow_slot = 0
        # --report-matched: readers matched to the victim publisher, to the shadow
        # one (if it has its own topic) and to the main one, in the status and, as
        # trailing column matched_victim, in the crc log. Without it both are
        # unchanged.
        self._report_matched = bool(getattr(args, "report_matched", False))
        self._crc_log = (ShadowCrcLog(args.shadow_crc_log,
                                      extra_columns=("matched_victim",) if self._report_matched else ())
                         if getattr(args, "shadow_crc_log", None) else None)
        self.create_subscription(
            Float32MultiArray, "/attack/victim_genome", self._sotto_lock(self._victim_genome_cb), 10,
            callback_group=self._misc_group)
        self.create_subscription(
            Bool, "/attack/victim_stream", self._victim_stream_cb, 10,
            callback_group=self._misc_group)
        self.create_subscription(
            Bool, "/attack/victim_attack", self._victim_attack_cb, 10,
            callback_group=self._misc_group)
        self.create_subscription(
            Empty, "/attack/victim_drift_reset", self._sotto_lock(self._victim_drift_reset_cb), 10,
            callback_group=self._misc_group)
        self.create_subscription(
            String, "/attack/twin_replay", self._twin_replay_cb, 10,
            callback_group=self._misc_group)

        self.create_subscription(
            Float32MultiArray, "/attack/genome", self._sotto_lock(self._genome_cb), 10,
            callback_group=self._misc_group)
        self.create_subscription(
            Bool, "/attack/enabled", self._enabled_cb, 10,
            callback_group=self._misc_group)
        # Resets only the accumulated temporal drift (not the frame counter): the
        # orchestrator uses it to start the drift when motion starts rather than
        # when the genome is loaded.
        self.create_subscription(
            Empty, "/attack/drift_reset", self._sotto_lock(self._drift_reset_cb), 10,
            callback_group=self._misc_group)
        self.create_subscription(
            Bool, "/attack/drift_keep", self._drift_keep_cb, 10,
            callback_group=self._misc_group)
        if self._fresh_on:
            from rclpy.qos import qos_profile_sensor_data
            from rosgraph_msgs.msg import Clock
            from std_msgs.msg import UInt32
            self.create_subscription(Clock, "/clock", self._fresh_clock_cb, qos_profile_sensor_data,
                                     callback_group=self._misc_group)
            self.create_subscription(UInt32, "/attack/fresh_after_restore", self._fresh_arm_cb, 10,
                                     callback_group=self._misc_group)
        if self._twin_clock_base_on:
            self.create_subscription(String, "/attack/twin_clock_base", self._twin_clock_base_cb, 10,
                                     callback_group=self._misc_group)
        if self._banco_twin_drift_on:
            self.create_subscription(Float32MultiArray, "/attack/twin_drift_init",
                                     self._sotto_lock(self._twin_drift_init_cb), 10, callback_group=self._misc_group)
        if self._operator_mode == "insieme":
            self.create_subscription(String, "/attack/metro", self._sotto_lock(self._metro_cb), 10,
                                     callback_group=self._misc_group)
        if getattr(self, "_zone_on", False):
            self.create_subscription(String, "/attack/zone", self._zone_cb, 10,
                                     callback_group=self._misc_group)
            self.create_subscription(String, "/attack/fotografia", self._foto_cb, 10,
                                     callback_group=self._misc_group)
        if self._salva is not None:
            self.create_subscription(String, "/attack/salva_nuvole_tag", self._salva_tag_cb, 10,
                                     callback_group=self._misc_group)
        if self._rampa_R > 1 or self._gim:
            self.create_subscription(String, "/attack/rampa", self._rampa_cb, 10,
                                     callback_group=self._misc_group)
        if self._stato_gen_on:
            self.create_subscription(String, "/attack/stato_generatore", self._stato_gen_cb, 10,
                                     callback_group=self._misc_group)
        self.status_pub = self.create_publisher(String, "/attack/status", 10)

        # 10 Hz like the clouds: the status also carries the accumulated drift and
        # the frame index, which the orchestrator writes into every trace row.
        self.create_timer(0.1, self._publish_status, callback_group=self._misc_group)
        # Deferred publishing: the callback processes and enqueues with the release
        # time (receive + minimum latency); this thread sleeps until then and
        # publishes in order. The receiving thread never blocks on the latency,
        # and nominal and candidates keep the same delay.
        self._out_q = queue.Queue()
        self._pub_thread = threading.Thread(target=self._pub_loop, daemon=True)
        self._pub_thread.start()

        self.get_logger().info("=" * 64)
        self.get_logger().info("  PERTURBATION NODE")
        self.get_logger().info("=" * 64)
        self.get_logger().info(f"  in  : {args.input_topic}")
        self.get_logger().info(f"  out : {args.output_topic}")
        self.get_logger().info(
            f"  attacco {'ATTIVO' if self.enabled else 'PASSTHROUGH (nuvole invariate)'}")
        if self._min_latency_ms > 0:
            self.get_logger().info(f"  latenza minima per scan: {self._min_latency_ms:.0f} ms")
        if self._max_drift_lag_s != float("inf"):
            self.get_logger().info(
                f"  tetto alla deriva temporale: {self._max_drift_lag_s:.2f} s "
                f"({self.generator.max_drift_m*100:.0f} cm)")
        if self._fresh_on:
            self.get_logger().info(
                f"  dopo un ripristino solo nuvole con stamp > pausa + {self._fresh_gate.margin_s:.3f} s "
                f"(al piu' {self._fresh_gate.max_drop} scarti)")
        if self._operator_mode != "random":
            self.get_logger().info(f"  operatori: {self._operator_mode}")
        if self._mol_on:
            self.get_logger().info(
                "  moltiplicatore dei limiti: " + ", ".join(
                    f"{f} x{k:g}" for f, k in self.generator.moltiplicatori_attivi().items()))
        if self._struct_log is not None:
            self.get_logger().info(f"  registro delle zone: {self._struct_log.path}")
        if self._salva is not None:
            self.get_logger().info(f"  nuvole salvate: prime {self._salva.k_max} per genoma "
                                   f"in {self._salva.cartella}")

    # ------------------------------------------------------------------
    # Genome handling
    # ------------------------------------------------------------------

    def _set_genome(self, genome, source: str):
        self.genome = np.asarray(genome, dtype=np.float64)
        self.params = self.generator.encode_perturbation(self.genome)
        if getattr(self, "_zone_on", False):
            self._genoma_sha["gemello"] = za.genoma_sha(self.genome)
        self.base_seed = (genome_seed(self.genome) + int(getattr(self, '_test_seed', 0))) % (2**31 - 1)
        # with --measures thread, under the measures lock: a measurement of the
        # previous genome is never assigned after the reset
        lock = getattr(self, "_meas_lock", None)
        with (lock if lock is not None else contextlib.nullcontext()):
            self.genome_id += 1
            self._chamfer = float("nan")
            self._perc_terms = None
            if getattr(self, "_cpf", False):
                self._gim_acc["gemello"] = self._gim_acc_vuoto(self.genome_id)
        # Per-cloud seed is base_seed + frame_idx: with a cumulative counter the
        # same genome applied at different frames would produce different clouds
        # (other points removed, other noise, other ghosts) and the deviation
        # measured in evaluation would not recur when applied. Resetting the
        # counter makes evaluation and application of the same genome produce the
        # same cloud sequence for the same input, so fitness is a function of the
        # genome.
        self.frame_idx = 0
        # --stato-generatore: with an armed slot and the same genome the twin takes
        # the saved state of a stream (index, drift, tracked zones, ramp) instead
        # of the one below; takes precedence over --banco-twin-drift and keep.
        caricato = getattr(self, "_stato_gen_on", False) and self._stato_gen_carica()
        # Temporal drift accumulates a bias frame after frame: it is per-genome
        # state and must not leak into the next perturbation. With keep the twin
        # instead starts from the victim's drift, frozen during the search (stream
        # closed): every candidate starts from the state the victim will resume
        # from. With --banco-twin-drift and a received value, it starts from that.
        if caricato:
            pass
        elif getattr(self, "_twin_drift_init", None) is not None and hasattr(self.generator, "set_temporal_state"):
            self.generator.set_temporal_state(self._twin_drift_init)
        elif getattr(self, "_drift_keep", False) and hasattr(self.generator, "set_temporal_state"):
            self.generator.set_temporal_state(self.victim_generator.accumulated_drift)
        elif hasattr(self.generator, "reset_temporal_state"):
            self.generator.reset_temporal_state()
        # Ramp or step: previous genome received for this genome, before the reset
        if not caricato and (getattr(self, "_rampa_R", 1) > 1 or getattr(self, "_gim", None)):
            self._rampa_arma("gemello", self.generator, self.genome)
        # insieme mode: one metre per genome, zones chosen on the first cloud after it.
        if not caricato and hasattr(self.generator, "reset_structures"):
            self.generator.reset_structures()
        p = self.params
        self.get_logger().info(
            f"[genoma #{self.genome_id} da {source}]  "
            f"noise {p['noise_intensity']*100:.2f} cm  "
            f"dropout {p['dropout_rate']*100:.1f}%  "
            f"ghost {p['ghost_ratio']*100:.1f}%  "
            f"geom {p.get('geometric_distortion', 0):.3f}  "
            f"edge {p.get('edge_attack_strength', 0):.2f}  "
            f"drift {p.get('temporal_drift_strength', 0):.2f}"
        )

    def _genome_cb(self, msg: Float32MultiArray):
        g = np.array(msg.data, dtype=np.float64)
        expected = self.generator.get_genome_size()
        if len(g) != expected:
            self.get_logger().error(
                f"Genoma di lunghezza {len(g)}, attesi {expected}. Ignorato.")
            return
        self._set_genome(g, source="/attack/genome")

    def _enabled_cb(self, msg: Bool):
        if msg.data != self.enabled:
            self.enabled = msg.data
            self._frames_since_enable = 0
            self.get_logger().info(
                f"Attacco {'ATTIVATO' if self.enabled else 'DISATTIVATO'}")

    def _drift_reset_cb(self, _msg: Empty):
        if hasattr(self.generator, "reset_temporal_state"):
            self.generator.reset_temporal_state()

    # Continuous victim stream.
    def _victim_genome_cb(self, msg: Float32MultiArray):
        g = np.array(msg.data, dtype=np.float64)
        if len(g) != self.victim_generator.get_genome_size():
            self.get_logger().error("Genoma vittima di lunghezza errata. Ignorato.")
            return
        self.victim_params = self.victim_generator.encode_perturbation(g)
        self.victim_seed = (genome_seed(g) + int(getattr(self, '_test_seed', 0))) % (2**31 - 1)
        if getattr(self, "_zone_on", False):
            self._genoma_sha["vittima"] = za.genoma_sha(g)
        if getattr(self, "_stato_gen_on", False):
            self._stato_gen_sha_vittima = za.genoma_sha(g)
        self.victim_genome_id += 1
        self.victim_frame_idx = 0
        if getattr(self, "_cpf", False):
            lock = getattr(self, "_meas_lock", None)
            with (lock if lock is not None else contextlib.nullcontext()):
                self._gim_acc["vittima"] = self._gim_acc_vuoto(self.victim_genome_id)
        # With keep the victim drift carries over across genomes and windows.
        if not getattr(self, "_drift_keep", False):
            self.victim_generator.reset_temporal_state()
        if getattr(self, "_rampa_R", 1) > 1 or getattr(self, "_gim", None):
            self._rampa_arma("vittima", self.victim_generator, g)
        if hasattr(self.victim_generator, "reset_structures"):
            self.victim_generator.reset_structures()
        self.get_logger().info(
            f"[vittima: genoma #{self.victim_genome_id}]  drift "
            f"{self.victim_params['temporal_drift_strength']:.2f}")

    def _twin_drift_init_cb(self, msg: Float32MultiArray):
        """--banco-twin-drift: 3 values (m, sensor frame) = drift the twin starts
        from at the next genome; empty = reset."""
        v = np.asarray(msg.data, dtype=np.float64)
        if len(v) == 0:
            self._twin_drift_init = None
        elif len(v) == 3 and np.all(np.isfinite(v)):
            self._twin_drift_init = v
        else:
            self.get_logger().warn(f"deriva iniziale del gemello non valida ({len(v)} valori): ignorata")
            return
        self.get_logger().info(f"Deriva iniziale del gemello: {self._twin_drift_init}")

    def _drift_keep_cb(self, msg: Bool):
        """Drift keep mode (orchestrator --drift-reset-on keep)."""
        self._drift_keep = bool(msg.data)
        for g in (self.generator, self.victim_generator):
            g.keep_drift = self._drift_keep
        self.get_logger().info(f"Deriva conservata fra genomi: {'SI' if self._drift_keep else 'NO'}")

    def _salva_tag_cb(self, msg: String):
        """--salva-nuvole: file prefix from the next cloud on."""
        self._salva.tag = msg.data
        self.get_logger().info(f"Tag delle nuvole salvate: {salva_nuvole_tag(msg.data)}")

    def _metro_cb(self, msg: String):
        """insieme mode: metre index per stream, JSON {"gemello": k} and/or
        {"vittima": k}. Applies from the first cloud after the next genome."""
        try:
            d = json.loads(msg.data)
            nuovi = {s: int(d[s]) for s in ("gemello", "vittima") if d.get(s) is not None}
        except (ValueError, TypeError, AttributeError) as e:
            self.get_logger().warn(f"metro non valido: {e}")
            return
        self._metro.update(nuovi)

    def _zone_fornitore(self, flusso, seme, metro, params):
        """Zones received for the stream, if valid for the current genome (genome,
        seed, metre and decoded parameters): (zones, "ok") or (None, reason)."""
        spec = self._zone_spec.get(flusso)
        if spec is None:
            return None, "assenti"
        if spec["stato"] != "ok":
            return None, spec["stato"]
        z = spec["zone"]
        if z.genoma_sha != self._genoma_sha.get(flusso):
            return None, "genoma_diverso"
        if seme is None or int(seme) != z.seme:
            return None, "seme_diverso"
        if metro is not None and int(metro) != z.metro:
            return None, "metro_diverso"
        if za.firma_parametri(params) != z.firma:
            return None, "parametri_diversi"
        if not moltiplicatori_uguali(z, self.generator):
            return None, "moltiplicatori_diversi"
        return z, "ok"

    def _zone_cb(self, msg: String):
        """Attacker zones for a stream: JSON {seq, flusso, path, sha256, hash,
        genoma_sha}. The file is read and checked here, outside the generator
        lock (50-53 ms without the snapshot cached), so clouds do not wait. Zones
        and receipt are installed together under the lock, receipt last (the
        orchestrator sends the genome only after seeing it); if another message
        for the same stream arrived meanwhile, that one wins. They apply from the
        first cloud after the genome."""
        try:
            d = json.loads(msg.data)
            flusso, seq = d["flusso"], int(d["seq"])
            if flusso not in ("gemello", "vittima"):
                raise ValueError(f"flusso {flusso}")
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f"zone non valide: {e}")
            return
        if (self._zone_ric.get(flusso) or {}).get("seq") == seq:
            return
        with self._zone_arrivi_lock:
            self._zone_arrivi[flusso] += 1
            arrivo = self._zone_arrivi[flusso]
        z = None
        try:
            with self._foto_lock:
                z = za.carica_zone(d["path"], sha256=d.get("sha256"), hash_atteso=d.get("hash"),
                                   cache=self._foto_cache)
            stato = "ok" if z.genoma_sha == d.get("genoma_sha") else "genoma_diverso"
            if stato == "ok" and not moltiplicatori_uguali(z, self.generator):
                # zones computed with a different limit multiplier
                stato = "moltiplicatori_diversi"
        except za.ZoneNonValide as e:
            stato = e.stato
            self.get_logger().warn(f"zone {flusso} non valide: {e}")
        except Exception as e:
            stato = "illeggibile"
            self.get_logger().warn(f"zone {flusso} illeggibili: {e}")
        with self._gen_lock:
            if self._zone_arrivi[flusso] != arrivo:
                return
            self._zone_spec[flusso] = {"zone": z if stato == "ok" else None, "stato": stato, "seq": seq}
            self._zone_ric[flusso] = {"seq": seq, "stato": stato,
                                      "hash": z.hash if z is not None else None,
                                      "genoma_sha": d.get("genoma_sha")}

    @staticmethod
    def _rampa_ricevuta(seq, stato, motivo=None):
        """Receipt for a stream without previous-genome data (cancel or malformed
        message)."""
        r = {"seq": seq, "stato": stato, "genoma_prec_sha": None, "genoma_sha_nuovo": None,
             "zone_prec_hash": None, "prec": None}
        if motivo is not None:
            r["motivo"] = motivo
        return r

    def _rampa_cb(self, msg: String):
        """Ramp (--rampa-genoma): previous genome P of a stream.

        JSON {seq, flusso, genoma_prec (17 values), genoma_sha_nuovo (sha of the
        genome N it applies to), zone_prec: {path, sha256, hash}}, sent before N.
        The zone file of P (snapshot of N's metre) is read and checked here, off
        the cloud thread, with genome, seed and parameters as if P had arrived on
        /attack/genome or /attack/victim_genome. {seq, flusso, annulla: true}
        clears the stream's data: later genomes get no ramp (prec_annullata) until
        the next valid message. A malformed message sets state "malformato" and
        the reason in the stream receipt, or in rampa.malformato if the stream is
        unreadable. As in _zone_cb the file is loaded outside the generator lock;
        the receipt is written under the lock, and a newer message for the same
        stream wins."""
        flusso, seq = None, None
        try:
            d = json.loads(msg.data)
            if not isinstance(d, dict):
                raise ValueError("non e' un oggetto JSON")
            if d.get("flusso") in ("gemello", "vittima"):
                flusso = d["flusso"]
            seq = int(d["seq"])
            if flusso is None:
                raise ValueError(f"flusso {d.get('flusso')!r}")
            if (self._rampa_ric.get(flusso) or {}).get("seq") == seq:
                return
            arrivo = self._rampa_conta(flusso)
            if d.get("annulla") is True:
                self._rampa_scrivi(flusso, arrivo, self._rampa_ricevuta(seq, "annullata"))
                self.get_logger().info(f"rampa {flusso}: dati del genoma precedente annullati (seq {seq})")
                return
            g = za.genoma_f32(d["genoma_prec"])
            if g.shape != (self.generator.get_genome_size(),) or not np.all(np.isfinite(g)):
                raise ValueError(f"genoma_prec di forma {g.shape}")
            sha_nuovo = str(d["genoma_sha_nuovo"])
            zp = dict(d["zone_prec"])
            zpath = str(zp["path"])
        except (ValueError, KeyError, TypeError) as e:
            motivo = f"{type(e).__name__}: {e}"
            self.get_logger().warn(f"rampa non valida: {motivo}")
            if flusso is not None:
                self._rampa_scrivi(flusso, self._rampa_conta(flusso), self._rampa_ricevuta(seq, "malformato", motivo))
            else:
                with self._gen_lock:
                    self._rampa_malformato = {"seq": seq, "motivo": motivo}
            return
        gen = self.generator if flusso == "gemello" else self.victim_generator
        params = gen.encode_perturbation(g)
        seme = (genome_seed(g) + int(getattr(self, "_test_seed", 0))) % (2**31 - 1)
        sha = za.genoma_sha(g)
        z, prec = None, None
        try:
            with self._foto_lock:
                z = za.carica_zone(zpath, sha256=zp.get("sha256"), hash_atteso=zp.get("hash"),
                                   cache=self._foto_cache)
            if z.genoma_sha != sha:
                stato = "genoma_diverso"
            elif z.seme != seme:
                stato = "seme_diverso"
            elif za.firma_parametri(params) != z.firma:
                stato = "parametri_diversi"
            elif not moltiplicatori_uguali(z, gen):
                stato = "moltiplicatori_diversi"
            else:
                stato = "ok"
                prec = {"params": params, "seme": seme, "zone": z, "sha": sha}
        except za.ZoneNonValide as e:
            stato = e.stato
            self.get_logger().warn(f"rampa {flusso}: zone del genoma precedente non valide: {e}")
        except Exception as e:
            stato = "illeggibile"
            self.get_logger().warn(f"rampa {flusso}: zone del genoma precedente illeggibili: {e}")
        self._rampa_scrivi(flusso, arrivo, {"seq": seq, "stato": stato, "genoma_prec_sha": sha,
                                            "genoma_sha_nuovo": sha_nuovo,
                                            "zone_prec_hash": z.hash if z is not None else None, "prec": prec})

    def _rampa_conta(self, flusso):
        """Ramp message received for the stream: returns its arrival number."""
        with self._rampa_arrivi_lock:
            self._rampa_arrivi[flusso] += 1
            return self._rampa_arrivi[flusso]

    def _rampa_scrivi(self, flusso, arrivo, ricevuta):
        """Writes the ramp receipt under the generator lock, unless another message
        for the same stream arrived meanwhile."""
        with self._gen_lock:
            if self._rampa_arrivi[flusso] == arrivo:
                self._rampa_ric[flusso] = ricevuta

    def _rampa_arma(self, flusso, gen, genoma):
        """On the stream's new genome, before the zone reset: ramp from the previous
        genome received for this genome (sha), or no ramp, only N and the reason
        in the status. Same function for victim and twin."""
        sha = za.genoma_sha(genoma)
        ric = self._rampa_ric.get(flusso)
        prec, stato = None, None
        if ric is None:
            stato = "prec_assente"
        elif ric["stato"] in ("annullata", "malformato"):
            stato = "prec_" + ric["stato"]
        elif ric["genoma_sha_nuovo"] != sha:
            stato = "genoma_nuovo_diverso"
        elif ric["stato"] != "ok":
            stato = "prec_" + ric["stato"]
        elif ric["genoma_prec_sha"] == sha:
            stato = "prec_uguale_nuovo"
        else:
            prec = ric["prec"]
        gim = getattr(self, "_gim", None)
        if gim:
            # genome in motion: no P on metre 0 (the victim MOLA starts with N)
            if self._metro.get(flusso) == 0:
                prec, stato = None, "metro0"
            gen.imposta_gradino(gim["S_cm"], gim["kmin"], gim["kmax"], prec, stato, genoma_sha=sha,
                                seq=None if ric is None else ric["seq"])
            if prec is None and stato != "metro0":
                self.get_logger().warn(f"genoma in moto {flusso}: nessun genoma precedente per {sha} ({stato}), "
                                       f"solo il genoma nuovo")
            return
        gen.imposta_rampa(self._rampa_R, prec, stato, genoma_sha=sha,
                          seq=None if ric is None else ric["seq"])
        if prec is None:
            self.get_logger().warn(f"rampa {flusso}: nessuna rampa per il genoma {sha} ({stato}), "
                                   f"solo il genoma nuovo")

    @staticmethod
    def _gim_acc_vuoto(gid):
        """Per-phase Chamfer accumulator of the stream's genome gid: sum and count."""
        return {"genome_id": gid, "P": [0.0, 0], "N": [0.0, 0]}

    def _cpf_stato(self):
        """Per-phase Chamfer status (--chamfer-per-fase): per stream, current genome
        and sum and count of the values computed in phases P and N."""
        return {f: {"genome_id": acc["genome_id"], "chamfer_P_somma": acc["P"][0], "n_chamfer_P": acc["P"][1],
                    "chamfer_N_somma": acc["N"][0], "n_chamfer_N": acc["N"][1]}
                for f, acc in self._gim_acc.items()}

    def _gim_fase(self, gen):
        """Phase P or N of the cloud just processed by the generator (Chamfer)."""
        sg = gen.stato_gradino()
        return "P" if (sg or {}).get("fase") == "P" else "N"

    def _gim_chamfer(self, flusso, gid, fase, valore):
        """Chamfer computed on a stream cloud in the given phase: added only if the
        stream's genome is still the one of that cloud."""
        acc = self._gim_acc[flusso]
        if acc["genome_id"] != gid or valore != valore:
            return
        acc[fase][0] += float(valore)
        acc[fase][1] += 1

    def _gim_stato(self):
        """Genome-in-motion status: threshold and limits, previous-genome data
        received per stream (as for the ramp) and, per stream, step state of the
        last processed cloud and per-phase Chamfer of the current genome."""
        st = dict(self._gim)
        st["ricevuta"] = {f: (None if r is None else
                              {k: r[k] for k in ("seq", "stato", "genoma_prec_sha", "genoma_sha_nuovo",
                                                 "zone_prec_hash", "motivo") if k in r})
                          for f, r in self._rampa_ric.items()}
        st["malformato"] = self._rampa_malformato
        for f, gen in (("gemello", self.generator), ("vittima", self.victim_generator)):
            sg = gen.stato_gradino()
            acc = self._gim_acc[f]
            st[f] = dict(sg or {}, genome_id=acc["genome_id"],
                         chamfer_P_somma=acc["P"][0], n_chamfer_P=acc["P"][1],
                         chamfer_N_somma=acc["N"][0], n_chamfer_N=acc["N"][1])
        return st

    def _rampa_stato(self):
        """Ramp status (only with --rampa-genoma): R, data received per stream and,
        per stream, genome, previous genome, state, k and alpha of the last
        processed cloud (alpha None = no ramp in the metre)."""
        st = {"R": self._rampa_R,
              "ricevuta": {f: (None if r is None else
                               {k: r[k] for k in ("seq", "stato", "genoma_prec_sha", "genoma_sha_nuovo",
                                                  "zone_prec_hash", "motivo") if k in r})
                           for f, r in self._rampa_ric.items()},
              "malformato": self._rampa_malformato}
        for f, gen in (("gemello", self.generator), ("vittima", self.victim_generator)):
            ri = getattr(gen, "rampa_info", None)
            st[f] = (None if ri is None else
                     {k: ri.get(k) for k in ("genoma_sha", "genoma_prec_sha", "stato", "k", "alpha", "seq")})
        return st

    # Generator state of a stream (--stato-generatore)

    def _sotto_lock(self, f):
        """Wraps a message callback under the generator lock."""
        def cb(msg):
            with self._gen_lock:
                return f(msg)
        return cb

    def _con_lock_generatore(self, f, *a):
        """Runs the cloud callback under the generator-state lock."""
        with self._gen_lock:
            return f(*a)

    def _stato_gen_cb(self, msg: String):
        """--stato-generatore: JSON {seq, azione, nome, flusso}. "salva" (flusso
        "vittima" or "gemello") copies the stream's generator state into slot nome,
        between two clouds; "carica" arms the slot for the twin's next genome
        (_stato_gen_carica). The status confirms seq only for an executed action;
        otherwise the reason goes into ultimo_errore."""
        try:
            d = json.loads(msg.data)
            seq, azione, nome = int(d["seq"]), str(d["azione"]), str(d["nome"])
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f"stato del generatore: comando non valido: {e}")
            return
        sg = self._stato_gen
        if seq == sg["seq"]:
            return
        motivo = None
        if azione == "salva":
            flusso = d.get("flusso")
            if flusso not in ("gemello", "vittima"):
                motivo = "flusso"
            else:
                with self._gen_lock:
                    posto = self._stato_gen_posto(flusso)
                sg["posti"][nome] = posto
        elif azione == "carica":
            if nome not in sg["posti"]:
                motivo = "posto_assente"
            else:
                sg["armato"], sg["armato_seq"] = nome, seq
        else:
            motivo = "azione"
        if motivo is not None:
            sg["ultimo_errore"] = {"seq": seq, "azione": azione, "nome": nome, "motivo": motivo}
            self.get_logger().warn(f"stato del generatore: {azione} {nome} non eseguito ({motivo})")
            return
        sg["seq"] = seq
        self.get_logger().info(f"stato del generatore: {azione} {nome} (seq {seq})")

    def _stato_gen_posto(self, flusso):
        """Slot with the stream state, built under the lock: cloud index since the
        genome, metre, genome and seed of the stream, copy of the generator state
        (drift, tracker with k, anchors and tracked zones, installed zones, ramp)
        and its fingerprint."""
        gemello = flusso == "gemello"
        gen = self.generator if gemello else self.victim_generator
        posto = {"flusso": flusso,
                 "frame_idx": int(self.frame_idx if gemello else self.victim_frame_idx),
                 "metro": self._metro.get(flusso),
                 "genoma_sha": za.genoma_sha(self.genome) if gemello else self._stato_gen_sha_vittima,
                 "seme": int(self.base_seed if gemello else self.victim_seed),
                 "gen": gen.stato_flusso()}
        posto["impronta"] = impronta_posto(posto)
        return posto

    def _stato_gen_carica(self):
        """Called in the twin's _set_genome after genome and seed: with an armed slot
        (consumed here) and the same genome and seed, loads cloud index, metre and
        generator state from the slot instead of resetting; True if loaded.
        Different genome or seed: nothing loaded, outcome in the status."""
        sg = self._stato_gen
        nome = sg["armato"]
        if nome is None:
            return False
        sg["armato"] = None
        posto = sg["posti"][nome]
        rec = {"nome": nome, "seq": sg["armato_seq"], "genome_id": self.genome_id, "impronta": None}
        sha = za.genoma_sha(self.genome)
        if sha != posto["genoma_sha"]:
            rec["esito"] = "genoma_diverso"
        elif int(self.base_seed) != posto["seme"]:
            rec["esito"] = "seme_diverso"
        else:
            with self._gen_lock:
                self.generator.imposta_stato_flusso(posto["gen"])
                self.frame_idx = int(posto["frame_idx"])
                self._metro["gemello"] = posto["metro"]
                # fingerprint of the loaded state, before resetting the motion reference
                # (the reset only changes the step's "riprendi" flag)
                rec["impronta"] = impronta_posto({"frame_idx": self.frame_idx, "metro": self._metro["gemello"],
                                                  "genoma_sha": sha, "seme": int(self.base_seed),
                                                  "gen": self.generator.stato_flusso(copia=False)})
                if getattr(self, "_gim", None):
                    # genome in motion: motion reference on the first cloud after loading
                    self.generator.riprendi_riferimento_moto()
            rec["esito"] = "ok"
        sg["ultimo_caricato"] = rec
        if rec["esito"] == "ok":
            self.get_logger().info(f"stato del generatore: posto {nome} caricato nel gemello "
                                   f"(frame_idx {self.frame_idx})")
        else:
            self.get_logger().warn(f"stato del generatore: posto {nome} non caricato ({rec['esito']})")
        return rec["esito"] == "ok"

    def _stato_gen_status(self):
        """stato_generatore key of the status (only with the option)."""
        sg = self._stato_gen
        posti = {}
        for nome, p in list(sg["posti"].items()):
            g = p["gen"]
            tr = g.get("_structs")
            posti[nome] = {"flusso": p["flusso"], "frame_idx": p["frame_idx"],
                           "k_strutture": None if tr is None else int(tr.k),
                           "drift": [float(x) for x in g["_accumulated_drift"]],
                           "genoma_sha": p["genoma_sha"], "metro": p["metro"], "impronta": p["impronta"]}
            if getattr(self, "_gim", None):
                gr = g.get("_gradino") or {}
                posti[nome]["gradino_fase"] = gr.get("fase")
                posti[nome]["gradino_k_ingresso"] = gr.get("k_ingresso")
        return {"seq": sg["seq"], "posti": posti, "armato": sg["armato"],
                "ultimo_caricato": sg["ultimo_caricato"], "ultimo_errore": sg["ultimo_errore"]}

    def _foto_cb(self, msg: String):
        """Attacker snapshot: JSON {seq, stamp, path, meta}. Saves the clean cloud
        sent to the attacker's MOLA with that stamp; no zone choice here."""
        try:
            d = json.loads(msg.data)
            seq, stamp, path = int(d["seq"]), float(d["stamp"]), str(d["path"])
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f"fotografia non valida: {e}")
            return
        if (self._foto_stato or {}).get("seq") == seq:
            return
        buf = list(self._att_buf)
        trov = [(i, s_, c) for i, (s_, c) in enumerate(buf) if abs(s_ - stamp) <= za.STAMP_TOL_S]
        rec = {"seq": seq, "path": path, "stamp": stamp, "sha256": None, "hash": None}
        if not trov:
            rec["esito"] = "nuvola_assente"
            rec["stamp_buffer"] = [buf[0][0], buf[-1][0]] if buf else None
        else:
            i, s_, c = trov[-1]
            meta = dict(d.get("meta") or {})
            meta.update({"stamp_nuvola": s_, "nuvole_dopo": len(buf) - 1 - i})
            try:
                rec["sha256"], rec["hash"] = za.salva_fotografia(path, c, meta)
                rec["esito"] = "ok"
            except FileExistsError:
                rec["esito"] = "file_esistente"
            except Exception as e:
                rec["esito"] = f"errore: {e}"
        self._foto_stato = rec
        self.get_logger().info(f"fotografia {seq}: {rec['esito']} ({path})")

    def _zone_conta(self, flusso, gen):
        """Counts stream clouds processed with or without valid zones (status)."""
        a = self._zone_app[flusso]
        a["stato"], a["hash"] = gen.zone_stato, gen.zone_hash
        a["n_ok" if gen.zone_stato == "ok" else "n_err"] += 1

    def _victim_stream_cb(self, msg: Bool):
        if msg.data != self.victim_stream:
            if msg.data and getattr(self, "_gim", None):
                # genome in motion: motion reference on the first cloud after reopening
                with self._gen_lock:
                    self.victim_generator.riprendi_riferimento_moto()
            self.victim_stream = msg.data
            if self.victim_stream:
                self._twin_clock_on = False
            self.get_logger().info(
                f"Flusso vittima {'ATTIVO' if self.victim_stream else 'FERMO'}")

    def _twin_clock_base_cb(self, msg):
        """Arms the twin clock from the base sent by the orchestrator (JSON {"tick",
        "seq"}, mrpt::Clock tick of the state saved by the shadow). The first twin
        header is base + 0.1 s, in integer ns that MOLA converts exactly to
        tick + 1000000; later ones follow the simulation dt. The same seq does
        not re-arm."""
        from src.optimization.twin_variants import twin_clock_first_ns
        try:
            d = json.loads(msg.data)
            seq = int(d["seq"])
            first = twin_clock_first_ns(int(d["tick"]))
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f"base dell'orologio del gemello non valida: {e}")
            return
        if seq == self._twin_clock_seq:
            return
        self._twin_first_ns = first
        self._twin_last_sim = None
        self._twin_last_clock = first / 1e9
        self._twin_clock_first_out_ns = None
        self._twin_clock_on = True
        self._twin_clock_seq = seq

    def _fresh_clock_cb(self, msg):
        self._fresh_last_clock = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _fresh_arm_cb(self, msg):
        """Arms the filter with the simulation stopped. The command is repeated
        until the status confirms the sequence number: a repeat of the same number
        (even one arriving after play) does not re-arm."""
        seq = int(msg.data)
        with self._fresh_lock:
            if seq == self._fresh_seq:
                return
            cand = [(v, src) for v, src in ((self._fresh_last_clock, "clock"),
                                            (self._fresh_last_stamp_in, "stamp_in")) if v is not None]
            base = src = thr = None
            if cand:
                base, src = max(cand)
                self._fresh_gate.arm(base, src)
                thr = self._fresh_gate.threshold
            self._fresh_seq = seq
        flog = getattr(self, "_frame_log", None)
        if flog is not None:
            # one frame-log row per arm, without cloud: base, source and threshold
            flog.add([time.time(), time.monotonic(), None] + [None] * (len(FrameLog.COLUMNS) - 3)
                     + ["armato" if cand else "non_armato", thr, base, src, seq])
        if cand:
            self.get_logger().info(
                f"Filtro dopo il ripristino armato: soglia {self._fresh_gate.threshold:.4f} "
                f"({src} {base:.4f})")
        else:
            self.get_logger().warn("Filtro dopo il ripristino non armato: nessun /clock e nessuna nuvola ricevuti")

    def _victim_attack_cb(self, msg: Bool):
        if msg.data != self.victim_attack:
            self.victim_attack = msg.data
            self.get_logger().info(
                f"Attacco sulla vittima {'ATTIVATO' if self.victim_attack else 'DISATTIVATO'}")

    def _victim_drift_reset_cb(self, _msg: Empty):
        self.victim_generator.reset_temporal_state()

    def _twin_replay_cb(self, msg: String):
        """Replays the clouds recorded for the victim to the twin (main output):
        all of them, or those of the last last_s seconds of clock."""
        try:
            req = json.loads(msg.data) if msg.data else {}
        except ValueError:
            req = {}
        if self._replay_active:
            self.get_logger().warn("Rigioco gia' in corso: comando ignorato")
            return
        last_s = float(req.get("last_s", 0.0) or 0.0)
        rate_hz = float(req.get("rate_hz", 10.0) or 10.0)
        rec = list(self._victim_rec)
        if last_s > 0.0 and rec:
            t_end = rec[-1][0]
            rec = [r for r in rec if r[0] >= t_end - last_s]
        self._replay_active = True
        self._replay_total = len(rec)
        self._replay_n = 0
        self._replay_first_stamp = rec[0][0] if rec else None
        self.get_logger().info(
            f"Rigioco al twin: {len(rec)} nuvole (ultimi {last_s:.0f} s) a {rate_hz:.0f} Hz")
        threading.Thread(target=self._replay_run, args=(rec, rate_hz), daemon=True).start()

    def _replay_run(self, rec, rate_hz):
        from std_msgs.msg import Header
        period = 1.0 / max(rate_hz, 0.5)
        last = None
        for stamp, frame_id, arr in rec:
            t0 = time.perf_counter()
            h = Header()
            h.frame_id = frame_id
            h.stamp.sec = int(stamp)
            h.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
            try:
                self._publish_cloud(arr, h)
            except Exception as e:
                self.get_logger().warn(f"rigioco interrotto: {e}")
                break
            last = stamp
            self._replay_n += 1
            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        if last is not None:
            self._twin_last_sim = None
            self._twin_last_clock = last
            self._twin_clock_on = True
            self._replay_last_stamp = last
        # Live clouds that arrived during replay were not published: the twin's
        # metre starts from the first cloud after replay.
        if hasattr(self.generator, "reset_structures"):
            self.generator.reset_structures()
        self._replay_active = False
        self._replay_seq += 1

    # ------------------------------------------------------------------
    # Cloud processing
    # ------------------------------------------------------------------

    def _cloud_cb_strumenti(self, msg, info):
        """Cloud callback with delay instrumentation: start and end times and a
        watchdog around _cloud_cb, which marks the phases."""
        st = self._strum
        r = st.inizio_callback(info, msg, self.frame_idx)
        self._strum_rec = r
        try:
            self._cloud_cb(msg, info)
        except Exception:
            r.dati["esito"] = "errore"
            raise
        finally:
            self._strum_rec = None
            st.fine_callback(r, "accodata" in r.t)

    def _cloud_cb(self, msg: PointCloud2, info=None):
        t0 = time.perf_counter()
        sn = getattr(self, "_salva", None)       # None without --salva-nuvole
        t_arr_sn = time.monotonic() if sn is not None else None
        sr = getattr(self, "_strum_rec", None)   # None without --strumenti-ritardo
        # frame log: arrival (wall and monotonic) and input stamp
        flog = ([time.time(), time.monotonic(), _stamp_s(msg.header)]
                if self._frame_log is not None else None)
        if flog is not None and getattr(self, "_delay_log", False):
            flog = _FlogRow(flog)
            flog.t_shadow_end = None
            flog.dds = ((info.get("received_timestamp", 0) * 1e-9 or None,
                         info.get("source_timestamp", 0) * 1e-9 or None)
                        if isinstance(info, dict) else (None, None))
        self.n_in += 1
        if getattr(self, "_fresh_on", False):
            stamp_in = _stamp_s(msg.header)
            with self._fresh_lock:
                thr = self._fresh_gate.threshold
                fbase, fsrc, fseq = self._fresh_gate.base, self._fresh_gate.base_src, self._fresh_seq
                keep, ev = self._fresh_gate.check(stamp_in)
                self._fresh_last_stamp_in = stamp_in
            if not keep:
                # no output, frame counters frozen: the cloud does not exist for victim and twin
                if flog is not None:
                    self._frame_log.add(flog + [self.frame_idx, None, self.enabled, self.victim_stream,
                                                self.victim_attack, msg.width * msg.height]
                                        + [None] * (len(FrameLog.COLUMNS) - 9)
                                        + ["scartata_ripristino", thr, fbase, fsrc, fseq])
                self.get_logger().info(
                    f"Nuvola scartata dopo il ripristino: stamp {stamp_in:.4f} <= soglia {thr:.4f}")
                return
            if ev == "tetto_scarti":
                self.get_logger().warn(
                    f"Filtro dopo il ripristino disarmato dopo {self._fresh_gate.max_drop} scarti: "
                    f"stamp {stamp_in:.4f}, soglia {thr:.4f} (orologio ripartito?)")
            elif ev == "prima_fresca":
                self.get_logger().info(
                    f"Prima nuvola dopo il ripristino: stamp {stamp_in:.4f} > soglia {thr:.4f} "
                    f"({self._fresh_gate.dropped} scartate)")

        if sr is not None and getattr(self, "_fresh_on", False):
            sr.fase("filtro_fine")
        arr = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        if arr is None or len(arr) == 0:
            if flog is not None:
                self._frame_log.add(flog + [self.frame_idx, None, self.enabled, self.victim_stream,
                                            self.victim_attack, 0])
            return

        # Structured array: columns extracted by name, no per-point loop.
        cloud = np.empty((len(arr), 4), dtype=np.float64)
        cloud[:, 0] = arr["x"]
        cloud[:, 1] = arr["y"]
        cloud[:, 2] = arr["z"]
        cloud[:, 3] = arr["intensity"]
        if sr is not None:
            sr.fase("lettura_fine")

        # Attack switches and frame indices are read once, at the start of
        # processing: they decide the perturbation and travel with the cloud to
        # the log. The switches live in a reentrant group and may change during
        # processing; read later, a cloud computed with the attack and published
        # after the attack was switched off would be logged as clean.
        enabled, victim_stream, victim_attack = self.enabled, self.victim_stream, self.victim_attack
        frame_idx, victim_frame_idx = self.frame_idx, self.victim_frame_idx
        sn_ids = ((self.genome_id, self.base_seed, self.victim_genome_id, self.victim_seed)
                  if sn is not None else None)
        metro = getattr(self, "_metro", None) or {}
        metro_g, metro_v = metro.get("gemello"), metro.get("vittima")
        # With --measures thread the cloud's genome travels with its measurements
        # (see _meas_loop); None in inline mode.
        mst = self.genome_id if getattr(self, "_meas_q", None) is not None else None

        victim_out = None
        dropped = False
        victim_pert = False
        gim_tag = None      # --chamfer-per-fase: (stream, genome, phase) of the cloud
        if victim_stream:
            # One perturbation per cloud: the victim's. The main output stays in
            # passthrough (no twin MOLA listens to it during the applied segment;
            # it only feeds the stall check).
            if victim_attack and (self.victim_params is not None or self._test_jitter > 0 or self._test_offset is not None or self._test_sector > 0 or self._test_ghost_wall > 0 or self._test_slow_drift != 0.0):
                victim_pert = True
                if (self._test_jitter > 0 or self._test_offset is not None or self._test_sector > 0
                        or self._test_ghost_wall > 0 or self._test_slow_drift != 0.0):
                    victim_out = cloud.copy()
                    self._frames_since_enable += 1
                    if self._test_offset is not None and self._frames_since_enable > self._test_offset_after:
                        victim_out[:, :3] += self._test_offset
                    if self._test_jitter > 0 and self._frames_since_enable > self._test_offset_after:
                        victim_out[:, 0] += self._test_jitter * (1.0 if self.victim_frame_idx % 2 == 0 else -1.0)
                    victim_out = apply_test_struct(victim_out, self.victim_frame_idx, self._test_sector, self._test_ghost_wall, self._test_slow_drift, self._test_drift_until)
                elif self._gaussian_sigma > 0.0:
                    rng = np.random.default_rng(
                        (self.victim_seed + self.victim_frame_idx) % (2**31 - 1))
                    victim_out = cloud.copy()
                    victim_out[:, :3] += rng.normal(0.0, self._gaussian_sigma, size=(len(cloud), 3))
                elif getattr(self, "_operator_mode", "random") == "insieme":
                    # insieme mode: zones from the genome seed and the metre
                    victim_out = self.victim_generator.apply_perturbation(
                        cloud, self.victim_params, ctx={"stamp": _stamp_s(msg.header)},
                        seme_genoma=self.victim_seed, metro=metro_v)
                    self._struct_row("vittima", self.victim_generator, self.victim_genome_id,
                                     self.victim_frame_idx, msg.header, self.victim_seed, metro_v)
                    if getattr(self, "_zone_on", False):
                        self._zone_conta("vittima", self.victim_generator)
                    if getattr(self, "_cpf", False):
                        gim_tag = ("vittima", self.victim_genome_id, self._gim_fase(self.victim_generator))
                else:
                    victim_out = self.victim_generator.apply_perturbation(
                        cloud, self.victim_params,
                        seed=(self.victim_seed + self.victim_frame_idx) % (2**31 - 1))
                    if getattr(self, "_cpf", False):
                        gim_tag = ("vittima", self.victim_genome_id, "N")
                if sr is not None:
                    sr.fase("perturbazione_fine")
                if (self._chamfer_every > 0
                        and self.victim_frame_idx % self._chamfer_every == 0):
                    if mst is not None:
                        self._perc_measure(self.victim_generator, cloud, victim_out, self.victim_params, mst,
                                           gim=gim_tag)
                    else:
                        try:
                            nn = (self.victim_generator.nn_distances(cloud, victim_out)
                                  if getattr(self, "_shared_nn", False) else None)
                            self._chamfer = self._magnitude(
                                self.victim_generator, cloud, victim_out, self.victim_params, nn=nn)
                            if gim_tag is not None:
                                self._gim_chamfer(*gim_tag, self._chamfer)
                            self._perc = self._perception_metrics(cloud, victim_out, nn=nn)
                        except Exception:
                            pass
            elif getattr(self, "_never_clean", False):
                # no output to victim and shadow, measurements and delivered count
                # frozen; the frame index still advances, so the seeds of the next
                # attacked clouds stay the same
                dropped = True
                self.victim_clean_dropped = getattr(self, "victim_clean_dropped", 0) + 1
                self.victim_frame_idx += 1
            else:
                victim_out = cloud
                self.victim_clean_out = getattr(self, "victim_clean_out", 0) + 1
                if mst is not None:
                    self._perc_zero(mst)
                else:
                    self._chamfer = 0.0
                    self._perc_terms = dict(ZERO_TERMS) if self._perc_terms_on else None
                    self._perc = {"shift_frac": 0.0, "shift_med_cm": 0.0,
                                  "n_added": 0, "n_removed": 0}
            if not dropped:
                if mst is not None:
                    self._tc_measure("victim", cloud, victim_out, self.victim_frame_idx)
                else:
                    self._update_tc("victim", cloud, victim_out, self.victim_frame_idx)
                self.victim_frame_idx += 1
                self.victim_frames += 1
            out = cloud
        elif enabled:
            self._frames_since_enable += 1
            if (self._test_offset is not None or self._test_keep_points > 0 or self._test_double_scan > 0
                    or self._test_jitter > 0 or self._test_sector > 0 or self._test_ghost_wall > 0 or self._test_slow_drift != 0.0):
                out = cloud.copy()
                if self._test_offset is not None and self._frames_since_enable > self._test_offset_after:
                    out[:, :3] += self._test_offset
                if self._test_jitter > 0 and self._frames_since_enable > self._test_offset_after:
                    out[:, 0] += self._test_jitter * (1.0 if self.frame_idx % 2 == 0 else -1.0)
                out = apply_test_struct(out, self.frame_idx, self._test_sector, self._test_ghost_wall, self._test_slow_drift, self._test_drift_until)
                if self._test_keep_points > 0 and len(out) > self._test_keep_points:
                    keep = np.random.default_rng(self.frame_idx).choice(len(out), self._test_keep_points, replace=False)
                    out = out[np.sort(keep)]
                if self._test_double_scan > 0 and self.frame_idx % self._test_double_scan == 0:
                    self._double_pending = True
            elif self._gaussian_sigma > 0.0:
                # Comparison arm: isotropic per-point Gaussian noise, no genome and
                # no scene analysis. Same Chamfer, same per-frame seed.
                rng = np.random.default_rng((self.base_seed + self.frame_idx) % (2**31 - 1))
                out = cloud.copy()
                out[:, :3] += rng.normal(0.0, self._gaussian_sigma, size=(len(cloud), 3))
            elif getattr(self, "_operator_mode", "random") == "insieme":
                out = self.generator.apply_perturbation(
                    cloud, self.params, ctx={"stamp": _stamp_s(msg.header)},
                    seme_genoma=self.base_seed, metro=metro_g)
                self._struct_row("gemello", self.generator, self.genome_id, self.frame_idx,
                                 msg.header, self.base_seed, metro_g)
                if getattr(self, "_zone_on", False):
                    self._zone_conta("gemello", self.generator)
                if getattr(self, "_cpf", False):
                    gim_tag = ("gemello", self.genome_id, self._gim_fase(self.generator))
            else:
                out = self.generator.apply_perturbation(
                    cloud, self.params,
                    # Reproducible for the same (genome, frame), different across frames.
                    seed=(self.base_seed + self.frame_idx) % (2**31 - 1),
                )
                if getattr(self, "_cpf", False):
                    gim_tag = ("gemello", self.genome_id, "N")
            # Perceptibility sampled every chamfer_every frames: two KD-trees per
            # cloud would exceed the 100 ms budget, and the orchestrator only needs
            # a mean over the window.
            if sr is not None:
                sr.fase("perturbazione_fine")
            if (self._chamfer_every > 0
                    and self.frame_idx % self._chamfer_every == 0):
                if mst is not None:
                    self._perc_measure(self.generator, cloud, out, self.params, mst, gim=gim_tag)
                else:
                    try:
                        nn = (self.generator.nn_distances(cloud, out)
                              if getattr(self, "_shared_nn", False) else None)
                        self._chamfer = self._magnitude(self.generator, cloud, out, self.params, nn=nn)
                        if gim_tag is not None:
                            self._gim_chamfer(*gim_tag, self._chamfer)
                        self._perc = self._perception_metrics(cloud, out, nn=nn)
                    except Exception:
                        pass
        else:
            out = cloud
            if mst is not None:
                self._perc_zero(mst)
            else:
                self._chamfer = 0.0
                self._perc_terms = dict(ZERO_TERMS) if self._perc_terms_on else None
                self._perc = {"shift_frac": 0.0, "shift_med_cm": 0.0,
                              "n_added": 0, "n_removed": 0}
            # Artificial delay with the attack off: the nominal run pays the same
            # latency as the candidates, so processing delay does not enter the
            # damage. Take the value from proc_ms_mean in the status. (The legacy
            # --passthrough-delay-ms delay is applied at release.)

        if sr is not None:
            sr.fase("ramo_fine")
        if not self.victim_stream:
            # same inputs threaded and inline: only where it is computed changes
            if mst is not None:
                self._tc_measure("main", cloud, out, self.frame_idx)
            else:
                self._update_tc("main", cloud, out, self.frame_idx)
        if sr is not None:
            sr.fase("coerenza_fine")
        t_proc = time.perf_counter() - t0
        # Same minimum latency for nominal and candidates, whatever the arm: the
        # compute-time difference does not enter the damage. Scans already above
        # it are not shortened. The wait happens in the publishing thread, not
        # here.
        t_release = t0 + max(self._min_latency_ms / 1000.0,
                             self._passthrough_delay if not self.enabled else 0.0)
        # cloud dropped for the victim: continuous header still computed (the
        # attacker uses it), no stamp_victim in the log
        vh = self._victim_header(msg.header) if (victim_out is not None or dropped) else None
        main_h = (self._twin_header(msg.header)
                  if (self._twin_clock_on and victim_out is None and not dropped) else msg.header)
        if flog is not None:
            flog += [frame_idx, victim_frame_idx if victim_out is not None else None,
                     enabled, victim_stream, victim_attack, len(cloud), t_proc * 1000.0,
                     flog[1] + (t_release - t0), _stamp_s(vh) if victim_out is not None else None,
                     _stamp_s(main_h)]
        if sn is not None:
            sc = salva_nuvole_scelta(victim_stream, victim_pert, enabled, dropped,
                                     getattr(self, "_replay_active", False),
                                     frame_idx, victim_frame_idx, sn.k_max)
            if sc is not None:
                fl, k = sc
                if fl == "vittima":
                    rec = sn.record(fl, sn_ids[2], sn_ids[3], metro_v, k, msg.header, vh, t_arr_sn,
                                    msg.width * msg.height, cloud, victim_out)
                else:
                    rec = sn.record(fl, sn_ids[0], sn_ids[1], metro_g, k, msg.header, main_h, t_arr_sn,
                                    msg.width * msg.height, cloud, out)
                sn.accoda(rec)
        self.frame_idx += 1
        self.proc_times.append(t_proc)
        if len(self.proc_times) > 200:
            self.proc_times.pop(0)
        if sr is None:
            self._out_q.put((t_release, t0, out, main_h, victim_out, vh, cloud, flog))
        else:
            # the instrumentation record travels at the end of the tuple to the
            # publishing thread; the row is written when both parts have finished
            sr.parti = 2
            self._out_q.put((t_release, t0, out, main_h, victim_out, vh, cloud, flog, sr))
            sr.fase("accodata")

    def _struct_row(self, stream, gen, gid, fidx, header, seme, metro):
        """Zone-log row for the generator's last cloud."""
        inf = gen.last_struct_info
        if getattr(self, "_struct_log", None) is None or inf is None:
            return
        nh = inf.get("n_hit", {})
        px, py, pyaw = inf.get("pose", (None, None, None))
        self._struct_log.add([
            time.monotonic(), stream, gid, fidx, _stamp_s(header), seme, metro,
            inf.get("per_op"), inf.get("k"), px, py, pyaw, inf.get("icp_pairs"), inf.get("icp_rms_m"),
            nh.get("g5"), nh.get("g6"), nh.get("g10"), nh.get("g13"), nh.get("g16"),
            "|".join(f"{k}:" + " ".join(str(c) for c in v) for k, v in inf.get("anchors", {}).items()),
        ] + ([inf.get("zone_stato"), inf.get("zone_hash"),
               " ".join(f"{k}:{v}" for k, v in sorted((inf.get("zone_attesi") or {}).items())),
               " ".join(f"{k}:{v}" for k, v in sorted((inf.get("zone_scarto") or {}).items()))]
              if getattr(self, "_zone_on", False) else [])
          + (rampa_riga(getattr(gen, "rampa_info", None), nh)
             if getattr(self, "_rampa_R", 1) > 1 else [])
          + (gradino_riga(gen.stato_gradino()) if getattr(self, "_gim", None) else [])
          + (limiti_riga(gen.ultimi_conteggi()) if getattr(self, "_mol_on", False) else []))

    def _pub_loop(self):
        # Minimum spacing between publishes: after a backlog clouds do not go out
        # in a burst. MOLA receives them best-effort and a burst of ~720 KB
        # messages loses UDP fragments, i.e. whole clouds.
        t_last = 0.0
        st = getattr(self, "_strum", None)       # None without --strumenti-ritardo
        while True:
            item = self._out_q.get()
            if len(item) == 8:
                t_release, t0, out, header, victim_out, vh, cloud, flog = item
                sr = None
            else:
                t_release, t0, out, header, victim_out, vh, cloud, flog, sr = item
            if sr is not None:
                st.presa(sr)
            wait = max(t_release, t_last + self._min_spacing) - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            t_last = time.perf_counter()
            if sr is not None:
                sr.fase("rilascio")
            if flog is not None:
                flog += [time.monotonic(), self._replay_active, None, None, None, None]
            try:
                if not self._replay_active:
                    if sr is not None:
                        st.pub_inizio(sr, "main")
                    self._publish_cloud(out, header)
                    if sr is not None:
                        st.pub_fine(sr, "main")
                    if self._double_pending:
                        self._double_pending = False
                        from std_msgs.msg import Header
                        h2 = Header(); h2.frame_id = header.frame_id
                        t2 = header.stamp.sec + header.stamp.nanosec * 1e-9 + 0.05
                        h2.stamp.sec = int(t2); h2.stamp.nanosec = int((t2 - int(t2)) * 1e9)
                        self._publish_cloud(out, h2)
                    if flog is not None:
                        flog[-4] = time.monotonic()
                if victim_out is not None:
                    if sr is not None:
                        st.pub_inizio(sr, "victim")
                    vmsg = self._publish_cloud(victim_out, vh, self.victim_pub)
                    if sr is not None:
                        st.pub_fine(sr, "victim")
                    if flog is not None:
                        flog[-3] = time.monotonic()
                    if sr is not None:
                        st.pub_inizio(sr, "attacker")
                    self._publish_cloud(cloud, vh, self.attacker_pub)
                    if sr is not None:
                        st.pub_fine(sr, "attacker")
                    if flog is not None:
                        flog[-2] = time.monotonic()
                    if getattr(self, "_zone_on", False):
                        self._att_buf.append((_stamp_s(vh), cloud))
                    # shadow after the attacker: the attacker cloud is not delayed
                    smsg, omitted = None, False
                    shadow_pub, crc_log = getattr(self, "shadow_pub", None), getattr(self, "_crc_log", None)
                    if shadow_pub is not None:
                        omitted = self._shadow_slot == self._shadow_omit
                        if not omitted:
                            if sr is not None:
                                st.pub_inizio(sr, "shadow")
                            smsg = self._publish_cloud(victim_out, vh, shadow_pub)
                            if sr is not None:
                                st.pub_fine(sr, "shadow")
                            if isinstance(flog, _FlogRow):
                                flog.t_shadow_end = time.monotonic()
                    self._victim_rec.append((vh.stamp.sec + vh.stamp.nanosec * 1e-9, vh.frame_id,
                                             np.ascontiguousarray(victim_out[:, :4], dtype=np.float32)))
                    self._victim_rec_total += 1
                    if crc_log is not None:
                        # crc after publishing: does not delay the messages
                        cv = cloud_crc(vmsg)
                        cs = (cloud_crc(smsg) if smsg is not None else
                              None if shadow_pub is not None else cv)
                        row = [self._shadow_slot, repr(_stamp_s(vh)), flog[4] if flog is not None else None,
                               len(victim_out), cv, cs, self.shadow_topic or "", omitted, time.time()]
                        if getattr(self, "_report_matched", False):
                            row.append(self.victim_pub.get_subscription_count())
                        crc_log.add(row)
                    if shadow_pub is not None or crc_log is not None:
                        self._shadow_slot += 1
                elif vh is not None:
                    # dropped for victim and shadow (--victim-never-clean): attacker only
                    if sr is not None:
                        st.pub_inizio(sr, "attacker")
                    self._publish_cloud(cloud, vh, self.attacker_pub)
                    if sr is not None:
                        st.pub_fine(sr, "attacker")
                    if flog is not None:
                        flog[-2] = time.monotonic()
                    if getattr(self, "_zone_on", False):
                        self._att_buf.append((_stamp_s(vh), cloud))
                if flog is not None:
                    flog[-1] = time.time()
                    self._frame_log.add(self._delay_row(flog) if isinstance(flog, _FlogRow) else flog)
                if sr is not None:
                    sr.fase("registri_fine")
                    st.fine_pubblicazione(sr)
            except Exception as e:        # ROS context closed during shutdown
                if flog is not None:
                    flog[-1] = time.time()   # row still written, with the times reached
                    self._frame_log.add(flog)
                if sr is not None:
                    st.fine_pubblicazione(sr, "errore")
                self.get_logger().warn(f"pubblicazione fallita: {e}")
                return
            self.n_out += 1
            self.lat_times.append(time.perf_counter() - t0)
            if len(self.lat_times) > 200:
                self.lat_times.pop(0)

    def _delay_row(self, flog):
        """Frame-log row with the effective-delay columns appended (only with
        --frame-log-delay; otherwise the row is unchanged)."""
        if not isinstance(flog, _FlogRow):
            return flog
        C = FrameLog.COLUMNS
        row = list(flog) + [None] * (len(self._frame_log.columns) - len(DELAY_COLUMNS) - len(flog))
        t_arr, t_rel = flog[C.index("t_arr_mono")], flog[C.index("t_release_target_mono")]
        d_min = (t_rel - t_arr) if (t_arr is not None and t_rel is not None) else None

        def delta(t_end):
            if t_end is None or d_min is None:
                return None
            return (t_end - t_arr - d_min) * 1000.0
        ends = [flog[C.index(c)] if len(flog) > C.index(c) else None
                for c in ("t_pub_main_end_mono", "t_pub_victim_end_mono", "t_pub_attacker_end_mono")]
        return row + [None if d_min is None else d_min * 1000.0] + [delta(t) for t in ends] + [
            flog.t_shadow_end, delta(flog.t_shadow_end), flog.dds[0], flog.dds[1]]

    # Jobs of the measures thread (--measures thread). st is the genome
    # (genome_id) read at the start of cloud processing (mst in _cloud_cb).

    def _perc_zero(self, st):
        """Zero perceptibility (unperturbed cloud), queued to stay in order with
        pending measurements."""
        self._meas_q.put(("zero", st, None))

    def _perc_measure(self, gen, clean, pert, params, st, gim=None):
        """Chamfer (P) and metrics of a measured cloud. Clouds are not modified
        after enqueueing (publish and replay recording make float32 copies).
        gim (--genoma-in-moto): (stream, genome, phase) of the cloud; the Chamfer
        is also added to the phase sum."""
        # shallow copy of the generator: freezes the counts of the last
        # apply_perturbation (gross term), which the next cloud would overwrite
        self._meas_q.put(("perc", st, (copy.copy(gen), clean, pert, params) + ((gim,) if gim is not None else ())))

    def _tc_measure(self, stream, clean_cloud, out_cloud, frame_idx):
        """Temporal coherence, with the same frame index as inline mode."""
        self._meas_q.put(("tc", None, (stream, clean_cloud, out_cloud, frame_idx)))

    def _meas_loop(self):
        """Measures thread (--measures thread): one job at a time, in cloud order,
        with the same values and ordering as inline mode. The one case to
        reproduce: inline, a cloud's measurement always precedes the next genome
        change, which resets P and the terms (not the metrics). If the genome
        changed after enqueueing, only the metrics are assigned here: P and terms
        stay reset, as inline."""
        lock = self._meas_lock if self._meas_lock is not None else threading.Lock()
        while True:
            kind, st, a = self._meas_q.get()
            t = time.perf_counter()
            try:
                if kind == "tc":
                    self._update_tc(*a)
                elif kind == "zero":
                    with lock:
                        if st == self.genome_id:
                            self._chamfer = 0.0
                            self._perc_terms = dict(ZERO_TERMS) if self._perc_terms_on else None
                        self._perc = {"shift_frac": 0.0, "shift_med_cm": 0.0,
                                      "n_added": 0, "n_removed": 0}
                else:
                    gen, clean, pert, params = a[:4]
                    gim = a[4] if len(a) > 4 else None
                    nn = gen.nn_distances(clean, pert) if self._shared_nn else None
                    ch, terms = self._magnitude_terms(gen, clean, pert, params, nn=nn)
                    try:
                        perc, perc_ok = self._perception_metrics(clean, pert, nn=nn), True
                    except Exception:
                        perc, perc_ok = None, False
                    # values assigned together, once computed
                    with lock:
                        if st == self.genome_id:
                            self._chamfer = ch
                            if terms is not None:
                                self._perc_terms = terms
                        if gim is not None:
                            self._gim_chamfer(*gim, ch)
                        if perc_ok:
                            self._perc = perc
            except Exception:
                pass
            if kind == "perc":
                self._meas_ms.append((time.perf_counter() - t) * 1000.0)
            self._meas_q.task_done()

    def _update_tc(self, stream, clean_cloud, out_cloud, frame_idx):
        """Temporal coherence of the output scan against the previous one of the
        same stream, every chamfer_every frames, also with the attack off. The
        same measure on the clean stream (same motion, same scene) is the
        per-frame null: the exposed values are differences output - clean, plus
        the clean fraction as reference."""
        try:
            def sub(c):
                x = c[:, :3]
                if len(x) > 4000:
                    x = x[np.random.default_rng(frame_idx).choice(len(x), 4000, replace=False)]
                return np.ascontiguousarray(x, dtype=np.float32)
            cur_out, cur_clean = sub(out_cloud), sub(clean_cloud)
            prev = self._tc_prev[stream]
            if (self._chamfer_every > 0 and frame_idx % self._chamfer_every == 0
                    and prev is not None and self._tc_frame_gap(stream, frame_idx)):
                to = temporal_coherence(prev[0], cur_out, seed=frame_idx)
                tc = temporal_coherence(prev[1], cur_clean, seed=frame_idx)
                if to is not None and tc is not None:
                    self._tc = {"tc_frac8": to["tc_frac8"] - tc["tc_frac8"], "tc_frac16": to["tc_frac16"] - tc["tc_frac16"],
                                "tc_med_cm": to["tc_med_cm"] - tc["tc_med_cm"], "tc_p90_cm": to["tc_p90_cm"] - tc["tc_p90_cm"],
                                "tc_rot_deg": to["tc_rot_deg"], "tc_trans_cm": to["tc_trans_cm"],
                                "tc_clean_frac8": tc["tc_frac8"]}
            self._tc_prev[stream] = (cur_out, cur_clean)
            self._tc_last_idx[stream] = frame_idx
        except Exception:
            pass

    def _tc_frame_gap(self, stream, frame_idx):
        """True if the stream's previous scan is the immediately preceding one."""
        if not hasattr(self, "_tc_last_idx"):
            self._tc_last_idx = {"main": None, "victim": None}
        last = self._tc_last_idx.get(stream)
        return last is not None and frame_idx - last == 1

    def _magnitude(self, gen, clean, pert, params, nn=None):
        """P published as chamfer_cm. With --perc-terms also the separate terms
        (same computation, compute_perturbation_terms), published in the status
        of the last sample."""
        if not self._perc_terms_on:
            return float(gen.compute_perturbation_magnitude(clean, pert, params, nn=nn))
        t = gen.compute_perturbation_terms(clean, pert, params, nn=nn)
        self._perc_terms = t
        return float(t["magnitude_cm"])

    def _magnitude_terms(self, gen, clean, pert, params, nn=None):
        """Like _magnitude, but returns (P, terms or None without --perc-terms)
        without touching node state: for the measures thread."""
        if not self._perc_terms_on:
            return float(gen.compute_perturbation_magnitude(clean, pert, params, nn=nn)), None
        t = gen.compute_perturbation_terms(clean, pert, params, nn=nn)
        return float(t["magnitude_cm"]), t

    def _perception_metrics(self, clean: np.ndarray, pert: np.ndarray, nn=None):
        """Geometric metrics between clean and perturbed cloud, on exact nearest
        neighbours (5 cm threshold):
          shift_frac    fraction of perturbed points farther than 5 cm from the
                        nearest clean point
          shift_med_cm  median distance of a perturbed point from the nearest
                        clean point
          n_added       perturbed points with no clean point within 5 cm (ghosts
                        or points moved far)
          n_removed     clean points with no perturbed point within 5 cm (dropout
                        or moved surfaces)
        Two KD-trees on ~45k points: same cost as the Chamfer, computed at the
        same rate.
        """
        from scipy.spatial import cKDTree
        a, b = clean[:, :3], pert[:, :3]
        if len(a) == 0 or len(b) == 0:
            return None
        if nn is not None:
            d_b, d_a = nn        # same queries as the Chamfer (generator nn_distances)
        else:
            d_b, _ = cKDTree(a).query(b, k=1, workers=-1)   # perturbed -> clean
            d_a, _ = cKDTree(b).query(a, k=1, workers=-1)   # clean -> perturbed
        thr = 0.05
        return {
            "shift_frac": float(np.mean(d_b > thr)),
            "shift_med_cm": float(np.median(d_b) * 100.0),
            "n_added": int(np.sum(d_b > thr)),
            "n_removed": int(np.sum(d_a > thr)),
        }

    def _victim_header(self, header):
        """Header stamped on the continuous victim clock."""
        from std_msgs.msg import Header
        sim = header.stamp.sec + header.stamp.nanosec * 1e-9
        if self._victim_last_sim is None:
            clock = sim
        else:
            dt = sim - self._victim_last_sim
            if dt <= 0.0 or dt > 1.0:        # pause (or non-monotonic stamp): one period
                dt = 0.1
            clock = self._victim_last_clock + dt
        self._victim_last_sim = sim
        self._victim_last_clock = clock
        self._victim_offset = sim - clock
        h = Header()
        h.frame_id = header.frame_id
        h.stamp.sec = int(clock)
        h.stamp.nanosec = int(round((clock - int(clock)) * 1e9))
        return h

    def _twin_header(self, header):
        """Header on the twin clock: continues from the last replayed stamp with the
        true simulation intervals (like the victim clock)."""
        from std_msgs.msg import Header
        sim = header.stamp.sec + header.stamp.nanosec * 1e-9
        first_ns = getattr(self, "_twin_first_ns", None)
        if first_ns is not None:
            # first scan after the --twin-clock-base base: exact stamp in integer ns
            self._twin_first_ns = None
            self._twin_clock_first_out_ns = first_ns
            self._twin_last_sim = sim
            self._twin_last_clock = first_ns / 1e9
            self._twin_offset = sim - self._twin_last_clock
            h = Header()
            h.frame_id = header.frame_id
            h.stamp.sec = first_ns // 1_000_000_000
            h.stamp.nanosec = first_ns % 1_000_000_000
            return h
        if self._twin_last_sim is None or self._twin_last_clock is None:
            clock = (self._twin_last_clock + 0.1) if self._twin_last_clock is not None else sim
        else:
            dt = sim - self._twin_last_sim
            if dt <= 0.0 or dt > 1.0:
                dt = 0.1
            clock = self._twin_last_clock + dt
        self._twin_last_sim = sim
        self._twin_last_clock = clock
        self._twin_offset = sim - clock
        h = Header()
        h.frame_id = header.frame_id
        h.stamp.sec = int(clock)
        h.stamp.nanosec = int(round((clock - int(clock)) * 1e9))
        return h

    def _publish_cloud(self, cloud: np.ndarray, header, pub=None):
        if cloud.shape[1] == 3:
            cloud = np.hstack([cloud, np.full((len(cloud), 1), 100.0)])
        data = np.ascontiguousarray(cloud[:, :4], dtype=np.float32)

        out = PointCloud2()
        out.header = header
        out.height = 1
        out.width = len(data)
        out.fields = FIELDS
        out.is_bigendian = False
        out.point_step = POINT_STEP
        out.row_step = POINT_STEP * len(data)
        out.data = data.tobytes()
        out.is_dense = True
        st = getattr(self, "_strum", None)
        if st is not None:
            st.chiamata()
        (pub or self.pub).publish(out)
        return out

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _publish_status(self):
        # With --measures thread, P, terms and metrics are read together under the
        # lock the measures thread writes them with. Works on the class, so also
        # for the partial nodes built in tests.
        lock = getattr(self, "_meas_lock", None)
        if lock is None:
            return PerturbationNode._publish_status_body(self)
        with lock:
            return PerturbationNode._publish_status_body(self)

    def _publish_status_body(self):
        if not self.proc_times or not self.lat_times:
            return
        mean_ms = float(np.mean(self.proc_times)) * 1000.0
        max_ms = float(np.max(self.proc_times)) * 1000.0
        st = {
            "enabled": self.enabled,
            "genome_id": self.genome_id,
            "frames_in": self.n_in,
            "frames_out": self.n_out,
            "proc_ms_mean": round(mean_ms, 2),
            "proc_ms_max": round(max_ms, 2),
            "proc_ms_last": round(float(self.proc_times[-1]) * 1000.0, 2),
            "latency_ms_mean": round(float(np.mean(self.lat_times)) * 1000.0, 2),
            "latency_ms_max": round(float(np.max(self.lat_times)) * 1000.0, 2),
            "min_latency_ms": self._min_latency_ms,
            "chamfer_cm": (None if self._chamfer != self._chamfer
                           else round(self._chamfer, 3)),
            "frame_idx": self.frame_idx,
            "max_drift_lag_s": (None if self._max_drift_lag_s == float("inf")
                                else self._max_drift_lag_s),
            # Effective genome decoding limits (the orchestrator's printouts use
            # them instead of the generator defaults).
            "gen_limits": {k: float(getattr(self.generator, k)) for k in
                           ("max_point_shift", "noise_std", "max_dropout_rate",
                            "max_ghost_points_ratio") if hasattr(self.generator, k)},
            # Form of the structural term of P (--structural).
            "structural": getattr(self.generator, "structural", "net"),
        }
        mol = getattr(self.generator, "moltiplicatori_attivi", lambda: None)()
        if mol:
            # --moltiplicatore-limiti on (key absent when off): the orchestrator
            # compares it with its own before computing the zones
            st["moltiplicatori_limiti"] = mol
        if hasattr(self.generator, "accumulated_drift"):
            d = self.generator.accumulated_drift
            st["drift_cm"] = round(float(np.linalg.norm(d)) * 100.0, 3)
            st["drift_xy_cm"] = round(float(np.hypot(d[0], d[1])) * 100.0, 3)
        dv = self.victim_generator.accumulated_drift
        st.update({
            "victim_stream": self.victim_stream,
            "victim_attack": self.victim_attack,
            "victim_genome_id": self.victim_genome_id,
            "victim_frames": self.victim_frames,
            "victim_frame_idx": self.victim_frame_idx,
            "victim_drift_cm": round(float(np.linalg.norm(dv)) * 100.0, 3),
            "victim_drift_xy_cm": round(float(np.hypot(dv[0], dv[1])) * 100.0, 3),
            "victim_stamp_offset_s": round(self._victim_offset, 4),
            "victim_rec_n": len(self._victim_rec),
            "replay_seq": self._replay_seq,
            "replay_active": self._replay_active,
            "replay_n": self._replay_n,
            "replay_total": self._replay_total,
            "replay_last_stamp": self._replay_last_stamp,
            "twin_clock_on": self._twin_clock_on,
            "twin_stamp_offset_s": round(self._twin_offset, 4),
            "victim_rec_total": getattr(self, "_victim_rec_total", None),
            "victim_rec_max": getattr(self._victim_rec, "maxlen", None),
            "replay_first_stamp": getattr(self, "_replay_first_stamp", None),
            "drift_keep": bool(getattr(self, "_drift_keep", False)),
        })
        if getattr(self, "_twin_clock_base_on", False):
            st.update({"twin_clock_seq": self._twin_clock_seq,
                       "twin_clock_first_ns": self._twin_clock_first_out_ns})
        if getattr(self, "_banco_twin_drift_on", False):
            d0 = self._twin_drift_init
            st["twin_drift_init"] = None if d0 is None else [round(float(x), 6) for x in d0]
        if getattr(self, "_report_matched", False):
            st["victim_pub_matched"] = self.victim_pub.get_subscription_count()
            # main publisher (--output-topic): twin readers
            st["main_pub_matched"] = self.pub.get_subscription_count()
            if getattr(self, "shadow_pub", None) is not None:
                st["shadow_pub_matched"] = self.shadow_pub.get_subscription_count()
        if getattr(self, "_fresh_on", False):
            g = self._fresh_gate
            st.update({"fresh_cloud": True, "fresh_seq": self._fresh_seq, "fresh_armed": g.armed,
                       "fresh_threshold": g.threshold, "fresh_base_src": g.base_src,
                       "fresh_dropped_total": g.dropped_total, "fresh_forced": g.n_forced})
        if getattr(self, "_operator_mode", "random") == "insieme":
            # metre received per stream (confirmation for the orchestrator) and last choice
            st["operator_mode"] = "insieme"
            st["metro_gemello"] = self._metro["gemello"]
            st["metro_vittima"] = self._metro["vittima"]
            g = self.victim_generator if self.victim_stream else self.generator
            inf = g.last_struct_info or {}
            st["struct_k"] = inf.get("k")
            st["struct_per_op"] = inf.get("per_op")
            st["struct_metro"] = inf.get("metro")
            st["struct_n_hit"] = inf.get("n_hit")
        if getattr(self, "_zone_on", False):
            st["zone_attaccante"] = True
            st["fotografia"] = self._foto_stato
            st["zone_ricevute"] = dict(self._zone_ric)
            st["zone_applicate"] = {f: dict(v) for f, v in self._zone_app.items()}
        if getattr(self, "_rampa_R", 1) > 1:
            st["rampa"] = self._rampa_stato()
        if getattr(self, "_gim", None):
            st["genoma_in_moto"] = self._gim_stato()
        if getattr(self, "_cpf", False):
            st["chamfer_fasi"] = self._cpf_stato()
        if getattr(self, "_stato_gen_on", False):
            st["stato_generatore"] = self._stato_gen_status()
        if getattr(self, "_salva", None) is not None:
            st["salva_nuvole"] = self._salva.stato()
        if getattr(self, "_never_clean", False):
            st.update({"victim_never_clean": True,
                       "victim_clean_dropped": self.victim_clean_dropped,
                       "victim_clean_out": self.victim_clean_out})
        if self._perc is not None:
            st["shift_frac"] = round(self._perc["shift_frac"], 4)
            st["shift_med_cm"] = round(self._perc["shift_med_cm"], 3)
            st["n_added"] = self._perc["n_added"]
            st["n_removed"] = self._perc["n_removed"]
        if self._perc_terms_on and self._perc_terms is not None:
            t = self._perc_terms
            st["chamfer_pure_cm"] = round(t["chamfer_cm"], 3)
            st["structural_cm"] = round(t["structural_cm"], 4)
            st["point_ratio"] = round(t["point_ratio"], 6)
            # Gross term: points removed and added by the generator; None if the
            # cloud does not come from apply_perturbation. The n_pts_* names are
            # distinct from n_added/n_removed (points beyond 5 cm).
            rg = t.get("point_ratio_gross")
            st["point_ratio_gross"] = None if rg is None else round(rg, 6)
            st["n_pts_removed"] = t.get("n_removed")
            st["n_pts_added"] = t.get("n_added")
        if self._tc is not None:
            for k in TC_KEYS:
                st[k] = round(self._tc[k], 4)
        if getattr(self, "_meas_q", None) is not None:
            # measurements pending in the thread and duration of the last ones (ms)
            st["meas_pending"] = self._meas_q.qsize()
            if self._meas_ms:
                st["meas_ms_max"] = round(max(self._meas_ms), 2)
        self.status_pub.publish(String(data=json.dumps(st)))
        # At 10 Hz the per-cloud budget is 100 ms: beyond it the node itself drops
        # scans and the attack effect cannot be separated from the delay.
        if mean_ms > 80.0:
            self.get_logger().warn(
                f"Elaborazione {mean_ms:.0f} ms per nuvola: vicino al budget di "
                f"100 ms a 10 Hz. Rischio di scartare frame."
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-topic", type=str, default="/carter/lidar_with_intensity")
    ap.add_argument("--output-topic", type=str, default="/carter/lidar_perturbed")
    ap.add_argument("--victim-topic", type=str, default="/carter/lidar_victim",
                    help="uscita per la MOLA vittima continua (attiva solo con "
                         "/attack/victim_stream)")
    ap.add_argument("--attacker-topic", type=str, default="/carter/lidar_attacker",
                    help="nuvola pulita per la MOLA dell'attaccante, con lo stesso "
                         "gating del flusso vittima")
    ap.add_argument("--genome-file", type=str, default=None)
    ap.add_argument("--passthrough", action="store_true",
                    help="non perturba: le nuvole passano invariate. Serve a "
                         "misurare la baseline in anello chiuso e il costo del "
                         "nodo. NON usare un genoma di zeri per questo: lo zero "
                         "cade a meta' scala di ogni parametro ed e' un attacco "
                         "di media intensita'.")
    ap.add_argument("--perturbation-level", type=float, default=0.5)
    ap.add_argument("--max-point-shift", type=float, default=0.05)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--dropout-rate", type=float, default=0.15)
    ap.add_argument("--gaussian-sigma", type=float, default=0.0,
                    help="se > 0, con attacco acceso applica solo rumore gaussiano "
                         "isotropo per punto con questa deviazione standard (m), "
                         "ignorando il genoma: braccio di confronto")
    ap.add_argument("--passthrough-delay-ms", type=float, default=0.0,
                    help="ritardo aggiunto a ogni nuvola quando l'attacco e' spento, "
                         "per dare al rollout nominale la latenza dei candidati "
                         "(tipicamente proc_ms_mean dello status, ~85 ms). Tenuto per "
                         "compatibilita': per la campagna --min-latency-ms")
    ap.add_argument("--min-latency-ms", type=float, default=0.0,
                    help="durata minima di ogni scan dalla ricezione alla pubblicazione, "
                         "ad attacco acceso o spento e per ogni braccio; gli scan piu' "
                         "lenti non vengono accorciati (campagna: 85)")
    ap.add_argument("--victim-record-max", type=int, default=1200,
                    help="nuvole attaccate della vittima tenute in memoria per il rigioco al twin "
                         "(~0.7 MB l'una)")
    ap.add_argument("--perc-terms", action="store_true",
                    help="pubblica nello status anche i termini della percettibilita' "
                         "(chamfer_pure_cm, structural_cm, point_ratio) oltre a chamfer_cm (P); "
                         "spento di default: status come prima.")
    ap.add_argument("--structural", choices=("net", "gross"), default="net",
                    help="termine strutturale della percettibilita' P: net = |M - N| / N "
                         "(default, serie A), gross = (punti tolti + aggiunti) / N. Con "
                         "--perc-terms lo status porta sempre tutti e due i rapporti.")
    ap.add_argument("--chamfer-every", type=int, default=5,
                    help="calcola la percettibilita' ogni N frame (0 = mai). "
                         "Costa due KD-tree su ~45k punti: ad ogni nuvola "
                         "sforerebbe il budget di 100 ms.")
    ap.add_argument("--min-spacing-ms", type=float, default=40.0,
                    help="intervallo minimo fra due pubblicazioni (default 40): un "
                         "arretrato viene smaltito a non piu' di 25 Hz, senza raffiche")
    ap.add_argument("--queue-depth", type=int, default=50,
                    help="profondita' della coda d'ingresso delle nuvole (default 50: "
                         "assorbe raffiche di ~5 s senza perdere scan)")
    ap.add_argument("--test-offset", type=str, default="",
                    help="collaudo dei controlli di MOLA: traslazione rigida dx,dy,dz [m] applicata a ogni nuvola "
                         "dopo --test-offset-after frame dall'accensione dell'attacco (al posto del genoma)")
    ap.add_argument("--test-offset-after", type=int, default=0)
    ap.add_argument("--test-keep-points", type=int, default=0,
                    help="collaudo: tiene solo N punti per nuvola ad attacco acceso (0 = disattivo)")
    ap.add_argument("--test-double-scan", type=int, default=0,
                    help="collaudo: ogni K frame pubblica la nuvola due volte, la seconda con stamp +0.05 s (0 = disattivo)")
    ap.add_argument("--test-seed", type=int, default=0,
                    help="collaudo: offset del seme delle perturbazioni casuali (rumore), per ripetizioni indipendenti")
    ap.add_argument("--test-sector", type=float, default=0.0,
                    help="collaudo V4: rimuove un settore azimutale di N gradi (attorno alla direzione +x)")
    ap.add_argument("--test-ghost-wall", type=float, default=0.0,
                    help="collaudo V4: fantasma di una parete piana a N m davanti (~3000 punti)")
    ap.add_argument("--test-slow-drift", type=float, default=0.0,
                    help="collaudo V5: deriva laterale cumulativa y += RATE*frame [m/frame]")
    ap.add_argument("--test-drift-until", type=int, default=0,
                    help="collaudo V5: oltre questo frame l'attacco e' tolto (nuvola pulita): per V5")
    ap.add_argument("--test-jitter", type=float, default=0.0,
                    help="collaudo: traslazione rigida di ampiezza A [m] sull'asse x con segno alternato a ogni frame "
                         "(scansioni consecutive a 2A, mappa incoerente): sonda la soglia di accoppiamento 2sigma a regime")
    ap.add_argument("--frame-log", type=str, default=None, metavar="CSV",
                    help="registro per frame: una riga per nuvola ricevuta con arrivo, rilascio, "
                         "fine di publish() verso uscita principale, vittima e attaccante, stamp in "
                         "ingresso e pubblicati, frame_idx, stato dell'attacco, proc_ms "
                         "(colonne in FrameLog.COLUMNS). Scritto a blocchi da un thread. Spento di default.")
    ap.add_argument("--fresh-cloud-after-restore", action="store_true",
                    help="dopo ogni ripristino di Isaac (armato dall'orchestratore con la stessa opzione) "
                         "inoltra a vittima, attaccante e uscita principale solo nuvole con stamp > tempo "
                         "simulato della pausa (ultimo /clock) + --fresh-cloud-margin-s; le altre si scartano "
                         "e, con --frame-log, si registrano insieme a ogni arm (colonne in FreshCloudGate.COLUMNS). "
                         "Spento di default.")
    ap.add_argument("--fresh-cloud-margin-s", type=float, default=0.1,
                    help="margine sul tempo della pausa (default 0.1 s, un periodo del LiDAR a 10 Hz)")
    ap.add_argument("--fresh-cloud-max-drop", type=int, default=5,
                    help="scarti consecutivi oltre i quali il filtro si disarma comunque (default 5)")
    ap.add_argument("--start-disabled", action="store_true",
                    help="parte in passthrough, si attiva via /attack/enabled")
    ap.add_argument("--max-drift-lag-s", type=float, default=float("inf"),
                    help="tetto al modulo della deriva temporale accumulata, "
                         "espresso come ritardo a 0.8 m/s (0.3 = 24 cm); inf = "
                         "nessun tetto, regime fino a 2.5 m")
    ap.add_argument("--twin-clock-base", action="store_true",
                    help="variante G: accetta su /attack/twin_clock_base la base dell'orologio del gemello "
                         "(stamp salvato dall'ombra); il primo header del gemello vale quella base + 0,1 s. "
                         "Spento di default: header del gemello come prima.")
    ap.add_argument("--banco-twin-drift", action="store_true",
                    help="prova al banco G (braccio B): accetta su /attack/twin_drift_init la deriva "
                         "temporale (3 valori, m) da cui parte il gemello al prossimo genoma, al posto "
                         "dell'azzeramento; vuoto = come prima. Lo status la riporta (twin_drift_init). "
                         "Spento di default: nessuna sottoscrizione, status e nuvole invariati.")
    ap.add_argument("--shadow-topic", type=str, default=None,
                    help="pubblica la nuvola della vittima anche su questo topic, per l'ombra "
                         "(stesso messaggio, subito dopo). Spento di default: l'ombra ascolta il topic della vittima.")
    ap.add_argument("--shadow-omit-index", type=int, default=-1,
                    help="controllo positivo del pilota: non pubblica all'ombra lo slot con questo indice "
                         "(slot = nuvole pubblicate alla vittima dall'avvio del nodo, da 0; va scelto in moto). "
                         "Richiede --shadow-topic. -1 (default) = spento.")
    ap.add_argument("--shadow-crc-log", type=str, default=None, metavar="CSV",
                    help="registro per slot del crc dei messaggi inviati a vittima e ombra "
                         "(colonne in ShadowCrcLog.COLUMNS). Spento di default.")
    ap.add_argument("--report-matched", action="store_true",
                    help="lettori agganciati al publisher della vittima (get_subscription_count) nello "
                         "status (victim_pub_matched; shadow_pub_matched con --shadow-topic; "
                         "main_pub_matched per il publisher principale, --output-topic) e nella "
                         "colonna matched_victim del registro dei crc. Serve a --sync-wait publisher "
                         "dell'orchestratore. Spento di default: status e registro invariati.")
    ap.add_argument("--measures", choices=("inline", "thread"), default="inline",
                    help="dove si calcolano Chamfer, metriche e coerenza temporale: inline (default, "
                         "nel percorso della nuvola, come prima) o thread (dopo la messa in coda, in un "
                         "thread dedicato e in ordine: stessi valori, fuori dal ritardo della nuvola)")
    ap.add_argument("--fast-ops", action="store_true",
                    help="raggruppamento per voxel della correlazione spaziale con chiave 1D e PCA "
                         "dell'attacco agli edge a blocchi: stesse nuvole bit per bit, meno tempo con "
                         "il GIL tenuto. Spento di default.")
    ap.add_argument("--shared-nn", action="store_true",
                    help="Chamfer e metriche di percettibilita' dalle stesse due query al vicino piu' "
                         "prossimo (due KD-tree invece di quattro, stessi valori). Spento di default.")
    ap.add_argument("--frame-log-delay", action="store_true",
                    help="con --frame-log, aggiunge in coda le colonne del ritardo effettivo "
                         "(DELAY_COLUMNS: D_min, delta per uscita, fine della publish all'ombra, "
                         "istanti DDS di ricezione e invio). Spento di default.")
    ap.add_argument("--victim-never-clean", action="store_true",
                    help="vittima e ombra non ricevono mai una nuvola pulita: a flusso della vittima aperto, "
                         "una nuvola senza attacco o senza genoma si scarta (contata nello status, "
                         "victim_clean_dropped) invece di passare invariata; l'attaccante la riceve come prima. "
                         "Va con la stessa opzione dell'orchestratore. Spento di default.")
    ap.add_argument("--operator-mode", choices=("random", "insieme"), default="random",
                    help="random (default, serie A): estrazioni con seme hash(genoma) + frame; "
                         "insieme: versione I della serie B, insieme fisso di zone scelto alla prima "
                         "nuvola del metro con il seme del genoma e il metro di /attack/metro, "
                         "inseguito sulle nuvole pulite")
    ap.add_argument("--struct-log", type=str, default=None, metavar="CSV",
                    help="con --operator-mode insieme: registro per nuvola delle zone "
                         "(colonne in StructLog.COLUMNS). Spento di default.")
    ap.add_argument("--zone-attaccante", action="store_true",
                    help="con --operator-mode insieme (serie B): le zone le sceglie l'attaccante "
                         "sulla fotografia del metro (/attack/fotografia, /attack/zone); i flussi le "
                         "installano e le inseguono senza scegliere. Zone assenti o non valide: nessuna "
                         "zona, stato nel registro (zone_stato, zone_hash) e nello status. Spento di default.")
    ap.add_argument("--strumenti-ritardo", type=str, default=None, metavar="CSV",
                    help="strumenti del ritardo (src/nodes/strumenti_ritardo.py): per nuvola istanti DDS, "
                         "inizio e fine della callback, fine delle fasi, inizio e fine di ogni publish; "
                         "raccolte del garbage collector; campioni di /proc/stat e /proc/loadavg; stack di "
                         "tutti i thread se la callback parte o dura oltre 200 ms (file <CSV>_gc.csv, "
                         "_cpu.csv, _cane.csv, _stack.txt). Spento di default.")
    ap.add_argument("--salva-nuvole", type=str, default=None, metavar="DIR",
                    help="salva in DIR, per flusso (vittima e gemello), la nuvola d'ingresso e quella "
                         "d'uscita delle prime --salva-nuvole-k nuvole perturbate e pubblicate di ogni genoma "
                         "(.npz non compresso, piu' indice.csv); prefisso dei file dall'ultimo "
                         "/attack/salva_nuvole_tag, contatori nello status. Scrittura in un thread con coda "
                         "limitata: a coda piena il record si scarta e si conta. Spento di default.")
    ap.add_argument("--salva-nuvole-k", type=int, default=20, metavar="N",
                    help="con --salva-nuvole: nuvole salvate per genoma e flusso (k < N)")
    ap.add_argument("--rampa-genoma", type=int, default=1, metavar="R",
                    help="rampa A3 al genoma nuovo (richiede --zone-attaccante): nelle R - 1 nuvole miste del "
                         "metro (k = 0..R-2) il genoma precedente del flusso, ricevuto su /attack/rampa prima "
                         "del genoma nuovo, e il nuovo si applicano insieme con pesi 1 - alfa_k e alfa_k, "
                         "alfa_k = (k+1)/R; da k = R-1 il solo nuovo. Senza i dati del precedente per quel "
                         "genoma: solo il nuovo, motivo nello status (rampa) e nel registro delle zone. "
                         "1 (default) = spenta: nessuna sottoscrizione, status, registri e nuvole invariati.")
    ap.add_argument("--genoma-in-moto", type=float, default=0.0, metavar="S_CM",
                    help="variante E1 (richiede --zone-attaccante, non con --rampa-genoma R > 1): da k = 0 il "
                         "genoma precedente del flusso (ricevuto su /attack/rampa come per la rampa) da solo, il "
                         "genoma nuovo dalla prima nuvola con k >= --genoma-in-moto-kmin in cui l'inseguitore "
                         "dell'attaccante si e' spostato di almeno S_CM cm dal riferimento (prima nuvola del "
                         "tratto: genoma nuovo, carico D-31c, riapertura della vittima), al piu' tardi a k = "
                         "--genoma-in-moto-kmax. Al metro 0 il solo genoma nuovo. Status genoma_in_moto, colonne "
                         "gradino_* nel registro delle zone. 0 (default) = spento: status, registri e nuvole "
                         "invariati.")
    ap.add_argument("--genoma-in-moto-kmin", type=int, default=20, metavar="K",
                    help="con --genoma-in-moto: prima nuvola (k) in cui il genoma nuovo puo' entrare")
    ap.add_argument("--genoma-in-moto-kmax", type=int, default=40, metavar="K",
                    help="con --genoma-in-moto: nuvola (k) in cui il genoma nuovo entra comunque (ripiego k_max)")
    ap.add_argument("--chamfer-per-fase", action="store_true",
                    help="Chamfer per flusso (vittima, gemello) e per fase P/N del gradino: somma e numero dei valori "
                         "calcolati sul genoma corrente del flusso, nello status (chamfer_fasi); senza gradino tutte "
                         "le nuvole sono di fase N. Acceso da --genoma-in-moto. Spento di default: status invariato.")
    ap.add_argument("--stato-generatore", action="store_true",
                    help="D-31c: su /attack/stato_generatore \"salva\" copia fra due nuvole lo stato del "
                         "generatore di un flusso (indice di nuvola, deriva, inseguitore con zone e ancore, "
                         "zone installate, rampa) in un posto con nome, \"carica\" lo arma per il prossimo "
                         "genoma del gemello, che lo prende al posto dell'azzeramento se il genoma e' lo stesso. "
                         "Chiave stato_generatore nello status; la callback delle nuvole gira sotto un lock. "
                         "Spento di default: nessuna sottoscrizione, status e nuvole invariati.")
    ap.add_argument("--moltiplicatore-limiti", type=float, default=1.0, metavar="K",
                    help="studio dei limiti: moltiplica per K >= 1 i limiti delle famiglie di "
                         "--famiglie-limiti (perturbation_generator.FAMIGLIE_LIMITI), uguale per vittima e "
                         "gemello; status moltiplicatori_limiti e colonne n_in, n_tolti, n_aggiunti, "
                         "frac_tolti nel registro delle zone. 1 (default) = spento: status, registri e "
                         "nuvole invariati.")
    ap.add_argument("--famiglie-limiti", type=str, default=",".join(FAMIGLIE_LIMITI), metavar="LISTA",
                    help="famiglie a cui si applica --moltiplicatore-limiti, separate da virgole: "
                         + ",".join(FAMIGLIE_LIMITI) + " (default tutte)")
    cli, ros_args = ap.parse_known_args(sys.argv[1:])
    if cli.shadow_omit_index >= 0 and not cli.shadow_topic:
        print("ERRORE: --shadow-omit-index richiede --shadow-topic")
        sys.exit(2)
    try:
        moltiplicatori_da_opzioni(cli.moltiplicatore_limiti, cli.famiglie_limiti)
    except ValueError as e:
        print(f"ERRORE: {e}")
        sys.exit(2)

    rclpy.init(args=ros_args)
    node = PerturbationNode(cli)
    # Separate threads for clouds (one at a time, in their group) and for status
    # and commands: a slow cloud does not delay status or commands.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node._frame_log is not None:
            node._frame_log.close()
        if node._crc_log is not None:
            node._crc_log.close()
        if node._struct_log is not None:
            node._struct_log.close()
        if node._strum is not None:
            node._strum.chiudi()
        if node._salva is not None:
            node._salva.chiudi()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
