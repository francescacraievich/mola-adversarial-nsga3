#!/usr/bin/env python3
"""Attack state in the per-frame log of the perturbation node.

The switches (/attack/enabled, /attack/victim_attack) and genome changes arrive in a
reentrant group and can change while a cloud is being computed. The cloud goes out as it
was computed, and the log must say the same. Pure tests: no ROS node created (no
rclpy.init), no topic.

    python3 -m pytest tests/test_stato_attacco_registro.py -q
"""

import csv
import queue
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pn = pytest.importorskip("src.nodes.perturbation_node")
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402

METODI = ("_cloud_cb", "_pub_loop", "_victim_header", "_twin_header", "_update_tc",
          "_tc_frame_gap", "_publish_cloud")
GENOMA = np.random.default_rng(3).uniform(-1.0, 1.0, 17)


class _Pub:
    """Fake publisher: keeps the message bytes; fails after fail_after
    publications (so _pub_loop exits).
    """

    def __init__(self, fail_after=None):
        self.data = []
        self.fail_after = fail_after

    def publish(self, msg):
        if self.fail_after is not None and len(self.data) >= self.fail_after:
            raise RuntimeError("fine della prova")
        self.data.append(bytes(msg.data))


class _Interruttore:
    """Generator that, after computing and before returning to _cloud_cb,
    runs dopo() once: a command arriving mid-computation.
    """

    def __init__(self, gen, dopo):
        self.gen = gen
        self.dopo = dopo

    def apply_perturbation(self, *a, **k):
        out = self.gen.apply_perturbation(*a, **k)
        if self.dopo is not None:
            self.dopo()
            self.dopo = None
        return out


def _generatore():
    # same limits as the node with default options (main(): --max-point-shift,
    # --noise-std, --dropout-rate, --max-drift-lag-s, --structural)
    return PerturbationGenerator(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15,
                                 max_drift_lag_s=float("inf"), structural="net")


def _nodo(frame_log, victim):
    cls = type("NodoFinto", (), {m: getattr(pn.PerturbationNode, m) for m in METODI})
    n = cls()
    n._frame_log = frame_log
    n.n_in = n.n_out = 0
    n.frame_idx = n.victim_frame_idx = n.victim_frames = 0
    g = _generatore()
    n.generator, n.victim_generator = g, _generatore()
    n.params = n.victim_params = g.encode_perturbation(GENOMA)
    n.base_seed = n.victim_seed = pn.genome_seed(GENOMA)
    n.enabled = not victim
    n.victim_stream = victim
    n.victim_attack = victim
    n._test_jitter = 0.0; n._test_offset = None; n._test_sector = 0.0; n._test_ghost_wall = 0.0
    n._test_slow_drift = 0.0; n._test_keep_points = 0; n._test_double_scan = 0
    n._frames_since_enable = 0
    n._gaussian_sigma = 0.0
    n._chamfer_every = 0
    n._perc_terms_on = False
    n._min_latency_ms = 0.0
    n._passthrough_delay = 0.0
    n._min_spacing = 0.0
    n._twin_clock_on = False
    n._victim_last_sim = n._victim_last_clock = None
    n._victim_offset = 0.0
    n._tc_prev = {"main": None, "victim": None}
    n._tc_last_idx = {"main": None, "victim": None}
    n.proc_times = []
    n.lat_times = []
    n._out_q = queue.Queue()
    n._replay_active = False
    n._double_pending = False
    n._victim_rec = deque(maxlen=10)
    n._victim_rec_total = 0
    n.pub, n.victim_pub, n.attacker_pub = _Pub(), _Pub(), _Pub()
    n.get_logger = lambda: SimpleNamespace(warn=lambda s: None, info=lambda s: None)
    return n


def _nuvola(stamp, n_pts=3000, seed=0):
    from sensor_msgs.msg import PointCloud2
    pts = np.random.default_rng(seed).uniform(-20, 20, (n_pts, 4)).astype(np.float32)
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.header.stamp.sec = int(stamp)
    m.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    m.height = 1; m.width = n_pts; m.fields = pn.FIELDS; m.is_bigendian = False
    m.point_step = pn.POINT_STEP; m.row_step = pn.POINT_STEP * n_pts
    m.data = pts.tobytes(); m.is_dense = True
    return m, pts.astype(np.float64)


def _corsa(tmp_path, victim, dopo, nome, frame_idx0=0):
    """One cloud with dopo() run mid-computation, plus a cloud that closes _pub_loop.
    Returns node, log rows and input cloud.
    """
    fl = pn.FrameLog(tmp_path / f"{nome}.csv", flush_s=0.05)
    n = _nodo(fl, victim)
    n.frame_idx = n.victim_frame_idx = frame_idx0
    gen = "victim_generator" if victim else "generator"
    setattr(n, gen, _Interruttore(getattr(n, gen), (lambda: dopo(n)) if dopo else None))
    msg, cloud = _nuvola(10.0)
    n._cloud_cb(msg)
    n._cloud_cb(_nuvola(10.1, seed=1)[0])        # last: its publication fails
    n.pub = _Pub(fail_after=1)
    n._pub_loop()
    fl.close()
    return n, list(csv.DictReader(open(fl.path))), cloud


def _attesa(cloud, seed):
    out = _generatore().apply_perturbation(cloud, _generatore().encode_perturbation(GENOMA), seed=seed)
    return np.ascontiguousarray(out[:, :4], dtype=np.float32).tobytes()


def test_vittima_spenta_a_calcolo_in_corso_registrata_attaccata(tmp_path):
    def spegni(n):
        n.victim_attack = False                  # /attack/victim_attack False
    n_rif, r_rif, cloud = _corsa(tmp_path, True, None, "rif")
    n, rows, cloud2 = _corsa(tmp_path, True, spegni, "spenta")
    assert n.victim_attack is False              # the switch really changed
    attesa = _attesa(cloud, pn.genome_seed(GENOMA) % (2**31 - 1))
    # output: the cloud computed with the attack goes out perturbed, identical to the one
    # without switch-off and to the genome perturbation with the frame-0 seed
    assert n.victim_pub.data[0] == n_rif.victim_pub.data[0] == attesa
    assert n.victim_pub.data[0] != np.ascontiguousarray(cloud2, dtype=np.float32).tobytes()
    assert n.attacker_pub.data[0] == np.ascontiguousarray(cloud2, dtype=np.float32).tobytes()
    # log: attacked, as the row without switch-off
    assert rows[0]["victim_stream"] == "1" and rows[0]["victim_attack"] == "1"
    assert r_rif[0]["victim_attack"] == "1"
    assert rows[0]["victim_frame_idx"] == r_rif[0]["victim_frame_idx"] == "0"
    # the next cloud, computed with the attack off, goes out clean and is logged clean
    assert rows[1]["victim_attack"] == "0"


def test_principale_spenta_a_calcolo_in_corso_registrata_attaccata(tmp_path):
    def spegni(n):
        n.enabled = False                        # /attack/enabled False
    n_rif, r_rif, cloud = _corsa(tmp_path, False, None, "rif")
    n, rows, _ = _corsa(tmp_path, False, spegni, "spenta")
    attesa = _attesa(cloud, pn.genome_seed(GENOMA) % (2**31 - 1))
    assert n.pub.data[0] == n_rif.pub.data[0] == attesa
    assert rows[0]["enabled"] == r_rif[0]["enabled"] == "1"
    assert rows[1]["enabled"] == "0"


def test_indici_di_frame_del_calcolo(tmp_path):
    # genome change mid-computation: the counter restarts from 0, the row
    # carries the index used for the seed of the computed cloud
    def nuovo_genoma(n):
        n.victim_frame_idx = 0                   # _victim_genome_cb
    n, rows, cloud = _corsa(tmp_path, True, nuovo_genoma, "genoma", frame_idx0=7)
    assert rows[0]["victim_frame_idx"] == "7"
    assert n.victim_pub.data[0] == _attesa(cloud, (pn.genome_seed(GENOMA) + 7) % (2**31 - 1))
    assert n.victim_frame_idx == 2               # node counter: 0 + 1 + 1, as usual

    def nuovo_genoma_principale(n):
        n.frame_idx = 0                          # _set_genome
    n, rows, cloud = _corsa(tmp_path, False, nuovo_genoma_principale, "genoma_p", frame_idx0=5)
    assert rows[0]["frame_idx"] == "5"
    assert n.pub.data[0] == _attesa(cloud, (pn.genome_seed(GENOMA) + 5) % (2**31 - 1))
    assert n.frame_idx == 2


def test_senza_cambi_uscita_e_righe_come_prima(tmp_path):
    # no command during the computation: state and indices logged as read after it
    n, rows, cloud = _corsa(tmp_path, True, None, "fermo", frame_idx0=3)
    assert [r["frame_idx"] for r in rows] == ["3", "4"]
    assert [r["victim_frame_idx"] for r in rows] == ["3", "4"]
    assert [(r["enabled"], r["victim_stream"], r["victim_attack"]) for r in rows] == [("0", "1", "1")] * 2
    assert n.victim_pub.data[0] == _attesa(cloud, (pn.genome_seed(GENOMA) + 3) % (2**31 - 1))
