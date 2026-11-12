"""Perceptibility measures off the cloud path (--measures thread), shared nearest-neighbour
queries (--shared-nn) and effective-delay columns (--frame-log-delay).

Pure tests: no ROS node created (no rclpy.init), no topic. The node is built with __new__
and the minimal state read by _cloud_cb and _pub_loop; publishes go to fake objects. With
the recorded bag in /tmp recorded clouds are also used (otherwise those tests are
skipped)."""
import inspect
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pn = pytest.importorskip("src.nodes.perturbation_node")
PointCloud2 = pn.PointCloud2

BAG = "/tmp/prova_strutture_v4_odom/bag33_det.npz"
GENOMA_MASSIMO = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]   # --fixed-genome


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


def _msg(c, st):
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.header.stamp.sec = int(st)
    m.header.stamp.nanosec = int(round((st - int(st)) * 1e9)) % 1000000000
    m.height, m.width = 1, len(c)
    m.fields = pn.FIELDS
    m.is_bigendian = False
    m.point_step = pn.POINT_STEP
    m.row_step = pn.POINT_STEP * len(c)
    m.data = np.ascontiguousarray(c[:, :4], dtype=np.float32).tobytes()
    m.is_dense = True
    return m


def _nuvole_sintetiche(n=12, seed=3):
    """Corridor with walls, floor and boxes; the sensor advances 8 cm per scan."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([
        np.c_[rng.uniform(0, 20, 9000), np.full(9000, 3.0), rng.uniform(0, 3, 9000)],
        np.c_[rng.uniform(0, 20, 9000), np.full(9000, -3.0), rng.uniform(0, 3, 9000)],
        np.c_[rng.uniform(0, 20, 12000), rng.uniform(-3, 3, 12000), np.zeros(12000)],
        np.c_[rng.uniform(4, 5, 3000), rng.uniform(1, 2, 3000), rng.uniform(0, 1, 3000)],
    ])
    out = []
    for k in range(n):
        c = base.copy()
        c[:, 0] -= 0.08 * k
        c = c + rng.normal(0, 0.003, c.shape)
        out.append((np.c_[c, rng.uniform(0, 255, len(c))].astype(np.float32), 10.0 + 0.1 * k))
    return out


def _nuvole_bag(n):
    if not os.path.exists(BAG):
        pytest.skip("bag della 3.3 assente in /tmp")
    z = np.load(BAG)
    off, xyzi, st = z["offsets"], z["xyzi"], z["stamps"]
    return [(np.asarray(xyzi[off[i]:off[i + 1]], dtype=np.float32), float(st[i])) for i in range(n)]


def _nodo(measures="inline", shared_nn=False, victim=True, frame_log=None, delay=False,
          min_latency_ms=0.0, perc_terms=True, structural="net"):
    N = pn.PerturbationNode
    n = N.__new__(N)
    kw = dict(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15, max_drift_lag_s=0.5,
              structural=structural)
    n.generator = pn.PerturbationGenerator(**kw)
    n.victim_generator = pn.PerturbationGenerator(**kw)
    g = np.asarray(GENOMA_MASSIMO, dtype=np.float64)
    n.genome = g
    n.params = n.generator.encode_perturbation(g)
    n.base_seed = pn.genome_seed(g)
    n.genome_id = 1
    n.victim_params = n.victim_generator.encode_perturbation(g)
    n.victim_seed = pn.genome_seed(g)
    n.victim_genome_id = 1
    n.victim_frame_idx = 0
    n.victim_frames = 0
    n.victim_stream = victim
    n.victim_attack = victim
    n.enabled = True
    n.frame_idx = 0
    n.n_in = n.n_out = 0
    n.proc_times, n.lat_times = [], []
    n._fresh_on = False
    n._frame_log = frame_log
    n._delay_log = bool(delay and frame_log is not None)
    for a in ("_test_jitter", "_test_sector", "_test_ghost_wall", "_test_slow_drift", "_gaussian_sigma"):
        setattr(n, a, 0.0)
    n._test_offset = None
    n._test_keep_points = n._test_double_scan = 0
    n._frames_since_enable = 0
    n._double_pending = False
    n._chamfer_every = 5
    n._chamfer = float("nan")
    n._perc = None
    n._perc_terms_on = perc_terms
    n._perc_terms = None
    n._tc = None
    n._tc_prev = {"main": None, "victim": None}
    n._tc_last_idx = {"main": None, "victim": None}
    n._min_latency_ms = min_latency_ms
    n._passthrough_delay = 0.0
    n._min_spacing = 0.0
    n._victim_last_sim = n._victim_last_clock = None
    n._victim_offset = 0.0
    n._twin_clock_on = False
    n._replay_active = False
    n._victim_rec = deque(maxlen=50)
    n._victim_rec_total = 0
    n.shadow_pub, n._crc_log, n.shadow_topic = None, None, None
    n._shadow_slot, n._shadow_omit = 0, -1
    n.pub, n.victim_pub, n.attacker_pub = _Pub(), _Pub(), _Pub()
    n.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)
    n._out_q = queue.Queue()
    n._measures = measures
    n._shared_nn = shared_nn
    n._meas_q = None
    n._meas_ms = deque(maxlen=200)
    if measures == "thread":
        n._meas_q = queue.Queue()
        threading.Thread(target=n._meas_loop, daemon=True).start()
    return n


def _stato(n):
    return (n._chamfer, None if n._perc_terms is None else dict(n._perc_terms),
            None if n._perc is None else dict(n._perc), None if n._tc is None else dict(n._tc))


def _corri(n, nuvole, cambio_genoma_a=None):
    """One cloud at a time; after each, the state of the measures (thread awaited)."""
    stati, uscite = [], []
    for k, (c, st) in enumerate(nuvole):
        if cambio_genoma_a is not None and k == cambio_genoma_a:
            n._set_genome(np.asarray(GENOMA_MASSIMO, dtype=np.float64) * 0.5, source="test")
        n._cloud_cb(_msg(c, st))
        if n._meas_q is not None:
            n._meas_q.join()
        stati.append(_stato(n))
        job = n._out_q.get_nowait()
        uscite.append((job[2].tobytes(), None if job[4] is None else job[4].tobytes()))
    return stati, uscite


def _uguali(a, b):
    # exact (==) comparison of floats, dicts and None; nan equals nan
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (a != a and b != b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_uguali(a[k], b[k]) for k in a)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_uguali(x, y) for x, y in zip(a, b))
    return a == b


# --- shared queries --------------------------------------------------------------

@pytest.mark.parametrize("structural", ["net", "gross"])
def test_nn_condivise_stessi_valori(structural):
    gen = pn.PerturbationGenerator(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15,
                                   structural=structural)
    params = gen.encode_perturbation(np.asarray(GENOMA_MASSIMO, dtype=np.float64))
    for c, _ in _nuvole_sintetiche(3):
        clean = c.astype(np.float64)
        pert = gen.apply_perturbation(clean, params, seed=7)
        nn = gen.nn_distances(clean, pert)
        assert gen.compute_chamfer_distance(clean, pert, nn=nn) == gen.compute_chamfer_distance(clean, pert)
        assert (gen.compute_perturbation_magnitude(clean, pert, params, nn=nn)
                == gen.compute_perturbation_magnitude(clean, pert, params))
        assert _uguali(gen.compute_perturbation_terms(clean, pert, params, nn=nn),
                       gen.compute_perturbation_terms(clean, pert, params))
        fin = SimpleNamespace()
        m = pn.PerturbationNode._perception_metrics
        assert m(fin, clean, pert, nn=nn) == m(fin, clean, pert)


def test_nn_nuvola_vuota():
    gen = pn.PerturbationGenerator()
    vuota = np.zeros((0, 4))
    assert gen.nn_distances(vuota, np.ones((5, 4))) is None
    assert gen.compute_chamfer_distance(vuota, np.ones((5, 4)), nn=None) == float("inf")


# --- measures in the thread: same values, same output cloud ----------------------

@pytest.mark.parametrize("victim", [True, False])
@pytest.mark.parametrize("modo", [("thread", False), ("thread", True), ("inline", True)])
def test_misure_stessi_valori_sintetico(victim, modo):
    nuvole = _nuvole_sintetiche(12)
    s0, u0 = _corri(_nodo(victim=victim), nuvole)
    s1, u1 = _corri(_nodo(measures=modo[0], shared_nn=modo[1], victim=victim), nuvole)
    assert u0 == u1                       # identical output cloud, byte for byte
    assert sum(1 for s in s0 if s[0] == s[0]) >= 3      # at least 3 frames measured
    for k, (a, b) in enumerate(zip(s0, s1)):
        assert _uguali(a, b), f"frame {k}: {a} != {b}"


def test_misure_stessi_valori_bag():
    nuvole = _nuvole_bag(16)
    s0, u0 = _corri(_nodo(), nuvole)
    s1, u1 = _corri(_nodo(measures="thread", shared_nn=True), nuvole)
    assert u0 == u1
    for k, (a, b) in enumerate(zip(s0, s1)):
        assert _uguali(a, b), f"frame {k}"


def test_gross_conteggi_congelati():
    # gross term: the counts of the last apply_perturbation must be those of the measured
    # cloud even if the thread processes it after the next one
    nuvole = _nuvole_sintetiche(11)
    s0, _ = _corri(_nodo(structural="gross"), nuvole)
    n = _nodo(measures="thread", shared_nn=True, structural="gross")
    blocco = threading.Event()
    orig = n._meas_loop
    n._meas_q = queue.Queue()
    # thread stopped until all clouds are queued, then measures
    threading.Thread(target=lambda: (blocco.wait(), orig()), daemon=True).start()
    for c, st in nuvole:
        n._cloud_cb(_msg(c, st))
    blocco.set()
    n._meas_q.join()
    assert _uguali(_stato(n), s0[-1])
    assert n._perc_terms["point_ratio_gross"] is not None


def test_cambio_genoma_scarta_misura_vecchia():
    nuvole = _nuvole_sintetiche(8)
    n = _nodo(measures="thread", victim=False)
    blocco = threading.Event()
    orig = n._meas_loop
    n._meas_q = queue.Queue()
    threading.Thread(target=lambda: (blocco.wait(), orig()), daemon=True).start()
    n._cloud_cb(_msg(*nuvole[0]))                # frame 0 measured, genome 1, still queued
    n._set_genome(np.asarray(GENOMA_MASSIMO, dtype=np.float64) * 0.5, source="test")
    blocco.set()
    n._meas_q.join()
    assert n._chamfer != n._chamfer and n._perc_terms is None   # reset, not overwritten
    n._cloud_cb(_msg(*nuvole[1]))                # frame 0 of the new genome: measured
    n._meas_q.join()
    assert n._chamfer == n._chamfer and n._chamfer > 0


def test_misure_spente_zero_in_ordine():
    nuvole = _nuvole_sintetiche(6)
    for modo in ("inline", "thread"):
        n = _nodo(measures=modo, victim=False)
        n._cloud_cb(_msg(*nuvole[0]))
        n.enabled = False
        n._cloud_cb(_msg(*nuvole[1]))
        if n._meas_q is not None:
            n._meas_q.join()
        assert n._chamfer == 0.0 and n._perc["n_added"] == 0, modo


def test_percorso_critico_senza_misure():
    # with the thread stopped the callback runs neither Chamfer nor temporal coherence
    n = _nodo(measures="thread", shared_nn=True)
    n._meas_q = queue.Queue()                    # no thread: measures stay queued
    chiamate = []
    n._update_tc = lambda *a: chiamate.append("tc")
    n._magnitude_terms = lambda *a, **k: chiamate.append("ch")
    for c, st in _nuvole_sintetiche(6):
        n._cloud_cb(_msg(c, st))
    assert chiamate == []
    tipi = [n._meas_q.get_nowait()[0] for _ in range(n._meas_q.qsize())]
    assert tipi.count("perc") == 2 and tipi.count("tc") == 6


def test_default_invariato():
    n = _nodo()
    assert n._meas_q is None and not n._shared_nn and not n._delay_log
    # constructor: without the new options in the namespace the defaults are unchanged
    src = inspect.getsource(pn.PerturbationNode.__init__)
    assert 'getattr(args, "measures", "inline")' in src
    assert 'getattr(args, "shared_nn", False)' in src


# --- effective delay in the per-frame log ----------------------------------------

def test_colonne_ritardo(tmp_path):
    flog = pn.FrameLog(tmp_path / "f.csv", extra_columns=pn.DELAY_COLUMNS)
    n = _nodo(measures="thread", shared_nn=True, frame_log=flog, delay=True, min_latency_ms=30.0)
    t = threading.Thread(target=n._pub_loop, daemon=True)
    t.start()
    info = {"received_timestamp": 1_700_000_000_123_456_789, "source_timestamp": 1_700_000_000_100_000_000}
    for c, st in _nuvole_sintetiche(4):
        n._cloud_cb(_msg(c, st), info)
    deadline = time.time() + 10
    while n.n_out < 4 and time.time() < deadline:
        time.sleep(0.01)
    flog.close()
    import csv
    righe = list(csv.DictReader(open(flog.path)))
    assert len(righe) == 4
    assert tuple(righe[0].keys()) == pn.FrameLog.COLUMNS + pn.DELAY_COLUMNS
    for r in righe:
        assert abs(float(r["d_min_ms"]) - 30.0) < 0.01
        dv = float(r["delta_victim_ms"])
        assert abs(dv - ((float(r["t_pub_victim_end_mono"]) - float(r["t_arr_mono"])) * 1000 - 30.0)) < 1e-6
        assert dv >= -0.1
        assert float(r["delta_attacker_ms"]) >= dv
        assert r["t_pub_shadow_end_mono"] == "" and r["delta_shadow_ms"] == ""
        assert abs(float(r["t_dds_recv"]) - 1700000000.1234568) < 1e-6
    # output clouds: three publishers, four clouds each
    assert len(n.victim_pub.msgs) == len(n.attacker_pub.msgs) == 4


def test_registro_senza_opzione_invariato(tmp_path):
    flog = pn.FrameLog(tmp_path / "g.csv")
    n = _nodo(frame_log=flog, delay=False)
    t = threading.Thread(target=n._pub_loop, daemon=True)
    t.start()
    for c, st in _nuvole_sintetiche(2):
        n._cloud_cb(_msg(c, st))
    deadline = time.time() + 10
    while n.n_out < 2 and time.time() < deadline:
        time.sleep(0.01)
    flog.close()
    righe = open(flog.path).read().splitlines()
    assert righe[0] == ",".join(pn.FrameLog.COLUMNS)
    assert all(len(r.split(",")) == len(pn.FrameLog.COLUMNS) for r in righe)


def test_callback_con_messageinfo_solo_con_opzione():
    # rclpy (subscription.py, callback setter): one argument -> message only,
    # two -> message and MessageInfo
    n = _nodo()
    inspect.signature(n._cloud_cb).bind(object())            # default: message only
    cb = (lambda msg, info: n._cloud_cb(msg, info))
    with pytest.raises(TypeError):
        inspect.signature(cb).bind(object())
    inspect.signature(cb).bind(object(), object())


# --- mean P of the window: thread identical to inline, bit for bit ---------------

class _CodaConCancello(queue.Queue):
    """Measure queue that holds every job while the gate is closed: the thread lags behind
    the clouds and commands, as in a live run."""

    def __init__(self):
        super().__init__()
        self.cancello = threading.Event()

    def get(self, *a, **k):
        it = super().get(*a, **k)
        self.cancello.wait()
        return it


# status keys that depend on computation times or exist only in thread mode
_STATUS_TEMPI = {"proc_ms_mean", "proc_ms_max", "proc_ms_last", "latency_ms_mean", "latency_ms_max",
                 "meas_pending", "meas_ms_max"}


def _orchestratore():
    return SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                           perc_samples=[], victim_stamp_offset=0.0, node_gen_limits=None,
                           node_structural=None)


def _sequenza(n, eventi, nuvole, ao):
    """Run the events on a node; at every "status" the thread (if any) drains the queued
    jobs, then the status goes to the real orchestrator (AttackNode._status_cb)."""
    import json
    n.status_pub = _Pub()
    n.lat_times = [0.0]
    # state read by the status and not set by _nodo
    n._max_drift_lag_s = 0.5
    n._replay_seq = n._replay_n = n._replay_total = 0
    n._replay_last_stamp = None
    n._twin_offset = 0.0
    orch = _orchestratore()
    stati = []
    for ev in eventi:
        tipo, val = ev[0], (ev[1] if len(ev) > 1 else None)
        if tipo == "nuvola":
            n._cloud_cb(_msg(*nuvole[val]))
            n._out_q.get_nowait()
        elif tipo == "enabled":
            n._enabled_cb(SimpleNamespace(data=val))
        elif tipo == "victim_attack":
            n.victim_attack = val
        elif tipo == "genoma":
            n._set_genome(np.asarray(GENOMA_MASSIMO, dtype=np.float64) * val, source="test")
        elif tipo == "genoma_vittima":
            n._victim_genome_cb(SimpleNamespace(data=list(np.asarray(GENOMA_MASSIMO, dtype=np.float64) * val)))
        elif tipo == "status":
            if n._meas_q is not None:
                n._meas_q.cancello.set()
                n._meas_q.join()
                n._meas_q.cancello.clear()
            n._publish_status()
            m = n.status_pub.msgs[-1]
            ao.AttackNode._status_cb(orch, m)
            d = json.loads(m.data)
            stati.append(({k: v for k, v in d.items() if k not in _STATUS_TEMPI}, _stato(n)))
    return orch, stati


def _eventi(victim):
    """Attack on and off, clean clouds measured at 0.0 with the switch already on at the
    status, genome change with a measure of the previous genome still queued."""
    E = []
    k = 0

    def nuvole(m, status_ogni=1):
        nonlocal k
        for i in range(m):
            E.append(("nuvola", k))
            k += 1
            if (i + 1) % status_ogni == 0:
                E.append(("status",))
    sw = "victim_attack" if victim else "enabled"
    E.append((sw, False))
    nuvole(3)
    E.append(("nuvola", k)); k += 1                 # clean cloud: 0.0 queued
    E.append((sw, True))                            # on before the status: the 0.0 sample enters
    E.append(("status",))
    nuvole(6, status_ogni=2)                        # attacked measures, thread two clouds behind
    nuvole(4)
    if victim:
        E.append(("genoma_vittima", 0.6))           # new victim genome: P not reset
        nuvole(6)
    E.append(("nuvola", k)); k += 1
    E.append(("genoma", 0.5))                       # genome change with the measure still queued
    E.append(("status",))
    nuvole(6)
    E.append(("nuvola", k)); k += 1
    E.append((sw, False))                           # off before the status: sample excluded
    E.append(("status",))
    nuvole(3)
    E.append((sw, True))
    nuvole(6, status_ogni=3)
    return E, k


@pytest.mark.parametrize("victim", [False, True])
@pytest.mark.parametrize("shared_nn", [False, True])
def test_p_media_thread_uguale_inline(victim, shared_nn):
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    eventi, m = _eventi(victim)
    nuvole = _nuvole_sintetiche(m)
    n0 = _nodo(victim=victim)
    n0._chamfer_every = 2
    o0, s0 = _sequenza(n0, eventi, nuvole, ao)
    n1 = _nodo(victim=victim, shared_nn=shared_nn)
    n1._chamfer_every = 2
    # measure thread as in the constructor, with the queue held until the status
    n1._measures, n1._meas_lock, n1._meas_q = "thread", threading.Lock(), _CodaConCancello()
    threading.Thread(target=n1._meas_loop, daemon=True).start()
    o1, s1 = _sequenza(n1, eventi, nuvole, ao)
    # the sequence exercises the cases to cover
    assert 0.0 in o0.chamfer and any(v > 0 for v in o0.chamfer)
    assert any(s[0].get("chamfer_cm") is None for s in s0)        # reset on genome change
    # orchestrator samples and their mean identical, bit for bit
    assert len(o0.chamfer) == len(o1.chamfer) >= 10
    assert [v.hex() for v in o0.chamfer] == [v.hex() for v in o1.chamfer]
    assert float(np.mean(o0.chamfer)).hex() == float(np.mean(o1.chamfer)).hex()
    assert o0.perc_samples == o1.perc_samples
    assert (o0.drift_cm, o0.drift_xy_cm) == (o1.drift_cm, o1.drift_xy_cm)
    # every status (without times) and every measure state, in the same order
    assert len(s0) == len(s1)
    for k, (a, b) in enumerate(zip(s0, s1)):
        assert a[0] == b[0], f"status {k}"
        assert _uguali(a[1], b[1]), f"stato {k}"


def test_status_senza_stato_della_misura():
    # in thread mode the status carries no field that changes the sample choice
    src = inspect.getsource(pn.PerturbationNode._publish_status_body)
    assert "meas_state" not in src
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    assert "meas_state" not in inspect.getsource(ao.AttackNode._status_cb)
