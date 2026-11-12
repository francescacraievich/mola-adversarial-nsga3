#!/usr/bin/env python3
"""
PERC_TERMS values, warm-start front after re-evaluations, single random search as main calls
it, per-frame node log (--frame-log). No active ROS or Isaac.

    python3 -m pytest tests/test_integrazione_B.py -q
"""

import csv
import importlib.util
import inspect
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimization import window_record as wr  # noqa: E402
from src.analysis.fronte_pareto import candidato_valido, fronte_pareto  # noqa: E402


def _ao():
    from src.optimization import attack_orchestrator as ao
    return ao


def _modulo_A():
    spec = importlib.util.spec_from_file_location("tia_b", ROOT / "tests" / "test_integrazione_A.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TIA = _modulo_A()
CAMPAGNA = ROOT / "scripts" / "run_campaign.sh"


def _rec(idx, damage, chamfer, status="ok", **kw):
    r = {"status": status, "counted": True, "admissible": True, "twin_misaligned": False,
         "damage": damage, "dev_mean": damage, "chamfer": chamfer,
         "genome": [float(idx)] * 4, "t_wall": 100.0 + idx}
    r.update(kw)
    return r


# ---------------------------------------------------------------------------
# PERC_TERMS only empty, 0 or 1
# ---------------------------------------------------------------------------

def test_perc_terms_validato(tmp_path):
    for bad in ("yes", "2", "true", "01"):
        rc, o = TIA._comando(CAMPAGNA, tmp_path, {"SERIE_B": "1", "PERC_TERMS": bad}, arm="nsga3")
        assert rc == 1 and "PERC_TERMS sconosciuto" in o, bad
        rc, _ = TIA._comando(CAMPAGNA, tmp_path, {"PERC_TERMS": bad}, arm="nsga3")
        assert rc == 1
    for ok in ("0", "1"):
        rc, o = TIA._comando(CAMPAGNA, tmp_path, {"SERIE_B": "1", "PERC_TERMS": ok}, arm="nsga3")
        assert rc == 0 and f"_pt{ok}_" in o and f"PT={ok}" in o
    assert TIA._comando(CAMPAGNA, tmp_path, {}, arm="nsga3")[0] == 0


# ---------------------------------------------------------------------------
# Warm-start front before or after the re-evaluations
# ---------------------------------------------------------------------------

def _pool_codice_vecchio(eval_log, select, budget, active_genes):
    """Older block of main (attack_orchestrator.py), copied."""
    adm_ev = [e for e in eval_log if e.get("admissible") and e.get("dev_mean") is not None and e.get("genome")]
    if select == "pareto_maxdamage":
        fr_w, _ = fronte_pareto(eval_log, budget=budget)
        warm_pool = [(np.array(p["genome"], dtype=float)[active_genes], p["damage"] * 100, p["chamfer"])
                     for p in fr_w] or None
        warm_best = None
    elif adm_ev:
        pts = [((e["dev_mean"] * 100), e["chamfer"]) for e in adm_ev]
        fidx = _ao().pareto_idx_maxmin([p[0] for p in pts], [p[1] for p in pts])
        warm_pool = [(np.array(adm_ev[i]["genome"], dtype=float)[active_genes],
                      adm_ev[i]["dev_mean"] * 100, adm_ev[i]["chamfer"]) for i in fidx]
        warm_best = sorted([(np.array(e["genome"], dtype=float)[active_genes], e["dev_mean"] * 100)
                            for e in adm_ev], key=lambda t: -t[1])
    else:
        warm_pool = None; warm_best = None
    return warm_pool, warm_best


def _log_con_degrado():
    # idx 1 dominates all (high damage); it is measured during the degradation
    return [_rec(0, 0.02, 1.0), _rec(1, 0.09, 1.5), _rec(2, 0.05, 2.0),
            _rec(3, 0.01, 0.5), _rec(4, 0.03, 3.0, status="stopped_early(0.1m)")]


@pytest.mark.parametrize("select", ["pareto_maxdamage", "maxdamage", "ratio"])
def test_fronte_uguale_al_codice_vecchio(select):
    ao = _ao()
    ag = [0, 1, 2, 3]
    rng = np.random.default_rng(3)
    logs = [_log_con_degrado(), [], [_rec(0, 0.02, 1.0, admissible=False)]]
    for _ in range(20):
        logs.append([_rec(i, float(rng.uniform(0, .1)), float(rng.uniform(0, 5)),
                          admissible=bool(rng.random() > .2)) for i in range(int(rng.integers(1, 12)))])
    for log in logs:
        for budget in (None, 2.5):
            vec, best_v = _pool_codice_vecchio(log, select, budget, ag)
            nuovo, best_n, idx = ao.warm_pool_from_log(log, select, budget, ag)
            assert (vec is None) == (nuovo is None)
            if vec is None:
                assert idx == []
                continue
            assert len(vec) == len(nuovo) == len(idx)
            for a, b in zip(vec, nuovo):
                assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
                assert log[b[3]]["chamfer"] == b[2]          # index in eval_log
            assert (best_v is None) == (best_n is None)
            if best_v is not None:
                assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(best_v, best_n))
            # same warm-start rows with and without the index in the tuples
            r3, c3 = ao.warm_front_rows(vec, 4, np.random.default_rng(9), 4)
            r4, c4 = ao.warm_front_rows(nuovo, 4, np.random.default_rng(9), 4)
            assert all(np.array_equal(a, b) for a, b in zip(r3, r4)) and len(r3) == len(r4)
            assert {k: v for k, v in c4.items() if k != "warm_front_copied_idx"} == c3
            assert len(c4["warm_front_copied_idx"]) == c4["warm_front_n"]


def test_fronte_dopo_le_rivalutazioni_senza_invalidati(capsys):
    ao = _ao()
    ag = [0, 1, 2, 3]
    log = _log_con_degrado()
    prima, _, idx_prima = ao.warm_pool_from_log(log, "pareto_maxdamage", None, ag)
    assert 1 in idx_prima                                   # the degraded one dominates the front

    def eval_genome(x):                                     # re-evaluation: lower damage
        log.append(_rec(len(log), 0.04, 1.5, genome=[float(v) for v in x]))
        return (-0.04, 1.5, 0.0)
    ev = [{"t_prev_clean": 100.5, "t_restart": 101.5}]
    ao.reevaluate_after_adapter_restart(log, ev, 99.0, eval_genome, ag)
    assert log[1]["measured_during_adapter_degradation"] and log[1]["superseded_by"] == 5
    assert log[5]["reevaluation_of"] == 1 and len(log) == 6
    dopo, _, idx_dopo = ao.warm_pool_from_log(log, "pareto_maxdamage", None, ag)
    assert 1 not in idx_dopo and 5 in idx_dopo
    assert all(candidato_valido(log[i]) for i in idx_dopo)
    assert not any(log[i].get("measured_during_adapter_degradation") for i in idx_dopo)
    # same outcome with maxdamage selection (admissibles)
    _, _, idx_md = ao.warm_pool_from_log(log, "maxdamage", None, ag)
    assert 1 not in idx_md
    assert "ADATTATORE riavviato" in capsys.readouterr().out


def test_main_ordine_del_fronte_e_history():
    src = inspect.getsource(_ao().main)
    i_prima = src.index("if not args.warm_front_after_reeval:")
    i_reeval = src.index("reevaluate_after_adapter_restart(eval_log, events, t_win, eval_genome, active_genes)")
    i_dopo = src.index('if args.search == "nsga3" and args.warm_front_after_reeval:')
    i_ripresa = src.index("random_first_resume(")
    assert i_prima < i_reeval < i_dopo < i_ripresa
    assert '"computed": "before_reeval"' in src and '"computed": "after_reeval"' in src
    assert '"warm_front_used": warm_used' in src and '"warm_front_next":' in src
    # the warm start uses the front computed in the previous window
    assert "warm_used = dict(warm_src or {}, copied_idx=" in src
    full = Path(_ao().__file__).read_text()
    assert '"--warm-front-after-reeval", action="store_true"' in full


def test_campagna_serie_b_accende_il_fronte_dopo(tmp_path):
    rc, o = TIA._comando(CAMPAGNA, tmp_path, {"SERIE_B": "1"}, arm="nsga3")
    assert rc == 0 and o.count("--warm-front-after-reeval") == 3       # three repetitions
    assert "DIR=nsga3_g3_p32x16_rkeep_sel2_warm_serieB_s1001-2001-3001_pt1_opinsieme\n" in o
    rc, o = TIA._comando(CAMPAGNA, tmp_path, {"SERIE_B": "1"}, arm="random")
    assert rc == 0 and "--warm-front-after-reeval" not in o
    rc, o = TIA._comando(CAMPAGNA, tmp_path, {"SERIE_B": "1", "WARM_FRONT_AFTER_REEVAL": "0"}, arm="nsga3")
    assert rc == 0 and "--warm-front-after-reeval" not in o and "_wfr0_serieB" in o
    rc, o = TIA._comando(CAMPAGNA, tmp_path, {"WARM_FRONT_AFTER_REEVAL": "1"}, arm="nsga3")
    assert rc == 0 and "--warm-front-after-reeval" in o and "DIR=nsga3_g3_p32x16_sel2_warm_wfr\n" in o
    rc, o = TIA._comando(CAMPAGNA, tmp_path, {}, arm="nsga3")
    assert rc == 0 and "--warm-front-after-reeval" not in o and "_wfr" not in o
    assert TIA._comando(CAMPAGNA, tmp_path, {"WARM_FRONT_AFTER_REEVAL": "si"})[0] == 1


# ---------------------------------------------------------------------------
# Single random: the wiring in main (search, re-evaluation, resume, choice)
# ---------------------------------------------------------------------------

def test_random_singolo_collegamento_di_main(capsys):
    ao = _ao()
    n_search = 4
    ag = list(range(n_search))
    off = np.array([-1.0] * n_search)
    log = []
    esiti = iter([False, True,       # search: valid at the second attempt
                  False,             # re-evaluation of the valid one after the restart: invalid
                  False, True])      # resume: two attempts, the second valid
    budget = 10.0

    def eval_genome(x):              # as in main: appends the record and returns 3 objectives
        v = next(esiti)
        i = len(log)
        log.append(_rec(i, 0.01 * (i + 1), 1.0 + i, status="ok" if v else "stopped_early(0.1m)",
                        genome=[float(g) for g in x]))
        return (-0.01 * (i + 1), 1.0 + i, 0.0)

    rng = np.random.default_rng(np.random.SeedSequence([1001, 2]))
    rf, X, F, draw = ao.random_first_search(rng, n_search, "uniform", off, eval_genome, log, budget, 5)
    assert rf == {"k_max": 5, "n_tries": 2, "found": True, "eval_idx": 1, "sampling": "uniform"}
    assert X.shape == (2, n_search) and F.shape == (2, 2)
    gemello = np.random.default_rng(np.random.SeedSequence([1001, 2]))
    attesi = [wr.sample_genomes(gemello, 1, n_search, "uniform", off)[0] for _ in range(4)]
    assert np.array_equal(X[0], attesi[0]) and np.array_equal(X[1], attesi[1])
    # adapter restart during the measurement of the valid one (t_wall 101)
    ao.reevaluate_after_adapter_restart(log, [{"t_prev_clean": 100.5, "t_restart": 101.5}],
                                        99.0, eval_genome, ag)
    assert log[1]["measured_during_adapter_degradation"] and log[2]["reevaluation_of"] == 1
    rf, X, F, n_more = ao.random_first_resume(rf, draw, eval_genome, log, budget, 5, X, F)
    assert n_more == 2 and X.shape == (4, n_search) and F.shape == (4, 2)
    assert np.array_equal(X[2], attesi[2]) and np.array_equal(X[3], attesi[3])   # same generator
    assert rf["eval_idx"] == 4 and rf["n_tries"] == 4 and rf["n_tries_resumed"] == 2
    fronte, scelto = fronte_pareto(log, budget=budget)
    ch = ao.random_first_choice(fronte, log, rf)
    assert ch["idx"] == 4 and ch is scelto
    out = capsys.readouterr().out
    assert "random singolo con budget: valido dopo 2 tentativi su 5" in out
    assert "ripresa dopo il riavvio dell'adattatore, 2 tentativi in piu' (valido)" in out


def test_main_chiama_il_collegamento():
    src = inspect.getsource(_ao().main)
    assert ("random_first_search(\n                        rng, n_search, mode_s, off, eval_genome, eval_log,\n"
            "                        args.chamfer_budget, args.random_first_admissible)") in src
    assert ("random_first_resume(\n                    rf_rec, rf_draw, eval_genome, eval_log, args.chamfer_budget,\n"
            "                    args.random_first_admissible, X_all, F_all)") in src
    assert "scelto = random_first_choice(fronte, eval_log, rf_rec)" in src
    # no copy of the old block left in main
    assert "rf_rec, rows_x, rows_f = random_first_run(" not in src
    assert "restarts = [ev for ev in events" not in src


# ---------------------------------------------------------------------------
# Per-frame node log (--frame-log)
# ---------------------------------------------------------------------------

def _pn():
    return pytest.importorskip("src.nodes.perturbation_node")


METODI = ("_cloud_cb", "_pub_loop", "_victim_header", "_twin_header", "_update_tc",
          "_tc_frame_gap", "_publish_cloud")


class _Pub:
    def __init__(self, fail_after=None):
        self.stamps = []
        self.fail_after = fail_after

    def publish(self, msg):
        if self.fail_after is not None and len(self.stamps) >= self.fail_after:
            raise RuntimeError("fine della prova")
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)


def _nodo(pn, frame_log=None, victim=True):
    import queue
    from collections import deque
    cls = type("NodoFinto", (), {m: getattr(pn.PerturbationNode, m) for m in METODI})
    n = cls()
    n._frame_log = frame_log
    n.n_in = n.n_out = 0
    n.frame_idx = n.victim_frame_idx = n.victim_frames = 0
    n.enabled = False
    n.victim_stream = victim
    n.victim_attack = False
    n.victim_params = None
    n._test_jitter = 0.0; n._test_offset = None; n._test_sector = 0.0; n._test_ghost_wall = 0.0
    n._test_slow_drift = 0.0; n._test_keep_points = 0; n._test_double_scan = 0
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
    n.get_logger = lambda: SimpleNamespace(warn=lambda s: None)
    return n


def _nuvola(pn, stamp, n_pts=45000, seed=0):
    from sensor_msgs.msg import PointCloud2
    pts = np.random.default_rng(seed).uniform(-20, 20, (n_pts, 4)).astype(np.float32)
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.header.stamp.sec = int(stamp)
    m.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    m.height = 1; m.width = n_pts; m.fields = pn.FIELDS; m.is_bigendian = False
    m.point_step = pn.POINT_STEP; m.row_step = pn.POINT_STEP * n_pts
    m.data = pts.tobytes(); m.is_dense = True
    return m


def _svuota(n, fail_pub):
    # _pub_loop exits when a publish fails: last cloud with a failing publisher
    n.pub = _Pub(fail_after=fail_pub)
    n._pub_loop()


def test_frame_log_spento_di_default(tmp_path):
    pn = _pn()
    n = _nodo(pn)
    n._cloud_cb(_nuvola(pn, 10.0, 1000))
    item = n._out_q.get_nowait()
    assert len(item) == 8 and item[7] is None
    src = (ROOT / "src" / "nodes" / "perturbation_node.py").read_text()
    assert 'ap.add_argument("--frame-log", type=str, default=None' in src
    # extra columns only with --fresh-cloud-after-restore (test_prima_nuvola_fresca.py)
    # or with --frame-log-delay (test_misure_fuori_percorso.py), in this order
    assert ('extra_columns=((FreshCloudGate.COLUMNS if self._fresh_on else ())\n'
            '                                                   + (DELAY_COLUMNS if getattr(args, '
            '"frame_log_delay", False)\n'
            '                                                      else ())))\n'
            '                           if getattr(args, "frame_log", None) else None)') in src
    assert list(tmp_path.iterdir()) == []


def test_frame_log_righe(tmp_path):
    pn = _pn()
    fl = pn.FrameLog(tmp_path / "frame_log.csv", flush_s=0.05)
    n = _nodo(pn, fl, victim=True)
    n._min_latency_ms = 20.0
    stamps = [10.0, 10.1, 13.0, 13.1]           # 2.9 s jump: the victim advances by 0.1
    for s in stamps:
        n._cloud_cb(_nuvola(pn, s, 2000))
    n.victim_stream = False                     # one cloud with the victim stream closed
    n._cloud_cb(_nuvola(pn, 13.2, 2000))
    n._cloud_cb(_nuvola(pn, 13.3, 2000))        # last: makes _pub_loop exit
    _svuota(n, fail_pub=5)
    fl.close()
    rows = list(csv.DictReader(open(tmp_path / "frame_log.csv")))
    # 6 rows: including the last, whose publish fails
    assert len(rows) == 6 and tuple(rows[0].keys()) == pn.FrameLog.COLUMNS
    assert all(None not in r and len(r) == 19 for r in rows)
    ult = rows.pop()
    assert float(ult["stamp_in"]) == pytest.approx(13.3) and ult["t_pub_main_end_mono"] == ""
    assert float(ult["t_pub_end_wall"]) >= float(ult["t_arr_wall"])
    assert [float(r["stamp_in"]) for r in rows] == pytest.approx([10.0, 10.1, 13.0, 13.1, 13.2])
    assert [float(r["stamp_victim"]) for r in rows[:4]] == pytest.approx([10.0, 10.1, 10.2, 10.3])
    assert rows[4]["stamp_victim"] == "" and rows[4]["victim_stream"] == "0"
    assert [r["frame_idx"] for r in rows] == ["0", "1", "2", "3", "4"]
    assert [r["victim_frame_idx"] for r in rows] == ["0", "1", "2", "3", ""]
    assert n.victim_pub.stamps == pytest.approx([10.0, 10.1, 10.2, 10.3])
    for r in rows:
        t_arr, t_rel = float(r["t_arr_mono"]), float(r["t_release_mono"])
        assert float(r["t_release_target_mono"]) == pytest.approx(t_arr + 0.020, abs=2e-3)
        assert t_rel >= float(r["t_release_target_mono"]) - 1e-3  # release after the minimum delay
        assert float(r["t_pub_main_end_mono"]) >= t_rel
        assert float(r["proc_ms"]) > 0 and r["n_points"] == "2000" and r["enabled"] == "0"
        assert float(r["t_pub_end_wall"]) >= float(r["t_arr_wall"])
    for r in rows[:4]:
        assert float(r["t_pub_attacker_end_mono"]) >= float(r["t_pub_victim_end_mono"]) >= float(r["t_pub_main_end_mono"])
    assert rows[4]["t_pub_victim_end_mono"] == ""


def _costo_callback(pn, tmp_path, n_rep=120):
    """Per-cloud time of _cloud_cb (45 000 points) with and without the log, alternated, and
    per-cloud time of _pub_loop with and without the log."""
    fl = pn.FrameLog(tmp_path / "costo.csv", flush_s=0.2)
    on, off = _nodo(pn, fl), _nodo(pn, None)
    msgs = [_nuvola(pn, 10.0 + 0.1 * i, 45000, seed=i % 3) for i in range(3)]
    t_on, t_off = [], []
    for i in range(n_rep):
        for nodo, acc in ((on, t_on), (off, t_off)) if i % 2 == 0 else ((off, t_off), (on, t_on)):
            m = msgs[i % 3]
            t = time.perf_counter()
            nodo._cloud_cb(m)
            acc.append(time.perf_counter() - t)
    # publish side: zero-cost publisher, no wait
    p_on, p_off = [], []
    for nodo, acc in ((on, p_on), (off, p_off)):
        q_n = nodo._out_q.qsize()
        nodo.pub = _Pub(fail_after=q_n - 1)
        nodo._publish_cloud = lambda cloud, header, pub=None, _n=nodo: (pub or _n.pub).publish(
            SimpleNamespace(header=header))
        t = time.perf_counter()
        nodo._pub_loop()
        acc.append((time.perf_counter() - t) / q_n)
    t = time.perf_counter()
    fl.close()
    t_close = time.perf_counter() - t
    return t_on, t_off, p_on[0], p_off[0], fl.n_written, t_close


def test_frame_log_costo_sotto_1_ms(tmp_path, capsys):
    pn = _pn()
    t_on, t_off, p_on, p_off, n_rows, t_close = _costo_callback(pn, tmp_path)
    med_on, med_off = statistics.median(t_on) * 1e3, statistics.median(t_off) * 1e3
    d_cb = med_on - med_off
    d_pub = (p_on - p_off) * 1e3
    # isolated cost of the added statements (callback and publish), over 10 000 repetitions
    hdr = _nuvola(pn, 10.0, 10).header
    fl = pn.FrameLog(tmp_path / "micro.csv", flush_s=0.2)
    t = time.perf_counter()
    for i in range(10000):
        r = [time.time(), time.monotonic(), pn._stamp_s(hdr)]
        r += [i, i, False, True, False, 45000, 3.1, r[1] + 0.085, pn._stamp_s(hdr), pn._stamp_s(hdr)]
        r += [time.monotonic(), False, None, None, None, None]
        r[-4] = time.monotonic(); r[-3] = time.monotonic(); r[-2] = time.monotonic(); r[-1] = time.time()
        fl.add(r)
    micro_ms = (time.perf_counter() - t) / 10000 * 1e3
    t = time.perf_counter()
    fl.close()
    scrittura_ms = (time.perf_counter() - t) / 10000 * 1e3
    with capsys.disabled():
        print(f"\n[costo frame-log] _cloud_cb 45k punti, mediana su {len(t_on)}: con {med_on:.3f} ms, "
              f"senza {med_off:.3f} ms, differenza {d_cb:+.3f} ms")
        print(f"[costo frame-log] _pub_loop per nuvola: con {p_on*1e3:.4f} ms, senza {p_off*1e3:.4f} ms, "
              f"differenza {d_pub:+.4f} ms; righe scritte {n_rows}, close {t_close*1e3:.2f} ms")
        print(f"[costo frame-log] istruzioni aggiunte isolate: {micro_ms*1e3:.2f} us per nuvola; "
              f"scrittura su file (thread) {scrittura_ms*1e3:.2f} us per riga")
    assert n_rows == len(t_on)  # including the last, which closes _pub_loop with an error
    assert micro_ms < 1.0
    assert d_cb + max(d_pub, 0.0) < 1.0


# ---------------------------------------------------------------------------
# Twin variants: joints, MOLA configuration, fingerprint cost, replay gate
# ---------------------------------------------------------------------------

def test_e_giunti_sempre_azzerati_con_attacker_est():
    from src.optimization import twin_variants as tv
    saved, scene, believed = (1.0, 0.0, 0.0), (5.0, 5.0, 0.3), (1.02, 0.01, 0.005)
    for vel in (None, (0.001, 0.0, 0.0), (0.586, 0.0, 0.0), (0.0, 0.0, 0.4), (0.3, 0.1, -0.2)):
        for mode in ("zero", "poses"):
            r = tv.twin_init_target(saved, scene, believed, vel, vel_mode=mode)
            assert r["zero_joints"] is True, (vel, mode)
    # moving with poses the base velocity stays the estimated one
    r = tv.twin_init_target(saved, scene, believed, (0.586, 0.0, 0.0), vel_mode="poses")
    assert math_hypot(r["v_set_scene"]) == pytest.approx(0.586)
    # the rollout passes the target's zero_joints to set_state
    src = inspect.getsource(_ao()._rollout_once)
    assert 'zero_joints=tgt["zero_joints"]' in src


def math_hypot(v):
    return float(np.hypot(v[0], v[1]))


def test_f_configurazione_di_mola_a_tutti_i_ruoli(tmp_path):
    ao = _ao()
    a0 = SimpleNamespace(lidar_pose_x=None, initial_sigma=None)
    assert ao.mola_config_env(a0) == {}
    v = ao.VictimRunner("s.sh", tmp_path / "l", ros_args=None)
    t = ao.AttackerRunner("s.sh", tmp_path / "l", ros_args=None)
    assert v.extra_env == {"ROS_ARGS": ao.VICTIM_ROS_ARGS}           # default unchanged
    assert t.extra_env == {"ROS_ARGS": ao.ATTACKER_ROS_ARGS}
    v0 = ao.VictimRunner("s.sh", tmp_path / "l", ros_args=None, env=ao.mola_config_env(a0))
    assert v0.extra_env == v.extra_env
    a1 = SimpleNamespace(lidar_pose_x=-0.25, initial_sigma=0.04)
    cfg = ao.mola_config_env(a1)
    assert cfg == {"LIDAR_POSE_X": -0.25, "MOLA_INITIAL_SIGMA": 0.04}
    v1 = ao.VictimRunner("s.sh", tmp_path / "l", ros_args="X", env=cfg)
    t1 = ao.AttackerRunner("s.sh", tmp_path / "l", ros_args="Y", env=cfg)
    assert v1.extra_env == {**cfg, "ROS_ARGS": "X"} and t1.extra_env == {**cfg, "ROS_ARGS": "Y"}
    src = inspect.getsource(ao.main)
    assert "cfg_env = mola_config_env(args)" in src and "extra_env = dict(cfg_env)" in src
    # victim: cfg_env plus --victim-mola-env (empty by default); attacker and shadow: cfg_env
    assert src.count("env={**cfg_env, **victim_env}") == 1
    assert src.count("env=cfg_env") == 2


def test_g_costo_dell_impronta_con_log_da_10000_righe(tmp_path, capsys):
    from src.optimization import twin_variants as tv
    log = tmp_path / "mola_w0_e00.log"
    righe = []
    for i in range(5000):
        righe.append(TIA.riga(i, 1000.0 + 0.1 * i))
        righe.append(f"[12:00:00.0000|DEBUG|LidarOdometry] Since last KF: dist=0.050 m rotation=0.0 deg "
                     f"updateLocalMap={'YES' if i % 4 == 0 else 'NO'} updateSimpleMap=NO\n")
    log.write_text("".join(righe))
    n_righe = sum(1 for _ in open(log))
    tempi = []
    for _ in range(5):
        t = time.perf_counter()
        fp = tv.start_fingerprint_from_log(log, 1000.0 + 0.1 * 2499)
        tempi.append(time.perf_counter() - t)
    med = statistics.median(tempi) * 1e3
    with capsys.disabled():
        print(f"\n[costo impronta] log di {n_righe} righe ({log.stat().st_size} B): mediana su 5 letture "
              f"{med:.1f} ms (min {min(tempi)*1e3:.1f}, max {max(tempi)*1e3:.1f})")
    assert n_righe == 10000
    assert fp["printf_rows_upto"] == 2500 and fp["kf_lines_upto"] == 2500 and fp["kf_updates"] == 625
    assert med < 1000.0


def test_h_replay_gate_rifiutato_con_replay():
    from src.optimization import twin_variants as tv
    base = dict(twin_state="replay", victim="continuous", twin_align="attacker",
                drift_reset_on="genome", attack_from="spawn", twin_init="true")
    assert tv.check_twin_args(SimpleNamespace(**base, replay_gate=0)) is None
    err = tv.check_twin_args(SimpleNamespace(**base, replay_gate=3))
    assert err and "--replay-gate" in err
    for ts in ("clean", "victim", "maprestore"):
        assert tv.check_twin_args(SimpleNamespace(**dict(base, twin_state=ts), replay_gate=3)) is None
    # main rejects with rc 2 before touching Isaac
    src = inspect.getsource(_ao().main)
    assert "err = check_random_first_args(args) or tv.check_twin_args(args)" in src


def test_frame_log_nuvola_vuota_e_nessuna_sovrascrittura(tmp_path):
    pn = _pn()
    f = tmp_path / "frame_log.csv"
    a = pn.FrameLog(f, flush_s=0.05)
    n = _nodo(pn, a)
    n._cloud_cb(_nuvola(pn, 10.0, 0))           # empty cloud: 19-field row
    a.close()
    rows = list(csv.reader(open(f)))
    assert len(rows) == 2 and len(rows[1]) == 19 and rows[1][2] == "10.0" and rows[1][8] == "0"
    b = pn.FrameLog(f, flush_s=0.05)            # node restarted on the same output
    c = pn.FrameLog(f, flush_s=0.05)
    b.close(); c.close()
    assert b.path.name == "frame_log_1.csv" and c.path.name == "frame_log_2.csv"
    assert len(list(csv.reader(open(f)))) == 2  # the first log was not touched
