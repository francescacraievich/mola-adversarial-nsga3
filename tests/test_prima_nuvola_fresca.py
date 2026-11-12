"""First-cloud filter after a restore (--fresh-cloud-after-restore).

Pure tests: no ROS node created (no rclpy.init), no topic.
"""
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pn = pytest.importorskip("src.nodes.perturbation_node")


def _gate(margin=0.1, max_drop=5):
    return pn.FreshCloudGate(margin, max_drop)


def test_disarmato_inoltra_tutto():
    g = _gate()
    assert not g.armed
    for s in (0.0, 35.9, 1e6):
        assert g.check(s) == (True, None)
    assert g.dropped_total == 0


def test_pausa_dentro_la_scansione_3_3():
    # recorded opening (line 316): /clock stopped at 35.8833, k0 with stamp 35.9, k1 36.0
    g = _gate()
    g.arm(35.8833, "clock")
    assert g.check(35.9) == (False, "scartata")
    assert g.check(36.0) == (True, "prima_fresca")
    assert not g.armed and g.dropped == 1
    assert g.check(36.1) == (True, None)


def test_pausa_sul_bordo_della_scansione():
    # pause at the end instant of a scan: the next one contains the step t_pause + 1/60,
    # still in the old pose, and must be dropped
    g = _gate()
    g.arm(35.9, "clock")
    assert g.check(36.0)[0] is False
    assert g.check(36.1) == (True, "prima_fresca")


def test_k0_a_piu_0_2():
    # k0 = last twin stamp + 0.2 s: already beyond the threshold
    g = _gate()
    g.arm(21.9833, "clock")
    assert g.check(22.1) == (True, "prima_fresca")
    assert g.dropped == 0


def test_nuvole_rimaste_di_prima_della_pausa_scartate():
    g = _gate()
    g.arm(50.0, "clock")
    assert g.check(49.9)[0] is False
    assert g.check(50.0)[0] is False
    assert g.check(50.1)[0] is False
    assert g.check(50.2) == (True, "prima_fresca")
    assert g.dropped == 3 and g.dropped_total == 3


def test_tetto_degli_scarti():
    # clock restarted (Isaac restarted): after max_drop drops the filter disarms
    g = _gate(max_drop=3)
    g.arm(500.0, "clock")
    assert [g.check(s)[0] for s in (1.0, 1.1, 1.2)] == [False] * 3
    assert g.check(1.3) == (True, "tetto_scarti")
    assert not g.armed and g.n_forced == 1
    assert g.check(1.4) == (True, None)


def test_riarmo_azzera_il_conteggio():
    g = _gate()
    g.arm(10.0, "clock")
    g.check(10.05)
    g.arm(20.0, "clock")
    assert g.dropped == 0 and g.dropped_total == 1 and g.n_arm == 2
    assert g.threshold == pytest.approx(20.1)


def _nodo_finto(clock=None, stamp_in=None):
    log = SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)
    return SimpleNamespace(_fresh_on=True, _fresh_gate=_gate(), _fresh_lock=pn.threading.Lock(),
                           _fresh_seq=None, _fresh_last_clock=clock, _fresh_last_stamp_in=stamp_in,
                           get_logger=lambda: log)


def test_arm_dal_clock_e_ripetizione_ignorata():
    n = _nodo_finto(clock=35.8833, stamp_in=35.8)
    pn.PerturbationNode._fresh_arm_cb(n, SimpleNamespace(data=7))
    assert n._fresh_gate.base_src == "clock"
    assert n._fresh_gate.threshold == pytest.approx(35.9833)
    assert n._fresh_gate.check(36.0) == (True, "prima_fresca")
    # same sequence arriving after play, with /clock advanced: no re-arm
    n._fresh_last_clock = 40.0
    pn.PerturbationNode._fresh_arm_cb(n, SimpleNamespace(data=7))
    assert not n._fresh_gate.armed and n._fresh_seq == 7


def test_arm_senza_clock_dall_ultimo_stamp():
    # without /clock: base = last received stamp (scan finished before the pause)
    n = _nodo_finto(clock=None, stamp_in=20.9)
    pn.PerturbationNode._fresh_arm_cb(n, SimpleNamespace(data=1))
    assert n._fresh_gate.base_src == "stamp_in"
    assert n._fresh_gate.check(21.0)[0] is False
    assert n._fresh_gate.check(21.1) == (True, "prima_fresca")


def test_arm_senza_riferimenti_non_arma():
    n = _nodo_finto()
    pn.PerturbationNode._fresh_arm_cb(n, SimpleNamespace(data=1))
    assert not n._fresh_gate.armed and n._fresh_seq == 1


class _LogFinto:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


def _msg(stamp, w=45000):
    sec = int(stamp)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))),
                           width=w, height=1)


def test_cloud_cb_scarta_e_registra():
    n = _nodo_finto()
    n._fresh_gate.arm(35.8833, "clock")
    n._frame_log = _LogFinto()
    n.n_in = 0
    n.frame_idx = 12
    n.enabled = False
    n.victim_stream = True
    n.victim_attack = True
    n._out_q = []      # a put() would fail the test: the cloud must not go out
    pn.PerturbationNode._cloud_cb(n, _msg(35.9))
    assert n.n_in == 1 and n.frame_idx == 12 and n._out_q == []
    row = n._frame_log.rows[0]
    assert len(row) == len(pn.FrameLog.COLUMNS) + len(pn.FreshCloudGate.COLUMNS)
    assert row[-5] == "scartata_ripristino" and row[-4] == pytest.approx(35.9833)
    assert row[-3] == pytest.approx(35.8833) and row[-2] == "clock" and row[-1] is None
    assert row[2] == pytest.approx(35.9) and row[3] == 12 and row[4] is None and row[8] == 45000
    assert n._fresh_last_stamp_in == pytest.approx(35.9)


def test_arm_registra_la_base_nel_registro():
    n = _nodo_finto(clock=35.8833, stamp_in=35.8)
    n._frame_log = _LogFinto()
    pn.PerturbationNode._fresh_arm_cb(n, SimpleNamespace(data=7))
    pn.PerturbationNode._fresh_arm_cb(n, SimpleNamespace(data=7))     # repetition: no row
    vuoto = _nodo_finto()
    vuoto._frame_log = _LogFinto()
    pn.PerturbationNode._fresh_arm_cb(vuoto, SimpleNamespace(data=8))
    assert len(n._frame_log.rows) == 1 and len(vuoto._frame_log.rows) == 1
    row = n._frame_log.rows[0]
    assert len(row) == len(pn.FrameLog.COLUMNS) + len(pn.FreshCloudGate.COLUMNS)
    assert row[2] is None
    assert row[-5:-3] == ["armato", pytest.approx(35.9833)]
    assert row[-3:] == [pytest.approx(35.8833), "clock", 7]
    assert vuoto._frame_log.rows[0][-5:] == ["non_armato", None, None, None, 8]


def test_frame_log_colonne(tmp_path):
    a = pn.FrameLog(tmp_path / "a.csv")
    a.add([1.0, 2.0, 3.0])
    a.close()
    b = pn.FrameLog(tmp_path / "b.csv", extra_columns=("fresh_gate", "fresh_threshold"))
    b.add([1.0, 2.0, 3.0])
    b.close()
    la = (tmp_path / "a.csv").read_text().splitlines()
    lb = (tmp_path / "b.csv").read_text().splitlines()
    # without the option, header and rows unchanged
    assert la[0] == ",".join(pn.FrameLog.COLUMNS) and la[1].count(",") == len(pn.FrameLog.COLUMNS) - 1
    assert lb[0] == ",".join(pn.FrameLog.COLUMNS) + ",fresh_gate,fresh_threshold"
    assert lb[1].count(",") == len(pn.FrameLog.COLUMNS) + 1


def test_status_senza_opzione_senza_campi_nuovi():
    st = []
    n = SimpleNamespace(
        proc_times=[0.01], lat_times=[0.02], enabled=False, genome_id=1, n_in=1, n_out=1,
        _min_latency_ms=0.0, _chamfer=0.0, frame_idx=1, _max_drift_lag_s=float("inf"),
        generator=SimpleNamespace(accumulated_drift=pn.np.zeros(3)),
        victim_generator=SimpleNamespace(accumulated_drift=pn.np.zeros(3)),
        victim_stream=False, victim_attack=False, victim_genome_id=0, victim_frames=0,
        victim_frame_idx=0, _victim_offset=0.0, _victim_rec=[], _replay_seq=0, _replay_active=False,
        _replay_n=0, _replay_total=0, _replay_last_stamp=None, _twin_clock_on=False, _twin_offset=0.0,
        _perc=None, _perc_terms_on=False, _perc_terms=None, _tc=None,
        status_pub=SimpleNamespace(publish=lambda m: st.append(json.loads(m.data))),
        get_logger=lambda: SimpleNamespace(warn=lambda *a: None))
    pn.PerturbationNode._publish_status(n)
    assert not any(k.startswith("fresh_") for k in st[0])
    n._fresh_on = True
    n._fresh_seq = 3
    n._fresh_gate = _gate()
    pn.PerturbationNode._publish_status(n)
    assert st[1]["fresh_cloud"] is True and st[1]["fresh_seq"] == 3 and st[1]["fresh_armed"] is False


def test_orologio_della_vittima_con_k0_scartata():
    # _victim_header compresses the pause to 0.1 s. With k0 dropped the compressed pause
    # falls between the last cloud of the previous window and k1; intervals seen by the
    # victim stay 0.1 s, and the victim clock lags by 0.1 s per dropped cloud.
    def nodo():
        return SimpleNamespace(_victim_last_sim=12.0, _victim_last_clock=12.0, _victim_offset=0.0)

    def hdr(s):
        sec = int(s)
        return SimpleNamespace(frame_id="base_link",
                               stamp=SimpleNamespace(sec=sec, nanosec=int(round((s - sec) * 1e9))))

    st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
    prima = nodo()
    c0 = st(pn.PerturbationNode._victim_header(prima, hdr(35.9)))
    c1 = st(pn.PerturbationNode._victim_header(prima, hdr(36.0)))
    dopo = nodo()
    d1 = st(pn.PerturbationNode._victim_header(dopo, hdr(36.0)))
    d2 = st(pn.PerturbationNode._victim_header(dopo, hdr(36.1)))
    assert (c0, c1) == (pytest.approx(12.1), pytest.approx(12.2))
    assert (d1, d2) == (pytest.approx(12.1), pytest.approx(12.2))
    assert dopo._victim_offset - prima._victim_offset == pytest.approx(0.1)


def test_orchestratore_senza_opzione_non_pubblica():
    from src.optimization import attack_orchestrator as ao
    n = SimpleNamespace(fresh_cloud=False)
    assert ao.AttackNode.arm_fresh_cloud(n) is None
    assert not hasattr(n, "_fresh_pub")


def test_orchestratore_status_e_ordine():
    from src.optimization import attack_orchestrator as ao
    n = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0)
    ao.AttackNode._status_cb(n, SimpleNamespace(data=json.dumps(
        {"enabled": False, "victim_stream": False, "fresh_cloud": True, "fresh_seq": 42})))
    assert n.node_fresh_cloud is True and n.fresh_seq == 42
    ao.AttackNode._status_cb(n, SimpleNamespace(data=json.dumps({"enabled": False, "victim_stream": False})))
    assert n.node_fresh_cloud is None
    # after restore and before play, for victim and twin
    src = inspect.getsource(ao._rollout_once)
    i_r, i_a = src.index("isaac.restore()"), src.index("node.arm_fresh_cloud()")
    assert i_r < i_a < src.index("_victim_rollout(") < src.index("isaac.play()")
    g = inspect.getsource(ao._gate_replay_once)
    assert g.index("isaac.restore()") < g.index("node.arm_fresh_cloud()")


def _orch_finto(ok, fresh_cloud=True):
    n = SimpleNamespace(fresh_cloud=fresh_cloud, _fresh_pub=SimpleNamespace(publish=lambda m: None),
                        _fresh_seq_sent=10, fresh_seq=None)
    n._publish_confirmed = lambda pub, m, conf, what: ok
    return n


def test_esito_dell_arm_registrato():
    pytest.importorskip("std_msgs.msg")
    from src.optimization import attack_orchestrator as ao
    n = _orch_finto(False)
    assert ao.AttackNode.arm_fresh_cloud(n) is False
    assert n.last_fresh_arm["seq"] == 11 and n.last_fresh_arm["confirmed"] is False
    n._publish_confirmed = lambda pub, m, conf, what: True
    assert ao.AttackNode.arm_fresh_cloud(n) is True
    assert n.last_fresh_arm["seq"] == 12 and n.last_fresh_arm["confirmed"] is True
    assert (n.fresh_arm_n, n.fresh_arm_ok) == (2, 1)
    # per rollout: only with the option
    assert "fresh_arm" not in ao._twin_start_fields(SimpleNamespace(fresh_cloud=False, last_fresh_arm={"x": 1}))
    assert ao._twin_start_fields(n)["fresh_arm"] == n.last_fresh_arm
    # per window (applied segment) and per gate replay
    assert '"fresh_arm_victim": getattr(node, "last_victim_fresh_arm", None)' in inspect.getsource(ao.main)
    assert 'rec["fresh_arm"] = fresh_arm' in inspect.getsource(ao._gate_replay_once)
    assert "node.last_victim_fresh_arm = node.last_fresh_arm" in inspect.getsource(ao._rollout_once)


def test_controllo_all_avvio_nei_due_versi():
    from src.optimization import attack_orchestrator as ao

    import time

    def nodo(fresh):
        # slow, counted spin: a wait would show in the time and in the spin count
        n = SimpleNamespace(node_fresh_cloud=fresh, n_spin=0)

        def spin(s):
            n.n_spin += 1
            time.sleep(s)
        n.spin = spin
        return n

    def cronometra(n, want, **kw):
        t0 = time.perf_counter()
        r = ao.fresh_cloud_mismatch(n, want, **kw)
        return r, time.perf_counter() - t0

    # option off: no wait, with or without status (default timeout 10 s)
    for fresh in (None, False):
        n = nodo(fresh)
        r, dt = cronometra(n, False)
        assert r is None and n.n_spin == 0 and dt < 0.05
    n = nodo(True)
    r, dt = cronometra(n, False)
    assert "l'orchestratore no" in r and n.n_spin == 0 and dt < 0.05
    # option on: waits for the status up to the timeout
    assert cronometra(nodo(True), True)[0] is None
    n = nodo(None)
    r, dt = cronometra(n, True, timeout=0.3)
    assert "richiede il nodo" in r and n.n_spin >= 2 and dt >= 0.3
    # at start and again after victim.start
    src = inspect.getsource(ao.main)
    assert src.count("fresh_cloud_mismatch(node, args.fresh_cloud_after_restore)") == 2
    assert src.index("victim.start(") < src.rindex("fresh_cloud_mismatch(node, args.fresh_cloud_after_restore)")


def test_contatore_dei_restore():
    from src.optimization import attack_orchestrator as ao
    k = ao.restore_kind
    assert k("w2_nominal", False) == "nominale" and k("w2_nominal_r1", False) == "nominale"
    assert k("w2_e00", False) == "valutazione" and k("w0_advance", False) == "avanzamento"
    assert k("w2_reeval", False) == k("w2_sel1", False) == k("w2_sel1b", False) == "rivalutazione"
    assert k("w2_applied", True) == k("w2_victim", True) == "vittima"
    assert k("gate_twin", False) == "gate" and k("w2_e00r", False, retry=True) == "ripetuto"
    assert ao.restore_window("w12_e3") == 12 and ao.restore_window("rep_nominal") is None
    n = SimpleNamespace()
    for tipo, w in (("nominale", 0), ("valutazione", 0), ("rivalutazione", 0), ("vittima", 0),
                    ("nominale", 1), ("gate", 1)):
        ao.count_restore(n, tipo, w)
    c0, c1 = ao.restore_counts(n, 0), ao.restore_counts(n, 1)
    assert c0["finestra"] == 4 and c0["per_tipo"]["vittima"] == 1 and c0["totale"] == 6
    assert c1["per_tipo"] == {**{t: 0 for t in ao.RESTORE_KINDS}, "nominale": 1, "gate": 1}
    # the arm pairs with the last recorded restore
    pytest.importorskip("std_msgs.msg")
    o = _orch_finto(True)
    o.restore_log = n.restore_log
    ao.AttackNode.arm_fresh_cloud(o)
    assert o.last_fresh_arm["restore_i"] == 5 and o.last_fresh_arm["window"] == 1
    assert o.fresh_arm_log == [o.last_fresh_arm]
    # the count is right after the restore, before the arm, and only with the option
    for f in (ao._rollout_once, ao._gate_replay_once):
        s = inspect.getsource(f)
        assert s.index("isaac.restore()") < s.index("count_restore(") < s.index("node.arm_fresh_cloud()")
    assert '"restore_n": restore_counts(node, k)' in inspect.getsource(ao.main)
