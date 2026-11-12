#!/usr/bin/env python3
"""
Twin variants, without ROS or Isaac: consistent replay, isolated maprestore, keep drift,
start from the attacker estimate, uniform use_sim_time, fingerprint and alignment.

    python3 -m pytest tests/test_gemello_2E.py -q
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimization import twin_variants as tv  # noqa: E402

VERIFICA_A_LOG = Path.home() / ("mola-adversarial-nsga3/data/attack/campaign/straight/"
                                "none_g1_sel2_warm_nomreps1_verifica_A/rep_01/logs/mola_w0_nominal.log")


def _ao():
    from src.optimization import attack_orchestrator as ao
    return ao


def _main(monkeypatch, argv):
    ao = _ao()
    monkeypatch.setattr(sys, "argv", ["attack_orchestrator.py"] + argv)
    return ao.main()


def riga_printf(step, t, sigma):
    ts = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y/%m/%d,%H:%M:%S.%f")
    return (f"[LidarOdometry] pathStep={step} timestamp={ts} goodness=0.990 minRequired=0.200 "
            f"iters=1 termReason=IterTermReason::Stalled isGood=1 "
            f"adapt_thres_sigma={sigma:.4f} consecutive_bad=0\n")


def riga_kf(yes):
    return (f"[11:31:30.6983|DEBUG|LidarOdometry] Since last KF: dist=0.000 m rotation=0.0 deg "
            f"updateLocalMap={'YES' if yes else 'NO'} updateSimpleMap=NO\n")


# ---------------------------------------------------------------------------
# Options: defaults unchanged and rejected combinations
# ---------------------------------------------------------------------------

def test_default_delle_opzioni_nuove():
    ao = _ao()
    import argparse
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    for opt, default in (("--replay-frame", '"twin"'), ("--twin-init", '"true"'),
                         ("--uniform-sim-time", '"off"')):
        i = src.index(f'"{opt}"')
        assert f"default={default}" in src[i:i + 200], opt
    i = src.index('"--twin-state"')
    assert 'default="clean"' in src[i:i + 200]
    i = src.index('"--drift-reset-on"')
    assert 'default="genome"' in src[i:i + 200]
    assert tv.check_twin_args(argparse.Namespace()) is None
    assert ao.TWIN_STATE["mode"] == "clean"


@pytest.mark.parametrize("argv", [
    ["--twin-state", "replay", "--twin-align", "true"],
    ["--twin-state", "replay", "--victim", "restart"],
    ["--drift-reset-on", "keep", "--attack-from", "motion"],
    ["--twin-init", "attacker_est", "--twin-align", "true"],
])
def test_main_rifiuta_combinazioni(monkeypatch, capsys, argv):
    assert _main(monkeypatch, argv) == 2
    assert "ERRORE" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Consistent replay
# ---------------------------------------------------------------------------

def test_frame_del_rigioco():
    origin = (1.0, 2.0, math.pi / 2)
    rec = {"status": "ok", "twin_est_raw": [0.5, 0.0, 0.1], "history_truncated": False}
    fr, src, why = tv.replay_frame("replay", "twin", rec, origin, (9.0, 9.0, 0.0))
    assert src == "twin_replay_end" and why is None
    assert fr == pytest.approx((1.0, 2.5, math.pi / 2 + 0.1))
    # truncated history: fall back to the attacker estimate, with the reason
    fr, src, why = tv.replay_frame("replay", "twin", dict(rec, history_truncated=True), origin, (9.0, 9.0, 0.0))
    assert fr == (9.0, 9.0, 0.0) and src == "attacker_est" and "troncata" in why
    fr, src, why = tv.replay_frame("replay", "attacker", rec, origin, (9.0, 9.0, 0.0))
    assert fr == (9.0, 9.0, 0.0) and src == "attacker_est" and why is None
    fr, src, why = tv.replay_frame("replay", "twin", {"status": "stale_pose"}, origin, (9.0, 9.0, 0.0))
    assert src == "attacker_est" and "stale_pose" in why
    assert tv.replay_frame("victim", "twin", rec, origin, None) == (None, None, None)


class _NodoRigioco:
    """Fake node: the replay brings the twin estimate to ests[i]."""

    def __init__(self, ests, rec_n=5, rec_total=5):
        self.ests = list(ests)
        self.victim_rec_n = rec_n
        self.victim_rec_total = rec_total
        self.replay_first_stamp = 1.0
        self.est_raw = (7.0, 7.0, 0.0)          # estimate left by the previous twin
        self.est_stamp_raw = 99.0
        self.est = None; self.est_t = 0.0; self.est_origin = None; self.est_stamp = float("nan")
        self.pose_quality = float("nan")
        self.calls = 0

    def twin_replay(self, last_s, rate_hz):
        assert self.est_raw is None          # cleared before the replay
        e = self.ests[min(self.calls, len(self.ests) - 1)]
        self.calls += 1
        self.est_raw = e
        self.est_stamp_raw = 1.4
        return self.victim_rec_n, 1.4, 0.5

    def spin(self, s):
        pass

    def drain(self):
        pass


class _Mola:
    log_path = None

    def wait_ready(self, check=None):
        return True

    def stop(self):
        pass

    def spawn(self, tag):
        pass


def test_rigioco_replay_senza_informazioni_della_vittima(monkeypatch):
    ao = _ao()
    monkeypatch.setitem(ao.TWIN_STATE, "mode", "replay")
    monkeypatch.setitem(ao.TWIN_STATE, "victim", None)      # no victim: must work
    monkeypatch.setitem(ao.TWIN_STATE, "last_s", 0.0)
    monkeypatch.setitem(ao.TWIN_STATE, "rate_hz", 10.0)
    node = _NodoRigioco([(0.5, 0.0, 0.0)])
    rec = ao._replay_to_twin(node, _Mola(), "w1_e0")
    assert rec["status"] == "ok" and rec["n"] == 5
    assert rec["first_stamp"] == 1.0 and rec["last_stamp"] == 1.4
    assert rec["history_truncated"] is False
    assert not any(k.startswith("diff") or k.startswith("victim_est") for k in rec)
    # with the victim present its estimate is only a lab measurement
    v = SimpleNamespace(last_est_end=(0.6, 0.0, 0.0), last_quality=0.9)
    monkeypatch.setitem(ao.TWIN_STATE, "victim", v)
    rec = ao._replay_to_twin(_NodoRigioco([(0.5, 0.0, 0.0)]), _Mola(), "w1_e1")
    assert rec["lab_diff_cm"] == pytest.approx(10.0) and "diff_cm" not in rec
    # full queue: truncated history
    rec = ao._replay_to_twin(_NodoRigioco([(0.5, 0.0, 0.0)], rec_n=1200, rec_total=1300), _Mola(), "w9")
    assert rec["history_truncated"] is True


def test_rigioco_victim_invariato(monkeypatch):
    ao = _ao()
    v = SimpleNamespace(last_est_end=(0.6, 0.0, 0.0), last_quality=0.9)
    monkeypatch.setitem(ao.TWIN_STATE, "mode", "victim")
    monkeypatch.setitem(ao.TWIN_STATE, "victim", v)
    monkeypatch.setitem(ao.TWIN_STATE, "last_s", 0.0)
    monkeypatch.setitem(ao.TWIN_STATE, "rate_hz", 10.0)
    node = _NodoRigioco([(0.5, 0.0, 0.0)])
    node.est_raw = None
    rec = ao._replay_to_twin(node, _Mola(), "w1_e0")
    assert rec["diff_cm"] == pytest.approx(10.0) and rec["victim_est_end"] == [0.6, 0.0, 0.0]
    # without a victim estimate the victim mode does not replay
    monkeypatch.setitem(ao.TWIN_STATE, "victim", SimpleNamespace(last_est_end=None))
    assert ao._replay_to_twin(_NodoRigioco([(0.5, 0.0, 0.0)]), _Mola(), "w0") is None


def test_guardia_del_replay_usa_solo_i_gemelli(monkeypatch):
    ao = _ao()
    for k, v in {"mode": "replay", "victim": None, "last_s": 0.0, "rate_hz": 10.0,
                 "guard": True, "guard_cm": 5.0, "window": 3, "ref": {}}.items():
        monkeypatch.setitem(ao.TWIN_STATE, k, v)
    # first candidate: two agreeing replays
    node = _NodoRigioco([(0.50, 0.0, 0.0), (0.51, 0.0, 0.0)])
    rec = ao._replay_with_guard(node, _Mola(), "w3_e0")
    assert rec["replay_attempts"] == 2 and rec["replay_outlier"] is False
    # second candidate out of consensus twice: excluded
    node = _NodoRigioco([(0.80, 0.0, 0.0), (0.80, 0.0, 0.0)])
    rec = ao._replay_with_guard(node, _Mola(), "w3_e1")
    assert rec["replay_outlier"] is True and rec["replay_attempts"] == 2
    ok, why = ao._twin_misaligned(SimpleNamespace(last_replay=rec),
                                  SimpleNamespace(twin_gap0_max=0.0), None)
    assert ok is True and "consenso" in why


def test_status_del_nodo_registrazione_e_keep():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    import importlib.util
    spec = importlib.util.spec_from_file_location("trs", ROOT / "tests" / "test_registro_serieB.py")
    trs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trs)
    n, st = trs._nodo_finto(False)
    pn.PerturbationNode._publish_status(n)
    assert st[0]["victim_rec_total"] is None and st[0]["drift_keep"] is False
    n._victim_rec_total, n._replay_first_stamp, n._drift_keep = 1300, 12.3, True
    pn.PerturbationNode._publish_status(n)
    assert st[1]["victim_rec_total"] == 1300 and st[1]["replay_first_stamp"] == 12.3
    assert st[1]["drift_keep"] is True
    ao = _ao()
    a = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0, node_gen_limits=None)
    ao.AttackNode._status_cb(a, SimpleNamespace(data=json.dumps(st[1])))
    assert a.victim_rec_total == 1300 and a.replay_first_stamp == 12.3 and a.st_drift_keep is True
    assert a.st_drift_xy_cm == 0.0 and a.st_victim_drift_xy_cm == 0.0


def test_registrazione_e_primo_stamp_nel_nodo():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    from collections import deque
    log = SimpleNamespace(info=lambda *a: None, warn=lambda *a: None)
    n = SimpleNamespace(_replay_active=False, _victim_rec=deque([(1.0, "f", None), (1.1, "f", None),
                                                                 (1.2, "f", None)], maxlen=10),
                        _replay_total=0, _replay_n=0, get_logger=lambda: log, _replay_run=lambda r, h: None)
    import threading
    orig = threading.Thread
    threading.Thread = lambda target, args, daemon: SimpleNamespace(start=lambda: None)
    try:
        pn.PerturbationNode._twin_replay_cb(n, SimpleNamespace(data=json.dumps({"last_s": 0.15})))
    finally:
        threading.Thread = orig
    assert n._replay_first_stamp == 1.1 and n._replay_total == 2


# ---------------------------------------------------------------------------
# Fingerprint of the initial state
# ---------------------------------------------------------------------------

def test_impronta_dal_log_sintetico(tmp_path):
    p = tmp_path / "mola_w1_e0.log"
    righe = ["[DEBUG] First pointcloud: skipping ICP and directly adding to local map.\n"]
    for i, (t, s, kf) in enumerate([(10.0, 0.1975, True), (10.1, 0.1950, False),
                                     (10.2, 0.1925, True), (10.3, 0.1900, False)]):
        righe += [riga_printf(i + 2, t, s), riga_kf(kf)]
    p.write_text("".join(righe))
    fp = tv.start_fingerprint_from_log(p, 10.2)
    assert fp == {"sigma": 0.1925, "printf_rows_upto": 3, "kf_updates": 2, "kf_lines_upto": 3}
    # no DEBUG lines: keyframe None
    q = tmp_path / "solo_printf.log"
    q.write_text("".join(riga_printf(i, 10 + i / 10, 0.2) for i in range(3)))
    fp = tv.start_fingerprint_from_log(q, 10.1)
    assert fp["sigma"] == 0.2 and fp["kf_updates"] is None
    # no log or no stamp
    assert tv.start_fingerprint_from_log(None, 10.0)["sigma"] is None
    assert tv.start_fingerprint_from_log(p, None)["sigma"] is None


@pytest.mark.skipif(not VERIFICA_A_LOG.is_file(), reason="log di verifica_A assente")
def test_impronta_dal_log_vero():
    # verifica_A, w0_nominal: 63 printf lines and 63 updateLocalMap lines (DEBUG);
    # first printf at 13.9 s with sigma 0.1975 and updateLocalMap=YES.
    fp = tv.start_fingerprint_from_log(VERIFICA_A_LOG, 13.9)
    assert fp["printf_rows_upto"] == 1 and fp["sigma"] == pytest.approx(0.1975)
    assert fp["kf_lines_upto"] == 1 and fp["kf_updates"] == 1
    tutto = tv.start_fingerprint_from_log(VERIFICA_A_LOG, 1e9)
    assert tutto["printf_rows_upto"] == 63 and tutto["kf_lines_upto"] == 63


def test_impronta_completata_a_rollout_chiuso(tmp_path):
    ao = _ao()
    p = tmp_path / "mola_w1_e0.log"
    p.write_text(riga_printf(2, 5.0, 0.18) + riga_kf(True))
    node = SimpleNamespace(last_twin_fp={"est_stamp": 5.0, "source": "prima_posa_dopo_avvio"})
    ao._note_twin_fp(node, SimpleNamespace(log_path=p))
    assert node.last_twin_fp["sigma"] == 0.18 and node.last_twin_fp["kf_updates"] == 1
    assert ao._twin_start_fields(SimpleNamespace(last_twin_fp=None, last_twin_align=None)) == \
        {"twin_start": None, "twin_alignment": None, "mola_stop": None}


# ---------------------------------------------------------------------------
# Kept drift (keep)
# ---------------------------------------------------------------------------

BASE_GEN = (Path(os.environ.get("SERIEB_LAVORO", str(Path.home() / "mola_lavoro"))) / "base/src/perturbations/perturbation_generator.py")


def _genoma(drift_on, rng):
    g = rng.uniform(-1, 1, 17)
    g[14] = 1.0 if drift_on else -1.0          # temporal_drift_strength 1 or 0
    return g


@pytest.mark.skipif(not BASE_GEN.is_file(), reason="copia della base assente")
def test_generatore_senza_keep_identico_alla_base():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_base", BASE_GEN)
    gb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gb)
    from src.perturbations.perturbation_generator import PerturbationGenerator
    rng = np.random.default_rng(5)
    cloud = rng.uniform(-8, 8, (2000, 4))
    a, b = PerturbationGenerator(max_drift_lag_s=0.5), gb.PerturbationGenerator(max_drift_lag_s=0.5)
    for k, on in enumerate([True, True, False, True, False]):
        g = _genoma(on, rng)
        pa, pb = a.encode_perturbation(g), b.encode_perturbation(g)
        for f in range(4):
            oa = a.apply_perturbation(cloud, pa, seed=100 + k * 10 + f)
            ob = b.apply_perturbation(cloud, pb, seed=100 + k * 10 + f)
            assert np.array_equal(oa, ob)
        if k == 2:
            a.reset_temporal_state(); b.reset_temporal_state()


def test_keep_prosegue_con_genoma_senza_deriva():
    from src.perturbations.perturbation_generator import PerturbationGenerator
    rng = np.random.default_rng(7)
    cloud = rng.uniform(-8, 8, (500, 4))
    gen = PerturbationGenerator()
    p_on = gen.encode_perturbation(_genoma(True, rng))
    for f in range(5):
        gen.apply_perturbation(cloud, p_on, seed=f)
    d5 = gen.accumulated_drift
    assert np.linalg.norm(d5) > 0
    p_off = gen.encode_perturbation(_genoma(False, rng))
    # without keep: with a genome without drift the drift is not applied (jump to zero)
    gen.keep_drift = False
    st = gen.accumulated_drift.copy()
    gen.apply_perturbation(cloud, p_off, seed=9)
    assert np.array_equal(gen.accumulated_drift, st)
    # with keep: decays by 0.98 per frame and is still applied
    gen.keep_drift = True
    gen.apply_perturbation(cloud, p_off, seed=9)
    assert gen.accumulated_drift == pytest.approx(0.98 * d5)
    gen.set_temporal_state(np.zeros(3))
    assert not gen._drift_carried()


def _nodo_keep(keep, vdrift):
    from src.perturbations.perturbation_generator import PerturbationGenerator
    log = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)
    n = SimpleNamespace(generator=PerturbationGenerator(), victim_generator=PerturbationGenerator(),
                        genome_id=0, victim_genome_id=0, _drift_keep=keep, _test_seed=0,
                        get_logger=lambda: log)
    n.victim_generator.set_temporal_state(vdrift)
    n.generator.set_temporal_state([0.3, 0.0, 0.0])      # drift of the previous candidate
    return n


def test_nodo_keep_gemello_parte_dalla_deriva_della_vittima():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    g = _genoma(True, np.random.default_rng(1))
    for keep, atteso_twin, atteso_vitt in ((False, [0, 0, 0], [0, 0, 0]),
                                            (True, [0.05, -0.02, 0.0], [0.05, -0.02, 0.0])):
        n = _nodo_keep(keep, [0.05, -0.02, 0.0])
        pn.PerturbationNode._set_genome(n, g, "prova")
        assert n.generator.accumulated_drift == pytest.approx(atteso_twin)
        pn.PerturbationNode._victim_genome_cb(n, SimpleNamespace(data=list(g)))
        assert n.victim_generator.accumulated_drift == pytest.approx(atteso_vitt)
    n = _nodo_keep(False, [0, 0, 0])
    pn.PerturbationNode._drift_keep_cb(n, SimpleNamespace(data=True))
    assert n._drift_keep and n.generator.keep_drift and n.victim_generator.keep_drift


def test_main_keep_accettato_con_spawn(monkeypatch):
    import argparse
    ns = argparse.Namespace(drift_reset_on="keep", attack_from="spawn")
    assert tv.check_twin_args(ns) is None


# ---------------------------------------------------------------------------
# Twin-victim alignment at the start of the window
# ---------------------------------------------------------------------------

def test_allineamento_puro():
    r = tv.twin_alignment((1.0, 0.0, 0.0), (1.0, 0.03, 0.0), (2.0, 0.0, 0.01), (2.0, 0.04, 0.0))
    assert r["true_gap_cm"] == pytest.approx(3.0) and r["est_gap_cm"] == pytest.approx(4.0)
    assert r["est_gap_deg"] == pytest.approx(math.degrees(0.01), abs=1e-3)
    r = tv.twin_alignment((1.0, 0.0, 0.0), None, None, (2.0, 0.0, 0.0))
    assert r["true_gap_cm"] is None and r["est_gap_cm"] is None


@pytest.mark.parametrize("mode,src,atteso_frame", [
    ("replay", "fine_rigioco", "origine dell'attaccante"),
    ("victim", "fine_rigioco", "origine della vittima"),
    ("clean", "prima_posa_dopo_avvio", None),
])
def test_allineamento_per_modo(monkeypatch, mode, src, atteso_frame):
    ao = _ao()
    v = SimpleNamespace(run_origin=(0.0, 0.0, 0.0))
    for k, val in {"mode": mode, "victim": v, "attacker_origin": (0.0, 0.0, 0.0),
                   "saved": (1.0, 0.0, 0.0), "attacker_belief": (1.005, 0.0, 0.0), "init": "true"}.items():
        monkeypatch.setitem(ao.TWIN_STATE, k, val)
    fp = {"source": src, "est_raw": [1.01, 0.0, 0.0]}
    r = ao._twin_alignment_record(None, (1.002, 0.0, 0.0), fp)
    assert r["true_gap_cm"] == pytest.approx(0.2)
    assert r["twin_est_frame"] == atteso_frame
    if atteso_frame is None:
        assert r["est_gap_cm"] is None          # clean twin: no common frame
    else:
        assert r["est_gap_cm"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ROS_ARGS (remap of /initialpose, use_sim_time) and consistency
# ---------------------------------------------------------------------------

def test_ros_args_default_invariati():
    ao = _ao()
    assert tv.mola_ros_args("victim") == ao.VICTIM_ROS_ARGS
    assert tv.mola_ros_args("attacker") == ao.ATTACKER_ROS_ARGS
    assert tv.mola_ros_args("twin") is None          # twin: launch command line


def test_ros_args_maprestore_e_sim_time():
    tw = json.loads(tv.mola_ros_args("twin", "off", isolate_initialpose=True))
    assert tw == ["--ros-args", "-p", "use_sim_time:=true", "-r", "/initialpose:=/twin/initialpose"]
    assert json.loads(tv.mola_ros_args("twin", "false")) == ["--ros-args", "-p", "use_sim_time:=false"]
    assert json.loads(tv.mola_ros_args("twin", "true")) == ["--ros-args", "-p", "use_sim_time:=true"]
    assert json.loads(tv.mola_ros_args("victim", "true")) == ["--ros-args", "-r", "__ns:=/victim",
                                                             "-p", "use_sim_time:=true"]
    assert json.loads(tv.mola_ros_args("attacker", "false"))[-1] == "use_sim_time:=false"


def test_publisher_di_initialpose_privato():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert 'PoseWithCovarianceStamped, "/initialpose"' not in src
    assert "PoseWithCovarianceStamped, tv.TWIN_INITIALPOSE" in src


def test_coerenza_con_il_threat_model():
    c = tv.coherence_record("maprestore")
    assert c["twin_state_coerente_threat_model"] is False
    assert tv.coherence_record("victim")["twin_state_coerente_threat_model"] is False
    assert tv.coherence_record("replay")["twin_state_coerente_threat_model"] is True
    assert tv.coherence_record("clean")["twin_state_coerente_threat_model"] is True
    assert tv.coherence_record("clean")["twin_init_coerente_threat_model"] is False
    assert tv.coherence_record("clean", "attacker_est")["twin_init_coerente_threat_model"] is True


def test_config_in_history():
    ao = _ao()
    import argparse
    a = argparse.Namespace(twin_state="maprestore", replay_frame="twin", twin_init="true",
                           uniform_sim_time="off", drift_reset_on="genome", twin_align="attacker")
    rec = ao.twin_config_fields(a)
    assert rec["twin_state_coerente_threat_model"] is False and rec["replay_frame"] is None


# ---------------------------------------------------------------------------
# Twin from the attacker estimate
# ---------------------------------------------------------------------------

def test_velocita_dalle_pose():
    buf = [(10.0 + 0.1 * i, 0.5 * 0.1 * i, 0.0, 0.2 * 0.1 * i) for i in range(10)]
    vx, vy, wz, n = tv.velocity_from_poses(buf)
    assert (vx, vy, wz) == pytest.approx((0.5, 0.0, 0.2)) and n == 6     # last 0.5 s
    assert tv.velocity_from_poses(buf[:1]) is None and tv.velocity_from_poses([]) is None
    # yaw across +-pi without a jump
    buf = [(0.0, 0, 0, math.pi - 0.01), (0.1, 0, 0, -math.pi + 0.01)]
    assert tv.velocity_from_poses(buf)[2] == pytest.approx(0.2)


def test_bersaglio_del_gemello():
    # odom and scene rotated by 90 deg and translated: the scene gets the same offset
    saved, scene = (1.0, 0.0, 0.0), (5.0, 5.0, math.pi / 2)
    believed = (1.02, 0.01, 0.005)
    r = tv.twin_init_target(saved, scene, believed, (0.001, 0.0, 0.0))
    assert r["target_scene"] == pytest.approx((5.0 - 0.01, 5.02, math.pi / 2 + 0.005))
    assert r["rule"] == "robot fermo: zero" and r["zero_joints"] is True
    assert r["v_set_scene"] == [0.0, 0.0] and r["w_set"] == 0.0
    # moving: velocity rotated into the scene, joints zeroed
    r = tv.twin_init_target(saved, scene, believed, (0.586, 0.0, 0.0))
    assert r["v_set_scene"] == pytest.approx([0.0, 0.586]) and r["zero_joints"] is True
    assert "pose consecutive" in r["rule"]
    assert tv.twin_init_target(saved, scene, believed, None)["rule"].startswith("velocita' assente")


def test_bersaglio_con_origine_dell_attaccante():
    ao = _ao()
    from collections import deque
    node = SimpleNamespace(attacker_buf=deque([(t / 10, 0.0, -0.06 * t / 10, 0.0) for t in range(6)]))
    att = SimpleNamespace(run_origin=(0.0, 0.0, -math.pi / 2))      # rotated MOLA frame
    r = ao.attacker_init_target(node, att, (0, 0, 0), (0, 0, 0), (0, 0, 0), "poses")
    assert r["v_att"] == pytest.approx([-0.06, 0.0], abs=1e-9) and r["vel_poses_n"] == 6
    assert r["rule"].startswith("da pose") and r["zero_joints"] is True
    # default zero: estimated velocity recorded, set to zero
    r = ao.attacker_init_target(node, att, (0, 0, 0), (0, 0, 0), (0, 0, 0))
    assert r["v_att"] == pytest.approx([-0.06, 0.0], abs=1e-9)
    assert r["v_set_scene"] == [0.0, 0.0] and r["zero_joints"] is True and "zero" in r["rule"]


def test_client_set_state_e_caps(monkeypatch):
    ao = _ao()
    c = ao.IsaacClient.__new__(ao.IsaacClient)
    sent = []
    monkeypatch.setattr(c, "call", lambda cmd, **kw: sent.append((cmd, kw)) or {"pose": [1, 2, 3], "caps": ["set_state"]},
                        raising=False)
    assert c.set_state(1, 2, 0.1, 0.5, 0.0, 0.0, zero_joints=False) == [1, 2, 3]
    assert sent[0] == ("set_state", {"x": 1.0, "y": 2.0, "yaw": 0.1, "vx": 0.5, "vy": 0.0,
                                     "wz": 0.0, "zero_joints": False})
    assert c.caps() == ["set_state"]

    def vecchio(cmd, **kw):
        raise RuntimeError("Isaac 'caps': comando sconosciuto: caps")
    monkeypatch.setattr(c, "call", vecchio, raising=False)
    assert c.caps() == []


def test_server_ha_set_state():
    src = (ROOT / "isaac/run_isaac_standalone.py").read_text()
    assert 'if op == "set_state":' in src and '"caps": ["set_state"]' in src
