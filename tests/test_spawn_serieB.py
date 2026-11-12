"""Victim never clean, deferred start and settling counted in clouds (--victim-never-clean,
--victim-start w0, --settle-clouds, --reopen-clouds).

Pure tests, no running ROS or processes: the perturbation node is built with __new__ (helper
of test_misure_fuori_percorso); twin and victim rollouts run with a fake node, Isaac and MOLA
that deliver one cloud per 0.1 s of spin while the simulation plays.
"""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

pn = pytest.importorskip("src.nodes.perturbation_node")
ao = pytest.importorskip("src.optimization.attack_orchestrator")
tm = pytest.importorskip("test_misure_fuori_percorso")


# --- perturbation node ---

def _sequenza():
    """Attack off (3), on (3), on without genome (2), on again (2)."""
    return [("off", 3), ("on", 3), ("nogenome", 2), ("on", 2)]


def _corri_nodo(never_clean, shadow=True):
    n = tm._nodo(victim=True)
    n._never_clean = never_clean
    n.victim_clean_dropped = 0
    n.victim_clean_out = 0
    n.victim_frame_idx = 0
    if shadow:
        n.shadow_pub = tm._Pub()
    params = n.victim_params
    nuvole = tm._nuvole_sintetiche(10)
    t = threading.Thread(target=n._pub_loop, daemon=True)
    t.start()
    attese = []
    i = 0
    for stato, k in _sequenza():
        for _ in range(k):
            n.victim_attack = stato != "off"
            n.victim_params = None if stato == "nogenome" else params
            c, st = nuvole[i]
            n._cloud_cb(tm._msg(c, st))
            attese.append((stato, c))
            i += 1
    t_end = time.time() + 10
    while n._out_q.qsize() > 0 or n.n_out < i:
        assert time.time() < t_end, "pubblicazione ferma"
        time.sleep(0.01)
    n.victim_params = params
    return n, attese


def _pulita(msg, c):
    a = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(-1, 4)
    return a.shape == c[:, :4].shape and np.array_equal(a, c[:, :4].astype(np.float32))


def test_mai_pulite_a_vittima_e_ombra():
    n, attese = _corri_nodo(True)
    attaccate = [c for s, c in attese if s == "on"]
    # only the 5 attacked clouds go to the victim and the shadow, no clean one
    assert len(n.victim_pub.msgs) == 5
    assert len(n.shadow_pub.msgs) == 5
    for m, c in zip(n.victim_pub.msgs, attaccate):
        assert not _pulita(m, c)
    assert [bytes(m.data) for m in n.shadow_pub.msgs] == [bytes(m.data) for m in n.victim_pub.msgs]
    # counters: 5 dropped, 0 clean delivered
    assert n.victim_clean_dropped == 5
    assert n.victim_clean_out == 0
    # the attacker receives all 10 clouds, clean, as usual
    assert len(n.attacker_pub.msgs) == 10
    assert all(_pulita(m, c) for m, (_, c) in zip(n.attacker_pub.msgs, attese))
    # replay log: only the delivered ones
    assert n._victim_rec_total == 5


def test_senza_opzione_come_prima():
    n, attese = _corri_nodo(False)
    assert len(n.victim_pub.msgs) == 10
    pulite = [m for m, (s, c) in zip(n.victim_pub.msgs, attese) if s != "on" and _pulita(m, c)]
    assert len(pulite) == 5
    assert n.victim_clean_out == 5 and n.victim_clean_dropped == 0
    assert len(n.attacker_pub.msgs) == 10


def test_nuvole_attaccate_identiche_a_prima():
    """Attacked clouds have the same content and stamp with and without the option."""
    a, _ = _corri_nodo(True)
    b, attese = _corri_nodo(False)
    on = [k for k, (s, _) in enumerate(attese) if s == "on"]
    vb = [b.victim_pub.msgs[k] for k in on]
    assert [bytes(m.data) for m in a.victim_pub.msgs] == [bytes(m.data) for m in vb]
    assert [(m.header.stamp.sec, m.header.stamp.nanosec) for m in a.victim_pub.msgs] == \
           [(m.header.stamp.sec, m.header.stamp.nanosec) for m in vb]
    # the attacker stamps are unchanged too
    assert [(m.header.stamp.sec, m.header.stamp.nanosec) for m in a.attacker_pub.msgs] == \
           [(m.header.stamp.sec, m.header.stamp.nanosec) for m in b.attacker_pub.msgs]


def _status(n):
    out = []
    n.status_pub = SimpleNamespace(publish=lambda m: out.append(m.data))
    n._meas_lock = None
    n._max_drift_lag_s = float("inf")
    n._replay_seq, n._replay_n, n._replay_total = 0, 0, 0
    n._replay_last_stamp = n._replay_first_stamp = None
    n._twin_offset = 0.0
    n._drift_keep = False
    pn.PerturbationNode._publish_status(n)
    import json
    return json.loads(out[-1])


def test_status_contatori_solo_con_opzione():
    a, _ = _corri_nodo(True)
    st = _status(a)
    assert st["victim_never_clean"] is True
    assert st["victim_clean_dropped"] == 5 and st["victim_clean_out"] == 0
    b, _ = _corri_nodo(False)
    st = _status(b)
    assert not any(k in st for k in ("victim_never_clean", "victim_clean_dropped", "victim_clean_out"))


def test_registro_per_frame_scartata_senza_stamp_vittima(tmp_path):
    fl = pn.FrameLog(tmp_path / "f.csv")
    n = tm._nodo(victim=True, frame_log=fl)
    n._never_clean = True
    n.victim_attack = False
    c, st = tm._nuvole_sintetiche(1)[0]
    n._cloud_cb(tm._msg(c, st))
    job = n._out_q.get_nowait()
    row = job[7]
    cols = pn.FrameLog.COLUMNS
    assert row[cols.index("victim_stream")] is True
    assert row[cols.index("victim_frame_idx")] is None
    assert row[cols.index("stamp_victim")] is None
    fl.close()


def test_opzione_del_nodo_spenta_di_default():
    import inspect
    src = inspect.getsource(pn.main)
    assert '"--victim-never-clean", action="store_true"' in src
    assert pn.PerturbationNode._never_clean is False


# --- orchestrator: options ---

def _args(**kw):
    a = dict(victim_never_clean=False, victim_start="run", settle_clouds=None, reopen_clouds=None,
             attack_from="spawn", warmup=0.0, passthrough=False, map_gate_test=False,
             victim="continuous", twin_state="ombra_e")
    a.update(kw)
    return SimpleNamespace(**a)


def test_default_invariato_orchestratore():
    a = _args()
    assert ao.check_spawn_b_args(a) is None
    assert (a.victim_start, a.settle_clouds, a.reopen_clouds) == ("run", 0, 0)
    ao.CLOUD_SETTLE.update({"n": 0, "reopen": 0, "never_clean": False})
    assert ao.settle_target(False) == 0 and ao.settle_target(True) == 0
    node = SimpleNamespace()
    assert "settle_clouds" not in ao._twin_start_fields(node)


def test_opzione_insieme_completa_i_default():
    a = _args(victim_never_clean=True)
    assert ao.check_spawn_b_args(a) is None
    assert (a.victim_start, a.settle_clouds, a.reopen_clouds) == ("w0", 20, 8)
    a = _args(victim_never_clean=True, settle_clouds=30, reopen_clouds=0)
    assert ao.check_spawn_b_args(a) is None
    assert (a.settle_clouds, a.reopen_clouds) == (30, 0)


@pytest.mark.parametrize("kw", [dict(attack_from="motion"), dict(warmup=0.5), dict(passthrough=True),
                                dict(map_gate_test=True), dict(victim="restart")])
def test_opzione_insieme_rifiuta(kw):
    assert "--victim-never-clean" in ao.check_spawn_b_args(_args(victim_never_clean=True, **kw))


def test_avvio_rimandato_rifiuta_stato_dalla_vittima():
    for ts in ("victim", "replay", "maprestore"):
        assert ao.check_spawn_b_args(_args(victim_start="w0", twin_state=ts))
    assert ao.check_spawn_b_args(_args(victim_start="w0", twin_state="ombra_e")) is None
    assert ao.check_spawn_b_args(_args(reopen_clouds=5))


def test_controllo_nodo_orchestratore():
    nodo = SimpleNamespace(node_never_clean=None, spin=lambda s: None)
    assert ao.never_clean_mismatch(nodo, False) is None
    assert ao.never_clean_mismatch(nodo, True, timeout=0.05)
    nodo.node_never_clean = True
    assert ao.never_clean_mismatch(nodo, True) is None
    assert ao.never_clean_mismatch(nodo, False)


def test_settle_target():
    ao.CLOUD_SETTLE.update({"n": 20, "reopen": 8, "never_clean": True})
    try:
        assert ao.settle_target(False) == 20      # twins before the start, victim at its start
        assert ao.settle_target(True) == 8        # reopening and twins of the later windows
        ao.CLOUD_SETTLE["reopen"] = 0
        assert ao.settle_target(True) == 20
    finally:
        ao.CLOUD_SETTLE.update({"n": 0, "reopen": 0, "never_clean": False})


def test_wait_clouds_conta_dal_nodo():
    c = {"n": 7}

    def spin(_dt):
        c["n"] += 1
    rec = ao.wait_clouds(None, lambda: c["n"], 7, 20, time.time(), spin=spin)
    assert rec["clouds"] == 20 and rec["ok"] is True and rec["target"] == 20
    c["n"] = 0
    rec = ao.wait_clouds(None, lambda: c["n"], 0, 5, time.time(), spin=lambda dt: time.sleep(dt),
                         timeout=0.1)
    assert rec["ok"] is False and rec["clouds"] == 0


def test_finestra_trattenuta():
    v = SimpleNamespace(started=False)
    f = ao.victim_hold_fields("nessun_ammissibile", v)
    assert f == {"victim_held": True, "victim_hold_reason": "nessun_ammissibile", "victim_started": False}


# --- twin and victim on w0: same settling ---

class _Isaac:
    def __init__(self):
        self.playing = False

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def restore(self):
        return (0.0, 0.0, 0.0)


class _Mola:
    def spawn(self, tag):
        pass

    def stop(self):
        pass

    def wait_ready(self, check=None):
        return True


class _Nodo:
    """Delivers one cloud per 0.1 s of spin in play: to the twin (main output) with the
    flow closed, to the victim with the flow open; with victim-never-clean, clouds
    without attack do not arrive. Logs who receives each delivered cloud and whether
    it is attacked.
    """
    compose = staticmethod(ao.AttackNode.compose)

    def __init__(self, isaac, never_clean):
        self.isaac = isaac
        self.never_clean = never_clean
        self.frames_out = 0
        self.victim_rec_total = 0
        self.stream = False
        self.enabled = False
        self.victim_attack = False
        self.victim_genome = None
        self.acc = 0.0
        self.log = []
        self.est = self.est_raw = None
        self.est_t = 0.0
        self.est_stamp_raw = float("nan")
        self.pose_quality = float("nan")
        self.last_victim_close_wall = time.time()
        # graph: only the perturbation node publishes to the victim; each subscriber
        # read shows a new GID (MOLA just launched)
        self._gid = [0]

        def _subs(topic):
            self._gid[0] += 1
            return [SimpleNamespace(endpoint_gid=[self._gid[0]])]
        self.node = SimpleNamespace(
            get_publishers_info_by_topic=lambda t: [SimpleNamespace(node_name="perturbation_node",
                                                                    node_namespace="/")],
            get_subscriptions_info_by_topic=_subs)

    # commands to the perturbation node
    def set_genome(self, g):
        return True

    def set_attack_enabled(self, on):
        self.enabled = bool(on)

    def set_victim_genome(self, g):
        self.victim_genome = g
        return True

    def set_victim_attack(self, on):
        self.victim_attack = bool(on)
        return True

    def set_victim_stream(self, on):
        self.stream = bool(on)

    def reset_victim_drift(self):
        pass

    def count_mola_publishers(self):
        return 1

    def settled_pose(self, isaac):
        self.spin(0.02)
        return (0.0, 0.0, 0.0)

    def spin(self, dt):
        if not self.isaac.playing:
            return
        self.acc += dt
        while self.acc >= 0.1 - 1e-9:
            self.acc -= 0.1
            if self.stream:
                att = self.victim_attack and self.victim_genome is not None
                if att or not self.never_clean:
                    self.victim_rec_total += 1
                    self.log.append(("vittima", att))
            else:
                self.frames_out += 1
                self.log.append(("gemello", self.enabled))
            self.est_raw = (0.0, 0.0, 0.0)
            self.est_t = time.time()
            self.est_stamp_raw = 1.0

    def drain(self, budget=0.02):
        self.spin(budget)


def _guida_finta(registro):
    def f(node, isaac, mola, local_wps, horizon, true_start, trace_path, *a, **kw):
        registro.append(list(node.log))
        if kw.get("finish"):
            kw["finish"]()
        return true_start, true_start, 1.0, "ok", 1.0, 10
    return f


@pytest.mark.parametrize("n_settle", [20, 12])
def test_gemello_e_vittima_stesso_assestamento_su_w0(monkeypatch, tmp_path, n_settle):
    ao.CLOUD_SETTLE.update({"n": n_settle, "reopen": 8, "never_clean": True})
    saved = dict(ao.TWIN_STATE)
    try:
        guide = []
        monkeypatch.setattr(ao, "_drive_and_measure", _guida_finta(guide))
        monkeypatch.setattr(ao, "_replay_with_guard", lambda node, mola, tag: None)
        monkeypatch.setattr(ao, "_twin_alignment_record", lambda node, ts, fp: None)
        isaac = _Isaac()
        victim = ao.VictimRunner("x.sh", tmp_path / "logs")
        victim.deferred = {"shadow": None, "origin_mode": "true"}
        victim.spawn_deferred = lambda node, shadow: {"rimandato": True}
        victim.wait_ready = lambda check=None: True
        ao.TWIN_STATE.update({"mode": "clean", "victim": victim, "init": "true", "init_target": None})
        g = np.ones(17)

        # twin of window 0 (victim not started yet)
        nt = _Nodo(isaac, True)
        ao._rollout_once(g, isaac, _Mola(), nt, (2.0, 0.0), 1.0, "w0_e00", settle_fixed=0.4)
        rec_t = nt.last_settle_clouds
        # victim at the first application (deferred start)
        nv = _Nodo(isaac, True)
        ao._rollout_once(g, isaac, _Mola(), nv, (2.0, 0.0), 1.0, "w0_applied", settle_fixed=0.4,
                         victim=victim)
        rec_v = nv.last_victim_settle_clouds

        twin_log, vict_log = guide[0], guide[1]
        # same number of clouds before driving, all attacked, from the first one
        assert len(twin_log) == len(vict_log) == n_settle
        assert all(who == "gemello" and att for who, att in twin_log)
        assert all(who == "vittima" and att for who, att in vict_log)
        assert rec_t["target"] == rec_v["target"] == n_settle
        assert rec_t["clouds_at_drive"] == rec_v["clouds_at_drive"] == n_settle
        assert rec_t["exact"] and rec_v["exact"] and rec_t["excess"] == rec_v["excess"] == 0
        assert rec_t["subscription"]["ok"] is True       # subscription before play, as the victim
        assert rec_t["sim_s_nominal"] == rec_v["sim_s_nominal"] == n_settle / 10.0
        assert victim.started and victim.start_info["window"] == 0
        assert victim.start_info["genome_on_first_cloud"] is True
        assert victim.run_origin == (0.0, 0.0, 0.0)

        # reopening of window 1: M clouds, all attacked, no clean one
        nv2 = _Nodo(isaac, True)
        ao._rollout_once(g, isaac, _Mola(), nv2, (2.0, 0.0), 1.0, "w1_applied", settle_fixed=0.4,
                         victim=victim)
        assert len(guide[2]) == 8 and all(att for _, att in guide[2])
        assert nv2.last_victim_settle_clouds["phase"] == "riapertura"
        # twin of window 1: same standstill segment as the reopening
        nt2 = _Nodo(isaac, True)
        ao._rollout_once(g, isaac, _Mola(), nt2, (2.0, 0.0), 1.0, "w1_e00", settle_fixed=0.4)
        assert len(guide[3]) == 8
    finally:
        ao.CLOUD_SETTLE.update({"n": 0, "reopen": 0, "never_clean": False})
        ao.TWIN_STATE.clear()
        ao.TWIN_STATE.update(saved)


def test_vittima_senza_genoma_non_guida(monkeypatch, tmp_path):
    ao.CLOUD_SETTLE.update({"n": 20, "reopen": 8, "never_clean": True})
    try:
        guide = []
        monkeypatch.setattr(ao, "_drive_and_measure", _guida_finta(guide))
        isaac = _Isaac()
        victim = ao.VictimRunner("x.sh", tmp_path / "logs")
        nv = _Nodo(isaac, True)
        res = ao._victim_rollout(None, isaac, victim, nv, (2.0, 0.0), 1.0, "w3_advance", None,
                                 0.6, None, None, 0.4, 0.5, 0.8, "estimate", "spawn", "genome")
        assert res[3] == "held_no_genome" and ao.outcome_kind(res[3]) == "infra"
        assert not res[3].startswith("victim_")       # not a pose loss
        assert guide == [] and nv.log == [] and not isaac.playing
    finally:
        ao.CLOUD_SETTLE.update({"n": 0, "reopen": 0, "never_clean": False})


def test_attacco_non_confermato_non_apre_il_flusso(monkeypatch, tmp_path):
    ao.CLOUD_SETTLE.update({"n": 20, "reopen": 8, "never_clean": True})
    try:
        guide = []
        monkeypatch.setattr(ao, "_drive_and_measure", _guida_finta(guide))
        isaac = _Isaac()
        victim = ao.VictimRunner("x.sh", tmp_path / "logs")
        victim.started = True
        nv = _Nodo(isaac, True)
        nv.set_victim_attack = lambda on: False
        res = ao._victim_rollout(np.ones(17), isaac, victim, nv, (2.0, 0.0), 1.0, "w2_applied", None,
                                 0.6, None, None, 0.4, 0.5, 0.8, "estimate", "spawn", "genome")
        assert res[3] == "attack_unconfirmed" and ao.outcome_kind(res[3]) == "infra"
        assert not nv.stream and guide == []
    finally:
        ao.CLOUD_SETTLE.update({"n": 0, "reopen": 0, "never_clean": False})


# --- held window restores the saved state ---

class _IsaacFisico:
    """Fake Isaac with physical state: play moves the robot, save and
    restore store and restore it.
    """
    def __init__(self):
        self.x = 0.0
        self.saved = None
        self.playing = False

    def play(self):
        self.playing = True
        self.x += 1.0

    def pause(self):
        self.playing = False

    def save(self):
        self.saved = self.x
        return (self.x, 0.0, 0.0)

    def restore(self):
        self.x = self.saved
        return (self.x, 0.0, 0.0)


def test_finestra_trattenuta_ripristina_lo_stato_salvato():
    isaac = _IsaacFisico()
    isaac.save()                     # window start
    isaac.play(); isaac.pause()      # last twin rollout: robot moved
    assert isaac.x != isaac.saved
    fermate = []
    mola = SimpleNamespace(stop=lambda: fermate.append(1))
    node = SimpleNamespace(fresh_cloud=True)
    rec = ao.hold_restore(isaac, mola, node, 3)
    assert isaac.x == isaac.saved and not isaac.playing and fermate == [1]
    assert rec["restored"] is True and rec["restore_pose"] == [isaac.saved, 0.0, 0.0]
    assert node.restore_log[-1]["kind"] == "trattenuta" and node.restore_log[-1]["window"] == 3
    # the next window saves the same state
    assert isaac.save()[0] == 0.0


def test_tre_rami_trattenuti_ripristinano():
    import inspect
    src = inspect.getsource(ao.main)
    for motivo in ("nominale_fallito", "watchdog", "nessun_ammissibile"):
        assert f'victim_hold_fields("{motivo}", victim)' in src
    assert src.count("hold_restore(isaac, mola, node, k, saved)") == 3


# --- true pose after a hold, foreign publisher, subscriber wait ---

def _msg_odom(x, y, yaw, t):
    import math
    return SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=0.8, y=0.0))),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=int((t % 1) * 1e9))))


class _IsaacOdom:
    """Fake Isaac with /chassis/odom: in play the robot moves 1 m along x and the pose
    reaches the node cache (_true_cb) at 10 Hz; paused, the topic is silent. save and
    restore return the pose in the scene (odom shifted by (10, 5)).
    """
    def __init__(self, node, x=1.0):
        self.node = node
        self.x = x
        self.saved = None
        self.playing = False
        self.t = 0.0

    def play(self):
        self.playing = True
        for i in range(1, 11):
            self.t += 0.1
            self.node._true_cb(_msg_odom(self.x + 0.1 * i, 0.0, 0.0, self.t))
        self.x += 1.0

    def pause(self):
        self.playing = False

    def _scena(self):
        return (self.x + 10.0, 5.0, 0.0)

    def pose_stream(self):
        return self._scena()

    def save(self):
        self.saved = self.x
        return self._scena()

    def restore(self):
        self.x = self.saved
        return self._scena()


def _nodo_odom():
    from collections import deque
    n = ao.AttackNode.__new__(ao.AttackNode)
    n.true_source = "ros"
    n.true, n.true_t, n.true_z = None, 0.0, float("nan")
    n.true_v = (float("nan"), float("nan"))
    n.true_buf = deque(maxlen=400)
    n._n_cb = 0
    n.attacker_buf = []
    return n


def _finestra_dopo_trattenuta(passa_saved):
    """Window start as in main (true_pose, pause, save), last twin rollout, hold, start of
    the next window. Returns the two saved positions and what follows from them with
    --twin-init attacker_est and with D2.
    """
    node = _nodo_odom()
    isaac = _IsaacOdom(node)
    isaac.play()                                   # robot at x = 2 at window start
    saved = node.true_pose(isaac)
    isaac.pause()
    saved_scene = isaac.save()
    isaac.play(); isaac.pause()                    # last twin rollout: cache at x = 3
    assert node.true_pose(isaac)[0] == pytest.approx(3.0)
    mola = SimpleNamespace(stop=lambda: None)
    if passa_saved:
        ao.hold_restore(isaac, mola, node, 2, saved)
    else:
        ao.hold_restore(isaac, mola, node, 2)
    saved2 = node.true_pose(isaac)                  # next window
    isaac.pause()
    scene2 = isaac.save()
    attacker = SimpleNamespace(run_origin=(0.0, 0.0, 0.0))
    believed = (2.3, 0.1, 0.05)                    # attacker estimate: 30 cm ahead of the robot
    ti = ao.attacker_init_target(node, attacker, saved2, scene2, believed, "zero")
    p0, u = ao.direction_line([(0.0, 0.0), (5.0, 0.0)], (0.0, 0.0))
    carota = ao.direction_window_goal(p0, u, saved2[:2], 5.0)
    return saved, saved_scene, saved2, scene2, ti, carota


def test_trattenuta_riporta_la_posa_vera_salvata():
    saved, saved_scene, saved2, scene2, ti, carota = _finestra_dopo_trattenuta(True)
    assert saved[0] == pytest.approx(2.0)
    # true_pose of the next window = saved pose, consistent with the physical state
    assert saved2 == pytest.approx(saved) and scene2 == pytest.approx(saved_scene)
    # twin base (--twin-init attacker_est): saved in the scene + believed offset
    assert ti["target_scene"] == pytest.approx([12.3, 5.1, 0.05])
    # D2 carrot from the saved pose: 2 + 5 m
    assert carota == pytest.approx((7.0, 0.0))


def test_trattenuta_senza_posa_salvata_cache_del_gemello():
    """Test control: without the saved pose the cache stays the twin's (1 m offset)."""
    saved, _, saved2, _, ti, carota = _finestra_dopo_trattenuta(False)
    assert saved2[0] - saved[0] == pytest.approx(1.0)
    assert ti["target_scene"][0] == pytest.approx(11.3) and carota[0] == pytest.approx(8.0)


class _NodoEstraneo:
    """Fake AttackNode: the first command inside the try block of main finds a foreign
    publisher on the victim topic, as check_victim_topic at a reopening.
    """
    def __init__(self, *a, **kw):
        self.true = None
        self.stopped = 0

    def halt(self):
        raise ao.VictimTopicForeign("publisher estraneo su /carter/lidar_victim: intruso")

    def stop_robot(self):
        self.stopped += 1

    def close(self):
        pass


class _IsaacVuoto:
    def play(self):
        pass


def test_publisher_estraneo_a_run_avviata_codice_2(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "IsaacClient", _IsaacVuoto)
    monkeypatch.setattr(ao.MolaRunner, "kill_orphans", staticmethod(lambda: 0))
    monkeypatch.setattr(ao, "AttackNode", _NodoEstraneo)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["attack_orchestrator.py", "--victim", "restart",
                                      "--true-pose", "file", "--out", str(out)])
    # global state written by main before the run: restored at the end of the test
    globali = {n: dict(getattr(ao, n)) for n in ("TWIN_STATE", "STALL_POLICY", "CLOUD_SETTLE", "MOLA_STOP")}
    try:
        rc = ao.main()
    finally:
        for n, v in globali.items():
            getattr(ao, n).clear()
            getattr(ao, n).update(v)
    assert rc == 2                                    # run_campaign.sh does not write completata
    import json
    h = json.loads((out / "history.json").read_text())
    assert h[-1]["status"] == "victim_topic_foreign_publisher"
    assert ao.outcome_kind(h[-1]["status"]) == "infra"
    # the other codes are unchanged
    assert ao.exit_code(None, {"any": False}, False) == 0


def test_search_none_rifiutata():
    msg = ao.check_spawn_b_args(_args(victim_never_clean=True, search="none"))
    assert msg and "--search none" in msg
    assert ao.check_spawn_b_args(_args(victim_never_clean=True, search="nsga3")) is None
    assert ao.check_spawn_b_args(_args(search="none")) is None


def _grafo(pubs=None, subs=None):
    pubs = pubs if pubs is not None else {}
    subs = subs if subs is not None else {}
    g = SimpleNamespace(
        get_publishers_info_by_topic=lambda t: [SimpleNamespace(node_name=n, node_namespace="/")
                                                for n in pubs.get(t, [])],
        get_subscriptions_info_by_topic=lambda t: [SimpleNamespace(endpoint_gid=list(x))
                                                   for x in subs.get(t, [])])
    return SimpleNamespace(node=g, spin=lambda dt: time.sleep(0.001), drain=lambda budget=0.02: None)


def test_publisher_del_topic_vittima():
    assert ao.victim_topic_problem(_grafo({ao.VICTIM_TOPIC: ["perturbation_node"]})) is None
    err = ao.victim_topic_problem(_grafo({ao.VICTIM_TOPIC: ["perturbation_node", "intruso"]}))
    assert err and "intruso" in err
    assert ao.victim_topic_problem(_grafo({}), timeout=0.05)
    with pytest.raises(ao.VictimTopicForeign):
        ao.check_victim_topic(_grafo({ao.VICTIM_TOPIC: ["perturbation_node", "perturbation_node"]}))


def test_apertura_con_publisher_estraneo_si_ferma(tmp_path):
    ao.CLOUD_SETTLE.update({"n": 20, "reopen": 8, "never_clean": True})
    try:
        isaac = _Isaac()
        victim = ao.VictimRunner("x.sh", tmp_path / "logs")
        victim.started = True
        nv = _Nodo(isaac, True)
        nv.node = _grafo({ao.VICTIM_TOPIC: ["perturbation_node", "intruso"]}).node
        with pytest.raises(ao.VictimTopicForeign):
            ao._victim_rollout(np.ones(17), isaac, victim, nv, (2.0, 0.0), 1.0, "w2_applied", None,
                               0.6, None, None, 0.4, 0.5, 0.8, "estimate", "spawn", "genome")
        assert not nv.stream and nv.log == []          # flow never opened
    finally:
        ao.CLOUD_SETTLE.update({"n": 0, "reopen": 0, "never_clean": False})


def test_attesa_iscrizione_nuova_mola():
    subs = {"/t": [(1, 2)]}
    nodo = _grafo(subs=subs)
    before = ao.topic_sub_gids(nodo, "/t")
    # only the previous MOLA entry: not enough
    r = ao.wait_new_subscriber(nodo, "/t", before, timeout=0.05)
    assert r["ok"] is False
    subs["/t"].append((3, 4))
    r = ao.wait_new_subscriber(nodo, "/t", before, timeout=0.05)
    assert r["ok"] is True
    # MOLA dead: exit at once
    subs["/t"].pop()
    r = ao.wait_new_subscriber(nodo, "/t", before, alive=lambda: False, timeout=5.0)
    assert r["ok"] is False and r["s"] < 1.0


def test_record_assestamento_esatto():
    rec = {"target": 20}
    ao.close_settle_record(rec, lambda: 27, 7, time.time())
    assert rec["clouds_at_drive"] == 20 and rec["exact"] is True and rec["excess"] == 0
    rec = {"target": 20}
    ao.close_settle_record(rec, lambda: 32, 7, time.time())
    assert rec["exact"] is False and rec["excess"] == 5
