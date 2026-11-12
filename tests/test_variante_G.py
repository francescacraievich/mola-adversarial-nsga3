"""Variant G (--twin-state ombra_g): constraints, consistency, overlay environment, .molastate
header, twin clock, signatures and hash chain, ready signal. Pure functions, no ROS or MOLA.
"""

import json
import os
import random
import types
from pathlib import Path

import pytest

import src.optimization.twin_variants as tv

REAL_STATE = "/tmp/G1_prove/salvacarica/victim/C_t1/stato_116444736118000000.molastate"


def _args(**kw):
    base = dict(twin_state="ombra_g", victim="continuous", twin_align="attacker",
                uniform_sim_time="true", twin_init="true", drift_reset_on="genome",
                attack_from="spawn", repeat_genome=None, repeat_spec=None,
                map_gate_test=False, replay_gate=0, g_overlay=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _overlay(tmp_path):
    root = tmp_path / "install_G1"
    for p in tv.G_OVERLAY_PKGS:
        (root / p / "lib" / "x86_64-linux-gnu").mkdir(parents=True)
    return str(root)


def _state(path, tick=116444736118000000, version=2, stamp=True, tail=b"x" * 64):
    b = tv.MOLASTATE_MAGIC + version.to_bytes(4, "little") + bytes([1 if stamp else 0])
    if stamp:
        b += int(tick).to_bytes(8, "little", signed=True)
    Path(path).write_bytes(b + tail)
    return str(path)


# --- choices and constraints ---

def test_scelte_e_default_invariato():
    assert tv.TWIN_STATES[:5] == ("clean", "victim", "replay", "maprestore", "ombra_e")
    assert "ombra_g" in tv.TWIN_STATES and tv.SHADOW_MODES == ("ombra_e", "ombra_g")


def test_vincoli_come_ombra_e(tmp_path):
    ov = _overlay(tmp_path)
    assert tv.check_twin_args(_args(g_overlay=ov)) is None
    assert "continuous" in tv.check_twin_args(_args(g_overlay=ov, victim="restart"))
    assert "use_sim_time" in tv.check_twin_args(_args(g_overlay=ov, uniform_sim_time="off"))
    assert "--repeat-genome" in tv.check_twin_args(_args(g_overlay=ov, repeat_genome="x"))
    assert "--repeat-genome" in tv.check_twin_args(_args(g_overlay=ov, map_gate_test=True))
    assert tv.check_twin_args(_args(g_overlay=ov, twin_init="ombra_est")) is None
    assert "ombra_e o ombra_g" in tv.check_twin_args(_args(twin_state="clean", twin_init="ombra_est"))


def test_vincoli_ombra_e_invariati():
    assert tv.check_twin_args(_args(twin_state="ombra_e")) is None
    assert tv.check_twin_args(_args(twin_state="ombra_e", twin_init="ombra_est")) is None
    assert tv.check_twin_args(_args(twin_state="clean", uniform_sim_time="off")) is None


def test_overlay_assente(tmp_path):
    msg = tv.check_twin_args(_args(g_overlay=str(tmp_path / "manca")))
    assert msg.startswith("--twin-state ombra_g: overlay di MOLA assente")


def test_coerenza():
    r = tv.coherence_record("ombra_g", "ombra_est")
    assert r["twin_state_coerente_threat_model"] is True
    assert "stato completo" in r["twin_state_motivo"] and "lab_*" in r["twin_state_motivo"]
    assert tv.coherence_record("ombra_e")["twin_state_coerente_threat_model"] is True


# --- environment ---

def test_ambiente_overlay_in_testa(tmp_path):
    ov = _overlay(tmp_path)
    env = tv.g_overlay_env({"LD_LIBRARY_PATH": "/opt/ros/jazzy/lib"}, ov)
    d = tv.g_overlay_dirs(ov)
    assert env["LD_LIBRARY_PATH"] == f"{d[0]}:{d[1]}:/opt/ros/jazzy/lib"
    assert env["MOLA_MODULES_LIB_PATH"] == f"{d[0]}:{d[1]}"


def test_ambiente_gemello_e_ombra(tmp_path):
    ov = _overlay(tmp_path)
    t = tv.g_twin_env("/s/a.molastate", "/s/r.molastate", {}, ov)
    assert t["MOLA_FULL_STATE"] == "1" and t["MOLA_FULL_STATE_LOAD"] == "/s/a.molastate"
    assert t["MOLA_FULL_STATE_RESAVE"] == "/s/r.molastate"
    vietate = ("MOLA_FULL_STATE_SAVE_AT_TICKS", "MOLA_FULL_STATE_TRACE", "MOLA_FULL_STATE_PROVA_TWIST_ZERO",
               "MOLA_FULL_STATE_SKIP_UNTIL_TICKS", "MOLA_LOAD_MM")
    assert not any(k in t for k in vietate) and not any(k.startswith("MOLA_INITIAL_") for k in t)
    assert "MOLA_FULL_STATE_RESAVE" not in tv.g_twin_env("/s/a.molastate", None, {}, ov)
    sh = tv.g_shadow_env({}, ov)
    assert sh["MOLA_FULL_STATE"] == "1" and "MOLA_FULL_STATE_LOAD" not in sh


# --- .molastate ---

def test_intestazione_sintetica(tmp_path):
    h = tv.molastate_header(_state(tmp_path / "a.molastate", tick=123))
    assert h == {"magic": "MOLAFS01", "version": 2, "has_stamp": True, "stamp_tick": 123}
    h = tv.molastate_header(_state(tmp_path / "b.molastate", stamp=False))
    assert h["has_stamp"] is False and h["stamp_tick"] is None


def test_stato_non_caricabile(tmp_path):
    assert "non esiste" in tv.g_state_problem(str(tmp_path / "manca"))
    (tmp_path / "vuoto").write_bytes(b"")
    assert "vuoto" in tv.g_state_problem(str(tmp_path / "vuoto"))
    (tmp_path / "corto").write_bytes(b"MOLAFS")
    assert "troncata" in tv.g_state_problem(str(tmp_path / "corto"))
    (tmp_path / "magic").write_bytes(b"XXXXXXXX" + bytes(20))
    assert "magic" in tv.g_state_problem(str(tmp_path / "magic"))
    assert "versione 1" in tv.g_state_problem(_state(tmp_path / "v1", version=1))
    assert "senza stamp" in tv.g_state_problem(_state(tmp_path / "ns", stamp=False))
    assert tv.g_state_problem(_state(tmp_path / "ok")) is None


@pytest.mark.skipif(not os.path.isfile(REAL_STATE), reason="file della G1 assente")
def test_intestazione_file_vero():
    h = tv.molastate_header(REAL_STATE)
    assert (h["version"], h["stamp_tick"]) == (2, 116444736118000000)
    assert tv.g_state_problem(REAL_STATE) is None


# --- clock ---

def test_conversione_come_mrpt():
    # 11.8 s of simulation -> G1 state file name (stato_116444736118000000)
    assert tv.ros_stamp_to_tick(11, 800_000_000) == 116444736118000000
    assert tv.stamp_s_to_tick(11.8) == 116444736118000000
    assert tv.stamp_s_to_tick(None) is None


def test_primo_stamp_esatto_1e5():
    r = random.Random(20261025)
    for _ in range(100_000):
        t = tv.MRPT_EPOCH_TICKS + r.randrange(0, 3600 * 10**7)
        ns = tv.twin_clock_first_ns(t)
        assert tv.ros_stamp_to_tick(ns // 10**9, ns % 10**9) == t + tv.TWIN_CLOCK_STEP_TICKS


def _node_cls():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    return next(c for c in vars(pn).values() if isinstance(c, type) and hasattr(c, "_twin_header"))


def _fake_node():
    log = types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)
    return types.SimpleNamespace(
        _twin_clock_on=False, _twin_last_sim=None, _twin_last_clock=None, _twin_offset=0.0,
        _twin_clock_base_on=True, _twin_clock_seq=None, _twin_first_ns=None,
        _twin_clock_first_out_ns=None, victim_stream=False, get_logger=lambda: log)


def _hdr(sec, nsec):
    return types.SimpleNamespace(stamp=types.SimpleNamespace(sec=sec, nanosec=nsec), frame_id="base_link")


def test_nodo_orologio_armato():
    cls = _node_cls()
    n = _fake_node()
    saved = tv.MRPT_EPOCH_TICKS + 118_000_000          # 11.8 s on the victim clock
    cls._twin_clock_base_cb(n, types.SimpleNamespace(data=json.dumps({"tick": saved, "seq": 7})))
    assert n._twin_clock_on and n._twin_clock_seq == 7
    h1 = cls._twin_header(n, _hdr(503, 250_000_000))    # Isaac time, far ahead
    assert tv.ros_stamp_to_tick(h1.stamp.sec, h1.stamp.nanosec) == saved + 1_000_000
    assert n._twin_clock_first_out_ns == h1.stamp.sec * 10**9 + h1.stamp.nanosec
    h2 = cls._twin_header(n, _hdr(503, 350_000_000))    # simulation dt 0.1 s
    t1 = h1.stamp.sec + h1.stamp.nanosec * 1e-9
    t2 = h2.stamp.sec + h2.stamp.nanosec * 1e-9
    assert abs((t2 - t1) - 0.1) < 2e-9
    # same seq: no re-arm
    cls._twin_clock_base_cb(n, types.SimpleNamespace(data=json.dumps({"tick": saved + 5, "seq": 7})))
    assert n._twin_first_ns is None
    # victim reopening: twin clock off (as after a replay)
    cls._victim_stream_cb(n, types.SimpleNamespace(data=True))
    assert n._twin_clock_on is False


def test_nodo_senza_base_header_come_prima():
    cls = _node_cls()
    n = _fake_node()
    n._twin_last_clock = 20.0
    h = cls._twin_header(n, _hdr(503, 0))
    assert (h.stamp.sec, h.stamp.nanosec) == (20, 100_000_000)
    cls._twin_clock_base_cb(n, types.SimpleNamespace(data="non json"))
    assert n._twin_first_ns is None and n._twin_clock_seq is None


# --- signatures and hash chain ---

def _log(path, extra=""):
    return (f"[LidarOdometry] [full_state] caricato {path} (4600000 byte) cache icp 3\n"
            f"[LidarOdometry] [full_state] salvato /r/x.molastate (4600000 byte, stamp 116444736118000000)\n"
            + extra)


def test_controllo_fine_rollout_buono(tmp_path):
    sp = _state(tmp_path / "ombra_w1.molastate")
    rs = tmp_path / "resave.molastate"
    rs.write_bytes(Path(sp).read_bytes())
    st = {"state_path": sp, "state_sha256": tv.file_sha256(sp), "state_stamp_tick": 116444736118000000}
    first = tv.twin_clock_first_ns(116444736118000000)
    c = tv.g_rollout_check(_log(sp), st, str(rs), first)
    assert c["ok"] and c["firme_ok"] and c["hash_ok"] and c["clock_ok"]
    assert c["first_minus_saved_ticks"] == 1_000_000


def test_controllo_fine_rollout_cattivo(tmp_path):
    sp = _state(tmp_path / "ombra_w1.molastate")
    st = {"state_path": sp, "state_sha256": tv.file_sha256(sp), "state_stamp_tick": 116444736118000000}
    first = tv.twin_clock_first_ns(116444736118000000)
    rs = tmp_path / "resave.molastate"
    rs.write_bytes(b"diverso")
    assert tv.g_rollout_check(_log(sp), st, str(rs), first)["hash_ok"] is False
    assert tv.g_rollout_check(_log(sp, "Initial re-localization done\n"), st, None, first)["firme_ok"] is False
    assert tv.g_rollout_check(_log(sp, "[full_state] caricamento fallito: x\n"), st, None, first)["ok"] is False
    assert tv.g_rollout_check(_log("/altro"), st, None, first)["loaded_is_shadow_file"] is False
    c = tv.g_rollout_check(_log(sp), st, None, first + 10**8)
    assert c["clock_ok"] is False and c["ok"] is False
    assert tv.g_rollout_check(_log(sp), st, None, None)["ok"] is False


# --- ready signal ---

def _runner(tmp_path, text):
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    r = ao.MolaRunner("x.sh", tmp_path / "logs", ready_timeout=0.3)
    r.log_path = tmp_path / "mola.log"
    r.log_path.write_text(text)
    r.proc = types.SimpleNamespace(poll=lambda: None)
    return r


def test_pronto_default_invariato(tmp_path):
    assert _runner(tmp_path, "... Initial re-localization done\n").wait_ready() is True
    with pytest.raises(TimeoutError):
        _runner(tmp_path, f"{tv.G_READY_MARKER} /a (1 byte)\n").wait_ready()


def test_pronto_gemello_g(tmp_path):
    r = _runner(tmp_path, f"[full_state] caricato /a.molastate (10 byte)\n")
    assert r.wait_ready(marker=tv.G_READY_MARKER, fail_markers=tv.G_ERROR_MARKERS) is True
    with pytest.raises(TimeoutError):
        _runner(tmp_path, "Initial re-localization done\n").wait_ready(marker=tv.G_READY_MARKER)
    with pytest.raises(RuntimeError, match="caricamento fallito"):
        _runner(tmp_path, "[full_state] caricamento fallito: x\n").wait_ready(
            marker=tv.G_READY_MARKER, fail_markers=tv.G_ERROR_MARKERS)


# --- orchestrator branches (second half) ---

import time  # noqa: E402


def _ao():
    from src.optimization import attack_orchestrator as ao
    return ao


SAVED_TICK = tv.MRPT_EPOCH_TICKS + 118_000_000          # 11.8 s


def test_argparse_accetta_ombra_g(monkeypatch, capsys, tmp_path):
    ao = _ao()
    monkeypatch.setattr("sys.argv", ["attack_orchestrator.py", "--twin-state", "ombra_g",
                                     "--uniform-sim-time", "true", "--g-overlay", str(tmp_path / "manca")])
    assert ao.main() == 2
    assert "overlay di MOLA assente" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["attack_orchestrator.py", "--twin-state", "ombra_g"])
    assert ao.main() == 2
    assert "ombra_g richiede --uniform-sim-time" in capsys.readouterr().out


def test_ambiente_overlay_solo_ombra(tmp_path):
    ao = _ao()
    root = _overlay(tmp_path)
    sg = ao.ShadowRunner("s.sh", tmp_path / "logs", env={"A": "1"}, g_root=root)
    assert sg.extra_env["MOLA_FULL_STATE"] == "1" and sg.extra_env["A"] == "1"
    assert sg.extra_env["MOLA_MODULES_LIB_PATH"].startswith(tv.g_overlay_dirs(root)[0])
    # default unchanged: without g_root the shadow is the usual one
    se = ao.ShadowRunner("s.sh", tmp_path / "logs", env={"A": "1"})
    assert se.extra_env == ao.shadow_extra_env({"A": "1"}, tv.mola_ros_args("shadow"))
    assert se.g_root is None
    # the victim, even with --victim-shadow-env, does not get the overlay
    v = ao.victim_extra_env({"A": "1"}, {}, "--ros-args", shadow_env=True)
    assert not any(k.startswith("MOLA_FULL_STATE") for k in v)
    assert "install_G1" not in v.get("MOLA_MODULES_LIB_PATH", "") + v.get("LD_LIBRARY_PATH", "")


class NodoOmbra:
    def __init__(self, stamp=11.8, scrivi_stato=True, tick=SAVED_TICK):
        self.shadow_est_t = time.time() - 5.0
        self.shadow_pose6 = (0.5, 0.25, 0.0, 0.3, 0.0, 0.0)
        self.shadow_est_raw = (0.5, 0.25, 0.3)
        self.shadow_est_stamp = stamp
        self.scrivi_stato, self.tick = scrivi_stato, tick

    def spin(self, s):
        time.sleep(s)

    def shadow_map_save(self, path):
        Path(f"{path}.mm").write_bytes(b"mappa")
        if self.scrivi_stato:
            _state(f"{path}.molastate", tick=self.tick)
        return True, ""

    compose = staticmethod(lambda b, l: _ao().AttackNode.compose(b, l))


def _ombra_g(tmp_path):
    ao = _ao()
    r = ao.ShadowRunner("s.sh", tmp_path / "logs", g_root=_overlay(tmp_path))
    r.log_path = tmp_path / "logs" / "mola_shadow.log"
    r.log_path.write_text("")                 # no printf: sigma only informative
    r.alive = lambda: True
    r.run_origin = (0.0, 0.0, 0.0)
    (tmp_path / "maps").mkdir()
    return r


def test_salvataggio_stato_g(tmp_path):
    r = _ombra_g(tmp_path)
    st = r.save_state(NodoOmbra(), tmp_path / "maps" / "ombra_w1")
    sp = f"{tmp_path / 'maps' / 'ombra_w1'}.molastate"
    assert st["status"] == "ok" and st["sigma"] is None
    assert st["state_path"] == sp and st["state_sha256"] == tv.file_sha256(sp)
    assert st["state_stamp_tick"] == SAVED_TICK and st["pose_stamp_tick"] == SAVED_TICK
    assert st["stamp_diff_ticks"] == 0 and st["stamp_ok"]
    assert st["env"]["MOLA_FULL_STATE_LOAD"] == sp and "MOLA_LOAD_MM" not in st["env"]
    # different pose stamp: logged, not ok
    st = _ombra_g(tmp_path / "b").save_state(NodoOmbra(stamp=11.9), tmp_path / "b" / "maps" / "o")
    assert st["status"] == "ok" and st["stamp_diff_ticks"] == -1_000_000 and not st["stamp_ok"]
    # .molastate missing: state not usable
    st = _ombra_g(tmp_path / "c").save_state(NodoOmbra(scrivi_stato=False), tmp_path / "c" / "maps" / "o")
    assert st["status"] == "g_state_failed" and "non esiste" in st["error"]


class IsaacFinto:
    def pause(self): pass
    def restore(self): return [0, 0, 0]


class MolaFinta:
    def __init__(self, ordine):
        self.extra_env = {"BASE": "1"}
        self.spawned = None
        self.last_stop = None
        self.ordine = ordine

    def stop(self): pass

    def spawn(self, tag):
        self.ordine.append("spawn")
        self.spawned = (tag, dict(self.extra_env))
        raise RuntimeError("fermata del test dopo lo spawn")


class NodoArmo:
    def __init__(self, ordine, ok=True):
        self.ordine, self.ok, self.ticks = ordine, ok, []

    def arm_twin_clock(self, tick):
        self.ordine.append("arm")
        self.ticks.append(tick)
        return self.ok


def test_rollout_g_stato_mancante(monkeypatch):
    ao = _ao()
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_g", "shadow_state": {"status": "g_state_failed"}})
    o = []
    mola = MolaFinta(o)
    r = ao._rollout_once(None, IsaacFinto(), mola, NodoArmo(o), (1, 0), 1.0, "w1_e0")
    assert r[3].startswith("ombra_g_state_missing") and mola.spawned is None and o == []
    assert ao.outcome_kind(r[3]) == "infra"


def test_rollout_g_ambiente_e_orologio(monkeypatch, tmp_path):
    ao = _ao()
    root = _overlay(tmp_path)
    sp = _state(tmp_path / "ombra_w1.molastate", tick=SAVED_TICK)
    st = {"status": "ok", "state_path": sp, "state_stamp_tick": SAVED_TICK,
          "pose6": [0.5, 0.25, 0, 0.3, 0, 0], "env": tv.g_twin_env(sp, None, root=root)}
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_g", "shadow_state": st, "g_root": root,
                                           "g_resave": "ogni", "twin_env_base": {"BASE": "1"}})
    monkeypatch.setattr(ao, "topic_sub_gids", lambda node, topic: set())
    o = []
    mola, node = MolaFinta(o), NodoArmo(o)
    r = ao._rollout_once(None, IsaacFinto(), mola, node, (1, 0), 1.0, "w1_e3")
    assert r[3].startswith("mola_spawn_fail")
    assert o == ["arm", "spawn"] and node.ticks == [SAVED_TICK]
    tag, e = mola.spawned
    assert tag == "w1_e3" and e["BASE"] == "1" and e["MOLA_FULL_STATE"] == "1"
    assert e["MOLA_FULL_STATE_LOAD"] == sp and e["MOLA_FULL_STATE_RESAVE"] == f"{sp}.risalvato_w1_e3"
    assert e["MOLA_MODULES_LIB_PATH"].startswith(tv.g_overlay_dirs(root)[0])
    assert not any(k.startswith("MOLA_INITIAL_") or k == "MOLA_LOAD_MM" for k in e)
    # re-save off
    ao.TWIN_STATE["g_resave"] = "off"
    o.clear()
    mola = MolaFinta(o)
    ao._rollout_once(None, IsaacFinto(), mola, NodoArmo(o), (1, 0), 1.0, "w1_e4")
    assert "MOLA_FULL_STATE_RESAVE" not in mola.spawned[1]
    # clock not confirmed: rollout failed before spawn
    o.clear()
    mola = MolaFinta(o)
    r = ao._rollout_once(None, IsaacFinto(), mola, NodoArmo(o, ok=False), (1, 0), 1.0, "w1_e5")
    assert r[3] == "ombra_g_clock_not_armed" and mola.spawned is None


def test_campi_infra_g():
    ao = _ao()
    rec = ao.skipped_window_record(1, "ombra_g_state_missing: x", (1, 0), 0, "ok", 1.0, {"status": "x"})
    assert rec["infra_failure"] == "ombra_g_state_missing" and rec["applied"] is False
    f = ao.ombra_infra_fields([{"status": "ombra_g_state_missing: x"}] * 2)
    assert f == {"n_ombra_g_state_missing": 2, "infra_failure": "ombra_g_state_missing", "applied": False}
    # variant E unchanged
    f = ao.ombra_infra_fields([{"status": "ombra_e_state_missing: x"}, {"status": "ok"}])
    assert f == {"n_ombra_e_state_missing": 1}
    assert ao.ombra_infra_fields([{"status": "ok"}]) == {}


def test_impronta_g_a_rollout_chiuso(monkeypatch, tmp_path):
    ao = _ao()
    sp = _state(tmp_path / "ombra_w1.molastate", tick=SAVED_TICK)
    rs = tmp_path / "ombra_w1.molastate.risalvato_w1_e0"
    rs.write_bytes(Path(sp).read_bytes())
    st = {"status": "ok", "state_path": sp, "state_sha256": tv.file_sha256(sp), "state_stamp_tick": SAVED_TICK}
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_g", "shadow_state": st})
    log = tmp_path / "mola_w1_e0.log"
    log.write_text(_log(sp))
    node = types.SimpleNamespace(last_twin_fp={"source": "ombra_g", "g_resave_path": str(rs), "est_stamp": None},
                                 twin_clock_first_ns=tv.twin_clock_first_ns(SAVED_TICK),
                                 last_twin_clock_arm={"seq": 3, "tick": SAVED_TICK, "confirmed": True})
    ao._note_twin_fp(node, types.SimpleNamespace(log_path=log))
    c = node.last_twin_fp["g_check"]
    assert c["ok"] and c["hash_ok"] and c["clock_ok"], c
    assert node.last_twin_fp["g_clock_arm"]["confirmed"]


# --- twin hash: map publication fields ignored ---

_VERIFICA = Path(__file__).resolve().parents[1] / "data/attack/verifica"
_CASI_CONTATORE = [  # shadow state and re-saved file differ only in the counter (cmp offset 0x5BA)
    ("2026-11-01_0930_rigioco_c2ra_r01", "ombra_w2.molastate", "ombra_w2.molastate.risalvato_w2_e0"),
    ("2026-11-01_1006_rigioco_c2rb_r01", "ombra_w1.molastate", "ombra_w1.molastate.risalvato_w1_e0"),
]


def _maps(run):
    m = sorted((_VERIFICA / run / "campagna/straight").glob("*/rep_01/maps"))
    return m[0] if m else None


@pytest.mark.parametrize("run,ombra,risalvato", _CASI_CONTATORE)
def test_hash_ignora_solo_il_contatore_di_pubblicazione(run, ombra, risalvato):
    m = _maps(run)
    if m is None or not (m / ombra).is_file():
        pytest.skip("run di riferimento non presente")
    c = tv.campi_pubblicazione_mappa((m / ombra).read_bytes())
    assert c["advertise_counter"] == (0x5BA, 4) and c["needs_publish"] == (0x5BF, 1)
    r = tv.confronto_stato_mascherato(str(m / ombra), str(m / risalvato))
    assert r["uguale"] is True and r["byte_diversi"] == [] and r["byte_diversi_ignorati"] == [0x5BA]
    st = {"state_path": str(m / ombra), "state_sha256": tv.file_sha256(str(m / ombra)), "state_stamp_tick": None}
    k = tv.g_rollout_check(_log(str(m / ombra)), st, str(m / risalvato), None)
    assert k["hash_esatto_ok"] is False and k["hash_ok"] is True


def test_hash_mascherato_non_nasconde_altri_byte(tmp_path):
    m = _maps(_CASI_CONTATORE[0][0])
    if m is None:
        pytest.skip("run di riferimento non presente")
    a = bytearray((m / _CASI_CONTATORE[0][1]).read_bytes())
    off = tv.campi_pubblicazione_mappa(a)["advertise_counter"][0]
    for d in (off - 1, off + 4, off + 6, len(a) - 1):      # outside both fields, tail
        b = bytearray(a)
        b[d] ^= 0xFF
        pa, pb = tmp_path / "a.molastate", tmp_path / f"b{d}.molastate"
        pa.write_bytes(a)
        pb.write_bytes(b)
        assert tv.confronto_stato_mascherato(str(pa), str(pb))["uguale"] is False, d
    b = bytearray(a)
    b[off] ^= 0x01
    b[off + 5] ^= 0x01
    pb = tmp_path / "campi.molastate"
    pb.write_bytes(b)
    assert tv.confronto_stato_mascherato(str(tmp_path / "a.molastate"), str(pb))["uguale"] is True


def test_hash_mascherato_rifiuta_formati_diversi(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"MOLAFS01" + (9).to_bytes(4, "little") + b"\0" * 64)
    b.write_bytes(b"MOLAFS01" + (9).to_bytes(4, "little") + b"\1" * 64)
    r = tv.confronto_stato_mascherato(str(a), str(b))
    assert r["uguale"] is False and "versione" in r["motivo"]
    b.write_bytes(b"x" * 10)
    assert tv.confronto_stato_mascherato(str(a), str(b))["uguale"] is False


def test_hash_mascherato_file_troncato_e_ombra_cambiata(tmp_path):
    m = _maps(_CASI_CONTATORE[0][0])
    if m is None:
        pytest.skip("run di riferimento non presente")
    a = (m / _CASI_CONTATORE[0][1]).read_bytes()
    pa, pb = tmp_path / "a.molastate", tmp_path / "b.molastate"
    pa.write_bytes(a[:200])                         # truncated inside section 1
    pb.write_bytes(a[:200])
    r = tv.confronto_stato_mascherato(str(pa), str(pb))
    assert r["uguale"] is False and r["motivo"]
    # shadow rewritten after the logged hash: the masked comparison does not use it
    pa.write_bytes(a)
    sha = tv.file_sha256(str(pa))
    b = bytearray(a)
    b[tv.campi_pubblicazione_mappa(a)["advertise_counter"][0]] ^= 0x01
    pb.write_bytes(b)
    pa.write_bytes(b)                               # now the shadow on disk equals the re-saved one
    st = {"state_path": str(pa), "state_sha256": sha, "state_stamp_tick": None}
    k = tv.g_rollout_check(_log(str(pa)), st, str(pb), None)
    assert k["hash_esatto_ok"] is False and k["hash_ok"] is False and "hash_mascherato" not in k
