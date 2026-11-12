#!/usr/bin/env python3
"""Tests of twin variant E ("save and load", --twin-state ombra_e) and of the options
--mola-stop-signal, --fixed-genome-seq, --victim-mola-env, without ROS or Isaac. Real data
from earlier runs are read only.

    python3 -m pytest tests/test_variante_E.py -q
"""

import inspect
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimization import twin_variants as tv  # noqa: E402

DATI = Path.home() / "mola-adversarial-nsga3/data/attack/campaign"
LOG_VITTIMA = DATI / "straight/none_g3_sel2_H1.0_baseline_live_r3/rep_01/logs/mola_victim.log"
MM_VERO = DATI / "L/nsga3_g3_p24x12_lag0.5_warm_fase3_H1_maprestore/rep_01/maps/victim_w10.mm"
NB = DATI / "straight/nsga3_g3_p32x16_lag0.5_sel3_warm_H1.0_nobudget_dannobaseline/rep_01/history.json"


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


# ---------------------------------------------------------------------------
# Option, combinations, shadow ROS_ARGS, environment
# ---------------------------------------------------------------------------

def test_default_invariato_e_ombra_tra_gli_stati():
    ao = _ao()
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    i = src.index('"--twin-state"')
    assert 'default="clean"' in src[i:i + 200]
    assert "ombra_e" in tv.TWIN_STATES and ao.TWIN_STATE["mode"] == "clean"


@pytest.mark.parametrize("argv", [
    ["--twin-state", "ombra_e", "--uniform-sim-time", "true", "--victim", "restart"],
    ["--twin-state", "ombra_e"],                       # uniform-sim-time off
])
def test_main_rifiuta_ombra_e_senza_vittima_o_con_sim_time_misto(monkeypatch, capsys, argv):
    assert _main(monkeypatch, argv) == 2
    assert "ERRORE" in capsys.readouterr().out


def test_ros_args_dell_ombra():
    assert json.loads(tv.mola_ros_args("shadow")) == ["--ros-args", "-r", "__ns:=/ombra"]
    # same use_sim_time as the victim
    for u in ("off", "true", "false"):
        v = json.loads(tv.mola_ros_args("victim", u))
        s = json.loads(tv.mola_ros_args("shadow", u))
        assert s[3:] == v[3:] and s[2] == "__ns:=/ombra"


def test_ambiente_dell_ombra():
    ao = _ao()
    env = ao.shadow_extra_env({"LIDAR_POSE_X": -0.25}, "RA", environ={})
    assert env["ROS_ARGS"] == "RA" and env["LIDAR_POSE_X"] == -0.25
    assert env["MOLA_LO_DEBUG_ICP_QUALITY"] == "1"
    if os.path.isfile(tv.STDBUF_LIB):
        assert env["LD_PRELOAD"] == tv.STDBUF_LIB and env["_STDBUF_O"] == "L"
        env2 = ao.shadow_extra_env({}, "RA", environ={"LD_PRELOAD": "/x.so"})
        assert env2["LD_PRELOAD"] == f"/x.so:{tv.STDBUF_LIB}"
    r = ao.ShadowRunner("s.sh", Path("/tmp") / "nonusata_ombra_test", ros_args=None)
    assert r.args == [ao.VICTIM_TOPIC] and r.extra_env["ROS_ARGS"] == tv.mola_ros_args("shadow")


def _ambiente_effettivo(runner):
    # as MolaRunner.spawn: inherited environment plus the runner extra_env
    env = os.environ.copy()
    env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    env.update({k: str(v) for k, v in runner.extra_env.items()})
    return env


@pytest.mark.parametrize("ld_preload", ["", "/x.so", "/a.so:/b.so"])
def test_vittima_con_ambiente_dell_ombra(monkeypatch, ld_preload):
    ao = _ao()
    from src.optimization import shadow_check as shc
    if ld_preload:
        monkeypatch.setenv("LD_PRELOAD", ld_preload)
    else:
        monkeypatch.delenv("LD_PRELOAD", raising=False)
    cfg = {"LIDAR_POSE_X": -0.25}
    d = Path("/tmp") / "nonusata_ombra_test"
    ra_v, ra_s = tv.mola_ros_args("victim"), tv.mola_ros_args("shadow")
    # default: victim environment unchanged
    assert ao.victim_extra_env(cfg, {"A": "1"}, ra_v) == {**cfg, "A": "1"}
    sh = ao.ShadowRunner("s.sh", d, ros_args=ra_s, env=cfg)
    for opz in (False, True):
        # as in main: runner with the usual environment, then the option
        v = ao.VictimRunner("s.sh", d, ros_args=ra_v, env={**cfg, **{}})
        if opz:
            v.extra_env = ao.victim_extra_env(cfg, {}, ra_v, shadow_env=True)
        # "identical environment" check of --shadow-check (same lines as main)
        inh = {k: x for k, x in os.environ.items() if k.startswith(("MOLA_", "LD_PRELOAD", "_STDBUF"))}
        r = shc.env_config_diff({**inh, **v.extra_env}, {**inh, **sh.extra_env})
        ev, es = _ambiente_effettivo(v), _ambiente_effettivo(sh)
        diverse = sorted(k for k in set(ev) | set(es) if ev.get(k) != es.get(k))
        if opz:
            assert r["identica"] and r["diverse"] == []
            assert diverse == ["ROS_ARGS"]                 # namespace only
            assert v.extra_env["ROS_ARGS"] == ra_v and v.args == [ao.VICTIM_TOPIC]
            if os.path.isfile(tv.STDBUF_LIB):
                atteso = f"{ld_preload}:{tv.STDBUF_LIB}" if ld_preload else tv.STDBUF_LIB
                assert ev["LD_PRELOAD"] == es["LD_PRELOAD"] == atteso
        else:
            assert "MOLA_LO_DEBUG_ICP_QUALITY" in r["diverse"]
            assert "MOLA_LO_DEBUG_ICP_QUALITY" not in v.extra_env


def test_opzione_ambiente_ombra_spenta_di_default():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    i = src.index('ap.add_argument("--victim-shadow-env"')
    assert 'action="store_true"' in src[i:i + 80]
    m = inspect.getsource(_ao().main)
    assert re.search(r"if victim is not None and args\.victim_shadow_env:\s+#[^\n]*\n\s+victim\.extra_env = "
                     r"victim_extra_env\(cfg_env, victim_env, ros_args\[\"victim\"\], shadow_env=True\)", m)
    # the runner before the option: ROS_ARGS equal to what victim_extra_env sets
    v = _ao().VictimRunner("s.sh", Path("/tmp") / "nonusata_ombra_test", ros_args=None, env={})
    assert _ao().victim_extra_env({}, {}, None, shadow_env=True)["ROS_ARGS"] == v.extra_env["ROS_ARGS"]
    ao = _ao()
    a = SimpleNamespace(twin_state="clean", replay_frame="twin", twin_init="true", uniform_sim_time="off",
                        drift_reset_on="genome", twin_align="attacker")
    assert ao.twin_config_fields(a)["victim_shadow_env"] is False


def test_coerenza_ombra_e():
    c = tv.coherence_record("ombra_e")
    assert c["twin_state_coerente_threat_model"] is True
    assert "lab_" in c["twin_state_motivo"]


# ---------------------------------------------------------------------------
# Pose, sigma, twin variables, .mm check
# ---------------------------------------------------------------------------

def test_posa_dal_quaternione():
    yaw = 0.7
    p = tv.pose6_from_quat(1.0, 2.0, 0.01, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    assert p[:3] == (1.0, 2.0, 0.01)
    assert p[3] == pytest.approx(yaw) and p[4] == pytest.approx(0.0) and p[5] == pytest.approx(0.0)


def test_variabili_del_gemello_in_gradi_e_senza_esponente():
    env = tv.e_initial_env("/a/b.mm", (1e-6, -2.5, 0.0, math.pi / 2, 0.0, 0.0), 0.04)
    assert env["MOLA_LOAD_MM"] == "/a/b.mm"
    assert float(env["MOLA_INITIAL_YAW"]) == pytest.approx(90.0)
    assert env["MOLA_INITIAL_X"] == "0.000001000" and env["MOLA_INITIAL_Y"] == "-2.500000000"
    assert not any("e" in v.lower() for k, v in env.items() if k.startswith("MOLA_INITIAL"))
    assert env["MOLA_INITIAL_SIGMA"] == "0.040000"
    assert "MOLA_INITIAL_SIGMA" not in tv.e_initial_env("/a/b.mm", (0,) * 6, None)


def test_controllo_del_mm(tmp_path):
    assert "assente" in tv.mm_problem(None)
    assert "non esiste" in tv.mm_problem(str(tmp_path / "x.mm"))
    (tmp_path / "v.mm").write_bytes(b"")
    assert "vuoto" in tv.mm_problem(str(tmp_path / "v.mm"))
    (tmp_path / "p.mm").write_bytes(b"abc")
    assert tv.mm_problem(str(tmp_path / "p.mm")) is None
    assert tv.file_sha256(tmp_path / "p.mm") == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sigma_dal_log_sintetico(tmp_path):
    p = tmp_path / "mola_shadow.log"
    p.write_text("".join(riga_printf(i, 20.0 + 0.1 * i, 0.2 - 0.01 * i) for i in range(6)))
    s = tv.shadow_sigma_from_log(p, 20.5)
    assert s["sigma"] == pytest.approx(0.15) and s["sigma_aligned"] and s["sigma_pathstep"] == 5
    # pose older than the last row: the row of the same scan is taken
    s = tv.shadow_sigma_from_log(p, 20.3)
    assert s["sigma"] == pytest.approx(0.17) and s["sigma_aligned"]
    # pose newer than the last row (row still in the buffer): not aligned
    s = tv.shadow_sigma_from_log(p, 21.0)
    assert s["sigma"] == pytest.approx(0.15) and not s["sigma_aligned"]
    assert tv.shadow_sigma_from_log(tmp_path / "manca.log", 20.0)["sigma"] is None


@pytest.mark.skipif(not LOG_VITTIMA.is_file(), reason="log della baseline live r3 assente")
def test_sigma_dal_log_vero():
    # last printf rows of the log (grep): pathStep 135 at 25.8 s and 136 at 25.9 s, sigma 0.0400
    s = tv.shadow_sigma_from_log(LOG_VITTIMA, 25.9)
    assert s["sigma"] == 0.04 and s["sigma_pathstep"] == 136 and s["sigma_aligned"]
    s = tv.shadow_sigma_from_log(LOG_VITTIMA, 25.84)
    assert s["sigma_pathstep"] == 135 and s["sigma_aligned"]
    # 135 rows with "adapt_thres_sigma=", 2 of them broken by ROS stderr warnings (start at
    # log rows 229 and 350, continued at 245 and 365): leggi_printf rebuilds them all
    n = sum(1 for r in LOG_VITTIMA.read_text(errors="replace").splitlines() if "adapt_thres_sigma=" in r)
    assert n == 135 and s["sigma_rows"] == 135


@pytest.mark.skipif(not MM_VERO.is_file(), reason=".mm della L maprestore assente")
def test_mm_vero_controllato_e_hash():
    assert tv.mm_problem(str(MM_VERO)) is None
    assert os.path.getsize(MM_VERO) == 8609059          # size of the recorded map (windows w8-w10)
    assert len(tv.file_sha256(MM_VERO)) == 64


# ---------------------------------------------------------------------------
# Shadow state saving (fake node)
# ---------------------------------------------------------------------------

class NodoFinto:
    def __init__(self, ok=True, scrivi=True):
        self.shadow_est_t = time.time() - 5.0
        self.shadow_pose6 = (0.5, 0.25, 0.0, 0.3, 0.0, 0.0)
        self.shadow_est_raw = (0.5, 0.25, 0.3)
        self.shadow_est_stamp = 20.5
        self.ok, self.scrivi, self.saved = ok, scrivi, []

    def spin(self, s):
        time.sleep(s)

    def shadow_map_save(self, path):
        self.saved.append(str(path))
        if self.scrivi:
            Path(f"{path}.mm").write_bytes(b"mappa")
        return self.ok, "" if self.ok else "errore finto"

    compose = staticmethod(lambda b, l: _ao().AttackNode.compose(b, l))


def _ombra(tmp_path, log_righe=True):
    ao = _ao()
    r = ao.ShadowRunner("s.sh", tmp_path / "logs")
    r.log_path = tmp_path / "logs" / "mola_shadow.log"
    r.log_path.write_text("".join(riga_printf(i, 20.0 + 0.1 * i, 0.2 - 0.01 * i) for i in range(6))
                          if log_righe else "")
    r.alive = lambda: True
    r.run_origin = (1.0, 2.0, math.pi / 2)
    return r


def test_salvataggio_dello_stato(tmp_path):
    r, node = _ombra(tmp_path), NodoFinto()
    (tmp_path / "maps").mkdir()
    st = r.save_state(node, tmp_path / "maps" / "ombra_w1")
    assert st["status"] == "ok" and st["quiet_ok"] and node.saved == [str(tmp_path / "maps" / "ombra_w1")]
    assert st["mm_bytes"] == 5 and len(st["mm_sha256"]) == 64
    assert st["sigma"] == pytest.approx(0.15) and st["sigma_aligned"]
    env = st["env"]
    # shadow pose passed as is (shadow frame = twin frame)
    assert float(env["MOLA_INITIAL_X"]) == 0.5 and float(env["MOLA_INITIAL_Y"]) == 0.25
    assert float(env["MOLA_INITIAL_YAW"]) == pytest.approx(math.degrees(0.3))
    assert env["MOLA_LOAD_MM"] == f"{tmp_path / 'maps' / 'ombra_w1'}.mm"
    # believed pose in the working frame: shadow origin composed with its estimate
    b = st["believed"]
    assert b[0] == pytest.approx(1.0 - 0.25) and b[1] == pytest.approx(2.0 + 0.5)
    assert b[2] == pytest.approx(math.pi / 2 + 0.3)
    assert not any(k.startswith("lab_") or "victim" in k for k in st)


def test_salvataggio_fallito(tmp_path):
    (tmp_path / "maps").mkdir()
    assert _ombra(tmp_path).save_state(NodoFinto(ok=False), tmp_path / "maps" / "a")["status"] == "map_save_failed"
    st = _ombra(tmp_path).save_state(NodoFinto(scrivi=False), tmp_path / "maps" / "b")
    assert st["status"] == "map_save_failed" and "non esiste" in st["error"]
    assert _ombra(tmp_path, log_righe=False).save_state(NodoFinto(), tmp_path / "maps" / "c")["status"] == "no_sigma"


def test_attesa_di_quiete(tmp_path, monkeypatch):
    ao = _ao()
    monkeypatch.setattr(ao, "SHADOW_QUIET_S", 0.3)
    node = NodoFinto()
    node.shadow_est_t = time.time()
    t0 = time.time()
    st = _ombra(tmp_path).save_state(node, tmp_path / "maps_q")
    assert st["quiet_ok"] and time.time() - t0 >= 0.3


# ---------------------------------------------------------------------------
# Per-candidate twin, fingerprint, alignment
# ---------------------------------------------------------------------------

class IsaacFinto:
    def pause(self): pass
    def restore(self): return [0, 0, 0]


class MolaFinta:
    def __init__(self):
        self.extra_env = {"BASE": "1"}
        self.spawned = None
        self.last_stop = None

    def stop(self): pass

    def spawn(self, tag):
        self.spawned = (tag, dict(self.extra_env))
        raise RuntimeError("fermata del test dopo lo spawn")


def test_gemello_senza_stato_non_parte(monkeypatch, tmp_path):
    ao = _ao()
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_e", "shadow_state": {"status": "map_save_failed"}})
    mola = MolaFinta()
    r = ao._rollout_once(None, IsaacFinto(), mola, SimpleNamespace(), (1, 0), 1.0, "w0_e0")
    assert r[3].startswith("ombra_e_state_missing") and mola.spawned is None
    st = {"status": "ok", "mm_path": str(tmp_path / "manca.mm"), "env": {}}
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_e", "shadow_state": st})
    r = ao._rollout_once(None, IsaacFinto(), mola, SimpleNamespace(), (1, 0), 1.0, "w0_e0")
    assert "non esiste" in r[3] and mola.spawned is None
    # the outcome class is an infrastructure failure (discarded)
    assert ao.outcome_kind(r[3]) == "infra"


def test_gemello_parte_con_le_variabili_dell_ombra(monkeypatch, tmp_path):
    ao = _ao()
    (tmp_path / "o.mm").write_bytes(b"m")
    env = tv.e_initial_env(str(tmp_path / "o.mm"), (0.5, 0.25, 0.0, 0.3, 0.0, 0.0), 0.04)
    st = {"status": "ok", "mm_path": str(tmp_path / "o.mm"), "env": env, "pose6": [0.5, 0.25, 0, 0.3, 0, 0]}
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_e", "shadow_state": st,
                                           "twin_env_base": {"BASE": "1", "MOLA_INITIAL_SIGMA": 0.2}})
    mola = MolaFinta()
    r = ao._rollout_once(None, IsaacFinto(), mola, SimpleNamespace(), (1, 0), 1.0, "w0_e3")
    assert r[3].startswith("mola_spawn_fail")
    tag, e = mola.spawned
    assert tag == "w0_e3" and e["BASE"] == "1"
    assert e["MOLA_INITIAL_SIGMA"] == "0.040000" and e["MOLA_LOAD_MM"] == str(tmp_path / "o.mm")
    # the other modes do not touch the twin environment
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "clean"})
    mola2 = MolaFinta()
    ao._rollout_once(None, IsaacFinto(), mola2, SimpleNamespace(), (1, 0), 1.0, "w0_e0")
    assert mola2.spawned[1] == {"BASE": "1"}


def test_impronta_del_candidato():
    st = {"status": "ok", "mm_path": "/m.mm", "mm_sha256": "ab", "mm_bytes": 3,
          "pose6": [1, 2, 0, 0.1, 0, 0], "pose_stamp": 20.5, "sigma": 0.04, "sigma_aligned": True,
          "env": {"MOLA_LOAD_MM": "/m.mm"}}
    fp = tv.e_fingerprint(st)
    assert fp["source"] == "ombra_e" and fp["mm_sha256"] == "ab" and fp["sigma_loaded"] == 0.04
    assert fp["pose_loaded"] == [1, 2, 0, 0.1, 0, 0]
    assert set(fp["campi_diversi"]) == {"twist", "step_counter_post_relocalization",
                                        "kf_decider_local_map", "estimated_observation_radius"}
    src = inspect.getsource(_ao()._rollout_once)
    assert "fp.update(tv.e_fingerprint(e_state))" in src


def test_allineamento_nel_frame_dell_ombra(monkeypatch):
    ao = _ao()
    st = {"origin": [1.0, 2.0, 0.0], "believed": [1.5, 2.0, 0.0]}
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_e", "shadow_state": st, "saved": (1.5, 2.0, 0.0),
                                           "attacker_belief": (9.0, 9.0, 0.0)})
    rec = ao._twin_alignment_record(None, (1.5, 2.0, 0.0), {"source": "ombra_e", "est_raw": [0.5, 0.01, 0.0]})
    assert rec["twin_est_frame"] == "origine dell'ombra"
    assert rec["est_gap_cm"] == pytest.approx(1.0) and rec["attacker_world"] == [1.5, 2.0, 0.0]


def test_campi_lab_e_ombra_in_history():
    ao = _ao()
    src = inspect.getsource(ao.main)
    i = src.index("sh = shadow.save_state(node, mp)")
    # up to the physical target of ombra_est, which uses saved/saved_scene only to
    # go from the odom frame to the scene frame (as attacker_est)
    blocco = src[i:src.index('if args.twin_init == "ombra_est"', i)]
    # true pose and victim only in lab_* fields
    for parola in ("saved", "victim.last_est_end"):
        for riga in blocco.splitlines():
            if parola in riga and "sh[" in riga:
                assert 'sh["lab_' in riga, riga
    import argparse
    a = argparse.Namespace(twin_state="ombra_e", replay_frame="twin", twin_init="true",
                           uniform_sim_time="true", drift_reset_on="genome", twin_align="attacker")
    rec = ao.twin_config_fields(a)
    assert "shadow_state" in rec and rec["twin_state_coerente_threat_model"] is True


# ---------------------------------------------------------------------------
# (a) --mola-stop-signal
# ---------------------------------------------------------------------------

def _processo(tmp_path, script):
    f = open(tmp_path / "p.log", "w")
    p = subprocess.Popen(["bash", "-c", script], stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    time.sleep(0.3)        # traps installed
    return p, f


def test_stop_con_sigint_misura_l_uscita(tmp_path, monkeypatch):
    ao = _ao()
    monkeypatch.setitem(ao.MOLA_STOP, "signal", "int")
    r = ao.MolaRunner("s.sh", tmp_path / "l")
    r.proc, r._f = _processo(tmp_path, 'trap "exit 0" INT; trap "" TERM; while true; do sleep 0.05; done')
    r.log_path = tmp_path / "p.log"
    r.stop()
    assert r.last_stop["signal"] == "int" and not r.last_stop["killed"]
    assert r.last_stop["exit_s"] is not None and r.last_stop["exit_s"] < 2.0
    assert r.last_stop["grace_s"] == ao.MOLA_STOP_GRACE_SEC and r.proc is None


def test_stop_con_sigterm_ignorato_finisce_al_sigkill(tmp_path, monkeypatch):
    ao = _ao()
    assert ao.MOLA_STOP["signal"] == "term"           # default
    monkeypatch.setattr(ao, "MOLA_STOP_GRACE_SEC", 0.5)
    r = ao.MolaRunner("s.sh", tmp_path / "l")
    r.proc, r._f = _processo(tmp_path, 'trap "" TERM; while true; do sleep 0.05; done')
    r.stop()
    assert r.last_stop["signal"] == "term" and r.last_stop["killed"] and r.last_stop["exit_s"] is None


def test_stop_opzione_e_history(monkeypatch):
    ao = _ao()
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    i = src.index('"--mola-stop-signal"')
    assert 'default="term"' in src[i:i + 200]
    assert 'MOLA_STOP["signal"] = args.mola_stop_signal' in inspect.getsource(ao.main)
    doc = ao.MolaRunner.kill_orphans.__doc__
    assert "chiudendo le connessioni DDS" not in doc and "SIGINT" in doc

    class M:
        last_stop = None

    def finto(genome, isaac, mola, node, *a, **kw):
        mola.last_stop = {"signal": "int", "exit_s": 0.4}
        return (None,) * 6
    monkeypatch.setattr(ao, "_rollout_once", finto)
    monkeypatch.setattr(ao, "_note_twin_log", lambda n, m: None)
    monkeypatch.setattr(ao, "_note_twin_fp", lambda n, m: None)
    node = SimpleNamespace()
    ao.rollout(None, None, M(), node, (1, 0), 1.0, "w0_e0")
    assert node.last_mola_stop == {"signal": "int", "exit_s": 0.4}
    assert ao._twin_start_fields(node)["mola_stop"]["exit_s"] == 0.4


# ---------------------------------------------------------------------------
# (b) --fixed-genome-seq
# ---------------------------------------------------------------------------

def test_sequenza_da_json_e_da_stringa(tmp_path):
    ao = _ao()
    g1, g2 = [0.1] * 17, [-1.0] * 16 + [0.5]
    h = tmp_path / "history.json"
    h.write_text(json.dumps([{"window": 0, "genome": g1,
                              "evaluations": [{"genome": g2}]}]))
    f = tmp_path / "seq.json"
    f.write_text(json.dumps([g1, ",".join(map(str, g2)), f"{h}:0", f"{h}:0:0"]))
    seq = ao.load_genome_seq(str(f))
    assert len(seq) == 4
    assert np.array_equal(seq[0][0], g1) and np.array_equal(seq[1][0], g2)
    assert np.array_equal(seq[2][0], g1) and seq[2][1]["source_window"] == 0
    assert seq[2][1]["source_evaluation"] is None
    assert np.array_equal(seq[3][0], g2) and seq[3][1]["source_evaluation"] == 0
    seq2 = ao.load_genome_seq(f"{h}:0;{h}:0:0")       # file format of a recorded run
    assert np.array_equal(seq2[1][0], g2)
    with pytest.raises(ValueError):
        ao.load_genome_seq("0.1,0.2")
    f.write_text("{}")
    with pytest.raises(ValueError):
        ao.load_genome_seq(str(f))


@pytest.mark.skipif(not NB.is_file(), reason="history della nobudget assente")
def test_sequenza_dalla_history_vera():
    ao = _ao()
    # recorded entries: history index 2 candidate 127, index 3 candidate 330
    seq = ao.load_genome_seq(f"{NB}:2:127;{NB}:3:330")
    hist = json.loads(NB.read_text())
    assert np.array_equal(seq[0][0], hist[2]["evaluations"][127]["genome"])
    assert np.array_equal(seq[1][0], hist[3]["evaluations"][330]["genome"])
    assert seq[0][1]["source_window"] == hist[2].get("window")
    # in history the position in the sequence is not hidden by the history index
    recs = [ao.genome_seq_entry(seq, k)[1] for k in range(4)]
    assert [r["seq_pos"] for r in recs] == [0, 1, 1, 1]
    assert [r["source_index"] for r in recs] == [2, 3, 3, 3]
    assert [r["source_evaluation"] for r in recs] == [127, 330, 330, 330]
    assert all("index" not in r for r in recs)


@pytest.mark.parametrize("argv", [
    ["--fixed-genome-seq", "x;y"],                                         # without --search fixed
    ["--search", "fixed", "--fixed-genome", ",".join(["0"] * 17), "--fixed-genome-seq", ",".join(["0"] * 17)],
    ["--search", "fixed", "--fixed-genome-seq", "0,1"],                    # 2 values
])
def test_main_rifiuta_sequenze_sbagliate(monkeypatch, capsys, argv):
    assert _main(monkeypatch, argv) == 2
    assert "ERRORE" in capsys.readouterr().out


def test_scelta_della_voce_per_finestra():
    ao = _ao()
    src = inspect.getsource(ao.main)
    assert "g_fix, seq_rec = genome_seq_entry(genome_seq, k)" in src
    assert '**({"fixed_genome_seq": seq_rec} if seq_rec is not None else {})' in src
    seq = [(np.zeros(17), {"entry": "a", "history": "h.json", "source_index": 2, "source_window": 2}),
           (np.ones(17), {"entry": "b"})]
    g, r = ao.genome_seq_entry(seq, 0)
    assert r["seq_pos"] == 0 and r["source_index"] == 2 and r["n"] == 2
    g, r = ao.genome_seq_entry(seq, 7)
    assert r["seq_pos"] == 1 and np.array_equal(g, np.ones(17))


# ---------------------------------------------------------------------------
# (c) --victim-mola-env
# ---------------------------------------------------------------------------

def test_variabili_della_sola_vittima(tmp_path, monkeypatch, capsys):
    ao = _ao()
    assert ao.parse_env_pairs([]) == {}
    assert ao.parse_env_pairs(["MOLA_VERBOSITY_MOLA_LO=DEBUG", "A=b=c"]) == \
        {"MOLA_VERBOSITY_MOLA_LO": "DEBUG", "A": "b=c"}
    for bad in (["X"], ["=1"], ["ROS_ARGS=[]"], ["TRACES=/x"]):
        with pytest.raises(ValueError):
            ao.parse_env_pairs(bad)
    assert _main(monkeypatch, ["--victim-mola-env", "SENZAUGUALE"]) == 2
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    i = src.index('"--victim-mola-env"')
    assert "default=[]" in src[i:i + 200]
    m = inspect.getsource(ao.main)
    assert "env={**cfg_env, **victim_env})" in m
    v = ao.VictimRunner("s.sh", tmp_path / "l", ros_args=None, env={**{}, **{"MOLA_VERBOSITY_MOLA_LO": "DEBUG"}})
    assert v.extra_env == {"MOLA_VERBOSITY_MOLA_LO": "DEBUG", "ROS_ARGS": ao.VICTIM_ROS_ARGS}
    # shadow and attacker receive only cfg_env
    assert re.search(r"ShadowRunner\(script.*?env=cfg_env\)", m, re.S)
    assert re.search(r"AttackerRunner\(script.*?env=cfg_env\)", m, re.S)


# ---------------------------------------------------------------------------
# Shadow with repetitions, ombra_est target, missing shadow state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [
    ["--repeat-genome", "x/history.json:0"],
    ["--repeat-spec", "x/spec.json"],
    ["--map-gate-test"],
])
def test_ombra_e_rifiutata_con_le_ripetizioni(monkeypatch, capsys, extra):
    argv = ["--twin-state", "ombra_e", "--uniform-sim-time", "true"] + extra
    assert _main(monkeypatch, argv) == 2
    assert "non va con --repeat-genome" in capsys.readouterr().out
    import argparse
    a = argparse.Namespace(twin_state="ombra_e", victim="continuous", uniform_sim_time="true",
                           repeat_genome=None, repeat_spec=None, map_gate_test=False)
    assert tv.check_twin_args(a) is None
    a.repeat_spec = "s.json"
    assert "--repeat-spec" in tv.check_twin_args(a)


def test_twin_init_ombra_est_opzione_e_coerenza(monkeypatch, capsys):
    assert _main(monkeypatch, ["--twin-init", "ombra_est"]) == 2      # without ombra_e
    assert "ombra_est" in capsys.readouterr().out
    import argparse
    a = argparse.Namespace(twin_state="ombra_e", victim="continuous", uniform_sim_time="true",
                           twin_init="ombra_est")
    assert tv.check_twin_args(a) is None
    c = tv.coherence_record("ombra_e", "ombra_est")
    assert c["twin_init_coerente_threat_model"] is True and "ombra" in c["twin_init_motivo"]
    assert tv.coherence_record("clean")["twin_init_coerente_threat_model"] is False   # unchanged
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    i = src.index('"--twin-init"')
    assert 'default="true"' in src[i:i + 200]


def test_bersaglio_fisico_dalla_posa_dell_ombra():
    ao = _ao()
    saved, saved_scene = (1.0, 0.0, 0.0), (11.0, 5.0, 0.5)
    believed = (1.2, 0.1, 0.05)                       # shadow origin composed with p_S
    buf = [(10.0 + 0.1 * i, 0.4 * 0.1 * i, 0.0, 0.0) for i in range(10)]   # 0.4 m/s in frame S
    node = SimpleNamespace(shadow_buf=buf, attacker_buf=[])
    shadow = SimpleNamespace(run_origin=(0.0, 0.0, math.pi / 2))
    r0 = ao.attacker_init_target(node, shadow, saved, saved_scene, believed, "zero", buf_name="shadow_buf")
    ref = tv.twin_init_target(saved, saved_scene, believed, None)
    assert r0["target_scene"] == pytest.approx(ref["target_scene"]) and r0["zero_joints"] is True
    # poses of the last 0.5 s (tv.VEL_SPAN_S): 6 of the 10
    assert r0["v_set_scene"] == [0.0, 0.0] and r0["vel_poses_n"] == 6
    assert r0["v_att"][1] == pytest.approx(0.4)       # rotated with the shadow origin
    r1 = ao.attacker_init_target(node, shadow, saved, saved_scene, believed, "poses", buf_name="shadow_buf")
    assert math.hypot(*r1["v_set_scene"]) == pytest.approx(0.4)
    # the attacker buffer is not used
    r2 = ao.attacker_init_target(node, shadow, saved, saved_scene, believed, "poses")
    assert r2["vel_poses_n"] == 0


def test_rollout_applica_il_bersaglio_ombra_est(monkeypatch, tmp_path):
    ao = _ao()
    (tmp_path / "o.mm").write_bytes(b"m")
    st = {"status": "ok", "mm_path": str(tmp_path / "o.mm"), "pose6": [0, 0, 0, 0, 0, 0],
          "env": tv.e_initial_env(str(tmp_path / "o.mm"), (0,) * 6, 0.04)}
    tgt = {"target_scene": [11.2, 5.1, 0.55], "v_set_scene": [0.0, 0.0], "w_set": 0.0, "zero_joints": True}
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_e", "shadow_state": st, "init": "ombra_est",
                                           "init_target": tgt, "twin_env_base": {}})
    chiamate = []
    isaac = IsaacFinto()
    isaac.set_state = lambda *a, **kw: chiamate.append((a, kw))
    ao._rollout_once(None, isaac, MolaFinta(), SimpleNamespace(), (1, 0), 1.0, "w0_e0")
    assert chiamate == [((11.2, 5.1, 0.55, 0.0, 0.0, 0.0), {"zero_joints": True})]


def test_finestra_con_stato_dell_ombra_mancante_e_un_guasto_marcato():
    ao = _ao()
    st = "ombra_e_state_missing: stato dell'ombra non salvato (no_sigma)"
    assert ao.outcome_kind(st) == "infra"
    sh = {"status": "no_sigma"}
    rec = ao.skipped_window_record(3, st, (5.0, 0.0), 0, "ok", 12.0, sh)
    assert rec["status"].startswith("nominal_failed(ombra_e_state_missing")
    assert rec["infra_failure"] == "ombra_e_state_missing" and rec["applied"] is False
    assert rec["shadow_state"] == sh and rec["victim_status"] == "ok"
    # other nominal failures: record unchanged
    rec2 = ao.skipped_window_record(3, "mola_not_ready: x", (5.0, 0.0), 0, None, 1.0, None)
    assert set(rec2) == {"window", "status", "goal", "waypoint_index", "victim_status", "window_time_s"}
    # all candidates without state after a successful nominal (no_admissible)
    log = [{"status": st}, {"status": st}]
    assert ao.ombra_infra_fields(log) == {"n_ombra_e_state_missing": 2,
                                         "infra_failure": "ombra_e_state_missing", "applied": False}
    assert ao.ombra_infra_fields([{"status": st}, {"status": "ok"}]) == {"n_ombra_e_state_missing": 1}
    assert ao.ombra_infra_fields([{"status": "ok"}]) == {}
    m = inspect.getsource(ao.main)
    assert "skipped_window_record(k, st, goal, wp_idx," in m and "**ombra_infra_fields(eval_log)," in m
    # the nominal with missing state fails before starting MOLA (as the candidates)
    src = inspect.getsource(ao._rollout_once)
    assert src.index("ombra_e_state_missing") < src.index("mola.spawn(tag)")
