"""Bench G, --g-banco: spec, physical targets, rollout order, printf lines and, with the
option off, unchanged orchestrator behaviour. Fake objects, no ROS node or MOLA."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.optimization import attack_orchestrator as ao
from src.optimization import banco_g
from src.optimization import twin_variants as tv

ROOT = Path(__file__).resolve().parents[1]
RUN_G_S1001 = (Path(__file__).resolve().parents[1] / "data/attack/verifica/2026-10-26_pilota_G/s1001/"
                   "campagna/straight/nsga3_g3_p4x2_lag0.5_sel2_warm_stopramp_k1.0_reliable_serieB_s1001_"
                   "pt1_oprandom_sgross_pilotaT5_dmin130/rep_01")
RIGHE = (
    "[LidarOdometry] pathStep=230 timestamp=1970/01/01,00:00:31.900000 goodness=0.000 minRequired=0.200 "
    "iters=7 termReason=IterTermReason::MaxIterations isGood=0 adapt_thres_sigma=0.0400 consecutive_bad=0\n"
    "[04:10:00.0950|INFO |LidarOdometry] [full_state] caricato /x.molastate (6654775 byte)\n"
    "[LidarOdometry] pathStep=231 timestamp=1970/01/01,00:00:32.000000 goodness=0.406 minRequired=0.200 "
    "iters=14 termReason=IterTermReason::MaxIterations isGood=1 adapt_thres_sigma=0.0400 consecutive_bad=0\n")


def _state(path, tick=116444736318000000):
    b = tv.MOLASTATE_MAGIC + (2).to_bytes(4, "little") + bytes([1]) + int(tick).to_bytes(8, "little", signed=True)
    Path(path).write_bytes(b + b"x" * 64)
    return str(path)


def _overlay(tmp_path):
    root = tmp_path / "install_G1"
    for p in tv.G_OVERLAY_PKGS:
        (root / p / "lib" / "x86_64-linux-gnu").mkdir(parents=True)
    return str(root)


def _spec(tmp_path):
    sp = _state(tmp_path / "ombra_w4.molastate")
    return {"versione": 1, "ripetizioni": 2, "casi": [
        {"nome": "guasta", "state_path": sp, "state_sha256": tv.file_sha256(sp),
         "saved_odom": [3.947, -0.278, -0.092], "saved_scene": [-2.0529, -1.2781, -0.092],
         "origin": [0.0177, 0.0, 0.0], "believed": [4.085, 0.519, -0.119],
         "pose6": [4.067, 0.5193, -0.04, -0.1189, -0.0137, -0.0034],
         "attacker_target_scene": [-2.046, -1.278, -0.092],
         "bracci": [{"nome": "A", "init": "attacker_est"}, {"nome": "C", "init": "ombra_est"},
                    {"nome": "D", "init": "attacker_est",
                     "mola_env": {"MOLA_FULL_STATE_PROVA_TWIST_ZERO": "1"}}]},
        {"nome": "controllo", "state_path": sp, "state_sha256": tv.file_sha256(sp),
         "saved_odom": [2.95, -0.1887, -0.0667], "saved_scene": [-3.0499, -1.1887, -0.0667],
         "origin": [0.0177, 0.0, 0.0], "believed": [3.1472, 0.168, -0.0322],
         "pose6": [3.1294, 0.168, -0.0087, -0.0322, -0.0051, -0.0032],
         "attacker_target_scene": [-3.0435, -1.1886, -0.0667],
         "bracci": [{"nome": "A", "init": "attacker_est"}]}]}


def _args(**kw):
    base = dict(g_banco=None, victim="restart", twin_state="clean", twin_init="true",
                uniform_sim_time="true", attack_from="spawn", shadow_check=False,
                repeat_genome=None, repeat_spec=None, map_gate_test=False, g_overlay=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- option off -------------------------------------------------------------------

def test_default_argparse_spenta(monkeypatch):
    """Default orchestrator arguments: --g-banco None and no extra error."""
    preso = {}

    class _Stop(Exception):
        pass

    orig = ao.argparse.ArgumentParser.parse_args

    def _parse(self, *a, **k):
        preso["args"] = orig(self, *a, **k)
        raise _Stop

    monkeypatch.setattr(ao.argparse.ArgumentParser, "parse_args", _parse)
    monkeypatch.setattr(sys, "argv", ["attack_orchestrator.py"])
    with pytest.raises(_Stop):
        ao.main()
    args = preso["args"]
    assert args.g_banco is None
    assert banco_g.check_args(args) is None


@pytest.mark.parametrize("kw", [dict(victim="continuous"), dict(twin_state="ombra_g"),
                                dict(uniform_sim_time="off"), dict(shadow_check=True)])
def test_spenta_nessun_vincolo(kw):
    assert banco_g.check_args(_args(**kw)) is None


class _Isaac:
    def pause(self):
        pass

    def restore(self):
        return [0.0, 0.0, 0.0]


class _Mola:
    def __init__(self):
        self.extra_env = {"PRIMA": "1"}

    def stop(self):
        pass


class _Node:
    fresh_cloud = False

    def arm_twin_clock(self, tick):
        return False


def _ambiente_gemello(tmp_path, monkeypatch, extra=None):
    """Environment given to twin G by _rollout_once up to arming the clock (which fails here
    and closes the rollout)."""
    sp = _state(tmp_path / "s.molastate")
    ts = {"mode": "ombra_g", "init": "true", "g_root": _overlay(tmp_path), "g_resave": "off",
          "twin_env_base": {"ROS_ARGS": "x"},
          "shadow_state": {"status": "ok", "state_path": sp, "state_stamp_tick": 1}}
    if extra is not None:
        ts["g_env_extra"] = extra
    monkeypatch.setattr(ao, "TWIN_STATE", ts)
    mola = _Mola()
    res = ao._rollout_once(None, _Isaac(), mola, _Node(), [(1.0, 0.0)], 1.0, "w4_nominal")
    assert res[3] == "ombra_g_clock_not_armed"
    return mola.extra_env, sp, ts["g_root"]


def test_ambiente_gemello_invariato_senza_banco(tmp_path, monkeypatch):
    env, sp, root = _ambiente_gemello(tmp_path, monkeypatch)
    assert env == {"ROS_ARGS": "x", **tv.g_twin_env(sp, None, root=root)}
    assert "MOLA_FULL_STATE_PROVA_TWIST_ZERO" not in env


def test_ambiente_gemello_braccio_d(tmp_path, monkeypatch):
    env, sp, root = _ambiente_gemello(tmp_path, monkeypatch, {"MOLA_FULL_STATE_PROVA_TWIST_ZERO": "1"})
    assert env == {"ROS_ARGS": "x", **tv.g_twin_env(sp, None, root=root),
                   "MOLA_FULL_STATE_PROVA_TWIST_ZERO": "1"}


# --- constraints with the option on -----------------------------------------------

def test_vincoli_accesa(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec(tmp_path)))
    ov = _overlay(tmp_path)
    assert banco_g.check_args(_args(g_banco=str(p), g_overlay=ov)) is None
    assert "--victim restart" in banco_g.check_args(_args(g_banco=str(p), g_overlay=ov, victim="continuous"))
    assert "--twin-state clean" in banco_g.check_args(_args(g_banco=str(p), g_overlay=ov, twin_init="attacker_est"))
    assert "uniform-sim-time" in banco_g.check_args(_args(g_banco=str(p), g_overlay=ov, uniform_sim_time="off"))
    assert "overlay" in banco_g.check_args(_args(g_banco=str(p), g_overlay=str(tmp_path / "manca")))
    assert "--g-banco" in banco_g.check_args(_args(g_banco=str(tmp_path / "manca.json"), g_overlay=ov))


@pytest.mark.parametrize("guasto, msg", [
    (lambda s: s["casi"][0].update(state_sha256="bb7d38802409"), "64 cifre"),
    (lambda s: s["casi"][0]["bracci"][0].update(genoma=[0.0] * 17), "chiavi non gestite"),
    (lambda s: s["casi"][0]["bracci"][0].update(mola_env={"MOLA_LOAD_MM": "x"}), "non ammesse"),
    (lambda s: s["casi"][0]["bracci"][0].update(init="true"), "init"),
    (lambda s: s["casi"][1].update(nome="guasta"), "ripetuto"),
    (lambda s: s.update(ripetizioni=0), "ripetizioni"),
    (lambda s: s["casi"][0].update(pose6=[1, 2, 3]), "pose6"),
])
def test_spec_rifiutate(tmp_path, guasto, msg):
    s = _spec(tmp_path)
    guasto(s)
    with pytest.raises(ValueError, match=msg):
        banco_g.check_spec(s)


# --- targets, order, printf lines -------------------------------------------------

def test_bersagli_fisici(tmp_path):
    s = banco_g.check_spec(_spec(tmp_path))
    c = s["casi"][0]
    a = banco_g.init_target(c, c["bracci"][0])
    assert a["target_scene"] == c["attacker_target_scene"] and a["zero_joints"] is True
    assert a["v_set_scene"] == [0.0, 0.0] and a["w_set"] == 0.0
    cc = banco_g.init_target(c, c["bracci"][1])
    ref = tv.twin_init_target(c["saved_odom"], c["saved_scene"], c["believed"], None)
    assert cc["target_scene"] == ref["target_scene"] and cc["stimatore"] == "ombra"
    # the believed pose is 80.9 cm from the true one (G s1001 w4)
    d = ((cc["target_scene"][0] - c["saved_scene"][0]) ** 2 + (cc["target_scene"][1] - c["saved_scene"][1]) ** 2) ** 0.5
    assert abs(100 * d - 80.9) < 0.2


def test_ordine_alternato(tmp_path):
    s = banco_g.check_spec(_spec(tmp_path))
    tags = [banco_g.tag(r, c, b) for r, c, b in banco_g.schedule(s)]
    assert tags[:5] == ["banco_guasta_A_r0", "banco_guasta_C_r0", "banco_guasta_D_r0",
                        "banco_controllo_A_r0", "banco_guasta_A_r1"]
    assert len(tags) == 8 and len(set(tags)) == 8


def test_righe_printf_ed_esito():
    righe = banco_g.icp_lines(RIGHE)
    assert [r["good"] for r in righe] == [False, True] and righe[1]["q"] == 0.406
    e = banco_g.esito(righe)
    assert e["passa"] is True and e["q1"] == 0.0 and e["buone_prime_n"] == 1
    e = banco_g.esito([{"q": 0.0, "good": False}] * 10 + [{"q": 0.5, "good": True}])
    assert e["passa"] is False and e["buone"] == 1 and e["prime_n"] == 10
    e = banco_g.esito([])
    assert e["passa"] is False and e["q1"] is None


def test_shadow_state_per_rollout(tmp_path):
    s = banco_g.check_spec(_spec(tmp_path))
    c = s["casi"][0]
    st = banco_g.shadow_state(c, "/o/stati/guasta.molastate", "ab" * 32, 85, 116444736318000000)
    fp = tv.g_fingerprint(st)
    assert fp["state_path"] == "/o/stati/guasta.molastate" and fp["pose_loaded"] == c["pose6"]
    assert st["status"] == "ok" and st["source_path"] == c["state_path"]


# --- spec from real data (read only; skipped if the run is missing) ---------------

@pytest.mark.skipif(not RUN_G_S1001.is_dir(), reason="run G s1001 del pilota assente")
def test_spec_dai_dati_veri(tmp_path):
    out = tmp_path / "spec.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out", str(out),
                        "--caso", f"guasta_s1001_w4:{RUN_G_S1001}:4:A,C,D",
                        "--caso", f"controllo_s1001_w3:{RUN_G_S1001}:3:A"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    s = banco_g.load_spec(out)
    g, c = s["casi"]
    # sha256 of the states: w4 = bb7d38802409, w3 = cf45bd69db1c
    assert g["state_sha256"].startswith("bb7d38802409") and c["state_sha256"].startswith("cf45bd69db1c")
    assert g["state_stamp_tick"] == 116444736318000000
    assert [b["nome"] for b in g["bracci"]] == ["A", "C", "D"]
    assert g["saved_odom"] == [3.947, -0.278, -0.092] and g["attacker_target_scene"] == [-2.046, -1.278, -0.092]
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out", str(out),
                         "--caso", f"x:{RUN_G_S1001}:4:A"], capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode != 0 and "non si sovrascrivono" in r2.stdout + r2.stderr


# --- bench loop with fake objects: copy, sha256, outcomes, analysis ---------------

class _IsaacBanco:
    def __init__(self, caps=("set_state",)):
        self._caps, self.chiamate = list(caps), []

    def caps(self):
        return self._caps

    def pause(self):
        self.chiamate.append("pause")

    def play(self):
        self.chiamate.append("play")

    def set_pose(self, x, y, yaw):
        self.chiamate.append(("set_pose", x, y, yaw))
        self.posa = (x, y, yaw)
        return list(self.posa)

    def save(self):
        return list(self.posa)


class _NodeBanco:
    last_twin_fp = None

    def spin(self, t):
        pass

    def settled_pose(self, isaac):
        # odom = scene - run offset
        return (isaac.posa[0] + 5.99994, isaac.posa[1] + 1.00002, isaac.posa[2])


def test_ciclo_banco_finto(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    s["casi"][0]["saved_scene"] = [3.947 - 5.99994, -0.278 - 1.00002, -0.092]
    s["casi"][1]["saved_scene"] = [2.95 - 5.99994, -0.1887 - 1.00002, -0.0667]
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    out = tmp_path / "prova" / "campagna" / "straight" / "fixed_g3" / "rep_01"
    out.mkdir(parents=True)
    visti = []

    def _rollout(genome, isaac, mola, node, goal, horizon, tag, **kw):
        ts = ao.TWIN_STATE
        visti.append((tag, genome, ts["init"], ts["g_env_extra"], tuple(ts["init_target"]["target_scene"]),
                      kw["frame_start"]))
        log = tmp_path / f"mola_{tag}.log"
        good = "1" if ("controllo" in tag or tag.endswith(("C_r0", "C_r1"))) else "0"
        log.write_text(RIGHE.replace("isGood=1", f"isGood={good}"))
        mola.log_path = str(log)
        node.last_twin_fp = {"g_check": {"firme_ok": True, "loaded_is_shadow_file": True,
                                         "hash_ok": True, "clock_ok": True}}
        return None, (1.0, 0.0, 0.0), 0.0, "ok", 1.0, 10

    monkeypatch.setattr(ao, "rollout", _rollout)
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    args = types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni", horizon=1.0, settle_sec=2.0,
                                 catchup_sec=0.8, attack_from="spawn", stop_on="estimate",
                                 drift_reset_on="genome", trace=True)
    mola = types.SimpleNamespace()
    assert ao._g_banco_run(args, _IsaacBanco(), mola, _NodeBanco(), out, [(0.0, 0.0), (5.0, 0.0)]) == 0
    assert [v[0] for v in visti][:4] == ["banco_guasta_A_r0", "banco_guasta_C_r0", "banco_guasta_D_r0",
                                         "banco_controllo_A_r0"]
    assert all(v[1] is None for v in visti)                      # nominals only
    d = [v for v in visti if v[0] == "banco_guasta_D_r0"][0]
    assert d[3] == {"MOLA_FULL_STATE_PROVA_TWIST_ZERO": "1"} and d[2] == "attacker_est"
    c = [v for v in visti if v[0] == "banco_guasta_C_r0"][0]
    assert c[2] == "ombra_est" and c[5] == tuple(s["casi"][0]["believed"])
    assert (out / "stati" / "guasta.molastate").is_file()
    assert ao.TWIN_STATE["shadow_state"]["state_path"] == str(out / "stati" / "controllo.molastate")
    es = json.loads((out / "banco_G.json").read_text())["esiti"]
    assert len(es) == 8 and all(e["saved_err_cm"] < 0.1 for e in es)
    # reading: A and D reproduce, C passes 2/3, control passes (V3 without libraries: invalid)
    (tmp_path / "prova" / "librerie_mola.txt").write_text(
        "10:00:00 1 gemello_G /ovl/mola_lidar_odometry/lib/x86_64-linux-gnu/libmola_lidar_odometry.so\n"
        "10:00:00 1 gemello_G /ovl/mola_state_estimation_simple/lib/x86_64-linux-gnu/"
        "libmola_state_estimation_simple.so\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analisi_G_d_guasto.py"), str(tmp_path / "prova")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    esito = [x for x in r.stdout.splitlines() if x.startswith("ESITO")][0]
    assert "guasta: A riproduce 0/2, C passa 2/2, D riproduce 0/2" in esito


def test_ciclo_banco_sha_diverso(tmp_path, monkeypatch):
    s = _spec(tmp_path)
    s["casi"][0]["state_sha256"] = "0" * 64
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    monkeypatch.setattr(ao, "rollout", lambda *a, **k: pytest.fail("rollout con sha256 sbagliato"))
    args = types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni")
    assert ao._g_banco_run(args, _IsaacBanco(), None, _NodeBanco(), tmp_path, []) == 2
    assert ao._g_banco_run(args, _IsaacBanco(caps=()), None, _NodeBanco(), tmp_path, []) == 2
