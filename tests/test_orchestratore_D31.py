"""Orchestrator options: pareto_maxdamage selection without re-evaluations, winner trials
(--prove-vincitore), realignment after settling (--riallinea-dopo-assestamento, realigned bench
arm), attitude log (--registra-assetto), twin attitude (--twin-init-assetto), the twin that chose
and the comparisons (--confronto-vincitore). With the options off, arguments, Isaac messages,
_rollout_once calls and the orchestrator line of run_campaign.sh are unchanged.
Fake objects, no ROS node, Isaac or MOLA; window-loop segments are taken from the source and run
as they are, with --search fixed as in the winner replay. Tests on a recorded run are skipped if
the run is missing.
"""

import csv
import json
import math
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import banco_g  # noqa: E402
from src.optimization import twin_variants as tv  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402
from src.analysis.baseline_distance import load_baseline  # noqa: E402
from src.analysis.fronte_pareto import fronte_pareto  # noqa: E402

ORCH = ROOT / "src" / "optimization" / "attack_orchestrator.py"
SERVER = ROOT / "isaac" / "run_isaac_standalone.py"
DATI = ROOT / "tests" / "dati" / "biforcazione_31ott"
REP = (ROOT / "data/attack/serieB/nsga3_senzabudget/s1001_accettazione_30ott/campagna/straight/"
       "nsga3_g3_p32x16_lag0.5_rkeep_sel2_warm_stopramp_k1.0_reliable_serieB_s1001_pt1_opinsieme_sgross_"
       "coda_N5_dmin140/rep_01")
BASELINE = ROOT / "data/attack/baseline/rettilineo_direction_N5_media3semi_dmin140_sosta1800.csv"
serve_run = pytest.mark.skipif(not (REP / "traces" / "w4_e480.csv").is_file() or not BASELINE.is_file(),
                               reason="run della prova di accettazione assente")
NUOVE = ("twin_init_assetto", "registra_assetto", "prove_vincitore", "riallinea_dopo_assestamento",
         "confronto_vincitore")


# --- arguments ---

def _args_main(argv):
    """Arguments of main for one command line, without starting anything."""
    preso = {}

    class Stop(Exception):
        pass
    orig = ao.argparse.ArgumentParser.parse_args

    def parse(self, *a, **k):
        preso["a"] = orig(self, *a, **k)
        raise Stop
    ao.argparse.ArgumentParser.parse_args = parse
    vecchio = sys.argv
    sys.argv = ["attack_orchestrator.py", *argv]
    try:
        ao.main()
    except Stop:
        pass
    finally:
        ao.argparse.ArgumentParser.parse_args = orig
        sys.argv = vecchio
    return preso["a"]


FISSO = ["--search", "fixed", "--fixed-genome-seq", "h.json:4:480", "--select", "pareto_maxdamage",
         "--victim", "continuous", "--twin-state", "ombra_g", "--twin-init", "attacker_est", "--trace",
         "--damage-ref", "baseline", "--baseline-path", "b.csv"]


def test_argomenti_spenti_di_default():
    a = _args_main([])
    assert (a.twin_init_assetto, a.registra_assetto, a.prove_vincitore, a.riallinea_dopo_assestamento,
            a.confronto_vincitore) == ("azzerato", False, 0, False, False)
    assert ao.problema_opzioni_d31(a) is None
    assert ao.problema_capacita_d31(a, ["set_state"]) is None


def test_argomenti_con_search_fixed():
    a = _args_main(FISSO + ["--prove-vincitore", "3", "--riallinea-dopo-assestamento", "--registra-assetto",
                            "--twin-init-assetto", "attaccante", "--confronto-vincitore",
                            "--fresh-cloud-after-restore"])
    assert a.search == "fixed" and a.select == "pareto_maxdamage" and a.prove_vincitore == 3
    assert ao.problema_opzioni_d31(a) is None
    caps = ["set_state", "set_state_assetto", "assetto", "save_nome"]
    assert ao.problema_capacita_d31(a, caps) is None
    assert "save_nome" in ao.problema_capacita_d31(a, caps[:3])
    assert "set_state_assetto" in ao.problema_capacita_d31(a, ["set_state", "assetto", "save_nome"])
    assert "assetto" in ao.problema_capacita_d31(a, ["set_state", "set_state_assetto", "save_nome"])


@pytest.mark.parametrize("extra, msg", [
    (["--prove-vincitore", "-1"], ">= 0"),
    (["--twin-init", "true", "--twin-init-assetto", "conservato"], "twin-init attacker_est o ombra_est"),
    (["--twin-init", "ombra_est", "--twin-init-assetto", "attaccante"], "attaccante richiede --twin-init"),
    (["--riallinea-dopo-assestamento", "--victim", "restart"], "--victim continuous"),
    (["--riallinea-dopo-assestamento", "--twin-state", "ombra_e"], "ombra_g"),
    (["--confronto-vincitore", "--damage-ref", "window"], "--trace"),
])
def test_argomenti_rifiutati(extra, msg):
    assert msg in ao.problema_opzioni_d31(_args_main(FISSO + extra))


def test_banco_rifiuta_le_opzioni_del_flusso():
    a = types.SimpleNamespace(g_banco="s.json", prove_vincitore=2, riallinea_dopo_assestamento=False,
                              confronto_vincitore=False, twin_init_assetto="azzerato")
    assert "--g-banco" in ao.problema_opzioni_d31(a)


# --- Isaac client: messages unchanged without the new arguments ---

def _client(monkeypatch):
    c = ao.IsaacClient.__new__(ao.IsaacClient)       # no __init__: no /tmp file touched
    sent = []
    monkeypatch.setattr(c, "call", lambda cmd, **kw: sent.append((cmd, kw)) or {"pose": [1, 2, 3]}, raising=False)
    return c, sent


def test_client_messaggi(monkeypatch):
    c, sent = _client(monkeypatch)
    c.save()
    c.restore()
    c.set_state(1, 2, 0.1, 0.5, 0.0, 0.0, zero_joints=False)
    assert sent == [("save", {}), ("restore", {}),
                    ("set_state", {"x": 1.0, "y": 2.0, "yaw": 0.1, "vx": 0.5, "vy": 0.0, "wz": 0.0,
                                   "zero_joints": False})]
    sent.clear()
    c.save(nome="riallineo")
    c.restore(nome="riallineo")
    c.set_state(1, 2, 0.1, roll=0.001, pitch=-0.002)
    c.set_state(1, 2, 0.1, keep_tilt=True)
    c.assetto()
    assert sent[0] == ("save", {"nome": "riallineo"}) and sent[1] == ("restore", {"nome": "riallineo"})
    assert sent[2][1]["roll"] == 0.001 and sent[2][1]["pitch"] == -0.002 and "keep_tilt" not in sent[2][1]
    assert sent[3][1]["keep_tilt"] is True and "roll" not in sent[3][1]
    assert sent[4] == ("assetto", {})


# --- Isaac server: named snapshots ---

def _server_handle():
    """Server _handle with a fake articulation (as in test_assetto_isaac)."""
    import ast
    albero = ast.parse(SERVER.read_text())
    nomi = {"_handle", "_snapshot", "_restore", "_pose_of", "_yaw"}
    corpo = [n for n in albero.body
             if (isinstance(n, ast.FunctionDef) and n.name in nomi)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_CHIAVI_ASSETTO", "_CAPS_ASSETTO",
                                                                             "_saved", "_saved_nomi")
                                                   for t in n.targets))]

    class Art:
        def __init__(self):
            self.pos, self.q = np.array([1.0, 2.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])
            self.scritture = []

        def get_world_pose(self):
            return self.pos.copy(), self.q.copy()

        def set_world_pose(self, position, orientation):
            self.scritture.append("pose")
            self.pos, self.q = np.array(position), np.array(orientation)

        def get_linear_velocity(self):
            return np.zeros(3)

        def get_angular_velocity(self):
            return np.zeros(3)

        def get_joint_positions(self):
            return np.zeros(7)

        def get_joint_velocities(self):
            return np.zeros(7)

        def set_joint_positions(self, v):
            self.scritture.append("jpos")

        def set_joint_velocities(self, v):
            self.scritture.append("jvel")

        def set_linear_velocity(self, v):
            self.scritture.append("lin")

        def set_angular_velocity(self, v):
            self.scritture.append("ang")
    art = Art()
    g = {"np": np, "_get_art": lambda: art, "_timeline": types.SimpleNamespace(is_playing=lambda: False)}
    exec(compile(ast.Module(body=corpo, type_ignores=[]), str(SERVER), "exec"), g)
    return g, art


def test_server_istantanee_con_nome():
    g, art = _server_handle()
    h = g["_handle"]
    assert h({"cmd": "save"})[0]["pose"] == [1.0, 2.0, 0.0]
    art.pos = np.array([5.0, 6.0, 0.3])
    assert h({"cmd": "save", "nome": "riallineo"})[0]["pose"] == [5.0, 6.0, 0.0]
    art.pos = np.array([9.0, 9.0, 0.3])
    assert h({"cmd": "restore", "nome": "riallineo"}) == (None, 1)
    assert list(art.pos) == [5.0, 6.0, 0.3]
    assert h({"cmd": "restore"}) == (None, 1) and list(art.pos) == [1.0, 2.0, 0.3]   # slot kept
    r, _ = h({"cmd": "restore", "nome": "altro"})
    assert r["ok"] is False and "altro" in r["error"]
    caps = h({"cmd": "caps"})[0]["caps"]
    assert caps[0] == "set_state" and {"set_state_assetto", "assetto", "save_nome"} <= set(caps)


# --- _rollout_once with fake objects ---

class _Isaac:
    def __init__(self):
        self.chiamate = []
        self.playing = False

    def pause(self):
        self.chiamate.append(("pause",))

    def play(self):
        self.chiamate.append(("play",))

    def restore(self, nome=None):
        self.chiamate.append(("restore",) + ((nome,) if nome else ()))
        return [0.0, 0.0, 0.0]

    def save(self, nome=None):
        self.chiamate.append(("save",) + ((nome,) if nome else ()))
        return [-2.05, -1.28, -0.09]

    def set_state(self, x, y, yaw, vx=0.0, vy=0.0, wz=0.0, zero_joints=True, **kw):
        self.chiamate.append(("set_state", x, y, yaw, vx, vy, wz, zero_joints, tuple(sorted(kw.items()))))
        return [x, y, yaw]

    def call(self, cmd, **kw):
        self.chiamate.append((cmd,))
        if cmd == "assetto":
            return {"ok": True, "pos": [0, 0, 0.3], "quat_wxyz": [1, 0, 0, 0], "roll": 1e-3, "pitch": -2e-3,
                    "yaw": 0.0, "lin": [0, 0, 0], "ang": [0, 0, 0], "playing": False}
        raise AssertionError(cmd)


class _Mola:
    log_path = None
    args = ["/carter/lidar_perturbed"]

    def stop(self):
        pass

    def spawn(self, tag):
        self.tag = tag

    def wait_ready(self, check=None, marker=None, fail_markers=()):
        return True


class _Node:
    fresh_cloud = False
    true_source = "ros"

    def __init__(self):
        self.est_t = 0.0
        self.est_raw = None
        self.frames_out = 0
        self.genomi = []

    def set_genome(self, g):
        self.genomi.append(list(g))

    def set_attack_enabled(self, on):
        return True

    def spin(self, s=0.0):
        self.est_t = time.time()
        self.est_raw = (0.1, 0.0, 0.0)
        self.est_stamp_raw = 38.7

    drain = spin

    def count_mola_publishers(self):
        return 1

    def settled_pose(self, isaac):
        return (3.95, -0.28, -0.09)

    def reset_drift(self):
        pass


def _rollout_finto(monkeypatch, tgt, registra=False, **kw):
    """Twin _rollout_once up to driving (replaced): calls to Isaac."""
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "clean", "init": "attacker_est", "init_target": tgt})
    monkeypatch.setitem(ao.ASSETTO, "registra", registra)
    monkeypatch.setattr(ao, "_replay_with_guard", lambda node, mola, tag: None)
    monkeypatch.setattr(ao, "_twin_alignment_record", lambda node, ts, fp: {})
    monkeypatch.setattr(ao, "topic_sub_gids", lambda node, topic: None)
    guida = {}

    def _drive(node, isaac, mola, local_wps, horizon, true_start, *a, **k):
        guida.update(true_start=true_start, frame_pose=k.get("frame_pose"), settle=node.last_settle_waited)
        return true_start, true_start, 1.0, "ok", 1.0, 10
    monkeypatch.setattr(ao, "_drive_and_measure", _drive)
    isaac, node = _Isaac(), _Node()
    res = ao._rollout_once([0.0] * 17, isaac, _Mola(), node, [(1.0, 0.0)], 1.0, "w4_e0", settle_sec=0.0, **kw)
    return res, isaac, node, guida


TGT = {"target_scene": [-2.046, -1.278, -0.092], "v_set_scene": [0.0, 0.0], "w_set": 0.0, "zero_joints": True}


def test_rollout_spento_invariato(monkeypatch):
    """Without options: restore, set_state with the usual 7 arguments, play; no attitude command."""
    res, isaac, node, guida = _rollout_finto(monkeypatch, dict(TGT))
    assert res[3] == "ok"
    assert isaac.chiamate == [("pause",), ("restore",),
                              ("set_state", -2.046, -1.278, -0.092, 0.0, 0.0, 0.0, True, ()), ("play",)]
    assert not hasattr(node, "last_assetto")


def test_registra_assetto_search_fixed(monkeypatch, tmp_path):
    """--registra-assetto, --search fixed: readings after restore, set_state and settling end."""
    res, isaac, node, _ = _rollout_finto(monkeypatch, dict(TGT), registra=True)
    assert [c[0] for c in isaac.chiamate] == ["pause", "restore", "assetto", "set_state", "assetto", "play",
                                              "assetto"]
    assert [x["fase"] for x in node.last_assetto] == ["dopo_restore", "dopo_set_state", "fine_assestamento"]
    assert node.last_assetto[0]["roll"] == 1e-3 and node.last_assetto[0]["pitch"] == -2e-3
    assert ao._twin_start_fields(node)["assetto"] is node.last_assetto
    monkeypatch.setitem(ao.ASSETTO, "registra", False)
    assert "assetto" not in ao._twin_start_fields(node)


def test_twin_init_assetto_nel_set_state_search_fixed(monkeypatch):
    for kw_a in ({"roll": 0.001, "pitch": -0.002}, {"keep_tilt": True}):
        _, isaac, _, _ = _rollout_finto(monkeypatch, {**TGT, "assetto_set_state": kw_a})
        ss = [c for c in isaac.chiamate if c[0] == "set_state"]
        assert len(ss) == 1 and dict(ss[0][8]) == kw_a


def test_assestamento_zero_e_aggancio(monkeypatch):
    """Realigned run: driving from the first pose (no wait), hook-up after settling."""
    visti = []
    res, isaac, node, guida = _rollout_finto(monkeypatch, dict(TGT), assestamento_nuvole=0,
                                             dopo_assestamento=lambda i, n, info: visti.append(info["ruolo"]))
    assert guida["settle"] == 0.0 and visti == ["gemello"] and res[3] == "ok"


# --- twin attitude from the attacker MOLA ---

def _carica_assetto_isaac():
    p = ROOT / "isaac" / "assetto.py"
    m = types.ModuleType("assetto_isaac")
    exec(compile(p.read_text(), str(p), "exec"), m.__dict__)
    return m


def test_assetto_attaccante_composto():
    att = (4.0, 0.5, 0.01, math.radians(9.206027), math.radians(-0.166056), math.radians(0.028472))
    node = types.SimpleNamespace(attacker_est6=att, attacker_est_raw=(att[0], att[1], att[3]),
                                 attacker_est_stamp=38.6)
    spawn = {"roll": 1.2e-3, "pitch": -0.8e-3, "yaw": 0.05}
    kw, rec = ao.assetto_gemello("attaccante", node, spawn)
    r, p, y = banco_g.assetto_composto((1.2e-3, -0.8e-3, 0.05), (att[5], att[4], att[3]))
    assert kw == {"roll": r, "pitch": p} and rec["composto"]["yaw"] == y
    assert rec["attaccante"]["stesso_messaggio_della_posa"] is True and rec["attaccante"]["stamp"] == 38.6
    # composition with the server quaternions: R = R_spawn R_att
    a = _carica_assetto_isaac()
    q = a.prodotto(a.quat_da_euler(*[spawn[k] for k in ("roll", "pitch", "yaw")]),
                   a.quat_da_euler(att[5], att[4], att[3]))
    rq, pq, _ = a.euler_da_quat(q)
    assert r == pytest.approx(rq, abs=1e-12) and p == pytest.approx(pq, abs=1e-12)
    # kept and zeroed
    assert ao.assetto_gemello("azzerato", node) == ({}, None)
    assert ao.assetto_gemello("conservato", node)[0] == {"keep_tilt": True}
    # without spawn or without the attacker pose: configuration error
    with pytest.raises(ao.ErroreAssetto, match="spawn"):
        ao.assetto_gemello("attaccante", node, {})
    with pytest.raises(ao.ErroreAssetto, match="nessuna posa"):
        ao.assetto_gemello("attaccante", types.SimpleNamespace(attacker_est6=None), spawn)
    big = types.SimpleNamespace(attacker_est6=(0, 0, 0, 0, 0.2, 0.0), attacker_est_raw=(0, 0, 0))
    with pytest.raises(ao.ErroreAssetto, match="oltre"):
        ao.assetto_gemello("attaccante", big, {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})


def test_posa_completa_dell_attaccante():
    """The attacker MOLA callback keeps x, y, yaw and the 6D pose from the same message."""
    q = _carica_assetto_isaac().quat_da_euler(0.01, -0.02, 0.3)   # [w, x, y, z]
    msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=38, nanosec=600000000)),
        pose=types.SimpleNamespace(pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=1.0, y=2.0, z=0.1),
            orientation=types.SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3]))))
    node = types.SimpleNamespace(_n_cb=0, attacker_buf=[], _pose_of=ao.AttackNode._pose_of)
    ao.AttackNode._attacker_est_cb(node, msg)
    assert node.attacker_est_raw[:2] == (1.0, 2.0) and node.attacker_est_raw[2] == pytest.approx(0.3)
    x, y, z, yaw, pitch, roll = node.attacker_est6
    assert (roll, pitch, yaw) == pytest.approx((0.01, -0.02, 0.3), abs=1e-12)
    assert node.attacker_est6[3] == node.attacker_est_raw[2] and node.attacker_est_stamp == pytest.approx(38.6)


def test_segmento_inizio_finestra_twin_init_assetto():
    """Window-start segment: with --twin-init-assetto the target carries parameters and source."""
    src = ORCH.read_text().splitlines()
    i0 = next(i for i, x in enumerate(src) if x.strip().startswith('if args.twin_init_assetto != "azzerato" and '
                                                                    'TWIN_STATE.get("init_target")'))
    i1 = next(i for i in range(i0 + 1, len(src)) if src[i].strip().startswith("if args.replay_gate > 0"))
    corpo = textwrap.dedent("\n".join(src[i0:i1]))
    att = (4.0, 0.5, 0.0, 0.16, 0.001, -0.002)
    node = types.SimpleNamespace(attacker_est6=att, attacker_est_raw=(4.0, 0.5, 0.16), attacker_est_stamp=1.0)
    for modo, attesi in (("azzerato", None), ("attaccante", ("roll", "pitch")), ("conservato", ("keep_tilt",))):
        ts = {"init_target": dict(TGT)}
        g = dict(ao.__dict__, TWIN_STATE=ts, node=node,
                 args=types.SimpleNamespace(twin_init_assetto=modo, search="fixed"))
        ao.ASSETTO["spawn"], vecchio = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, ao.ASSETTO["spawn"]
        try:
            exec(corpo, g)
        finally:
            ao.ASSETTO["spawn"] = vecchio
        if attesi is None:
            assert ts["init_target"] == TGT
        else:
            assert tuple(ts["init_target"]["assetto_set_state"]) == attesi
            assert ts["init_target"]["assetto"]["modo"] == modo


# --- pareto_maxdamage selection ---

def test_d31a_lento_non_escluso_e_nessuna_rivalutazione():
    ev = [{"genome": [0.1] * 17, "status": "ok", "counted": True, "admissible": True, "damage": 0.40,
           "chamfer": 30.0, "tau_s": 0.7, "slow": True},
          {"genome": [0.2] * 17, "status": "ok", "counted": True, "admissible": True, "damage": 0.30,
           "chamfer": 10.0, "tau_s": 0.3, "slow": False}]
    fr, scelto = fronte_pareto(ev, budget=None, solo_validi=True)
    assert scelto["idx"] == 0 and scelto["slow"] is True        # no filter on slow candidates
    # in the loop, the sel re-evaluations are only in the maxdamage branch
    src = ORCH.read_text()
    i_p = src.index('elif args.select == "pareto_maxdamage":')
    i_e = src.index("top = top_maxdamage(eval_log, args)")
    assert i_p < i_e and src.index('f"w{k}_sel{j}"') > i_e
    assert src[i_p:i_e].count("rollout(") == 0
    # the "slow" filter of --select-only-valid applies only with maxdamage
    assert 'lenti_fuori = (args.select == "maxdamage"' in src


# --- winner trials ---

def test_prove_del_vincitore_search_fixed(tmp_path):
    """Fixed arm: the winner is the only candidate; N trials run and keep the true damage."""
    log = tmp_path / "mola_w4_prova.log"
    log.write_text((DATI / "mola_w4_sel0_38.7-39.6.log").read_text())
    mola = types.SimpleNamespace(log_path=str(log))
    node = types.SimpleNamespace(last_tau=float("nan"), last_twin_fp={"log": str(log)})
    taus = {"w4_prova0": 0.3, "w4_prova1": 0.7, "w4_prova2": 0.3}
    stati = {"w4_prova0": "ok", "w4_prova1": "ok", "w4_prova2": "timeout(45s)"}
    tags = []

    def esegui(tag):
        tags.append(tag)
        node.last_tau = taus[tag]
        return (0, 0, 0), (4.9, 0.4, 0.0), 12.0, stati[tag], 0.95, 10

    def misura(res):
        return {"damage": 0.42, "dev_mean": 0.41}, {"mean_cm": 41.0}
    rec = ao.prove_del_vincitore(3, 4, esegui, misura, node, mola, danno_ricerca_cm=73.033)
    assert tags == ["w4_prova0", "w4_prova1", "w4_prova2"] and rec["n"] == 3
    p0, p1, p2 = rec["prove"]
    assert p1["lento"] is True and p1["danno_cm"] == pytest.approx(42.0) and p1["tau_s"] == 0.7
    assert p2["valido"] is False and p2["danno_cm"] == pytest.approx(42.0) and p2["status"] == "timeout(45s)"
    assert p0["ramo"]["classe"] == "ramo_gemello" and p0["ramo"]["dy_cm"] == pytest.approx(-19.2292, abs=1e-3)
    assert rec["rami"] == {"ramo_gemello": 3}
    q = p0["icp_prime"]
    assert q["n"] >= 1 and q["q_min"] <= q["q_media"] and q["fonte"] in ("printf", "icp_kind0")
    assert rec["danno_ricerca_cm"] == 73.033
    # rollout without a final pose: no damage, no 0.0
    rec2 = ao.prove_del_vincitore(1, 4, lambda t: (None, None, float("nan"), "mola_crash", 0.0, 0),
                                  misura, node, mola)
    assert rec2["prove"][0]["danno_cm"] is None and rec2["prove"][0]["chamfer"] is None


def test_restore_kind_dei_tag_nuovi():
    assert ao.restore_kind("w4_prova0", False) == "prova_vincitore"
    assert ao.restore_kind("w4_riallineato", False) == "riallineato"
    assert ao.restore_kind("w4_e0", False) == "valutazione" and ao.restore_kind("w4_sel1", False) == "rivalutazione"
    node = types.SimpleNamespace(restore_log=[])
    ao.count_restore(node, "valutazione", 4)
    assert list(ao.restore_counts(node, 4)["per_tipo"]) == list(ao.RESTORE_KINDS)   # no unused type


# --- loop segment: trials, realigned twin, chosen twin and comparisons with --search fixed ---

def _segmento_applicazione():
    src = ORCH.read_text().splitlines()
    i0 = next(i for i, x in enumerate(src) if x.strip() == "# --prove-vincitore N: N identical twin rollouts with the winner genome from the same")
    i1 = next(i for i in range(i0, len(src)) if src[i].strip().startswith("# Waypoints passed by the applied rollout"))
    return textwrap.dedent("\n".join(src[i0:i1]))


class _IsaacR(_Isaac):
    def __init__(self):
        super().__init__()
        self.playing = True


class _NodeR:
    """Victim-flow node with the fields that the realigned run saves and restores."""
    fresh_cloud = False
    true_source = "ros"

    def __init__(self):
        self.est_source = "victim"
        self.est_raw, self.est_t, self.est_stamp_raw = (3.68, -0.70, 0.16), 1.0, 38.6
        self.est, self.est_origin = (0.0, 0.0, 0.0), (3.68, -0.70, 0.16)
        self.chamfer, self.perc_samples, self.proc_ms_roll = [46.7], [{"a": 1}], [80.0]
        self.flusso = []
        self.attacker_buf = []
        self.attacker_est6 = None
        self.last_victim_open_stamp = 38.7
        self.st_stato_generatore = None
        self.last_trace = []
        self.last_tau = 0.3

    def true_pose(self, isaac, max_age=0.5):
        return (3.947, -0.278, -0.092)

    def set_victim_stream(self, on):
        self.flusso.append(on)

    def stato_generatore(self, azione, nome, flusso=None, timeout=3.0):
        return None

    def spin(self, s=0.0):
        if self.est_source == "victim":
            self.est_t = time.time()


class _Ombra:
    def __init__(self, tmp_path):
        self.tmp = tmp_path

    def save_state(self, node, path):
        return {"status": "ok", "state_path": str(path) + ".molastate", "state_stamp_tick": 1, "pose6": [0] * 6,
                "pose_stamp": 38.6, "believed": [4.085, 0.519, -0.119], "origin": [0.0177, 0.0, 0.0]}


@serve_run
@pytest.mark.parametrize("riallinea", [False, True])
def test_segmento_applicazione_search_fixed(tmp_path, monkeypatch, riallinea):
    """--search fixed: winner w4_e0, trials, realignment, chosen twin and comparisons."""
    out = tmp_path / "rep_01"
    (out / "traces").mkdir(parents=True)
    (out / "maps").mkdir()
    # chosen twin: real trace and log of e480 (w4); victim: real trace and log of the run
    (out / "traces" / "w4_e00.csv").write_text((REP / "traces" / "w4_e480.csv").read_text())
    log_g = tmp_path / "mola_w4_e0.log"
    log_g.write_text((DATI / "mola_w4_sel0_38.7-39.6.log").read_text())
    log_v = tmp_path / "mola_victim.log"
    log_v.write_text((DATI / "mola_victim_38.6_38.7-39.6.log").read_text())
    righe_v = list(csv.DictReader(open(REP / "traces" / "w4_applied.csv")))
    righe_r = list(csv.DictReader(open(REP / "traces" / "w4_e480.csv")))
    poly = load_baseline(str(BASELINE))
    g = [0.5] * 17
    eval_log = [{"genome": g, "damage": 0.73033, "status": "ok", "twin_start": {"log": str(log_g)},
                 "danno_metro": {"mean_cm": 73.033}}]
    isaac, node = _IsaacR(), _NodeR()
    mola = types.SimpleNamespace(log_path=str(log_g))
    victim = types.SimpleNamespace(log_path=str(log_v))
    attacker = types.SimpleNamespace(run_origin=(0.0177, 0.0, 0.0),
                                     believed_pose=lambda n: (4.0855, 0.519, -0.1189))
    tags = []

    def _rollout(genome, isa, mola_, nd, goal, horizon, tag, **kw):
        tags.append((tag, sorted(k for k in ("dopo_assestamento", "assestamento_nuvole", "victim") if k in kw)))
        if tag.endswith("_applied"):
            hook = kw.get("dopo_assestamento")
            if hook is not None:
                hook(isa, nd, {"tag": tag, "ruolo": "vittima", "victim": kw["victim"]})
                assert nd.est_source == "victim" and nd.chamfer == [46.7] and nd.proc_ms_roll == [80.0]
            nd.last_trace = righe_v
            return (0, 0, 0), (4.9, 0.37, 0.0), 40.0, "ok", 0.95, 12
        if tag.endswith("_riallineato"):
            assert kw["assestamento_nuvole"] == 0 and kw["frame_start"] == (4.085, 0.519, -0.119)
            ts = ao.TWIN_STATE
            assert ts["mode"] == "ombra_g" and ts["shadow_state"]["pose_stamp"] == 38.6
            assert ts["init_target"]["target_scene"] is not None
            isa.pause()
            isa.restore()                           # _IsaacPosto: the named slot
            nd.est_source = "twin"
            nd.chamfer = []
            mola_.log_path = str(log_g)
            nd.last_trace = righe_r
            return (0, 0, 0), (4.9, 0.4, 0.0), 41.0, "ok", 0.95, 11
        assert tag.startswith("w4_prova")
        nd.last_trace = righe_r
        mola_.log_path = str(log_g)
        return (0, 0, 0), (4.9, 0.4, 0.0), 41.0, "ok", 0.95, 11

    args = types.SimpleNamespace(prove_vincitore=2, riallinea_dopo_assestamento=riallinea,
                                 confronto_vincitore=True, search="fixed", select="pareto_maxdamage",
                                 warmup=0.0, settle_sec=2.0, catchup_sec=0.8, attack_from="spawn",
                                 stop_on="estimate", untracked="discard", drift_reset_on="keep", trace=True,
                                 damage="mean", dir_weight=0.0, twin_init_vel="zero", horizon=1.0,
                                 chiudi_attacco_scoperto=True, frame_log_nodo=None)
    ts0 = {"mode": "ombra_g", "init": "attacker_est", "init_target": {"x": 1}, "shadow_state": {"s": 0}}
    monkeypatch.setattr(ao, "TWIN_STATE", dict(ts0))
    monkeypatch.setitem(ao.ASSETTO, "registra", False)
    loc = dict(ao.__dict__, rollout=_rollout, args=args, best_g=np.array(g), best_F=np.array([-0.73033, 40.0]),
               k=4, isaac=isaac, mola=mola, node=node, wps=[(0.0, 0.0)], nominal_ok=True,
               nom_ratio=float("nan"), nom_settle_s=None, frame_start=(4.085, 0.519, -0.119), out=out,
               nom_end=None, goal=(5.0, 0.0), nom_path=[], baseline_poly=poly, victim=victim,
               shadow=_Ombra(tmp_path), attacker=attacker, eval_log=eval_log,
               sel_rec={"chosen_idx": 0})
    exec(_segmento_applicazione(), loc)
    d = loc["d31_rec"]
    # the realigned twin runs inside the victim rollout, at the end of settling
    attesi = ["w4_prova0", "w4_prova1", "w4_applied"] + (["w4_riallineato"] if riallinea else [])
    assert [t for t, _ in tags] == attesi
    assert tags[2][1] == (["dopo_assestamento", "victim"] if riallinea else ["victim"])
    assert [p["tag"] for p in d["prove_vincitore"]["prove"]] == ["w4_prova0", "w4_prova1"]
    sc = d["gemello_scelto"]
    assert sc["tag"] == "w4_e0" and sc["traccia"].endswith("traces/w4_e00.csv") and sc["mola_log"] == str(log_g)
    a = d["confronti_d31"]["a"]
    assert (a["R1_cm"], a["laterale_cm"], a["danno_cm"]) == (1.66, 7.405, 4.574)   # per-metre table
    assert a["ramo_vittima"] == "ramo_vittima" and a["ramo_gemello"] == "ramo_gemello" and not a["segue"]
    # full criterion: without a per-frame log the incremental E1 cannot be measured
    assert a["criterio_completo"] is True and a["dm_real"] is True and a["E1_inc_cm"] is None
    assert "E1_inc_cm" in a["non_misurati"] and "registro per frame" in a["e1"]["motivo"]
    if not riallinea:
        assert "riallineato" not in d and "b" not in d["confronti_d31"] and loc["d31_errore"] is None
        assert isaac.chiamate == [] and node.flusso == []
        return
    r = d["riallineato"]
    assert r["generatore"] == "nodo_senza_stato_generatore" and r["gemello"]["status"] == "ok"
    # realignment without the generator state: configuration error at window end
    assert "senza lo stato del generatore" in loc["d31_errore"]
    assert r["gemello"]["danno_metro"]["mean_cm"] == 73.033 and r["gemello"]["traccia"].endswith("w4_riallineato.csv")
    # sequence: pause, named snapshot, twin from the slot, resume from the slot, play
    assert isaac.chiamate == [("pause",), ("save", "riallineo"), ("pause",), ("restore", "riallineo"),
                              ("pause",), ("restore", "riallineo"), ("play",)]
    assert node.flusso == [False, True] and r["ripresa"]["attesa"] == "senza_filtro"
    assert set(r["conteggi"]) == {"pausa", "ripresa", "dopo_attesa", "fine"}
    assert ao.TWIN_STATE == ts0                                   # window twin state restored
    b = d["confronti_d31"]["b"]
    assert b["gemello"] == "riallineato" and d["confronti_d31"]["stamp_ripresa"] == 38.6
    assert b["ramo_vittima"] == "ramo_vittima" and b["R1_cm"] == 1.66


def test_segmento_applicazione_spento_vuoto(tmp_path):
    """Without options the segment adds no rollout and no history field."""
    tags = []

    def _rollout(genome, isa, mola_, nd, goal, horizon, tag, **kw):
        tags.append((tag, sorted(kw)))
        nd.last_trace = []
        return (0, 0, 0), (1.0, 0.0, 0.0), 1.0, "ok", 1.0, 10
    args = types.SimpleNamespace(prove_vincitore=0, riallinea_dopo_assestamento=False, confronto_vincitore=False,
                                 warmup=0.0, settle_sec=2.0, catchup_sec=0.8, attack_from="spawn",
                                 stop_on="estimate", untracked="discard", drift_reset_on="keep", trace=True,
                                 horizon=1.0)
    loc = dict(ao.__dict__, rollout=_rollout, args=args, best_g=np.zeros(17), k=4, isaac=None, mola=None,
               node=types.SimpleNamespace(), wps=[], nominal_ok=True, nom_ratio=1.0, nom_settle_s=None,
               out=tmp_path, victim=None)
    exec(_segmento_applicazione(), loc)
    assert loc["d31_rec"] == {} and len(tags) == 1 and tags[0][0] == "w4_applied"
    assert "dopo_assestamento" not in tags[0][1]
    assert tags[0][1] == sorted(["warmup", "settle_sec", "catchup_sec", "attack_from", "stop_on", "untracked",
                                 "nominal_ok", "track_ref", "settle_fixed", "drift_reset_on", "victim",
                                 "trace_path"])


def test_nodo_rimesso_dopo_il_riallineato():
    node = types.SimpleNamespace(est=(1, 2, 3), chamfer=[1.0], est_source="victim")
    foto = ao.fotografa_nodo(node, ("est", "chamfer", "est_source", "last_assetto"))
    node.chamfer.append(2.0)
    node.est, node.est_source, node.last_assetto = None, "twin", [{"fase": "x"}]
    ao.rimetti_nodo(node, foto)
    assert node.est == (1, 2, 3) and node.chamfer == [1.0] and node.est_source == "victim"
    assert not hasattr(node, "last_assetto")


# --- bench: realigned arm ---

tg = pytest.importorskip("test_banco_G_d")


def _spec_trio(tmp_path, **gr):
    s = tg._spec(tmp_path)
    g = [0.5] * 17
    s["casi"] = s["casi"][:1]
    s["casi"][0]["bracci"] = [{"nome": "G", "init": "attacker_est", "genome": g},
                              {"nome": "V", "init": "ripristino", "genome": g},
                              {"nome": "Gr", "init": "riallineato", "riallinea_da": "V", "genome": g, **gr}]
    s["casi"][0]["confronti"] = [{"nome": "a", "vittima": "V", "gemello": "G"},
                                 {"nome": "b", "vittima": "V", "gemello": "Gr"}]
    s["ripetizioni"] = 1
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s))
    return p


def test_banco_specifica_riallineato(tmp_path):
    p = _spec_trio(tmp_path)
    spec = banco_g.load_spec(p)
    gr = spec["casi"][0]["bracci"][2]
    assert banco_g.riallinea_posa(gr) == "attacker_est"
    assert banco_g.init_target(spec["casi"][0], gr)["target_scene"] == spec["casi"][0]["attacker_target_scene"]
    assert banco_g.init_target(spec["casi"][0], {**gr, "riallinea_posa": "ripristino"})["target_scene"] is None
    assert banco_g.bracci_riallineati(spec["casi"][0]) == {"V"}
    # enabled: needs the settling length in clouds and the named slot in the server
    base = dict(damage_ref="baseline", baseline_path="b.csv")
    assert banco_g.problema_estensioni(spec, tg._args(settle_clouds=20, **base)) is None
    assert "D-31c" in banco_g.problema_estensioni(spec, tg._args(**base))
    assert "save_nome" in banco_g.problema_capacita(spec, ["set_state"])
    assert banco_g.problema_capacita(spec, ["set_state", "save_nome"]) is None
    # attitude on the realigned arm only with the attacker pose
    banco_g.load_spec(_spec_trio(tmp_path, assetto={"modo": "conservato"}))
    with pytest.raises(ValueError, match="assetto solo"):
        banco_g.load_spec(_spec_trio(tmp_path, riallinea_posa="ripristino", assetto={"modo": "conservato"}))
    with pytest.raises(ValueError, match="riallinea_posa"):
        banco_g.load_spec(_spec_trio(tmp_path, riallinea_posa="vera"))


class _IsaacTrio(tg._IsaacBanco):
    def __init__(self):
        super().__init__(caps=("set_state", "save_nome"))
        self.comandi = []

    def save(self, nome=None):
        self.comandi.append(("save", nome))
        return list(self.posa)

    def restore(self, nome=None):
        self.comandi.append(("restore", nome))
        return list(self.posa)

    def call(self, cmd, **kw):
        self.comandi.append((cmd, kw.get("nome")))
        if cmd == "set_state":
            return {"ok": True, "pose": [kw["x"], kw["y"], kw["yaw"]]}
        raise AssertionError(cmd)


class _NodeTrio(tg._NodeBanco):
    def __init__(self):
        self.st_stato_generatore = None
        self.est_t, self.est_raw, self.est_quat, self.est_z = 0.0, (4.067, 0.5193, -0.1189), (0, 0, 0, 1), -0.04
        self.est_stamp_raw = 38.6
        self.last_trace = []

    def true_pose(self, isaac, max_age=0.5):
        return self.settled_pose(isaac)

    def stato_generatore(self, azione, nome, flusso=None, timeout=3.0):
        return None

    def twin_map_save(self, path):
        tg._state(str(path) + ".molastate", tick=116444736386000000)
        return True, ""


def test_banco_ciclo_trio_finto(tmp_path, monkeypatch):
    """G, V (saves at settling end), Gr (restarts from the named slot with V state, no settling)."""
    p = _spec_trio(tmp_path)
    out = tmp_path / "rep_01"
    out.mkdir()
    isaac, node = _IsaacTrio(), _NodeTrio()
    visti = []

    def _rollout(genome, isa, mola, nd, goal, horizon, tag, **kw):
        isa.pause()
        isa.restore()
        ts = ao.TWIN_STATE
        if ts["init"] == "attacker_est":
            x, y, yaw = ts["init_target"]["target_scene"]
            isa.set_state(x, y, yaw, 0.0, 0.0, 0.0, zero_joints=True)
        visti.append((tag, ts["init"], kw.get("assestamento_nuvole"), "dopo_assestamento" in kw,
                      kw["frame_start"], ts["shadow_state"]["state_path"]))
        if "dopo_assestamento" in kw:
            kw["dopo_assestamento"](isa, nd, {"tag": tag, "ruolo": "gemello", "mola": mola})
        lp = tmp_path / f"{tag}.log"
        lp.write_text("")
        mola.log_path = str(lp)
        return None, (1.0, 0.0, 0.0), 0.0, "ok", 1.0, 40

    monkeypatch.setattr(ao, "rollout", _rollout)
    monkeypatch.setattr(ao, "TWIN_STATE", {})
    monkeypatch.setattr(ao, "SHADOW_QUIET_S", 0.0)
    args = types.SimpleNamespace(g_banco=str(p), g_overlay="/ovl", g_resave="ogni", horizon=1.0, settle_sec=2.0,
                                 catchup_sec=0.8, attack_from="spawn", stop_on="estimate", drift_reset_on="keep",
                                 trace=False, start_pose="-5.99,-1.0,0", damage_ref=None, baseline_path=None,
                                 settle_clouds=20)
    assert ao._g_banco_run(args, isaac, types.SimpleNamespace(alive=lambda: True), node, out, [(0.0, 0.0)]) == 0
    (tg_, ig, ag, hg, fg, sg), (tv_, iv, av, hv, fv, sv), (tr, ir, ar, hr, fr, sr) = visti
    assert (ig, iv, ir) == ("attacker_est", "true", "attacker_est") and (hg, hv, hr) == (False, True, False)
    assert ar == 0 and ag is None and av is None
    assert sr.endswith("banco_guasta_V_r0_riallineo.molastate") and sg == sv != sr
    assert fr == tuple(ao.AttackNode.compose((0.0177, 0.0, 0.0), node.est_raw))
    # V saves to the named slot; Gr restores it; no set_pose before Gr
    assert ("save", "riallineo") in isaac.comandi and ("restore", "riallineo") in isaac.comandi
    i_v = isaac.comandi.index(("save", "riallineo"))
    assert isaac.comandi[i_v + 1:].count(("restore", "riallineo")) == 1
    d = json.loads((out / "banco_G.json").read_text())
    es = {e["braccio"]: e for e in d["esiti"]}
    assert es["V"]["riallineo"]["stato"]["status"] == "ok" and es["Gr"]["riallinea_da"] == "V"
    assert es["Gr"]["generatore"] == "nodo_senza_stato_generatore"
    assert es["Gr"]["stato_caricato"]["state_stamp_tick"] == 116444736386000000


# --- launchers: variables off, line unchanged ---

def _blocco_d31(env):
    """Options block of run_campaign.sh run alone: resulting ORCH_EXTRA."""
    src = (ROOT / "scripts" / "run_campaign.sh").read_text().splitlines()
    i0 = next(i for i, x in enumerate(src) if x.startswith("# Options off by default: with default values the orchestrator"))
    i1 = next(i for i in range(i0, len(src)) if src[i].startswith('[[ -n "$D31_FLAGS" ]]'))
    codice = "set -euo pipefail\nORCH_EXTRA=\"${ORCH_EXTRA:-}\"\n" + "\n".join(src[i0:i1 + 1]) + '\nprintf "[%s]" "$ORCH_EXTRA"\n'
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env})
    return r.returncode, r.stdout + r.stderr


def test_run_campaign_spento_e_acceso():
    assert _blocco_d31({}) == (0, "[]")
    assert _blocco_d31({"ORCH_EXTRA": "--repeat-genome x"}) == (0, "[--repeat-genome x]")
    rc, txt = _blocco_d31({"PROVE_VINCITORE": "3", "RIALLINEA": "1", "REGISTRA_ASSETTO": "1",
                           "TWIN_INIT_ASSETTO": "attaccante", "CONFRONTO_VINCITORE": "1", "ORCH_EXTRA": "--x"})
    assert rc == 0 and txt == ("[--prove-vincitore 3 --riallinea-dopo-assestamento --registra-assetto "
                               "--twin-init-assetto attaccante --confronto-vincitore --x]")
    assert _blocco_d31({"TWIN_INIT_ASSETTO": "azzerato"}) == (0, "[]")
    assert _blocco_d31({"PROVE_VINCITORE": "tre"})[0] != 0
    assert _blocco_d31({"TWIN_INIT_ASSETTO": "vero"})[0] != 0


def test_coda_spenta_invariata():
    src = (ROOT / "scripts" / "coda_serieB.sh").read_text()
    # command, environment and expected options: each entry appears only when enabled
    assert 'MOLA_BINARIO_G1=$MOLA_BINARIO_G1${D31_DESC})"' in src
    assert '(( ${#D31_ENV[@]} )) && CAMP_ENV+=("${D31_ENV[@]}")' in src
    assert '(( ${#D31_ATTESE[@]} )) && ATTESE+=("${D31_ATTESE[@]}")' in src
    i0 = src.index("D31_ENV=(); D31_ATTESE=(); D31_DESC=\"\"")
    i1 = src.index("\n# --- Arguments ---")
    codice = "set -uo pipefail\n" + src[i0:i1] + '\necho "${#D31_ENV[@]}|${#D31_ATTESE[@]}|${D31_DESC}|"\n'
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.stdout.strip() == "0|0||"
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PROVE_VINCITORE": "3", "RIALLINEA": "1"})
    assert r.stdout.strip() == "2|2| PROVE_VINCITORE=3 RIALLINEA=1|"


# --- realigned run: resume, restored state, E1, warm start ---

class _Registri:
    def __init__(self):
        self.tag, self._t0_rollout = "w4_applied", 123.0

    def inizio_rollout(self, tag):
        self.tag, self._t0_rollout = tag, 999.0


class _NodeRipresa(_NodeR):
    """Victim node with the first-cloud filter: on resume the scan in progress is dropped
    (or, with consegna=True, delivered to the victim).
    """
    fresh_cloud = True

    def __init__(self, consegna=False):
        super().__init__()
        self.registri = _Registri()
        self.victim_rec_total, self.fresh_dropped_total = 220, 7
        self.restore_log = []
        self.consegna = consegna
        self.in_play = False

    def arm_fresh_cloud(self):
        self.last_fresh_arm = {"seq": 1, "confirmed": True}
        return True

    def spin(self, s=0.0):
        # no new pose: the estimate stays at the settling one
        if self.in_play and self.est_source == "victim":
            if self.consegna:
                self.victim_rec_total += 1
            else:
                self.fresh_dropped_total += 1
            self.in_play = False


class _IsaacRipresa(_IsaacR):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def play(self):
        super().play()
        self.node.in_play = True


def _riallineo_finto(monkeypatch, consegna=False):
    node = _NodeRipresa(consegna)
    isaac = _IsaacRipresa(node)
    mola = types.SimpleNamespace(log_path=None, last_stop={"vittima": "chiusura del gemello prima della vittima"})
    monkeypatch.setitem(ao.STALL_POLICY, "window", 4)
    monkeypatch.setitem(ao.TWIN_LOG_PREV, "last_stamp", 38.5)
    monkeypatch.setitem(ao.CONTROLLO_ETA, "finestre", {"4": {"tentativi": 10, "rifatti": 0}})
    monkeypatch.setattr(ao, "TWIN_STATE", {"mode": "ombra_g"})
    monkeypatch.setitem(ao.ASSETTO, "registra", False)

    def esegui(isa_p, tag, fs):
        # as in the twin rollout/_rollout_once: logs, MOLA shutdown, previous stamp, age
        node.registri.inizio_rollout(tag)
        mola.last_stop = {"riallineato": True}
        ao.TWIN_LOG_PREV["last_stamp"] = 99.9
        ao.CONTROLLO_ETA["finestre"]["4"]["tentativi"] += 1
        node.est_source, node.est_t, node.chamfer = "twin", 5.0, []
        node.last_trace = []
        return (0, 0, 0), (4.9, 0.4, 0.0), 41.0, "ok", 0.95, 11
    attacker = types.SimpleNamespace(run_origin=(0.0177, 0.0, 0.0), believed_pose=lambda n: (4.0855, 0.519, -0.1189))
    rial = ao.RiallineoVittima(4, esegui, _Ombra(None), attacker, Path("/nonesiste"), mola=mola)
    rial.out = Path(pytest.importorskip("tempfile").mkdtemp())
    est_t0 = node.est_t
    rial(isaac, node, {"tag": "w4_applied"})
    return rial, node, mola, est_t0


def test_d1_ripresa_senza_attesa_della_posa(monkeypatch):
    """Victim resumes from the last settling pose; it waits only for the dropped in-flight scan."""
    rial, node, _, est_t0 = _riallineo_finto(monkeypatch)
    r = rial.rec
    assert r["ripresa"]["attesa"] == "scartata" and node.est_t == est_t0 and node.est_raw == (3.68, -0.70, 0.16)
    assert r["conteggi"]["pausa"]["victim_rec_total"] == 220 and r["conteggi"]["ripresa"]["fresh_dropped_total"] == 7
    rial.alla_guida(node)
    assert r["nuvole_vittima_ripresa_guida"] == 0 and r["scartate_ripresa_guida"] == 1
    node.victim_rec_total += 9
    rial.alla_fine(node)
    assert r["conteggi"]["fine"]["victim_rec_total"] == 229 and r["scartate_ripresa_fine"] == 1
    # no drop and one cloud to the victim: driving starts at once, the record says so
    rial, node, _, _ = _riallineo_finto(monkeypatch, consegna=True)
    rial.alla_guida(node)
    assert rial.rec["ripresa"]["attesa"] == "nuvola_alla_vittima" and rial.rec["nuvole_vittima_ripresa_guida"] == 1
    # victim driving starts right after the counts, in order
    src = ORCH.read_text()
    i = src.index("def _victim_rollout(")
    j = src.index("return _drive_and_measure(node, isaac, victim, local_wps", i)
    assert "dopo_assestamento.alla_guida(node)" in src[j - 200:j]
    assert "while time.time() < deadline and node.est_t < t_res" not in src


def test_d1_richiede_il_filtro_della_prima_nuvola():
    a = _args_main(FISSO + ["--riallinea-dopo-assestamento"])
    assert "--fresh-cloud-after-restore" in ao.problema_opzioni_d31(a)
    assert ao.problema_opzioni_d31(_args_main(FISSO + ["--riallinea-dopo-assestamento",
                                                       "--fresh-cloud-after-restore"])) is None


def test_d2_d5_stato_rimesso_dopo_il_riallineato(monkeypatch):
    """Per-frame log tag and start, twin MOLA shutdown, last twin stamp, window age counts."""
    rial, node, mola, _ = _riallineo_finto(monkeypatch)
    assert node.registri.tag == "w4_applied" and node.registri._t0_rollout == 123.0
    assert mola.last_stop == {"vittima": "chiusura del gemello prima della vittima"}
    assert ao.TWIN_LOG_PREV["last_stamp"] == 38.5
    assert ao.CONTROLLO_ETA["finestre"]["4"]["tentativi"] == 10
    g = rial.rec["gemello"]
    assert g["mola_stop"] == {"riallineato": True} and g["controllo_eta_conteggi"]["tentativi"] == 11
    assert node.est_source == "victim" and node.chamfer == [46.7]


def test_d3_confronto_criterio_completo():
    p0, u = np.zeros(2), np.array([1.0, 0.0])
    tr = [{"travelled": 0.5 * i / 10, "true_x": 0.5 * i / 10, "true_y": 0.0} for i in range(11)]
    g = banco_g.grandezze_traccia(tr, p0, u)
    ramo = {"classe": "nessun_salto", "dy_cm": 0.0}
    d = {"mean_cm": 10.0}
    vecchio = banco_g.confronto_coppia(g, g, ramo, ramo, d, d)
    assert vecchio["segue"] and "criterio_completo" not in vecchio and "E1_inc_cm" not in vecchio
    assert vecchio["soglie"] == {"R1_cm": 3.0, "laterale_cm": 3.0, "danno_cm": 1.5}
    ok = banco_g.confronto_coppia(g, g, ramo, ramo, d, d, e1={"E1_inc_cm": 1.0}, dm_real=True)
    assert ok["segue"] and ok["criterio_completo"] and ok["soglie"]["E1_inc_cm"] == 3.5
    oltre = banco_g.confronto_coppia(g, g, ramo, ramo, d, d, e1={"E1_inc_cm": 6.1327}, dm_real=True)
    assert not oltre["segue"] and oltre["oltre"] == ["E1_inc_cm"]
    senza = banco_g.confronto_coppia(g, g, ramo, ramo, d, d, e1=None, dm_real=False)
    assert not senza["segue"] and senza["danno_cm"] is None and senza["non_misurati"] == ["E1_inc_cm", "danno_cm"]
    assert banco_g.e1_incrementale(None, None, None, None, None)["motivo"].startswith("mancano")


RUN_ACC = REP.parents[3]


@pytest.mark.skipif(not (RUN_ACC / "frame_log.csv").is_file() or not (REP / "logs" / "mola_victim.log").is_file(),
                    reason="run della prova di accettazione assente")
def test_d3_e1_incrementale_come_esito_ufficiale():
    """Incremental E1 of w4 computed as the official outcome, against tabella_per_metro.csv."""
    tab = {r["metro"]: r for r in csv.DictReader(open(
        ROOT / "data/attack/dossier/2026-10-23_serieB/prova_accettazione/esito_s1001_accettazione_30ott/"
        "tabella_per_metro.csv"))}
    for k, idx in ((4, 480), (3, 465)):
        e = banco_g.e1_incrementale(REP / "traces" / f"w{k}_applied.csv", REP / "logs" / "mola_victim.log",
                                    REP / "traces" / f"w{k}_e{idx}.csv", REP / "logs" / f"mola_w{k}_e{idx}.log",
                                    RUN_ACC / "frame_log.csv", True, False)
        t = tab[str(k)]
        assert e["E1_inc_cm"] == pytest.approx(float(t["E1_inc_cm"]), abs=1e-9)
        assert e["E1_ass_cm"] == pytest.approx(float(t["E1_ass_cm"]), abs=1e-9)
        assert e["R2_inc_cm"] == pytest.approx(float(t["R2_inc_cm"]), abs=1e-9)
    assert not list((ROOT / "scripts").glob("__pycache__/analisi_accettazione_esito30ott*"))


def test_d6_partenza_calda_senza_solo_validi_in_tutte_e_due_le_chiamate():
    """Pareto warm start uses fronte_pareto without solo_validi, before and after re-evaluation."""
    src = ORCH.read_text()
    i = src.index("def warm_pool_from_log(")
    assert "fronte_pareto(eval_log, budget=budget)" in src[i:i + 1500]
    assert src.count("warm_pool_from_log(\n                        eval_log, args.select, args.chamfer_budget, active_genes)") == 1
    assert src.count("warm_pool, warm_best, w_idx = warm_pool_from_log(") == 2


def test_d7_vittima_del_banco_in_pausa_senza_restore(tmp_path, monkeypatch):
    """With D31C=si the bench victim arm pauses at settling end and resumes without restore."""
    isaac, node = _IsaacTrio(), _NodeTrio()
    isaac.posa = (-2.05, -1.28, -0.09)
    node.st_stato_generatore = None
    monkeypatch.setattr(ao, "SHADOW_QUIET_S", 0.0)
    h = ao.RiallineoBanco(tmp_path / "V_riallineo", (0.0177, 0.0, 0.0))
    h(isaac, node, {"tag": "banco_c3_V_r0", "mola": types.SimpleNamespace(alive=lambda: True)})
    assert ("save", "riallineo") in isaac.comandi and not any(c[0] == "restore" for c in isaac.comandi)
    assert h.rec["stato"]["status"] == "ok" and h.rec["pausa_s"] >= 0.0


# --- twin attitude relative to the attacker estimate (attaccante_relativo, variant 3a) ---

def test_assetto_attaccante_relativo_3a():
    att = (4.0, 0.5, 0.01, math.radians(9.206027), math.radians(-0.166056), math.radians(0.028472))
    node = types.SimpleNamespace(attacker_est6=att, attacker_est_raw=(att[0], att[1], att[3]),
                                 attacker_est_stamp=38.6)
    kw, rec = ao.assetto_gemello("attaccante_relativo", node, {})      # no reading at spawn
    assert kw == {"roll": att[5], "pitch": att[4]} and rec["variante"].startswith("3a")
    assert rec["attaccante"]["stesso_messaggio_della_posa"] is True and "spawn" not in rec
    kw3b, rec3b = ao.assetto_gemello("attaccante", node, {"roll": 1e-3, "pitch": 0.0, "yaw": 0.0})
    assert rec3b["variante"].startswith("3b diagnostica, non adottabile") and kw3b != kw
    big = types.SimpleNamespace(attacker_est6=(0, 0, 0, 0, 0.2, 0.0), attacker_est_raw=(0, 0, 0))
    with pytest.raises(ao.ErroreAssetto, match="oltre"):
        ao.assetto_gemello("attaccante_relativo", big)
    a = _args_main(FISSO + ["--twin-init-assetto", "attaccante_relativo"])
    assert ao.problema_opzioni_d31(a) is None
    assert ao.problema_capacita_d31(a, ["set_state", "set_state_assetto"]) is None   # no attitude
    a = _args_main(FISSO + ["--twin-init", "ombra_est", "--twin-init-assetto", "attaccante_relativo"])
    assert "attaccante_relativo richiede --twin-init attacker_est" in ao.problema_opzioni_d31(a)
    assert _blocco_d31({"TWIN_INIT_ASSETTO": "attaccante_relativo"}) == (0, "[--twin-init-assetto attaccante_relativo]")


def test_assetto_3a_al_banco():
    a = {"modo": "esplicito_attaccante_relativo", "roll_rad": 4.9e-4, "pitch_rad": -2.9e-3, "fonte": "log"}
    assert banco_g.parametri_set_state(a) == {"roll": 4.9e-4, "pitch": -2.9e-3}
    assert banco_g._check_assetto(dict(a), "b")["modo"] == "esplicito_attaccante_relativo"
    with pytest.raises(ValueError, match="senza fonte"):
        banco_g._check_assetto({k: v for k, v in a.items() if k != "fonte"}, "b")
    assert banco_g.VARIANTE_ASSETTO == {"esplicito_attaccante_relativo": "3a, adottabile",
                                        "esplicito_relativo": "3b diagnostica, non adottabile"}
    assert '"assetto_variante": banco_g.VARIANTE_ASSETTO' in ORCH.read_text()


@pytest.mark.skipif(not (REP / "logs" / "mola_attacker.log").is_file(), reason="run della prova di accettazione assente")
def test_spec_assetto_3a_dalla_run(tmp_path):
    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "banco_G_d_spec.py"), "--out", str(out),
                        "--piano", "configurazioni", "--rep", str(REP), "--metro", "4", "--configurazioni", "3",
                        "--assetto-attaccante", "esplicito_attaccante_relativo", "--ripetizioni", "1"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    g = json.loads(out.read_text())["casi"][0]["bracci"][0]["assetto"]
    assert g["modo"] == "esplicito_attaccante_relativo" and "yaw_rad" not in g
    assert g["roll_rad"] == pytest.approx(math.radians(0.028472)) and g["pitch_rad"] == pytest.approx(math.radians(-0.166056))


def test_rigioco_accetta_attaccante_relativo():
    src = (ROOT / "scripts" / "prova_banco_serieB_rigioco_vincitori.sh").read_text()
    riga = next(x for x in src.splitlines() if x.startswith('case "$P_TIA" in'))
    for v, rc in (("attaccante_relativo", 0), ("attaccante", 0), ("vero", 2)):
        r = subprocess.run(["bash", "-c", f'P_TIA={v}\n{riga}\nexit 0'], capture_output=True, text=True)
        assert r.returncode == rc, (v, r.stdout)
    assert '"$P_TIA" == "attaccante_relativo"' in src


def test_stato_generatore_conferma_ed_errore():
    """--stato-generatore: seq confirmed, error with its reason, silent without support."""
    class N:
        st_stato_generatore = {"seq": None, "posti": {}, "armato": None, "ultimo_caricato": None,
                               "ultimo_errore": None}
        node = types.SimpleNamespace(create_publisher=lambda *a: "pub")

        def __init__(self, sbaglia):
            self.sbaglia, self.mandati = sbaglia, []

        def _publish_confirmed(self, pub, msg, confirmed, what, timeout=3.0):
            d = json.loads(msg.data)
            self.mandati.append(d)
            if self.sbaglia:
                self.st_stato_generatore = {**self.st_stato_generatore, "ultimo_errore": {
                    "seq": d["seq"], "azione": d["azione"], "nome": d["nome"], "motivo": "posto assente"}}
            else:
                self.st_stato_generatore = {**self.st_stato_generatore, "seq": d["seq"],
                                            "posti": {d["nome"]: {"flusso": d.get("flusso")}}}
            return confirmed()
    pytest.importorskip("std_msgs.msg")
    ok = N(False)
    r = ao.AttackNode.stato_generatore(ok, "salva", "riallineo", flusso="vittima")
    assert r["confermato"] and r["posto"] == {"flusso": "vittima"} and "errore" not in r
    ko = N(True)
    r = ao.AttackNode.stato_generatore(ko, "carica", "riallineo")
    assert not r["confermato"] and r["errore"]["motivo"] == "posto assente"
    senza = types.SimpleNamespace(st_stato_generatore=None)
    assert ao.AttackNode.stato_generatore(senza, "salva", "riallineo") is None


# --- realignment: generator state required, clouds saved ---

def test_stato_generatore_obbligatorio_con_d31c():
    assert "--stato-generatore" in ao.stato_generatore_mancante(types.SimpleNamespace(
        st_stato_generatore=None, spin=lambda s: None), timeout=0.0)
    assert ao.stato_generatore_mancante(types.SimpleNamespace(st_stato_generatore={"seq": None}), timeout=0.0) is None
    assert "non chiamato" in ao.errore_riallineo(None)
    assert "non eseguito" in ao.errore_riallineo({"gemello": None, "saltato": "ombra"})
    assert "nodo_senza_stato_generatore" in ao.errore_riallineo({"gemello": {}, "generatore": "nodo_senza_stato_generatore"})
    assert ao.errore_riallineo({"gemello": {}, "generatore": "trasferito"}) is None
    src = ORCH.read_text()
    i = src.index("if d31_errore:")
    assert "json.dump(history, f, indent=2)" in src[i - 200:i] and "raise ErroreConfigurazioneZone" in src[i:i + 150]
    assert "err = stato_generatore_mancante(node)" in src


def _node_start(env):
    """node_start of run_campaign.sh with the options block: node line."""
    src = (ROOT / "scripts" / "run_campaign.sh").read_text().splitlines()
    i0 = next(i for i, x in enumerate(src) if x.startswith("# Options off by default: with default values the orchestrator"))
    i1 = next(i for i in range(i0, len(src)) if src[i].startswith('[[ -n "$D31_FLAGS" ]]'))
    j0 = next(i for i, x in enumerate(src) if x.startswith("node_start() {"))
    j1 = next(i for i in range(j0, len(src)) if src[i] == "}")
    fn = "\n".join(src[j0:j1 + 1]).replace("exec python3 -u src/nodes/perturbation_node.py", "exec echo NODO")
    codice = ("set -uo pipefail\nORCH_EXTRA=\"\"; MAX_DRIFT_LAG_S=\"\"; STRUMENTI_NODO=0; PERC_TERMS=0; STRUCTURAL=\"\"; "
              "NODE_SB_FLAGS=\"\"; OPERATOR_NODE_FLAGS=\"\"; SERIE_B=0; MIN_LATENCY_MS=140; OUT=/r/rep_01\n"
              + "\n".join(src[i0:i1 + 1]) + "\n" + fn + '\nnode_start /dev/stdout ""; wait\n')
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_run_campaign_nodo_stato_generatore_e_nuvole():
    assert "--stato-generatore" not in _node_start({}) and "--salva-nuvole" not in _node_start({})
    riga = _node_start({"RIALLINEA": "1", "SALVA_NUVOLE": "1", "SALVA_NUVOLE_K": "15"})
    assert " --stato-generatore" in riga and " --salva-nuvole /r/rep_01/nuvole --salva-nuvole-k 15" in riga
    # already among the node options (replay launcher): not repeated
    riga = _node_start({"RIALLINEA": "1", "SALVA_NUVOLE": "1",
                        "NODE_EXTRA": "--stato-generatore --salva-nuvole /x/nuvole --salva-nuvole-k 20"})
    assert riga.count("--stato-generatore") == 1 and riga.count("--salva-nuvole ") == 1
    assert _blocco_d31({"SALVA_NUVOLE": "2"})[0] != 0 and _blocco_d31({"SALVA_NUVOLE_K": "0"})[0] != 0


def test_coda_attese_del_nodo():
    src = (ROOT / "scripts" / "coda_serieB.sh").read_text()
    i0 = src.index("D31_ENV=(); D31_ATTESE=(); D31_DESC=\"\"")
    i1 = src.index("\n# --- Arguments ---")
    codice = "set -uo pipefail\n" + src[i0:i1] + '\nprintf "[%s]" "${D31_NODO_ATTESE[@]}" "${D31_ENV[@]}"\n'
    r = subprocess.run(["bash", "-c", codice], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "RIALLINEA": "1", "SALVA_NUVOLE": "1"})
    assert r.stdout == ("[--stato-generatore][--salva-nuvole][--salva-nuvole-k 20]"
                        "[RIALLINEA=1][SALVA_NUVOLE=1][SALVA_NUVOLE_K=20]")
    rg = (ROOT / "scripts" / "prova_banco_serieB_rigioco_vincitori.sh").read_text()
    assert '[[ "$P_RIALL" == "1" ]] && NODO_ATTESE+=("--stato-generatore")' in rg
    assert 'in_help "$HELP_NODO" "--stato-generatore"' in rg
