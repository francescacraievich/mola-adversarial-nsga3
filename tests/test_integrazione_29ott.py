#!/usr/bin/env python3
"""
Single MOLA binary (--mola-binario-g1), queue with CONTROLLO_ETA, DDS_PROFILE, SONDA and
MOLA_BINARIO_G1, cloud-age failure as chain failure, probe in run_campaign.sh. No ROS or
Isaac.

    python3 -m pytest tests/test_integrazione_29ott.py -q
"""

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.optimization import twin_variants as tv  # noqa: E402
from src.optimization import shadow_check as shc  # noqa: E402
import coda_serieB_stato as coda  # noqa: E402

TRE = ("LD_LIBRARY_PATH", "MOLA_MODULES_LIB_PATH", "MOLA_FULL_STATE")


def _ao():
    from src.optimization import attack_orchestrator as ao
    return ao


def _overlay(tmp_path):
    root = tmp_path / "install_G1"
    for d in tv.g_overlay_dirs(str(root)):
        os.makedirs(d)
    return str(root)


def _mole(tmp_path, monkeypatch, attivo):
    """MOLA environments composed as in main (victim with --victim-shadow-env, shadow G,
    attacker, twin G)."""
    ao = _ao()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ros/jazzy/lib")
    monkeypatch.delenv("MOLA_MODULES_LIB_PATH", raising=False)
    root = _overlay(tmp_path)
    args = SimpleNamespace(mola_binario_g1=attivo, g_overlay=root)
    cfg_env = {}
    cfg_env.update(ao.ambiente_binario_g1(args))
    log = tmp_path / "logs"
    rv, rs = tv.mola_ros_args("victim", "true"), tv.mola_ros_args("shadow", "true")
    victim = ao.VictimRunner("s.sh", log, ros_args=rv, env=dict(cfg_env))
    victim.extra_env = ao.victim_extra_env(cfg_env, {}, rv, shadow_env=True)
    shadow = ao.ShadowRunner("s.sh", log, ros_args=rs, env=cfg_env, g_root=root)
    attacker = ao.AttackerRunner("s.sh", log, ros_args="--ros-args", env=cfg_env)
    twin_base = {**cfg_env, "ROS_ARGS": "--ros-args"}
    twin_g = {**twin_base, **tv.g_twin_env(str(tmp_path / "w0.molastate"), root=root)}
    return {"vittima": victim.extra_env, "ombra": shadow.extra_env, "attaccante": attacker.extra_env,
            "gemello_base": twin_base, "gemello_g": twin_g}


def test_binario_unico_stesse_tre_variabili(tmp_path, monkeypatch):
    m = _mole(tmp_path, monkeypatch, True)
    for k in TRE:
        valori = {r: e.get(k) for r, e in m.items()}
        assert len(set(valori.values())) == 1 and None not in valori.values(), (k, valori)
    assert m["vittima"]["MOLA_FULL_STATE"] == "1"
    assert m["vittima"]["LD_LIBRARY_PATH"].endswith(":/opt/ros/jazzy/lib")
    d = shc.env_config_diff(m["vittima"], m["ombra"])
    assert d["identica"] is True and d["diverse"] == []


def test_senza_opzione_come_oggi(tmp_path, monkeypatch):
    ao = _ao()
    assert ao.ambiente_binario_g1(SimpleNamespace(mola_binario_g1=False, g_overlay="x")) == {}
    assert ao.ambiente_binario_g1(SimpleNamespace()) == {}
    m = _mole(tmp_path, monkeypatch, False)
    for r in ("vittima", "attaccante", "gemello_base"):
        assert not any(k in m[r] for k in TRE), r
    assert all(k in m["ombra"] for k in TRE)
    d = shc.env_config_diff(m["vittima"], m["ombra"])
    assert d["identica"] is False and sorted(d["diverse"]) == sorted(TRE)   # as in s1001


def test_main_usa_il_binario_per_tutte(tmp_path):
    ao = _ao()
    src = inspect.getsource(ao.main)
    i = src.index("cfg_env = mola_config_env(args)")
    j = src.index("TWIN_STATE[\"twin_env_base\"]")
    assert "cfg_env.update(g1_env)" in src[i:j]
    # victim, shadow and attacker get cfg_env after the update
    assert src.index("env={**cfg_env, **victim_env}") > i
    assert src.index("env=cfg_env, topic=args.shadow_topic") > i
    assert src.index("ros_args=ros_args[\"attacker\"], env=cfg_env") > i


def test_argomento_e_overlay_assente(monkeypatch, capsys, tmp_path):
    ao = _ao()
    monkeypatch.setattr("sys.argv", ["attack_orchestrator.py", "--mola-binario-g1",
                                     "--g-overlay", str(tmp_path / "manca")])
    assert ao.main() == 2
    assert "--mola-binario-g1: overlay di MOLA assente" in capsys.readouterr().out


def test_history_registra_il_binario(monkeypatch):
    ao = _ao()
    base = dict(twin_state="clean", replay_frame=None, twin_init="true", uniform_sim_time="false",
                drift_reset_on="genome", twin_align="none")
    monkeypatch.setitem(ao.TWIN_STATE, "binario_g1", {"overlay": "R", "env": {"MOLA_FULL_STATE": "1"}})
    assert "mola_binario_g1" not in ao.twin_config_fields(SimpleNamespace(**base))
    rec = ao.twin_config_fields(SimpleNamespace(**base, mola_binario_g1=True))
    assert rec["mola_binario_g1"]["env"]["MOLA_FULL_STATE"] == "1"


def test_guasto_eta_vittima_scrive_da_rifare():
    src = inspect.getsource(_ao().main)
    i = src.index("if isinstance(e, GuastoEtaNuvola):")
    assert "da_rifare.json" in src[i:i + 600] and '"causa": "eta_nuvola"' in src[i:i + 600]


# --- coda_serieB_stato.py ---

def _rep(tmp_path, hist):
    rep = tmp_path / "rep_01"
    rep.mkdir()
    (rep / "history.json").write_text(json.dumps(hist))
    return rep


def test_stallo_eta_nuvola_e_guasto_catena(tmp_path):
    rep = _rep(tmp_path, [{"window": 0}, {"window": 1, "status": "isaac_stall", "error": "eta",
                                           "causa": "eta_nuvola"}])
    s = coda.storia(str(rep))
    assert s["guasto_catena"] is True and "eta_nuvola" in " ".join(s["run_end"])
    e = coda.classifica({"rc": 3, "storia": s, "completata": False})
    assert e["esito"] == "guasto_catena" and "eta_nuvola" in e["motivo"]


def test_stallo_di_isaac_senza_causa_come_prima(tmp_path):
    rep = _rep(tmp_path, [{"window": 1, "status": "isaac_stall", "error": "x"}])
    s = coda.storia(str(rep))
    assert s["guasto_catena"] is False and s["run_end"] == []


def test_riepilogo_conteggi_controllo_eta(tmp_path):
    c = {"tentativi": 40, "rifatti": 3, "ripetuti": 1, "guasti_vittima": 0, "non_determinati": 0,
         "quota_rifatti": 0.075}
    rep = _rep(tmp_path, [{"window": 0, "controllo_eta": c}, {"window": 1}])
    d = coda.riepilogo_run(str(tmp_path), rep=str(rep), usa_bag=False)
    assert d["controllo_eta"] == [{"finestra": 0, "tentativi": 40, "rifatti": 3, "guasto_eta_ripetuto": 1,
                                   "quota_rifatti": 0.075, "guasti_vittima": 0, "non_determinati": 0}]
    t = coda.testo_riepilogo_run(d)
    assert "w0: tentativi 40, rifatti 3, guasto_eta_ripetuto 1, quota rifatti 0.075" in t
    rep2 = tmp_path / "senza"
    rep2.mkdir()
    (rep2 / "history.json").write_text(json.dumps([{"window": 0}]))
    d2 = coda.riepilogo_run(str(tmp_path), rep=str(rep2), usa_bag=False)
    assert "controllo_eta" not in d2 and "eta' della nuvola" not in coda.testo_riepilogo_run(d2)


# --- coda_serieB.sh a secco ---

def _modulo(nome):
    spec = importlib.util.spec_from_file_location(nome, ROOT / "tests" / f"{nome}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_coda_a_secco_con_i_controlli(tmp_path):
    (tmp_path / "b_s1001.csv").write_text("x,y\n")
    profilo = str(ROOT / "config" / "dds" / "fastdds_buffer4M_shm16M.xml")
    tcs = _modulo("test_coda_serieB")
    p = tcs._a_secco(tmp_path, BASELINE_SERIEB=str(tmp_path / "b_s{seme}.csv"), CONTROLLO_ETA="0.40",
                     DDS_PROFILE=profilo, SONDA="1", MOLA_BINARIO_G1="1")
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert "CONTROLLO_ETA=0.40 DDS_PROFILE=" + os.path.realpath(profilo) + " SONDA=1 MOLA_BINARIO_G1=1" in out
    assert "[--controllo-eta 0.40]" in out and "[--mola-binario-g1]" in out
    assert f"[--dds-profile {os.path.realpath(profilo)}]" in out
    # queue baseline without --victim-shadow-env: the age check cannot run
    assert "CONTROLLO_ETA con la baseline della coda" in out
    p0 = tcs._a_secco(tmp_path, BASELINE_SERIEB=str(tmp_path / "b_s{seme}.csv"))
    assert "--controllo-eta" not in p0.stdout and "--mola-binario-g1" not in p0.stdout
    assert "SONDA=0 MOLA_BINARIO_G1=0" in p0.stdout


# --- run_campaign.sh ---

def test_comando_con_binario_g1(tmp_path):
    tia = _modulo("test_integrazione_A")
    rc, out = tia._comando(ROOT / "scripts" / "run_campaign.sh", tmp_path, {"MOLA_BINARIO_G1": "1"},
                           arm="none", reps=1)
    assert rc == 0, out
    assert re.search(r"CMD .*--mola-binario-g1", out)
    rc, out = tia._comando(ROOT / "scripts" / "run_campaign.sh", tmp_path, {}, arm="none", reps=1)
    assert rc == 0 and "--mola-binario-g1" not in out
    src = (ROOT / "scripts" / "run_campaign.sh").read_text()
    assert src.count("$ETA_FLAG $G1_FLAG $REG_FLAG $ORCH_EXTRA") == 2  # command logged and launched


def _funzione(src, nome):
    m = re.search(rf"^{nome}\(\) \{{.*?^\}}", src, re.S | re.M)
    return m.group(0)


def test_sonda_prima_e_dopo_l_orchestratore(tmp_path):
    src = (ROOT / "scripts" / "run_campaign.sh").read_text()
    i_start = src.index('sonda_start "$OUT/sonda"')
    i_orch = src.index("exec python3 -u src/optimization/attack_orchestrator.py")
    i_wait = src.index('wait "$ORCH_PID"')
    i_stop = src.index("sonda_stop", i_wait)
    assert i_start < i_orch < i_wait < i_stop < src.index("node_stop", i_wait)
    # sonda_start with fake python3: arguments, shadow topic, ready file
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text('#!/bin/bash\necho "$@" >> "$ARGS_FILE"\n'
                                    'for ((i=1;i<=$#;i++)); do [[ "${!i}" == --out ]] && { j=$((i+1)); touch "${!j}/pronta"; }; done\n')
    (bindir / "python3").chmod(0o755)
    prova = tmp_path / "prova.sh"
    prova.write_text("set -euo pipefail\nstop_pid() { :; }\nSONDA_PID=''; SONDA_UDP_PID=''\n" + _funzione(src, "sonda_start") + "\n"
                     + _funzione(src, "sonda_stop") + "\n"
                     'sonda_start "$1"; wait; sonda_stop; echo fatto\n')
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "ARGS_FILE": str(tmp_path / "args.txt"),
           "SONDA": "1", "ORCH_EXTRA": "--shadow-check --shadow-topic /carter/lidar_shadow --x 1"}
    r = subprocess.run(["bash", str(prova), str(tmp_path / "rep" / "sonda")], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 0 and "fatto" in r.stdout, r.stdout + r.stderr
    righe = (tmp_path / "args.txt").read_text()
    assert "scripts/sonda_udp.py --out" in righe
    assert ("scripts/sonda_nuvole.py --out " + str(tmp_path / "rep" / "sonda") + " --topic /carter/lidar_victim "
            "--topic /carter/lidar_attacker --topic /carter/lidar_perturbed --topic /carter/lidar_shadow") in righe
    # off: no process
    (tmp_path / "args.txt").unlink()
    env["SONDA"] = "0"
    r = subprocess.run(["bash", str(prova), str(tmp_path / "rep2")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and not (tmp_path / "args.txt").exists()


def test_prove_al_banco_ereditano_le_variabili():
    for nome in ("prova_banco_serieB_pilota_zone.sh", "prova_banco_serieB_baseline_definitiva.sh"):
        src = (ROOT / "scripts" / nome).read_text()
        # env without -i: environment inherited
        assert 'exec env "${CAMP_ENV[@]}" scripts/run_campaign.sh' in src, nome
        camp = re.search(r"CAMP_ENV=\((.*?)\)\n", src, re.S).group(1)
        for v in ("CONTROLLO_ETA=", "DDS_PROFILE=", "SONDA=", "MOLA_BINARIO_G1="):
            assert v not in camp, (nome, v)   # not overwritten: they come from the environment


# --- winner damage with failed re-evaluations ---

def test_topk_rivalutazione_guasta_esclusa():
    ao = _ao()
    # normal case: same values as before
    v = ao.valori_topk(10.0, 6.0, "ok", "min")
    assert v == {"reeval_dev_mean_cm": 6.0, "mean_cm": 8.0, "min_cm": 6.0, "agg_cm": 6.0}
    assert ao.valori_topk(10.0, 6.0, "ok", "mean")["agg_cm"] == 8.0
    # failed re-evaluation: only the search damage counts, declared
    g = ao.valori_topk(10.0, 0.0, ao.ESITO_ETA_RIPETUTO, "min")
    assert g["agg_cm"] == g["mean_cm"] == g["min_cm"] == 10.0 and g["reeval_dev_mean_cm"] is None
    assert g["reeval_escluso"] == ao.ESITO_ETA_RIPETUTO and g["valutato_su"] == "solo_ricerca"
    # the choice is not driven by the failure: chosen among valid re-evaluations
    cand = [{"g": "A", **g}, {"g": "B", **ao.valori_topk(8.0, 6.0, "ok", "min")}]
    assert ao.scegli_topk(cand)["g"] == "B" and "scelta_fra_esclusi" not in ao.scegli_topk(cand)
    # all failed: maximum of the search damage only, declared
    tutti = [{"g": "A", **g}, {"g": "C", **ao.valori_topk(12.0, 0.0, ao.ESITO_ETA_RIPETUTO, "min")}]
    s = ao.scegli_topk(tutti)
    assert s["g"] == "C" and s["scelta_fra_esclusi"] is True
    # no failure: as before (maximum of agg_cm)
    ok = [{"g": "D", **ao.valori_topk(9.0, 7.0, "ok", "mean")}, {"g": "E", **ao.valori_topk(10.0, 2.0, "ok", "mean")}]
    assert ao.scegli_topk(ok)["g"] == "D"


def test_main_usa_valori_topk_e_ramo_ratio():
    src = inspect.getsource(_ao().main)
    assert "v_top = valori_topk(search_cm, rev, re_st, args.topk_agg)" in src
    assert "best_rec = scegli_topk(topk_rec)" in src
    assert "agg_cm = min(search_cm, rev)" not in src
    i = src.index("rivalutazione: SCARTATA")
    assert '"escluso": ESITO_ETA_RIPETUTO' in src[i - 300:i]


def _eta(monkeypatch, tmp_path, esiti, stalli):
    ao = _ao()
    m = _modulo("test_controllo_eta_dds")
    chiamate, _ = m._prepara(monkeypatch, tmp_path, esiti)
    orig = ao._rollout_once
    tutte = []

    def _once(*a, **k):
        tutte.append(a[6])
        if len(tutte) - 1 in stalli:
            raise ao.IsaacStall("stallo finto")
        return orig(*a, **k)

    monkeypatch.setattr(ao, "_rollout_once", _once)
    monkeypatch.setattr(ao, "_isaac_recovery", lambda *a, **k: None)
    monkeypatch.setattr(ao, "STALL_POLICY", {**ao.STALL_POLICY, "window": 1})
    return m, tutte


def test_stallo_poi_eta_e_primo_guasto(monkeypatch, tmp_path):
    ao = _ao()
    m, tutte = _eta(monkeypatch, tmp_path, ["guasto", "ok"], {0})
    res = ao.rollout(None, None, m._Mola(None), m._Node(), [(1, 0)], 1.0, "w1_e2")
    assert tutte == ["w1_e2", "w1_e2r", "w1_e2rr"] and res[3] == "ok"
    c = ao.CONTROLLO_ETA["finestre"]["1"]
    assert (c["tentativi"], c["rifatti"], c["ripetuti"]) == (2, 1, 0)


def test_stallo_poi_due_guasti_eta(monkeypatch, tmp_path):
    ao = _ao()
    m, tutte = _eta(monkeypatch, tmp_path, ["guasto", "guasto"], {0})
    res = ao.rollout(None, None, m._Mola(None), m._Node(), [(1, 0)], 1.0, "w1_e2")
    assert len(tutte) == 3 and res[3] == ao.ESITO_ETA_RIPETUTO
    c = ao.CONTROLLO_ETA["finestre"]["1"]
    assert (c["rifatti"], c["ripetuti"]) == (1, 1)


def test_eta_poi_stallo_poi_ok_e_due_stalli(monkeypatch, tmp_path):
    ao = _ao()
    m, tutte = _eta(monkeypatch, tmp_path, ["guasto", "ok"], {1})
    assert ao.rollout(None, None, m._Mola(None), m._Node(), [(1, 0)], 1.0, "w1_e4")[3] == "ok"
    assert len(tutte) == 3
    m, tutte = _eta(monkeypatch, tmp_path, ["ok"], {0, 1})
    import pytest
    with pytest.raises(ao.IsaacStall):
        ao.rollout(None, None, m._Mola(None), m._Node(), [(1, 0)], 1.0, "w1_e5")
    assert len(tutte) == 2


def test_senza_controllo_eta_due_tentativi(monkeypatch):
    ao = _ao()
    monkeypatch.setattr(ao, "CONTROLLO_ETA", {"soglia": None, "frame_log": None, "registro": None})
    monkeypatch.setattr(ao, "_note_twin_log", lambda n, m: None)
    monkeypatch.setattr(ao, "_note_twin_fp", lambda n, m: None)
    monkeypatch.setattr(ao, "_isaac_recovery", lambda *a, **k: None)
    tag = []

    def _once(*a, **k):
        tag.append(a[6])
        raise ao.IsaacStall("x")

    monkeypatch.setattr(ao, "_rollout_once", _once)
    m = _modulo("test_controllo_eta_dds")
    import pytest
    with pytest.raises(ao.IsaacStall):
        ao.rollout(None, None, m._Mola(None), m._Node(), [(1, 0)], 1.0, "w0_e1")
    assert tag == ["w0_e1", "w0_e1r"]


def _preventivo(tmp_path, profilo, env_proc):
    vivo = subprocess.Popen(["sleep", "60"], env={**env_proc, "PATH": os.environ["PATH"]})
    try:
        cmd = (f'ROOT={ROOT}; OUT={tmp_path}; A_SECCO=0; source {ROOT}/scripts/prova_banco_serieB_lib.sh; '
               f'pid_adattatore() {{ echo {vivo.pid}; }}; riavvia_adattatore() {{ echo RIAVVIO; }}; '
               f'riavvio_preventivo {tmp_path}/adattatore.log')
        env = {**os.environ, "DDS_PROFILE": profilo}
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)
    finally:
        vivo.kill()
        vivo.wait()


def test_riavvio_preventivo_con_profilo(tmp_path):
    profilo = os.path.realpath(ROOT / "config" / "dds" / "fastdds_buffer4M_shm16M.xml")
    r = _preventivo(tmp_path, profilo, {})
    assert "RIAVVIO" in r.stdout and "senza FASTRTPS_DEFAULT_PROFILES_FILE" in r.stdout, r.stdout + r.stderr
    r = _preventivo(tmp_path, profilo, {"FASTRTPS_DEFAULT_PROFILES_FILE": profilo})
    assert "RIAVVIO" not in r.stdout and "tenuto" in r.stdout, r.stdout + r.stderr
    # no profile: as before, young adapter kept
    r = _preventivo(tmp_path, "", {})
    assert "RIAVVIO" not in r.stdout and "tenuto" in r.stdout, r.stdout + r.stderr


# --- fallback fitness ---

def test_causa_e_blocco_della_finestra():
    ao = _ao()
    assert ao.causa_riserva(ao.ESITO_ETA_RIPETUTO, "infra") == "guasto_eta_ripetuto"
    assert ao.causa_riserva("isaac_stall", "infra") == "infra:isaac_stall"
    assert ao.causa_riserva("timeout(3.0)", "countable") == "scartato:timeout"
    assert ao.causa_riserva("no_pose", "infra", guasto_g=True) == "guasto_gemello"
    assert ao.causa_riserva("victim_lost", "infra", dos=True) == "dos_vittima_senza_posa"
    ev = [{"genome": [0]}, {"fitness_riserva": "guasto_eta_ripetuto"}, {"fitness_riserva": "non_furtivo"},
          {"fitness_riserva": "guasto_eta_ripetuto"}]
    b = ao.fitness_riserva_finestra(ev)
    assert b == {"n": 3, "n_candidati": 4, "cause": {"guasto_eta_ripetuto": {"n": 2, "indici": [1, 3]},
                                                     "non_furtivo": {"n": 1, "indici": [2]}}}
    assert ao.fitness_riserva_finestra([])["n"] == 0


def test_ogni_fitness_di_riserva_registra_la_causa():
    src = inspect.getsource(_ao().main)
    righe = src.splitlines()
    ritorni = [i for i, r in enumerate(righe) if "1e6)" in r and "return" in r]
    assert len(ritorni) == 3
    for i in ritorni:
        assert any("fitness_riserva" in r for r in righe[i - 4:i]), righe[i]
    assert '"fitness_riserva": fitness_riserva_finestra(eval_log)' in src


def _genomi(rng, n, sposta=None):
    out = []
    for _ in range(n):
        g = [rng.uniform(-1, 1) for _ in range(17)]
        if sposta is not None:
            g[sposta] = min(1.0, g[sposta] * 0.2 + 0.8)
        out.append(g)
    return out


def _storia_riserva(tmp_path, sposta):
    import random
    rng = random.Random(7)
    ev = ([{"genome": g, "fitness_riserva": "guasto_eta_ripetuto"} for g in _genomi(rng, 12, sposta)]
          + [{"genome": g, "counted": True} for g in _genomi(rng, 60)])
    rep = tmp_path / ("rep_sposta" if sposta is not None else "rep_uguali")
    rep.mkdir()
    (rep / "history.json").write_text(json.dumps([{"window": 0, "evaluations": ev}]))
    return rep


def test_confronto_geni_spostato_e_uguali(tmp_path):
    m = coda.fitness_riserva_rep(str(_storia_riserva(tmp_path, 3)))[0]
    assert (m["n_riserva"], m["n_candidati"], m["fonte"]) == (12, 72, "registrata")
    g3 = m["confronto"]["geni"][3]
    assert g3["p"] < 0.001 and g3["differenza"] > 0.5
    assert m["confronto"]["attivi"]["p"] is not None
    u = coda.fitness_riserva_rep(str(_storia_riserva(tmp_path, None)))[0]
    assert u["confronto"]["geni"][3]["p"] > 0.05


def test_confronto_senza_scipy(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    m = coda.fitness_riserva_rep(str(_storia_riserva(tmp_path, 3)))[0]
    g3 = m["confronto"]["geni"][3]
    assert g3["metodo"].startswith("permutazioni") and g3["p"] < 0.001
    p1, _ = coda.mann_whitney([0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45])
    assert p1 > 0.5


def test_pochi_candidati_e_ricostruzione(tmp_path):
    rep = tmp_path / "r"
    rep.mkdir()
    ev = [{"genome": [0.0] * 17, "counted": False}, {"genome": [0.1] * 17, "counted": True, "stealth_ok": False},
          {"genome": [0.2] * 17, "counted": True}, {"genome": [0.3] * 17, "counted": True, "superseded_by": 4}]
    (rep / "history.json").write_text(json.dumps([{"window": 2, "evaluations": ev}, {"window": None}]))
    m = coda.fitness_riserva_rep(str(rep))[0]
    assert m["n_riserva"] == 2 and m["fonte"].startswith("ricostruita") and m["confronto"] is None
    assert m["cause"] == {"ricostruita": {"n": 2, "indici": [0, 1]}}


def test_riga_di_comando_e_riepilogo(tmp_path):
    rep = _storia_riserva(tmp_path, 3)
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "coda_serieB_stato.py"), "fitness-riserva",
                        str(rep), "--json", str(tmp_path / "fr.json")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "w0: 12/72 (guasto_eta_ripetuto 12)" in r.stdout and "g3 (mediane" in r.stdout
    assert json.loads((tmp_path / "fr.json").read_text())[0]["n_riserva"] == 12
    d = coda.riepilogo_run(str(tmp_path), rep=str(rep), usa_bag=False)
    assert d["fitness_riserva"][0]["n_riserva"] == 12
    assert "fitness di riserva per metro" in coda.testo_riepilogo_run(d)
