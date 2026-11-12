"""Chain rule: a chain failure is checked before a run is closed as attack detected. With fake
objects: (1) victim deliveries stalled until the decision, measured on the per-frame log rows
(stamp_victim, t_pub_victim_end_mono); (2) victim poses with good ICP not received by the
orchestrator (DDS); (3) unmonitored adapter: undetermined, written in the record; (4) chain
failure = repetition to redo (da_rifare.json, exit code 4 of run_campaign.sh), seen by the
queue as run_end guasto_catena; (5) MOLA slowdown as the only signal: --rallentamento-mola
guasto|scoperto. run_campaign.sh runs with fake python3, pgrep, ros2 and sleep in PATH.
"""

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization import attack_orchestrator as ao  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = shutil.which("python3")
SLEEP = shutil.which("sleep")


def _modulo(nome):
    spec = importlib.util.spec_from_file_location(f"_t_{nome}", ROOT / "tests" / f"{nome}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RN = _modulo("test_regole_notte")      # per-frame log and fake traces, _catena, _chiusura


def _riga(t, st, aperta=True, cons=True, attacco=True, fresh=""):
    """Per-frame log row: arrival t, input stamp st."""
    r = {c: "" for c in RN.COLONNE_FL}
    r.update(t_arr_wall=str(t), stamp_in=str(st), victim_stream="1" if aperta else "0",
             victim_attack="1" if attacco else "0", fresh_gate=fresh,
             stamp_victim=str(round(st - 60.0, 3)) if cons else "",
             t_pub_victim_end_mono="5.0" if cons else "")
    return r


# --- (1) deliveries stalled until the decision ---

def test_consegne_ferme_in_coda_sono_guasto():
    # 10 s delivered, then 20 s of arrivals without deliveries until the decision
    righe = [_riga(100 + 0.1 * i, 70 + 0.1 * i, cons=i < 100) for i in range(300)]
    b = wrec.buchi_consegna_vittima(righe, 100.0)
    assert b["esito"] == "guasto" and b["buco_in_coda_s"] == 20.1 and b["buco_max_s"] == 20.1
    assert b["buco_da_stamp"] == round(70 + 9.9 - 60.0, 3)    # victim clock
    # before the rule: only between two deliveries, hence "ok"
    assert b["n_consegnate"] == 100 and b["n_attese"] == 300
    # regular flow to the end: ok, tail of one period
    ok = wrec.buchi_consegna_vittima([_riga(100 + 0.1 * i, 70 + 0.1 * i) for i in range(50)], 100.0)
    assert ok["esito"] == "ok" and ok["buco_max_s"] == 0.1 and ok["buco_in_coda_s"] == 0.1


def test_tratto_senza_consegne_e_righe_non_attese():
    # attack-on segment with no delivery for 2 s: failure
    nulla = [_riga(100 + 0.1 * i, 70 + 0.1 * i, cons=False) for i in range(20)]
    b = wrec.buchi_consegna_vittima(nulla, 100.0)
    assert b["esito"] == "guasto" and b["buco_max_s"] == 2.1 and b["buco_da_stamp"] is None
    # --victim-never-clean: clean clouds (attack off) do not reach the victim, pilot w0
    w0 = [_riga(100 + 0.1 * i, 5 + 0.1 * i, cons=False, attacco=False) for i in range(32)]
    dopo = ([_riga(110, 73.6, cons=False, fresh="scartata_ripristino")]
            + [_riga(110.1 + 0.1 * i, 73.7 + 0.1 * i) for i in range(50)])
    assert wrec.buchi_consegna_vittima(w0 + dopo, 100.0, never_clean=True)["esito"] == "ok"
    # without never_clean the same clean rows are expected and not delivered
    assert wrec.buchi_consegna_vittima(w0 + dopo, 100.0)["esito"] == "guasto"
    # the cloud dropped by the post-restore filter does not open a gap at the head
    assert wrec.buchi_consegna_vittima(dopo, 100.0)["buco_max_s"] == 0.1
    # no expected row in the window: undetermined
    assert wrec.buchi_consegna_vittima(w0, 100.0, never_clean=True)["esito"] == "non_determinato"


def test_controlla_catena_consegne_ferme(tmp_path, monkeypatch):
    t0 = time.time() - 60
    ferme = [(t0 + 1 + 0.1 * i, True, i < 10, 20 + 0.1 * i) for i in range(50)]
    c = RN._catena(tmp_path, monkeypatch, frame=ferme)
    assert c["guasto"] is True and c["controlli"]["nuvole_vittima"]["esito"] == "guasto"
    assert c["controlli"]["nuvole_vittima"]["buco_in_coda_s"] == 4.1


# --- (2) poses lost between the victim MOLA and the orchestrator ---

def _printf(stamps_buoni, stamps_rifiutati=()):
    return ([{"stamp_s": round(s, 3), "is_good": 1} for s in stamps_buoni]
            + [{"stamp_s": round(s, 3), "is_good": 0} for s in stamps_rifiutati])


def test_pose_non_ricevute():
    buone = [20 + 0.1 * i for i in range(51)]                 # good ICP up to 25.0
    assert wrec.pose_non_ricevute(_printf(buone), 25.0)["esito"] == "ok"
    # pose lost because ICP was rejected after the last received one: not a chain failure
    r = wrec.pose_non_ricevute(_printf(buone, [25.1 + 0.1 * i for i in range(30)]), 25.0)
    assert r["esito"] == "ok" and r["n_buone_dopo"] == 0
    # MOLA published up to 25.0 but the orchestrator saw only up to 22.0: failure
    r = wrec.pose_non_ricevute(_printf(buone), 22.0)
    assert r["esito"] == "guasto" and r["non_ricevute_s"] == 3.0 and r["fonte"] == "printf"
    # below the 1 s threshold (poses in flight): ok
    assert wrec.pose_non_ricevute(_printf(buone), 24.2)["esito"] == "ok"
    # no pose ever received, MOLA with good ICP for 2 s: failure
    assert wrec.pose_non_ricevute(_printf(buone[:21]), None)["esito"] == "guasto"
    # without printf: traces with quality above min_icp_goodness
    tr = RN._tracce([0.02] * 50)                              # q 0.8, stamps up to 25.0
    r = wrec.pose_non_ricevute([], 22.0, tracce=tr, q_min=0.2)
    assert r["esito"] == "guasto" and r["fonte"] == "tracce"
    assert wrec.pose_non_ricevute([], 22.0, tracce=tr, q_min=0.9)["esito"] == "ok"
    # no data: undetermined
    assert wrec.pose_non_ricevute([], 22.0)["esito"] == "non_determinato"
    assert wrec.pose_non_ricevute([], 22.0, tracce=tr, q_min=None)["esito"] == "non_determinato"


def test_soglia_icp_buona_dalla_pipeline():
    v, fonte = wrec.soglia_icp_buona({})
    assert v == 0.20 and "lidar3d-gicp-optimize-twist-local.yaml" in fonte
    assert wrec.soglia_icp_buona({"MOLA_MINIMUM_ICP_QUALITY": "0.3"})[0] == 0.3


def test_controlla_catena_posa_persa(tmp_path, monkeypatch):
    log = tmp_path / "mola_victim.log"
    log.write_text("".join(_riga_printf_vera(i, 20 + 0.1 * i) + "\n" for i in range(51)))
    c = RN._catena(tmp_path, monkeypatch, ricevuta=22.0, log_path=str(log))
    assert c["guasto"] is True and c["controlli"]["posa_vittima"]["esito"] == "guasto"
    assert c["controlli"]["posa_vittima"]["fonte"] == "printf"
    c = RN._catena(tmp_path, monkeypatch, ricevuta=25.0, log_path=str(log))
    assert c["guasto"] is False and c["controlli"]["posa_vittima"]["esito"] == "ok"
    # without printf: victim traces and pipeline threshold
    c = RN._catena(tmp_path, monkeypatch, ricevuta=22.0)
    assert c["controlli"]["posa_vittima"]["fonte"] == "tracce" and c["guasto"] is True


def _riga_printf_vera(i, stamp):
    s = f"1970/01/01,00:00:{stamp:09.6f}"
    return (f"[LidarOdometry] pathStep={i} timestamp={s} goodness=0.700 minRequired=0.200 iters=12 "
            f"termReason=IterTermReason::Stalled isGood=1 adapt_thres_sigma=0.040 consecutive_bad=0")


# --- (3) unmonitored adapter ---

def test_adattatore_senza_sorveglianza_nel_record():
    ev = [{"t_check": 1792800000.0, "p99": 30}]                 # last check is old
    a = wrec.eventi_adattatore(ev, 1793000000.0, 1793000100.0)
    assert a["esito"] == "non_determinato" and a["sorveglianza_attiva"] is False
    assert a["motivo"].startswith("sorveglianza dell'adattatore assente") and "2026-" in a["motivo"]
    assert a["ultimo_evento_wall"] == 1792800000.0
    assert wrec.eventi_adattatore(None, 0, 1)["sorveglianza_attiva"] is False
    assert wrec.eventi_adattatore([{"t_check": 50, "p99": 30}], 0, 100)["sorveglianza_attiva"] is True
    c = wrec.guasto_catena({"adattatore": a, "isaac": {"esito": "guasto", "motivo": "x"}})
    assert c["sorveglianza_adattatore"] == "assente" and c["non_determinati"] == ["adattatore"]
    rec = wrec.guasto_catena_record(2, "pose_stale", c, seed=1001)
    assert rec["sorveglianza_adattatore"] == "assente" and rec["controlli"]["adattatore"]["sorveglianza_attiva"] is False
    # alone it does not declare a failure
    assert wrec.guasto_catena({"adattatore": a})["guasto"] is False


# --- (5) MOLA slowdown ---

SANI = {"isaac": {"esito": "ok"}, "nuvole_vittima": {"esito": "ok"}, "perturbation": {"esito": "ok"},
        "posa_vittima": {"esito": "ok"}, "adattatore": {"esito": "non_determinato"}}
LENTO = {"esito": "guasto", "motivo": "time_onLidar p99 250 ms oltre 200 ms (n 50)"}


def test_rallentamento_mola_regola():
    c = {**SANI, "mola_vittima": LENTO}
    g = wrec.guasto_catena(c)                                    # default: guasto
    assert g["guasto"] is True and g["rallentamento_mola"] is True and g["regola_rallentamento"] == "guasto"
    s = wrec.guasto_catena(c, "scoperto")
    assert s["guasto"] is False and s["rallentamento_mola"] is True and s["motivi"] == []
    assert s["segnali"] == ["mola_vittima: " + LENTO["motivo"]]
    # not the only signal: failure even with "scoperto"
    altro = wrec.guasto_catena({**c, "isaac": {"esito": "guasto", "motivo": "stallo"}}, "scoperto")
    assert altro["guasto"] is True and altro["rallentamento_mola"] is False
    # undetermined deliveries: the chain is not shown healthy, failure stays
    nd = wrec.guasto_catena({**c, "nuvole_vittima": {"esito": "non_determinato"}}, "scoperto")
    assert nd["guasto"] is True and nd["rallentamento_mola"] is False
    with pytest.raises(ValueError):
        wrec.guasto_catena(c, "altro")


def test_rallentamento_da_controlla_catena(tmp_path, monkeypatch):
    lento = RN._tracce([0.3] * 20)
    g = RN._catena(tmp_path, monkeypatch, tracce=lento)
    assert g["guasto"] is True and g["rallentamento_mola"] is True
    s = RN._catena(tmp_path, monkeypatch, tracce=lento, opz={"rallentamento_mola": "scoperto"})
    assert s["guasto"] is False and s["rallentamento_mola"] is True


def test_rallentamento_scoperto_chiude_come_attacco_scoperto(tmp_path):
    s = wrec.guasto_catena({**SANI, "mola_vittima": LENTO}, "scoperto")
    _, k, h = RN._chiusura(tmp_path, s)
    fine = h[2]["run_end"]
    assert fine["reason"] == "attacco_scoperto" and fine["rallentamento_mola"] is True
    assert not (tmp_path / "da_rifare.json").exists()
    g = wrec.guasto_catena({**SANI, "mola_vittima": LENTO})
    _, _, h = RN._chiusura(tmp_path, g)
    assert h[2]["run_end"]["reason"] == "guasto_catena" and h[2]["run_end"]["rallentamento_mola"] is True


# --- (4) repetition to redo ---

def test_spento_senza_controllo_catena(tmp_path):
    # --controllo-catena off: closing as without the rule, no check
    ts = _modulo("test_attacco_scoperto")
    rotta = wrec.guasto_catena({"isaac": {"esito": "guasto", "motivo": "x"}})
    _, _, h = ts._chiudi(tmp_path, True, ["ok", "ok", "pose_stale", "ok"], catena=rotta, controllo=False)
    fine = h[2]["run_end"]
    assert fine["reason"] == "attacco_scoperto" and "controllo_catena" not in fine
    assert not (tmp_path / "da_rifare.json").exists()


def test_guasto_scrive_da_rifare(tmp_path):
    rotta = wrec.guasto_catena({"isaac": {"esito": "guasto", "motivo": "1 stalli di Isaac nella finestra"}})
    _, _, h = RN._chiusura(tmp_path, rotta)
    m = json.loads((tmp_path / "da_rifare.json").read_text())
    assert m["reason"] == "guasto_catena" and m["seme"] == 1001 and m["window"] == 2
    assert h[2]["run_end"]["stamp_ultima_posa_ricevuta"] is None


def _finti(d, reg):
    """Fake python3, pgrep, ros2 and sleep; the fake orchestrator logs its arguments and,
    with _GUASTO=1, exits as a chain failure (code 2).
    """
    d.mkdir()
    orch = f"""import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]; os.makedirs(out, exist_ok=True)
open("{reg}", "a").write(json.dumps(a) + "\\n")
if os.environ.get("_GUASTO") == "1":
    fine = {{"reason": "guasto_catena", "window": 2, "seme": int(a[a.index("--seed") + 1])}}
    json.dump([{{"window": 0}}, {{"window": 1}}, {{"window": 2, "run_end": fine}}], open(out + "/history.json", "w"))
    json.dump(fine, open(out + "/da_rifare.json", "w"))
sys.exit(int(os.environ.get("_RC", "0")))
"""
    (d / "orch.py").write_text(orch)
    files = {"python3": f"""#!/bin/bash
case "$*" in
  *run_isaac_standalone.py*) echo "[standalone] pronto"; exec {SLEEP} 600 ;;
  *perturbation_node.py*) exec {SLEEP} 600 ;;
  *attack_orchestrator.py*) [[ "$1" == "-u" ]] && shift; shift; exec {PY} "{d / 'orch.py'}" "$@" ;;
esac
exec {PY} "$@"
""",
             "pgrep": '#!/bin/bash\ncase "$*" in *add_intensity*) echo 1; exit 0 ;; esac\nexit 1\n',
             "ros2": '#!/bin/bash\necho "{\\"min_latency_ms\\": ${_LAT_FINTA}}"\n',
             "sleep": "#!/bin/bash\nexit 0\n"}
    for nome, testo in files.items():
        (d / nome).write_text(testo)
        (d / nome).chmod((d / nome).stat().st_mode | stat.S_IEXEC)


def _campagna(tmp_path, env_extra, arm="nsga3"):
    reg = tmp_path / "argomenti.jsonl"
    if not (tmp_path / "bin").exists():
        _finti(tmp_path / "bin", reg)
    env = {k: v for k, v in os.environ.items() if k not in RN.TOLTE + ("MOLA_DEBUG", "MOLA_VERBOSITY_MOLA_LO",
                                                                      "RALLENTAMENTO_MOLA")}
    env.update({"PATH": f"{tmp_path / 'bin'}:{env['PATH']}", "CAMPAIGN_DIR": str(tmp_path / "camp")})
    env.update(env_extra)
    env["_LAT_FINTA"] = env.get("MIN_LATENCY_MS") or ("140" if env.get("SERIE_B") == "1" else "85")
    r = subprocess.run(["bash", str(ROOT / "scripts" / "run_campaign.sh"), "straight", arm, "3", "1", "4", "2"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    righe = [json.loads(x) for x in reg.read_text().splitlines()] if reg.exists() else []
    return r, righe


def test_campagna_guasto_catena_da_rifare(tmp_path):
    env = {"SERIE_B": "1", "SEEDS": "1001", "_GUASTO": "1", "_RC": "2"}
    r, righe = _campagna(tmp_path, env)
    assert r.returncode == 4, r.stdout[-3000:] + r.stderr[-3000:]
    assert "guasto della catena nella ripetizione 1 (seme 1001): da rifare" in r.stdout
    assert "--controllo-catena" in righe[-1]
    rep = Path(righe[-1][righe[-1].index("--out") + 1])
    assert (rep / "da_rifare.json").exists() and not (rep / "completata").exists()
    # relaunch: the repetition is redone with the same seed, the failed attempt stays renamed
    r2, righe = _campagna(tmp_path, {"SERIE_B": "1", "SEEDS": "1001"})
    assert r2.returncode == 0, r2.stdout[-3000:]
    assert "incompleta da un lancio precedente: rinominata e rifatta" in r2.stdout
    assert righe[-1][righe[-1].index("--seed") + 1] == "1001" and (rep / "completata").exists()
    vecchie = [p for p in rep.parent.glob("rep_01_incompleta_*") if p.is_dir()]
    assert len(vecchie) == 1 and (vecchie[0] / "da_rifare.json").exists()


def test_campagna_senza_serie_b_codici_come_prima(tmp_path):
    # rc 2 without marker (infrastructure error): the campaign exits with 0 as usual
    r, righe = _campagna(tmp_path, {"_RC": "2"})
    assert r.returncode == 0 and "orchestratore uscito con codice 2" in r.stdout
    assert "--controllo-catena" not in righe[-1] and "--rallentamento-mola" not in righe[-1]


@pytest.mark.parametrize("arm,valore,atteso", [("nsga3", "scoperto", True), ("random", "guasto", True),
                                                ("none", "scoperto", False), ("nsga3", "", False)])
def test_campagna_rallentamento_mola(tmp_path, arm, valore, atteso):
    r, righe = _campagna(tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "RALLENTAMENTO_MOLA": valore}, arm)
    assert r.returncode == 0, r.stdout[-3000:]
    a = righe[-1]
    assert (("--rallentamento-mola" in a) and a[a.index("--rallentamento-mola") + 1] == valore) == atteso
    assert ("--controllo-catena" in a) == (arm != "none")


def test_campagna_rallentamento_mola_non_valido(tmp_path):
    r, _ = _campagna(tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "RALLENTAMENTO_MOLA": "forse"})
    assert r.returncode == 1 and "RALLENTAMENTO_MOLA sconosciuto" in r.stdout


CODA = Path(os.environ.get("CODA_SERIEB_STATO", ROOT / "scripts" / "coda_serieB_stato.py"))


@pytest.mark.skipif(not CODA.exists(), reason="coda_serieB_stato.py assente")
def test_coda_riconosce_guasto_catena(tmp_path):
    # aligned name: the queue RUN_END_GUASTO (default guasto_catena) on run_end.reason
    spec = importlib.util.spec_from_file_location("_coda", CODA)
    coda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coda)
    r, righe = _campagna(tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "_GUASTO": "1", "_RC": "2"})
    rep = righe[-1][righe[-1].index("--out") + 1]
    s = coda.storia(rep)
    assert s["guasto_catena"] is True and s["run_end"] == ["guasto_catena"]
    e = coda.classifica({"storia": s, "rc": r.returncode, "completata": False})
    assert e["esito"] == "guasto_catena" and "guasto_catena" in coda.ESITI_RIPETIBILI
    # slowdown with "scoperto": attack detected, run done
    s2 = {"history": True, "run_end": ["attacco_scoperto"], "guasto_catena": False, "attacco_scoperto": True}
    assert coda.classifica({"storia": s2, "rc": 0, "completata": True, "adattatore_vivo": True,
                            "ritardo_rc": 0, "ombra": {"esito": "identica"}})["esito"] == "fatta"
    # without the shadow check the run is not valid
    e3 = coda.classifica({"storia": s2, "rc": 0, "completata": True, "adattatore_vivo": True, "ritardo_rc": 0})
    assert e3["esito"] == "guasto_catena" and "ombra non misurata" in e3["motivo"]
