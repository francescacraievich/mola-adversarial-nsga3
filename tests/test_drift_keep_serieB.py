"""Drift (gene 14) kept across meters with SERIE_B=1, and marking of the first ICP of the
meter.

run_campaign.sh: with SERIE_B=1 in the random/nsga3/fixed arms and DRIFT_RESET_ON unset,
keep applies (folder _rkeep); an explicit value is respected; none and gaussian arms are
unchanged; without SERIE_B identical to the original copy. Real node (tests/zone_nodo.py,
deriva_* modes, environment ROS_DOMAIN_ID, no spin): with keep every twin starts from the
victim drift at the start of the meter and the winner produces the same clouds as the
victim; without keep both restart from zero. First ICP: window_record.prima_icp /
prima_icp_gemello, orchestrator history and queue run summary, against the values of a
recorded diagnosis (divergenza_stime_w2_w3, read only).

    ROS_DOMAIN_ID=88 python3 -m pytest tests/test_drift_keep_serieB.py -q
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.optimization import window_record as wrec  # noqa: E402

QUI = Path(__file__).resolve().parent
ORIGINALE = Path(os.environ.get("DRIFT_KEEP_ORIGINALE",
                                os.path.expanduser("~/mola_lavoro/lavoro_drift_keep_originale")))
NPZ = Path(os.environ.get("VERSIONEI_NUVOLE",
                          os.path.expanduser("~/mola_lavoro/riferimento_serieA_reale"))) / "svil1.npz"
REPO = Path(os.path.expanduser("~/mola-adversarial-nsga3"))
PROVA = REPO / "data/attack/verifica/2026-10-27_prova_breve_zone_attaccante"
DIAG = REPO / "data/attack/dossier/2026-10-23_serieB/divergenza_stime_w2_w3/controllore_e_prima_icp.tsv"
ARMS = ("none", "gaussian", "random", "nsga3", "fixed")


# ----------------------------------------------------------------------
# run_campaign.sh
# ----------------------------------------------------------------------

def _blocco(tmp_path, env_extra, arm, radice=ROOT):
    src = (radice / "scripts" / "run_campaign.sh").read_text().splitlines()
    fine = next(i for i, l in enumerate(src) if l.startswith("if pgrep -f"))
    corpo = "\n".join(src[:fine]).replace('cd "$ROOT"', ": cd")
    corpo += ('\necho "FLAGS=$EXTRA $SERIEB_FLAGS"\necho "NODO=$OPERATOR_NODE_FLAGS"\n'
              'echo "DRIFT=$DRIFT_RESET_ON"\necho "SUFFISSO=$MODE_SUFFIX"\n')
    f = tmp_path / f"blocco_{radice.name}_{arm}.sh"
    f.write_text(corpo)
    tolte = ("SERIE_B", "OPERATOR_MODE", "ZONE_ATTACCANTE", "SEEDS", "ORCH_EXTRA", "NODE_EXTRA",
             "CHAMFER_BUDGET", "MAX_DRIFT_LAG_S", "RUN_SUFFIX", "SELECT", "MIN_LATENCY_MS", "FAST_OPS",
             "REPORT_MATCHED", "SYNC_WAIT", "SYNC_GUARD", "WINDOW_RNG", "PERC_TERMS", "ICP_QUALITY_LOG",
             "DRIFT_RESET_ON", "ATTACK_FROM")
    env = {k: v for k, v in os.environ.items() if k not in tolte}
    env.update(env_extra)
    r = subprocess.run(["bash", str(f), "straight", arm, "3", "1", "4", "2"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)
    o = dict(l.split("=", 1) for l in r.stdout.splitlines()
             if "=" in l and l.split("=", 1)[0] in ("FLAGS", "NODO", "DRIFT", "SUFFISSO"))
    o["rc"], o["stdout"] = r.returncode, r.stdout
    return o


@pytest.mark.parametrize("arm", ARMS)
def test_run_campaign_serieB_keep(tmp_path, arm):
    o = _blocco(tmp_path, {"SERIE_B": "1", "SEEDS": "1001"}, arm)
    assert o["rc"] == 0, o["stdout"]
    if arm in ("random", "nsga3", "fixed"):
        assert o["DRIFT"] == "keep" and "_rkeep" in o["SUFFISSO"]
    else:
        assert o["DRIFT"] == "genome" and "_rkeep" not in o["SUFFISSO"]
    # explicit value respected
    for v in ("genome", "drive", "keep"):
        o = _blocco(tmp_path, {"SERIE_B": "1", "SEEDS": "1001", "DRIFT_RESET_ON": v}, arm)
        assert o["rc"] == 0 and o["DRIFT"] == v
        assert ("_rkeep" in o["SUFFISSO"]) == (v == "keep")
        assert ("_rdrive" in o["SUFFISSO"]) == (v == "drive")


def _togli_incompleti(a, b, env, arm):
    """--incompleti-regola: once in the attack arms of the copy with SERIE_B=1; removed from
    both sides before the comparison (an older reference lacks it, a later one has it)."""
    n = a["FLAGS"].split().count("--incompleti-regola")
    assert n == (1 if env.get("SERIE_B") == "1" and arm != "none" else 0), a["FLAGS"]
    for o in (a, b):
        o["FLAGS"] = " ".join(x for x in o["FLAGS"].split(" ") if x != "--incompleti-regola")


@pytest.mark.skipif(not (ORIGINALE / "scripts" / "run_campaign.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("arm", ARMS)
def test_run_campaign_rispetto_all_originale(tmp_path, arm):
    """Without SERIE_B everything identical; with SERIE_B=1 only DRIFT_RESET_ON (keep) and
    the suffix (_rkeep) change in the generator arms."""
    for env in ({}, {"DRIFT_RESET_ON": "genome"}, {"DRIFT_RESET_ON": "drive"}, {"DRIFT_RESET_ON": "keep"},
                {"CHAMFER_BUDGET": "70"}, {"SERIE_B": "1", "SEEDS": "1001", "DRIFT_RESET_ON": "genome"}):
        a, b = _blocco(tmp_path, env, arm), _blocco(tmp_path, env, arm, ORIGINALE)
        assert a["rc"] == b["rc"] == 0
        _togli_incompleti(a, b, env, arm)
        assert {k: a[k] for k in ("FLAGS", "NODO", "DRIFT", "SUFFISSO")} == \
               {k: b[k] for k in ("FLAGS", "NODO", "DRIFT", "SUFFISSO")}, (env, a["stdout"], b["stdout"])
    env = {"SERIE_B": "1", "SEEDS": "1001"}
    a, b = _blocco(tmp_path, env, arm), _blocco(tmp_path, env, arm, ORIGINALE)
    _togli_incompleti(a, b, env, arm)
    assert a["FLAGS"] == b["FLAGS"] and a["NODO"] == b["NODO"]
    if arm in ("random", "nsga3", "fixed"):
        assert b["DRIFT"] == "genome" and a["DRIFT"] == "keep"
        assert a["SUFFISSO"] == b["SUFFISSO"].replace("_sel", "_rkeep_sel", 1) if "_sel" in b["SUFFISSO"] \
            else a["SUFFISSO"].replace("_rkeep", "", 1) == b["SUFFISSO"]
    else:
        assert a["DRIFT"] == b["DRIFT"] == "genome" and a["SUFFISSO"] == b["SUFFISSO"]


def test_run_campaign_passa_la_deriva_all_orchestratore():
    s = (ROOT / "scripts/run_campaign.sh").read_text()
    assert s.count('--drift-reset-on "$DRIFT_RESET_ON"') == 1
    assert s.count('--drift-reset-on \\"$DRIFT_RESET_ON\\"') == 1


# ----------------------------------------------------------------------
# Real node: victim, twin (and shadow) drift at the start of the meter
# ----------------------------------------------------------------------

def _genomi(path):
    rng = np.random.default_rng(20261027)

    def g(d):
        v = rng.uniform(-1, 1, 17)
        v[14] = d
        return v.tolist()
    # first-meter victim with full drift; candidates: full, half, off drift (winner)
    gg = [{"id": "vittima0", "genoma": g(1.0)}, {"id": "c_deriva", "genoma": g(1.0)},
          {"id": "c_media", "genoma": g(0.0)}, {"id": "vincitore", "genoma": g(-1.0)}]
    path.write_text(json.dumps(gg))
    return path


def _nodo_deriva(tmp_path, radice, modo, zone=True, metri="2,3,4", n=6):
    args = ["--max-drift-lag-s", "0.5", "--perc-terms", "--structural", "gross", "--operator-mode", "insieme"]
    if zone:
        args.append("--zone-attaccante")
    nome = f"{radice.name}_{modo}_{int(zone)}"
    out = tmp_path / f"{nome}.json"
    cmd = [sys.executable, str(QUI / "zone_nodo.py"), str(radice), str(NPZ), str(tmp_path / nome), str(out),
           modo, str(_genomi(tmp_path / "genomi.json")), metri, str(n), "--"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())["deriva"]


nodo_vero = pytest.mark.skipif(not NPZ.exists(), reason="nuvole svil1 assenti")


@nodo_vero
@pytest.mark.parametrize("zone", [True, False])
def test_nodo_keep_stessa_deriva_a_inizio_metro(tmp_path, zone):
    d = _nodo_deriva(tmp_path, ROOT, "deriva_keep", zone)
    assert np.linalg.norm(d[0]["vittima"]["fine"]) > 0.2          # drift accumulated in meter 1
    for prec, m in zip(d[:-1], d[1:]):
        D = prec["vittima"]["fine"]
        assert m["vittima_fine_prec"] == D
        # the victim resumes where it was (no reset)
        assert m["vittima"]["inizio"] == D
        # each candidate starts from the victim drift, not from the previous candidate's
        for k, g in enumerate(m["gemelli"]):
            assert g["inizio"] == D, (m["metro"], g["id"])
            if k:
                assert m["gemelli"][k - 1]["fine"] != D
        # winner and victim: same clouds, bit for bit (the shadow receives the same array as
        # the victim, perturbation_node.py:1355), same drift at the end of the meter
        assert m["vittima"]["uscite"] == m["gemelli"][-1]["uscite"]
        assert m["vittima"]["fine"] == m["gemelli"][-1]["fine"]
        # winner with gene 14 off: the kept drift decays by 0.98 per cloud
        assert np.allclose(m["vittima"]["fine"], np.asarray(D) * 0.98 ** len(m["vittima"]["uscite"]))


@nodo_vero
def test_nodo_senza_keep_azzera(tmp_path):
    d = _nodo_deriva(tmp_path, ROOT, "deriva_genome")
    assert np.linalg.norm(d[0]["vittima"]["fine"]) > 0.2
    for m in d[1:]:
        assert m["vittima"]["inizio"] == [0.0, 0.0, 0.0]
        assert all(g["inizio"] == [0.0, 0.0, 0.0] for g in m["gemelli"])
        assert m["vittima"]["uscite"] == m["gemelli"][-1]["uscite"]


@nodo_vero
@pytest.mark.skipif(not (ORIGINALE / "src/nodes/perturbation_node.py").exists(), reason="copia originale assente")
@pytest.mark.parametrize("modo", ["deriva_genome", "deriva_keep"])
def test_nodo_identico_all_originale(tmp_path, modo):
    assert _nodo_deriva(tmp_path, ROOT, modo) == _nodo_deriva(tmp_path, ORIGINALE, modo)


# ----------------------------------------------------------------------
# First ICP of the meter
# ----------------------------------------------------------------------

def _r(stamp, q, it=5, term="Stalled"):
    return {"stamp_s": stamp, "goodness": q, "iters": it, "term_reason": term, "is_good": 1}


def test_prima_icp_soglia_e_campi_mancanti():
    rr = [_r(10.0, 0.9), _r(10.1, 0.34), _r(10.2, 0.341)]
    assert wrec.PRIMA_ICP_Q_DEBOLE == 0.34
    p = wrec.prima_icp(rr, 10.1)
    assert p["q"] == 0.34 and p["prima_icp_debole"] is True and p["stamp"] == 10.1 and p["soglia_q"] == 0.34
    assert wrec.prima_icp(rr, 10.2)["prima_icp_debole"] is False
    assert wrec.prima_icp(rr, 10.2 + 0.04)["q"] == 0.341          # within half a period
    for st, rec in ((10.5, rr), (None, rr), (10.1, [])):
        p = wrec.prima_icp(rec, st)
        assert p["q"] is None and p["prima_icp_debole"] is None
    g = wrec.prima_icp_gemello({"twin_icp_first_q": 0.318, "twin_icp_first_stamp": 28.5,
                                "twin_icp_first_iters": 10, "twin_icp_first_term": "Stalled"})
    assert g == {"stamp": 28.5, "q": 0.318, "iters": 10, "term": "Stalled", "prima_icp_debole": True,
                 "soglia_q": 0.34}
    assert wrec.prima_icp_gemello(None)["prima_icp_debole"] is None
    assert wrec.prima_icp_gemello({"twin_icp_first_q": 0.45})["prima_icp_debole"] is False


def _run(seme):
    rr = sorted((PROVA / seme / "campagna/straight").glob("nsga3*/rep_01"))
    return rr[0] if rr else None


dati_veri = pytest.mark.skipif(not (PROVA.exists() and DIAG.exists()), reason="prova breve o diagnosi assenti")


def _diagnosi():
    out = {}
    with open(DIAG) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["q1_v"] != "None":
                out[(r["seme"], int(r["w"]))] = (float(r["q1_v"]), float(r["q1_g"]))
    return out


@dati_veri
def test_riepilogo_prima_icp_come_la_diagnosi():
    import coda_serieB_stato as cs
    diag = _diagnosi()
    assert len(diag) == 8
    segnati = set()
    for seme in ("s1001", "s2001"):
        rep = _run(seme)
        hist = json.loads((rep / "history.json").read_text())
        pi, err = cs.prima_icp_finestre(str(rep), hist)
        assert err is None and len(pi) == 5
        for p in pi:
            k = (seme, p["finestra"])
            if k in diag:
                # printf with 3 decimals, diagnosis CSV with 4
                assert (p["vittima"]["q"], p["vincitore"]["q"]) == pytest.approx(diag[k], abs=5e-4)
            if p["vittima"]["prima_icp_debole"] or p["vincitore"]["prima_icp_debole"]:
                segnati.add(k)
        d = cs.riepilogo_run(str(PROVA / seme), usa_bag=False)
        assert d["prima_icp"] == pi
        assert "prima ICP del metro" in cs.testo_riepilogo_run(d)
    # meters with q <= 0.34 in the diagnosis: s1001 w1 (twin), w2, s2001 w3
    assert segnati == {("s1001", 1), ("s1001", 2), ("s2001", 3)}


@dati_veri
def test_history_finestra_vittima_e_ricalcolo():
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    rep = _run("s2001")
    victim = SimpleNamespace(log_path=rep / "logs/mola_victim.log", traces_path=rep / "logs/traces_victim.csv")
    node = SimpleNamespace(last_trace=None, last_drive_stamps=(28.5, 34.0), last_victim_open_stamp=28.5)
    rec = ao._victim_window_record(node, victim, None)
    assert "error" not in rec
    assert rec["prima_icp"]["q"] == pytest.approx(0.338) and rec["prima_icp"]["prima_icp_debole"] is True
    # window recorded before the log arrived: q is filled on recompute
    vw = {"open_stamp": 28.5, "drive_stamps": [28.5, 34.0], "scans_drive": {"log_complete": False},
          "prima_icp": wrec.prima_icp([], 28.5)}
    ao._refresh_victim_scans([{"victim_window": vw}], victim)
    assert vw["prima_icp"]["q"] == pytest.approx(0.338)


def test_orchestratore_vincitore_nel_danno_metro():
    s = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert '"prima_icp_vincitore": wrec.prima_icp_gemello(e_win),' in s
    assert 'rec["prima_icp"] = wrec.prima_icp(rows, rec["open_stamp"])' in s
