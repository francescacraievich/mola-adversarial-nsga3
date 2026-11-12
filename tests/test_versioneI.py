#!/usr/bin/env python3
"""Operator version I (node --operator-mode insieme, orchestrator --metro-al-nodo,
OPERATOR_MODE=insieme in run_campaign.sh). No Isaac or MOLA; the tests on real clouds read
the clouds extracted from recordings svil1 and svil2 read-only and are skipped if missing.

    ROS_DOMAIN_ID=88 python3 -m pytest tests/test_versioneI.py -q

The full measurements (14 genomes, 6 metres, two recordings) use the same helper scripts:
versioneI_nodo.py and versioneI_punti.py.
"""

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

pn = pytest.importorskip("src.nodes.perturbation_node")
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402

QUI = Path(__file__).resolve().parent
NPZ = Path(os.environ.get("VERSIONEI_NUVOLE",
                          os.path.expanduser("~/mola_lavoro/riferimento_serieA_reale")))
# reference root without the change (optional): untouched copy of the repo
ORIGINALE = Path(os.environ.get("VERSIONEI_ORIGINALE", str(ROOT.parent / "originale")))
KW = dict(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15, max_drift_lag_s=0.5)
INS = dict(operator_mode="coherent", struct_track="scan", struct_rule="caso",
           struct_g10_pavimento=True, struct_ripiego=True, struct_g13_punteggio=True,
           struct_insieme=True)
RNG = np.random.default_rng(20261027)
GENOMI = [{"id": "uno", "genoma": [0.5] * 16 + [-1.0]},
          {"id": "casuale", "genoma": RNG.uniform(-1, 1, 17).tolist()},
          {"id": "strategico", "genoma": [0.3] * 16 + [1.0]}]
NODE_KW = ["--max-drift-lag-s", "0.5", "--perc-terms"]

nuvole_vere = pytest.mark.skipif(not (NPZ / "svil1.npz").exists(),
                                 reason="nuvole delle registrazioni svil1/svil2 assenti")


def _args(**kw):
    a = dict(max_point_shift=0.05, noise_std=0.02, dropout_rate=0.15, max_drift_lag_s=0.5,
             structural="net", fast_ops=False, operator_mode="random")
    a.update(kw)
    return SimpleNamespace(**a)


def _nuvole(reg="svil1", metro=0, n=4):
    z = np.load(NPZ / f"{reg}.npz")
    order = np.argsort(z["t_arr_mono"], kind="stable")
    fr = z["finestra_registro"][order]
    w = np.unique(fr[fr >= 0])[metro]
    idx = order[fr == w][:n]
    return [(z["xyzi"][z["offsets"][i]:z["offsets"][i + 1]].astype(np.float32).astype(np.float64),
             float(z["stamps"][i])) for i in idx]


def _uguali(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _nodo(tmp_path, nome, radice, flusso, genomi, metri, n, node_args, scarto=0):
    gj = tmp_path / f"{nome}_genomi.json"
    gj.write_text(json.dumps(genomi))
    out = tmp_path / f"{nome}.json"
    cmd = [sys.executable, str(QUI / "versioneI_nodo.py"), str(radice), str(NPZ / "svil1.npz"),
           str(out), flusso, str(gj), metri, str(n)]
    if scarto:
        cmd += ["--metro-scarto", str(scarto)]
    return subprocess.Popen(cmd + ["--"] + node_args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL), out


# ----------------------------------------------------------------------
# Option off
# ----------------------------------------------------------------------

def test_default_random():
    g = pn.make_generator(_args())
    assert g.operator_mode == "random" and g.struct_insieme is False and g._structs is None
    assert pn.make_generator(_args(operator_mode="insieme")).struct_insieme is True


@nuvole_vere
def test_make_generator_come_costruzione_storica():
    """Option off: generator output matches the explicit construction, with or without fast_ops."""
    for fo in (False, True):
        a = pn.make_generator(_args(fast_ops=fo))
        b = PerturbationGenerator(structural="net", fast_ops=fo, **KW)
        g = np.asarray(GENOMI[1]["genoma"])
        pa, pb = a.encode_perturbation(g), b.encode_perturbation(g)
        for k, (c, _) in enumerate(_nuvole(n=3)):
            assert _uguali(a.apply_perturbation(c, pa, seed=100 + k),
                           b.apply_perturbation(c, pb, seed=100 + k))


@nuvole_vere
@pytest.mark.skipif(not (ORIGINALE / "src/nodes/perturbation_node.py").exists(),
                    reason="radice originale assente")
@pytest.mark.parametrize("extra", [[], ["--fast-ops"]])
def test_nodo_spento_uguale_originale(tmp_path, extra):
    """Twin and victim outputs bit-identical to the unmodified node, in separate processes."""
    pa, oa = _nodo(tmp_path, "orig", ORIGINALE, "entrambi", GENOMI, "0,5", 4, NODE_KW + extra)
    pb, ob = _nodo(tmp_path, "copia", ROOT, "entrambi", GENOMI, "0,5", 4, NODE_KW + extra)
    assert pa.wait() == 0 and pb.wait() == 0
    ra, rb = json.loads(oa.read_text())["righe"], json.loads(ob.read_text())["righe"]
    assert len(ra) == len(rb) == len(GENOMI) * 2 * 4 * 2
    for x, y in zip(ra, rb):
        assert (x["md5_main"], x["md5_victim"]) == (y["md5_main"], y["md5_victim"]), (x, y)


# ----------------------------------------------------------------------
# Version I
# ----------------------------------------------------------------------

@nuvole_vere
def test_stessa_zona_vittima_gemello_due_processi(tmp_path):
    """Same seed and metre: same zones and outputs for victim flow and twin, in two processes."""
    args = ["--operator-mode", "insieme", "--victim-never-clean"] + NODE_KW
    pv, ov = _nodo(tmp_path, "vittima", ROOT, "vittima", GENOMI, "0,3", 5, args)
    pg, og = _nodo(tmp_path, "gemello", ROOT, "gemello", GENOMI, "0,3", 5, args)
    ps, os_ = _nodo(tmp_path, "scarto", ROOT, "gemello", GENOMI, "0,3", 1, args, scarto=1)
    assert pv.wait() == 0 and pg.wait() == 0 and ps.wait() == 0
    rv, rg = json.loads(ov.read_text())["righe"], json.loads(og.read_text())["righe"]
    assert len(rv) == len(rg) == len(GENOMI) * 2 * 5
    for x, y in zip(rv, rg):
        assert x["per_op"] is True and y["per_op"] is True
        assert x["anchors"] == y["anchors"] and x["uscita"] == y["uscita"], (x, y)
        assert x["metro"] == y["metro"] and x["k"] == y["k"]
    # zones fixed within the metre
    for gid in {r["genoma"] for r in rv}:
        for w in (0, 3):
            an = {r["anchors"] for r in rv if r["genoma"] == gid and r["metro"] == w}
            assert len(an) == 1
    # another metre (same clouds) changes the choice in at least one case
    rs = json.loads(os_.read_text())["righe"]
    prime = {(r["genoma"], r["metro"]): r["anchors"] for r in rg if r["k"] == 0}
    assert any(prime[(r["genoma"], r["metro"])] != r["anchors"] for r in rs)


@nuvole_vere
def test_seme_diverso_cambia_insieme():
    c = _nuvole(n=1)[0]
    g = np.asarray(GENOMI[0]["genoma"])
    scelte = []
    for seme in (11, 12):
        gen = PerturbationGenerator(**INS, **KW)
        gen.apply_perturbation(c[0], gen.encode_perturbation(g), ctx={"stamp": c[1]},
                               seme_genoma=seme, metro=0)
        scelte.append(gen.last_struct_info["anchors"])
    assert scelte[0] != scelte[1]


@nuvole_vere
def test_punti_toccati_non_oltre_serieA(tmp_path):
    out = tmp_path / "punti.json"
    gj = tmp_path / "g.json"
    gj.write_text(json.dumps(GENOMI))
    r = subprocess.run([sys.executable, str(QUI / "versioneI_punti.py"), str(ROOT),
                        str(NPZ / "svil1.npz"), str(out), str(gj), "0", "5"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert r.returncode == 0
    righe = json.loads(out.read_text())["righe"]
    gruppi = {}
    for x in righe:
        gruppi.setdefault((x["genoma"], x["op"]), []).append(x)
    assert len(gruppi) >= 4
    for key, rr in gruppi.items():
        a = [x["A"] for x in rr]
        i = [x["I"] for x in rr]
        assert all(x["per_op"] for x in rr)
        assert max(i) <= max(a) and np.median(i) <= np.median(a), key


@nuvole_vere
def test_insieme_fast_ops_identico():
    """--fast-ops does not affect the coherent mode: same outputs with and without."""
    g = np.asarray(GENOMI[1]["genoma"])
    a = PerturbationGenerator(**INS, **KW)
    b = PerturbationGenerator(**INS, fast_ops=True, **KW)
    pa, pb = a.encode_perturbation(g), b.encode_perturbation(g)
    for c, s in _nuvole(n=3):
        assert _uguali(a.apply_perturbation(c, pa, ctx={"stamp": s}, seme_genoma=5, metro=2),
                       b.apply_perturbation(c, pb, ctx={"stamp": s}, seme_genoma=5, metro=2))


@nuvole_vere
def test_stato_globale_numpy_intatto():
    g = np.asarray(GENOMI[0]["genoma"])
    gen = PerturbationGenerator(**INS, **KW)
    p = gen.encode_perturbation(g)
    np.random.seed(123)
    s0 = np.random.get_state()[1].copy()
    for c, s in _nuvole(n=2):
        gen.apply_perturbation(c, p, ctx={"stamp": s}, seme_genoma=5, metro=1)
    assert np.array_equal(np.random.get_state()[1], s0)


@nuvole_vere
def test_conteggi_gross_nel_modo_coerente():
    g = np.asarray(GENOMI[2]["genoma"])
    gen = PerturbationGenerator(**INS, structural="gross", **KW)
    p = gen.encode_perturbation(g)
    c, s = _nuvole(n=1)[0]
    out = gen.apply_perturbation(c, p, ctx={"stamp": s}, seme_genoma=5, metro=0)
    cnt = gen.point_counts(c, out)
    assert cnt is not None
    assert cnt["n_removed"] == gen.last_struct_info["n_hit"].get("g5", 0)
    assert cnt["n_added"] == gen.last_struct_info["n_hit"].get("g16", 0)


def test_metro_senza_metro_vale_R1():
    """Without a metre the 'caso' rule does not apply (per_op false)."""
    rng = np.random.default_rng(0)
    c = np.c_[rng.uniform(-10, 10, (5000, 3)), rng.uniform(0, 255, 5000)]
    gen = PerturbationGenerator(**INS, **KW)
    gen.apply_perturbation(c, gen.encode_perturbation(np.full(17, 0.5)), ctx={"stamp": 1.0},
                           seme_genoma=5, metro=None)
    assert gen.last_struct_info["per_op"] is False


# ----------------------------------------------------------------------
# Node and orchestrator: metre command
# ----------------------------------------------------------------------

def test_metro_cb():
    warn = []
    n = SimpleNamespace(_metro={"gemello": None, "vittima": None},
                        get_logger=lambda: SimpleNamespace(warn=warn.append))
    pn.PerturbationNode._metro_cb(n, SimpleNamespace(data='{"gemello": 3}'))
    assert n._metro == {"gemello": 3, "vittima": None}
    pn.PerturbationNode._metro_cb(n, SimpleNamespace(data='{"vittima": 4}'))
    assert n._metro == {"gemello": 3, "vittima": 4}
    pn.PerturbationNode._metro_cb(n, SimpleNamespace(data="non json"))
    assert n._metro == {"gemello": 3, "vittima": 4} and len(warn) == 1


def test_orchestratore_set_metro():
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    pubblicati = []
    n = SimpleNamespace(node_metro=None)

    def confermato(pub, msg, ok, what, **kw):
        pubblicati.append(json.loads(msg.data))
        n.node_metro = {"gemello": None, "vittima": None, **pubblicati[-1]}
        return ok()

    n._publish_confirmed = confermato
    n._metro_pub = object()
    ao.METRO["on"] = False
    assert ao.AttackNode.set_metro(n, "gemello") is None and not pubblicati
    try:
        ao.METRO["on"] = True
        ao.TWIN_STATE["window"] = 2
        assert ao.AttackNode.set_metro(n, "vittima") is True
        assert pubblicati == [{"vittima": 2}]
        ao.TWIN_STATE["window"] = -1
        assert ao.AttackNode.set_metro(n, "gemello") is True
        assert pubblicati[-1] == {"gemello": 0}
        n._publish_confirmed = lambda *a, **k: False
        assert ao.AttackNode.set_metro(n, "gemello") is False
        assert ao.METRO["non_confermati"][-1] == {"flusso": "gemello", "metro": 0}
    finally:
        ao.METRO.update({"on": False, "non_confermati": []})
        ao.TWIN_STATE["window"] = None


def test_orchestratore_status_metro():
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    n = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0)
    ao.AttackNode._status_cb(n, SimpleNamespace(data=json.dumps(
        {"enabled": True, "victim_stream": False, "metro_gemello": 1, "metro_vittima": None})))
    assert n.node_metro == {"gemello": 1, "vittima": None}
    m = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0)
    ao.AttackNode._status_cb(m, SimpleNamespace(data=json.dumps({"enabled": True, "victim_stream": False})))
    assert not hasattr(m, "node_metro")


def test_run_campaign_operator_mode(tmp_path):
    """OPERATOR_MODE: insieme by default with SERIE_B=1 on random/nsga3/fixed arms only."""
    trs = pytest.importorskip("test_registro_serieB")
    o = trs._blocco_campagna(tmp_path, {}, arm="nsga3")
    assert o["rc"] == 0 and "--metro-al-nodo" not in o["FLAGS"] and "_op" not in o["DIR"]
    for arm in ("nsga3", "random", "fixed"):
        o = trs._blocco_campagna(tmp_path, {"SERIE_B": "1"}, arm=arm)
        assert o["rc"] == 0 and o["DIR"].endswith("_opinsieme"), (arm, o["DIR"])
        assert o["FLAGS"].count("--metro-al-nodo") == 1
    for arm in ("none", "gaussian"):
        o = trs._blocco_campagna(tmp_path, {"SERIE_B": "1"}, arm=arm)
        assert o["rc"] == 0 and o["DIR"].endswith("_oprandom") and "--metro-al-nodo" not in o["FLAGS"]
        o = trs._blocco_campagna(tmp_path, {"SERIE_B": "1", "OPERATOR_MODE": "insieme"}, arm=arm)
        assert o["rc"] == 1
    # with SERIE_B=1 on attack arms the run does not start with operators other
    # than insieme
    o = trs._blocco_campagna(tmp_path, {"SERIE_B": "1", "OPERATOR_MODE": "random"}, arm="nsga3")
    assert o["rc"] == 1 and "OPERATOR_MODE deve essere insieme" in o["stdout"]
    o = trs._blocco_campagna(tmp_path, {"OPERATOR_MODE": "insieme"}, arm="nsga3")
    assert o["rc"] == 0 and "--metro-al-nodo" in o["FLAGS"]
    o = trs._blocco_campagna(tmp_path, {"OPERATOR_MODE": "coherent"}, arm="nsga3")
    assert o["rc"] == 1
    s = (ROOT / "scripts/run_campaign.sh").read_text()
    assert 'extra="$extra$OPERATOR_NODE_FLAGS"' in s


@pytest.mark.skipif(not (ORIGINALE / "scripts" / "run_campaign.sh").exists(), reason="radice originale assente")
@pytest.mark.parametrize("arm,atteso", [("nsga3", True), ("random", True), ("gaussian", False), ("none", False)])
def test_campagna_rispetto_all_originale(tmp_path, arm, atteso):
    """Orchestrator options: those of the unmodified repo plus --metro-al-nodo with SERIE_B=1."""
    tas = pytest.importorskip("test_attacco_scoperto")
    for env in ({}, {"SERIE_B": "1", "SEEDS": "1001"}):
        nuovo = tas._flags_campagna(ROOT, tmp_path, env, arm).split()
        vecchio = tas._flags_campagna(ORIGINALE, tmp_path, env, arm).split()
        aggiunte = ["--metro-al-nodo"] if (env and atteso) else []
        # removed on both sides, together with --incompleti-regola (SERIE_B=1 on attack arms):
        # the reference can be the copy before the change or a later one that already has them
        tolte = ("--metro-al-nodo", "--incompleti-regola")
        assert [x for x in nuovo if x not in tolte] == [x for x in vecchio if x not in tolte]
        assert [x for x in nuovo if x == "--metro-al-nodo"] == aggiunte
        assert nuovo.count("--incompleti-regola") == (1 if (env and arm != "none") else 0)
