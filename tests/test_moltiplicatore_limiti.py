#!/usr/bin/env python3
"""
Limit study: per-family limit multiplier (--moltiplicatore-limiti, --famiglie-limiti of node
and orchestrator, MOLTIPLICATORE_LIMITI and FAMIGLIE_LIMITI of the queue), gene 14 blocked in
the search (--blocca-gene 14=-1, BLOCCA_G14) and victim crash rule (REGOLA_CRASH_LIMITI,
classifica --regola-crash). All off by default.

    ROS_DOMAIN_ID=88 python3 -m pytest tests/test_moltiplicatore_limiti.py -q

Real node in a separate process (tests/limiti_nodo.py, domain 88, no spin) on the recorded
clean clouds svil1 (read only); tests are skipped if missing. Reference for x1: output
fingerprints of the code before the multiplier, and a direct comparison with that copy if
present (LIMITI_ORIGINALE). Queue dry runs: RADICE inside tmp_path, never created.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.perturbations import perturbation_generator as pg  # noqa: E402
import coda_serieB_stato as cs  # noqa: E402
from src.perturbations.perturbation_generator import (  # noqa: E402
    FAMIGLIE_LIMITI, PerturbationGenerator, moltiplicatori_da_opzioni)

QUI = Path(__file__).resolve().parent
NPZ = Path(os.environ.get("VERSIONEI_NUVOLE",
                          os.path.expanduser("~/mola_lavoro/riferimento_serieA_reale"))) / "svil1.npz"
ORIGINALE = Path(os.environ.get("LIMITI_ORIGINALE",
                                os.path.expanduser("~/mola_lavoro/fase1_limiti/repo_originale")))
AMBIENTE_CODA = Path(os.path.expanduser("~/mola_lavoro/coda_05nov_genomanullo/ambiente_coda.env"))
nuvole_vere = pytest.mark.skipif(not NPZ.exists(), reason="nuvole svil1 assenti")

# Genomes applied with seed 1001 (history.json, genome field of the window): massimo w0,
# nsga3_eps5 w2, casuale_senzabudget w4.
GENOMI_S1001 = [
    {"id": "massimo_w0", "genoma": [
        0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    {"id": "nsga3_eps5_w2", "genoma": [
        0.03384720349616552, -0.9130946801540737, -0.029331881003834263, -0.6715105372192895,
        -0.8063236070038831, -0.822178946558384, -0.4119718945609446, 0.2635189247539448, -0.748915113074247,
        -0.036700355794072305, 0.8491975817378428, -0.41175694276013647, -0.9943000599521141,
        -0.864280317962095, -0.8662979493269731, -0.5973339184903654, -0.8243997549930512]},
    {"id": "casuale_senzabudget_w4", "genoma": [
        0.9775170653795289, -0.9055283070864251, -0.11390256311636215, 0.9314455328073217, 0.451084117639063,
        -0.32154436190099345, 0.7560604947311893, -0.3912214310934332, -0.3887983994005981,
        -0.8027896906225214, 0.7097750462861707, -0.7937921146104105, -0.814488553969251, -0.13315641680246926,
        -0.5985812497951513, -0.2691322145122752, 0.008859595889390004]},
]
# Node output fingerprints with the code before the multiplier: sha256 of the cloud md5s
# (3 genomes x meters 0 and 2 x twin and victim x 12 clouds, keep on) and of the zone
# hashes, in order.
IMPRONTA_X1_ZONE = "ee0c3631433036156582a39890c08c21c9fc3371fa5f03e321225f2077f76cd2"
# same with the random mode (no zones), 6 clouds per stream
IMPRONTA_X1_RANDOM = "717816c7e6bb8ab235b42b171bdca3fa7bc2a2d84461ba19d57e53954666ce2b"
NODO_ZONE = ["--max-drift-lag-s", "0.5", "--perc-terms", "--structural", "gross", "--fast-ops",
             "--operator-mode", "insieme", "--zone-attaccante"]
NODO_RANDOM = ["--max-drift-lag-s", "0.5", "--perc-terms", "--structural", "gross", "--fast-ops"]

MASSIMO = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# genes of each family: at -1 the family is off
GENI_FAMIGLIA = {"deriva": [14], "rumore": [3], "geometria": [12, 15],
                 "strutturale": [6, 10, 13, 16], "dropout": [5]}


def _variante(spenti):
    g = list(MASSIMO)
    for i in spenti:
        g[i] = -1.0
    return g


GENOMI_FAMIGLIE = ([{"id": "massimo", "genoma": list(MASSIMO)},
                    {"id": "senza_fantasmi", "genoma": _variante([6])}]
                   + [{"id": f"senza_{f}", "genoma": _variante(GENI_FAMIGLIA[f])} for f in FAMIGLIE_LIMITI])


def _nodo(tmp_path, nome, radice, genomi, metri, n, keep, args, struct_log=True):
    d = tmp_path / nome
    d.mkdir(parents=True, exist_ok=True)
    gj = d / "genomi.json"
    gj.write_text(json.dumps(genomi))
    out = d / "uscita.json"
    extra = ["--struct-log", str(d / "struct_log.csv")] if struct_log else []
    cmd = [sys.executable, str(QUI / "limiti_nodo.py"), str(radice), str(NPZ), str(d), str(out), str(gj),
           metri, str(n), "1" if keep else "0", "--"] + list(args) + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def _impronta(d):
    t = "\n".join([r["uscita"] for r in d["righe"]] + [z["hash"] for z in d["zone"]])
    return hashlib.sha256(t.encode()).hexdigest()


def _per_genoma(d, campo="uscita"):
    out = {}
    for r in d["righe"]:
        out.setdefault(r["genoma"], []).append(r[campo])
    return out


# ----------------------------------------------------------------------
# Options and effective generator limits
# ----------------------------------------------------------------------

def test_opzioni_moltiplicatore():
    assert moltiplicatori_da_opzioni(1) == {} and moltiplicatori_da_opzioni("1.0", "deriva") == {}
    assert moltiplicatori_da_opzioni(2) == {f: 2.0 for f in FAMIGLIE_LIMITI}
    # x6 and non-integer values are allowed
    assert moltiplicatori_da_opzioni(6, "rumore,geometria,strutturale,dropout") == {
        "rumore": 6.0, "geometria": 6.0, "strutturale": 6.0, "dropout": 6.0}
    assert moltiplicatori_da_opzioni("2.5", ["dropout", "deriva"]) == {"deriva": 2.5, "dropout": 2.5}
    for k, f in ((0.5, None), (0, None), ("x", None), (float("inf"), None), (float("nan"), None),
                 (2, "deriva,deriva"), (2, "pippo"), (2, ""), (2, "rumore,")):
        with pytest.raises(ValueError):
            moltiplicatori_da_opzioni(k, f)
    with pytest.raises(ValueError):
        PerturbationGenerator(moltiplicatori={"pippo": 2.0})
    with pytest.raises(ValueError):
        PerturbationGenerator(moltiplicatori={"deriva": 0.9})


@pytest.mark.parametrize("mol", [None, {}, {f: 1.0 for f in FAMIGLIE_LIMITI}])
def test_x1_limiti_effettivi_uguali_ai_limiti_di_oggi(mol):
    g = PerturbationGenerator(max_dropout_rate=0.15, max_drift_lag_s=0.5, moltiplicatori=mol)
    assert g.moltiplicatori_attivi() is None
    assert (g._lim_point_shift, g._lim_dropout_rate, g._lim_ghost_ratio) == (0.05, 0.05, 0.02)
    assert (g._lim_cluster_std, g._lim_edge_shift, g._lim_drift_step) == (0.03, 0.08, 0.05)
    assert (g._lim_geo_range, g._lim_geo_yaw, g._lim_geo_scala, g._lim_geo_bias) == (0.05, pg.GEOM_MAX_YAW_RAD,
                                                                                   pg.GEOM_MAX_SCALE, 0.02)
    assert (g._lim_scan_sist, g._lim_scan_rand, g._lim_g13_punti, g._min_keep) == (0.02, 0.03, 500, 0.90)
    assert g.max_drift_m == 0.5 * pg.DRIFT_REF_SPEED_MPS
    # public attributes (status gen_limits, zones) unchanged
    assert (g.max_point_shift, g.max_dropout_rate, g.max_ghost_points_ratio) == (0.05, 0.05, 0.02)


def test_limiti_effettivi_per_famiglia():
    g = PerturbationGenerator(max_dropout_rate=0.15, max_drift_lag_s=0.5,
                              moltiplicatori={"deriva": 2, "dropout": 8, "strutturale": 3})
    assert g.moltiplicatori_attivi() == {"deriva": 2.0, "strutturale": 3.0, "dropout": 8.0}
    assert abs(g.max_drift_m - 0.80) < 1e-12 and abs(g._lim_drift_step - 0.10) < 1e-12
    # dropout: 5 % x 8 = 40 %, minimum kept points 1 - 0.8 -> capped at 50 %
    assert abs(g._lim_dropout_rate - 0.40) < 1e-12 and g._min_keep == 0.5
    assert PerturbationGenerator(max_dropout_rate=0.15, moltiplicatori={"dropout": 20})._lim_dropout_rate == 0.5
    assert PerturbationGenerator(moltiplicatori={"dropout": 2})._min_keep == pytest.approx(0.8)
    assert g._lim_g13_punti == 1500 and abs(g._lim_ghost_ratio - 0.06) < 1e-12
    assert abs(g._lim_edge_shift - 0.24) < 1e-12 and abs(g._lim_cluster_std - 0.09) < 1e-12
    # families at x1: current limits; directions never touched
    assert (g._lim_point_shift, g._lim_geo_yaw, g._lim_scan_sist) == (0.05, pg.GEOM_MAX_YAW_RAD, 0.02)
    p = g.encode_perturbation(np.array(MASSIMO))
    p1 = PerturbationGenerator(max_dropout_rate=0.15).encode_perturbation(np.array(MASSIMO))
    assert np.array_equal(p["noise_direction"], p1["noise_direction"])
    assert np.array_equal(p["cluster_direction"], p1["cluster_direction"])
    assert p["dropout_rate"] == pytest.approx(0.40) and p["ghost_ratio"] == pytest.approx(0.06)


def _nuvola_sintetica(n=4000, seme=7):
    rng = np.random.default_rng(seme)
    xyz = np.c_[rng.uniform(-8, 8, n), rng.uniform(-8, 8, n), rng.uniform(-0.5, 2.0, n)]
    return np.c_[xyz, rng.uniform(0, 255, n)]


@pytest.mark.parametrize("K", [1, 2, 4, 6])
def test_deriva_tetto_generatore(K):
    """Drift only (g14 = 1, direction (0, 1, 0)): every cloud is translated by the
    accumulated drift; cap 0.40 K m reached at the ninth cloud, as at x1 (step and cap
    scaled together)."""
    g = [-1.0] * 17
    g[0:3] = [0.0, 1.0, 0.0]
    g[14] = 1.0
    gen = PerturbationGenerator(max_drift_lag_s=0.5, moltiplicatori={"deriva": K} if K != 1 else None)
    p = gen.encode_perturbation(np.array(g))
    c = _nuvola_sintetica()
    norme = []
    for i in range(30):
        out = gen.apply_perturbation(c, p, seed=i)
        sp = out[:, :3] - c[:, :3]
        assert np.allclose(sp, sp[0], atol=1e-12)              # rigid translation
        norme.append(float(np.linalg.norm(sp[0])))
    assert max(norme) == pytest.approx(0.40 * K, abs=1e-9)
    assert norme.index(max(norme)) == 8 and norme[7] < 0.40 * K
    ref = [min(0.05 * (1 - 0.98 ** (i + 1)) / 0.02, 0.40) * K for i in range(30)]
    assert norme == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_dropout_stop_fisico_mai_oltre_meta():
    """_coherent_struct_dropout: with a target equal to all points at most half of the scan
    points are removed; below half the target is removed."""
    gen = PerturbationGenerator(operator_mode="coherent")
    n, nv, per = 1000, 100, 10
    gen._structs = SimpleNamespace(anchors={"g5": [{"axis1": np.array([1.0, 0, 0])}]}, voxel=0.2, per_op=True)
    gen._n_zone = lambda: None
    fr = {"order": np.arange(n), "starts": np.arange(nv) * per, "counts": np.full(nv, per),
          "hits": {"g5": [np.arange(nv)]}, "keys_a": np.arange(nv), "cent_a": np.zeros((nv, 3))}
    for quanti, atteso in ((n, n // 2), (int(0.7 * n), n // 2), (300, 300), (0, 0)):
        drop = gen._coherent_struct_dropout(fr, n, 0.05, quanti=quanti, chiave="g5")
        assert int(drop.sum()) == atteso <= n // 2
    assert int(gen._coherent_struct_dropout(fr, n, 1.0, chiave="g5").sum()) == n // 2


# ----------------------------------------------------------------------
# Real node: x1 identical to the current code; drift, dropout, x6 and families
# ----------------------------------------------------------------------

@nuvole_vere
@pytest.mark.parametrize("modo", ["zone", "random"])
def test_x1_impronte_uguali_al_codice_di_prima(tmp_path, modo):
    """Multiplier off: node outputs (md5 of each cloud) and zones identical to those of the
    code before the multiplier for the three seed-1001 genomes."""
    args, n, imp = (NODO_ZONE, 12, IMPRONTA_X1_ZONE) if modo == "zone" else (NODO_RANDOM, 6, IMPRONTA_X1_RANDOM)
    d = _nodo(tmp_path, "x1", ROOT, GENOMI_S1001, "0,2", n, True, args)
    assert len(d["righe"]) == 3 * 2 * 2 * n
    assert _impronta(d) == imp
    assert d["moltiplicatori"] is None and "moltiplicatori_limiti" not in d["status"]
    assert all(z["moltiplicatori"] is None for z in d["zone"])
    if modo == "zone":
        assert "frac_tolti" not in d["struct_log_colonne"]


@nuvole_vere
@pytest.mark.skipif(not (ORIGINALE / "src/nodes/perturbation_node.py").exists(), reason="copia originale assente")
def test_x1_uguale_alla_copia_originale(tmp_path):
    a = _nodo(tmp_path, "orig", ORIGINALE, GENOMI_S1001, "0,2", 8, True, NODO_ZONE)
    b = _nodo(tmp_path, "copia", ROOT, GENOMI_S1001, "0,2", 8, True, NODO_ZONE)
    assert [r["uscita"] for r in a["righe"]] == [r["uscita"] for r in b["righe"]]
    assert [z["hash"] for z in a["zone"]] == [z["hash"] for z in b["zone"]]
    assert sorted(a["status"]) == sorted(b["status"])
    assert a["struct_log_colonne"] == b["struct_log_colonne"]
    togli = lambda rr: [{k: v for k, v in r.items() if k != "t_arr_mono"} for r in rr]  # noqa: E731
    assert togli(a["struct_log"]) == togli(b["struct_log"])


@nuvole_vere
def test_deriva_x2_tetto_080_nel_nodo(tmp_path):
    """Drift x2 with the massimo genome and keep: twin and victim drift reaches 0.80 m at the
    ninth cloud and no further; x1 at 0.40 m."""
    g = [GENOMI_S1001[0]]
    d1 = _nodo(tmp_path, "x1", ROOT, g, "2", 12, True, NODO_ZONE, struct_log=False)
    d2 = _nodo(tmp_path, "x2", ROOT, g, "2", 12, True,
               NODO_ZONE + ["--moltiplicatore-limiti", "2", "--famiglie-limiti", "deriva"], struct_log=False)
    for d, tetto in ((d1, 0.40), (d2, 0.80)):
        gem = [r["deriva_m"] for r in d["righe"] if r["stream"] == "gemello"]
        vit = [r["deriva_m"] for r in d["righe"] if r["stream"] == "vittima"]
        assert max(gem) == pytest.approx(tetto, abs=1e-9) and max(vit) == pytest.approx(tetto, abs=1e-9)
        assert gem.index(max(gem)) == 8 and gem[7] < tetto
    assert d2["status"]["moltiplicatori_limiti"] == {"deriva": 2.0}
    assert all(z["moltiplicatori"] == {"deriva": 2.0} for z in d2["zone"])


@nuvole_vere
def test_dropout_xK_misurato_e_registrato(tmp_path):
    """Dropout x2, x4, x8 (massimo genome without ghosts): never more than half of the scan
    points removed; in the zone log n_tolti and frac_tolti match those measured on the clouds
    (n_in - n_out without ghosts)."""
    g = [GENOMI_FAMIGLIE[1]]
    medie = {}
    for K in (1, 2, 4, 8):
        mol = [] if K == 1 else ["--moltiplicatore-limiti", str(K), "--famiglie-limiti", "dropout"]
        d = _nodo(tmp_path, f"x{K}", ROOT, g, "2", 8, False, NODO_ZONE + mol)
        fr = []
        for r in d["righe"]:
            tolti = r["n_in"] - r["n_out"]
            assert r["conteggi"]["n_added"] == 0 and r["conteggi"]["n_removed"] == tolti
            assert 0 <= tolti <= r["n_in"] // 2
            fr.append(tolti / r["n_in"])
        medie[K] = float(np.mean(fr))
        if K == 1:
            assert "frac_tolti" not in d["struct_log_colonne"]
            continue
        assert d["struct_log_colonne"][-4:] == ["n_in", "n_tolti", "n_aggiunti", "frac_tolti"]
        assert len(d["struct_log"]) == len(d["righe"])
        for s, r in zip(d["struct_log"], d["righe"]):
            assert s["stream"] == r["stream"]
            assert int(s["n_in"]) == r["n_in"] and int(s["n_tolti"]) == r["n_in"] - r["n_out"]
            assert int(s["n_aggiunti"]) == 0 and int(s["n_tolti"]) == int(s["n_hit_g5"])
            assert float(s["frac_tolti"]) == (r["n_in"] - r["n_out"]) / r["n_in"]
        assert d["status"]["moltiplicatori_limiti"] == {"dropout": float(K)}
        # per meter (run summary and frazione-tolta subcommand): victim and twin
        ft = cs.frazione_tolta(str(tmp_path / f"x{K}"), str(tmp_path / f"x{K}"))
        assert ft["colonne_moltiplicatore"] and {x["flusso"] for x in ft["per_metro"]} == {"vittima", "gemello"}
        for x in ft["per_metro"]:
            v = sorted((r["n_in"] - r["n_out"]) / r["n_in"] for r in d["righe"] if r["stream"] == x["flusso"])
            assert x["metro"] == "2" and x["n_con_frazione"] == len(v) == 8
            assert (x["min"], x["max"]) == (v[0], v[-1]) and x["mediana"] == pytest.approx(float(np.median(v)), abs=1e-15)
    assert medie[2] > medie[1] and max(medie.values()) <= 0.5


@nuvole_vere
def test_x6_accettato_dal_nodo(tmp_path):
    g = [GENOMI_FAMIGLIE[0], GENOMI_FAMIGLIE[2 + FAMIGLIE_LIMITI.index("deriva")]]
    d1 = _nodo(tmp_path, "x1", ROOT, g, "2", 6, True, NODO_ZONE, struct_log=False)
    d6 = _nodo(tmp_path, "x6", ROOT, g, "2", 6, True,
               NODO_ZONE + ["--moltiplicatore-limiti", "6", "--famiglie-limiti", "rumore,geometria,strutturale,dropout"])
    att = {"rumore": 6.0, "geometria": 6.0, "strutturale": 6.0, "dropout": 6.0}
    assert d6["status"]["moltiplicatori_limiti"] == att and all(z["moltiplicatori"] == att for z in d6["zone"])
    a, b = _per_genoma(d1), _per_genoma(d6)
    assert all(x != y for x, y in zip(a["massimo"], b["massimo"]))
    assert all(r["zone_stato"] == "ok" for r in d6["righe"])
    # drift outside the families: same accumulated drift as x1, cloud by cloud
    assert [r["deriva_m"] for r in d6["righe"]] == [r["deriva_m"] for r in d1["righe"]]
    assert max(r["deriva_m"] for r in d6["righe"]) <= 0.40 + 1e-12


@nuvole_vere
def test_famiglie_indipendenti(tmp_path):
    """x4 on a single family: with that family's genes at -1 the clouds are identical to x1
    (the other families stay at x1); with the massimo genome they change."""
    base = _per_genoma(_nodo(tmp_path, "x1", ROOT, GENOMI_FAMIGLIE, "2", 6, False, NODO_ZONE, struct_log=False))
    for f in FAMIGLIE_LIMITI:
        d = _nodo(tmp_path, f, ROOT, GENOMI_FAMIGLIE, "2", 6, False,
                  NODO_ZONE + ["--moltiplicatore-limiti", "4", "--famiglie-limiti", f], struct_log=False)
        u = _per_genoma(d)
        assert u[f"senza_{f}"] == base[f"senza_{f}"], f
        assert all(x != y for x, y in zip(u["massimo"], base["massimo"])), f
        assert d["status"]["moltiplicatori_limiti"] == {f: 4.0}


# nsga3_eps5 w3 with seed 1001 (history.json): g5 = -0.81 and low g6/g16, dropout and ghosts
# off at x1
NSGA3_EPS5_W3 = [-0.01808672964610908, -0.901656303229961, 0.1207365567616652, 0.04345277071123099,
                 -0.8237591469162449, -0.8133776860138648, -0.8239001634759636, -0.9197237449624227,
                 -0.8550300565313802, -0.11225216521972092, -0.9603047675482947, -0.2947605416858594,
                 -0.9998658935471787, -0.8905658726115625, -0.9950564163619185, -0.8196990116148815,
                 -0.9717902203731605]
# operators of each family (g4 and g11 modulate g3)
OPERATORI_FAMIGLIA = {"deriva": {"g14"}, "rumore": {"g3", "g4", "g11"}, "geometria": {"g12", "g15"},
                      "strutturale": {"g6", "g16", "g10", "g13"}, "dropout": {"g5"}}


def _genomi_accensione():
    """Genomes for the thresholds: the 3 seed-1001 genomes above, nsga3_eps5 w3, the 20
    applied ones from the seed-1001 histories if present, uniform random and random with low
    genes, and genomes straddling the thresholds of g3 (-0.96), g5 (-0.6) and g6 (0)."""
    gen = [g["genoma"] for g in GENOMI_S1001] + [NSGA3_EPS5_W3]
    radice = Path(os.path.expanduser("~/mola-adversarial-nsga3/data/attack/serieB_finale_corretta_genomanullo"))
    for c in ("massimo", "nsga3_eps5", "casuale_senzabudget", "casuale_eps5"):
        for f in radice.glob(f"{c}/s1001/campagna/straight/*/rep_01/history.json"):
            gen += [r["genome"] for r in json.loads(f.read_text()) if r.get("genome") is not None]
    rng = np.random.default_rng(20261107)
    gen += list(rng.uniform(-1, 1, (150, 17))) + list(rng.uniform(-1, -0.5, (150, 17)))
    for i, soglia in ((3, -0.96), (5, -0.6), (6, 0.0), (16, 0.0)):
        for d in (-1e-6, 1e-6, -0.05, 0.05):
            g = rng.uniform(-1, 1, 17)
            g[i] = soglia + d
            if i == 16:
                g[6] = 0.5
            gen.append(g)
    return [np.asarray(g, dtype=float) for g in gen]


@pytest.mark.parametrize("K", [2, 4, 8])
def test_soglie_di_accensione_come_x1(K):
    """The multiplier changes operator strength, never which ones turn on: for every family
    (and all together) the active operators are those of x1, genome by genome."""
    g1 = PerturbationGenerator(max_dropout_rate=0.15, max_drift_lag_s=0.5)
    gen = _genomi_accensione()
    assert len(gen) >= 4 + 300 + 16
    acc1 = [g1.operatori_accesi(g1.encode_perturbation(g)) for g in gen]
    for fam in [(f,) for f in FAMIGLIE_LIMITI] + [FAMIGLIE_LIMITI]:
        gk = PerturbationGenerator(max_dropout_rate=0.15, max_drift_lag_s=0.5, moltiplicatori={f: K for f in fam})
        diversi = [i for i, g in enumerate(gen) if gk.operatori_accesi(gk.encode_perturbation(g)) != acc1[i]]
        assert not diversi, (fam, diversi[:5])
    w3 = g1.operatori_accesi(g1.encode_perturbation(np.array(NSGA3_EPS5_W3)))
    assert "g5" not in w3 and "g6" not in w3 and "g16" not in w3
    # at x1 the parameters are the usual ones (no _x1 key: zone signature unchanged)
    assert not any(k.endswith("_x1") for k in g1.encode_perturbation(np.array(NSGA3_EPS5_W3)))


@pytest.mark.parametrize("K", [2, 4, 8])
def test_famiglia_spenta_a_x1_resta_spenta(K):
    """Random mode (fixed seed, synthetic cloud): if at x1 no operator of a family is on, the
    family at xK gives the same cloud as x1 (one extra active operator would also change the
    random sequence of the others)."""
    c = _nuvola_sintetica()
    gen = [np.array(NSGA3_EPS5_W3)] + _genomi_accensione()[4:40]
    controllati = 0
    for g in gen:
        g1 = PerturbationGenerator(max_dropout_rate=0.15, max_drift_lag_s=0.5)   # drift from zero
        p1 = g1.encode_perturbation(g)
        out1 = g1.apply_perturbation(c, p1, seed=7)
        acc = set(g1.operatori_accesi(p1))
        for f in FAMIGLIE_LIMITI:
            if acc & OPERATORI_FAMIGLIA[f]:
                continue
            gk = PerturbationGenerator(max_dropout_rate=0.15, max_drift_lag_s=0.5, moltiplicatori={f: K})
            outk = gk.apply_perturbation(c, gk.encode_perturbation(g), seed=7)
            assert outk.shape == out1.shape and np.array_equal(outk, out1), (f, K)
            controllati += 1
    assert controllati >= 10


@nuvole_vere
def test_operatori_accesi_nel_nodo_come_x1(tmp_path):
    """Real node, zone mode: for nsga3_eps5 w3 dropout x4 and structural x6 (off at x1) give
    the x1 clouds; with all families at x8 the active localized operators (n_hit keys) are
    those of x1, cloud by cloud, for three genomes."""
    w3 = [{"id": "w3", "genoma": NSGA3_EPS5_W3}]
    base = _per_genoma(_nodo(tmp_path, "w3_x1", ROOT, w3, "2", 6, False, NODO_ZONE, struct_log=False))
    for nome, mol in (("drop4", ["--moltiplicatore-limiti", "4", "--famiglie-limiti", "dropout"]),
                      ("strut6", ["--moltiplicatore-limiti", "6", "--famiglie-limiti", "strutturale"])):
        d = _nodo(tmp_path, f"w3_{nome}", ROOT, w3, "2", 6, False, NODO_ZONE + mol, struct_log=False)
        assert _per_genoma(d) == base, nome
        assert all(not r["n_hit"].get("g5") and not r["n_hit"].get("g6") and not r["n_hit"].get("g16")
                   for r in d["righe"])
    gen = w3 + [GENOMI_S1001[1], GENOMI_S1001[2]]
    a = _per_genoma(_nodo(tmp_path, "tre_x1", ROOT, gen, "2", 4, False, NODO_ZONE, struct_log=False), "n_hit")
    b = _per_genoma(_nodo(tmp_path, "tre_x8", ROOT, gen, "2", 4, False,
                          NODO_ZONE + ["--moltiplicatore-limiti", "8"], struct_log=False), "n_hit")
    for g in a:
        assert [sorted(x) for x in a[g]] == [sorted(x) for x in b[g]], g


def test_nodo_zone_con_altro_moltiplicatore():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    g1 = PerturbationGenerator()
    g2 = PerturbationGenerator(moltiplicatori={"dropout": 2})
    z1 = SimpleNamespace(moltiplicatori=None)
    z2 = SimpleNamespace(moltiplicatori={"dropout": 2.0})
    assert pn.moltiplicatori_uguali(z1, g1) and pn.moltiplicatori_uguali(z2, g2)
    assert not pn.moltiplicatori_uguali(z1, g2) and not pn.moltiplicatori_uguali(z2, g1)
    # zones without the attribute (older objects) = off
    assert pn.moltiplicatori_uguali(SimpleNamespace(), g1)
    a = SimpleNamespace(moltiplicatore_limiti=4.0, famiglie_limiti="rumore,dropout", max_point_shift=0.05,
                        noise_std=0.02, dropout_rate=0.15, max_drift_lag_s=0.5)
    assert pn.make_generator(a).moltiplicatori_attivi() == {"rumore": 4.0, "dropout": 4.0}
    a0 = SimpleNamespace(max_point_shift=0.05, noise_std=0.02, dropout_rate=0.15, max_drift_lag_s=0.5)
    assert pn.make_generator(a0).moltiplicatori_attivi() is None


# ----------------------------------------------------------------------
# Orchestrator: zone limits, gene 14 blocked, victim process
# ----------------------------------------------------------------------

@pytest.fixture
def orch():
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    salvato = dict(ao.LIMITI)
    yield ao
    ao.LIMITI.clear()
    ao.LIMITI.update(salvato)


def _args(**kw):
    d = dict(moltiplicatore_limiti=1.0, famiglie_limiti=None, blocca_gene=None, fixed_genome=None,
             fixed_genome_seq=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_orchestratore_opzioni_limiti(orch):
    assert orch.problema_limiti(_args()) is None and orch.LIMITI == {"mol": {}, "bloccati": []}
    assert orch.problema_limiti(_args(moltiplicatore_limiti=6.0, famiglie_limiti="rumore,dropout")) is None
    assert orch.LIMITI["mol"] == {"rumore": 6.0, "dropout": 6.0}
    assert "numero >= 1" in orch.problema_limiti(_args(moltiplicatore_limiti=0.5))
    assert orch.problema_limiti(_args(blocca_gene=["14=-1"])) is None and orch.LIMITI["bloccati"] == [14]
    for bad in (["14=1"], ["17=-1"], ["14"], ["14=-1", "14=-1"], ["a=-1"]):
        assert orch.problema_limiti(_args(blocca_gene=bad)) is not None, bad
    gm = ",".join(str(v) for v in MASSIMO)
    assert "conflitto" in orch.problema_limiti(_args(blocca_gene=["14=-1"], fixed_genome=gm))
    assert orch.problema_limiti(_args(blocca_gene=["14=-1"], fixed_genome=",".join(str(v) for v in _variante([14])))) is None
    assert orch.problema_limiti(_args(blocca_gene=["14=-1"], fixed_genome_seq="x.json")) is not None


def test_orchestratore_limiti_zone_e_controllo(orch):
    lim = {"max_point_shift": 0.05, "noise_std": 0.02, "max_dropout_rate": 0.05, "max_ghost_points_ratio": 0.02}
    nodo = SimpleNamespace(node_gen_limits=dict(lim), node_moltiplicatori=None)
    orch.problema_limiti(_args())
    assert orch.limiti_zone(nodo, "prova") == orch.node_gen_limits(nodo)      # off: unchanged
    assert orch.limiti_fields(nodo) == {}
    nodo.node_moltiplicatori = {"dropout": 4.0}
    with pytest.raises(orch.ErroreMoltiplicatori):
        orch.limiti_zone(nodo, "prova")
    orch.problema_limiti(_args(moltiplicatore_limiti=4.0, famiglie_limiti="dropout"))
    lz = orch.limiti_zone(nodo, "prova")
    assert lz["moltiplicatori"] == {"dropout": 4.0} and lz["max_dropout_rate"] == 0.05
    assert orch.limiti_fields(nodo)["moltiplicatori_limiti"] == {"dropout": 4.0}
    nodo.node_moltiplicatori = None
    with pytest.raises(orch.ErroreConfigurazioneZone):      # subclass: same run shutdown
        orch.controlla_moltiplicatori(nodo, "finestra 0")
    gen = orch.print_generator(SimpleNamespace(node_gen_limits=dict(lim)))
    assert gen.moltiplicatori_attivi() == {"dropout": 4.0}


def test_blocco_g14_in_ogni_candidato(orch):
    """With --blocca-gene 14=-1 gene 14 leaves the search space: cold start, sparse
    population, warm start (also from a front with g14 != -1), NSGA-III offspring, random
    search and single random always have g14 = -1."""
    from pymoo.optimize import minimize
    from src.optimization import window_record as wrec
    act = orch.geni_attivi(3, [14])
    assert act == [i for i in range(17) if i != 14] and orch.GENOME_GROUPS[3] == list(range(17))
    assert orch.geni_attivi(3) == list(range(17))
    ns = len(act)
    x = np.arange(ns) / 20.0
    full = orch.expand_genome(x, act)
    assert full[14] == -1.0 and full[15] == x[14] and full[16] == x[15] and np.array_equal(full[:14], x[:14])
    visti = []

    def valuta(xa):
        g = orch.expand_genome(xa, act)
        visti.append(g)
        return (-float(np.sum(g)), float(np.linalg.norm(g)), -1.0)

    off = np.array([0.0 if a in orch.DIRECTION_GENES else -1.0 for a in act])
    rng = np.random.default_rng(5)
    # previous-window front with any g14: projected onto the search space
    log = [{"genome": rng.uniform(-1, 1, 17).tolist(), "admissible": True, "dev_mean": float(rng.uniform(0, .3)),
            "chamfer": float(rng.uniform(0, 30)), "idx": i} for i in range(12)]
    assert any(e["genome"][14] != -1.0 for e in log)
    pool, _, _ = orch.warm_pool_from_log(log, "maxdamage", None, act)
    assert pool and all(len(t[0]) == ns for t in pool)
    righe, _ = orch.warm_front_rows(pool, 8, np.random.default_rng(1), ns)
    calda = np.array(righe + list(orch.legacy_random_genomes(np.random.default_rng(2), 16 - len(righe), ns, True, off)))
    for init in (None, orch.legacy_random_genomes(np.random.default_rng(3), 16, ns, True, off), calda):
        res = minimize(orch.build_problem(ns, valuta), orch.build_nsga3(16, init_X=init), ("n_gen", 4),
                       seed=1001, verbose=False)
        assert res.X is not None
    for modo in ("uniform", "sparse"):
        for xa in wrec.sample_genomes(np.random.default_rng(4), 64, ns, modo, off):
            valuta(xa)
    for xa in orch.legacy_random_genomes(np.random.default_rng(6), 64, ns, False, off):
        valuta(xa)
    V = np.array(visti)
    assert len(V) > 3 * 16 * 2 + 192
    assert np.all(V[:, 14] == -1.0)
    # the other genes move (none stuck at -1 because of a wrong index)
    assert all(np.any(V[:, i] != -1.0) for i in range(17) if i != 14)


def test_sorgente_orchestratore_usa_i_limiti():
    s = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert 'active_genes = geni_attivi(args.genome_group, LIMITI["bloccati"])' in s
    assert s.count("limiti_zone(") == 3          # definition, genome zones, ramp zones
    assert "node_gen_limits(self))" not in s and "node_gen_limits(node))\n" not in s
    assert "**limiti_fields(node)," in s and "err = problema_limiti(args)" in s
    # victim-without-pose fields in the three run_end branches (attack discovered, with and
    # without chain check, and chain failure), never inside controlla_catena
    assert s.count("**proc_v})") == 3 and "proc_v = campi_vittima_persa(victim)" in s
    i = s.index("def controlla_catena(")
    assert "processo_mola_vittima" not in s[i:s.index("\ndef ", i + 10)]


# ICP printf lines as MOLA writes them (format of tests/test_g_nominale_senza_posa.py)
def _printf(stamp, good, q=None):
    q = (0.95 if good else 0.05) if q is None else q
    return (f"[LidarOdometry] pathStep=1 timestamp=1970/01/01,00:00:{stamp:09.6f} goodness={q:.3f} "
            f"minRequired=0.200 iters=3 termReason=X isGood={1 if good else 0} adapt_thres_sigma=0.0400 "
            f"consecutive_bad=0\n")


def test_processo_mola_vittima(orch, tmp_path):
    viva = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None, pid=11))
    morta = SimpleNamespace(proc=SimpleNamespace(poll=lambda: -11, pid=12))
    assert orch.processo_mola_vittima(viva) == {"vivo": True, "rc": None, "pid": 11, "esito": "vivo"}
    assert orch.processo_mola_vittima(morta) == {"vivo": False, "rc": -11, "pid": 12, "esito": "terminato"}
    # process not found: not a crash (vivo None, distinct outcome)
    assert orch.processo_mola_vittima(SimpleNamespace(proc=None)) == {"vivo": None, "rc": None, "pid": None,
                                                                    "esito": "non_trovato"}
    rif = tmp_path / "rifiuto.log"
    rif.write_text(_printf(10.0, True) + _printf(10.1, True) + _printf(10.2, False, 0.12) + _printf(10.3, False))
    buono = tmp_path / "buono.log"
    buono.write_text(_printf(10.0, True) + _printf(10.1, True))
    casi = [(viva, rif, "rifiuto_da_attacco"), (viva, buono, "senza_posa_da_confermare"),
            (morta, rif, "crash_processo"), (SimpleNamespace(proc=None), rif, "processo_non_trovato")]
    for v, log, atteso in casi:
        v.log_path = str(log)
        c = orch.campi_vittima_persa(v)
        assert c["esito_vittima"] == atteso, (log.name, c)
    c = orch.campi_vittima_persa(SimpleNamespace(proc=viva.proc, log_path=str(rif)))
    assert c["rifiuto_in_linea"]["rifiutate_dopo_ultima_buona"] == 2
    assert c["rifiuto_in_linea"]["goodness_min_dopo_ultima_buona"] == 0.05


# ----------------------------------------------------------------------
# Queue: victim crash rule
# ----------------------------------------------------------------------

def _storia(tmp_path, nome, finestre):
    rep = tmp_path / nome
    rep.mkdir()
    (rep / "history.json").write_text(json.dumps(finestre))
    return str(rep)


def _fine(status, vivo=None, rc=None, window=2, non_trovato=False, rifiuto=None):
    f = {"reason": "attacco_scoperto", "window": window, "status": status}
    if non_trovato:
        f["processo_mola_vittima"] = {"vivo": None, "rc": None, "pid": None, "esito": "non_trovato"}
    elif vivo is not None:
        f["processo_mola_vittima"] = {"vivo": vivo, "rc": rc, "pid": 1, "esito": "vivo" if vivo else "terminato"}
    if rifiuto is not None:
        f["rifiuto_in_linea"] = {"rifiuto": rifiuto}
    return f


def test_storia_rifiuto_e_crash(tmp_path):
    G = [[float(i), 0.5] for i in range(3)]
    # process alive, no log: the inline signs apply
    s = cs.storia(_storia(tmp_path, "a", [{"window": 0, "genome": G[0]}, {"window": 1, "genome": G[1]},
                                          {"window": 2, "genome": G[2],
                                           "run_end": _fine("victim_no_pose", True, rifiuto=True)}]))
    vp = s["vittima_persa"]
    assert s["attacco_scoperto"] and vp["crash"] is False and vp["marcatore"] == "rifiuto_da_attacco"
    assert vp["esito_vittima"] == "rifiuto da attacco" and vp["genoma"] == G[2]
    s = cs.storia(_storia(tmp_path, "a2", [{"window": 0, "genome": G[0],
                                            "run_end": _fine("victim_no_pose", True, window=0, rifiuto=False)}]))
    assert s["vittima_persa"]["marcatore"] == "senza_posa_da_confermare" and not s["vittima_persa"]["crash"]
    s = cs.storia(_storia(tmp_path, "b", [{"window": 0, "genome": G[0]},
                                          {"window": 1, "genome": G[1], "run_end": _fine("pose_stale", False, -11, 1)}]))
    assert s["vittima_persa"]["crash"] and s["vittima_persa"]["processo"]["rc"] == -11
    assert s["vittima_persa"]["genoma"] == G[1] and s["vittima_persa"]["marcatore"] == "crash_processo"
    # process not found: distinct marker, never a crash
    s = cs.storia(_storia(tmp_path, "n", [{"window": 0, "genome": G[0],
                                           "run_end": _fine("victim_no_pose", window=0, non_trovato=True)}]))
    assert s["vittima_persa"]["marcatore"] == "processo_non_trovato" and s["vittima_persa"]["crash"] is False
    # victim_crash: process found dead before the segment, genome of the previous window
    s = cs.storia(_storia(tmp_path, "c", [{"window": 0, "genome": G[0]},
                                          {"window": 1, "genome": G[1], "run_end": _fine("victim_crash", window=1)}]))
    assert s["vittima_persa"]["crash"] and s["vittima_persa"]["fonte"] == "status del rollout"
    assert s["vittima_persa"]["genoma"] == G[0] and s["vittima_persa"]["genoma_finestra"] == 0
    # run without lost victim: no new key
    s = cs.storia(_storia(tmp_path, "d", [{"window": 0, "genome": G[0], "run_end": {"reason": "finestre_completate"}}]))
    assert "vittima_persa" not in s


def test_storia_rifiuto_dal_log_completo(tmp_path):
    """At the end of the run the queue re-reads the full victim log: an attack rejection
    requires rejected ICPs after the last good one, whatever the inline signs say."""
    for nome, righe, in_linea, atteso in (
            ("r1", _printf(5.0, True) + _printf(5.1, False, 0.1), False, "rifiuto_da_attacco"),
            ("r2", _printf(5.0, True) + _printf(5.1, True), True, "senza_posa_senza_rifiuto"),
            ("r3", _printf(5.0, False) + _printf(5.1, True), True, "senza_posa_senza_rifiuto")):
        rep = Path(_storia(tmp_path, nome, [{"window": 0, "genome": [1.0], "moltiplicatori_limiti": {"dropout": 4.0},
                                             "run_end": _fine("victim_no_pose", True, window=0, rifiuto=in_linea)}]))
        (rep / "logs").mkdir()
        (rep / "logs" / "mola_victim.log").write_text(righe)
        vp = cs.storia(str(rep))["vittima_persa"]
        assert vp["marcatore"] == atteso and vp["fonte_segni_rifiuto"] == "log completo a fine run", (nome, vp)
        assert vp["moltiplicatori"] == {"dropout": 4.0}


def _fatti(storia, condizione="massimo"):
    return {"rc": 0, "completata": True, "adattatore_vivo": True, "storia": storia, "baseline": False,
            "ritardo_rc": 0, "ombra": {"esito": "identica"}, "watchdog_interventi": 0, "condizione": condizione}


def test_classifica_regola_crash(tmp_path):
    crash = cs.storia(_storia(tmp_path, "c", [{"window": 0, "genome": [1.0],
                                               "run_end": _fine("mola_crash", False, -6, 0)}]))
    rifiuto = cs.storia(_storia(tmp_path, "r", [{"window": 0, "genome": [1.0],
                                                 "run_end": _fine("victim_no_pose", True, None, 0, rifiuto=True)}]))
    non_trovato = cs.storia(_storia(tmp_path, "n", [{"window": 0, "genome": [1.0],
                                                     "run_end": _fine("victim_no_pose", window=0, non_trovato=True)}]))
    # rule off: unchanged (fatta, attacco scoperto), with the victim outcome in the details
    e = cs.classifica(_fatti(crash))
    assert (e["esito"], e["motivo"], e["vittima"]) == ("fatta", "attacco scoperto", "crash di processo")
    e = cs.classifica(_fatti(rifiuto), regola_crash=True)
    assert (e["esito"], e["vittima"]) == ("fatta", "rifiuto da attacco")
    e = cs.classifica(_fatti(crash), regola_crash=True)
    assert e["esito"] == "crash_processo" and e["crash"]["genoma"] == [1.0] and e["crash"]["rc"] == -6
    # process not found: not a crash even with the rule
    e = cs.classifica(_fatti(non_trovato), regola_crash=True)
    assert (e["esito"], e["vittima_marcatore"]) == ("fatta", "processo_non_trovato")
    # rule only with a fixed genome: does not apply to a search
    e = cs.classifica(_fatti(crash, "nsga3_eps5"), regola_crash=True)
    assert e["esito"] == "fatta" and e["vittima_marcatore"] == "crash_processo" and "regola_crash" in e
    # chain before crash: a failure stays a failure
    f = _fatti(crash)
    f["rc"] = cs.RC_GUASTO_CATENA
    assert cs.classifica(f, regola_crash=True)["esito"] == "guasto_catena"
    completa = cs.storia(_storia(tmp_path, "k", [{"window": 0, "genome": [1.0],
                                                  "run_end": {"reason": "finestre_completate"}}]))
    assert cs.classifica(_fatti(completa)) == cs.classifica(_fatti(completa), regola_crash=True)


def _esito(tmp_path, nome, esito, genoma=None, mol=None):
    e = {"esito": esito, "motivo": nome, "cartella": str(tmp_path / nome)}
    if genoma is not None:
        e["dettaglio"] = {"crash": {"genoma": genoma, "finestra": 1, "moltiplicatori": mol}}
    return e


@pytest.mark.parametrize("secondo,stato,esito2", [
    ("stesso", "crash_da_attacco", "crash_da_attacco"),
    ("diverso", "da_rifare", "guasto_catena"),
    ("altro_K", "da_rifare", "guasto_catena"),
    ("fatta", "fatta", "fatta"),
])
def test_chiudi_regola_crash(tmp_path, secondo, stato, esito2):
    sf = str(tmp_path / "coda" / "stato_coda.json")
    cs.inizia(sf, "massimo", "1001", str(tmp_path / "s1001"))
    m4 = {"dropout": 4.0}
    assert cs.chiudi(sf, "massimo", "1001", _esito(tmp_path, "s1001", "crash_processo", [1.0, -1.0], m4)) == "da_rifare"
    cs.inizia(sf, "massimo", "1001", str(tmp_path / "s1001_rip1"))
    # same level = same K (and families) with the same genome
    e2 = {"stesso": _esito(tmp_path, "s1001_rip1", "crash_processo", [1.0, -1.0], m4),
          "diverso": _esito(tmp_path, "s1001_rip1", "crash_processo", [0.5, -1.0], m4),
          "altro_K": _esito(tmp_path, "s1001_rip1", "crash_processo", [1.0, -1.0], {"dropout": 8.0}),
          "fatta": _esito(tmp_path, "s1001_rip1", "fatta")}[secondo]
    assert cs.chiudi(sf, "massimo", "1001", e2) == stato
    t = cs.leggi(sf)["runs"]["s1001/massimo"]["tentativi"]
    assert t[0]["esito"] == "crash_processo" and t[1]["esito"] == esito2
    testo = cs.riepilogo(cs.leggi(sf), [("1001", "massimo")], per_run=False)
    assert ("Crash da attacco (1):" in testo) == (secondo == "stesso")


def test_frazione_tolta_senza_colonne(tmp_path):
    """Run at x1 (zone log without the multiplier columns): for the victim n_hit_g5 divided by
    n_points of the per-frame log, for the twin only the removed points."""
    rep = tmp_path / "rep_01"
    rep.mkdir()
    (rep / "struct_log.csv").write_text(
        "stream,genome_id,frame_idx,stamp,metro,n_hit_g5\n"
        "gemello,2,0,12.4,0,300\nvittima,1,0,20.1,0,400\nvittima,1,1,20.2,0,500\nvittima,2,0,30.1,1,250\n")
    (tmp_path / "frame_log.csv").write_text(
        "stamp_in,victim_stream,n_points\n12.4,0,4000\n20.1,1,5000\n20.2,1,5000\n30.1,1,5000\n")
    ft = cs.frazione_tolta(str(tmp_path), str(rep))
    assert not ft["colonne_moltiplicatore"]
    pm = {(x["flusso"], x["metro"]): x for x in ft["per_metro"]}
    assert pm[("vittima", "0")]["mediana"] == pytest.approx(0.09) and pm[("vittima", "0")]["n_nuvole"] == 2
    assert pm[("vittima", "1")]["min"] == 0.05 and pm[("gemello", "0")]["mediana"] is None
    assert cs.frazione_tolta(str(tmp_path), str(tmp_path / "manca")) is None


def test_regola_crash_spenta_esiti_di_sempre(tmp_path):
    sf = str(tmp_path / "stato.json")
    cs.inizia(sf, "nsga3_eps5", "1001", "x")
    assert cs.chiudi(sf, "nsga3_eps5", "1001", {"esito": "fatta", "motivo": "attacco scoperto"}) == "fatta"
    assert "Crash da attacco" not in cs.riepilogo(cs.leggi(sf), per_run=False)


# ----------------------------------------------------------------------
# Queue dry run: variables on, off and wrong
# ----------------------------------------------------------------------

def _a_secco(tmp_path, cond, radice_nome="radice_limiti", **var):
    rad = tmp_path / radice_nome
    env = {k: v for k, v in os.environ.items()
           if k not in ("MOLTIPLICATORE_LIMITI", "FAMIGLIE_LIMITI", "BLOCCA_G14", "REGOLA_CRASH_LIMITI")}
    for riga in AMBIENTE_CODA.read_text().splitlines():
        if "=" in riga and not riga.startswith("#"):
            k, v = riga.split("=", 1)
            env[k] = v
    env["RADICE"] = str(rad)
    env.update({k: str(v) for k, v in var.items()})
    r = subprocess.run(["bash", "scripts/coda_serieB.sh", "--a-secco", cond], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert not rad.exists()
    return r.stdout


# failed checks that concern the limit study (others, such as the watchdog or a run in
# progress, depend on the machine state)
CHIAVI_LIMITI = ("LIMITI", "BLOCCA_G14", "REGOLA_CRASH", "RADICE", "GEN_MAX", "moltiplicatore", "blocca-gene",
                 "regola-crash", "famiglie", "processo_mola_vittima")


def _falliti(out):
    return [ln.strip() for ln in out.splitlines()
            if "CONTROLLO FALLITO" in ln and any(c in ln for c in CHIAVI_LIMITI)]


def _riga(out, inizio):
    """First line starting with inizio, without the printf %q escapes (spaces and commas)."""
    r = next(ln.strip() for ln in out.splitlines() if ln.strip().startswith(inizio))
    return r.replace("\\ ", " ").replace("\\,", ",")


coda_pronta = pytest.mark.skipif(not AMBIENTE_CODA.exists(), reason="ambiente della coda assente")


@coda_pronta
def test_coda_spenta_nessuna_traccia(tmp_path):
    for var in ({}, {"MOLTIPLICATORE_LIMITI": "1.0", "FAMIGLIE_LIMITI": "rumore", "BLOCCA_G14": "0",
                     "REGOLA_CRASH_LIMITI": "0"}):
        out = _a_secco(tmp_path, "nsga3_senzabudget", **var)
        env = _riga(out, "+ env CAMPAIGN_DIR")
        assert "--moltiplicatore-limiti" not in out and "--blocca-gene" not in out and "_limx" not in out
        assert " RUN_SUFFIX=_coda_N5_dmin140 " in env and "studio dei limiti" not in out
        assert "regola-crash" not in out and "REGOLA_CRASH" not in out and not _falliti(out)


@coda_pronta
def test_coda_accesa(tmp_path):
    out = _a_secco(tmp_path, "nsga3_senzabudget", radice_nome="radice_limiti_x4", MOLTIPLICATORE_LIMITI="4",
                   FAMIGLIE_LIMITI="dropout,rumore,geometria,strutturale", BLOCCA_G14="1")
    assert not _falliti(out), _falliti(out)
    env = _riga(out, "+ env CAMPAIGN_DIR")
    assert " RUN_SUFFIX=_limx4_rumore-geometria-strutturale-dropout_g14bloccato_coda_N5_dmin140 " in env
    nodo_x = env.split(" NODE_EXTRA=")[1].split(" ORCH_EXTRA=")[0]
    assert nodo_x.endswith("--moltiplicatore-limiti 4 --famiglie-limiti rumore,geometria,strutturale,dropout")
    orch_x = env.split(" ORCH_EXTRA=")[1]
    assert ("--moltiplicatore-limiti 4 --famiglie-limiti rumore,geometria,strutturale,dropout "
            "--blocca-gene 14=-1 CONTROLLO_ETA=") in orch_x
    att = _riga(out, "+ appena esiste rep_01/comando.txt")
    for o in ("[--moltiplicatore-limiti 4]", "[--famiglie-limiti rumore,geometria,strutturale,dropout]",
              "[--blocca-gene 14=-1]"):
        assert o in att
    nodo = _riga(out, "+ nel comando del perturbation_node")
    assert "[--moltiplicatore-limiti 4] [--famiglie-limiti rumore,geometria,strutturale,dropout]" in nodo
    riga = _riga(out, "studio dei limiti:")
    assert riga == ("studio dei limiti: moltiplicatore x4 su rumore,geometria,strutturale,dropout (suffisso "
                    "_limx4_rumore-geometria-strutturale-dropout); g14 bloccato a -1 in tutti i candidati (suffisso "
                    "_g14bloccato)")
    assert "regola-crash" not in out


@coda_pronta
def test_coda_regola_crash_solo_a_genoma_fisso(tmp_path):
    gm = ",".join(str(int(v)) for v in _variante([14]))
    out = _a_secco(tmp_path, "massimo", radice_nome="radice_limiti_x4", MOLTIPLICATORE_LIMITI="4",
                   FAMIGLIE_LIMITI="rumore,geometria,strutturale,dropout", REGOLA_CRASH_LIMITI="1", GEN_MAX=gm)
    assert not _falliti(out), _falliti(out)
    assert "+ classifica con --regola-crash" in out and "crash_processo (REGOLA_CRASH_LIMITI=1)" in out
    assert _riga(out, "studio dei limiti:").endswith("regola del crash della vittima accesa (una ripetizione, "
                                                     "poi crash_da_attacco o guasto)")
    for cond in ("nsga3_senzabudget", "casuale_eps5", "randpuro_eps10"):
        out = _a_secco(tmp_path, cond, REGOLA_CRASH_LIMITI="1")
        assert any("REGOLA_CRASH_LIMITI=1 vale solo a genoma fisso" in f for f in _falliti(out)), cond


@coda_pronta
def test_coda_K_nel_nome_della_radice(tmp_path):
    casi = (("radice_limiti_x4", "4", True), ("radice_limiti", "4", False), ("radice_limiti_x40", "4", False),
            ("limiti/senza_deriva_x4", "4", True), ("radice_limiti_x4p5", "4", False),
            ("radice_limiti_x2p5", "2.5", True), ("radice_limiti_x2", "2.5", False),
            ("x4_limiti/senza_deriva", "4", False))     # K goes in the last folder
    for nome, k, ok in casi:
        out = _a_secco(tmp_path, "massimo", radice_nome=nome, MOLTIPLICATORE_LIMITI=k, FAMIGLIE_LIMITI="deriva")
        f = [x for x in _falliti(out) if "della RADICE deve contenere x" in x]
        assert (not f) == ok, (nome, k, f)
    # x1: no constraint on K
    assert not _falliti(_a_secco(tmp_path, "massimo", radice_nome="radice_limiti", BLOCCA_G14="0"))


@coda_pronta
def test_coda_x6_e_valori_non_interi(tmp_path):
    out = _a_secco(tmp_path, "massimo", radice_nome="radice_limiti_x6", MOLTIPLICATORE_LIMITI="6",
                   FAMIGLIE_LIMITI="geometria")
    assert not _falliti(out) and "RUN_SUFFIX=_limx6_geometria_coda_N5_dmin140 " in out
    assert _riga(out, "studio dei limiti:") == ("studio dei limiti: moltiplicatore x6 su geometria "
                                                "(suffisso _limx6_geometria)")
    out = _a_secco(tmp_path, "massimo", radice_nome="radice_limiti_x2p5", MOLTIPLICATORE_LIMITI="2.50",
                   FAMIGLIE_LIMITI="deriva")
    assert not _falliti(out) and "RUN_SUFFIX=_limx2p5_deriva_coda_N5_dmin140 " in out
    assert "[--moltiplicatore-limiti 2.5]" in out


@coda_pronta
@pytest.mark.parametrize("var,attesa", [
    ({"MOLTIPLICATORE_LIMITI": "0.5"}, "MOLTIPLICATORE_LIMITI: numero >= 1"),
    ({"MOLTIPLICATORE_LIMITI": "4x"}, "MOLTIPLICATORE_LIMITI: numero >= 1"),
    ({"MOLTIPLICATORE_LIMITI": "2", "FAMIGLIE_LIMITI": "rumore,pippo"}, "FAMIGLIE_LIMITI: fra"),
    ({"BLOCCA_G14": "si"}, "BLOCCA_G14: 0 o 1"),
    ({"REGOLA_CRASH_LIMITI": "2"}, "REGOLA_CRASH_LIMITI: 0 o 1"),
])
def test_coda_valori_sbagliati(tmp_path, var, attesa):
    out = _a_secco(tmp_path, "nsga3_senzabudget", **var)
    assert any(attesa in f for f in _falliti(out)), _falliti(out)
    if "MOLTIPLICATORE_LIMITI" in attesa or "FAMIGLIE_LIMITI" in attesa:
        assert "studio dei limiti: valori non validi" in out and "--moltiplicatore-limiti" not in out


@coda_pronta
def test_coda_radice_del_seme_e_massimo_con_g14(tmp_path):
    # the word of the dedicated root must not appear in the test name (and so in tmp_path)
    out = _a_secco(tmp_path, "nsga3_senzabudget", radice_nome="radice_seme_x2", MOLTIPLICATORE_LIMITI="2")
    assert any("RADICE senza 'limiti'" in f for f in _falliti(out))
    out = _a_secco(tmp_path, "massimo", BLOCCA_G14="1")
    assert any("il gene 14 di GEN_MAX deve essere -1" in f for f in _falliti(out))
    out = _a_secco(tmp_path, "massimo", BLOCCA_G14="1", GEN_MAX=",".join(
        str(int(v)) for v in _variante([14])))
    assert not _falliti(out) and "_g14bloccato_coda_N5" in out


def _normalizza(testo, radice):
    t = testo.replace(str(radice), "REPO")
    t = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "DATA", t)
    t = re.sub(r"battito vecchio di \d+ s", "battito vecchio di N s", t)
    t = re.sub(r"acceso da \d+ s", "acceso da N s", t)
    t = re.sub(r"md5 dei file: [0-9a-f]+", "md5 dei file: X", t)
    t = re.sub(r"disco \d+ GB liberi \(\d+% usato\)|disco: \d+ GB liberi", "disco", t)
    return t


@coda_pronta
@pytest.mark.skipif(not (ORIGINALE / "scripts/coda_serieB.sh").exists(), reason="copia originale assente")
@pytest.mark.parametrize("cond", ["massimo", "nsga3_senzabudget", "casuale_eps5"])
def test_coda_spenta_uguale_alla_copia_originale(tmp_path, cond):
    """Options off: dry-run output identical to that of the earlier scripts, apart from
    times, heartbeat and adapter age, free space and code md5."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MOLTIPLICATORE_LIMITI", "FAMIGLIE_LIMITI", "BLOCCA_G14", "REGOLA_CRASH_LIMITI")}
    for riga in AMBIENTE_CODA.read_text().splitlines():
        if "=" in riga:
            k, v = riga.split("=", 1)
            env[k] = v
    env["RADICE"] = str(tmp_path / "radice_limiti")
    uscite = []
    for radice in (ORIGINALE, ROOT):
        r = subprocess.run(["bash", "scripts/coda_serieB.sh", "--a-secco", cond], cwd=radice, env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0
        uscite.append(_normalizza(r.stdout, radice))
    assert uscite[0] == uscite[1]
