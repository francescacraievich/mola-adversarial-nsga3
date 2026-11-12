#!/usr/bin/env python3
"""Structural perceptibility term, --structural {net,gross}, without Isaac or MOLA.

    python3 -m pytest tests/test_strutturale_P15.py -q
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402

CLOUDS = os.environ.get("TEST_CLOUDS_DIR", str(ROOT / "data" / "attack" / "verifica" / "clouds"))


def nuvola(n=6000, seed=2):
    rng = np.random.default_rng(seed)
    p = np.c_[rng.uniform(-10, 10, n), rng.uniform(-5, 5, n), rng.uniform(-0.5, 2.0, n)]
    return np.hstack([p, np.full((n, 1), 100.0)])


def genoma(dropout=-1.0, fantasmi=-1.0, rumore=-1.0):
    g = np.full(17, -1.0)
    g[0:3] = [0.3, 0.9, 0.1]
    g[7:10] = [0.8, -0.5, 0.2]
    g[3] = rumore
    g[5] = dropout
    g[6] = fantasmi
    return g


def P_atteso(ch_cm, r):
    return float(np.sqrt(ch_cm ** 2 + (20.0 * r) ** 2))


def test_forma_sconosciuta_rifiutata():
    with pytest.raises(ValueError):
        PerturbationGenerator(structural="lordo")
    assert PerturbationGenerator().structural == "net"


def test_uscita_del_generatore_indipendente_dalla_forma():
    c = nuvola()
    g = genoma(dropout=1.0, fantasmi=1.0, rumore=0.5)
    a = PerturbationGenerator(max_dropout_rate=0.15)
    b = PerturbationGenerator(max_dropout_rate=0.15, structural="gross")
    oa = a.apply_perturbation(c, a.encode_perturbation(g), seed=11)
    ob = b.apply_perturbation(c, b.encode_perturbation(g), seed=11)
    assert np.array_equal(oa, ob)


def test_net_invariato_e_gross_con_tolti_e_aggiunti():
    c = nuvola()
    g = genoma(dropout=1.0, fantasmi=1.0)
    net = PerturbationGenerator(max_dropout_rate=0.15)
    gro = PerturbationGenerator(max_dropout_rate=0.15, structural="gross")
    on = net.apply_perturbation(c, net.encode_perturbation(g), seed=5)
    og = gro.apply_perturbation(c, gro.encode_perturbation(g), seed=5)
    cnt = gro.point_counts(c, og)
    N = len(c)
    m1 = N - cnt["n_removed"]
    assert cnt["n_removed"] > 0 and cnt["n_added"] == int(m1 * 0.02)   # g = 0.02 at gene +1
    assert len(og) == N - cnt["n_removed"] + cnt["n_added"]
    r_net = abs(len(og) - N) / N
    r_gross = (cnt["n_removed"] + cnt["n_added"]) / N
    assert r_gross > r_net
    ch = float(np.sqrt(gro.compute_chamfer_distance(c, og) * 10000))
    # net: unchanged formula
    assert float(net.compute_perturbation_magnitude(c, on, {})) == pytest.approx(P_atteso(ch, r_net), rel=1e-12)
    pg = float(gro.compute_perturbation_magnitude(c, og, {}))
    assert pg == pytest.approx(P_atteso(ch, r_gross), rel=1e-12)
    t = gro.compute_perturbation_terms(c, og, {})
    assert t["magnitude_cm"] == pg
    assert t["point_ratio"] == r_net and t["point_ratio_gross"] == pytest.approx(r_gross)
    assert t["structural_cm"] == pytest.approx(20.0 * r_gross) and t["structural"] == "gross"
    assert t["n_removed"] == cnt["n_removed"] and t["n_added"] == cnt["n_added"]
    tn = net.compute_perturbation_terms(c, on, {})
    assert tn["structural_cm"] == pytest.approx(20.0 * r_net) and tn["point_ratio_gross"] == pytest.approx(r_gross)


def test_net_identico_alla_formula_della_serie_A():
    # same expression as compute_perturbation_magnitude without the gross term
    c = nuvola()
    gen = PerturbationGenerator(max_dropout_rate=0.15)
    for g in (genoma(dropout=1.0, fantasmi=1.0, rumore=0.3), genoma(rumore=0.3), genoma(fantasmi=1.0)):
        o = gen.apply_perturbation(c, gen.encode_perturbation(g), seed=21)
        cd = gen.compute_chamfer_distance(c, o)
        vecchio = np.sqrt(np.sqrt(cd * 10000) ** 2 + (abs(len(o) - len(c)) / max(len(c), 1) * 20.0) ** 2)
        assert gen.compute_perturbation_magnitude(c, o, {}) == vecchio
        assert gen.compute_perturbation_terms(c, o, {})["magnitude_cm"] == float(vecchio)


def test_tolti_uguali_agli_aggiunti(monkeypatch):
    # M = N: net zero, gross 2k/N
    c = nuvola(n=2000)
    gro = PerturbationGenerator(structural="gross")
    net = PerturbationGenerator()
    k = 40
    for gen in (gro, net):
        monkeypatch.setattr(gen, "_apply_dropout", lambda p, n, w, prm: p[k:])
        monkeypatch.setattr(gen, "_add_ghost_points", lambda p, prm: np.vstack([p, p[:k] + 0.03]))
    g = genoma()
    og = gro.apply_perturbation(c, gro.encode_perturbation(g), seed=1)
    on = net.apply_perturbation(c, net.encode_perturbation(g), seed=1)
    assert len(og) == len(c)
    ch = float(np.sqrt(gro.compute_chamfer_distance(c, og) * 10000))
    t = gro.compute_perturbation_terms(c, og, {})
    assert t["point_ratio"] == 0.0 and t["point_ratio_gross"] == pytest.approx(2 * k / len(c))
    assert t["magnitude_cm"] == pytest.approx(P_atteso(ch, 2 * k / len(c)))
    assert float(net.compute_perturbation_magnitude(c, on, {})) == pytest.approx(ch)


def test_solo_dropout_o_solo_fantasmi_net_uguale_gross():
    c = nuvola()
    for g in (genoma(dropout=1.0), genoma(fantasmi=1.0)):
        gen = PerturbationGenerator(max_dropout_rate=0.15, structural="gross")
        o = gen.apply_perturbation(c, gen.encode_perturbation(g), seed=3)
        t = gen.compute_perturbation_terms(c, o, {})
        assert len(o) != len(c)
        assert t["point_ratio_gross"] == pytest.approx(t["point_ratio"], abs=1e-15)


def test_conteggi_solo_per_l_uscita_del_generatore():
    c = nuvola()
    gen = PerturbationGenerator(max_dropout_rate=0.15, structural="gross")
    o = gen.apply_perturbation(c, gen.encode_perturbation(genoma(dropout=1.0, fantasmi=1.0)), seed=4)
    assert gen.point_counts(c, o) is not None
    assert gen.point_counts(c, o.copy()) is None            # another cloud, even if equal
    assert gen.point_counts(c[:-1], o) is None              # different input
    altra = c[:-30]                                          # no generator: removed only
    t = gen.compute_perturbation_terms(c, altra, {})
    assert t["point_ratio_gross"] is None and t["n_removed"] is None
    assert t["structural_cm"] == pytest.approx(20.0 * 30 / len(c))   # fallback to net
    assert PerturbationGenerator(structural="gross").point_counts(c, o) is None


@pytest.mark.skipif(not os.path.exists(os.path.join(CLOUDS, "cloud_0.npy")), reason="nuvole registrate assenti")
def test_nuvola_registrata_conteggi_coerenti():
    a = np.load(os.path.join(CLOUDS, "cloud_0.npy"))
    c = np.hstack([a[:, :3], np.full((len(a), 1), 100.0)])
    gen = PerturbationGenerator(max_dropout_rate=0.15, structural="gross")
    g = genoma(dropout=0.2, fantasmi=0.6)                    # d = 0.03, g = 0.016
    o = gen.apply_perturbation(c, gen.encode_perturbation(g), seed=7)
    t = gen.compute_perturbation_terms(c, o, {})
    d = 0.6 * 0.05
    delta = t["n_removed"] / len(c)
    assert d - 0.005 <= delta <= 2 * d + 0.005               # removed fraction in [d, 2d]
    assert t["n_added"] == int((len(c) - t["n_removed"]) * 0.8 * 0.02)
    assert t["point_ratio_gross"] - t["point_ratio"] == pytest.approx(
        2 * min(t["n_removed"], t["n_added"]) / len(c))


# ---------------------------------------------------------------------------
# Node, orchestrator, campaign
# ---------------------------------------------------------------------------

def _trs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("trs", ROOT / "tests" / "test_registro_serieB.py")
    trs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trs)
    return trs


def test_status_del_nodo_con_la_forma_e_i_due_rapporti():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    trs = _trs()
    n, st = trs._nodo_finto(True)
    n.generator = PerturbationGenerator(structural="gross")
    n._perc_terms = dict(n._perc_terms, point_ratio_gross=0.09, n_removed=3000, n_added=900)
    pn.PerturbationNode._publish_status(n)
    assert st[0]["structural"] == "gross"
    assert st[0]["point_ratio"] == 0.075 and st[0]["point_ratio_gross"] == 0.09
    assert st[0]["n_pts_removed"] == 3000 and st[0]["n_pts_added"] == 900
    assert st[0]["n_added"] == 3 and st[0]["n_removed"] == 4     # 5 cm metrics unchanged
    # without --perc-terms: only the shape, no ratio
    n, st = trs._nodo_finto(False)
    pn.PerturbationNode._publish_status(n)
    assert st[0]["structural"] == "net" and "point_ratio_gross" not in st[0]
    # terms with the attack off
    assert pn.ZERO_TERMS["point_ratio_gross"] == 0.0 and pn.ZERO_TERMS["n_removed"] == 0


def test_magnitudine_del_nodo_in_gross():
    pn = pytest.importorskip("src.nodes.perturbation_node")
    c = nuvola()
    gen = PerturbationGenerator(max_dropout_rate=0.15, structural="gross")
    o = gen.apply_perturbation(c, gen.encode_perturbation(genoma(dropout=1.0, fantasmi=1.0)), seed=9)
    on = SimpleNamespace(_perc_terms_on=True, _perc_terms=None)
    off = SimpleNamespace(_perc_terms_on=False, _perc_terms=None)
    p_on = pn.PerturbationNode._magnitude(on, gen, c, o, {})
    p_off = pn.PerturbationNode._magnitude(off, gen, c, o, {})
    assert p_on == p_off
    assert on._perc_terms["point_ratio_gross"] > on._perc_terms["point_ratio"]


def test_argomento_del_nodo(monkeypatch):
    # The parser is read by stopping main() before rclpy.init: no ROS node
    # is created (double guard: parse_known_args and rclpy.init).
    pn = pytest.importorskip("src.nodes.perturbation_node")
    import argparse
    seen = {}
    real = argparse.ArgumentParser.parse_known_args

    class Fermo(Exception):
        pass

    def parse(self, *a, **kw):
        seen["ns"] = real(self, [])[0]
        seen["gross"] = real(self, ["--structural", "gross"])[0]
        with pytest.raises(SystemExit):
            real(self, ["--structural", "lordo"])
        raise Fermo()

    def init_vietato(*a, **kw):
        raise RuntimeError("rclpy.init non deve essere chiamato dal test")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", parse)
    monkeypatch.setattr(pn.rclpy, "init", init_vietato)
    with pytest.raises(Fermo):
        pn.main()
    assert seen["ns"].structural == "net" and seen["gross"].structural == "gross"


def test_orchestratore_registra_forma_e_campioni():
    ao = pytest.importorskip("src.optimization.attack_orchestrator")
    n = SimpleNamespace(_n_cb=0, frames_out=0, node_frame_idx=-1, proc_ms_roll=[], chamfer=[],
                        perc_samples=[], victim_stamp_offset=0.0, node_gen_limits=None, node_structural=None)
    d = {"enabled": True, "victim_stream": False, "chamfer_cm": 3.0, "shift_frac": 0.1,
         "n_added": 1, "n_removed": 2, "chamfer_pure_cm": 2.0, "structural_cm": 1.0,
         "point_ratio": 0.05, "point_ratio_gross": 0.08, "n_pts_removed": 2900, "n_pts_added": 700,
         "structural": "gross", "drift_cm": 0.0}
    ao.AttackNode._status_cb(n, SimpleNamespace(data=json.dumps(d)))
    assert n.node_structural == "gross"
    p = n.perc_samples[0]
    assert p["point_ratio_gross"] == 0.08 and p["n_pts_removed"] == 2900 and p["n_pts_added"] == 700
    f = ao.structural_fields(n, SimpleNamespace(structural="gross"))
    assert f == {"node_structural": "gross", "structural_expected": "gross", "structural_ok": True}
    assert ao.structural_fields(n, SimpleNamespace(structural="net"))["structural_ok"] is False
    vecchio = SimpleNamespace(node_structural=None)       # node without the key: net
    assert ao.structural_fields(vecchio, SimpleNamespace(structural="net"))["structural_ok"] is True
    assert ao.structural_fields(vecchio, SimpleNamespace(structural=None))["structural_ok"] is True


def test_argomento_dell_orchestratore():
    src = (ROOT / "src" / "optimization" / "attack_orchestrator.py").read_text()
    assert 'ap.add_argument("--structural", choices=["net", "gross"], default=None,' in src
    assert "**structural_fields(node, args)," in src


def test_campagna_structural(tmp_path):
    trs = _trs()
    o = trs._blocco_campagna(tmp_path, {})
    assert o["rc"] == 0 and o["DIR"] == "random_g3_p32x16_sel2_warm" and o["FLAGS"] == ""
    o = trs._blocco_campagna(tmp_path, {"STRUCTURAL": "net"})
    assert o["rc"] == 0 and o["DIR"] == "random_g3_p32x16_sel2_warm" and "--structural net" in o["FLAGS"]
    o = trs._blocco_campagna(tmp_path, {"STRUCTURAL": "gross"})
    assert o["rc"] == 0 and o["DIR"] == "random_g3_p32x16_sel2_warm_sgross"
    assert "--structural gross" in o["FLAGS"]
    o = trs._blocco_campagna(tmp_path, {"SERIE_B": "1", "STRUCTURAL": "gross"}, arm="nsga3")
    assert o["DIR"].endswith("_serieB_s1001-2001-3001_pt1_opinsieme_sgross")
    assert trs._blocco_campagna(tmp_path, {"STRUCTURAL": "lordo"})["rc"] == 1
    # the node receives the same shape
    src = (ROOT / "scripts" / "run_campaign.sh").read_text()
    assert '[[ -n "$STRUCTURAL" ]] && extra="$extra --structural $STRUCTURAL"' in src
