#!/usr/bin/env python3
"""
Generator fast_ops (node option --fast-ops): voxel grouping with a 1D key and blockwise
edge PCA must give the same clouds, bit for bit, as the default path. No Isaac or MOLA.

    python3 -m pytest tests/test_fast_ops_identita.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402


def _stanza(n, seed):
    """Walls, floor and corners with little noise: many edges close to each other."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    c = np.zeros((n, 4))
    a, b = n // 3, 2 * n // 3
    c[:a, 0] = t[:a] * 20 - 10
    c[:a, 1] = 5.0
    c[:a, 2] = rng.uniform(0, 3, a)
    c[a:b, 0] = -10.0
    c[a:b, 1] = t[a:b] * 10 - 5
    c[a:b, 2] = rng.uniform(0, 3, b - a)
    c[b:, :2] = rng.uniform(-10, 10, (n - b, 2))
    c[:, :3] += rng.normal(0, 0.003, (n, 3))
    c[:, 3] = rng.uniform(0, 255, n)
    return c


def _uguali(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _coppia(**kw):
    return PerturbationGenerator(**kw), PerturbationGenerator(fast_ops=True, **kw)


def test_default_spento():
    assert PerturbationGenerator().fast_ops is False


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_inverse_voxel_come_axis0(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-40, 40, (20000, 3))
    pts[::3] = pts[1::3][: len(pts[::3])]
    for vox in (0.2, 0.5, 1.3):
        keys = np.floor(pts / vox).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        assert _uguali(inv, PerturbationGenerator._voxel_inverse(keys))


def test_inverse_voxel_fuori_int64_ricade():
    big = np.array([[0, 0, 0], [2**40, 2**40, 1], [5, -2**40, 3]], dtype=np.int64)
    assert PerturbationGenerator._voxel_inverse(big) is None
    g0, g1 = _coppia()
    pts = big.astype(np.float64) * 0.5
    pts = np.repeat(pts, 40, axis=0)
    noise = np.random.default_rng(3).normal(0, 0.02, pts.shape)
    assert _uguali(g0._apply_spatial_correlation(pts, noise, 0.7),
                   g1._apply_spatial_correlation(pts, noise, 0.7))


@pytest.mark.parametrize("seed", [0, 5])
def test_correlazione_spaziale_identica(seed):
    g0, g1 = _coppia()
    c = _stanza(30000, seed)
    noise = np.random.default_rng(seed + 100).normal(0, 0.02, (len(c), 3))
    for corr in (0.3, 0.6, 1.0):
        assert _uguali(g0._apply_spatial_correlation(c, noise, corr),
                       g1._apply_spatial_correlation(c, noise, corr))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_covarianze_a_blocchi_come_np_cov(dtype):
    rng = np.random.default_rng(11)
    nb = (rng.normal(0, 1, (300, 10, 3)) * np.array([3.0, 0.1, 0.01]) + np.array([12.0, -4.0, 1.5])).astype(dtype)
    cov = PerturbationGenerator._cov_blocks(nb)
    assert cov.dtype == np.float64
    for j in range(len(nb)):
        centered = nb[j] - nb[j].mean(axis=0)
        assert _uguali(cov[j], np.cov(centered.T))


@pytest.mark.parametrize("seed", [0, 2, 9])
@pytest.mark.parametrize("strength", [0.15, 1.0])
def test_attacco_edge_identico(seed, strength):
    g0, g1 = _coppia()
    c = _stanza(20000, seed)
    d = np.random.default_rng(seed).normal(size=3)
    for direction in (d / np.linalg.norm(d), np.zeros(3), np.array([0.0, 0.0, 1.0])):
        p = {"edge_attack_strength": strength, "noise_direction": direction}
        np.random.seed(seed)
        a = g0._apply_edge_attack(c, p)
        np.random.seed(seed)
        b = g1._apply_edge_attack(c, p)
        assert _uguali(a, b)
        assert not _uguali(a, c)


@pytest.mark.parametrize("seed", [0, 2, 9])
def test_attacco_edge_identico_float32(seed):
    # float32 cloud (real scene scale, 20 m): without the conversion in _cov_blocks the
    # blockwise edge drifted by up to about 7e-6 m
    g0, g1 = _coppia()
    c = _stanza(20000, seed).astype(np.float32)
    d = np.random.default_rng(seed).normal(size=3)
    for strength in (0.15, 1.0):
        for direction in (d / np.linalg.norm(d), np.array([0.0, 0.0, 1.0])):
            p = {"edge_attack_strength": strength, "noise_direction": direction}
            np.random.seed(seed)
            a = g0._apply_edge_attack(c, p)
            np.random.seed(seed)
            b = g1._apply_edge_attack(c, p)
            assert a.dtype == np.float32
            assert _uguali(a, b)
            assert not _uguali(a, c)


@pytest.mark.parametrize("seed", range(3))
def test_pipeline_identica_float32(seed):
    kw = dict(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15,
              max_drift_lag_s=0.5, structural="gross")
    g0, g1 = _coppia(**kw)
    rng = np.random.default_rng(seed)
    c = _stanza(25000, seed).astype(np.float32)
    for _ in range(3):
        genome = rng.uniform(-1, 1, g0.get_genome_size())
        genome[3] = max(genome[3], 0.0)
        genome[11] = max(genome[11], 0.2)
        genome[13] = max(genome[13], 0.0)
        s = int(rng.integers(0, 2**31 - 1))
        a = g0.apply_perturbation(c, g0.encode_perturbation(genome), seed=s)
        b = g1.apply_perturbation(c, g1.encode_perturbation(genome), seed=s)
        assert _uguali(a, b)


@pytest.mark.parametrize("seed", range(6))
def test_pipeline_identica(seed):
    kw = dict(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15,
              max_drift_lag_s=0.5, structural="gross")
    g0, g1 = _coppia(**kw)
    rng = np.random.default_rng(seed)
    c = _stanza(25000, seed)
    for _ in range(4):
        genome = rng.uniform(-1, 1, g0.get_genome_size())
        # spatial correlation and edge attack on (genes 3, 11, 13)
        genome[3] = max(genome[3], 0.0)
        genome[11] = max(genome[11], 0.2)
        genome[13] = max(genome[13], 0.0)
        p = g0.encode_perturbation(genome)
        assert p["spatial_correlation"] >= 0.3 and p["edge_attack_strength"] > 0.1
        s = int(rng.integers(0, 2**31 - 1))
        a = g0.apply_perturbation(c, g0.encode_perturbation(genome), seed=s)
        b = g1.apply_perturbation(c, g1.encode_perturbation(genome), seed=s)
        assert _uguali(a, b)


def _args(**kw):
    a = dict(max_point_shift=0.05, noise_std=0.02, dropout_rate=0.03,
             max_drift_lag_s=0.5, structural="net")
    a.update(kw)
    return SimpleNamespace(**a)


@pytest.mark.parametrize("valore", [None, False, True])
def test_nodo_passa_l_opzione(monkeypatch, valore):
    pn = pytest.importorskip("src.nodes.perturbation_node")
    # rclpy Node.__init__ skipped: the two generators, built first, are enough; the rest of
    # __init__ (ROS) may fail.
    monkeypatch.setattr(pn.Node, "__init__", lambda self, *a, **k: None)
    args = _args() if valore is None else _args(fast_ops=valore)
    n = pn.PerturbationNode.__new__(pn.PerturbationNode)
    try:
        pn.PerturbationNode.__init__(n, args)
    except Exception:
        pass
    atteso = bool(valore)
    assert n.generator.fast_ops is atteso and n.victim_generator.fast_ops is atteso


def test_argomento_fast_ops(monkeypatch):
    pn = pytest.importorskip("src.nodes.perturbation_node")
    visti = []

    class _Fermo(Exception):
        pass

    def _nodo(cli):
        visti.append(cli)
        raise _Fermo()

    monkeypatch.setattr(pn.rclpy, "init", lambda *a, **k: None)
    monkeypatch.setattr(pn, "PerturbationNode", _nodo)
    for argv, atteso in ((["perturbation_node.py"], False),
                         (["perturbation_node.py", "--fast-ops"], True)):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(_Fermo):
            pn.main()
        assert visti[-1].fast_ops is atteso
