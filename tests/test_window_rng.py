"""Per-window generator of the random arm (--window-rng).

'sum' reproduces the genomes of the earlier runs (default_rng(seed + k)); 'seedseq' gives
different genomes to the (repetition, window) pairs that coincide with 'sum'
(seed 1000 + repetition). Sampling is the one of attack_orchestrator.py (random branch,
--random-sparse).
"""
import json
import os

import numpy as np
import pytest

DIRECTION_GENES = (0, 1, 2, 7, 8, 9)
N = 17
RIF = os.path.join(os.path.dirname(__file__), "..", "data", "attack", "campaign", "straight",
                   "random_g3_p1x1_lag0.5_sel2_H1.0_rif")


def genoma(seed, k, modo):
    rng = (np.random.default_rng(np.random.SeedSequence([seed, k])) if modo == "seedseq"
           else np.random.default_rng(seed + k))
    x = np.array([0.0 if g in DIRECTION_GENES else -1.0 for g in range(N)])
    kk = int(rng.integers(1, N + 1))
    idx = rng.choice(N, size=kk, replace=False)
    x[idx] = rng.uniform(-1.0, 1.0, size=kk)
    return x


def test_sum_coincide_con_le_ripetizioni_sfasate():
    assert np.allclose(genoma(1001, 1, "sum"), genoma(1002, 0, "sum"))
    assert np.allclose(genoma(1001, 2, "sum"), genoma(1003, 0, "sum"))


def test_seedseq_separa_le_ripetizioni():
    gs = {(s, k): genoma(s, k, "seedseq") for s in (1001, 1002, 1003) for k in range(8)}
    chiavi = list(gs)
    for i, a in enumerate(chiavi):
        for b in chiavi[i + 1:]:
            assert not np.allclose(gs[a], gs[b]), (a, b)


def test_seedseq_deterministico():
    assert np.allclose(genoma(1001, 3, "seedseq"), genoma(1001, 3, "seedseq"))


@pytest.mark.skipif(not os.path.isdir(RIF), reason="run random di riferimento assente")
def test_sum_riproduce_la_history_del_23_ottobre():
    for r in (1, 2, 3):
        with open(os.path.join(RIF, f"rep_0{r}", "history.json")) as f:
            for w in json.load(f):
                assert np.allclose(np.array(w["genome"]), genoma(1000 + r, w["window"], "sum"), atol=1e-12)
