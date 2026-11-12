#!/usr/bin/env python3
"""
Pareto front (maximum damage, minimum Chamfer) and the chosen point.

    python3 -m pytest tests/test_fronte_pareto.py -q    # with pytest
    python3 tests/test_fronte_pareto.py                  # without
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.fronte_pareto import (  # noqa: E402
    candidato_valido, fronte_pareto, indici_non_dominati)


def rec(damage, chamfer, status="ok", admissible=True, twin=False, counted=True,
        tau=0.3, slow=False, genome=None):
    """Minimal record like those in history.json['evaluations']."""
    return {"genome": genome if genome is not None else [damage, chamfer],
            "damage": damage, "chamfer": chamfer, "status": status,
            "admissible": admissible, "twin_misaligned": twin, "counted": counted,
            "tau_s": tau, "slow": slow}


def test_punto_dominato_escluso():
    # (0.10, 5) dominates (0.05, 8): more damage with less Chamfer
    fr, sc = fronte_pareto([rec(0.10, 5.0), rec(0.05, 8.0), rec(0.20, 20.0)])
    assert [p["idx"] for p in fr] == [0, 2]
    assert sc["idx"] == 2 and sc["damage"] == 0.20


def test_ordinamento_per_chamfer():
    fr, _ = fronte_pareto([rec(0.30, 30.0), rec(0.10, 5.0), rec(0.20, 15.0)])
    assert [p["chamfer"] for p in fr] == [5.0, 15.0, 30.0]


def test_pareggio_sul_danno_vince_chamfer_minore():
    fr, sc = fronte_pareto([rec(0.20, 20.0), rec(0.20, 12.0), rec(0.05, 1.0)])
    # (0.20, 20) is dominated by (0.20, 12)
    assert [p["idx"] for p in fr] == [2, 1]
    assert sc["idx"] == 1


def test_duplicati_identici_restano_entrambi_e_vince_indice_minore():
    fr, sc = fronte_pareto([rec(0.05, 1.0), rec(0.20, 12.0), rec(0.20, 12.0)])
    assert sorted(p["idx"] for p in fr) == [0, 1, 2]
    assert sc["idx"] == 1


def test_budget_esclude_il_massimo():
    recs = [rec(0.10, 5.0), rec(0.50, 80.0), rec(0.25, 40.0)]
    fr, sc = fronte_pareto(recs)
    assert sc["idx"] == 1
    fr_b, sc_b = fronte_pareto(recs, budget=70.0)
    assert [p["idx"] for p in fr_b] == [0, 2]
    assert sc_b["idx"] == 2
    # budget on the threshold: a candidate with Chamfer equal to the budget stays
    _, sc_eq = fronte_pareto(recs, budget=80.0)
    assert sc_eq["idx"] == 1


def test_record_non_validi_esclusi():
    recs = [
        rec(0.90, 1.0, status="stopped_early(0.43m)"),     # invalid outcome
        rec(0.80, 1.0, admissible=False),                  # not admissible
        rec(0.70, 1.0, twin=True),                         # twin not aligned
        rec(0.60, 1.0, counted=False),                     # measurement not counted
        rec(0.50, float("nan")),                           # non-finite Chamfer
        {"genome": [0.0], "dev": None, "chamfer": 2.0, "status": "ok",
         "counted": False},                                # dropped: no damage
        rec(0.10, 3.0, status="arrived", slow=True, tau=0.9),
    ]
    assert [candidato_valido(r) for r in recs] == [False] * 6 + [True]
    fr, sc = fronte_pareto(recs)
    assert [p["idx"] for p in fr] == [6]
    assert sc["idx"] == 6 and sc["slow"] is True and sc["tau_s"] == 0.9


def test_lenti_non_filtrati():
    fr, sc = fronte_pareto([rec(0.10, 5.0), rec(0.40, 30.0, slow=True, tau=1.2)])
    assert sc["idx"] == 1 and sc["slow"]


def test_fronte_vuoto():
    assert fronte_pareto([]) == ([], None)
    assert fronte_pareto([rec(0.1, 2.0, admissible=False)]) == ([], None)
    assert fronte_pareto([rec(0.1, 90.0)], budget=70.0) == ([], None)


def test_scelto_sta_sul_fronte():
    recs = [rec(0.1 * i, 3.0 * i + (i % 3)) for i in range(1, 12)] + [rec(0.3, 1.0)]
    fr, sc = fronte_pareto(recs)
    assert any(p["idx"] == sc["idx"] for p in fr)
    assert sc["damage"] == max(p["damage"] for p in fr)
    assert all(math.isfinite(p["chamfer"]) for p in fr)


def test_indici_non_dominati():
    assert indici_non_dominati([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == [0, 1, 2]
    assert indici_non_dominati([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == [2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} test passati")
