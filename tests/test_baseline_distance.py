#!/usr/bin/env python3
"""
Point-to-polyline distance and damage relative to the baseline.

    python3 -m pytest tests/ -q                # with pytest
    python3 tests/test_baseline_distance.py    # without
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.baseline_distance import (  # noqa: E402
    baseline_damage, dist_to_polyline, load_baseline)

RETTA = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
ELLE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])


def test_punto_sulla_retta():
    d = dist_to_polyline([[1.5, 0.0], [0.0, 0.0], [3.0, 0.0]], RETTA)
    assert np.allclose(d, 0.0)


def test_spostato_di_dieci_cm():
    d = dist_to_polyline([[1.5, 0.10], [0.7, -0.10]], RETTA)
    assert np.allclose(d, 0.10)


def test_oltre_la_fine_distanza_dall_estremo():
    # past the last point: distance from (3,0), not from the extended line
    d = dist_to_polyline([[3.5, 0.0], [3.3, 0.4]], RETTA)
    assert np.allclose(d, [0.5, 0.5])
    # before the first point: distance from (0,0)
    d = dist_to_polyline([[-1.0, 0.0], [-0.3, -0.4]], RETTA)
    assert np.allclose(d, [1.0, 0.5])


def test_polilinea_a_elle():
    # on the first segment, on the second, at the inner corner, at the outer corner
    d = dist_to_polyline([[1.0, 0.2], [2.3, 1.0], [1.9, 0.1], [2.5, -0.5]], ELLE)
    assert np.allclose(d, [0.2, 0.3, 0.1, math.hypot(0.5, 0.5)])


def test_punti_duplicati():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    d = dist_to_polyline([[1.0, 0.3], [0.5, 0.0], [2.5, 0.0]], poly)
    assert np.allclose(d, [0.3, 0.0, 0.5])
    # all vertices coincident: distance from the point
    d = dist_to_polyline([[3.0, 4.0]], np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert np.allclose(d, 5.0)


def test_polilinea_di_un_solo_punto():
    d = dist_to_polyline([[3.0, 4.0], [0.0, 0.0]], np.array([[0.0, 0.0]]))
    assert np.allclose(d, [5.0, 0.0])


def test_baseline_damage():
    cand = np.array([[0.5, 0.0], [1.0, 0.1], [1.5, 0.3], [3.5, 0.0]])
    r = baseline_damage(cand, RETTA)
    assert math.isclose(r["dev_end"], 0.5)
    assert math.isclose(r["dev_max"], 0.5)
    assert math.isclose(r["dev_mean"], (0.0 + 0.1 + 0.3 + 0.5) / 4)
    r1 = baseline_damage([[0.0, 0.2]], RETTA)
    assert r1["dev_end"] == r1["dev_mean"] == r1["dev_max"] == 0.2


def test_pose_non_finite_scartate():
    # first row NaN: dropped, the rest counts as if it were not there
    cand = np.array([[np.nan, np.nan], [1.0, 0.1], [2.0, 0.3], [3.0, 0.0]])
    r = baseline_damage(cand, RETTA)
    assert math.isclose(r["dev_end"], 0.0)
    assert math.isclose(r["dev_max"], 0.3)
    assert math.isclose(r["dev_mean"], (0.1 + 0.3 + 0.0) / 3)
    # inf and NaN on a single coordinate too
    cand = np.array([[1.0, np.inf], [np.nan, 0.0], [1.5, 0.1]])
    r = baseline_damage(cand, RETTA)
    assert math.isclose(r["dev_mean"], 0.1) and math.isclose(r["dev_end"], 0.1)
    # no finite pose: explicit NaN on all three
    r = baseline_damage(np.array([[np.nan, 0.0], [np.inf, 1.0]]), RETTA)
    assert all(math.isnan(r[k]) for k in ("dev_end", "dev_mean", "dev_max"))


def test_load_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "baseline.csv"
        p.write_text("s_m,x_m,y_m\n# commento\n0.0,0.0,0.0\n1.0,1.0,0.0\n2.0,2.0,0.0\n")
        poly = load_baseline(p)
        assert poly.shape == (3, 2)
        assert np.allclose(poly, [[0, 0], [1, 0], [2, 0]])
        # comment before the header and empty line at the end
        p.write_text("# prodotto da analisi\ns_m,x_m,y_m\n0.0,-5.99,-1.0\n0.5,-5.49,-1.0\n\n")
        assert np.allclose(load_baseline(p), [[-5.99, -1.0], [-5.49, -1.0]])
        for txt in ("s_m,x_m,y_m\n", "s_m,x_m\n0,0\n1,1\n", "s_m,x_m,y_m\n0,a,b\n",
                    "s_m,x_m,y_m\n1,0,0\n0,1,0\n"):
            p.write_text(txt)
            try:
                load_baseline(p)
            except ValueError:
                pass
            else:
                raise AssertionError(f"atteso ValueError per {txt!r}")
        try:
            load_baseline(Path(tmp) / "manca.csv")
        except ValueError:
            pass
        else:
            raise AssertionError("atteso ValueError per file assente")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} test superati")
