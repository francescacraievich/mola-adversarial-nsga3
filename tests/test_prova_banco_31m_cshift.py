#!/usr/bin/env python3
"""Check of how time_onLidar is matched to its scan: in the row of scan k MOLA writes the
duration of the previous scan (LidarOdometry_ProcessScan.cpp:542, :1627;
LidarOdometry_Main.cpp:744). Synthetic data with a known shift, generated here: no real numbers.

    python3 -m pytest tests/test_prova_banco_31m_cshift.py -q
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prova_banco_31m import ROOT, an, corsa  # noqa: E402

TRASP = 3.0      # ms between the end of onLidar and pose reception
D0 = 5.0         # ms, duration of the first-row scan (no ICP)


def scrivi_traccia_mola(path, righe, tronca=True):
    """CSV as MOLA writes it: trailing comma on each row; last row truncated on request."""
    with open(path, "w") as f:
        f.write('"icp_iterations","icp_quality","time_onLidar","timestamp",\n')
        for r in righe:
            f.write(",".join(f"{x:f}" for x in r) + ",\n")
        if tronca:
            f.write("3.000000,0.990000,0.0")


def test_durate_onlidar_riga_successiva(tmp_path):
    p = tmp_path / "traces.csv"
    d = {100: 11.0, 101: 354.0, 102: 12.0, 103: 13.0}
    # row k: duration of the previous scan (the first row, without ICP, holds 0)
    righe = [(0, 0, 0.0, 9.9)]
    prec = D0
    for k in sorted(d):
        righe.append((3, 0.99, prec / 1000.0, k / 10.0))
        prec = d[k]
    scrivi_traccia_mola(p, righe)
    elab, C = an.durate_onlidar(p)
    assert elab == {100, 101, 102, 103}
    assert C[100] == pytest.approx(11.0) and C[101] == pytest.approx(354.0) and C[102] == pytest.approx(12.0)
    assert 103 not in C                      # followed by a truncated row: no duration
    elab0, C0 = an.durate_onlidar(p, riga_successiva=False)
    assert elab0 == elab and C0[101] == pytest.approx(11.0) and C0[100] == pytest.approx(D0)


def durata(k, k_ep):
    return 354.0 if k == k_ep else 10.0 + 30.0 * (k % 2)


def sfasa(spec, k_ep=None):
    """Rewrites traces and poses of a synthetic run: pose received TRASP ms after
    the end of onLidar, which lasts durata(k); the trace row of k holds the
    duration of the previous scan.
    """
    cfg, rep, fl, inv = spec
    rep, inv = Path(rep), Path(inv)
    F = list(csv.DictReader(open(fl)))
    tp = {chi: {an.slot(r["stamp_victim"]): float(r[f"t_pub_{chi}_end_mono"]) for r in F}
          for chi in ("victim", "attacker")}
    for chi in ("victim", "attacker"):
        p = rep / "logs" / f"traces_{chi}.csv"
        ks = [an.slot(r["timestamp"]) for r in csv.DictReader(open(p)) if float(r["timestamp"]) > 0]
        righe, prec = [(0, 0, 0.0, (ks[0] - 1) / 10.0)], D0
        for k in ks:
            righe.append((3, 0.99, prec / 1000.0, k / 10.0))
            prec = durata(k, k_ep)
        scrivi_traccia_mola(p, righe)
    P = list(csv.reader(open(inv / "pose.csv")))
    for r in P[1:]:
        k = an.slot(r[1])
        r[3] = repr(tp[r[0]][k] + (TRASP + durata(k, k_ep)) / 1000.0)
    with open(inv / "pose.csv", "w", newline="") as f:
        csv.writer(f).writerows(P)


def _quattro_sfasate(tmp, k_ep):
    spec = []
    for i, cfg in enumerate("XYXY"):
        s, _ = corsa(tmp, f"c{i + 1}_{cfg}", cfg, mono0=1000.0 + 5000 * i, k0=1000 + 10000 * i)
        sfasa(s, k_ep if i == 0 else None)
        spec.append(s)
    return spec


def test_31m_eta_e_coda_con_sfasamento(tmp_path):
    k_ep = 1000 + 3 * 60 + 20          # moving (window 3, position 20) in the first run X
    spec = _quattro_sfasate(tmp_path, k_ep)
    nuovo = an.analizza(spec, tasso31_sup_x=0.05)
    vecchio = an.analizza(spec, tasso31_sup_x=0.05, c_riga_successiva=False)
    v1, v0 = nuovo["corse"][0]["victim"], vecchio["corse"][0]["victim"]
    assert v1["n_S"] == v0["n_S"] == 275 and v1["perse"] == v0["perse"] == 11
    # c from the next row: A = TRASP for every scan, no tail
    A1 = [r["A"] for r in v1["_recs"] if r["A"] is not None]
    assert len(A1) == v1["n_L_e2e"] and all(a == pytest.approx(TRASP, abs=1e-6) for a in A1)
    assert v1["B_ge2"] == 0 and v1["B_eq1"] == 0
    ep = next(r for r in v1["_recs"] if r["k"] == k_ep)
    assert ep["c"] == pytest.approx(354.0) and ep["A"] == pytest.approx(TRASP, abs=1e-6) and ep["B"] == 0
    # c from the same row: negative A and B >= 2 over the episode (the known artefact)
    assert sum(1 for r in v0["_recs"] if r["A"] is not None and r["A"] < 0) > 100
    ep0 = next(r for r in v0["_recs"] if r["k"] == k_ep)
    assert ep0["B"] == 2 and v0["B_ge2"] == 1
    assert vecchio["verdetto"]["righe"][5]["ok"] is False and nuovo["verdetto"]["righe"][5]["ok"] is True


def test_31m_cli_regola_c(tmp_path):
    spec = _quattro_sfasate(tmp_path, 1000 + 3 * 60 + 20)
    args = [sys.executable, str(ROOT / "scripts" / "prova_banco_serieB_31m_analisi.py")]
    for s in spec:
        args += ["--corsa", *s]
    r = subprocess.run(args + ["--tasso31-sup-X", "0.05", "--out", str(tmp_path / "a")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "c (time_onLidar): riga successiva" in r.stdout
    r = subprocess.run(args + ["--tasso31-sup-X", "0.05", "--out", str(tmp_path / "b"), "--c-stesso-slot"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "c (time_onLidar): stessa riga" in r.stdout
    assert json.loads((tmp_path / "b" / "analisi_31m.json").read_text())["verdetto"]["righe"]["5"]["ok"] is False
    assert json.loads((tmp_path / "a" / "analisi_31m.json").read_text())["verdetto"]["righe"]["5"]["ok"] is True
