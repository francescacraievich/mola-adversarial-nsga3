#!/usr/bin/env python3
"""
Tempo di assestamento dello stato interno di MOLA, per componente.

Legge le tracce per scan prodotte da mola-cli con MOLA_SAVE_DEBUG_TRACES
(file logs/traces_*.csv di una run) e riporta, per ciascun file, l'evoluzione
delle grandezze che definiscono lo stato interno:

  icp_quality              qualita' del matching; se e' gia' alta al secondo
                           scan, il transitorio non e' in ICP.

  ADAPTIVE_THRESHOLD_SIGMA soglia adattiva, aggiornata con alpha=0.95: ogni
                           scan la sposta del 5% verso il valore corrente,
                           quindi servono circa 20 scan per assestarsi. Parte
                           dal valore iniziale del pipeline YAML.

  vx, wz                   twist stimato; converge solo con il robot in moto,
                           perche' da fermo il modello di velocita' impara
                           velocita' zero e sbaglia la prima predizione alla
                           partenza.

Il confronto fra i tempi di assestamento indica se un riscaldamento a robot
fermo basta o se serve un tratto in movimento.

Uso:
    python3 src/analysis/analyse_mola_traces.py data/attack/<run>/logs/traces_w0_nominal.csv
    python3 src/analysis/analyse_mola_traces.py data/attack/<run>/logs/traces_*.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for k in rows[0]:
        k2 = k.strip().strip('"')
        if not k2:
            continue
        vals = []
        for r in rows:
            v = r[k]
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k2] = np.array(vals)
    return out


def settle_index(x, tol_frac=0.05, window=5):
    """Primo indice da cui x resta entro tol_frac del suo valore finale.

    Misura in scan, non in secondi: e' l'unita' in cui MOLA aggiorna lo stato
    e rende il numero confrontabile fra rollout di durata diversa.
    """
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if good.sum() < window + 2:
        return None
    x = x[good]
    final = np.median(x[-window:])
    scale = max(abs(final), 1e-6)
    for i in range(len(x) - window):
        if np.all(np.abs(x[i:i + window] - final) / scale <= tol_frac):
            return i
    return None


def analyse(path):
    d = load(path)
    if d is None:
        print(f"{path}: vuoto")
        return

    n = len(next(iter(d.values())))
    t = d.get("current_relative_timestamp", np.arange(n) * 0.1)

    print(f"\n{'=' * 74}")
    print(f"  {Path(path).name}   {n} scan, {t[-1] - t[0]:.1f}s")
    print(f"{'=' * 74}")

    q = d.get("icp_quality")
    sig = d.get("ADAPTIVE_THRESHOLD_SIGMA")
    vx = d.get("vx")
    wz = d.get("wz")
    it = d.get("icp_iterations")

    # Il primo scan non ha nulla con cui accoppiarsi: qualita' 0 per costruzione,
    # escluso dalle statistiche.
    if q is not None and len(q) > 2:
        print(f"  icp_quality   scan1 {q[0]:.3f}  scan2 {q[1]:.3f}  "
              f"mediana {np.nanmedian(q[1:]):.3f}  min {np.nanmin(q[1:]):.3f}")
        bad = int((q[1:] < 0.5).sum())
        if bad:
            print(f"    {bad} scan sotto 0.5 di qualita'")
        else:
            print("    ICP converge bene da subito: il transitorio non e' qui")

    if sig is not None:
        i = settle_index(sig)
        print(f"  sigma         iniziale {sig[0]:.4f}  finale {sig[-1]:.4f}  "
              f"escursione {np.nanmax(sig) - np.nanmin(sig):.4f}")
        if np.nanmax(sig) - np.nanmin(sig) < 1e-4:
            print("    costante: non si sta adattando in questo rollout")
        elif i is not None:
            print(f"    assestato dopo ~{i} scan ({i * 0.1:.1f}s di sim-time)")

    if vx is not None:
        i = settle_index(vx, tol_frac=0.15)
        print(f"  vx            iniziale {vx[0]:+.4f}  finale {vx[-1]:+.4f} m/s")
        if i is not None:
            print(f"    assestato dopo ~{i} scan ({i * 0.1:.1f}s)")
        else:
            print("    non si assesta entro il rollout")

    if wz is not None:
        print(f"  wz            finale {wz[-1]:+.4f} rad/s   "
              f"max |wz| {np.nanmax(np.abs(wz)):.4f}")

    if it is not None and len(it) > 1:
        print(f"  iterazioni ICP  mediana {np.nanmedian(it[1:]):.0f}  "
              f"max {np.nanmax(it):.0f}")

    tw = d.get("twistCorrectionCount")
    if tw is not None and np.nanmax(tw) > 0:
        print(f"  correzioni twist  mediana {np.nanmedian(tw):.1f}  "
              f"max {np.nanmax(tw):.0f}")

    # Andamento nei primi scan, dove si concentra il transitorio.
    print(f"\n  {'scan':>5} {'t':>6} {'quality':>8} {'sigma':>8} "
          f"{'vx':>8} {'wz':>8} {'x':>8} {'yaw':>8}")
    show = min(n, 20)
    for i in range(show):
        print(f"  {i:>5} {t[i]:6.2f} "
              f"{q[i] if q is not None else float('nan'):8.3f} "
              f"{sig[i] if sig is not None else float('nan'):8.4f} "
              f"{vx[i] if vx is not None else float('nan'):+8.4f} "
              f"{wz[i] if wz is not None else float('nan'):+8.4f} "
              f"{d.get('robot_x', np.zeros(n))[i]:+8.4f} "
              f"{d.get('robot_yaw', np.zeros(n))[i]:+8.4f}")
    if n > show:
        print(f"  ... {n - show} scan successivi")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    args = ap.parse_args()
    for p in args.traces:
        analyse(p)
    print("\nCome leggerlo:")
    print("  Se icp_quality e' alta dal secondo scan e sigma resta costante,")
    print("  il transitorio e' tutto nel TWIST: il modello di velocita' non ha")
    print("  ancora osservato movimento. In quel caso un riscaldamento a robot")
    print("  fermo NON aiuta, perche' da fermo il twist corretto e' zero.")
    print("  Se invece sigma impiega molti scan ad assestarsi, anche il fermo")
    print("  contribuisce e si puo' accorciare il tratto di riscaldamento.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
