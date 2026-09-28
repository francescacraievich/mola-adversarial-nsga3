#!/usr/bin/env python3
"""
Andamento temporale: perturbazione -> errore MOLA -> deviazione.

Legge le tracce per-tick prodotte dall'orchestratore con --trace e mostra su un
unico asse dei tempi tre pannelli:

  1. perturbazione applicata (Chamfer, cm)
  2. errore di MOLA (divario stima-verita', cm) e qualita' ICP
  3. deviazione della traiettoria vera dalla rotta nominale (cm)

Serve a vedere se una perturbazione piccola e costante produce un errore che si
accumula nel closed loop. Piu' tracce vengono concatenate in sequenza sull'asse
dei tempi, con una linea ai confini di finestra.

Il divario stima-verita' e' quello registrato nella traccia (est_true_gap). La
deviazione dal nominale e' approssimata con la distanza laterale dalla direzione
iniziale del moto.
Limite noto: stima e posa vera sono in frame diversi (MOLA riparte da (0,0,0) a
ogni riavvio), l'allineamento e' da implementare.

Uso:
    python3 src/plots/plot_temporal.py data/attack/<run>/traces/w{0,1,2}_applied.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    d = {}
    for k in rows[0]:
        d[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                         for r in rows])
    return d


def deviation_from_nominal(d):
    """Distanza laterale della posa vera dalla direzione iniziale del moto (m)."""
    tx, ty = d["true_x"], d["true_y"]
    ok = ~np.isnan(tx)
    if ok.sum() < 2:
        return np.full(len(tx), np.nan)
    p0 = np.array([tx[ok][0], ty[ok][0]])
    # Il waypoint e' nel frame di MOLA, non nel mondo: la direzione dei primi campioni
    # veri fa da proxy della rotta nominale.
    head = np.array([tx[ok][min(5, ok.sum() - 1)] - p0[0],
                     ty[ok][min(5, ok.sum() - 1)] - p0[1]])
    n = np.linalg.norm(head)
    if n < 1e-6:
        return np.full(len(tx), np.nan)
    head /= n
    normal = np.array([-head[1], head[0]])
    dev = np.full(len(tx), np.nan)
    for i in range(len(tx)):
        if not np.isnan(tx[i]):
            dev[i] = abs(np.dot(np.array([tx[i] - p0[0], ty[i] - p0[1]]), normal))
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Concatena le finestre su un unico asse temporale continuo.
    T, cham, gap, qual, dev = [], [], [], [], []
    t_off = 0.0
    boundaries = []
    for p in args.traces:
        d = load(p)
        if d is None:
            continue
        t = d["t"] + t_off
        T.append(t)
        cham.append(d.get("chamfer_cm", np.full(len(t), np.nan)))
        gap.append(d.get("est_true_gap", np.full(len(t), np.nan)))
        qual.append(d.get("pose_quality", np.full(len(t), np.nan)))
        dev.append(deviation_from_nominal(d))
        t_off = t[-1] + 0.1
        boundaries.append(t_off)

    if not T:
        print("nessuna traccia valida")
        return 1

    T = np.concatenate(T)
    cham = np.concatenate(cham)
    gap = np.concatenate(gap) * 100      # m -> cm
    qual = np.concatenate(qual)
    dev = np.concatenate(dev) * 100      # m -> cm

    out = Path(args.out) if args.out else Path(args.traces[0]).parent.parent / "plot_temporal.png"

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1. Perturbazione applicata.
    ax[0].plot(T, cham, color="#9467bd", lw=1.4)
    ax[0].set_ylabel("perturbazione\nChamfer [cm]")
    ax[0].set_title("Catena causale dell'attacco nel closed loop")
    ax[0].grid(alpha=0.3)

    # 2. Errore di MOLA e qualita' ICP su asse secondario.
    ax[1].plot(T, gap, color="#ff7f0e", lw=1.4, label="divario stima-verita'")
    ax[1].set_ylabel("errore MOLA [cm]", color="#ff7f0e")
    ax[1].tick_params(axis="y", labelcolor="#ff7f0e")
    ax[1].grid(alpha=0.3)
    ax1b = ax[1].twinx()
    ax1b.plot(T, qual, color="#1f77b4", lw=1.0, alpha=0.6, label="qualita' ICP")
    ax1b.set_ylabel("qualita' ICP", color="#1f77b4")
    ax1b.tick_params(axis="y", labelcolor="#1f77b4")
    ax1b.set_ylim(0, 1.05)

    # 3. Deviazione della traiettoria vera.
    ax[2].plot(T, dev, color="#d62728", lw=1.6)
    ax[2].set_ylabel("deviazione\ntraiettoria [cm]")
    ax[2].set_xlabel("tempo [s]")
    ax[2].grid(alpha=0.3)

    # Linee ai confini di finestra.
    for b in boundaries[:-1]:
        for a in ax:
            a.axvline(b, color="#cccccc", ls=":", lw=0.8)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Salvato: {out}")

    # Sintesi: confronto fra primo e ultimo quarto della deviazione.
    finite = ~np.isnan(dev)
    if finite.sum() > 10:
        first = np.nanmean(dev[:len(dev) // 4])
        last = np.nanmean(dev[-len(dev) // 4:])
        print(f"\n  deviazione media primo quarto : {first:.1f} cm")
        print(f"  deviazione media ultimo quarto: {last:.1f} cm")
        if last > first * 1.5:
            print("  -> la deviazione CRESCE nel tempo: accumulo nel closed loop")
        else:
            print("  -> deviazione stabile: nessun accumulo evidente su questo tratto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
