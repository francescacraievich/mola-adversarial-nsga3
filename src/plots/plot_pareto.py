#!/usr/bin/env python3
"""
Fronte di Pareto: danno della traiettoria vs percettibilita'.

Per ogni finestra dell'attacco mostra i candidati valutati nel piano
(percettibilita', danno), il fronte di Pareto non dominato e la soluzione
scelta. Un pannello per finestra; il genoma e' quello dell'orchestratore.

Legge history.json prodotto dall'orchestratore (campo "evaluations").

Assi:
  x = percettibilita' (Chamfer, cm)  -> minimizzare
  y = danno (deviazione, cm)         -> massimizzare
La soluzione ideale sta in alto a sinistra: molto danno, poco visibile.

Uso:
    python3 src/plots/plot_pareto.py data/attack/<run>/history.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pareto_mask(pts):
    """Maschera dei punti non dominati: pts[:,0] = pert (min), pts[:,1] = danno (max)."""
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1] and (
                    pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]):
                keep[i] = False
                break
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.history) as f:
        H = json.load(f)
    W = [w for w in H if w.get("evaluations")]
    if not W:
        print("nessuna finestra con valutazioni")
        return 1

    out = Path(args.out) if args.out else Path(args.history).parent / "plot_pareto.png"

    n = len(W)
    ncol = min(n, 3)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.5 * nrow),
                             squeeze=False)

    for idx, w in enumerate(W):
        ax = axes[idx // ncol][idx % ncol]
        evs = w["evaluations"]

        # Solo i candidati con danno misurato; gli scartati (stopped_early, timeout,
        # untracked) non hanno un punto nel piano.
        valid = [(e["chamfer"], e["damage"] * 100)
                 for e in evs
                 if e.get("damage") is not None and e.get("status") in ("ok", "arrived")]
        scarted = sum(1 for e in evs
                      if e.get("damage") is None or e.get("status") not in ("ok", "arrived"))

        if not valid:
            ax.set_title(f"finestra {w['window']}: nessun candidato valido")
            continue

        pts = np.array(valid)
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color="#aaaaaa",
                   label="candidati", zorder=2)

        # Fronte di Pareto fra i candidati validi, ordinato per percettibilita'.
        m = pareto_mask(pts)
        front = pts[m]
        front = front[np.argsort(front[:, 0])]
        ax.plot(front[:, 0], front[:, 1], "-o", color="#d62728", lw=1.5,
                markersize=8, label="fronte di Pareto", zorder=3)

        # Soluzione scelta dall'orchestratore.
        chosen_dmg = w.get("damage_cm")
        chosen_pert = w.get("perturbation")
        if chosen_dmg is not None and chosen_pert is not None:
            ax.scatter([chosen_pert], [chosen_dmg], s=180, marker="*",
                       color="#2ca02c", edgecolor="k", linewidth=0.5,
                       label="scelto", zorder=4)

        ax.set_xlabel("percettibilita' — Chamfer [cm]")
        ax.set_ylabel("danno — deviazione [cm]")
        ax.set_title(f"finestra {w['window']}  "
                     f"({len(valid)} validi, {scarted} scartati)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Pannelli vuoti della griglia.
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("Fronte di Pareto per finestra — danno vs percettibilita'\n"
                 "(ideale: alto a sinistra)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140)
    print(f"Salvato: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
