#!/usr/bin/env python3
"""
Traiettoria nominale vs attaccata.

Mostra il percorso che il robot avrebbe tenuto senza attacco (nominale) contro
quello tenuto sotto attacco, finestra per finestra, con il bersaglio.

Legge history.json prodotto dall'orchestratore. Le posizioni sono pose vere
lette da Isaac Sim, non stime di MOLA. Assi: x, y in metri.

Uso:
    python3 src/plots/plot_trajectory.py data/attack/<run>/history.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Traiettoria: nominale vs attaccata")
    args = ap.parse_args()

    with open(args.history) as f:
        H = json.load(f)
    if not H:
        print("history vuoto")
        return 1

    # Solo le finestre con un attacco applicato (l'ultima puo' essere saltata).
    W = [w for w in H if w.get("applied_end") and w.get("nominal_end")]
    if not W:
        print("nessuna finestra con attacco applicato")
        return 1

    out = Path(args.out) if args.out else Path(args.history).parent / "plot_trajectory.png"

    # Gli applied_end formano una catena continua; i nominal_end no: ogni nominale
    # e' ricalcolato dal punto reale corrente, quindi e' un raggio dalla traiettoria attaccata.
    start = W[0].get("goal")  # solo per riferimento assi
    applied = np.array([w["applied_end"][:2] for w in W])
    nominal = np.array([w["nominal_end"][:2] for w in W])
    goal = np.array(W[0]["goal"][:2]) if W[0].get("goal") else None

    # Il punto di partenza della prima finestra non e' registrato: si disegna dalla seconda.
    fig, ax = plt.subplots(figsize=(11, 6))

    # Traiettoria attaccata: catena continua.
    ax.plot(applied[:, 0], applied[:, 1], "-o", color="#d62728", lw=2,
            markersize=7, label="attaccata (posa vera)", zorder=3)

    # Segmento nominale di ogni finestra: dall'applied_end precedente al nominal_end.
    for i, w in enumerate(W):
        p_nom = np.array(w["nominal_end"][:2])
        p_start = applied[i - 1] if i > 0 else None
        if p_start is not None:
            ax.plot([p_start[0], p_nom[0]], [p_start[1], p_nom[1]],
                    "--", color="#1f77b4", lw=1.3, alpha=0.8,
                    label="nominale (senza attacco)" if i == 1 else None,
                    zorder=2)
            ax.plot(p_nom[0], p_nom[1], "s", color="#1f77b4",
                    markersize=5, zorder=2)

    if goal is not None:
        ax.plot(goal[0], goal[1], "*", color="#2ca02c", markersize=22,
                label="bersaglio C", zorder=4)

    # Frecce di deviazione da nominal_end ad applied_end, con lo scarto in cm.
    for i, w in enumerate(W):
        p_nom = np.array(w["nominal_end"][:2])
        p_att = applied[i]
        d = float(np.linalg.norm(p_att - p_nom)) * 100
        ax.annotate("", xy=p_att, xytext=p_nom,
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
        mid = (p_nom + p_att) / 2
        ax.text(mid[0], mid[1] + 0.03, f"{d:.0f} cm", fontsize=8,
                color="#555555", ha="center")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(args.title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Salvato: {out}")

    # Riepilogo numerico per finestra.
    print(f"\n  {'finestra':>8} {'dev applicata':>14} {'stato':>10}")
    for i, w in enumerate(W):
        d = float(np.linalg.norm(applied[i] - nominal[i])) * 100
        print(f"  {w['window']:>8} {d:>12.1f}cm {w.get('applied_status',''):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
