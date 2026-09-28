#!/usr/bin/env python3
"""
Confronto fra regimi di mappa locale: ancorato vs scorrevole.

Stabilisce se il drift di MOLA si accumula lungo il percorso o viene riassorbito
quando il robot rientra nel raggio dei keyframe iniziali. Solo nel primo caso lo
schema receding-horizon puo' costruire danno segmento dopo segmento: se ogni
deviazione viene corretta dalla stessa geometria di riferimento, non c'e' nulla
da accumulare. Con MOLA_LOCAL_MAP_MAX_SIZE=25 m e un rettangolo di diagonale
4.7 m il punto di partenza non esce mai dalla mappa locale; con 8 m si'.

Legge, per le due run indicate, data/trajectories/mola/run_<id>.tum e
data/trajectories/gt/run_<id>_aligned.tum; scrive un grafico a tre pannelli
(errore lungo il percorso, errore vs distanza dalla partenza, riassorbimento
finale) e stampa picco, errore finale e frazione residua per regime.

Uso:
    # dopo aver eseguito le due run con --local-map-size 25 e 8
    python3 src/analysis/compare_localmap_regimes.py --anchored <id> --sliding <id> \
        [--dir data/trajectories] [--out data/plots/localmap_regimes.png]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pair(base: Path, run_id: int):
    """Carica traiettoria MOLA e riferimento interpolato agli stessi timestamp."""
    m = np.loadtxt(base / "mola" / f"run_{run_id}.tum", comments="#")
    g = np.loadtxt(base / "gt" / f"run_{run_id}_aligned.tum", comments="#")
    n = min(len(m), len(g))
    return m[:n], g[:n]


def error_profile(m, g):
    """Errore istantaneo, frazione di percorso e ascissa curvilinea lungo la traiettoria.

    L'ascissa e' la frazione di percorso e non il tempo, cosi' due run di durata
    diversa restano confrontabili punto per punto.
    """
    err = np.linalg.norm(m[:, 1:3] - g[:, 1:3], axis=1)
    seg = np.linalg.norm(np.diff(g[:, 1:3], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    frac = s / max(s[-1], 1e-9)
    return frac, err, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchored", type=int, required=True,
                    help="run-id con local map grande (es. 25 m)")
    ap.add_argument("--sliding", type=int, required=True,
                    help="run-id con local map piccola (es. 8 m)")
    ap.add_argument("--dir", type=str, default="data/trajectories")
    ap.add_argument("--out", type=str, default="data/plots/localmap_regimes.png")
    args = ap.parse_args()

    base = Path(args.dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = {}
    for label, rid in [("ancorato (25 m)", args.anchored),
                       ("scorrevole (8 m)", args.sliding)]:
        try:
            m, g = load_pair(base, rid)
        except OSError as e:
            print(f"  run {rid}: {e}")
            continue
        runs[label] = error_profile(m, g) + (m, g)

    if len(runs) < 2:
        print("Servono entrambe le run.")
        return 1

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    colors = {"ancorato (25 m)": "#1f77b4", "scorrevole (8 m)": "#d62728"}

    # 1. Errore lungo il percorso
    for label, (frac, err, s, m, g) in runs.items():
        ax[0].plot(frac, err * 1000, color=colors[label], lw=1.4, label=label)
    ax[0].set_title("Errore MOLA lungo il percorso")
    ax[0].set_xlabel("frazione di percorso")
    ax[0].set_ylabel("errore [mm]")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # 2. Errore vs distanza dalla partenza. Se l'errore dipende dalla distanza
    # dall'origine invece che dal percorso fatto, domina l'ancoraggio alla mappa
    # locale: il riassorbimento avviene al rientro nel raggio dei keyframe iniziali.
    for label, (frac, err, s, m, g) in runs.items():
        d_from_start = np.linalg.norm(g[:, 1:3] - g[0, 1:3], axis=1)
        ax[1].scatter(d_from_start, err * 1000, s=4, alpha=0.4,
                      color=colors[label], label=label)
    ax[1].set_title("Errore vs distanza dalla partenza")
    ax[1].set_xlabel("distanza dal punto di partenza [m]")
    ax[1].set_ylabel("errore [mm]")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    # 3. Riassorbimento finale: quota dell'errore di picco che resta alla fine.
    # Un rapporto basso significa che la mappa ha cancellato il drift accumulato.
    labels, peaks, finals = [], [], []
    for label, (frac, err, s, m, g) in runs.items():
        labels.append(label)
        peaks.append(err.max() * 1000)
        finals.append(float(np.mean(err[-10:]) * 1000))

    x = np.arange(len(labels))
    ax[2].bar(x - 0.18, peaks, 0.36, label="picco", color="#888888")
    ax[2].bar(x + 0.18, finals, 0.36, label="finale",
              color=[colors[l] for l in labels])
    for i, (p, f) in enumerate(zip(peaks, finals)):
        ax[2].text(i, max(p, f) * 1.02, f"resta {100*f/max(p,1e-9):.0f}%",
                   ha="center", fontsize=9)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, fontsize=9)
    ax[2].set_ylabel("errore [mm]")
    ax[2].set_title("Riassorbimento del drift")
    ax[2].grid(alpha=0.3, axis="y")
    ax[2].legend()

    fig.suptitle("Regime della mappa locale \u2014 il drift accumula o viene riassorbito?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)

    print(f"\n{'='*66}")
    print(f"  {'regime':<20} {'picco':>10} {'finale':>10} {'resta':>8}")
    print(f"{'='*66}")
    for label, p, f in zip(labels, peaks, finals):
        print(f"  {label:<20} {p:>8.1f}mm {f:>8.1f}mm {100*f/max(p,1e-9):>7.0f}%")

    print(f"\nSalvato: {out}\n")
    print("Come leggerlo:")
    print("  Se nel regime ancorato l'errore sale e poi ricade, e nello")
    print("  scorrevole sale e resta alto, il drift accumula solo nel secondo.")
    print("  E' quello il regime in cui il receding-horizon puo' costruire")
    print("  danno segmento dopo segmento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
