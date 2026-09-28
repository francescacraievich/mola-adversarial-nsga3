#!/usr/bin/env python3
"""
Grafici di baseline per le run del percorso chiuso.

Legge le traiettorie TUM in data/trajectories/{mola,gt}/run_<n>.tum (e, se
presente, gt/run_<n>_aligned.tum per l'ATE) e produce una figura a quattro
pannelli:

  1. Traiettorie XY sovrapposte (MOLA e riferimento)
  2. Errore MOLA-riferimento in funzione della frazione di percorso
  3. Scarto fra ogni run e la mediana delle run, per ascissa curvilinea
  4. ATE medio per run con la banda a 3 sigma

I pannelli 3 e 4 accoppiano le run per ascissa curvilinea, non per tempo: le run
partono in istanti diversi rispetto allo zero di MOLA e l'accoppiamento per
timestamp confronterebbe punti diversi dello stesso percorso.

Uso:
    python3 src/plots/plot_loop_baseline.py --runs 2 3 4 5 6
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p: Path):
    return np.loadtxt(p, comments="#")


def arclen_resample(xy: np.ndarray, n: int = 400):
    """Ricampiona una polilinea a n punti equispaziati in lunghezza d'arco.

    Ritorna (traiettoria ricampionata, frazione di percorso in [0,1]).
    """
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return None, None
    u = np.linspace(0.0, s[-1], n)
    out = np.column_stack([np.interp(u, s, xy[:, 0]),
                           np.interp(u, s, xy[:, 1])])
    return out, u / s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", required=True)
    ap.add_argument("--dir", type=str, default="data/trajectories")
    ap.add_argument("--out", type=str, default="data/plots/loop_baseline.png")
    ap.add_argument("--samples", type=int, default=400)
    args = ap.parse_args()

    base = Path(args.dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    mola, gt, ates, labels = [], [], [], []
    for r in args.runs:
        mp = base / "mola" / f"run_{r}.tum"
        gp = base / "gt" / f"run_{r}.tum"
        ga = base / "gt" / f"run_{r}_aligned.tum"
        if not (mp.exists() and gp.exists()):
            print(f"  run {r}: file mancanti, saltata")
            continue
        m, g = load(mp), load(gp)
        mola.append(m)
        gt.append(g)
        labels.append(r)
        if ga.exists():
            a = load(ga)
            n = min(len(m), len(a))
            ates.append(np.linalg.norm(m[:n, 1:3] - a[:n, 1:3], axis=1))
        else:
            ates.append(None)

    if not mola:
        print("Nessuna run valida.")
        return

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    cmap = plt.get_cmap("tab10")

    # 1. Traiettorie XY.
    a0 = ax[0][0]
    for i, (m, g, r) in enumerate(zip(mola, gt, labels)):
        c = cmap(i % 10)
        a0.plot(m[:, 1], m[:, 2], color=c, lw=1.4, label=f"MOLA run {r}")
        a0.plot(g[:, 1], g[:, 2], color=c, lw=0.9, ls="--", alpha=0.55)
    a0.plot(mola[0][0, 1], mola[0][0, 2], "ko", ms=8, label="partenza")
    a0.set_title("Traiettorie  (continuo = MOLA, tratteggio = riferimento)")
    a0.set_xlabel("x [m]")
    a0.set_ylabel("y [m]")
    a0.axis("equal")
    a0.grid(alpha=0.3)
    a0.legend(fontsize=7, ncol=2)

    # 2. Errore MOLA vs riferimento lungo il percorso.
    a1 = ax[0][1]
    for i, (e, r) in enumerate(zip(ates, labels)):
        if e is None:
            continue
        frac = np.linspace(0, 1, len(e))
        a1.plot(frac, e * 1000, color=cmap(i % 10), lw=1.2, label=f"run {r}")
    a1.set_title("Errore MOLA − riferimento lungo il percorso")
    a1.set_xlabel("frazione di percorso")
    a1.set_ylabel("errore [mm]")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)

    # 3. Scarto di ogni run dalla mediana, per ascissa curvilinea.
    a2 = ax[1][0]
    res = [arclen_resample(m[:, 1:3], args.samples)[0] for m in mola]
    res = [x for x in res if x is not None]
    if len(res) >= 2:
        stack = np.stack(res)                 # (n_run, n_samples, 2)
        med = np.median(stack, axis=0)
        frac = np.linspace(0, 1, args.samples)
        for i, (rr, r) in enumerate(zip(res, labels)):
            d = np.linalg.norm(rr - med, axis=1)
            a2.plot(frac, d * 1000, color=cmap(i % 10), lw=1.2,
                    label=f"run {r}  (med {d.mean()*1000:.0f} mm)")
        a2.set_title("Scarto di ogni run dalla mediana  (per ascissa curvilinea)")
        a2.set_xlabel("frazione di percorso")
        a2.set_ylabel("scarto [mm]")
        a2.grid(alpha=0.3)
        a2.legend(fontsize=8)

    # 4. ATE medio per run con banda a 3 sigma.
    a3 = ax[1][1]
    means = np.array([e.mean() * 1000 for e in ates if e is not None])
    labs = [r for r, e in zip(labels, ates) if e is not None]
    if len(means):
        mu, sd = means.mean(), means.std(ddof=1) if len(means) > 1 else 0.0
        a3.bar(range(len(means)), means,
               color=[cmap(i % 10) for i in range(len(means))], alpha=0.85)
        a3.axhline(mu, color="k", lw=1.2, label=f"media {mu:.1f} mm")
        a3.axhline(mu + 3 * sd, color="r", lw=1.2, ls="--",
                   label=f"soglia attacco 3σ = {mu+3*sd:.1f} mm")
        a3.axhspan(mu - sd, mu + sd, color="gray", alpha=0.18, label="±1σ")
        a3.set_xticks(range(len(means)))
        a3.set_xticklabels([f"run {r}" for r in labs])
        a3.set_title("ATE medio per run")
        a3.set_ylabel("ATE [mm]")
        a3.grid(alpha=0.3, axis="y")
        a3.legend(fontsize=8)

    fig.suptitle(
        f"Baseline percorso chiuso — run {', '.join(map(str, labels))}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
