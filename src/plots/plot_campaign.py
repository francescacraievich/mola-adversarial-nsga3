#!/usr/bin/env python3
"""
Figure della campagna su uno scenario.

Legge le ripetizioni complete (cartelle rep_NN con il file "completata") sotto
la cartella dello scenario e scrive in <scenario>/plots/:
  traiettorie_nsga3.png    traiettorie vere nominale e attaccata di ogni run
                           nsga3, sul piano x-y, con il bersaglio
  danno_totale.png         box plot del danno applicato totale per braccio
  danno_per_finestra.png   danno applicato per finestra e braccio
  chamfer_danno.png        Chamfer contro danno di tutte le valutazioni nsga3,
                           con i quattro punti del gaussiano (curva del rumore
                           casuale) e i genomi applicati evidenziati
  pareto_<run>.png         fronte di Pareto per finestra di una run di esempio
  tempo_per_finestra.png   tempo per finestra e braccio

Uso:
    python3 src/plots/plot_campaign.py data/attack/campaign/straight [--esempio rep_01]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP_RE = re.compile(r"^rep_\d{2}$")
VALID = ("ok", "arrived")
# I PNG di matplotlib portano solo un campo "Software"; a None non viene scritto
# alcun metadato di provenienza.
NO_META = {"Software": None}


def salva(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata=NO_META)
    plt.close(fig)
    print(f"  {path}")


def reps_complete(arm_dir: Path):
    """Cartelle rep_NN con il file 'completata', in ordine."""
    out = []
    for d in sorted(arm_dir.glob("rep_[0-9][0-9]")):
        if REP_RE.match(d.name) and (d / "completata").exists():
            out.append(d)
    return out


def arms(root: Path):
    """Cartelle di braccio con almeno una ripetizione completa, ordinate."""
    found = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name != "plots" and reps_complete(d):
            found[d.name] = reps_complete(d)
    return found


def order_arms(names):
    """none, poi gaussian per sigma crescente, poi random/nsga3."""
    def key(n):
        if n.startswith("none"):
            return (0, n)
        if n.startswith("gaussian_s"):
            try:
                return (1, float(n.split("_s")[1]))
            except ValueError:
                return (1, n)
        return (2, n)
    return sorted(names, key=key)


def leggi_traccia(path: Path):
    """Colonne true_x, true_y da una traccia; array Nx2 o vuoto."""
    if not path.exists():
        return np.empty((0, 2))
    xy = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                xy.append((float(row["true_x"]), float(row["true_y"])))
            except (KeyError, ValueError):
                pass
    return np.array(xy) if xy else np.empty((0, 2))


def percorso_vero(rep: Path, suffisso: str):
    """Traiettoria vera concatenata sulle finestre (nominal o applied)."""
    tr = rep / "traces"
    parti = []
    k = 0
    while True:
        p = tr / f"w{k}_{suffisso}.csv"
        if not p.exists():
            break
        xy = leggi_traccia(p)
        if len(xy):
            parti.append(xy)
        k += 1
    return np.vstack(parti) if parti else np.empty((0, 2))


def carica(root: Path):
    """Per ogni braccio: lista di (nome_rep, history)."""
    dati = {}
    for arm, reps in arms(root).items():
        dati[arm] = [(r.name, json.load(open(r / "history.json")), r) for r in reps]
    return dati


def somma(hist, campo):
    v = [w[campo] for w in hist if w.get(campo) is not None]
    return float(np.sum(v)) if v else np.nan


# ---------------------------------------------------------------------------

def plot_traiettorie(dati, out):
    runs = dati.get("nsga3_g3_p8x4") or next(
        (dati[a] for a in dati if a.startswith("nsga3")), None)
    if not runs:
        return
    n = len(runs)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    goal = None
    for h, _, _ in [(h, 0, 0) for _, h, _ in runs]:
        for w in h:
            if w.get("goal"):
                goal = w["goal"]
    for ax in axes.ravel():
        ax.axis("off")
    for i, (nome, hist, rep) in enumerate(runs):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        nom = percorso_vero(rep, "nominal")
        app = percorso_vero(rep, "applied")
        if len(nom):
            ax.plot(nom[:, 0], nom[:, 1], color="#2563eb", lw=1.6, label="nominale")
        if len(app):
            ax.plot(app[:, 0], app[:, 1], color="#dc2626", lw=1.6, label="attaccata")
        g = next((w["goal"] for w in hist if w.get("goal")), goal)
        if g:
            ax.plot(g[0], g[1], marker="*", color="#111827", markersize=13,
                    linestyle="none", label="bersaglio")
        ax.set_title(nome, fontsize=9)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Traiettorie vere: nominale contro attaccata (NSGA-III, gruppo 3)",
                 fontsize=12)
    fig.supxlabel("x [m] (frame odom)", fontsize=9)
    fig.supylabel("y [m]", fontsize=9)
    fig.tight_layout()
    salva(fig, out / "traiettorie_nsga3.png")


def plot_danno_totale(dati, out):
    noms = order_arms([a for a in dati if not a.startswith("none")])
    serie = [[somma(h, "damage_real_cm") for _, h, _ in dati[a]] for a in noms]
    serie = [[v for v in s if np.isfinite(v)] for s in serie]
    fig, ax = plt.subplots(figsize=(1.6 * len(noms) + 2, 5))
    ax.boxplot(serie, tick_labels=noms, showmeans=True)
    ax.set_ylabel("danno applicato totale sulla run [cm]")
    ax.set_title("Danno applicato totale per braccio")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    salva(fig, out / "danno_totale.png")


def plot_danno_per_finestra(dati, out):
    noms = order_arms([a for a in dati if not a.startswith("none")])
    finestre = sorted({w["window"] for a in noms for _, h, _ in dati[a] for w in h
                       if w.get("damage_real_cm") is not None})
    fig, ax = plt.subplots(figsize=(2.0 * len(finestre) + 2, 5))
    cmap = plt.get_cmap("tab10")
    width = 0.8 / max(len(noms), 1)
    for j, a in enumerate(noms):
        medie, errori, xs = [], [], []
        for fi, k in enumerate(finestre):
            vals = [w["damage_real_cm"] for _, h, _ in dati[a] for w in h
                    if w["window"] == k and w.get("damage_real_cm") is not None]
            if vals:
                medie.append(np.mean(vals))
                errori.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                xs.append(fi + j * width - 0.4 + width / 2)
        ax.bar(xs, medie, width, yerr=errori, capsize=3, color=cmap(j % 10), label=a)
    ax.set_xticks(range(len(finestre)))
    ax.set_xticklabels([f"finestra {k + 1}" for k in finestre])
    ax.set_ylabel("danno applicato [cm]")
    ax.set_title("Danno applicato per finestra e braccio (media ± dev.std)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    salva(fig, out / "danno_per_finestra.png")


def plot_chamfer_danno(dati, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    nsga = dati.get("nsga3_g3_p8x4") or next(
        (dati[a] for a in dati if a.startswith("nsga3")), [])
    # Tutte le valutazioni valide.
    cx, cy = [], []
    ax_applied_x, ax_applied_y = [], []
    for _, hist, _ in nsga:
        for w in hist:
            for e in w.get("evaluations", []):
                if e["status"] in VALID and np.isfinite(e.get("chamfer", np.nan)):
                    cx.append(e["chamfer"])
                    cy.append(e["damage"] * 100.0)
            if w.get("perturbation") is not None and w.get("damage_real_cm") is not None:
                ax_applied_x.append(w["perturbation"])
                ax_applied_y.append(w["damage_real_cm"])
    if cx:
        ax.scatter(cx, cy, s=10, color="#9ca3af", alpha=0.5,
                   label="valutazioni NSGA-III")
    if ax_applied_x:
        ax.scatter(ax_applied_x, ax_applied_y, s=42, color="#dc2626",
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label="genomi applicati")
    # Bracci gaussiani: un punto per sigma (media delle finestre e delle run).
    gx, gy, gl = [], [], []
    for a in order_arms([a for a in dati if a.startswith("gaussian_s")]):
        ch = [w["perturbation"] for _, h, _ in dati[a] for w in h
              if w.get("perturbation") is not None]
        dm = [w["damage_real_cm"] for _, h, _ in dati[a] for w in h
              if w.get("damage_real_cm") is not None]
        if ch and dm:
            gx.append(np.mean(ch))
            gy.append(np.mean(dm))
            gl.append(a.split("_s")[1])
    if gx:
        order = np.argsort(gx)
        gx = np.array(gx)[order]; gy = np.array(gy)[order]
        gl = [gl[i] for i in order]
        ax.plot(gx, gy, "-o", color="#2563eb", zorder=4,
                label="gaussiano (rumore casuale)")
        # Etichette alternate sopra e sotto: a basso danno i punti sono vicini.
        for i, (x, y, l) in enumerate(zip(gx, gy, gl)):
            dy = 9 if i % 2 == 0 else -14
            ax.annotate(f"σ={l}", (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8, color="#2563eb")
    ax.set_xlabel("Chamfer (percettibilità) [cm]")
    ax.set_ylabel("danno [cm]")
    ax.set_title("Chamfer contro danno: valutazioni NSGA-III e rumore gaussiano")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    salva(fig, out / "chamfer_danno.png")


def plot_pareto(dati, out, esempio):
    nsga = dati.get("nsga3_g3_p8x4") or next(
        (dati[a] for a in dati if a.startswith("nsga3")), [])
    scelta = next((t for t in nsga if t[0] == esempio), None) or (nsga[0] if nsga else None)
    if scelta is None:
        return
    nome, hist, _ = scelta
    fin = [w for w in hist if w.get("evaluations")]
    if not fin:
        return
    ncol = min(len(fin), 5)
    nrow = int(np.ceil(len(fin) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, w in enumerate(fin):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        cx = [e["chamfer"] for e in w["evaluations"]
              if e["status"] in VALID and np.isfinite(e.get("chamfer", np.nan))]
        cy = [e["damage"] * 100.0 for e in w["evaluations"]
              if e["status"] in VALID and np.isfinite(e.get("chamfer", np.nan))]
        if cx:
            ax.scatter(cx, cy, s=16, color="#9ca3af", alpha=0.7, label="valutate")
        pf = np.atleast_2d(w.get("pareto_F", []))
        if pf.size:
            fx = pf[:, 1]
            fy = -pf[:, 0] * 100.0
            o = np.argsort(fx)
            ax.plot(fx[o], fy[o], "-o", color="#2563eb", ms=5, label="fronte")
        if w.get("perturbation") is not None:
            ax.scatter([w["perturbation"]], [w["damage_cm"]], marker="*",
                       s=150, color="#dc2626", zorder=5, label="scelto")
        ax.set_title(f"finestra {w['window'] + 1}", fontsize=9)
        ax.set_xlabel("Chamfer [cm]", fontsize=8)
        ax.set_ylabel("danno [cm]", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Fronte di Pareto per finestra — {nome}", fontsize=12)
    fig.tight_layout()
    salva(fig, out / f"pareto_{nome}.png")


def plot_tempo_per_finestra(dati, out):
    noms = order_arms(list(dati))
    finestre = sorted({w["window"] for a in noms for _, h, _ in dati[a] for w in h
                       if w.get("window_time_s") is not None})
    if not finestre:
        return
    fig, ax = plt.subplots(figsize=(2.0 * len(finestre) + 2, 5))
    cmap = plt.get_cmap("tab10")
    width = 0.8 / max(len(noms), 1)
    for j, a in enumerate(noms):
        medie, errori, xs = [], [], []
        for fi, k in enumerate(finestre):
            vals = [w["window_time_s"] / 60.0 for _, h, _ in dati[a] for w in h
                    if w["window"] == k and w.get("window_time_s") is not None]
            if vals:
                medie.append(np.mean(vals))
                errori.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                xs.append(fi + j * width - 0.4 + width / 2)
        ax.bar(xs, medie, width, yerr=errori, capsize=3, color=cmap(j % 10), label=a)
    ax.set_xticks(range(len(finestre)))
    ax.set_xticklabels([f"finestra {k + 1}" for k in finestre])
    ax.set_ylabel("tempo [min]")
    ax.set_title("Tempo per finestra e braccio (media ± dev.std)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    salva(fig, out / "tempo_per_finestra.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="cartella di uno scenario, es. data/attack/campaign/straight")
    ap.add_argument("--esempio", default="rep_01", help="run per il fronte di Pareto (default rep_01)")
    args = ap.parse_args()

    dati = carica(args.root)
    if not dati:
        print(f"nessuna run completa sotto {args.root}")
        return 1
    out = args.root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    print(f"bracci: {', '.join(order_arms(list(dati)))}")
    plot_traiettorie(dati, out)
    plot_danno_totale(dati, out)
    plot_danno_per_finestra(dati, out)
    plot_chamfer_danno(dati, out)
    plot_pareto(dati, out, args.esempio)
    plot_tempo_per_finestra(dati, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
