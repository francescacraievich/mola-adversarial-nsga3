#!/usr/bin/env python3
"""
Aggregazione delle ripetizioni di una campagna: una riga per run, medie e
deviazioni standard per braccio, distribuzioni in figura.

Legge tutti gli history.json sotto data/attack/campaign/<scenario>/<arm>_g<gruppo>/rep_*/
e per ogni run calcola:
  damage_total_cm   somma sulle finestre della deviazione media applicata
                    (dev_real_mean_cm; per il braccio "none" e' 0 per definizione)
  damage_search_cm  somma del danno stimato in ricerca (damage_cm)
  final_dist_m      distanza vera fra l'ultimo punto d'arrivo e il bersaglio finale
  windows           finestre eseguite
  heading_deg       errore di prua medio per finestra (valore assoluto)
  frames_per_m      frame per metro medio nei tratti applicati
  paralysis_frac    frazione di candidati con stopped_early
  untracked_frac    frazione di candidati con tracking perso
  chamfer           percettibilita' media dei genomi applicati
  time_min          tempo totale della run

Uso:
    python3 src/analysis/aggregate_campaign.py data/attack/campaign [--plot]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def summarize_run(path: Path):
    with open(path) as f:
        hist = json.load(f)
    if not hist:
        return None
    goal_final = None
    dmg_real, dmg_search, heading, fpm, cham, times = [], [], [], [], [], []
    n_eval = n_par = n_untr = 0
    last_end = None
    for w in hist:
        goal_final = w.get("goal", goal_final)
        if w.get("applied_end"):
            last_end = w["applied_end"]
        elif w.get("nominal_end") and "genome" not in w:
            last_end = w["nominal_end"]
        if w.get("dev_real_mean_cm") is not None:
            dmg_real.append(w["dev_real_mean_cm"])
        if w.get("damage_cm") is not None:
            dmg_search.append(w["damage_cm"])
        if w.get("heading_error_deg") is not None:
            heading.append(abs(w["heading_error_deg"]))
        if w.get("frames_per_m_applied") is not None:
            fpm.append(w["frames_per_m_applied"])
        elif w.get("frames") and w.get("travelled"):
            fpm.append(w["frames"] / max(w["travelled"], 1e-6))
        if w.get("perturbation") is not None:
            cham.append(w["perturbation"])
        if w.get("window_time_s") is not None:
            times.append(w["window_time_s"])
        n_eval += w.get("n_evaluations", 0)
        n_par += w.get("n_stopped_early", 0)
        n_untr += w.get("n_untracked", 0)
    final_dist = (math.hypot(goal_final[0] - last_end[0], goal_final[1] - last_end[1])
                  if goal_final and last_end else float("nan"))
    return {
        "damage_total_cm": float(np.sum(dmg_real)) if dmg_real else 0.0,
        "damage_search_cm": float(np.sum(dmg_search)) if dmg_search else 0.0,
        "final_dist_m": final_dist,
        "windows": len(hist),
        "heading_deg": float(np.mean(heading)) if heading else float("nan"),
        "frames_per_m": float(np.mean(fpm)) if fpm else float("nan"),
        "paralysis_frac": n_par / n_eval if n_eval else float("nan"),
        "untracked_frac": n_untr / n_eval if n_eval else float("nan"),
        "chamfer": float(np.mean(cham)) if cham else 0.0,
        "time_min": float(np.sum(times)) / 60.0 if times else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--plot", action="store_true", help="salva le distribuzioni in <root>/summary.png")
    args = ap.parse_args()

    rows = []
    for hist in sorted(args.root.glob("*/*/rep_*/history.json")):
        rep = hist.parent.name
        arm = hist.parent.parent.name
        scenario = hist.parent.parent.parent.name
        r = summarize_run(hist)
        if r is None:
            continue
        rows.append({"scenario": scenario, "arm": arm, "rep": rep, **r})
    if not rows:
        print(f"nessuna run sotto {args.root}")
        return 1

    keys = ["damage_total_cm", "damage_search_cm", "final_dist_m", "windows", "heading_deg",
            "frames_per_m", "paralysis_frac", "untracked_frac", "chamfer", "time_min"]
    print(f"{'scenario':<10}{'braccio':<14}{'n':>3}  " + "  ".join(f"{k:>16}" for k in keys))
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["arm"]), []).append(r)
    for (sc, arm), rs in sorted(groups.items()):
        cells = []
        for k in keys:
            v = np.array([x[k] for x in rs], dtype=float)
            v = v[np.isfinite(v)]
            cells.append(f"{v.mean():8.2f}±{v.std(ddof=1) if len(v) > 1 else 0:6.2f}" if len(v) else f"{'-':>16}")
        print(f"{sc:<10}{arm:<14}{len(rs):>3}  " + "  ".join(cells))

    with open(args.root / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nrighe per run in {args.root/'summary.json'}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scenarios = sorted({r["scenario"] for r in rows})
        metrics = [("damage_total_cm", "danno applicato totale [cm]"),
                   ("final_dist_m", "distanza finale dal bersaglio [m]"),
                   ("chamfer", "percettibilità media [cm]")]
        fig, axes = plt.subplots(len(metrics), len(scenarios),
                                 figsize=(5.2 * len(scenarios), 3.6 * len(metrics)), squeeze=False)
        for j, sc in enumerate(scenarios):
            arms = sorted({r["arm"] for r in rows if r["scenario"] == sc})
            for i, (k, lab) in enumerate(metrics):
                ax = axes[i][j]
                data = [[r[k] for r in rows if r["scenario"] == sc and r["arm"] == a
                         and np.isfinite(r[k])] for a in arms]
                ax.boxplot(data, tick_labels=arms, showmeans=True)
                ax.set_ylabel(lab)
                if i == 0:
                    ax.set_title(sc)
                ax.tick_params(axis="x", rotation=30)
                ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.root / "summary.png", dpi=160)
        print(f"figura in {args.root/'summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
