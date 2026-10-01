#!/usr/bin/env python3
"""
Aggregazione delle ripetizioni di una campagna: una riga per run, media,
deviazione standard e numero di ripetizioni per braccio, danno per finestra,
distribuzioni in figura.

Considera solo le run complete: cartelle rep_NN (due cifre) con il file
"completata" scritto da scripts/run_campaign.sh. Le cartelle
rep_NN_incompleta_* e rep_NN_stallo* sono ignorate. La radice puo' essere
data/attack/campaign (tutti gli scenari) o la cartella di uno scenario.

Per ogni run, sommando sulle finestre:
  damage_real_cm     danno del genoma applicato (deviazione media lungo il
                     tratto + termine direzionale), come nella ricerca
  dev_real_mean_cm   deviazione media lungo il tratto dell'applicato dal nominale
  dev_real_end_cm    deviazione al termine del tratto dell'applicato dal nominale
e inoltre:
  damage_search_cm   somma del danno del genoma scelto in ricerca
  final_dist_m       distanza vera fra l'ultimo punto d'arrivo e il bersaglio finale
  windows            finestre eseguite
  discarded          valutazioni di ricerca scartate (esito diverso da ok/arrived)
  evaluations        valutazioni di ricerca totali
  chamfer_cm         Chamfer media dei genomi applicati, misurata in ricerca
                     (la history non registra quella del rollout applicato)
  time_min           somma dei tempi di finestra (avvio e chiusura di Isaac esclusi)
  heading_deg, frames_per_m, paralysis_frac, untracked_frac
Per il braccio "none" (solo nominali) danni, Chamfer e tempo non esistono.

Uso:
    python3 src/analysis/aggregate_campaign.py data/attack/campaign/straight [--plot]
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

REP_RE = re.compile(r"^rep_\d{2}$")
VALID = ("ok", "arrived")
NAN = float("nan")


def summarize_run(path: Path):
    with open(path) as f:
        hist = [w for w in json.load(f) if w.get("status") != "isaac_stall"]
    if not hist:
        return None, []
    goal_final = None
    sums = {"damage_real_cm": [], "dev_real_mean_cm": [], "dev_real_cm": [], "damage_cm": []}
    heading, fpm, cham, times = [], [], [], []
    n_eval = n_disc = n_par = n_untr = 0
    last_end = None
    per_window = []
    from collections import Counter
    outcomes = Counter()
    loc_app, loc_nom, loc_nom_raw = [], [], []
    for w in hist:
        goal_final = w.get("goal", goal_final)
        if w.get("applied_end"):
            last_end = w["applied_end"]
        elif w.get("nominal_end") and "genome" not in w:
            last_end = w["nominal_end"]
        for k in sums:
            if w.get(k) is not None:
                sums[k].append(w[k])
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
        ev = w.get("evaluations", [])
        n_eval += len(ev)
        for e in ev:
            base = str(e.get("status", "?")).split("(")[0]
            outcomes[base] += 1
            counted = e.get("counted", e.get("status") in VALID)
            if not counted:
                n_disc += 1
        n_par += w.get("n_stopped_early", 0)
        n_untr += w.get("n_untracked", 0)
        for k in ("applied_loc_err_mean_cm", "loc_err_mean_cm"):
            if w.get(k) is not None:
                loc_app.append(w[k]); break
        if w.get("nominal_loc_err_mean_cm") is not None:
            loc_nom.append(w["nominal_loc_err_mean_cm"])
        if w.get("loc_err_mean_cm") is not None and w.get("search") == "none":
            loc_nom.append(w["loc_err_mean_cm"])
        for k in ("nominal_loc_err_mean_raw_cm", "loc_err_mean_raw_cm"):
            if w.get(k) is not None:
                loc_nom_raw.append(w[k]); break
        if w.get("damage_real_cm") is not None or w.get("search") == "none":
            per_window.append({"window": w["window"],
                               "damage_real_cm": w.get("damage_real_cm", 0.0),
                               "chamfer_cm": w.get("perturbation", NAN),
                               "loc_err_app_cm": w.get("applied_loc_err_mean_cm", NAN),
                               "loc_err_nom_cm": w.get("nominal_loc_err_mean_cm", NAN)})
    final_dist = (math.hypot(goal_final[0] - last_end[0], goal_final[1] - last_end[1])
                  if goal_final and last_end else NAN)
    total = lambda k: float(np.sum(sums[k])) if sums[k] else NAN
    return {
        "damage_real_cm": total("damage_real_cm"),
        "dev_real_mean_cm": total("dev_real_mean_cm"),
        "dev_real_end_cm": total("dev_real_cm"),
        "damage_search_cm": total("damage_cm"),
        "final_dist_m": final_dist,
        "windows": len(hist),
        "discarded": n_disc,
        "evaluations": n_eval,
        "chamfer_cm": float(np.mean(cham)) if cham else NAN,
        "time_min": float(np.sum(times)) / 60.0 if times else NAN,
        "heading_deg": float(np.mean(heading)) if heading else NAN,
        "frames_per_m": float(np.mean(fpm)) if fpm else NAN,
        "paralysis_frac": n_par / n_eval if n_eval else NAN,
        "untracked_frac": n_untr / n_eval if n_eval else NAN,
        "loc_err_app_cm": float(np.sum(loc_app)) if loc_app else NAN,
        "loc_err_nom_cm": float(np.sum(loc_nom)) if loc_nom else NAN,
        "loc_err_nom_raw_cm": float(np.sum(loc_nom_raw)) if loc_nom_raw else NAN,
        "outcomes": dict(outcomes),
    }, per_window


def stat(values, fmt="{:.2f}"):
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "-"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{fmt.format(v.mean())} ± {fmt.format(sd)}"


def main():
    from collections import Counter
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--plot", action="store_true", help="salva le distribuzioni in <root>/summary.png")
    args = ap.parse_args()

    rows, windows = [], []
    skipped = []
    for d in sorted(p.parent for p in args.root.rglob("history.json")):
        if not REP_RE.match(d.name):
            continue
        if not (d / "completata").exists():
            skipped.append(str(d))
            continue
        arm, scenario = d.parent.name, d.parent.parent.name
        r, pw = summarize_run(d / "history.json")
        if r is None:
            continue
        rows.append({"scenario": scenario, "arm": arm, "rep": d.name, **r})
        windows += [{"scenario": scenario, "arm": arm, "rep": d.name, **w} for w in pw]
    if not rows:
        print(f"nessuna run completa sotto {args.root}")
        return 1

    cols = [("damage_real_cm", "danno appl. cm"), ("dev_real_mean_cm", "dev media cm"),
            ("dev_real_end_cm", "dev fine cm"), ("final_dist_m", "dist. finale m"),
            ("windows", "finestre"), ("discarded", "scartati"), ("chamfer_cm", "Chamfer cm"),
            ("loc_err_app_cm", "err.localizz cm"), ("time_min", "tempo min")]
    print("Medie per braccio (media ± dev.std sulle ripetizioni; danni e deviazioni sommati sulle finestre)")
    print(f"{'scenario':<10}{'braccio':<16}{'n':>3}  " + "  ".join(f"{h:>17}" for _, h in cols))
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["arm"]), []).append(r)
    for (sc, arm), rs in sorted(groups.items()):
        cells = [stat([x[k] for x in rs]) for k, _ in cols]
        print(f"{sc:<10}{arm:<16}{len(rs):>3}  " + "  ".join(f"{c:>17}" for c in cells))

    print("\nFrequenza degli esiti delle valutazioni per braccio (quota sul totale)")
    for (sc, arm), rs in sorted(groups.items()):
        tot = Counter()
        for r in rs:
            tot.update(r.get("outcomes", {}))
        n = sum(tot.values())
        if n:
            quote = "  ".join(f"{k} {v}/{n} ({100*v/n:.0f}%)" for k, v in sorted(tot.items()))
            print(f"  {sc}/{arm}: {quote}")

    print("\nErrore di localizzazione (divario stima-verità, somma sulle finestre, cm). "
          "nom allineato = allo stamp dello scan (errore vero); nom grezzo = alla posa corrente (con latenza)")
    print(f"{'scenario':<10}{'braccio':<16}{'n':>3}  {'nom allineato':>17}  {'nom grezzo':>17}  {'applicato':>17}")
    for (sc, arm), rs in sorted(groups.items()):
        print(f"{sc:<10}{arm:<16}{len(rs):>3}  {stat([x['loc_err_nom_cm'] for x in rs]):>17}  "
              f"{stat([x['loc_err_nom_raw_cm'] for x in rs]):>17}  "
              f"{stat([x['loc_err_app_cm'] for x in rs]):>17}")

    if windows:
        print("\nDanno applicato e Chamfer del genoma applicato per finestra")
        print(f"{'scenario':<10}{'braccio':<16}{'fin.':>4}{'n':>4}  {'danno cm media ± std':>22}  "
              f"{'min':>6} {'mediana':>7} {'max':>6}  {'Chamfer cm media ± std':>24}")
        wg = {}
        for w in windows:
            wg.setdefault((w["scenario"], w["arm"], w["window"]), []).append(w)
        for (sc, arm, k), ws in sorted(wg.items()):
            d = np.array([w["damage_real_cm"] for w in ws], dtype=float)
            print(f"{sc:<10}{arm:<16}{k + 1:>4}{len(ws):>4}  {stat(d):>22}  {d.min():6.2f} "
                  f"{np.median(d):7.2f} {d.max():6.2f}  {stat([w['chamfer_cm'] for w in ws]):>24}")

        print("\nErrore di localizzazione per finestra (divario stima-verità, cm): "
              "nominale e applicato")
        print(f"{'scenario':<10}{'braccio':<16}{'fin.':>4}{'n':>4}  {'nominale':>17}  {'applicato':>17}")
        for (sc, arm, k), ws in sorted(wg.items()):
            print(f"{sc:<10}{arm:<16}{k + 1:>4}{len(ws):>4}  {stat([w['loc_err_nom_cm'] for w in ws]):>17}  "
                  f"{stat([w['loc_err_app_cm'] for w in ws]):>17}")

    if skipped:
        print(f"\nignorate {len(skipped)} cartelle rep_NN senza 'completata': " + ", ".join(skipped))

    with open(args.root / "summary.json", "w") as f:
        json.dump({"runs": rows, "windows": windows}, f, indent=2)
    print(f"\nrighe per run e per finestra in {args.root/'summary.json'}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scenarios = sorted({r["scenario"] for r in rows})
        metrics = [("damage_real_cm", "danno applicato totale [cm]"),
                   ("final_dist_m", "distanza finale dal bersaglio [m]"),
                   ("chamfer_cm", "Chamfer media dei genomi applicati [cm]")]
        fig, axes = plt.subplots(len(metrics), len(scenarios),
                                 figsize=(6.0 * len(scenarios), 3.6 * len(metrics)), squeeze=False)
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
