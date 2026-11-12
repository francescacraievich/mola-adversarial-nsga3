#!/usr/bin/env python3
"""Per-seed figures from the inventory data (inventario_fronti/run.csv):
(a) grid of Pareto fronts as NSGA-III sees them: P = evaluations[i].chamfer on x, cumulative fitness
    damage = 100 * evaluations[i].damage on y, front = selection.front of the run, chosen genome =
    selection.chosen_idx with the twin's evaluation value; below, the added damage measured on the
    victim (danno_metro_vittima.added_cm) per metre and condition;
(b) true victim trajectories (true_y of w*_applied.csv) against the damage baseline, same scale
    for all seeds.
Usage: figure_per_seme.py <output dir> [seeds to mark "not yet verified", e.g. 2001,3001]"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "attack", "")
INV = os.path.join(A, "..", "..", "results", "inventario", "run.csv")
BASE = A + "baseline/rettilineo_direction_N5_media3semi_dmin140_sosta1800.csv"
OUT = sys.argv[1]
NON_VERIF = set((sys.argv[2] if len(sys.argv) > 2 else "").split(",")) - {""}
COND = [("nsga3_senzabudget", "NSGA-III, no budget", "#08306b", "-"), ("nsga3_eps50", "NSGA-III, budget 50", "#2171b5", "-"),
        ("nsga3_eps20", "NSGA-III, budget 20", "#6baed6", "-"), ("casuale_senzabudget", "random search, no budget", "#7f2704", "--"),
        ("casuale_eps50", "random search, budget 50", "#d94801", "--"), ("casuale_eps20", "random search, budget 20", "#fd8d3c", "--"),
        ("randpuro_senzabudget", "pure random, no budget", "#00441b", ":"), ("randpuro_eps50", "pure random, budget 50", "#238b45", ":"),
        ("randpuro_eps20", "pure random, budget 20", "#74c476", ":"), ("massimo", "maximum (all at limit)", "#000000", "-.")]
GRIGIO = "#bdbdbd"

runs = {(r["seme"], r["condizione"]): r for r in csv.DictReader(open(INV))}
dati = {}
for (s, c), r in runs.items():
    if not r["cartella"]:
        continue
    h = json.load(open(A + r["cartella"] + "/history.json"))
    tr = []
    for k in range(len(h)):
        for q in csv.DictReader(open(f"{A}{r['cartella']}/traces/w{k}_applied.csv")):
            tr.append((float(q["true_x"]), 100 * float(q["true_y"])))
    dati[(s, c)] = (h, np.array(tr), r["cartella"].split("/")[0])

righe_csv = []
for s in ("1001", "2001", "3001"):
    nv = s in NON_VERIF
    conds = [c for c in COND if (s, c[0]) in dati]
    nr = len(conds)
    fig = plt.figure(figsize=(25, 3.0 * nr + 4.8), dpi=110)
    gs = fig.add_gridspec(nr + 1, 5, height_ratios=[1] * nr + [1.3], hspace=0.35, wspace=0.18, top=0.935, bottom=0.045)
    senza_valore = {}
    ylim_col = []
    for k in range(5):
        ys = [100 * e["damage"] for (cc, *_r) in conds for e in dati[(s, cc)][0][k]["evaluations"]
              if e.get("damage") is not None and e.get("chamfer") is not None]
        ylim_col.append((min(ys + [0]) - 1, max(ys) + 2))
    for i, (cc, lab, col, _ls) in enumerate(conds):
        h = dati[(s, cc)][0]
        for k in range(5):
            ax = fig.add_subplot(gs[i, k])
            w = h[k]; ev = w["evaluations"]; sel = w.get("selection") or {}
            nvuoti = sum(1 for e in ev if e.get("damage") is None or e.get("chamfer") is None)
            senza_valore[cc] = senza_valore.get(cc, 0) + nvuoti
            ok = [e for e in ev if e.get("damage") is not None and e.get("chamfer") is not None]
            val = [e for e in ok if not e.get("selection_invalid_reasons")]
            inv = [e for e in ok if e.get("selection_invalid_reasons")]
            ax.scatter([e["chamfer"] for e in inv], [100 * e["damage"] for e in inv], s=5, c=GRIGIO, linewidths=0, zorder=1)
            ax.scatter([e["chamfer"] for e in val], [100 * e["damage"] for e in val], s=6, c=col, alpha=0.55, linewidths=0, zorder=2)
            fr = [f["idx"] for f in (sel.get("front") or [])]
            if fr:
                pf = sorted((ev[j]["chamfer"], 100 * ev[j]["damage"]) for j in fr)
                ax.plot([p[0] for p in pf], [p[1] for p in pf], "-", color="#222222", lw=1.0, zorder=3)
                ax.scatter([p[0] for p in pf], [p[1] for p in pf], s=9, c="#222222", zorder=3)
            ci = sel.get("chosen_idx")
            if isinstance(ci, int):
                e = ev[ci]
                ax.scatter([e["chamfer"]], [100 * e["damage"]], s=130, facecolors="none", edgecolors=col, linewidths=2.4, zorder=5)
                ax.scatter([e["chamfer"]], [100 * e["damage"]], s=130, facecolors="none", edgecolors="#000000", linewidths=0.6, zorder=6)
            else:
                ax.text(0.03, 0.92, "no valid candidate:\nnull genome applied\n(no twin evaluation)", transform=ax.transAxes,
                        fontsize=8, va="top", color="#444444")
            b = w.get("chamfer_budget")
            if b:
                ax.axvline(float(b), color="#555555", ls="--", lw=1.0, zorder=0)
            ax.set_xlim(0, 100); ax.set_ylim(*ylim_col[k]); ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(f"metre {k + 1}", fontsize=13)
            if k == 0:
                ax.set_ylabel(f"{lab}\ncumulative damage [cm]", fontsize=9, color=col if col != "#000000" else "#000000")
            if i == nr - 1:
                ax.set_xlabel("perturbation P (Chamfer) [cm]", fontsize=9)
            righe_csv.append(dict(seme=s, condizione=cc, metro=k + 1, n_candidati=len(ev), n_validi=len(val), n_non_validi_disegnati=len(inv),
                                  n_senza_valore=nvuoti, n_fronte=len(fr), scelto_idx=ci,
                                  scelto_P_valutazione=(ev[ci]["chamfer"] if isinstance(ci, int) else None),
                                  scelto_danno_cumulativo_valutazione_cm=(100 * ev[ci]["damage"] if isinstance(ci, int) else None),
                                  vittima_danno_aggiunto_cm=(w.get("danno_metro_vittima") or {}).get("added_cm"),
                                  vittima_P_cm=w.get("chamfer_applicato_cm"), budget=w.get("chamfer_budget")))
    # bar row: added damage measured on the victim
    allv = [(dati[(s, cc)][0][k].get("danno_metro_vittima") or {}).get("added_cm") for (cc, *_r) in conds for k in range(5)]
    allv = [v for v in allv if v is not None]
    for k in range(5):
        ax = fig.add_subplot(gs[nr, k])
        vals = [(dati[(s, cc)][0][k].get("danno_metro_vittima") or {}).get("added_cm") for (cc, *_r) in conds]
        ax.bar(range(len(conds)), [v if v is not None else 0 for v in vals], color=[c[2] for c in conds], edgecolor="#333333", linewidth=0.4)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_ylim(min(allv + [0]) - 0.5, max(allv) + 0.8); ax.set_xticks([]); ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.set_ylabel("added damage\nmeasured on victim [cm]", fontsize=10)
        ax.set_title(f"metre {k + 1}: measured on the victim", fontsize=10, color="#444444")
    handles = [plt.Line2D([], [], ls="", marker="s", ms=9, color=c[2], label=c[1]) for c in conds]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=4, color=GRIGIO, label="invalid candidate"),
                plt.Line2D([], [], ls="-", marker="o", ms=3, color="#222222", label="Pareto front computed by the run"),
                plt.Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec="#000000", mew=1.6, label="chosen genome, as evaluated (twin)"),
                plt.Line2D([], [], ls="--", color="#555555", label="Chamfer budget")]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.982))
    radici = sorted({dati[(s, cc)][2] for (cc, *_r) in conds})
    tot_vuoti = sum(senza_valore.values())
    tit = f"Seed {s}: Pareto fronts as seen by NSGA-III (twin evaluation) and added damage measured on the victim"
    if nv:
        tit += "   [NOT YET VERIFIED]"
    fig.suptitle(tit, fontsize=16, y=0.997, color="#b30000" if nv else "#000000")
    cap = ("Top rows: values from the twin's evaluation of each candidate (P = Chamfer, cumulative damage = fitness); "
           "front = the one computed by the run; ring = chosen genome as evaluated (twin). Bottom row: added damage measured on the victim in that metre. "
           f"Candidates without a value, not drawn: {tot_vuoti}. Data from: {', '.join(radici)}.")
    if nv:
        cap += " Values not yet verified."
    fig.text(0.5, 0.012, cap, ha="center", va="bottom", fontsize=10, wrap=True)
    nome = f"{OUT}/pareto_fronts_twin_evaluation_and_victim_added_damage_seed{s}.png"
    fig.savefig(nome, bbox_inches="tight"); plt.close(fig)
    print("->", nome, "| candidati senza valore", tot_vuoti, senza_valore, "| radici", radici)

with open(f"{OUT}/pareto_fronts_per_seed_numbers.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(righe_csv[0])); wr.writeheader(); wr.writerows(righe_csv)

# (b) trajectories
b = np.loadtxt(BASE, delimiter=",", skiprows=1)
yy = [v for (s, c), d in dati.items() for v in d[1][:, 1]]
ylo, yhi = min(yy) - 5, max(yy) + 5
for s in ("1001", "2001", "3001"):
    nv = s in NON_VERIF
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=150)
    ax.plot(b[:, 1], 100 * b[:, 2], color="#999999", ls="--", lw=1.4, label="damage baseline (no attack)")
    for cc, lab, col, ls in COND:
        if (s, cc) not in dati:
            continue
        tr = dati[(s, cc)][1]
        ax.plot(tr[:, 0], tr[:, 1], color=col, ls=ls, lw=2.2, label=f"{lab}  (end {tr[-1, 1]:+.1f} cm)")
        ax.plot(tr[-1, 0], tr[-1, 1], "o", color=col, ms=6)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_ylim(ylo, yhi); ax.set_xlim(-0.1, 5.6); ax.grid(alpha=0.3)
    ax.set_xlabel("true x along the straight [m]", fontsize=12); ax.set_ylabel("true lateral offset of the victim [cm] (left +)", fontsize=12)
    tit = f"Seed {s}: true trajectory of the victim, all conditions vs the damage baseline (same scale for all seeds)"
    ax.set_title(tit + ("   [NOT YET VERIFIED]" if nv else ""), fontsize=13, color="#b30000" if nv else "#000000")
    ax.legend(fontsize=9, ncol=2, frameon=False, loc="best")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    nome = f"{OUT}/true_trajectories_all_conditions_seed{s}.png"
    fig.tight_layout(); fig.savefig(nome); plt.close(fig)
    print("->", nome)
print("scala comune delle traiettorie:", round(ylo, 1), round(yhi, 1), "cm")
