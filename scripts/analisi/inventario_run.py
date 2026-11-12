#!/usr/bin/env python3
"""Inventory for the per-seed Pareto-front and trajectory figures: per seed and condition,
completed run and directory; per metre, recorded and valid candidates, damage fields and P,
applied genome and its origin, values measured on the victim; final true lateral offset from the
traces compared with the previously reported value. Read only."""
import csv, glob, json, os

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "attack")
OUT = os.path.join(A, "..", "..", "results", "inventario")
COND = ["nsga3_senzabudget", "casuale_senzabudget", "nsga3_eps20", "casuale_eps20", "nsga3_eps50", "casuale_eps50",
        "randpuro_senzabudget", "randpuro_eps50", "randpuro_eps20", "massimo"]
RADICI = ["serieB_finale_corretta", "serieB_finale_corretta_genomanullo"]
# final true lateral offset already reported (cm); source in the line comment
RIPORTATA = {
    ("1001", "nsga3_senzabudget"): -25.51, ("1001", "nsga3_eps50"): 27.33, ("1001", "nsga3_eps20"): 5.83,
    ("1001", "casuale_eps50"): 26.02, ("1001", "casuale_eps20"): 7.31, ("1001", "casuale_senzabudget"): -42.94,
    ("1001", "massimo"): 15.63, ("1001", "randpuro_senzabudget"): -6.58, ("1001", "randpuro_eps50"): -7.65,
    ("1001", "randpuro_eps20"): -2.13,         # laterale_prevista_per_run.csv
    ("2001", "nsga3_senzabudget"): -46.21, ("2001", "casuale_senzabudget"): -23.74, ("2001", "nsga3_eps20"): 12.74,
    ("2001", "casuale_eps20"): -8.42, ("2001", "nsga3_eps50"): -26.50,       # same CSV
    ("2001", "casuale_eps50"): -29.59, ("2001", "massimo"): 15.45,           # earlier report
    ("3001", "nsga3_senzabudget"): 84.83, ("3001", "casuale_senzabudget"): -40.53, ("3001", "nsga3_eps20"): -12.49,
    ("3001", "casuale_eps20"): 9.68, ("3001", "nsga3_eps50"): 20.34, ("3001", "casuale_eps50"): -36.27,   # earlier report
}

def trova(seme, cond):
    cand = []
    for rad in RADICI:
        for s in (f"s{seme}", f"s{seme}_rip1", f"s{seme}_rip2"):
            for rep in glob.glob(f"{A}/{rad}/{cond}/{s}/campagna/straight/*/rep_01"):
                if os.path.exists(f"{rep}/completata"):
                    cand.append(rep)
    return cand

righe_run, righe_metro = [], []
for seme in ("1001", "2001", "3001"):
    for cond in COND:
        reps = trova(seme, cond)
        if not reps:
            righe_run.append(dict(seme=seme, condizione=cond, stato="assente o non completa", cartella="", n_complete=0))
            continue
        rep = reps[-1]
        rel = os.path.relpath(rep, A)
        h = json.load(open(f"{rep}/history.json"))
        tr = [f"{rep}/traces/w{k}_applied.csv" for k in range(len(h))]
        manca_tr = [os.path.basename(t) for t in tr if not os.path.exists(t)]
        lat = None
        if not manca_tr:
            r = list(csv.DictReader(open(tr[-1])))[-1]
            lat = round(100 * float(r["true_y"]), 2)
        rip = RIPORTATA.get((seme, cond))
        righe_run.append(dict(seme=seme, condizione=cond, stato="completa", cartella=rel, n_complete=len(reps), finestre=len(h),
                              tracce_mancanti=";".join(manca_tr), laterale_tracce_cm=lat, laterale_riportata_cm=rip,
                              coincide=(None if rip is None or lat is None else abs(lat - rip) < 0.006)))
        for k, w in enumerate(h):
            ev = w.get("evaluations", [])
            validi = [e for e in ev if not e.get("selection_invalid_reasons")]
            vi = w.get("vincitore_idx")
            gen_vinc = ev[vi]["genome"] if isinstance(vi, int) and 0 <= vi < len(ev) else None
            stesso = (None if gen_vinc is None or w.get("genome") is None
                      else all(abs(a - b) < 1e-9 for a, b in zip(gen_vinc, w["genome"])))
            dmv = w.get("danno_metro_vittima") or {}
            righe_metro.append(dict(
                seme=seme, condizione=cond, metro=k, n_candidati=len(ev), n_validi=len(validi), n_valid_selection=w.get("n_valid_selection"),
                cand_danno_aggiunto=sum(1 for e in ev if (e.get("danno_metro") or {}).get("added_cm") is not None),
                cand_danno_fitness=sum(1 for e in ev if e.get("damage") is not None),
                cand_P=sum(1 for e in ev if e.get("chamfer") is not None),
                budget=w.get("chamfer_budget"), genoma_presente=w.get("genome") is not None, vincitore_idx=vi,
                genoma_uguale_vincitore=stesso,
                vittima_added_cm=dmv.get("added_cm"), vittima_P_cm=w.get("chamfer_applicato_cm"),
                vittima_danno_fitness_cm=w.get("damage_real_cm"),
                traccia=os.path.exists(f"{rep}/traces/w{k}_applied.csv")))

os.makedirs(OUT, exist_ok=True)
for nome, rr in (("run.csv", righe_run), ("metri.csv", righe_metro)):
    campi = sorted({c for r in rr for c in r}, key=lambda c: list(rr[0]).index(c) if c in rr[0] else 99)
    with open(f"{OUT}/{nome}", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=campi); wr.writeheader(); wr.writerows(rr)
for r in righe_run:
    print(f"{r['seme']} {r['condizione']:21s} {r['stato']:24s} lat tracce {r.get('laterale_tracce_cm')} riportata {r.get('laterale_riportata_cm')} coincide {r.get('coincide')} tracce mancanti [{r.get('tracce_mancanti','')}] {r['cartella']}")
print()
for r in righe_metro:
    flag = []
    if r["n_validi"] != r["n_valid_selection"]: flag.append("validi!=n_valid_selection")
    if r["cand_danno_aggiunto"] != r["n_candidati"]: flag.append(f"added mancante in {r['n_candidati'] - r['cand_danno_aggiunto']}")
    if r["genoma_uguale_vincitore"] is False: flag.append("genoma != vincitore")
    if r["vittima_added_cm"] is None or r["vittima_P_cm"] is None: flag.append("misura vittima mancante")
    if not r["traccia"]: flag.append("traccia mancante")
    print(f"{r['seme']} {r['condizione']:21s} w{r['metro']} cand {r['n_candidati']:3d} validi {r['n_validi']:3d} budget {r['budget']} vinc {r['vincitore_idx']} "
          f"vitt added {r['vittima_added_cm']} P {r['vittima_P_cm'] and round(r['vittima_P_cm'], 2)} {' | '.join(flag)}")
