#!/usr/bin/env python3
"""Paired comparison of NSGA-III against random search over seeds 1001, 2001, 3001, using the
statistical criterion fixed before the data. Run only when no simulation is active.

Usage: python3 confronto_semi.py [--semi 1001,2001,3001] [--out <dir>]
Reads from each run's history.json: added_cm per metre (danno_metro_vittima), lat_end_cm
(victim_window.baseline), genoma_nullo, victim_held; from the repeated trials of the winner,
the added damage per metre of the 3 trials.
The floor F_base comes from the three baseline runs against the media3semi polyline (same metric).
"""
import argparse, csv, glob, json, math, statistics as st, sys
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RADICI = ["serieB_finale_corretta_genomanullo", "serieB_finale_corretta"]   # search order
# eps 30 is not part of the comparison
BUDGET = {"senzabudget": None, "eps5": 5, "eps20": 20, "eps50": 50}
POLY = REPO / "data/attack/baseline/rettilineo_direction_N5_media3semi_dmin140_sosta1800.csv"
BASELINE_RUNS = {s: REPO / f"data/attack/verifica/2026-10-26_baseline_definitiva_s{s}_N5_dmin140_sosta1800" for s in (1001, 2001, 3001)}
V4 = Path(__file__).resolve().parent / "danno_aggiunto.py"


def rep_dir(cond, seme):
    """Completed run (file completata) for the seed: among s<seme>, s<seme>_rip1, s<seme>_rip2
    the latest completed repetition wins. Never directories under data/attack/verifica (pilot runs)."""
    for r in RADICI:
        for s in (f"s{seme}_rip2", f"s{seme}_rip1", f"s{seme}"):
            reps = sorted(glob.glob(str(REPO / f"data/attack/{r}/{cond}/{s}/campagna/straight/*/rep_01")))
            reps = [p for p in reps if (Path(p) / "completata").exists() and "/data/attack/verifica/" not in p]
            if reps:
                return Path(reps[-1]), r
    return None, None


def leggi_run(cond, seme):
    rep, rad = rep_dir(cond, seme)
    if rep is None or not (rep / "history.json").exists():
        return None
    H = [w for w in json.load(open(rep / "history.json")) if "window" in w]
    metri = []
    for w in H:
        dm = w.get("danno_metro_vittima") or {}
        vb = (w.get("victim_window") or {}).get("baseline") or {}
        prove = [(c.get("danno_metro") or {}).get("added_cm") for c in ((w.get("prove_vincitore") or {}).get("prove") or [])]
        prove = [a for a in prove if a is not None]
        # metres with the victim held: counted with their measured added damage and flagged
        ev = w.get("evaluations") or []
        metri.append(dict(k=w["window"], added=dm.get("added_cm"), lat_end=vb.get("lat_end_cm"),
                          nullo=bool(w.get("genoma_nullo")), trattenuta=bool(w.get("victim_held")), prove=prove,
                          n_cand=len(ev), n_validi=sum(1 for e in ev if not e.get("selection_invalid_reasons"))))
    metri = [m for m in metri if m["added"] is not None]
    if not metri:
        return None
    # 5-metre check: D enters the verdicts only with 5 metres; otherwise D_k over k metres, flagged
    cinque = len(metri) == 5
    D_k = sum(m["added"] for m in metri)
    # F_rip of the run: sample standard deviation of the added damage across the winner's trials
    # in each metre, summed in quadrature over the 5 metres; only if all 5 metres have >= 2 trials
    rip = [st.stdev(m["prove"]) for m in metri if len(m["prove"]) >= 2]
    f_rip = math.sqrt(sum(x * x for x in rip)) if cinque and len(rip) == 5 else None
    return dict(cond=cond, seme=seme, radice=rad, rep=str(rep), D=(D_k if cinque else None), D_k=D_k, n_metri=len(metri),
                lat_fine=metri[-1]["lat_end"], metri=metri, n_nullo=sum(m["nullo"] for m in metri),
                n_trattenuta=sum(m["trattenuta"] for m in metri), f_rip=f_rip)


def carica_v4():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v4", str(V4)); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def f_base(v4):
    poly = np.array([[float(r["x_m"]), float(r["y_m"])] for r in csv.DictReader(open(POLY))])
    out = {}
    for s, cart in BASELINE_RUNS.items():
        reps = sorted(glob.glob(str(cart / "campagna/straight/*/rep_01")))
        if not reps:
            out[s] = None; continue
        tot = 0.0; ok = True
        for k in range(5):
            cand = sorted(glob.glob(f"{reps[0]}/traces/w{k}_applied.csv")) or sorted(glob.glob(f"{reps[0]}/traces/w{k}_victim.csv"))
            if not cand:
                ok = False; break
            P = v4.pose(cand[0]); d, *_ = v4.geom(P, poly); d = d * 100
            tot += d.mean() - d[0]
        out[s] = tot if ok else None
    return out


def verdetto(delta, F, n_atteso=3):
    d = [x for x in delta if x is not None]
    if len(d) < 2:
        return "dati insufficienti"
    stesso = all(x > 0 for x in d) or all(x < 0 for x in d)
    if len(d) == n_atteso and stesso and min(abs(x) for x in d) > F and abs(st.mean(d)) > 2 * F:
        return "differenza reale"
    sopra = [x for x in d if abs(x) > F]
    segni = {math.copysign(1, x) for x in sopra}
    if stesso and len(d) == n_atteso:
        return "tendenza" + (" (tutti entro il fondo)" if all(abs(x) <= F for x in d) else "")
    # all above the floor with the same sign except at most one within the floor (2 of 3, 3 of 4)
    if len(sopra) >= 2 and len(segni) == 1 and len(d) - len(sopra) <= 1:
        entro = [x for x in d if abs(x) <= F]
        return "tendenza" + (" (un seme discorde entro il fondo)" if entro and math.copysign(1, entro[0]) not in segni else "")
    if len(segni) == 2:
        return "non distinguibili (eterogeneo fra semi)"
    if all(abs(x) <= F for x in d):
        return "non distinguibili (tutti entro il fondo)"
    return "non distinguibili"


ORDINE = ["dati insufficienti", "non distinguibili", "tendenza", "differenza reale"]


def piu_prudente(a, b):
    """Between two verdicts (with F3 and with F4) the lower category wins."""
    ca = next(i for i, c in enumerate(ORDINE) if a.startswith(c)); cb = next(i for i, c in enumerate(ORDINE) if b.startswith(c))
    return a if ca <= cb else b


def effetto(D, F, n_atteso):
    """Effect against the baseline: D(s) > F in all seeds, <= F in all, otherwise mixed."""
    d = [x for x in D if x is not None]
    if not d:
        return "dati insufficienti", 0
    k = sum(1 for x in d if x > F)
    sotto = sum(1 for x in d if x < -F)        # D < -F: attack pulls back to the baseline (counted)
    n = f" (solo {len(d)} semi su {n_atteso})" if len(d) < n_atteso else ""
    if k == len(d):
        return "effetto oltre la variabilita' nominale" + n, sotto
    if k == 0:
        return "nessun effetto oltre la variabilita' nominale" + n, sotto
    return f"misto, {k} su {len(d)} sopra F" + n, sotto


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--semi", default="1001,2001,3001")
    ap.add_argument("--out", default=str(REPO / "results" / "confronto_semi"))
    a = ap.parse_args(); semi = [int(x) for x in a.semi.split(",")]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for b, s in product(BUDGET, semi):
        for braccio in ("nsga3", "casuale", "randpuro"):
            runs[(braccio, b, s)] = leggi_run(f"{braccio}_{b}", s)
    v4 = carica_v4(); fb = f_base(v4)
    F_base = max(abs(x) for x in fb.values() if x is not None) if any(x is not None for x in fb.values()) else float("nan")
    # F_rip: mean over runs with the same seed and budget, then median over all groups
    gruppi = {}
    for (braccio, b, s), r in runs.items():
        if r and r["f_rip"] is not None:
            gruppi.setdefault((b, s), []).append(r["f_rip"])
    fr = [st.mean(v) for v in gruppi.values()]
    F_rip = st.median(fr) if fr else float("nan")
    # F3: F_rip over seeds 1001-3001 only; F4: over all seeds read
    fr3 = [st.mean(v) for (b, s), v in gruppi.items() if s <= 3001]
    F_rip3 = st.median(fr3) if fr3 else float("nan")
    F3 = max(x for x in (F_base, F_rip3) if x == x)
    F4 = max(x for x in (F_base, F_rip) if x == x)
    incomplete = {f"{k[0]}_{k[1]}_s{k[2]}": r["n_metri"] for k, r in runs.items() if r and r["n_metri"] != 5}
    if incomplete:
        print("run con meno di 5 metri, escluse dai verdetti:", incomplete)
    F = F4
    D = lambda br, b: [runs[(br, b, s)]["D"] if runs[(br, b, s)] else None for s in semi]
    def riga_conf(b, br_a, br_b):
        Da, Db = D(br_a, b), D(br_b, b)
        delta = [x - y if (x is not None and y is not None) else None for x, y in zip(Da, Db)]
        v3, v4 = verdetto(delta, F3, len(semi)), verdetto(delta, F4, len(semi))
        return dict(budget=b, confronto=f"{br_a} - {br_b}", **{f"D_{br_b}_s{s}": x for s, x in zip(semi, Db)},
                    **{f"D_{br_a}_s{s}": x for s, x in zip(semi, Da)}, **{f"delta_s{s}": x for s, x in zip(semi, delta)},
                    media_delta=st.mean([x for x in delta if x is not None]) if any(x is not None for x in delta) else None,
                    verdetto_F3=v3, verdetto_F4=v4, verdetto=piu_prudente(v3, v4),
                    regime=("verdetto a tre semi, provvisorio fino al quarto seme"
                            if len(semi) < 4 else "verdetto a quattro semi"))
    righe = [riga_conf(b, "casuale", "nsga3") for b in BUDGET]
    righe_rp = [riga_conf(b, a, "randpuro") for b in BUDGET for a in ("casuale", "nsga3")]
    with open(out / "confronto_appaiato.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0])); w.writeheader(); w.writerows(righe)
    with open(out / "confronto_randpuro.csv", "w", newline="") as f:
        campi = list(dict.fromkeys(k for r in righe_rp for k in r))
        w = csv.DictWriter(f, fieldnames=campi); w.writeheader(); w.writerows(righe_rp)
    eff = []
    for br, b in product(("nsga3", "casuale", "randpuro"), BUDGET):
        lab, sotto = effetto(D(br, b), F, len(semi))
        eff.append(dict(braccio=br, budget=b, **{f"D_s{s}": x for s, x in zip(semi, D(br, b))}, effetto=lab, n_sotto_meno_F=sotto,
                        n_metri_genoma_nullo={f"s{s}": runs[(br, b, s)]["n_nullo"] for s in semi if runs[(br, b, s)]}))
    with open(out / "effetto_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eff[0])); w.writeheader(); w.writerows(eff)
    # valid candidates per metre, NSGA-III against random (reported separately)
    cv = []
    for b, s in product(BUDGET, semi):
        rn, rc = runs[("nsga3", b, s)], runs[("casuale", b, s)]
        if not rn and not rc:
            continue
        for k in range(5):
            mn = rn["metri"][k] if rn and k < len(rn["metri"]) else None
            mc = rc["metri"][k] if rc and k < len(rc["metri"]) else None
            cv.append(dict(budget=b, seme=s, metro=k + 1, candidati_nsga3=mn and mn["n_cand"], validi_nsga3=mn and mn["n_validi"],
                           candidati_casuale=mc and mc["n_cand"], validi_casuale=mc and mc["n_validi"]))
    with open(out / "candidati_validi_per_metro.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cv[0])); w.writeheader(); w.writerows(cv)
    with open(out / "runs.json", "w") as f:
        json.dump({f"{k[0]}_{k[1]}_s{k[2]}": v for k, v in runs.items()}, f, indent=1)
    with open(out / "pavimento.json", "w") as f:
        json.dump(dict(F_base_per_seme=fb, F_base=F_base, F_rip=F_rip, F_rip_3semi=F_rip3, F3=F3, F4=F4, F=F, n_gruppi_f_rip=len(fr),
                       f_rip_per_gruppo={f"{b}_s{s}": st.mean(v) for (b, s), v in gruppi.items()},
                       run_incomplete=incomplete), f, indent=1)
    print(f"F_base {F_base:.3f}  F_rip {F_rip:.3f}  F3 {F3:.3f}  F4 {F4:.3f}  F {F:.3f} cm")
    for r in righe + righe_rp:
        print(r["confronto"], r["budget"], r["verdetto"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k.startswith("delta")})
    for e in eff:
        print("effetto", e["braccio"], e["budget"], e["effetto"], "sotto -F:", e["n_sotto_meno_F"])


if __name__ == "__main__":
    sys.exit(main())
