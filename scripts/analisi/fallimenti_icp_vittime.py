#!/usr/bin/env python3
"""ICP rejections and failures of the victim's MOLA instance, over all runs of the three seeds
(inventario_fronti/run.csv) and the completed ablation runs. From logs/mola_victim.log, lines
"[LidarOdometry] pathStep=... timestamp=... goodness=G ... isGood=B" (one per scan, first
occurrence):
- ICP failure: isGood=0 with goodness = 0 (no matching);
- quality rejection: isGood=0 with 0 < goodness < threshold (minRequired, 0.20).
Episodes: consecutive scans with gaps <= 0.15 s; duration = last - first + 0.1 s.
Metre: victim stamp interval in the pose.csv log (tag w<k>_applied); outside the intervals the
previous metre is assigned. State: "moving" if in controllore.csv (victim role) the last command
with pose stamp within 0.5 s before the scan has cmd_v > 0; otherwise "stopped" (settling, coast,
pause). Read only."""
import bisect, csv, glob, os, re, sys

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "attack", "")
INV = os.path.join(A, "..", "..", "results", "inventario", "run.csv")
OUT = os.path.join(A, "..", "..", "results", "fallimenti_icp")
PAT = re.compile(r"\[LidarOdometry\] pathStep=\d+ timestamp=1970/01/01,00:00:([0-9.]+) goodness=([0-9.]+) minRequired=([0-9.]+).*?isGood=(\d)")

runs = [(r["seme"], r["condizione"], A + r["cartella"]) for r in csv.DictReader(open(INV)) if r["cartella"]]
for rep in sorted(glob.glob(A + "serieB_finale_corretta_orizzonte_H*/*/s*/campagna/straight/*/rep_01")):
    if os.path.exists(rep + "/completata"):
        p = rep.split("/data/attack/")[1].split("/")
        runs.append((p[2].lstrip("s"), f"{p[1]}_{p[0].split('_')[-1]}", rep))

episodi, riepilogo = [], []
for seme, cond, rep in runs:
    lg = rep + "/logs/mola_victim.log"
    if not os.path.exists(lg):
        riepilogo.append(dict(seme=seme, condizione=cond, scansioni=None, nota="log della vittima assente")); continue
    scan = {}
    for l in open(lg, errors="replace"):
        m = PAT.search(l)
        if m and float(m.group(1)) not in scan:
            scan[float(m.group(1))] = (float(m.group(2)), float(m.group(3)), int(m.group(4)))
    # metres from the victim pose stamps
    intervalli = []
    pf = rep + "/registri_dal_vivo/pose.csv"
    if os.path.exists(pf):
        per = {}
        for q in csv.DictReader(open(pf)):
            if q["ruolo"] == "vittima":
                per.setdefault(q["tag"], []).append(float(q["stamp_nuvola_s"]))
        intervalli = sorted((min(v), max(v), t) for t, v in per.items())
    cmd = []
    cf = rep + "/registri_dal_vivo/controllore.csv"
    if os.path.exists(cf):
        for q in csv.DictReader(open(cf)):
            if q["ruolo"] == "vittima" and q["stamp_posa_grezzo_s"]:
                cmd.append((float(q["stamp_posa_grezzo_s"]), float(q["cmd_v"])))
        cmd.sort()
    cs = [c[0] for c in cmd]
    def metro(s):
        if not intervalli: return None
        prec = None
        for a, b, t in intervalli:
            if a <= s <= b: return t
            if a <= s: prec = t
        return prec
    def stato(s):
        if not cmd: return None
        i = bisect.bisect_right(cs, s) - 1
        return "in moto" if i >= 0 and s - cs[i] <= 0.5 and cmd[i][1] > 0 else "fermo"
    cattive = sorted((s, g, mr) for s, (g, mr, ok) in scan.items() if ok == 0)
    for tipo, sel in (("fallimento ICP", [x for x in cattive if x[1] == 0.0]), ("rifiuto qualita'", [x for x in cattive if x[1] > 0.0])):
        grp = []
        for s, g, mr in sel:
            if grp and s - grp[-1][-1] <= 0.15: grp[-1].append(s)
            else: grp.append([s])
        for gg in grp:
            gs = [scan[s][0] for s in gg]
            episodi.append(dict(seme=seme, condizione=cond, tipo=tipo, metro=metro(gg[0]), stato_inizio=stato(gg[0]), stato_fine=stato(gg[-1]),
                                stamp_inizio=gg[0], scansioni=len(gg), durata_s=round(gg[-1] - gg[0] + 0.1, 2),
                                goodness_max=round(max(gs), 3), soglia=scan[gg[0]][1]))
    n_f = sum(1 for x in cattive if x[1] == 0.0); n_r = len(cattive) - n_f
    riepilogo.append(dict(seme=seme, condizione=cond, scansioni=len(scan), fallimenti_icp=n_f, rifiuti_qualita=n_r,
                          registri=bool(intervalli and cmd), cartella=rep.split("/data/attack/")[1]))

os.makedirs(OUT, exist_ok=True)
for nome, rr in (("episodi.csv", episodi), ("riepilogo_per_run.csv", riepilogo)):
    if rr:
        campi = list(dict.fromkeys(k for r in rr for k in r))
        with open(f"{OUT}/{nome}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campi); w.writeheader(); w.writerows(rr)
tot_s = sum(r["scansioni"] or 0 for r in riepilogo)
print(f"run {len(riepilogo)}, scansioni della vittima {tot_s}, fallimenti ICP {sum(r.get('fallimenti_icp', 0) for r in riepilogo)}, "
      f"rifiuti per qualita' {sum(r.get('rifiuti_qualita', 0) for r in riepilogo)}, episodi {len(episodi)}")
for e in episodi:
    print(f"  {e['seme']} {e['condizione']:24s} {e['tipo']:16s} {e['metro']} {e['stato_inizio']}->{e['stato_fine']} "
          f"stamp {e['stamp_inizio']:.1f} scansioni {e['scansioni']} durata {e['durata_s']} s goodness max {e['goodness_max']}")
for r in riepilogo:
    if r.get("scansioni") is None or not r.get("registri"):
        print("  nota:", r["seme"], r["condizione"], r.get("nota", "registri assenti: metro o stato non assegnabili"))
