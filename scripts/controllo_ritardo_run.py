#!/usr/bin/env python3
"""Per-run delay check on the node outputs towards the victim and the shadow MOLA.

Episodes and per-output conformity come from episodi() and valuta_uscita() of analisi_pilota.py.

Inputs: <dir>/frame_log.csv (--frame-log-delay) and the run in campagna/*/*/rep_NN/:
logs/mola_shadow.log (shadow topic), logs/mola_victim.log (victim topic),
traces/w<k>_applied.csv or, without attack, w<k>_victim.csv (windows);
shadow_crc.csv for the clouds omitted to the shadow.

  C1  d_min_ms = D_min within 0.01 ms on every released row (t_release_mono present)
  C2  (t_release_target_mono - t_arr_mono) = d_min_ms within 0.01 ms
  C3  delta >= -0.1 ms on victim and shadow
  C4  every row with stamp_victim has delta_victim_ms; with --shadow-topic every
      non-omitted row has delta_shadow_ms
  C5  column delta_victim_ms exists and has at least one value (otherwise not measurable)
  per output: share above 5 ms <= 2.5 % and at most 1 severe episode (max > 50 ms)
  Pass if victim and shadow both conform; a window is flagged "delay" if the victim has
  at least one cloud above 50 ms or more than 2 between 5 and 50 ms (flagged, never excluded).

Usage: python3 scripts/controllo_ritardo_run.py <run or seed dir> [--json out.json]
Exit code: 0 conforming, 3 rit4 (rerun), 4 invalid configuration, 5 not measurable.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisi_pilota import (GAP_S, LIEVI_MAX_FINESTRA, RIT_GRAVE_MS, RIT_ORARIO_MS,  # noqa: E402
                            TOPIC_USCITA, intervalli_rollout, num, valuta_uscita, vero)

D_MIN_MS = 140.0
CONF_DMIN_MS = 0.01
DELTA_MIN_MS = -0.1
MARGINE_CAUSA_MS = 5.0          # cause "compute" if proc_ms > D_min - 5 ms
ESITO_CONFORME, ESITO_RIT4, ESITO_CONFIG, ESITO_NON_MISURABILE = 0, 3, 4, 5
NOMI_ESITO = {ESITO_CONFORME: "conforme", ESITO_RIT4: "rit4 (ritardo oltre il limite, run da rifare)",
              ESITO_CONFIG: "configurazione non valida (run da rifare)",
              ESITO_NON_MISURABILE: "non misurabile (la run non vale)"}


# ----------------------------------------------------------------------------- input
def trova_cartelle(p):
    """Return (seed dir with frame_log.csv, rep_NN dir or None, notes)."""
    note = []
    if (p / "frame_log.csv").is_file():
        reps = [r for r in sorted(p.glob("campagna/*/*/rep_[0-9][0-9]")) if r.is_dir()]
        if len(reps) > 1:
            note.append(f"{len(reps)} cartelle rep_NN: si usa l'ultima, come trova_run() di analisi_pilota")
        return p, (reps[-1] if reps else None), note
    # rep dir given: the log is in the seed dir, a few levels up
    for su in list(p.parents)[:4]:
        if (su / "frame_log.csv").is_file():
            return su, p, note
    return None, (p if (p / "logs").is_dir() else None), note


def leggi_frame_log(f):
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


def topic_log(p):
    """LiDAR topic written by the launch script at the top of a MOLA log ("topic LiDAR : ...",
    after a separator line), as in vite_mola/topic_uscite of analisi_pilota."""
    if p is None or not p.is_file():
        return None
    with open(p, errors="replace") as fh:
        for i, line in enumerate(fh):
            if "topic LiDAR" in line:
                return line.split(":", 1)[1].strip()
            if i > 50:
                break
    return None


def chiave_stamp(v):
    """Join key between frame_log.csv (stamp_victim) and shadow_crc.csv (stamp)."""
    x = num(v)
    return None if x is None else round(x, 6)


def slot_omessi(seme):
    """Stamps of the clouds omitted to the shadow (shadow_crc.csv); None if the file is missing.
    The join is on the stamp, unique within a run (victim_frame_idx restarts at 0 per genome)."""
    f = seme / "shadow_crc.csv"
    if not f.is_file():
        return None
    with open(f, newline="") as fh:
        return {chiave_stamp(r.get("stamp")) for r in csv.DictReader(fh) if vero(r.get("shadow_omitted"))}


# ----------------------------------------------------------------------------- checks
def controlli(righe, colonne, d_min, ombra_propria, omesse):
    """C1-C5 on all rows: {Ck: (ok, text)}."""
    V = [r for r in righe if r.get("stamp_victim")]
    c = {}
    dv = [num(r.get("delta_victim_ms")) for r in V]
    c5 = "delta_victim_ms" in colonne and any(x is not None for x in dv)
    c["C5"] = (c5, f"colonna delta_victim_ms {'presente' if 'delta_victim_ms' in colonne else 'assente'}, "
                   f"righe con valore {sum(x is not None for x in dv)}")
    ril = [r for r in righe if num(r.get("t_release_mono")) is not None]
    fuori1 = [r for r in ril if num(r.get("d_min_ms")) is None or abs(num(r["d_min_ms"]) - d_min) > CONF_DMIN_MS]
    vals = sorted({round(num(r["d_min_ms"]), 8) for r in ril if num(r.get("d_min_ms")) is not None})
    c["C1"] = (bool(ril) and not fuori1,
               f"righe rilasciate {len(ril)}, fuori da {d_min:g} +- {CONF_DMIN_MS} ms: {len(fuori1)}; "
               f"valori {vals[:3]}{' ...' if len(vals) > 3 else ''}")
    tgt = [r for r in righe if num(r.get("t_release_target_mono")) is not None]
    fuori2 = 0
    for r in tgt:
        ta, dm = num(r.get("t_arr_mono")), num(r.get("d_min_ms"))
        if ta is None or dm is None or abs((num(r["t_release_target_mono"]) - ta) * 1000.0 - dm) > CONF_DMIN_MS:
            fuori2 += 1
    c["C2"] = (bool(tgt) and fuori2 == 0, f"righe con rilascio programmato {len(tgt)}, fuori tolleranza {fuori2}")
    neg_v = sum(1 for x in dv if x is not None and x < DELTA_MIN_MS)
    neg_o, manca_o = 0, 0
    if ombra_propria:
        for r in V:
            if omesse is not None and chiave_stamp(r.get("stamp_victim")) in omesse:
                continue
            x = num(r.get("delta_shadow_ms"))
            if x is None:
                manca_o += 1
            elif x < DELTA_MIN_MS:
                neg_o += 1
    c["C3"] = (neg_v == 0 and neg_o == 0,
               f"delta < {DELTA_MIN_MS} ms: vittima {neg_v}"
               + (f", ombra {neg_o}" if ombra_propria else " (ombra sul topic della vittima o non letta)"))
    manca_v = sum(1 for x in dv if x is None)
    c["C4"] = (manca_v == 0 and manca_o == 0,
               f"righe con stamp_victim senza delta_victim_ms {manca_v} su {len(V)}"
               + (f"; righe dell'ombra non omesse senza delta_shadow_ms {manca_o}" if ombra_propria else ""))
    return c


def stat_uscita(d, proc, d_min):
    """Per-output statistics and the (descriptive) cause of each cloud above the margin."""
    if not d:
        return None
    q, n_ep, ep_g, conf = valuta_uscita(d)
    a = np.asarray(d, float)
    oltre = [(x, p) for x, p in zip(d, proc) if x > RIT_ORARIO_MS]
    calcolo = sum(1 for _, p in oltre if p is not None and p > d_min - MARGINE_CAUSA_MS)
    nd = sum(1 for _, p in oltre if p is None)
    return {"n": len(d), "oltre_5ms": len(oltre), "quota": q,
            "p50": float(np.median(a)), "p99": float(np.percentile(a, 99, method="inverted_cdf")),
            "max": float(a.max()), "episodi": n_ep, "gravi": ep_g,
            "causa_calcolo": calcolo, "causa_catena": len(oltre) - calcolo - nd, "causa_nd": nd,
            "conforme": bool(conf)}


def finestre_vittima(V, dV, rep):
    """Windows flagged "delay". As in etichetta() of analisi_pilota, a victim cloud belongs to the
    first driven window (traces/w<k>_applied.csv) ending after its arrival, within GAP_S. Runs
    without attack use traces/w<k>_victim.csv, only when no applied file exists.
    Returns (dict, reason if not possible)."""
    if rep is None:
        return None, "cartella rep_NN non trovata"
    if not (rep / "traces").is_dir():
        return None, f"{rep / 'traces'} assente"
    ivs = intervalli_rollout(rep)
    app, fonte = sorted(x for x in ivs if x[3] == "vittima"), "w<k>_applied.csv"
    if not app:
        app, fonte = sorted(x for x in ivs if x[4].endswith("_victim")), "w<k>_victim.csv (run senza attacco)"
    if not app:
        return None, "nessun traces/w<k>_applied.csv o w<k>_victim.csv con orari"
    per_w, senza = {}, 0
    for r, x in zip(V, dV):
        t = num(r.get("t_arr_wall"))
        k = None if t is None else next((k for _, t1, k, _, _ in app if t <= t1 + GAP_S), None)
        if k is None:
            senza += 1
        else:
            per_w.setdefault(k, []).append(x)
    marc = []
    for k, v in sorted(per_w.items()):
        gravi = sum(1 for x in v if x > RIT_GRAVE_MS)
        lievi = sum(1 for x in v if RIT_ORARIO_MS < x <= RIT_GRAVE_MS)
        if gravi >= 1 or lievi > LIEVI_MAX_FINESTRA:
            marc.append({"finestra": k, "gravi": gravi, "lievi": lievi})
    return {"finestre": len(per_w), "marcate": marc, "nuvole_senza_finestra": senza, "fonte": fonte}, None


# ----------------------------------------------------------------------------- analysis
def analizza(cartella, d_min=D_MIN_MS):
    """Full delay check for one run; returns the report dict, including "codice"."""
    cartella = Path(cartella)
    seme, rep, note = trova_cartelle(cartella)
    res = {"cartella": str(cartella), "seme": str(seme) if seme else None, "rep": str(rep) if rep else None,
           "d_min_ms": d_min, "note": note}
    if seme is None:
        res.update(codice=ESITO_NON_MISURABILE, motivo="frame_log.csv non trovato")
        return res
    colonne, righe = leggi_frame_log(seme / "frame_log.csv")
    V = [r for r in righe if r.get("stamp_victim")]
    res["righe_registro"] = len(righe)
    res["victim_stream_senza_stamp_victim"] = sum(1 for r in righe
                                                  if vero(r.get("victim_stream")) and not r.get("stamp_victim"))
    t_ombra = topic_log(rep / "logs" / "mola_shadow.log") if rep else None
    t_vitt = topic_log(rep / "logs" / "mola_victim.log") if rep else None
    fonte_v = "logs/mola_victim.log"
    if t_vitt is None:
        t_vitt, fonte_v = TOPIC_USCITA["victim"], "default del nodo (mola_victim.log senza topic)"
    res["topic_vittima"], res["fonte_topic_vittima"], res["topic_ombra"] = t_vitt, fonte_v, t_ombra
    ombra_propria = t_ombra is not None and t_ombra != t_vitt
    omesse = slot_omessi(seme) if ombra_propria else set()
    if ombra_propria and omesse is None:
        note.append("shadow_crc.csv assente: nessuna nuvola considerata omessa all'ombra")
        omesse = set()
    res["ombra_propria"] = ombra_propria
    res["controlli"] = {k: {"ok": ok, "testo": t} for k, (ok, t) in
                        sorted(controlli(righe, colonne, d_min, ombra_propria, omesse).items())}

    # victim output: rows with stamp_victim, in log order
    Vd = [(r, num(r.get("delta_victim_ms"))) for r in V]
    Vd = [(r, x) for r, x in Vd if x is not None]
    dV = [x for _, x in Vd]
    res["vittima"] = stat_uscita(dV, [num(r.get("proc_ms")) for r, _ in Vd], d_min)
    if t_ombra is None:
        res["ombra"] = None
        res["ombra_modo"] = "non misurabile: topic dell'ombra non letto (logs/mola_shadow.log assente o senza topic)"
    elif not ombra_propria:
        res["ombra"] = res["vittima"]
        res["ombra_modo"] = "ombra = stesso messaggio della vittima"
    else:
        Od = [(r, num(r.get("delta_shadow_ms"))) for r in V
              if chiave_stamp(r.get("stamp_victim")) not in omesse]
        Od = [(r, x) for r, x in Od if x is not None]
        res["ombra"] = stat_uscita([x for _, x in Od], [num(r.get("proc_ms")) for r, _ in Od], d_min)
        res["ombra_modo"] = f"topic proprio {t_ombra}, delta_shadow_ms; nuvole omesse {len(omesse)}"
    fin, motivo = finestre_vittima([r for r, _ in Vd], dV, rep) if dV else (None, "nessun delta della vittima")
    res["finestre"], res["finestre_motivo"] = fin, motivo

    c = res["controlli"]
    if not c["C5"]["ok"]:
        codice, motivo = ESITO_NON_MISURABILE, "C5: delta_victim_ms assente o vuota"
    elif not all(c[k]["ok"] for k in ("C1", "C2", "C3", "C4")):
        codice = ESITO_CONFIG
        motivo = "configurazione: " + ", ".join(k for k in ("C1", "C2", "C3", "C4") if not c[k]["ok"])
    elif res["ombra"] is None:
        codice, motivo = ESITO_NON_MISURABILE, res["ombra_modo"]
    elif res["vittima"]["conforme"] and res["ombra"]["conforme"]:
        # delay only: pose identity of the shadow is checked elsewhere
        codice, motivo = ESITO_CONFORME, "ritardo di vittima e ombra conforme"
    else:
        codice = ESITO_RIT4
        motivo = "non conformi: " + ", ".join(u for u in ("vittima", "ombra") if not res[u]["conforme"])
    res["codice"], res["motivo"] = codice, motivo
    # warning only: in a rit4 run, most clouds above the margin have cause "compute"
    if codice == ESITO_RIT4:
        usc = [res["vittima"]] + ([res["ombra"]] if ombra_propria else [])
        oltre = sum(u["oltre_5ms"] for u in usc)
        calc = sum(u["causa_calcolo"] for u in usc)
        res["avviso_calcolo"] = oltre > 0 and calc > oltre / 2
    return res


def fmt_uscita(nome, s):
    if s is None:
        return f"  {nome}: non misurabile"
    return (f"  {nome}: n {s['n']}, oltre 5 ms {s['oltre_5ms']}, quota {100 * s['quota']:.2f} %, "
            f"delta p50 {s['p50']:.2f} p99 {s['p99']:.2f} max {s['max']:.2f} ms, episodi {s['episodi']}, "
            f"gravi {s['gravi']}, causa calcolo {s['causa_calcolo']} / catena {s['causa_catena']}"
            + (f" / n.d. {s['causa_nd']}" if s["causa_nd"] else "")
            + f" -> {'conforme' if s['conforme'] else 'NON conforme'}")


def stampa(res):
    print("cartella:", res["cartella"])
    print("registro:", (res["seme"] + "/frame_log.csv") if res["seme"] else "non trovato")
    print("run:", res["rep"] or "non trovata")
    for n in res["note"]:
        print("nota:", n)
    if res["seme"] is None:
        print(f"ESITO [4]: {NOMI_ESITO[res['codice']]} ({res['motivo']})")
        return
    print(f"D_min atteso {res['d_min_ms']:g} ms; righe del registro {res['righe_registro']}; "
          f"victim_stream senza stamp_victim (fuori dal denominatore) {res['victim_stream_senza_stamp_victim']}")
    print(f"topic vittima {res['topic_vittima']} ({res['fonte_topic_vittima']}); topic ombra {res['topic_ombra']}")
    print("\nconfigurazione (100 % delle righe)")
    for k, v in res["controlli"].items():
        print(f"  {k}: {'OK' if v['ok'] else 'NON OK'}  ({v['testo']})")
    print("\nuscite (margine 5 ms, quota <= 2.5 %, episodi gravi <= 1)")
    print(fmt_uscita("vittima", res["vittima"]))
    print(f"  ombra: {res['ombra_modo']}")
    if res["ombra"] is not None and res["ombra_propria"]:
        print(fmt_uscita("ombra", res["ombra"]))
    print("\nfinestre (vittima: 0 gravi e al piu' 2 lievi)")
    f = res["finestre"]
    if f is None:
        print(f"  attribuzione alle finestre non possibile: {res['finestre_motivo']}")
    else:
        marc = ", ".join(f"w{m['finestra']} (gravi {m['gravi']}, lievi {m['lievi']})" for m in f["marcate"])
        print(f"  finestre {f['finestre']}; marcate 'ritardo' {marc or 'nessuna'} (marcate, non escluse); "
              f"nuvole senza finestra {f['nuvole_senza_finestra']}; tracce {f['fonte']}")
    print(f"\nESITO [4]: {NOMI_ESITO[res['codice']]} ({res['motivo']})")
    if res["codice"] == ESITO_RIT4:
        print("  etichetta rit4: quarantena (mai cancellata), rifare con lo stesso seme come guasto della catena,"
              " al piu' 2 ripetizioni automatiche; al terzo fallimento sullo stesso seme voce [!] e stop")
        if res.get("avviso_calcolo"):
            print("  [!] senza stop: la maggioranza delle nuvole oltre il margine ha causa 'calcolo'")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cartella", help="cartella del seme (con frame_log.csv) o della run (rep_NN)")
    ap.add_argument("--json", default=None, metavar="FILE", help="scrive il rapporto in JSON")
    ap.add_argument("--d-min", type=float, default=D_MIN_MS, help="D_min atteso in ms per C1 e la causa (140)")
    a = ap.parse_args(argv)
    res = analizza(a.cartella, a.d_min)
    stampa(res)
    if a.json:
        res["esito"] = NOMI_ESITO[res["codice"]]
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
    return res["codice"]


if __name__ == "__main__":
    sys.exit(main())
