#!/usr/bin/env python3
"""
Python side of the campaign queue (scripts/coda_serieB.sh): list of conditions, order in
blocks per seed, file-based state with resume, outcome of each run, hours and disk
estimate, daily text summary.

No chain process is started here: the script only reads and writes files (queue state,
esito_run.json, summaries).

Subcommands (internal use by the queue script):
  condizioni                          table of conditions
  seleziona --semi S... ID...          runs in (seed, condition) order, one per line
  stima --pilota D --ore CSV ... RUN   hours and disk per run and in total
  cartella --radice R COND SEME        first free directory for the next attempt
  prossima STATO RUN...                next run to do (empty if none)
  inizia STATO COND SEME CARTELLA      mark an attempt as running
  chiudi STATO COND SEME ESITO_JSON    record the outcome of an attempt
  ripara STATO                         attempts left "running" by a crashed queue
  classifica ... [--regola-crash]      run outcome from the collected facts (esito_run.json);
                                       --regola-crash: victim MOLA process crash = crash_processo
  storia REP_DIR                       end-of-run reasons in history.json
  ombra REP_DIR                        per-metre shadow/victim identity (shadow_check), JSON
  mostra STATO RUN...                  state of the selected runs
  riepilogo STATO CARTELLA [RUN...]    daily text summary (--disco: free GB and disk estimate
                                       for the remaining runs)
  frazione-tolta CARTELLA_RUN          fraction of points removed by dropout per cloud and per
                                       metre, victim and twin (zone log; --csv for the clouds)
  riepilogo-run CARTELLA_RUN           victim rejections and pose losses, delay check, end of
                                       run, Chamfer and damage, victim MOLA time per run/window
"""

import argparse
import csv
import fcntl
import glob
import json
import os
import re
import sys
import time
from contextlib import contextmanager

SOGLIE = ("5", "10", "20", "30", "50")
SEMI_D5 = ("1001", "2001", "3001")

# Automatic repetitions with the same seed after a chain failure.
RIPETIZIONI_MAX = 2
# Safety cap on the attempts of one run, interruptions included.
TENTATIVI_MAX = 6

ESITI_RIPETIBILI = ("guasto_catena", "oltre_ritardo", "crash_processo")
ESITI_FINALI = ("fatta", "fuori_spazio", "saltata", "crash_da_attacco")
# Victim crash rule (only with classifica --regola-crash, i.e. REGOLA_CRASH_LIMITI=1 in the
# queue, and only with a fixed genome): victim without pose and mola-cli terminated =
# crash_processo, repeated once (counts as a repetition); a second crash at the same level
# (same multiplier and same genome) = crash_da_attacco (final), otherwise guasto_catena.
# Without the option neither outcome appears.
# Victim-lost statuses that imply a terminated process when the history has no
# processo_mola_vittima field (older orchestrator).
STATI_CRASH_VITTIMA = ("victim_crash", "mola_crash")
# ritardo_*: exit codes 4 and 5 of controllo_ritardo_run.py (or a failed check): the run is
# invalid and the queue stops with a [!] entry; on restart it is redone in a new directory
ESITI_RITARDO_FERMANO = ("ritardo_configurazione", "ritardo_non_misurabile", "ritardo_errore")
ESITI_FERMANO_CODA = ("interrotta", "bloccata", "configurazione", "disco") + ESITI_RITARDO_FERMANO

# Exit codes of scripts/controllo_ritardo_run.py
RIT_CONFORME, RIT_RIT4, RIT_CONFIG, RIT_NON_MISURABILE = 0, 3, 4, 5
# Exit code of run_campaign.sh when its repetitions ended on a chain failure
RC_GUASTO_CATENA = 4

# Estimated hours (5 windows) of the no-budget conditions missing from T6: sources in _condizioni
ORE_MISURATE_CASUALE = 7.867
ORE_STIMA_RANDPURO_SENZABUDGET = 0.5


def _condizioni():
    """Campaign conditions in design order: id -> parameters."""
    c = [{"id": "baseline", "braccio": "none", "budget": "", "tipo": "baseline",
          "t6": "baseline", "descrizione": "baseline, braccio senza attacco"}]
    c.append({"id": "nsga3_senzabudget", "braccio": "nsga3", "budget": "", "tipo": "ricerca",
              "t6": "NSGA-III senza budget", "descrizione": "NSGA-III senza budget"})
    for s in SOGLIE:
        c.append({"id": f"nsga3_eps{s}", "braccio": "nsga3", "budget": s, "tipo": "ricerca",
                  "t6": f"NSGA-III soglia {s} cm", "descrizione": f"NSGA-III, budget di Chamfer {s} cm"})
    # no budget: not in T6. Max hours (run time limit) from the T6 row named by t6_come;
    # estimated hours from ore_stima_5 (5 windows)
    c.append({"id": "casuale_senzabudget", "braccio": "random", "budget": "", "tipo": "ricerca",
              "t6": "ricerca casuale senza budget", "t6_come": "ricerca casuale soglia 5 cm",
              "ore_stima_5": ORE_MISURATE_CASUALE,
              "fonte_ore": "media delle durate misurate di nsga3_senzabudget, casuale_eps20 e casuale_eps50 s1001 "
                           "(serieB_finale_corretta, eventi.txt: 7,736, 8,092, 7,774 h); massimo dalla riga T6 "
                           "'ricerca casuale soglia 5 cm' (uguale per tutte le soglie)",
              "descrizione": "ricerca casuale pop x gen, senza budget di Chamfer"})
    for s in SOGLIE:
        c.append({"id": f"casuale_eps{s}", "braccio": "random", "budget": s, "tipo": "ricerca",
                  "t6": f"ricerca casuale soglia {s} cm",
                  "descrizione": f"ricerca casuale pop x gen, budget di Chamfer {s} cm"})
    c.append({"id": "randpuro_senzabudget", "braccio": "random", "budget": "", "tipo": "randpuro",
              "t6": "random puro senza budget", "t6_come": "random puro soglia 50 cm",
              "ore_stima_5": ORE_STIMA_RANDPURO_SENZABUDGET,
              "fonte_ore": "stima, non misurata (con budget, misurate s1001: "
                           "randpuro_eps5 1,489 h, randpuro_eps10 0,858 h); massimo dalla riga T6 "
                           "'random puro soglia 50 cm' (K estrazioni)",
              "descrizione": "random puro (primo valido, al piu' K estrazioni), senza budget"})
    for s in SOGLIE:
        c.append({"id": f"randpuro_eps{s}", "braccio": "random", "budget": s, "tipo": "randpuro",
                  "t6": f"random puro soglia {s} cm",
                  "descrizione": f"random puro (primo valido, al piu' K estrazioni), budget {s} cm"})
    c.append({"id": "massimo", "braccio": "fixed", "budget": "", "tipo": "massimo",
              "t6": "massimo", "descrizione": "massimo, genoma fisso"})
    return c


CONDIZIONI = _condizioni()
PER_ID = {c["id"]: c for c in CONDIZIONI}


def espandi(voci):
    """Command-line items -> ids in canonical order. Accepts ids, 'tutte' and * patterns
    (e.g. 'nsga3_*'). An unknown id is an error."""
    scelti = set()
    for v in voci:
        for parte in re.split(r"[,\s]+", v.strip()):
            if not parte:
                continue
            if parte == "tutte":
                scelti.update(PER_ID)
                continue
            rx = re.compile("^" + re.escape(parte).replace(r"\*", ".*") + "$")
            trovati = [i for i in PER_ID if rx.match(i)]
            if not trovati:
                raise ValueError(f"condizione sconosciuta: {parte}")
            scelti.update(trovati)
    return [c["id"] for c in CONDIZIONI if c["id"] in scelti]


def ordina(semi, ids):
    """Blocks per seed: all conditions of the first seed, then of the second, and so on."""
    return [(str(s), i) for s in semi for i in ids]


def chiave(cond, seme):
    return f"s{seme}/{cond}"


# --------------------------------------------------------------- file-based state
@contextmanager
def bloccato(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fl, fcntl.LOCK_UN)


def leggi(path):
    if not os.path.exists(path):
        return {"versione": 1, "runs": {}}
    with open(path) as f:
        return json.load(f)


def scrivi(path, st):
    st["aggiornato"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def voce(st, cond, seme):
    return st["runs"].setdefault(chiave(cond, seme), {
        "condizione": cond, "seme": str(seme), "stato": "da_fare", "tentativi": []})


def conta_guasti(v):
    return sum(1 for t in v["tentativi"] if t.get("esito") in ESITI_RIPETIBILI)


def stato_run(v):
    """State from the attempts (the last one decides; chain failures are counted)."""
    if not v["tentativi"]:
        return "da_fare"
    ult = v["tentativi"][-1]
    e = ult.get("esito")
    if e is None:
        return "in_corso"
    if e in ESITI_FINALI:
        return e
    if e in ESITI_RIPETIBILI:
        if conta_guasti(v) > RIPETIZIONI_MAX:
            return "guasto_definitivo"
    if len(v["tentativi"]) >= TENTATIVI_MAX:
        return "tentativi_esauriti"
    return "da_rifare"


def da_fare(v):
    return stato_run(v) in ("da_fare", "da_rifare")


def prossima(st, runs):
    for seme, cond in runs:
        v = st["runs"].get(chiave(cond, seme))
        if v is None or da_fare(v):
            return seme, cond
    return None


def cartella_libera(radice, cond, seme):
    base = os.path.join(radice, cond, f"s{seme}")
    if not os.path.exists(base):
        return base
    n = 1
    while os.path.exists(f"{base}_rip{n}"):
        n += 1
    return f"{base}_rip{n}"


def inizia(path, cond, seme, cartella):
    with bloccato(path):
        st = leggi(path)
        v = voce(st, cond, seme)
        v["tentativi"].append({"cartella": cartella, "inizio": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "inizio_epoch": time.time(), "esito": None})
        v["stato"] = "in_corso"
        scrivi(path, st)


def chiudi(path, cond, seme, esito):
    with bloccato(path):
        st = leggi(path)
        v = voce(st, cond, seme)
        if not v["tentativi"] or v["tentativi"][-1].get("esito") is not None:
            v["tentativi"].append({"cartella": esito.get("cartella"), "inizio": None, "inizio_epoch": None})
        t = v["tentativi"][-1]
        t.update({"esito": esito.get("esito"), "motivo": esito.get("motivo", ""),
                  "dettaglio": esito.get("dettaglio", {}), "fine": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "fine_epoch": time.time()})
        if t.get("inizio_epoch"):
            t["ore"] = round((t["fine_epoch"] - t["inizio_epoch"]) / 3600.0, 3)
        regola_crash_tentativo(v)
        v["stato"] = stato_run(v)
        scrivi(path, st)
        return v["stato"]


def regola_crash_tentativo(v):
    """Crash rule (only with crash_processo outcomes, i.e. --regola-crash with a fixed genome):
    if the previous attempt was also crash_processo, the last one becomes crash_da_attacco
    (same level: same genome and multiplier, final) or guasto_catena otherwise. The raw
    classification is kept in esito_classifica."""
    if len(v["tentativi"]) < 2:
        return
    t, p = v["tentativi"][-1], v["tentativi"][-2]
    if t.get("esito") != "crash_processo" or p.get("esito") != "crash_processo":
        return
    c = [((x.get("dettaglio") or {}).get("crash") or {}) for x in (p, t)]
    t["esito_classifica"] = "crash_processo"
    # same level = same multiplier (K and families; None = x1) with the same genome
    if c[0].get("genoma") is not None and c[0].get("genoma") == c[1].get("genoma") \
            and c[0].get("moltiplicatori") == c[1].get("moltiplicatori"):
        t.update(esito="crash_da_attacco",
                 motivo=f"crash della vittima ripetuto allo stesso livello (stesso genoma, moltiplicatore "
                        f"{c[1].get('moltiplicatori') or 'x1'}; {p.get('cartella')}, {t.get('cartella')}): "
                        f"crash da attacco")
    else:
        t.update(esito="guasto_catena",
                 motivo=f"crash della vittima ripetuto con un altro genoma o un altro moltiplicatore "
                        f"({p.get('cartella')}, {t.get('cartella')}): guasto")


def ripara(path):
    """Mark attempts without outcome (queue crash, machine reboot) as interrupted.
    Call only while holding the queue lock (no other live queue)."""
    riparati = []
    with bloccato(path):
        st = leggi(path)
        for k, v in st["runs"].items():
            if v["tentativi"] and v["tentativi"][-1].get("esito") is None:
                t = v["tentativi"][-1]
                t.update({"esito": "interrotta", "motivo": "coda caduta durante la run (ripresa)",
                          "fine": time.strftime("%Y-%m-%dT%H:%M:%S"), "fine_epoch": None})
                v["stato"] = stato_run(v)
                riparati.append(f"{k} {t.get('cartella')}")
        if riparati:
            scrivi(path, st)
    return riparati


# ------------------------------------------------------------ run outcome
def storia(rep_dir, guasto_re="guasto_catena"):
    """End-of-run reasons (run_end.reason) and window count in history.json."""
    h_path = os.path.join(rep_dir, "history.json") if rep_dir else ""
    out = {"history": bool(h_path) and os.path.exists(h_path), "run_end": [], "finestre": 0,
           "guasto_catena": False, "attacco_scoperto": False, "da_rifare": None}
    prec = None       # previous window record (genome of the crash with victim_crash)
    # chain failure marker written by the orchestrator (--controllo-catena)
    dr = os.path.join(rep_dir, "da_rifare.json") if rep_dir else ""
    if dr and os.path.exists(dr):
        try:
            with open(dr) as f:
                out["da_rifare"] = json.load(f)
        except ValueError:
            out["da_rifare"] = {"errore": "da_rifare.json illeggibile"}
        out["guasto_catena"] = True
    if not out["history"]:
        return out
    try:
        with open(h_path) as f:
            h = json.load(f)
    except Exception as e:   # history partially written or unreadable
        out["errore"] = f"history illeggibile: {e}"
        return out
    rx = re.compile(guasto_re)
    for rec in h if isinstance(h, list) else []:
        out["finestre"] += 1
        p_rec = prec
        if isinstance(rec, dict) and rec.get("genome"):
            prec = rec
        if isinstance(rec, dict) and rec.get("status") == "isaac_stall" and rec.get("causa") == "eta_nuvola":
            # --controllo-eta: victim cloud age above threshold = chain failure, the whole
            # run is redone (not an Isaac stall)
            out["run_end"].append(f"isaac_stall eta_nuvola (finestra {rec.get('window')})")
            out["guasto_catena"] = True
            continue
        fine = rec.get("run_end")
        if isinstance(fine, dict):
            r = str(fine.get("reason", ""))
        elif fine:
            r = str(fine)
        else:
            continue
        out["run_end"].append(r)
        if rx.search(r):
            out["guasto_catena"] = True
        if r == "attacco_scoperto":
            out["attacco_scoperto"] = True
            out["vittima_persa"] = vittima_persa_rec(fine, rec, p_rec, rep_dir)
    return out


def _wrec():
    """The repo window_record module (None if not importable)."""
    radice = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if radice not in sys.path:
        sys.path.insert(0, radice)
    try:
        from src.optimization import window_record as wrec
        return wrec
    except Exception:     # numpy or repo modules unavailable
        return None


# Victim-lost markers with their description (copy of window_record.ESITI_VITTIMA_PERSA,
# for use when the repo modules are not importable).
TESTI_VITTIMA_PERSA = {
    "crash_processo": "crash di processo",
    "rifiuto_da_attacco": "rifiuto da attacco",
    "senza_posa_da_confermare": "senza posa, rifiuto da confermare",
    "senza_posa_senza_rifiuto": "senza posa, nessuna ICP rifiutata",
    "processo_non_trovato": "processo non trovato",
}


def vittima_persa_rec(fine, rec, prec=None, rep_dir=None):
    """Victim without pose at end of run (run_end attacco_scoperto), with a distinct marker
    (window_record.ESITI_VITTIMA_PERSA): process crash (mola-cli terminated; without
    processo_mola_vittima, status victim_crash or mola_crash), process not found (not a
    crash), attack rejection (process alive and ICP rejected after the last good one), no
    pose without rejection. Rejection is decided on the full victim log
    (rep_dir/logs/mola_victim.log, MOLA already closed); without it, the online signs from
    the orchestrator (rifiuto_in_linea) are used. Genome: that of the window, or of the
    previous window for victim_crash (process found dead before the stream opened).
    Multipliers: those of the window (None = x1)."""
    fine = fine if isinstance(fine, dict) else {}
    st = str(fine.get("status", "")).split("(")[0].strip()
    pv = fine.get("processo_mola_vittima")
    seg, fonte_seg = fine.get("rifiuto_in_linea"), "righe lette in linea dall'orchestratore"
    log = os.path.join(rep_dir, "logs", "mola_victim.log") if rep_dir else None
    wrec = _wrec() if log and os.path.exists(log) else None
    completo = False
    if wrec is not None:
        seg, fonte_seg, completo = wrec.segnali_rifiuto(wrec.read_printf(log)), "log completo a fine run", True
    if isinstance(pv, dict):
        # same rule as window_record.esito_vittima_persa (which may not be importable)
        if pv.get("esito") == "non_trovato" or pv.get("vivo") is None:
            marc = "processo_non_trovato"
        elif pv.get("vivo") is False:
            marc = "crash_processo"
        elif (seg or {}).get("rifiuto"):
            marc = "rifiuto_da_attacco"
        else:
            marc = "senza_posa_senza_rifiuto" if completo else "senza_posa_da_confermare"
        fonte = "processo_mola_vittima"
    else:
        marc = ("crash_processo" if st in STATI_CRASH_VITTIMA
                else "rifiuto_da_attacco" if (seg or {}).get("rifiuto")
                else "senza_posa_senza_rifiuto" if completo else "senza_posa_da_confermare")
        fonte = "status del rollout"
    g_rec = prec if (st == "victim_crash" and isinstance(prec, dict) and prec.get("genome")) else rec
    return {"finestra": fine.get("window"), "status": fine.get("status"),
            "crash": marc == "crash_processo", "marcatore": marc,
            "esito_vittima": TESTI_VITTIMA_PERSA.get(marc, marc),
            "processo": pv if isinstance(pv, dict) else None, "fonte": fonte,
            "segni_rifiuto": seg, "fonte_segni_rifiuto": fonte_seg if seg is not None else None,
            "genoma": (g_rec or {}).get("genome"), "genoma_finestra": (g_rec or {}).get("window"),
            "moltiplicatori": (rec or {}).get("moltiplicatori_limiti")}


def classifica(f, regola_crash=False):
    """Run outcome from the facts collected by the queue script (dict). The chain is checked
    before closing as attack detected. With regola_crash (--regola-crash, fixed genome only,
    condition type massimo) a detected attack with the victim MOLA process terminated becomes
    crash_processo (repeated), before the delay and shadow checks, which stay in the facts.
    The rule does not apply to search conditions (the queue rejects them before the run)."""
    def es(e, m):
        return {"esito": e, "motivo": m}
    if f.get("configurazione"):
        return es("configurazione", f["configurazione"])
    if f.get("bloccata"):
        return es("bloccata", f["bloccata"])
    mi = f.get("motivo_interruzione") or ""
    if mi.startswith("x odom vera oltre"):
        return es("fuori_spazio", mi)
    if f.get("tempo_scaduto"):
        return es("guasto_catena", "tempo massimo della run superato")
    if mi:
        return es("interrotta", mi)
    if f.get("segnale"):
        return es("interrotta", f"segnale {f['segnale']}")
    if int(f.get("watchdog_interventi") or 0) > 0:
        return es("guasto_catena", "Isaac terminato dal watchdog durante la run")
    s = f.get("storia") or {}
    rc = f.get("rc")
    dr = s.get("da_rifare")
    if dr is not None or (rc is not None and int(rc) == RC_GUASTO_CATENA):
        dett = []
        if rc is not None and int(rc) == RC_GUASTO_CATENA:
            dett.append(f"run_campaign.sh uscito con {RC_GUASTO_CATENA}")
        if isinstance(dr, dict):
            dett.append("da_rifare.json: " + "; ".join(str(m) for m in (dr.get("motivi") or [dr.get("status") or dr.get("errore")])))
        return es("guasto_catena", "guasto della catena (" + ", ".join(dett) + ")")
    if s.get("guasto_catena"):
        return es("guasto_catena", "run_end guasto catena nella history (" + ", ".join(s.get("run_end", [])) + ")")
    if rc is None or int(rc) != 0:
        return es("guasto_catena", f"run_campaign.sh uscito con codice {rc}")
    if not f.get("completata"):
        return es("guasto_catena", "ripetizione senza il file completata")
    if f.get("adattatore_vivo") is False:
        return es("guasto_catena", "adattatore assente a fine run")
    if f.get("baseline") and f.get("baseline_rc") not in (None, 0):
        return es("guasto_catena", f"polilinea della baseline non scritta (codice {f.get('baseline_rc')})")
    vp = s.get("vittima_persa") or {}
    fisso = (PER_ID.get(f.get("condizione")) or {}).get("tipo") == "massimo"
    if regola_crash and fisso and s.get("attacco_scoperto") and vp.get("crash"):
        pv = vp.get("processo") or {}
        out = es("crash_processo", f"crash del processo MOLA della vittima nella finestra {vp.get('finestra')} "
                                   f"({vp.get('status')}, rc {pv.get('rc')}, fonte {vp.get('fonte')}): "
                                   f"run ripetuta una volta (regola del crash)")
        out["crash"] = {k: vp.get(k) for k in ("finestra", "status", "genoma", "genoma_finestra", "fonte",
                                               "moltiplicatori", "marcatore")}
        out["crash"]["rc"] = pv.get("rc")
        return out
    out = es("fatta", "attacco scoperto" if s.get("attacco_scoperto") else "completa")
    if s.get("attacco_scoperto") and vp:
        # attack rejection, process crash, no pose without rejection, or process not found
        out["vittima"] = vp.get("esito_vittima")
        out["vittima_marcatore"] = vp.get("marcatore")
        if regola_crash and not fisso:
            out["regola_crash"] = "non applicata: vale solo a genoma fisso"
    if f.get("baseline"):
        out["ritardo"] = "non_previsto"     # no delay measurement on the baseline
    else:
        rr = f.get("ritardo_rc")
        if rr is None:
            out["ritardo"] = "non_controllato"
        elif int(rr) == RIT_CONFORME:
            out["ritardo"] = "entro_limite"
        elif int(rr) == RIT_RIT4:
            return es("oltre_ritardo", "rit4: ritardo oltre il limite (blocco [4], codice 3)")
        elif int(rr) == RIT_CONFIG:
            return es("ritardo_configurazione", "[!] blocco [4]: configurazione non valida (codice 4)")
        elif int(rr) == RIT_NON_MISURABILE:
            return es("ritardo_non_misurabile", "[!] blocco [4]: non misurabile (codice 5)")
        else:
            return es("ritardo_errore", f"[!] blocco [4]: controllo non riuscito (codice {rr})")
        # per-metre shadow/victim identity: a diverging shadow in any metre, or no comparison,
        # makes the run invalid; it is redone like a chain failure (same repetition limit)
        om = f.get("ombra")
        e_om = om.get("esito") if isinstance(om, dict) else None
        if e_om == "ombra_divergente":
            return es("guasto_catena", om.get("testo") or "ombra divergente")
        if e_om != "identica":
            dett = om.get("testo") if isinstance(om, dict) and om.get("testo") else "controllo senza uscita"
            return es("guasto_catena", f"ombra non misurata: {dett}")
        out["ombra"] = "identica"
    return out


# ------------------------------------------------------------ fallback fitness
# Candidate genes from history.json: field evaluations of each window entry (search
# candidate log, eval_log of the orchestrator), genome = full 17-gene vector. A candidate
# has the fallback fitness when its record has fitness_riserva (with the cause); in older
# histories it is reconstructed (counted false, or stealth_ok, calc_ok or accorge_ok false)
# and the block says so. Redone records (superseded_by) are in neither group.
GENI_DIREZIONE = (0, 1, 2, 7, 8, 9)   # as DIRECTION_GENES: off = 0, the other genes off = -1
RISERVA_N_MIN = 3


def _ranghi(v):
    """Average ranks (1..n) with ties."""
    ordine = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(ordine):
        j = i
        while j + 1 < len(ordine) and v[ordine[j + 1]] == v[ordine[i]]:
            j += 1
        for t in range(i, j + 1):
            r[ordine[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def mann_whitney(a, b, seme=20261029, n_perm=5000):
    """Two-sided Mann-Whitney p (descriptive, not corrected for multiple comparisons): scipy if
    available, otherwise a rank-sum permutation test with a fixed seed. Returns (p, method)."""
    a, b = [float(x) for x in a], [float(x) for x in b]
    if len(set(a + b)) == 1:
        return 1.0, "valori tutti uguali"
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue), "scipy mannwhitneyu"
    except ImportError:
        pass
    import random
    r = _ranghi(a + b)
    n1 = len(a)
    atteso = n1 * (len(r) + 1) / 2
    oss = abs(sum(r[:n1]) - atteso)
    rng = random.Random(seme)
    conta = 0
    for _ in range(n_perm):
        rng.shuffle(r)
        conta += abs(sum(r[:n1]) - atteso) >= oss - 1e-9
    return (conta + 1) / (n_perm + 1), f"permutazioni ({n_perm}, seme {seme})"


def _mediana(v):
    s = sorted(v)
    n = len(s)
    return None if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def frazione_attivi(g):
    """Fraction of active genes: off = -1, or 0 for direction genes."""
    att = [abs(x) > 1e-9 if i in GENI_DIREZIONE else x > -1 + 1e-9 for i, x in enumerate(g)]
    return sum(att) / len(att) if att else None


def fitness_riserva_metro(rec):
    """For one window entry: count with fallback fitness over all candidates, causes and,
    with at least RISERVA_N_MIN candidates per group, a gene comparison."""
    ev = [e for e in (rec.get("evaluations") or []) if isinstance(e, dict)]
    registrata = isinstance(rec.get("fitness_riserva"), dict) or any("fitness_riserva" in e for e in ev)
    ris, altri, cause = [], [], {}
    for i, e in enumerate(ev):
        if e.get("superseded_by") is not None or not isinstance(e.get("genome"), list):
            continue
        if registrata:
            c = e.get("fitness_riserva")
        else:
            c = ("ricostruita" if e.get("counted") is False or False in
                 (e.get("stealth_ok"), e.get("calc_ok"), e.get("accorge_ok")) else None)
        if c:
            ris.append(e["genome"])
            cause.setdefault(c, []).append(i)
        else:
            altri.append(e["genome"])
    out = {"finestra": rec.get("window"), "n_riserva": len(ris), "n_candidati": len(ev),
           "fonte": "registrata" if registrata else "ricostruita (history senza fitness_riserva)",
           "cause": {c: {"n": len(v), "indici": v} for c, v in sorted(cause.items())}}
    if len(ris) < RISERVA_N_MIN or len(altri) < RISERVA_N_MIN:
        out["confronto"] = None
        out["nota"] = f"confronto non fatto: servono almeno {RISERVA_N_MIN} candidati per gruppo"
        return out
    geni = []
    for j in range(min(len(g) for g in ris + altri)):
        a, b = [g[j] for g in ris], [g[j] for g in altri]
        p, metodo = mann_whitney(a, b)
        ma, mb = _mediana(a), _mediana(b)
        geni.append({"gene": j, "mediana_riserva": round(ma, 4), "mediana_altri": round(mb, 4),
                     "differenza": round(ma - mb, 4), "p": round(p, 6), "metodo": metodo})
    fa, fb = [frazione_attivi(g) for g in ris], [frazione_attivi(g) for g in altri]
    p, metodo = mann_whitney(fa, fb)
    out["confronto"] = {"geni": geni, "attivi": {"mediana_riserva": round(_mediana(fa), 4),
                                                 "mediana_altri": round(_mediana(fb), 4), "p": round(p, 6),
                                                 "metodo": metodo},
                        "nota": "p non corretti per confronti multipli: lettura descrittiva"}
    return out


def fitness_riserva_rep(rep):
    """Per-metre fallback fitness of one repetition (history.json), read-only."""
    with open(os.path.join(rep, "history.json")) as f:
        hist = json.load(f)
    return [fitness_riserva_metro(h) for h in hist if isinstance(h, dict) and h.get("window") is not None
            and "evaluations" in h]


def testo_fitness_riserva(metri):
    L = ["  fitness di riserva per metro (p non corretti, descrittivi; tabella completa nel JSON):"]
    for m in metri:
        cause = ", ".join(f"{c} {v['n']}" for c, v in m["cause"].items())
        L.append(f"    w{m['finestra']}: {m['n_riserva']}/{m['n_candidati']}" + (f" ({cause})" if cause else "")
                 + ("" if m["fonte"] == "registrata" else f" [{m['fonte']}]"))
        cf = m.get("confronto")
        if cf is None:
            if m["n_riserva"]:
                L.append(f"      {m['nota']}")
            continue
        sig = [g for g in cf["geni"] if g["p"] < 0.05]
        L.append("      geni con p < 0,05: " + (", ".join(
            f"g{g['gene']} (mediane {g['mediana_riserva']} / {g['mediana_altri']}, p {g['p']})" for g in sig)
            if sig else "nessuno"))
        at = cf["attivi"]
        L.append(f"      frazione di geni attivi: mediane {at['mediana_riserva']} / {at['mediana_altri']}, p {at['p']}")
    return L


# ------------------------------------------------------------ estimates
def disco_pilota(pilota):
    """MB per twin rollout and number of runs read from the pilot (du of the files)."""
    per, n = [], 0
    for seme in sorted(glob.glob(os.path.join(pilota, "s*"))):
        reps = glob.glob(os.path.join(seme, "campagna", "straight", "*", "rep_01"))
        if not reps:
            continue
        ris = sum(len([x for x in os.listdir(os.path.join(r, "maps")) if "risalvato" in x])
                  for r in reps if os.path.isdir(os.path.join(r, "maps")))
        tot = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(seme) for x in fs)
        if ris:
            per.append(tot / ris / 1e6)
            n += 1
    if not per:
        return None, 0
    return sum(per) / len(per), n


def mb_cartella(d):
    if not d or not os.path.isdir(d):
        return None
    return sum(os.path.getsize(os.path.join(a, x)) for a, _, fs in os.walk(d) for x in fs) / 1e6


def ore_t6(path, scenario="b"):
    """ore_condizioni.csv di T6 -> {nome: {finestre: (ore_per_run, ore_kmax)}}."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["scenario"] != scenario:
                continue
            kmax = float(r["ore_per_run_Kmax"]) if r.get("ore_per_run_Kmax") else None
            out.setdefault(r["condizione"], {})[int(r["finestre"])] = (float(r["ore_per_run"]), kmax)
    return out


def stima_run(cond, finestre, extra, pop, gen, topk, k_max, mb_rollout, mb_fisso, ore):
    """Twin rollouts per window, MB and hours of one run: [estimate, maximum]."""
    c = PER_ID[cond]
    t = c["tipo"]
    if t == "ricerca":
        rol_min = rol_max = pop * gen + topk
    elif t == "randpuro":
        # mean draws per feasible candidate (sparse generator); at most K
        rol_min, rol_max = RANDPURO_ESTRAZIONI.get(c["budget"], 2.35), k_max
    else:
        rol_min = rol_max = 0
    # windows with a re-saved shadow state: all except w0
    w_min, w_max = max(finestre - 1, 0), max(finestre + extra - 1, 0)
    mb_min = (mb_fisso or 0) + (mb_rollout or 0) * rol_min * w_min
    mb_max = (mb_fisso or 0) + (mb_rollout or 0) * rol_max * w_max
    o = ore.get(c["t6"], {})
    stima_5 = None
    if not o and c.get("t6_come") and ore.get(c["t6_come"]):
        # condition missing from T6: maximum from the t6_come row, estimate from ore_stima_5;
        # without T6, as the others (no estimate, default time limit)
        o = ore[c["t6_come"]]
        stima_5 = c.get("ore_stima_5")
    o_min = o.get(finestre, (None, None))[0]
    if stima_5 is not None:
        o_min = stima_5 * finestre / 5
    o8 = o.get(finestre + extra)
    if o8 is not None:
        o_max = o8[1] if (t == "randpuro" and o8[1] is not None) else o8[0]
    elif o_min is not None:
        o_max = o_min * (finestre + extra) / finestre
    else:
        o_max = None
    return {"condizione": cond, "rollout_per_finestra": [rol_min, rol_max],
            "mb": [round(mb_min, 1), round(mb_max, 1)], "ore": [o_min, o_max]}


def gb_liberi(percorso):
    """Free GB (10^9 bytes) for the user on the disk of percorso (as df --output=avail)."""
    s = os.statvfs(percorso)
    return s.f_bavail * s.f_frsize / 1e9


def righe_disco(disco, rimaste):
    """Disk lines of the summary. disco: percorso, gb_liberi, ora and optional stima (stima_run
    arguments without the condition), fonte and margine_gb; rimaste: (seed, condition) runs
    still to do, or None."""
    righe = [f"Disco della repo ({disco['percorso']}): {disco['gb_liberi']:.1f} GB liberi "
             f"(10^9 byte, letti alle {disco['ora']})"]
    st = disco.get("stima")
    if rimaste is None:
        righe.append("  stima per le run rimaste non calcolabile: selezione delle run non data")
    elif not st or st.get("mb_rollout") is None:
        righe.append(f"  stima per le {len(rimaste)} run rimaste non calcolabile: MB per rollout del pilota "
                     f"non misurati ({disco.get('fonte') or 'pilota non dato'})")
    else:
        mb = [0.0, 0.0]
        for _, cond in rimaste:
            r = stima_run(cond, ore={}, **st)
            mb[0] += r["mb"][0]
            mb[1] += r["mb"][1]
        riga = (f"  run rimaste {len(rimaste)}: servono circa {mb[0] / 1000:.1f} GB, al piu' {mb[1] / 1000:.1f} GB "
                f"(stima di --lista; {disco.get('fonte', '')})")
        righe.append(riga)
        m = disco.get("margine_gb")
        if m is not None:
            resta = disco["gb_liberi"] - mb[1] / 1000 - m
            righe.append(f"  dopo il massimo stimato e {m} GB di margine restano {resta:.1f} GB"
                         + (" - NON BASTA" if resta < 0 else ""))
    return righe


def tempo_max_s(ore_max, fattore=1.5, minimo_s=3600):
    """Run time limit: factor x hours of the longest expected run (T6, retained windows
    included), at least minimo_s. Without an estimate: 12 h."""
    if ore_max is None:
        return 12 * 3600
    return max(int(minimo_s), int(round(fattore * ore_max * 3600)))


# mean draws per feasible candidate of pure random search
RANDPURO_ESTRAZIONI = {"5": 2.35, "10": 2.35, "20": 2.11, "30": 1.43, "50": 1.43}


# ------------------------------------------------------------ per-run summary
# Victim MOLA rejections (isGood=0) from the MOLA_LO_DEBUG_ICP_QUALITY printf, read with
# leggi_log() of analisi_32r.py (lines split by stderr rejoined, i-th printf paired with the
# i-th logger pair). True speed at frame arrival (frame_log.csv, stamp_victim -> t_arr_wall):
# from the bag (/chassis/odom) if readable, otherwise from the nearest row of
# traces/w*_applied.csv (true_vx, true_vy) within 0.5 s.
V_MOTO = 0.05          # m/s: moving above this true speed
TOLL_TRACCIA_S = 0.5   # s, nearest trace row (analisi_32r.moto_vittima)
SENZA_POSA_S = 1.0     # pose loss: more than 1 s between two good ICPs
MAX_EPISODI = 25       # episodes listed per run in the text (all in the JSON)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def rep_della_run(cartella):
    """Last rep_NN of the run (as trova_run of analisi_pilota); None if missing."""
    reps = sorted(r for r in glob.glob(os.path.join(cartella, "campagna", "*", "*", "rep_[0-9][0-9]"))
                  if os.path.isdir(r))
    return reps[-1] if reps else None


def _printf_vittima(rep):
    """Victim printf records via analisi_32r.leggi_log; (records, info) or (None, error)."""
    log = os.path.join(rep, "logs", "mola_victim.log")
    if not os.path.exists(log):
        return None, "logs/mola_victim.log assente"
    qui = os.path.dirname(os.path.abspath(__file__))
    if qui not in sys.path:
        sys.path.insert(0, qui)
    try:
        from analisi_32r import leggi_log
    except Exception as e:   # numpy or the parser unavailable
        return None, f"parser di analisi_32r non importabile: {e}"
    r = leggi_log(log)
    # check on the raw text: a printf line following a WARN on the same line is lost when
    # testo_log strips stderr (typically 1-2 lines per run, isGood=1)
    with open(log, errors="replace") as f:
        grezzo = f.read()
    info = {"n_printf": r["n_printf"], "n_coppie_logger": r["n_coppie"],
            "non_accoppiati": r["non_accoppiati"], "dropping": r["dropping"],
            "isgood_grezzo": grezzo.count("isGood="), "isgood0_grezzo": grezzo.count("isGood=0")}
    return r["rec"], info


def _arrivi(cartella):
    """stamp_victim (0.1 s) -> t_arr_wall of the first frame with that stamp (velocita_bag.py)."""
    f = os.path.join(cartella, "frame_log.csv")
    if not os.path.exists(f):
        return None
    arr = {}
    for r in _csv(f):
        s, w = _num(r.get("stamp_victim")), _num(r.get("t_arr_wall"))
        if s is not None and w is not None:
            arr.setdefault(round(s, 1), w)
    return arr


def _tracce(rep):
    """Rows (wall, |true v|, window) of traces/w*_applied.csv and per-window intervals."""
    righe, inter = [], {}
    for f in sorted(glob.glob(os.path.join(rep, "traces", "w*_applied.csv"))):
        m = re.match(r"w(\d+)_applied\.csv$", os.path.basename(f))
        if not m:
            continue
        w = int(m.group(1))
        rr = []
        for r in _csv(f):
            t, vx, vy = _num(r.get("wall")), _num(r.get("true_vx")), _num(r.get("true_vy"))
            if t is not None and vx is not None and vy is not None:
                rr.append((t, (vx * vx + vy * vy) ** 0.5, w))
        if rr:
            rr.sort()
            inter[w] = (rr[0][0], rr[-1][0], rr[0][1], rr[-1][1])
            righe += rr
    righe.sort()
    return righe, inter


def _bag(cartella):
    """(times, |v|) of /chassis/odom from the run bag; (None, reason) if missing or unreadable."""
    d = os.path.join(cartella, "bag")
    if not os.path.isdir(d):
        return None, "bag assente"
    try:
        import rosbag2_py
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message
    except Exception as e:
        return None, f"bag non leggibile ({type(e).__name__})"
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=d, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    rd.set_filter(rosbag2_py.StorageFilter(topics=["/chassis/odom"]))
    T, V = [], []
    while rd.has_next():
        _, data, t = rd.read_next()
        m = deserialize_message(data, Odometry)
        T.append(t * 1e-9)
        V.append((m.twist.twist.linear.x ** 2 + m.twist.twist.linear.y ** 2) ** 0.5)
    return (T, V), f"bag, /chassis/odom ({len(T)} messaggi)"


def _v_bag(bag, w):
    """Max |v| of /chassis/odom around the arrival (window as in velocita_bag.py)."""
    import bisect
    T, V = bag
    for a, b in ((0.15, 0.05), (1.0, 1.0)):
        vs = V[bisect.bisect_left(T, w - a):bisect.bisect_right(T, w + b)]
        if vs:
            return max(vs)
    return None


def _dove(w, inter, pause):
    """Window of instant w: inside an applied window, in its pause, or between two."""
    if w is None:
        return "?"
    for k in sorted(inter):
        if inter[k][0] <= w <= inter[k][1]:
            return f"w{k}"
    for k, (a, b) in sorted(pause.items()):
        if a is not None and b is not None and a <= w <= b:
            return f"pausa dopo w{k}"
    ks = sorted(inter)
    if not ks:
        return "?"
    if w < inter[ks[0]][0]:
        return f"prima di w{ks[0]}"
    prec = [k for k in ks if inter[k][1] < w]
    succ = [k for k in ks if inter[k][0] > w]
    if prec and succ:
        return f"fra w{prec[-1]} e w{succ[0]}"
    return f"dopo w{prec[-1]}" if prec else "?"


def _moto(w, righe, inter, bag):
    """(category, v, source): measured in_moto / fermo, fermo_dedotto, non_misurato."""
    if w is None:
        return "non_misurato", None, "stamp senza frame nel registro"
    if bag is not None:
        v = _v_bag(bag, w)
        if v is not None:
            return ("in_moto" if v > V_MOTO else "fermo"), v, "bag"
    if righe:
        import bisect
        i = bisect.bisect_left(righe, (w,))
        cand = [righe[j] for j in (i - 1, i) if 0 <= j < len(righe)]
        t, v, _ = min(cand, key=lambda x: abs(x[0] - w))
        if abs(t - w) <= TOLL_TRACCIA_S:
            return ("in_moto" if v > V_MOTO else "fermo"), v, "tracce"
    # between windows: inferred still only if the previous ends and the next starts still
    ks = sorted(inter)
    prec = [k for k in ks if inter[k][1] < w]
    succ = [k for k in ks if inter[k][0] > w]
    if prec and succ and inter[prec[-1]][3] <= V_MOTO and inter[succ[0]][2] <= V_MOTO:
        return "fermo_dedotto", None, f"fra w{prec[-1]} e w{succ[0]}, ferme agli estremi"
    return "non_misurato", None, "nessuna riga delle tracce entro 0,5 s"


def ombra_run(rep):
    """Per-metre shadow/victim identity of a rep (src/optimization/shadow_check.py,
    identita_da_rep); reads shadow_check_scans.csv and history.json only."""
    radice = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if radice not in sys.path:
        sys.path.insert(0, radice)
    try:
        from src.optimization import shadow_check as shc
        return shc.identita_da_rep(rep)
    except Exception as e:   # repo modules not importable or files unreadable
        return {"esito": "non_misurata", "testo": f"OMBRA NON MISURATA ({type(e).__name__}: {e}): run non valida",
                "righe": [], "metri_divergenti": []}


def _ora(w):
    return time.strftime("%H:%M:%S", time.localtime(w)) if w is not None else "?"


def tempi_mola_run(rep, hist):
    """Victim MOLA time per run and per window, treated as an attack effect and not a chain
    failure: p50, p99 and max of c = 1000 * time_onLidar of the next line, and ICP restarts
    from the DEBUG log (window_record.tempi_mola_finestra), on end-of-run files. A window
    spans the scans from the stream reopening (open_stamp of victim_window, or segment start)
    up to and including the next reopening; the last one runs to the end."""
    radice = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if radice not in sys.path:
        sys.path.insert(0, radice)
    try:
        from src.optimization import window_record as wrec
    except Exception as e:   # numpy or repo modules unavailable
        return {"note": [f"window_record non importabile: {e}"]}
    tracce = wrec.leggi_csv(os.path.join(rep, "logs", "traces_victim.csv"))
    rip = wrec.ripartenze_icp(os.path.join(rep, "logs", "mola_victim.log"))
    out = {"run": wrec.tempi_mola_finestra(tracce, None, None, rip), "finestre": [], "note": []}
    if tracce is None:
        out["note"].append("logs/traces_victim.csv assente")
    if rip is None:
        out["note"].append("ripartenze non contate: log della vittima assente o senza righe DEBUG")
    con_vw = [h for h in hist if h.get("window") is not None and isinstance(h.get("victim_window"), dict)]
    for i, h in enumerate(con_vw):
        vw = h["victim_window"]
        a = vw.get("open_stamp")
        t0 = round(a + wrec.PERIODO_S, 1) if a is not None else (vw.get("drive_stamps") or [None])[0]
        t1 = con_vw[i + 1]["victim_window"].get("open_stamp") if i + 1 < len(con_vw) else None
        if t0 is not None:
            out["finestre"].append({"finestra": h["window"], **wrec.tempi_mola_finestra(tracce, t0, t1, rip)})
    return out


def prima_icp_finestre(rep, hist):
    """First ICP of each metre, victim and winning twin, with the prima_icp_debole flag
    (window_record.prima_icp): from the history if recorded, else from the victim log
    (open_stamp) and twin_icp_first_* of the winner. Flag only: no window is excluded."""
    radice = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if radice not in sys.path:
        sys.path.insert(0, radice)
    try:
        from src.optimization import window_record as wrec
    except Exception as e:
        return None, f"window_record non importabile: {e}"
    rows = None
    out = []
    for h in hist:
        if h.get("window") is None:
            continue
        vw = h.get("victim_window") if isinstance(h.get("victim_window"), dict) else {}
        pv = vw.get("prima_icp")
        if not isinstance(pv, dict) or pv.get("q") is None:
            if rows is None:
                rows = wrec.read_printf(os.path.join(rep, "logs", "mola_victim.log"))
            pv = wrec.prima_icp(rows, _num(vw.get("open_stamp")))
        pg = h.get("prima_icp_vincitore")
        if not isinstance(pg, dict):
            evs, vi = h.get("evaluations"), h.get("vincitore_idx")
            ev = evs[vi] if isinstance(evs, list) and isinstance(vi, int) and 0 <= vi < len(evs) else None
            pg = wrec.prima_icp_gemello(ev) if isinstance(ev, dict) else None
        out.append({"finestra": h.get("window"), "vittima": pv, "vincitore": pg})
    return out, None


def frazione_tolta(cartella, rep):
    """Fraction of points removed by dropout, per cloud and per metre, victim and twin, from
    the zone log (rep/struct_log.csv), read-only. With the multiplier on, uses the node
    columns n_in, n_tolti, frac_tolti; otherwise, for the victim n_hit_g5 / n_points from
    frame_log.csv (victim-stream rows with the same stamp), for the twin only the removed
    count (n_in not logged). None if the zone log is missing. per_metro: stream, metre,
    cloud count, median, min, max; the twin pools all candidates of the metre,
    per_genoma_gemello splits them by genome_id."""
    sl = os.path.join(rep, "struct_log.csv") if rep else ""
    if not sl or not os.path.exists(sl):
        return None
    righe = _csv(sl)
    colonne = bool(righe) and "frac_tolti" in righe[0]
    n_pts = {}
    if not colonne:
        fl = os.path.join(cartella, "frame_log.csv")
        if os.path.exists(fl):
            for r in _csv(fl):
                if str(r.get("victim_stream")).strip() in ("1", "True") and _num(r.get("stamp_in")) is not None:
                    n_pts[round(float(r["stamp_in"]), 3)] = _num(r.get("n_points"))
    nuvole = []
    for r in righe:
        fl_, metro = r.get("stream"), r.get("metro")
        if colonne:
            n_in, tolti, fr = _num(r.get("n_in")), _num(r.get("n_tolti")), _num(r.get("frac_tolti"))
        else:
            tolti = _num(r.get("n_hit_g5")) or 0.0
            n_in = n_pts.get(round(float(r["stamp"]), 3)) if fl_ == "vittima" and _num(r.get("stamp")) is not None else None
            fr = tolti / n_in if n_in else None
        nuvole.append({"flusso": fl_, "metro": metro, "genome_id": r.get("genome_id"), "frame_idx": r.get("frame_idx"),
                       "stamp": r.get("stamp"), "n_in": n_in, "n_tolti": tolti, "frac_tolti": fr})

    def agg(gruppo):
        v = sorted(x["frac_tolti"] for x in gruppo if x["frac_tolti"] is not None)
        return {"n_nuvole": len(gruppo), "n_con_frazione": len(v), "mediana": _mediana(v) if v else None,
                "min": v[0] if v else None, "max": v[-1] if v else None}
    per_metro, per_gen = [], []
    for chiave in sorted({(x["flusso"], x["metro"]) for x in nuvole}, key=lambda t: (str(t[0]), _num(t[1]) or -1)):
        per_metro.append({"flusso": chiave[0], "metro": chiave[1],
                          **agg([x for x in nuvole if (x["flusso"], x["metro"]) == chiave])})
    for chiave in sorted({(x["metro"], x["genome_id"]) for x in nuvole if x["flusso"] == "gemello"},
                         key=lambda t: (_num(t[0]) or -1, _num(t[1]) or -1)):
        per_gen.append({"metro": chiave[0], "genome_id": chiave[1],
                        **agg([x for x in nuvole if x["flusso"] == "gemello" and (x["metro"], x["genome_id"]) == chiave])})
    return {"fonte": ("struct_log.csv, colonne del moltiplicatore (n_in, n_tolti, frac_tolti)" if colonne else
                      "struct_log.csv n_hit_g5 / frame_log.csv n_points (solo vittima); gemello senza n_in"),
            "colonne_moltiplicatore": colonne, "per_metro": per_metro, "per_genoma_gemello": per_gen,
            "nuvole": nuvole}


def riepilogo_run(cartella, rep=None, usa_bag=True, ritardo_txt=None):
    """Per-run summary (dict): victim rejections and pose losses with time, window and motion;
    delay check; end-of-run reason; winner Chamfer per window; final damage. Read-only."""
    rep = rep or rep_della_run(cartella)
    out = {"cartella": cartella, "rep": rep, "note": []}
    if not rep:
        out["note"].append("nessuna rep_NN nella run")
        return out
    # history: end of run, Chamfer and damage per window, pauses
    hist = []
    hp = os.path.join(rep, "history.json")
    try:
        with open(hp) as f:
            hist = json.load(f)
    except (OSError, ValueError) as e:
        out["note"].append(f"history.json non letta: {e}")
    hist = [h for h in hist if isinstance(h, dict)] if isinstance(hist, list) else []
    st = storia(rep)
    out["run_end"] = [h["run_end"] for h in hist if h.get("run_end")]
    if st.get("vittima_persa"):
        out["vittima_persa"] = st["vittima_persa"]
    out["da_rifare"] = st.get("da_rifare")
    out["completata"] = os.path.exists(os.path.join(rep, "completata"))
    pause = {h.get("window"): (_num(h.get("pause_start_wall")), _num(h.get("pause_end_wall")))
             for h in hist if h.get("window") is not None}
    fin = []
    for h in hist:
        if h.get("window") is None:
            continue
        dm = h.get("danno_metro_vittima") if isinstance(h.get("danno_metro_vittima"), dict) else None
        fin.append({"finestra": h.get("window"), "chamfer_vincitore_cm": h.get("chamfer_vincitore_cm"),
                    "chamfer_applicato_cm": h.get("chamfer_applicato_cm"),
                    "chamfer_budget": h.get("chamfer_budget"),
                    "danno_metro_vittima_mean_cm": dm.get("mean_cm") if dm else None,
                    "danno_metro_vittima_end_cm": dm.get("end_cm") if dm else None,
                    "victim_lost": h.get("victim_lost"), "applied_status": h.get("applied_status"),
                    "danno_metro": "danno_metro_vittima" in h or "chamfer_vincitore_cm" in h})
    out["finestre"] = fin
    # --controllo-eta: per-window counts (history field controllo_eta), only if present
    ce = [{"finestra": h.get("window"), "tentativi": c.get("tentativi"), "rifatti": c.get("rifatti"),
           "guasto_eta_ripetuto": c.get("ripetuti"), "quota_rifatti": c.get("quota_rifatti"),
           "guasti_vittima": c.get("guasti_vittima"), "non_determinati": c.get("non_determinati")}
          for h in hist if h.get("window") is not None
          for c in [h.get("controllo_eta")] if isinstance(c, dict)]
    if ce:
        out["controllo_eta"] = ce
    fr = [fitness_riserva_metro(h) for h in hist if h.get("window") is not None and "evaluations" in h]
    if fr:
        out["fitness_riserva"] = fr
    ft = frazione_tolta(cartella, rep)
    if ft is not None and ft["colonne_moltiplicatore"]:
        # multiplier on: removed fraction per metre (per-cloud values stay in the zone log
        # and the frazione-tolta subcommand)
        out["frazione_tolta"] = {k: v for k, v in ft.items() if k != "nuvole"}
    out["prima_icp"], err = prima_icp_finestre(rep, hist)
    if err:
        out["note"].append(err)
    out["danno_run"] = next((h["danno_run"] for h in reversed(hist) if h.get("danno_run") is not None), None)
    # delay check: output of controllo_ritardo_run.py and code in esito_run.json
    rt = ritardo_txt or os.path.join(cartella, "controllo_ritardo.txt")
    out["ritardo"] = {"file": rt if os.path.exists(rt) else None, "esito": None, "codice": None}
    if os.path.exists(rt):
        with open(rt, errors="replace") as f:
            esiti = [ln.strip() for ln in f if ln.startswith("ESITO [4]")]
        out["ritardo"]["esito"] = esiti[-1] if esiti else "riga ESITO [4] assente"
    ej = os.path.join(cartella, "esito_run.json")
    if os.path.exists(ej):
        try:
            with open(ej) as f:
                e = json.load(f)
            out["esito_coda"] = f"{e.get('esito')}: {e.get('motivo', '')}"
            out["ritardo"]["codice"] = (e.get("fatti") or {}).get("ritardo_rc")
        except ValueError:
            out["note"].append("esito_run.json illeggibile")
    out["ombra"] = ombra_run(rep)
    try:
        out["tempi_mola"] = tempi_mola_run(rep, hist)
    except Exception as e:
        out["tempi_mola"] = {"note": [f"tempi di MOLA non calcolati: {type(e).__name__}: {e}"]}
    # victim rejections
    rec, info = _printf_vittima(rep)
    if rec is None:
        out["note"].append(f"scarti non letti: {info}")
        return out
    out["printf"] = info
    arr = _arrivi(cartella)
    if arr is None:
        out["note"].append("frame_log.csv assente: niente istanti di parete ne' velocita'")
        arr = {}
    righe, inter = _tracce(rep)
    bag, fonte_bag = (None, "bag non usato")
    if usa_bag:
        bag, fonte_bag = _bag(cartella)
    out["fonte_velocita"] = fonte_bag if bag is not None else f"tracce applicate ({fonte_bag})"
    conteggi = {"in_moto": 0, "fermo": 0, "fermo_dedotto": 0, "non_misurato": 0}
    for r in rec:
        r["wall"] = arr.get(r["stamp"])
        if r["isGood"] == 0:
            r["moto"], r["v"], r["fonte"] = _moto(r["wall"], righe, inter, bag)
            conteggi[r["moto"]] += 1
    out["scarti"] = {"totale": sum(conteggi.values()), **conteggi}
    # runs of consecutive isGood=0; time without pose between the good ICP before and after
    ep, cur = [], []
    for i, r in enumerate(rec):
        if r["isGood"] == 0:
            cur.append(i)
        elif cur:
            ep.append(cur)
            cur = []
    if cur:
        ep.append(cur)
    episodi = []
    for idx in ep:
        a, b = idx[0], idx[-1]
        prima = next((rec[j]["stamp"] for j in range(a - 1, -1, -1) if rec[j]["isGood"] == 1), None)
        dopo = next((rec[j]["stamp"] for j in range(b + 1, len(rec)) if rec[j]["isGood"] == 1), None)
        fine = dopo if dopo is not None else rec[-1]["stamp"]
        senza = round(fine - prima, 1) if prima is not None else None
        mot = {}
        for j in idx:
            mot[rec[j]["moto"]] = mot.get(rec[j]["moto"], 0) + 1
        w0 = rec[a]["wall"]
        episodi.append({
            "stamp_inizio": rec[a]["stamp"], "stamp_fine": rec[b]["stamp"], "n": len(idx),
            "ora": _ora(w0), "wall": w0, "finestra": _dove(w0, inter, pause), "moto": mot,
            "v_max": max((rec[j]["v"] for j in idx if rec[j]["v"] is not None), default=None),
            "senza_posa_s": senza, "recuperata": dopo is not None,
            "perdita_di_posa": dopo is None or (senza is not None and senza > SENZA_POSA_S)})
    out["episodi"] = episodi
    out["perdite_di_posa"] = [e for e in episodi if e["perdita_di_posa"]]
    return out


def testo_riepilogo_run(d, titolo=None):
    """Text block of a per-run summary."""
    L = [titolo or f"Run {d['cartella']}"]
    if not d.get("rep"):
        return "\n".join(L + [f"  {n}" for n in d["note"]]) + "\n"
    re_ = d.get("run_end") or []
    if re_:
        fr = re_[-1]
        dett = ", ".join(f"{k} {fr[k]}" for k in ("window", "status", "windows_driven", "windows_held",
                                                  "stamp_ultima_posa", "wall_iso") if fr.get(k) is not None)
        L.append(f"  fine run: {fr.get('reason')}" + (f" ({dett})" if dett else ""))
    else:
        L.append("  fine run: nessun run_end nella history")
    vp = d.get("vittima_persa")
    if vp:
        pv = vp.get("processo") or {}
        sr = vp.get("segni_rifiuto") or {}
        L.append(f"  vittima senza posa: {vp.get('esito_vittima')} [{vp.get('marcatore')}] (finestra "
                 f"{vp.get('finestra')}, {vp.get('status')}"
                 + (f", mola-cli rc {pv.get('rc')}" if pv.get("vivo") is False else "")
                 + (f", ICP rifiutate dopo l'ultima buona {sr.get('rifiutate_dopo_ultima_buona')}, goodness minima "
                    f"{sr.get('goodness_min_dopo_ultima_buona')} ({vp.get('fonte_segni_rifiuto')})" if sr else "")
                 + f", processo da {vp.get('fonte')})")
    if d.get("da_rifare"):
        L.append(f"  da_rifare.json: {json.dumps(d['da_rifare'], ensure_ascii=False)}")
    L.append(f"  completata: {'si' if d.get('completata') else 'no'}"
             + (f"; esito della coda: {d['esito_coda']}" if d.get("esito_coda") else ""))
    ft = d.get("frazione_tolta")
    if ft:
        L.append(f"  frazione tolta dal dropout per metro ({ft.get('fonte')}):")
        for x in ft.get("per_metro") or []:
            if x.get("mediana") is not None:
                L.append(f"    {x['flusso']} metro {x['metro']}: mediana {x['mediana']:.4f}, min {x['min']:.4f}, "
                         f"max {x['max']:.4f} ({x['n_con_frazione']} nuvole)")
    rt = d.get("ritardo") or {}
    if rt.get("esito") or rt.get("codice") is not None:
        L.append(f"  blocco [4]: {rt.get('esito') or ''}" + (f" (codice {rt['codice']})" if rt.get("codice") is not None else ""))
    else:
        L.append("  blocco [4]: non eseguito (controllo_ritardo.txt assente)")
    om = d.get("ombra")
    if isinstance(om, dict) and om.get("testo"):
        L.append(f"  {om['testo']}")
        L.extend(f"    {r}" for r in om.get("righe") or [])
    if "scarti" in d:
        s, p = d["scarti"], d["printf"]
        L.append(f"  scarti della vittima (isGood=0): {s['totale']} su {p['n_printf']} righe printf "
                 f"(non accoppiate al logger {p['non_accoppiati']}); in moto {s['in_moto']}, da fermo {s['fermo']}, "
                 f"da fermo dedotto {s['fermo_dedotto']}, velocita' non misurata {s['non_misurato']} "
                 f"[velocita': {d['fonte_velocita']}; in moto se > {V_MOTO} m/s]")
        if p.get("isgood_grezzo") is not None and (p["isgood_grezzo"] != p["n_printf"]
                                                    or p["isgood0_grezzo"] != s["totale"]):
            L.append(f"  attenzione: nel testo grezzo 'isGood=' {p['isgood_grezzo']} volte, 'isGood=0' "
                     f"{p['isgood0_grezzo']} (righe printf spezzate o perse dal parser)")
        pp = d["perdite_di_posa"]
        L.append(f"  perdite di posa (oltre {SENZA_POSA_S:g} s senza ICP buona o non recuperate): {len(pp)}")
        for e in pp:
            L.append(f"    ore {e['ora']}, stamp {e['stamp_inizio']}-{e['stamp_fine']}, {e['finestra']}: "
                     f"{e['n']} scarti, senza posa {e['senza_posa_s']} s"
                     f"{'' if e['recuperata'] else ' (non recuperata fino all ultima riga)'}, "
                     + ", ".join(f"{k} {v}" for k, v in sorted(e["moto"].items())))
        ep = d["episodi"]
        L.append(f"  episodi di scarti: {len(ep)}" + (f" (primi {MAX_EPISODI})" if len(ep) > MAX_EPISODI else ""))
        for e in ep[:MAX_EPISODI]:
            vm = f", v max {e['v_max']:.3f} m/s" if e["v_max"] is not None else ""
            L.append(f"    ore {e['ora']}, stamp {e['stamp_inizio']}-{e['stamp_fine']}, {e['finestra']}: "
                     f"{e['n']} scarti, " + ", ".join(f"{k} {v}" for k, v in sorted(e["moto"].items())) + vm
                     + (f", senza posa {e['senza_posa_s']} s" if e["senza_posa_s"] is not None else ""))
    fin = d.get("finestre") or []
    if fin and any(f["danno_metro"] for f in fin):
        def c(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "-"
        L.append("  per finestra: Chamfer del vincitore / applicata (cm), danno della vittima nel metro media / fine (cm)")
        for f in fin:
            L.append(f"    w{f['finestra']}: {c(f['chamfer_vincitore_cm'])} / {c(f['chamfer_applicato_cm'])}; "
                     f"{c(f['danno_metro_vittima_mean_cm'])} / {c(f['danno_metro_vittima_end_cm'])}"
                     + (f"; budget {f['chamfer_budget']}" if f.get("chamfer_budget") is not None else "")
                     + (f"; {f['applied_status']}" if f.get("applied_status") not in (None, "ok") else ""))
    elif fin:
        L.append(f"  Chamfer del vincitore per finestra: non registrata (run senza --danno-metro, {len(fin)} finestre)")
    pi = d.get("prima_icp") or []
    if pi:
        def q(x):
            if not isinstance(x, dict) or x.get("q") is None:
                return "-"
            return f"{x['q']:.3f}" + (" DEBOLE" if x.get("prima_icp_debole") else "")
        soglia = next((x.get("soglia_q") for p in pi for x in (p["vittima"], p["vincitore"])
                       if isinstance(x, dict) and x.get("soglia_q") is not None), None)
        segnati = [p["finestra"] for p in pi
                   if any(isinstance(x, dict) and x.get("prima_icp_debole") for x in (p["vittima"], p["vincitore"]))]
        L.append(f"  prima ICP del metro, q vittima / gemello vincitore (debole se <= {soglia}; segnata, "
                 f"non esclusa): finestre segnate {len(segnati)} su {len(pi)}"
                 + (f" ({', '.join(f'w{w}' for w in segnati)})" if segnati else ""))
        for p in pi:
            L.append(f"    w{p['finestra']}: {q(p['vittima'])} / {q(p['vincitore'])}")
    tm = d.get("tempi_mola") or {}
    if tm.get("run"):
        def t(x):
            if not x.get("n") and x.get("ripartenze") is None:
                return "nessuna scansione"
            r = (f"ripartenze dell'ICP {x['ripartenze']} su {x['n_scansioni_debug']} scansioni del log "
                 f"(scansioni con ripartenze {x['scansioni_con_ripartenze']}, massimo {x['ripartenze_max']}, "
                 f"sulle scansioni con c {x['ripartenze_su_c']})" if x.get("ripartenze") is not None
                 else "ripartenze non contate")
            c = (f"n {x['n']} p50 {x['p50_ms']} p99 {x['p99_ms']} max {x['max_ms']} ms"
                 if x.get("n") else "n 0")
            return f"{c}; {r}"
        L.append("  tempo di MOLA della vittima (c = 1000 * time_onLidar della riga successiva; "
                 "risultato dell'attacco, non guasto della catena):")
        L.append(f"    run: {t(tm['run'])}")
        for f in tm.get("finestre", []):
            L.append(f"    w{f['finestra']} [{f['t0']}, {f['t1']}]: {t(f)}")
    L += [f"  nota sui tempi di MOLA: {n}" for n in tm.get("note", [])]
    if d.get("controllo_eta"):
        L.append("  controllo dell'eta' della nuvola (rollout del gemello per finestra):")
        for c in d["controllo_eta"]:
            L.append(f"    w{c['finestra']}: tentativi {c['tentativi']}, rifatti {c['rifatti']}, "
                     f"guasto_eta_ripetuto {c['guasto_eta_ripetuto']}, quota rifatti {c['quota_rifatti']}"
                     f"; guasti della vittima {c['guasti_vittima']}, non determinati {c['non_determinati']}")
    if d.get("fitness_riserva"):
        L += testo_fitness_riserva(d["fitness_riserva"])
    dr = d.get("danno_run")
    if dr:
        L.append(f"  danno a fine run: fine {dr.get('victim_end_cm')} cm (w{dr.get('victim_end_window')}), media "
                 f"{dr.get('victim_mean_cm')} cm, massimo {dr.get('victim_max_cm')} cm su {dr.get('n_windows')} metri; "
                 f"vincitori nella ricerca, media {dr.get('winner_mean_cm')} cm")
    else:
        L.append("  danno a fine run: non registrato (danno_run assente: run senza --danno-metro o senza metri con danno)")
    L += [f"  nota: {n}" for n in d.get("note", [])]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------ daily summary
def riepilogo(st, runs=None, oggi=None, per_run=True, disco=None):
    oggi = oggi or time.strftime("%Y-%m-%d")
    righe = [f"Riepilogo della coda della serie B - {oggi} (scritto {time.strftime('%Y-%m-%d %H:%M:%S')})", ""]
    voci = list(st["runs"].values())
    if runs:
        ordine = {chiave(c, s): i for i, (s, c) in enumerate(runs)}
        voci.sort(key=lambda v: ordine.get(chiave(v["condizione"], v["seme"]), 10**6))
    fatte, in_corso, guasti, altre, crash = [], [], [], [], []
    ore_tot = ore_oggi = 0.0
    for v in voci:
        s = stato_run(v)
        nome = f"s{v['seme']} {v['condizione']}"
        for t in v["tentativi"]:
            o = t.get("ore") or 0.0
            ore_tot += o
            if (t.get("fine") or "").startswith(oggi) or (t.get("inizio") or "").startswith(oggi):
                ore_oggi += o
            if t.get("esito") in ESITI_RIPETIBILI or t.get("esito") in ESITI_FERMANO_CODA:
                guasti.append(f"  {nome}: {t.get('esito')} - {t.get('motivo', '')} ({t.get('cartella')}, "
                              f"fine {t.get('fine')})")
        ult = v["tentativi"][-1] if v["tentativi"] else {}
        if s == "fatta":
            extra = ult.get("dettaglio", {}).get("ritardo", "")
            fatte.append(f"  {nome}: {ult.get('motivo', '')}{', ritardo ' + extra if extra else ''}, "
                         f"{ult.get('ore', '?')} h, {ult.get('cartella')}")
        elif s == "in_corso":
            in_corso.append(f"  {nome}: dal {ult.get('inizio')}, {ult.get('cartella')}")
        elif s == "crash_da_attacco":
            crash.append(f"  {nome}: {ult.get('motivo', '')}, {ult.get('ore', '?')} h")
        else:
            altre.append(f"  {nome}: {s} ({len(v['tentativi'])} tentativi, guasti {conta_guasti(v)})")
    righe += [f"Fatte ({len(fatte)}):"] + (fatte or ["  nessuna"]) + [""]
    righe += [f"In corso ({len(in_corso)}):"] + (in_corso or ["  nessuna"]) + [""]
    righe += [f"Guasti e interruzioni ({len(guasti)} tentativi):"] + (guasti or ["  nessuno"]) + [""]
    if crash:
        # crash rule: only if there is at least one
        righe += [f"Crash da attacco ({len(crash)}):"] + crash + [""]
    righe += [f"Da fare o chiuse senza dato ({len(altre)}):"] + (altre or ["  nessuna"]) + [""]
    righe += [f"Ore di banco: oggi {ore_oggi:.2f} h, in totale {ore_tot:.2f} h (tentativi chiusi)"]
    if runs:
        rest = [r for r in runs if da_fare(st["runs"].get(chiave(r[1], r[0]), {"tentativi": []}))]
        righe.append(f"Run della selezione ancora da fare: {len(rest)} su {len(runs)}")
    if disco is not None:
        righe += righe_disco(disco, rest if runs else None)
    if per_run:
        # details of attempts closed today: the run's riepilogo_run.json (written by the
        # queue at close), otherwise computed here read-only
        blocchi = []
        for v in voci:
            for t in v["tentativi"]:
                c = t.get("cartella")
                if not (t.get("fine") or "").startswith(oggi) or not c or not os.path.isdir(c):
                    continue
                titolo = f"s{v['seme']} {v['condizione']} - {t.get('esito')} ({c})"
                j = os.path.join(c, "riepilogo_run.json")
                try:
                    with open(j) as f:
                        d = json.load(f)
                except (OSError, ValueError):
                    d = riepilogo_run(c, usa_bag=False)
                blocchi.append(testo_riepilogo_run(d, titolo))
        righe += ["", f"Dettaglio delle run chiuse oggi ({len(blocchi)} tentativi):"] + (blocchi or ["  nessuno\n"])
    return "\n".join(righe) + "\n"


# ------------------------------------------------------------ command line
def _runs(args_runs):
    out = []
    for r in args_runs:
        seme, cond = r.split(":", 1)
        out.append((seme, cond))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("condizioni")
    p = sp.add_parser("seleziona"); p.add_argument("--semi", default=" ".join(SEMI_D5), help="semi separati da spazi")
    p.add_argument("voci", nargs="+")
    p = sp.add_parser("stima")
    p.add_argument("--pilota"); p.add_argument("--ore"); p.add_argument("--baseline-dir")
    p.add_argument("--scenario", default="b"); p.add_argument("--finestre", type=int, default=5)
    p.add_argument("--extra", type=int, default=3); p.add_argument("--pop", type=int, default=32)
    p.add_argument("--gen", type=int, default=16); p.add_argument("--topk", type=int, default=2)
    p.add_argument("--k-max", type=int, default=64); p.add_argument("--json", action="store_true")
    p.add_argument("runs", nargs="*", help="seme:condizione")
    p = sp.add_parser("tempo-max", parents=[]); p.add_argument("--ore"); p.add_argument("--scenario", default="b")
    p.add_argument("--finestre", type=int, default=5); p.add_argument("--extra", type=int, default=3)
    p.add_argument("--fattore", type=float, default=1.5); p.add_argument("--minimo-s", type=int, default=3600)
    p.add_argument("cond")
    p = sp.add_parser("cartella"); p.add_argument("--radice", required=True); p.add_argument("cond"); p.add_argument("seme")
    p = sp.add_parser("prossima"); p.add_argument("stato"); p.add_argument("runs", nargs="*")
    p = sp.add_parser("inizia"); p.add_argument("stato"); p.add_argument("cond"); p.add_argument("seme"); p.add_argument("cartella")
    p = sp.add_parser("chiudi"); p.add_argument("stato"); p.add_argument("cond"); p.add_argument("seme"); p.add_argument("esito_json")
    p = sp.add_parser("ripara"); p.add_argument("stato")
    p = sp.add_parser("classifica"); p.add_argument("fatti_json"); p.add_argument("--uscita", required=True)
    p.add_argument("--regola-crash", action="store_true",
                   help="studio dei limiti: vittima senza posa con mola-cli terminato = crash_processo")
    p = sp.add_parser("storia"); p.add_argument("rep"); p.add_argument("--guasto-re", default="guasto_catena")
    p = sp.add_parser("ombra"); p.add_argument("rep")
    p = sp.add_parser("fitness-riserva"); p.add_argument("rep"); p.add_argument("--json", help="uscita JSON su file")
    p = sp.add_parser("mostra"); p.add_argument("stato"); p.add_argument("runs", nargs="*")
    p = sp.add_parser("riepilogo"); p.add_argument("stato"); p.add_argument("cartella"); p.add_argument("runs", nargs="*")
    p.add_argument("--stampa", action="store_true")
    p.add_argument("--disco", help="percorso del disco della repo: GB liberi e stima per le run rimaste")
    p.add_argument("--pilota"); p.add_argument("--baseline-dir")
    p.add_argument("--finestre", type=int, default=5); p.add_argument("--extra", type=int, default=3)
    p.add_argument("--pop", type=int, default=32); p.add_argument("--gen", type=int, default=16)
    p.add_argument("--topk", type=int, default=2); p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--margine-gb", type=float, help="margine della coda (DISCO_MIN_GB)")
    p = sp.add_parser("frazione-tolta"); p.add_argument("cartella"); p.add_argument("--rep")
    p.add_argument("--csv", help="nuvole su file CSV (flusso, metro, genome_id, frame_idx, stamp, n_in, n_tolti, frac_tolti)")
    p = sp.add_parser("riepilogo-run"); p.add_argument("cartella"); p.add_argument("--rep")
    p.add_argument("--senza-bag", action="store_true", help="velocita' vera solo dalle tracce")
    p.add_argument("--ritardo", help="uscita di controllo_ritardo_run.py (default <cartella>/controllo_ritardo.txt)")
    p.add_argument("--json", help="scrive anche il riepilogo in JSON")
    a = ap.parse_args(argv)

    if a.cmd == "condizioni":
        for c in CONDIZIONI:
            print(f"{c['id']}\t{c['braccio']}\t{c['budget'] or '-'}\t{c['tipo']}\t{c['descrizione']}")
    elif a.cmd == "seleziona":
        try:
            ids = espandi(a.voci)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        semi = a.semi.split()
        for s in semi:
            if not re.fullmatch(r"[0-9]+", s):
                print(f"seme non intero: {s}", file=sys.stderr)
                return 2
        for seme, cond in ordina(semi, ids):
            print(f"{seme}:{cond}")
    elif a.cmd == "stima":
        mb_r, n_pil = disco_pilota(a.pilota) if a.pilota else (None, 0)
        mb_f = mb_cartella(a.baseline_dir)
        ore = ore_t6(a.ore, a.scenario)
        tot = {"mb": [0.0, 0.0], "ore": [0.0, 0.0]}
        righe = []
        for seme, cond in _runs(a.runs):
            r = stima_run(cond, a.finestre, a.extra, a.pop, a.gen, a.topk, a.k_max, mb_r, mb_f, ore)
            r["seme"] = seme
            righe.append(r)
            for i in (0, 1):
                tot["mb"][i] += r["mb"][i]
                tot["ore"][i] += r["ore"][i] or 0.0
        if a.json:
            print(json.dumps({"mb_per_rollout": mb_r, "run_pilota": n_pil, "mb_fisso": mb_f,
                              "runs": righe, "totale": tot}, indent=1))
            return 0
        print(f"fonte del disco: {a.pilota} ({n_pil} run lette): "
              + (f"{mb_r:.2f} MB per rollout del gemello (du del seme / stati risalvati)" if mb_r else "non misurato")
              + (f"; parte fissa {mb_f:.1f} MB ({a.baseline_dir})" if mb_f is not None else "; parte fissa non misurata"))
        print(f"fonte delle ore: {a.ore} (scenario {a.scenario}); finestre {a.finestre}, al piu' {a.finestre + a.extra}")
        print(f"{'seme':>6} {'condizione':<20} {'GB stima':>9} {'GB max':>8} {'ore stima':>9} {'ore max':>8}")
        for r in righe:
            o0, o1 = r["ore"]
            print(f"{r['seme']:>6} {r['condizione']:<20} {r['mb'][0]/1000:9.2f} {r['mb'][1]/1000:8.2f} "
                  f"{(o0 if o0 is not None else float('nan')):9.2f} {(o1 if o1 is not None else float('nan')):8.2f}")
        print(f"{'':>6} {'TOTALE':<20} {tot['mb'][0]/1000:9.2f} {tot['mb'][1]/1000:8.2f} "
              f"{tot['ore'][0]:9.2f} {tot['ore'][1]:8.2f}")
        # conditions missing from T6: source of the hours estimate
        for cond in dict.fromkeys(r["condizione"] for r in righe):
            c = PER_ID[cond]
            if c.get("fonte_ore") and not ore.get(c["t6"]) and ore.get(c.get("t6_come")):
                print(f"ore di {cond}: {c['fonte_ore']}")
    elif a.cmd == "tempo-max":
        r = stima_run(a.cond, a.finestre, a.extra, 32, 16, 2, 64, None, None, ore_t6(a.ore, a.scenario))
        print(tempo_max_s(r["ore"][1], a.fattore, a.minimo_s))
    elif a.cmd == "cartella":
        print(cartella_libera(a.radice, a.cond, a.seme))
    elif a.cmd == "prossima":
        p = prossima(leggi(a.stato), _runs(a.runs))
        if p:
            print(f"{p[0]}:{p[1]}")
    elif a.cmd == "inizia":
        inizia(a.stato, a.cond, a.seme, a.cartella)
    elif a.cmd == "chiudi":
        with open(a.esito_json) as f:
            e = json.load(f)
        print(chiudi(a.stato, a.cond, a.seme, e))
    elif a.cmd == "ripara":
        for r in ripara(a.stato):
            print(r)
    elif a.cmd == "classifica":
        with open(a.fatti_json) as f:
            fatti = json.load(f)
        e = classifica(fatti, regola_crash=a.regola_crash)
        e["dettaglio"] = {k: v for k, v in e.items() if k not in ("esito", "motivo")}
        e["cartella"] = fatti.get("cartella")
        e["fatti"] = fatti
        tmp = a.uscita + ".tmp"
        with open(tmp, "w") as f:
            json.dump(e, f, indent=1, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, a.uscita)
        print(f"{e['esito']}: {e['motivo']}")
    elif a.cmd == "storia":
        print(json.dumps(storia(a.rep, a.guasto_re)))
    elif a.cmd == "fitness-riserva":
        metri = fitness_riserva_rep(a.rep)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(metri, f, indent=1, ensure_ascii=False)
        print("\n".join(testo_fitness_riserva(metri)))
    elif a.cmd == "ombra":
        print(json.dumps(ombra_run(a.rep), ensure_ascii=False))
    elif a.cmd == "mostra":
        st = leggi(a.stato)
        for seme, cond in _runs(a.runs):
            v = st["runs"].get(chiave(cond, seme))
            if v is None:
                print(f"s{seme} {cond}: da_fare")
            else:
                ult = v["tentativi"][-1] if v["tentativi"] else {}
                print(f"s{seme} {cond}: {stato_run(v)} ({len(v['tentativi'])} tentativi, guasti {conta_guasti(v)}"
                      + (f", ultimo: {ult.get('esito')} {ult.get('motivo', '')}" if ult else "") + ")")
    elif a.cmd == "riepilogo":
        disco = None
        if a.disco:
            mb_r, n_pil = disco_pilota(a.pilota) if a.pilota else (None, 0)
            disco = {"percorso": a.disco, "gb_liberi": gb_liberi(a.disco), "ora": time.strftime("%H:%M:%S"),
                     "margine_gb": a.margine_gb,
                     "fonte": f"pilota {a.pilota}, {n_pil} run lette" if a.pilota else "pilota non dato",
                     "stima": {"finestre": a.finestre, "extra": a.extra, "pop": a.pop, "gen": a.gen,
                               "topk": a.topk, "k_max": a.k_max, "mb_rollout": mb_r,
                               "mb_fisso": mb_cartella(a.baseline_dir)}}
        testo = riepilogo(leggi(a.stato), _runs(a.runs) or None, disco=disco)
        if a.stampa:
            print(testo, end="")
        else:
            os.makedirs(a.cartella, exist_ok=True)
            dest = os.path.join(a.cartella, f"riepilogo_{time.strftime('%Y-%m-%d')}.txt")
            tmp = dest + f".tmp_{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(testo)
            os.replace(tmp, dest)
            print(dest)
    elif a.cmd == "frazione-tolta":
        ft = frazione_tolta(a.cartella, a.rep or rep_della_run(a.cartella))
        if ft is None:
            print("registro delle zone assente")
            return 1
        if a.csv:
            campi = ("flusso", "metro", "genome_id", "frame_idx", "stamp", "n_in", "n_tolti", "frac_tolti")
            with open(a.csv, "x", newline="") as f:
                w = csv.DictWriter(f, fieldnames=campi)
                w.writeheader()
                w.writerows(ft["nuvole"])
        print(json.dumps({k: v for k, v in ft.items() if k != "nuvole"}, indent=1, ensure_ascii=False))
    elif a.cmd == "riepilogo-run":
        d = riepilogo_run(a.cartella, a.rep, usa_bag=not a.senza_bag, ritardo_txt=a.ritardo)
        print(testo_riepilogo_run(d), end="")
        if a.json:
            tmp = a.json + f".tmp_{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=1, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
