#!/usr/bin/env python3
"""
Actual MOLA scan rejections on the victim: reading and cross-checking the two per-scan sources.

Sources (both written by mola-cli in the logs/ folder of a run):

  mola_<tag>.log      with MOLA_LO_DEBUG_ICP_QUALITY=1 (LidarOdometry_ProcessScan.cpp,
                      lines 1291-1302) every scan that runs ICP prints a line
                      "[LidarOdometry] pathStep=N timestamp=... goodness=... isGood=0|1 ...":
                      isGood=0 is a real rejection (ICP refused, pose not published).
  traces_<tag>.csv    with MOLA_SAVE_DEBUG_TRACES (--trace): one row per processed scan,
                      with a fresh icp_quality also for rejected scans; rejected if
                      icp_quality < min_icp_goodness. The first row
                      (current_relative_timestamp = 0) is the first scan, without ICP, and is
                      excluded. robot_x is the pose of the previous ICP (one-scan lag).

Known limits, to be stated with every output: on SIGKILL the last CSV block (<= 8 KiB, last
row often truncated) and the last printf lines (4 KiB stdout buffer, about 21 lines) are
lost; runs without MOLA_LO_DEBUG_ICP_QUALITY have the CSV only.

Usage:
    python3 src/analysis/scarti_mola.py <run or rep_01> [--tag victim] [--passo 1.0] [--csv OUT]
As a module:
    from scarti_mola import analizza_run
    records, riepilogo, bins = analizza_run(run_dir, soglia)
"""

import argparse
import bisect
import csv
import json
import math
import os
import re
from datetime import datetime, timezone

PERIODO_SCAN_S = 0.1          # Hesai XT-32 at 10 Hz
TOLLERANZA_Q = 0.0005         # printf has goodness to 3 decimals, the CSV to 4
RIGHE_PRINTF_PERSE_MAX = 21   # 4 KiB stdout buffer / ~190 bytes per line

RX_PRINTF = re.compile(
    r"\[LidarOdometry\] pathStep=(\d+) timestamp=(\S+) goodness=([\d.]+) minRequired=([\d.]+) "
    r"iters=(\d+) termReason=(\S+) isGood=([01]) adapt_thres_sigma=([\d.]+) consecutive_bad=(-?\d+)")
RX_TF_OLD = re.compile(r"TF_OLD_DATA .*? at time ([\d.]+)")
# Whole rcutils stderr line, plus the second line of the tf2 message. In the mola-cli log
# stdout and stderr share one file and the block-buffered stdout can be cut mid-line by
# TF_OLD_DATA warnings or by the signal_handler line: removing these lines rebuilds the
# printf line. A stderr line that carries a whole printf record (the warning itself cut
# by stdout) is not removed, otherwise the record would be lost.
_SENZA_PRINTF = r"(?:(?!\[LidarOdometry\] pathStep=)[^\n])*\n"
RX_STDERR = re.compile(r"\[(?:DEBUG|INFO|WARN|ERROR|FATAL)\] \[\d+\.\d+\] \[[^\]\n]*\]: " + _SENZA_PRINTF
                       + r"(?:Possible reasons are listed at " + _SENZA_PRINTF + ")?")
RX_YAML_SOGLIA = re.compile(r"^\s*min_icp_goodness:\s*\"?(\$\{([A-Z0-9_]+)\|([^}]*)\}|[0-9.]+)\"?")
_EPOCA = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _stamp(v):
    """Stamp rounded to 0.1 s: the join key between printf and CSV."""
    return round(float(v), 1)


def _stamp_da_data(s):
    """'1970/01/01,00:00:12.400000' (mrpt dateTimeToString) -> seconds since epoch."""
    d = datetime.strptime(s, "%Y/%m/%d,%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return (d - _EPOCA).total_seconds()


# ----------------------------------------------------------------------------- reading
def testo_log(log_path):
    """(text, riga_nel_file) of the log without the ROS stderr lines.

    text is RX_STDERR.sub("", file content), with stdout lines rebuilt; riga_nel_file(p)
    gives the 1-based file line holding character p of text. Without stderr lines cutting
    stdout the numbers match enumerate(f, 1).
    """
    with open(log_path, encoding="utf-8", errors="replace") as f:
        grezzo = f.read()
    # kept spans: start in the clean text and matching start in the file
    inizi_pulito, inizi_file, parti = [0], [0], []
    fine, lun = 0, 0
    for m in RX_STDERR.finditer(grezzo):
        parti.append(grezzo[fine:m.start()])
        lun += m.start() - fine
        inizi_pulito.append(lun)
        inizi_file.append(m.end())
        fine = m.end()
    parti.append(grezzo[fine:])
    a_capo = [m.start() for m in re.finditer("\n", grezzo)]

    def riga_nel_file(p):
        k = bisect.bisect_right(inizi_pulito, p) - 1
        return bisect.bisect_left(a_capo, inizi_file[k] + (p - inizi_pulito[k])) + 1

    return "".join(parti), riga_nel_file


def leggi_printf(log_path):
    """One record per ICP printf line (accepted or rejected), in log order.

    ROS stderr lines are removed first (testo_log), so a record split by a TF_OLD_DATA
    warning or by the signal_handler line is read whole; riga is the file line where the
    record starts.
    """
    testo, riga_nel_file = testo_log(log_path)
    out, p = [], 0
    for riga in testo.split("\n"):
        m = RX_PRINTF.search(riga)
        if m:
            out.append(dict(
                stamp_s=_stamp(_stamp_da_data(m.group(2))), goodness=float(m.group(3)),
                min_required=float(m.group(4)), iters=int(m.group(5)),
                term_reason=m.group(6).replace("IterTermReason::", ""), is_good=int(m.group(7)),
                sigma=float(m.group(8)), consecutive_bad=int(m.group(9)), pathstep=int(m.group(1)),
                riga=riga_nel_file(p + m.start())))
        p += len(riga) + 1
    return out


def ultimo_stamp_tf_old_data(log_path):
    """Last stamp quoted by a TF_OLD_DATA warning in the log (None if absent)."""
    ultimo = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for riga in f:
            m = RX_TF_OLD.search(riga)
            if m:
                ultimo = float(m.group(1))
    return ultimo


def _righe_csv(csv_path):
    """Header, complete rows (dicts) and number of truncated rows (tail lost on SIGKILL)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = [h for h in next(rd) if h != ""]
        righe, tronche = [], 0
        for i, r in enumerate(rd, 2):
            if len(r) > len(hdr):
                r = r[:len(hdr)] if r[len(hdr):] == [""] * (len(r) - len(hdr)) else r
            if len(r) < len(hdr):
                tronche += 1
                continue
            d = dict(zip(hdr, r[:len(hdr)]))
            d["riga"] = i
            righe.append(d)
    return hdr, righe, tronche


def leggi_csv_mola(csv_path, soglia):
    """One record per processed scan; the first row (relative_timestamp = 0) is excluded."""
    _, righe, _ = _righe_csv(csv_path)
    out = []
    for r in righe:
        if float(r.get("current_relative_timestamp", "0")) == 0.0:
            continue
        q = float(r["icp_quality"])
        out.append(dict(
            stamp_s=_stamp(r["timestamp"]), q=q, iters=int(float(r["icp_iterations"])),
            robot_x=float(r["robot_x"]), robot_y=float(r["robot_y"]),
            sigma=float(r["ADAPTIVE_THRESHOLD_SIGMA"]), time_onLidar=float(r["time_onLidar"]),
            scartato=bool(q < soglia), riga=r["riga"]))
    return out


def soglia_min_icp_goodness(config_yaml, env=None):
    """(value, source) of min_icp_goodness: ${VAR|default} resolved from env, else the default."""
    env = env or {}
    with open(config_yaml, encoding="utf-8") as f:
        for i, riga in enumerate(f, 1):
            m = RX_YAML_SOGLIA.match(riga)
            if not m:
                continue
            if m.group(2):
                var, default = m.group(2), m.group(3).strip()
                if var in env:
                    return float(env[var]), f"{config_yaml}:{i} ({var}={env[var]} dall'ambiente)"
                return float(default), f"{config_yaml}:{i} (default di ${{{var}|{default}}})"
            return float(m.group(1)), f"{config_yaml}:{i}"
    raise ValueError(f"min_icp_goodness non trovato in {config_yaml}")


# ----------------------------------------------------------------------------- join
def unisci(printf, csv_rec, soglia=None):
    """1:1 join on the stamp (0.1 s) between printf and CSV rows, in stamp order.

    Each record has fonte 'printf+csv', 'csv' or 'printf'. Where both exist q and iters must
    match (q within TOLLERANZA_Q), otherwise incoerente = True. The primary rejection flag is
    printf isGood=0 when present, otherwise q < soglia from the CSV.
    """
    per_stamp = {}
    for p in printf:
        per_stamp.setdefault(p["stamp_s"], []).append(p)
    out, usati = [], set()
    for c in csv_rec:
        cand = [p for p in per_stamp.get(c["stamp_s"], []) if id(p) not in usati]
        p = cand[0] if cand else None
        r = dict(c)
        r["scartato_csv"] = c["scartato"]
        if p is None:
            r.update(fonte="csv", is_good=None, goodness=float("nan"), term_reason="",
                     consecutive_bad=None, pathstep=None, incoerente=False, nota="")
            r["scartato"] = c["scartato"]
        else:
            usati.add(id(p))
            dq = abs(p["goodness"] - c["q"])
            incoerente = dq > TOLLERANZA_Q or p["iters"] != c["iters"]
            r.update(fonte="printf+csv", is_good=p["is_good"], goodness=p["goodness"],
                     term_reason=p["term_reason"], consecutive_bad=p["consecutive_bad"],
                     pathstep=p["pathstep"], incoerente=incoerente,
                     nota=(f"q {c['q']:.4f} vs goodness {p['goodness']:.3f}, iters {c['iters']} vs "
                           f"{p['iters']}") if incoerente else "")
            r["scartato"] = (p["is_good"] == 0)
        out.append(r)
    for p in printf:
        if id(p) in usati:
            continue
        out.append(dict(
            stamp_s=p["stamp_s"], q=p["goodness"], iters=p["iters"], robot_x=float("nan"),
            robot_y=float("nan"), sigma=p["sigma"], time_onLidar=float("nan"),
            scartato_csv=(p["goodness"] < soglia) if soglia is not None else None,
            fonte="printf", is_good=p["is_good"], goodness=p["goodness"],
            term_reason=p["term_reason"], consecutive_bad=p["consecutive_bad"],
            pathstep=p["pathstep"], incoerente=False, nota="solo printf (riga CSV assente)",
            riga=p["riga"], scartato=(p["is_good"] == 0)))
    out.sort(key=lambda r: (r["stamp_s"], 0 if r["fonte"] != "printf" else 1))
    return out


# ----------------------------------------------------------------------------- counts
def scarti_per_metro(records, passo_m=1.0):
    """Rejections per robot_x bin (pose of the previous ICP: one-scan lag).

    Per bin: processed scans (rows with robot_x), scans expected at 10 Hz from min to max
    stamp in the bin, rejections (primary flag), rejections by the CSV (q < soglia) and
    min q. Records without robot_x (printf only) are left out.
    """
    con_x = [r for r in records if math.isfinite(r["robot_x"])]
    if not con_x:
        return []
    x_max = max(r["robot_x"] for r in con_x)
    x_min = min(0.0, min(r["robot_x"] for r in con_x))
    nb = max(1, int(math.floor((x_max - x_min) / passo_m)) + 1)
    bins = [dict(bin_inizio_m=x_min + b * passo_m, bin_fine_m=x_min + (b + 1) * passo_m,
                 n_scan=0, n_attesi_10hz=0, n_scartati=0, n_scartati_csv=0, q_min=float("nan"),
                 stamp_min_s=float("nan"), stamp_max_s=float("nan")) for b in range(nb)]
    for r in con_x:
        b = bins[min(int((r["robot_x"] - x_min) // passo_m), nb - 1)]
        b["n_scan"] += 1
        b["n_scartati"] += int(bool(r["scartato"]))
        b["n_scartati_csv"] += int(bool(r.get("scartato_csv")))
        b["q_min"] = r["q"] if math.isnan(b["q_min"]) else min(b["q_min"], r["q"])
        b["stamp_min_s"] = (r["stamp_s"] if math.isnan(b["stamp_min_s"])
                            else min(b["stamp_min_s"], r["stamp_s"]))
        b["stamp_max_s"] = (r["stamp_s"] if math.isnan(b["stamp_max_s"])
                            else max(b["stamp_max_s"], r["stamp_s"]))
    for b in bins:
        if b["n_scan"]:
            b["n_attesi_10hz"] = int(round((b["stamp_max_s"] - b["stamp_min_s"]) / PERIODO_SCAN_S)) + 1
    return bins


def buchi(records, periodo_s=PERIODO_SCAN_S):
    """Gaps between consecutive stamps above 1.5 periods: (stamp_before, stamp_after, lost)."""
    stamps = sorted({r["stamp_s"] for r in records})
    out = []
    for a, b in zip(stamps, stamps[1:]):
        if b - a > 1.5 * periodo_s:
            out.append((a, b, int(round((b - a) / periodo_s)) - 1))
    return out


def riepilogo(records, printf, csv_rec, n_righe_tronche=0, stamp_tf_old=None, soglia=None):
    """Run summary: scans, rejections, gaps and how much of the tail is missing."""
    csv_st = [r["stamp_s"] for r in csv_rec]
    pf_st = [p["stamp_s"] for p in printf]
    tutti = sorted({r["stamp_s"] for r in records})
    n_attesi = int(round((tutti[-1] - tutti[0]) / PERIODO_SCAN_S)) + 1 if tutti else 0
    bh = buchi(records)
    fonte = "printf (isGood=0)" if printf else "csv (q < soglia, senza printf)"
    nan = float("nan")
    ultimo_csv = max(csv_st) if csv_st else nan
    ultimo_pf = max(pf_st) if pf_st else nan
    noti = [v for v in (ultimo_csv, ultimo_pf, stamp_tf_old if stamp_tf_old is not None else nan)
            if not math.isnan(v)]
    riferimento = max(noti) if noti else nan
    return dict(
        fonte_principale=fonte, soglia_q=soglia,
        n_scan_processati=len(csv_rec), n_righe_printf=len(printf), n_record=len(records),
        n_scartati=sum(1 for r in records if r["scartato"]),
        n_scartati_csv=sum(1 for r in records if r.get("scartato_csv")),
        n_scartati_printf=sum(1 for p in printf if p["is_good"] == 0),
        n_incoerenti=sum(1 for r in records if r["incoerente"]),
        n_solo_csv=sum(1 for r in records if r["fonte"] == "csv"),
        n_solo_printf=sum(1 for r in records if r["fonte"] == "printf"),
        primo_stamp_s=tutti[0] if tutti else nan, ultimo_stamp_s=tutti[-1] if tutti else nan,
        n_attesi_10hz=n_attesi, n_buchi=len(bh), scan_persi_nei_buchi=sum(b[2] for b in bh),
        buchi=bh, n_righe_csv_tronche=n_righe_tronche,
        ultimo_stamp_csv_s=ultimo_csv, ultimo_stamp_printf_s=ultimo_pf,
        ultimo_stamp_tf_old_data_s=stamp_tf_old if stamp_tf_old is not None else nan,
        coda_persa_csv_s=(riferimento - ultimo_csv) if csv_st else nan,
        coda_persa_printf_s=(riferimento - ultimo_pf) if pf_st else nan,
        nota_coda=("coda persa stimata rispetto all'ultimo stamp noto (CSV, printf o TF_OLD_DATA); "
                   f"al SIGKILL si perdono <= 8 KiB di CSV e <= {RIGHE_PRINTF_PERSE_MAX} righe printf"),
        **statistiche_time_on_lidar(records))


# ----------------------------------------------------------------------------- alignment
def allinea_q(records, sequenza_q, inizio=0, finestra=400, tolleranza=1e-4, tempi=None,
              penalita_s=0.3):
    """Align a sequence of observed q (consecutive distinct pose_quality values in an
    orchestrator trace) to the MOLA records, with increasing index.

    Returns a list as long as sequenza_q of indices into records (None if not found).
    Without `tempi`, greedy: first occurrence of each q after the last match. With `tempi`
    (trace time of each q) a repeated q may have several candidates: the monotone path that
    minimises the sum of jumps of the offset stamp - t between consecutive matches is chosen
    (the offset is nearly constant within a window), with `penalita_s` per unmatched q.
    Rejected scans publish no pose, so they never appear in the sequence and fall between
    two consecutive matches.
    """
    n = len(sequenza_q)
    if tempi is None:
        out, p = [], inizio
        for q in sequenza_q:
            k = next((j for j in range(p, min(p + finestra, len(records)))
                      if abs(records[j]["q"] - q) <= tolleranza), None)
            out.append(k)
            if k is not None:
                p = k + 1
        return out
    fine = min(len(records), inizio + finestra + n)
    cand = [[j for j in range(inizio, fine) if abs(records[j]["q"] - q) <= tolleranza]
            for q in sequenza_q]
    # dynamic programming: costo[i][c] = min cost with the i-th q matched to candidate c
    costo = [dict() for _ in range(n)]
    prec = [dict() for _ in range(n)]
    for i in range(n):
        for j in cand[i]:
            off_j = records[j]["stamp_s"] - tempi[i]
            best, da = penalita_s * i, None
            for a in range(i):
                for ja, ca in costo[a].items():
                    if ja >= j:
                        continue
                    c = ca + abs(off_j - (records[ja]["stamp_s"] - tempi[a])) + penalita_s * (i - a - 1)
                    if c < best:
                        best, da = c, (a, ja)
            costo[i][j] = best
            prec[i][j] = da
    best, ultimo = penalita_s * n, None
    for i in range(n):
        for j, c in costo[i].items():
            if c + penalita_s * (n - 1 - i) < best:
                best, ultimo = c + penalita_s * (n - 1 - i), (i, j)
    out = [None] * n
    while ultimo is not None:
        i, j = ultimo
        out[i] = j
        ultimo = prec[i][j]
    return out


def statistiche_time_on_lidar(records):
    """Mean and p99 of time_onLidar (ms) over records that have it; p99 by 'nearest'."""
    v = sorted(r["time_onLidar"] * 1000.0 for r in records if math.isfinite(r.get("time_onLidar", float("nan"))))
    if not v:
        return dict(time_onLidar_media_ms=float("nan"), time_onLidar_p99_ms=float("nan"), time_onLidar_n=0,
                    time_onLidar_p99_metodo="nearest (indice round(0.99*(n-1)) dei valori ordinati)")
    return dict(time_onLidar_media_ms=sum(v) / len(v), time_onLidar_p99_ms=v[int(round(0.99 * (len(v) - 1)))],
                time_onLidar_n=len(v), time_onLidar_p99_metodo="nearest (indice round(0.99*(n-1)) dei valori ordinati)")


# ----------------------------------------------------------------------------- whole run
def file_di_run(run_dir, tag="victim"):
    """(log, csv) of the victim in <run>/logs or <run>/rep_01/logs; None where missing."""
    for base in (run_dir, os.path.join(run_dir, "rep_01")):
        logs = os.path.join(base, "logs")
        if os.path.isdir(logs):
            log = os.path.join(logs, f"mola_{tag}.log")
            csv_p = os.path.join(logs, f"traces_{tag}.csv")
            return (log if os.path.exists(log) else None), (csv_p if os.path.exists(csv_p) else None)
    return None, None


def analizza_run(run_dir, soglia, tag="victim", passo_m=1.0):
    """(records, riepilogo, bins) of a run; records empty if the CSV is missing too."""
    log, csv_p = file_di_run(run_dir, tag)
    printf = leggi_printf(log) if log else []
    stamp_tf = ultimo_stamp_tf_old_data(log) if log else None
    csv_rec, tronche = [], 0
    if csv_p:
        csv_rec = leggi_csv_mola(csv_p, soglia)
        tronche = _righe_csv(csv_p)[2]
    records = unisci(printf, csv_rec, soglia)
    rp = riepilogo(records, printf, csv_rec, tronche, stamp_tf, soglia)
    rp.update(log=log, csv=csv_p)
    return records, rp, scarti_per_metro(records, passo_m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="cartella della run (o rep_01)")
    ap.add_argument("--tag", default="victim")
    ap.add_argument("--soglia", type=float, default=None,
                    help="min_icp_goodness; se assente, letta da --config con l'ambiente corrente")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "..", "config", "mola",
                                                     "lidar3d-gicp-optimize-twist-local.yaml"))
    ap.add_argument("--passo", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="scrive i record per scan in questo file")
    a = ap.parse_args()
    soglia = a.soglia
    fonte = "riga di comando"
    if soglia is None:
        soglia, fonte = soglia_min_icp_goodness(os.path.abspath(a.config), dict(os.environ))
    records, rp, bins = analizza_run(a.run, soglia, a.tag, a.passo)
    rp["fonte_soglia"] = fonte
    print(json.dumps(rp, indent=1, ensure_ascii=False, default=str))
    print("scarti per metro (robot_x di MOLA, ritardo di uno scan):")
    for b in bins:
        print(f"  [{b['bin_inizio_m']:.1f}, {b['bin_fine_m']:.1f}) scan {b['n_scan']:3d} attesi "
              f"{b['n_attesi_10hz']:3d} scartati {b['n_scartati']:2d} (csv {b['n_scartati_csv']:2d}) "
              f"q min {b['q_min']:.3f}")
    for r in records:
        if r["scartato"] or r["incoerente"]:
            print(f"  stamp {r['stamp_s']:.1f} q {r['q']:.4f} iters {r['iters']} robot_x "
                  f"{r['robot_x']:.3f} fonte {r['fonte']} scartato {r['scartato']} {r['nota']}")
    if a.csv:
        colonne = ["stamp_s", "fonte", "scartato", "scartato_csv", "is_good", "q", "goodness",
                   "iters", "term_reason", "consecutive_bad", "pathstep", "robot_x", "robot_y",
                   "sigma", "time_onLidar", "incoerente", "nota", "riga"]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colonne, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        print("record scritti in", a.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
