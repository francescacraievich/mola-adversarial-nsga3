#!/usr/bin/env python3
"""
Straight-line baseline in direction mode: polyline of the victim's true pose
from a run without attack, with --goal-mode direction and N windows.

Same recipe as the multi-run baseline (true path per run, resampling, mean
polyline up to the shortest length, CSV writer), reduced to a single run:
  - the traces traces/w<k>_victim.csv of the driven windows are concatenated in
    window order (true_x, true_y in the odom frame), dropping consecutive
    duplicates, also at the joints (true_path in src/analysis/path_deviation.py);
  - arc length s, resampling at a fixed step (default 0.05 m) and a last point
    at s = run length (with a single run the mean changes nothing);
  - CSV `s_m,x_m,y_m` with the same number of digits, the format read by
    load_baseline (src/analysis/baseline_distance.py), used by
    --damage-ref baseline --baseline-path.

Checks (all required: if one fails the CSV is not written; the JSON report is
always written):
  - run directory completed (`completata` file from run_campaign.sh);
  - driven windows = --finestre, no held window (victim_held),
    run_end.reason = finestre_completate;
  - no victim pose loss: victim_status "ok" in every driven window
    ("arrived" is not expected in direction mode), victim_lost never true,
    no record with a victim or infrastructure outcome;
  - victim trace present for every driven window;
  - final true odom x of the victim <= --x-max, and maximum true x over all
    traces of the repetition (nominals included: they move the robot in Isaac) <= --x-max;
  - run seed equal to --seme, if given ("seed" line of the repetition log).
The orchestrator's TRUE_CAP is per rollout (2 m of true path), not a maximum x:
the 7.0 m limit is enforced here.

Files are never overwritten: if the CSV or the JSON exists the script stops.

Usage:
    python3 scripts/costruisci_baseline_direction.py REP_DIR --csv OUT.csv
        [--json OUT.json] [--finestre 6] [--x-max 7.0] [--passo-m 0.05]
        [--seme 1001] [--stato pilota]
    python3 scripts/costruisci_baseline_direction.py --controllo-x REP_DIR [--x-max 7.0]
REP_DIR = repetition directory (history.json, traces/, completata).
Exit codes: 0 checks passed and CSV written; 1 checks failed;
2 usage error or file already present. With --controllo-x: 0 within the limit, 1 beyond.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from path_deviation import true_path, arc_length  # noqa: E402
from baseline_distance import load_baseline, dist_to_polyline  # noqa: E402

PASSO_M = 0.05
X_MAX_M = 7.0
FINESTRE = 6
STATI_VITTIMA_OK = ("ok",)


def leggi_traccia(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def finestre_guidate(history):
    """Indices of driven and held (victim_held) windows, as in the orchestrator's
    direction_window_count."""
    held = sorted({r.get("window") for r in history if r.get("victim_held") and isinstance(r.get("window"), int)})
    guidate = sorted({r.get("window") for r in history
                      if isinstance(r.get("window"), int) and r.get("window") not in held})
    return guidate, held


def percorso_vero(rep, finestre):
    """Distinct true victim poses concatenated over the given windows."""
    pts, n_righe, file_usati = [], 0, []
    for k in finestre:
        p = rep / "traces" / f"w{k}_victim.csv"
        if not p.exists():
            continue
        file_usati.append(p.name)
        rows = leggi_traccia(p)
        n_righe += len(rows)
        for q in true_path(rows):
            if not (math.isfinite(q[0]) and math.isfinite(q[1])):
                continue
            if not pts or (q[0] != pts[-1][0] or q[1] != pts[-1][1]):
                pts.append((float(q[0]), float(q[1])))
    return np.array(pts, dtype=np.float64).reshape(-1, 2), n_righe, file_usati


def polilinea(xy, passo=PASSO_M):
    """Resample at a fixed step, with the last point at s = length."""
    s = arc_length(xy)
    L = float(s[-1])
    sq = np.arange(0.0, L + 1e-9, passo)
    sq = sq[sq <= L]
    if L > sq[-1] + 1e-9:
        sq = np.concatenate([sq, [L]])
    return sq, np.stack([np.interp(sq, s, xy[:, 0]), np.interp(sq, s, xy[:, 1])], axis=1), L


def scrivi_polilinea(path, s, xy):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_m", "x_m", "y_m"])
        for si, (x, y) in zip(s, xy):
            w.writerow([f"{si:.4f}", f"{x:.6f}", f"{y:.6f}"])


def x_max_tracce(rep):
    """Maximum true x (odom frame) over all CSV traces of the repetition."""
    best, best_f = None, None
    for p in sorted((Path(rep) / "traces").glob("*.csv")):
        try:
            rows = leggi_traccia(p)
        except OSError:
            continue
        for r in rows:
            try:
                x = float(r["true_x"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and (best is None or x > best):
                best, best_f = x, p.name
    return best, best_f


def seme_dal_log(rep):
    """Seed printed by the orchestrator (line 'seed : N') in the repetition log."""
    log = rep.parent / f"{rep.name}.log"
    if not log.exists():
        return None
    m = re.search(r"^\s*seed\s*:\s*(\d+)", log.read_text(errors="replace"), re.M)
    return int(m.group(1)) if m else None


def controlla(rep, history, finestre_attese, x_max, seme=None):
    """Baseline checks: list of dicts (name, value, expected, ok) plus auxiliary data."""
    c = []

    def add(nome, valore, atteso, ok):
        c.append({"controllo": nome, "valore": valore, "atteso": atteso, "ok": bool(ok)})

    add("cartella completata", (rep / "completata").exists(), True, (rep / "completata").exists())
    guidate, held = finestre_guidate(history)
    add("finestre guidate", len(guidate), finestre_attese, len(guidate) == finestre_attese)
    add("finestre trattenute (victim_held)", len(held), 0, len(held) == 0)
    run_end = next((r.get("run_end") for r in reversed(history) if r.get("run_end")), None)
    reason = (run_end or {}).get("reason")
    add("run_end.reason", reason, "finestre_completate", reason == "finestre_completate")
    modi = sorted({(r.get("goal_mode") or {}).get("mode") for r in history if r.get("goal_mode")})
    add("goal_mode nei record (se presente)", modi or None, "direction o assente",
        all(m == "direction" for m in modi))

    per_finestra = []
    stati_ko = []
    for r in history:
        w = r.get("window")
        st = str(r.get("status", ""))
        if st.startswith("victim_") or st in ("baseline_frame_mismatch", "victim_topic_foreign_publisher"):
            stati_ko.append(f"w{w}: status {st}")
        if w not in guidate:
            continue
        vst = r.get("victim_status")
        if vst not in STATI_VITTIMA_OK:
            stati_ko.append(f"w{w}: victim_status {vst}")
        if r.get("victim_lost"):
            stati_ko.append(f"w{w}: victim_lost")
        tp = rep / "traces" / f"w{w}_victim.csv"
        xy_w = true_path(leggi_traccia(tp)) if tp.exists() else np.zeros((0, 2))
        per_finestra.append({"finestra": w, "status_nominale": r.get("status"), "victim_status": vst,
                             "victim_travelled_m": r.get("victim_travelled"),
                             "victim_track_ratio": r.get("victim_track_ratio"),
                             "traccia": tp.name if tp.exists() else None, "pose_distinte": int(len(xy_w)),
                             "x_fine_m": round(float(xy_w[-1][0]), 4) if len(xy_w) else None,
                             "window_time_s": r.get("window_time_s")})
    add("esiti della vittima (nessuna perdita di posa, nessun arrived)", stati_ko or "nessun esito anomalo",
        "nessun esito anomalo", not stati_ko)
    manca = [f["finestra"] for f in per_finestra if f["pose_distinte"] < 2]
    add("traccia della vittima per ogni finestra guidata (>= 2 pose)", manca or "tutte", "tutte", not manca)

    xy, n_righe, file_usati = percorso_vero(rep, guidate)
    x_fin = float(xy[-1, 0]) if len(xy) else None
    add("x odom vera finale della vittima [m]", round(x_fin, 4) if x_fin is not None else None,
        f"<= {x_max}", x_fin is not None and x_fin <= x_max)
    xm, xm_f = x_max_tracce(rep)
    add("x odom vera massima su tutte le tracce [m]", (round(xm, 4), xm_f) if xm is not None else None,
        f"<= {x_max}", xm is not None and xm <= x_max)
    if seme is not None:
        s_log = seme_dal_log(rep)
        add("seme della run (log della ripetizione)", s_log, seme, s_log == seme)
    dati = {"guidate": guidate, "trattenute": held, "run_end": run_end, "per_finestra": per_finestra,
            "xy": xy, "n_righe": n_righe, "file_usati": file_usati}
    return c, dati


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ora():
    return subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rep", nargs="?", type=Path, help="cartella della ripetizione (rep_NN)")
    ap.add_argument("--csv", type=Path, help="polilinea di uscita (nome nuovo, mai sovrascritto)")
    ap.add_argument("--json", type=Path, default=None, help="rapporto (default: il CSV con estensione .json)")
    ap.add_argument("--finestre", type=int, default=FINESTRE)
    ap.add_argument("--x-max", type=float, default=X_MAX_M, help="x odom vera massima ammessa (m)")
    ap.add_argument("--passo-m", type=float, default=PASSO_M)
    ap.add_argument("--seme", type=int, default=None, help="seme atteso della run")
    ap.add_argument("--stato", default="pilota", help="etichetta nel rapporto (es. pilota, serieB)")
    ap.add_argument("--controllo-x", type=Path, default=None, metavar="REP_DIR",
                    help="solo la x vera massima delle tracce di REP_DIR contro --x-max")
    a = ap.parse_args(argv)

    if a.controllo_x is not None:
        xm, f = x_max_tracce(a.controllo_x)
        if xm is None:
            print("x_max nessuna_traccia")
            return 0
        print(f"x_max {xm:.4f} {f}")
        return 1 if xm > a.x_max else 0

    if a.rep is None or a.csv is None:
        ap.error("servono REP_DIR e --csv")
    rep = a.rep.resolve()
    f_csv = a.csv.resolve()
    f_json = (a.json or f_csv.with_suffix(".json")).resolve()
    for f in (f_csv, f_json):
        if f.exists():
            print(f"ERRORE: {f} esiste gia': i file di baseline non si sovrascrivono")
            return 2
    hp = rep / "history.json"
    if not hp.exists():
        print(f"ERRORE: history assente: {hp}")
        return 2
    history = json.load(open(hp))

    controlli, d = controlla(rep, history, a.finestre, a.x_max, a.seme)
    ok = all(x["ok"] for x in controlli)
    for x in controlli:
        print(f"  {'ok' if x['ok'] else 'KO'}  {x['controllo']}: {x['valore']} (atteso {x['atteso']})")

    meta = {
        "descrizione": ("baseline del rettilineo in modo direction: polilinea della posa vera della vittima "
                        "di una run senza attacco (--goal-mode direction, finestre fisse)"),
        "stato": a.stato,
        "data_costruzione": ora(),
        "comando": "python3 " + " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        "ripetizione": str(rep),
        "controlli_superati": ok,
        "controlli": controlli,
        "finestre_guidate": d["guidate"], "finestre_trattenute": d["trattenute"], "run_end": d["run_end"],
        "per_finestra": d["per_finestra"],
        "tracce_concatenate": d["file_usati"], "righe_csv": d["n_righe"], "pose_vere_distinte": int(len(d["xy"])),
        "x_max_m": a.x_max, "passo_m": a.passo_m,
        "csv": None,
    }
    xy = d["xy"]
    if ok and len(xy) >= 2:
        s, poly, L = polilinea(xy, a.passo_m)
        f_csv.parent.mkdir(parents=True, exist_ok=True)
        scrivi_polilinea(f_csv, s, poly)
        # read back with the orchestrator's reader and measure the offset from the true poses
        letta = load_baseline(f_csv)
        scarto = dist_to_polyline(xy, letta)
        meta.update({
            "csv": str(f_csv), "csv_sha256": sha256(f_csv),
            "n_punti": int(len(s)), "lunghezza_m": round(L, 4),
            "primo_punto": [round(float(poly[0, 0]), 4), round(float(poly[0, 1]), 4)],
            "ultimo_punto": [round(float(poly[-1, 0]), 4), round(float(poly[-1, 1]), 4)],
            "y_min_m": round(float(xy[:, 1].min()), 4), "y_max_m": round(float(xy[:, 1].max()), 4),
            "lettore": "load_baseline (src/analysis/baseline_distance.py)",
            "riletti_punti": int(len(letta)),
            "scarto_pose_dalla_polilinea_cm": {"media": round(float(scarto.mean() * 100), 4),
                                               "max": round(float(scarto.max() * 100), 4), "n": int(len(xy))},
        })
        print(f"polilinea: {len(s)} punti, passo {a.passo_m} m, lunghezza {L:.4f} m, "
              f"da ({poly[0,0]:.4f}, {poly[0,1]:.4f}) a ({poly[-1,0]:.4f}, {poly[-1,1]:.4f}) -> {f_csv}")
    elif ok:
        meta["controlli_superati"] = ok = False
        print("KO: meno di 2 pose vere distinte")
    f_json.parent.mkdir(parents=True, exist_ok=True)
    with open(f_json, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"rapporto: {f_json}")
    if not ok:
        print("controlli falliti: CSV non scritto")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
