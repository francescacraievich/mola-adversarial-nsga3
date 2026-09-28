#!/usr/bin/env python3
"""
Diagnostica di un rollout: confronto fra stima MOLA, posa vera e comandi.

Legge le tracce CSV per tick prodotte da attack_orchestrator.py --trace
(est_x/y/yaw, true_x/y/yaw, waypoint, cmd_v/cmd_w) e per ciascuna riporta:

  1. rapporto fra spostamento stimato e vero (sotto/sovrastima di MOLA);
  2. se la rotazione vera cresce sempre nello stesso verso (bias sistematico)
     oppure oscilla (errore casuale);
  3. errore di puntamento sulla stima e statistiche dei comandi.

La distinzione fra errore casuale e bias orienta la ricerca: un errore casuale
punta a MOLA in transitorio e si riduce con piu' riscaldamento o un orizzonte
piu' lungo; un bias sempre nello stesso verso punta a un problema di frame o di
convenzione di segno, che il riscaldamento non corregge.

Uso:
    python3 src/analysis/diagnose_rollout.py data/attack/<run>/traces/w0_nominal.csv
    python3 src/analysis/diagnose_rollout.py data/attack/<run>/traces/*.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    d = {}
    for k in rows[0]:
        d[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                         for r in rows])
    return d


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def analyse(path, verbose=False):
    d = load(path)
    if d is None:
        print(f"{path}: vuoto")
        return

    est = np.column_stack([d["est_x"], d["est_y"]])
    tru = np.column_stack([d["true_x"], d["true_y"]])
    est_yaw = unwrap_deg(d["est_yaw_deg"])
    tru_yaw = unwrap_deg(d["true_yaw_deg"])

    ok = ~np.isnan(tru[:, 0])
    if ok.sum() < 3:
        print(f"{path}: troppe pose vere mancanti")
        return

    # Spostamento dall'inizio, nei due sistemi. Non si confrontano le posizioni
    # assolute: il frame di MOLA ha origine nella posa in cui si e'
    # rilocalizzata, quello vero e' il mondo di Isaac Sim.
    est_d = np.linalg.norm(est - est[0], axis=1)
    tru_d = np.linalg.norm(tru[ok] - tru[ok][0], axis=1)

    est_rot = est_yaw - est_yaw[0]
    tru_rot = tru_yaw[ok] - tru_yaw[ok][0]

    n = min(len(est_d), len(tru_d))
    scale = est_d[:n] / np.maximum(tru_d[:n], 1e-6)
    valid = tru_d[:n] > 0.05          # sotto 5 cm il rapporto e' rumore

    print(f"\n{'='*70}")
    print(f"  {Path(path).name}   ({len(d['t'])} tick, "
          f"{d['t'][-1]:.1f}s)")
    print(f"{'='*70}")

    print(f"  spostamento     vero {tru_d[-1]:6.3f} m    "
          f"stimato {est_d[-1]:6.3f} m")
    if valid.any():
        sc = float(np.median(scale[valid]))
        print(f"  rapporto stima/verita' (mediano): {sc:.3f}")
        if sc < 0.85:
            print("    -> MOLA SOTTOSTIMA lo spostamento: il robot va oltre "
                  "prima di credersi arrivato")
        elif sc > 1.15:
            print("    -> MOLA SOVRASTIMA: il robot si ferma prima")

    print(f"  rotazione       vera {tru_rot[-1]:+7.1f}\u00b0   "
          f"stimata {est_rot[-1]:+7.1f}\u00b0")

    # Segnale che distingue bias da rumore: frazione di passi di rotazione vera
    # con lo stesso segno. Vicina a 0 o a 1 indica un verso preferenziale.
    if len(tru_rot) > 5:
        steps = np.diff(tru_rot)
        frac_pos = float((steps > 0).mean())
        print(f"  passi di rotazione positivi: {100*frac_pos:.0f}%")
        if frac_pos > 0.8 or frac_pos < 0.2:
            print("    -> BIAS SISTEMATICO: il robot curva sempre dallo stesso")
            print("       lato. Non e' MOLA in transitorio, e' un problema di")
            print("       frame o di convenzione di segno. Verificare")
            print("       LIDAR_POSE_X (--lidar-pose-x +0.2317) e il segno di")
            print("       angular.z fra follower e differential_controller.")
        else:
            print("    -> errore senza verso preferenziale: compatibile con")
            print("       MOLA in transitorio. Alzare --warmup o --horizon.")

    # Errore di puntamento calcolato sulla stima: se il controllo comanda
    # rotazione con puntamento gia' piccolo il problema e' nel controllo, se la
    # comanda perche' la stima colloca il waypoint altrove il problema e' a monte.
    bearing = np.degrees(np.arctan2(d["wp_y"] - d["est_y"],
                                    d["wp_x"] - d["est_x"])) - d["est_yaw_deg"]
    bearing = (bearing + 180) % 360 - 180
    print(f"  errore di puntamento (sulla stima): "
          f"medio {np.nanmean(np.abs(bearing)):5.1f}\u00b0   "
          f"max {np.nanmax(np.abs(bearing)):5.1f}\u00b0")
    print(f"  comandi: v medio {np.nanmean(d['cmd_v']):.3f} m/s   "
          f"w medio {np.nanmean(d['cmd_w']):+.3f} rad/s   "
          f"w>0 nel {100*float((d['cmd_w']>0).mean()):.0f}% dei tick")

    if abs(np.nanmean(d["cmd_w"])) > 0.05:
        print("    -> il controllo comanda rotazione quasi sempre nello stesso")
        print("       verso: sta inseguendo un waypoint che la stima colloca")
        print("       costantemente di lato.")

    if verbose:
        print(f"\n  {'t':>6} {'est_d':>7} {'tru_d':>7} {'est_rot':>8} "
              f"{'tru_rot':>8} {'bear':>7} {'cmd_w':>7}")
        step = max(1, len(d["t"]) // 25)
        for i in range(0, len(d["t"]), step):
            j = min(i, len(tru_d) - 1)
            print(f"  {d['t'][i]:6.2f} {est_d[i]:7.3f} {tru_d[j]:7.3f} "
                  f"{est_rot[i]:+8.1f} {tru_rot[j]:+8.1f} "
                  f"{bearing[i]:+7.1f} {d['cmd_w'][i]:+7.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--verbose", action="store_true",
                    help="stampa l'andamento nel tempo, non solo il riepilogo")
    args = ap.parse_args()

    for p in args.traces:
        analyse(p, args.verbose)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
