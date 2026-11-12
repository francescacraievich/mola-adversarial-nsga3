#!/usr/bin/env python3
"""
Deviation between two true trajectories aligned by arc length.

The end-point deviation ignores what happens along the way: an attack that
pushes the robot off course mid-window and lets it come back scores zero. Here
the nominal and attacked trajectories (both true poses from the same initial
state) are compared at equal travelled distance: the deviation at s is the
distance between where the attacked robot is after s metres and where the
nominal one was after the same s metres.

Alignment is by distance, not time, because under attack the robot is slower
(about twice the frames per metre in the recorded runs): at equal time one would
measure the delay rather than the offset.

Command line, on a traces/ folder written by the orchestrator:

    python3 src/analysis/path_deviation.py data/attack/attack_v5/traces
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


def true_path(rows):
    """Distinct true positions of a trace (list of dicts or Nx2 array).

    The true pose in the traces is updated at intervals and repeated on the
    ticks in between: consecutive duplicates are dropped, otherwise the
    stretches where the robot stood still would weigh more.
    """
    if len(rows) == 0:
        return np.zeros((0, 2))
    if isinstance(rows[0], dict):
        pts = [(float(r["true_x"]), float(r["true_y"])) for r in rows]
    else:
        pts = [(float(p[0]), float(p[1])) for p in rows]
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1] and p[0] == p[0]:
            out.append(p)
    return np.array(out, dtype=np.float64)


def arc_length(xy):
    if len(xy) < 2:
        return np.zeros(len(xy))
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def sample_at(xy, s, s_query):
    """Position along polyline xy at arc length s_query (past the end: last point)."""
    s_query = np.clip(s_query, 0.0, s[-1])
    x = np.interp(s_query, s, xy[:, 0])
    y = np.interp(s_query, s, xy[:, 1])
    return np.stack([x, y], axis=1)


def path_deviation(nom_xy, cand_xy, step=0.02):
    """Deviation between candidate and nominal trajectory at equal path length.

    Both polylines are resampled every `step` m up to the longer length; past
    its end the shorter one is held at its last point, so a robot that stops
    early or spins in place is not rewarded by a shorter comparison.

    Returns a dict with dev_mean, dev_max, dev_end (m), len_nom and len_cand
    (travelled m) and the profile (s, dev) for plots.
    """
    nom_xy = np.asarray(nom_xy, dtype=np.float64)
    cand_xy = np.asarray(cand_xy, dtype=np.float64)
    if len(nom_xy) < 2 or len(cand_xy) < 2:
        return None
    s_n, s_c = arc_length(nom_xy), arc_length(cand_xy)
    s_max = max(s_n[-1], s_c[-1])
    if s_max <= 0:
        return None
    s = np.arange(0.0, s_max + step, step)
    diff = sample_at(cand_xy, s_c, s) - sample_at(nom_xy, s_n, s)
    dev = np.linalg.norm(diff, axis=1)
    return {
        "dev_mean": float(dev.mean()),
        "dev_max": float(dev.max()),
        "dev_end": float(np.linalg.norm(cand_xy[-1] - nom_xy[-1])),
        "len_nom": float(s_n[-1]),
        "len_cand": float(s_c[-1]),
        "profile_s": s,
        "profile_dev": dev,
    }


def read_trace(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", type=Path, help="cartella traces/ di una run")
    ap.add_argument("--step", type=float, default=0.02, help="passo di ricampionamento in metri")
    args = ap.parse_args()

    files = sorted(args.traces.glob("w*_nominal.csv"),
                   key=lambda p: int(re.match(r"w(\d+)_", p.name).group(1)))
    if not files:
        print(f"nessuna traccia nominale in {args.traces}")
        return 1

    print(f"{'finestra':>8}  {'rollout':<12} {'punti':>5} {'percorso':>8} "
          f"{'dev_end':>8} {'dev_mean':>8} {'dev_max':>8}")
    for nom_file in files:
        k = re.match(r"w(\d+)_", nom_file.name).group(1)
        nom = true_path(read_trace(nom_file))
        others = sorted(args.traces.glob(f"w{k}_*.csv"))
        for f in others:
            if f == nom_file:
                continue
            cand = true_path(read_trace(f))
            r = path_deviation(nom, cand, args.step)
            tag = f.name[len(f"w{k}_"):-4]
            if r is None:
                print(f"{k:>8}  {tag:<12} {len(cand):>5}   troppo corta")
                continue
            print(f"{k:>8}  {tag:<12} {len(cand):>5} {r['len_cand']:>7.2f}m "
                  f"{r['dev_end']*100:>7.1f} {r['dev_mean']*100:>8.1f} "
                  f"{r['dev_max']*100:>8.1f}   cm   (nominale: {len(nom)} punti, "
                  f"{r['len_nom']:.2f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
