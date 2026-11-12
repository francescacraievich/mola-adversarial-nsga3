#!/usr/bin/env python3
"""
Distance of a true trajectory from a reference polyline (baseline).

The per-window damage compares the rollout with the nominal of the same window
(src/analysis/path_deviation.py). Here the reference is a fixed polyline in the
Isaac odom frame, built by the analysis scripts from unattacked traversals: the
distance of each true pose from the nearest point of the polyline does not
depend on the window nominal and is not affected by its noise.

Point-to-polyline distance is the distance to the nearest segment. The polyline
is not extended before its first or past its last point: the distance from the
end point applies, so a robot that overshoots the target accumulates damage.

Baseline CSV format (one row per sample, sorted by increasing s, lines starting
with '#' ignored):

    s_m,x_m,y_m
"""

import io
from pathlib import Path
import warnings

import numpy as np


def load_baseline(path):
    """Mx2 polyline (x, y in m) from the `s_m,x_m,y_m` CSV.

    Raises ValueError with a readable message if the file is missing, empty or
    lacks three numeric columns.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"baseline non trovata: {path}")
    # Comment lines anywhere, optional header (first non-numeric line).
    rows = [l for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if rows and not rows[0].lstrip()[0] in "0123456789+-.":
        rows = rows[1:]
    if not rows:
        raise ValueError(f"baseline vuota: {path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.loadtxt(io.StringIO("\n".join(rows)), delimiter=",", ndmin=2)
    except Exception as e:  # non-numeric columns, ragged rows, etc.
        raise ValueError(f"baseline malformata ({path}): {e}") from e
    if data.size == 0:
        raise ValueError(f"baseline vuota: {path}")
    if data.shape[1] < 3:
        raise ValueError(f"baseline malformata ({path}): attese 3 colonne s_m,x_m,y_m, "
                         f"trovate {data.shape[1]}")
    if not np.all(np.isfinite(data[:, :3])):
        raise ValueError(f"baseline malformata ({path}): valori non finiti")
    s = data[:, 0]
    if np.any(np.diff(s) < 0):
        raise ValueError(f"baseline malformata ({path}): s_m non e' crescente")
    return np.ascontiguousarray(data[:, 1:3], dtype=np.float64)


def dist_to_polyline(points, poly):
    """Euclidean distance of each point (Nx2) from the polyline (Mx2).

    Each point is projected on every segment (t clipped to [0, 1]) and the
    minimum is kept. Degenerate segments (duplicate points) count as their single
    point; with one vertex it is the distance from that vertex.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    poly = np.atleast_2d(np.asarray(poly, dtype=np.float64))
    if points.shape[0] == 0:
        return np.zeros(0)
    if poly.shape[0] == 0:
        raise ValueError("polilinea vuota")
    if poly.shape[0] == 1:
        return np.linalg.norm(points - poly[0], axis=1)
    a = poly[:-1]                      # (M-1, 2) segment starts
    d = poly[1:] - a                   # (M-1, 2) segment vectors
    len2 = np.einsum("ij,ij->i", d, d)  # (M-1,)
    # Projection of every point on every segment: (N, M-1).
    ap = points[:, None, :] - a[None, :, :]
    t = np.einsum("nmj,mj->nm", ap, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(len2 > 0, t / len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * d[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return dist.min(axis=1)


def baseline_damage(cand_xy, poly):
    """Deviation of a rollout from the baseline: dev_end, dev_mean, dev_max (m).

    cand_xy are the true rollout poses (Nx2, odom frame), poly the reference
    polyline (Mx2). dev_end is the distance of the last pose, dev_mean and
    dev_max the mean and maximum over all poses. Non-finite poses are dropped;
    if none is left all three are NaN and the caller picks the fallback (as
    window_damage does with the final rollout pose).
    """
    cand_xy = np.atleast_2d(np.asarray(cand_xy, dtype=np.float64)).reshape(-1, 2)
    cand_xy = cand_xy[np.all(np.isfinite(cand_xy), axis=1)]
    if cand_xy.shape[0] == 0:
        return {"dev_end": float("nan"), "dev_mean": float("nan"), "dev_max": float("nan")}
    dist = dist_to_polyline(cand_xy, poly)
    return {"dev_end": float(dist[-1]),
            "dev_mean": float(dist.mean()),
            "dev_max": float(dist.max())}
