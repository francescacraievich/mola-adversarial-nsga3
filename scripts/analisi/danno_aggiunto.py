#!/usr/bin/env python3
"""Independent check of the victim's added damage per metre.

Usage: danno_aggiunto.py <baseline.csv> <rep_01 A> <rep_01 B>
For each run and window k: true poses from traces/w{k}_applied.csv (true_x,
true_y), with consecutive duplicates and non-finite rows removed; unsigned
distance from the baseline polyline (projection clamped to the segment, minimum
over segments); signed lateral offset from the line of the nearest segment
(positive to the left), extending the end segments beyond the endpoints. The
computation is written here without using the repository code. Compared with
history[k]['danno_metro_vittima'], history[k]['victim_window']['baseline'] and danno_run.
Breakdown at w4: longitudinal component beyond the last vertex.
"""
import csv, json, math, sys
import numpy as np


def leggi_baseline(p):
    pts = []
    for row in csv.reader(open(p)):
        if not row or row[0].startswith('#'):
            continue
        try:
            pts.append((float(row[1]), float(row[2])))
        except ValueError:
            continue
    return np.array(pts)


def pose(p):
    out = []
    for r in csv.DictReader(open(p)):
        x, y = float(r['true_x']), float(r['true_y'])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if out and (x, y) == out[-1]:
            continue
        out.append((x, y))
    return np.array(out)


def geom(P, poly):
    """Distance (m), signed lateral offset (m), index of the nearest segment, unclamped t."""
    dist = np.empty(len(P)); lat = np.empty(len(P)); seg = np.empty(len(P), int); tt = np.empty(len(P))
    for i, p in enumerate(P):
        best = None
        for j in range(len(poly) - 1):
            a, b = poly[j], poly[j + 1]
            d = b - a
            L2 = d @ d
            if L2 <= 0:
                continue
            t = ((p - a) @ d) / L2
            c = a + min(max(t, 0.0), 1.0) * d
            dd = math.hypot(*(p - c))
            if best is None or dd < best[0]:
                cr = d[0] * (p - a)[1] - d[1] * (p - a)[0]
                best = (dd, cr / math.sqrt(L2), j, t)
        dist[i], lat[i], seg[i], tt[i] = best
    return dist, lat, seg, tt


def closest_point(p, poly):
    best = None
    for j in range(len(poly) - 1):
        a, b = poly[j], poly[j + 1]
        d = b - a; L2 = d @ d
        if L2 <= 0:
            continue
        t = min(max(((p - a) @ d) / L2, 0.0), 1.0)
        c = a + t * d
        dd = math.hypot(*(p - c))
        if best is None or dd < best[0]:
            best = (dd, c)
    return best[1]


def main():
    poly = leggi_baseline(sys.argv[1])
    print(f'baseline: {len(poly)} vertici, da {poly[0]} a {poly[-1]}')
    tot = {}
    for lab, rep in (('A', sys.argv[2]), ('B', sys.argv[3])):
        H = json.load(open(f'{rep}/history.json'))
        s_add = 0.0
        print(f'\n== {lab}  {rep.split("/")[-2][:40]}')
        print(' k   n  start   mean  added | H: n start mean added | lat_s   lat_e  dlat(e-s)  lat_mean-lat_s | Hlat_s Hlat_e Hlatadd | shift H_shift')
        for k, h in enumerate(H):
            P = pose(f'{rep}/traces/w{k}_applied.csv')
            d, lat, seg, tt = geom(P, poly)
            d *= 100; lat *= 100
            s, m, e = d[0], d.mean(), d[-1]
            v = P[0] - closest_point(P[0], poly)
            sh, _, _, _ = geom(P, poly + v)
            sh = sh.mean() * 100
            dm = h['danno_metro_vittima']
            vb = (h.get('victim_window') or {}).get('baseline') or {}
            s_add += round(m - s, 3)
            print(f' w{k} {len(P):3d} {s:6.3f} {m:6.3f} {m-s:6.3f} | {dm["n_points"]:3d} {dm["start_cm"]:6.3f} {dm["mean_cm"]:6.3f} {dm["added_cm"]:6.3f} |'
                  f' {lat[0]:7.3f} {lat[-1]:7.3f} {lat[-1]-lat[0]:7.3f} {lat.mean()-lat[0]:7.3f} |'
                  f' {vb.get("lat_start_cm")} {vb.get("lat_end_cm")} {vb.get("lat_added_mean_cm")} | {sh:6.3f} {dm.get("added_shift_cm")}')
            if k == len(H) - 1:
                # breakdown: poses beyond the last vertex (t > 1 on the last segment)
                oltre = (seg == len(poly) - 2) & (tt > 1.0)
                perp = np.abs(lat)
                print(f'   w{k}: pose oltre l\'estremo della baseline {oltre.sum()}/{len(P)}; x finale {P[-1,0]:.4f} (estremo {poly[-1,0]:.4f})')
                print(f'   media distanza {m:.3f}  media |laterale| {perp.mean():.3f}  parte longitudinale {m-perp.mean():.3f} cm')
                print(f'   fine: distanza {e:.3f}  |laterale| {perp[-1]:.3f}  differenza {e-perp[-1]:.3f} cm')
                print(f'   added (distanza) {m-s:.3f}; added sulla |laterale| {perp.mean()-perp[0]:.3f}; dlat fine-inizio {lat[-1]-lat[0]:.3f}; shift {sh:.3f}')
                print(f'   fine percorso: lat_end {lat[-1]:.3f} cm')
        dr = H[-1].get('danno_run') or {}
        print(f' somma added (arrotondati come nella history) {s_add:.3f}; history danno_run.victim_added_sum_cm {dr.get("victim_added_sum_cm")}')
        tot[lab] = s_add
    print(f'\nB - A somma added {tot["B"] - tot["A"]:.3f}')


if __name__ == '__main__':
    main()
