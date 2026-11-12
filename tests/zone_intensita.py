#!/usr/bin/env python3
"""Points touched per operator: attacker zones against random mode with the same genome and
clouds, from the outputs of vz_nodo.py (mode "uscite"). Per operator (g5, ghosts g6+g16,
g10, g13): ratio of the sums zones/random, mean and 95 % CI of the per-scan ratio, share of
scans with insufficient zones (scarto > 0, from the generator log).

Usage: zone_intensita.py <random.json> <zone.json> [uscita.json [cartella_zone genomi.json]]
"""
import json
import sys

import numpy as np

OPS = ("g5", "fantasmi", "g10", "g13")


def _conta(h, serie_a):
    h = h or {}
    gh = h.get("gh", 0) if serie_a else h.get("g6", 0) + h.get("g16", 0)
    return {"g5": h.get("g5", 0), "fantasmi": gh, "g10": h.get("g10", 0), "g13": h.get("g13", 0)}


def _scarto(r):
    s = r.get("zone_scarto") or {}
    return {"g5": s.get("g5"), "fantasmi": s.get("g6", s.get("g16")), "g10": s.get("g10"), "g13": s.get("g13")}


def _attesi(dir_zone, genomi_json):
    """Expected points per cloud from the zone files (same rule as the generator):
    function (genome, metre, n) -> {operator: expected}.
    """
    import glob
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.perturbations import zone_attaccante as za
    gen = {g["id"]: za.genoma_sha(g["genoma"]) for g in json.load(open(genomi_json))}
    cache = {}

    def f(gid, w, n):
        key = (gid, w)
        if key not in cache:
            fz = glob.glob(f"{dir_zone}/w{w}_{gen[gid]}.npz")
            if not fz:
                cache[key] = None
            else:
                m_, a_, sha_, h_ = za._leggi(fz[0])
                z = za.Zone(m_, a_, sha_, h_)
                p = za.generatore(z.meta["limiti"]).encode_perturbation(np.asarray(z.meta["genoma_f32"]))
                cache[key] = (z, p)
        if cache[key] is None:
            return {}
        z, p = cache[key]
        out = {}
        t5 = int(round(z.quota_g5 * n)) if (z.quota_g5 is not None and p["dropout_rate"] > 0.01) else 0
        if z.quota_g5 is not None and p["dropout_rate"] > 0.01:
            out["g5"] = t5
        if p["ghost_ratio"] > 0.01:
            out["fantasmi"] = int((n - t5) * p["ghost_ratio"])
        if z.quota_g10 is not None:
            out["g10"] = int(round(z.quota_g10 * n))
        if "g13" in z.obiettivi:
            out["g13"] = z.obiettivi["g13"][0]
        return out
    return f


def confronta(a_json, z_json, dir_zone=None, genomi_json=None):
    A = {(r["g"], r["w"], r["j"], r["stream"]): r for r in json.load(open(a_json))["righe"]}
    Z = {(r["g"], r["w"], r["j"], r["stream"]): r for r in json.load(open(z_json))["righe"]}
    att_f = _attesi(dir_zone, genomi_json) if dir_zone else None
    chiavi = [k for k in Z if k in A]
    out = {"n_scansioni": len(chiavi), "per_operatore": {}}
    for op in OPS:
        a = np.array([_conta(A[k]["n_hit"], True)[op] for k in chiavi], dtype=float)
        z = np.array([_conta(Z[k]["n_hit"], False)[op] for k in chiavi], dtype=float)
        att = a > 0
        if not att.any():
            out["per_operatore"][op] = {"attivo": 0}
            continue
        rr = z[att] / a[att]
        if att_f is not None:
            sc = []
            for k, x in zip(chiavi, att):
                if x:
                    t = att_f(k[0], k[1], Z[k]["n_in"]).get(op)
                    if t is not None:
                        sc.append(t - _conta(Z[k]["n_hit"], False)[op])
        else:
            sc = [_scarto(Z[k]).get(op) for k, x in zip(chiavi, att) if x]
            sc = [v for v in sc if v is not None]
        m, sd = float(rr.mean()), float(rr.std(ddof=1)) if len(rr) > 1 else 0.0
        out["per_operatore"][op] = {
            "attivo": int(att.sum()), "somma_serieA": int(a.sum()), "somma_zone": int(z.sum()),
            "rapporto_somme": float(z.sum() / a.sum()),
            "rapporto_medio": m, "ic95_media": [m - 1.96 * sd / np.sqrt(len(rr)), m + 1.96 * sd / np.sqrt(len(rr))],
            "rapporto_p2_5_p97_5": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))],
            "zone_insufficienti": int(sum(v > 0 for v in sc)), "con_scarto": len(sc),
            "per_genoma": {g: float(z[[x and k[0] == g for k, x in zip(chiavi, att)]].sum()
                                    / max(a[[x and k[0] == g for k, x in zip(chiavi, att)]].sum(), 1))
                           for g in sorted({k[0] for k, x in zip(chiavi, att) if x})},
        }
    return out


if __name__ == "__main__":
    r = confronta(sys.argv[1], sys.argv[2], *(sys.argv[4:6] if len(sys.argv) > 5 else ()))
    if len(sys.argv) > 3:
        json.dump(r, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(r, indent=1))
