#!/usr/bin/env python3
"""Points touched per scan and per operator: random mode (per-cloud seed as in the node) and
version I (insieme, genome seed and metre), with the same genome restricted to the operator
and the same recorded clouds.

Usage: versioneI_punti.py <root> <npz> <output.json> <genomes.json> <metres> <clouds per metre>

Restricted genome (as in the effect test): genes of the operator and the directions it uses
from the full genome, all others at -1; seed = genome_seed of the full genome. Points
touched: g5 removed, g6/g16 added, g10/g13 moved (random mode: changed rows; version I: hit
points declared by the generator).
"""
import json
import sys

import numpy as np

radice, npz, uscita, genomi_json, metri_s, n_nuv = sys.argv[1:7]
sys.path.insert(0, radice)
from src.nodes.perturbation_node import genome_seed  # noqa: E402
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402

KW = dict(max_point_shift=0.05, noise_std=0.02, max_dropout_rate=0.15, max_drift_lag_s=0.5)
INS = dict(operator_mode="coherent", struct_track="scan", struct_rule="caso",
           struct_g10_pavimento=True, struct_ripiego=True, struct_g13_punteggio=True,
           struct_insieme=True)
GENI = {"g5": [5], "g6": [6, 0, 1, 2], "g16": [6, 16, 0, 1, 2], "g10": [10, 7, 8, 9], "g13": [13, 0, 1, 2]}


def operatori(p):
    ops = []
    if p["dropout_rate"] > 0.01:
        ops.append("g5")
    if p["ghost_ratio"] > 0.01:
        ops.append("g16" if p.get("strategic_ghost", 0) > 0.5 else "g6")
    if p["cluster_strength"] > 0.1:
        ops.append("g10")
    if p.get("edge_attack_strength", 0) > 0.1:
        ops.append("g13")
    return ops


def ristretto(G, op):
    g = np.full(17, -1.0)
    g[GENI[op]] = G[GENI[op]]
    if op == "g6":
        g[16] = -1.0
    return g


def toccati_A(c, out, op):
    if op == "g5":
        return len(c) - len(out)
    if op in ("g6", "g16"):
        return len(out) - len(c)
    assert len(out) == len(c)
    return int(np.any(out[:, :3] != c[:, :3], axis=1).sum())


z = np.load(npz)
order = np.argsort(z["t_arr_mono"], kind="stable")
fr = z["finestra_registro"][order]
metri_idx = [order[fr == w] for w in np.unique(fr[fr >= 0])]
off, xyzi, stamps = z["offsets"], z["xyzi"], z["stamps"]
metri = [int(m) for m in metri_s.split(",")]
n_nuv = int(n_nuv)
ris = []
for gi in json.load(open(genomi_json)):
    G = np.asarray(gi["genoma"], dtype=np.float64)
    seme = genome_seed(G) % (2**31 - 1)
    for op in operatori(PerturbationGenerator(**KW).encode_perturbation(G)):
        g = ristretto(G, op)
        for w in metri:
            a = PerturbationGenerator(**KW)
            b = PerturbationGenerator(**INS, **KW)
            pa, pb = a.encode_perturbation(g), b.encode_perturbation(g)
            idx = metri_idx[w] if n_nuv <= 0 else metri_idx[w][:n_nuv]
            for k, i in enumerate(idx):
                c = xyzi[off[i]:off[i + 1]].astype(np.float32).astype(np.float64)
                oa = a.apply_perturbation(c, pa, seed=(seme + k) % (2**31 - 1))
                b.apply_perturbation(c, pb, ctx={"stamp": float(stamps[i])}, seme_genoma=seme, metro=w)
                ris.append({"genoma": gi["id"], "op": op, "metro": w, "k": k, "n": len(c),
                            "A": toccati_A(c, oa, op), "I": int(b.last_struct_info["n_hit"].get(op, 0)),
                            "per_op": b.last_struct_info["per_op"]})
json.dump({"npz": npz, "righe": ris}, open(uscita, "w"))
print(len(ris), "righe")
