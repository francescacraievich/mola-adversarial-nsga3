#!/usr/bin/env python3
"""Real perturbation node (no spin, ROS_DOMAIN_ID 88) on recorded clouds, for the tests of
the limit multiplier (test_moltiplicatore_limiti.py). As in tests/zone_nodo.py, callbacks
are called as the executor would and outputs are read from the publish queue; snapshot
and per-genome zones come from zone_attaccante, with the node's limits and multiplier
(as limiti_zone in the orchestrator). Also works with the code before the multiplier.

Usage:
  limiti_nodo.py <root> <npz> <folder> <out.json> <genomes.json>
                 <meters, e.g. 0,2> <clouds per stream> <keep 0|1> -- <node arguments>

Per meter and genome: snapshot (with --zone-attaccante), zones, twin and then victim on
the same clouds. Per cloud: md5 of the output, input and output points, counts of the last
apply_perturbation, accumulated drift. At the end: node status and rows of the zone log
(--struct-log), if requested.
"""
import csv
import hashlib
import json
import os
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if os.environ.get("ROS_DOMAIN_ID") != "88":
    sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: i nodi dei test girano solo nel dominio 88, non parto")

sep = sys.argv.index("--")
pos, node_args = sys.argv[1:sep], sys.argv[sep + 1:]
radice, npz, cartella, uscita, genomi_json, metri_s, n_nuv, keep = pos
sys.path.insert(0, radice)
import src.nodes.perturbation_node as pn  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

assert pn.__file__.startswith(radice), pn.__file__
ZONE_ON = "--zone-attaccante" in node_args
from src.perturbations import zone_attaccante as za  # noqa: E402


class _Cli(Exception):
    pass


def _cattura(cli):
    raise _Cli(cli)


_orig = pn.PerturbationNode
pn.PerturbationNode = _cattura
sys.argv = ["perturbation_node.py"] + node_args
try:
    pn.main()
    raise SystemExit("main non ha creato il nodo")
except _Cli as e:
    cli = e.args[0]
pn.PerturbationNode = _orig
node = _orig(cli)
node._out_q = queue.Queue()      # the publisher thread stays blocked on the old queue
INSIEME = getattr(cli, "operator_mode", "random") == "insieme"
if keep == "1":
    node._drift_keep_cb(SimpleNamespace(data=True))

z = np.load(npz)
order = np.argsort(z["t_arr_mono"], kind="stable")
fr = z["finestra_registro"][order]
gruppi = {int(w): order[fr == w] for w in np.unique(fr)}       # -1 = settling before w0
off, xyzi, stamps, att = z["offsets"], z["xyzi"], z["stamps"], z["att"]
pose_per_stamp = {round(float(r[0]), 6): (float(r[1]), float(r[2]), float(r[3])) for r in att}
genomi = json.load(open(genomi_json))
metri = [int(m) for m in metri_s.split(",")]
n_nuv = int(n_nuv)
DIR = Path(cartella)
SEQ = [0]
LIMITI = {k: float(getattr(node.generator, k)) for k in ("max_point_shift", "noise_std",
                                                             "max_dropout_rate", "max_ghost_points_ratio")}
MOL = getattr(node.generator, "moltiplicatori_attivi", lambda: None)()
if MOL:
    # as limiti_zone in the orchestrator: node limits times its multiplier
    LIMITI["moltiplicatori"] = dict(MOL)


def nuvola(i):
    return xyzi[off[i]:off[i + 1]].astype(np.float32).astype(np.float64), float(stamps[i])


def msg_di(c, s):
    h = Header()
    h.frame_id = "base_link"
    h.stamp.sec = int(s)
    h.stamp.nanosec = int(round((s - int(s)) * 1e9))
    m = PointCloud2()
    m.header = h
    m.height, m.width = 1, len(c)
    m.fields = pn.FIELDS
    m.is_bigendian = False
    m.point_step = pn.POINT_STEP
    m.row_step = pn.POINT_STEP * len(c)
    m.data = np.ascontiguousarray(c, dtype="<f4").tobytes()
    m.is_dense = True
    return m


def md5(a):
    return None if a is None else hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def seq():
    SEQ[0] += 1
    return SEQ[0]


def fotografia(w):
    """Snapshot of meter w as the orchestrator takes it (zone_nodo.py)."""
    prec = gruppi[w - 1] if (w - 1) in gruppi else gruppi[-1]
    cand = [i for i in prec if round(float(stamps[i]), 6) in pose_per_stamp]
    i_foto = cand[-1]
    node._att_buf.clear()
    for i in prec[-5:]:
        c, s = nuvola(i)
        node._att_buf.append((pn._stamp_s(msg_di(c, s).header), c))
    s = pn._stamp_s(msg_di(*nuvola(i_foto)).header)
    path = DIR / "zone" / f"w{w}_fotografia.npz"
    q = seq()
    node._foto_cb(SimpleNamespace(data=json.dumps({
        "seq": q, "stamp": s, "path": str(path),
        "meta": {"metro": w, "stamp_posa": s, "posa_base_odom": list(pose_per_stamp[round(float(stamps[i_foto]), 6)])}})))
    rec = node._foto_stato
    assert rec["seq"] == q and rec["esito"] == "ok", rec
    return za.carica_fotografia(path, rec["sha256"], rec["hash"])


def manda_zone(flusso, zz):
    d = {"seq": seq(), "flusso": flusso, "path": str(zz.path), "sha256": zz.sha256, "hash": zz.hash,
         "genoma_sha": zz.genoma_sha, "seme": zz.seme, "metro": zz.metro}
    node._zone_cb(SimpleNamespace(data=json.dumps(d)))
    return dict(node._zone_ric[flusso])


def conteggi(gen):
    f = getattr(gen, "ultimi_conteggi", None)
    if f is not None:
        return f()
    lc = getattr(gen, "_last_counts", None)
    return None if lc is None else dict(lc[1])


def flusso(stream, g, w, nuvole):
    g32 = np.asarray(g, dtype=np.float64).astype(np.float32).astype(np.float64)
    if stream == "gemello":
        node.victim_stream = False
        node.enabled = True
        if INSIEME:
            node._metro_cb(SimpleNamespace(data=json.dumps({"gemello": w})))
        node._set_genome(g32, source="prova")
        gen = node.generator
    else:
        node.victim_stream = True
        node.victim_attack = True
        if INSIEME:
            node._metro_cb(SimpleNamespace(data=json.dumps({"vittima": w})))
        node._victim_genome_cb(SimpleNamespace(data=list(g32)))
        gen = node.victim_generator
    righe = []
    for c, s in nuvole:
        node._cloud_cb(msg_di(c, s))
        item = node._out_q.get_nowait()
        o = item[4] if stream == "vittima" else item[2]
        inf = (gen.last_struct_info or {}) if INSIEME else {}
        righe.append({"stream": stream, "metro": w, "n_in": len(c), "n_out": None if o is None else len(o),
                      "uscita": md5(o), "conteggi": conteggi(gen), "n_hit": inf.get("n_hit") or {},
                      "zone_stato": inf.get("zone_stato"), "zone_hash": inf.get("zone_hash"),
                      "deriva_m": float(np.linalg.norm(gen.accumulated_drift))})
    return righe


ris = {"radice": radice, "args": node_args, "keep": keep, "moltiplicatori": MOL, "righe": [], "zone": []}
for w in metri:
    idx = gruppi[w][:n_nuv]
    nv = [nuvola(i) for i in idx]
    foto = fotografia(w) if ZONE_ON else None
    for gi in genomi:
        zz = za.zone_genoma(DIR / "zone", foto, gi["genoma"], w, LIMITI) if ZONE_ON else None
        if zz is not None:
            ris["zone"].append({"genoma": gi["id"], "metro": w, "hash": zz.hash,
                                "moltiplicatori": zz.meta.get("moltiplicatori"),
                                "obiettivi": zz.meta.get("obiettivi"), "quota_g5": zz.meta.get("quota_g5")})
        for stream in ("gemello", "vittima"):
            if zz is not None:
                ric = manda_zone(stream, zz)
                assert ric["stato"] == "ok", (stream, ric)
            for r in flusso(stream, gi["genoma"], w, nv):
                r["genoma"] = gi["id"]
                ris["righe"].append(r)

# node status as published (latency is filled by the publisher thread, which is
# stopped here: a dummy value only to get the status out)
if not node.lat_times:
    node.lat_times.append(0.0)
pubblicati = []
node.status_pub = SimpleNamespace(publish=lambda m: pubblicati.append(m.data))
node._publish_status()
ris["status"] = json.loads(pubblicati[-1]) if pubblicati else None
if getattr(node, "_struct_log", None) is not None:
    node._struct_log.close()
    with open(node._struct_log.path, newline="") as f:
        ris["struct_log"] = list(csv.DictReader(f))
    ris["struct_log_colonne"] = list(node._struct_log.columns)
Path(uscita).write_text(json.dumps(ris))
print(f"righe {len(ris['righe'])}, zone {len(ris['zone'])}")
