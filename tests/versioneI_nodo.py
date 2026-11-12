#!/usr/bin/env python3
"""Real perturbation node (no spin, ROS_DOMAIN_ID from the environment) on recorded clouds:
calls _cloud_cb as the executor would and reads the outputs from the publish queue. Used by
the version I tests (test_versioneI.py).

Usage:
  versioneI_nodo.py <root> <npz> <output.json> <gemello|vittima|entrambi>
                    <genomes.json> <metres, e.g. 0,2,5> <clouds per metre, 0 = all>
                    [--metro-scarto N] -- <node arguments>

For each genome and metre: metre to the node (only with --operator-mode insieme, value =
metre + N), genome, then the first clouds of the metre. JSON output per cloud: md5 of the
outputs, chosen zones, hit points and _cloud_cb time.
"""
import hashlib
import json
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np

import os  # noqa: E402
if os.environ.get("ROS_DOMAIN_ID") != "88":
    # real node: outside domain 88 it would land on the bench topics (domain 0)
    sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: i nodi dei test girano solo nel dominio 88, non parto")

sep = sys.argv.index("--")
pos, node_args = sys.argv[1:sep], sys.argv[sep + 1:]
scarto = 0
if "--metro-scarto" in pos:
    i = pos.index("--metro-scarto")
    scarto = int(pos[i + 1])
    del pos[i:i + 2]
radice, npz, uscita, flusso, genomi_json, metri_s, n_nuv = pos
sys.path.insert(0, radice)
import src.nodes.perturbation_node as pn  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

assert pn.__file__.startswith(radice), pn.__file__


class _Cli(Exception):
    pass


def _cattura(cli):
    raise _Cli(cli)


# arguments read by the node's own parser (main stops before creating the node)
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
node._out_q = queue.Queue()      # the publishing thread stays blocked on the old queue
insieme = getattr(cli, "operator_mode", "random") == "insieme"

z = np.load(npz)
order = np.argsort(z["t_arr_mono"], kind="stable")
fr = z["finestra_registro"][order]
metri_idx = [order[fr == w] for w in np.unique(fr[fr >= 0])]
off, xyzi, stamps = z["offsets"], z["xyzi"], z["stamps"]
genomi = json.load(open(genomi_json))
metri = [int(m) for m in metri_s.split(",")]
n_nuv = int(n_nuv)


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


def passo(stream, gid, g, w):
    righe = []
    if stream == "gemello":
        node.victim_stream = False
        node.enabled = True
        if insieme:
            node._metro_cb(SimpleNamespace(data=json.dumps({"gemello": w + scarto})))
        node._set_genome(np.asarray(g, dtype=np.float64), source="prova")
        gen = node.generator
    else:
        node.victim_stream = True
        node.victim_attack = True
        if insieme:
            node._metro_cb(SimpleNamespace(data=json.dumps({"vittima": w + scarto})))
        node._victim_genome_cb(SimpleNamespace(data=list(g)))
        gen = node.victim_generator
    idx = metri_idx[w] if n_nuv <= 0 else metri_idx[w][:n_nuv]
    for k, i in enumerate(idx):
        c = xyzi[off[i]:off[i + 1]].astype(np.float32).astype(np.float64)
        m = msg_di(c, float(stamps[i]))
        t0 = time.perf_counter()
        node._cloud_cb(m)
        dt = (time.perf_counter() - t0) * 1000.0
        item = node._out_q.get_nowait()
        out, victim_out = item[2], item[4]
        inf = getattr(gen, "last_struct_info", None) if insieme else None
        righe.append({
            "genoma": gid, "metro": w, "k": k, "stream": stream,
            "md5_main": md5(out), "md5_victim": md5(victim_out),
            "uscita": md5(victim_out if stream == "vittima" else out),
            "anchors": None if inf is None else json.dumps(inf.get("anchors"), sort_keys=True),
            "per_op": None if inf is None else inf.get("per_op"),
            "n_hit": None if inf is None else inf.get("n_hit"),
            "proc_ms": node.proc_times[-1] * 1000.0, "cb_ms": dt,
        })
    return righe


stato0 = np.random.get_state()[1][:8].tolist()
ris = []
streams = ["gemello", "vittima"] if flusso == "entrambi" else [flusso]
for gi in genomi:
    for w in metri:
        for s in streams:
            ris += passo(s, gi["id"], gi["genoma"], w)
json.dump({"radice": radice, "npz": npz, "args": node_args, "righe": ris,
           "insieme": insieme}, open(uscita, "w"))
print(len(ris), "nuvole")
node.destroy_node()
