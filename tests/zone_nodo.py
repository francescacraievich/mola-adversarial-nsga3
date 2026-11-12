#!/usr/bin/env python3
"""Real perturbation node (no spin, ROS_DOMAIN_ID from the environment) on recorded clouds,
for the attacker-zone tests (test_zone_attaccante.py). Calls the callbacks as the executor
would and reads the outputs from the publish queue; the orchestrator part (snapshot and
zones per genome) uses the same functions as zone_attaccante.

Usage:
  zone_nodo.py <root> <npz> <folder> <output.json> <mode> <genomes.json>
               <metres, e.g. 0,2,5> <clouds per flow, 0 = all> -- <node arguments>

mode:
  casi    per metre and genome: snapshot (last cloud of the previous segment and
          pose logged with the same stamp), zones, twin with the first cloud
          normal, removed, changed and moved, victim with the winner
  errori  missing zones, wrong hash or sha256, corrupt or missing file, different
          genome, missing snapshot, missing snapshot cloud
  tempi   all clouds of the metres, twin and victim, at 10 Hz (time per cloud
          with the measurement thread on)
  spento  outputs only (node without zones), for bit-exact comparisons
  deriva_keep, deriva_genome
          time drift of twin and victim at the start and end of each
          rollout, with /attack/drift_keep on or off
"""
import hashlib
import json
import queue
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import os  # noqa: E402
if os.environ.get("ROS_DOMAIN_ID") != "88":
    # real node: outside domain 88 it would land on the bench topics (domain 0)
    sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: i nodi dei test girano solo nel dominio 88, non parto")

sep = sys.argv.index("--")
pos, node_args = sys.argv[1:sep], sys.argv[sep + 1:]
radice, npz, cartella, uscita, modo, genomi_json, metri_s, n_nuv = pos
sys.path.insert(0, radice)
import src.nodes.perturbation_node as pn  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

assert pn.__file__.startswith(radice), pn.__file__
ZONE_ON = "--zone-attaccante" in node_args
if ZONE_ON or modo != "spento":
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
node._out_q = queue.Queue()      # the publishing thread stays blocked on the old queue
INSIEME = getattr(cli, "operator_mode", "random") == "insieme"

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
    """Snapshot of metre w as the orchestrator takes it: last attacker pose (logged) and
    the cloud with the same stamp, saved by the node from its buffer of clouds sent to
    the attacker.
    """
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
    return za.carica_fotografia(path, rec["sha256"], rec["hash"]), int(i_foto)


def manda_zone(flusso, zz, **sovr):
    d = {"seq": seq(), "flusso": flusso, "path": str(zz.path), "sha256": zz.sha256, "hash": zz.hash,
         "genoma_sha": zz.genoma_sha, "seme": zz.seme, "metro": zz.metro}
    d.update(sovr)
    node._zone_cb(SimpleNamespace(data=json.dumps(d)))
    return dict(node._zone_ric[flusso])


def flusso(stream, g, w, nuvole, ritmo=0.0):
    """One flow rollout: metre, genome, clouds; rows per cloud."""
    g32 = np.asarray(g, dtype=np.float64).astype(np.float32).astype(np.float64)
    if stream == "gemello":
        node.victim_stream = False
        node.enabled = True
        if INSIEME:
            node._metro_cb(SimpleNamespace(data=json.dumps({"gemello": w})))
        node._set_genome(g32, source="prova")
        gen, params = node.generator, node.params
    else:
        node.victim_stream = True
        node.victim_attack = True
        if INSIEME:
            node._metro_cb(SimpleNamespace(data=json.dumps({"vittima": w})))
        node._victim_genome_cb(SimpleNamespace(data=list(g32)))
        gen, params = node.victim_generator, node.victim_params
    righe = []
    for c, s in nuvole:
        t0 = time.perf_counter()
        node._cloud_cb(msg_di(c, s))
        dt = time.perf_counter() - t0
        item = node._out_q.get_nowait()
        out, victim_out = item[2], item[4]
        o = victim_out if stream == "vittima" else out
        inf = (gen.last_struct_info or {}) if INSIEME else {}
        nh = inf.get("n_hit") or {}
        n = len(c)
        n_rm = int(nh.get("g5", 0))
        righe.append({
            "stream": stream, "metro": w, "n_in": n, "n_out": None if o is None else len(o),
            "uscita": md5(o), "zone_stato": inf.get("zone_stato"), "zone_hash": inf.get("zone_hash"),
            "k": inf.get("k"), "anchors": json.dumps(inf.get("anchors"), sort_keys=True) if inf else None,
            "n_hit": nh, "proc_ms": node.proc_times[-1] * 1000.0,
            "zone_attesi": inf.get("zone_attesi"), "zone_scarto": inf.get("zone_scarto"),
            # random-mode caps per cloud (g5, ghosts) and snapshot targets (g10, g13)
            "tetto": {"g5": int(n * params["dropout_rate"]),
                      "g6": int((n - n_rm) * params["ghost_ratio"]),
                      "g16": int((n - n_rm) * params["ghost_ratio"]),
                      "g10": (gen._obiettivi or {}).get("g10", [None])[0] if INSIEME else None,
                      "g13": (gen._obiettivi or {}).get("g13", [None])[0] if INSIEME else None},
        })
        if ritmo > 0:
            time.sleep(max(0.0, ritmo - dt))
    return righe


def varianti(idx):
    """First rollout cloud: normal, removed, changed (1 cm in x and one point in ten
    removed), moved (swapped with the second).
    """
    nv = [nuvola(i) for i in idx]
    c0, s0 = nv[0]
    cam = c0.copy()
    cam[:, 0] += 0.01
    cam = cam[np.arange(len(cam)) % 10 != 0]
    return {"normale": nv[:-1], "tolta": nv[1:], "cambiata": [(cam, s0)] + nv[1:-1],
            "spostata": [nv[1], nv[0]] + nv[2:-1]}


ris = {"radice": radice, "npz": npz, "args": node_args, "modo": modo, "casi": [], "errori": [],
       "righe": [], "foto": {}}

if modo in ("casi", "tempi", "spento"):
    for w in metri:
        idx = gruppi[w] if n_nuv <= 0 else gruppi[w][:n_nuv + 1]
        foto = None
        if ZONE_ON:
            foto, i_foto = fotografia(w)
            ris["foto"][w] = {"i": i_foto, "sha256": foto.sha256, "hash": foto.hash,
                              "n": foto.meta["n_punti"]}
        for gi in genomi:
            g = gi["genoma"]
            zz = None
            if ZONE_ON:
                zz = za.zone_genoma(DIR / "zone", foto, g, w, LIMITI)
                # second computation from a re-read snapshot: same hash
                foto2 = za.carica_fotografia(foto.path, foto.sha256, foto.hash)
                m2, a2, _, _ = za.calcola_zone(foto2, g, w, LIMITI)
                h2 = za.hash_contenuto(m2, a2)
            if modo == "casi":
                vv = varianti(idx)
                caso = {"genoma": gi["id"], "metro": w, "hash_file": zz.hash if zz else None,
                        "hash_ricalcolo": h2 if zz else None, "rollout": {}}
                for nome, nuv in vv.items():
                    if zz:
                        manda_zone("gemello", zz)
                    righe = flusso("gemello", g, w, nuv)
                    caso["rollout"][nome] = righe
                # victim with the winner: zones from the genome file, re-read like the orchestrator
                for nome in ("normale", "tolta"):
                    if zz:
                        zv = za.zone_genoma(DIR / "zone", foto, g, w, LIMITI)
                        caso["hash_vittima_file"] = zv.hash
                        manda_zone("vittima", zv)
                    caso["rollout"]["vittima_" + nome] = flusso("vittima", g, w, vv[nome])
                ris["casi"].append(caso)
            else:
                nv = [nuvola(i) for i in idx]
                for stream in ("gemello", "vittima"):
                    if zz:
                        manda_zone(stream, zz)
                    ris["righe"] += flusso(stream, g, w, nv, ritmo=0.1 if modo == "tempi" else 0.0)

if modo == "errori":
    w = metri[0]
    idx = gruppi[w][:4]
    nv = [nuvola(i) for i in idx]
    foto, _ = fotografia(w)
    gA, gB = genomi[0]["genoma"], genomi[1]["genoma"]
    zA = za.zone_genoma(DIR / "zone", foto, gA, w, LIMITI)
    rotto = DIR / "zone" / f"w{w}_rotto.npz"
    data = bytearray(Path(zA.path).read_bytes())
    data[len(data) // 2] ^= 0xFF
    rotto.write_bytes(bytes(data))
    altrove = DIR / "senza_foto"
    altrove.mkdir(parents=True, exist_ok=True)
    shutil.copy(zA.path, altrove / Path(zA.path).name)

    def prova(nome, stream, g, msg_sovr=None, manda=True, svuota=False):
        if svuota:
            node._foto_cache.clear()     # snapshot not yet read by the node
        app0 = dict(node._zone_app[stream])
        ric = manda_zone(stream, zA, **(msg_sovr or {})) if manda else None
        righe = flusso(stream, g, w, nv)
        app1 = dict(node._zone_app[stream])
        ris["errori"].append({"caso": nome, "stream": stream, "ricevute": ric,
                              "stati": [r["zone_stato"] for r in righe],
                              "hash": [r["zone_hash"] for r in righe],
                              "n_hit": [r["n_hit"] for r in righe],
                              "n_err": app1["n_err"] - app0["n_err"], "n_ok": app1["n_ok"] - app0["n_ok"]})

    for stream in ("gemello", "vittima"):
        prova("assenti", stream, gA, manda=False)
        prova("controllo_valide", stream, gA)
        prova("hash_sbagliato", stream, gA, {"hash": "0" * 64})
        prova("sha256_sbagliato", stream, gA, {"sha256": "0" * 64})
        prova("file_corrotto", stream, gA, {"path": str(rotto),
                                            "sha256": hashlib.sha256(bytes(data)).hexdigest()})
        prova("file_assente", stream, gA, {"path": str(DIR / "zone" / "nessuno.npz")})
        prova("fotografia_assente", stream, gA, {"path": str(altrove / Path(zA.path).name)}, svuota=True)
        prova("genoma_diverso", stream, gB)
    # snapshot with a stamp not in the buffer
    q = seq()
    node._foto_cb(SimpleNamespace(data=json.dumps({"seq": q, "stamp": -1.0,
                                                   "path": str(DIR / "zone" / "w99_fotografia.npz")})))
    ris["foto_assente"] = dict(node._foto_stato)
    ris["foto_99_esiste"] = (DIR / "zone" / "w99_fotografia.npz").exists()

if modo in ("deriva_keep", "deriva_genome"):
    # Drift (gene 14) across metres: first genome = victim of the first metre, the
    # others = search candidates of each later metre, winner the last one.
    # Starting drift logged right after the genome, before the cloud.
    node._drift_keep_cb(SimpleNamespace(data=modo == "deriva_keep"))
    INIZIO = {}
    _set, _vcb = node._set_genome, node._victim_genome_cb

    def _set_reg(g, source):
        _set(g, source=source)
        INIZIO["gemello"] = node.generator.accumulated_drift.tolist()

    def _vcb_reg(msg):
        _vcb(msg)
        INIZIO["vittima"] = node.victim_generator.accumulated_drift.tolist()
    node._set_genome, node._victim_genome_cb = _set_reg, _vcb_reg
    ris["deriva"] = []
    for j, w in enumerate(metri):
        nv = [nuvola(i) for i in (gruppi[w] if n_nuv <= 0 else gruppi[w][:n_nuv])]
        foto = fotografia(w)[0] if ZONE_ON else None
        rec = {"metro": w, "vittima_fine_prec": node.victim_generator.accumulated_drift.tolist(),
               "gemelli": []}
        cand = genomi[1:] if j > 0 else []
        for gi in cand:
            if ZONE_ON:
                manda_zone("gemello", za.zone_genoma(DIR / "zone", foto, gi["genoma"], w, LIMITI))
            r = flusso("gemello", gi["genoma"], w, nv)
            rec["gemelli"].append({"id": gi["id"], "inizio": INIZIO["gemello"],
                                   "fine": node.generator.accumulated_drift.tolist(),
                                   "uscite": [x["uscita"] for x in r]})
        gv = cand[-1]["genoma"] if cand else genomi[0]["genoma"]
        if ZONE_ON:
            manda_zone("vittima", za.zone_genoma(DIR / "zone", foto, gv, w, LIMITI))
        r = flusso("vittima", gv, w, nv)
        rec["vittima"] = {"inizio": INIZIO["vittima"], "fine": node.victim_generator.accumulated_drift.tolist(),
                          "uscite": [x["uscita"] for x in r]}
        ris["deriva"].append(rec)

json.dump(ris, open(uscita, "w"))
node.destroy_node()
print("fatto")
