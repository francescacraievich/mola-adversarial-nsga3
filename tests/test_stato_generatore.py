#!/usr/bin/env python3
"""Generator state for the realigned twin (node --stato-generatore; generator stato_flusso,
imposta_stato_flusso and impronta_stato). Interface: /attack/stato_generatore with "salva"
(flow, slot) and "carica" (slot, for the next twin genome), key stato_generatore in the status.

No Isaac or MOLA: synthetic scene and genomes of test_rampa_genoma.py; the real node runs in
a separate process (this file with --figlio, domain 88, no spin). Key test: save from the
victim -> load into the twin -> the twin clouds are those the victim produces from the next
cloud on, also halfway through the ramp.

    ROS_DOMAIN_ID=88 python3 -B -m pytest -q -p no:cacheprovider tests/test_stato_generatore.py
"""

import json
import os
import queue
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

QUI = Path(__file__).resolve().parent
ROOT = QUI.parent
sys.path.insert(0, str(QUI))

import test_rampa_genoma as tr  # noqa: E402  (scene, genomes, messages; no import from src)

R = tr.R_PROVA
STG = ["--stato-generatore"]


# ----------------------------------------------------------------------
# Child process
# ----------------------------------------------------------------------

def _figlio(argv):
    sep = argv.index("--")
    radice, scenario, uscita = argv[:sep]
    node_args = list(argv[sep + 1:])
    if os.environ.get("ROS_DOMAIN_ID") != "88":
        sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: il nodo dei test gira solo nel dominio 88")
    sys.path.insert(0, radice)
    import src.nodes.perturbation_node as pn
    from src.perturbations import zone_attaccante as za
    from std_msgs.msg import Bool, String
    import rclpy
    assert pn.__file__.startswith(radice), pn.__file__
    lavoro = Path(uscita).with_suffix("")
    zd = lavoro / "zone"
    zd.mkdir(parents=True, exist_ok=True)

    class _Cli(Exception):
        pass

    def _cattura(cli):
        raise _Cli(cli)

    orig = pn.PerturbationNode
    pn.PerturbationNode = _cattura
    sys.argv = ["perturbation_node.py"] + node_args
    try:
        pn.main()
        raise SystemExit("main non ha creato il nodo")
    except _Cli as e:
        cli = e.args[0]
    pn.PerturbationNode = orig
    node = orig(cli)
    node._out_q = queue.Queue()
    seqn = [0]

    def seq():
        seqn[0] += 1
        return seqn[0]

    class _P:
        def publish(self, m):
            self.m = m

    def stato():
        node.status_pub = _P()
        if not node.lat_times:
            node.lat_times.append(0.01)
        pn.PerturbationNode._publish_status_body(node)
        return json.loads(node.status_pub.m.data.replace(str(lavoro), "<DIR>"))

    def nuvola(flusso, c, s):
        if flusso == "gemello":
            node.victim_stream, node.enabled = False, True
        else:
            node.victim_stream, node.victim_attack = True, True
        # the callback as subscribed (with the option: under the generator lock)
        cb = node.sub.callback
        cb(tr._msg(pn, c, s))
        item = node._out_q.get_nowait()
        o = item[4] if flusso == "vittima" else item[2]
        gen = node.victim_generator if flusso == "vittima" else node.generator
        inf = gen.last_struct_info or {}
        return {"flusso": flusso, "md5": tr._md5(o), "k": inf.get("k"),
                "deriva": [float(x) for x in gen.accumulated_drift],
                "rampa": None if getattr(gen, "rampa_info", None) is None else dict(gen.rampa_info)}

    def manda_zone(flusso, z):
        node._zone_cb(String(data=json.dumps({"seq": seq(), "flusso": flusso, "path": str(z.path),
                                              "sha256": z.sha256, "hash": z.hash,
                                              "genoma_sha": z.genoma_sha})))

    def manda_rampa(flusso, g_prec, sha_nuovo, z_prec):
        node._rampa_cb(String(data=json.dumps({
            "seq": seq(), "flusso": flusso, "genoma_prec": [float(x) for x in g_prec],
            "genoma_sha_nuovo": sha_nuovo,
            "zone_prec": {"path": str(z_prec.path), "sha256": z_prec.sha256, "hash": z_prec.hash}})))

    def stg(**d):
        d.setdefault("seq", seq())
        node._stato_gen_cb(String(data=json.dumps(d)))
        return d["seq"], stato().get("stato_generatore")

    def metro(flusso, w):
        node._metro_cb(String(data=json.dumps({flusso: w})))

    rampa = "--rampa-genoma" in node_args
    node._drift_keep_cb(Bool(data=True))
    s_foto = 9.0
    node._att_buf.append((pn._stamp_s(tr._msg(pn, tr.scena(-1), s_foto).header), tr.scena(-1)))
    node._foto_cb(String(data=json.dumps({
        "seq": 1, "stamp": pn._stamp_s(tr._msg(pn, tr.scena(-1), s_foto).header),
        "path": str(zd / "w1_fotografia.npz"), "meta": {"metro": 1, "posa_base_odom": [0.0, 0.0, 0.0]}})))
    fs = node._foto_stato
    foto = za.carica_fotografia(zd / "w1_fotografia.npz", fs["sha256"], fs["hash"])
    limiti = {k: float(getattr(node.generator, k)) for k in za.LIMITI}
    zP, zN = (za.zone_genoma(zd, foto, g, 1, limiti) for g in (tr.GP, tr.GN))
    sha_n = za.genoma_sha(tr.GN)
    out = {"righe": {}, "stg": {}}
    # victim: P for 3 clouds, then N (with the ramp if on) for n_prima clouds
    metro("vittima", 1)
    manda_zone("vittima", zP)
    node._victim_genome_cb(SimpleNamespace(data=list(tr.GP)))
    for i in range(3):
        nuvola("vittima", tr.scena(100 + i), 1.0 + 0.1 * i)
    if rampa:
        manda_rampa("vittima", tr.GP, sha_n, zP)
    manda_zone("vittima", zN)
    node._victim_genome_cb(SimpleNamespace(data=list(tr.GN)))
    n_prima = 2                                      # k = 0, 1: halfway through the ramp with R = 4
    out["righe"]["vittima_prima"] = [nuvola("vittima", tr.scena(i), 2.0 + 0.1 * i) for i in range(n_prima)]
    if scenario == "senza_opzione":
        # realigned without transfer: victim genome to the twin, as without the option
        out["righe"]["vittima_dopo"] = [nuvola("vittima", tr.scena(i), 2.0 + 0.1 * i)
                                        for i in range(n_prima, n_prima + 4)]
        metro("gemello", 1)
        manda_zone("gemello", zN)
        node._set_genome(tr.GN, source="prova")
        # same clouds and stamps as the victim (identical input: the tracker
        # predicts the pose from the ratio of stamp intervals)
        out["righe"]["gemello"] = [nuvola("gemello", tr.scena(i), 2.0 + 0.1 * i)
                                   for i in range(n_prima, n_prima + 4)]
    else:
        out["stg"]["salva"] = stg(azione="salva", nome="riallineo", flusso="vittima")
        out["stg"]["salva_flusso_errato"] = stg(azione="salva", nome="x", flusso="ombra")
        out["stg"]["carica_assente"] = stg(azione="carica", nome="nessuno")
        # the victim continues from the next cloud: the attack the realigned twin must reproduce
        out["righe"]["vittima_dopo"] = [nuvola("vittima", tr.scena(i), 2.0 + 0.1 * i)
                                        for i in range(n_prima, n_prima + 4)]
        out["posto_dopo_vittima"] = stato()["stato_generatore"]["posti"]["riallineo"]
        # load with a different genome: nothing loaded, twin as without the option
        out["stg"]["carica_1"] = stg(azione="carica", nome="riallineo")
        metro("gemello", 1)
        manda_zone("gemello", zP)
        node._set_genome(tr.GP, source="prova")
        out["dopo_genoma_diverso"] = stato()["stato_generatore"]
        out["righe"]["gemello_diverso"] = [nuvola("gemello", tr.scena(i), 3.0 + 0.1 * i) for i in range(2)]
        # load with the victim genome: the twin takes over the victim state
        out["stg"]["carica_2"] = stg(azione="carica", nome="riallineo")
        out["stg"]["carica_2_ripetuto"] = stg(azione="carica", nome="riallineo",
                                              seq=out["stg"]["carica_2"][0])
        metro("gemello", 1)
        manda_zone("gemello", zN)
        node._set_genome(tr.GN, source="prova")
        out["dopo_carica"] = stato()["stato_generatore"]
        # same clouds and stamps as the victim (identical input: the tracker
        # predicts the pose from the ratio of stamp intervals)
        out["righe"]["gemello"] = [nuvola("gemello", tr.scena(i), 2.0 + 0.1 * i)
                                   for i in range(n_prima, n_prima + 4)]
        # a second twin genome without load: the usual reset
        node._set_genome(tr.GN, source="prova")
        out["righe"]["gemello_dopo"] = [nuvola("gemello", tr.scena(i), 3.0 + 0.1 * i) for i in range(2)]
        out["finale"] = stato()["stato_generatore"]
    out["status_chiavi"] = sorted(stato().keys())
    json.dump(out, open(uscita, "w"))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__" and "--figlio" in sys.argv:
    _figlio(sys.argv[sys.argv.index("--figlio") + 1:])
    sys.exit(0)


# ----------------------------------------------------------------------
# Tests (pytest)
# ----------------------------------------------------------------------

import pytest  # noqa: E402


def _nodo_figlio(tmp_path, nome, radice, scenario, args):
    out = tmp_path / f"{nome}.json"
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--figlio", str(radice), scenario, str(out),
           "--"] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, ROS_DOMAIN_ID="88"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def test_argomenti_e_sottoscrizioni(monkeypatch):
    pn = pytest.importorskip("src.nodes.perturbation_node")
    rclpy = pytest.importorskip("rclpy")

    class _Preso(Exception):
        pass

    def cli(argv):
        def nodo(c):
            raise _Preso(c)
        with monkeypatch.context() as m:
            m.setattr(pn.rclpy, "init", lambda **k: None)
            m.setattr(pn, "PerturbationNode", nodo)
            m.setattr(sys, "argv", ["perturbation_node.py"] + argv)
            with pytest.raises(_Preso) as e:
                pn.main()
        return e.value.args[0]

    assert cli([]).stato_generatore is False and cli(STG).stato_generatore is True
    assert pn.PerturbationNode._stato_gen_on is False
    rclpy.init()
    try:
        topic, tipo = {}, {}
        for nome, argv in (("spento", tr.NODO), ("acceso", tr.NODO + STG), ("random", STG)):
            n = pn.PerturbationNode(cli(argv))
            topic[nome] = sorted(s.topic_name for s in n.subscriptions)
            tipo[nome] = [s._callback_type for s in n.subscriptions if s.topic_name == "/carter/lidar_with_intensity"]
            n.destroy_node()
        assert "/attack/stato_generatore" not in topic["spento"]
        assert topic["acceso"] == sorted(topic["spento"] + ["/attack/stato_generatore"])
        assert tipo["acceso"] == tipo["spento"]          # same shape as the cloud callback
    finally:
        rclpy.shutdown()


def test_generatore_trasferito_prosegue_identico(Z):
    """Transferred state continues identically: outputs, drift and ramp state; slot unchanged."""
    pg = Z["pg"]
    a = tr._gen_rampa(Z, R)
    for i in range(2):
        tr._passo(a, Z["pN"], Z["sN"], i)
    posto = a.stato_flusso()
    imp = pg.impronta_stato(posto)
    assert imp == pg.impronta_stato(a.stato_flusso(copia=False))
    b = tr._gen(Z, Z["zP"], deriva=(0.3, -0.2, 0.0))       # other genome, drift, zones
    tr._passo(b, Z["pP"], Z["sP"], 50)
    b.imposta_stato_flusso(posto)
    assert pg.impronta_stato(b.stato_flusso(copia=False)) == imp
    for i in range(2, R + 3):
        x, y = tr._passo(a, Z["pN"], Z["sN"], i), tr._passo(b, Z["pN"], Z["sN"], i)
        tr._uguali(x, y)
        assert x["n_hit"] == y["n_hit"] and x["rampa"] == y["rampa"] and x["k"] == y["k"] == i
    assert pg.impronta_stato(posto) == imp                 # the slot stays as saved
    c = tr._gen(Z, Z["zN"])
    c.imposta_stato_flusso(posto)                          # a second load from the same slot
    a2 = tr._gen_rampa(Z, R)
    for i in range(2):
        tr._passo(a2, Z["pN"], Z["sN"], i)
    for i in range(2, 4):
        tr._uguali(tr._passo(a2, Z["pN"], Z["sN"], i), tr._passo(c, Z["pN"], Z["sN"], i))


@pytest.fixture(scope="module")
def Z(tmp_path_factory):
    """Snapshot of metre 1, zones of P and N, parameters and seeds (as in test_rampa_genoma)."""
    pg, za = tr._moduli()
    d = tmp_path_factory.mktemp("stg_zone")
    s1 = za.salva_fotografia(d / "w1_fotografia.npz", tr.scena(-1), {"metro": 1, "posa_base_odom": [0, 0, 0]})
    f1 = za.carica_fotografia(d / "w1_fotografia.npz", *s1)
    zz = {nome: za.zone_genoma(d, f1, g, 1, tr.LIM) for nome, g in (("P", tr.GP), ("N", tr.GN))}
    cache = {}
    zone = {nome: za.carica_zone(z.path, z.sha256, z.hash, cache=cache) for nome, z in zz.items()}
    g = pg.PerturbationGenerator(**tr.LIM)
    return {"pg": pg, "za": za, "zP": zone["P"], "zN": zone["N"],
            "pP": g.encode_perturbation(tr.GP), "pN": g.encode_perturbation(tr.GN),
            "sP": za.seme_genoma(tr.GP), "sN": za.seme_genoma(tr.GN),
            "shaP": za.genoma_sha(tr.GP), "shaN": za.genoma_sha(tr.GN)}


@pytest.mark.parametrize("con_rampa", [False, True], ids=["senza_rampa", "rampa"])
def test_nodo_salva_carica_come_la_vittima(tmp_path, con_rampa):
    """Real node: save from the victim, load into the twin, same 4 clouds; seq and slot rules."""
    args = tr.NODO + STG + (["--rampa-genoma", str(R)] if con_rampa else [])
    o = _nodo_figlio(tmp_path, "stg", tr.ROOT, "stg", args)
    v, g = o["righe"]["vittima_dopo"], o["righe"]["gemello"]
    assert [r["md5"] for r in g] == [r["md5"] for r in v]
    assert [r["deriva"] for r in g] == [r["deriva"] for r in v]
    assert [r["k"] for r in g] == [r["k"] for r in v] == [2, 3, 4, 5]
    assert [r["rampa"] for r in g] == [r["rampa"] for r in v]
    if con_rampa:
        assert [r["rampa"]["alpha"] for r in v] == [0.75, 1.0, 1.0, 1.0]
        assert [r["rampa"]["stato"] for r in o["righe"]["vittima_prima"]] == ["attiva", "attiva"]
    s_seq, st = o["stg"]["salva"]
    assert st["seq"] == s_seq
    p = st["posti"]["riallineo"]
    assert p["flusso"] == "vittima" and p["frame_idx"] == 2 and p["k_strutture"] == 1 and p["metro"] == 1
    assert p["drift"] == o["righe"]["vittima_prima"][-1]["deriva"]
    assert o["posto_dopo_vittima"] == p                     # victim continues, slot does not
    # actions not executed: seq not confirmed, reason in ultimo_errore
    for chiave, motivo in (("salva_flusso_errato", "flusso"), ("carica_assente", "posto_assente")):
        q, s2 = o["stg"][chiave]
        assert s2["seq"] == s_seq and s2["ultimo_errore"]["seq"] == q and s2["ultimo_errore"]["motivo"] == motivo
    q1, s1 = o["stg"]["carica_1"]
    assert s1["seq"] == q1 and s1["armato"] == "riallineo"
    d = o["dopo_genoma_diverso"]
    assert d["armato"] is None and d["ultimo_caricato"]["esito"] == "genoma_diverso"
    assert d["ultimo_caricato"]["seq"] == q1 and d["ultimo_caricato"]["impronta"] is None
    assert [r["k"] for r in o["righe"]["gemello_diverso"]] == [0, 1]
    q2, s2 = o["stg"]["carica_2"]
    assert s2["armato"] == "riallineo"
    assert o["stg"]["carica_2_ripetuto"][1]["seq"] == q2      # same seq: no new action
    c = o["dopo_carica"]
    assert c["armato"] is None and c["ultimo_caricato"]["esito"] == "ok" and c["ultimo_caricato"]["seq"] == q2
    assert c["ultimo_caricato"]["impronta"] == p["impronta"] == c["posti"]["riallineo"]["impronta"]
    assert [r["k"] for r in o["righe"]["gemello_dopo"]] == [0, 1]
    assert "stato_generatore" in o["status_chiavi"]


def test_nodo_senza_opzione_riallineato_diverso(tmp_path):
    """Control: without --stato-generatore the realigned twin restarts from k = 0 and differs."""
    o = _nodo_figlio(tmp_path, "senza", tr.ROOT, "senza_opzione", tr.NODO)
    v, g = o["righe"]["vittima_dopo"], o["righe"]["gemello"]
    assert [r["k"] for r in g] == [0, 1, 2, 3] and [r["k"] for r in v] == [2, 3, 4, 5]
    assert all(x["md5"] != y["md5"] for x, y in zip(g, v))
    assert "stato_generatore" not in o["status_chiavi"]


@pytest.mark.skipif(not (tr.ORIGINALE / "src/nodes/perturbation_node.py").exists(), reason="copia originale assente")
def test_opzione_accesa_non_usata_uscite_identiche(tmp_path):
    """--stato-generatore without messages: outputs identical to the untouched copy."""
    a = tr._nodo_figlio(tmp_path, "orig", tr.ORIGINALE, "spento", tr.NODO)
    b = tr._nodo_figlio(tmp_path, "acceso", tr.ROOT, "spento", tr.NODO + STG)
    assert [r["md5"] for r in b["righe"]] == [r["md5"] for r in a["righe"]]
    assert [r["deriva"] for r in b["righe"]] == [r["deriva"] for r in a["righe"]]
    assert [r[1:] for r in b["struct"]] == [r[1:] for r in a["struct"]]
    for x, y in zip(a["righe"], b["righe"]):
        sy = dict(y["stato"])
        assert sy.pop("stato_generatore")["posti"] == {}
        assert sy == x["stato"]


def test_genoma_a_meta_nuvola_aspetta_la_fine_della_nuvola(monkeypatch):
    """A genome arriving mid-cloud waits for the cloud end, so its frame_idx reset holds."""
    import threading
    import time
    pn = pytest.importorskip("src.nodes.perturbation_node")
    rclpy = pytest.importorskip("rclpy")
    from std_msgs.msg import Float32MultiArray

    class _Preso(Exception):
        pass

    def cli(argv):
        def nodo(c):
            raise _Preso(c)
        with monkeypatch.context() as m:
            m.setattr(pn.rclpy, "init", lambda **k: None)
            m.setattr(pn, "PerturbationNode", nodo)
            m.setattr(sys, "argv", ["perturbation_node.py"] + argv)
            with pytest.raises(_Preso) as e:
                pn.main()
        return e.value.args[0]

    rclpy.init()
    try:
        for argv in (tr.NODO, tr.NODO + STG):
            n = pn.PerturbationNode(cli(argv))
            assert n._gen_lock.acquire(blocking=False)              # reentrant, always present
            n._gen_lock.release()
            cb = {s.topic_name: s.callback for s in n.subscriptions}
            for topic in ("/attack/genome", "/attack/victim_genome"):
                g = np.full(n.generator.get_genome_size(), 0.1)
                n.frame_idx = n.victim_frame_idx = 5
                n._gen_lock.acquire()                               # a cloud in progress
                t = threading.Thread(target=cb[topic], args=(Float32MultiArray(data=list(g)),))
                t.start()
                time.sleep(0.3)
                assert t.is_alive(), topic                          # the genome waits
                n.frame_idx += 1                                    # end of the cloud
                n.victim_frame_idx += 1
                n._gen_lock.release()
                t.join(5)
                assert not t.is_alive()
                idx = n.frame_idx if topic == "/attack/genome" else n.victim_frame_idx
                assert idx == 0, (topic, idx)                       # reset from the next cloud
            n.destroy_node()
    finally:
        rclpy.shutdown()
