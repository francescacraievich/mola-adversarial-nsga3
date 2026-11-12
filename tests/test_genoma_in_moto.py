#!/usr/bin/env python3
"""
Genome in motion: step ramp of the generator (imposta_gradino), --genoma-in-moto options of
node and orchestrator, launchers.

Synthetic scene and genomes from test_rampa_genoma.py; the robot "moves" by shifting the
cloud along x (the sensor advances by dx, the scene moves back by dx). The real node runs
in a separate process (this file with --figlio-gim, domain 88, no spin). Generator tests
run with PERT_OTTIMIZZAZIONI off and "tutte".

    ROS_DOMAIN_ID=88 python3 -B -m pytest -q -p no:cacheprovider tests/test_genoma_in_moto.py

Reference for the option off: an untouched copy of src/ from before the option
(GIM_ORIGINALE); clouds recorded from run c0ra_1119 with the node-profiling identity
script, if present.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

QUI = Path(__file__).resolve().parent
ROOT = QUI.parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(ROOT))

import test_rampa_genoma as tr  # noqa: E402

ORIGINALE = Path(os.environ.get("GIM_ORIGINALE", os.path.expanduser("~/mola_lavoro/lavoro_E1_originale")))
PROFILO = ROOT / "data/attack/dossier/2026-10-23_serieB/profilo_nodo_02nov"
RUN_1119 = ROOT / "data/attack/verifica/2026-11-01_1119_rigioco_c0ra_r01"
BANCO_PROFILO = Path(os.path.expanduser("~/mola_lavoro/profilo_nodo_02nov/banco.py"))

S_NODO, KMIN_NODO, KMAX_NODO = 2.0, 3, 12
NODO_GIM = tr.NODO + ["--genoma-in-moto", str(S_NODO), "--genoma-in-moto-kmin", str(KMIN_NODO),
                      "--genoma-in-moto-kmax", str(KMAX_NODO), "--stato-generatore"]
# robot motion (m) per cloud: stopped, then 1.5 cm per cloud
FERMO, PASSO = 5, 0.015
SEQ = [0.0] * FERMO + [PASSO * (i + 1) for i in range(6)]
# after the realignment pause: stopped 1.5 cm further (offset at load), then moving
SEQ_DOPO = [0.015] * 3 + [0.015 + PASSO * (i + 1) for i in range(5)]


def mossa(i, dx):
    """Synthetic cloud i seen from the sensor shifted by dx along x."""
    c = np.array(tr.scena(i))
    c[:, 0] -= dx
    c.setflags(write=False)
    return c


# ----------------------------------------------------------------------
# Child process: real node with --genoma-in-moto, callbacks called by hand
# ----------------------------------------------------------------------

def _figlio_gim(argv):
    sep = argv.index("--")
    radice, uscita = argv[:sep]
    node_args = list(argv[sep + 1:])
    if os.environ.get("ROS_DOMAIN_ID") != "88":
        sys.exit(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}: il nodo dei test gira solo nel dominio 88")
    sys.path.insert(0, radice)
    import src.nodes.perturbation_node as pn
    from src.perturbations import zone_attaccante as za
    from std_msgs.msg import Bool, Float32MultiArray, String
    import rclpy
    lavoro = Path(uscita).with_suffix("")
    zd = lavoro / "zone"
    zd.mkdir(parents=True, exist_ok=True)
    node_args += ["--struct-log", str(lavoro / "struct.csv")]

    class _Cli(Exception):
        pass

    orig = pn.PerturbationNode
    pn.PerturbationNode = lambda cli: (_ for _ in ()).throw(_Cli(cli))
    sys.argv = ["perturbation_node.py"] + node_args
    try:
        pn.main()
        raise SystemExit("main non ha creato il nodo")
    except _Cli as e:
        cli = e.args[0]
    pn.PerturbationNode = orig
    node = orig(cli)
    node._out_q = queue.Queue()
    cb = {s.topic_name: s.callback for s in node.subscriptions}
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
        d = json.loads(node.status_pub.m.data.replace(str(lavoro), "<DIR>"))
        for k in tr.VOLATILI:
            d.pop(k, None)
        return d

    stamp = [10.0]

    def nuvola(flusso, c, s=None):
        """One cloud to the stream; s = stamp (default: 0.1 s after the previous one)."""
        if flusso == "gemello":
            if node.victim_stream:
                node._victim_stream_cb(Bool(data=False))
            node.enabled = True
        else:
            if not node.victim_stream:
                node._victim_stream_cb(Bool(data=True))
            node.victim_attack = True
        stamp[0] = stamp[0] + 0.1 if s is None else s
        fidx = node.victim_frame_idx if flusso == "vittima" else node.frame_idx
        node.sub.callback(tr._msg(pn, c, stamp[0]))
        item = node._out_q.get_nowait()
        o = item[4] if flusso == "vittima" else item[2]
        gen = node.victim_generator if flusso == "vittima" else node.generator
        inf = gen.last_struct_info or {}
        return {"flusso": flusso, "md5": tr._md5(o), "k": inf.get("k"), "fidx": fidx,
                "pose": list(inf.get("pose") or ()), "gradino": gen.stato_gradino(),
                "deriva": [float(x) for x in gen.accumulated_drift], "stato": stato()}

    def manda_zone(flusso, z):
        node._zone_cb(String(data=json.dumps({"seq": seq(), "flusso": flusso, "path": str(z.path),
                                              "sha256": z.sha256, "hash": z.hash, "genoma_sha": z.genoma_sha})))

    def manda_rampa(flusso, g_prec, sha_nuovo, z_prec):
        cb["/attack/rampa"](String(data=json.dumps({
            "seq": seq(), "flusso": flusso, "genoma_prec": [float(x) for x in g_prec], "genoma_sha_nuovo": sha_nuovo,
            "zone_prec": {"path": str(z_prec.path), "sha256": z_prec.sha256, "hash": z_prec.hash}})))

    def annulla(flusso):
        cb["/attack/rampa"](String(data=json.dumps({"seq": seq(), "flusso": flusso, "annulla": True})))

    def metro(flusso, w):
        node._metro_cb(String(data=json.dumps({flusso: w})))

    def stato_gen(azione, nome, flusso=None):
        node._stato_gen_cb(String(data=json.dumps({"seq": seq(), "azione": azione, "nome": nome,
                                                   **({"flusso": flusso} if flusso else {})})))

    def genoma(g):
        cb["/attack/genome"](Float32MultiArray(data=[float(x) for x in g]))

    node._drift_keep_cb(Bool(data=True))
    zone = {}
    for w, idx in ((1, -1), (0, -3)):
        s_f = 9.0 + w
        nome = f"w{w}_fotografia.npz"
        node._att_buf.append((pn._stamp_s(tr._msg(pn, tr.scena(idx), s_f).header), tr.scena(idx)))
        node._foto_cb(String(data=json.dumps({
            "seq": seq(), "stamp": pn._stamp_s(tr._msg(pn, tr.scena(idx), s_f).header), "path": str(zd / nome),
            "meta": {"metro": w, "posa_base_odom": [0.0, 0.0, 0.0]}})))
        fs = node._foto_stato
        assert fs["esito"] == "ok", fs
        foto = za.carica_fotografia(zd / nome, fs["sha256"], fs["hash"])
        limiti = {k: float(getattr(node.generator, k)) for k in za.LIMITI}
        zone[w] = {nm: za.zone_genoma(zd, foto, g, w, limiti) for nm, g in (("P", tr.GP), ("N", tr.GN))}
    zP, zN = zone[1]["P"], zone[1]["N"]
    sha_n = za.genoma_sha(tr.GN)
    R = {}

    # meter 0: P data sent anyway, the step does not use it
    manda_rampa("gemello", tr.GP, sha_n, zone[0]["P"])
    metro("gemello", 0)
    manda_zone("gemello", zone[0]["N"])
    genoma(tr.GN)
    R["metro0"] = [nuvola("gemello", mossa(i, dx)) for i, dx in enumerate(SEQ[:FERMO + 2])]

    # victim with P at meter 1 (victim drift, from which the twins start with keep)
    metro("vittima", 1)
    manda_zone("vittima", zP)
    cb["/attack/victim_genome"](Float32MultiArray(data=list(tr.GP)))
    R["vittima_P"] = [nuvola("vittima", mossa(100 + i, 0.0)) for i in range(2)]

    # twin and victim with P -> N and the same clouds
    manda_rampa("gemello", tr.GP, sha_n, zP)
    metro("gemello", 1)
    manda_zone("gemello", zN)
    genoma(tr.GN)
    # same stamps: the follower predicts the pose from the intervals between stamps
    t0 = stamp[0]
    R["gemello"] = [nuvola("gemello", mossa(i, dx), t0 + 0.1 * (i + 1)) for i, dx in enumerate(SEQ)]
    manda_rampa("vittima", tr.GP, sha_n, zP)
    manda_zone("vittima", zN)
    cb["/attack/victim_genome"](Float32MultiArray(data=list(tr.GN)))
    R["vittima"] = [nuvola("vittima", mossa(i, dx), t0 + 0.1 * (i + 1)) for i, dx in enumerate(SEQ)]

    # realignment: victim with the same genome from scratch, save with the robot stopped
    # (phase P), stream closed, realigned run with the loaded slot (ramp cancelled first, as
    # the orchestrator does with RAMPA["sospesa"]), victim reopened
    cb["/attack/victim_genome"](Float32MultiArray(data=list(tr.GN)))
    R["vittima_prima"] = [nuvola("vittima", mossa(i, 0.0)) for i in range(FERMO)]
    stato_gen("salva", "riallineo", "vittima")
    node._victim_stream_cb(Bool(data=False))
    R["posto"] = stato()["stato_generatore"]["posti"]["riallineo"]
    annulla("gemello")
    stato_gen("carica", "riallineo")
    metro("gemello", 1)
    manda_zone("gemello", zN)
    genoma(tr.GN)
    R["caricato"] = stato()["stato_generatore"]["ultimo_caricato"]
    # same stamps for the realigned run and the resumed victim (Isaac restored to the same time)
    t_pausa = stamp[0]
    R["riallineato"] = [nuvola("gemello", mossa(200 + i, dx), t_pausa + 0.1 * (i + 1))
                        for i, dx in enumerate(SEQ_DOPO)]
    R["vittima_dopo"] = [nuvola("vittima", mossa(200 + i, dx), t_pausa + 0.1 * (i + 1))
                         for i, dx in enumerate(SEQ_DOPO)]

    # genome arriving in the middle of a cloud (lock held by the cloud)
    j = 2
    manda_rampa("gemello", tr.GP, sha_n, zP)
    manda_zone("gemello", zN)
    genoma(tr.GN)
    t1 = stamp[0]
    R["d1_rif"] = [nuvola("gemello", mossa(300 + i, 0.0), t1 + 0.1 * (i + 1)) for i in range(j + 3)]
    genoma(tr.GN)
    R["d1"] = [nuvola("gemello", mossa(300 + i, 0.0), t1 + 0.1 * (i + 1)) for i in range(j)]
    ap = node.generator.apply_perturbation
    d1 = {}

    def a_meta(*a, **k):
        out = ap(*a, **k)
        t = threading.Thread(target=genoma, args=(tr.GN,))
        t.start()
        time.sleep(0.3)
        d1["aspetta"] = t.is_alive()
        d1["t"] = t
        return out

    node.generator.apply_perturbation = a_meta
    R["d1"].append(nuvola("gemello", mossa(300 + j, 0.0), t1 + 0.1 * (j + 1)))
    node.generator.apply_perturbation = ap
    d1["t"].join(10)
    R["d1_aspetta"] = d1["aspetta"]
    R["d1_frame_idx"] = node.frame_idx
    R["d1_dopo"] = [nuvola("gemello", mossa(300 + i, 0.0), t1 + 0.1 * (i + 1)) for i in range(3)]

    node._struct_log.close()
    with open(node._struct_log.path) as f:
        R["struct"] = [r.rstrip("\n").split(",") for r in f]
    json.dump(R, open(uscita, "w"))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__" and "--figlio-gim" in sys.argv:
    _figlio_gim(sys.argv[sys.argv.index("--figlio-gim") + 1:])
    sys.exit(0)


# ----------------------------------------------------------------------
# Tests (pytest)
# ----------------------------------------------------------------------

import pytest  # noqa: E402
from test_rampa_genoma import Z  # noqa: E402,F401  (zone fixture)

from src.perturbations import structures as st  # noqa: E402


@pytest.fixture(params=["spente", "tutte"])
def ott(request, monkeypatch):
    """Step tests with the generator optimizations off and on."""
    monkeypatch.setattr(st, "OTT", frozenset() if request.param == "spente" else frozenset(st.OTT_NOMI))
    return request.param


def _gen_gradino(Z, S=2.0, kmin=3, kmax=40, zona_n="zN", prec="zP", deriva=tr.DERIVA0, stato=None):
    g = tr._gen(Z, Z[zona_n], deriva)
    p = None if prec is None else tr._prec(Z, prec)
    g.imposta_gradino(S, kmin, kmax, p, stato, genoma_sha=Z["shaN"], seq=1)
    return g


def _passo(g, params, seme, nuvola, i, metro=1):
    o = g.apply_perturbation(nuvola, params, ctx={"stamp": 2.0 + 0.1 * i}, seme_genoma=seme, metro=metro)
    inf = g.last_struct_info
    return {"md5": tr._md5(o), "n_hit": dict(inf["n_hit"]), "k": inf["k"], "pose": inf["pose"],
            "deriva": g.accumulated_drift, "gradino": None if g._gradino is None else g.stato_gradino()}


def _sequenza(Z, g, profilo, params="pN", seme="sN", inizio=0):
    return [_passo(g, Z[params], Z[seme], mossa(inizio + i, dx), inizio + i) for i, dx in enumerate(profilo)]


# --- off by default -------------------------------------------------------------------

def test_spento_generatore(Z):
    pg = Z["pg"]
    g = tr._gen(Z, Z["zN"])
    assert g._gradino is None and g.stato_gradino() is None
    tr._passo(g, Z["pN"], Z["sN"], 0)
    assert set(g.stato_flusso(copia=False)) == set(pg.PerturbationGenerator.STATO_FLUSSO)
    with pytest.raises(ValueError):
        g.imposta_gradino(0.0, 3, 40, tr._prec(Z))
    with pytest.raises(ValueError):
        g.imposta_gradino(2.0, 5, 4, tr._prec(Z))
    gi = pg.PerturbationGenerator(**tr.LIM, **Z["za"].KW_INSIEME)
    with pytest.raises(ValueError):
        gi.imposta_gradino(2.0, 3, 40, tr._prec(Z))            # without attacker zones
    gr = tr._gen_rampa(Z, 4)
    with pytest.raises(ValueError):
        gr.imposta_gradino(2.0, 3, 40, tr._prec(Z))            # ramp and step together


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

    c = cli([])
    assert (c.genoma_in_moto, c.genoma_in_moto_kmin, c.genoma_in_moto_kmax) == (0.0, 20, 40)
    assert pn.PerturbationNode._gim is None
    rclpy.init()
    try:
        topic = {}
        for nome, argv in (("spento", tr.NODO), ("zero", tr.NODO + ["--genoma-in-moto", "0"]),
                           ("acceso", tr.NODO + ["--genoma-in-moto", "2.0"])):
            n = pn.PerturbationNode(cli(argv))
            topic[nome] = sorted(s.topic_name for s in n.subscriptions)
            assert (n._gim is not None) == (nome == "acceso")
            if nome == "acceso":
                assert n._gim == {"S_cm": 2.0, "kmin": 20, "kmax": 40}
                assert n._struct_log is None                       # no log without --struct-log
            n.destroy_node()
        assert "/attack/rampa" not in topic["spento"] and topic["zero"] == topic["spento"]
        assert topic["acceso"] == sorted(topic["spento"] + ["/attack/rampa"])
        for argv in (["--genoma-in-moto", "2.0"],                                      # no zones
                     tr.NODO + ["--genoma-in-moto", "2.0", "--rampa-genoma", "4"],
                     tr.NODO + ["--genoma-in-moto", "2.0", "--genoma-in-moto-kmin", "41"],
                     tr.NODO + ["--genoma-in-moto", "-1"]):
            with pytest.raises(ValueError):
                pn.PerturbationNode(cli(argv))
    finally:
        rclpy.shutdown()


@pytest.mark.skipif(not (ORIGINALE / "src/nodes/perturbation_node.py").exists(), reason="copia originale assente")
def test_spento_identico_alla_copia_originale(tmp_path):
    """Without the option and with --genoma-in-moto 0: victim and twin outputs, status at
    every cloud, zone log identical to the node of the untouched copy."""
    a = tr._nodo_figlio(tmp_path, "orig", ORIGINALE, "spento", tr.NODO)
    b = tr._nodo_figlio(tmp_path, "nuovo", ROOT, "spento", tr.NODO)
    c = tr._nodo_figlio(tmp_path, "zero", ROOT, "spento", tr.NODO + ["--genoma-in-moto", "0"])
    assert len(a["righe"]) == 3 + 2 * tr.NK
    for x in (b, c):
        assert x["zone"] == a["zone"]
        assert [r["md5"] for r in x["righe"]] == [r["md5"] for r in a["righe"]]
        assert [r["stato"] for r in x["righe"]] == [r["stato"] for r in a["righe"]]
        assert [r["deriva"] for r in x["righe"]] == [r["deriva"] for r in a["righe"]]
        assert x["struct"][0] == a["struct"][0]
        assert [r[1:] for r in x["struct"][1:]] == [r[1:] for r in a["struct"][1:]]
    assert all("genoma_in_moto" not in r["stato"] and "chamfer_fasi" not in r["stato"] for r in b["righe"] + c["righe"])
    assert not any(h.startswith("gradino_") for h in b["struct"][0])


def test_chamfer_per_fase_senza_gradino(tmp_path):
    """--chamfer-per-fase without step: clouds, log and status equal to the node without the
    option except the chamfer_fasi key; all Chamfers in phase N, per stream, reset at the
    stream's new genome."""
    s = tr._nodo_figlio(tmp_path, "spento", ROOT, "spento", tr.NODO)
    c = tr._nodo_figlio(tmp_path, "cpf", ROOT, "spento", tr.NODO + ["--chamfer-per-fase"])
    assert [r["md5"] for r in c["righe"]] == [r["md5"] for r in s["righe"]]
    assert c["struct"][0] == s["struct"][0] and [r[1:] for r in c["struct"][1:]] == [r[1:] for r in s["struct"][1:]]
    for x, y in zip(c["righe"], s["righe"]):
        sx = dict(x["stato"])
        assert set(sx.pop("chamfer_fasi")) == {"gemello", "vittima"} and sx == y["stato"]
        assert "genoma_in_moto" not in sx
    # rows: victim with P (3), twin with N (NK), victim with N (NK); Chamfer every 5 stream clouds
    vit_p, gem, vit = c["righe"][:3], c["righe"][3:3 + tr.NK], c["righe"][3 + tr.NK:]
    for flusso, rr, fine in (("gemello", gem, gem[-1]), ("vittima", vit, vit[-1]), ("vittima", vit_p, vit_p[-1])):
        cf = fine["stato"]["chamfer_fasi"][flusso]
        att = [r["stato"]["chamfer_cm"] for i, r in enumerate(rr) if i % 5 == 0]
        assert cf["n_chamfer_P"] == 0 and cf["chamfer_P_somma"] == 0.0
        assert cf["n_chamfer_N"] == len(att) >= 1
        assert abs(cf["chamfer_N_somma"] - sum(att)) <= 0.0005 * len(att) + 1e-9
    # the twin does not touch the victim sum and vice versa
    assert gem[-1]["stato"]["chamfer_fasi"]["vittima"] == vit_p[-1]["stato"]["chamfer_fasi"]["vittima"]


@pytest.mark.skipif(not ((RUN_1119 / "nuvole/indice.csv").is_file() and BANCO_PROFILO.is_file()
                         and (PROFILO / "integrazione_repo/id_repo_spente_18run.json").is_file()),
                    reason="nuvole registrate o banco della profilatura assenti")
@pytest.mark.parametrize("ottim", ["spente", "tutte"])
def test_spento_nuvole_registrate(ottim, tmp_path):
    """Repo generator without the option on the clouds saved by the node in run c0ra_1119:
    outputs equal byte for byte to the saved ones, and per stream sha256 of the outputs and
    state fingerprints equal to those measured on the code before the option."""
    out = tmp_path / "id.json"
    with open(out, "w") as f:
        r = subprocess.run([sys.executable, "-B", str(PROFILO / "script/identita.py"), str(ROOT), str(RUN_1119)],
                           stdout=f, stderr=subprocess.PIPE, text=True, timeout=1200, cwd=PROFILO / "script",
                           env=dict(os.environ, PERT_OTTIMIZZAZIONI=ottim, ROS_DOMAIN_ID="88"))
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["nuvole"] == d["uguali"] == 200
    rif = json.loads((PROFILO / f"integrazione_repo/id_repo_{ottim}_18run.json").read_text())
    rif = {(x["flusso"], x["genome_id"]): x for x in rif["flussi"] if x["run"] == str(RUN_1119)}
    assert len(rif) == len(d["flussi"]) == 10
    for x in d["flussi"]:
        y = rif[(x["flusso"], x["genome_id"])]
        assert (x["sha256_uscite"], x["impronte_stato"], x["k"]) == (y["sha256_uscite"], y["impronte_stato"], y["k"])


# --- phase P = P alone ----------------------------------------------------------------

def test_fase_P_uguale_a_P_da_solo(Z, ott):
    """Before entry: clouds equal byte for byte to a generator with only P on the P zones of
    the same snapshot (same drift); P hits with the prefix, no N operator."""
    pre = Z["pg"].RAMPA_PREFISSO
    ref, g = tr._gen(Z, Z["zP"]), _gen_gradino(Z, kmin=6)
    for i in range(6):
        a = _passo(ref, Z["pP"], Z["sP"], mossa(i, 0.0), i)
        b = _passo(g, Z["pN"], Z["sN"], mossa(i, 0.0), i)
        assert b["gradino"]["fase"] == "P" and b["k"] == i
        assert a["md5"] == b["md5"] and np.array_equal(a["deriva"], b["deriva"])
        assert all(k.startswith(pre) for k in b["n_hit"])
        assert {k[len(pre):]: v for k, v in b["n_hit"].items()} == a["n_hit"]
    assert any(k.startswith(pre) for k in g._structs.anchors)


# --- after entry = N alone ------------------------------------------------------------

@pytest.fixture(scope="module")
def Z0(tmp_path_factory):
    """As Z, with P and N without drift (gene 14 at -1: s14 < 0.1)."""
    pg, za = tr._moduli()
    d = tmp_path_factory.mktemp("gim_zone0")
    s1 = za.salva_fotografia(d / "w1_fotografia.npz", tr.scena(-1), {"metro": 1, "posa_base_odom": [0, 0, 0]})
    f1 = za.carica_fotografia(d / "w1_fotografia.npz", *s1)
    GP0, GN0 = tr.GP.copy(), tr.GN.copy()
    GP0[14] = GN0[14] = -1.0
    cache = {}
    zz = {nome: za.carica_zone(z.path, z.sha256, z.hash, cache=cache)
          for nome, z in (("P", za.zone_genoma(d, f1, GP0, 1, tr.LIM)), ("N", za.zone_genoma(d, f1, GN0, 1, tr.LIM)))}
    g = pg.PerturbationGenerator(**tr.LIM)
    Zd = {"pg": pg, "za": za, "zP": zz["P"], "zN": zz["N"], "pP": g.encode_perturbation(GP0),
          "pN": g.encode_perturbation(GN0), "sP": za.seme_genoma(GP0), "sN": za.seme_genoma(GN0),
          "shaP": za.genoma_sha(GP0), "shaN": za.genoma_sha(GN0)}
    assert Zd["pP"]["temporal_drift_strength"] < 0.1 and Zd["pN"]["temporal_drift_strength"] < 0.1
    return Zd


def _confronta_anchors(ref, g, pre):
    ta, tb = ref._structs.anchors, g._structs.anchors
    assert not any(k.startswith(pre) for k in tb)
    assert sorted(ta) == sorted(tb)
    for op in ta:
        for A, B in zip(ta[op], tb[op]):
            assert np.array_equal(A["keys"], B["keys"]) and A["lost"] == B["lost"]


def test_dopo_ingresso_uguale_a_N_da_solo_senza_deriva(Z0, ott):
    """Drift off (s14 < 0.1, zero starting drift): from the entry cloud the clouds are those
    of N alone on the same clouds, with no adjustment; not before."""
    pre = Z0["pg"].RAMPA_PREFISSO
    ref = tr._gen(Z0, Z0["zN"], deriva=(0.0, 0.0, 0.0))
    g = _gen_gradino(Z0, kmin=3, deriva=(0.0, 0.0, 0.0))
    a = _sequenza(Z0, ref, SEQ)
    b = _sequenza(Z0, g, SEQ)
    fasi = [x["gradino"]["fase"] for x in b]
    ki = g.stato_gradino()["k_ingresso"]
    assert ki is not None and fasi[ki] == "ingresso" and set(fasi[:ki]) == {"P"} and set(fasi[ki + 1:]) == {"N"}
    assert all(x["md5"] != y["md5"] for x, y in zip(a[:ki], b[:ki]))
    for x, y in zip(a[ki:], b[ki:]):
        assert x["md5"] == y["md5"] and x["n_hit"] == y["n_hit"]
        assert np.array_equal(x["deriva"], y["deriva"]) and not np.any(y["deriva"])
    _confronta_anchors(ref, g, pre)


def test_dopo_ingresso_con_deriva_cambia_solo_accumulatore(Z, ott):
    """Drift on: before entry the accumulator is P's; with the step accumulator given to N
    alone at the entry cloud, the clouds are identical from there on."""
    pre = Z["pg"].RAMPA_PREFISSO
    ref, g = tr._gen(Z, Z["zN"]), _gen_gradino(Z, kmin=3)
    ki = None
    for i, dx in enumerate(SEQ):
        d_prima = g.accumulated_drift
        b = _passo(g, Z["pN"], Z["sN"], mossa(i, dx), i)
        if b["gradino"]["fase"] == "ingresso":
            ki = i
            assert not np.array_equal(ref.accumulated_drift, d_prima)     # P and N drift differ
            ref.set_temporal_state(d_prima)
        a = _passo(ref, Z["pN"], Z["sN"], mossa(i, dx), i)
        if ki is None:
            assert a["md5"] != b["md5"]
        else:
            assert a["md5"] == b["md5"] and a["n_hit"] == b["n_hit"] and np.array_equal(a["deriva"], b["deriva"])
    assert ki is not None and ki < len(SEQ) - 1
    _confronta_anchors(ref, g, pre)


# --- trigger --------------------------------------------------------------------------

def test_innesco_alla_prima_nuvola_oltre_S(Z, ott):
    """Entry at the first cloud with k >= kmin and follower displacement from the reference
    >= S; distance from the threshold on both sides, stamp gap, reference at k = 0.
    Recomputed independently from the follower poses."""
    prof = [0.0] * 5 + [0.005, 0.01, 0.015, 0.03, 0.045]
    g = _gen_gradino(Z, S=2.0, kmin=3)
    b = _sequenza(Z, g, prof)
    p0 = np.array(b[0]["pose"][:2])
    spost = [float(np.hypot(*(np.array(x["pose"][:2]) - p0)) * 100.0) for x in b]
    atteso = next(k for k, s in enumerate(spost) if k >= 3 and s >= 2.0)
    assert atteso == 8  # 3 cm after 1.5 cm: far from threshold
    sg = g.stato_gradino()
    assert sg["k_ingresso"] == atteso and sg["stato"] == "attivo" and sg["rif_k"] == 0
    assert sg["spost_prima_cm"] < 2.0 <= sg["spost_ingresso_cm"]
    assert abs(sg["spost_ingresso_cm"] - spost[atteso]) < 1e-9 and abs(sg["spost_prima_cm"] - spost[atteso - 1]) < 1e-9
    assert abs(sg["buco_stamp_s"] - 0.1) < 1e-9
    assert [x["gradino"]["fase"] for x in b] == ["P"] * atteso + ["ingresso", "N"]
    # per-cloud columns: displacement of the cloud and of the previous one, then empty in N
    assert b[0]["gradino"]["spost_cm"] == 0.0 and b[0]["gradino"]["spost_prec_cm"] is None
    assert abs(b[5]["gradino"]["spost_prec_cm"] - spost[4]) < 1e-9
    assert b[9]["gradino"]["spost_cm"] is None and b[9]["gradino"]["k_ingresso"] == atteso


def test_innesco_mai_prima_di_kmin(Z, ott):
    g = _gen_gradino(Z, S=2.0, kmin=4)
    b = _sequenza(Z, g, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    assert [x["gradino"]["fase"] for x in b] == ["P"] * 4 + ["ingresso", "N"]
    sg = g.stato_gradino()
    assert sg["k_ingresso"] == 4 and sg["stato"] == "attivo" and sg["spost_prima_cm"] > 2.0


def test_ripiego_a_kmax(Z, ott):
    g = _gen_gradino(Z, S=2.0, kmin=2, kmax=5)
    b = _sequenza(Z, g, [0.0] * 7)
    assert [x["gradino"]["fase"] for x in b] == ["P"] * 5 + ["ingresso", "N"]
    sg = g.stato_gradino()
    assert sg["k_ingresso"] == 5 and sg["stato"] == "k_max" and sg["spost_ingresso_cm"] < 2.0


@pytest.mark.parametrize("stato,fase", [("metro0", "metro0"), ("prec_assente", None)])
def test_senza_P_solo_N(Z, ott, stato, fase):
    """Meter 0 (and no previous genome): N alone from k = 0, equal to N alone."""
    ref, g = tr._gen(Z, Z["zN"]), _gen_gradino(Z, prec=None, stato=stato)
    a, b = _sequenza(Z, ref, SEQ), _sequenza(Z, g, SEQ)
    assert [x["md5"] for x in a] == [x["md5"] for x in b]
    assert all(x["gradino"]["fase"] == fase and x["gradino"]["stato"] == stato
               and x["gradino"]["k_ingresso"] is None for x in b)


def test_zone_di_P_di_un_altro_metro(Z, ott):
    ref, g = tr._gen(Z, Z["zN"]), _gen_gradino(Z, prec="zP2")
    a, b = _sequenza(Z, ref, SEQ[:3]), _sequenza(Z, g, SEQ[:3])
    assert [x["md5"] for x in a] == [x["md5"] for x in b]
    assert g.stato_gradino()["stato"] == "metro_diverso" and g.stato_gradino()["fase"] is None


# --- reference reset (mandatory) ------------------------------------------------------

def test_riferimento_ripreso_dopo_il_carico(Z, ott):
    """An offset at standstill at load (1.2 cm, more than the test's S = 1.0 cm) does not
    trigger the step if the reference is reset; without the reset it would trigger at the
    first cloud after the load, with the robot stopped."""
    pg = Z["pg"]
    a = _gen_gradino(Z, S=1.0, kmin=3, kmax=30)
    _sequenza(Z, a, [0.0] * 5)
    posto = a.stato_flusso()
    assert posto["_gradino"]["fase"] == "P" and posto["_gradino"]["rif_k"] == 0
    dopo = [0.012] * 4 + [0.012 + 0.006 * (i + 1) for i in range(3)]
    esiti = {}
    for nome, riprendi in (("ripresa", True), ("senza", False)):
        b = tr._gen(Z, Z["zN"])
        b.imposta_stato_flusso(posto)
        assert pg.impronta_stato(b.stato_flusso(copia=False)) == pg.impronta_stato(posto)
        if riprendi:
            b.riprendi_riferimento_moto()
        esiti[nome] = (_sequenza(Z, b, dopo, inizio=5), b.stato_gradino())
    xs, sg = esiti["ripresa"]
    assert [x["gradino"]["fase"] for x in xs[:5]] == ["P"] * 5  # stopped and first step: P
    assert sg["rif_k"] == 5 and sg["k_ingresso"] == 10 and sg["stato"] == "attivo"
    xs, sg = esiti["senza"]
    assert sg["rif_k"] == 0 and sg["k_ingresso"] == 5  # would trigger at standstill


def test_riferimento_ripreso_alla_riapertura(Z, ott):
    """Victim: the reset applies to the cloud after the call; without step it does nothing."""
    g = _gen_gradino(Z, S=2.0, kmin=0, kmax=30)
    _sequenza(Z, g, [0.0] * 3)
    g.riprendi_riferimento_moto()
    b = _sequenza(Z, g, [0.015, 0.015, 0.03], inizio=3)
    assert g.stato_gradino()["rif_k"] == 3 and [x["gradino"]["fase"] for x in b] == ["P", "P", "P"]
    g0 = tr._gen(Z, Z["zN"])
    g0.riprendi_riferimento_moto()
    assert g0._gradino is None


# --- state round trip -----------------------------------------------------------------

def test_andata_e_ritorno_dello_stato(Z, ott):
    """Save in phase P at k = 21, load into another generator, reference reset in both
    (victim reopening, realigned load): same clouds, identical outputs and same entry k."""
    pg = Z["pg"]
    prof = [0.0] * 24 + [PASSO * (i + 1) for i in range(6)]
    a = _gen_gradino(Z, S=2.0, kmin=20, kmax=40)
    _sequenza(Z, a, prof[:22])
    posto = a.stato_flusso()
    assert posto["_gradino"]["fase"] == "P" and posto["_structs"].k == 21
    b = tr._gen(Z, Z["zP2"])                    # different zones: the loaded state replaces them
    b.imposta_stato_flusso(posto)
    assert pg.impronta_stato(b.stato_flusso(copia=False)) == pg.impronta_stato(a.stato_flusso(copia=False))
    a.riprendi_riferimento_moto()
    b.riprendi_riferimento_moto()
    xa = _sequenza(Z, a, prof[22:], inizio=22)
    xb = _sequenza(Z, b, prof[22:], inizio=22)
    assert [x["md5"] for x in xa] == [x["md5"] for x in xb]
    assert [x["gradino"] for x in xa] == [x["gradino"] for x in xb]
    ki = a.stato_gradino()["k_ingresso"]
    assert ki == b.stato_gradino()["k_ingresso"] and 24 <= ki < 30
    assert pg.impronta_stato(b.stato_flusso(copia=False)) == pg.impronta_stato(a.stato_flusso(copia=False))


# --- node: victim and twin, realignment, mid-cloud genome, Chamfer per phase ----------

@pytest.fixture(scope="module")
def figlio(tmp_path_factory):
    out = tmp_path_factory.mktemp("gim_nodo") / "gim.json"
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--figlio-gim", str(ROOT), str(out), "--"] + NODO_GIM
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=dict(os.environ, ROS_DOMAIN_ID="88"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def test_nodo_vittima_uguale_al_gemello(figlio):
    gem, vit = figlio["gemello"], figlio["vittima"]
    assert [r["md5"] for r in gem] == [r["md5"] for r in vit]
    assert [r["deriva"] for r in gem] == [r["deriva"] for r in vit]
    # all equal except the ramp message number, one per stream
    senza_seq = lambda rr: [{k: v for k, v in r["gradino"].items() if k != "seq"} for r in rr]  # noqa: E731
    assert senza_seq(gem) == senza_seq(vit)
    assert gem[0]["gradino"]["seq"] != vit[0]["gradino"]["seq"]
    ki = gem[-1]["gradino"]["k_ingresso"]
    assert ki is not None and KMIN_NODO <= ki and [r["gradino"]["fase"] for r in gem[:ki]] == ["P"] * ki
    assert gem[ki]["gradino"]["fase"] == "ingresso" and gem[-1]["gradino"]["stato"] == "attivo"
    st_ = gem[-1]["stato"]["genoma_in_moto"]
    assert (st_["S_cm"], st_["kmin"], st_["kmax"]) == (S_NODO, KMIN_NODO, KMAX_NODO)
    assert st_["ricevuta"]["gemello"]["stato"] == "ok" and st_["gemello"]["k_ingresso"] == ki
    assert vit[-1]["stato"]["genoma_in_moto"]["vittima"]["k_ingresso"] == ki
    assert "rampa" not in gem[-1]["stato"]


def test_nodo_metro0(figlio):
    rr = figlio["metro0"]
    assert all(r["gradino"]["fase"] == "metro0" and r["gradino"]["stato"] == "metro0" for r in rr)
    assert all(r["gradino"]["k_ingresso"] is None for r in rr)


def test_nodo_D31c_riallineato_uguale_alla_vittima(figlio):
    """Victim save in phase P, twin ramp cancelled before the load (does not touch the
    loaded state), reference reset after the load and at reopening: realigned and victim
    identical, entry at the same cloud, after the standstill offset at load (1.5 cm) and not
    before."""
    posto = figlio["posto"]
    assert posto["gradino_fase"] == "P" and posto["k_strutture"] == FERMO - 1
    assert figlio["caricato"]["esito"] == "ok"
    # load fingerprint equal to the slot's (computed before the reference reset)
    assert figlio["caricato"]["impronta"] == posto["impronta"]
    ria, vit = figlio["riallineato"], figlio["vittima_dopo"]
    assert [r["md5"] for r in ria] == [r["md5"] for r in vit]
    assert [r["gradino"] for r in ria] == [r["gradino"] for r in vit]
    assert [r["k"] for r in ria] == list(range(FERMO, FERMO + len(SEQ_DOPO)))
    g = ria[-1]["gradino"]
    # reference at the first cloud after the load; 1.5 cm of standstill offset is not enough
    assert g["rif_k"] == FERMO and g["k_ingresso"] == FERMO + 4 and g["stato"] == "attivo"
    assert ria[0]["stato"]["genoma_in_moto"]["ricevuta"]["gemello"]["stato"] == "annullata"


def test_nodo_D1_genoma_a_meta_nuvola(figlio):
    """A genome arriving mid-cloud waits for the end of the cloud: the current cloud is the
    undisturbed one (same k, same output), then k = 0 and phase P."""
    rif, d1 = figlio["d1_rif"], figlio["d1"]
    assert figlio["d1_aspetta"] is True and figlio["d1_frame_idx"] == 0
    assert [r["md5"] for r in d1] == [r["md5"] for r in rif[:len(d1)]]
    assert [r["k"] for r in d1] == list(range(len(d1)))
    dopo = figlio["d1_dopo"]
    assert [r["k"] for r in dopo] == [0, 1, 2] and all(r["gradino"]["fase"] == "P" for r in dopo)
    assert [r["md5"] for r in dopo] == [r["md5"] for r in rif[:3]]


def test_nodo_chamfer_per_fase(figlio):
    """Sum and count of the stream Chamfers per phase, on the clouds where they are computed
    (every 5 stream clouds), reset at the new genome."""
    for flusso, rr in (("gemello", figlio["gemello"]), ("vittima", figlio["vittima"])):
        att = {"P": [0.0, 0], "N": [0.0, 0]}
        for r in rr:
            if r["fidx"] % 5 == 0:
                f = "P" if r["gradino"]["fase"] == "P" else "N"
                att[f][0] += r["stato"]["chamfer_cm"]
                att[f][1] += 1
        s = rr[-1]["stato"]["genoma_in_moto"][flusso]
        assert (s["n_chamfer_P"], s["n_chamfer_N"]) == (att["P"][1], att["N"][1])
        assert att["P"][1] >= 1 and att["N"][1] >= 1
        for f in "PN":
            assert abs(s[f"chamfer_{f}_somma"] - att[f][0]) <= 0.0005 * att[f][1] + 1e-9
    # realigned: reset at its genome; victim: its clouds before and after the pause
    s = figlio["riallineato"][-1]["stato"]["genoma_in_moto"]
    n_ria = sum(1 for r in figlio["riallineato"] if r["fidx"] % 5 == 0)
    assert s["gemello"]["n_chamfer_P"] + s["gemello"]["n_chamfer_N"] == n_ria
    n_vit = sum(1 for r in figlio["vittima_prima"] + figlio["vittima_dopo"] if r["fidx"] % 5 == 0)
    sv = figlio["vittima_dopo"][-1]["stato"]["genoma_in_moto"]["vittima"]
    assert sv["n_chamfer_P"] + sv["n_chamfer_N"] == n_vit


def test_nodo_chamfer_fasi_con_il_gradino(figlio):
    """With --genoma-in-moto the chamfer_fasi key is present and matches the step sums."""
    for r in figlio["gemello"] + figlio["vittima"] + figlio["riallineato"]:
        cf, gm = r["stato"]["chamfer_fasi"], r["stato"]["genoma_in_moto"]
        for f in ("gemello", "vittima"):
            assert cf[f] == {k: gm[f][k] for k in ("genome_id", "chamfer_P_somma", "n_chamfer_P",
                                                   "chamfer_N_somma", "n_chamfer_N")}


def test_nodo_registro_delle_zone(figlio):
    pn = pytest.importorskip("src.nodes.perturbation_node")
    hdr = figlio["struct"][0]
    assert tuple(hdr[-6:]) == pn.GRADINO_COLONNE
    righe = [dict(zip(hdr, r)) for r in figlio["struct"][1:]]
    gem = [r for r in righe if r["stream"] == "gemello" and r["metro"] == "1"][:len(SEQ)]
    ki = figlio["gemello"][-1]["gradino"]["k_ingresso"]
    assert [r["gradino_fase"] for r in gem] == ["P"] * ki + ["ingresso"] + ["N"] * (len(SEQ) - ki - 1)
    assert all(r["gradino_k_ingresso"] == "" for r in gem[:ki])
    assert all(r["gradino_k_ingresso"] == str(ki) for r in gem[ki:])
    e = gem[ki]
    assert float(e["gradino_spost_cm"]) >= S_NODO > float(e["gradino_spost_prima_cm"])
    assert abs(float(e["gradino_buco_stamp_s"]) - 0.1) < 1e-6 and e["gradino_rif_k"] == "0"
    assert gem[ki + 1]["gradino_spost_cm"] == ""
    m0 = [r for r in righe if r["metro"] == "0"]
    assert m0 and all(r["gradino_fase"] == "metro0" for r in m0)


def test_nodo_chamfer_nel_thread_delle_misure():
    """--measures thread: a cloud's Chamfer goes into the sum of its phase only if the
    stream genome has not changed meanwhile."""
    pn = pytest.importorskip("src.nodes.perturbation_node")
    n = pn.PerturbationNode.__new__(pn.PerturbationNode)
    n._meas_lock, n._meas_q, n._meas_ms = threading.Lock(), queue.Queue(), deque(maxlen=10)
    n._shared_nn, n._perc_terms_on, n.genome_id, n._chamfer, n._perc = False, False, 7, float("nan"), None
    n._gim, n._cpf = {"S_cm": 2.0, "kmin": 20, "kmax": 40}, True
    n._gim_acc = {"gemello": n._gim_acc_vuoto(7), "vittima": n._gim_acc_vuoto(3)}
    gen = SimpleNamespace(compute_perturbation_magnitude=lambda c, p, prm, nn=None: 1.25)
    threading.Thread(target=n._meas_loop, daemon=True).start()
    c = np.zeros((20, 4))
    n._perc_measure(gen, c, c, {}, 7, gim=("gemello", 7, "P"))
    n._perc_measure(gen, c, c, {}, 7, gim=("vittima", 3, "N"))
    n._perc_measure(gen, c, c, {}, 7, gim=("vittima", 2, "N"))      # old victim genome
    n._perc_measure(gen, c, c, {}, 7)  # without the option: usual tuple
    n._meas_q.join()
    assert n._gim_acc["gemello"]["P"] == [1.25, 1] and n._gim_acc["gemello"]["N"] == [0.0, 0]
    assert n._gim_acc["vittima"]["N"] == [1.25, 1] and n._chamfer == 1.25


# --- orchestrator ---------------------------------------------------------------------

def _ao():
    from src.optimization import attack_orchestrator as ao
    return ao


@pytest.fixture
def gim_pulito(monkeypatch):
    ao = _ao()
    monkeypatch.setattr(ao, "GIM", {"S": 0.0, "kmin": 20, "kmax": 40, "finestra": None, "cpf": False,
                                    "chamfer_fasi": None})
    monkeypatch.setattr(ao, "RAMPA", {"R": 1, "vittima_ultimo": None, "prec": None, "prec_sha": None,
                                      "zone_prec": None, "metro": None, "sospesa": False, "seq": 0,
                                      "finestra": None, "ultimo": {}, "preinviata": {}, "ultimi_dati": None})
    monkeypatch.setattr(ao, "TWIN_STATE", {"window": 3})
    return ao


def test_orchestratore_argomenti_e_controlli(gim_pulito):
    ao = gim_pulito
    t31 = pytest.importorskip("test_orchestratore_D31")
    base = t31.FISSO + ["--zone-attaccante", "--metro-al-nodo", "--settle-clouds", "20", "--reopen-clouds", "20",
                        "--attack-from", "spawn"]
    a = t31._args_main([])
    assert (a.genoma_in_moto, a.genoma_in_moto_kmin, a.genoma_in_moto_kmax) == (0.0, 20, 40)
    assert a.chamfer_per_fase is False and t31._args_main(["--chamfer-per-fase"]).chamfer_per_fase is True
    assert ao.problema_genoma_in_moto(a) is None
    assert ao.problema_genoma_in_moto(t31._args_main(base + ["--genoma-in-moto", "2.0"])) is None
    assert "--zone-attaccante" in ao.problema_genoma_in_moto(t31._args_main(t31.FISSO + ["--genoma-in-moto", "2"]))
    assert "spawn" in ao.problema_genoma_in_moto(
        t31._args_main(base + ["--genoma-in-moto", "2", "--attack-from", "motion"]))
    assert "--rampa-genoma" in ao.problema_genoma_in_moto(
        t31._args_main(base + ["--genoma-in-moto", "2", "--rampa-genoma", "9"]))
    assert "kmin" in ao.problema_genoma_in_moto(
        t31._args_main(base + ["--genoma-in-moto", "2", "--genoma-in-moto-kmin", "41"]))
    assert ">= 0" in ao.problema_genoma_in_moto(t31._args_main(base + ["--genoma-in-moto", "-1"]))


def test_orchestratore_controllo_contro_lo_status(gim_pulito):
    ao = gim_pulito
    n = types.SimpleNamespace(status_t=1.0, node_gim_on=True, node_gim={"S_cm": 2.0, "kmin": 20, "kmax": 40},
                              spin=lambda s: None)
    assert ao.gim_mismatch(n, 2.0, 20, 40) is None
    assert "S 2.0" in ao.gim_mismatch(n, 2.1, 20, 40, timeout=0.0)
    assert "stessi valori" in ao.gim_mismatch(n, 2.0, 21, 40, timeout=0.0)
    assert "e l'orchestratore no" in ao.gim_mismatch(n, 0.0, 20, 40)
    n0 = types.SimpleNamespace(status_t=1.0, node_gim_on=False, node_gim=None, spin=lambda s: None)
    assert ao.gim_mismatch(n0, 0.0, 20, 40) is None and "spento" in ao.gim_mismatch(n0, 2.0, 20, 40, timeout=0.0)


def test_orchestratore_chamfer_per_fase(gim_pulito):
    """--chamfer-per-fase without step: chamfer_fasi record per stream, no genoma_in_moto
    record; off: no record; check against the status."""
    ao = gim_pulito
    sc = {"genome_id": 3, "chamfer_P_somma": 0.0, "n_chamfer_P": 0, "chamfer_N_somma": 7.5, "n_chamfer_N": 3}
    n = types.SimpleNamespace(status_t=time.time() + 5, node_cpf={"gemello": sc, "vittima": sc},
                              node_gim=None, spin=lambda s: None)
    assert ao.gim_inizio_finestra() is None and ao.GIM["chamfer_fasi"] is None
    assert ao.gim_registra(n, "gemello", "w2_e0") is None and ao.GIM["chamfer_fasi"] is None
    ao.GIM["cpf"] = True
    assert ao.gim_inizio_finestra() is None and ao.GIM["chamfer_fasi"] == {} and ao.GIM["finestra"] is None
    assert ao.gim_registra(n, "gemello", "w2_e0") is None
    ao.gim_registra(n, "vittima", "vittima")
    assert ao.GIM["chamfer_fasi"] == {
        nome: {"chamfer_P_cm": None, "n_chamfer_P": 0, "chamfer_N_cm": 2.5, "n_chamfer_N": 3}
        for nome in ("w2_e0", "vittima")}
    on = types.SimpleNamespace(node_cpf_on=True, spin=lambda s: None)
    off = types.SimpleNamespace(node_cpf_on=False, spin=lambda s: None)
    assert ao.cpf_mismatch(on, True) is None and ao.cpf_mismatch(off, False) is None
    assert "status senza chamfer_fasi" in ao.cpf_mismatch(off, True, timeout=0.0)
    assert "e l'orchestratore no" in ao.cpf_mismatch(on, False)


def test_orchestratore_record_e_chamfer(gim_pulito):
    ao = gim_pulito
    sg = {"stato": "attivo", "fase": "N", "k_ingresso": 26, "spost_ingresso_cm": 2.4, "spost_prima_cm": 1.7,
          "buco_stamp_s": 0.1, "rif_k": 21, "chamfer_P_somma": 3.0, "n_chamfer_P": 2,
          "chamfer_N_somma": 9.0, "n_chamfer_N": 4}
    r = ao.gim_flusso(sg, "P")
    assert r == {"stato": "attivo", "k_ingresso": 26, "spost_ingresso_cm": 2.4, "spost_prima_cm": 1.7,
                 "buco_stamp_s": 0.1, "rif_k": 21, "fase_al_salvataggio_D31c": "P",
                 "chamfer_P_cm": 1.5, "n_chamfer_P": 2, "chamfer_N_cm": 2.25, "n_chamfer_N": 4}
    r0 = ao.gim_flusso({"stato": "metro0", "chamfer_P_somma": 0.0, "n_chamfer_P": 0,
                        "chamfer_N_somma": 2.0, "n_chamfer_N": 1})
    assert r0["chamfer_P_cm"] is None and r0["n_chamfer_P"] == 0 and r0["chamfer_N_cm"] == 2.0
    assert r0["k_ingresso"] is None and r0["fase_al_salvataggio_D31c"] is None
    assert [ao.gim_fase_salvataggio({"gradino_fase": f}) for f in ("P", "ingresso", "N", "metro0", None)] == \
        ["P", "N", "N", None, None]
    assert ao.gim_fase_salvataggio(None) is None
    # window log from the status (off: nothing)
    n = types.SimpleNamespace(status_t=time.time() + 5, node_gim={"gemello": sg}, spin=lambda s: None)
    assert ao.gim_registra(n, "gemello", "w3_e0") is None
    ao.GIM["S"], ao.GIM["cpf"] = 2.0, True
    n.node_cpf = {"gemello": sg}
    fin = ao.gim_inizio_finestra()
    assert fin == {"S_cm": 2.0, "kmin": 20, "kmax": 40, "flussi": {}}
    ao.gim_registra(n, "gemello", "w3_e0")
    assert fin["flussi"]["w3_e0"]["k_ingresso"] == 26 and fin["flussi"]["w3_e0"]["chamfer_N_cm"] == 2.25
    assert ao.GIM["chamfer_fasi"]["w3_e0"] == {"chamfer_P_cm": 1.5, "n_chamfer_P": 2, "chamfer_N_cm": 2.25,
                                               "n_chamfer_N": 4}
    assert set(fin["flussi"]["w3_e0"]) == {"stato", "k_ingresso", "spost_ingresso_cm", "spost_prima_cm",
                                           "buco_stamp_s", "rif_k", "fase_al_salvataggio_D31c", "chamfer_P_cm",
                                           "n_chamfer_P", "chamfer_N_cm", "n_chamfer_N"}


class _NodoGim:
    """Fake node with --genoma-in-moto: received in genoma_in_moto.ricevuta."""

    def __init__(self, ao):
        self.node_gim = {"S_cm": 2.0, "ricevuta": {}}
        self.mandati = []
        self.manda_rampa = types.MethodType(ao.AttackNode.manda_rampa, self)

    def _zone_pub(self, topic):
        return topic, types.SimpleNamespace

    def _publish_confirmed(self, pub, msg, confirmed, what, **kw):
        d = json.loads(msg.data)
        self.mandati.append(d)
        if d.get("annulla"):
            self.node_gim["ricevuta"][d["flusso"]] = {"seq": d["seq"], "stato": "annullata",
                                                       "genoma_sha_nuovo": None}
        else:
            self.node_gim["ricevuta"][d["flusso"]] = {"seq": d["seq"], "stato": "ok",
                                                       "genoma_sha_nuovo": d["genoma_sha_nuovo"]}
        return confirmed()


def test_orchestratore_canale_della_rampa(gim_pulito, monkeypatch):
    """With the step the previous genome goes to the node as for the ramp; off: no message;
    meter 0 without P; the realigned run (sospesa) gets the cancel only if the node holds
    data that would apply to it."""
    ao = gim_pulito
    za = tr._moduli()[1]
    N, P = [0.5] * 17, [0.2] * 17
    nodo = _NodoGim(ao)
    assert nodo.manda_rampa("gemello", N) is None and nodo.mandati == []       # off
    ao.GIM["S"] = 2.0
    assert ao.canale_rampa()
    fin = ao.prepara_rampa_metro(None, 0)
    assert fin["motivo"] == "metro 0" and ao.RAMPA["prec"] is None
    ao.TWIN_STATE["window"] = 0
    assert nodo.manda_rampa("gemello", N)["esito"] == "senza_precedente" and nodo.mandati == []
    ao.RAMPA.update({"prec": list(P), "prec_sha": za.genoma_sha(P), "metro": 3,
                     "zone_prec": {"path": "/z/w3_p.npz", "sha256": "a" * 64, "hash": "b" * 64},
                     "finestra": {"messaggi": []}})
    ao.TWIN_STATE["window"] = 3
    rec = nodo.manda_rampa("gemello", N)
    assert rec["esito"] == "mandata" and nodo.mandati[-1]["genoma_sha_nuovo"] == za.genoma_sha(N)
    ao.RAMPA["sospesa"] = True
    rec = nodo.manda_rampa("gemello", N)
    assert rec["esito"] == "annullata" and nodo.mandati[-1] == {"seq": nodo.mandati[-1]["seq"], "flusso": "gemello",
                                                               "annulla": True}
    n = len(nodo.mandati)
    assert nodo.manda_rampa("gemello", N)["esito"] == "sospesa" and len(nodo.mandati) == n


def test_orchestratore_registro_controllore_stamp_grezzo():
    """_registra_ciclo passes the raw pose stamp and the offset to the log."""
    ao = _ao()
    chiamate = []
    reg = types.SimpleNamespace(ciclo=lambda *a, **k: chiamate.append((a, k)), _disattiva=lambda *a: None)
    n = types.SimpleNamespace(registri=reg, est_source="victim", est_stamp=24.6, est_stamp_raw=14.6, est_t=1.0,
                              _t_sim=lambda: 24.7)
    ao.AttackNode._registra_ciclo(n, 1.0, 1.1, (0.0, 0.0, 0.0), 0.0, 0.8, 0.0, "crociera")
    a, k = chiamate[0]
    assert a[0] == "vittima" and k["stamp_grezzo"] == 14.6 and abs(k["offset"] - 10.0) < 1e-12
    n.est_stamp, n.est_stamp_raw = float("nan"), float("nan")
    ao.AttackNode._registra_ciclo(n, 1.0, 1.1, None, 0.0, 0.0, 0.0, "stop")
    assert chiamate[1][1]["offset"] != chiamate[1][1]["offset"]  # NaN: empty field in the log


# --- launchers ------------------------------------------------------------------------

def _blocco_rc():
    """run_campaign.sh lines from RAMPA= to the genome-in-motion block included."""
    testo = (ROOT / "scripts/run_campaign.sh").read_text().split("\n")
    i = next(n for n, r in enumerate(testo) if r.startswith('RAMPA="${RAMPA:-}"'))
    j = next(n for n, r in enumerate(testo) if n > i and r.startswith('case "$CHAMFER_PER_FASE" in'))
    k = next(n for n, r in enumerate(testo) if n > j and r.endswith("esac"))
    suff = [r for r in testo if "_gim${GENOMA_IN_MOTO}" in r]
    assert len(suff) == 1
    return "\n".join(testo[i:k + 1] + ["MODE_SUFFIX=''"] + suff)


@pytest.mark.parametrize("env,flags,nodo,suffisso", [
    ({}, "", "", ""), ({"GENOMA_IN_MOTO": "0"}, "", "", ""), ({"GENOMA_IN_MOTO": "0.0"}, "", "", ""),
    ({"GENOMA_IN_MOTO": "2.0"}, " --genoma-in-moto 2.0", " --genoma-in-moto 2.0", "_gim2.0"),
    ({"GENOMA_IN_MOTO": "2.0", "RAMPA": "1"}, " --genoma-in-moto 2.0", " --genoma-in-moto 2.0", "_gim2.0"),
    ({"CHAMFER_PER_FASE": "0"}, "", "", ""),
    ({"CHAMFER_PER_FASE": "1"}, " --chamfer-per-fase", " --chamfer-per-fase", ""),
    ({"GENOMA_IN_MOTO": "2.0", "CHAMFER_PER_FASE": "1"}, " --genoma-in-moto 2.0 --chamfer-per-fase",
     " --genoma-in-moto 2.0 --chamfer-per-fase", "_gim2.0")])
def test_run_campaign_genoma_in_moto(env, flags, nodo, suffisso):
    prog = f'set -e\n{_blocco_rc()}\necho "[$D31_FLAGS][$RAMPA_NODO][$MODE_SUFFIX]"'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], **env})
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == f"[{flags}][{nodo}][{suffisso}]"


@pytest.mark.parametrize("env,msg", [({"GENOMA_IN_MOTO": "2,0"}, "GENOMA_IN_MOTO: soglia"),
                                     ({"GENOMA_IN_MOTO": "-2"}, "GENOMA_IN_MOTO: soglia"),
                                     ({"GENOMA_IN_MOTO": "2.0", "RAMPA": "9"}, "GENOMA_IN_MOTO e RAMPA"),
                                     ({"CHAMFER_PER_FASE": "si"}, "CHAMFER_PER_FASE: 0 o 1")])
def test_run_campaign_genoma_in_moto_rifiutato(env, msg):
    prog = f'set -e\n{_blocco_rc()}\necho fine'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], **env})
    assert r.returncode == 1 and msg in r.stdout and "fine" not in r.stdout


def test_run_campaign_passa_il_valore_al_nodo():
    testo = (ROOT / "scripts/run_campaign.sh").read_text()
    assert '[[ -n "$RAMPA_NODO" ]] && extra="$extra$RAMPA_NODO"' in testo
    assert 'GENOMA_IN_MOTO=$GENOMA_IN_MOTO, --genoma-in-moto $GENOMA_IN_MOTO a nodo e orchestratore' in testo
    assert "CHAMFER_PER_FASE=1, --chamfer-per-fase a nodo e orchestratore" in testo
