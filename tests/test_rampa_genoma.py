#!/usr/bin/env python3
"""Genome ramp towards the new genome (node --rampa-genoma; generator imposta_rampa and
_coherent_pesato). No Isaac or MOLA: synthetic scene (floor, walls, poles, boxes; robot still,
2 mm noise between clouds) with snapshot and attacker zones computed in a temporary folder.
The real node runs in a separate process (this file with --figlio, domain 88, no spin), as in
tests/zone_nodo.py. The test on the real w4 snapshot (e465 -> e480) is skipped if the files
are missing.

    ROS_DOMAIN_ID=88 python3 -B -m pytest -q -p no:cacheprovider tests/test_rampa_genoma.py

Reference for the option off: an untouched copy of src/ without the ramp (RAMPA_ORIGINALE).
"""

import functools
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINALE = Path(os.environ.get("RAMPA_ORIGINALE",
                                os.path.expanduser("~/mola_lavoro/lavoro_rampa_A3_originale")))
REP = ROOT / ("data/attack/serieB/nsga3_senzabudget/s1001_accettazione_30ott/campagna/straight/"
              "nsga3_g3_p32x16_lag0.5_rkeep_sel2_warm_stopramp_k1.0_reliable_serieB_s1001_pt1_opinsieme_"
              "sgross_coda_N5_dmin140/rep_01")
LIM = {"max_point_shift": 0.05, "noise_std": 0.02, "max_dropout_rate": 0.15, "max_ghost_points_ratio": 0.02}
NODO = ["--max-drift-lag-s", "0.5", "--perc-terms", "--structural", "gross",
        "--operator-mode", "insieme", "--zone-attaccante"]
R_PROVA = 4          # ramp of the synthetic tests
NK = R_PROVA + 3     # clouds per flow after the genome change
DERIVA0 = (0.0, 0.12, 0.0)
Z0 = -0.526          # floor in the sensor frame


# ----------------------------------------------------------------------
# Scene and genomes (shared by the pytest process and the children)
# ----------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def scena(i):
    """Synthetic cloud i (read-only): same scene, different 2 mm noise for each i."""
    base = np.random.default_rng(7)
    parti = []
    n = 7000
    parti.append(np.c_[base.uniform(-9, 9, n), base.uniform(-5, 5, n), np.full(n, Z0)])
    for y in (-5.0, 5.0):
        n = 4000
        parti.append(np.c_[base.uniform(-8, 9, n), np.full(n, y), base.uniform(Z0, 2.0, n)])
    n = 2500
    parti.append(np.c_[np.full(n, 9.0), base.uniform(-5, 5, n), base.uniform(Z0, 2.0, n)])
    for cx, cy in ((3, 2), (5, -2.5), (-3, 3), (6, 3.5), (-5, -2), (2, -3.5), (7, 0.5), (-6, 1)):
        n = 500
        a = base.uniform(0, 2 * np.pi, n)
        parti.append(np.c_[cx + 0.06 * np.cos(a), cy + 0.06 * np.sin(a), base.uniform(Z0, 1.6, n)])
    for cx, cy in ((4.0, 0.5), (-4.0, -1.0)):
        n = 2400
        f = base.integers(0, 5, n)
        u, v = base.uniform(0, 1, n), base.uniform(0, 1, n)
        x = np.where(f == 0, 0.0, np.where(f == 1, 1.0, u))
        y = np.where(f == 2, 0.0, np.where(f == 3, 1.0, np.where(f < 2, u, v)))
        z = np.where(f == 4, 1.0, np.where(f < 4, v, 0))
        parti.append(np.c_[cx + x, cy + y, Z0 + z])
    xyz = np.concatenate(parti)
    xyz = xyz + np.random.default_rng(1000 + i).normal(0, 0.002, xyz.shape)
    c = np.c_[xyz, base.uniform(0, 255, len(xyz))].astype(np.float32).astype(np.float64)
    c.setflags(write=False)
    return c


def _genoma(d):
    g = np.full(17, -1.0)
    for k, v in d.items():
        g[k] = v
    return np.asarray(g, dtype=np.float32).astype(np.float64)


# P: noise, g5, g6 on the walls, g10 (balls), g12, drift. N: noise, g5, g16, g13, g12, drift, g15.
GP = _genoma({0: 1.0, 1: 0.2, 2: 0.0, 3: 0.2, 4: 0.6, 5: 0.2, 6: 0.8, 7: 0.3, 8: 1.0, 9: 0.0, 10: 0.6,
              11: 0.5, 12: -0.2, 14: -0.4, 16: -0.6})
GN = _genoma({0: 0.1, 1: 1.0, 2: 0.0, 3: 0.4, 4: -0.5, 5: 0.0, 6: 0.6, 7: 1.0, 8: 0.2, 9: 0.0, 10: -1.0,
              11: 0.0, 12: 0.2, 13: 0.0, 14: -0.6, 15: 0.0, 16: 0.9})


def _md5(a):
    return None if a is None else hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------
# Child process: real node with callbacks called by hand
# ----------------------------------------------------------------------

VOLATILI = ("proc_ms_mean", "proc_ms_max", "proc_ms_last", "latency_ms_mean", "latency_ms_max")


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
    node_args += ["--struct-log", str(lavoro / "struct.csv")]

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
    node._out_q = queue.Queue()        # the publishing thread stays blocked on the old queue
    seqn = [0]

    def seq():
        seqn[0] += 1
        return seqn[0]

    def msg_di(c, s):
        return _msg(pn, c, s)

    class _P:
        def publish(self, m):
            self.m = m

    def stato():
        node.status_pub = _P()
        if not node.lat_times:
            node.lat_times.append(0.01)
        pn.PerturbationNode._publish_status_body(node)
        d = json.loads(node.status_pub.m.data.replace(str(lavoro), "<DIR>"))
        for k in VOLATILI:
            d.pop(k, None)
        return d

    def nuvola(flusso, c, s):
        if flusso == "gemello":
            node.victim_stream, node.enabled = False, True
        else:
            node.victim_stream, node.victim_attack = True, True
        node.sub.callback(msg_di(c, s))          # the callback as subscribed
        item = node._out_q.get_nowait()
        o = item[4] if flusso == "vittima" else item[2]
        gen = node.victim_generator if flusso == "vittima" else node.generator
        inf = gen.last_struct_info or {}
        return {"flusso": flusso, "md5": _md5(o), "n": int(len(o)), "k": inf.get("k"),
                "n_hit": inf.get("n_hit"), "deriva": [float(x) for x in gen.accumulated_drift],
                "stato": stato()}

    def manda_zone(flusso, z):
        node._zone_cb(String(data=json.dumps({"seq": seq(), "flusso": flusso, "path": str(z.path),
                                              "sha256": z.sha256, "hash": z.hash,
                                              "genoma_sha": z.genoma_sha})))

    def manda_rampa(flusso, g_prec, sha_nuovo, z_prec, **sovr):
        d = {"seq": seq(), "flusso": flusso, "genoma_prec": [float(x) for x in g_prec],
             "genoma_sha_nuovo": sha_nuovo,
             "zone_prec": {"path": str(z_prec.path), "sha256": z_prec.sha256, "hash": z_prec.hash}}
        d.update(sovr)
        node._rampa_cb(String(data=json.dumps(d)))

    def metro(flusso, w):
        node._metro_cb(String(data=json.dumps({flusso: w})))

    node._drift_keep_cb(Bool(data=True))
    s_foto = 9.0
    node._att_buf.append((pn._stamp_s(msg_di(scena(-1), s_foto).header), scena(-1)))
    node._foto_cb(String(data=json.dumps({
        "seq": 1, "stamp": pn._stamp_s(msg_di(scena(-1), s_foto).header), "path": str(zd / "w1_fotografia.npz"),
        "meta": {"metro": 1, "posa_base_odom": [0.0, 0.0, 0.0]}})))
    fs = node._foto_stato
    assert fs["esito"] == "ok", fs
    foto = za.carica_fotografia(zd / "w1_fotografia.npz", fs["sha256"], fs["hash"])
    limiti = {k: float(getattr(node.generator, k)) for k in za.LIMITI}
    zP, zN = (za.zone_genoma(zd, foto, g, 1, limiti) for g in (GP, GN))
    sha_n = za.genoma_sha(GN)
    rampa = scenario in ("rampa", "errori")
    righe = []
    # victim with P (victim drift, from which the twin starts with keep)
    metro("vittima", 1)
    manda_zone("vittima", zP)
    node._victim_genome_cb(SimpleNamespace(data=list(GP)))
    for i in range(3):
        righe.append(nuvola("vittima", scena(100 + i), 1.0 + 0.1 * i))
    casi = {}
    if scenario == "errori":
        # twin with N: cases without a valid ramp, then a valid one
        za2 = za.salva_fotografia(zd / "w2_fotografia.npz", scena(-2), {"metro": 2, "posa_base_odom": [0, 0, 0]})
        foto2 = za.carica_fotografia(zd / "w2_fotografia.npz", *za2)
        zP2 = za.zone_genoma(zd, foto2, GP, 2, limiti)
        varianti = [("assente", None),
                    ("altro_genoma", lambda: manda_rampa("gemello", GP, za.genoma_sha(GP), zP)),
                    ("uguale", lambda: manda_rampa("gemello", GN, sha_n, zN)),
                    ("altro_metro", lambda: manda_rampa("gemello", GP, sha_n, zP2)),
                    ("sha256", lambda: manda_rampa("gemello", GP, sha_n, zP,
                                                   zone_prec={"path": str(zP.path), "sha256": "0" * 64,
                                                              "hash": zP.hash})),
                    # malformed message with a readable flow: 3-value genome
                    ("malformato", lambda: manda_rampa("gemello", [0.5] * 3, sha_n, zP)),
                    # valid data for N, then cancelled
                    ("annullata", lambda: (manda_rampa("gemello", GP, sha_n, zP),
                                           node._rampa_cb(String(data=json.dumps(
                                               {"seq": seq(), "flusso": "gemello", "annulla": True}))))),
                    ("valida", lambda: manda_rampa("gemello", GP, sha_n, zP))]
        # message with an unreadable flow: no receipt touched, reason in the status
        node._rampa_cb(String(data=json.dumps({"seq": 990, "flusso": "ombra"})))
        for nome, invia in varianti:
            if invia is not None:
                invia()
            metro("gemello", 1)
            manda_zone("gemello", zN)
            node._set_genome(GN, source="prova")
            casi[nome] = [nuvola("gemello", scena(i), 2.0 + 0.1 * i) for i in range(2)]
    else:
        if rampa:
            manda_rampa("gemello", GP, sha_n, zP)
        metro("gemello", 1)
        manda_zone("gemello", zN)
        node._set_genome(GN, source="prova")
        for i in range(NK):
            righe.append(nuvola("gemello", scena(i), 2.0 + 0.1 * i))
        # victim with N, same clouds as the twin
        if rampa:
            manda_rampa("vittima", GP, sha_n, zP)
        manda_zone("vittima", zN)
        node._victim_genome_cb(SimpleNamespace(data=list(GN)))
        for i in range(NK):
            righe.append(nuvola("vittima", scena(i), 4.0 + 0.1 * i))
    node._struct_log.close()
    with open(node._struct_log.path) as f:
        struct = [r.rstrip("\n").split(",") for r in f]
    json.dump({"righe": righe, "casi": casi, "struct": struct,
               "zone": {"P": zP.hash, "N": zN.hash}}, open(uscita, "w"))
    node.destroy_node()
    rclpy.shutdown()


def _msg(pn, c, s):
    from sensor_msgs.msg import PointCloud2
    m = PointCloud2()
    m.header.frame_id = "base_link"
    m.header.stamp.sec = int(s)
    m.header.stamp.nanosec = int(round((s - int(s)) * 1e9)) % 1000000000
    m.height, m.width = 1, len(c)
    m.fields = pn.FIELDS
    m.is_bigendian = False
    m.point_step = pn.POINT_STEP
    m.row_step = pn.POINT_STEP * len(c)
    m.data = np.ascontiguousarray(c[:, :4], dtype="<f4").tobytes()
    m.is_dense = True
    return m


if __name__ == "__main__" and "--figlio" in sys.argv:
    _figlio(sys.argv[sys.argv.index("--figlio") + 1:])
    sys.exit(0)


# ----------------------------------------------------------------------
# Tests (pytest)
# ----------------------------------------------------------------------

import pytest  # noqa: E402


def _moduli():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.perturbations import perturbation_generator as pg
    from src.perturbations import zone_attaccante as za
    return pg, za


def _nodo_figlio(tmp_path, nome, radice, scenario, args):
    out = tmp_path / f"{nome}.json"
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--figlio", str(radice), scenario, str(out),
           "--"] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, ROS_DOMAIN_ID="88"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def Z(tmp_path_factory):
    """Snapshot of metre 1 (and metre 2 for P), zones of P and N, parameters and seeds."""
    pg, za = _moduli()
    d = tmp_path_factory.mktemp("rampa_zone")
    s1 = za.salva_fotografia(d / "w1_fotografia.npz", scena(-1), {"metro": 1, "posa_base_odom": [0, 0, 0]})
    s2 = za.salva_fotografia(d / "w2_fotografia.npz", scena(-2), {"metro": 2, "posa_base_odom": [0, 0, 0]})
    f1 = za.carica_fotografia(d / "w1_fotografia.npz", *s1)
    f2 = za.carica_fotografia(d / "w2_fotografia.npz", *s2)
    zz = {nome: za.zone_genoma(d, f, g, w, LIM) for nome, f, g, w in
          (("P", f1, GP, 1), ("N", f1, GN, 1), ("P2", f2, GP, 2))}
    cache = {}
    zone = {nome: za.carica_zone(z.path, z.sha256, z.hash, cache=cache) for nome, z in zz.items()}
    g = pg.PerturbationGenerator(**LIM)
    return {"pg": pg, "za": za, "zP": zone["P"], "zN": zone["N"], "zP2": zone["P2"],
            "pP": g.encode_perturbation(GP), "pN": g.encode_perturbation(GN),
            "sP": za.seme_genoma(GP), "sN": za.seme_genoma(GN),
            "shaP": za.genoma_sha(GP), "shaN": za.genoma_sha(GN)}


def _gen(Z, zona, deriva=DERIVA0):
    pg, za = Z["pg"], Z["za"]
    g = pg.PerturbationGenerator(max_point_shift=LIM["max_point_shift"], noise_std=LIM["noise_std"],
                                 max_dropout_rate=LIM["max_dropout_rate"], max_drift_lag_s=0.5,
                                 structural="gross", zone_attaccante=True, **za.KW_INSIEME)
    g.zone_fornitore = lambda s_, m_, p_: (zona, "ok")
    g.keep_drift = True
    g.set_temporal_state(np.asarray(deriva, dtype=np.float64))
    return g


def _prec(Z, zona="zP"):
    return {"params": Z["pP"], "seme": Z["sP"], "zone": Z[zona], "sha": Z["shaP"]}


def _gen_rampa(Z, R, deriva=DERIVA0, zona="zP"):
    g = _gen(Z, Z["zN"], deriva)
    g.imposta_rampa(R, _prec(Z, zona), genoma_sha=Z["shaN"], seq=1)
    return g


def _passo(g, params, seme, i, metro=1):
    o = g.apply_perturbation(scena(i), params, ctx={"stamp": 2.0 + 0.1 * i}, seme_genoma=seme, metro=metro)
    inf = g.last_struct_info
    return {"out": o, "md5": _md5(o), "n_hit": dict(inf["n_hit"]), "attesi": dict(inf.get("zone_attesi") or {}),
            "k": inf["k"], "deriva": g.accumulated_drift, "hits": {k: v.copy() for k, v in g.last_hits.items()},
            "counts": dict(g._last_counts[1]),
            "rampa": None if g.rampa_info is None else dict(g.rampa_info)}


# ----------------------------------------------------------------------
# Pure functions and defaults
# ----------------------------------------------------------------------

def test_alfa_e_conteggi_puri():
    pg, _ = _moduli()
    for R in (2, 3, 4, 9, 12):
        assert pg.rampa_alfa(0, R) == 1.0 / R
        assert pg.rampa_alfa(R - 2, R) == (R - 1) / R
        assert pg.rampa_alfa(R - 1, R) == 1.0 and pg.rampa_alfa(R + 5, R) == 1.0
    assert pg.rampa_alfa(0, 1) == 1.0
    # round half up, exact (integer arithmetic)
    assert pg.rampa_conta(1, 2, 1) == 1 and pg.rampa_conta(1, 4, 2) == 1 and pg.rampa_conta(3, 4, 2) == 2
    assert pg.rampa_conta(1, 3, 1) == 0 and pg.rampa_conta(2, 3, 1) == 1
    assert pg.rampa_conta(8, 9, 806) == 716 and pg.rampa_conta(1, 9, 783) == 87
    assert pg.rampa_conta(5, 9, 0) == 0 and pg.rampa_conta(5, 9, -3) == 0
    for R in range(2, 13):
        for T in list(range(0, 60)) + [499, 500, 783, 806, 1315, 1505]:
            serie = [pg.rampa_conta(m, R, T) for m in range(R + 1)]
            assert serie[0] == 0 and serie[R] == T
            for m, v in enumerate(serie):
                assert v == math.floor(Fraction(m * T, R) + Fraction(1, 2))
            # max step per cloud: ceil(T / R), non-decreasing series
            passi = np.diff(serie)
            assert passi.min() >= 0 and passi.max() <= math.ceil(T / R)


def test_default_spento_generatore(Z):
    pg = Z["pg"]
    g = pg.PerturbationGenerator()
    assert g.rampa_spec is None and g.rampa_info is None and g._rampa is None
    gz = _gen(Z, Z["zN"])
    with pytest.raises(ValueError):
        gz.imposta_rampa(1, _prec(Z))            # R = 1 is the ramp off: the node does not arm it
    gi = pg.PerturbationGenerator(**LIM, **{k: v for k, v in Z["za"].KW_INSIEME.items()})
    with pytest.raises(ValueError):
        gi.imposta_rampa(4, _prec(Z))            # without attacker zones
    gz.imposta_rampa(4, None, "prec_assente")    # no previous genome: no ramp, only the state
    assert gz.rampa_spec is None and gz.rampa_info["stato"] == "prec_assente"


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

    assert cli([]).rampa_genoma == 1
    assert cli(["--rampa-genoma", "9"]).rampa_genoma == 9
    assert pn.PerturbationNode._rampa_R == 1
    rclpy.init()
    try:
        topic = {}
        for nome, argv in (("spento", NODO), ("uno", NODO + ["--rampa-genoma", "1"]),
                           ("acceso", NODO + ["--rampa-genoma", "4"])):
            n = pn.PerturbationNode(cli(argv))
            topic[nome] = sorted(s.topic_name for s in n.subscriptions)
            assert n._rampa_R == (4 if nome == "acceso" else 1)
            n.destroy_node()
        assert "/attack/rampa" not in topic["spento"] and topic["uno"] == topic["spento"]
        assert topic["acceso"] == sorted(topic["spento"] + ["/attack/rampa"])
        for argv in (["--rampa-genoma", "4"], ["--operator-mode", "insieme", "--rampa-genoma", "4"],
                     NODO + ["--rampa-genoma", "0"]):
            with pytest.raises(ValueError):
                pn.PerturbationNode(cli(argv))
    finally:
        rclpy.shutdown()


# ----------------------------------------------------------------------
# Option off: node identical to the untouched copy
# ----------------------------------------------------------------------

@pytest.mark.skipif(not (ORIGINALE / "src/nodes/perturbation_node.py").exists(), reason="copia originale assente")
def test_spento_identico_alla_copia_originale(tmp_path):
    """Ramp off: outputs, status, zone log and files identical to the untouched copy."""
    a = _nodo_figlio(tmp_path, "orig", ORIGINALE, "spento", NODO)
    b = _nodo_figlio(tmp_path, "nuovo", ROOT, "spento", NODO)
    c = _nodo_figlio(tmp_path, "uno", ROOT, "spento", NODO + ["--rampa-genoma", "1"])
    assert len(a["righe"]) == 3 + 2 * NK
    for x in (b, c):
        assert x["zone"] == a["zone"]
        assert [r["md5"] for r in x["righe"]] == [r["md5"] for r in a["righe"]]
        assert [r["stato"] for r in x["righe"]] == [r["stato"] for r in a["righe"]]
        assert [r["deriva"] for r in x["righe"]] == [r["deriva"] for r in a["righe"]]
        # zone log: same header, same rows except the arrival time
        assert x["struct"][0] == a["struct"][0]
        assert [r[1:] for r in x["struct"][1:]] == [r[1:] for r in a["struct"][1:]]
    assert all("rampa" not in r["stato"] for r in b["righe"] + c["righe"])


# ----------------------------------------------------------------------
# Generator: identity at the extremes, ramp end, counts, alpha
# ----------------------------------------------------------------------

def _uguali(a, b):
    assert a["md5"] == b["md5"], (a["k"], a["n_hit"], b["n_hit"])
    assert np.array_equal(a["deriva"], b["deriva"])


def test_estremi_un_solo_genoma(Z):
    """_coherent_pesato with one term at weight 1 gives that genome alone, bit for bit."""
    pg = Z["pg"]
    pre = pg.RAMPA_PREFISSO
    for solo in ("N", "P"):
        ref = _gen(Z, Z["zN"] if solo == "N" else Z["zP"])
        g = _gen_rampa(Z, 1000)                  # ramp that does not end within the test clouds
        w = 7

        def pesato(pc, pert, fr, p, seme, metro, g=g, solo=solo):
            rp = g._rampa
            termini = [{"pre": pre, "p": rp["params"], "seme": rp["seme"], "ob": rp["obiettivi"],
                        "zi": rp["zi"], "m": w if solo == "P" else 0},
                       {"pre": "", "p": p, "seme": seme, "ob": g._obiettivi, "zi": g._zone_int,
                        "m": w if solo == "N" else 0}]
            return g._coherent_pesato(pc, pert, fr, termini, w, metro)

        g._apply_coherent_rampa = pesato
        pr, sr = (Z["pN"], Z["sN"]) if solo == "N" else (Z["pP"], Z["sP"])
        for i in range(6):
            a = _passo(ref, pr, sr, i)
            b = _passo(g, Z["pN"], Z["sN"], i)
            _uguali(a, b)
            nh = {(k[len(pre):] if k.startswith(pre) else k): v for k, v in b["n_hit"].items()}
            assert nh == a["n_hit"]
            if solo == "P":
                assert all(k.startswith(pre) for k in b["n_hit"])


def test_rampa_attiva_poi_identica_al_solo_N(Z):
    """During the ramp the output changes; from k = R - 1 it equals N alone, same drift."""
    pre = Z["pg"].RAMPA_PREFISSO
    for R in (2, R_PROVA):
        ref, g = _gen(Z, Z["zN"]), _gen_rampa(Z, R)
        for i in range(R + 3):
            if i >= R - 1:
                ref.set_temporal_state(g.accumulated_drift)
            a = _passo(ref, Z["pN"], Z["sN"], i)
            b = _passo(g, Z["pN"], Z["sN"], i)
            assert a["k"] == b["k"] == i
            if i < R - 1:
                assert a["md5"] != b["md5"]
                assert any(k.startswith(pre) for k in g._structs.anchors)
            else:
                _uguali(a, b)
                assert b["n_hit"] == a["n_hit"] and b["attesi"] == a["attesi"]
                assert not any(k.startswith(pre) for k in g._structs.anchors)
        ta, tb = ref._structs.anchors, g._structs.anchors
        assert sorted(ta) == sorted(tb)
        for op in ta:
            for A, B in zip(ta[op], tb[op]):
                assert np.array_equal(A["keys"], B["keys"]) and A["lost"] == B["lost"]


def test_alfa_ai_bordi_e_stato(Z):
    for R in (2, R_PROVA):
        g = _gen_rampa(Z, R)
        assert g.rampa_info["stato"] == "armata" and g.rampa_info["k"] is None
        for i in range(R + 1):
            ri = _passo(g, Z["pN"], Z["sN"], i)["rampa"]
            assert ri["k"] == i and ri["alpha"] == min(1.0, (i + 1) / R)
            assert ri["stato"] == ("attiva" if i < R - 1 else "finita")
            assert ri["genoma_prec_sha"] == Z["shaP"] and ri["genoma_sha"] == Z["shaN"]
        assert _passo(g, Z["pN"], Z["sN"], R + 1)["rampa"]["alpha"] == 1.0


def test_conteggi_scalino_e_deriva(Z):
    """Per-genome rampa_conta points, step <= ceil(T / R) per cloud, removed set, drift."""
    pg = Z["pg"]
    pre = pg.RAMPA_PREFISSO
    R = R_PROVA
    gP, gN, g = _gen(Z, Z["zP"]), _gen(Z, Z["zN"]), _gen_rampa(Z, R)
    prima = None
    serie = {}
    for i in range(R):
        a, b = _passo(gP, Z["pP"], Z["sP"], i), _passo(gN, Z["pN"], Z["sN"], i)
        acc0 = g.accumulated_drift
        c = _passo(g, Z["pN"], Z["sN"], i)
        mP, mN = R - i - 1, i + 1
        for op in ("g5", "g6"):
            T = a["attesi"][op]
            v = c["attesi"].get(pre + op, 0)
            assert v == pg.rampa_conta(mP, R, T)
            serie.setdefault(pre + op, []).append((v, T))
        for op in ("g5", "g16"):
            T = b["attesi"][op]
            assert c["attesi"][op] == pg.rampa_conta(mN, R, T)
            serie.setdefault(op, []).append((c["attesi"][op], T))
        if i == R - 1:
            break                                # k = R - 1: N alone (separate test)
        # displacement operators: same points as without the ramp (g10 of P, g13 of N)
        assert c["n_hit"][pre + "g10"] == a["n_hit"]["g10"] and c["n_hit"]["g13"] == b["n_hit"]["g13"]
        for key in (pre + "g6", "g16", pre + "g5", "g5"):
            assert c["n_hit"][key] == c["attesi"][key]          # in the scene the zones suffice
        tolti = np.union1d(c["hits"][pre + "g5"], c["hits"]["g5"])
        assert c["counts"]["n_removed"] == len(tolti) <= c["n_hit"][pre + "g5"] + c["n_hit"]["g5"]
        assert c["counts"]["n_added"] == c["n_hit"][pre + "g6"] + c["n_hit"]["g16"]
        fP = np.asarray(Z["pP"]["noise_direction"]) * Z["pP"]["temporal_drift_strength"] * 0.05
        fN = np.asarray(Z["pN"]["noise_direction"]) * Z["pN"]["temporal_drift_strength"] * 0.05
        att = acc0 * 0.98 + (mP / R) * fP + (mN / R) * fN
        assert np.allclose(c["deriva"], att, rtol=0, atol=1e-15)
        prima = c
    assert prima is not None
    for key, v in serie.items():
        conti = [x for x, _ in v]
        T = {t for _, t in v}
        assert len(T) == 1 and len(conti) == R, (key, v)        # same point count per cloud
        assert max(abs(int(d)) for d in np.diff(conti)) <= math.ceil(T.pop() / R), (key, v)


def test_determinismo_due_generatori(Z):
    g1, g2 = _gen_rampa(Z, R_PROVA), _gen_rampa(Z, R_PROVA)
    for i in range(R_PROVA + 2):
        a, b = _passo(g1, Z["pN"], Z["sN"], i), _passo(g2, Z["pN"], Z["sN"], i)
        _uguali(a, b)
        assert a["n_hit"] == b["n_hit"] and a["rampa"] == b["rampa"]


def test_zone_di_un_altro_metro_rifiutate(Z):
    """P zones from another snapshot and metre: no ramp, reason in the state, N alone."""
    ref, g = _gen(Z, Z["zN"]), _gen_rampa(Z, R_PROVA, zona="zP2")
    for i in range(2):
        a, b = _passo(ref, Z["pN"], Z["sN"], i), _passo(g, Z["pN"], Z["sN"], i)
        _uguali(a, b)
        assert b["rampa"]["stato"] == "metro_diverso" and b["rampa"]["alpha"] is None
    gf = _gen(Z, Z["zN"])
    zf = SimpleNamespace(**{k: getattr(Z["zP"], k) for k in ("metro", "center0", "max_range0", "anchors",
                                                              "obiettivi", "quota_g5", "palle_g10", "quota_g10",
                                                              "palle_centri", "palle_raggio", "hash")})
    zf.meta = dict(Z["zP"].meta, fotografia_sha256="0" * 64)
    gf.imposta_rampa(R_PROVA, dict(_prec(Z), zone=zf), genoma_sha=Z["shaN"])
    ref = _gen(Z, Z["zN"])
    a, b = _passo(ref, Z["pN"], Z["sN"], 0), _passo(gf, Z["pN"], Z["sN"], 0)
    _uguali(a, b)
    assert b["rampa"]["stato"] == "fotografia_diversa"


# ----------------------------------------------------------------------
# Real node: victim = twin, state, missing previous genome, two processes
# ----------------------------------------------------------------------

def test_nodo_rampa_vittima_uguale_al_gemello(tmp_path):
    """Same data: victim and twin give identical clouds; status, log, two processes agree."""
    args = NODO + ["--rampa-genoma", str(R_PROVA)]
    a = _nodo_figlio(tmp_path, "rampa1", ROOT, "rampa", args)
    b = _nodo_figlio(tmp_path, "rampa2", ROOT, "rampa", args)
    s = _nodo_figlio(tmp_path, "spento", ROOT, "spento", NODO)
    assert [r["md5"] for r in a["righe"]] == [r["md5"] for r in b["righe"]]

    def senza_avvio(rr):
        # first 3 rows (victim with P): the twin still has the node start genome,
        # drawn at random in each process (random_genome)
        out = json.loads(json.dumps([r["stato"] for r in rr]))
        for st in out[:3]:
            st["rampa"]["gemello"]["genoma_sha"] = None
        return out

    assert senza_avvio(a["righe"]) == senza_avvio(b["righe"])
    gem = [r for r in a["righe"][3:] if r["flusso"] == "gemello"]
    vit = [r for r in a["righe"][3:] if r["flusso"] == "vittima"]
    assert len(gem) == len(vit) == NK
    assert [r["md5"] for r in gem] == [r["md5"] for r in vit]
    assert [r["deriva"] for r in gem] == [r["deriva"] for r in vit]
    # the victim with P (before the change) matches the node without the option
    assert [r["md5"] for r in a["righe"][:3]] == [r["md5"] for r in s["righe"][:3]]
    # during the ramp the output differs from N alone
    s_gem = [r for r in s["righe"][3:] if r["flusso"] == "gemello"]
    assert all(x["md5"] != y["md5"] for x, y in zip(gem[:R_PROVA - 1], s_gem))
    for flusso, rr in (("gemello", gem), ("vittima", vit)):
        for i, r in enumerate(rr):
            st = r["stato"]["rampa"]
            assert st["R"] == R_PROVA and st["ricevuta"][flusso]["stato"] == "ok"
            f = st[flusso]
            assert f["k"] == i and f["alpha"] == min(1.0, (i + 1) / R_PROVA)
            assert f["stato"] == ("attiva" if i < R_PROVA - 1 else "finita")
            assert f["genoma_prec_sha"] == st["ricevuta"][flusso]["genoma_prec_sha"]
    hdr = a["struct"][0]
    assert hdr[-4:] == ["rampa_stato", "rampa_alfa", "rampa_prec_sha", "n_hit_prec"]
    righe_g = [r for r in a["struct"][1:] if r[1] == "gemello"]
    assert righe_g[0][-4] == "attiva" and float(righe_g[0][-3]) == 1.0 / R_PROVA
    assert "g6:" in righe_g[0][-1] and "g10:" in righe_g[0][-1]
    assert righe_g[R_PROVA - 1][-4] == "finita" and righe_g[R_PROVA - 1][-1] == ""


def test_nodo_precedente_mancante_segnalato(tmp_path):
    """Option on without valid P data for the new genome: N only, reason in status."""
    e = _nodo_figlio(tmp_path, "errori", ROOT, "errori", NODO + ["--rampa-genoma", str(R_PROVA)])
    s = _nodo_figlio(tmp_path, "spento", ROOT, "spento", NODO)
    s_gem = [r["md5"] for r in s["righe"][3:5]]
    attesi = {"assente": "prec_assente", "altro_genoma": "genoma_nuovo_diverso",
              "uguale": "prec_uguale_nuovo", "altro_metro": "metro_diverso", "sha256": "prec_sha256_diverso",
              "malformato": "prec_malformato", "annullata": "prec_annullata"}
    for nome, stato in attesi.items():
        rr = e["casi"][nome]
        assert [r["md5"] for r in rr] == s_gem, nome
        for r in rr:
            f = r["stato"]["rampa"]["gemello"]
            assert f["stato"] == stato and f["alpha"] is None, (nome, f)
    v = e["casi"]["valida"]
    assert [r["stato"]["rampa"]["gemello"]["alpha"] for r in v] == [1 / R_PROVA, 2 / R_PROVA]
    assert v[0]["md5"] != s_gem[0]
    # malformed and cancel: in the flow receipt, with seq; unreadable flow handled apart
    ric = e["casi"]["malformato"][0]["stato"]["rampa"]["ricevuta"]["gemello"]
    assert ric["stato"] == "malformato" and isinstance(ric["seq"], int) and "genoma_prec" in ric["motivo"]
    assert ric["genoma_sha_nuovo"] is None and ric["genoma_prec_sha"] is None
    ric = e["casi"]["annullata"][0]["stato"]["rampa"]["ricevuta"]["gemello"]
    assert ric["stato"] == "annullata" and ric["genoma_sha_nuovo"] is None and "motivo" not in ric
    mal = e["casi"]["assente"][0]["stato"]["rampa"]["malformato"]
    assert mal["seq"] == 990 and "flusso" in mal["motivo"]
    assert e["casi"]["assente"][0]["stato"]["rampa"]["ricevuta"]["gemello"] is None


# ----------------------------------------------------------------------
# Real w4 snapshot of the recorded acceptance run: P = e465, N = e480
# ----------------------------------------------------------------------

@pytest.mark.skipif(not (REP / "zone/w4_fotografia.npz").exists(), reason="run di riferimento assente")
def test_fotografia_vera_w4():
    pg, za = _moduli()
    h = json.load(open(REP / "history.json"))
    gP, gN = za.genoma_f32(h[3]["genome"]), za.genoma_f32(h[4]["genome"])
    cache = {}
    zP = za.carica_zone(REP / f"zone/w4_{za.genoma_sha(gP)}.npz", cache=cache)
    zN = za.carica_zone(REP / f"zone/w4_{za.genoma_sha(gN)}.npz", cache=cache)
    assert (zP.genoma_sha, zN.genoma_sha) == ("57e2b5a6e61e41ea", "ebe99146f954b663")
    foto = za.carica_fotografia(REP / "zone/w4_fotografia.npz").cloud
    lim = zN.meta["limiti"]

    def gen(z):
        g = pg.PerturbationGenerator(max_point_shift=lim["max_point_shift"], noise_std=lim["noise_std"],
                                     max_dropout_rate=lim["max_dropout_rate"], max_drift_lag_s=0.5,
                                     structural="gross", zone_attaccante=True, **za.KW_INSIEME)
        g.zone_fornitore = lambda s_, m_, p_: (z, "ok")
        g.keep_drift = True
        g.set_temporal_state(np.array([0.0, 0.35306, 0.0]))
        return g

    R = 9
    ref, g = gen(zN), gen(zN)
    pP, pN = g.encode_perturbation(gP), g.encode_perturbation(gN)
    g.imposta_rampa(R, {"params": pP, "seme": za.seme_genoma(gP), "zone": zP, "sha": zP.genoma_sha},
                    genoma_sha=zN.genoma_sha)
    sN = za.seme_genoma(gN)
    for i in range(R + 1):
        if i >= R - 1:
            ref.set_temporal_state(g.accumulated_drift)
        a = ref.apply_perturbation(foto, pN, ctx={"stamp": 38.7 + 0.1 * i}, seme_genoma=sN, metro=4)
        na = dict(ref.last_struct_info["n_hit"])
        b = g.apply_perturbation(foto, pN, ctx={"stamp": 38.7 + 0.1 * i}, seme_genoma=sN, metro=4)
        nb = dict(g.last_struct_info["n_hit"])
        if i < R - 1:
            assert nb == {"p:g6": pg.rampa_conta(R - i - 1, R, 806), "g16": pg.rampa_conta(i + 1, R, 783),
                          "g13": 500}
            assert len(b) == len(foto) + nb["p:g6"] + nb["g16"]
        else:
            assert _md5(a) == _md5(b) and na == nb == {"g16": 783, "g13": 500}
    assert g.rampa_info["stato"] == "finita"
