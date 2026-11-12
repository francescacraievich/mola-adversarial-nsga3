"""Input and output clouds saved per genome and flow (--salva-nuvole).

The partial node is the one of test_misure_fuori_percorso.py (__new__, fake publish, no
rclpy.init); only one test creates a real node, in the conftest domain 88, for the
subscriptions. Without the option: same arguments, same status, same output clouds byte
for byte.
"""
import csv
import json
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

pn = pytest.importorskip("src.nodes.perturbation_node")
from test_misure_fuori_percorso import GENOMA_MASSIMO, _Pub, _msg, _nodo, _nuvole_sintetiche  # noqa: E402

String = pn.String


# --- arguments ---

class _Preso(Exception):
    pass


def _cli(monkeypatch, argv):
    """Arguments as main() sees them: fake rclpy.init, the node is not created."""
    def nodo(cli):
        raise _Preso(cli)
    monkeypatch.setattr(pn.rclpy, "init", lambda **k: None)
    monkeypatch.setattr(pn, "PerturbationNode", nodo)
    monkeypatch.setattr(sys, "argv", ["perturbation_node.py"] + argv)
    with pytest.raises(_Preso) as e:
        pn.main()
    return e.value.args[0]


def test_argomenti_default_spenti(monkeypatch):
    cli = _cli(monkeypatch, [])
    assert cli.salva_nuvole is None and cli.salva_nuvole_k == 20
    cli = _cli(monkeypatch, ["--salva-nuvole", "/x/y", "--salva-nuvole-k", "7"])
    assert cli.salva_nuvole == "/x/y" and cli.salva_nuvole_k == 7


# --- pure functions ---

def test_scelta():
    s = pn.salva_nuvole_scelta
    # victim: only perturbed, not dropped, k < N
    assert s(True, True, False, False, False, 99, 3, 20) == ("vittima", 3)
    assert s(True, False, True, False, False, 0, 0, 20) is None        # clean
    assert s(True, False, True, True, False, 0, 0, 20) is None         # dropped (never_clean)
    assert s(True, True, True, False, False, 0, 20, 20) is None        # k = N
    # twin: attack on, outside the victim flow and the replay
    assert s(False, False, True, False, False, 5, 0, 20) == ("gemello", 5)
    assert s(False, False, False, False, False, 0, 0, 20) is None      # passthrough
    assert s(False, False, True, False, True, 0, 0, 20) is None        # replay
    assert s(False, False, True, False, False, 19, 0, 20) == ("gemello", 19)
    assert s(False, False, True, False, False, 20, 0, 20) is None
    assert s(False, False, True, False, False, 0, 0, 0) is None


def test_nome_e_tag():
    assert pn.salva_nuvole_tag("") == pn.salva_nuvole_tag(None) == pn.salva_nuvole_tag("  ") == "senza_tag"
    assert pn.salva_nuvole_tag("w4 e480/sel0") == "w4_e480_sel0"
    assert pn.salva_nuvole_tag("K1-P1.a") == "K1-P1.a"
    assert pn.salva_nuvole_tag("è") == "_"
    assert (pn.salva_nuvole_nome("w4", "vittima", 12, 3, 1234, 5000000)
            == "w4_vittima_g12_k03_1234.005000000.npz")
    assert pn.salva_nuvole_nome("", "gemello", 1, 17, 7, 0) == "senza_tag_gemello_g1_k17_7.000000000.npz"


# --- option off ---

def _uscite(n, nuvole, cambio_a=None):
    out = []
    for i, (c, st) in enumerate(nuvole):
        if cambio_a is not None and i == cambio_a:
            n._set_genome(np.asarray(GENOMA_MASSIMO, dtype=np.float64) * 0.5, source="test")
        n._cloud_cb(_msg(c, st))
        job = n._out_q.get_nowait()
        assert len(job) == 8
        out.append((job[2].tobytes(), None if job[4] is None else job[4].tobytes(),
                    job[3].stamp.sec, job[3].stamp.nanosec,
                    None if job[5] is None else (job[5].stamp.sec, job[5].stamp.nanosec)))
    return out


def _status(n):
    for a, v in (("_max_drift_lag_s", 0.5), ("_replay_seq", 0), ("_replay_n", 0), ("_replay_total", 0),
                 ("_replay_last_stamp", None), ("_twin_offset", 0.0)):
        if not hasattr(n, a):
            setattr(n, a, v)
    if not n.lat_times:                        # _pub_loop not running: a fake latency
        n.lat_times.append(0.01)
    n.status_pub = _Pub()
    pn.PerturbationNode._publish_status_body(n)
    return json.loads(n.status_pub.msgs[-1].data)


@pytest.mark.parametrize("victim", [True, False])
def test_spenta_uscite_identiche(tmp_path, victim):
    nuvole = _nuvole_sintetiche(8)
    u0 = _uscite(_nodo(victim=victim), nuvole, cambio_a=5)       # node without the attribute
    n1 = _nodo(victim=victim)
    n1._salva = None                                               # option off
    u1 = _uscite(n1, nuvole, cambio_a=5)
    n2 = _nodo(victim=victim)
    n2._salva = pn.SalvaNuvole(tmp_path / "s", 3)                  # option on: same outputs
    u2 = _uscite(n2, nuvole, cambio_a=5)
    n2._salva.chiudi()
    assert u0 == u1 == u2
    # twin: k 0-2 of two genomes; victim: the twin genome does not reset its k
    assert n2._salva.salvati == (3 if victim else 6) and n2._salva.scartati_coda == 0


def test_spenta_status_invariato():
    n0, n1 = _nodo(), _nodo()
    n1._salva = None
    for n in (n0, n1):
        n._cloud_cb(_msg(*_nuvole_sintetiche(1)[0]))
    s0, s1 = _status(n0), _status(n1)
    assert "salva_nuvole" not in s0 and s0.keys() == s1.keys()


def test_nodo_vero_sottoscrizioni(tmp_path, monkeypatch):
    rclpy = pytest.importorskip("rclpy")
    argv0 = []
    argv1 = ["--salva-nuvole", str(tmp_path / "s"), "--salva-nuvole-k", "4"]
    clis = [_cli(monkeypatch, a) for a in (argv0, argv1)]
    monkeypatch.undo()
    rclpy.init()
    try:
        topic = {}
        for i, cli in enumerate(clis):
            n = pn.PerturbationNode(cli)
            topic[i] = sorted(s.topic_name for s in n.subscriptions)
            if i == 0:
                assert n._salva is None
            else:
                assert n._salva.k_max == 4 and n._salva.cartella == tmp_path / "s"
                n._salva.chiudi()
            n.destroy_node()
        assert "/attack/salva_nuvole_tag" not in topic[0]
        assert topic[1] == sorted(topic[0] + ["/attack/salva_nuvole_tag"])
    finally:
        rclpy.shutdown()


# --- option on ---

def _carica(cartella):
    with open(cartella / "indice.csv") as f:
        righe = list(csv.DictReader(f))
    return righe, {r["file"]: np.load(cartella / r["file"]) for r in righe}


def test_gemello_record_e_file(tmp_path):
    nuvole = _nuvole_sintetiche(9)
    n = _nodo(victim=False)
    n._salva = sn = pn.SalvaNuvole(tmp_path, 4)
    pn.PerturbationNode._salva_tag_cb(n, String(data="w4 e480"))
    attese = []
    for i, (c, st) in enumerate(nuvole):
        if i == 6:
            n._set_genome(np.asarray(GENOMA_MASSIMO, dtype=np.float64) * 0.5, source="test")
        n._cloud_cb(_msg(c, st))
        job = n._out_q.get_nowait()
        attese.append((n.genome_id, job[6], job[2], job[3]))
    sn.chiudi()
    righe, z = _carica(tmp_path)
    # genome 1: k 0-3 of clouds 0-5; genome 2: k 0-2 of clouds 6-8
    assert [(r["genome_id"], r["k"]) for r in righe] == [
        ("1", "0"), ("1", "1"), ("1", "2"), ("1", "3"), ("2", "0"), ("2", "1"), ("2", "2")]
    usate = [0, 1, 2, 3, 6, 7, 8]
    for r, i in zip(righe, usate):
        gid, cloud, out, h = attese[i]
        k = int(r["k"])
        assert r["file"] == pn.salva_nuvole_nome("w4 e480", "gemello", gid, k, h.stamp.sec, h.stamp.nanosec)
        d = z[r["file"]]
        assert str(d["flusso"]) == "gemello" and str(d["tag"]) == "w4_e480"
        assert int(d["genome_id"]) == gid and int(d["k"]) == k and int(d["metro"]) == -1
        g = np.asarray(GENOMA_MASSIMO, dtype=np.float64) * (1.0 if gid == 1 else 0.5)
        assert int(d["seme"]) == pn.genome_seed(g) % (2**31 - 1)
        assert d["ingresso"].dtype == np.float32 and d["uscita"].dtype == np.float32
        # input = cloud read from the message; output = published bytes
        assert np.array_equal(d["ingresso"], np.ascontiguousarray(cloud[:, :4], dtype=np.float32))
        assert np.array_equal(d["ingresso"], np.asarray(nuvole[i][0], dtype=np.float32))
        assert d["uscita"].tobytes() == np.ascontiguousarray(out[:, :4], dtype=np.float32).tobytes()
        assert int(d["n_in"]) == len(cloud) and int(d["n_out"]) == len(out) == int(r["n_out"])
        assert list(d["stamp_in"]) == [h.stamp.sec, h.stamp.nanosec] and list(d["stamp_out"]) == [-1, -1]
        assert r["stamp_out_sec"] == "" and r["tag"] == "w4_e480"
    st = _status(n)["salva_nuvole"]
    assert st == {"dir": str(tmp_path), "k": 4, "tag": "w4 e480", "salvati": 7, "scartati_coda": 0,
                  "errori": 0}


def test_vittima_ristampata_pulita_scartata_e_rigioco(tmp_path):
    nuvole = _nuvole_sintetiche(6)
    n = _nodo(victim=True)
    n._salva = sn = pn.SalvaNuvole(tmp_path, 20)
    n._metro = {"gemello": None, "vittima": 4}
    # continuous victim clock 6.9 s behind: different output stamp
    n._victim_last_sim, n._victim_last_clock = 5.0, 3.0
    n._cloud_cb(_msg(*nuvole[0]))
    job = n._out_q.get_nowait()
    n.victim_attack = False                      # clean: not saved
    n._cloud_cb(_msg(*nuvole[1]))
    n._never_clean = True                        # dropped: not saved (k advances)
    n.victim_clean_dropped = 0
    n._cloud_cb(_msg(*nuvole[2]))
    n._never_clean = False
    n.victim_attack = True
    n._cloud_cb(_msg(*nuvole[3]))
    # twin during replay: not saved
    n.victim_stream = False
    n._replay_active = True
    n._cloud_cb(_msg(*nuvole[4]))
    sn.chiudi()
    righe, z = _carica(tmp_path)
    assert [(r["flusso"], r["k"]) for r in righe] == [("vittima", "0"), ("vittima", "3")]
    d = z[righe[0]["file"]]
    vh = job[5]
    assert list(d["stamp_out"]) == [vh.stamp.sec, vh.stamp.nanosec] == [3, 100000000]
    assert list(d["stamp_in"]) == [10, 0] and int(d["metro"]) == 4
    assert righe[0]["file"].startswith("senza_tag_vittima_g1_k00_10.000000000")
    assert d["uscita"].tobytes() == np.ascontiguousarray(job[4][:, :4], dtype=np.float32).tobytes()
    assert int(d["seme"]) == n.victim_seed and str(d["flusso"]) == "vittima"


def test_nessun_file_sovrascritto(tmp_path):
    sn = pn.SalvaNuvole(tmp_path, 20)
    h = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2))
    c = np.ones((10, 4))
    for v in (1.0, 2.0):
        sn.accoda(sn.record("gemello", 1, 5, None, 0, h, h, 0.0, 10, c * v, c * v))
    sn.chiudi()
    righe, z = _carica(tmp_path)
    assert [r["file"] for r in righe] == ["senza_tag_gemello_g1_k00_1.000000002.npz",
                                          "senza_tag_gemello_g1_k00_1.000000002_1.npz"]
    assert [float(z[r["file"]]["uscita"][0, 0]) for r in righe] == [1.0, 2.0]


def test_coda_piena_scarti_contati(tmp_path):
    class Lento(pn.SalvaNuvole):
        def _scrivi(self, rec):
            preso.set()
            via.wait(10)
            super()._scrivi(rec)
    preso, via = threading.Event(), threading.Event()
    sn = Lento(tmp_path, 20, coda=2)
    h = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))
    rec = lambda k: sn.record("gemello", 1, 5, None, k, h, h, 0.0, 3, np.zeros((3, 4)), np.zeros((3, 4)))
    t0 = time.perf_counter()
    assert sn.accoda(rec(0))
    assert preso.wait(5)                          # the thread took the first one and writes
    esiti = [sn.accoda(rec(k)) for k in range(1, 6)]
    assert time.perf_counter() - t0 < 5.0         # enqueue does not wait for the thread
    assert esiti == [True, True, False, False, False]
    assert sn.stato()["scartati_coda"] == 3
    via.set()
    sn.chiudi()
    assert sn.salvati == 3 and sn.errori == 0
    assert [r["k"] for r in _carica(tmp_path)[0]] == ["0", "1", "2"]


def test_errori_contati(tmp_path):
    sn = pn.SalvaNuvole(tmp_path, 20)
    h = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))
    rec = sn.record("gemello", 1, 5, None, 0, h, h, 0.0, 3, np.zeros((3, 4)), np.zeros((3, 4)))
    rec["k"] = "non intero"
    sn.accoda(rec)
    sn.chiudi()
    assert sn.stato()["errori"] == 1 and sn.salvati == 0


# --- time added to the callback ---

def test_tempo_accodare_sotto_un_ms(tmp_path):
    rng = np.random.default_rng(0)
    n_pt = 45000
    cloud = np.c_[rng.uniform(-20, 20, (n_pt, 3)), rng.uniform(0, 255, n_pt)]
    out = cloud + rng.normal(0, 0.02, cloud.shape)
    h = SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0))
    ho = SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=1))
    sn = pn.SalvaNuvole(tmp_path, 20, coda=64)
    ts = []
    for k in range(60):
        t = time.perf_counter()
        sn.accoda(sn.record("vittima", 1, 5, 2, k % 20, h, ho, 0.0, n_pt, cloud, out))
        ts.append((time.perf_counter() - t) * 1000.0)
        time.sleep(0.02)                          # the thread writes meanwhile, as on the bench
    sn.chiudi()
    med, p95 = float(np.median(ts)), float(np.percentile(ts, 95))
    print(f"accodare 45000 punti: mediana {med:.3f} ms, p95 {p95:.3f} ms, max {max(ts):.3f} ms, "
          f"salvati {sn.salvati}, scartati {sn.scartati_coda}")
    assert med < 1.0
    assert sn.salvati + sn.scartati_coda == 60 and sn.errori == 0
