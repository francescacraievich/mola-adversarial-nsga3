#!/usr/bin/env python3
"""Generator optimisations with identical output (PERT_OTTIMIZZAZIONI) and zone loading outside
the generator lock.

Synthetic scene and genomes of test_rampa_genoma.py; the real node (in process, no spin) only
in domain 88:

    ROS_DOMAIN_ID=88 python3 -B -m pytest -q -p no:cacheprovider tests/test_ottimizzazioni_nodo.py
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent
ROOT = QUI.parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(ROOT))

import test_rampa_genoma as tr  # noqa: E402
from test_rampa_genoma import Z  # noqa: E402,F401  (zone fixture)

from src.perturbations import structures as st  # noqa: E402


def test_valori_della_variabile():
    assert st._leggi_ottimizzazioni(None) == frozenset()
    assert st._leggi_ottimizzazioni("spente") == frozenset()
    assert st._leggi_ottimizzazioni("tutte") == frozenset(st.OTT_NOMI)
    assert st._leggi_ottimizzazioni("O4, O1") == frozenset({"O1", "O4"})
    for v in ("O6", "O8", "O1,O99", "nessuna", "tutto", ","):
        with pytest.raises(ValueError):
            st._leggi_ottimizzazioni(v)


def test_spente_di_default():
    if "PERT_OTTIMIZZAZIONI" in os.environ:
        pytest.skip("PERT_OTTIMIZZAZIONI impostata nell'ambiente dei test")
    assert st.OTT == frozenset()


def test_eigh_a_blocchi_e_in_ordinati_bit_per_bit():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(10559, 16, 3))
    a -= a.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", a, a) / 15.0
    lam0, vec0 = np.linalg.eigh(cov)
    lam1, vec1 = st._eigh_blocchi(cov)
    assert lam1.tobytes() == lam0.tobytes() and vec1.tobytes() == vec0.tobytes()
    s = np.unique(rng.integers(0, 1 << 40, 50000))
    x = np.r_[rng.choice(s, 3000), rng.integers(0, 1 << 41, 3000), [s[0], s[-1], s[-1] + 1, -1]]
    assert np.array_equal(st._in_ordinati(x, s), np.isin(x, s))
    assert np.array_equal(st._in_ordinati(x, s[:0]), np.isin(x, s[:0]))


def _sequenza(Z, ott, monkeypatch, n=14, cambio=8, salva_a=None, carica=None):
    """Outputs and state fingerprints per cloud; P until cambio, then N.
    salva_a: index after which the flow state is saved; carica: state to impose before the
    first cloud (starting then from salva_a + 1).
    """
    pg = Z["pg"]
    monkeypatch.setattr(st, "OTT", ott)
    g = tr._gen(Z, Z["zP"])
    righe, salvato = [], None
    inizio = 0
    if carica is not None:
        g.imposta_stato_flusso(carica[1])
        inizio = carica[0] + 1
    for i in range(inizio, n):
        if i == cambio:
            g.zone_fornitore = lambda s_, m_, p_: (Z["zN"], "ok")
        params, seme = (Z["pP"], Z["sP"]) if i < cambio else (Z["pN"], Z["sN"])
        p = tr._passo(g, params, seme, i)
        trk = g._structs
        assert "_albero" not in trk.__dict__                    # O4 closed before returning
        assert trk.prev is None or trk.prev["tree"] is not None
        righe.append((i, p["md5"], pg.impronta_stato(g.stato_flusso(copia=False))))
        if salva_a is not None and i == salva_a:
            salvato = (i, g.stato_flusso(copia=True))
    return righe, salvato


@pytest.mark.parametrize("nomi", [st.OTT_NOMI, ("O1",), ("O2",), ("O3",), ("O4",), ("O5",), ("O7",), ("O9",)])
def test_uscita_e_stato_identici(Z, monkeypatch, nomi):
    base, _ = _sequenza(Z, frozenset(), monkeypatch)
    ott, _ = _sequenza(Z, frozenset(nomi), monkeypatch)
    assert ott == base


def test_salva_e_carica_D31c_fra_le_due_versioni(Z, monkeypatch):
    """State saved with optimisations and loaded without (and vice versa): continuous sequence."""
    tutte = frozenset(st.OTT_NOMI)
    base, s_base = _sequenza(Z, frozenset(), monkeypatch, salva_a=5)
    ott, s_ott = _sequenza(Z, tutte, monkeypatch, salva_a=5)
    assert ott == base
    da_ott, _ = _sequenza(Z, frozenset(), monkeypatch, carica=s_ott)
    da_base, _ = _sequenza(Z, tutte, monkeypatch, carica=s_base)
    assert da_ott == base[6:]
    assert da_base == base[6:]


def test_errore_del_corpo_non_coperto_da_quello_del_thread(Z, monkeypatch):
    monkeypatch.setattr(st, "OTT", frozenset(st.OTT_NOMI))
    g = tr._gen(Z, Z["zP"])
    tr._passo(g, Z["pP"], Z["sP"], 0)
    fut = Future()
    fut.set_exception(RuntimeError("kd-tree"))
    g._structs._albero = fut

    def corpo(*a, **k):
        raise ValueError("corpo")
    monkeypatch.setattr(g, "_apply_coherent_corpo", corpo)
    with pytest.raises(ValueError, match="corpo"):
        g._apply_coherent(tr.scena(1), Z["pP"], {"stamp": 2.1}, Z["sP"], 1)
    assert "_albero" not in g._structs.__dict__


def test_avvia_pool_crea_i_thread(monkeypatch):
    # new pool, independent of those created by earlier tests
    monkeypatch.setattr(st, "_POOL", None)
    monkeypatch.setattr(st, "OTT", frozenset())
    st.avvia_pool()
    assert st._POOL is None                                 # off: no pool
    monkeypatch.setattr(st, "OTT", frozenset({"O1"}))
    st.avvia_pool()
    try:
        assert st._POOL is not None and len(st._POOL._threads) == st.WORKERS
    finally:
        st._POOL.shutdown(wait=True)


# ----------------------------------------------------------------------
# Zones outside the lock (real node, domain 88)
# ----------------------------------------------------------------------

def _crea_nodo(monkeypatch, argv):
    pn = pytest.importorskip("src.nodes.perturbation_node")

    class _Preso(Exception):
        pass

    def _cattura(c):
        raise _Preso(c)
    with monkeypatch.context() as m:
        m.setattr(pn.rclpy, "init", lambda **k: None)
        m.setattr(pn, "PerturbationNode", _cattura)
        m.setattr(sys, "argv", ["perturbation_node.py"] + argv)
        with pytest.raises(_Preso) as e:
            pn.main()
    return pn, pn.PerturbationNode(e.value.args[0])


@pytest.fixture
def nodo_gim(monkeypatch):
    if os.environ.get("ROS_DOMAIN_ID") != "88":
        pytest.skip("il nodo dei test gira solo nel dominio 88")
    rclpy = pytest.importorskip("rclpy")
    rclpy.init()
    pn, n = _crea_nodo(monkeypatch, tr.NODO + ["--genoma-in-moto", "2.0"])
    yield pn, n
    n.destroy_node()
    rclpy.shutdown()


def _msg_rampa(seq, flusso, path):
    return SimpleNamespace(data=json.dumps({"seq": seq, "flusso": flusso, "genoma_prec": [0.1] * 17,
                                            "genoma_sha_nuovo": "n",
                                            "zone_prec": {"path": path, "sha256": "s", "hash": "h"}}))


def test_rampa_caricata_mentre_una_nuvola_tiene_il_lock(nodo_gim, monkeypatch):
    """With the genome in motion the ramp carries P: zones read without the lock, receipt waits."""
    pn, n = nodo_gim
    caricate = threading.Event()

    def carica(path, sha256=None, hash_atteso=None, cache=None):
        caricate.set()
        return SimpleNamespace(genoma_sha="x", seme=0, firma="f", hash="h:" + path)
    monkeypatch.setattr(pn.za, "carica_zone", carica)
    cb = {s.topic_name: s.callback for s in n.subscriptions}["/attack/rampa"]
    n._gen_lock.acquire()
    try:
        t = threading.Thread(target=cb, args=(_msg_rampa(1, "vittima", "p"),), daemon=True)
        t.start()
        assert caricate.wait(5)
        time.sleep(0.2)
        assert t.is_alive() and n._rampa_ric["vittima"] is None
    finally:
        n._gen_lock.release()
    t.join(5)
    assert not t.is_alive()
    assert n._rampa_ric["vittima"]["seq"] == 1 and n._rampa_ric["vittima"]["zone_prec_hash"] == "h:p"


def test_rampa_vince_l_ultimo_messaggio_arrivato(nodo_gim, monkeypatch):
    pn, n = nodo_gim
    lento, dentro = threading.Event(), threading.Event()

    def carica(path, sha256=None, hash_atteso=None, cache=None):
        if path == "vecchio":
            dentro.set()
            assert lento.wait(5)
        return SimpleNamespace(genoma_sha="x", seme=0, firma="f", hash="h:" + path)
    monkeypatch.setattr(pn.za, "carica_zone", carica)
    t1 = threading.Thread(target=n._rampa_cb, args=(_msg_rampa(1, "gemello", "vecchio"),), daemon=True)
    t1.start()
    assert dentro.wait(5)
    t2 = threading.Thread(target=n._rampa_cb, args=(_msg_rampa(2, "gemello", "nuovo"),), daemon=True)
    t2.start()
    t_lim = time.monotonic() + 5
    while n._rampa_arrivi["gemello"] < 2 and time.monotonic() < t_lim:
        time.sleep(0.01)
    assert n._rampa_arrivi["gemello"] == 2
    lento.set()
    t1.join(5)
    t2.join(5)
    assert n._rampa_ric["gemello"]["seq"] == 2 and n._rampa_ric["gemello"]["zone_prec_hash"] == "h:nuovo"
    # cancel and malformed message go through the same receipt
    n._rampa_cb(SimpleNamespace(data=json.dumps({"seq": 3, "flusso": "gemello", "annulla": True})))
    assert n._rampa_ric["gemello"]["stato"] == "annullata"
    n._rampa_cb(SimpleNamespace(data=json.dumps({"seq": 4, "flusso": "gemello"})))
    assert n._rampa_ric["gemello"]["stato"] == "malformato"


def test_rampa_ricevuta_di_un_arrivo_superato_scartata(nodo_gim):
    """Arrival check: the receipt of a message superseded by a later one is never written."""
    pn, n = nodo_gim
    a1 = n._rampa_conta("vittima")
    a2 = n._rampa_conta("vittima")
    n._rampa_scrivi("vittima", a2, {"seq": 2})
    n._rampa_scrivi("vittima", a1, {"seq": 1})
    assert n._rampa_ric["vittima"] == {"seq": 2}
    n._rampa_scrivi("vittima", a1, {"seq": 1})
    assert n._rampa_ric["vittima"] == {"seq": 2}


@pytest.fixture
def nodo(monkeypatch):
    if os.environ.get("ROS_DOMAIN_ID") != "88":
        pytest.skip("il nodo dei test gira solo nel dominio 88")
    pn = pytest.importorskip("src.nodes.perturbation_node")
    rclpy = pytest.importorskip("rclpy")

    class _Preso(Exception):
        pass

    def _cattura(c):
        raise _Preso(c)
    with monkeypatch.context() as m:
        m.setattr(pn.rclpy, "init", lambda **k: None)
        m.setattr(pn, "PerturbationNode", _cattura)
        m.setattr(sys, "argv", ["perturbation_node.py"] + tr.NODO)
        with pytest.raises(_Preso) as e:
            pn.main()
    rclpy.init()
    n = pn.PerturbationNode(e.value.args[0])
    yield pn, n
    n.destroy_node()
    rclpy.shutdown()


def _msg_zone(seq, flusso, path):
    return SimpleNamespace(data=json.dumps({"seq": seq, "flusso": flusso, "path": path,
                                            "sha256": "s", "hash": "h", "genoma_sha": "g"}))


def test_zone_caricate_mentre_una_nuvola_tiene_il_lock(nodo, monkeypatch):
    pn, n = nodo
    caricate = threading.Event()

    def carica(path, sha256=None, hash_atteso=None, cache=None):
        caricate.set()
        return SimpleNamespace(genoma_sha="g", hash="h:" + path)
    monkeypatch.setattr(pn.za, "carica_zone", carica)
    cb = {s.topic_name: s.callback for s in n.subscriptions}["/attack/zone"]
    n._gen_lock.acquire()                                   # a cloud in progress
    try:
        t = threading.Thread(target=cb, args=(_msg_zone(1, "gemello", "a"),), daemon=True)
        t.start()
        assert caricate.wait(5)                             # the file is read without the lock
        time.sleep(0.2)
        assert t.is_alive() and n._zone_ric["gemello"] is None   # zones, receipt wait for the cloud
    finally:
        n._gen_lock.release()
    t.join(5)
    assert not t.is_alive()
    assert n._zone_ric["gemello"]["seq"] == 1 and n._zone_ric["gemello"]["stato"] == "ok"
    assert n._zone_spec["gemello"]["zone"].hash == "h:a"


def test_vince_l_ultimo_messaggio_arrivato(nodo, monkeypatch):
    """The last message to arrive wins: the first finishes loading and installs nothing."""
    pn, n = nodo
    lento = threading.Event()
    dentro = threading.Event()

    def carica(path, sha256=None, hash_atteso=None, cache=None):
        if path == "vecchio":
            dentro.set()
            assert lento.wait(5)
        return SimpleNamespace(genoma_sha="g", hash="h:" + path)
    monkeypatch.setattr(pn.za, "carica_zone", carica)
    t1 = threading.Thread(target=n._zone_cb, args=(_msg_zone(1, "vittima", "vecchio"),), daemon=True)
    t1.start()
    assert dentro.wait(5)
    t2 = threading.Thread(target=n._zone_cb, args=(_msg_zone(2, "vittima", "nuovo"),), daemon=True)
    t2.start()
    t_lim = time.monotonic() + 5
    while n._zone_arrivi["vittima"] < 2 and time.monotonic() < t_lim:
        time.sleep(0.01)
    assert n._zone_arrivi["vittima"] == 2                   # second arrived before first ended
    lento.set()
    t1.join(5)
    t2.join(5)
    assert not t1.is_alive() and not t2.is_alive()
    assert n._zone_ric["vittima"]["seq"] == 2 and n._zone_ric["vittima"]["stato"] == "ok"
    assert n._zone_spec["vittima"]["zone"].hash == "h:nuovo"
    assert n._zone_ric["gemello"] is None                   # the other flow is unchanged


def test_cache_delle_fotografie_un_caricamento_alla_volta(nodo, monkeypatch):
    pn, n = nodo
    dentro, insieme = [0], [0]
    lk = threading.Lock()

    def carica(path, sha256=None, hash_atteso=None, cache=None):
        with lk:
            dentro[0] += 1
            insieme[0] = max(insieme[0], dentro[0])
        time.sleep(0.05)
        with lk:
            dentro[0] -= 1
        return SimpleNamespace(genoma_sha="g", hash="h")
    monkeypatch.setattr(pn.za, "carica_zone", carica)
    ts = [threading.Thread(target=n._zone_cb, args=(_msg_zone(i, f, "p"),))
          for i, f in ((1, "gemello"), (2, "vittima"), (3, "gemello"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert insieme[0] == 1
