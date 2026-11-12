"""Delay instruments of perturbation_node (--strumenti-ritardo, off by default).

Without ROS: per-cloud rows, watchdog (duration, wait, false wait, rate limit, signature B1
with a long GC collection and B2 with the GIL held by a C call), garbage collector, CPU
sampler, file names. With ROS, only in domain 88 (conftest): the real node in a subprocess
with and without the option, same clouds in passthrough, same outputs.
"""
import csv
import gc
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nodes import strumenti_ritardo as sr  # noqa: E402


def leggi(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


class _Msg:
    class header:
        class stamp:
            sec, nanosec = 12, 500_000_000
    width, height = 100, 1


def _nuvola_completa(st, info=None, pausa_cb=0.0):
    r = st.inizio_callback(info, _Msg, 7)
    r.fase("lettura_fine")
    if pausa_cb:
        time.sleep(pausa_cb)
    r.parti = 2
    r.fase("accodata")
    st.fine_callback(r, True)
    st.presa(r)
    r.fase("rilascio")
    for t in ("main", "victim"):
        st.pub_inizio(r, t)
        st.chiamata()
        st.pub_fine(r, t)
    r.fase("registri_fine")
    st.fine_pubblicazione(r)
    return r


def test_colonne_e_riga_per_nuvola(tmp_path):
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    adesso = time.time_ns()
    info = {"source_timestamp": adesso - 30_000_000, "received_timestamp": adesso - 10_000_000,
            "publication_sequence_number": 5, "reception_sequence_number": 4}
    _nuvola_completa(st, info)
    st.chiudi()
    righe = leggi(tmp_path / "rit.csv")
    assert list(righe[0].keys()) == list(sr.COLONNE)
    assert len(righe) == 1
    r = righe[0]
    assert r["esito"] == "pubblicata" and r["frame_idx"] == "7" and r["n_punti"] == "100"
    assert float(r["stamp_in"]) == pytest.approx(12.5)
    assert int(r["dds_recv_ns"]) == adesso - 10_000_000 and r["pub_seq"] == "5"
    assert 9.0 <= float(r["ritardo_ricezione_ms"]) < 200.0
    m = {e: int(r[f"{e}_mono_ns"]) for e in ("cb_inizio", "lettura_fine", "cb_fine", "presa", "rilascio",
                                              "main_inizio", "main_chiamata", "main_fine",
                                              "victim_inizio", "victim_chiamata", "victim_fine",
                                              "registri_fine")}
    assert m["cb_inizio"] <= m["lettura_fine"] <= m["cb_fine"] <= m["presa"] <= m["rilascio"]
    assert m["main_inizio"] <= m["main_chiamata"] <= m["main_fine"] <= m["victim_inizio"]
    assert m["victim_fine"] <= m["registri_fine"]
    assert r["attacker_inizio_mono_ns"] == "" and r["shadow_fine_wall_ns"] == ""
    # no stack for a short callback (the wait after the last one is disarmed at close)
    assert "--- stack" not in (tmp_path / "rit_stack.txt").read_text()


def test_nuvola_senza_uscita(tmp_path):
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    r = st.inizio_callback(None, _Msg, 3)
    st.fine_callback(r, False)
    st.chiudi()
    righe = leggi(tmp_path / "rit.csv")
    assert len(righe) == 1 and righe[0]["esito"] == "senza_uscita"
    assert righe[0]["dds_recv_ns"] == "" and righe[0]["presa_mono_ns"] == ""


def test_cane_durata(tmp_path):
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    _nuvola_completa(st, pausa_cb=0.35)
    st.chiudi()
    testo = (tmp_path / "rit_stack.txt").read_text()
    assert "Timeout (0:00:00.200000)!" in testo
    assert "--- stack 1 tipo=durata nuvola=0" in testo and "_nuvola_completa" in testo
    cane = leggi(tmp_path / "rit_cane.csv")
    assert [c["evento"] for c in cane] == ["stack", "durata"]
    assert cane[1]["n_stack"] == "1" and float(cane[1]["valore_ms"]) >= 300
    assert leggi(tmp_path / "rit.csv")[0]["stack_durata"] == "1"


def test_cane_con_un_thread_appena_avviato(tmp_path, monkeypatch):
    # threading.enumerate() includes starting threads, still without ident: the
    # stack must not kill the callback
    class _InAvvio:
        name, ident, native_id = "in_avvio", None, None
    vero = sr.threading.enumerate
    monkeypatch.setattr(sr.threading, "enumerate", lambda: list(vero()) + [_InAvvio()])
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    _nuvola_completa(st, pausa_cb=0.35)
    st.chiudi()
    testo = (tmp_path / "rit_stack.txt").read_text()
    assert "--- stack 1 tipo=durata nuvola=0" in testo and " in_avvio:-:-" in testo


def test_cane_attesa_e_limite_di_frequenza(tmp_path):
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    _nuvola_completa(st)
    time.sleep(0.35)                     # the next cloud does not arrive: "attesa" stack
    _nuvola_completa(st, pausa_cb=0.25)  # less than MIN_INTERVALLO_STACK_S later: no stack
    st.chiudi()
    testo = (tmp_path / "rit_stack.txt").read_text()
    assert testo.count("--- stack ") == 1 and "tipo=attesa nuvola=0" in testo
    righe = leggi(tmp_path / "rit.csv")
    assert righe[1]["stack_attesa"] == "1" and righe[1]["stack_durata"] == ""
    cane = leggi(tmp_path / "rit_cane.csv")
    assert [(c["evento"], c["seq"]) for c in cane] == [("stack", "0"), ("durata", "1")]
    assert st.cane.n_saltati >= 1


def test_attesa_falsa_non_consuma_tetto_ne_limite(tmp_path):
    """Input gap: the "attesa" stack is written but counts towards neither cap nor rate limit."""
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    st.cane._max = 1
    _nuvola_completa(st)
    time.sleep(0.3)
    adesso = time.time_ns()
    info = {"source_timestamp": adesso - 20_000_000, "received_timestamp": adesso - 5_000_000}
    _nuvola_completa(st, info, pausa_cb=0.3)
    st.chiudi()
    testo = (tmp_path / "rit_stack.txt").read_text()
    assert "--- stack 1 tipo=attesa nuvola=0" in testo and "--- stack 2 tipo=durata nuvola=1" in testo
    cane = leggi(tmp_path / "rit_cane.csv")
    assert [(c["evento"], c["seq"]) for c in cane] == [("stack", "0"), ("attesa_falsa", "1"),
                                                      ("stack", "1"), ("durata", "1")]
    assert st.cane.n_falsi == 1 and st.cane.n_veri == 1 and st.cane.n_stack == 2
    r = leggi(tmp_path / "rit.csv")[1]
    assert r["stack_attesa"] == "1" and r["stack_durata"] == "2"
    assert float(r["ritardo_dds_ms"]) == pytest.approx(15.0)


def _stack_scattati(testo):
    """(type, scatto_mono_ns) for each '--- stack' row."""
    out = []
    for riga in testo.splitlines():
        if riga.startswith("--- stack "):
            campi = dict(c.split("=", 1) for c in riga.split()[3:] if "=" in c)
            out.append((campi["tipo"], int(campi["scatto_mono_ns"])))
    return out


def test_cane_durante_una_raccolta_lunga_del_gc(tmp_path):
    """Signature B1: the stack fires inside a gen-2 GC collection longer than 200 ms."""
    prog = textwrap.dedent(f"""
        import sys, gc, time, json
        sys.path.insert(0, {str(ROOT)!r})
        from src.nodes import strumenti_ritardo as sr
        gc.disable()
        tenuti = []

        def cresci(n):
            for _ in range(n):
                a = []
                a.append(a)
                tenuti.append(a)

        # taratura: contenitori finche' una raccolta completa dura almeno 0,35 s
        dur = 0.0
        while dur < 0.35 and len(tenuti) < 20_000_000:
            cresci(1_000_000)
            t = time.monotonic()
            gc.collect(2)
            dur = time.monotonic() - t

        class M:
            class header:
                class stamp:
                    sec, nanosec = 1, 0
            width, height = 1, 1

        st = sr.StrumentiRitardo({str(tmp_path / 'gc.csv')!r})

        def callback_con_gc():
            r = st.inizio_callback(None, M, 0)
            gc.collect(2)
            st.fine_callback(r, False)

        callback_con_gc()
        st.chiudi()
        print(json.dumps({{"n": len(tenuti), "dur_s": dur}}))
    """)
    p = subprocess.run([sys.executable, "-c", prog], check=True, timeout=300, capture_output=True, text=True)
    tar = json.loads(p.stdout.strip().splitlines()[-1])
    assert tar["dur_s"] >= 0.25, f"raccolta troppo breve per la prova: {tar}"
    testo = (tmp_path / "gc_stack.txt").read_text()
    scatti = _stack_scattati(testo)
    assert len(scatti) == 1 and scatti[0][0] == "durata"
    assert "callback_con_gc" in testo
    lunghe = [r for r in leggi(tmp_path / "gc_gc.csv") if r["generazione"] == "2" and float(r["durata_ms"]) > 200]
    assert len(lunghe) == 1
    assert int(lunghe[0]["inizio_mono_ns"]) <= scatti[0][1] <= int(lunghe[0]["fine_mono_ns"])
    assert leggi(tmp_path / "gc.csv")[0]["stack_durata"] == "1"


def test_cane_con_il_gil_tenuto_da_una_chiamata_c(tmp_path):
    """Signature B2: the stack fires inside a C call that holds the GIL for at least 0.5 s."""
    prog = textwrap.dedent(f"""
        import sys, threading, time, json
        sys.path.insert(0, {str(ROOT)!r})
        from src.nodes import strumenti_ritardo as sr
        t = time.monotonic()
        sum(range(10_000_000))
        N = int(10_000_000 * 0.8 / (time.monotonic() - t))     # una chiamata da circa 0,8 s

        class M:
            class header:
                class stamp:
                    sec, nanosec = 1, 0
            width, height = 1, 1

        st = sr.StrumentiRitardo({str(tmp_path / 'gil.csv')!r})
        c = {{}}
        pronto = threading.Event()

        def tieni_gil():
            pronto.wait()
            c["t0"] = time.monotonic_ns()
            sum(range(N))
            c["t1"] = time.monotonic_ns()

        def prova_python(fino_ns):
            prec = time.monotonic_ns()
            buco = (0, prec, prec)
            while prec < fino_ns:
                ora = time.monotonic_ns()
                if ora - prec > buco[0]:
                    buco = (ora - prec, prec, ora)
                prec = ora
            return buco

        th = threading.Thread(target=tieni_gil, name="tiene_gil")
        th.start()
        r = st.inizio_callback(None, M, 0)
        pronto.set()
        buco = prova_python(time.monotonic_ns() + 2_000_000_000)
        th.join()
        st.fine_callback(r, False)
        st.chiudi()
        print(json.dumps({{"t0": c["t0"], "t1": c["t1"], "buco": buco, "N": N}}))
    """)
    p = subprocess.run([sys.executable, "-c", prog], check=True, timeout=120, capture_output=True, text=True)
    m = json.loads(p.stdout.strip().splitlines()[-1])
    t0, t1, (buco, b0, b1) = m["t0"], m["t1"], m["buco"]
    assert t1 - t0 >= 500_000_000, m
    # GIL really held: the callback thread stalled for nearly the whole call
    assert buco >= 400_000_000 and b0 >= t0 - 20_000_000 and b1 <= t1 + 20_000_000, m
    testo = (tmp_path / "gil_stack.txt").read_text()
    scatti = _stack_scattati(testo)
    assert len(scatti) == 1 and scatti[0][0] == "durata" and t0 <= scatti[0][1] <= t1
    assert "tieni_gil" in testo and "prova_python" in testo


def test_garbage_collector(tmp_path):
    st = sr.StrumentiRitardo(tmp_path / "rit.csv")
    n0 = len(gc.callbacks)
    gc.collect(2)
    st.chiudi()
    assert len(gc.callbacks) == n0 - 1
    righe = leggi(tmp_path / "rit_gc.csv")
    assert any(r["generazione"] == "2" for r in righe)
    r = [r for r in righe if r["generazione"] == "2"][-1]
    assert int(r["fine_mono_ns"]) >= int(r["inizio_mono_ns"]) and float(r["durata_ms"]) >= 0


def test_campionatore_cpu(tmp_path):
    st = sr.StrumentiRitardo(tmp_path / "rit.csv", periodo_cpu_s=0.05)
    time.sleep(0.4)
    st.chiudi()
    righe = leggi(tmp_path / "rit_cpu.csv")
    assert len(righe) >= 3
    r = righe[-1]
    assert int(r["procs_running"]) >= 1 and float(r["load1"]) >= 0
    assert int(r["nodo_thread"]) >= 2 and int(r["cpu_idle"]) > 0


def test_nomi_liberi(tmp_path):
    a = sr.StrumentiRitardo(tmp_path / "rit.csv")
    a.chiudi()
    b = sr.StrumentiRitardo(tmp_path / "rit.csv")
    b.chiudi()
    assert b.path.name == "rit_1.csv" and (tmp_path / "rit_1_stack.txt").exists()


def test_spento_di_default_nessun_import():
    prog = textwrap.dedent(f"""
        import sys, gc, threading
        sys.path.insert(0, {str(ROOT)!r})
        n_gc, n_thr = len(gc.callbacks), threading.active_count()
        from src.nodes import perturbation_node as pn
        assert "src.nodes.strumenti_ritardo" not in sys.modules
        assert pn.PerturbationNode._strum is None and pn.PerturbationNode._strum_rec is None
        assert len(gc.callbacks) == n_gc and threading.active_count() == n_thr
    """)
    subprocess.run([sys.executable, "-c", prog], check=True, timeout=120)


# --- real node, domain 88 ---

def _nodo_e_nuvole(tmp_path, nome, strumenti, n=30):
    """Passthrough node (--start-disabled) on its own topics; n clouds at 10 Hz;
    returns the output data received per stamp and the node exit code.
    """
    assert os.environ.get("ROS_DOMAIN_ID") == "88"
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import PointCloud2
    from src.nodes import perturbation_node as pn

    cart = tmp_path / nome
    cart.mkdir()
    pre = f"/prova_strumenti_{os.getpid()}"
    cmd = [sys.executable, "-u", str(ROOT / "src" / "nodes" / "perturbation_node.py"),
           "--start-disabled", "--min-latency-ms", "140",
           "--input-topic", f"{pre}/in", "--output-topic", f"{pre}/out",
           "--victim-topic", f"{pre}/victim", "--attacker-topic", f"{pre}/att",
           "--frame-log", str(cart / "frame_log.csv"), "--frame-log-delay"]
    if strumenti:
        cmd += ["--strumenti-ritardo", str(cart / "ritardo_nodo.csv")]
    env = dict(os.environ, ROS_DOMAIN_ID="88")
    nodo = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=open(cart / "nodo.log", "w"),
                            stderr=subprocess.STDOUT)
    ctx = rclpy.Context()
    rclpy.init(context=ctx)
    uscite = {}
    try:
        n_ros = rclpy.create_node("prova_strumenti", context=ctx)
        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(n_ros)
        n_ros.create_subscription(
            PointCloud2, f"{pre}/out",
            lambda m: uscite.__setitem__(m.header.stamp.sec * 10 + m.header.stamp.nanosec // 100_000_000,
                                         bytes(m.data)), 50)
        pub = n_ros.create_publisher(PointCloud2, f"{pre}/in", 10)
        t0 = time.time()
        while (pub.get_subscription_count() == 0 or n_ros.count_publishers(f"{pre}/out") == 0) \
                and time.time() - t0 < 60:
            ex.spin_once(timeout_sec=0.1)
        assert pub.get_subscription_count() > 0, "nodo non agganciato"
        time.sleep(1.0)
        rng = np.random.default_rng(0)
        dati = {}
        for k in range(n):
            xyz = np.c_[rng.uniform(-10, 10, (20000, 3)), np.full(20000, 100.0)].astype(np.float32)
            m = PointCloud2()
            m.header.frame_id = "base_link"
            m.header.stamp.sec, m.header.stamp.nanosec = 100 + k // 10, (k % 10) * 100_000_000
            m.height, m.width = 1, len(xyz)
            m.fields, m.point_step = pn.FIELDS, pn.POINT_STEP
            m.row_step, m.is_bigendian, m.is_dense = pn.POINT_STEP * len(xyz), False, True
            m.data = xyz.tobytes()
            dati[1000 + k] = xyz.tobytes()
            pub.publish(m)
            t = time.time()
            while time.time() - t < 0.1:
                ex.spin_once(timeout_sec=0.01)
        t = time.time()
        while len(uscite) < n and time.time() - t < 5:
            ex.spin_once(timeout_sec=0.05)
    finally:
        nodo.send_signal(signal.SIGINT)
        try:
            rc = nodo.wait(timeout=30)
        except subprocess.TimeoutExpired:
            nodo.kill()
            rc = None
        rclpy.shutdown(context=ctx)
    return cart, dati, uscite, rc


def test_nodo_con_e_senza_strumenti(tmp_path):
    cart_s, dati_s, usc_s, rc_s = _nodo_e_nuvole(tmp_path, "senza", False)
    cart_c, dati_c, usc_c, rc_c = _nodo_e_nuvole(tmp_path, "con", True)
    # outputs identical to the input in both (passthrough)
    for dati, usc in ((dati_s, usc_s), (dati_c, usc_c)):
        assert len(usc) >= 25
        assert all(usc[k] == dati[k] for k in usc)
    # without the option no instrument file, and a per-frame log with the same columns
    assert sorted(p.name for p in cart_s.iterdir()) == ["frame_log.csv", "nodo.log"]
    assert "strumenti del ritardo" not in (cart_s / "nodo.log").read_text()
    with open(cart_s / "frame_log.csv") as a, open(cart_c / "frame_log.csv") as b:
        assert a.readline() == b.readline()
    # with the option one row per cloud, DDS times present and ordered
    righe = leggi(cart_c / "ritardo_nodo.csv")
    assert len(righe) == len(leggi(cart_c / "frame_log.csv")) >= 25
    for r in righe:
        assert r["esito"] == "pubblicata"
        assert int(r["dds_recv_ns"]) >= int(r["dds_src_ns"]) > 0
        # received_timestamp on the wall clock: a few ms before the callback starts
        assert -5.0 < float(r["ritardo_ricezione_ms"]) < 1000.0
        assert int(r["cb_inizio_mono_ns"]) <= int(r["cb_fine_mono_ns"]) <= int(r["rilascio_mono_ns"])
        assert int(r["rilascio_mono_ns"]) - int(r["cb_inizio_mono_ns"]) >= 139_000_000
        assert (int(r["main_inizio_mono_ns"]) <= int(r["main_chiamata_mono_ns"])
                <= int(r["main_fine_mono_ns"]) <= int(r["registri_fine_mono_ns"]))
    assert np.median([float(r["ritardo_ricezione_ms"]) for r in righe]) < 50.0
    assert len(leggi(cart_c / "ritardo_nodo_cpu.csv")) >= 10
    assert (cart_c / "ritardo_nodo_gc.csv").exists() and (cart_c / "ritardo_nodo_cane.csv").exists()
    assert (cart_c / "ritardo_nodo_stack.txt").read_text().startswith("# stack del cane da guardia")
