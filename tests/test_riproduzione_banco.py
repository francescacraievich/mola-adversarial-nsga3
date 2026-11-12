"""Pure parts of the reproduction bench: DDS profiles, environments, player, analysis."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _carica(nome):
    spec = importlib.util.spec_from_file_location(nome, ROOT / "scripts" / f"{nome}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


reg = _carica("riproduzione_regista")
pla = _carica("riproduzione_player")
ana = _carica("riproduzione_analisi")


def test_profilo_completo_come_quello_della_serie_b():
    import xml.etree.ElementTree as ET
    t = reg.profilo_xml("completo")
    ET.fromstring(t.split("\n", 1)[1])
    assert "<segment_size>16777216</segment_size>" in t and "<sendBufferSize>4194304</sendBufferSize>" in t
    assert "<useBuiltinTransports>false</useBuiltinTransports>" in t


def _valori(tipo):
    """Effective profile values: SHM segment, descriptor UDP buffers, builtin transports."""
    import xml.etree.ElementTree as ET
    ns = {"p": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    r = ET.fromstring(reg.profilo_xml(tipo).split("\n", 1)[1])
    seg = [int(x.text) for x in r.findall(".//p:transport_descriptor[p:type='SHM']/p:segment_size", ns)]
    udp = r.find(".//p:transport_descriptor[p:type='UDPv4']", ns)
    snd = udp.find("p:sendBufferSize", ns) if udp is not None else None
    rcv = udp.find("p:receiveBufferSize", ns) if udp is not None else None
    usati = [x.text for x in r.findall(".//p:userTransports/p:transport_id", ns)]
    ids = [x.text for x in r.findall(".//p:transport_descriptor/p:transport_id", ns)]
    return {"seg": seg, "snd": int(snd.text) if snd is not None else None,
            "rcv": int(rcv.text) if rcv is not None else None,
            "builtin": r.find(".//p:useBuiltinTransports", ns).text, "usati_dichiarati": sorted(usati) == sorted(ids)}


@pytest.mark.parametrize("tipo,seg,buf", [
    ("completo", 16777216, 4194304), ("buffer", 524288, 4194304), ("shm", 16777216, None),
    ("shm1M", 1048576, None), ("shm2M", 2097152, None)])
def test_profili_valori_effettivi(tipo, seg, buf):
    v = _valori(tipo)
    # builtin transports off: no segment derived from the participant buffers
    assert v["builtin"] == "false" and v["usati_dichiarati"]
    assert v["seg"] == [seg]
    assert v["snd"] == buf and v["rcv"] == buf


def test_niente_buffer4M():
    assert "buffer4M" not in reg.PROFILI


def test_profili_su_file_nuovi_e_non_sovrascritti(tmp_path):
    p = reg.scrivi_profili(tmp_path)
    assert set(p) == set(reg.PROFILI)
    reg.scrivi_profili(tmp_path)          # equal: no error
    Path(p["shm"]).write_text("altro")
    with pytest.raises(RuntimeError):
        reg.scrivi_profili(tmp_path)


def test_profili_nel_repo_uguali_al_generatore():
    for tipo in reg.PROFILI:
        f = reg.percorso_profilo(tipo)
        assert f.exists() and f.read_text() == reg.profilo_xml(tipo)


def test_ambiente_dds():
    base = {"FASTRTPS_DEFAULT_PROFILES_FILE": "x", "RMW_FASTRTPS_PUBLICATION_MODE": "SYNCHRONOUS", "A": "1"}
    e = reg.ambiente_dds(base, None, "default")
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in e and "RMW_FASTRTPS_PUBLICATION_MODE" not in e and e["A"] == "1"
    e = reg.ambiente_dds({}, "/p.xml", "async")
    assert e["FASTRTPS_DEFAULT_PROFILES_FILE"] == e["FASTDDS_DEFAULT_PROFILES_FILE"] == "/p.xml"
    assert e["RMW_FASTRTPS_PUBLICATION_MODE"] == "ASYNCHRONOUS"
    assert reg.ambiente_dds({}, None, "sync")["RMW_FASTRTPS_PUBLICATION_MODE"] == "SYNCHRONOUS"


def test_ambienti_mola_per_ruolo():
    env = {"LD_LIBRARY_PATH": "/opt/ros/jazzy/lib", "MOLA_MODULES_LIB_PATH": "/opt/ros/jazzy/lib"}
    a = reg.ambienti_mola(False, environ=env)
    assert "/victim" in a["victim"]["ROS_ARGS"] and "/ombra" in a["shadow"]["ROS_ARGS"]
    assert "LD_LIBRARY_PATH" not in a["victim"] and "LD_LIBRARY_PATH" not in a["attacker"]
    assert a["shadow"]["MOLA_FULL_STATE"] == "1" and a["twin"]["MOLA_FULL_STATE"] == "1"
    assert a["shadow"]["LD_LIBRARY_PATH"] == a["twin"]["LD_LIBRARY_PATH"] != env["LD_LIBRARY_PATH"]
    assert "use_sim_time:=true" in a["twin"]["ROS_ARGS"] and "use_sim_time:=true" in a["victim"]["ROS_ARGS"]
    u = reg.ambienti_mola(True, environ=env)
    assert u["victim"]["LD_LIBRARY_PATH"] == u["shadow"]["LD_LIBRARY_PATH"]


def test_comando_nodo_serie_b(tmp_path):
    c = reg.comando_nodo(tmp_path)
    for o in ("--victim-never-clean", "--frame-log-delay", "--report-matched", "--twin-clock-base"):
        assert o in c
    assert c[c.index("--min-latency-ms") + 1] == "140"


def test_player_stamp_cdr_e_ordine():
    raw = b"\x00\x01\x00\x00" + struct.pack("<iI", 5, 7) + b"resto"
    nuovo = pla.stamp_cdr(raw, 12, 345)
    assert pla.leggi_stamp_cdr(nuovo) == pytest.approx(12.000000345) and nuovo.endswith(b"resto")
    assert pla.ordine(4, "pingpong", 0) == [0, 1, 2, 3]
    assert pla.ordine(4, "pingpong", 1) == [2, 1, 0]
    assert pla.ordine(4, "pingpong", 2) == [1, 2, 3]
    assert pla.ordine(3, "loop", 1) == [0, 1, 2]


def test_player_intervalli():
    assert pla.intervalli([1.0, 1.2, 1.2, 1.5, 4.5]) == pytest.approx([0.2, 0.3, 3.0])
    assert pla.intervalli([1.0, 1.2, 4.2], taglio=1.0) == pytest.approx([0.2, 1.0])
    passi, med = pla.passi_stamp([0.0, 0.1, 0.1, 0.3])
    assert med == pytest.approx(0.2) and passi == pytest.approx([0.1, 0.2, 0.2])


def test_analisi_eta_mai_prese_e_finestra():
    righe = [(10.0, 1.0, 9.9), (10.2, 1.1, 10.1), (10.4, 1.2, 10.3), (10.6, 1.3, 10.5)]
    prese = [(10.05, 1.0), (10.9, 1.2)]
    e = ana.eta_prese(prese, righe)
    assert [round(x[2], 3) for x in e] == [0.15, 0.6]
    assert [s for _, s, _ in ana.mai_prese(prese, righe, 10.0, 10.5)] == [1.1]
    assert ana.finestra_gemello([], 20.0, 0.4) is None
    assert ana.finestra_gemello(prese, 12.0, 0.4) == (10.05, pytest.approx(11.6))


def test_analisi_stalli_e_avvii():
    ev = [(1.0, 0.5, "victim"), (3.0, 0.9, "shadow"), (20.0, 1.1, "g3")]
    assert ana.stalli(ev, 5.0) == [(1.0, 3.0, 0.9, 2, {"victim", "shadow"}), (20.0, 20.0, 1.1, 1, {"g3"})]
    assert ana.stalli(ev, 300.0)[0][:4] == (1.0, 20.0, 1.1, 3)
    # by start count: 2 twins between 3 and 20 s, merged with at most 2, separate with at most 1
    assert len(ana.stalli(ev, 5.0, [5.0, 10.0], 2)) == 1
    assert len(ana.stalli(ev, 5.0, [5.0, 10.0], 1)) == 2
    avvii = [{"t_avvio": 0.0, "consegnate": 50}, {"t_avvio": 5.0, "consegnate": 40},
             {"t_avvio": 30.0, "consegnate": 10}]
    assert ana.avvii_prima(avvii, 10.0) == (2, 90)


def test_associazione_gemello():
    righe = [(10.0, 1.0, 10.0), (10.2, 1.1, 10.2), (10.4, 1.2, 10.4), (10.6, 1.3, 10.6)]
    prese = [(11.0, 1.0), (11.1, 1.2), (11.2, 1.3)]
    pp, up = ana.associazione(prese, righe, 9.9)
    assert pp == pytest.approx(1.1) and up == pytest.approx(0.3)
    assert ana.associazione([], righe, 9.9) == (None, None)


def test_analisi_frame_log_e_publish(tmp_path):
    intest = ("t_arr_wall,stamp_victim,stamp_main_out,t_dds_src,t_release_mono,"
              "t_pub_main_end_mono,t_pub_victim_end_mono\n")
    (tmp_path / "frame_log.csv").write_text(intest + "10,1.0,1.0,9.9,100.0,100.002,100.004\n")
    (tmp_path / "frame_log_1.csv").write_text(intest + "11,2.0,,10.9,101.0,101.001,\n")
    assert len(ana.righe_frame_log(tmp_path / "frame_log.csv", "stamp_victim")) == 2
    assert len(ana.righe_frame_log(tmp_path / "frame_log.csv", "stamp_main_out")) == 1
    d = ana.durate_publish(tmp_path / "frame_log.csv")
    assert d["main"] == pytest.approx([2.0, 1.0], abs=1e-6) and d["victim"] == pytest.approx([4.0], abs=1e-6)
    assert ana.quantili(d["main"])["max"] == pytest.approx(2.0)


def test_genoma_da_history_e_file(tmp_path):
    h = [{"window": 3, "genome": [0.0] * 17}, {"window": 4, "genome": [0.5] * 17, "vincitore_idx": 45}]
    (tmp_path / "history.json").write_text(__import__("json").dumps(h))
    g, f = reg.genoma_da_history(tmp_path / "history.json", 4)
    assert g == [0.5] * 17 and f["vincitore_idx"] == 45
    with pytest.raises(ValueError):
        reg.genoma_da_history(tmp_path / "history.json", 7)
    assert reg.leggi_genoma("nullo", 17)[0] == [-1.0] * 17
    (tmp_path / "g.json").write_text("[0.1, 0.2]")
    assert reg.leggi_genoma(str(tmp_path / "g.json"), 2)[0] == [0.1, 0.2]
    assert "--operator-mode" in reg.comando_nodo(tmp_path, "insieme")
    assert "--operator-mode" not in reg.comando_nodo(tmp_path, "random")


def test_istogramma_e_ripartenza():
    assert ana.istogramma_eta([0.5, 0.9, 1.0, 1.3, 0.3]) == {"0.4-0.8": 1, "0.8-1.2": 2, ">1.2": 1}
    serie = [(10.0, 1.0, 0.9), (11.0, 1.1, 0.2), (12.0, 1.2, 0.1)]
    assert ana.prima_presa_dopo(serie, 10.0, 0.4) == 11.0
    assert ana.prima_presa_dopo(serie, 12.0, 0.4) is None


def test_media_colonna(tmp_path):
    (tmp_path / "s.csv").write_text("nodo_proc_ms_mean\n80\n\n76\n")
    assert ana.media_colonna(tmp_path / "s.csv", "nodo_proc_ms_mean") == (78.0, 2)


def test_stalli_senza_avvii_non_si_uniscono_per_avvii():
    # check without twins: two events 1000 s apart stay two periods
    p = ana.stalli([(0.0, 0.5, "v"), (1000.0, 0.6, "v")], 300.0, t_avvii=(), unisci_avvii=60)
    assert len(p) == 2
    # with a start in between they merge even beyond 300 s
    p = ana.stalli([(0.0, 0.5, "v"), (1000.0, 0.6, "v")], 300.0, t_avvii=(500.0,), unisci_avvii=60)
    assert len(p) == 1


def test_riavvio_mola_dovuto_e_tag():
    assert not any(reg.riavvio_mola_dovuto(k, 0, 1200) for k in range(1200))
    assert [k for k in range(1200) if reg.riavvio_mola_dovuto(k, 300, 1200)] == [299]
    # not needed after the last twin
    assert not any(reg.riavvio_mola_dovuto(k, 10, 10) for k in range(10))
    assert reg.tag_riavvio("victim", 1) == "victim" and reg.tag_riavvio("victim", 2) == "victim_2"


def test_file_log_mola(tmp_path):
    assert ana.file_log_mola(tmp_path, "victim") == []
    for n in ("mola_victim.log", "mola_victim_2.log", "mola_victim_4.log", "mola_shadow.log"):
        (tmp_path / n).write_text("")
    # consecutive: _4 without _3 does not count
    assert [p.name for p in ana.file_log_mola(tmp_path, "victim")] == ["mola_victim.log", "mola_victim_2.log"]
    assert [p.name for p in ana.file_log_mola(tmp_path, "shadow")] == ["mola_shadow.log"]


def test_concatenazione_e_finestre_di_riavvio():
    l1 = [(10.0, 1.0), (10.1, 1.1)]
    l2 = [(25.0, 2.0), (25.1, 2.1)]
    assert ana.concatena_prese([l1, l2]) == l1 + l2
    assert ana.concatena_prese([l1]) == l1
    assert ana.finestre_riavvio([], "victim", [l1], 0.4, 1.6) == []
    r = [{"segnale_wall": 12.0, "avvio_wall": 20.0, "avvio_ruolo_wall": {"victim": 21.0}}]
    # cap from the start of the role process, not from the first take
    assert ana.finestre_riavvio(r, "victim", [l1, l2], 0.4, 1.6) == [
        (pytest.approx(11.6), pytest.approx(22.6), "segnale", "avvio_piu_tetto")]
    assert ana.finestre_riavvio(r, "shadow", [l1, l2], 0.4, 1.6)[0][1] == pytest.approx(21.6)
    # without times: from the last previous take to the first new one; without takes: to the end
    assert ana.finestre_riavvio([], "victim", [l1, l2], 0.4, 1.6) == [(10.1, 25.0, "ultima_presa", "prima_presa")]
    assert ana.finestre_riavvio([], "victim", [l1, []], 0.4, 1.6)[0][1] == float("inf")


def test_separa_prese_per_arrivo():
    righe = [(10.0, 1.0, 9.9), (20.5, 2.0, 20.4), (23.0, 2.1, 22.9)]
    eta = [(10.1, 1.0, 0.2), (21.7, 2.0, 1.3), (23.6, 2.1, 0.7)]
    assert ana.separa_prese(eta, righe, []) == (eta, [])
    contate, tolte = ana.separa_prese(eta, righe, [(11.6, 22.6)])
    # first old take (arrived inside the window) removed; take delayed past the cap counted
    assert tolte == [eta[1]] and contate == [eta[0], eta[2]]


def test_mai_prese_con_finestra_esclusa():
    righe = [(10.0, 1.0, 9.9), (11.0, 1.1, 10.9), (12.0, 1.2, 11.9), (13.0, 1.3, 12.9), (14.0, 1.4, 13.9)]
    prese = [(10.1, 1.0), (14.1, 1.4)]
    # default: unchanged
    assert ana.mai_prese(prese, righe, 10.0, 14.0) == ana.mai_prese(prese, righe, 10.0, 14.0, ())
    assert [s for _, s, _ in ana.mai_prese(prese, righe, 10.0, 14.0)] == [1.1, 1.2, 1.3]
    # window [11.5, 13.0): 1.2 excluded, 1.3 counted (right end open)
    assert [s for _, s, _ in ana.mai_prese(prese, righe, 10.0, 14.0, [(11.5, 13.0)])] == [1.1, 1.3]


def _riga_presa(t, stamp):
    import time
    lt = time.localtime(t)
    return (f"[{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec + (t % 1):09.6f}|DEBUG|Generator (raw)] "
            f"Processing observation x stamp={stamp:.6f}\n")


def _prova_sintetica(cartella, riavvio):
    """Victim taking one cloud every 0.1 s; with restart: close at t0+5, new process
    taking from t0+20; clouds in between are not taken.
    """
    import json
    t0 = 1761750000.0
    (cartella / "mola").mkdir(parents=True)
    (cartella / "analisi").mkdir()
    righe = ["t_arr_wall,stamp_victim,stamp_main_out,t_dds_src"]
    p1, p2 = [], []
    for i in range(300):
        t = t0 + i * 0.1
        stamp = 100.0 + i * 0.1
        righe.append(f"{t:.3f},{stamp:.3f},,{t - 0.05:.3f}")
        if not riavvio or t < t0 + 5.0:
            p1.append(_riga_presa(t + 0.05, stamp))
        elif t >= t0 + 20.0:
            p2.append(_riga_presa(t + 0.05, stamp))
    (cartella / "frame_log.csv").write_text("\n".join(righe) + "\n")
    (cartella / "mola" / "mola_victim.log").write_text("".join(p1))
    orari = {"inizio_wall": t0}
    if riavvio:
        (cartella / "mola" / "mola_victim_2.log").write_text("".join(p2))
        orari["riavvii_mola"] = [{"dopo_avvii": 300, "indice": 2, "segnale_wall": t0 + 5.0,
                                  "avvio_wall": t0 + 6.05, "avvio_ruolo_wall": {"victim": t0 + 6.05}}]
    (cartella / "orari.json").write_text(json.dumps(orari))


def test_analizza_senza_e_con_riavvio_mola(tmp_path):
    _prova_sintetica(tmp_path / "a", False)
    rie, _, _ = ana.analizza(tmp_path / "a")
    assert rie["victim"]["n_prese"] == 300 and rie["victim"]["n_mai_prese"] == 0
    assert "log" not in rie["victim"] and "riavvio_mola" not in rie
    _prova_sintetica(tmp_path / "b", True)
    rie, serie, _ = ana.analizza(tmp_path / "b")
    v = rie["victim"]
    # chained takes; window [t0+4.6; t0+7.65): the 27 clouds that arrived there and were
    # never taken are removed; the 123 after the cap and before the first take (t0+20)
    # count as never taken
    assert v["n_prese"] == 50 + 100 and [x["n_prese"] for x in v["log"]] == [50, 100]
    assert v["n_mai_prese_nelle_finestre_escluse"] == 27 and v["n_mai_prese"] == 123
    f = v["finestre_escluse"][0]
    assert f["inizio_da"] == "segnale" and f["fine_da"] == "avvio_piu_tetto"
    assert f["durata_s"] == pytest.approx(3.05, abs=0.01)
    # the 4 takes of the old process arrived in the last 0.4 s before the signal are not counted
    assert v["n_prese_nelle_finestre_escluse"] == 4 and "riavvio_mola" in rie


PROFILO_GEMELLI = ROOT / "config" / "dds" / "riproduzione_gemelli_rcvbuf_ridotto.xml"
MAX_MESSAGGIO = 65500     # s_maximumMessageSize of Fast DDS 2.14.6, default maxMessageSize
RB_FABBRICA = 212992      # rmem_default: rb of the UDP sockets with default buffers


def test_profilo_gemelli_valori_dal_file():
    import xml.etree.ElementTree as ET
    ns = {"p": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    r = ET.parse(PROFILO_GEMELLI).getroot()
    shm = r.findall(".//p:transport_descriptor[p:type='SHM']", ns)
    udp = r.findall(".//p:transport_descriptor[p:type='UDPv4']", ns)
    assert len(shm) == 1 and len(udp) == 1
    assert [int(x.text) for x in shm[0].findall("p:segment_size", ns)] == [reg.SHM_FABBRICA]
    # one lever only: receiveBufferSize, with default send size and max message size
    figli = {c.tag.split("}")[1] for c in udp[0]}
    assert figli == {"transport_id", "type", "receiveBufferSize"}
    rcv = int(udp[0].find("p:receiveBufferSize", ns).text)
    usati = [x.text for x in r.findall(".//p:userTransports/p:transport_id", ns)]
    ids = [x.text for x in r.findall(".//p:transport_descriptor/p:transport_id", ns)]
    assert sorted(usati) == sorted(ids)
    assert r.find(".//p:useBuiltinTransports", ns).text == "false"
    parts = r.findall(".//p:participant", ns)
    assert len(parts) == 1 and parts[0].get("is_default_profile") == "true"
    assert r.find(".//p:listenSocketBufferSize", ns) is None and r.find(".//p:sendSocketBufferSize", ns) is None
    # accepted by UDPTransportInterface::init (receiveBufferSize >= maxMessageSize);
    # the kernel doubles it
    assert rcv == MAX_MESSAGGIO
    assert 2 * rcv == 131000 < RB_FABBRICA


def test_ambienti_processi_profilo_ai_soli_gemelli():
    base = reg.ambiente_dds({"A": "1", "FASTRTPS_DEFAULT_PROFILES_FILE": "x"}, None, "default")
    e = reg.ambienti_processi(base, PROFILO_GEMELLI)
    p = str(PROFILO_GEMELLI.resolve())
    assert e["twin"]["FASTRTPS_DEFAULT_PROFILES_FILE"] == e["twin"]["FASTDDS_DEFAULT_PROFILES_FILE"] == p
    assert {k: v for k, v in e["twin"].items() if not k.endswith("_DEFAULT_PROFILES_FILE")} == base
    for r_ in ("player", "nodo", "victim", "shadow", "attacker"):
        assert e[r_] == base and "FASTRTPS_DEFAULT_PROFILES_FILE" not in e[r_]
        assert "FASTDDS_DEFAULT_PROFILES_FILE" not in e[r_]
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in base


def test_ambienti_processi_invarianza_senza_profilo():
    base = reg.ambiente_dds({"A": "1"}, "/p.xml", "sync")
    for prof in (None, ""):
        e = reg.ambienti_processi(base, prof)
        assert set(e) == set(reg.RUOLI_PROCESSI) and all(v == base for v in e.values())


def test_comando_nodo_riavviato_usa_un_registro_crc_nuovo(tmp_path):
    # the restarted node (RIAVVIO_NODO_A) must not find the crc log of the first node
    c0 = reg.comando_nodo(tmp_path)
    c1 = reg.comando_nodo(tmp_path, "insieme", 1)
    assert c0[c0.index("--shadow-crc-log") + 1] == str(tmp_path / "shadow_crc.csv")
    assert c1[c1.index("--shadow-crc-log") + 1] == str(tmp_path / "shadow_crc_1.csv")
