"""Shadow-victim comparison, edge cases: pure tests, no ROS node."""
import csv
import time
from pathlib import Path

import pytest

from src.optimization import shadow_check as shc

DATI = Path(__file__).parent / "dati"
REALI = DATI / "traces_reali_37c_victim_righe1-51.csv"   # recorded traces_victim.csv


def _scrivi(path, rows, hdr_raw, tronca=None):
    with open(path, "w", newline="") as f:
        f.write(",".join(f'"{h}"' if h else "" for h in hdr_raw) + "\n")
        for r in rows:
            f.write(",".join(r.get(h, "") if h else "" for h in hdr_raw) + "\n")
        if tronca:
            f.write(tronca)


def _reali():
    with open(REALI) as f:
        rd = csv.reader(f)
        hdr_raw = next(rd)
        hdr = [h for h in hdr_raw if h]
        rows = [dict(zip(hdr, r)) for r in rd if len(r) == len(hdr_raw)]
    return hdr_raw, [r for r in rows if float(r["current_relative_timestamp"]) != 0.0]


def test_B1_righe_reali_diverse_solo_in_time_onLidar(tmp_path):
    hdr_raw, rows = _reali()
    assert len(rows) == 50 and "time_onLidar" in rows[0]
    altre = [dict(r, time_onLidar=f"{float(r['time_onLidar']) + 1e-5:.6f}") for r in rows]
    _scrivi(tmp_path / "v.csv", rows, hdr_raw)
    _scrivi(tmp_path / "s.csv", altre, hdr_raw)
    vt, st = shc.read_traces(str(tmp_path / "v.csv")), shc.read_traces(str(tmp_path / "s.csv"))
    scans, info = shc.compare([], [], [], [], vt, st, None, None, final=True)
    assert info["n_comuni"] == 50
    # time_onLidar excluded: the rows stay identical (class A)
    assert {s["livello"] for s in scans.values()} == {"A"}


def test_B1_tutte_le_colonne_classificate():
    hdr_raw, _ = _reali()
    cols = {h for h in hdr_raw if h}
    assert len(cols) == 25
    assert cols == set(shc.CSV_CONT) | set(shc.CSV_DISC) | set(shc.CSV_EXCLUDED)


def test_B1_colonna_di_stato_oltre_unita_e_C(tmp_path):
    hdr_raw, rows = _reali()
    altre = [dict(r) for r in rows]
    altre[20]["robot_z"] = f"{float(rows[20]['robot_z']) + 5e-6:.6f}"      # class "altro": no B'
    altre[30]["twistCorrectionCount"] = "9.000000"                        # discrete
    _scrivi(tmp_path / "v.csv", rows, hdr_raw)
    _scrivi(tmp_path / "s.csv", altre, hdr_raw)
    scans, _ = shc.compare([], [], [], [], shc.read_traces(str(tmp_path / "v.csv")),
                           shc.read_traces(str(tmp_path / "s.csv")), None, None, final=True)
    lv = [s["livello"] for _, s in sorted(scans.items())]
    assert lv[20] == "C" and lv[30] == "C" and lv.count("A") == 48


def test_B2_avviso_nel_formato_MRPT(tmp_path):
    # real row from a recorded mola_victim.log
    p = tmp_path / "mola_shadow.log"
    p.write_text("[10:59:21.5424|WARN |LidarOdometry] Dropping LiDAR scan (worker busy: keeping only the "
                 "freshest); dropped frames: 0.15%\n[INFO] [1792832356.598667114] [x]: altro\n")
    assert shc.read_drop_warn(str(p)) == [pytest.approx(10 * 3600 + 59 * 60 + 21.5424)]


def _episodio(drop_s, imu_assente, t_pub):
    v = [{"stamp_s": 10.0 + 0.1 * i, "raw": {"timestamp": f"{10.0 + 0.1 * i:.6f}", "vx": "0.1"}} for i in range(10)]
    s = [r for i, r in enumerate(v) if i != 5]
    crc = {shc.key10(r["stamp_s"]): {"crc_victim": "1", "crc_shadow": "1", "shadow_omitted": "0",
                                      "t_pub_wall": repr(t_pub)} for r in v}
    _, info = shc.compare([], [], [], [], v, s, None, None, final=True, crc=crc, drop_s=drop_s,
                          imu_assente=imu_assente)
    return info["episodi"][0]["causa"]


def test_B2_causa_worker_trasporto_non_determinata():
    t_pub = time.mktime((2026, 10, 24, 10, 59, 21, 0, 0, -1)) + 0.2
    tod = shc.seconds_of_day(t_pub)
    assert _episodio([tod - 0.4], False, t_pub) == "worker"
    assert _episodio([tod - 30.0], False, t_pub) == "non determinata"
    assert _episodio([], True, t_pub) == "trasporto"
    assert _episodio([tod - 0.4], False, None) == "non determinata"    # without publication time


def test_B3_coda_tronca_non_e_episodio(tmp_path):
    hdr_raw, rows = _reali()
    _scrivi(tmp_path / "v.csv", rows, hdr_raw)
    # the shadow has 10 fewer rows and the last is truncated, as when closing without flush
    tronca = ",".join(rows[40].get(h, "") if h else "" for h in hdr_raw)[:57]
    _scrivi(tmp_path / "s.csv", rows[:40], hdr_raw, tronca=tronca)
    st = shc.read_traces(str(tmp_path / "s.csv"))
    assert len(st) == 40                                                  # truncated row dropped
    scans, info = shc.compare([], [], [], [], shc.read_traces(str(tmp_path / "v.csv")), st, None, None, final=True)
    assert info["episodi"] == [] and info["solo_vittima"]["coda"] == 10
    assert sum(1 for s in scans.values() if s.get("coda_non_confrontabile")) == 10


def test_B4_finestra_con_episodio_marcata_a_livello_A():
    v = [{"stamp_s": 13.0 + 0.1 * i, "raw": {"timestamp": f"{13.0 + 0.1 * i:.6f}", "vx": "0.1"}} for i in range(15)]
    s = [r for i, r in enumerate(v) if i != 7]                            # 13.7 s victim only
    scans, info = shc.compare([], [], [], [], v, s, None, None, final=True)
    assert [e["stamp_s"] for e in info["episodi"]] == [[13.7]]
    w = shc.window_summary(scans, info, None, None, False)
    assert w["livello_peggiore"] == "A" and w["episodi_nella_finestra"] == 1 and w["marcata"]


def test_C1_livello_sull_ultima_posa_senza_traccia():
    import math
    p = lambda t, x: {"stamp_ns": int(round(t * 1e9)), "t_recv": 0.0, "x": x, "y": 0.0, "z": 0.0,
                      "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "cov": [0.0] * 36}
    v = [{"stamp_s": 10.0, "raw": {"timestamp": "10.000000"}}]
    scans, info = shc.compare([p(10.0, 1.0), p(10.1, 1.0)], [p(10.0, 1.0), p(10.1, 1.0 + 5e-4)], [], [],
                              v, v, None, None)
    w = shc.window_summary(scans, info, None, None, False)
    assert w["livello_ultimo_scan"] == "A" and w["stamp_ultima_posa"] == 10.1
    assert w["livello_ultima_posa"] == "C+"
    del math


def test_C6_registro_crc_esistente_nome_nuovo(tmp_path):
    # node restarted at attempt 2 after a stall: the attempt-1 file stays intact
    # and the node writes crc_1.csv (no FileExistsError)
    pn = pytest.importorskip("src.nodes.perturbation_node")
    f = tmp_path / "crc.csv"
    f.write_text("vecchio\n")
    log = pn.ShadowCrcLog(str(f))
    log.close()
    assert f.read_text() == "vecchio\n"
    assert log.path == tmp_path / "crc_1.csv" and log.path.exists()
