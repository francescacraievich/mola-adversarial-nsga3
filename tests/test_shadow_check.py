"""Shadow-victim comparison (src/optimization/shadow_check.py): pure tests, no ROS."""
import csv
import math

import pytest

from src.optimization import shadow_check as shc


def _pose(t, x=1.0, y=2.0, yaw=0.1):
    return {"stamp_ns": int(round(t * 1e9)), "t_recv": 0.0, "x": x, "y": y, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": math.sin(yaw / 2), "qw": math.cos(yaw / 2), "cov": [0.0] * 36}


def _tr(t, sigma="0.200000", vx="0.500000", it="5.000000"):
    raw = {"timestamp": f"{t:.6f}", "ADAPTIVE_THRESHOLD_SIGMA": sigma, "vx": vx, "vy": "0.000000",
           "wz": "0.000000", "icp_quality": "0.950000", "icp_iterations": it, "twistCorrectionCount": "1.000000",
           "ESTIMATED_OBSERVATION_RADIUS": "25.000000", "INSTANTANEOUS_OBSERVATION_RADIUS": "25.000000"}
    return {"stamp_s": t, "raw": raw}


def _run(vt, st, vp=None, sp=None, final=True, crc=None, tol=None):
    return shc.compare(vp or [], sp or [], [], [], vt, st, None, None, tol=tol, final=final, crc=crc)


def test_abbinamento_per_stamp_e_livello_A():
    ts = [10.0 + 0.1 * i for i in range(5)]
    scans, info = _run([_tr(t) for t in ts], [_tr(t) for t in ts],
                       [_pose(t) for t in ts], [_pose(t) for t in ts])
    assert info["n_comuni"] == 5 and info["episodi"] == []
    assert {s["livello"] for s in scans.values()} == {"A"}
    assert info["primo_stamp_vittima_s"] == info["primo_stamp_ombra_s"] == 10.0


def test_livelli_B_Bprimo_C_Cpiu():
    t = [10.0]
    lv = lambda dx=0.0, dyaw=0.0, sig="0.200000", it="5.000000": _run(
        [_tr(10.0)], [_tr(10.0, sigma=sig, it=it)], [_pose(10.0)], [_pose(10.0, x=1.0 + dx, yaw=0.1 + dyaw)])[0][100]["livello"]
    assert lv(dx=5e-10) == "B"                      # pose not bit-exact, within 1e-9
    assert lv(dx=5e-5) == "B'"                      # beyond B, within 1e-4
    assert lv(sig="0.200001") == "B"                # one unit in the sixth digit, isolated scan
    assert lv(sig="0.200005") == "B'"               # |dsigma| 5e-6: beyond B, within B' (1e-5)
    assert lv(sig="0.200020") == "C"                # |dsigma| 2e-5 > 1e-5
    assert lv(it="6.000000") == "C"                 # different discrete quantity
    assert lv(dx=5e-4) == "C+"                      # |dxy| > 4e-4
    assert lv(dyaw=math.radians(0.02)) == "C+"
    del t


def test_una_unita_non_isolata_non_e_B():
    ts = [10.0, 10.1]
    scans, _ = _run([_tr(t) for t in ts], [_tr(t, sigma="0.200001") for t in ts])
    assert scans[100]["livello"] == "B'" and scans[101]["livello"] == "B'"


def test_scan_di_una_sola_ed_episodi():
    v = [_tr(10.0 + 0.1 * i) for i in range(30)]
    s = [r for i, r in enumerate(v) if i not in (12, 13, 25)]
    scans, info = _run(v, s)
    assert info["solo_vittima"]["in_corsa"] == 3
    assert [e["stamp_s"] for e in info["episodi"]] == [[11.2, 11.3], [12.5]]   # second episode
    assert shc.run_verdict(info)["esito"].startswith("ombra non affidabile")


def test_avvio_e_coda():
    v = [_tr(10.0 + 0.1 * i) for i in range(10)]
    s = v[3:8]
    for final in (False, True):
        _, info = _run(v, s, final=final)
        # beyond the last common stamp: tail not comparable, never an episode (even at the end)
        assert info["solo_vittima"] == {"avvio": 3, "coda": 2, "in_corsa": 0}
        assert info["episodi"] == [] and info["ultimo_stamp_confrontabile_s"] == 10.7


def test_crc_omesso_non_e_episodio():
    v = [_tr(10.0 + 0.1 * i) for i in range(10)]
    s = [r for i, r in enumerate(v) if i != 5]
    crc = {shc.key10(10.0 + 0.1 * i): {"crc_victim": "1", "crc_shadow": "" if i == 5 else "1",
                                        "shadow_omitted": "1" if i == 5 else "0", "t_pub_wall": "0"}
           for i in range(10)}
    scans, info = _run(v, s, crc=crc)
    assert info["episodi"] == [] and scans[105]["crc"] == "omesso" and info["crc"]["omessi"] == 1


def test_finestra_marcata_e_ingresso():
    v = [_tr(10.0 + 0.1 * i) for i in range(10)]
    s = [_tr(10.0 + 0.1 * i, sigma="0.300000" if i == 7 else "0.200000") for i in range(10)]
    scans, info = _run(v, s)
    w1 = shc.window_summary(scans, info, None, 10.5, False)
    w2 = shc.window_summary(scans, info, 10.5, None, True)
    assert w1["livello_peggiore"] == "A" and not w1["marcata"]
    assert w2["livello_peggiore"] == "C+" and w2["marcata"] and w2["ingresso"]


def test_keyframe_non_misurato(tmp_path):
    p = tmp_path / "mola_victim.log"
    p.write_text("riga senza debug\n")
    assert shc.read_kf(str(p)) is None
    _, info = shc.compare([], [], [], [], [_tr(10.0)], [_tr(10.0)], None, (3, 10))
    assert info["keyframe_stato"] == "non misurato" and info["keyframe"] is None


def test_soglie_configurabili():
    tol = shc.parse_tol("bp_xy_m=1e-3, cp_sigma=none")
    assert tol["bp_xy_m"] == 1e-3 and tol["cp_sigma"] is None and tol["b_xy_m"] == 1e-9
    with pytest.raises(ValueError):
        shc.parse_tol("pippo=1")
    with pytest.raises(ValueError):
        shc.parse_tol("bp_v=-1")


def test_pose_17_cifre_rilette_identiche(tmp_path):
    r = _pose(10.0, x=1.0 / 3.0, y=-2.0e-17)
    p = tmp_path / "p.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(shc.POSE_COLS)
        w.writerow(shc.pose_row("victim", r))
    back = shc.read_pose_csv(str(p))["victim"][0]
    assert back["x"] == r["x"] and back["y"] == r["y"] and back["qz"] == r["qz"]
    assert len(shc.pose_row("victim", r)[3].replace("0.", "").lstrip("0")) >= 16


def test_ambiente_identico_a_meno_del_namespace():
    d = shc.env_config_diff({"ROS_ARGS": '["--ros-args", "-r", "__ns:=/victim"]', "A": "1"},
                            {"ROS_ARGS": '["--ros-args", "-r", "__ns:=/ombra"]', "A": "1"})
    assert d["identica"]
    assert shc.env_config_diff({"A": "1"}, {"A": "2"})["diverse"] == ["A"]
