"""Live logs (--registri-dal-vivo, REGISTRI_DAL_VIVO=1): format of the three CSV files, option
off with no files or calls, pose age, command reason, two-argument callback for MessageInfo,
command wait while paused. No ROS node, no simulation.

    python3 -m pytest tests/test_registri_dal_vivo.py -q
"""
import csv
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimization import registri_dal_vivo as re_  # noqa: E402


def _msg(sec=12, nsec=500_000_000, x=1.0, y=2.0, yaw=0.5):
    q = SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nsec)),
                           pose=SimpleNamespace(pose=SimpleNamespace(
                               position=SimpleNamespace(x=x, y=y, z=0.0), orientation=q)))


def _righe(path):
    with open(path) as f:
        return list(csv.reader(f))


def test_intestazioni_esatte():
    assert re_.POSE_COLONNE == ("ruolo", "tag", "stamp_nuvola_s", "t_pub_src_ns", "t_ric_dds_ns",
                                "t_ric_wall_s", "t_ric_sim_s", "x", "y", "yaw")
    assert re_.CONTROLLORE_COLONNE == (
        "ruolo", "tag", "t_wall_s", "t_sim_s", "stamp_posa_usata_s", "eta_posa_s",
        "eta_posa_wall_s", "est_x", "est_y", "est_yaw", "est_dist", "cmd_v", "cmd_w",
        "t_pub_cmd_wall_s", "motivo", "stamp_posa_grezzo_s", "offset_stamp_s")
    assert re_.ISAAC_COLONNE == (
        "t_ric_wall_s", "t_ric_sim_s", "cmd_v", "cmd_w", "ruota_sx_rad_s", "ruota_dx_rad_s",
        "step", "t_step_succ_sim_s", "v_vera_step_succ", "w_vera_step_succ")


def test_pose_csv(tmp_path):
    reg = re_.RegistriDalVivo(tmp_path / "registri_dal_vivo")
    reg.tag = "w2_e17"
    reg.posa("gemello", _msg(), {"source_timestamp": 1_700_000_000_123_456_789,
                                     "received_timestamp": 1_700_000_000_125_000_000}, 1000.25, 12.55)
    reg.posa("vittima", _msg(), {}, 1000.30, float("nan"))
    reg.close()
    r = _righe(tmp_path / "registri_dal_vivo" / "pose.csv")
    assert tuple(r[0]) == re_.POSE_COLONNE
    assert len(r) == 3 and all(len(x) == len(re_.POSE_COLONNE) for x in r)
    d = dict(zip(r[0], r[1]))
    assert d["ruolo"] == "gemello" and d["tag"] == "w2_e17"
    assert float(d["stamp_nuvola_s"]) == pytest.approx(12.5)
    assert d["t_pub_src_ns"] == "1700000000123456789"
    assert d["t_ric_dds_ns"] == "1700000000125000000"
    assert float(d["t_ric_wall_s"]) == pytest.approx(1000.25)
    assert float(d["yaw"]) == pytest.approx(0.5)
    d2 = dict(zip(r[0], r[2]))
    assert d2["t_pub_src_ns"] == "" and d2["t_ric_dds_ns"] == "" and d2["t_ric_sim_s"] == ""


def test_intestazione_una_volta_in_aggiunta(tmp_path):
    for _ in range(2):
        reg = re_.RegistriDalVivo(tmp_path)
        reg.posa("gemello", _msg(), None, 1.0, 2.0)
        reg.close()
    r = _righe(tmp_path / "pose.csv")
    assert [x[0] for x in r] == ["ruolo", "gemello", "gemello"]


def test_righe_leggibili_durante_la_run(tmp_path):
    """Line buffering: the row is on disk before closing."""
    reg = re_.RegistriDalVivo(tmp_path)
    reg.ciclo("gemello", 10.0, 5.0, 4.9, 9.95, (0.1, 0.0, 0.0), 0.1, 0.8, 0.0, 10.001, "crociera")
    assert len(_righe(tmp_path / "controllore.csv")) == 2
    reg.close()


def test_eta_e_controllore(tmp_path):
    assert re_.eta(10.5, 10.2) == pytest.approx(0.3)
    assert math.isnan(re_.eta(float("nan"), 1.0)) and math.isnan(re_.eta(1.0, None))
    reg = re_.RegistriDalVivo(tmp_path)
    reg.tag = "w3_victim"
    reg.ciclo("vittima", 100.0, 50.25, 50.0, 99.9, (0.3, 0.01, 0.02), 0.31, 0.8, -0.1,
              100.002, "rampa")
    # no estimate: stamp and age empty even with t_sim known (the old stamp is rejected)
    reg.ciclo("gemello", 101.0, 50.3, 50.0, 99.9, None, float("nan"), 0.0, 0.0,
              101.001, "stop")
    reg.close()
    r = _righe(tmp_path / "controllore.csv")
    d = dict(zip(r[0], r[1]))
    assert float(d["eta_posa_s"]) == pytest.approx(0.25)
    assert float(d["eta_posa_s"]) == pytest.approx(float(d["t_sim_s"]) - float(d["stamp_posa_usata_s"]))
    assert float(d["eta_posa_wall_s"]) == pytest.approx(0.1)
    assert d["motivo"] == "rampa" and d["ruolo"] == "vittima" and d["tag"] == "w3_victim"
    assert float(d["t_pub_cmd_wall_s"]) >= float(d["t_wall_s"])
    d2 = dict(zip(r[0], r[2]))
    assert d2["eta_posa_s"] == "" and d2["eta_posa_wall_s"] == "" and d2["est_x"] == ""
    assert d2["stamp_posa_usata_s"] == "" and d2["motivo"] == "stop" and float(d2["t_sim_s"]) == 50.3


def test_controllore_stamp_grezzo_e_offset(tmp_path):
    """Trailing columns for the corrected age: raw stamp and offset, empty when not given."""
    reg = re_.RegistriDalVivo(tmp_path)
    reg.tag = "w1_applied"
    # first pose after resume: offset from before the pause (a log artefact)
    reg.ciclo("vittima", 100.0, 39.433333, 36.0, 99.9, (0.0, 0.0, 0.0), 0.0, 0.8, 0.0,
              100.001, "crociera", stamp_grezzo=20.2, offset=15.8)
    reg.ciclo("vittima", 100.05, 39.45, 36.0, 99.9, (0.0, 0.0, 0.0), 0.0, 0.8, 0.0,
              100.051, "crociera")
    reg.ciclo("vittima", 100.1, 39.5, 36.0, 99.9, None, 0.0, 0.8, 0.0,
              100.101, "crociera", stamp_grezzo=20.2, offset=15.8)
    reg.close()
    r = _righe(tmp_path / "controllore.csv")
    assert tuple(r[0]) == re_.CONTROLLORE_COLONNE
    assert all(len(x) == len(re_.CONTROLLORE_COLONNE) for x in r)
    assert r[0][:15] == ["ruolo", "tag", "t_wall_s", "t_sim_s", "stamp_posa_usata_s", "eta_posa_s",
                         "eta_posa_wall_s", "est_x", "est_y", "est_yaw", "est_dist", "cmd_v", "cmd_w",
                         "t_pub_cmd_wall_s", "motivo"]
    d1, d2, d3 = (dict(zip(r[0], x)) for x in r[1:])
    assert float(d1["stamp_posa_grezzo_s"]) == pytest.approx(20.2)
    assert float(d1["offset_stamp_s"]) == pytest.approx(15.8)
    assert float(d1["eta_posa_s"]) == pytest.approx(3.433333)     # logged age, with the artefact
    assert float(d1["stamp_posa_grezzo_s"]) + float(d1["offset_stamp_s"]) == pytest.approx(
        float(d1["stamp_posa_usata_s"]))
    assert d2["stamp_posa_grezzo_s"] == "" and d2["offset_stamp_s"] == ""
    assert d3["stamp_posa_grezzo_s"] == "" and d3["offset_stamp_s"] == "" and d3["stamp_posa_usata_s"] == ""


def test_posa_del_rollout_precedente_non_usata(tmp_path):
    """Stop row at rollout start with the previous rollout's estimate: estimate and age empty."""
    ora = [200.0]
    reg = re_.RegistriDalVivo(tmp_path, orologio=lambda: ora[0])
    reg.inizio_rollout("w2_e01")
    assert reg.tag == "w2_e01"
    reg.ciclo("gemello", 201.0, 60.0, 59.0, 150.0, (1.0, 0.0, 0.0), 1.0, 0.0, 0.0, 201.001, "stop")
    reg.ciclo("gemello", 202.0, 61.0, 60.9, 201.5, (0.1, 0.0, 0.0), 0.1, 0.8, 0.0, 202.001, "crociera")
    reg.close()
    r = _righe(tmp_path / "controllore.csv")
    d1, d2 = dict(zip(r[0], r[1])), dict(zip(r[0], r[2]))
    assert d1["stamp_posa_usata_s"] == "" and d1["eta_posa_s"] == "" and d1["est_x"] == ""
    assert d1["est_dist"] == "" and d1["motivo"] == "stop" and d1["tag"] == "w2_e01"
    assert float(d2["eta_posa_s"]) == pytest.approx(0.1)


def test_errore_di_scrittura_disattiva_senza_eccezioni(tmp_path, capsys):
    reg = re_.RegistriDalVivo(tmp_path)
    reg._pose.close()
    reg._ctrl.close()
    reg.posa("gemello", _msg(), {}, 1.0, 2.0)
    reg.ciclo("gemello", 1.0, 2.0, 1.9, 0.9, (0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 1.0, "stop")
    cb = reg.callback_posa("vittima", lambda m: None, lambda: 1.0, lambda: 2.0)
    cb(_msg(), {})
    assert reg.attivo is False
    assert capsys.readouterr().err.count("registro disattivato") == 1


def test_callback_originale_eseguito_anche_con_errore(tmp_path):
    reg = re_.RegistriDalVivo(tmp_path)
    visti = []

    def rotto():
        raise RuntimeError("orologio")
    cb = reg.callback_posa("gemello", visti.append, rotto, lambda: 0.0)
    cb(_msg(), {})
    assert len(visti) == 1 and reg.attivo is False


def test_isaac_errore_di_scrittura(tmp_path, capsys):
    reg = re_.RegistroIsaacCmd(tmp_path)
    reg.comando(0.8, 0.0, t_ric_wall=1.0)
    reg._f.close()
    assert reg.dopo_lo_step(reg.prima_dello_step(), 1, 0.1, lambda: (1.0, 1.0),
                            lambda: (0.8, 0.0)) == 0
    assert reg.attivo is False
    reg.comando(0.8, 0.0)
    assert reg.prima_dello_step() == 0
    assert capsys.readouterr().err.count("registro disattivato") == 1


@pytest.mark.parametrize("fase,motivo", [("crociera", "crociera"), ("rampa", "rampa"),
                                         ("coast", "coast"), ("fermo", "stop"),
                                         ("stop", "stop"), ("", "altro"), (None, "altro")])
def test_motivo(fase, motivo):
    assert re_.motivo_da_fase(fase) == motivo
    assert motivo in re_.MOTIVI


def test_callback_a_due_argomenti(tmp_path):
    """rclpy (subscription.py) passes MessageInfo only to two-argument callbacks."""
    reg = re_.RegistriDalVivo(tmp_path)
    visti = []
    cb = reg.callback_posa("gemello", visti.append, lambda: 5.0, lambda: 6.0)
    with pytest.raises(TypeError):
        inspect.signature(cb).bind(object())
    inspect.signature(cb).bind(object(), object())
    m = _msg()
    cb(m, {"source_timestamp": 42, "received_timestamp": 43})
    reg.close()
    assert visti == [m]
    d = dict(zip(*_righe(tmp_path / "pose.csv")[:2]))
    assert d["t_pub_src_ns"] == "42" and float(d["t_ric_wall_s"]) == 5.0


def test_isaac_cmd_e_pausa(tmp_path):
    reg = re_.RegistroIsaacCmd(tmp_path)
    chiamate = []

    def ruote():
        chiamate.append("r")
        return 3.2, 3.4

    def vel():
        return 0.79, 0.01

    assert reg.dopo_lo_step(0, 100, 5.0, ruote, vel) == 0
    reg.comando(0.8, 0.0, t_ric_wall=1000.0)
    n = reg.prima_dello_step()
    assert n == 1
    # while paused the step does not advance: the command waits
    assert reg.dopo_lo_step(n, 100, 5.0, ruote, vel) == 0
    assert chiamate == []
    assert reg.dopo_lo_step(reg.prima_dello_step(), 101, 5.0167, ruote, vel) == 1
    reg.close()
    r = _righe(tmp_path / "isaac_cmd.csv")
    assert tuple(r[0]) == re_.ISAAC_COLONNE and len(r) == 2
    d = dict(zip(r[0], r[1]))
    assert float(d["t_ric_wall_s"]) == 1000.0 and float(d["t_ric_sim_s"]) == 5.0
    assert float(d["ruota_sx_rad_s"]) == pytest.approx(3.2) and d["step"] == "101"
    assert float(d["t_step_succ_sim_s"]) == pytest.approx(5.0167)
    assert float(d["v_vera_step_succ"]) == pytest.approx(0.79)


# --- orchestrator ---

ao = pytest.importorskip("src.optimization.attack_orchestrator")


class _Pub:
    def __init__(self):
        self.n = 0

    def publish(self, m):
        self.n += 1


def _nodo(registri=None):
    node = ao.AttackNode.__new__(ao.AttackNode)
    node.cmd_pub = _Pub()
    node._Twist = lambda: SimpleNamespace(linear=SimpleNamespace(x=0.0),
                                          angular=SimpleNamespace(z=0.0))
    node.est = (0.4, 0.0, 0.0)
    node.est_t = 99.0
    node.est_stamp = 49.5
    node.est_source = "twin"
    from collections import deque
    node.true_buf = deque([(50.0, 0.0, 0.0)])
    if registri is not None:
        node.registri = registri
    return node


def test_opzione_spenta_nessuna_chiamata(tmp_path, monkeypatch):
    def vietato(*a, **k):
        raise AssertionError("registro chiamato con l'opzione spenta")
    monkeypatch.setattr(re_.RegistriDalVivo, "ciclo", vietato)
    monkeypatch.setattr(re_.RegistriDalVivo, "posa", vietato)
    monkeypatch.setattr(re_.RegistriDalVivo, "__init__", vietato)
    node = _nodo()
    assert node.registri is None
    node._stop_registrato("stop", (0.0, 0.0))
    node._stop_registrato("coast")
    assert node.cmd_pub.n == 2
    assert list(tmp_path.iterdir()) == []


def test_opzione_spenta_default_e_campagna():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert 'ap.add_argument("--registri-dal-vivo", action="store_true"' in src
    camp = (ROOT / "scripts/run_campaign.sh").read_text()
    assert 'REGISTRI_DAL_VIVO="${REGISTRI_DAL_VIVO:-0}"' in camp
    assert 'REG_FLAG=""; [[ "$REGISTRI_DAL_VIVO" == "1" ]] && REG_FLAG="--registri-dal-vivo"' in camp
    assert 'else unset REGISTRI_DAL_VIVO_DIR; fi' in camp
    isaac = (ROOT / "isaac/run_isaac_standalone.py").read_text()
    assert 'if os.environ.get("REGISTRI_DAL_VIVO_DIR"):' in isaac
    assert "_n_reg = _reg.prima_dello_step() if _reg is not None else 0" in isaac
    # repo root appended to sys.path; start and write guarded; thread closed
    assert "sys.path.append(str(Path(__file__).resolve().parent.parent))" in isaac
    assert "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))" not in isaac
    assert "registri non avviati" in isaac and '_reg_spegni("scrittura", e)' in isaac
    assert "_reg_exec.shutdown(timeout_sec=1.0)" in isaac and "_reg_ctx.try_shutdown()" in isaac


def test_ciclo_registrato_dallo_stop(tmp_path):
    reg = re_.RegistriDalVivo(tmp_path, orologio=lambda: 50.0)
    reg.inizio_rollout("w1_e03")
    node = _nodo(reg)
    node._stop_registrato("fermo", (0.0, 0.0))
    node.est_source = "victim"
    node._registra_ciclo(100.0, 100.001, (0.5, 0.0, 0.0), 0.5, 0.8, 0.0, "crociera")
    reg.close()
    r = _righe(tmp_path / "controllore.csv")
    d = dict(zip(r[0], r[1]))
    assert node.cmd_pub.n == 1
    assert d["ruolo"] == "gemello" and d["tag"] == "w1_e03" and d["motivo"] == "stop"
    assert float(d["est_dist"]) == pytest.approx(0.4)
    assert float(d["eta_posa_s"]) == pytest.approx(0.5)      # 50.0 - 49.5
    assert float(d["cmd_v"]) == 0.0
    d2 = dict(zip(r[0], r[2]))
    assert d2["ruolo"] == "vittima" and d2["motivo"] == "crociera"
