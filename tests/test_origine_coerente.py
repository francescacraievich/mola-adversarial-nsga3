"""--origine-coerente-vittima: estimate origin and victim control frame from the same est_raw
reading. Without the option the frame is the default one.
"""
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.optimization import attack_orchestrator as ao  # noqa: E402

RUN_ORIGIN = (0.0166, -0.0003, 0.0)
PRIMA = (2.6883, -0.3817, math.radians(4.43))      # est_raw at the origin row (w3, stamp 33.2)
DOPO = (2.6828, -0.6804, math.radians(4.65))       # est_raw after the standstill jump (stamp 33.4)


def nodo():
    n = types.SimpleNamespace(est_raw=DOPO, est_origin=PRIMA, est=(0.0, 0.0, 0.0), last_frame_coerenza="vecchio")
    n.compose = ao.AttackNode.compose
    return n


@pytest.fixture
def coerente(monkeypatch):
    monkeypatch.setitem(ao.ORIGINE_VITTIMA, "coerente", True)


def msg_posa(x, y, yaw):
    h = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=33, nanosec=400000000))
    o = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    p = types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=o)
    return types.SimpleNamespace(header=h, pose=types.SimpleNamespace(pose=p))


def creduta(n, frame, raw):
    """Believed pose = frame or local estimate, with the local estimate computed by the real
    AttackNode._store_est (lines 819-835) from the raw pose.
    """
    n._n_cb = 0
    ao.AttackNode._store_est(n, msg_posa(*raw))
    return ao.AttackNode.compose(frame, n.est)


def test_spenta_uguale_a_prima():
    n = nodo()
    f = ao.frame_di_controllo_vittima(n, RUN_ORIGIN, (9, 9, 9))
    assert f == ao.AttackNode.compose(RUN_ORIGIN, DOPO)
    assert n.est_origin == PRIMA and n.last_frame_coerenza == "vecchio"
    assert ao.frame_di_controllo_vittima(nodo(), None, (9, 9, 9)) == (9, 9, 9)


def test_spenta_riproduce_l_incoerenza_del_03_11():
    n = nodo()
    f = ao.frame_di_controllo_vittima(n, RUN_ORIGIN, (0, 0, 0))
    p = creduta(n, f, DOPO)
    # the victim thinks it moved from the frame: about the distance between the two readings
    salto = math.hypot(DOPO[0] - PRIMA[0], DOPO[1] - PRIMA[1])
    assert math.hypot(p[0] - f[0], p[1] - f[1]) == pytest.approx(salto, rel=1e-9)
    assert salto > 0.29


def test_accesa_origine_e_frame_dalla_stessa_lettura(coerente):
    n = nodo()
    f = ao.frame_di_controllo_vittima(n, RUN_ORIGIN, (0, 0, 0))
    assert n.est_origin == DOPO and n.est == (0.0, 0.0, 0.0)
    assert f == ao.AttackNode.compose(RUN_ORIGIN, DOPO)
    p = creduta(n, f, DOPO)
    assert math.hypot(p[0] - f[0], p[1] - f[1]) < 1e-12 and abs(p[2] - f[2]) < 1e-12
    d = n.last_frame_coerenza
    assert d["attiva"] is True
    assert d["spostamento_origine_cm"] == pytest.approx(100 * math.hypot(DOPO[0] - PRIMA[0], DOPO[1] - PRIMA[1]), abs=1e-3)


def test_accesa_senza_run_origin_come_prima(coerente):
    n = nodo()
    assert ao.frame_di_controllo_vittima(n, None, (9, 9, 9)) == (9, 9, 9)
    assert n.est_origin == PRIMA and n.last_frame_coerenza is None


def _nodo_record(est_source):
    return types.SimpleNamespace(last_drift_start_cm=0.0, last_drift_end_cm=0.0, last_motion={}, last_align_err_cm=0.0,
                                 last_align_err_deg=0.0, last_stop_info={}, est_source=est_source,
                                 last_frame_coerenza={"attiva": True, "spostamento_origine_cm": 1.0,
                                                      "spostamento_origine_deg": 0.0})


def test_record_della_vittima_con_est_source_twin_come_nel_flusso_vero(coerente):
    # after rollout() finish() resets est_source = "twin": the field must still arrive
    n = _nodo_record("twin")
    assert ao.motion_vittima(n)["frame_coerenza"]["spostamento_origine_cm"] == 1.0
    # twin and nominal records use drift_fields: never the field
    assert "frame_coerenza" not in ao.drift_fields(n)


def test_record_assente_da_spenta():
    n = _nodo_record("twin")
    assert "frame_coerenza" not in ao.motion_vittima(n)
    assert ao.motion_vittima(n) == ao.drift_fields(n)


def test_accesa_senza_stima_come_prima(coerente):
    n = nodo()
    n.est_raw = None
    f = ao.frame_di_controllo_vittima(n, None, (9, 9, 9))
    assert f == (9, 9, 9) and n.est_origin == PRIMA and n.last_frame_coerenza is None


def test_i_record_della_vittima_usano_motion_vittima():
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert '"victim_motion": motion_vittima(node)' in src
    assert '"applied_motion": motion_vittima(node) if real_end is not None else None' in src
    # calls (definition excluded): only the two victim records
    assert src.count("motion_vittima(node)") - src.count("def motion_vittima(node)") == 2


def test_opzioni_nei_lanciatori():
    sh = (ROOT / "scripts/coda_serieB.sh").read_text()
    assert 'ORIGINE_COERENTE="${ORIGINE_COERENTE:-0}"' in sh and 'ATTESE+=("--origine-coerente-vittima")' in sh
    bl = (ROOT / "scripts/prova_banco_serieB_baseline_definitiva.sh").read_text()
    assert 'ORIGINE_COERENTE="${ORIGINE_COERENTE:-0}"' in bl and '$ORIG_X")' in bl
    src = (ROOT / "src/optimization/attack_orchestrator.py").read_text()
    assert 'ap.add_argument("--origine-coerente-vittima", action="store_true"' in src
