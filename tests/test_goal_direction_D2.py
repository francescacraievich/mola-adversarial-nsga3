"""--goal-mode direction: the goal only gives the direction, every window ends with the ramp
on the estimate, the run has a fixed number of driven windows.

Pure tests on orchestrator functions and on the real drive loop (AttackNode.drive) with the
fake node of test_arresto_finestra (1D robot, estimate delayed by 0.2 s, a few seconds of
real time per drive). No process and no ROS node.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

ao = pytest.importorskip("src.optimization.attack_orchestrator")
taf = pytest.importorskip("test_arresto_finestra")

from src.optimization.trajectory_predictor import FollowerModel  # noqa: E402


# --- options ------------------------------------------------------------------

def _args(**kw):
    a = dict(goal_mode="direction", max_windows=6, stop_on="estimate", window_stop="ramp",
             goals="5,0", direction_lookahead=ao.DIRECTION_LOOKAHEAD)
    a.update(kw)
    return SimpleNamespace(**a)


def test_default_invariato():
    # waypoints (default) checks nothing, also without --max-windows
    assert ao.check_goal_mode_args(_args(goal_mode="waypoints", max_windows=None,
                                         window_stop="legacy", goals="5,0; 5,2.5")) is None
    # a node without the attribute (older nodes) drives in waypoints mode
    node = taf._nodo_finto("ramp")
    assert not hasattr(node, "goal_mode")
    assert ao.DIRECTION_LOOKAHEAD == 5.0 and ao.ARRIVAL_TOL == 0.30


def test_configurazione_serie_b_accettata():
    assert ao.check_goal_mode_args(_args()) is None
    assert ao.check_goal_mode_args(_args(goals=None)) is None        # only --goal
    assert ao.check_goal_mode_args(_args(direction_lookahead=1.0)) is None


@pytest.mark.parametrize("kw,parola", [
    (dict(max_windows=None), "--max-windows"),
    (dict(max_windows=0), "--max-windows"),
    (dict(stop_on="true"), "--stop-on estimate"),
    (dict(window_stop="legacy"), "--window-stop ramp"),
    (dict(goals="5,0; 5,2.5"), "un solo bersaglio"),
    (dict(direction_lookahead=0.9), "--direction-lookahead"),
])
def test_opzioni_rifiutate(kw, parola):
    err = ao.check_goal_mode_args(_args(**kw))
    assert err is not None and parola in err


# --- carrot and line ------------------------------------------------------------

def test_retta_solo_direzione():
    for g in ((0.5, 0.0), (5.0, 0.0), (50.0, 0.0)):
        p0, u = ao.direction_line([(0.0, 0.0), g], None)
        assert np.allclose(p0, (0.0, 0.0)) and np.allclose(u, (1.0, 0.0))
    # a single point: from the initial estimate to the point
    p0, u = ao.direction_line([(3.0, 4.0)], (0.0, 0.0))
    assert np.allclose(p0, (0.0, 0.0)) and np.allclose(u, (0.6, 0.8))
    with pytest.raises(ValueError):
        ao.direction_line([(1.0, 1.0), (1.0, 1.0)], None)


def test_carota_sulla_retta_a_L_oltre_la_proiezione():
    p0, u = np.array([1.0, -1.0]), np.array([math.cos(0.3), math.sin(0.3)])
    n = np.array([-u[1], u[0]])
    for s in (-2.0, 0.0, 0.7, 4.9, 12.0):          # also before the start and past the goal
        for e in (-0.4, 0.0, 0.25):                # lateral offset of the estimate
            xy = p0 + s * u + e * n
            c = np.array(ao.direction_carrot(xy, p0, u, 5.0))
            assert np.allclose(c, p0 + (s + 5.0) * u)
            assert math.hypot(*(c - xy)) == pytest.approx(math.hypot(5.0, e))


def _comandi(goal, L=5.0, follower=None):
    """Follower commands toward the carrot along a sequence of estimates with lateral offset
    and yaw (line from the start (0,0) toward goal)."""
    follower = follower or FollowerModel()
    p0, u = ao.direction_line([(0.0, 0.0), goal], None)
    out = []
    for i in range(200):
        s = 0.05 * i                                # 0 to ~10 m, past the near goals
        e = 0.3 * math.sin(0.11 * i)
        yaw = math.atan2(u[1], u[0]) + 0.2 * math.cos(0.07 * i)
        x, y = p0 + s * u + e * np.array([-u[1], u[0]])
        c = ao.direction_carrot((x, y), p0, u, L)
        out.append(follower.command(x, y, yaw, c[0], c[1]))
    return out


def test_bersaglio_vicino_o_lontano_stesso_moto():
    ref = _comandi((5.0, 0.0))
    for g in ((0.5, 0.0), (1.2, 0.0), (50.0, 0.0), (500.0, 0.0)):
        assert _comandi(g) == ref                  # identical, not just close


def test_follower_senza_rallentamento():
    f = FollowerModel()
    for L in (1.0, 5.0):
        for v, w, dist in _comandi((0.5, 0.0), L=L, follower=f):
            assert dist >= L - 1e-9
            assert v == 0.0 or v == pytest.approx(f.max_speed)   # 0 only when rotating in place
            assert v > 0.0 or abs(w) > 0.0


def test_bersaglio_di_finestra_davanti_al_robot():
    u = np.array([1.0, 0.0])
    # robot past the given goal (5,0): the window goal stays L ahead
    g = ao.direction_window_goal((0.0, 0.0), u, (6.0, 0.4), 5.0)
    assert g == pytest.approx((11.0, 0.0))


# --- number of windows ----------------------------------------------------------

def _corri(max_windows, extra, trattenute):
    """Window loop reduced to what decides the end: one record per index, victim_held for
    the indices in `trattenute`."""
    history = []
    k = -1
    while True:
        k += 1
        fine = ao.direction_run_end(history, k, max_windows, extra)
        if fine is not None:
            return fine, history, k
        rec = {"window": k, "status": "ok"}
        if k in trattenute:
            rec.update(ao.victim_hold_fields("nessun_ammissibile", SimpleNamespace(started=True)))
        history.append(rec)


@pytest.mark.parametrize("n", [1, 4, 6, 9])
def test_numero_di_finestre_esatto(n):
    fine, hist, k = _corri(n, 3, set())
    assert fine == "finestre_completate" and len(hist) == n and k == n
    assert ao.direction_window_count(hist, k) == (n, 0)


def test_trattenute_non_contano():
    fine, hist, k = _corri(6, 3, {1, 3})
    assert fine == "finestre_completate"
    assert ao.direction_window_count(hist, k) == (6, 2) and len(hist) == 8
    # last one held: one more index needed
    fine, hist, k = _corri(6, 3, {5})
    assert fine == "finestre_completate" and ao.direction_window_count(hist, k) == (6, 1)


def test_troppe_trattenute_run_incompleta():
    fine, hist, k = _corri(6, 3, {0, 1, 2, 3})
    assert fine == "limite_finestre_trattenute"
    assert k == 9 and ao.direction_window_count(hist, k) == (5, 4)
    # with 3 held (= --extra-windows) the run completes
    fine, hist, k = _corri(6, 3, {0, 1, 2})
    assert fine == "finestre_completate" and ao.direction_window_count(hist, k) == (6, 3)


def test_fine_senza_posizione():
    # the end depends only on the window records, not on where the robot is
    import inspect
    assert list(inspect.signature(ao.direction_run_end).parameters) == [
        "history", "k", "max_windows", "extra_windows"]


# --- real drive loop ------------------------------------------------------------

def _guida(goal, goal_mode, k_ramp=1.0, L=5.0):
    node = taf._nodo_finto("ramp")
    node.stop_ramp_k = k_ramp
    if goal_mode is not None:
        node.goal_mode = goal_mode
        node.direction_lookahead = L
    isaac = SimpleNamespace(playing=True, stream_available=lambda: True,
                            pose_stream=lambda: node.true)
    mola = SimpleNamespace(crashed=lambda: False)
    # as _victim_rollout: [start, goal] in the local frame
    st, trav = node.drive([(0.0, 0.0), goal], 1.0, isaac, mola, timeout=15.0,
                          catchup_sec=0.8, stop_on="estimate")
    return node, st, trav


def test_drive_waypoints_bersaglio_vicino_arrived():
    # bench check: with the default a goal at 0.5 m closes as arrived
    node, st, trav = _guida((0.5, 0.0), None)
    assert st == "arrived" and node.last_stop_reason == "arrivato_waypoint"
    assert trav < 0.5


@pytest.mark.parametrize("goal", [(0.5, 0.0), (50.0, 0.0)])
def test_drive_direction_chiude_con_la_rampa(goal):
    node, st, trav = _guida(goal, "direction")
    info = node.last_stop_info
    assert st == "ok" and node.last_stop_reason == "stima_orizzonte"
    assert info["stop_mode"] == "ramp" and info["stop_k"] == 1.0
    assert info["coast_end_reason"] == "fermo_stima_nuova"
    assert -0.5 <= info["scarto_H_true_cm"] <= 4.0       # past the goal at 0.5 m
    rows = node.trace_rows
    guida = [r for r in rows if r["fase"] in ("crociera", "rampa")]
    assert guida and "rampa" in {r["fase"] for r in guida}
    # carrot always L ahead on the line, whatever the goal
    assert all(r["wp_x"] - r["est_x"] == pytest.approx(5.0, abs=2e-3) and r["wp_y"] == 0.0
               for r in guida)
    # no follower slowdown: outside the ramp always 0.8 m/s
    assert all(r["cmd_v"] == pytest.approx(0.8) for r in guida if r["fase"] == "crociera")
    assert max(r["cmd_v"] for r in guida) == pytest.approx(0.8)
