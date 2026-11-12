#!/usr/bin/env python3
"""
Tests of the end-of-window stop (--window-stop legacy|ramp).

Three levels: the pure functions in src/optimization/window_stop.py; a kinematic
simulation of the window (estimate delayed by 0.2 s, stepped every 0.1 s, braking at
2 m/s^2, control at 20 Hz) comparing the two rules; the real drive loop of AttackNode.drive
with a fake node simulating robot and estimate in real time (a few seconds).
"""

import math
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization import window_stop as ws  # noqa: E402


# ---------------------------------------------------------------------------
# Speed law
# ---------------------------------------------------------------------------

def test_rampa_lontano_velocita_del_follower():
    assert ws.ramp_speed(0.8, 0.0, 1.0, k=1.5, vmin=0.05) == 0.8
    # the ramp starts at H - v/k = 1 - 0.8/1.5 = 0.467 m
    assert ws.ramp_speed(0.8, 0.46, 1.0, k=1.5, vmin=0.05) == 0.8


def test_rampa_vicino_proporzionale():
    assert ws.ramp_speed(0.8, 0.8, 1.0, k=1.5, vmin=0.05) == pytest.approx(0.3)
    assert ws.ramp_speed(0.8, 0.9, 1.0, k=2.0, vmin=0.05) == pytest.approx(0.2)


def test_rampa_mai_sotto_vmin_prima_di_H():
    for d in (0.97, 0.99, 0.9999):
        assert ws.ramp_speed(0.8, d, 1.0, k=1.5, vmin=0.05) == pytest.approx(0.05)


def test_rampa_zero_da_H():
    assert ws.ramp_speed(0.8, 1.0, 1.0) == 0.0
    assert ws.ramp_speed(0.8, 1.2, 1.0) == 0.0


def test_rampa_non_supera_il_follower():
    # in-place rotation (v = 0) or follower already slow near the waypoint
    assert ws.ramp_speed(0.0, 0.9, 1.0) == 0.0
    assert ws.ramp_speed(0.02, 0.5, 1.0, k=1.5, vmin=0.05) == 0.02


def test_comando_e_fase():
    v, vr, fase = ws.ramp_command(0.8, 0.2, 1.0, 1.5, 0.05, 0.05, None)
    assert (v, fase) == (0.8, "crociera") and vr == pytest.approx(1.2)
    v, vr, fase = ws.ramp_command(0.8, 0.9, 1.0, 1.5, 0.05, 0.05, vr)
    assert v == pytest.approx(0.15) and fase == "rampa"


def test_stima_ferma_tiene_la_velocita_di_rampa():
    # estimate 0.4 s old: keep the previous tick's ramp, even if the (old) estimate
    # would allow a higher speed
    v, vr, fase = ws.ramp_command(0.8, 0.5, 1.0, 1.5, 0.05, 0.4, 0.12)
    assert v == pytest.approx(0.12) and vr == pytest.approx(0.12) and fase == "rampa"
    # fresh estimate: recomputed
    v, _, _ = ws.ramp_command(0.8, 0.5, 1.0, 1.5, 0.05, 0.1, 0.12)
    assert v == pytest.approx(0.75)
    # old estimate but no previous ramp (first tick): computed
    v, _, _ = ws.ramp_command(0.8, 0.9, 1.0, 1.5, 0.05, 0.4, None)
    assert v == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# End of coast and log
# ---------------------------------------------------------------------------

def test_fine_coast_fermo_e_stima_nuova():
    assert ws.coast_end_ramp(0.3, 3, True) == "fermo_stima_nuova"
    assert ws.coast_end_ramp(0.3, 3, False) is None
    assert ws.coast_end_ramp(0.3, 2, True) is None


def test_fine_coast_tetto():
    assert ws.coast_end_ramp(2.0, 0, False) == "tetto"
    assert ws.coast_end_ramp(1.99, 5, False) is None


def test_conteggio_fermo():
    n = 0
    for v in (0.05, 0.01, 0.015, 0.03, 0.01, 0.0, 0.0):
        n = ws.still_count(n, v)
    assert n == 3
    assert ws.still_count(2, float("nan")) == 0


def test_stima_dopo_arresto():
    assert ws.est_after_stop(10.30, 10.25, 1.0, 0.9) is True
    assert ws.est_after_stop(10.20, 10.25, 1.0, 0.9) is False
    # no stamp: reception time
    assert ws.est_after_stop(float("nan"), None, 1.0, 0.9) is True
    assert ws.est_after_stop(float("nan"), None, 0.8, 0.9) is False
    assert ws.est_after_stop(10.3, 10.25, 1.0, None) is False


def test_scarti_e_registro():
    info = ws.new_stop_info("ramp", 1.5, 0.05)
    assert set(info) == set(ws.STOP_INFO_KEYS)
    info.update({"est_dist_end_m": 1.00123456, "true_dist_end_m": 1.0149,
                 "tau_cond_s": 2.71828, "est_jump_at_cond_cm": float("nan")})
    ws.finalize_stop_info(info, 1.0)
    assert info["scarto_H_est_cm"] == pytest.approx(0.12, abs=0.01)
    assert info["scarto_H_true_cm"] == pytest.approx(1.49, abs=0.01)
    assert info["est_dist_end_m"] == 1.0012 and info["tau_cond_s"] == 2.718
    assert info["est_jump_at_cond_cm"] is None
    assert info["stop_k"] == 1.5 and info["stop_vmin"] == 0.05
    leg = ws.new_stop_info("legacy", 1.5, 0.05)
    assert leg["stop_k"] is None and leg["stop_vmin"] is None
    assert ws.scarto_cm(None, 1.0) is None and ws.scarto_cm(1.38, 1.0) == pytest.approx(38.0)


# ---------------------------------------------------------------------------
# Kinematic simulation of the window
# ---------------------------------------------------------------------------

def simula_finestra(modo, k=ws.DEFAULT_K, vmin=ws.DEFAULT_VMIN, H=1.0, v_fol=0.8,
                    lat=0.2, upd=0.1, fase=0.0, a_max=2.0, ctrl=0.05, dt=1.0 / 600,
                    catchup=0.8, perse=(), rit_cmd=0.0, t_max=20.0):
    """One window from standstill: 1D robot with acceleration limited to a_max, estimate =
    true pose `lat` seconds earlier, updated in steps at phase + n*upd (except indices in
    `perse`), control every `ctrl` seconds with the drive() rule and the _coast() coast.
    rit_cmd: delay between command and actuator. Returns the stop-log quantities."""
    n_ctrl = int(round(ctrl / dt))
    xs = [0.0]
    x = v = 0.0
    cmd_q = deque()
    v_att = 0.0
    est, est_stamp, est_t = 0.0, -1.0, 0.0
    est_prev = None
    n_upd = 0
    next_upd = fase
    stato = "guida"
    v_prev = None
    r = {"modo": modo}
    t_c0 = arrest_t = None
    hits = still = 0
    i = 0

    def x_at(tq):
        if tq <= 0:
            return 0.0
        j = tq / dt
        j0 = int(j)
        if j0 + 1 >= len(xs):
            return xs[-1]
        return xs[j0] + (xs[j0 + 1] - xs[j0]) * (j - j0)

    while i * dt < t_max:
        t = i * dt
        while next_upd <= t + 1e-9:
            if n_upd not in perse:
                est_prev = est
                est = x_at(next_upd - lat)
                est_stamp = next_upd - lat
                est_t = next_upd
            n_upd += 1
            next_upd = fase + n_upd * upd
        if i % n_ctrl == 0:
            if stato == "guida":
                if est >= H:
                    stato = "coast"
                    t_c0 = t
                    cmd_q.append((t + rit_cmd, 0.0))
                    r.update(t_cond=t, x_cond=x, est_cond=est,
                             salto_cm=100 * (est - est_prev) if est_prev is not None else None)
                else:
                    vc = v_fol
                    if modo == "ramp":
                        vc, v_prev, _ = ws.ramp_command(v_fol, est, H, k, vmin, t - est_t, v_prev)
                    cmd_q.append((t + rit_cmd, vc))
            if stato == "coast":
                cap = ws.COAST_CAP_S if modo == "ramp" else catchup
                if t - t_c0 >= cap:
                    r["fine"] = "tetto" if modo == "ramp" else "scadenza"
                    break
                spd = abs(v)
                if arrest_t is None and spd < ws.STILL_V:
                    arrest_t = t
                    r.update(t_arresto=t, x_arresto=x, est_arresto=est)
                r.update(x_fine=x, est_fine=est, t_fine=t)
                if modo == "ramp":
                    still = ws.still_count(still, spd)
                    nuova = ws.est_after_stop(est_stamp, arrest_t, est_t, arrest_t)
                    fine = ws.coast_end_ramp(t - t_c0, still, nuova)
                    if fine is not None:
                        r["fine"] = fine
                        break
                else:
                    hits = hits + 1 if abs(est - x) < 0.03 else 0
                    if hits >= 3:
                        r["fine"] = "gap_stima_vera"
                        break
        while cmd_q and cmd_q[0][0] <= t + 1e-9:
            v_att = cmd_q.popleft()[1]
        dv = max(-a_max * dt, min(a_max * dt, v_att - v))
        v += dv
        x += v * dt
        xs.append(x)
        i += 1
    r["scarto_vero_cm"] = 100 * (r["x_fine"] - H)
    r["scarto_stima_cm"] = 100 * (r["est_fine"] - H)
    r["coast_s"] = r["t_fine"] - r["t_cond"]
    return r


FASI = [j * 0.005 for j in range(20)]      # estimate phase relative to control


def _serie(modo, **kw):
    return [simula_finestra(modo, fase=f, **kw) for f in FASI]


def test_simulazione_legacy_supera_H_di_35_40_cm():
    rs = _serie("legacy")
    sc = [r["scarto_vero_cm"] for r in rs]
    assert 34.0 <= float(np.mean(sc)) <= 41.0
    assert min(sc) > 28.0 and max(sc) < 46.0


def test_simulazione_rampa_entro_3_cm():
    rs = _serie("ramp", k=1.5, vmin=0.05)
    sc = [r["scarto_vero_cm"] for r in rs]
    assert min(sc) >= 0.0 and max(sc) <= 3.0
    assert all(r["fine"] == "fermo_stima_nuova" for r in rs)
    # at the trigger the estimate exceeds H by at most one step at v_min (0.5 cm)
    assert all(0.0 <= 100 * (r["est_cond"] - 1.0) <= 1.0 for r in rs)


def test_simulazione_rampa_k_fino_a_1_5_entro_3_cm():
    for k in (1.0, 1.5):
        for vmin in (0.03, 0.05, 0.08):
            sc = [r["scarto_vero_cm"] for r in _serie("ramp", k=k, vmin=vmin)]
            assert max(sc) <= 3.0, (k, vmin, max(sc))


def test_simulazione_sensibilita_a_k_e_al_ritardo():
    # With k * total delay above ~1/e the true robot passes H before the estimate does:
    # the overshoot grows with k and with the actuator delay (0.05-0.10 s in the traces).
    for rit in (0.0, 0.1):
        m = [float(np.mean([r["scarto_vero_cm"] for r in _serie("ramp", k=k, rit_cmd=rit)]))
             for k in (1.0, 1.5, 2.0)]
        assert m[0] < m[1] < m[2], (rit, m)
    sc = [r["scarto_vero_cm"] for r in _serie("ramp", k=1.0, rit_cmd=0.1)]
    assert max(sc) <= 3.0


def test_simulazione_rampa_con_due_scan_persi():
    # missing the update that would trigger the condition and the next one: estimate
    # frozen for 0.3 s, the ramp stays at v_min and does not accelerate; the condition
    # comes one update later, ~1 cm further
    for f in (0.0, 0.025, 0.05, 0.075):
        base = simula_finestra("ramp", fase=f)
        n = int(round((base["t_cond"] - f) / 0.1 - 0.5))
        r = simula_finestra("ramp", fase=f, perse=(n, n + 1))
        assert r["t_cond"] > base["t_cond"]
        assert 0.0 <= r["scarto_vero_cm"] - base["scarto_vero_cm"] <= 2.0
        assert r["scarto_vero_cm"] <= 4.0


def test_simulazione_tempo_in_piu():
    # Rollout duration up to the end of the coast: the ramp lengthens the approach but
    # shortens braking and coast; lower k costs more.
    leg = np.mean([r["t_fine"] for r in _serie("legacy")])
    r15 = np.mean([r["t_fine"] for r in _serie("ramp", k=1.5)])
    r10 = np.mean([r["t_fine"] for r in _serie("ramp", k=1.0)])
    assert -0.5 < r15 - leg < 1.0
    assert r10 > r15 and r10 - leg < 2.5


# ---------------------------------------------------------------------------
# Real drive loop (AttackNode.drive) with a fake real-time node
# ---------------------------------------------------------------------------

class _Mondo:
    """1D robot and delayed estimate, advanced on wall-clock time."""

    def __init__(self, lat=0.2, upd=0.1, a_max=2.0, dt=1.0 / 600):
        self.t0 = time.time()
        self.lat, self.upd, self.a, self.dt = lat, upd, a_max, dt
        self.t = 0.0
        self.x = self.v = self.v_cmd = 0.0
        self.xs = [0.0]
        self.n_upd = 1

    def x_at(self, tq):
        if tq <= 0:
            return 0.0
        j = min(len(self.xs) - 1, int(tq / self.dt))
        return self.xs[j]

    def avanza(self):
        """Advance to the current time; return the new estimates (x, stamp)."""
        now = time.time() - self.t0
        while self.t + self.dt <= now:
            dv = max(-self.a * self.dt, min(self.a * self.dt, self.v_cmd - self.v))
            self.v += dv
            self.x += self.v * self.dt
            self.t += self.dt
            self.xs.append(self.x)
        nuove = []
        while self.n_upd * self.upd <= now:
            tu = self.n_upd * self.upd
            nuove.append((self.x_at(tu - self.lat), tu - self.lat))
            self.n_upd += 1
        return nuove


def _nodo_finto(modo):
    from src.optimization.attack_orchestrator import AttackNode
    from src.optimization.trajectory_predictor import FollowerModel

    class NodoFinto(AttackNode):
        def __init__(self):    # no ROS
            self.mondo = _Mondo()
            self.follower = FollowerModel()
            self.window_stop = modo
            self.stop_ramp_k = ws.DEFAULT_K
            self.stop_ramp_vmin = ws.DEFAULT_VMIN
            self.last_stop_info = None
            self.true_source = "ros"
            self.true_buf = deque(maxlen=400)
            self.est = None
            self.est_t = 0.0
            self.est_stamp = float("nan")
            self.true = None
            self.true_t = 0.0
            self.true_v = (0.0, 0.0)
            self._last_true = None
            self._warned_no_stream = False
            self.trace_rows = []
            self.pose_quality = float("nan")
            self.chamfer = []
            self.drift_cm = self.drift_xy_cm = 0.0
            self.node_frame_idx = 0
            self.true_z = self.est_z = 0.0
            self._Twist = lambda: SimpleNamespace(linear=SimpleNamespace(x=0.0),
                                                  angular=SimpleNamespace(z=0.0))
            mondo = self.mondo
            self.cmd_pub = SimpleNamespace(
                publish=lambda m: setattr(mondo, "v_cmd", float(m.linear.x)))
            self._exec = SimpleNamespace(spin_once=lambda timeout_sec=0.0: self.drain())
            self.drain()

        def check_stall(self):
            pass

        def drain(self, budget=0.0):
            m = self.mondo
            for ex, st in m.avanza():
                self.est = (ex, 0.0, 0.0)
                self.est_t = time.time()
                self.est_stamp = st
            self.true = (m.x, 0.0, 0.0)
            self.true_t = time.time()
            self.true_v = (m.v, 0.0)
            self.true_buf.append((m.t, m.x, 0.0))

    return NodoFinto()


def _guida(modo):
    node = _nodo_finto(modo)
    isaac = SimpleNamespace(playing=True, stream_available=lambda: True,
                            pose_stream=lambda: node.true)
    mola = SimpleNamespace(crashed=lambda: False)
    status, trav = node.drive([(4.0, 0.0)], 1.0, isaac, mola, timeout=15.0,
                              catchup_sec=0.8, stop_on="estimate")
    return node, status, trav


def test_drive_vero_legacy_e_rampa():
    from src.optimization.attack_orchestrator import drift_fields  # noqa: F401
    node, st, trav = _guida("legacy")
    info = node.last_stop_info
    assert st == "ok" and node.last_stop_reason == "stima_orizzonte"
    assert info["stop_mode"] == "legacy" and info["stop_k"] is None
    assert info["coast_end_reason"] in ("gap_stima_vera", "scadenza")
    assert 25.0 <= info["scarto_H_true_cm"] <= 50.0
    assert info["true_dist_end_m"] == pytest.approx(trav, abs=1e-4)
    assert {r["fase"] for r in node.trace_rows} <= {"crociera", "coast", "fermo"}
    assert max(r["cmd_v"] for r in node.trace_rows) == pytest.approx(0.8)
    # new columns appended, old ones in the same order
    cols = list(node.trace_rows[0].keys())
    assert cols[-2:] == ["fase", "est_dist"] and cols[0] == "t" and cols[-3] == "wall"
    leg_s = info["drive_s"]

    node, st, trav = _guida("ramp")
    info = node.last_stop_info
    assert st == "ok" and node.last_stop_reason == "stima_orizzonte"
    assert info["stop_mode"] == "ramp" and info["stop_k"] == 1.5 and info["stop_vmin"] == 0.05
    assert info["coast_end_reason"] == "fermo_stima_nuova"
    assert -0.5 <= info["scarto_H_true_cm"] <= 4.0
    assert 0.0 <= info["scarto_H_est_cm"] <= 2.0
    fasi = [r["fase"] for r in node.trace_rows]
    assert "rampa" in fasi and "fermo" in fasi
    rampa = [r["cmd_v"] for r in node.trace_rows if r["fase"] == "rampa"]
    assert min(rampa) == pytest.approx(0.05, abs=1e-3) and max(rampa) < 0.8
    assert info["drive_s"] > leg_s
    # fields reach the history through drift_fields
    node.last_drift_start_cm = node.last_drift_end_cm = float("nan")
    node.last_motion = {}
    d = drift_fields(node)
    assert all(kk in d for kk in ws.STOP_INFO_KEYS)
