"""
Base attitude helpers for the rollout server (run_isaac_standalone.py):
quaternions, Euler angles and the orientation chosen by set_state. numpy only, no
Isaac imports, so it can be tested outside the simulator
(tests/test_assetto_isaac.py). The server loads this file from its own directory.

Conventions: quaternions are [w, x, y, z] as in Isaac get_world_pose /
set_world_pose, scene world frame. Euler ZYX in rad: R = Rz(yaw) Ry(pitch)
Rx(roll), i.e. yaw about world z, then pitch about y and roll about x of the yawed
frame. Roll and pitch are thus the tilt expressed in the yaw frame; yaw matches the
server's _yaw.
"""

import math

import numpy as np

# Beyond this (rad, ~28.6 deg) roll and pitch cannot belong to a robot on the floor:
# almost certainly degrees passed as radians.
LIMITE_TILT = 0.5

CONVENZIONE = "ZYX: R = Rz(yaw) Ry(pitch) Rx(roll), rad, quaternione [w,x,y,z], terna del mondo"


def quat_da_yaw(yaw):
    """Yaw-only quaternion, same expression as set_state/set_pose."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_da_euler(roll, pitch, yaw):
    """Quaternion [w,x,y,z] from ZYX Euler. With zero roll and pitch it returns
    quat_da_yaw(yaw) bit for bit (no signed zeros)."""
    if roll == 0.0 and pitch == 0.0:
        return quat_da_yaw(yaw)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def euler_da_quat(q):
    """ZYX (roll, pitch, yaw) from a unit quaternion [w,x,y,z]. Yaw uses the
    server's _yaw formula; pitch is clamped to [-pi/2, pi/2]."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def prodotto(a, b):
    """Hamilton product a*b of quaternions [w,x,y,z] (b applied first, then a)."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def normalizza(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_keep_tilt(q_attuale, yaw_nuovo):
    """New yaw with the roll and pitch of q_attuale.

    q_attuale = Rz(yaw_att) * T, with T the tilt in the yaw frame; the result is
    Rz(yaw_nuovo) * T, i.e. q_attuale rotated about world z by
    (yaw_nuovo - yaw_att). ZYX roll and pitch are kept, yaw becomes yaw_nuovo."""
    q = normalizza(q_attuale)
    yaw_att = euler_da_quat(q)[2]
    tilt = prodotto(quat_da_yaw(-yaw_att), q)
    return normalizza(prodotto(quat_da_yaw(yaw_nuovo), tilt))


def _finito(cmd, chiave):
    try:
        v = float(cmd[chiave])
    except (TypeError, ValueError):
        raise ValueError(f"set_state: {chiave} non numerico: {cmd[chiave]!r}")
    if not math.isfinite(v):
        raise ValueError(f"set_state: {chiave} non finito: {v}")
    return v


def quat_set_state(cmd, q_attuale):
    """Base orientation for set_state with its optional parameters.

    roll and pitch (rad, both required) give an explicit 6D pose, ZYX with the
    command yaw; keep_tilt true keeps the roll and pitch of q_attuale (from the
    restore) with the command yaw; neither (or keep_tilt false) gives yaw only.
    Raises ValueError before any write for: roll/pitch with keep_tilt, only one of
    roll and pitch, non-numeric or non-finite values, tilt over LIMITE_TILT,
    non-boolean keep_tilt."""
    yaw = float(cmd.get("yaw", 0.0))
    keep = cmd.get("keep_tilt", False)
    if not isinstance(keep, bool):
        raise ValueError(f"set_state: keep_tilt deve essere true o false, non {keep!r}")
    ha_rp = "roll" in cmd or "pitch" in cmd
    if ha_rp and keep:
        raise ValueError("set_state: roll/pitch e keep_tilt sono alternativi")
    if ha_rp:
        if not ("roll" in cmd and "pitch" in cmd):
            raise ValueError("set_state: servono sia roll sia pitch")
        roll, pitch = _finito(cmd, "roll"), _finito(cmd, "pitch")
        if abs(roll) > LIMITE_TILT or abs(pitch) > LIMITE_TILT:
            raise ValueError(f"set_state: |roll|, |pitch| oltre {LIMITE_TILT} rad "
                             f"(roll={roll}, pitch={pitch}): gradi al posto di radianti?")
        return quat_da_euler(roll, pitch, yaw)
    if keep:
        return quat_keep_tilt(q_attuale, yaw)
    return quat_da_yaw(yaw)


def risposta_assetto(pos, quat, lin, ang, nomi_giunti=None, jpos=None, jvel=None):
    """Reply of the assetto command, JSON types only. Velocities in the world
    frame, as returned by the Isaac articulation."""
    roll, pitch, yaw = euler_da_quat(quat)
    r = {
        "ok": True,
        "pose": [float(pos[0]), float(pos[1]), yaw],
        "pos": [float(v) for v in pos],
        "quat_wxyz": [float(v) for v in quat],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "convenzione": CONVENZIONE,
        "lin": [float(v) for v in lin],
        "ang": [float(v) for v in ang],
    }
    if jpos is not None:
        r["giunti"] = {
            "nomi": None if nomi_giunti is None else [str(n) for n in nomi_giunti],
            "pos": [float(v) for v in jpos],
            "vel": None if jvel is None else [float(v) for v in jvel],
        }
    return r
