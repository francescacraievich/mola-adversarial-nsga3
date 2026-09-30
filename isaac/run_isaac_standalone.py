"""
Avvio di Isaac Sim da terminale con scena, bridge ROS 2, Play e server dei
rollout nello stesso processo: sostituisce l'apertura manuale della scena e
lo Script Editor, cosi' l'intera catena si lancia da riga di comando.

    source /opt/ros/jazzy/setup.bash
    python3 isaac/run_isaac_standalone.py [--headless] [--scene isaac/carter_warehouse.usd]

Con --headless niente finestra (piu' veloce, stesso rendering RTX del LiDAR);
senza, si apre la finestra di Isaac Sim come al solito. Il server risponde ai
comandi dell'orchestratore su /tmp/isaac_cmd.json come isaac_rollout_server.py
(save, restore, set_pose, step, pose, play, pause). Ctrl-C chiude tutto.

Da verificare al primo avvio: che il bridge ROS 2 trovi le librerie di Jazzy
(sourcing di /opt/ros/jazzy prima del lancio) e che /front_3d_lidar/lidar_points
arrivi a 10 Hz (`ros2 topic hz`).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default=str(Path(__file__).resolve().parent / "carter_warehouse.usd"))
ap.add_argument("--headless", action="store_true")
ap.add_argument("--no-play", action="store_true", help="carica la scena senza premere Play")
ap.add_argument("--no-rate-limit", action="store_true",
                help="ciclo principale senza limite di frequenza (RTF > 1)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

# L'app con finestra (isaacsim.exp.full.kit) limita il ciclo principale a
# 60 Hz; l'esperienza di SimulationApp no, e la simulazione corre a RTF ~1.2.
# Stesso limite dell'app, cosi' il tempo reale per scan resta quello delle
# run fatte con la finestra.
if not args.no_rate_limit:
    import carb  # noqa: E402
    _s = carb.settings.get_settings()
    _s.set("/app/runLoops/main/rateLimitEnabled", True)
    _s.set("/app/runLoops/main/rateLimitFrequency", 60)
    _s.set("/app/runLoops/main/rateLimitUseBusyLoop", False)

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
app.update()

print(f"[standalone] apro {args.scene}")
open_stage(args.scene)
for _ in range(10):
    app.update()

from isaacsim.core.prims import SingleArticulation  # noqa: E402

ROBOT = "/World/Nova_Carter_ROS"
CMD = "/tmp/isaac_cmd.json"
REPLY = "/tmp/isaac_reply.json"
POSE = "/tmp/isaac_pose.json"

_timeline = omni.timeline.get_timeline_interface()
_art = None
_saved = None


def _get_art():
    global _art
    if _art is None:
        _art = SingleArticulation(prim_path=ROBOT, name="attack_ctrl")
        _art.initialize()
    return _art


def _snapshot(art):
    pos, orient = art.get_world_pose()
    return {
        "pos": np.array(pos, dtype=np.float64),
        "orient": np.array(orient, dtype=np.float64),
        "lin": np.array(art.get_linear_velocity(), dtype=np.float64),
        "ang": np.array(art.get_angular_velocity(), dtype=np.float64),
        "jpos": np.array(art.get_joint_positions(), dtype=np.float64),
        "jvel": np.array(art.get_joint_velocities(), dtype=np.float64),
    }


def _restore(art, s):
    art.set_world_pose(position=s["pos"], orientation=s["orient"])
    art.set_joint_positions(s["jpos"])
    art.set_joint_velocities(s["jvel"])
    art.set_linear_velocity(s["lin"])
    art.set_angular_velocity(s["ang"])


def _yaw(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _pose_of(s):
    p = s["pos"]
    return [float(p[0]), float(p[1]), _yaw(s["orient"])]


def _reply(payload):
    tmp = REPLY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, REPLY)


def _write_pose():
    try:
        s = _snapshot(_get_art())
        tmp = POSE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pose": _pose_of(s), "wall": time.time(),
                       "playing": bool(_timeline.is_playing())}, f)
        os.replace(tmp, POSE)
    except Exception:
        pass


def _handle(cmd):
    """Esegue un comando; i comandi che richiedono frame ne restituiscono il numero."""
    global _saved
    art = _get_art()
    op = cmd.get("cmd")
    if op == "save":
        _saved = _snapshot(art)
        return {"ok": True, "pose": _pose_of(_saved)}, 0
    if op == "restore":
        if _saved is None:
            return {"ok": False, "error": "nessuno stato salvato"}, 0
        _restore(art, _saved)
        return None, 1                     # un frame, poi la posa
    if op == "set_pose":
        yaw = float(cmd.get("yaw", 0.0))
        pos, _ = art.get_world_pose()
        art.set_world_pose(position=np.array([float(cmd["x"]), float(cmd["y"]), float(pos[2])]),
                           orientation=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
        art.set_joint_velocities(np.zeros_like(art.get_joint_velocities()))
        art.set_linear_velocity(np.zeros(3))
        art.set_angular_velocity(np.zeros(3))
        return None, 1
    if op == "step":
        return None, int(cmd.get("n", 60))
    if op == "pose":
        return {"ok": True, "pose": _pose_of(_snapshot(art)),
                "playing": bool(_timeline.is_playing())}, 0
    if op == "play":
        _timeline.play()
        return {"ok": True}, 0
    if op == "pause":
        _timeline.pause()
        return {"ok": True}, 0
    return {"ok": False, "error": f"comando sconosciuto: {op}"}, 0


for _f in (CMD, REPLY, POSE):
    if os.path.exists(_f):
        os.remove(_f)

if not args.no_play:
    _timeline.play()
print("[standalone] pronto: comandi su", CMD)

pending = None      # (cmd, frame rimanenti) per i comandi che aspettano frame
try:
    while app.is_running():
        app.update()
        _write_pose()
        if pending is not None:
            cmd, n = pending
            n -= 1
            if n <= 0:
                res = {"ok": True, "pose": _pose_of(_snapshot(_get_art()))}
                if cmd.get("cmd") == "step":
                    res["frames"] = int(cmd.get("n", 60))
                res["id"] = cmd.get("id")
                _reply(res)
                pending = None
            else:
                pending = (cmd, n)
            continue
        if os.path.exists(CMD):
            try:
                with open(CMD) as f:
                    cmd = json.load(f)
            except (OSError, ValueError):
                continue
            os.remove(CMD)
            try:
                res, frames = _handle(cmd)
            except Exception as e:
                import traceback
                res, frames = {"ok": False, "error": f"{type(e).__name__}: {e}",
                               "trace": traceback.format_exc()}, 0
            if frames > 0:
                pending = (cmd, frames)
            else:
                res["id"] = cmd.get("id")
                _reply(res)
except KeyboardInterrupt:
    pass
finally:
    app.close()
