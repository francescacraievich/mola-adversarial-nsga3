"""
Isaac Sim launched from the terminal with scene, ROS 2 bridge, Play and rollout
server in one process, replacing manual scene loading and the Script Editor.

    source /opt/ros/jazzy/setup.bash
    python3 isaac/run_isaac_standalone.py [--headless] [--scene isaac/carter_warehouse.usd]

--headless runs without a window (faster, same RTX LiDAR rendering). The server
answers orchestrator commands on /tmp/isaac_cmd.json like isaac_rollout_server.py
(save, restore, set_pose, step, pose, play, pause). Ctrl-C closes everything.

Attitude (isaac/assetto.py, ZYX Euler): the assetto command reads the pose with
full quaternion, roll, pitch, yaw, base velocities and joints without stepping;
set_state accepts roll and pitch (6D pose) or keep_tilt (roll and pitch from the
restore). Without them set_state sets yaw only.

Named snapshots: save and restore with "nome" use a separate slot, so the
unnamed start-of-segment snapshot stays valid.

With REGISTRI_DAL_VIVO_DIR=<dir> writes <dir>/isaac_cmd.csv: every /cmd_vel
received (own rclpy node, in a thread), the wheel speeds from the graph's
differential controller and the true robot velocity at the next step
(src/optimization/registri_dal_vivo.py).

The ROS 2 bridge needs the Jazzy libraries (source /opt/ros/jazzy before launch);
/front_3d_lidar/lidar_points should arrive at 10 Hz (`ros2 topic hz`).
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

# The windowed app (isaacsim.exp.full.kit) caps the main loop at 60 Hz; the
# SimulationApp experience does not, and runs at RTF ~1.2. Apply the same cap so the
# wall time per scan matches windowed runs.
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
_saved_nomi = {}         # named snapshots (save/restore with "nome")


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


_mod_assetto = None
_CHIAVI_ASSETTO = ("roll", "pitch", "keep_tilt")
_CAPS_ASSETTO = ["set_state_assetto", "assetto"]


def _assetto():
    """Load isaac/assetto.py from this directory, independent of sys.path, no .pyc."""
    global _mod_assetto
    if _mod_assetto is None:
        import types
        p = Path(__file__).resolve().parent / "assetto.py"
        m = types.ModuleType("assetto")
        m.__file__ = str(p)
        exec(compile(p.read_text(), str(p), "exec"), m.__dict__)
        _mod_assetto = m
    return _mod_assetto


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


_t_pose_err = [0.0]


def _write_pose():
    try:
        s = _snapshot(_get_art())
        tmp = POSE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pose": _pose_of(s), "wall": time.time(),
                       "playing": bool(_timeline.is_playing())}, f)
        os.replace(tmp, POSE)
    except Exception as e:
        # A failure here leaves a stale pose file that the orchestrator reads as a
        # stall: log it (at most one line every 2 s).
        now = time.monotonic()
        if now - _t_pose_err[0] > 2.0:
            _t_pose_err[0] = now
            print(f"[standalone] file della posa non scritto: {type(e).__name__}: {e}", flush=True)


def _handle(cmd):
    """Run a command; commands that need frames return how many."""
    global _saved
    art = _get_art()
    op = cmd.get("cmd")
    if op == "save" and "nome" in cmd:
        s = _snapshot(art)
        _saved_nomi[str(cmd["nome"])] = s
        return {"ok": True, "pose": _pose_of(s)}, 0
    if op == "restore" and "nome" in cmd:
        s = _saved_nomi.get(str(cmd["nome"]))
        if s is None:
            return {"ok": False, "error": f"nessuno stato salvato con nome {cmd['nome']}"}, 0
        _restore(art, s)
        return None, 1
    if op == "save":
        _saved = _snapshot(art)
        return {"ok": True, "pose": _pose_of(_saved)}, 0
    if op == "restore":
        if _saved is None:
            return {"ok": False, "error": "nessuno stato salvato"}, 0
        _restore(art, _saved)
        return None, 1                     # one frame, then the pose
    if op == "set_pose":
        yaw = float(cmd.get("yaw", 0.0))
        pos, _ = art.get_world_pose()
        art.set_world_pose(position=np.array([float(cmd["x"]), float(cmd["y"]), float(pos[2])]),
                           orientation=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
        art.set_joint_velocities(np.zeros_like(art.get_joint_velocities()))
        art.set_linear_velocity(np.zeros(3))
        art.set_angular_velocity(np.zeros(3))
        return None, 1
    if op == "set_state":
        # Twin from the attacker's estimate: after restore, move the base to scene
        # (x, y, yaw) with velocity (vx, vy, wz); joints keep the restored state, or
        # are stopped with zero_joints. With roll/pitch or keep_tilt the orientation
        # comes from assetto.quat_set_state, which rejects inconsistent parameters
        # before any write; z stays that of the restore.
        yaw = float(cmd.get("yaw", 0.0))
        pos, q_att = art.get_world_pose()
        if any(k in cmd for k in _CHIAVI_ASSETTO):
            q = _assetto().quat_set_state(cmd, q_att)
        else:
            q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        art.set_world_pose(position=np.array([float(cmd["x"]), float(cmd["y"]), float(pos[2])]),
                           orientation=q)
        if cmd.get("zero_joints", True):
            art.set_joint_velocities(np.zeros_like(art.get_joint_velocities()))
        art.set_linear_velocity(np.array([float(cmd.get("vx", 0.0)), float(cmd.get("vy", 0.0)), 0.0]))
        art.set_angular_velocity(np.array([0.0, 0.0, float(cmd.get("wz", 0.0))]))
        return None, 1
    if op == "caps":
        # save_nome: named snapshots (save and restore with "nome")
        return {"ok": True, "caps": ["set_state"] + _CAPS_ASSETTO + ["save_nome"]}, 0
    if op == "assetto":
        # Read-only, no frame: 6D pose, velocities (world frame) and joints.
        pos, q = art.get_world_pose()
        try:
            nomi = list(art.dof_names)
        except Exception:
            nomi = None
        return (_assetto().risposta_assetto(pos, q, art.get_linear_velocity(),
                                            art.get_angular_velocity(), nomi,
                                            art.get_joint_positions(), art.get_joint_velocities())
                | {"playing": bool(_timeline.is_playing())}), 0
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


# Live logs, only with REGISTRI_DAL_VIVO_DIR. A start-up or write error disables the
# log with a warning; Isaac carries on without it.
_reg = None
_reg_exec = _reg_ctx = None
_reg_frame = [0]


def _reg_spegni(dove, e):
    global _reg
    if _reg is not None:
        print(f"[standalone] registri: {dove}: {type(e).__name__}: {e}; registro disattivato",
              flush=True)
        try:
            _reg.close()
        except Exception:
            pass
        _reg = None


if os.environ.get("REGISTRI_DAL_VIVO_DIR"):
    try:
        # repo root appended last so Isaac packages keep precedence
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        import threading  # noqa: E402
        import rclpy  # noqa: E402
        from rclpy.executors import SingleThreadedExecutor  # noqa: E402
        from geometry_msgs.msg import Twist  # noqa: E402
        import omni.graph.core as og  # noqa: E402
        from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
        from src.optimization.registri_dal_vivo import RegistroIsaacCmd  # noqa: E402

        _reg = RegistroIsaacCmd(os.environ["REGISTRI_DAL_VIVO_DIR"])
        # differential_drive graph node of the asset (docs/catena_ros.md).
        _REG_DIFF = os.environ.get("REGISTRI_DAL_VIVO_DIFF_NODE",
                                   ROBOT + "/differential_drive/differential_controller_01")
        _reg_avvisi = set()

        def _reg_avviso(chi, e):
            if chi not in _reg_avvisi:
                _reg_avvisi.add(chi)
                print(f"[standalone] registri: {chi} non letto: {type(e).__name__}: {e}", flush=True)

        def _reg_ruote():
            try:
                vc = og.Controller.attribute(_REG_DIFF + ".outputs:velocityCommand").get()
                return float(vc[0]), float(vc[1])
            except Exception as e:
                _reg_avviso("velocityCommand", e)
                raise

        def _reg_vel():
            # linear velocity along the robot axis and angular velocity about z
            try:
                art = _get_art()
                lin = art.get_linear_velocity()
                ang = art.get_angular_velocity()
                _, q = art.get_world_pose()
                yaw = _yaw(q)
                return (float(lin[0] * np.cos(yaw) + lin[1] * np.sin(yaw)), float(ang[2]))
            except Exception as e:
                _reg_avviso("velocita' vera", e)
                raise

        def _reg_tempo():
            try:
                return (float(SimulationManager.get_simulation_time()),
                        int(SimulationManager.get_num_physics_steps()))
            except Exception as e:
                _reg_avviso("tempo simulato", e)
                return float("nan"), _reg_frame[0]   # frame counter instead of physics steps

        def _reg_cb(msg, info):
            # DDS reception time if available, else the callback time
            try:
                r = _reg
                if r is None:
                    return
                try:
                    t = int(info["received_timestamp"]) * 1e-9
                except Exception:
                    t = 0.0
                r.comando(msg.linear.x, msg.angular.z, t_ric_wall=t if t > 0 else time.time())
            except Exception as e:
                _reg_spegni("ricezione di /cmd_vel", e)

        _reg_ctx = rclpy.Context()
        rclpy.init(context=_reg_ctx)
        _reg_node = rclpy.create_node("registri_isaac_cmd", context=_reg_ctx)
        _reg_node.create_subscription(Twist, "/cmd_vel", _reg_cb, 10)
        _reg_exec = SingleThreadedExecutor(context=_reg_ctx)
        _reg_exec.add_node(_reg_node)
        threading.Thread(target=_reg_exec.spin, daemon=True).start()
        print(f"[standalone] registri dal vivo in {os.environ['REGISTRI_DAL_VIVO_DIR']}/isaac_cmd.csv",
              flush=True)
    except Exception as e:
        print(f"[standalone] registri non avviati: {type(e).__name__}: {e}; Isaac continua senza",
              flush=True)
        if _reg is not None:
            _reg.close()
            _reg = None
        if _reg_ctx is not None:
            try:
                _reg_ctx.try_shutdown()
            except Exception:
                pass
        _reg_exec = _reg_ctx = None


def _reg_dopo_step(n):
    try:
        _reg_frame[0] += 1
        t_sim, step = _reg_tempo()
        _reg.dopo_lo_step(n, step, t_sim, _reg_ruote, _reg_vel)
    except Exception as e:
        _reg_spegni("scrittura", e)


for _f in (CMD, REPLY, POSE):
    if os.path.exists(_f):
        os.remove(_f)

if not args.no_play:
    _timeline.play()
print("[standalone] pronto: comandi su", CMD)

pending = None      # (cmd, frames left) for commands waiting on frames
try:
    while app.is_running():
        _n_reg = _reg.prima_dello_step() if _reg is not None else 0
        _t_upd = time.monotonic()
        app.update()
        _dt_upd = time.monotonic() - _t_upd
        if _dt_upd > 1.0:
            print(f"[standalone] app.update lento: {_dt_upd:.2f} s", flush=True)
        if _reg is not None:
            _reg_dopo_step(_n_reg)
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
            _t_cmd = time.monotonic()
            try:
                res, frames = _handle(cmd)
            except Exception as e:
                import traceback
                res, frames = {"ok": False, "error": f"{type(e).__name__}: {e}",
                               "trace": traceback.format_exc()}, 0
                print(f"[standalone] comando '{cmd.get('cmd')}' fallito: {type(e).__name__}: {e}", flush=True)
            _dt_cmd = time.monotonic() - _t_cmd
            if _dt_cmd > 0.5:
                print(f"[standalone] comando '{cmd.get('cmd')}' eseguito in {_dt_cmd:.2f} s", flush=True)
            if frames > 0:
                pending = (cmd, frames)
            else:
                res["id"] = cmd.get("id")
                _reply(res)
except KeyboardInterrupt:
    pass
finally:
    if _reg is not None:
        _reg.close()
    for _chiudi in (lambda: _reg_exec.shutdown(timeout_sec=1.0), lambda: _reg_ctx.try_shutdown()):
        try:
            _chiudi()
        except Exception:
            pass
    app.close()
