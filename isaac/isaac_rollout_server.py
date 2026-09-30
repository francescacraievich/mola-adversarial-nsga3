"""
Server dei comandi per i rollout fisici dell'attacco receding-horizon.

Gira nello Script Editor di Isaac Sim: con la simulazione in Play, aprire questo
file nello Script Editor ed eseguirlo. Resta attivo finche' non viene fermato o
la scena ricaricata.

Comunicazione via file, senza rclpy dentro Isaac Sim:
    /tmp/isaac_cmd.json    comando scritto dall'orchestratore (src/optimization/)
    /tmp/isaac_reply.json  risposta di questo script, con lo stesso campo "id"
    /tmp/isaac_pose.json   posa vera corrente, riscritta a ogni frame senza richiesta

Comandi: save (memorizza posa, velocita' e giunti), restore (ripristina lo stato
salvato), set_pose x y yaw (riporta il robot a una posa nota, da fermo), step n
(avanza di n frame), pose (posa vera corrente), play, pause.
NSGA-III prova piu' candidati dallo stesso stato e il robot e' uno solo, quindi
dopo ogni prova viene riportato indietro. Il campo "id" permette all'orchestratore
di distinguere la risposta nuova da quella del comando precedente.
"""

import asyncio
import json
import os
import time

import numpy as np
import omni.kit.app
import omni.timeline
from isaacsim.core.prims import SingleArticulation

ROBOT = "/World/Nova_Carter_ROS"
CMD = "/tmp/isaac_cmd.json"
REPLY = "/tmp/isaac_reply.json"
POSE = "/tmp/isaac_pose.json"

_timeline = omni.timeline.get_timeline_interface()
_art = None
_saved = None
_busy = False


def _get_art():
    global _art
    if _art is None:
        _art = SingleArticulation(prim_path=ROBOT, name="attack_ctrl")
        _art.initialize()
    return _art


def _snapshot(art):
    """Stato completo: posa, velocita' del corpo, posizioni e velocita' dei giunti.

    Le velocita' sono necessarie: ripristinare la sola posa lascerebbe al robot la
    quantita' di moto della prova precedente.
    """
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


def _reply(payload):
    tmp = REPLY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    # Scrittura atomica: l'orchestratore non deve leggere un JSON troncato.
    os.replace(tmp, REPLY)


async def _handle(cmd):
    global _saved
    art = _get_art()
    op = cmd.get("cmd")

    if op == "save":
        _saved = _snapshot(art)
        p = _saved["pos"]
        return {"ok": True, "pose": [float(p[0]), float(p[1]),
                                     _yaw(_saved["orient"])]}

    if op == "restore":
        if _saved is None:
            return {"ok": False, "error": "nessuno stato salvato"}
        _restore(art, _saved)
        # Un frame perche' PhysX assorba lo stato scritto.
        await omni.kit.app.get_app().next_update_async()
        s = _snapshot(art)
        p = s["pos"]
        return {"ok": True, "pose": [float(p[0]), float(p[1]), _yaw(s["orient"])]}

    if op == "step":
        n = int(cmd.get("n", 60))
        for _ in range(n):
            await omni.kit.app.get_app().next_update_async()
        s = _snapshot(art)
        p = s["pos"]
        return {"ok": True, "pose": [float(p[0]), float(p[1]), _yaw(s["orient"])],
                "frames": n}

    if op == "pose":
        s = _snapshot(art)
        p = s["pos"]
        return {"ok": True, "pose": [float(p[0]), float(p[1]), _yaw(s["orient"])],
                "playing": bool(_timeline.is_playing())}

    if op == "set_pose":
        # Riporta il robot a una posa nota con velocita' e giunti azzerati:
        # inizio ripetibile per le run di una campagna.
        yaw = float(cmd.get("yaw", 0.0))
        pos, _ = art.get_world_pose()
        new_pos = np.array([float(cmd["x"]), float(cmd["y"]), float(pos[2])])
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])   # w, x, y, z
        art.set_world_pose(position=new_pos, orientation=quat)
        art.set_joint_velocities(np.zeros_like(art.get_joint_velocities()))
        art.set_linear_velocity(np.zeros(3))
        art.set_angular_velocity(np.zeros(3))
        await omni.kit.app.get_app().next_update_async()
        s = _snapshot(art)
        p = s["pos"]
        return {"ok": True, "pose": [float(p[0]), float(p[1]), _yaw(s["orient"])]}

    if op == "play":
        _timeline.play()
        return {"ok": True}

    if op == "pause":
        _timeline.pause()
        return {"ok": True}

    return {"ok": False, "error": f"comando sconosciuto: {op}"}


def _write_pose():
    """Posa vera a ogni frame, letta dall'orchestratore senza richiesta.

    Serve alla traccia per tick e al criterio d'arresto: una richiesta per
    frame bloccherebbe il ciclo di controllo per la latenza dello scambio a
    file. Il campo wall permette di riconoscere un file vecchio.
    """
    try:
        pos, orient = _get_art().get_world_pose()
        tmp = POSE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pose": [float(pos[0]), float(pos[1]), _yaw(orient)],
                       "wall": time.time(),
                       "playing": bool(_timeline.is_playing())}, f)
        os.replace(tmp, POSE)
    except Exception:
        pass


def _tick(_event):
    global _busy
    _write_pose()
    if _busy or not os.path.exists(CMD):
        return
    try:
        with open(CMD) as f:
            cmd = json.load(f)
    except (OSError, ValueError):
        return
    os.remove(CMD)
    _busy = True

    async def run():
        global _busy
        try:
            res = await _handle(cmd)
        except Exception as e:
            import traceback
            res = {"ok": False, "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()}
        res["id"] = cmd.get("id")
        _reply(res)
        _busy = False

    asyncio.ensure_future(run())


for _f in (CMD, REPLY, POSE):
    if os.path.exists(_f):
        os.remove(_f)

if "attack_ctrl_sub" in dir():
    try:
        attack_ctrl_sub.unsubscribe()
    except Exception:
        pass

attack_ctrl_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _tick, name="attack_rollout_server")

print("[attack_server] attivo")
print(f"[attack_server]   comandi  : {CMD}")
print(f"[attack_server]   risposte : {REPLY}")
print(f"[attack_server]   posa     : {POSE} (a ogni frame)")
print("[attack_server]   save | restore | set_pose | step n | pose | play | pause")
