"""
Test di fedelta' del salva/ripristina dello stato del robot in Isaac Sim.

Misura: salva lo stato, avanza di N_STEPS frame (passata A), ripristina, avanza
di nuovo con la stessa dinamica (passata B) e riporta lo scarto fra i due punti
d'arrivo, assoluto e relativo alla distanza percorsa. Se lo scarto e' molto
minore della deviazione indotta dall'attacco (indicativamente sotto il
millimetro su un metro), i candidati NSGA-III si possono valutare con rollout
fisici; altrimenti serve il replay di nuvole registrate con il modello uniciclo.

Uso: gira nello Script Editor con la simulazione in Play. Il robot deve muoversi
durante il test, comandato dalla stessa catena di controllo dell'attacco
(cmd_vel -> differential_controller). Prima di eseguire lo script, da un
terminale:

    source /opt/ros/jazzy/setup.bash
    ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \\
        "{linear: {x: 0.5}, angular: {z: 0.15}}"

La componente angolare rende il test sensibile allo stato dei giunti e alla
velocita' angolare, che in linea retta non contano. Al termine fermare il
publisher e fare Stop/Play per riportare il robot al punto di partenza.
Risultato in /tmp/restore_fidelity.txt.
"""

import asyncio

import numpy as np
import omni.kit.app
from isaacsim.core.prims import SingleArticulation

ROBOT = "/World/Nova_Carter_ROS"
N_STEPS = 400         # frame per passata: a ~60 fps e 0.5 m/s sono circa 1 m
PUSH = 0.0            # velocita' iniziale impressa qui; 0 = il moto arriva da cmd_vel
OUT = "/tmp/restore_fidelity.txt"


def snapshot(art):
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


def restore(art, s):
    art.set_world_pose(position=s["pos"], orientation=s["orient"])
    art.set_joint_positions(s["jpos"])
    art.set_joint_velocities(s["jvel"])
    art.set_linear_velocity(s["lin"])
    art.set_angular_velocity(s["ang"])


async def step(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def main():
    lines = []

    def log(s):
        lines.append(s)
        print(s)

    def dump():
        try:
            with open(OUT, "w") as f:
                f.write("\n".join(lines))
            print(f"\n[restore_test] scritto {OUT}")
        except OSError as e:
            print(f"[restore_test] impossibile scrivere {OUT}: {e}")

    try:
        await _body(log)
    except Exception:
        # Un'eccezione in una coroutine lanciata con ensure_future viene
        # inghiottita in silenzio: senza questa cattura il file non verrebbe scritto.
        import traceback
        log("\n" + "=" * 64)
        log("  ERRORE")
        log("=" * 64)
        log(traceback.format_exc())
    finally:
        dump()


async def _body(log):
    art = SingleArticulation(prim_path=ROBOT, name="restore_test")
    art.initialize()

    log("=" * 64)
    log("  TEST DI FEDELTA' DEL RIPRISTINO")
    log("=" * 64)

    # Il moto arriva da cmd_vel: serve la stessa catena di controllo dell'attacco,
    # comprese le rampe del differential_controller.
    if PUSH > 0:
        art.set_linear_velocity(np.array([PUSH, 0.0, 0.0]))
    await step(5)

    s0 = snapshot(art)
    if float(np.linalg.norm(s0["lin"][:2])) < 0.05:
        log("\n  ATTENZIONE: il robot e' fermo. Il test misurerebbe il")
        log("  ripristino su pochi millimetri di moto, senza dire nulla su")
        log("  un rollout da un metro. Pubblica cmd_vel da un terminale:")
        log("    ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \\")
        log("        \"{linear: {x: 0.5}, angular: {z: 0.15}}\"")
        log("  e rilancia.")

    log(f"\nStato salvato:")
    log(f"  posizione  {s0['pos']}")
    log(f"  vel lin    {s0['lin']}")
    log(f"  vel giunti {np.round(s0['jvel'], 4)}")

    # Passata A
    await step(N_STEPS)
    a = snapshot(art)
    log(f"\nPassata A dopo {N_STEPS} frame:")
    log(f"  posizione  {a['pos']}")

    # Ripristino
    restore(art, s0)
    await step(1)
    r = snapshot(art)
    err_pos = float(np.linalg.norm(r["pos"] - s0["pos"]))
    err_lin = float(np.linalg.norm(r["lin"] - s0["lin"]))
    log(f"\nRipristino immediato (scarto dallo stato salvato):")
    log(f"  posizione  {err_pos*1000:.4f} mm")
    log(f"  vel lin    {err_lin*1000:.4f} mm/s")

    # Passata B
    await step(N_STEPS)
    b = snapshot(art)
    log(f"\nPassata B dopo {N_STEPS} frame:")
    log(f"  posizione  {b['pos']}")

    # Verdetto
    d = float(np.linalg.norm(b["pos"][:2] - a["pos"][:2]))
    travel = float(np.linalg.norm(a["pos"][:2] - s0["pos"][:2]))

    log("\n" + "=" * 64)
    log(f"  distanza percorsa nella passata : {travel*100:.2f} cm")
    log(f"  SCARTO FRA LE DUE PASSATE       : {d*1000:.3f} mm")
    if travel > 1e-6:
        log(f"  scarto relativo                 : {100*d/travel:.3f}%")
    log("=" * 64)

    if travel < 0.20:
        log("\n  TEST NON CONCLUSIVO: il robot ha percorso meno di 20 cm.")
        log("  Serve che percorra circa un metro, come in un rollout vero.")
        log("  Pubblica cmd_vel da un terminale e rilancia (vedi l'intestazione).")
    elif d < 0.001:
        log("\n  OTTIMO. Il ripristino e' fedele sotto il millimetro.")
        log("  -> ROLLOUT FISICO: ogni candidato viene provato facendo")
        log("     avanzare davvero la simulazione. Si misura la deviazione")
        log("     invece di stimarla con un modello.")
    elif d < 0.010:
        log("\n  ACCETTABILE. Scarto di pochi millimetri.")
        log("  -> ROLLOUT praticabile purche' la deviazione attesa sotto")
        log("     attacco sia di parecchi centimetri. Da confrontare con")
        log("     l'ampiezza dell'effetto che si vuole misurare.")
    else:
        log("\n  TROPPO IMPRECISO. I candidati partirebbero da condizioni")
        log("  diverse e il confronto sarebbe viziato.")
        log("  -> REPLAY + modello uniciclo.")


asyncio.ensure_future(main())
