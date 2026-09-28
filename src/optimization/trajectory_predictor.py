#!/usr/bin/env python3
"""
Modello del follower e predittore a modello uniciclo.

FollowerModel replica la legge di controllo di src/nodes/waypoint_follower_node.py
ed e' usato dall'orchestratore (src/optimization/attack_orchestrator.py) per
ricostruire il comando che il follower calcola sulla posa stimata da MOLA.

predict_true_trajectory, damage_deviation e directional_bonus sono un'alternativa
a modello, non usata dall'orchestratore, mantenuta per confronto: propagano il
comando del follower con il modello uniciclo

    x <- x + v*cos(theta)*dt
    y <- y + v*sin(theta)*dt
    theta <- theta + omega*dt

e stimano la deviazione senza far girare la simulazione. L'orchestratore valuta
invece i candidati con rollout fisici in Isaac Sim. Il modello ignora inerzia,
slittamento e le rampe del differential_controller.
"""

import math

import numpy as np


def yaw_from_quat(qx, qy, qz, qw) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class FollowerModel:
    """Replica della legge di controllo di waypoint_follower_node.py.

    I valori di default devono coincidere con quelli del nodo reale: una
    divergenza renderebbe la predizione sbagliata senza che nulla lo segnali.
    """

    def __init__(self, max_speed=0.8, max_turn=0.5, align_threshold_deg=25.0,
                 k_lin=0.8, k_ang=1.2, tolerance=0.25):
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.align_threshold = math.radians(align_threshold_deg)
        self.k_lin = k_lin
        self.k_ang = k_ang
        self.tolerance = tolerance

    def command(self, est_x, est_y, est_yaw, wp_x, wp_y):
        """Comando (v, omega, distanza dal waypoint) calcolato sulla posa stimata."""
        dx, dy = wp_x - est_x, wp_y - est_y
        dist = math.hypot(dx, dy)
        if dist < self.tolerance:
            return 0.0, 0.0, dist

        bearing_err = angle_diff(math.atan2(dy, dx), est_yaw)

        if abs(bearing_err) > self.align_threshold:
            # Rotazione sul posto
            w = max(-self.max_turn, min(self.max_turn, self.k_ang * bearing_err))
            if 0 < abs(w) < 0.05:
                w = math.copysign(0.05, w)
            return 0.0, w, dist

        v = min(self.max_speed, self.k_lin * dist)
        w = max(-self.max_turn, min(self.max_turn, self.k_ang * bearing_err))
        return v, w, dist


def predict_true_trajectory(est_poses, true_start, waypoint, dt=0.1,
                            follower=None):
    """Traiettoria vera predetta dal modello uniciclo, date le pose stimate.

    Parameters
    ----------
    est_poses : (N,3) array
        Pose stimate (x, y, yaw) sulla finestra.
    true_start : (3,)
        Posa vera del robot all'inizio della finestra (x, y, yaw).
    waypoint : (2,)
        Bersaglio che il follower sta inseguendo.
    dt : float
        Passo temporale fra due scan (0.1 s a 10 Hz).

    Returns
    -------
    (N,3) array con la traiettoria vera predetta.

    Il follower vede ad ogni passo la posa stimata, cioe' la posa vera piu'
    l'errore indotto dalla perturbazione; il comando che ne risulta viene
    applicato alla posa vera.
    """
    follower = follower or FollowerModel()
    est_poses = np.asarray(est_poses, dtype=float)
    n = len(est_poses)
    if n == 0:
        return np.array([true_start], dtype=float)

    x, y, yaw = map(float, true_start)
    out = np.zeros((n, 3))

    for i in range(n):
        ex, ey, eyaw = est_poses[i]
        v, w, _ = follower.command(ex, ey, eyaw, waypoint[0], waypoint[1])

        # Integrazione con il punto medio dell'orientamento: piu' accurata
        # dell'Eulero esplicito durante le rotazioni.
        yaw_mid = yaw + 0.5 * w * dt
        x += v * math.cos(yaw_mid) * dt
        y += v * math.sin(yaw_mid) * dt
        yaw = angle_diff(yaw + w * dt, 0.0)

        out[i] = (x, y, yaw)

    return out


def damage_deviation(est_clean, est_attacked, true_start, waypoint, dt=0.1,
                     follower=None):
    """Distanza fra i punti d'arrivo della traiettoria nominale e di quella attaccata.

    Entrambe le traiettorie sono predette con lo stesso modello, cosi' gli errori
    del modello uniciclo si cancellano in buona parte e resta l'effetto della
    perturbazione. Si usa il punto d'arrivo perche' e' l'errore che il robot si
    porta nella finestra successiva. Ritorna (deviazione, nominale, attaccata).
    """
    n = min(len(est_clean), len(est_attacked))
    if n < 2:
        return 0.0, None, None
    nominal = predict_true_trajectory(est_clean[:n], true_start, waypoint, dt, follower)
    attacked = predict_true_trajectory(est_attacked[:n], true_start, waypoint, dt, follower)
    dev = float(np.linalg.norm(attacked[-1, :2] - nominal[-1, :2]))
    return dev, nominal, attacked


def directional_bonus(nominal, attacked, goal):
    """Variazione della distanza dal bersaglio: positiva se l'attacco allontana.

    Distingue una deviazione che allontana dal bersaglio da una della stessa
    entita' che gli gira intorno.
    """
    goal = np.asarray(goal, dtype=float)
    d_nom = np.linalg.norm(nominal[-1, :2] - goal)
    d_att = np.linalg.norm(attacked[-1, :2] - goal)
    return float(d_att - d_nom)


if __name__ == "__main__":
    # Prova rapida: un errore di stima costante di 20 cm in y deve far sterzare
    # il robot e dare una deviazione predetta diversa da zero.
    n = 20
    true_start = (0.0, 0.0, 0.0)
    wp = (4.0, 0.0)

    clean = np.array([[0.8 * 0.1 * i, 0.0, 0.0] for i in range(n)])
    attacked = clean.copy()
    attacked[:, 1] += 0.20          # stima spostata di 20 cm in y

    dev, nom, att = damage_deviation(clean, attacked, true_start, wp)
    print(f"errore di stima costante: 20.0 cm in y")
    print(f"arrivo nominale : ({nom[-1,0]:.3f}, {nom[-1,1]:.3f})")
    print(f"arrivo attaccato: ({att[-1,0]:.3f}, {att[-1,1]:.3f})")
    print(f"deviazione predetta: {dev*100:.1f} cm")
    print(f"variazione distanza dal bersaglio: "
          f"{directional_bonus(nom, att, wp)*100:+.1f} cm")
