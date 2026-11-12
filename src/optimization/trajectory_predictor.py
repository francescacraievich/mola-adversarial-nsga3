#!/usr/bin/env python3
"""
Waypoint follower model and unicycle-model predictor.

FollowerModel reproduces the waypoint follower's control law: on the estimated
pose it computes the bearing error to the waypoint; above an alignment threshold
it turns in place (omega = clip(k_ang * err), at least 0.05 rad/s), otherwise it
drives with v = min(max_speed, k_lin * dist) and the same clipped omega; inside
the tolerance it stops. The orchestrator's controller uses it to compute the
/cmd_vel it publishes from the MOLA estimate.

predict_true_trajectory, damage_deviation and directional_bonus are a model-based
alternative, not used by the orchestrator (which evaluates candidates with
physical rollouts in Isaac Sim), kept for comparison. They propagate the follower
command with the unicycle model, midpoint heading integration

    theta_mid <- theta + 0.5*omega*dt
    x <- x + v*cos(theta_mid)*dt
    y <- y + v*sin(theta_mid)*dt
    theta <- theta + omega*dt

and ignore inertia, wheel slip and the differential controller's ramps.
"""

import math

import numpy as np


def yaw_from_quat(qx, qy, qz, qw) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class FollowerModel:
    """Waypoint follower control law (m, m/s, rad/s).

    Defaults must match the running follower: a mismatch makes the prediction
    wrong without any visible error.
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
        """Command (v, omega, distance to waypoint) computed on the estimated pose."""
        dx, dy = wp_x - est_x, wp_y - est_y
        dist = math.hypot(dx, dy)
        if dist < self.tolerance:
            return 0.0, 0.0, dist

        bearing_err = angle_diff(math.atan2(dy, dx), est_yaw)

        if abs(bearing_err) > self.align_threshold:
            # Turn in place
            w = max(-self.max_turn, min(self.max_turn, self.k_ang * bearing_err))
            if 0 < abs(w) < 0.05:
                w = math.copysign(0.05, w)
            return 0.0, w, dist

        v = min(self.max_speed, self.k_lin * dist)
        w = max(-self.max_turn, min(self.max_turn, self.k_ang * bearing_err))
        return v, w, dist


def predict_true_trajectory(est_poses, true_start, waypoint, dt=0.1,
                            follower=None):
    """True trajectory predicted by the unicycle model from the estimated poses.

    Args: est_poses (N,3) estimated (x, y, yaw) over the window; true_start true
    pose at window start; waypoint (2,) target of the follower; dt step between
    scans (0.1 s at 10 Hz). Returns an (N,3) array. At each step the follower
    sees the estimated pose and its command is applied to the true pose.
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

        # Midpoint heading integration: more accurate than explicit Euler
        # while turning.
        yaw_mid = yaw + 0.5 * w * dt
        x += v * math.cos(yaw_mid) * dt
        y += v * math.sin(yaw_mid) * dt
        yaw = angle_diff(yaw + w * dt, 0.0)

        out[i] = (x, y, yaw)

    return out


def damage_deviation(est_clean, est_attacked, true_start, waypoint, dt=0.1,
                     follower=None):
    """Distance (m) between the end points of the nominal and attacked trajectories.

    Both are predicted with the same model, so most unicycle-model errors cancel
    and the perturbation effect remains. The end point is the error the robot
    carries into the next window. Returns (deviation, nominal, attacked).
    """
    n = min(len(est_clean), len(est_attacked))
    if n < 2:
        return 0.0, None, None
    nominal = predict_true_trajectory(est_clean[:n], true_start, waypoint, dt, follower)
    attacked = predict_true_trajectory(est_attacked[:n], true_start, waypoint, dt, follower)
    dev = float(np.linalg.norm(attacked[-1, :2] - nominal[-1, :2]))
    return dev, nominal, attacked


def directional_bonus(nominal, attacked, goal):
    """Change in distance to the goal (m): positive if the attack moves away.

    Separates a deviation away from the goal from one of equal size around it.
    """
    goal = np.asarray(goal, dtype=float)
    d_nom = np.linalg.norm(nominal[-1, :2] - goal)
    d_att = np.linalg.norm(attacked[-1, :2] - goal)
    return float(d_att - d_nom)


if __name__ == "__main__":
    # Quick check: a constant 20 cm estimate error in y must steer the robot
    # and give a non-zero predicted deviation.
    n = 20
    true_start = (0.0, 0.0, 0.0)
    wp = (4.0, 0.0)

    clean = np.array([[0.8 * 0.1 * i, 0.0, 0.0] for i in range(n)])
    attacked = clean.copy()
    attacked[:, 1] += 0.20          # estimate shifted by 20 cm in y

    dev, nom, att = damage_deviation(clean, attacked, true_start, wp)
    print(f"errore di stima costante: 20.0 cm in y")
    print(f"arrivo nominale : ({nom[-1,0]:.3f}, {nom[-1,1]:.3f})")
    print(f"arrivo attaccato: ({att[-1,0]:.3f}, {att[-1,1]:.3f})")
    print(f"deviazione predetta: {dev*100:.1f} cm")
    print(f"variazione distanza dal bersaglio: "
          f"{directional_bonus(nom, att, wp)*100:+.1f} cm")
