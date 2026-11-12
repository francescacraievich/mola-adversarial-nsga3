"""End-of-window stop on the estimated distance (--window-stop).

legacy: the robot drives at the follower speed until the MOLA estimated distance
(Euclidean from the window-start estimate) reaches the horizon H, then a zero
command. At the condition the robot is at 0.8 m/s: estimate latency (~0.2 s),
braking (~20 cm) and estimate steps (8 cm per scan) take it ~37-39 cm past H.

ramp: in the last metres the speed falls with the remaining estimated distance,
v = min(v_follower, max(v_min, k (H - d_est))). At the condition the robot is at
v_min, so latency and braking cost ~1-2 cm. The stop condition is the same.

Pure functions only (speed law, end of coast, offsets from H), testable without
ROS; the orchestrator calls them in its drive loop.
"""

import math

MODES = ("legacy", "ramp")
DEFAULT_K = 1.5          # 1/s: k * estimate latency (0.2 s) = 0.3, below 1/e
DEFAULT_VMIN = 0.05      # m/s
STALE_S = 0.3            # s: estimate older than this -> keep the last ramp speed
STILL_V = 0.02           # m/s: below this true speed the robot is still
STILL_TICKS = 3          # consecutive still ticks that end the coast (ramp)
COAST_CAP_S = 2.0        # s: coast cap in ramp mode


def ramp_speed(v_follower, est_dist, horizon, k=DEFAULT_K, vmin=DEFAULT_VMIN):
    """Ramp speed (m/s): v_follower far from H, k * remaining distance near it,
    never below vmin before H, zero from H on."""
    if est_dist >= horizon:
        return 0.0
    return min(v_follower, max(vmin, k * (horizon - est_dist)))


def ramp_command(v_follower, est_dist, horizon, k, vmin, est_age_s, v_ramp_prev,
                 stale_s=STALE_S):
    """Speed command of one tick with the ramp.

    If the estimate is older than stale_s (lost scans) the ramp speed of the
    previous tick is kept, so the robot does not speed up on a stale estimate.
    Returns (commanded v, ramp v to remember, phase), phase "rampa" when the
    ramp limits the follower, "crociera" otherwise.
    """
    if est_age_s > stale_s and v_ramp_prev is not None:
        v_ramp = v_ramp_prev
    elif est_dist >= horizon:
        v_ramp = 0.0
    else:
        v_ramp = max(vmin, k * (horizon - est_dist))
    v = min(v_follower, v_ramp)
    return v, v_ramp, ("rampa" if v_ramp < v_follower else "crociera")


def still_count(prev, speed, still_v=STILL_V):
    """Consecutive ticks with the robot still (NaN speed counts as moving)."""
    return prev + 1 if speed == speed and speed < still_v else 0


def coast_end_ramp(elapsed_s, still_ticks, est_new_after_stop,
                   cap_s=COAST_CAP_S, need=STILL_TICKS):
    """Reason the ramp-mode coast ends, None to continue: robot still for `need`
    ticks with an estimate received after the stop, or the time cap."""
    if still_ticks >= need and est_new_after_stop:
        return "fermo_stima_nuova"
    if elapsed_s >= cap_s:
        return "tetto"
    return None


def est_after_stop(est_stamp, arrest_stamp, est_wall, arrest_wall):
    """True if the estimate is newer than the stop: by stamp (simulation time)
    when both are known, otherwise by reception wall time."""
    if arrest_wall is None:
        return False
    if (est_stamp is not None and est_stamp == est_stamp
            and arrest_stamp is not None and arrest_stamp == arrest_stamp):
        return est_stamp > arrest_stamp
    return est_wall is not None and est_wall > arrest_wall


def scarto_cm(dist_m, horizon):
    """100 * (distance - H) in cm; None if the distance is missing."""
    if dist_m is None or dist_m != dist_m:
        return None
    return 100.0 * (dist_m - horizon)


STOP_INFO_KEYS = ("stop_mode", "stop_k", "stop_vmin",
                  "est_dist_at_cond_m", "true_dist_at_cond_m",
                  "est_dist_at_stop_m", "true_dist_at_stop_m",
                  "est_dist_end_m", "true_dist_end_m",
                  "scarto_H_est_cm", "scarto_H_true_cm",
                  "tau_cond_s", "est_jump_at_cond_cm", "coast_end_reason",
                  "drive_s")


def new_stop_info(mode, k, vmin):
    """Stop record of one rollout; k and vmin only in ramp mode."""
    d = {key: None for key in STOP_INFO_KEYS}
    d["stop_mode"] = mode
    if mode == "ramp":
        d["stop_k"] = float(k)
        d["stop_vmin"] = float(vmin)
    return d


def finalize_stop_info(info, horizon):
    """Offsets from H and rounding (m to 4 decimals, cm to 2, s to 3)."""
    info["scarto_H_est_cm"] = scarto_cm(info.get("est_dist_end_m"), horizon)
    info["scarto_H_true_cm"] = scarto_cm(info.get("true_dist_end_m"), horizon)
    for key, val in list(info.items()):
        if key in ("stop_k", "stop_vmin") or not isinstance(val, float):
            continue
        if not math.isfinite(val):
            info[key] = None
        elif key.endswith("_m"):
            info[key] = round(val, 4)
        elif key.endswith("_cm"):
            info[key] = round(val, 2)
        elif key.endswith("_s"):
            info[key] = round(val, 3)
    return info
