"""Twin variants: option checks, MOLA arguments and state handling.

Pure or read-only functions, testable without ROS: validation of option
combinations (replay, maprestore, keep drift, twin start from the attacker's
estimate, uniform use_sim_time), ROS arguments of the MOLA instances, coherence
of each variant with the threat model, the twin control frame after replay,
alignment checks at window start and twin start state from the attacker's
estimate. Variant E ("save and load") restores map, pose and sigma from the
shadow MOLA; variant G loads the full MOLA state of the shadow through a MOLA
overlay (G_OVERLAY_ROOT) and checks signatures, clock and hash chain at the end
of each rollout.
"""

import hashlib
import json
import math
import os
import re
import struct

from src.analysis.scarti_mola import leggi_printf
from src.optimization.window_record import compose_pose, pose_inverse

# Private /initialpose name for the twin: the name in the MOLA file is
# absolute (lidar_odometry_ros2_local.yaml:120), so __ns does not move it.
TWIN_INITIALPOSE = "/twin/initialpose"
TWIN_STATES = ("clean", "victim", "replay", "maprestore", "ombra_e", "ombra_g")
# Variants using the shadow: E (map, pose and sigma) and G (full state).
SHADOW_MODES = ("ombra_e", "ombra_g")
# Shadow: a MOLA that keeps running on the clouds the node publishes to the
# victim, in its own namespace (MOLA services have relative names).
SHADOW_NS = "/ombra"
# Line-buffered shadow stdout: MOLA's printf goes to stdout, which is block
# buffered when redirected to a file, and sigma is read while MOLA runs.
STDBUF_LIB = "/usr/libexec/coreutils/libstdbuf.so"
# Max gap (s) between the printf stamp and the last shadow pose stamp for the
# sigma to belong to the same scan: leggi_printf rounds stamps to 0.1 s, so the
# same scan is within 0.05 s and the neighbours at least 0.05 s away (10 Hz).
SHADOW_SIGMA_TOL_S = 0.05
# Differences between twin E and the victim that the environment variables
# cannot close (MOLA source references in the values).
E_FIELDS_DIFFERENT = {
    "twist": "0: FixedPose azzera lo stimatore (InitialLocalization.cpp:32-44); "
             "initial_twist letto ma non usato (Parameters.cpp:47-57)",
    "step_counter_post_relocalization": "5 nel gemello, 0 nella vittima (InitialLocalization.cpp:52-54)",
    "kf_decider_local_map": "vuoto: il primo scan del candidato diventa keyframe",
    "estimated_observation_radius": "inizializzato dal primo scan del candidato (AdaptiveVariables.cpp:228-268)",
}
# Thresholds for the start from the attacker's estimate: below them the robot
# is still and the velocity is zeroed. 0.05 m/s is above the finite-difference
# noise at rest (max 0.033 m/s, n = 60) and below the moving case (0.586 m/s).
TWIN_INIT_VMIN = 0.05
TWIN_INIT_WMIN = 0.05      # rad/s
VEL_SPAN_S = 0.5           # s of attacker MOLA poses used for the velocity
RX_KF = re.compile(r"updateLocalMap=(YES|NO)")


def check_no_nominal_args(args):
    """Allowed combinations of --no-nominal-rollout: error message or None.
    Without the nominal the damage comes only from the baseline and there are no
    clean-cloud twin references (internal f2, fixed settling, untracked
    nominal, lost twin ending the run)."""
    if not getattr(args, "no_nominal_rollout", False):
        return None
    sc = getattr(args, "settle_clouds", None)
    if sc is None and getattr(args, "victim_never_clean", False):
        sc = 20   # default of --victim-never-clean (check_spawn_b_args)
    fb = getattr(args, "f2_budget", None)
    bad = [o for o, on in (
        ("--damage-ref window", getattr(args, "damage_ref", "window") != "baseline"),
        ("--perceptibility internal", getattr(args, "perceptibility", "internal") == "internal"),
        ("--f2-budget", fb is not None and fb > 0),
        ("--untracked nominal", getattr(args, "untracked", "discard") == "nominal"),
        ("--settle-clouds 0", not sc or sc <= 0),
        ("--g-twin-lost-ends-run", getattr(args, "g_twin_lost_ends_run", False)),
        ("--passthrough", getattr(args, "passthrough", False)),
        ("--repeat-genome", getattr(args, "repeat_genome", None)),
        ("--repeat-spec", getattr(args, "repeat_spec", None)),
        ("--map-gate-test", getattr(args, "map_gate_test", False)),
        ("--victim restart", getattr(args, "victim", "continuous") != "continuous")) if on]
    if bad:
        return (f"--no-nominal-rollout non va con {', '.join(bad)}: servono --damage-ref baseline, "
                "--perceptibility chamfer, --settle-clouds > 0 e --victim continuous")
    return None


def check_twin_args(args):
    """Error message for an invalid option combination, None if valid."""
    err = check_no_nominal_args(args)
    if err:
        return err
    ts = getattr(args, "twin_state", "clean")
    cont = getattr(args, "victim", "continuous") == "continuous"
    att = getattr(args, "twin_align", "attacker") == "attacker"
    if ts == "replay" and not (cont and att):
        return ("--twin-state replay richiede --victim continuous e --twin-align attacker "
                "(il frame del gemello viene dalla sua stima o da quella dell'attaccante)")
    if getattr(args, "drift_reset_on", "genome") == "keep" and getattr(args, "attack_from", "spawn") != "spawn":
        return ("--drift-reset-on keep richiede --attack-from spawn: con motion l'assestamento "
                "vede la nuvola pulita e la deriva riappare a salto all'accensione")
    if getattr(args, "twin_init", "true") == "attacker_est" and not (cont and att):
        return "--twin-init attacker_est richiede --victim continuous e --twin-align attacker"
    if ts in SHADOW_MODES and not cont:
        return f"--twin-state {ts} richiede --victim continuous (l'ombra ascolta le nuvole della vittima)"
    if ts in SHADOW_MODES and getattr(args, "uniform_sim_time", "off") == "off":
        # with off the twin has use_sim_time true, victim and shadow do not
        return (f"--twin-state {ts} richiede --uniform-sim-time true o false: gemello e ombra "
                "devono avere lo stesso use_sim_time")
    if ts in SHADOW_MODES and (getattr(args, "repeat_genome", None) or getattr(args, "repeat_spec", None)
                               or getattr(args, "map_gate_test", False)):
        # these branches run before the window loop, where the shadow state is
        # saved: they would silently use the clean twin
        return (f"--twin-state {ts} non va con --repeat-genome, --repeat-spec o --map-gate-test "
                "(girano prima del ciclo delle finestre, con il gemello pulito)")
    if getattr(args, "twin_init", "true") == "ombra_est" and ts not in SHADOW_MODES:
        return "--twin-init ombra_est richiede --twin-state ombra_e o ombra_g"
    if getattr(args, "g_twin_lost_ends_run", False) and ts != "ombra_g":
        return "--g-twin-lost-ends-run richiede --twin-state ombra_g"
    if getattr(args, "g_nominale_senza_posa", "trattieni") == "continua":
        # without the nominal the damage must come from the baseline; ending the
        # run on a lost twin would contradict this rule
        if ts != "ombra_g":
            return "--g-nominale-senza-posa continua richiede --twin-state ombra_g"
        if getattr(args, "damage_ref", "window") != "baseline":
            return ("--g-nominale-senza-posa continua richiede --damage-ref baseline "
                    "(con window il danno e' misurato dal nominale)")
        if getattr(args, "g_twin_lost_ends_run", False):
            return "--g-nominale-senza-posa continua non va con --g-twin-lost-ends-run"
        if getattr(args, "passthrough", False):
            return "--g-nominale-senza-posa continua non va con --passthrough (nessun candidato)"
    if ts == "ombra_g":
        why = g_overlay_problem(getattr(args, "g_overlay", None) or G_OVERLAY_ROOT)
        if why:
            return f"--twin-state ombra_g: {why}"
    if ts == "replay" and (getattr(args, "replay_gate", 0) or 0) > 0:
        # the gate replays in victim mode from victim.last_est_end (the victim
        # estimate): not the threat-model-coherent replay
        return ("--replay-gate non e' compatibile con --twin-state replay: il gate usa il modo "
                "victim e la stima della vittima (victim.last_est_end)")
    return None


def ros_args_json(ns=None, sim_time=None, remaps=()):
    """ROS arguments for the ROS_ARGS variable, as a JSON list."""
    a = ["--ros-args"]
    if ns:
        a += ["-r", f"__ns:={ns}"]
    if sim_time is not None:
        a += ["-p", f"use_sim_time:={'true' if sim_time else 'false'}"]
    for src, dst in remaps:
        a += ["-r", f"{src}:={dst}"]
    return json.dumps(a)


def mola_ros_args(role, uniform_sim_time="off", isolate_initialpose=False):
    """ROS_ARGS for a MOLA instance, or None if it must not be set.

    Without ROS_ARGS mola-cli uses the launch command-line arguments
    (use_sim_time:=true); with ROS_ARGS it loses them. Hence with "off" the twin
    has use_sim_time true and victim and attacker false.
    uniform_sim_time: "off" (mixed, as above), "true" (all true), "false" (all false).
    """
    if role in ("victim", "attacker", "shadow"):
        # the shadow takes the victim's use_sim_time
        sim = {"off": None, "true": True, "false": False}[uniform_sim_time]
        return ros_args_json(ns=SHADOW_NS if role == "shadow" else f"/{role}", sim_time=sim)
    if role != "twin":
        raise ValueError(role)
    if uniform_sim_time == "off" and not isolate_initialpose:
        return None
    sim = False if uniform_sim_time == "false" else True
    remaps = [("/initialpose", TWIN_INITIALPOSE)] if isolate_initialpose else []
    return ros_args_json(sim_time=sim, remaps=remaps)


def coherence_record(twin_state, twin_init="true", twin_align="attacker", replay_frame="twin"):
    """Coherence of the variants with the threat model (the attacker knows
    neither the victim's true pose nor the internal state of its MOLA)."""
    state_ok = {"clean": True, "victim": False, "replay": True, "maprestore": False,
                "ombra_e": True, "ombra_g": True}[twin_state]
    why = {"clean": "MOLA del gemello vuota: nessuna informazione della vittima",
           "victim": "nuvole dell'attaccante, ma frame di controllo dalla posa pubblicata dalla MOLA della vittima",
           "replay": f"nuvole inviate dall'attaccante; frame dalla stima {'del gemello' if replay_frame == 'twin' else 'della MOLA dell attaccante'}",
           "maprestore": "mappa salvata dalla MOLA della vittima e sua posa creduta: solo confronto",
           "ombra_e": ("mappa, posa e sigma dall'ombra, una MOLA dell'attaccante sulle nuvole che lui "
                       "stesso invia alla vittima; posa vera solo nei campi lab_*"),
           "ombra_g": ("stato completo di MOLA dall'ombra (stesse nuvole inviate alla vittima); catena "
                       "degli hash; orologio del gemello dall'intestazione del file dell'ombra; posa "
                       "vera solo nei campi lab_*")}[twin_state]
    return {"twin_state_coerente_threat_model": state_ok,
            "twin_state_motivo": why,
            "twin_init_coerente_threat_model": twin_init in ("attacker_est", "ombra_est"),
            "twin_init_motivo": {"attacker_est": "posa dalla stima della MOLA dell'attaccante",
                                 "ombra_est": "posa dalla stima dell'ombra (stesso stimatore della MOLA del gemello)"}
                                .get(twin_init, "posa e velocita' vere della vittima ripristinate in Isaac (P13)"),
            "twin_align_coerente_threat_model": twin_align == "attacker"}


def replay_frame(mode, choice, rec, origin, believed):
    """Twin control frame after replay (--twin-state replay).

    choice "twin": twin estimate at the end of the replay composed with the
    attacker origin; valid only if the replay started from the first recorded
    cloud (same MOLA frame as victim and attacker, robot still at start).
    Otherwise, or with choice "attacker", the pose believed by the attacker's
    MOLA. Returns (frame, source, reason).
    """
    if mode != "replay":
        return None, None, None
    if choice == "twin":
        why = None
        if not rec or rec.get("status") != "ok" or rec.get("twin_est_raw") is None:
            why = f"rigioco {(rec or {}).get('status')}"
        elif rec.get("history_truncated"):
            why = "storia troncata (coda piena o last_s > 0): frame del gemello non ancorato"
        elif origin is None:
            why = "origine dell'attaccante assente"
        if why is None:
            return compose_pose(origin, rec["twin_est_raw"]), "twin_replay_end", None
        return believed, "attacker_est", why
    return believed, "attacker_est", None


def start_fingerprint_from_log(log_path, t_fp, tol=0.05):
    """Twin adaptive sigma and keyframe count at the first instant (stamp t_fp).

    sigma: adapt_thres_sigma of the last printf line (MOLA_LO_DEBUG_ICP_QUALITY=1)
    with stamp <= t_fp + tol. kf_updates: "updateLocalMap=YES" lines among the
    first n "updateLocalMap=" lines (DEBUG, same ICP block as the printf, one per
    scan), n = printf lines up to t_fp; the first cloud (no ICP) is not
    counted. None where the log lacks the lines."""
    out = {"sigma": None, "printf_rows_upto": None, "kf_updates": None, "kf_lines_upto": None}
    if not log_path or t_fp is None or t_fp != t_fp:
        return out
    try:
        rows = leggi_printf(str(log_path))
    except OSError:
        return out
    upto = [r for r in rows if r["stamp_s"] <= t_fp + tol]
    if rows:
        out["printf_rows_upto"] = len(upto)
        out["sigma"] = upto[-1]["sigma"] if upto else None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            kf = [m.group(1) for m in RX_KF.finditer(f.read())]
    except OSError:
        kf = []
    if kf and rows:
        first = kf[:len(upto)]
        out["kf_lines_upto"] = len(first)
        out["kf_updates"] = sum(1 for v in first if v == "YES")
    return out


def twin_alignment(true_twin, true_victim, twin_est_world, attacker_world):
    """Gaps at window start (cm, deg): twin true pose vs victim true pose, twin
    estimate (working frame) vs attacker estimate."""
    def gap(a, b):
        if a is None or b is None:
            return None, None
        d = 100 * math.hypot(a[0] - b[0], a[1] - b[1])
        e = math.degrees(math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2])))
        return round(d, 3), round(e, 3)
    tc, td = gap(true_twin, true_victim)
    ec, ed = gap(twin_est_world, attacker_world)
    f = lambda p: [float(v) for v in p] if p is not None else None
    return {"true_twin": f(true_twin), "true_victim": f(true_victim),
            "true_gap_cm": tc, "true_gap_deg": td,
            "twin_est_world": f(twin_est_world), "attacker_world": f(attacker_world),
            "est_gap_cm": ec, "est_gap_deg": ed}


def velocity_from_poses(buf, span_s=VEL_SPAN_S):
    """(vx, vy, wz, n) by least squares on the (stamp, x, y, yaw) poses of the
    last span_s seconds; None with fewer than two poses."""
    if not buf:
        return None
    t_end = buf[-1][0]
    pts = [p for p in buf if p[0] >= t_end - span_s]
    if len(pts) < 2 or pts[-1][0] - pts[0][0] <= 0:
        return None
    t0 = pts[0][0]
    ts = [p[0] - t0 for p in pts]
    tm = sum(ts) / len(ts)
    den = sum((t - tm) ** 2 for t in ts)
    yaws = [pts[0][3]]
    for p in pts[1:]:
        d = p[3] - yaws[-1]
        yaws.append(yaws[-1] + math.atan2(math.sin(d), math.cos(d)))

    def slope(vals):
        vm = sum(vals) / len(vals)
        return sum((t - tm) * (v - vm) for t, v in zip(ts, vals)) / den
    return slope([p[1] for p in pts]), slope([p[2] for p in pts]), slope(yaws), len(pts)


def twin_init_target(saved, saved_scene, believed, vel, vmin=TWIN_INIT_VMIN, wmin=TWIN_INIT_WMIN,
                     vel_mode="poses"):
    """Twin state in Isaac from the attacker's estimate.

    saved: true pose in the odom frame at window start; saved_scene: the same in
    the scene frame (save reply); believed: pose believed by the attacker (odom
    frame); vel: (vx, vy, wz) in the odom frame or None. The scene pose is
    saved_scene composed with the offset saved -> believed. vel_mode "zero":
    velocity always zero (the estimate is logged); "poses": zero if
    |v| <= vmin and |w| <= wmin or missing, otherwise the estimate."""
    if vel_mode == "zero" and vel is not None:
        rec = twin_init_target(saved, saved_scene, believed, None, vmin, wmin)
        rec.update({"v_att": [float(vel[0]), float(vel[1])], "w_att": float(vel[2]),
                    "rule": "zero per scelta (--twin-init-vel zero)"})
        return rec
    rel = compose_pose(pose_inverse(saved), believed)
    x, y, yaw = compose_pose(saved_scene, rel)
    rot = saved_scene[2] - saved[2]
    rec = {"target_odom": [float(v) for v in believed], "target_scene": [x, y, yaw],
           "v_att": None, "w_att": None, "vmin": vmin, "wmin": wmin}
    vx = vy = wz = 0.0
    rule = "velocita' assente: zero"
    if vel is not None:
        rec["v_att"] = [float(vel[0]), float(vel[1])]
        rec["w_att"] = float(vel[2])
        if math.hypot(vel[0], vel[1]) <= vmin and abs(vel[2]) <= wmin:
            rule = "robot fermo: zero"
        else:
            c, s = math.cos(rot), math.sin(rot)
            vx, vy, wz = c * vel[0] - s * vel[1], s * vel[0] + c * vel[1], float(vel[2])
            rule = "da pose consecutive della MOLA dell'attaccante"
    # Joint velocities always zeroed: otherwise the twin would keep the victim's
    # true wheel speeds inherited from the restore, unknown to the attacker.
    # Joint positions stay those of the restore.
    rec.update({"v_set_scene": [vx, vy], "w_set": wz, "rule": rule,
                "zero_joints": True})
    return rec


# ---------------------------------------------------------------------------
# Variant E: twin state from the shadow
# ---------------------------------------------------------------------------
#
# Frames. The shadow starts with the victim, robot still: its MOLA frame S has
# its origin at the robot pose of the first scan, like the victim's. The map
# saved by the shadow has its keyframes in S and its last pose p_S is in S.
# Twin E loads that map and starts from MOLA_INITIAL_* = p_S, so its MOLA frame
# is S and p_S is passed unchanged. The Isaac frame does not enter MOLA: the
# twin's physical pose is the restored one (--twin-init). The working frame
# (Isaac odom) is used only by the controller: believed pose = shadow origin
# composed with p_S, and the twin estimate is read relative to p_S
# (est_origin), as the clean twin's is relative to (0, 0, 0).

def pose6_from_quat(x, y, z, qx, qy, qz, qw):
    """(x, y, z, yaw, pitch, roll) from a quaternion, angles in rad (ZYX Euler
    angles, the CPose3D convention)."""
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    sp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sp)
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    return (float(x), float(y), float(z), yaw, pitch, roll)


def e_initial_env(mm_path, pose6, sigma):
    """Environment variables of twin E. Angles in deg
    (LidarOdometry_Parameters.cpp:333-336), fixed-point values without
    exponent notation. Without sigma the pipeline value (or --initial-sigma)
    is kept."""
    x, y, z, yaw, pitch, roll = pose6
    env = {"MOLA_LOAD_MM": str(mm_path),
           "MOLA_INITIAL_X": f"{x:.9f}", "MOLA_INITIAL_Y": f"{y:.9f}", "MOLA_INITIAL_Z": f"{z:.9f}",
           "MOLA_INITIAL_YAW": f"{math.degrees(yaw):.9f}",
           "MOLA_INITIAL_PITCH": f"{math.degrees(pitch):.9f}",
           "MOLA_INITIAL_ROLL": f"{math.degrees(roll):.9f}"}
    if sigma is not None:
        env["MOLA_INITIAL_SIGMA"] = f"{float(sigma):.6f}"
    return env


def mm_problem(path):
    """Reason the .mm cannot be loaded, None if it exists and is not empty.
    With a missing file MOLA stops on an ASSERT (Initialize.cpp:560-565)."""
    if not path:
        return "percorso del .mm assente"
    if not os.path.isfile(path):
        return f"{path} non esiste"
    if os.path.getsize(path) == 0:
        return f"{path} vuoto"
    return None


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def shadow_sigma_from_log(log_path, pose_stamp, tol=SHADOW_SIGMA_TOL_S):
    """Shadow adaptive sigma from the last printf line with stamp <= the last
    pose stamp (+ tol). The printf has 4 decimals (ProcessScan.cpp:1293-1296)
    and leggi_printf rounds stamps to 0.1 s. sigma_aligned: the line belongs to
    the same scan as the pose. Without lines sigma is None."""
    out = {"sigma": None, "sigma_stamp": None, "sigma_pathstep": None, "sigma_rows": 0,
           "sigma_aligned": False, "sigma_source": "printf"}
    try:
        rows = leggi_printf(str(log_path))
    except (OSError, TypeError):
        return out
    out["sigma_rows"] = len(rows)
    if pose_stamp is not None and pose_stamp == pose_stamp:
        rows = [r for r in rows if r["stamp_s"] <= pose_stamp + tol]
    if not rows:
        return out
    r = max(rows, key=lambda r: (r["stamp_s"], r["riga"]))
    out.update(sigma=r["sigma"], sigma_stamp=r["stamp_s"], sigma_pathstep=r["pathstep"],
               sigma_aligned=bool(pose_stamp is not None and pose_stamp == pose_stamp
                                  and abs(r["stamp_s"] - pose_stamp) <= tol))
    return out


def e_fingerprint(shadow_state):
    """Fingerprint of a candidate run with the shadow state: source, .mm hash,
    loaded pose and sigma, fields that remain different."""
    st = shadow_state or {}
    return {"source": "ombra_e", "mm_path": st.get("mm_path"), "mm_sha256": st.get("mm_sha256"),
            "mm_bytes": st.get("mm_bytes"), "pose_loaded": st.get("pose6"),
            "pose_stamp": st.get("pose_stamp"), "sigma_loaded": st.get("sigma"),
            "sigma_aligned": st.get("sigma_aligned"), "env": st.get("env"),
            "campi_diversi": dict(E_FIELDS_DIFFERENT)}


# --- Variant G -------------------------------------------------------------
# MOLA overlay with full-state save/load. Only shadow and twin use it; the
# victim stays on the /opt libraries.
G_OVERLAY_ROOT = os.path.expanduser("~/mola_ws/install_G1")
G_OVERLAY_PKGS = ("mola_lidar_odometry", "mola_state_estimation_simple")
# Overlay log lines (mola_lidar_odometry/module/src/LidarOdometry_FullState.cpp):
# state loaded (:1093-1094) and errors (:1184, :1212, :1231, :1243, :1250). A
# loaded twin skips the initial relocalization and does not print
# "Initial re-localization done".
G_READY_MARKER = "[full_state] caricato"
G_ERROR_MARKERS = ("[full_state] caricamento fallito", "[full_state] salvataggio fallito",
                   "[full_state] file di stato assente", "[full_state] traccia:",
                   "stato completo:")
RELOC_MARKER = "Initial re-localization done"
# .molastate header (LidarOdometry_FullState.cpp:97-99, :1120-1128): magic,
# u32 version, u8 has-stamp flag, int64 stamp in 100 ns ticks.
MOLASTATE_MAGIC = b"MOLAFS01"
MOLASTATE_VERSION = 2
# mrpt::Clock: 100 ns ticks since 1601. fromDouble(t) = trunc(t * 1e7) +
# 116444736000000000 (disassembly of libmrpt-core.so.2.15.21, 0x6e30), and
# ros2bridge::fromROS(Time) = fromDouble(Time::seconds())
# (libmrpt_libros_bridge.so, 0x1bfe0).
MRPT_EPOCH_TICKS = 116444736000000000
# First twin scan = saved stamp + 0.1 s
TWIN_CLOCK_STEP_TICKS = 1_000_000
RX_G_LOADED = re.compile(r"\[full_state\] caricato (\S+) \((\d+) byte\)")
RX_G_SAVED = re.compile(r"\[full_state\] salvato (\S+) \((\d+) byte, stamp (\d+|nessuno)\)")


def g_overlay_dirs(root=None):
    root = root or G_OVERLAY_ROOT
    return [os.path.join(root, p, "lib", "x86_64-linux-gnu") for p in G_OVERLAY_PKGS]


def g_overlay_problem(root=None):
    """Reason the overlay is unusable, None if both library folders exist."""
    for d in g_overlay_dirs(root):
        if not os.path.isdir(d):
            return f"overlay di MOLA assente: {d}"
    return None


def g_overlay_env(environ=None, root=None):
    """LD_LIBRARY_PATH and MOLA_MODULES_LIB_PATH with the overlay first (the
    inherited environment already holds /opt)."""
    environ = os.environ if environ is None else environ
    ovl = ":".join(g_overlay_dirs(root))
    out = {}
    for k in ("LD_LIBRARY_PATH", "MOLA_MODULES_LIB_PATH"):
        old = environ.get(k, "")
        out[k] = f"{ovl}:{old}" if old else ovl
    return out


def g_shadow_env(environ=None, root=None):
    """Extra environment of shadow G: overlay and full-state save at every
    map_save (<path>.molastate). No SAVE_AT_TICKS: live saves go through the
    service."""
    return {**g_overlay_env(environ, root), "MOLA_FULL_STATE": "1"}


def g_twin_env(state_path, resave_path=None, environ=None, root=None):
    """Environment of twin G: overlay, load of the shadow state and, if given,
    a re-save right after loading (hash chain). No variant E variables
    (MOLA_LOAD_MM, MOLA_INITIAL_*)."""
    env = {**g_overlay_env(environ, root), "MOLA_FULL_STATE": "1",
           "MOLA_FULL_STATE_LOAD": str(state_path)}
    if resave_path:
        env["MOLA_FULL_STATE_RESAVE"] = str(resave_path)
    return env


def molastate_header(path):
    """.molastate header: magic, version, stamp flag and value (mrpt::Clock
    ticks). ValueError if the file is short or not a state file."""
    with open(path, "rb") as f:
        b = f.read(8 + 4 + 1 + 8)
    if len(b) < 13:
        raise ValueError(f"{path}: intestazione troncata ({len(b)} byte)")
    if b[:8] != MOLASTATE_MAGIC:
        raise ValueError(f"{path}: magic {b[:8]!r} non riconosciuto")
    ver = int.from_bytes(b[8:12], "little")
    has = b[12] != 0
    if has and len(b) < 21:
        raise ValueError(f"{path}: stamp troncato")
    stamp = int.from_bytes(b[13:21], "little", signed=True) if has else None
    return {"magic": b[:8].decode(), "version": ver, "has_stamp": has, "stamp_tick": stamp}


def g_state_problem(path, version=MOLASTATE_VERSION):
    """Reason the twin cannot load the .molastate, None if it can. Without a
    stamp or with another version the overlay rejects it (FullState.cpp:1120-1127)."""
    if not path:
        return "percorso del .molastate assente"
    if not os.path.isfile(path):
        return f"{path} non esiste"
    if os.path.getsize(path) == 0:
        return f"{path} vuoto"
    try:
        h = molastate_header(path)
    except (OSError, ValueError) as e:
        return str(e)
    if h["version"] != version:
        return f"{path}: versione {h['version']}, attesa {version}"
    if not h["has_stamp"]:
        return f"{path}: senza stamp"
    return None


def ros_stamp_to_tick(sec, nanosec):
    """mrpt::Clock tick that MOLA assigns to a ROS stamp (BridgeROS2.cpp:635,
    fromROS(header.stamp); rclcpp::Time in integer ns, seconds() = ns / 1e9)."""
    ns = int(sec) * 1_000_000_000 + int(nanosec)
    return int((ns / 1e9) * 1e7) + MRPT_EPOCH_TICKS


def stamp_s_to_tick(stamp_s):
    """ros_stamp_to_tick for a stamp already in seconds (float)."""
    if stamp_s is None or stamp_s != stamp_s:
        return None
    ns = int(round(float(stamp_s) * 1e9))
    return ros_stamp_to_tick(ns // 1_000_000_000, ns % 1_000_000_000)


def twin_clock_first_ns(saved_tick, step_ticks=TWIN_CLOCK_STEP_TICKS):
    """Integer-ns stamp of the first twin G scan that MOLA converts exactly to
    saved_tick + step_ticks. Starts from the tick centre (+50 ns), away from
    the truncation edges, and checks the result."""
    target = int(saved_tick) + int(step_ticks)
    base = (target - MRPT_EPOCH_TICKS) * 100
    for d in (50, 0, 99, 25, 75):
        ns = base + d
        if ns >= 0 and ros_stamp_to_tick(ns // 1_000_000_000, ns % 1_000_000_000) == target:
            return ns
    raise ValueError(f"nessuno stamp ROS da {target} tick")


def g_log_signature(text):
    """Twin G signatures in its log: load lines, initial relocalization and
    overlay errors, loaded and re-saved file."""
    text = text or ""
    loaded = RX_G_LOADED.findall(text)
    saved = RX_G_SAVED.findall(text)
    return {"n_loaded": len(loaded), "n_reloc": text.count(RELOC_MARKER),
            "n_errors": sum(text.count(m) for m in G_ERROR_MARKERS),
            "loaded_path": loaded[0][0] if loaded else None,
            "loaded_bytes": int(loaded[0][1]) if loaded else None,
            "resaved_path": saved[0][0] if saved else None,
            "resaved_stamp_tick": (int(saved[0][2]) if saved and saved[0][2].isdigit() else None)}


# Per-scan lines with MOLA_LO_DEBUG_ICP_QUALITY=1 and totals printed when MOLA
# exits ("Run totals: registrations=... icp_rejected=...").
RX_G_ICP = re.compile(r"isGood=([01]) .*consecutive_bad=(-?\d+)")
RX_G_TOTALS = re.compile(r"Run totals: registrations=(\d+) no_motion_model=(\d+).*?icp_rejected=(\d+)")


def g_twin_lost_lines(text):
    """Twin G MOLA lines useful when the twin publishes no pose: state loads,
    accepted and rejected ICPs, consecutive_bad (last and max), end-of-run
    totals. None where the lines are missing."""
    text = text or ""
    icp = RX_G_ICP.findall(text)
    tot = RX_G_TOTALS.findall(text)
    bad = [int(c) for _, c in icp]
    return {"n_loaded": len(RX_G_LOADED.findall(text)),
            "n_icp_lines": len(icp),
            "n_icp_good": sum(1 for g, _ in icp if g == "1"),
            "n_icp_bad": sum(1 for g, _ in icp if g == "0"),
            "consecutive_bad_last": bad[-1] if bad else None,
            "consecutive_bad_max": max(bad) if bad else None,
            "registrations": int(tot[-1][0]) if tot else None,
            "no_motion_model": int(tot[-1][1]) if tot else None,
            "icp_rejected": int(tot[-1][2]) if tot else None}


def g_twin_lost(status, mode, lines):
    """True if twin G loaded the shadow state but does not localize: nominal
    failed with no_pose_from_new_mola and the load line is in the log."""
    return (mode == "ombra_g" and str(status or "").startswith("no_pose_from_new_mola")
            and bool(lines) and (lines.get("n_loaded") or 0) > 0)


def g_twin_lost_record(status, lines, log_path=None, attempts=2):
    """victim_lost_g history entry: cause, nominal outcome and MOLA lines. The
    log is that of the last attempt: the same tag overwrites the first one."""
    return {"cause": "nominale senza posa dopo [full_state] caricato: il gemello G non localizza",
            "nominal_status": str(status), "nominal_attempts": int(attempts),
            "mola_log": str(log_path) if log_path else None,
            "mola_log_attempt": "ultimo", "mola_lines": dict(lines or {})}


# goodness and isGood of the per-scan line (MOLA_LO_DEBUG_ICP_QUALITY=1)
RX_G_Q = re.compile(r"timestamp=(\S+) goodness=([-+.\deE]+) .*?isGood=([01])")


def g_nominale_non_localizza_record(window, status, text, log_path=None, attempts=2, t_wall=None):
    """nominale_non_localizza history entry (--g-nominale-senza-posa continua):
    the twin G nominal loaded the shadow state but publishes no pose. q = ICP
    goodness after the last load line, from the last attempt's log; the window
    goes on with the candidates."""
    text = text or ""
    i = text.rfind(G_READY_MARKER)
    righe = RX_G_Q.findall(text[i:] if i >= 0 else "")
    q = [float(g) for _, g, _ in righe]
    return {"window": int(window), "status": str(status), "t_wall": t_wall,
            "nominal_attempts": int(attempts),
            "q_n": len(q), "q_good": sum(1 for _, _, b in righe if b == "1"),
            "q_max": max(q) if q else None,
            "q_mean": round(sum(q) / len(q), 4) if q else None,
            "q_primi": q[:10],
            "stamp_primo": righe[0][0] if righe else None,
            "stamp_ultimo": righe[-1][0] if righe else None,
            "mola_log": str(log_path) if log_path else None,
            "mola_lines": g_twin_lost_lines(text)}


def g_nominale_senza_posa(args, status, nom_end, log_path, window, attempts, t_wall=None):
    """nominale_non_localizza entry if --g-nominale-senza-posa continua and the
    nominal failed because the loaded twin G publishes no pose; None otherwise
    (normal window flow). Also lists which nominal references the candidates
    lack. With --no-nominal-rollout the nominal is not run and the entry is
    nominale_assente (senza_nominale_record)."""
    if getattr(args, "no_nominal_rollout", False):
        return senza_nominale_record(args, window, t_wall)
    if getattr(args, "g_nominale_senza_posa", "trattieni") != "continua":
        return None
    if nom_end is not None and status in ("ok", "arrived"):
        return None
    try:
        text = open(log_path, errors="replace").read() if log_path else ""
    except OSError:
        text = ""
    if not g_twin_lost(status, getattr(args, "twin_state", "clean"), g_twin_lost_lines(text)):
        return None
    rec = g_nominale_non_localizza_record(window, status, text, log_path, attempts, t_wall)
    interna = getattr(args, "perceptibility", "internal") == "internal"
    rec["riferimenti_non_disponibili"] = {
        "f2_interna": "non definita" if interna else "non usata",
        "nominal_ok": "non disponibile, candidati non bloccati",
        "track_ref": "non disponibile, untracked sul rapporto del candidato",
        "settle_fixed": ("non usato (--settle-clouds)" if (getattr(args, "settle_clouds", None) or 0) > 0
                         else "non disponibile")}
    return rec


# --no-nominal-rollout: tracking reference of the candidates. The untracked
# threshold (0.60) is absolute on the candidate's estimate/truth ratio; the
# reference only needs to be non-None.
TRACK_REF_SENZA_NOMINALE = 1.0
ESITI_GEMELLO = ("controlli_ok", "gemello_perde_posa", "guasto_gemello")


def senza_nominale_record(args, window, t_wall=None):
    """nominale_assente history entry (--no-nominal-rollout): no twin nominal
    in the window and the references used instead."""
    return {"window": int(window), "senza_nominale": True, "t_wall": t_wall,
            "riferimenti": {
                "danno": "baseline (--damage-ref baseline)",
                "track_ref": TRACK_REF_SENZA_NOMINALE,
                "nominal_ok": "vero: nessuna misura a nuvola pulita",
                "settle_fixed": f"non usato (--settle-clouds {getattr(args, 'settle_clouds', None)})",
                "f2_interna": f"non usata (--perceptibility {getattr(args, 'perceptibility', None)})",
                "controllo_frame": "posa vera salvata a inizio w0 contro la polilinea, un punto"}}


def esito_gemello_g(mode, status, fp):
    """Twin G outcome of a candidate from its rollout checks (g_rollout_check:
    signatures, hash, clock, loaded file), without a nominal. guasto_gemello:
    checks missing or failed; gemello_perde_posa: checks passed but no pose
    from the new MOLA (a MOLA outcome, logged separately); controlli_ok
    otherwise. None if the twin is not G (windows before the shadow starts,
    other variants)."""
    if mode != "ombra_g":
        return None
    chk = fp.get("g_check") if isinstance(fp, dict) else None
    if not isinstance(chk, dict) or chk.get("ok") is not True:
        return "guasto_gemello"
    if str(status or "").startswith("no_pose_from_new_mola"):
        return "gemello_perde_posa"
    return "controlli_ok"


def conta_esiti_gemello(evaluations):
    """Window candidates counted by twin G outcome (esito_gemello field)."""
    n = {e: 0 for e in ESITI_GEMELLO}
    for ev in evaluations or []:
        e = ev.get("esito_gemello") if isinstance(ev, dict) else None
        if e in n:
            n[e] += 1
    return n


# Local-map publication fields of the full state (overlay
# LidarOdometry_FullState.cpp, section 1): the spinOnce thread can update them
# between load and re-save (seen in 2 of 392 cases, counter only), so the hash
# comparison ignores them.
FS_MAGIC = b"MOLAFS01"
FS_VERSIONI = (2,)
_FS_POSA = 3 * 8 + 9 * 8 + 1 + 3 * 8          # Scrittore::posa
_FS_PDF = _FS_POSA + 36 * 8                   # posaPdf / posaPdfInf (6x6)


def campi_pubblicazione_mappa(dati):
    """Offsets of localmap_advertise_updates_counter (uint32) and
    local_map_needs_publish (uint8) in a .molastate, found by following the
    write order of fullStateSerialize (section 1). Raises if the format is
    not the expected one."""
    b = bytes(dati)
    p = 0

    def leggi(fmt):
        nonlocal p
        v = struct.unpack_from(fmt, b, p)
        p += struct.calcsize(fmt)
        return v[0]

    def salta(n):
        nonlocal p
        if p + n > len(b):
            raise ValueError("stato completo troncato")
        p += n

    if b[:len(FS_MAGIC)] != FS_MAGIC:
        raise ValueError("stato completo: magic diverso")
    p = len(FS_MAGIC)
    ver = leggi("<I")
    if ver not in FS_VERSIONI:
        raise ValueError(f"stato completo: versione {ver} non prevista")
    salta(1 + 8)                                  # has-stamp flag, tick
    if leggi("<I") != 0x53450001:
        raise ValueError("stato completo: segnaposto della sezione 1 assente")
    leggi("<B")                                   # initial_localization_done
    if leggi("<B"):
        salta(16)                                 # gravity_calib_pitch_roll
    if leggi("<B"):
        salta(8)                                  # gravity_calib_first_available_time
    if leggi("<B"):
        salta(_FS_POSA)                           # gravity_calib_pose
    salta(_FS_PDF)                                # last_lidar_pose
    for _ in range(leggi("<Q")):                  # last_obs_tim_by_label
        salta(leggi("<Q"))
        salta(8)
    for _ in range(leggi("<Q")):                  # last_lidar_sensor_poses
        salta(leggi("<Q"))
        salta(_FS_POSA)
    salta(1 + 8 + 8)                              # last_icp_was_good, quality, iterations
    for _ in range(3):                            # first_ever, last_obs, last_icp (tempoOpt)
        if leggi("<B"):
            salta(8)
    salta(8)                                      # last_observed_scan_period_sec
    if leggi("<B"):
        salta(_FS_PDF + 6 * 8 + 36 * 8)           # last_motion_model_output
    salta(8 + 4 + 8 * 3)                          # adapt_thres_sigma ... registration_icp_rejected
    for _ in range(2):                            # estimated/instantaneous_observation_radius
        if leggi("<B"):
            salta(8)
    salta(1)                                      # map_has_been_loaded
    salta(4)                                      # localmap_check_removal_counter
    contatore = p
    return {"versione": ver, "advertise_counter": (contatore, 4), "needs_publish": (contatore + 5, 1)}


def confronto_stato_mascherato(path_a, path_b):
    """Byte-by-byte comparison of two .molastate files ignoring the map
    publication fields. uguale is True if everything else matches;
    byte_diversi lists the differing offsets outside the ignored fields."""
    a = open(path_a, "rb").read()
    b = open(path_b, "rb").read()
    out = {"campi_ignorati": None, "byte_diversi_ignorati": [], "byte_diversi": [], "uguale": False}
    if len(a) != len(b):
        out["motivo"] = f"dimensioni diverse ({len(a)} contro {len(b)})"
        return out
    try:
        ca, cb = campi_pubblicazione_mappa(a), campi_pubblicazione_mappa(b)
    except (ValueError, struct.error) as e:
        out["motivo"] = str(e)
        return out
    if ca != cb:
        out["motivo"] = "campi di pubblicazione in posizioni diverse"
        return out
    ign = set()
    for off, n in (ca["advertise_counter"], ca["needs_publish"]):
        ign.update(range(off, off + n))
    out["campi_ignorati"] = {k: list(v) for k, v in ca.items() if k != "versione"}
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            (out["byte_diversi_ignorati"] if i in ign else out["byte_diversi"]).append(i)
    out["byte_diversi"] = out["byte_diversi"][:20]
    out["uguale"] = not out["byte_diversi"]
    return out


def g_rollout_check(log_text, state, resave_path=None, first_ns=None):
    """End-of-rollout checks of twin G: one load line, no initial
    relocalization and no error; sha256 of the re-saved file equal to the
    shadow file's; first stamp published to the twin = saved stamp + 0.1 s.
    ok only if all pass; checks that cannot run (re-save off) are None."""
    st = state or {}
    sig = g_log_signature(log_text)
    out = {"firme": sig,
           "firme_ok": sig["n_loaded"] == 1 and sig["n_reloc"] == 0 and sig["n_errors"] == 0,
           "loaded_is_shadow_file": (sig["loaded_path"] == st.get("state_path")
                                     if sig["loaded_path"] is not None else False)}
    if resave_path:
        if os.path.isfile(resave_path):
            sha = file_sha256(resave_path)
            esatto = sha == st.get("state_sha256")
            out.update(resave_sha256=sha, resave_bytes=os.path.getsize(resave_path),
                       hash_esatto_ok=esatto, hash_ok=esatto)
            if (not esatto and st.get("state_path") and os.path.isfile(st["state_path"])
                    and file_sha256(st["state_path"]) == st.get("state_sha256")):
                # equal except for the map publication fields
                m = confronto_stato_mascherato(st["state_path"], resave_path)
                out.update(hash_mascherato=m, hash_ok=bool(m["uguale"]))
        else:
            out.update(resave_sha256=None, hash_ok=False)
    else:
        out["hash_ok"] = None
    saved_tick = st.get("state_stamp_tick")
    if first_ns is not None and saved_tick is not None:
        ft = ros_stamp_to_tick(int(first_ns) // 1_000_000_000, int(first_ns) % 1_000_000_000)
        out.update(first_stamp_tick=ft, first_minus_saved_ticks=ft - int(saved_tick),
                   clock_ok=(ft - int(saved_tick) == TWIN_CLOCK_STEP_TICKS))
    else:
        out.update(first_stamp_tick=None, clock_ok=None)
    out["ok"] = bool(out["firme_ok"] and out["loaded_is_shadow_file"]
                     and out["hash_ok"] is not False and out["clock_ok"] is True)
    return out


def g_fingerprint(shadow_state):
    """Fingerprint of a candidate run with the full shadow state: file, hash,
    bytes and stamp; the shadow pose is stored only as frame reference."""
    st = shadow_state or {}
    return {"source": "ombra_g", "state_path": st.get("state_path"),
            "state_sha256": st.get("state_sha256"), "state_bytes": st.get("state_bytes"),
            "state_stamp_tick": st.get("state_stamp_tick"), "pose_loaded": st.get("pose6"),
            "pose_stamp": st.get("pose_stamp"), "env": st.get("env")}
