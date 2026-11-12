#!/usr/bin/env python3
"""Receding-horizon adversarial attack orchestrator.

For every window of H metres: save the Isaac Sim state, evaluate N candidates with a physical
rollout (state restore, MOLA restart, genome load, H metres of driving, final pose and
perceptibility measurement), rank them with NSGA-III, pick one from the Pareto front and apply the
winner for a real stretch.

Fitness is measured by the rollout, not estimated with a kinematic model: the state restore is
faithful, so no assumption on future motion is needed. The damage of a candidate is the deviation of
its true trajectory from the nominal rollout of the same window, compared at equal travelled
distance (--damage mean|max|end), plus the increase of the distance to the goal. With --damage-ref
baseline the reference is instead a fixed polyline in the odom frame (--baseline-path, CSV
s_m,x_m,y_m) and the damage is the distance of the true poses from the closest point of the
polyline; the per-window nominal still runs but does not enter the damage (it stays in the *_window
fields).
MOLA has no reset service and is restarted for every candidate, otherwise the local map and adaptive
sigma of the previous candidate would bias the comparison. At every restart MOLA relocalises at
(0,0,0) in the current robot pose, so the waypoint is re-expressed in that frame from the true pose.

Usage:
    python3 -u src/optimization/attack_orchestrator.py --goal "5,0" --horizon 1.0 \
        --settle-sec 2.0 --pop 4 --gen 2 --trace --out data/attack/<name>

Requires Isaac Sim in Play with isaac/isaac_rollout_server.py loaded in the Script Editor, and
`python3 src/nodes/add_intensity_node.py ...` and `python3 src/nodes/perturbation_node.py` running.

Reads /lidar_odometry/pose, /lidar_odometry/pose_quality and /attack/status; publishes /cmd_vel,
/attack/genome and /attack/enabled. Talks to Isaac Sim through /tmp/isaac_cmd.json and
/tmp/isaac_reply.json. Writes history.json, logs/ and traces/ in the --out directory.
"""

import argparse
import json
import math
from collections import deque
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import PerturbationGenerator, MAX_DROPOUT_RATE  # noqa: E402
from src.perturbations.perturbation_generator import moltiplicatori_da_opzioni  # noqa: E402
from src.optimization.trajectory_predictor import FollowerModel, angle_diff  # noqa: E402
from src.analysis.path_deviation import path_deviation, true_path  # noqa: E402
from src.analysis.baseline_distance import load_baseline, baseline_damage, dist_to_polyline  # noqa: E402
from src.analysis.fronte_pareto import fronte_pareto, candidato_valido, motivi_non_valido  # noqa: E402
from src.optimization import window_stop as wstop  # noqa: E402
from src.optimization import window_record as wrec  # noqa: E402
from src.optimization import twin_variants as tv  # noqa: E402
from src.optimization import shadow_check as shc  # noqa: E402
from src.optimization import banco_g  # noqa: E402
from src.perturbations import zone_attaccante as za  # noqa: E402

CMD_FILE = "/tmp/isaac_cmd.json"
REPLY_FILE = "/tmp/isaac_reply.json"
POSE_FILE = "/tmp/isaac_pose.json"

# MOLA-estimated distance within which a waypoint counts as reached (m).
ARRIVAL_TOL = 0.30
# --goal-mode direction: distance of the carrot along the goal line (m), the same as the
# straight-line goal (run_campaign.sh, GOALS="5,0").
DIRECTION_LOOKAHEAD = 5.0
# Minimum fraction of the horizon travelled for a rollout to be valid.
MIN_TRAVEL_FRAC = 0.70
# Seconds without a pose from the Isaac loop or without perturbed clouds (while the simulation is
# running) after which Isaac is considered stalled.
STALL_SEC = 5.0
# Grace after SIGTERM before SIGKILL of mola-cli. MOLA never exits within the 12 s (measured 12.21
# +- 0.01 s: it always expires), but the 12 s are not wasted: with 2 s the killed MOLA stays in the
# DDS graph as a stale subscriber (liveliness lease ~10 s), the RELIABLE publisher of
# perturbation_node blocks on the full queue and clouds stop arriving ("no cloud" stalls within 1-3
# min). Do not go below the DDS lease without changing the node QoS.
MOLA_STOP_GRACE_SEC = 12.0
# MOLA shutdown signal (--mola-stop-signal): "term" by default; "int" is the only signal for which
# mola-cli installs a handler (mola-cli.cpp:51-60). The maximum wait stays MOLA_STOP_GRACE_SEC.
MOLA_STOP = {"signal": "term"}


class IsaacStall(RuntimeError):
    """Isaac loop stalled or clouds missing: the run cannot continue."""


class GuastoEtaNuvola(IsaacStall):
    """--controllo-eta: a cloud taken by the victim MOLA older than the age threshold, or never
    taken, in the measured stretch.
    Chain failure: the run stops as for an Isaac stall. The twin does not raise it (rollout
    repeated, then outcome guasto_eta_ripetuto).
    """


def scrivi_da_rifare_ombra(out, k, motivo, seme):
    """Write da_rifare.json for a run stopped by OmbraPersa: the queue repeats it as guasto_catena
    (coda_serieB_stato.storia reads the file).
    """
    rec = {"reason": "guasto_catena", "window": k, "status": "ombra_persa", "causa": "ombra_persa",
           "motivi": [motivo], "seme": seme, "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(Path(out) / "da_rifare.json", "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    return rec


class OmbraPersa(IsaacStall):
    """--controllo-ombra-viva: the shadow MOLA process terminated while the run was in progress
    (e.g. a crash of the mola-cli ROS bridge). Without the shadow the state for realignment
    cannot be saved: the run stops at once as a chain failure (da_rifare.json) instead of at the
    window realignment.
    """


# Outcome of a twin rollout whose age failure persists in the repeat: excluded from fitness,
# archive, selection and re-evaluations; the search goes on.
ESITO_ETA_RIPETUTO = "guasto_eta_ripetuto"


# --controllo-eta: threshold in s, None = off. frame_log: per-frame log of the node
# (--frame-log-nodo); registro: one JSON line per rollout in logs/controllo_eta.jsonl.
CONTROLLO_ETA = {"soglia": None, "frame_log": None, "registro": None, "finestre": {}}
# --dds-profile: Fast DDS XML profile given to all MOLA instances (absolute path, md5).
DDS_PROFILO = {"path": None, "md5": None}
# File that Fast DDS 2.14.6 loads from the working directory, in addition to the env variable.
DDS_FILE_CARTELLA = "DEFAULT_FASTRTPS_PROFILES.xml"


# Rollout outcomes. VALID: stretch completed. COUNTABLE: stretch not completed but measured (the
# robot moved and there is a final pose), counts as damage. Everything else is an infrastructure
# failure: always discarded.
VALID = ("ok", "arrived")
COUNTABLE = ("timeout", "stopped_early", "untracked", "est_capped")

# Motion class of a rollout (motion_class field in the history).
# A stop (v < 0.05 m/s and |w| > 0.3 rad/s for 10 ticks) after which the robot restarts and travels
# at least STOP_RESUME_M is the follower settling rotation (initial heading error); a stop with no
# restart, with accumulated temporal drift above threshold during the still phase, is drift-induced
# paralysis (in spawn mode the drift is already there at the stop, in motion mode it grows while the
# robot is still); the rest is "arresto_altro". Threshold from data: paralysis with |d_xy| at the
# stop p10 = 22 cm, min 11; below 20 cm the median damage is ~1 cm.
PARALYSIS_DRIFT_CM = 20.0
STOP_V, STOP_W, STOP_TICKS = 0.05, 0.3, 10
STOP_RESUME_M = 0.30

# Stealth admissibility: a candidate is admissible only if the attack is a diversion, not a denial
# of service. A stopped robot or a MOLA that loses the pose are separate outcomes (visible,
# fail-safe, obtainable by just not publishing clouds): they are measured and reported but do not
# compete as damage.
# Stealth (two conditions only): the victim publishes poses for the whole stretch and the mean ICP
# quality stays above STEALTH_MIN_QUALITY, the threshold MOLA actually acts on (not the 0.40
# setpoint of a controller). Denial of service (DoS), infeasible: the robot does not move (rotation
# in place or missing pose). No filter on the estimate/true ratio or on travelled fraction: a
# deceived robot that stops early has by definition travelled less with a high ratio (MOLA
# overestimates: that is the deception). The ratio stays a damage metric to report, not a filter: in
# this threat model (LiDAR-only MOLA, no IMU or wheel odometry) nothing in the system can notice it.
STEALTH_MIN_QUALITY = 0.20
DOS_TRAVEL_M = 0.10        # below this net displacement the robot "does not move"

# Stop on the estimate: cap on the true distance, in multiples of the horizon. The MOLA estimate
# lags by 0.3-0.5 s (0.25-0.4 m at 0.8 m/s) and stays almost still in the first half metre after the
# twin starts: with 1.5 even nominals ended `est_capped`. The cap closes rollouts where the estimate
# under-tracks a lot, it does not cut the latency.
class WindowTimeout(Exception):
    """The window exceeded the watchdog time cap: the search stops."""


class VictimTopicForeign(RuntimeError):
    """Publisher other than the perturbation node on the victim topic (--victim-never-clean):
    infrastructure outcome, the run stops.
    """


class ErroreConfigurazioneZone(RuntimeError):
    """--zone-attaccante: snapshot or zones invalid, not confirmed or not applied by a stream. Not
    an attack: the window is recorded as errore_configurazione and the run stops.
    """


class ErroreMoltiplicatori(ErroreConfigurazioneZone):
    """--moltiplicatore-limiti: node limit multiplier differs from the orchestrator one (zones and
    decoding would not match). Configuration error: the window is recorded as
    errore_configurazione and the run stops, as for the zones.
    """


class ShadowSyncFail(RuntimeError):
    """Synchronous start guard: first processed stamp differs between victim and shadow.
    Infrastructure outcome (shadow_sync_fail), the run stops.
    """


TRUE_CAP = 2.0
SLOW_TAU = 0.5                 # estimate delay (s) beyond which the rollout is in the slow regime
# With --damage-ref baseline: if the nominal of the first window is on average farther than this
# from the polyline, the frame does not match or the baseline is wrong, and the run stops before
# evaluating candidates.
BASELINE_FRAME_MAX_CM = 30.0
DIRECTION_GENES = (0, 1, 2, 7, 8, 9)   # direction genes: 'off' = zero vector, not -1
DRIVE_TIMEOUT = [45.0]        # set by --drive-timeout


def outcome_kind(status: str) -> str:
    if status in VALID:
        return "valid"
    if status.split("(")[0] in COUNTABLE:
        return "countable"
    return "infra"


# ---------------------------------------------------------------------------
# Isaac Sim side
# ---------------------------------------------------------------------------

class IsaacClient:
    """File-based communication with isaac/isaac_rollout_server.py.

    Files instead of ROS because rclpy is not available inside the Isaac Sim interpreter. The id
    field tells a new reply from a stale one left on disk.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._id = 0
        self.playing = True
        self.t_play = None         # last play issued by this client
        for f in (CMD_FILE, REPLY_FILE):
            if os.path.exists(f):
                os.remove(f)

    def call(self, cmd: str, **kw):
        self._id += 1
        payload = {"cmd": cmd, "id": self._id, **kw}
        if os.path.exists(REPLY_FILE):
            os.remove(REPLY_FILE)
        with open(CMD_FILE, "w") as f:
            json.dump(payload, f)

        # The server replies within one frame (step: n frames). A wait beyond STALL_SEC, or the loop
        # pose frozen for STALL_SEC during the wait, is a stall even if the loop restarts later: a
        # transient block while the orchestrator waits here would not be seen elsewhere.
        t0 = time.time()
        deadline = t0 + self.timeout
        while time.time() < deadline:
            if os.path.exists(REPLY_FILE):
                try:
                    with open(REPLY_FILE) as f:
                        r = json.load(f)
                except (OSError, ValueError):
                    time.sleep(0.02)
                    continue
                if r.get("id") == self._id:
                    if not r.get("ok"):
                        raise RuntimeError(
                            f"Isaac '{cmd}': {r.get('error')}\n{r.get('trace','')}")
                    waited = time.time() - t0
                    if cmd != "step" and waited > STALL_SEC:
                        raise IsaacStall(f"Isaac ha risposto a '{cmd}' dopo {waited:.1f} s")
                    return r
            age = time.time() - self._pose_file_wall()
            if age > STALL_SEC and time.time() - t0 > STALL_SEC:
                raise IsaacStall(f"il ciclo di Isaac non aggiorna {POSE_FILE} da {age:.1f} s "
                                 f"(in attesa della risposta a '{cmd}')")
            time.sleep(0.02)
        raise IsaacStall(
            f"Isaac non risponde a '{cmd}' da {self.timeout:.0f} s (server non in "
            f"esecuzione o ciclo di Isaac fermo)")

    @staticmethod
    def _pose_file_wall():
        try:
            with open(POSE_FILE) as f:
                return float(json.load(f).get("wall", 0.0))
        except (OSError, ValueError):
            return 0.0

    def save(self, nome=None):
        """Isaac snapshot; with a name, into the slot of that name (server with save_nome), without,
        into the single default slot (same message).
        """
        return self.call("save", **({"nome": str(nome)} if nome else {}))["pose"]
    def restore(self, nome=None):
        return self.call("restore", **({"nome": str(nome)} if nome else {}))["pose"]
    def pose(self):    return self.call("pose")["pose"]
    def set_pose(self, x, y, yaw):
        return self.call("set_pose", x=float(x), y=float(y), yaw=float(yaw))["pose"]
    def set_state(self, x, y, yaw, vx=0.0, vy=0.0, wz=0.0, zero_joints=True,
                  roll=None, pitch=None, keep_tilt=False):
        """Base pose and velocity in the scene (--twin-init attacker_est).
        roll and pitch (together) or keep_tilt only with --twin-init-assetto; without it the message
        carries yaw only.
        """
        kw = {}
        if roll is not None or pitch is not None:
            kw.update(roll=float(roll), pitch=float(pitch))
        if keep_tilt:
            kw["keep_tilt"] = True
        return self.call("set_state", x=float(x), y=float(y), yaw=float(yaw), vx=float(vx),
                         vy=float(vy), wz=float(wz), zero_joints=bool(zero_joints), **kw)["pose"]
    def assetto(self):
        """Server assetto command (read only, no frame)."""
        return self.call("assetto")
    def caps(self):
        """Optional server commands; empty for a server that does not know them."""
        try:
            return list(self.call("caps").get("caps", []))
        except RuntimeError:
            return []
    def play(self):
        self.call("play"); self.playing = True; self.t_play = time.time()
    def pause(self):   self.call("pause"); self.playing = False

    def pose_stream(self, max_age=0.5):
        """True pose from the file the server rewrites at every Isaac Sim frame.

        A read, not a request: it does not block the control loop and allows logging the true pose
        at every tick. If the file is missing or stale (older server, or stalled simulation) it
        falls back to the request.
        """
        p = self._read_stream(max_age)
        return p if p is not None else self.pose()

    def stream_available(self):
        return self._read_stream(0.5) is not None

    @staticmethod
    def _read_stream(max_age):
        try:
            with open(POSE_FILE) as f:
                r = json.load(f)
            if time.time() - float(r.get("wall", 0.0)) <= max_age:
                return r["pose"]
        except (OSError, ValueError, KeyError):
            pass
        return None


# ---------------------------------------------------------------------------
# MOLA side
# ---------------------------------------------------------------------------

class MolaRunner:
    """Starts and stops the MOLA instance under attack."""

    _live = set()      # instances started by this process, for the orphan check

    def __init__(self, script: str, log_dir: Path, ready_timeout: float = 40.0,
                 extra_env: dict | None = None, traces: bool = False,
                 args: list | None = None):
        self.script = script
        self.args = [str(a) for a in (args or [])]
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ready_timeout = ready_timeout
        self.extra_env = extra_env or {}
        self.traces = traces
        self.proc = None
        self.log_path = None
        self.last_stop = None      # outcome of the last shutdown (stop)

    def spawn(self, tag: str):
        """Start the process without waiting for it to be ready.

        MOLA declares itself ready only after the first cloud, which does not arrive while the
        simulation is stopped: starting and waiting in one go with Isaac Sim paused leads to a
        timeout.
        """
        self.stop()
        # Processes surviving stop() would publish on the same topic and the poses would mix. The
        # other known instances (continuous victim, attacker MOLA, in their own namespaces) are
        # expected.
        others = sum(1 for r in MolaRunner._live if r is not self and r.alive())
        left = self._count_processes() - others
        if left > 0:
            raise RuntimeError(
                f"{left} processi mola-cli ancora vivi prima dell'avvio oltre alle "
                f"{others} istanze attese: pubblicherebbero sullo stesso topic e le "
                f"pose si mescolerebbero. Controlla con: ps aux | grep '[m]ola-cli'")
        MolaRunner._live.add(self)
        self.log_path = self.log_dir / f"mola_{tag}.log"
        f = open(self.log_path, "w")
        self._f = f
        env = os.environ.copy()
        if DDS_PROFILO["path"]:
            # --dds-profile: the same profile as the rest of the chain
            env["FASTRTPS_DEFAULT_PROFILES_FILE"] = DDS_PROFILO["path"]
        else:
            env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)   # factory DDS transport for MOLA
        env.update({k: str(v) for k, v in self.extra_env.items()})
        if self.traces:
            # Per-scan MOLA traces: ICP quality, adaptive sigma, estimated twist.
            env["TRACES"] = str(self.log_dir / f"traces_{tag}.csv")
            self.traces_path = self.log_dir / f"traces_{tag}.csv"
        self.proc = subprocess.Popen(
            ["bash", self.script, *self.args], stdout=f, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, env=env)

    def wait_ready(self, check=None, marker=None, fail_markers=()):
        """Wait for the initial relocalisation. Requires the simulation in play.

        check, if given, is called at every iteration and may raise IsaacStall: without clouds MOLA
        never declares itself ready and the stall would turn into a plain timeout. marker: ready
        line (default the initial relocalisation); the twin G loaded from state does not print it
        and uses tv.G_READY_MARKER. fail_markers: lines that stop the wait with an error.
        """
        marker = marker or "Initial re-localization done"
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if check is not None:
                check()
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(f"mola-cli terminato subito ({self.log_path})")
            try:
                text = self.log_path.read_text()
                if marker in text:
                    return True
                bad = [m for m in fail_markers if m in text]
                if bad:
                    raise RuntimeError(f"MOLA: {bad[0]} ({self.log_path})")
            except OSError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"MOLA non pronto ({self.log_path})")

    def start(self, tag: str):
        """Start and wait. Use only with the simulation in play."""
        self.spawn(tag)
        return self.wait_ready()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def crashed(self) -> bool:
        """True if the process terminated on its own.

        An aggressive genome can produce degenerate clouds and crash MOLA: treated as a failed
        evaluation, not a fatal error.
        """
        return self.proc is not None and self.proc.poll() is not None

    @staticmethod
    def _count_processes() -> int:
        """Number of live mola-cli processes according to the process table.

        The ROS graph is not used: discovery takes seconds to drop a dead publisher and would give
        false positives right after a termination. Exact process name (-x), not the command line
        (-f): with -f any shell, grep or editor naming mola-cli would be counted, and killed by
        kill_orphans.
        """
        r = subprocess.run(["pgrep", "-x", "mola-cli"], capture_output=True,
                           text=True)
        return len([ln for ln in r.stdout.splitlines() if ln.strip()])

    @classmethod
    def kill_orphans(cls, timeout: float = 15.0) -> int:
        """Terminate every mola-cli and wait until it is gone from the process table.

        SIGTERM first, then SIGKILL if some remain after `timeout` seconds. mola-cli installs a
        handler only for SIGINT (mola-cli.cpp:51-60): SIGTERM goes to the rclcpp handler, and MOLA
        was never observed to exit within the 12 s of MolaRunner.stop (8222 shutdowns out of 8222).
        The /lidar_odometry/pose publisher has infinite liveliness: without the unregister message
        its entry stays in the ROS graph even after the process is dead.

        Returns the number of processes still alive.
        """
        if cls._count_processes() == 0:
            return 0
        subprocess.run(["pkill", "-TERM", "-x", "mola-cli"], capture_output=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cls._count_processes() == 0:
                return 0
            time.sleep(0.2)
        subprocess.run(["pkill", "-9", "-x", "mola-cli"], capture_output=True)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if cls._count_processes() == 0:
                return 0
            time.sleep(0.2)
        return cls._count_processes()

    def stop(self):
        """Stop the process leaving it time to unregister from DDS.

        last_stop keeps the signal, the measured exit time (0.1 s resolution; None if SIGKILL was
        needed) and the log of the stopped instance.
        """
        if self.proc is None:
            try:
                self._f.close()
            except Exception:
                pass
            return

        sig_name = MOLA_STOP.get("signal", "term")
        t0 = time.time()
        try:
            os.killpg(os.getpgid(self.proc.pid),
                      signal.SIGINT if sig_name == "int" else signal.SIGTERM)
        except Exception:
            pass

        exit_s = None
        deadline = t0 + MOLA_STOP_GRACE_SEC
        while time.time() < deadline:
            if self.proc.poll() is not None:
                exit_s = time.time() - t0
                break
            time.sleep(0.1)
        else:
            # SIGKILL leaves a stale entry in the ROS graph for a few seconds.
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=3.0)
            except Exception:
                pass
        self.last_stop = {"signal": sig_name, "exit_s": round(exit_s, 3) if exit_s is not None else None,
                          "killed": exit_s is None, "returncode": self.proc.returncode,
                          "grace_s": MOLA_STOP_GRACE_SEC,
                          "log": str(self.log_path) if self.log_path else None}

        self.proc = None
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Orchestrator ROS node
# ---------------------------------------------------------------------------

class AttackNode:
    """Drives the robot during rollouts and talks to the perturbation node.

    Control is internal and not delegated to waypoint_follower_node: the robot must stop precisely
    after H true metres and the waypoint must be re-expressed in the MOLA frame at every restart.
    """

    registri = None            # live logs (abilita_registri), off by default

    def __init__(self, follower: FollowerModel):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Bool, Float32MultiArray, String

        self._rclpy = rclpy
        self._Twist = Twist
        self._F32 = Float32MultiArray
        from rclpy.executors import SingleThreadedExecutor
        rclpy.init()
        self.node = Node("attack_orchestrator")
        # Own executor with the node added once: rclpy.spin_once(node) adds and removes the node
        # from the global executor at every call, and the resulting wake-up consumes iterations
        # without running callbacks.
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)

        self.est = None            # MOLA estimated pose
        self.est_t = 0.0
        self.chamfer = []          # perceptibility reported by the node
        self.frames_out = 0        # perturbed clouds, cumulative
        self.trace_rows = None     # if not None, drive() appends the trace to it
        self.last_trace = []       # trace of the last rollout, for the deviation along the stretch
        self._last_true = None
        self._warned_no_stream = False
        self._n_cb = 0             # callbacks run, to empty the queue in drain()
        # Registration quality reported by MOLA: a detectability channel independent of Chamfer,
        # since a defender has no access to the original cloud but sees whether SLAM registers well.
        self.pose_quality = float("nan")
        self.last_track_ratio = float("nan")   # estimate/true ratio of the last rollout
        # Localisation error (estimate-truth gap in world frame, m) and geometric perceptibility
        # metrics of the last rollout.
        self.last_loc_err_mean = float("nan")      # stamp-aligned error (m)
        self.last_loc_err_end = float("nan")
        self.last_loc_err_mean_raw = float("nan")  # not aligned (latency included)
        self.last_loc_err_end_raw = float("nan")
        self.last_track_thr = float("nan")         # tracking threshold used
        self.last_settle_waited = 0.0              # convergence wait (s)
        self.last_settle_ok = True
        self.est_stamp = float("nan")              # sim stamp of the last estimate
        self.true_buf = deque(maxlen=400)          # (stamp_sim, x, y) for interpolation
        self.status_t = None                       # wall time of the last /attack/status
        self.perc_samples = []                 # shift_frac etc. samples from the node
        self.last_perc = {}
        # Accumulated temporal drift from the node (cm) and frame index, from the status: they end
        # up in every trace row. Zero with the attack off.
        self.drift_cm = 0.0
        self.drift_xy_cm = 0.0
        self.node_frame_idx = -1
        self.proc_ms_roll = []        # per-cloud compute time in the current rollout (ms)
        self.node_enabled = None               # attack state reported by the node
        self.node_genome_id = None             # genome id loaded by the node
        self.victim_stream = None              # victim stream state in the node
        self.victim_attack = None
        self.victim_genome_id = None
        self.victim_drift_xy_cm = 0.0
        self.victim_stamp_offset = 0.0
        self.max_drift_lag_s = None
        self.last_drift_start_cm = float("nan")   # |d_xy| at the first driving tick
        self.last_drift_end_cm = float("nan")
        self.last_motion = {}                      # motion classification
        self.last_quality_mean = None              # mean ICP quality of the rollout
        self.last_stop_reason = None               # stop cause of the rollout
        # End-of-window stop (--window-stop) and stop record of the last rollout (fields in
        # drift_fields).
        self.window_stop = "legacy"
        self.stop_ramp_k = wstop.DEFAULT_K
        self.stop_ramp_vmin = wstop.DEFAULT_VMIN
        self.last_stop_info = None
        # --goal-mode: waypoints (default) or direction (carrot on the line).
        self.goal_mode = "waypoints"
        self.direction_lookahead = DIRECTION_LOOKAHEAD
        # Per-window record (window_record.py): first victim stamp after the stream reopens and flag
        # of the first twin ICP.
        self.last_victim_open_stamp = None
        self.last_twin_flags = None

        self._sub_gemello = self.node.create_subscription(Odometry, "/lidar_odometry/pose",
                                                          self._est_cb, 100)
        # Continuous victim: a second MOLA in the /victim namespace, never restarted, listening to
        # /carter/lidar_victim. The estimate used by the controller comes from the twin ("twin") or
        # from the victim ("victim").
        self.est_source = "twin"
        self.est_raw = None
        self.est_origin = None     # origin of the stretch in the victim frame
        self.victim_est_raw = None # last raw victim estimate (always)
        self.victim_est_t = 0.0
        self.victim_est_stamp = None   # stamp of the last victim pose received
        # Attacker MOLA: continuous, on the clean cloud, namespace /attacker.
        self.attacker_est_raw = None
        self.attacker_est_t = 0.0
        self.attacker_est6 = None          # (x, y, z, yaw, pitch, roll) from the same message
        self.attacker_est_stamp = None
        self.node.create_subscription(Odometry, "/attacker/lidar_odometry/pose",
                                      self._attacker_est_cb, 100)
        self._sub_vittima = self.node.create_subscription(Odometry, "/victim/lidar_odometry/pose",
                                                          self._victim_est_cb, 100)
        # --registri-dal-vivo: None = off (no extra lines, no extra calls)
        self.registri = None
        # True pose from the Isaac Sim topic: same physical source as the Script Editor server, but
        # over ROS. The file channel stays for pause, play, save and restore, which have no ROS
        # equivalent.
        self.true = None
        self.true_t = 0.0
        self.true_z = float("nan")
        self.true_v = (float("nan"), float("nan"))   # CoM velocity (odom), robot x and y
        self.est_z = float("nan")
        self.last_tau = float("nan")
        self.last_victim_open_wall = None
        self.last_victim_close_wall = None
        self.pause_start_wall = None
        self.victim_stamp_offset = None
        self.true_source = "ros"
        # Short queue: only the last message matters, and at ~50 messages/s a long undrained queue
        # turns into delay.
        self.node.create_subscription(Odometry, "/chassis/odom",
                                      self._true_cb, 5)
        self.node.create_subscription(String, "/attack/status",
                                      self._status_cb, 10)
        from std_msgs.msg import Float32
        self.node.create_subscription(Float32, "/lidar_odometry/pose_quality",
                                      self._quality_cb, 50)
        self.node.create_subscription(Float32, "/victim/lidar_odometry/pose_quality",
                                      self._victim_quality_cb, 50)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self._enable_pub = self.node.create_publisher(Bool, "/attack/enabled", 10)
        from std_msgs.msg import Empty
        self._drift_reset_pub = self.node.create_publisher(Empty, "/attack/drift_reset", 10)
        self._victim_stream_pub = self.node.create_publisher(Bool, "/attack/victim_stream", 10)
        self._victim_attack_pub = self.node.create_publisher(Bool, "/attack/victim_attack", 10)
        self._victim_genome_pub = self.node.create_publisher(
            Float32MultiArray, "/attack/victim_genome", 10)
        self._twin_replay_pub = self.node.create_publisher(String, "/attack/twin_replay", 10)
        # Service clients for map-based restore (--twin-state maprestore / --map-gate-test): save
        # the victim map (/victim/map_save) and reload it in the twin (/map_load), relocalising at
        # the pose believed by the victim (/relocalize_near_pose). In try/except: if mola_msgs is
        # not available the clients stay None and the other --twin-state modes are unaffected.
        self._map_restore_import_error = None
        self._victim_map_save_cli = self._twin_map_load_cli = self._twin_reloc_cli = None
        self._twin_initialpose_pub = None
        try:
            from mola_msgs.srv import MapSave, MapLoad, RelocalizeNearPose
            from geometry_msgs.msg import PoseWithCovarianceStamped
            self._MapSave, self._MapLoad, self._RelocNearPose = MapSave, MapLoad, RelocalizeNearPose
            self._PoseWithCovStamped = PoseWithCovarianceStamped
            self._victim_map_save_cli = self.node.create_client(MapSave, "/victim/map_save")
            self._twin_map_load_cli = self.node.create_client(MapLoad, "/map_load")
            self._twin_reloc_cli = self.node.create_client(RelocalizeNearPose, "/relocalize_near_pose")
            # The MOLA config relocalises from the /initialpose TOPIC (relocalize_from_topic), not
            # from the near-pose service (rejected). QoS transient_local (latched) so that the twin
            # subscriber, just started, still receives the last pose if it arrives late.
            from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
            qos_ip = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
            # /initialpose is an absolute name: published there it would also reach victim and
            # attacker. The twin receives it remapped (tv.TWIN_INITIALPOSE, twin ROS_ARGS in
            # maprestore).
            self._twin_initialpose_pub = self.node.create_publisher(
                PoseWithCovarianceStamped, tv.TWIN_INITIALPOSE, qos_ip)
        except Exception as e:  # mola_msgs missing or different interfaces
            self._map_restore_import_error = str(e)
        self.victim_est_frame_id = None   # frame of the victim pose (relocalize)
        self.node_gen_limits = None       # node generator limits (status gen_limits)
        self.node_moltiplicatori = None   # node limit multiplier (status moltiplicatori_limiti)
        # --zone-attaccante: node state (status) and expected zones per stream
        self.node_zone_on = None
        self.node_fotografia = None
        self.node_zone_ricevute = None
        self.node_zone_applicate = None
        self.zone_attese = {}
        self.node_structural = None       # shape of the structural term of P (status structural)
        self.victim_rec_n = 0
        self.replay_seq = None
        self.replay_active = None
        self.replay_n = 0
        self.replay_total = 0
        self.replay_last_stamp = None
        self.victim_rec_total = None
        self.replay_first_stamp = None
        self.st_drift_xy_cm = self.st_victim_drift_xy_cm = self.st_drift_keep = None
        self.st_victim_drift_cm = None   # victim |A| (null genome)
        # twin starting drift (node with --banco-twin-drift; absent without it)
        self.st_twin_drift_on = False
        self.st_twin_drift_init = None
        # --fresh-cloud-after-restore: node filter armed after every restore. Initial sequence
        # number from the clock (ms): it cannot match the one left in the node by a previous launch.
        self.fresh_cloud = False
        self._fresh_pub = None
        self._fresh_seq_sent = int(time.time() * 1000) % (2 ** 32)
        self.fresh_seq = None
        self.node_fresh_cloud = None
        self.last_fresh_arm = None        # outcome of the last arm (seq, confirmed)
        self.fresh_arm_n = self.fresh_arm_ok = 0
        self.last_twin_fp = None          # fingerprint of the twin initial state
        self.last_twin_align = None       # twin-victim alignment at rollout start
        # Attacker MOLA poses (stamp, x, y, yaw) for the velocity.
        self.attacker_buf = deque(maxlen=50)
        self.last_replay = None
        self.twin_clock_on = False
        self.twin_stamp_offset = 0.0
        self.est_stamp_raw = float("nan")
        self._victim_drift_reset_pub = self.node.create_publisher(
            Empty, "/attack/victim_drift_reset", 10)
        self.genome_pub = self.node.create_publisher(
            Float32MultiArray, "/attack/genome", 10)
        # Cloud arrivals at the MOLA input, to detect an Isaac stall. raw=True: only the arrival
        # time is logged, without deserialising ~700 kB per message.
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        self.cloud_t = 0.0
        self.node.create_subscription(PointCloud2, "/carter/lidar_perturbed",
                                      self._cloud_cb, qos_profile_sensor_data,
                                      raw=True)
        self.watch = None          # IsaacClient to watch, None = check off
        self.ombra = None          # ShadowRunner to watch (--controllo-ombra-viva), None = off
        self._t_check = 0.0
        self.follower = follower

    def _est_cb(self, msg):
        if self.est_source != "twin":
            self._n_cb += 1
            return
        # Twin replayed from the victim history: stamp on the twin clock, mapped back to simulation
        # time with the node offset.
        self._store_est(msg, stamp_offset=(self.twin_stamp_offset if self.twin_clock_on else 0.0))

    def _victim_est_cb(self, msg):
        self.victim_est_raw = self._pose_of(msg)
        self.victim_est_t = time.time()
        # raw stamp (victim clock, as in the printf lines): chain check
        self.victim_est_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if msg.header.frame_id:
            self.victim_est_frame_id = msg.header.frame_id
        if self.est_source != "victim":
            self._n_cb += 1
            return
        # Stamp on the continuous victim clock: mapped back to simulation time with the offset
        # published by the node.
        self._store_est(msg, stamp_offset=self.victim_stamp_offset)

    def _attacker_est_cb(self, msg):
        self._n_cb += 1
        self.attacker_est_raw = self._pose_of(msg)
        self.attacker_est_t = time.time()
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.attacker_buf.append((st, *self.attacker_est_raw))
        # full pose from the same message (x, y, z, yaw, pitch, roll), for --twin-init-assetto
        # attaccante: roll and pitch of the attacker MOLA
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.attacker_est6 = tv.pose6_from_quat(p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self.attacker_est_stamp = st

    @staticmethod
    def _pose_of(msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (p.position.x, p.position.y, yaw)

    @staticmethod
    def compose(base, local):
        """Pose `local` (in the frame with origin `base`) expressed in the base frame."""
        bx, by, byaw = base
        lx, ly, lyaw = local
        c, s_ = math.cos(byaw), math.sin(byaw)
        return (bx + c * lx - s_ * ly, by + s_ * lx + c * ly, byaw + lyaw)

    def _store_est(self, msg, stamp_offset=0.0):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.est_raw = (p.position.x, p.position.y, yaw)
        self.est_z = p.position.z
        self.est_quat = (q.x, q.y, q.z, q.w)   # twin 6D pose at save time (realignment, bench)
        # The controller works in the rollout frame (origin = pose at the start of the stretch). The
        # twin restarts from zero at every rollout; the continuous victim does not: its estimate is
        # mapped back to the stretch origin.
        if self.est_origin is None:
            self.est = self.est_raw
        else:
            ox, oy, oyaw = self.est_origin
            c, s_ = math.cos(oyaw), math.sin(oyaw)
            dx, dy = p.position.x - ox, p.position.y - oy
            self.est = (c * dx + s_ * dy, -s_ * dx + c * dy, angle_diff(yaw, oyaw))
        self.est_t = time.time()
        self.est_stamp_raw = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.est_stamp = self.est_stamp_raw + stamp_offset
        self._n_cb += 1

    def _true_cb(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.true = (p.position.x, p.position.y, yaw)
        self.true_t = time.time()
        self.true_z = p.position.z
        self.true_v = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.true_buf.append((st, p.position.x, p.position.y))
        self._n_cb += 1

    def true_pose(self, isaac, max_age=0.5):
        """True pose in the working frame.

        With true_source "ros" it is the last /chassis/odom message, which Isaac Sim expresses in
        the odom frame (origin where the robot was at Play), not in the scene: that is why all logic
        uses this function and never the poses returned by the server, which are in the scene frame.
        With the simulation paused the topic is silent and the last message is the still pose;
        max_age applies only while the simulation runs. With "file" it is the server pose, in the
        scene frame.
        """
        if self.true_source == "ros" and self.true is not None and (
                time.time() - self.true_t <= max_age or not isaac.playing):
            return self.true
        return isaac.pose_stream()

    def settled_pose(self, isaac):
        """True pose with the robot still and the simulation running: waits for a new message."""
        if self.true_source == "ros":
            t0 = self.true_t
            deadline = time.time() + 2.0
            while self.true_t <= t0 and time.time() < deadline:
                self.spin(0.05)
            if self.true is not None:
                return self.true
        return isaac.pose()

    def true_at(self, stamp):
        """True pose (x, y) interpolated at the sim stamp of the scan (odom frame).

        /chassis/odom and /lidar_odometry/pose are both in sim time: the true pose at the scan
        instant, not the current one, removes the latency from the estimate-truth gap. Outside the
        buffer interval: nearest end. Empty buffer or missing stamp: None.
        """
        buf = self.true_buf
        if not buf or stamp != stamp:
            return None
        if stamp <= buf[0][0]:
            return (buf[0][1], buf[0][2])
        if stamp >= buf[-1][0]:
            return (buf[-1][1], buf[-1][2])
        for i in range(len(buf) - 1):
            t0, x0, y0 = buf[i]
            t1, x1, y1 = buf[i + 1]
            if t0 <= stamp <= t1:
                a = (stamp - t0) / (t1 - t0) if t1 > t0 else 0.0
                return (x0 + a * (x1 - x0), y0 + a * (y1 - y0))
        return (buf[-1][1], buf[-1][2])

    def _status_cb(self, msg):
        self._n_cb += 1
        self.status_t = time.time()    # last perturbation_node status (chain check)
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        # frames_out is cumulative: the per-window count is the difference between rollout start and
        # end.
        if d.get("frames_out") is not None:
            self.frames_out = int(d["frames_out"])
        if d.get("frame_idx") is not None:
            new_idx = int(d["frame_idx"])
            # One entry per processed cloud: the genome compute cost in the rollout, to compare with
            # the scan period.
            if new_idx != self.node_frame_idx and d.get("proc_ms_last") is not None:
                self.proc_ms_roll.append(float(d["proc_ms_last"]))
            self.node_frame_idx = new_idx
        self.node_enabled = d.get("enabled")
        if d.get("genome_id") is not None:
            self.node_genome_id = int(d["genome_id"])
        if "metro_gemello" in d:
            self.node_metro = {"gemello": d.get("metro_gemello"), "vittima": d.get("metro_vittima")}
        if isinstance(d.get("gen_limits"), dict):
            self.node_gen_limits = d["gen_limits"]
            # node --moltiplicatore-limiti: key present only with the option
            self.node_moltiplicatori = d.get("moltiplicatori_limiti")
        # node --rampa-genoma: key present only with the option
        self.node_rampa_on = "rampa" in d
        self.node_rampa = d.get("rampa")
        # node --genoma-in-moto: key present only with the option
        self.node_gim_on = "genoma_in_moto" in d
        self.node_gim = d.get("genoma_in_moto")
        # node --chamfer-per-fase: key present only with the option
        self.node_cpf_on = "chamfer_fasi" in d
        self.node_cpf = d.get("chamfer_fasi")
        if "zone_applicate" in d:
            self.node_zone_on = bool(d.get("zone_attaccante"))
            self.node_fotografia = d.get("fotografia")
            self.node_zone_ricevute = d.get("zone_ricevute")
            self.node_zone_applicate = d.get("zone_applicate")
        if d.get("structural") is not None:
            self.node_structural = str(d["structural"])
        self.victim_stream = d.get("victim_stream")
        # readers matched to the node publishers (absent without --report-matched)
        self.victim_pub_matched = d.get("victim_pub_matched")
        self.shadow_pub_matched = d.get("shadow_pub_matched")
        self.main_pub_matched = d.get("main_pub_matched")
        self.victim_stamp_offset = d.get("victim_stamp_offset_s")
        self.victim_attack = d.get("victim_attack")
        if d.get("victim_genome_id") is not None:
            self.victim_genome_id = int(d["victim_genome_id"])
        if d.get("victim_stamp_offset_s") is not None:
            self.victim_stamp_offset = float(d["victim_stamp_offset_s"])
        if d.get("victim_rec_n") is not None:
            self.victim_rec_n = int(d["victim_rec_n"])
            self.replay_seq = d.get("replay_seq")
            self.replay_active = d.get("replay_active")
            self.replay_n = int(d.get("replay_n") or 0)
            self.replay_total = int(d.get("replay_total") or 0)
            self.replay_last_stamp = d.get("replay_last_stamp")
            self.twin_clock_on = bool(d.get("twin_clock_on"))
            # variant G (node with --twin-clock-base)
            self.twin_clock_seq = d.get("twin_clock_seq")
            self.twin_clock_first_ns = d.get("twin_clock_first_ns")
            if d.get("twin_stamp_offset_s") is not None:
                self.twin_stamp_offset = float(d["twin_stamp_offset_s"])
            self.victim_rec_total = d.get("victim_rec_total")
            self.replay_first_stamp = d.get("replay_first_stamp")
        # Raw drift of the two generators and keep mode, always (initial state fingerprint), whether
        # or not the attack is on.
        self.st_drift_xy_cm = d.get("drift_xy_cm")
        self.st_victim_drift_xy_cm = d.get("victim_drift_xy_cm")
        self.st_victim_drift_cm = d.get("victim_drift_cm")
        self.st_drift_keep = d.get("drift_keep")
        self.st_twin_drift_on = "twin_drift_init" in d
        self.st_twin_drift_init = d.get("twin_drift_init")
        # per-stream generator state (node with --stato-generatore; absent without it)
        self.st_stato_generatore = d.get("stato_generatore")
        self.node_fresh_cloud = d.get("fresh_cloud")
        self.fresh_seq = d.get("fresh_seq")
        self.fresh_dropped_total = d.get("fresh_dropped_total")   # dropped by the filter
        # node --victim-never-clean: option and counters (absent without it)
        self.node_never_clean = d.get("victim_never_clean")
        self.victim_clean_dropped = d.get("victim_clean_dropped")
        self.victim_clean_out = d.get("victim_clean_out")
        victim_on = bool(d.get("victim_stream")) and bool(d.get("victim_attack"))
        self.max_drift_lag_s = d.get("max_drift_lag_s")
        if victim_on and d.get("victim_drift_cm") is not None:
            self.drift_cm = float(d["victim_drift_cm"])
            self.drift_xy_cm = float(d["victim_drift_xy_cm"])
        elif d.get("enabled") and not d.get("victim_stream") and d.get("drift_cm") is not None:
            self.drift_cm = float(d["drift_cm"])
            self.drift_xy_cm = float(d.get("drift_xy_cm", d["drift_cm"]))
        else:
            self.drift_cm = 0.0
            self.drift_xy_cm = 0.0
        # chamfer_cm is None in passthrough or before the first sample: float(None) in the callback
        # would kill the executor spin.
        v = d.get("chamfer_cm")
        attacking = victim_on or (bool(d.get("enabled")) and not d.get("victim_stream"))
        if v is None or not attacking:
            return
        try:
            self.chamfer.append(float(v))
        except (TypeError, ValueError):
            pass
        if d.get("shift_frac") is not None or d.get("tc_frac8") is not None:
            keys = ("shift_frac", "shift_med_cm", "n_added", "n_removed",
                    "chamfer_pure_cm", "structural_cm", "point_ratio",
                    "point_ratio_gross", "n_pts_removed", "n_pts_added",
                    "tc_frac8", "tc_frac16", "tc_med_cm", "tc_p90_cm", "tc_rot_deg", "tc_trans_cm", "tc_clean_frac8")
            self.perc_samples.append({k: d[k] for k in keys if d.get(k) is not None})

    def _cloud_cb(self, _msg):
        self._n_cb += 1
        self.cloud_t = time.time()

    def check_stall(self):
        """Raise IsaacStall if the Isaac loop or the clouds have been frozen for STALL_SEC.

        The pose age in the server file always counts (the loop writes it even when paused); clouds
        are counted only while the simulation runs and after a play issued by this orchestrator. At
        most one check every 0.5 s.
        """
        isaac = self.watch
        now = time.time()
        if isaac is None or now - self._t_check < 0.5:
            return
        self._t_check = now
        self.check_ombra()
        try:
            with open(POSE_FILE) as f:
                age = now - float(json.load(f).get("wall", 0.0))
        except (OSError, ValueError):
            age = float("inf")
        if age > STALL_SEC:
            raise IsaacStall(f"il ciclo di Isaac non aggiorna {POSE_FILE} da {age:.1f} s")
        if isaac.playing and isaac.t_play is not None and now - isaac.t_play > STALL_SEC:
            last = max(self.cloud_t, isaac.t_play)
            if now - last > STALL_SEC:
                raise IsaacStall(f"nessuna nuvola su /carter/lidar_perturbed da {now - last:.1f} s "
                                 f"a simulazione in corso")

    def check_ombra(self):
        """--controllo-ombra-viva: OmbraPersa if the shadow process, already started, has
        terminated. No check with the option off (ombra None), before the start (proc None) or
        after the run is closed (check_stall with watch None does not get here).
        """
        r = getattr(self, "ombra", None)
        if r is None or getattr(r, "proc", None) is None:
            return
        rc = r.proc.poll()
        if rc is not None:
            raise OmbraPersa(f"ombra persa: il processo MOLA dell'ombra e' terminato (codice {rc}, "
                             f"log {getattr(r, 'log_path', '?')})")

    def _quality_cb(self, msg):
        self._n_cb += 1
        if self.est_source == "twin":
            self.pose_quality = float(msg.data)

    def _victim_quality_cb(self, msg):
        self._n_cb += 1
        if self.est_source == "victim":
            self.pose_quality = float(msg.data)

    def count_mola_publishers(self) -> int:
        """Publishers on /lidar_odometry/pose in the ROS graph (diagnostic, 1 expected)."""
        return self.node.count_publishers("/lidar_odometry/pose")

    def drain(self, budget: float = 0.02):
        """Run all queued callbacks, within budget seconds.

        spin_once runs only one: with /chassis/odom at ~50 messages/s, one per control tick (20 Hz)
        lets the queue grow and true pose and estimate arrive 1-2 s late. One empty iteration is not
        enough to say the queue is empty (wake-ups without callbacks): stop after three in a row.
        """
        t_end = time.time() + budget
        empty = 0
        while time.time() < t_end and empty < 3:
            n = self._n_cb
            self._exec.spin_once(timeout_sec=0.0)
            empty = empty + 1 if self._n_cb == n else 0
        self.check_stall()

    def spin(self, sec: float):
        t0 = time.time()
        while time.time() - t0 < sec:
            self._exec.spin_once(timeout_sec=0.01)
            self.check_stall()

    def set_genome(self, genome):
        """Load the genome into the node and wait for the status to confirm the change (genome_id
        incremented), republishing if needed: a message published before DDS has matched
        publisher and subscriber is lost without error.
        """
        self.manda_rampa("gemello", genome)
        self.set_metro("gemello")
        self.set_zone("gemello", genome)
        m = self._F32()
        m.data = [float(v) for v in genome]
        before = self.node_genome_id
        self._publish_confirmed(self.genome_pub, m,
                                lambda: self.node_genome_id is not None
                                and self.node_genome_id != before,
                                "genoma")

    def _publish_confirmed(self, pub, msg, confirmed, what, timeout=3.0, every=0.5):
        t0 = time.time()
        t_pub = 0.0
        while time.time() - t0 < timeout:
            if time.time() - t_pub >= every:
                pub.publish(msg)
                t_pub = time.time()
            self.spin(0.05)
            if confirmed():
                self.spin(0.1)
                return True
        print(f"    [nota] il nodo non ha confermato {what} entro {timeout:.0f} s "
              f"(status assente o stantio): si prosegue")
        return False

    def arm_fresh_cloud(self):
        """With --fresh-cloud-after-restore, with the simulation stopped and after the restore: the
        node will drop clouds acquired before the pause time + margin (the first scan after play
        is partly taken in the pose before the restore). No-op without the option.
        """
        if not self.fresh_cloud:
            return None
        from std_msgs.msg import UInt32
        if self._fresh_pub is None:
            self._fresh_pub = self.node.create_publisher(UInt32, "/attack/fresh_after_restore", 10)
        self._fresh_seq_sent = (self._fresh_seq_sent + 1) % (2 ** 32)
        seq = self._fresh_seq_sent
        m = UInt32(); m.data = seq
        t0 = time.time()
        ok = self._publish_confirmed(self._fresh_pub, m, lambda: self.fresh_seq == seq,
                                     "filtro della prima nuvola dopo il ripristino")
        # arm outcome for the history (window and rollout): an unconfirmed arm leaves the node
        # filter as it was
        rl = getattr(self, "restore_log", None) or []
        self.last_fresh_arm = {"seq": seq, "confirmed": bool(ok), "t_wall": round(t0, 3),
                               "wait_s": round(time.time() - t0, 3),
                               # restore it refers to (last recorded)
                               "restore_i": rl[-1]["i"] if rl else None,
                               "window": rl[-1]["window"] if rl else None}
        if getattr(self, "fresh_arm_log", None) is None:
            self.fresh_arm_log = []
        self.fresh_arm_log.append(self.last_fresh_arm)
        self.fresh_arm_n = getattr(self, "fresh_arm_n", 0) + 1
        self.fresh_arm_ok = getattr(self, "fresh_arm_ok", 0) + bool(ok)
        return ok

    def arm_twin_clock(self, tick):
        """Variant G: twin clock base (tick of the state saved by the shadow), repeated until the
        node status confirms the seq. Call with the simulation stopped, before spawning the twin.
        Never confirmed without --twin-clock-base in the node: False.
        """
        from std_msgs.msg import String as _Str
        if getattr(self, "_twin_clock_pub", None) is None:
            self._twin_clock_pub = self.node.create_publisher(_Str, "/attack/twin_clock_base", 10)
            self._twin_clock_seq_sent = int(time.time() * 1000) % (2 ** 31)
        self._twin_clock_seq_sent = (self._twin_clock_seq_sent + 1) % (2 ** 31)
        seq = self._twin_clock_seq_sent
        m = _Str()
        m.data = json.dumps({"tick": int(tick), "seq": seq})
        self.twin_clock_first_ns = None
        ok = self._publish_confirmed(self._twin_clock_pub, m,
                                     lambda: getattr(self, "twin_clock_seq", None) == seq,
                                     "base dell'orologio del gemello G")
        self.last_twin_clock_arm = {"seq": seq, "tick": int(tick), "confirmed": bool(ok)}
        return ok

    def reset_drift(self):
        """Reset the temporal drift accumulated by the node (not the frame counter)."""
        from std_msgs.msg import Empty
        self._drift_reset_pub.publish(Empty())
        # Local delivery: a short spin is enough. With 0.2 s the drift at the first driving tick was
        # already 1-2 frames (up to 9 cm at strength 1).
        self.spin(0.1)

    @staticmethod
    def classify_motion(trace):
        """Classify the motion of a trace: "moto", "paralisi_deriva" or "rotazione_assestamento"
        (see PARALYSIS_DRIFT_CM). Returns a dict with motion_class, stop_onset_s and
        drift_at_stop_cm.
        """
        onset = None
        for k in range(len(trace) - STOP_TICKS + 1):
            if all(abs(trace[j]["cmd_v"]) < STOP_V and abs(trace[j]["cmd_w"]) > STOP_W
                   for j in range(k, k + STOP_TICKS)):
                onset = k
                break
        if onset is None:
            return {"motion_class": "moto", "stop_onset_s": None, "drift_at_stop_cm": None,
                    "stop_still_s": None, "stop_drift_max_cm": None, "stop_resumed": None}
        j = onset
        while j < len(trace) and abs(trace[j]["cmd_v"]) < STOP_V:
            j += 1
        still = trace[min(j, len(trace) - 1)]["t"] - trace[onset]["t"]
        drifts = [r.get("drift_xy_cm", 0.0) for r in trace[onset:max(j, onset + 1)]]
        drifts = [d for d in drifts if d == d]
        d0 = drifts[0] if drifts else 0.0
        dmax = max(drifts) if drifts else 0.0
        resumed = j < len(trace) and (trace[-1]["travelled"] - trace[j]["travelled"]) >= STOP_RESUME_M
        if resumed:
            cls = "rotazione_assestamento"
        elif dmax >= PARALYSIS_DRIFT_CM:
            cls = "paralisi_deriva"
        else:
            cls = "arresto_altro"
        return {"motion_class": cls, "stop_onset_s": trace[onset]["t"],
                "drift_at_stop_cm": round(d0, 2), "stop_still_s": round(still, 2),
                "stop_drift_max_cm": round(dmax, 2), "stop_resumed": bool(resumed)}

    def set_victim_stream(self, on: bool):
        """Open or close the cloud stream to the continuous victim MOLA."""
        from std_msgs.msg import Bool
        m = Bool(); m.data = bool(on)
        self._publish_confirmed(self._victim_stream_pub, m,
                                lambda: self.victim_stream is bool(on),
                                f"flusso vittima {'aperto' if on else 'chiuso'}")

    def set_victim_attack(self, on: bool):
        from std_msgs.msg import Bool
        m = Bool(); m.data = bool(on)
        return self._publish_confirmed(self._victim_attack_pub, m,
                                       lambda: self.victim_attack is bool(on),
                                       f"attacco vittima {'acceso' if on else 'spento'}")

    def set_metro(self, flusso):
        """--metro-al-nodo: send the metre number (current window, from 0) to the node for the given
        stream, before the genome and confirmed by the status. Zones are chosen at the first
        cloud after the genome, so with the metre already received. No-op without the option.
        """
        if not METRO["on"]:
            return None
        from std_msgs.msg import String
        k = TWIN_STATE.get("window")
        k = 0 if k is None or k < 0 else int(k)
        if getattr(self, "_metro_pub", None) is None:
            self._metro_pub = self.node.create_publisher(String, "/attack/metro", 10)
        m = String()
        m.data = json.dumps({flusso: k})
        ok = self._publish_confirmed(
            self._metro_pub, m,
            lambda: (getattr(self, "node_metro", None) or {}).get(flusso) == k,
            f"metro {k} ({flusso})")
        if not ok:
            METRO["non_confermati"].append({"flusso": flusso, "metro": k})
        return ok

    def _zone_pub(self, topic):
        from std_msgs.msg import String
        pubs = self.__dict__.setdefault("_zone_pubs", {})
        if topic not in pubs:
            pubs[topic] = self.node.create_publisher(String, topic, 10)
        return pubs[topic], String

    def fotografia_metro(self, k, believed=None):
        """--zone-attaccante, start of metre k with the robot still and Isaac paused: the snapshot
        is the last attacker MOLA pose and the clean cloud with the same stamp, saved by the node
        in <out>/zone/w{k}_fotografia.npz. Configuration error if missing or if it does not
        reload with its hash.
        """
        self.spin(0.5)      # pending poses (no new cloud with Isaac stopped)
        if not self.attacker_buf:
            raise ErroreConfigurazioneZone(f"fotografia w{k}: nessuna posa della MOLA dell'attaccante")
        stamp, x, y, yaw = self.attacker_buf[-1]
        path = Path(ZONE["out"]) / "zone" / f"w{k}_fotografia.npz"
        pub, String = self._zone_pub("/attack/fotografia")
        ZONE["seq"] += 1
        seq = ZONE["seq"]
        meta = {"metro": int(k), "stamp_posa": stamp, "posa_base_odom": [x, y, yaw],
                "posa_lavoro": None if believed is None else [float(v) for v in believed],
                "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S")}
        m = String()
        m.data = json.dumps({"seq": seq, "stamp": stamp, "path": str(path), "meta": meta})
        ok = self._publish_confirmed(pub, m, lambda: (self.node_fotografia or {}).get("seq") == seq,
                                     f"fotografia del metro {k}", timeout=5.0)
        rec = dict(self.node_fotografia or {}) if ok else {}
        ZONE["foto"], ZONE["metro"] = None, None
        ZONE["finestra"] = {"fotografia": {"path": str(path), "stamp": stamp, "posa_base_odom": [x, y, yaw],
                                           "confermata": bool(ok), "esito": rec.get("esito"),
                                           "sha256": rec.get("sha256"), "hash": rec.get("hash")},
                            "zone": [], "applicate": []}
        if rec.get("esito") != "ok":
            raise ErroreConfigurazioneZone(f"fotografia w{k}: {rec.get('esito') or 'non confermata dal nodo'}")
        try:
            foto = za.carica_fotografia(path, rec["sha256"], rec["hash"])
        except za.ZoneNonValide as e:
            raise ErroreConfigurazioneZone(f"fotografia w{k}: {e}")
        ZONE["finestra"]["fotografia"].update({"n_punti": foto.meta.get("n_punti"),
                                               "nuvole_dopo": foto.meta.get("nuvole_dopo")})
        ZONE["foto"], ZONE["metro"] = foto, int(k)
        print(f"  fotografia dell'attaccante w{k}: stamp {stamp:.3f}, posa {np.round([x, y, yaw], 3)}, "
              f"{foto.meta.get('n_punti')} punti, sha256 {foto.sha256[:12]}")
        return foto

    def set_zone(self, flusso, genome):
        """--zone-attaccante: genome zones from the metre snapshot, saved and sent to the node for
        the stream, before the genome and confirmed by the status. No-op without the option.
        """
        if not ZONE["on"]:
            return None
        k = TWIN_STATE.get("window")
        k = 0 if k is None or k < 0 else int(k)
        if ZONE.get("foto") is None or ZONE.get("metro") != k:
            raise ErroreConfigurazioneZone(f"zone {flusso} w{k}: fotografia del metro assente")
        t0 = time.time()
        try:
            z = za.zone_genoma(Path(ZONE["out"]) / "zone", ZONE["foto"], genome, k,
                               limiti_zone(self, f"zone {flusso} w{k}"))
        except (za.ZoneNonValide, OSError) as e:
            raise ErroreConfigurazioneZone(f"zone {flusso} w{k}: {e}")
        t_calc = time.time() - t0
        pub, String = self._zone_pub("/attack/zone")
        ZONE["seq"] += 1
        seq = ZONE["seq"]
        m = String()
        m.data = json.dumps({"seq": seq, "flusso": flusso, "path": str(z.path), "sha256": z.sha256,
                             "hash": z.hash, "genoma_sha": z.genoma_sha, "seme": z.seme, "metro": k})
        ok = self._publish_confirmed(
            pub, m, lambda: ((self.node_zone_ricevute or {}).get(flusso) or {}).get("seq") == seq,
            f"zone {flusso} w{k}")
        ric = ((self.node_zone_ricevute or {}).get(flusso) or {}) if ok else {}
        app = ((self.node_zone_applicate or {}).get(flusso) or {})
        rec = {"flusso": flusso, "genoma_sha": z.genoma_sha, "hash": z.hash, "sha256": z.sha256,
               "file": str(z.path), "seme": z.seme, "n_zone": z.n_zone(), "t_calcolo_s": round(t_calc, 3),
               "confermate": bool(ok), "stato_nodo": ric.get("stato")}
        (ZONE.get("finestra") or {}).setdefault("zone", []).append(rec)
        if ric.get("stato") != "ok" or ric.get("hash") != z.hash:
            raise ErroreConfigurazioneZone(f"zone {flusso} w{k} ({z.genoma_sha}): nodo "
                                           f"{ric.get('stato') or 'nessuna conferma'}")
        self.zone_attese[flusso] = {"hash": z.hash, "genoma_sha": z.genoma_sha,
                                    "n_ok0": int(app.get("n_ok") or 0), "n_err0": int(app.get("n_err") or 0)}
        return z

    def zone_controllo(self, flusso, tag, frames=None):
        """--zone-attaccante, after a rollout: stream clouds processed without valid zones, with a
        hash other than the one sent, or no cloud with zones while clouds left the node (frames >
        0) = configuration error. Expected zones are valid for one rollout only. Counters not
        moving: wait for the status as for the other confirmations (up to 2 s); with frames == 0
        no cloud reached the node and the outcome stays the rollout one (nessuna_nuvola,
        recorded).
        """
        att = self.zone_attese.pop(flusso, None)
        if not ZONE["on"] or att is None:
            return None

        def conta():
            a = ((self.node_zone_applicate or {}).get(flusso) or {})
            return a, int(a.get("n_ok") or 0) - att["n_ok0"], int(a.get("n_err") or 0) - att["n_err0"]
        self.spin(0.3)
        app, n_ok, n_err = conta()
        t0 = time.time()
        while n_ok + n_err == 0 and (frames is None or frames > 0) and time.time() - t0 < 2.0:
            self.spin(0.1)
            app, n_ok, n_err = conta()
        nessuna = n_ok + n_err == 0
        esito = ("nessuna_nuvola" if nessuna and frames == 0 else
                 "nessuna_zona_applicata" if nessuna else
                 "errore" if (n_err > 0 or app.get("hash") != att["hash"]) else "ok")
        rec = {"flusso": flusso, "tag": tag, "genoma_sha": att["genoma_sha"], "hash_atteso": att["hash"],
               "hash_applicato": app.get("hash") if n_ok > 0 else None, "stato": app.get("stato"),
               "nuvole_ok": n_ok, "nuvole_err": n_err, "frames": frames, "esito": esito}
        fin = ZONE.get("finestra") or {}
        fin.setdefault("applicate", []).append(rec)
        tot = fin.setdefault("totali", {}).setdefault(flusso, {"nuvole_ok": 0, "nuvole_err": 0, "rollout": 0})
        tot["nuvole_ok"] += n_ok
        tot["nuvole_err"] += n_err
        tot["rollout"] += 1
        if esito in ("errore", "nessuna_zona_applicata"):
            raise ErroreConfigurazioneZone(f"{tag}: flusso {flusso}, {esito}: nuvole con zone {n_ok}, senza "
                                           f"zone valide {n_err} ({app.get('stato')}), nuvole uscite "
                                           f"{frames}, hash {str(app.get('hash'))[:12]} atteso {att['hash'][:12]}")
        return rec

    def manda_rampa(self, flusso, genome, sha_nuovo=None):
        """--rampa-genoma: data of the previous genome of the metre for the new genome of the
        stream, on /attack/rampa, confirmed by the status (rampa.ricevuta[flusso].seq, state ok).
        No message with R = 1, at metre 0 or without a previous genome (the node applies N only). If
        there is no ramp for this genome (metre 0, no previous one, realigned twin with
        RAMPA["sospesa"], bench arm without ramp) but the node holds data that would apply to it
        (same sha, e.g. the last candidate or the winner trials), send the node cancel command
        ({seq, flusso, annulla: true}; state "annullata", the next genome runs without ramp with
        prec_annullata). Record in RAMPA["ultimo"][flusso] and in the window; unconfirmed or
        rejected data = configuration error. With --genoma-in-moto the same message carries the
        previous genome P of the step (receipt in genoma_in_moto.ricevuta).
        """
        if not canale_rampa():
            return None
        k = TWIN_STATE.get("window")
        k = 0 if k is None or k < 0 else int(k)
        sha = za.genoma_sha(genome)
        if (RAMPA.get("preinviata") or {}).get(flusso) == sha:
            # G bench: sent before the metre and zones of the spec, not repeated (not even in
            # rollout repeats); the bench removes it after the rollout
            return RAMPA["ultimo"].get(flusso)
        rec = {"flusso": flusso, "metro": k, "genoma_sha_nuovo": sha, "genoma_prec_sha": RAMPA.get("prec_sha")}
        fin = (RAMPA.get("finestra") or {}).setdefault("messaggi", []) if RAMPA.get("finestra") is not None else []

        def _ric():
            st = getattr(self, "node_rampa" if RAMPA["R"] > 1 else "node_gim", None)
            return ((st or {}).get("ricevuta") or {}).get(flusso) or {}
        senza = RAMPA["sospesa"] or RAMPA["prec"] is None or RAMPA["metro"] != k
        if senza:
            motivo = "sospesa" if RAMPA["sospesa"] else "senza_precedente"
            if _ric().get("genoma_sha_nuovo") != sha:
                rec.update(esito=motivo, genoma_prec_sha=None)
                RAMPA["ultimo"][flusso] = rec
                fin.append(rec)
                return rec
            rec.update(esito="annullata", motivo=motivo, genoma_prec_sha=None)
        pub, String = self._zone_pub("/attack/rampa")
        RAMPA["seq"] += 1
        seq = RAMPA["seq"]
        m = String()
        if senza:
            m.data = json.dumps({"seq": seq, "flusso": flusso, "annulla": True})
        else:
            prec, zp = RAMPA["prec"], RAMPA["zone_prec"]
            m.data = json.dumps({"seq": seq, "flusso": flusso, "genoma_prec": [float(v) for v in prec],
                                 "genoma_sha_nuovo": sha,
                                 "zone_prec": {"path": zp["path"], "sha256": zp["sha256"], "hash": zp["hash"]}})
        ok = self._publish_confirmed(pub, m, lambda: _ric().get("seq") == seq, f"rampa {flusso} w{k}")
        ric = _ric() if ok else {}
        rec.update(seq=seq, confermata=bool(ok), stato_nodo=ric.get("stato"), zone_prec_hash=ric.get("zone_prec_hash"))
        rec.setdefault("esito", "mandata")
        RAMPA["ultimo"][flusso] = rec
        fin.append(rec)
        buono = (ric.get("stato") == RAMPA_ANNULLATA if rec["esito"] == "annullata"
                 else (ric.get("stato") == "ok" and ric.get("genoma_sha_nuovo") == sha))
        if not buono:
            raise ErroreConfigurazioneZone(f"rampa {flusso} w{k} ({sha}, {rec['esito']}): nodo "
                                           f"{ric.get('stato') or 'nessuna conferma'}")
        return rec

    def stato_rampa(self, flusso):
        """Ramp of the stream from the status (after a rollout): genome, previous, state, k and
        alpha of the last processed cloud (status at 10 Hz).
        """
        if RAMPA["R"] <= 1:
            return None
        return {"R": RAMPA["R"], "messaggio": RAMPA["ultimo"].get(flusso),
                "nodo": dict((getattr(self, "node_rampa", None) or {}).get(flusso) or {}) or None}

    def set_victim_genome(self, genome):
        self.manda_rampa("vittima", genome)
        self.set_metro("vittima")
        self.set_zone("vittima", genome)
        m = self._F32()
        m.data = [float(v) for v in genome]
        before = self.victim_genome_id
        ok = self._publish_confirmed(self._victim_genome_pub, m,
                                lambda: self.victim_genome_id is not None
                                and self.victim_genome_id != before, "genoma vittima")
        # previous ramp genome for the next metre: the last one given to the victim
        RAMPA["vittima_ultimo"] = [float(v) for v in genome]
        return ok

    def twin_replay(self, last_s=0.0, rate_hz=10.0, timeout=None):
        """Ask the node to replay the victim clouds to the twin and wait for the end (replay_seq
        incremented and replay not active). Returns (clouds replayed, last stamp, seconds taken).
        """
        from std_msgs.msg import String
        before = self.replay_seq
        m = String(); m.data = json.dumps({"last_s": float(last_s), "rate_hz": float(rate_hz)})
        t0 = time.time()
        self._twin_replay_pub.publish(m)
        t_pub = t0
        started = False
        if timeout is None:
            timeout = 30.0 + 1.5 * max(self.victim_rec_n, 1) / max(rate_hz, 0.5)
        while time.time() - t0 < timeout:
            self.spin(0.05)
            if self.replay_active:
                started = True
            if not started and time.time() - t_pub > 2.0:
                self._twin_replay_pub.publish(m); t_pub = time.time()
            if self.replay_seq is not None and before is not None and self.replay_seq > before and not self.replay_active:
                break
            if before is None and self.replay_seq is not None and self.replay_seq > 0 and started and not self.replay_active:
                break
        n = self.replay_n if self.replay_seq is not None and (before is None or self.replay_seq > before) else 0
        return n, self.replay_last_stamp, time.time() - t0

    def _call_srv(self, cli, req, timeout=20.0):
        """Call a ROS service and wait for the result spinning the executor.
        Returns the result, or None if the service is missing or the timeout expires.
        """
        if cli is None:
            return None
        if not cli.wait_for_service(timeout_sec=5.0):
            return None
        fut = cli.call_async(req)
        t0 = time.time()
        while time.time() - t0 < timeout and not fut.done():
            self.spin(0.05)
        return fut.result() if fut.done() else None

    def enable_shadow(self):
        """Subscription to the shadow pose and client of its map_save (--twin-state ombra_e), in
        namespace tv.SHADOW_NS. Created only here: without a shadow the graph is unchanged.
        """
        from nav_msgs.msg import Odometry
        self.shadow_est_raw = None     # (x, y, yaw) in the shadow MOLA frame
        self.shadow_pose6 = None       # (x, y, z, yaw, pitch, roll), radians
        self.shadow_est_stamp = None
        self.shadow_est_t = 0.0
        self.shadow_buf = deque(maxlen=50)   # (stamp, x, y, yaw) for the velocity (ombra_est)
        self.node.create_subscription(Odometry, f"{tv.SHADOW_NS}/lidar_odometry/pose",
                                      self._shadow_est_cb, 100)
        self._shadow_map_save_cli = None
        if self._map_restore_import_error is None:
            self._shadow_map_save_cli = self.node.create_client(self._MapSave, f"{tv.SHADOW_NS}/map_save")
        return self._map_restore_import_error

    def _shadow_est_cb(self, msg):
        self._n_cb += 1
        p = msg.pose.pose
        q = p.orientation
        self.shadow_pose6 = tv.pose6_from_quat(p.position.x, p.position.y, p.position.z,
                                               q.x, q.y, q.z, q.w)
        self.shadow_est_raw = (self.shadow_pose6[0], self.shadow_pose6[1], self.shadow_pose6[3])
        self.shadow_est_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.shadow_est_t = time.time()
        self.shadow_buf.append((self.shadow_est_stamp, *self.shadow_est_raw))

    def enable_shadow_check(self, recorder):
        """Subscriptions for the shadow-victim comparison (--shadow-check), separate from the
        control ones: the poses go only to the bench log.
        """
        from nav_msgs.msg import Odometry
        for who, topic in (("victim", "/victim/lidar_odometry/pose"),
                           ("shadow", f"{tv.SHADOW_NS}/lidar_odometry/pose")):
            self.node.create_subscription(
                Odometry, topic, lambda m, w=who: recorder.add_pose(w, shc.pose_record(m, time.time())), 1000)

    def shadow_map_save(self, path):
        """Shadow map_save to `path` (no extension): (success, error)."""
        cli = getattr(self, "_shadow_map_save_cli", None)
        if cli is None:
            return False, f"client assente ({self._map_restore_import_error})"
        res = self._call_srv(cli, self._MapSave.Request(map_path=str(path)))
        if res is None:
            return False, f"nessuna risposta da {tv.SHADOW_NS}/map_save"
        return bool(res.success), getattr(res, "error_message", "")

    def twin_map_save(self, path):
        """Twin map_save (default namespace, /map_save) to `path`: with MOLA_FULL_STATE=1 (variant
        G) it also writes <path>.molastate. Only for the realigned bench arm: client created on
        first use.
        """
        if getattr(self, "_twin_map_save_cli", None) is None:
            if self._map_restore_import_error is not None:
                return False, f"client assente ({self._map_restore_import_error})"
            self._twin_map_save_cli = self.node.create_client(self._MapSave, "/map_save")
        res = self._call_srv(self._twin_map_save_cli, self._MapSave.Request(map_path=str(path)))
        if res is None:
            return False, "nessuna risposta da /map_save"
        return bool(res.success), getattr(res, "error_message", "")

    def stato_generatore(self, azione, nome, flusso=None, timeout=3.0):
        """Node with --stato-generatore: "salva" snapshots the generator state of the stream (victim
        or twin) into slot `nome`; "carica" arms it for the next twin genome, which takes it
        instead of the reset. Confirmed by the status (stato_generatore.seq). None if the node
        lacks the capability (key missing from the status): no message.
        """
        if not isinstance(getattr(self, "st_stato_generatore", None), dict):
            return None
        from std_msgs.msg import String as _Str
        if getattr(self, "_stato_gen_pub", None) is None:
            self._stato_gen_pub = self.node.create_publisher(_Str, "/attack/stato_generatore", 10)
            self._stato_gen_seq = int(time.time() * 1000) % (2 ** 31)
        self._stato_gen_seq = (self._stato_gen_seq + 1) % (2 ** 31)
        seq = self._stato_gen_seq
        m = _Str()
        m.data = json.dumps({"seq": seq, "azione": azione, "nome": nome,
                             **({"flusso": flusso} if flusso else {})})
        # the node confirms the seq only for an executed action; an error goes to ultimo_errore with
        # the same seq: exit at once, unconfirmed
        errore = lambda: ((self.st_stato_generatore or {}).get("ultimo_errore") or {}).get("seq") == seq
        ok = self._publish_confirmed(self._stato_gen_pub, m,
                                     lambda: (self.st_stato_generatore or {}).get("seq") == seq or errore(),
                                     f"stato del generatore ({azione} {nome})", timeout=timeout)
        st = dict(self.st_stato_generatore or {})
        ok = bool(ok) and st.get("seq") == seq
        return {"azione": azione, "nome": nome, "flusso": flusso, "seq": seq, "confermato": ok,
                "posto": (st.get("posti") or {}).get(nome), "armato": st.get("armato"),
                "ultimo_caricato": st.get("ultimo_caricato"),
                **({"errore": st.get("ultimo_errore")} if errore() else {})}

    def victim_map_save(self, path):
        """Save the continuous victim map to `path` (no extension). (success, error)."""
        if self._victim_map_save_cli is None:
            return False, f"client assente ({self._map_restore_import_error})"
        res = self._call_srv(self._victim_map_save_cli, self._MapSave.Request(map_path=str(path)))
        if res is None:
            return False, "nessuna risposta da /victim/map_save"
        return bool(res.success), getattr(res, "error_message", "")

    def twin_map_load(self, path, mola=None):
        """Load into the twin (default namespace) the map saved in `path`. (success, error).
        If the service future does not resolve in our executor (happens with /map_load), success is
        checked by reading the twin MOLA log for '[map_load] Successful.' (up to 10 s), so no false
        failure is reported when MOLA did load.
        """
        if self._twin_map_load_cli is None:
            return False, f"client assente ({self._map_restore_import_error})"
        res = self._call_srv(self._twin_map_load_cli, self._MapLoad.Request(map_path=str(path)))
        if res is not None:
            return bool(res.success), getattr(res, "error_message", "")
        logp = getattr(mola, "log_path", None)
        if logp is not None:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                try:
                    if "[map_load] Successful." in Path(logp).read_text():
                        return True, "(confermato dal log MOLA; future del servizio non risolto)"
                except OSError:
                    pass
                self.spin(0.2)
        return False, "nessuna risposta da /map_load e log non conferma"

    def twin_relocalize(self, pose_xyth):
        """Relocalise the twin around the pose (x, y, yaw) believed by the victim by PUBLISHING on
        /initialpose: the MOLA config uses relocalize_from_topic:/initialpose and the near-pose
        service is rejected (no module accepts it). Publishes several times (QoS transient_local)
        and keeps the service as a non-blocking fallback. Returns True if published; actual
        fidelity is measured after play (diff_cm in _rollout_once). /initialpose is absolute and
        victim and attacker subscribe to it too, where it is not a no-op (it resets estimator and
        sigma): it is published on tv.TWIN_INITIALPOSE, where the twin receives it through the
        remap in its ROS_ARGS.
        """
        if self._twin_initialpose_pub is None:
            return False
        x, y, yaw = float(pose_xyth[0]), float(pose_xyth[1]), float(pose_xyth[2])
        msg = self._PoseWithCovStamped()
        # Frame: the one seen on the victim pose (odom); the twin loads the victim map, same frame.
        # Override with TWIN_STATE['reloc_frame']; if the diff stays large with "odom", try "map".
        msg.header.frame_id = TWIN_STATE.get("reloc_frame") or self.victim_est_frame_id or "odom"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.04   # x, y ~0.2 m
        cov[35] = 0.02           # yaw ~8 deg
        msg.pose.covariance = cov
        for _ in range(3):
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self._twin_initialpose_pub.publish(msg)
            self.spin(0.3)
        # Non-blocking fallback: also try the service if ready (outcome ignored).
        if self._twin_reloc_cli is not None and self._twin_reloc_cli.service_is_ready():
            try:
                self._twin_reloc_cli.call_async(self._RelocNearPose.Request(pose=msg))
                self.spin(0.05)
            except Exception:
                pass
        return True

    def set_drift_keep(self, on: bool):
        """Drift keep mode in the node, confirmed by the status (drift_keep). False if the node does
        not confirm (version without keep).
        """
        from std_msgs.msg import Bool
        if getattr(self, "_drift_keep_pub", None) is None:
            self._drift_keep_pub = self.node.create_publisher(Bool, "/attack/drift_keep", 10)
        m = Bool(); m.data = bool(on)
        return self._publish_confirmed(self._drift_keep_pub, m,
                                       lambda: self.st_drift_keep is bool(on),
                                       f"deriva conservata {'accesa' if on else 'spenta'}")

    def set_twin_drift_init(self, drift):
        """G bench (node with --banco-twin-drift): drift the twin starts from at the next genome (3
        values, m, sensor frame) or None for the usual reset. Confirmed by the status
        (twin_drift_init); False if the node lacks the option.
        """
        if getattr(self, "_twin_drift_pub", None) is None:
            self._twin_drift_pub = self.node.create_publisher(self._F32, "/attack/twin_drift_init", 10)
        m = self._F32()
        m.data = [] if drift is None else [float(v) for v in drift]
        return self._publish_confirmed(self._twin_drift_pub, m,
                                       lambda: twin_drift_confirmed(self.st_twin_drift_on,
                                                                    self.st_twin_drift_init, drift),
                                       f"deriva iniziale del gemello {list(m.data) or 'azzerata'}")

    def reset_victim_drift(self):
        from std_msgs.msg import Empty
        self._victim_drift_reset_pub.publish(Empty())
        self.spin(0.1)

    def set_attack_enabled(self, on: bool):
        """Turn the perturbation on or off without restarting the node, confirmed by the status (see
        set_genome: the first message of a just-started orchestrator can be lost and the nominal
        would run with the attack on).
        """
        from std_msgs.msg import Bool
        m = Bool()
        m.data = bool(on)
        return self._publish_confirmed(self._enable_pub, m,
                                       lambda: self.node_enabled is bool(on),
                                       f"attacco {'acceso' if on else 'spento'}")

    def abilita_registri(self, cartella):
        """Live logs (registri_dal_vivo.py): the twin and victim pose callbacks take two arguments
        to get the MessageInfo.
        """
        from src.optimization.registri_dal_vivo import RegistriDalVivo
        reg = RegistriDalVivo(cartella)
        self.registri = reg
        self._sub_gemello.callback = reg.callback_posa(
            "gemello", self._est_cb,
            lambda: self.est_t if self.est_source == "twin" else time.time(), self._t_sim)
        self._sub_vittima.callback = reg.callback_posa(
            "vittima", self._victim_est_cb, lambda: self.victim_est_t, self._t_sim)
        return reg

    def _t_sim(self):
        """Node simulated time: stamp of the last /chassis/odom received."""
        return self.true_buf[-1][0] if self.true_buf else float("nan")

    def _registra_ciclo(self, t_ciclo, t_pub, est, est_dist, v, w, fase):
        reg = self.registri
        try:
            ruolo = "vittima" if self.est_source == "victim" else "gemello"
            t_sim = self._t_sim()
        except Exception as e:
            reg._disattiva("controllore.csv", e)
            return
        # raw pose stamp (clock of the stream MOLA) and applied offset: correct age without
        # reconstruction
        grezzo = getattr(self, "est_stamp_raw", None)
        try:
            offset = self.est_stamp - grezzo
        except TypeError:
            offset = None
        reg.ciclo(ruolo, t_ciclo, t_sim, self.est_stamp, self.est_t, est,
                  est_dist, v, w, t_pub, fase, stamp_grezzo=grezzo, offset=offset)

    def _stop_registrato(self, fase="stop", est0=None):
        """stop_robot with the controller line if the logs are on."""
        if self.registri is None:
            self.stop_robot()
            return
        t_c = time.time()
        self.stop_robot()
        t_pub = time.time()
        est = self.est
        try:
            ed = (math.hypot(est[0] - est0[0], est[1] - est0[1])
                  if est is not None and est0 is not None else float("nan"))
        except Exception:
            ed = float("nan")
        self._registra_ciclo(t_c, t_pub, est, ed, 0.0, 0.0, fase)

    def stop_robot(self):
        self.cmd_pub.publish(self._Twist())

    def halt(self, sec: float = 0.5):
        """Null command repeated and delivered.

        The Isaac Sim differential_drive applies the last /cmd_vel received with no timeout: a
        single message published right before closing the node may not go out, and the robot keeps
        moving even after set_pose and into the next run.
        """
        t_end = time.time() + sec
        while time.time() < t_end:
            self.stop_robot()
            self.spin(0.05)

    def _trace_row(self, x, y, yaw, local_wp, v, w, travelled, start_pose, c0, s0, t0,
                   fase="", est_dist=float("nan")):
        """One trace row: MOLA estimate, true pose and commands on the same time axis."""
        if self.trace_rows is None:
            return
        tp = self._last_true if self._last_true else (float("nan"),) * 3
        est_wx = start_pose[0] + c0 * x - s0 * y
        est_wy = start_pose[1] + s0 * x + c0 * y
        ta = self.true_at(self.est_stamp) if self.true_source == "ros" else None
        self.trace_rows.append({
            "t": round(time.time() - t0, 3),
            "est_x": round(x, 4), "est_y": round(y, 4),
            "est_yaw_deg": round(math.degrees(yaw), 2),
            "est_wx": round(est_wx, 4), "est_wy": round(est_wy, 4),
            "true_x": round(tp[0], 4), "true_y": round(tp[1], 4),
            "true_yaw_deg": round(math.degrees(tp[2]), 2),
            "wp_x": round(local_wp[0], 3), "wp_y": round(local_wp[1], 3),
            "cmd_v": round(v, 3), "cmd_w": round(w, 3),
            "travelled": round(travelled, 4),
            "est_true_gap": round(math.hypot(est_wx - tp[0], est_wy - tp[1]), 4)
            if tp[0] == tp[0] else float("nan"),
            "est_true_gap_al": (round(math.hypot(est_wx - ta[0], est_wy - ta[1]), 4)
                                if ta is not None else float("nan")),
            "pose_quality": (round(self.pose_quality, 4)
                             if self.pose_quality == self.pose_quality else float("nan")),
            "chamfer_cm": (round(self.chamfer[-1], 3) if self.chamfer else float("nan")),
            "drift_cm": round(self.drift_cm, 3),
            "drift_xy_cm": round(self.drift_xy_cm, 3),
            "frame_idx": self.node_frame_idx,
            "true_z": round(self.true_z, 4), "est_z": round(self.est_z, 4),
            "true_vx": round(self.true_v[0], 4), "true_vy": round(self.true_v[1], 4),
            "wall": round(time.time(), 3),
            "fase": fase,
            "est_dist": round(est_dist, 4) if est_dist == est_dist else float("nan"),
        })

    def _coast(self, isaac, start_pose, true_start, c0, s0, t0, catchup_sec,
               local_wp, rate=20.0, tol=0.03, need=3, est0=None):
        """After the stop: keep logging ticks and estimate while the robot brakes and MOLA catches
        up, until the estimate tracks the truth within `tol` for `need` scans or catchup_sec
        expires. So final estimate and truth are at the same instant and MOLA has arrived.
        Updates the final true pose in _last_true and returns the true metres travelled.

        With --window-stop ramp it closes instead when the robot is still (true speed below 0.02 m/s
        for 3 ticks) and the estimate is later than the stop, with a 2 s cap. In both modes it
        records in last_stop_info the stop (first tick below 0.02 m/s) and the end.
        """
        dt = 1.0 / rate
        info = self.last_stop_info if self.last_stop_info is not None else {}
        ramp = info.get("stop_mode") == "ramp"
        t_c0 = time.time()
        t_end = t_c0 + (wstop.COAST_CAP_S if ramp else catchup_sec)
        hits = 0
        still = 0
        arrest_wall = arrest_stamp = None
        prev_p = prev_pt = None
        reason = "tetto" if ramp else "scadenza"
        travelled = float(np.linalg.norm(np.array(self._last_true[:2]) - true_start)) \
            if self._last_true else 0.0
        while time.time() < t_end:
            self._stop_registrato("fermo" if arrest_wall is not None else "coast", est0)
            self.drain()
            try:
                p = self.true_pose(isaac)
                self._last_true = p
                travelled = float(np.linalg.norm(np.array(p[:2]) - true_start))
            except IsaacStall:
                raise
            except Exception:
                p = self._last_true
            now = time.time()
            spd = self._true_speed(p, prev_p, (now - prev_pt) if prev_pt else 0.0)
            prev_p, prev_pt = p, now
            if self.est is not None and p is not None:
                x, y, _ = self.est
                est_wx = start_pose[0] + c0 * x - s0 * y
                est_wy = start_pose[1] + s0 * x + c0 * y
                gap = math.hypot(est_wx - p[0], est_wy - p[1])
                ed = math.hypot(x - est0[0], y - est0[1]) if est0 is not None else float("nan")
                if arrest_wall is None and spd == spd and spd < wstop.STILL_V:
                    arrest_wall = now
                    arrest_stamp = (self.true_buf[-1][0]
                                    if self.true_source == "ros" and self.true_buf else None)
                    info["est_dist_at_stop_m"] = ed
                    info["true_dist_at_stop_m"] = travelled
                info["est_dist_end_m"] = ed
                info["true_dist_end_m"] = travelled
                self._trace_row(x, y, self.est[2], local_wp, 0.0, 0.0, travelled,
                                start_pose, c0, s0, t0,
                                fase=("fermo" if arrest_wall is not None else "coast"),
                                est_dist=ed)
                if ramp:
                    still = wstop.still_count(still, spd)
                    nuova = wstop.est_after_stop(getattr(self, "est_stamp", float("nan")),
                                                 arrest_stamp, self.est_t, arrest_wall)
                    r = wstop.coast_end_ramp(now - t_c0, still, nuova)
                    if r is not None:
                        reason = r
                        break
                else:
                    hits = hits + 1 if gap < tol else 0
                    if hits >= need:
                        reason = "gap_stima_vera"
                        break
            time.sleep(dt)
        info["coast_end_reason"] = reason
        return travelled

    def _true_speed(self, p, prev_p, dt_p):
        """True speed (m/s) from the /chassis/odom twist; without the topic, by difference between
        two read poses (NaN if not computable).
        """
        if self.true_source == "ros" and self.true is not None:
            v = math.hypot(self.true_v[0], self.true_v[1])
            if v == v:
                return v
        if p is None or prev_p is None or dt_p <= 0:
            return float("nan")
        return math.hypot(p[0] - prev_p[0], p[1] - prev_p[1]) / dt_p

    def _true_dist_now(self, true_start):
        """True distance from the start point with the last pose already received."""
        p = self.true if (self.true_source == "ros" and self.true is not None) else self._last_true
        if p is None:
            return float("nan")
        return float(np.linalg.norm(np.array(p[:2]) - true_start))

    def _stop_and_coast(self, isaac, start_pose, true_start, c0, s0, t0, catchup_sec,
                        local_wp, est0, est_dist, est_pair, travelled, horizon):
        """Null command at the closing condition, record of the condition, coast and offsets from H
        in last_stop_info.
        """
        self._stop_registrato("stop", est0)
        info = self.last_stop_info
        info["est_dist_at_cond_m"] = est_dist
        info["true_dist_at_cond_m"] = self._true_dist_now(true_start)
        info["tau_cond_s"] = time.time() - t0
        prev_xy, last_xy = est_pair
        if prev_xy is not None and last_xy is not None:
            info["est_jump_at_cond_cm"] = 100.0 * math.hypot(last_xy[0] - prev_xy[0],
                                                             last_xy[1] - prev_xy[1])
        if catchup_sec > 0:
            travelled = self._coast(isaac, start_pose, true_start, c0, s0,
                                    t0, catchup_sec, local_wp, est0=est0)
        else:
            info["est_dist_end_m"] = est_dist
            info["true_dist_end_m"] = travelled
            info["coast_end_reason"] = "nessun_coast"
        info["drive_s"] = time.time() - t0
        wstop.finalize_stop_info(info, horizon)
        return travelled

    def wait_converged(self, isaac, max_sec, tol=0.015, need=4, rate=20.0):
        """Wait, with the robot still, until the MOLA estimate is stable (displacement between
        consecutive scans < tol) for `need` scans: MOLA has locked on before starting. Capped at
        max_sec. Returns the seconds waited and whether it converged.
        """
        dt = 1.0 / rate
        t0 = time.time()
        prev = self.est
        prev_t = self.est_t
        hits = 0
        while time.time() - t0 < max_sec:
            self.drain()
            if self.est is not None and self.est_t != prev_t:
                if prev is not None:
                    d = math.hypot(self.est[0] - prev[0], self.est[1] - prev[1])
                    hits = hits + 1 if d < tol else 0
                prev = self.est
                prev_t = self.est_t
                if hits >= need:
                    return time.time() - t0, True
            time.sleep(dt)
        return time.time() - t0, False

    def drive(self, local_wps, distance, isaac: IsaacClient,
              mola: MolaRunner, timeout=45.0, rate=20.0, catchup_sec=0.0,
              check_every=None, over_run=1.30, arrival_tol=ARRIVAL_TOL,
              stop_on="true", true_cap=TRUE_CAP, frame_pose=None):
        """Drive the robot up to `distance` true metres from the start point.

        The stop uses the Isaac Sim true pose, not the MOLA estimate: stopping when MOLA believes it
        has covered the stretch makes the fitness unbounded, because an attack that makes the
        translation underestimated keeps the robot moving and the deviation grows by itself. The
        true pose is read from the file the server rewrites at every frame, so at every control
        tick; if the server does not publish it, the blocking request is used every check_every
        seconds (0.25 s: 20 cm between reads at 0.8 m/s).

        local_wps is the list of remaining waypoints in the MOLA frame: when the first is within
        arrival_tol the next one is taken without stopping; `arrived` only on the last.

        Outcomes: ok, arrived (last waypoint within arrival_tol), overrun (stretch exceeded by the
        over_run factor without stopping), timeout, mola_crash, pose_stale, no_pose. Returns
        (outcome, metres travelled).

        With --window-stop ramp and stop_on "estimate" the speed decreases in the last metres with
        the remaining estimated distance (window_stop.py); the stop condition is the same. With
        goal_mode "direction" local_wps only gives the line (first to last point; with a single
        point, from the initial estimate to the point): the follower chases at every tick the carrot
        direction_lookahead metres beyond the projection of the estimate on the line, there is no
        "arrived" and the window closes only on the distance stop (or the true cap).
        """
        direction = getattr(self, "goal_mode", "waypoints") == "direction"
        mode = self.window_stop if stop_on == "estimate" else "legacy"
        ramp = mode == "ramp"
        self.last_stop_info = wstop.new_stop_info(mode, self.stop_ramp_k, self.stop_ramp_vmin)
        self.est = None
        t_wait = time.time()
        while self.est is None and time.time() - t_wait < 15.0:
            self._exec.spin_once(timeout_sec=0.05)
            self.check_stall()
            if mola.crashed():
                self.last_stop_reason = "mola_crash"
                return "mola_crash", 0.0
        if self.est is None:
            self.last_stop_reason = "no_pose"
            return "no_pose", 0.0

        ros_pose = self.true_source == "ros" and self.true is not None \
            and time.time() - self.true_t < 1.0
        streamed = ros_pose or isaac.stream_available()
        if check_every is None:
            check_every = 0.0 if streamed else 0.25
        if not streamed and not self._warned_no_stream:
            print("    [nota] posa vera ne' da /chassis/odom ne' in streaming: "
                  "letta dal server ogni 0.25 s")
            self._warned_no_stream = True

        # True pose for the measurement (travelled distance); control frame (origin of the MOLA
        # estimate in the world) equal to the true pose, or to the believed pose (attacker or
        # victim) with frame_pose.
        tp = self.true_pose(isaac)
        true_start = np.array(tp[:2])
        self._last_true = tp
        start_pose = frame_pose if frame_pose is not None else tp
        # Initial estimate: with --stop-on estimate the stretch closes when the distance ESTIMATED
        # by MOLA reaches the horizon (removing from the attack the advantage of our true distance
        # measurement).
        est0 = np.array(self.est[:2]) if self.est is not None else np.zeros(2)
        # The MOLA estimate starts at (0,0,0) in the initial true pose: for the estimate/truth
        # comparison in the trace it is mapped to the world frame.
        c0, s0 = math.cos(start_pose[2]), math.sin(start_pose[2])
        dt = 1.0 / rate
        t0 = time.time()
        t_last_check = -1.0
        travelled = 0.0
        self.last_wps_reached = 0      # waypoints passed in this drive
        wps = [tuple(w) for w in local_wps]
        local_wp = wps[0]
        if direction:
            look = float(getattr(self, "direction_lookahead", DIRECTION_LOOKAHEAD))
            line_p0, line_u = direction_line(wps, est0)
            local_wp = direction_carrot(est0, line_p0, line_u, look)
        # Last two distinct estimates (jump at the condition) and ramp of the previous tick.
        est_seen_t = None
        est_prev_xy = est_last_xy = None
        v_ramp_prev = None

        while time.time() - t0 < timeout:
            self.drain()
            if mola.crashed():
                self._stop_registrato("stop", est0)
                self.last_stop_reason = "mola_crash"
                return "mola_crash", travelled
            if self.est is None or time.time() - self.est_t > 3.0:
                self._stop_registrato("stop", est0)
                self.last_stop_reason = "pose_stale"
                return "pose_stale", travelled

            x, y, yaw = self.est
            if self.est_t != est_seen_t:
                est_seen_t = self.est_t
                est_prev_xy, est_last_xy = est_last_xy, (x, y)
            est_dist = float(math.hypot(x - est0[0], y - est0[1]))

            # Stop on the estimated distance (every tick) with a cap on the truth at true_cap x
            # horizon for cases where the estimate freezes.
            if stop_on == "estimate":
                if est_dist >= distance:
                    travelled = self._stop_and_coast(
                        isaac, start_pose, true_start, c0, s0, t0, catchup_sec, local_wp,
                        est0, est_dist, (est_prev_xy, est_last_xy), travelled, distance)
                    # The estimate reached the horizon: the robot believes it has arrived. If the
                    # true distance differs, that is the deception.
                    self.last_stop_reason = "stima_orizzonte"
                    return "ok", travelled

            # Waypoint within tolerance: the follower commands zero speed and the stretch would
            # never complete. In direction mode the carrot stays direction_lookahead metres ahead:
            # no arrival.
            if direction:
                local_wp = direction_carrot((x, y), line_p0, line_u, look)
            elif math.hypot(local_wp[0] - x, local_wp[1] - y) < arrival_tol:
                if len(wps) > 1:
                    wps.pop(0)
                    local_wp = wps[0]
                    self.last_wps_reached += 1
                else:
                    travelled = self._stop_and_coast(
                        isaac, start_pose, true_start, c0, s0, t0, catchup_sec, local_wp,
                        est0, est_dist, (est_prev_xy, est_last_xy), travelled, distance)
                    self.last_stop_reason = "arrivato_waypoint"
                    return "arrived", travelled

            now = time.time() - t0
            if now - t_last_check >= check_every:
                t_last_check = now
                try:
                    p = self.true_pose(isaac) if streamed else isaac.pose()
                    self._last_true = p
                    travelled = float(np.linalg.norm(
                        np.array(p[:2]) - true_start))
                except IsaacStall:
                    raise
                except Exception:
                    pass
                if stop_on == "estimate":
                    if travelled >= distance * true_cap:
                        travelled = self._stop_and_coast(
                            isaac, start_pose, true_start, c0, s0, t0, catchup_sec, local_wp,
                            est0, est_dist, (est_prev_xy, est_last_xy), travelled, distance)
                        self.last_stop_reason = "tetto_vero"
                        return "est_capped", travelled
                else:
                    if travelled >= distance:
                        travelled = self._stop_and_coast(
                            isaac, start_pose, true_start, c0, s0, t0, catchup_sec, local_wp,
                            est0, est_dist, (est_prev_xy, est_last_xy), travelled, distance)
                        self.last_stop_reason = "distanza_vera"
                        return "ok", travelled
                    if travelled >= distance * over_run:
                        self._stop_registrato("stop", est0)
                        self.last_stop_reason = "overrun"
                        return "overrun", travelled

            t_ciclo = time.time() if self.registri is not None else 0.0
            v, w, _ = self.follower.command(x, y, yaw, local_wp[0], local_wp[1])
            fase = "crociera"
            if ramp:
                # Before the trace row, so cmd_v records the ramp.
                v, v_ramp_prev, fase = wstop.ramp_command(
                    v, est_dist, distance, self.stop_ramp_k, self.stop_ramp_vmin,
                    time.time() - self.est_t, v_ramp_prev)
            self._trace_row(x, y, yaw, local_wp, v, w, travelled, start_pose, c0, s0, t0,
                            fase=fase, est_dist=est_dist)

            m = self._Twist()
            m.linear.x = v
            m.angular.z = w
            self.cmd_pub.publish(m)
            if self.registri is not None:
                self._registra_ciclo(t_ciclo, time.time(), (x, y, yaw), est_dist, v, w, fase)
            time.sleep(dt)

        self._stop_registrato("stop", est0)
        self.last_stop_reason = "timeout"
        return "timeout", travelled

    def close(self):
        try:
            self.halt()
        except Exception:
            pass
        if self.registri is not None:
            self.registri.close()
        try:
            self._exec.shutdown()
            self.node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# One rollout
# ---------------------------------------------------------------------------

def to_local(goal, true_pose):
    """Global waypoint expressed in the frame of the just-relocalised MOLA.

    At every restart MOLA puts its origin at the current robot pose, so the goal coordinates change
    at every rollout.
    """
    gx, gy = goal
    x, y, yaw = true_pose
    dx, dy = gx - x, gy - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return (dx * c - dy * s, dx * s + dy * c)


def direction_line(wps, origin):
    """Line for --goal-mode direction: start point and unit vector. From the first to the last point
    of wps; with a single point, from origin to the point. The goal distance does not matter:
    only the direction counts.
    """
    p0 = np.asarray(wps[0] if len(wps) > 1 else origin, dtype=np.float64)[:2]
    d = np.asarray(wps[-1], dtype=np.float64)[:2] - p0
    n = float(np.hypot(d[0], d[1]))
    if not n > 1e-9:
        raise ValueError("--goal-mode direction: bersaglio coincidente con l'inizio della retta")
    return p0, d / n


def direction_carrot(xy, p0, u, lookahead):
    """Carrot `lookahead` metres beyond the projection of xy on the line p0 + s u."""
    s = float((xy[0] - p0[0]) * u[0] + (xy[1] - p0[1]) * u[1])
    return (float(p0[0] + (s + lookahead) * u[0]), float(p0[1] + (s + lookahead) * u[1]))


def direction_window_count(history, k):
    """--goal-mode direction: driven and held windows among indices 0..k-1. A held window
    (victim_held, --victim-never-clean without genome) uses an index but does not move the robot,
    so it does not count towards --max-windows.
    """
    held = {r.get("window") for r in history if r.get("victim_held")}
    n_held = sum(1 for w in held if isinstance(w, int) and 0 <= w < k)
    return k - n_held, n_held


def spegni_attacco_fine_run(node):
    """End of run: turn off the twin attack (/attack/enabled False, confirmed by the status) before
    closing the node and putting Isaac back in play. The victim finish() closes the victim stream
    and attack but leaves the last twin genome on: without this step a cloud arriving after the
    last rollout would go out perturbed on the main topic. Called only in the finally of main,
    after the last window: nothing changes during the run. Returns True if the node confirmed.
    """
    try:
        return bool(node.set_attack_enabled(False))
    except Exception as e:
        print(f"    [nota] spegnimento dell'attacco a fine run non riuscito: {e}")
        return False


def direction_run_end(history, k, max_windows, extra_windows):
    """Reason for ending the run with --goal-mode direction before window k, None to go on:
    --max-windows driven windows, or more than --extra-windows held windows (incomplete run).
    """
    driven, _ = direction_window_count(history, k)
    if driven >= max_windows:
        return "finestre_completate"
    if k >= max_windows + max(0, extra_windows):
        return "limite_finestre_trattenute"
    return None


def check_goal_mode_args(args, follower=None):
    """Options of --goal-mode direction: error message or None."""
    if getattr(args, "goal_mode", "waypoints") != "direction":
        return None
    follower = follower or FollowerModel()
    bad = []
    if args.max_windows is None or args.max_windows < 1:
        bad.append("--max-windows >= 1 obbligatorio")
    if args.stop_on != "estimate":
        bad.append("--stop-on estimate")
    if args.window_stop != "ramp":
        bad.append("--window-stop ramp (tutte le finestre si chiudono con la rampa)")
    if args.goals and len([g for g in args.goals.split(";") if g.strip()]) > 1:
        bad.append("un solo bersaglio (--goals con piu' waypoint)")
    # below max_speed / k_lin the follower would slow down towards the carrot
    l_min = follower.max_speed / follower.k_lin
    if not args.direction_lookahead >= l_min:
        bad.append(f"--direction-lookahead >= {l_min:.2f} m (follower senza rallentamento)")
    if bad:
        return "--goal-mode direction richiede: " + "; ".join(bad)
    return None


def direction_window_goal(start_xy, u, saved_xy, lookahead):
    """Window goal with --goal-mode direction (damage, directional term, print): the carrot of the
    saved true pose, so it stays ahead of the robot even past the given goal.
    """
    return direction_carrot(saved_xy, np.asarray(start_xy, dtype=np.float64), u, lookahead)


# ROS arguments for the victim MOLA: namespace /victim, so it publishes /victim/lidar_odometry/pose
# next to the twin without touching the yaml files (the bridge reads ROS_ARGS from the environment).
VICTIM_ROS_ARGS = '["--ros-args", "-r", "__ns:=/victim"]'
VICTIM_TOPIC = "/carter/lidar_victim"
ATTACKER_ROS_ARGS = '["--ros-args", "-r", "__ns:=/attacker"]'
ATTACKER_TOPIC = "/carter/lidar_attacker"


class VictimRunner(MolaRunner):
    """Continuous victim MOLA: one instance for the whole run, never restarted.

    Listens to /carter/lidar_victim, which perturbation_node feeds only in the applied stretches:
    during the search (twins restarted, state restored) it receives nothing and stays where the
    robot will be restored. Local map and adaptive threshold build up along the path, as on the
    robot.
    """

    def __init__(self, script: str, log_dir: Path, traces: bool = False, ros_args=None, env=None):
        super().__init__(script, log_dir, ready_timeout=60.0,
                         extra_env={**(env or {}), "ROS_ARGS": ros_args or VICTIM_ROS_ARGS},
                         traces=traces, args=[VICTIM_TOPIC])
        self.settle_s = float("nan")
        self.settle_ok = None
        self.run_origin = None      # true pose at start: origin of the victim frame
        self.origin_info = None     # offset of the two origin rules (history: run_origin)
        self.last_est_end = None    # raw estimate at the last tick of the previous stretch
        self.sync_info = None       # synchronous start with the shadow (--shadow-check)
        self.started = False        # victim MOLA initialised (start or deferred start)
        self.deferred = None        # --victim-start w0: shadow, attacker, origin rule
        self.start_info = None      # --victim-start w0: window, clouds and duration of the start

    def spawn_deferred(self, node, shadow):
        """--victim-start w0: start victim (and shadow) with the stream closed and the simulation
        stopped, wait for the topic subscriptions (at most SYNC_SUB_TIMEOUT_S). The first cloud
        after opening is the first both MOLA instances see. With --sync-wait publisher also wait
        for the readers matched to the node publishers (sync_wait_publisher).
        """
        node.set_victim_stream(False)
        need = {VICTIM_TOPIC: ["/victim"]}
        if shadow is not None:
            need.setdefault(shadow.args[0], []).append(tv.SHADOW_NS)
        pub_need = publisher_needs(need) if SYNC_WAIT["mode"] == "publisher" else None
        pub_base = matched_base(node, pub_need) if pub_need else None
        self.spawn("victim")
        if shadow is not None:
            shadow.spawn("shadow")
        t0 = time.time()
        while True:
            node.drain()
            for r in (self, shadow):
                if r is not None and not r.alive():
                    raise RuntimeError(f"mola-cli terminato prima dell'iscrizione ({r.log_path})")
            missing = {t: shc.missing_subscribers(node.node.get_subscriptions_info_by_topic(t), ns)
                       for t, ns in need.items()}
            missing = {t: m for t, m in missing.items() if m}
            if not missing:
                break
            if time.time() - t0 > SYNC_SUB_TIMEOUT_S:
                raise RuntimeError(f"avvio rimandato: iscrizioni mancanti dopo {SYNC_SUB_TIMEOUT_S:.0f} s: {missing}")
            node.spin(0.1)
        t_sub = time.time() - t0
        pub = (sync_wait_publisher(node, pub_need, pub_base, (self, shadow), t0)
               if pub_need else None)
        return {"iscrizioni_s": round(t_sub, 3), "timeout_s": SYNC_SUB_TIMEOUT_S,
                "topic": need, "rimandato": True, **({"sync_wait": pub} if pub else {})}

    def set_deferred_origins(self, node, true_now, shadow, origin_mode):
        """--victim-start w0: victim and shadow origins after settling, with the rules of start();
        the attacker one is already fixed at run start (attacker_phase_start).
        """
        v_est = node.est_raw
        if origin_mode == "estimate" and v_est is not None:
            self.run_origin = wrec.origin_from_estimate(true_now, v_est)
        else:
            self.run_origin = true_now
        self.origin_info = dict(self.origin_info or {})
        self.origin_info["victim"] = wrec.origin_record(origin_mode, true_now, v_est, self.run_origin)
        if shadow is not None:
            s_est = getattr(node, "shadow_est_raw", None)
            if origin_mode == "estimate" and s_est is not None:
                shadow.run_origin = wrec.origin_from_estimate(true_now, s_est)
            else:
                shadow.run_origin = self.run_origin
            self.origin_info["shadow"] = wrec.origin_record(origin_mode, true_now, s_est, shadow.run_origin)

    def _start_with_shadow(self, node, shadow):
        """Synchronous start of victim and shadow (--shadow-check): stream closed, spawn both, wait
        until both are subscribed to their topics (endpoints in namespaces /victim and
        tv.SHADOW_NS, at most SYNC_SUB_TIMEOUT_S), then open the stream: the first cloud is the
        same for both. With --sync-wait publisher also wait for the readers matched to the node
        publishers; with the guard (sync_guard_active) the first processed stamp is compared at
        once.
        """
        node.set_victim_stream(False)
        need = {}
        need.setdefault(VICTIM_TOPIC, []).append("/victim")
        need.setdefault(shadow.args[0], []).append(tv.SHADOW_NS)
        pub_need = publisher_needs(need) if SYNC_WAIT["mode"] == "publisher" else None
        pub_base = matched_base(node, pub_need) if pub_need else None
        self.spawn("victim")
        shadow.spawn("shadow")
        t0 = time.time()
        while True:
            node.drain()
            for r in (self, shadow):
                if not r.alive():
                    raise RuntimeError(f"mola-cli terminato prima dell'iscrizione ({r.log_path})")
            missing = {t: shc.missing_subscribers(node.node.get_subscriptions_info_by_topic(t), ns)
                       for t, ns in need.items()}
            missing = {t: m for t, m in missing.items() if m}
            if not missing:
                break
            if time.time() - t0 > SYNC_SUB_TIMEOUT_S:
                raise RuntimeError(f"avvio sincrono: iscrizioni mancanti dopo {SYNC_SUB_TIMEOUT_S:.0f} s: {missing}")
            node.spin(0.1)
        t_sub = time.time() - t0
        pub = (sync_wait_publisher(node, pub_need, pub_base, (self, shadow), t0)
               if pub_need else None)
        node.set_victim_stream(True)
        self.wait_ready(check=node.drain)
        shadow.wait_ready(check=node.drain)
        shadow.wait_pose(node)
        print(f"  avvio sincrono vittima-ombra: iscritte in {t_sub:.2f} s, flusso aperto dopo"
              + (f" ({pub['aggancio_s']:.2f} s ai publisher del nodo)" if pub else ""))
        info = {"iscrizioni_s": round(t_sub, 3), "timeout_s": SYNC_SUB_TIMEOUT_S, "topic": need,
                **({"sync_wait": pub} if pub else {})}
        if sync_guard_active():
            g = sync_guard(node, self, shadow)
            info["guardia"] = g
            self.sync_info = info
            if g["esito"] == "diverso":
                node.set_victim_stream(False)
                raise ShadowSyncFail(f"primo stamp diverso: vittima {g['vittima']}, ombra {g['ombra']} "
                                     f"({g['fonte']})")
        return info

    def start(self, isaac, node, settle_sec: float, attacker=None, origin_mode="true", shadow=None,
              sync_shadow=False):
        """Single start with the simulation running: stream open in passthrough, relocalisation,
        estimate convergence with the robot still. The attacker MOLA and the shadow (--twin-state
        ombra_e), if present, start in the same stream; the shadow takes its origin with the
        attacker rule.

        origin_mode "true" (default): victim frame origin = true pose read after convergence, also
        copied to the attacker; the drift with the stream open recorded by the estimate is counted
        twice (+0.4..+1.1 cm in w0 of the live baselines). "estimate": origin = true pose composed
        with the inverse of the victim estimate at the same instant; the attacker has its own, from
        its estimate. In both cases origin_info records the offset between the two rules at the read
        instant.
        """
        node.set_victim_attack(False)
        if shadow is not None and sync_shadow:
            self.sync_info = self._start_with_shadow(node, shadow)
        else:
            node.set_victim_stream(True)
            self.spawn("victim")
            self.wait_ready(check=node.drain)
        if attacker is not None:
            attacker.start(node, None)
        if shadow is not None and not sync_shadow:
            shadow.start(node)
        node.est_source = "victim"
        node.est = None
        node.est_t = 0.0
        deadline = time.time() + 15.0
        while time.time() < deadline and node.est is None:
            node.spin(0.1)
        if node.est is None:
            raise RuntimeError("la MOLA vittima non pubblica su /victim/lidar_odometry/pose")
        self.settle_s, self.settle_ok = node.wait_converged(isaac, settle_sec)
        true_now = node.settled_pose(isaac)
        # Estimates read right after the true pose: robot still, up to the drift (0.6-0.8 mm/s)
        # between two messages.
        v_est = node.est_raw
        a_est = node.attacker_est_raw if attacker is not None else None
        if origin_mode == "estimate" and v_est is not None:
            self.run_origin = wrec.origin_from_estimate(true_now, v_est)
        else:
            self.run_origin = true_now
        self.origin_info = {"victim": wrec.origin_record(origin_mode, true_now, v_est, self.run_origin)}
        if attacker is not None:
            if origin_mode == "estimate" and a_est is not None:
                attacker.run_origin = wrec.origin_from_estimate(true_now, a_est)
            else:
                attacker.run_origin = self.run_origin
            self.origin_info["attacker"] = wrec.origin_record(origin_mode, true_now, a_est,
                                                              attacker.run_origin)
        if shadow is not None:
            s_est = getattr(node, "shadow_est_raw", None)
            if origin_mode == "estimate" and s_est is not None:
                shadow.run_origin = wrec.origin_from_estimate(true_now, s_est)
            else:
                shadow.run_origin = self.run_origin
            self.origin_info["shadow"] = wrec.origin_record(origin_mode, true_now, s_est, shadow.run_origin)
        for who, oi in self.origin_info.items():
            d0 = (oi["offset"] or {}).get("d_cm")
            d1 = (oi["offset_other_mode"] or {}).get("d_cm")
            print(f"  origine {who} ({origin_mode}): offset stima-vero {d0} cm; con l'altra regola {d1} cm")
        node.set_victim_stream(False)
        node.last_victim_close_wall = time.time()
        node.est_source = "twin"
        self.started = True
        print(f"  vittima continua avviata: convergenza in {self.settle_s:.2f} s "
              f"({'ok' if self.settle_ok else 'non convergiuta'})"
              + ("; MOLA dell'attaccante avviata" if attacker is not None else ""))


class AttackerRunner(MolaRunner):
    """Attacker MOLA: continuous like the victim, on the clean cloud (/carter/lidar_attacker, same
    gating), namespace /attacker. Its estimate is the pose the attacker believes true: with
    --twin-align attacker the search twins start from there, not from the simulator true pose.
    """

    def __init__(self, script: str, log_dir: Path, traces: bool = False, ros_args=None, env=None):
        super().__init__(script, log_dir, ready_timeout=60.0,
                         extra_env={**(env or {}), "ROS_ARGS": ros_args or ATTACKER_ROS_ARGS},
                         traces=traces, args=[ATTACKER_TOPIC])
        self.run_origin = None

    def start(self, node, run_origin):
        """Call with the victim stream open (clouds arriving)."""
        self.spawn("attacker")
        self.wait_ready(check=node.drain)
        deadline = time.time() + 15.0
        while time.time() < deadline and node.attacker_est_raw is None:
            node.spin(0.1)
        if node.attacker_est_raw is None:
            raise RuntimeError("la MOLA dell'attaccante non pubblica su /attacker/lidar_odometry/pose")
        self.run_origin = run_origin

    def believed_pose(self, node):
        """Robot pose according to the attacker, in the working frame."""
        if node.attacker_est_raw is None or self.run_origin is None:
            return None
        return node.compose(self.run_origin, node.attacker_est_raw)


# Shadow quiet before map_save: no new pose for this long (10 scans at 10 Hz); the victim stream is
# already closed and Isaac paused.
SHADOW_QUIET_S = 1.0
# Synchronous start (--shadow-check): maximum wait for victim and shadow subscriptions
SYNC_SUB_TIMEOUT_S = 30.0
# --sync-wait publisher: margin after the node publishers match (reader side) and maximum wait for
# the base reading from the status
SYNC_PUB_MARGIN_S = 0.5
SYNC_BASE_TIMEOUT_S = 3.0
# First-stamp guard: maximum wait for the first pose of both
SYNC_GUARD_TIMEOUT_S = 5.0
# Wait mode (--sync-wait), guard (--sync-guard) and comparison recorder, set by main.
SYNC_WAIT = {"mode": "graph", "guard": False, "recorder": None}
# --origine-coerente-vittima: origin of the estimate and victim control frame from the same est_raw
# reading, as for the twin
ORIGINE_VITTIMA = {"coerente": False}
# node status keys with the matched readers (--report-matched)
MATCHED_KEY_VICTIM = "victim_pub_matched"
MATCHED_KEY_SHADOW = "shadow_pub_matched"
# main node publisher (--output-topic, /carter/lidar_perturbed): the twin topic
MATCHED_KEY_MAIN = "main_pub_matched"


def sync_guard_active():
    return SYNC_WAIT["mode"] == "publisher" or bool(SYNC_WAIT["guard"])


def publisher_needs(need, victim_topic=VICTIM_TOPIC):
    """Extra readers expected per status key: on the victim topic one per subscribed namespace
    (victim and shadow, if the shadow shares it), on the separate shadow topic (--shadow-topic)
    one.
    """
    out = {}
    for t, ns in need.items():
        k = MATCHED_KEY_VICTIM if t == victim_topic else MATCHED_KEY_SHADOW
        out[k] = out.get(k, 0) + len(ns)
    return out


def matched_base(node, pub_need, timeout_s=None, clock=time.time):
    """Base counts of the matched readers, from the node status, before the spawn. RuntimeError if
    the status does not report them.
    """
    timeout_s = SYNC_BASE_TIMEOUT_S if timeout_s is None else timeout_s
    # fresh status before reading the base: the cached value may be old (status at 10 Hz,
    # perturbation_node.py:598)
    node.spin(0.3)
    t0 = clock()
    while True:
        base = {k: getattr(node, k, None) for k in pub_need}
        if all(v is not None for v in base.values()):
            return {k: int(v) for k, v in base.items()}
        if clock() - t0 > timeout_s:
            raise RuntimeError(f"--sync-wait publisher: lo status del nodo non riporta {sorted(pub_need)} "
                               f"dopo {timeout_s:.0f} s (perturbation_node.py va lanciato con --report-matched)")
        node.spin(0.1)


def wait_publisher_matched(read, base, pub_need, spin, timeout_s=None, margin_s=None,
                           clock=time.time, check=None, t0=None):
    """--sync-wait publisher: wait until for every key k the matched readers (read(k), None if not
    reported) reach base[k] + pub_need[k], then margin_s. t0: start of the wait (spawn), from
    which the timeout counts. check() at every loop (processes alive). RuntimeError on expiry.
    """
    timeout_s = SYNC_SUB_TIMEOUT_S if timeout_s is None else timeout_s
    margin_s = SYNC_PUB_MARGIN_S if margin_s is None else margin_s
    t0 = clock() if t0 is None else t0
    while True:
        if check is not None:
            check()
        cur = {k: read(k) for k in pub_need}
        if all(cur[k] is not None and int(cur[k]) >= base[k] + pub_need[k] for k in pub_need):
            break
        if clock() - t0 > timeout_s:
            raise RuntimeError(f"lettori agganciati ai publisher del nodo {cur} sotto base {base} + "
                               f"{pub_need} dopo {timeout_s:.0f} s")
        spin(0.1)
    t_match = clock() - t0
    t1 = clock()
    while clock() - t1 < margin_s:
        spin(min(0.1, margin_s))
    return {"modo": "publisher", "base": base, "attesi_in_piu": pub_need,
            "finale": {k: (None if v is None else int(v)) for k, v in cur.items()},
            "aggancio_s": round(t_match, 3), "margine_s": margin_s, "timeout_s": timeout_s}


def g_publisher_need(g_active, mode=None):
    """--sync-wait publisher with --twin-state ombra_g: one extra reader matched to the main node
    publisher (twin topic) before play. Subscription in the orchestrator graph is not enough:
    with volatile QoS the clouds published before the node-side match are lost and the first
    processed scan is no longer the saved stamp + 0.1 s. None otherwise.
    """
    mode = SYNC_WAIT["mode"] if mode is None else mode
    return {MATCHED_KEY_MAIN: 1} if g_active and mode == "publisher" else None


def sync_wait_publisher(node, pub_need, base, runners, t0):
    def check():
        node.drain()
        for r in runners:
            if r is not None and not r.alive():
                raise RuntimeError(f"mola-cli terminato prima dell'aggancio ({r.log_path})")
    return wait_publisher_matched(lambda k: getattr(node, k, None), base, pub_need, node.spin,
                                  check=check, t0=t0)


def first_stamp_compare(poses_v, poses_s):
    """First processed stamp, from the received poses (shc.pose_record records): key in tenths of a
    second of the minimum stamp of each. None if one of the two has no poses.
    """
    if not poses_v or not poses_s:
        return None
    kv = shc.key10(min(r["stamp_ns"] for r in poses_v) / 1e9)
    ks = shc.key10(min(r["stamp_ns"] for r in poses_s) / 1e9)
    return {"vittima": kv / 10, "ombra": ks / 10, "esito": "uguale" if kv == ks else "diverso",
            "fonte": "pose"}


def first_printf_compare(rows_v, rows_s):
    """Guard fallback on the logs (ICP printf lines, shc.read_printf): stamp of the first line with
    the same pathStep, the smallest common one. None if missing.
    """
    if not rows_v or not rows_s:
        return None
    ps = max(min(r["pathstep"] for r in rows_v), min(r["pathstep"] for r in rows_s))
    fv = [r["stamp_s"] for r in rows_v if r["pathstep"] == ps]
    fs = [r["stamp_s"] for r in rows_s if r["pathstep"] == ps]
    if not fv or not fs:
        return None
    kv, ks = shc.key10(fv[0]), shc.key10(fs[0])
    return {"vittima": kv / 10, "ombra": ks / 10, "esito": "uguale" if kv == ks else "diverso",
            "fonte": f"printf pathStep={ps}"}


def sync_guard(node, victim, shadow, recorder=None, timeout_s=None, clock=time.time):
    """Synchronous start guard: first stamp processed by victim and shadow, from the poses of the
    comparison recorder (--shadow-check) or, if one is missing within timeout_s, from the printf
    lines of the two logs. Outcome "uguale", "diverso" or "non verificabile" (neither source: the
    run goes on).
    """
    recorder = recorder if recorder is not None else SYNC_WAIT.get("recorder")
    timeout_s = SYNC_GUARD_TIMEOUT_S if timeout_s is None else timeout_s
    t0 = clock()
    g = None
    while recorder is not None:
        g = first_stamp_compare(recorder.poses["victim"], recorder.poses["shadow"])
        if g is not None or clock() - t0 > timeout_s:
            break
        node.spin(0.1)
    if g is None:
        try:
            g = first_printf_compare(shc.read_printf(str(getattr(victim, "log_path", "") or "")),
                                     shc.read_printf(str(getattr(shadow, "log_path", "") or "")))
        except Exception as e:
            g = {"vittima": None, "ombra": None, "esito": "non verificabile", "fonte": f"errore: {e}"}
    if g is None:
        g = {"vittima": None, "ombra": None, "esito": "non verificabile", "fonte": "nessuna"}
    g["attesa_s"] = round(clock() - t0, 3)
    print(f"  guardia avvio sincrono: primo stamp vittima {g['vittima']}, ombra {g['ombra']} "
          f"-> {g['esito']} ({g['fonte']})")
    return g
SHADOW_QUIET_MAX_S = 10.0


def shadow_extra_env(cfg_env, ros_args, environ=None):
    """Shadow environment: MOLA configuration equal to the victim, ROS_ARGS with its namespace, ICP
    printf on (sigma) and line-buffered stdout with libstdbuf (environment variables only, MOLA
    and launch unchanged).
    """
    environ = os.environ if environ is None else environ
    env = {**(cfg_env or {}), "ROS_ARGS": ros_args, "MOLA_LO_DEBUG_ICP_QUALITY": "1"}
    if os.path.isfile(tv.STDBUF_LIB):
        pre = environ.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{pre}:{tv.STDBUF_LIB}" if pre else tv.STDBUF_LIB
        env["_STDBUF_O"] = "L"
    return env


def victim_extra_env(cfg_env, victim_env, ros_args, shadow_env=False, environ=None):
    """Victim environment: MOLA configuration plus --victim-mola-env. With --victim-shadow-env the
    same computation as the shadow (shadow_extra_env, LD_PRELOAD appended to the inherited one):
    the two MOLA instances have the same environment.
    """
    env = {**(cfg_env or {}), **(victim_env or {})}
    if not shadow_env:
        return env
    return shadow_extra_env(env, ros_args or VICTIM_ROS_ARGS, environ=environ)


# Fallback fitness (inf, ..., 1e6) given to a candidate without a valid measurement: cause written
# in the candidate record (evaluations[i].fitness_riserva) and per-window counts in history
# (fitness_riserva). Possible causes: guasto_eta_ripetuto, guasto_gemello, mola_se_ne_accorge,
# dos_vittima_senza_posa, infra:<outcome> (infrastructure outcomes, outside VALID and COUNTABLE),
# scartato:<outcome> (measurable outcome discarded by --discard-policy or without a final pose),
# guasto_catena_incompleto_ripetuto, non_furtivo, calcolo_oltre_limite.
def causa_riserva(st, kind, guasto_g=False, dos=False, accorge=False):
    """Cause of the fallback fitness for a discarded candidate."""
    base = str(st).split("(")[0].strip()
    if st == ESITO_ETA_RIPETUTO:
        return ESITO_ETA_RIPETUTO
    if guasto_g:
        return "guasto_gemello"
    if accorge:
        return "mola_se_ne_accorge"
    if dos:
        return "dos_vittima_senza_posa"
    return f"infra:{base}" if kind == "infra" else f"scartato:{base}"


def fitness_riserva_finestra(eval_log):
    """Window block: candidates with the fallback fitness, by cause, with their indices in
    evaluations.
    """
    cause = {}
    for i, e in enumerate(eval_log):
        c = e.get("fitness_riserva")
        if c:
            cause.setdefault(c, []).append(i)
    return {"n": sum(len(v) for v in cause.values()), "n_candidati": len(eval_log),
            "cause": {c: {"n": len(v), "indici": v} for c, v in sorted(cause.items())}}


def valori_topk(search_cm, rev, re_st, topk_agg):
    """Values of a candidate in the top-k re-evaluation. A re-evaluation with outcome
    guasto_eta_ripetuto does not enter the choice (a failed rollout is never used for selection
    or re-evaluation): the candidate is worth its search damage only, declared in valutato_su.
    """
    if re_st == ESITO_ETA_RIPETUTO:
        return {"reeval_dev_mean_cm": None, "mean_cm": search_cm, "min_cm": search_cm, "agg_cm": search_cm,
                "reeval_escluso": ESITO_ETA_RIPETUTO, "valutato_su": "solo_ricerca"}
    return {"reeval_dev_mean_cm": rev, "mean_cm": (search_cm + rev) / 2, "min_cm": min(search_cm, rev),
            "agg_cm": min(search_cm, rev) if topk_agg == "min" else (search_cm + rev) / 2}


def scegli_topk(topk_rec):
    """Applied candidate among the top-k: the maximum agg_cm among those with a valid re-evaluation.
    A candidate with a failed re-evaluation (reeval_escluso) is chosen only if no other has a
    valid one, and the record says so (scelta_fra_esclusi): otherwise the failure would decide
    the choice, because the search damage alone is higher on average (winner's curse).
    """
    validi = [r for r in topk_rec if "reeval_escluso" not in r]
    best = max(validi or topk_rec, key=lambda r: r["agg_cm"])
    return dict(best, scelta_fra_esclusi=True) if not validi else best


def ambiente_binario_g1(args, environ=None):
    """--mola-binario-g1: the three variables the G shadow receives (tv.g_shadow_env:
    LD_LIBRARY_PATH and MOLA_MODULES_LIB_PATH with the --g-overlay first, MOLA_FULL_STATE=1), to
    give to all MOLA instances (victim, shadow, attacker, twins) with the same values. Empty
    without the option.
    """
    if not getattr(args, "mola_binario_g1", False):
        return {}
    return tv.g_shadow_env(environ=environ, root=getattr(args, "g_overlay", None) or tv.G_OVERLAY_ROOT)


class ShadowRunner(MolaRunner):
    """Shadow of variant E (--twin-state ombra_e): continuous MOLA like the victim, subscribed to
    the same clouds (/carter/lidar_victim, produced by the attacker node), namespace
    tv.SHADOW_NS. At the start of a window it saves the map and provides pose and sigma to the
    twin: no victim information is used.
    """

    def __init__(self, script: str, log_dir: Path, traces: bool = False, ros_args=None, env=None,
                 topic: str = VICTIM_TOPIC, g_root=None):
        # topic: VICTIM_TOPIC by default; with --shadow-topic the one on which the node republishes
        # the same cloud (perturbation_node.py --shadow-topic). g_root: with --twin-state ombra_g
        # the shadow runs on the MOLA overlay and at every map_save also writes <path>.molastate
        # (MOLA_FULL_STATE=1); the victim never gets these variables (it stays on /opt).
        ext = shadow_extra_env(env, ros_args or tv.mola_ros_args("shadow"))
        if g_root:
            ext.update(tv.g_shadow_env(root=g_root))
        super().__init__(script, log_dir, ready_timeout=60.0, extra_env=ext,
                         traces=traces, args=[topic])
        self.run_origin = None
        self.g_root = g_root

    def start(self, node):
        """Call with the victim stream open (clouds arriving)."""
        self.spawn("shadow")
        self.wait_ready(check=node.drain)
        self.wait_pose(node)

    def wait_pose(self, node):
        deadline = time.time() + 15.0
        while time.time() < deadline and getattr(node, "shadow_est_raw", None) is None:
            node.spin(0.1)
        if getattr(node, "shadow_est_raw", None) is None:
            raise RuntimeError(f"l'ombra non pubblica su {tv.SHADOW_NS}/lidar_odometry/pose")

    def believed_pose(self, node):
        """Robot pose according to the shadow, in the working frame."""
        if getattr(node, "shadow_est_raw", None) is None or self.run_origin is None:
            return None
        return node.compose(self.run_origin, node.shadow_est_raw)

    def save_state(self, node, path):
        """Shadow map_save to `path` (+ .mm, .simplemap), last pose and sigma from the log. status
        "ok" only with .mm present, pose and sigma.
        """
        t0 = time.time()
        rec = {"status": None, "mm_path": f"{path}.mm", "shadow_log": str(self.log_path)}
        if not self.alive():
            rec["status"] = "shadow_dead"
            return rec
        while time.time() - t0 < SHADOW_QUIET_MAX_S and time.time() - node.shadow_est_t < SHADOW_QUIET_S:
            node.spin(0.1)
        rec["quiet_wait_s"] = round(time.time() - t0, 2)
        rec["quiet_ok"] = time.time() - node.shadow_est_t >= SHADOW_QUIET_S
        t1 = time.time()
        ok, err = node.shadow_map_save(path)
        rec["map_save_s"] = round(time.time() - t1, 3)
        rec["map_save_ok"] = ok
        why = tv.mm_problem(rec["mm_path"]) if ok else f"map_save: {err}"
        if why:
            rec.update(status="map_save_failed", error=why)
            return rec
        rec["mm_bytes"] = os.path.getsize(rec["mm_path"])
        rec["mm_sha256"] = tv.file_sha256(rec["mm_path"])
        pose6, stamp = node.shadow_pose6, node.shadow_est_stamp
        if pose6 is None:
            rec["status"] = "no_pose"
            return rec
        rec["pose6"] = [float(v) for v in pose6]
        rec["pose_stamp"] = stamp
        rec.update(tv.shadow_sigma_from_log(self.log_path, stamp))
        if getattr(self, "g_root", None):
            # Variant G: full state in <path>.molastate (stamp = last scan processed by the shadow);
            # sigma from the log is informative only.
            sp = f"{path}.molastate"
            rec["state_path"] = sp
            why = tv.g_state_problem(sp)
            if why:
                rec.update(status="g_state_failed", error=why)
                return rec
            h = tv.molastate_header(sp)
            rec["state_bytes"] = os.path.getsize(sp)
            rec["state_sha256"] = tv.file_sha256(sp)
            rec["state_stamp_tick"] = h["stamp_tick"]
            # published pose: stamp passed through toROS (double) and converted back, exact equality
            # is not guaranteed; the difference is recorded
            pt = tv.stamp_s_to_tick(stamp)
            rec["pose_stamp_tick"] = pt
            rec["stamp_diff_ticks"] = (h["stamp_tick"] - pt) if pt is not None else None
            rec["stamp_ok"] = rec["stamp_diff_ticks"] is not None and abs(rec["stamp_diff_ticks"]) <= 1
            rec["env"] = tv.g_twin_env(sp, None, root=self.g_root)
        else:
            if rec["sigma"] is None:
                rec["status"] = "no_sigma"
                return rec
            rec["env"] = tv.e_initial_env(rec["mm_path"], pose6, rec["sigma"])
        b = self.believed_pose(node)
        rec["believed"] = [float(v) for v in b] if b is not None else None
        rec["origin"] = [float(v) for v in self.run_origin] if self.run_origin is not None else None
        rec["status"] = "ok"
        rec["total_s"] = round(time.time() - t0, 3)
        return rec


# Settling counted in clouds (--settle-clouds N, --reopen-clouds M; 0 = time-based rules) and
# never-clean victim (--victim-never-clean). Set in main.
CLOUD_SETTLE = {"n": 0, "reopen": 0, "never_clean": False}


def settle_target(victim_started):
    """Clouds while still, from play (twin) or from stream opening (victim) to driving. N for a MOLA
    starting from zero: the twins until the victim is started and the victim at its start.
    Afterwards, M for the victim reopening and for the twins (same still phase); with M = 0 it
    stays N. 0 = time-based settling (cap or nominal duration).
    """
    n, m = CLOUD_SETTLE["n"], CLOUD_SETTLE["reopen"]
    return m if (victim_started and m > 0) else n


def wait_clouds(node, count, c0, target, t0, spin=None, timeout=None):
    """Wait until the node has delivered `target` clouds since t0 (counter count() of its status,
    value c0 at t0). Returns the record for the history; ok False if the time cap expires first.
    """
    spin = spin or node.spin
    timeout = timeout if timeout is not None else 10.0 + 0.2 * target
    while time.time() - t0 < timeout and (count() or 0) - c0 < target:
        spin(0.02)
    n = (count() or 0) - c0
    return {"target": int(target), "clouds": int(n), "wall_s": round(time.time() - t0, 3),
            "ok": bool(n >= target)}


def close_settle_record(rec, count, c0, t0):
    """Clouds and duration at the driving instant (after reading the true pose). exact: driving
    starts at the N-th cloud; excess: clouds beyond N (pose arrived after the N-th or status one
    period late).
    """
    if rec is not None:
        rec["clouds_at_drive"] = int((count() or 0) - c0)
        rec["wall_at_drive_s"] = round(time.time() - t0, 3)
        rec["sim_s_nominal"] = round(rec["clouds_at_drive"] / 10.0, 2)   # LiDAR at 10 Hz
        rec["excess"] = rec["clouds_at_drive"] - int(rec.get("target", 0))
        rec["exact"] = rec["excess"] == 0
    return rec


# --registra-assetto: Isaac assetto command after the restore, after set_state and at the end of
# settling of every rollout; --twin-init-assetto: twin attitude in set_state (azzerato = quaternion
# from yaw only). spawn: base attitude read once at the run start pose.
ASSETTO = {"registra": False, "gemello": "azzerato", "spawn": None}
CHIAVI_ASSETTO = ("pos", "quat_wxyz", "roll", "pitch", "yaw", "lin", "ang", "playing")
# --riallinea-dopo-assestamento: Isaac snapshot slot at the end of victim settling (the
# start-of-metre slot is unchanged)
POSTO_RIALLINEO = "riallineo"
# Victim resume after the realigned twin: driving starts as soon as the filter has dropped the scan
# in progress at the restore (partly taken in the end pose of the realigned twin), without waiting
# for new victim clouds; cap in s.
RIPRESA_ATTESA_MAX_S = 1.0


class ErroreAssetto(RuntimeError):
    """--twin-init-assetto attaccante: attacker attitude unavailable or out of bounds. The run stops
    (configuration error, not an outcome).
    """


def leggi_assetto(isaac, fase, node=None):
    """Attitude reading (--registra-assetto), appended to node.last_assetto of the current rollout.
    Off: None, no command to Isaac.
    """
    if not ASSETTO["registra"]:
        return None
    t0 = time.time()
    try:
        r = isaac.call("assetto")
        rec = {"fase": fase, **{k: r.get(k) for k in CHIAVI_ASSETTO}}
    except IsaacStall:
        raise
    except RuntimeError as e:
        rec = {"fase": fase, "errore": str(e)}
    rec["t_wall"] = round(t0, 3)
    rec["wall_s"] = round(time.time() - t0, 3)
    reg = getattr(node, "last_assetto", None) if node is not None else None
    if isinstance(reg, list):
        reg.append(rec)
    return rec


def misura_assetto_spawn(isaac, start_pose):
    """Base attitude at the run start pose (after set_pose, play and 1 s, robot still), before the
    attacker MOLA starts: the starting frame of its map (zero roll and pitch).
    """
    r = isaac.call("assetto")
    return {"posa_partenza": [float(v) for v in start_pose], "t_wall": round(time.time(), 3),
            **{k: r.get(k) for k in CHIAVI_ASSETTO},
            "nota": "lettura unica a robot fermo dopo set_pose alla --start-pose, play e 1 s"}


def assetto_gemello(modo, node, spawn=None):
    """--twin-init-assetto: extra set_state parameters for the twin and their source. azzerato:
    none. conservato: keep_tilt, roll and pitch of the restore (saved Isaac state, i.e. the true
    attitude: comparison only, outside the threat model). attaccante_relativo (admissible): roll
    and pitch of the pose published by the attacker MOLA (same message as x, y, yaw) as they are,
    relative to the frame in which its map was born (FixedPose with zero roll and pitch,
    config/mola/lidar3d-gicp-optimize-twist-local.yaml:184-194): information the attacker owns.
    attaccante (diagnostic, not admissible): the same composed with the base attitude read from
    Isaac at spawn (banco_g.assetto_composto).
    """
    if modo in (None, "azzerato"):
        return {}, None
    if modo == "conservato":
        return {"keep_tilt": True}, {
            "modo": modo, "fonte": "restore: rollio e beccheggio dello stato salvato di Isaac (assetto vero "
                                   "della vittima a inizio rollout; fuori dal threat model D6, solo confronto)"}
    if modo not in ("attaccante", "attaccante_relativo"):
        raise ErroreAssetto(f"--twin-init-assetto {modo} sconosciuto")
    a6 = getattr(node, "attacker_est6", None)
    sp = spawn if spawn is not None else ASSETTO.get("spawn")
    if a6 is None:
        raise ErroreAssetto("assetto dell'attaccante: nessuna posa completa dalla sua MOLA")
    rel = (float(a6[5]), float(a6[4]), float(a6[3]))
    raw = getattr(node, "attacker_est_raw", None)
    stesso = raw is not None and (a6[0], a6[1], a6[3]) == tuple(raw)
    att = {"roll": rel[0], "pitch": rel[1], "yaw": rel[2], "stamp": getattr(node, "attacker_est_stamp", None),
           "stesso_messaggio_della_posa": bool(stesso)}
    if modo == "attaccante_relativo":
        if max(abs(rel[0]), abs(rel[1])) > banco_g.ASSETTO_MAX_RAD:
            raise ErroreAssetto(f"assetto dell'attaccante oltre {banco_g.ASSETTO_MAX_RAD} rad "
                                f"(rollio {rel[0]:.5f}, beccheggio {rel[1]:.5f})")
        return {"roll": rel[0], "pitch": rel[1]}, {
            "modo": modo, "variante": "3a, adottabile",
            "attaccante": att,
            "fonte": ("posa pubblicata dalla MOLA dell'attaccante (/attacker/lidar_odometry/pose, quaternione "
                      "completo), rollio e beccheggio cosi' come stimati, relativi alla terna di partenza della "
                      "sua mappa: nessuna informazione oltre la stima dell'attaccante (D6)")}
    if not sp or sp.get("roll") is None or sp.get("pitch") is None or sp.get("yaw") is None:
        raise ErroreAssetto("assetto dell'attaccante: assetto allo spawn non misurato")
    r, p, y = banco_g.assetto_composto((float(sp["roll"]), float(sp["pitch"]), float(sp["yaw"])), rel)
    if max(abs(r), abs(p)) > banco_g.ASSETTO_MAX_RAD:
        raise ErroreAssetto(f"assetto dell'attaccante composto oltre {banco_g.ASSETTO_MAX_RAD} rad "
                            f"(rollio {r:.5f}, beccheggio {p:.5f})")
    return {"roll": r, "pitch": p}, {
        "modo": modo, "variante": "3b diagnostica, non adottabile (assetto vero allo spawn letto da Isaac)",
        "attaccante": att,
        "spawn": {k: sp.get(k) for k in ("roll", "pitch", "yaw")},
        "composto": {"roll": r, "pitch": p, "yaw": y},
        "fonte": ("posa pubblicata dalla MOLA dell'attaccante (/attacker/lidar_odometry/pose, quaternione "
                  "completo) composta con l'assetto allo spawn: R = R_spawn R_attaccante (ZYX)")}


class _IsaacPosto:
    """Isaac client for a rollout restarting from a named snapshot: restore of the slot, everything
    else unchanged.
    """

    def __init__(self, cli, nome):
        self._cli, self._nome = cli, nome

    def __getattr__(self, nome):
        return getattr(self._cli, nome)

    def restore(self, nome=None):
        return self._cli.restore(nome=nome or self._nome)


# Node fields of the victim rollout that the realigned twin overwrites: saved before and put back
# after, so the victim stretch is measured as without the option.
CAMPI_NODO_RIALLINEO = (
    "est", "est_raw", "est_t", "est_stamp", "est_stamp_raw", "est_z", "est_quat", "est_origin",
    "est_source", "chamfer", "perc_samples", "proc_ms_roll", "pose_quality", "last_eta_nuvola",
    "last_drive_wall", "last_stop_info", "last_twin_flags", "last_twin_fp", "last_twin_align",
    "last_mola_stop", "last_replay", "last_settle_waited", "last_settle_ok", "last_settle_clouds",
    "last_align_err_cm", "last_align_err_deg", "last_resume_jump_cm", "last_trace", "last_tau",
    "last_loc_err_mean", "last_loc_err_end", "last_loc_err_mean_raw", "last_loc_err_end_raw",
    "last_track_ratio", "last_track_thr", "last_motion", "last_quality_mean", "last_quality_min",
    "last_perc", "last_drift_start_cm", "last_drift_end_cm", "last_stop_reason", "last_wps_reached",
    "last_drive_stamps", "last_fresh_arm", "rollout_retry", "last_assetto")
_ASSENTE = object()


def conteggi_vittima(node, t0=None):
    """Clouds delivered to the victim and dropped by the filter, from the status."""
    return {"victim_rec_total": getattr(node, "victim_rec_total", None),
            "fresh_dropped_total": getattr(node, "fresh_dropped_total", None),
            "t_wall": round(time.time(), 3),
            **({"dal_play_s": round(time.time() - t0, 3)} if t0 is not None else {})}


def _diff_conteggio(a, b, chiave):
    x, y = (a or {}).get(chiave), (b or {}).get(chiave)
    return int(y) - int(x) if x is not None and y is not None else None


def fotografa_nodo(node, campi=CAMPI_NODO_RIALLINEO):
    """Current values of the node fields (copies of the lists)."""
    out = {}
    for c in campi:
        v = getattr(node, c, _ASSENTE)
        out[c] = list(v) if isinstance(v, list) else v
    return out


def rimetti_nodo(node, foto):
    for c, v in foto.items():
        if v is _ASSENTE:
            if hasattr(node, c):
                try:
                    delattr(node, c)
                except AttributeError:
                    pass
        else:
            setattr(node, c, v)


def gemello_salva_stato(node, mola, path, origin, quiet_s=None, quiet_max_s=None):
    """Bench, realignment: full state of the twin MOLA (map_save on /map_save, <path>.molastate with
    MOLA_FULL_STATE=1), as ShadowRunner.save_state for the shadow. pose6 from the current twin
    estimate (frame of its MOLA, i.e. of the shadow it was loaded from), believed = origin
    composed with it.
    """
    quiet_s = SHADOW_QUIET_S if quiet_s is None else quiet_s
    quiet_max_s = SHADOW_QUIET_MAX_S if quiet_max_s is None else quiet_max_s
    t0 = time.time()
    rec = {"status": None, "mm_path": f"{path}.mm", "twin_log": str(getattr(mola, "log_path", None))}
    if hasattr(mola, "alive") and not mola.alive():
        rec["status"] = "twin_dead"
        return rec
    while time.time() - t0 < quiet_max_s and time.time() - node.est_t < quiet_s:
        node.spin(0.1)
    rec["quiet_wait_s"] = round(time.time() - t0, 2)
    rec["quiet_ok"] = time.time() - node.est_t >= quiet_s
    t1 = time.time()
    ok, err = node.twin_map_save(path)
    rec["map_save_s"] = round(time.time() - t1, 3)
    rec["map_save_ok"] = ok
    if not ok:
        rec.update(status="map_save_failed", error=f"map_save: {err}")
        return rec
    sp = f"{path}.molastate"
    rec["state_path"] = sp
    why = tv.g_state_problem(sp)
    if why:
        rec.update(status="g_state_failed", error=why)
        return rec
    h = tv.molastate_header(sp)
    rec.update(state_bytes=os.path.getsize(sp), state_sha256=tv.file_sha256(sp), state_stamp_tick=h["stamp_tick"])
    raw, q, z = getattr(node, "est_raw", None), getattr(node, "est_quat", None), getattr(node, "est_z", float("nan"))
    if raw is None or q is None:
        rec["status"] = "no_pose"
        return rec
    rec["pose6"] = [float(v) for v in tv.pose6_from_quat(raw[0], raw[1], z, *q)]
    stamp = getattr(node, "est_stamp_raw", float("nan"))
    rec["pose_stamp"] = stamp if stamp == stamp else None
    pt = tv.stamp_s_to_tick(rec["pose_stamp"])
    rec["pose_stamp_tick"] = pt
    rec["stamp_diff_ticks"] = (h["stamp_tick"] - pt) if pt is not None else None
    rec["origin"] = [float(v) for v in origin] if origin is not None else None
    rec["believed"] = ([float(v) for v in AttackNode.compose(origin, raw)] if origin is not None else None)
    rec["env"] = None
    rec["status"] = "ok"
    rec["total_s"] = round(time.time() - t0, 3)
    return rec


def topic_sub_gids(node, topic):
    """GIDs of the subscribers of `topic` in the graph, None if the graph is not readable."""
    try:
        infos = node.node.get_subscriptions_info_by_topic(topic)
    except Exception:
        return None
    return {tuple(getattr(i, "endpoint_gid", ()) or ()) for i in infos}


def wait_new_subscriber(node, topic, before, alive=None, timeout=None):
    """--settle-clouds, twin: with the simulation stopped, wait until the just-launched MOLA shows
    up among the subscribers of `topic` (GID not in `before`; the entry of the previous MOLA may
    stay in the graph). As for the victim in spawn_deferred: the first cloud after play is the
    first MOLA receives. On cap expiry it goes on: the record says so and the pose is the
    evidence.
    """
    timeout = SYNC_SUB_TIMEOUT_S if timeout is None else timeout
    t0 = time.time()
    while True:
        node.drain()
        now = topic_sub_gids(node, topic)
        if now is None:
            return {"topic": topic, "ok": None, "s": 0.0}
        if now - before:
            return {"topic": topic, "ok": True, "s": round(time.time() - t0, 3)}
        if (alive is not None and not alive()) or time.time() - t0 > timeout:
            return {"topic": topic, "ok": False, "s": round(time.time() - t0, 3)}
        node.spin(0.1)


VICTIM_PUBLISHER = "perturbation_node"


def victim_topic_problem(node, timeout=3.0):
    """--victim-never-clean: only the perturbation node may publish on the victim topic (a foreign
    publisher would give it clean clouds). Error message or None; unreadable graph = error.
    """
    t0 = time.time()
    while True:
        try:
            infos = list(node.node.get_publishers_info_by_topic(VICTIM_TOPIC))
        except Exception as e:
            return f"publisher di {VICTIM_TOPIC} non leggibili: {e}"
        names = [f"{(getattr(i, 'node_namespace', '') or '').rstrip('/')}/{getattr(i, 'node_name', '?')}"
                 for i in infos]
        foreign = [n for n in names if n.rsplit("/", 1)[-1] != VICTIM_PUBLISHER]
        if foreign:
            return f"publisher estranei su {VICTIM_TOPIC}: {foreign} (tutti: {names})"
        if len(infos) == 1:
            return None
        if time.time() - t0 > timeout:
            return f"{len(infos)} publisher su {VICTIM_TOPIC} (atteso 1, {VICTIM_PUBLISHER}): {names}"
        node.spin(0.1)


def check_victim_topic(node):
    """Check at victim stream opening: raises if it fails."""
    err = victim_topic_problem(node)
    if err:
        raise VictimTopicForeign(err)


def never_clean_mismatch(node, want, timeout=10.0):
    """--victim-never-clean must match between orchestrator and node: with the option only in the
    node the victim start (attack off) would get no clouds; with the option only in the
    orchestrator the tails and the windows without genome would pass clean. Error message or
    None.
    """
    if want:
        t0 = time.time()
        while getattr(node, "node_never_clean", None) is not True and time.time() - t0 < timeout:
            node.spin(0.1)
        if getattr(node, "node_never_clean", None) is not True:
            return ("--victim-never-clean richiede il nodo di perturbazione lanciato con la stessa "
                    "opzione (status senza victim_never_clean)")
        return None
    if getattr(node, "node_never_clean", None) is True:
        return ("il nodo di perturbazione ha --victim-never-clean e l'orchestratore no: l'avvio della "
                "vittima ad attacco spento non riceverebbe nuvole")
    return None


def never_clean_fields(node):
    """Cumulative node counters (history, with --victim-never-clean): clouds dropped for victim and
    shadow and clean clouds delivered (expected 0).
    """
    return {"dropped_total": getattr(node, "victim_clean_dropped", None),
            "clean_out_total": getattr(node, "victim_clean_out", None),
            "start_phase": getattr(node, "never_clean_start", None)}


def victim_hold_fields(reason, victim):
    """Window without genome with --victim-never-clean: the victim does not drive (it would get no
    clouds, or clean ones), the robot stays in the saved state and the next window restarts from
    there.
    """
    return {"victim_held": True, "victim_hold_reason": reason,
            "victim_started": bool(getattr(victim, "started", False))}


def genoma_nullo(active_genes, n_genes=17):
    """Null genome: direction genes at 0, all others at -1. It is the `off` vector of random
    sampling (in the group space), expanded like the candidates with expand_genome.
    """
    off = np.array([0.0 if ag in DIRECTION_GENES else -1.0 for ag in active_genes])
    return expand_genome(off, active_genes, n_genes)


def deriva_vittima_status(node):
    """Accumulated drift of the victim generator from the last node /attack/status (10 Hz):
    magnitude |A| and xy part in cm (victim_drift_cm, victim_drift_xy_cm) and keep mode. None in
    the fields the node does not report.
    """
    return {"modulo_cm": getattr(node, "st_victim_drift_cm", None),
            "xy_cm": getattr(node, "st_victim_drift_xy_cm", None),
            "drift_keep": getattr(node, "st_drift_keep", None),
            "fonte": "status del nodo: victim_drift_cm, victim_drift_xy_cm"}


def scelta_genoma_nullo(args, rf_rec, eval_log, n_valid_sel, active_genes, n_genes, res_F):
    """--genoma-nullo-se-nessun-ammissibile: window without an admissible genome.

    Returns (best_g, best_F, ratio, res_F, rec). best_g is the null genome, then applied through the
    normal path like a winner; best_F and ratio are NaN because the null genome is not evaluated in
    the twin (damage_cm, perturbation and ratio go to None in the history); res_F is the pymoo front
    as (n, 2), empty if pymoo gave none (None without feasible points). rec: reason, attempts
    (single random: n_tries, otherwise the window evaluations) and counts, for the
    genoma_nullo_dettagli entry.
    """
    if rf_rec is not None:
        n = int(rf_rec.get("n_tries", 0))
        motivo = f"nessun ammissibile in {n} tentativi"
    else:
        n = len(eval_log)
        motivo = "nessun candidato ammissibile"
    F = np.zeros((0, 2)) if res_F is None else np.asarray(res_F, dtype=float)
    F = np.zeros((0, 2)) if F.size == 0 else np.atleast_2d(F)
    g = genoma_nullo(active_genes, n_genes)
    rec = {"motivo": motivo, "tentativi": n, "ricerca": args.search,
           "random_first": rf_rec is not None,
           "k_max": int(rf_rec["k_max"]) if rf_rec is not None and rf_rec.get("k_max") is not None else None,
           "n_valutazioni": len(eval_log), "n_validi_selezione": n_valid_sel,
           "genoma": [float(v) for v in g],
           "regola": "nessun genoma ammissibile nella finestra: si applica il genoma nullo "
                     "(direzioni 0, altri geni -1) e la vittima percorre il metro; la finestra dopo "
                     "riprova la ricerca. La deriva accumulata (drift keep) resta applicata: residuo "
                     "registrato, non corretto"}
    return g, np.array([np.nan, np.nan]), float("nan"), F, rec


class RiallineoVittima:
    """Realignment after settling in the victim stream (--riallinea-dopo-assestamento): called by
    _victim_rollout at the end of settling, before driving.
    Robot still: true pose read in play, then pause and victim stream closed; Isaac snapshot in slot
    POSTO_RIALLINEO (the start-of-metre one is kept); full shadow state (variant G, as at metre
    start); victim generator state (node with --stato-generatore); physical target from the current
    attacker estimate (attacker_init_target, --twin-init-assetto); twin rollout with the victim
    genome loading the shadow state, without settling (esegui); restore of the victim snapshot,
    first-cloud filter, node fields put back, stream reopened, play and first victim pose. The
    record goes into history[k]["riallineato"]; no true pose enters the twin: the true pose is used
    only for the odom-scene frame change, as at metre start.
    """

    def __init__(self, k, esegui, shadow, attacker, out, vel_mode="zero", mola=None):
        self.k, self.esegui, self.shadow, self.attacker = k, esegui, shadow, attacker
        self.out, self.vel_mode, self.mola = Path(out), vel_mode, mola
        self.tag = f"w{k}_riallineato"
        self.rec = None
        self.trace_rows = None

    def __call__(self, isaac, node, info):
        t0 = time.time()
        rec = self.rec = {"tag": self.tag, "t_wall_inizio": round(t0, 3), "tag_vittima": info.get("tag")}
        saved_now = node.true_pose(isaac)
        isaac.pause()
        node.set_victim_stream(False)
        rec["conteggi"] = {"pausa": conteggi_vittima(node)}
        saved_scene_now = isaac.save(nome=POSTO_RIALLINEO)
        rec["istantanea"] = {"posto": POSTO_RIALLINEO, "posa_scena": [float(v) for v in saved_scene_now],
                             "posa_vera_odom_lab": [float(v) for v in saved_now]}
        leggi_assetto(isaac, "riallineo_pausa", node)
        # victim rollout fields, put back on resume
        foto = fotografa_nodo(node)
        node.est_source = "twin"
        er = foto.get("est_raw")
        rec["vittima_alla_pausa"] = {"est_raw": [float(v) for v in er] if er not in (None, _ASSENTE) else None,
                                     "est_stamp_raw": foto.get("est_stamp_raw") if foto.get("est_stamp_raw")
                                     is not _ASSENTE else None}
        mp = self.out / "maps" / f"ombra_w{self.k}_riallineo"
        mp.parent.mkdir(parents=True, exist_ok=True)
        sh = self.shadow.save_state(node, mp)
        rec["stato_ombra"] = sh
        rec["generatore_salvato"] = node.stato_generatore("salva", POSTO_RIALLINEO, flusso="vittima")
        believed_att = self.attacker.believed_pose(node) if self.attacker is not None else None
        rec["posa_creduta_attaccante"] = [float(v) for v in believed_att] if believed_att is not None else None
        why = None
        if sh.get("status") != "ok" or sh.get("believed") is None:
            why = f"stato dell'ombra non salvato ({sh.get('status')}: {sh.get('error')})"
        elif believed_att is None:
            why = "nessuna stima dell'attaccante"
        tgt = None
        if why is None:
            tgt = attacker_init_target(node, self.attacker, saved_now, saved_scene_now, believed_att, self.vel_mode)
            try:
                kw_a, rec_a = assetto_gemello(ASSETTO["gemello"], node)
            except ErroreAssetto as e:
                kw_a, rec_a, why = {}, {"modo": ASSETTO["gemello"], "errore": str(e)}, f"assetto: {e}"
            if kw_a:
                tgt["assetto_set_state"] = kw_a
            if rec_a is not None:
                tgt["assetto"] = rec_a
        rec["init_target"] = tgt
        if why is None:
            gs = rec["generatore_salvato"]
            rec["generatore_caricato"] = (node.stato_generatore("carica", POSTO_RIALLINEO)
                                          if gs is not None and gs.get("confermato") else None)
            rec["generatore"] = ("trasferito" if (rec["generatore_caricato"] or {}).get("confermato") else
                                 "nodo_senza_stato_generatore" if gs is None else "non_confermato")
            chiavi = ("mode", "shadow_state", "init", "init_target", "saved", "attacker_belief")
            ts_prima = {c: TWIN_STATE.get(c, _ASSENTE) for c in chiavi}
            TWIN_STATE.update({"mode": "ombra_g", "shadow_state": sh, "init": "attacker_est",
                               "init_target": tgt, "saved": saved_now, "attacker_belief": believed_att})
            # state that the nested rollout overwrites and that belongs to the victim rollout or to
            # the window: live log tag, twin MOLA shutdown, last stamp of the previous twin, age
            # counters
            reg = getattr(node, "registri", None)
            reg_prima = (reg.tag, getattr(reg, "_t0_rollout", None)) if reg is not None else None
            stop_prima = getattr(self.mola, "last_stop", _ASSENTE) if self.mola is not None else _ASSENTE
            log_prev = TWIN_LOG_PREV.get("last_stamp")
            ce_k = str(STALL_POLICY.get("window"))
            ce_prima = (json.loads(json.dumps(CONTROLLO_ETA["finestre"][ce_k]))
                        if ce_k in CONTROLLO_ETA["finestre"] else None)
            t1 = time.time()
            RAMPA["sospesa"] = True        # the realigned twin starts after settling: no ramp
            try:
                res = self.esegui(_IsaacPosto(isaac, POSTO_RIALLINEO), self.tag, tuple(sh["believed"]))
            finally:
                RAMPA["sospesa"] = False
                for c, v in ts_prima.items():
                    if v is _ASSENTE:
                        TWIN_STATE.pop(c, None)
                    else:
                        TWIN_STATE[c] = v
                stop_ria = getattr(self.mola, "last_stop", None) if self.mola is not None else None
                if stop_prima is not _ASSENTE:
                    self.mola.last_stop = stop_prima
                if reg_prima is not None:
                    reg.tag = reg_prima[0]
                    if reg_prima[1] is not None:
                        reg._t0_rollout = reg_prima[1]
                TWIN_LOG_PREV["last_stamp"] = log_prev
                ce_dopo = CONTROLLO_ETA["finestre"].get(ce_k)
                if ce_prima is None:
                    CONTROLLO_ETA["finestre"].pop(ce_k, None)
                else:
                    CONTROLLO_ETA["finestre"][ce_k] = ce_prima
            _, end, ch, st, trav, frames = res
            self.trace_rows = list(getattr(node, "last_trace", None) or [])
            mola = self.mola
            rec["gemello"] = {
                "status": st, "end": [float(v) for v in end] if end is not None else None,
                "travelled": trav, "frames": frames, "chamfer": ch if ch == ch else None,
                "tau_s": node.last_tau if getattr(node, "last_tau", None) == getattr(node, "last_tau", None) else None,
                "slow": bool(node.last_tau == node.last_tau and node.last_tau > SLOW_TAU),
                "quality_mean": getattr(node, "last_quality_mean", None),
                "quality_min": getattr(node, "last_quality_min", None),
                "mola_log": str(getattr(mola, "log_path", None)) if mola is not None else None,
                "twin_start": getattr(node, "last_twin_fp", None),
                "twin_alignment": getattr(node, "last_twin_align", None),
                "settle_clouds": getattr(node, "last_settle_clouds", None),
                "eta_nuvola": getattr(node, "last_eta_nuvola", None),
                "assetto": getattr(node, "last_assetto", None) if ASSETTO["registra"] else None,
                "mola_stop": stop_ria, "controllo_eta_conteggi": ce_dopo,
                **({"rampa": node.stato_rampa("gemello")} if RAMPA["R"] > 1 else {}),
                "durata_s": round(time.time() - t1, 3)}
            if rec["generatore_caricato"] is not None:
                rec["generatore_dopo"] = dict(getattr(node, "st_stato_generatore", None) or {}).get("ultimo_caricato")
        else:
            rec["gemello"] = None
            rec["saltato"] = why
            print(f"  riallineato w{self.k}: non eseguito ({why})")
        # victim resume from the end-of-settling snapshot
        isaac.pause()
        pose = isaac.restore(nome=POSTO_RIALLINEO)
        if getattr(node, "true_source", "ros") == "ros":
            node.true = tuple(saved_now)
        fresh = None
        if getattr(node, "fresh_cloud", False):
            count_restore(node, "riallineato_ripresa", self.k)
            node.arm_fresh_cloud()
            fresh = node.last_fresh_arm
        rimetti_nodo(node, foto)
        leggi_assetto(isaac, "riallineo_ripresa", node)
        node.set_victim_stream(True)
        c_ripresa = rec["conteggi"]["ripresa"] = conteggi_vittima(node)
        t_res = self.t_play = time.time()
        isaac.play()
        # No wait for new poses: the victim restarts from the last settling pose (put back above),
        # as without realignment. Only wait for the filter to drop the scan in progress at the
        # restore; if a cloud reaches the victim instead (no drop) driving starts at once and the
        # record says so.
        attesa = "senza_filtro"
        if getattr(node, "fresh_cloud", False):
            attesa = "scaduta"
            while time.time() - t_res < RIPRESA_ATTESA_MAX_S:
                node.spin(0.01)
                if (_diff_conteggio(c_ripresa, conteggi_vittima(node), "fresh_dropped_total") or 0) > 0:
                    attesa = "scartata"
                    break
                if (_diff_conteggio(c_ripresa, conteggi_vittima(node), "victim_rec_total") or 0) > 0:
                    attesa = "nuvola_alla_vittima"
                    break
        rec["conteggi"]["dopo_attesa"] = conteggi_vittima(node, t_res)
        rec["ripresa"] = {"restore_pose": [float(v) for v in pose] if pose is not None else None,
                          "fresh_arm": fresh, "attesa": attesa,
                          "attesa_s": round(time.time() - t_res, 3),
                          "regola": ("guida dall'ultima posa dell'assestamento; attesa solo dello scarto della "
                                     f"scansione in corso al ripristino (tetto {RIPRESA_ATTESA_MAX_S} s)")}
        rec["pausa_vittima_s"] = round(t_res - t0, 3)
        print(f"  riallineato w{self.k}: {((rec.get('gemello') or {}).get('status'))}, generatore "
              f"{rec.get('generatore')}, vittima ripresa dopo {rec['pausa_vittima_s']:.1f} s "
              f"(ripresa: {attesa} in {rec['ripresa']['attesa_s']} s)")

    def alla_guida(self, node):
        """Counts at the start of victim driving (from _victim_rollout)."""
        if self.rec is None:
            return
        c = self.rec.setdefault("conteggi", {})
        c["guida"] = conteggi_vittima(node, getattr(self, "t_play", None))
        self.rec["nuvole_vittima_ripresa_guida"] = _diff_conteggio(c.get("ripresa"), c["guida"], "victim_rec_total")
        self.rec["scartate_ripresa_guida"] = _diff_conteggio(c.get("ripresa"), c["guida"], "fresh_dropped_total")

    def alla_fine(self, node):
        """Counts at the end of the victim stretch (from the loop, after the rollout)."""
        if self.rec is None:
            return
        c = self.rec.setdefault("conteggi", {})
        c["fine"] = conteggi_vittima(node, getattr(self, "t_play", None))
        self.rec["scartate_ripresa_fine"] = _diff_conteggio(c.get("ripresa"), c["fine"], "fresh_dropped_total")


class RiallineoBanco:
    """Bench realignment: at the end of settling of the victim arm V (a twin G with restore init)
    pause, full state of its MOLA (map_save, .molastate), generator state of the twin stream
    (node with --stato-generatore), Isaac snapshot in slot POSTO_RIALLINEO, then play: V goes on.
    The realigned arm runs after V, from the state saved here.
    """

    def __init__(self, path, origin):
        self.path, self.origin = Path(path), origin
        self.rec = None

    def __call__(self, isaac, node, info):
        t0 = time.time()
        saved_now = node.true_pose(isaac)
        isaac.pause()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"tag_vittima": info.get("tag"), "t_wall_inizio": round(t0, 3)}
        rec["stato"] = gemello_salva_stato(node, info.get("mola"), self.path, self.origin)
        rec["generatore_salvato"] = node.stato_generatore("salva", POSTO_RIALLINEO, flusso="gemello")
        rec["saved_odom"] = [float(v) for v in saved_now]
        rec["saved_scene"] = [float(v) for v in isaac.save(nome=POSTO_RIALLINEO)]
        leggi_assetto(isaac, "riallineo_pausa", node)
        isaac.play()
        rec["pausa_s"] = round(time.time() - t0, 3)
        self.rec = rec


def stamp_dopo(stamp_s, passo=0.05):
    """First nuvole_icp stamp after the given one (stamps rounded to 0.1 s): threshold for
    esito_ramo that excludes the cloud at that stamp.
    """
    return None if stamp_s is None else round(float(stamp_s), 1) + passo


def leggi_traccia_csv(path):
    """Rows of a saved trace (dicts of strings; grandezze_traccia and true_path convert them). Empty
    list if the file is missing.
    """
    import csv
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _testo(path):
    try:
        return Path(path).read_text(errors="replace") if path else ""
    except OSError:
        return ""


def misure_icp(text, n=banco_g.RAMO_PRIME):
    """ICP quality of the first n clouds of a MOLA log: printf lines (MOLA_LO_DEBUG_ICP_QUALITY=1)
    and, if missing, ICP lines (kind=0) of the DEBUG output.
    """
    righe = banco_g.icp_lines(text)[:n]
    fonte = "printf"
    q = [r["q"] for r in righe]
    if not q:
        q = [r["q"] for r in banco_g.nuvole_icp(text)[:n] if r.get("q") is not None]
        fonte = "icp_kind0" if q else None
    return {"n": len(q), "fonte": fonte, "q_min": min(q) if q else None,
            "q_media": round(float(np.mean(q)), 4) if q else None,
            "buone": sum(r["good"] for r in righe) if righe else None, "q": q}


def record_prova_vincitore(j, tag, res, node, mola, dm=None, danno_m=None):
    """One identical trial of the winner: damage, branch during settling, ICP quality of the first
    20 clouds, outcome, tau. The damage is recorded anyway (even slow or with an invalid
    outcome): no 0.0 penalty.
    """
    _, end, ch, st, trav, frames = res
    lp = getattr(mola, "log_path", None)
    text = _testo(lp)
    tau = getattr(node, "last_tau", float("nan"))
    tau_ok = tau == tau
    return {"j": j, "tag": tag, "status": st, "valido": st in VALID,
            "end": [float(v) for v in end] if end is not None else None,
            "travelled": trav, "frames": frames, "chamfer": ch if ch == ch else None,
            "tau_s": tau if tau_ok else None, "lento": bool(tau_ok and tau > SLOW_TAU),
            "danno_cm": dm["damage"] * 100 if dm else None,
            "dev_mean_cm": dm["dev_mean"] * 100 if dm else None,
            "danno_metro": danno_m,
            "ramo": banco_g.esito_ramo(text), "icp_prime": misure_icp(text),
            "mola_log": str(lp) if lp else None,
            "twin_start": getattr(node, "last_twin_fp", None),
            **({"assetto": getattr(node, "last_assetto", None)} if ASSETTO["registra"] else {})}


def prove_del_vincitore(n, k, esegui, misura, node, mola, danno_ricerca_cm=None):
    """n identical twin rollouts with the winner genome from the same state (tag w{k}_prova{j}),
    before applying it. esegui(tag) -> rollout outcome; misura(res) -> (window_damage or None,
    danno_metro or None) on the trace just closed. The winner is applied anyway.
    """
    prove = []
    for j in range(int(n)):
        tag = f"w{k}_prova{j}"
        res = esegui(tag)
        end = res[1]
        dm, dmm = misura(res) if end is not None else (None, None)
        r = record_prova_vincitore(j, tag, res, node, mola, dm, dmm)
        prove.append(r)
        print(f"    prova {j}: [{r['status']}] danno {r['danno_cm']} cm, tau {r['tau_s']} s"
              f"{' (lento)' if r['lento'] else ''}, ramo {r['ramo']['classe']} (dy {r['ramo']['dy_cm']} cm), "
              f"q prime {r['icp_prime']['q_min']}/{r['icp_prime']['q_media']}")
    rami = [p["ramo"]["classe"] for p in prove]
    return {"n": int(n), "danno_ricerca_cm": danno_ricerca_cm, "prove": prove,
            "rami": {c: rami.count(c) for c in sorted(set(rami))},
            "regola": "D-31b: n prove identiche dallo stato di inizio metro; il vincitore si applica comunque; "
                      "danno vero registrato anche con esito lento o non valido (nessuna penalita' 0,0)"}


def gemello_scelto(eval_log, k, idx, out):
    """Twin that chose the attack (evaluation idx of window k): tag, saved trace, MOLA log, metre
    damage (comparison (a) with the victim).
    """
    if idx is None or not (0 <= idx < len(eval_log)):
        return None
    e = eval_log[idx]
    ts = e.get("twin_start") or {}
    return {"idx": int(idx), "tag": f"w{k}_e{idx}", "traccia": str(Path(out) / "traces" / f"w{k}_e{idx:02d}.csv"),
            "mola_log": ts.get("log"), "status": e.get("status"), "damage_cm": (e["damage"] * 100
                                                                                  if e.get("damage") is not None else None),
            "danno_metro": e.get("danno_metro")}


def confronti_d31(baseline_poly, vittima_rows, vittima_log, stamp_apertura, scelto, riallineato=None,
                  riallineato_rows=None, stamp_ripresa=None, dm_real=None, frame_log=None,
                  traccia_vittima=None, attendi_fino=None):
    """Victim-twin comparisons with the definitions of banco_g.confronto_coppia (incremental error
    and dm_real): (a) against the twin that chose (victim branch from the stream reopening), (b)
    against the realigned twin (victim branch from the first cloud after resume). Error from the
    saved traces, the MOLA logs and the per-frame log (waiting until attendi_fino for the log to
    be written). Without baseline: None.
    """
    if baseline_poly is None:
        return None
    if frame_log and attendi_fino is not None:
        attesa = attendi_fino - time.time()
        if 0 < attesa <= FRAME_LOG_FLUSH_S + 1.0:
            time.sleep(attesa)
    p0, u0 = banco_g.asse_baseline(baseline_poly)
    tv_ = _testo(vittima_log)
    gv = banco_g.grandezze_traccia(vittima_rows, p0, u0)
    dv = wrec.danno_metro(true_path(vittima_rows), baseline_poly) if vittima_rows else None
    out = {"definizioni": "criterio A (emendamento 8): banco_g.confronto_coppia", "stamp_apertura": stamp_apertura}
    if scelto is not None:
        rw = leggi_traccia_csv(scelto["traccia"])
        ramo_v = banco_g.esito_ramo(tv_, stamp0=stamp_dopo(stamp_apertura, -0.05) if stamp_apertura is not None else None)
        ramo_w = banco_g.esito_ramo(_testo(scelto.get("mola_log")))
        dw = wrec.danno_metro(true_path(rw), baseline_poly) if rw else None
        e1 = banco_g.e1_incrementale(traccia_vittima, vittima_log, scelto["traccia"], scelto.get("mola_log"),
                                     frame_log, True, False)
        out["a"] = {"gemello": scelto["tag"], "righe_traccia_gemello": len(rw),
                    **banco_g.confronto_coppia(gv, banco_g.grandezze_traccia(rw, p0, u0), ramo_v, ramo_w, dv, dw,
                                               e1=e1, dm_real=bool(dm_real))}
    if riallineato is not None and riallineato_rows is not None:
        ramo_v = banco_g.esito_ramo(tv_, stamp0=stamp_dopo(stamp_ripresa))
        ramo_r = banco_g.esito_ramo(_testo(riallineato.get("mola_log")))
        dr = wrec.danno_metro(true_path(riallineato_rows), baseline_poly) if riallineato_rows else None
        out["stamp_ripresa"] = stamp_ripresa
        e1 = banco_g.e1_incrementale(traccia_vittima, vittima_log, riallineato.get("traccia"),
                                     riallineato.get("mola_log"), frame_log, True, False)
        out["b"] = {"gemello": "riallineato", "righe_traccia_gemello": len(riallineato_rows),
                    **banco_g.confronto_coppia(gv, banco_g.grandezze_traccia(riallineato_rows, p0, u0),
                                               ramo_v, ramo_r, dv, dr, e1=e1, dm_real=bool(dm_real))}
    return out


def hold_restore(isaac, mola, node, k, saved=None):
    """Held window: robot back in the state saved at window start (the search left it where the last
    rollout ended) and twin MOLA stopped; the next window saves the same state.

    saved: true pose saved at window start (odom frame). While paused /chassis/odom is silent and
    the cache would keep the end pose of the last twin rollout: it is reset to the saved pose, so
    the next window (true_pose, twin base, carrot) starts from the restored state.
    """
    isaac.pause()
    pose = isaac.restore()
    mola.stop()
    if saved is not None and getattr(node, "true_source", "ros") == "ros":
        node.true = tuple(saved)
    if getattr(node, "fresh_cloud", False):
        count_restore(node, "trattenuta", k)
    return {"restored": True, "restore_pose": list(map(float, pose)) if pose is not None else None}


def attacker_phase_start(isaac, node, attacker, origin_mode, settle_s):
    """--victim-start w0: at run start only the attacker MOLA starts (clean clouds, same gating as
    the victim stream). Victim and shadow are not started; with --victim-never-clean the node
    drops the victim clouds (attack off), without the option they go to a topic with no
    subscribers. Attacker origin with the rules of VictimRunner.start.
    """
    node.set_victim_attack(False)
    node.set_victim_stream(True)
    attacker.start(node, None)
    if settle_s > 0:
        node.spin(settle_s)
    true_now = node.settled_pose(isaac)
    a_est = node.attacker_est_raw
    if origin_mode == "estimate" and a_est is not None:
        attacker.run_origin = wrec.origin_from_estimate(true_now, a_est)
    else:
        attacker.run_origin = true_now
    node.set_victim_stream(False)
    node.last_victim_close_wall = time.time()
    node.est_source = "twin"
    return {"attacker": wrec.origin_record(origin_mode, true_now, a_est, attacker.run_origin)}


def frame_di_controllo_vittima(node, run_origin, true_start):
    """Victim control frame: the pose it believes it has.

    Without --origine-coerente-vittima: the current estimate composed with the run origin, while
    est_origin stays the one read before waiting for the true pose and the realignment. With the
    option: estimate origin and frame from the same est_raw reading, as for the twin. If MOLA
    published a new pose between the two readings (settled_pose, assetto, realignment), without the
    option the victim believes it has moved by the jump (observed up to 30 cm).
    """
    if ORIGINE_VITTIMA["coerente"]:
        node.last_frame_coerenza = None      # no stale value from another metre
    if not ORIGINE_VITTIMA["coerente"] or not run_origin or node.est_raw is None:
        return node.compose(run_origin, node.est_raw) if run_origin else true_start
    prima = node.compose(run_origin, node.est_origin) if node.est_origin is not None else None
    lettura = node.est_raw
    node.est_origin = lettura
    node.est = (0.0, 0.0, 0.0)
    frame = node.compose(run_origin, lettura)
    node.last_frame_coerenza = {
        "attiva": True,
        "spostamento_origine_cm": (round(100 * math.hypot(frame[0] - prima[0], frame[1] - prima[1]), 3)
                                   if prima is not None else None),
        "spostamento_origine_deg": (round(math.degrees(angle_diff(frame[2], prima[2])), 3)
                                    if prima is not None else None)}
    return frame


def _victim_rollout(genome, isaac, victim, node, goal, horizon, tag, trace_path,
                    track_min, track_max, track_ref, settle_fixed, min_travel_frac,
                    catchup_sec, stop_on, attack_from, drift_reset_on, frame_start=None,
                    dopo_assestamento=None):
    """Rollout driven by the continuous victim (see rollout). Simulation paused and state already
    restored; the twin is stopped. dopo_assestamento (None = off): called at the end of settling,
    before driving, with (isaac, node, info); it brings the victim back to the same state before
    returning.
    """
    if ORIGINE_VITTIMA["coerente"]:
        node.last_frame_coerenza = None      # no value from a previous stretch
    if victim.crashed():
        return None, None, float("inf"), "victim_crash", 0.0, 0
    never_clean = CLOUD_SETTLE["never_clean"]
    if never_clean and genome is None:
        # defensive: callers do not drive the victim without a genome (victim_hold_fields)
        return None, None, float("inf"), "held_no_genome", 0.0, 0
    first = victim.deferred is not None and not victim.started
    shadow = (victim.deferred or {}).get("shadow")
    if first:
        # --victim-start w0: victim and shadow are born here, with the stream closed
        try:
            victim.sync_info = victim.spawn_deferred(node, shadow)
        except OmbraPersa:
            raise
        except Exception as e:
            return None, None, float("inf"), f"mola_spawn_fail: vittima {e}", 0.0, 0
    ok_g = node.set_victim_genome(genome) if genome is not None else True
    ok_a = node.set_victim_attack(attack_from == "spawn" and genome is not None)
    if never_clean and not (ok_g and ok_a):
        # genome or attack not confirmed: the stream does not open (infrastructure outcome)
        return None, None, float("inf"), "attack_unconfirmed", 0.0, 0
    node.chamfer = []; node.perc_samples = []; node.proc_ms_roll = []
    node.est_source = "victim"
    node.est = None
    node.est_t = 0.0
    target = settle_target(victim.started)
    c_count = lambda: getattr(node, "victim_rec_total", None)
    if never_clean:
        check_victim_topic(node)
    c0 = c_count() or 0
    t_open = time.time()
    node.pause_start_wall = node.last_victim_close_wall
    node.last_victim_open_wall = t_open
    node.set_victim_stream(True)
    isaac.play()
    if first:
        try:
            victim.wait_ready(check=node.drain)
            if shadow is not None:
                shadow.wait_ready(check=node.drain)
        except IsaacStall:
            raise
        except Exception as e:
            isaac.pause()
            node.set_victim_stream(False)
            node.est_source = "twin"
            return None, None, float("inf"), f"mola_not_ready: vittima {e}", 0.0, 0
    # First victim estimate after the stream reopens.
    deadline = time.time() + 15.0
    while time.time() < deadline and node.est_t < t_open:
        node.spin(0.1)
    if first and shadow is not None and node.est_t >= t_open:
        try:
            shadow.wait_pose(node)
        except OmbraPersa:
            raise
        except RuntimeError as e:
            isaac.pause()
            node.set_victim_stream(False)
            node.est_source = "twin"
            return None, None, float("inf"), f"no_pose: ombra {e}", 0.0, 0
        if sync_guard_active():
            g = sync_guard(node, victim, shadow)
            victim.sync_info = {**(victim.sync_info or {}), "guardia": g}
            if g["esito"] == "diverso":
                isaac.pause()
                node.set_victim_stream(False)
                node.est_source = "twin"
                raise ShadowSyncFail(f"primo stamp diverso: vittima {g['vittima']}, ombra {g['ombra']} "
                                     f"({g['fonte']})")
    # Estimate jump at resume: the victim restarts where it stopped (same physical state), but with
    # a stamp minutes ahead. A significant jump would mean MOLA reads the pause as a teleport.
    if node.est_t >= t_open and victim.last_est_end is not None and node.est_raw is not None:
        node.last_resume_jump_cm = 100 * math.hypot(node.est_raw[0] - victim.last_est_end[0],
                                                     node.est_raw[1] - victim.last_est_end[1])
    else:
        node.last_resume_jump_cm = None
    if node.est_t < t_open:
        isaac.pause()
        node.set_victim_stream(False)
        node.est_source = "twin"
        node.est_origin = None
        return None, None, float("inf"), "victim_no_pose", 0.0, 0
    node.last_victim_open_stamp = node.est_stamp_raw
    settle_rec = None
    if target > 0:
        # --settle-clouds: clouds delivered to the victim since opening, counted by the node
        first_pose_clouds = int((c_count() or 0) - c0)
        settle_rec = wait_clouds(node, c_count, c0, target, t_open)
        settle_rec["phase"] = "avvio" if first else "riapertura"
        settle_rec["first_pose_clouds"] = first_pose_clouds
        node.last_settle_waited = settle_rec["wall_s"]
        node.last_settle_ok = settle_rec["ok"]
    else:
        # Same fixed wait as the twins: the victim is at steady state and does not need it, but the
        # drift (reset at the genome) accumulates as in the candidates.
        if settle_fixed is not None and settle_fixed > 0:
            node.spin(settle_fixed)
        node.last_settle_waited = settle_fixed or 0.0
        node.last_settle_ok = True
    # Stretch origin in the victim frame: its current estimate.
    node.est_origin = node.est_raw
    node.est = (0.0, 0.0, 0.0)
    true_start = node.settled_pose(isaac)
    if first:
        victim.set_deferred_origins(node, true_start, shadow, victim.deferred.get("origin_mode", "true"))
        victim.settle_s = node.last_settle_waited
        victim.settle_ok = node.last_settle_ok
        victim.started = True
        victim.start_info = {"window": restore_window(tag), "tag": tag, "settle": settle_rec,
                             "sync": victim.sync_info, "genome_on_first_cloud": genome is not None}
    node.last_victim_settle_clouds = close_settle_record(settle_rec, c_count, c0, t_open)
    leggi_assetto(isaac, "fine_assestamento", node)
    if dopo_assestamento is not None:
        dopo_assestamento(isaac, node, {"tag": tag, "ruolo": "vittima", "victim": victim,
                                        "true_start": true_start, "genome": genome})
    # Control frame: the pose the victim believes it has (on the robot the controller only knows the
    # estimate), unless frame_start is given.
    if frame_start is None:
        frame_start = frame_di_controllo_vittima(node, victim.run_origin, true_start)
    node.last_align_err_cm = 100 * math.hypot(frame_start[0] - true_start[0], frame_start[1] - true_start[1])
    node.last_align_err_deg = math.degrees(angle_diff(frame_start[2], true_start[2]))
    goals = goal if isinstance(goal, list) else [goal]
    local_wps = [to_local(g, frame_start) for g in goals]
    if attack_from == "motion" and genome is not None:
        node.set_victim_attack(True)
        node.chamfer = []; node.perc_samples = []; node.proc_ms_roll = []
    if drift_reset_on == "drive":
        node.reset_victim_drift()

    def finish():
        victim.last_est_end = node.est_raw
        victim.last_quality = node.pose_quality if node.pose_quality == node.pose_quality else None
        node.set_victim_attack(False)
        node.set_victim_stream(False)
        node.last_victim_close_wall = time.time()
        node.est_source = "twin"
        node.est_origin = None

    if dopo_assestamento is not None and hasattr(dopo_assestamento, "alla_guida"):
        dopo_assestamento.alla_guida(node)
    return _drive_and_measure(node, isaac, victim, local_wps, horizon, true_start, trace_path,
                              track_min, track_max, track_ref, min_travel_frac,
                              catchup_sec, stop_on, finish=finish, frame_pose=frame_start)


# Twin starting state (set in main from --twin-state): with "victim" every twin receives, with the
# simulation stopped, the attacked clouds already sent to the victim (all or the last replay_last_s
# seconds) and starts from its state.
TWIN_STATE = {"mode": "clean", "last_s": 0.0, "rate_hz": 10.0, "victim": None}
# Last printf stamp of the previous twin log (window_record check).
TWIN_LOG_PREV = {"last_stamp": None}
# --metro-al-nodo: metre number sent to the node before every genome; metres not confirmed by the
# status, per window.
METRO = {"on": False, "non_confermati": []}
# --zone-attaccante: run directory, snapshot of the current metre, window record (snapshot, zones
# sent, zones applied per rollout).
ZONE = {"on": False, "out": None, "foto": None, "metro": None, "finestra": None, "seq": 0}
# Limits study (off by default): multiplier per family ({family: K}, empty = off;
# --moltiplicatore-limiti and --famiglie-limiti, equal to the node one) and genes fixed at -1 for
# all candidates (--blocca-gene, outside the search space).
LIMITI = {"mol": {}, "bloccati": []}
# --rampa-genoma R (node ramp): previous genome P of the metre (the last one applied to the victim,
# taken at metre start) and zones of P on the metre snapshot, sent on /attack/rampa before every new
# genome, per stream. R = 1: off, no message. sospesa: realigned twin (no ramp: it starts after
# settling).
RAMPA = {"R": 1, "vittima_ultimo": None, "prec": None, "prec_sha": None, "zone_prec": None,
         "metro": None, "sospesa": False, "seq": 0, "finestra": None, "ultimo": {}, "preinviata": {},
         "ultimi_dati": None}
RAMPA_ANNULLATA = "annullata"   # receipt state after the node cancel command
# --genoma-in-moto S (node step): same channel as the ramp for the previous genome P
# (prepara_rampa_metro, manda_rampa); S = 0: off. finestra: window record in history
# (genoma_in_moto), with the step state and the per-phase Chamfer per stream. cpf
# (--chamfer-per-fase, also turned on by --genoma-in-moto): chamfer_fasi = window record in history
# with the Chamfer per stream and phase.
GIM = {"S": 0.0, "kmin": 20, "kmax": 40, "finestra": None, "cpf": False, "chamfer_fasi": None}


def canale_rampa():
    """Previous genome sent to the node on /attack/rampa: ramp or step."""
    return RAMPA["R"] > 1 or GIM["S"] > 0


def gim_inizio_finestra():
    """--genoma-in-moto: empty window record (history[k]["genoma_in_moto"]); --chamfer-per-fase:
    empty record history[k]["chamfer_fasi"].
    """
    if GIM["S"] > 0:
        GIM["finestra"] = {"S_cm": GIM["S"], "kmin": GIM["kmin"], "kmax": GIM["kmax"], "flussi": {}}
    if GIM["cpf"]:
        GIM["chamfer_fasi"] = {}
    return GIM["finestra"]


GIM_STATI_FINALI = {"attivo", "k_max", "metro0"}


def gim_flusso(sg, fase_salvataggio=None):
    """Record of one stream from the node status (genoma_in_moto[gemello|vittima]) at rollout end:
    state, entry of N, reference, mean Chamfer per phase.
    """
    sg = sg or {}
    return {"stato": sg.get("stato"), "k_ingresso": sg.get("k_ingresso"),
            "spost_ingresso_cm": sg.get("spost_ingresso_cm"), "spost_prima_cm": sg.get("spost_prima_cm"),
            "buco_stamp_s": sg.get("buco_stamp_s"), "rif_k": sg.get("rif_k"),
            "fase_al_salvataggio_D31c": fase_salvataggio, **chamfer_fasi_flusso(sg)}


def chamfer_fasi_flusso(sc):
    """Mean Chamfer per phase of a stream from the node status (sum / number of computed values;
    None without values).
    """
    sc = sc or {}

    def media(f):
        n = int(sc.get(f"n_chamfer_{f}") or 0)
        return (float(sc[f"chamfer_{f}_somma"]) / n) if n > 0 else None

    return {"chamfer_P_cm": media("P"), "n_chamfer_P": int(sc.get("n_chamfer_P") or 0),
            "chamfer_N_cm": media("N"), "n_chamfer_N": int(sc.get("n_chamfer_N") or 0)}


def gim_fase_salvataggio(posto):
    """Step phase in the state saved for the realigned twin (status slot): "P" before N enters, "N"
    from the entry cloud, None without step.
    """
    f = (posto or {}).get("gradino_fase")
    return "P" if f == "P" else ("N" if f in ("ingresso", "N") else None)


def gim_registra(node, flusso_nodo, nome, attesa_s=0.2, tetto_s=1.0):
    """--genoma-in-moto, end of a successful rollout (Isaac paused): step state and per-phase
    Chamfer of the node stream, from a status published at least attesa_s after the end, in the
    window record under the stream name (vittima, w{k}_e{i}, w{k}_riallineato, ...). With
    --chamfer-per-fase also the per-phase Chamfer in the window chamfer_fasi record.
    """
    gim = GIM["S"] > 0 and GIM["finestra"] is not None
    cpf = GIM["cpf"] and GIM["chamfer_fasi"] is not None
    if not (gim or cpf):
        return None
    t0 = time.time()
    while time.time() - t0 < tetto_s and (getattr(node, "status_t", None) or 0.0) < t0 + attesa_s:
        node.spin(0.02)
    rec = None
    if cpf:
        GIM["chamfer_fasi"][nome] = chamfer_fasi_flusso((getattr(node, "node_cpf", None) or {}).get(flusso_nodo))
    if gim:
        rec = gim_flusso((getattr(node, "node_gim", None) or {}).get(flusso_nodo))
        GIM["finestra"]["flussi"][nome] = rec
    return rec


def cpf_mismatch(node, on, timeout=10.0):
    """Startup check of --chamfer-per-fase against the node status (chamfer_fasi): on or off in
    both.
    """
    t0 = time.time()
    if on:
        while not getattr(node, "node_cpf_on", False) and time.time() - t0 < timeout:
            node.spin(0.1)
        if not getattr(node, "node_cpf_on", False):
            return ("--chamfer-per-fase (o --genoma-in-moto) richiede il nodo di perturbazione con "
                    "--chamfer-per-fase o --genoma-in-moto (status senza chamfer_fasi)")
        return None
    if getattr(node, "node_cpf_on", False):
        return ("il nodo di perturbazione ha --chamfer-per-fase e l'orchestratore no: la Chamfer per fase non "
                "andrebbe nella history. Lanciare tutti e due con l'opzione o tutti e due senza")
    return None


def prepara_rampa_metro(node, k):
    """--rampa-genoma, start of metre k after the snapshot: P = last genome given to the victim
    (none at metre 0 or if the victim has not had one yet) and zones of P on the snapshot of
    metre k with the same limits as the N zones (file w{k}_{sha_P}.npz, created or reloaded).
    Window record in RAMPA["finestra"].
    """
    RAMPA.update({"prec": None, "prec_sha": None, "zone_prec": None, "metro": int(k), "ultimo": {}})
    P = RAMPA.get("vittima_ultimo")
    RAMPA["finestra"] = {"R": RAMPA["R"], "metro": int(k), "genoma_prec": P, "genoma_prec_sha": None,
                         "zone_prec": None, "messaggi": []}
    if P is None or k <= 0:
        RAMPA["finestra"]["motivo"] = "metro 0" if k <= 0 else "nessun genoma applicato alla vittima"
        return RAMPA["finestra"]
    if ZONE.get("foto") is None or ZONE.get("metro") != k:
        raise ErroreConfigurazioneZone(f"rampa w{k}: fotografia del metro assente")
    try:
        z = za.zone_genoma(Path(ZONE["out"]) / "zone", ZONE["foto"], np.asarray(P, dtype=float), k,
                           limiti_zone(node, f"rampa w{k}"))
    except (za.ZoneNonValide, OSError) as e:
        raise ErroreConfigurazioneZone(f"rampa w{k}: zone del genoma precedente: {e}")
    zp = {"path": str(z.path), "sha256": z.sha256, "hash": z.hash, "genoma_sha": z.genoma_sha}
    RAMPA.update({"prec": list(P), "prec_sha": z.genoma_sha, "zone_prec": zp})
    RAMPA["finestra"].update(genoma_prec_sha=z.genoma_sha, zone_prec=zp)
    print(f"  rampa w{k}: genoma precedente {z.genoma_sha}, zone {Path(z.path).name} (R {RAMPA['R']})")
    return RAMPA["finestra"]


def stato_generatore_mancante(node, timeout=10.0):
    """--riallinea-dopo-assestamento: the realigned twin restarts from the victim generator state,
    so the node must have --stato-generatore (key stato_generatore in the status). Error reason
    or None.
    """
    t0 = time.time()
    while not isinstance(getattr(node, "st_stato_generatore", None), dict) and time.time() - t0 < timeout:
        node.spin(0.1)
    if not isinstance(getattr(node, "st_stato_generatore", None), dict):
        return ("--riallinea-dopo-assestamento richiede il nodo di perturbazione con --stato-generatore (status "
                "senza stato_generatore): il riallineato non riceverebbe lo stato del generatore della vittima")
    return None


def errore_riallineo(rec):
    """Realignment requested: realigned twin not run or victim generator state not transferred =
    configuration error (reason), otherwise None.
    """
    if rec is None:
        return "riallineato: aggancio dopo l'assestamento non chiamato"
    if rec.get("gemello") is None:
        return f"riallineato non eseguito ({rec.get('saltato')})"
    if rec.get("generatore") != "trasferito":
        return f"riallineato senza lo stato del generatore della vittima ({rec.get('generatore')})"
    return None


def rampa_mismatch(node, R, timeout=10.0):
    """Startup check of --rampa-genoma against the node status: same R in both, or off in both."""
    t0 = time.time()
    if R > 1:
        while getattr(node, "status_t", None) is None and time.time() - t0 < timeout:
            node.spin(0.1)
        while not getattr(node, "node_rampa_on", False) and time.time() - t0 < timeout:
            node.spin(0.1)
        rn = (getattr(node, "node_rampa", None) or {}).get("R") if getattr(node, "node_rampa_on", False) else None
        if rn != R:
            return (f"--rampa-genoma {R} richiede il nodo di perturbazione con --rampa-genoma {R} "
                    f"(status: {'senza rampa' if rn is None else 'R ' + str(rn)})")
        return None
    # off: no extra wait, only the status messages already received
    if getattr(node, "node_rampa_on", False):
        return ("il nodo di perturbazione ha --rampa-genoma e l'orchestratore no: nessun genoma precedente "
                "verrebbe mandato. Lanciare tutti e due con lo stesso R o tutti e due senza")
    return None


def gim_mismatch(node, S, kmin, kmax, timeout=10.0):
    """Startup check of --genoma-in-moto against the node status: same S, kmin and kmax in both, or
    off in both.
    """
    t0 = time.time()
    if S > 0:
        while getattr(node, "status_t", None) is None and time.time() - t0 < timeout:
            node.spin(0.1)
        while not getattr(node, "node_gim_on", False) and time.time() - t0 < timeout:
            node.spin(0.1)
        gn = getattr(node, "node_gim", None) if getattr(node, "node_gim_on", False) else None
        att = (float(S), int(kmin), int(kmax))
        dn = None if gn is None else (float(gn.get("S_cm")), int(gn.get("kmin")), int(gn.get("kmax")))
        if dn != att:
            return (f"--genoma-in-moto {S} (kmin {kmin}, kmax {kmax}) richiede il nodo di perturbazione con gli "
                    f"stessi valori (status: {'spento' if dn is None else 'S %s, kmin %s, kmax %s' % dn})")
        return None
    if getattr(node, "node_gim_on", False):
        return ("il nodo di perturbazione ha --genoma-in-moto e l'orchestratore no: nessun genoma precedente "
                "verrebbe mandato. Lanciare tutti e due con gli stessi valori o tutti e due senza")
    return None


def zone_mismatch(node, want, timeout=10.0):
    """Startup check of --zone-attaccante against the node status."""
    if want:
        t0 = time.time()
        while node.node_zone_on is not True and time.time() - t0 < timeout:
            node.spin(0.1)
        if node.node_zone_on is not True:
            return ("--zone-attaccante richiede il nodo di perturbazione lanciato con la stessa "
                    "opzione (status senza zone_attaccante)")
        return None
    if node.node_zone_on is True:
        return ("il nodo di perturbazione ha --zone-attaccante e l'orchestratore no: nessuna zona "
                "verrebbe mandata. Lanciare tutti e due con l'opzione o tutti e due senza")
    return None


def _note_twin_log(node, mola):
    """First ICP of the just-closed twin and window_record flags, from its MOLA log (printf lines
    with MOLA_LO_DEBUG_ICP_QUALITY=1). Without lines the flags stay None. Does not change the
    rollout outcome.
    """
    try:
        rows = wrec.read_printf(getattr(mola, "log_path", None))
        node.last_twin_flags = wrec.twin_icp_flags(rows, TWIN_LOG_PREV["last_stamp"])
        TWIN_LOG_PREV["last_stamp"] = rows[-1]["stamp_s"] if rows else None
    except Exception as e:
        node.last_twin_flags = {"twin_flags_error": str(e)}


def _note_twin_fp(node, mola):
    """Complete the fingerprint of the just-closed twin with adaptive sigma and keyframes from its
    log (None if the log has no printf or DEBUG lines).
    """
    fp = getattr(node, "last_twin_fp", None)
    if not isinstance(fp, dict):
        return
    try:
        fp.update(tv.start_fingerprint_from_log(getattr(mola, "log_path", None), fp.get("est_stamp")))
        fp["log"] = str(getattr(mola, "log_path", None))
        if fp.get("source") == "ombra_g":
            lp = getattr(mola, "log_path", None)
            text = Path(lp).read_text(errors="replace") if lp and Path(lp).is_file() else ""
            fp["g_check"] = tv.g_rollout_check(text, TWIN_STATE.get("shadow_state"),
                                               fp.get("g_resave_path"),
                                               getattr(node, "twin_clock_first_ns", None))
            fp["g_clock_arm"] = getattr(node, "last_twin_clock_arm", None)
    except Exception as e:
        fp["fp_error"] = str(e)


def _twin_alignment_record(node, true_start, fp):
    """Twin alignment at rollout start: true pose against the victim true pose in the saved state,
    twin estimate (in the working frame) against the pose believed by the attacker. With the
    clean twin its estimate restarts from zero and has no common frame with the attacker: None.
    """
    mode = TWIN_STATE.get("mode")
    est = (fp or {}).get("est_raw")
    victim = TWIN_STATE.get("victim")
    origin = None
    ref = TWIN_STATE.get("attacker_belief")
    if mode == "replay":
        origin = TWIN_STATE.get("attacker_origin")
    elif mode in ("victim", "maprestore") and victim is not None:
        origin = victim.run_origin
    elif mode in tv.SHADOW_MODES:
        # twin estimate in the shadow frame, compared with the pose believed by the shadow
        origin = (TWIN_STATE.get("shadow_state") or {}).get("origin")
        ref = (TWIN_STATE.get("shadow_state") or {}).get("believed")
    est_w = wrec.compose_pose(origin, est) if (origin is not None and est is not None
                                               and (fp or {}).get("source") != "prima_posa_dopo_avvio") else None
    rec = tv.twin_alignment(true_start, TWIN_STATE.get("saved"), est_w, ref)
    rec["twin_est_frame"] = ({"replay": "origine dell'attaccante", "victim": "origine della vittima",
                              "maprestore": "origine della vittima",
                              "ombra_e": "origine dell'ombra",
                              "ombra_g": "origine dell'ombra"}.get(mode) if est_w is not None else None)
    if mode in tv.SHADOW_MODES:
        rec["attacker_world_is"] = "posa creduta dall'ombra"
    rec["twin_init"] = TWIN_STATE.get("init", "true")
    return rec


def attacker_init_target(node, attacker, saved, saved_scene, believed, vel_mode="zero",
                         buf_name="attacker_buf"):
    """Twin state in Isaac from the attacker estimate: velocity from the last poses of its MOLA
    (MOLA frame, rotated into the working frame with the attacker origin), zero with the robot
    still (tv.twin_init_target). With --twin-init ombra_est the same rule on the shadow: attacker
    is then the ShadowRunner and buf_name "shadow_buf".
    """
    vel = tv.velocity_from_poses(list(getattr(node, buf_name, [])))
    if vel is not None and attacker is not None and attacker.run_origin is not None:
        c, s_ = math.cos(attacker.run_origin[2]), math.sin(attacker.run_origin[2])
        vel = (c * vel[0] - s_ * vel[1], s_ * vel[0] + c * vel[1], vel[2], vel[3])
    rec = tv.twin_init_target(saved, saved_scene, believed, vel[:3] if vel is not None else None,
                              vel_mode=vel_mode)
    rec["vel_poses_n"] = vel[3] if vel is not None else 0
    return rec


def twin_config_fields(args):
    """Twin variants in use and their consistency with the threat model (history)."""
    rec = {"twin_state": args.twin_state,
           "replay_frame": args.replay_frame if args.twin_state == "replay" else None,
           "twin_init": args.twin_init,
           "twin_init_vel": (getattr(args, "twin_init_vel", None)
                             if args.twin_init in ("attacker_est", "ombra_est") else None),
           "uniform_sim_time": args.uniform_sim_time,
           "drift_reset_on": args.drift_reset_on, "mola_ros_args": TWIN_STATE.get("ros_args"),
           "twin_init_target": TWIN_STATE.get("init_target"),
           "mola_stop_signal": getattr(args, "mola_stop_signal", "term"),
           "victim_mola_env": list(getattr(args, "victim_mola_env", None) or []),
           "victim_shadow_env": bool(getattr(args, "victim_shadow_env", False))}
    if getattr(args, "mola_binario_g1", False):
        rec["mola_binario_g1"] = TWIN_STATE.get("binario_g1")
    if args.twin_state in tv.SHADOW_MODES:
        rec["shadow_state"] = TWIN_STATE.get("shadow_state")
    if getattr(args, "no_nominal_rollout", False):
        rec["no_nominal_rollout"] = True
    rec.update(TWIN_STATE.get("coherence") or tv.coherence_record(
        args.twin_state, args.twin_init, args.twin_align, args.replay_frame))
    return rec


RESTORE_KINDS = ("nominale", "valutazione", "rivalutazione", "vittima", "gate", "avanzamento", "ripetuto")


def restore_kind(tag, victim, retry=False):
    """Restore kind from the rollout tag (counter of --fresh-cloud-after-restore)."""
    if retry:
        return "ripetuto"
    import re
    if victim:
        return "vittima"
    t = re.sub(r"^w\d+_", "", str(tag))
    if "nominal" in t:
        return "nominale"
    if t.startswith(("reeval", "sel", "top")):
        return "rivalutazione"
    if t.startswith("prova"):
        return "prova_vincitore"      # --prove-vincitore
    if t.startswith("riallineato"):
        return "riallineato"          # --riallinea-dopo-assestamento
    if t.startswith("gate"):
        return "gate"
    if t.startswith("advance"):
        return "avanzamento"
    return "valutazione"


def restore_window(tag):
    import re
    m = re.match(r"^w(\d+)_", str(tag))
    return int(m.group(1)) if m else None


def count_restore(node, kind, window):
    """Record a restore command just sent to Isaac (only with --fresh-cloud-after-restore): restore
    source independent of the arms.
    """
    log = getattr(node, "restore_log", None)
    if log is None:
        log = node.restore_log = []
    log.append({"i": len(log), "window": window, "kind": kind, "t_wall": round(time.time(), 3)})


def restore_counts(node, window):
    """Window restores by kind and total since start (history)."""
    log = getattr(node, "restore_log", None) or []
    per = {k: 0 for k in RESTORE_KINDS}
    for r in log:
        if r["window"] == window:
            per[r["kind"]] = per.get(r["kind"], 0) + 1
    return {"per_tipo": per, "finestra": sum(per.values()), "totale": len(log)}


def fresh_cloud_mismatch(node, want, timeout=10.0):
    """Startup check of --fresh-cloud-after-restore against the node status: error message if the
    two options differ, otherwise None. With the option on it waits up to timeout for a status
    with fresh_cloud; off it does not wait: error only if a status already received says the node
    has the option (the check is repeated after the victim start).
    """
    if want:
        t0 = time.time()
        while node.node_fresh_cloud is not True and time.time() - t0 < timeout:
            node.spin(0.1)
        if node.node_fresh_cloud is not True:
            return ("--fresh-cloud-after-restore richiede il nodo di perturbazione lanciato con "
                    "la stessa opzione (status senza fresh_cloud)")
        return None
    if node.node_fresh_cloud is True:
        return ("il nodo di perturbazione ha --fresh-cloud-after-restore (status con fresh_cloud) e "
                "l'orchestratore no: il filtro non verrebbe mai armato. Lanciare tutti e due con "
                "l'opzione o tutti e due senza")
    return None


def _twin_start_fields(node):
    """Fingerprint and alignment of the last twin rollout (history)."""
    rec = {"twin_start": getattr(node, "last_twin_fp", None),
           "twin_alignment": getattr(node, "last_twin_align", None),
           "mola_stop": getattr(node, "last_mola_stop", None)}
    if CONTROLLO_ETA["soglia"] is not None:
        # --controllo-eta: measurements of every rollout attempt
        rec["eta_nuvola"] = getattr(node, "last_eta_nuvola", None)
    if getattr(node, "fresh_cloud", False):
        # filter arm outcome in this rollout (--fresh-cloud-after-restore)
        rec["fresh_arm"] = getattr(node, "last_fresh_arm", None)
    if CLOUD_SETTLE["n"] > 0:
        # --settle-clouds: clouds and duration from play to driving for this twin
        rec["settle_clouds"] = getattr(node, "last_settle_clouds", None)
    if ASSETTO["registra"]:
        # --registra-assetto: rollout readings (after restore, set_state, end of settling)
        rec["assetto"] = getattr(node, "last_assetto", None)
    if RAMPA["R"] > 1 and hasattr(node, "stato_rampa"):
        # --rampa-genoma: message sent and twin ramp state at rollout end
        rec["rampa"] = node.stato_rampa("gemello")
    return rec


def _replay_to_twin(node, mola, tag):
    """Replay the victim history to the just-started twin and wait until its estimate reaches the
    last replayed stamp. Returns the replay record (None if there is no history to replay).
    """
    victim = TWIN_STATE.get("victim")
    mode = TWIN_STATE.get("mode")
    # Replay mode (consistent with the threat model): the node recording is enough, the victim
    # estimate is needed neither to start nor to decide.
    if mode == "replay":
        if not node.victim_rec_n:
            return None
        # An estimate left over from the previous twin would satisfy the stamp criterion below at
        # once.
        node.est_raw = None
        node.est_stamp_raw = float("nan")
    elif mode != "victim" or victim is None or victim.last_est_end is None \
            or not node.victim_rec_n:
        return None
    t0 = time.time()
    n_rep, last_stamp, t_rep = node.twin_replay(TWIN_STATE["last_s"], TWIN_STATE["rate_hz"])
    rec = {"n": n_rep, "last_s": TWIN_STATE["last_s"], "rate_hz": TWIN_STATE["rate_hz"],
           "replay_s": round(t_rep, 2), "last_stamp": last_stamp,
           "first_stamp": getattr(node, "replay_first_stamp", None),
           "victim_rec_n": node.victim_rec_n,
           "victim_rec_total": getattr(node, "victim_rec_total", None)}
    tot = rec["victim_rec_total"]
    rec["history_truncated"] = bool(TWIN_STATE["last_s"] > 0 or tot is None
                                    or tot > node.victim_rec_n or n_rep < node.victim_rec_n)
    if n_rep <= 0 or last_stamp is None:
        rec["status"] = "no_clouds"
        return rec
    try:
        mola.wait_ready(check=node.drain)
    except OmbraPersa:
        raise
    except Exception as e:
        rec["status"] = f"mola_not_ready: {e}"
        return rec
    deadline = time.time() + 15.0
    while time.time() < deadline and not (node.est_stamp_raw == node.est_stamp_raw and node.est_raw is not None
                                          and node.est_stamp_raw >= last_stamp - 0.05):
        node.spin(0.1)
    rec["status"] = "ok" if (node.est_raw is not None and node.est_stamp_raw == node.est_stamp_raw
                             and node.est_stamp_raw >= last_stamp - 0.05) else "stale_pose"
    rec["twin_est_stamp"] = node.est_stamp_raw if node.est_stamp_raw == node.est_stamp_raw else None
    rec["twin_est_raw"] = list(map(float, node.est_raw)) if node.est_raw is not None else None
    vend = getattr(victim, "last_est_end", None) if victim is not None else None
    # In replay the victim estimate is only a lab measurement (lab_* fields): it does not enter
    # start, guard, frame or selection.
    pre = "lab_" if mode == "replay" else ""
    if vend is not None:
        rec[pre + "victim_est_end"] = list(map(float, vend))
        if node.est_raw is not None:
            rec[pre + "diff_cm"] = round(100 * math.hypot(node.est_raw[0] - vend[0],
                                                          node.est_raw[1] - vend[1]), 3)
            rec[pre + "diff_deg"] = round(math.degrees(angle_diff(node.est_raw[2], vend[2])), 3)
    rec["twin_quality"] = node.pose_quality if node.pose_quality == node.pose_quality else None
    rec[pre + "victim_quality_end"] = getattr(victim, "last_quality", None)
    rec["total_s"] = round(time.time() - t0, 2)
    return rec


def _maprestore_to_twin(node, mola, tag):
    """O(1) twin restore via map (--twin-state maprestore): load into the twin the victim map saved
    at window start and relocalise at the pose believed by the victim. Unlike the replay (which
    feeds clouds with the simulation stopped and makes the estimate converge at once),
    map_load/relocalize fix the state but the estimate is confirmed only by the first scan after
    isaac.play(): fidelity (twin-vs-victim diff_cm) is measured after wait_ready, in
    _rollout_once. Here only load+relocalize, returning a record with the service outcomes (None
    if not in maprestore mode).
    """
    victim = TWIN_STATE.get("victim")
    path = TWIN_STATE.get("victim_map_path")
    if TWIN_STATE.get("mode") != "maprestore" or victim is None \
            or victim.last_est_end is None or not path:
        return None
    t0 = time.time()
    rec = {"mode": "maprestore", "victim_map_path": str(path),
           "victim_est_end": list(map(float, victim.last_est_end))}
    # map_load with the simulation stopped (the twin service is up). Relocalisation on /initialpose
    # does NOT go here: the initial MOLA estimate at the first scan after play would bring it back
    # to the origin. /initialpose is published AFTER wait_ready, in _rollout_once, where fidelity is
    # measured.
    ok_load, err_load = node.twin_map_load(path, mola)
    rec["map_load_ok"] = ok_load
    rec["status"] = "loaded" if ok_load else "map_load_failed"
    if not ok_load:
        rec["error"] = err_load
    rec["total_s"] = round(time.time() - t0, 2)
    return rec


# Isaac stall policy (--stall-policy): with "tolerate" a transient stall (the loop restarts within
# max_wait_s) during a twin rollout or a replay is recorded and the step repeated; the victim is
# unaffected because the simulation is paused. A stall in the applied stretch, Isaac not restarting
# or more than max_per_window stalls in the same window end the run. Isaac blocks after a clock jump
# come in groups of 2-20 of 10 s each: wait up to 15 min.
STALL_POLICY = {"tolerate": False, "max_wait_s": 900.0, "max_per_window": 6, "events": [], "window": None}


def _isaac_stable(isaac, node, span_s=2.0):
    """True if the Isaac loop updates the pose file without interruption for span_s seconds: blocks
    come in groups of 10 s with a single update in between, so one update is not enough.
    """
    t0 = time.time()
    while time.time() - t0 < span_s:
        if time.time() - isaac._pose_file_wall() > 0.5:
            return False
        node.spin(0.1)
    return True


# Same pattern as the check in scripts/run_campaign.sh (pgrep -f).
_RX_PERTURBATION = r"^python3( -u)? .*perturbation_node\.py"


def _perturbation_vivo():
    """pids of the running perturbation_node processes, from the command line in /proc."""
    import re
    rx = re.compile(_RX_PERTURBATION)
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if rx.search(cmd):
            pids.append(int(d))
    return pids


FRAME_LOG_FLUSH_S = 1.0   # the per-frame log is written every second (FrameLog.flush_s)


def controlla_catena(args, node, victim, window, t0):
    """Chain checks in window `window` before closing a run as "attack detected"
    (--controllo-catena).
    t0: window start ({"wall", "stamp_tracce"}). Thresholds already in use: STALL_SEC and
    STALL_POLICY for Isaac, adapter events and p99 (--adapter-events; without monitoring it stays
    undetermined and the record says so), gaps over 1 s in the clouds delivered to the victim, also
    in the tail up to the decision (--frame-log-nodo), victim poses with good ICP not received by
    the orchestrator for over 1 s, victim time_onLidar over 2x 100 ms (with --rallentamento-mola
    scoperto, if it is the only signal, it is not a failure; with risultato it never is),
    perturbation node alive and with a status in the window.
    """
    # per-frame log rows up to the stream closing: wait for one flush
    chiuso = getattr(node, "last_victim_close_wall", None)
    if getattr(args, "frame_log_nodo", None) and isinstance(chiuso, (int, float)):
        attesa = chiuso + FRAME_LOG_FLUSH_S + 0.5 - time.time()
        if 0 < attesa <= FRAME_LOG_FLUSH_S + 0.5:
            time.sleep(attesa)
    now = time.time()
    t0_wall = (t0 or {}).get("wall") or 0.0
    c = {}
    # Isaac: tolerated stalls in the window (_isaac_recovery) and age of the loop pose
    stalli = [e for e in STALL_POLICY["events"] if e.get("window") == window]
    try:
        with open(POSE_FILE) as f:
            age = now - float(json.load(f).get("wall", 0.0))
    except (OSError, ValueError):
        age = None
    mot = []
    if stalli:
        mot.append(f"{len(stalli)} stalli di Isaac nella finestra")
    if age is not None and age > STALL_SEC:
        mot.append(f"posa del ciclo di Isaac ferma da {age:.1f} s (soglia {STALL_SEC:.0f} s)")
    c["isaac"] = {"stalli": stalli, "eta_posa_s": None if age is None else round(age, 2),
                  "esito": "guasto" if mot else ("ok" if age is not None else "non_determinato"),
                  "motivo": "; ".join(mot) or (None if age is not None else f"{POSE_FILE} non leggibile")}
    # adapter
    ev = None
    if args.adapter_events and os.path.exists(args.adapter_events):
        try:
            with open(args.adapter_events) as f:
                ev = [json.loads(r) for r in f if r.strip()]
        except (OSError, ValueError):
            ev = None
    c["adattatore"] = wrec.eventi_adattatore(ev, t0_wall, now)
    # clouds delivered to the victim
    c["nuvole_vittima"] = wrec.buchi_consegna_vittima(
        wrec.leggi_csv(getattr(args, "frame_log_nodo", None)), t0_wall, now,
        never_clean=bool(getattr(args, "victim_never_clean", False)))
    tracce = wrec.leggi_csv(getattr(victim, "traces_path", None))
    # victim poses lost between MOLA and orchestrator
    q_min, _ = wrec.soglia_icp_buona({**os.environ, **(getattr(victim, "extra_env", None) or {})})
    c["posa_vittima"] = wrec.pose_non_ricevute(
        wrec.read_printf(getattr(victim, "log_path", None)), getattr(node, "victim_est_stamp", None),
        tracce=tracce, q_min=q_min)
    # victim MOLA timings
    c["mola_vittima"] = wrec.tempi_mola_vittima(tracce, (t0 or {}).get("stamp_tracce"))
    # perturbation_node
    pids = _perturbation_vivo()
    st_t = getattr(node, "status_t", None)
    mot = []
    if not pids:
        mot.append("perturbation_node non in esecuzione")
    if st_t is None or st_t < t0_wall:
        mot.append("nessuno stato del perturbation_node nella finestra")
    c["perturbation"] = {"pid": pids, "ultimo_stato_wall": st_t,
                         "esito": "guasto" if mot else "ok", "motivo": "; ".join(mot) or None}
    return wrec.guasto_catena(c, getattr(args, "rallentamento_mola", "guasto"))


def controlli_incompleto(args, t0_wall, t1_wall):
    """Genome-independent chain signals for a twin rollout closed with timeout or stopped_early
    (--incompleti-regola): Isaac stalls and age of its pose, adapter events, gaps >= 1 s in the
    clouds entering the node (--frame-log-nodo). Waits for a flush of the per-frame log, as in
    controlla_catena.
    """
    if getattr(args, "frame_log_nodo", None):
        attesa = t1_wall + FRAME_LOG_FLUSH_S + 0.5 - time.time()
        if 0 < attesa <= FRAME_LOG_FLUSH_S + 0.5:
            time.sleep(attesa)
    try:
        with open(POSE_FILE) as f:
            age = time.time() - float(json.load(f).get("wall", 0.0))
    except (OSError, ValueError):
        age = None
    ev = None
    if args.adapter_events and os.path.exists(args.adapter_events):
        try:
            with open(args.adapter_events) as f:
                ev = [json.loads(r) for r in f if r.strip()]
        except (OSError, ValueError):
            ev = None
    return {"isaac": wrec.stalli_isaac_rollout(STALL_POLICY["events"], t0_wall, t1_wall, age, STALL_SEC),
            "adattatore": wrec.eventi_adattatore(ev, t0_wall, t1_wall),
            "nuvole_nodo": wrec.buchi_arrivi_nodo(
                wrec.leggi_csv(getattr(args, "frame_log_nodo", None)), t0_wall, t1_wall)}


def classe_rollout_incompleto(args, node, mola, status, t0_wall, t1_wall):
    """Class of a twin rollout outside eval_genome (top-K re-evaluation): same checks, same calc_ok
    and same rule 8 as the candidates.
    """
    proc = float(np.mean(node.proc_ms_roll)) if node.proc_ms_roll else float("nan")
    calc_ok = (args.max_proc_ms is None or args.max_proc_ms <= 0 or proc != proc
               or proc <= args.max_proc_ms)
    acc = wrec.mola_accorge(wrec.read_printf(getattr(mola, "log_path", None)),
                            args.accorge_icp_rifiutate, args.accorge_senza_posa_s)
    return wrec.classe_incompleto(status, controlli_incompleto(args, t0_wall, t1_wall), acc, calc_ok)


def _isaac_recovery(isaac, node, err, phase):
    """After a transient stall wait until the Isaac loop updates the pose file again, record the
    event and pause again. Re-raise the stall if the policy is abort, if Isaac does not restart
    or if the stalls in the window exceed the maximum.
    """
    ev = {"window": STALL_POLICY["window"], "phase": phase, "error": str(err), "t_wall": time.time()}
    STALL_POLICY["events"].append(ev)
    n_win = sum(1 for e in STALL_POLICY["events"] if e["window"] == STALL_POLICY["window"])
    if not STALL_POLICY["tolerate"] or n_win > STALL_POLICY["max_per_window"]:
        ev["outcome"] = "run_abort"
        raise err
    watch = node.watch
    node.watch = None
    try:
        node.stop_robot()
    except Exception:
        pass
    t0 = time.time()
    ok = False
    n_blocchi = 0
    while time.time() - t0 < STALL_POLICY["max_wait_s"]:
        if _isaac_stable(isaac, node, 2.0):
            try:
                isaac.pause()
                ok = True
                break
            except IsaacStall:
                n_blocchi += 1      # new block during the pause: wait again
                continue
        node.spin(0.2)
    ev["recovered_after_s"] = round(time.time() - t0, 1)
    ev["pause_retries"] = n_blocchi
    if not ok:
        ev["outcome"] = "run_abort_no_recovery"
        node.watch = watch
        raise err
    node.watch = watch
    ev["outcome"] = "retry"
    print(f"  STALLO TRANSITORIO di Isaac ({phase}): {err}; ciclo ripreso dopo "
          f"{ev['recovered_after_s']} s, {n_win} nella finestra: passo rifatto")
    return ev


def _twin_misaligned(node, args, gap0):
    """Twin not aligned: in victim mode the replay says so (out of consensus or failed), otherwise
    the offset from the truth at the first row.
    """
    rep = getattr(node, "last_replay", None)
    if TWIN_STATE.get("mode") in ("victim", "replay") and rep is not None:
        if rep.get("status") != "ok":
            return True, f"rigioco {rep.get('status')}"
        if rep.get("replay_outlier"):
            return True, f"rigioco fuori consenso {rep.get('replay_dev_cm')} cm"
        return False, ""
    if TWIN_STATE.get("mode") == "maprestore" and rep is not None:
        if rep.get("status") != "ok":
            return True, f"map-restore {rep.get('status')}"
        d = rep.get("diff_cm")
        if args.twin_gap0_max > 0 and d is not None and d > args.twin_gap0_max:
            return True, f"map-restore scarto {d:.0f} cm > {args.twin_gap0_max:.0f}"
        return False, ""
    if args.twin_gap0_max > 0 and gap0 is not None and gap0 > args.twin_gap0_max:
        return True, f"{gap0:.0f} cm > {args.twin_gap0_max:.0f}"
    return False, ""


def _replay_with_guard(node, mola, tag):
    """Twin replay with consensus guard: the estimate at the end of the replay must lie within
    guard_cm of the median of the valid replays of the same window. The first replay of a window
    is confirmed by a second one (a third decides if the two disagree); an out-of-consensus
    replay is repeated once with a new twin. The twin of the last replay stays running. About 4 %
    of MOLA restarts get the initial alignment wrong (1 in 24 and 2 in 50 in two gate runs):
    without the guard the candidate would be evaluated from a wrong state.
    """
    rec = _replay_to_twin(node, mola, tag)
    if rec is None or not TWIN_STATE.get("guard", False):
        return rec
    ref = TWIN_STATE.setdefault("ref", {})
    lst = ref.setdefault(TWIN_STATE.get("window"), [])
    guard_cm = float(TWIN_STATE.get("guard_cm", 5.0))

    def _est(r):
        e = r.get("twin_est_raw") if r else None
        return (float(e[0]), float(e[1])) if (e is not None and r.get("status") == "ok") else None

    def _dist(a, b):
        return 100 * math.hypot(a[0] - b[0], a[1] - b[1])

    def _median(pts):
        return (float(np.median([q[0] for q in pts])), float(np.median([q[1] for q in pts]))) if pts else None

    def _redo(suffix):
        mola.stop()
        node.est = None; node.est_t = 0.0; node.est_raw = None; node.est_origin = None
        node.est_stamp = float("nan"); node.est_stamp_raw = float("nan")
        mola.spawn(tag + suffix)
        return _replay_to_twin(node, mola, tag + suffix)

    attempts = 1
    p = _est(rec)
    if not lst:
        rec2 = _redo("c"); attempts += 1
        p2 = _est(rec2)
        if p is not None and p2 is not None and _dist(p, p2) <= guard_cm:
            lst.extend([p, p2])
            rec, p = rec2, p2
            dev = _dist(p, _median(lst))
            outlier = False
        else:
            rec3 = _redo("d"); attempts += 1
            p3 = _est(rec3)
            pts = [q for q in (p, p2, p3) if q is not None]
            med = _median(pts)
            for q in pts:
                if med is not None and _dist(q, med) <= guard_cm:
                    lst.append(q)
            rec, p = rec3, p3
            dev = _dist(p, med) if (p is not None and med is not None) else None
            outlier = dev is None or dev > guard_cm
    else:
        med = _median(lst)
        dev = _dist(p, med) if p is not None else None
        outlier = dev is None or dev > guard_cm
        if outlier:
            rec2 = _redo("b"); attempts += 1
            p = _est(rec2); rec = rec2
            dev = _dist(p, med) if p is not None else None
            outlier = dev is None or dev > guard_cm
        if not outlier:
            lst.append(p)
    rec["replay_attempts"] = attempts
    rec["replay_ref_n"] = len(lst)
    rec["replay_dev_cm"] = round(dev, 3) if dev is not None else None
    rec["replay_outlier"] = bool(outlier)
    if outlier:
        print(f"    rigioco fuori consenso ({rec['replay_dev_cm']} cm, {attempts} tentativi): candidato escluso dalla selezione")
    return rec


def _gate_replay_once(k, ls, i, isaac, mola, node, victim, saved, args):
    """One gate replay (--replay-gate): new twin, replay of the last ls seconds, comparison with the
    victim estimate; no guard (raw measurement).
    """
    isaac.pause(); isaac.restore(); mola.stop()
    fresh_arm = None
    if getattr(node, "fresh_cloud", False):
        count_restore(node, "gate", k)
        node.arm_fresh_cloud()
        fresh_arm = node.last_fresh_arm
    node.est = None; node.est_t = 0.0; node.est_raw = None; node.est_origin = None
    node.est_stamp = float("nan"); node.est_stamp_raw = float("nan")
    try:
        mola.spawn(f"w{k}_gate{ls:g}_{i}")
    except Exception as e:
        return {"last_s": ls, "i": i, "status": f"spawn_fail: {e}"}
    saved_mode = dict(TWIN_STATE)
    TWIN_STATE.update({"mode": "victim", "last_s": ls, "rate_hz": args.replay_rate_hz, "victim": victim, "guard": False})
    try:
        rec = _replay_to_twin(node, mola, f"w{k}_gate") or {"status": "no_history"}
    finally:
        TWIN_STATE.update(saved_mode)
    rec.update({"last_s": ls, "i": i})
    if fresh_arm is not None:
        rec["fresh_arm"] = fresh_arm
    if node.est_raw is not None and victim.run_origin is not None:
        tw = node.compose(victim.run_origin, node.est_raw)
        vw = node.compose(victim.run_origin, victim.last_est_end)
        rec["twin_gap_cm"] = round(100 * math.hypot(tw[0] - saved[0], tw[1] - saved[1]), 3)
        rec["victim_gap_cm"] = round(100 * math.hypot(vw[0] - saved[0], vw[1] - saved[1]), 3)
    mola.stop()
    return rec


def twin_drift_confirmed(on, cur, drift, tol=1e-5):
    """The node status confirms the requested starting drift (None = reset): option present and
    value equal within tol (m).
    """
    if not on:
        return False
    if drift is None:
        return cur is None
    return (cur is not None and len(cur) == 3
            and max(abs(float(a) - float(b)) for a, b in zip(cur, drift)) < tol)


# Cloud age when MOLA takes it (--controllo-eta). Age = time of the DEBUG line "Generator (raw)]
# Processing observation ... stamp=s" in the MOLA log minus t_dds_src of the per-frame node log row
# with that stamp; clouds published in the stretch before end - threshold and never taken count as a
# block. Twin stream: stamp_main_out (victim_stream 0); victim stream: stamp_victim (victim_stream
# 1). Same definition as scripts/analisi_prova_decisiva.py (RX_PRESA, prese_mola, righe_gemello,
# eta_alle_prese, mai_prese).
RX_PRESA = re.compile(r"\[(\d\d):(\d\d):(\d\d\.\d+)\|DEBUG\|Generator \(raw\)\] "
                      r"Processing observation .*?stamp=([\d.]+)")
# MOLA drop with a busy worker (LidarOdometry_SensorCallbacks.cpp:259-274): the queued cloud never
# reaches onLidar, so it stays never taken
RX_DROP = re.compile(r"\[(\d\d):(\d\d):(\d\d\.\d+)\|WARN \|LidarOdometry\] Dropping LiDAR scan")
# ROS log line written in the middle of a MOLA line (TF_OLD_DATA), or containing a MOLA line: take
# and drop are also searched mid-line (search), with the time of the MOLA header preceding the text
RX_ROS_LOG = re.compile(r"\[(?:DEBUG|INFO|WARN|ERROR|FATAL)\] \[\d+\.\d+\] \[")
RX_ORA_MOLA = re.compile(r"^\[\d\d:\d\d:\d\d\.\d+\|")


def ricuci_log(text):
    """MOLA lines with ROS lines inserted in the middle split off and the two pieces joined back
    (the rest of the MOLA line follows, after possible empty lines). The ROS line stays as a line
    of its own: it may contain a MOLA line.
    """
    out, pezzo = [], None
    for line in (text or "").splitlines():
        m = RX_ROS_LOG.search(line)
        if m is not None:
            if m.start() > 0:
                pezzo = (pezzo or "") + line[:m.start()]
            out.append(line[m.start():])
            continue
        if pezzo is not None:
            if not line.strip():
                continue
            if not RX_ORA_MOLA.match(line):
                line = pezzo + line
            else:
                out.append(pezzo)
            pezzo = None
        out.append(line)
    if pezzo is not None:
        out.append(pezzo)
    return out


def _ora_parete(m, mezzanotte, w_rif):
    t = mezzanotte + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return t + 86400 if t < w_rif - 43200 else t


def _mezzanotte(w_rif):
    import datetime
    d = datetime.datetime.fromtimestamp(w_rif)
    return datetime.datetime(d.year, d.month, d.day).timestamp()


def scarti_mola(text, w0, w1):
    """MOLA WARN "Dropping LiDAR scan" lines in the window [w0 - 0.5, w1 + 0.5] (the line is
    rate-limited to one per second: it counts the seconds with drops).
    """
    mz = _mezzanotte(w0)
    n = 0
    for line in ricuci_log(text):
        m = RX_DROP.search(line)
        if m and w0 - 0.5 <= _ora_parete(m, mz, w0) <= w1 + 0.5:
            n += 1
    return n


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def prese_mola(text, w_rif):
    """(wall time, stamp) of the MOLA takes. The log has only the local time: the day is that of
    w_rif (wall s); midnight wrap handled.
    """
    mezzanotte = _mezzanotte(w_rif)
    out = []
    for line in ricuci_log(text):
        for m in RX_PRESA.finditer(line):
            out.append((_ora_parete(m, mezzanotte, w_rif), float(m.group(4))))
    return out


def righe_flusso(frame_log, vittima=False):
    """(t_arr_wall, stamp, t_dds_src) of the twin stream (victim_stream 0, stamp_main_out) or of the
    victim stream (victim_stream 1, stamp_victim).
    """
    import csv
    flusso, col = (("1", "1.0"), "stamp_victim") if vittima else (("0", "0.0"), "stamp_main_out")
    out = []
    with open(frame_log) as f:
        for r in csv.DictReader(f):
            if r.get("victim_stream") not in flusso:
                continue
            t, s, src = _num(r.get("t_arr_wall")), _num(r.get(col)), _num(r.get("t_dds_src"))
            if not any(math.isnan(x) for x in (t, s, src)):
                out.append((t, s, src))
    return out


def eta_alle_prese(prese, righe, w0, w1):
    """Age (s) of every cloud when MOLA takes it in the window [w0, w1]."""
    sel = [(s, src) for t, s, src in righe if w0 - 5 <= t <= w1 + 1]
    out = []
    for tl, s in prese:
        if not (w0 - 0.5 <= tl <= w1 + 0.5):
            continue
        cand = [src for ss, src in sel if abs(ss - s) < 1e-3]
        if cand:
            out.append(tl - cand[-1])
    return out


def mai_prese(prese, righe, w0, w1, soglia):
    """Minimum age (w1 - t_dds_src) of the clouds published after the first take (or since w0
    without takes) and before w1 - threshold, never taken.
    """
    pw = [(tl, s) for tl, s in prese if w0 - 0.5 <= tl <= w1 + 0.5]
    t_da = min(tl for tl, _ in pw) if pw else w0
    stamp = [s for _, s in pw]
    return [w1 - src for t, s, src in righe
            if t_da <= t <= w1 - soglia and not any(abs(s - x) < 1e-3 for x in stamp)]


def misura_eta(text, righe, w0, w1, soglia):
    """Check outcome on a stretch: failure with maximum age over threshold or with clouds never
    taken; non_determinato without DEBUG lines in the log or without stream rows in the window
    (the rollout stays valid).
    """
    prese = prese_mola(text or "", w0)
    eta = eta_alle_prese(prese, righe, w0, w1)
    mai = mai_prese(prese, righe, w0, w1, soglia)
    n_righe = sum(1 for t, _, _ in righe if w0 <= t <= w1)
    m = {"w0": round(w0, 3), "w1": round(w1, 3), "soglia_s": soglia,
         "eta_max_s": round(max(eta), 4) if eta else None, "n_prese": len(eta),
         "n_mai_prese": len(mai), "eta_mai_prese_max_s": round(max(mai), 4) if mai else None,
         "n_righe_flusso": n_righe, "n_drop_mola": scarti_mola(text, w0, w1)}
    mot = []
    if eta and max(eta) > soglia:
        mot.append(f"eta' massima {max(eta):.3f} s oltre {soglia:.2f} s")
    if mai:
        mot.append(f"{len(mai)} nuvole mai prese")
    if mot:
        m.update(esito="guasto", motivo="; ".join(mot))
    elif not prese:
        m.update(esito="non_determinato", motivo="log di MOLA senza righe DEBUG di presa")
    elif not n_righe:
        m.update(esito="non_determinato", motivo="nessuna riga del flusso nel registro per frame")
    else:
        m.update(esito="ok", motivo=None)
    return m


def controlla_eta_rollout(runner, node, ruolo, tag, tentativo, res, guasti_prima=None):
    """Age check of one rollout attempt: wait for a per-frame log flush after the end of the
    stretch, measure, entry in node.last_eta_nuvola and in the file log. Without a stretch
    (rollout closed before driving) the outcome is non_misurato.
    """
    soglia = CONTROLLO_ETA["soglia"]
    win = getattr(node, "last_drive_wall", None)
    m = {"tag": tag, "ruolo": ruolo, "tentativo": tentativo,
         "status": res[3] if isinstance(res, tuple) and len(res) > 3 else None,
         "log": str(getattr(runner, "log_path", None))}
    if win is None:
        m.update(esito="non_misurato", motivo="nessun tratto guidato")
    else:
        w0, w1 = win
        attesa = w1 + FRAME_LOG_FLUSH_S + 0.5 - time.time()
        if 0 < attesa <= FRAME_LOG_FLUSH_S + 0.5:
            time.sleep(attesa)
        try:
            text = Path(runner.log_path).read_text(errors="replace")
            righe = righe_flusso(CONTROLLO_ETA["frame_log"], vittima=(ruolo == "vittima"))
            m.update(misura_eta(text, righe, w0, w1, soglia))
        except (OSError, TypeError, ValueError) as e:
            m.update(esito="non_determinato", motivo=f"lettura fallita: {e}")
    if isinstance(getattr(node, "last_eta_nuvola", None), list):
        node.last_eta_nuvola.append(m)
    # per-window counts (history: controllo_eta of the window)
    m["finestra"] = STALL_POLICY.get("window")
    c = CONTROLLO_ETA["finestre"].setdefault(str(m["finestra"]), {
        "tentativi": 0, "rifatti": 0, "ripetuti": 0, "guasti_vittima": 0, "non_determinati": 0})
    c["tentativi"] += 1
    c["non_determinati"] += m["esito"] in ("non_determinato", "non_misurato")
    if m["esito"] == "guasto":
        # guasti_prima: age failures the rollout already had (None = from the attempt number)
        primo = (tentativo == 1) if guasti_prima is None else (guasti_prima == 0)
        c["guasti_vittima" if ruolo == "vittima" else ("rifatti" if primo else "ripetuti")] += 1
    c["quota_rifatti"] = round(c["rifatti"] / c["tentativi"], 4)
    if CONTROLLO_ETA["registro"]:
        try:
            with open(CONTROLLO_ETA["registro"], "a") as f:
                f.write(json.dumps({"wall": round(time.time(), 3), **m}) + "\n")
        except OSError:
            pass
    return m


def frame_log_corrente(percorso):
    """Per-frame log actually written by the node: with a file already present the restarted node
    writes <name>_1, _2, ... (FrameLog.libero); the last one existing at orchestrator start is
    used (the node starts first).
    """
    p = Path(percorso)
    ult, i = p, 1
    while p.with_name(f"{p.stem}_{i}{p.suffix}").exists():
        ult = p.with_name(f"{p.stem}_{i}{p.suffix}")
        i += 1
    return str(ult)


def registri_nodo_correnti(args):
    """Per-frame log and crc log actually used by the node: at the next attempt after a stall the
    restarted node writes <name>_1, _2, ... in the same directory; the node starts before the
    orchestrator, so the last existing one is used. Age check, chain check and shadow comparison
    then read the already resolved args.
    """
    for nome in ("frame_log_nodo", "shadow_crc_log"):
        v = getattr(args, nome, None)
        if v:
            setattr(args, nome, frame_log_corrente(v))


def prepara_dds_profilo(percorso, cwd=None):
    """--dds-profile: absolute path and md5 of the profile. Error if the file does not exist or if
    DEFAULT_FASTRTPS_PROFILES.xml is in the working directory (Fast DDS would load the same
    profiles twice: "same id").
    """
    import hashlib
    pp = Path(percorso).expanduser().resolve()
    if not pp.is_file():
        raise ValueError(f"--dds-profile: {pp} non esiste")
    doppio = Path(cwd or os.getcwd()) / DDS_FILE_CARTELLA
    if doppio.exists():
        raise ValueError(f"--dds-profile: {doppio} presente nella cartella di lavoro (profili "
                         "caricati due volte): toglierlo o lanciare da un'altra cartella")
    return {"path": str(pp), "md5": hashlib.md5(pp.read_bytes()).hexdigest()}


def rollout(genome, isaac, mola, node, goal, horizon, tag, **kw):
    """Rollout with tolerance to transient Isaac stalls (STALL_POLICY): a twin rollout hit by a
    stall is repeated once after recovery; the victim stretch is not (invalid run).
    """
    victim = kw.get("victim")
    node.last_eta_nuvola = [] if CONTROLLO_ETA["soglia"] is not None else None
    # With --controllo-eta at most 3 attempts: stalls and age failures are counted separately
    # (second stall = exception, second age failure = guasto_eta_ripetuto). Without it: 2 attempts.
    n_tent = 3 if CONTROLLO_ETA["soglia"] is not None else 2
    n_eta = n_stalli = 0
    for attempt in range(n_tent):
        node.last_drive_wall = None
        node.last_stop_info = None
        node.last_twin_flags = None
        node.last_twin_fp = None
        node.last_twin_align = None
        node.last_mola_stop = None
        mola.last_stop = None
        if getattr(node, "fresh_cloud", False):
            node.rollout_retry = attempt > 0      # restore counter: repeated rollout
        try:
            res = _rollout_once(genome, isaac, mola, node, goal, horizon,
                                tag + "r" * attempt, **kw)
            # twin MOLA shutdown in this rollout (None for the victim)
            node.last_mola_stop = getattr(mola, "last_stop", None)
            if CONTROLLO_ETA["soglia"] is not None:
                # before any other use of the rollout: on failure its values reach neither fitness,
                # archive, selection nor re-evaluations
                m = controlla_eta_rollout(victim if victim is not None else mola, node,
                                          "vittima" if victim is not None else "gemello",
                                          tag + "r" * attempt, attempt + 1, res, guasti_prima=n_eta)
                if m["esito"] == "guasto":
                    if victim is not None:
                        raise GuastoEtaNuvola(f"guasto_eta_nuvola tratto della vittima {tag}: {m['motivo']}")
                    n_eta += 1
                    if n_eta == 2:
                        print(f"  GUASTO ETA' NUVOLA RIPETUTO (rollout {tag}): {m['motivo']}; "
                              f"candidato escluso ({ESITO_ETA_RIPETUTO})")
                        return None, None, float("inf"), ESITO_ETA_RIPETUTO, 0.0, 0
                    print(f"  GUASTO ETA' NUVOLA (rollout {tag}): {m['motivo']}; rollout rifatto")
                    continue
            if victim is None:
                _note_twin_log(node, mola)
                _note_twin_fp(node, mola)
            if ZONE["on"] and genome is not None:
                node.zone_controllo("vittima" if victim is not None else "gemello", tag,
                                    res[5] if isinstance(res, tuple) and len(res) > 5 else None)
            if (GIM["S"] > 0 or GIM["cpf"]) and genome is not None:
                # --genoma-in-moto, --chamfer-per-fase: step and per-phase Chamfer of the stream at
                # rollout end
                gim_registra(node, "vittima" if victim is not None else "gemello",
                             "vittima" if victim is not None else tag)
            return res
        except IsaacStall as e:
            n_stalli += 1
            if victim is not None or n_stalli == 2 or attempt == n_tent - 1:
                raise
            _isaac_recovery(isaac, node, e, f"rollout {tag}")


def _rollout_once(genome, isaac, mola, node, goal, horizon, tag, warmup=0.0,
            trace_path=None, track_min=0.60, track_max=None, track_ref=None,
            settle_sec=5.0, settle_fixed=None, min_travel_frac=MIN_TRAVEL_FRAC, catchup_sec=0.8,
            stop_on="true",
            attack_from="spawn", untracked="discard", nominal_ok=True,
            drift_reset_on="genome", victim=None, frame_start=None,
            assestamento_nuvole=None, dopo_assestamento=None):
    """Evaluate a candidate: restore, restart MOLA, drive H metres, measure.

    With warmup > 0 the robot first drives that stretch without perturbation and the measurement
    starts afterwards: MOLA restarts without a local map and with a velocity model to re-converge,
    and convergence comes with motion, not with time.

    min_travel_frac is the minimum fraction of the stretch to drive for the deviation to be
    measurable: below it the outcome is `stopped_early` (the attack paralysed the robot). It must be
    excluded from the front, otherwise the degenerate "robot still" solution gets the maximum
    deviation from the moving nominal.

    With victim (VictimRunner) the rollout is driven by the continuous victim: no MOLA restart, the
    robot goes back to the saved state (where the victim left it), the cloud stream to the victim
    reopens with the given genome and the controller reads its estimate. The twin is stopped.

    Returns (start_pose, true_end_pose, mean_chamfer, outcome, metres_travelled, perturbed_frames).

    Realignment (None = off): assestamento_nuvole sets the twin still clouds (0 = drive from the
    first pose, realigned twin); dopo_assestamento is called at the end of settling, before driving,
    with (isaac, node, info).
    """
    if getattr(node, "registri", None) is not None:
        node.registri.inizio_rollout(tag)
    if ASSETTO["registra"]:
        node.last_assetto = []
    isaac.pause()
    isaac.restore()
    leggi_assetto(isaac, "dopo_restore", node)
    mola.stop()
    if getattr(node, "fresh_cloud", False):
        count_restore(node, restore_kind(tag, victim is not None, getattr(node, "rollout_retry", False)),
                      restore_window(tag))
        node.arm_fresh_cloud()
        if victim is not None:
            node.last_victim_fresh_arm = node.last_fresh_arm
    if victim is not None:
        return _victim_rollout(genome, isaac, victim, node, goal, horizon, tag,
                               trace_path, track_min, track_max, track_ref,
                               settle_fixed, min_travel_frac, catchup_sec,
                               stop_on, attack_from, drift_reset_on, frame_start,
                               dopo_assestamento=dopo_assestamento)
    # --twin-init attacker_est: the physical twin starts from the pose estimated by the attacker
    # MOLA (target computed at window start).
    tgt = TWIN_STATE.get("init_target") if TWIN_STATE.get("init") in ("attacker_est", "ombra_est") else None
    if tgt is not None:
        x, y, yaw = tgt["target_scene"]
        # --twin-init-assetto: roll and pitch (or keep_tilt) only if given
        isaac.set_state(x, y, yaw, tgt["v_set_scene"][0], tgt["v_set_scene"][1], tgt["w_set"],
                        zero_joints=tgt["zero_joints"], **(tgt.get("assetto_set_state") or {}))
        leggi_assetto(isaac, "dopo_set_state", node)

    # Variant E: new MOLA with map, pose and sigma of the shadow saved at window start. The .mm is
    # checked before the start: if missing MOLA stops on an ASSERT (Initialize.cpp:560-565).
    e_state = TWIN_STATE.get("shadow_state") if TWIN_STATE.get("mode") == "ombra_e" else None
    if TWIN_STATE.get("mode") == "ombra_e":
        e_why = (tv.mm_problem((e_state or {}).get("mm_path")) if (e_state or {}).get("status") == "ok"
                 else f"stato dell'ombra non salvato ({(e_state or {}).get('status')})")
        if e_why:
            return None, None, float("inf"), f"ombra_e_state_missing: {e_why}", 0.0, 0
        mola.extra_env = {**TWIN_STATE.get("twin_env_base", {}), **e_state["env"]}

    # Variant G: new overlay MOLA that loads the full state saved by the shadow at window start and
    # saves it again at once (hash chain). The twin clock is armed with the simulation stopped,
    # before the spawn: the first scan has the saved stamp + 0.1 s.
    g_state = TWIN_STATE.get("shadow_state") if TWIN_STATE.get("mode") == "ombra_g" else None
    g_resave = None
    if TWIN_STATE.get("mode") == "ombra_g":
        g_st = g_state or {}
        g_why = (tv.g_state_problem(g_st.get("state_path")) if g_st.get("status") == "ok"
                 else f"stato dell'ombra non salvato ({g_st.get('status')})")
        if g_why:
            return None, None, float("inf"), f"ombra_g_state_missing: {g_why}", 0.0, 0
        if TWIN_STATE.get("g_resave", "ogni") == "ogni":
            g_resave = f"{g_st['state_path']}.risalvato_{tag}"
        # g_env_extra: only with --g-banco (variables allowed by banco_g), otherwise absent
        mola.extra_env = {**TWIN_STATE.get("twin_env_base", {}),
                          **tv.g_twin_env(g_st["state_path"], g_resave, root=TWIN_STATE.get("g_root")),
                          **TWIN_STATE.get("g_env_extra", {})}
        if not node.arm_twin_clock(g_st["state_stamp_tick"]):
            return None, None, float("inf"), "ombra_g_clock_not_armed", 0.0, 0
    shadow_st = e_state if e_state is not None else g_state

    # Start without waiting: the initial relocalisation needs a cloud, which does not arrive with
    # the simulation stopped. Spawn first, then play, then wait.
    t_spawn = time.time()
    twin_topic = (getattr(mola, "args", None) or ["/carter/lidar_perturbed"])[0]
    # always with G: the first scan (saved stamp + 0.1 s) must find MOLA subscribed
    gids_before = topic_sub_gids(node, twin_topic) if (settle_target(
        bool(getattr(TWIN_STATE.get("victim"), "started", False))) > 0 or g_state is not None) else None
    # with G and --sync-wait publisher: base of the readers matched to the main node publisher, read
    # before the spawn
    g_pub_need = g_publisher_need(g_state is not None)
    g_pub_base = None
    if g_pub_need:
        try:
            g_pub_base = matched_base(node, g_pub_need)
        except OmbraPersa:
            raise
        except RuntimeError as e:
            return None, None, float("inf"), f"mola_not_ready: gemello G, {e}", 0.0, 0
    try:
        mola.spawn(tag)
    except Exception as e:
        return None, None, float("inf"), f"mola_spawn_fail: {e}", 0.0, 0

    # The genome is loaded at once, the attack is turned on only at the start of the measured
    # stretch: during settling (and warm-up) MOLA sees clean clouds, the temporal drift does not
    # accumulate with the robot still and the mean Chamfer covers only the motion. With
    # attack_from="spawn" it is on from the start.
    if genome is not None:
        node.set_genome(genome)
    node.set_attack_enabled(attack_from == "spawn" and warmup <= 0.0
                            and genome is not None)
    node.chamfer = []; node.perc_samples = []; node.proc_ms_roll = []

    # Victim state: the recorded clouds reach the twin with the simulation stopped (the first
    # initialises it); the twin estimate at the end of the replay is the stretch origin, as for the
    # victim.
    node.est_origin = None
    node.est = None
    node.est_t = 0.0
    try:
        if TWIN_STATE.get("mode") == "maprestore":
            # Map restore: load+relocalize with the simulation stopped; estimate (and fidelity) are
            # confirmed after isaac.play(), below.
            node.last_replay = _maprestore_to_twin(node, mola, tag)
        else:
            node.last_replay = _replay_with_guard(node, mola, tag)
    except IsaacStall:
        raise
    except Exception as e:
        return None, None, float("inf"), f"mola_spawn_fail: {e}", 0.0, 0
    if node.last_replay is not None and node.last_replay.get("status") == "ok":
        node.est_origin = node.est_raw
    if shadow_st is not None:
        # twin estimate relative to the loaded pose (shadow frame), as the clean twin one is
        # relative to (0, 0, 0)
        p6 = shadow_st["pose6"]
        node.est_origin = (p6[0], p6[1], p6[3])
    rep = node.last_replay
    # Replay mode: control frame from the twin estimate at the end of the replay or from the
    # attacker MOLA one, never from the victim pose.
    if TWIN_STATE.get("mode") == "replay":
        fr, src, why = tv.replay_frame("replay", TWIN_STATE.get("replay_frame", "twin"), rep,
                                       TWIN_STATE.get("attacker_origin"), frame_start)
        if rep is not None:
            rep.update({"frame_source": src, "frame_fallback_why": why,
                        "frame": list(map(float, fr)) if fr is not None else None})
        if fr is not None:
            frame_start = fr
    # Initial state fingerprint: drift of the two generators before play (after loading the genome),
    # replay ends; estimated pose below, sigma and keyframes from the log once the rollout is closed
    # (_note_twin_fp).
    fp = {"drift_twin_xy_cm": getattr(node, "st_drift_xy_cm", None),
          "drift_victim_xy_cm": getattr(node, "st_victim_drift_xy_cm", None),
          "drift_keep": getattr(node, "st_drift_keep", None),
          "replay_n": (rep or {}).get("n"),
          "replay_first_stamp": (rep or {}).get("first_stamp"),
          "replay_last_stamp": (rep or {}).get("last_stamp"),
          "source": None, "est_raw": None, "est_stamp": None}
    if TWIN_STATE.get("mode") in ("victim", "replay") and rep is not None \
            and rep.get("status") == "ok" and rep.get("twin_est_raw") is not None:
        fp.update(source="fine_rigioco", est_raw=rep["twin_est_raw"], est_stamp=rep.get("twin_est_stamp"))
    if e_state is not None:
        fp.update(tv.e_fingerprint(e_state))
    if g_state is not None:
        fp.update(tv.g_fingerprint(g_state))
        fp["g_resave_path"] = g_resave
    node.last_twin_fp = fp

    # --settle-clouds: clouds delivered to the twin since play (main output)
    target = settle_target(bool(getattr(TWIN_STATE.get("victim"), "started", False)))
    if assestamento_nuvole is not None:
        target = int(assestamento_nuvole)
    c_count = lambda: getattr(node, "frames_out", None)
    sub_wait = None
    if (target > 0 or g_state is not None) and gids_before is not None:
        # as for the victim: the new MOLA subscribes before play
        sub_wait = wait_new_subscriber(node, twin_topic, gids_before, alive=getattr(mola, "alive", None))
    if g_pub_need:
        # node-side match of the main publisher (plus margin), since the spawn
        try:
            pub = sync_wait_publisher(node, g_pub_need, g_pub_base, (mola,), t_spawn)
        except OmbraPersa:
            raise
        except RuntimeError as e:
            mola.stop()
            return (None, None, float("inf"),
                    f"mola_not_ready: gemello G su {twin_topic}, {e}", 0.0, 0)
        sub_wait = {**(sub_wait or {"topic": twin_topic}), "sync_wait": pub}
    c0 = c_count() or 0
    t_play = time.time()
    node.last_settle_clouds = None
    isaac.play()
    try:
        # drain, not just the check: the wait does not spin the executor and without callbacks cloud
        # arrivals would not be recorded.
        if g_state is not None:
            # the loaded twin does not go through the initial relocalisation
            mola.wait_ready(check=node.drain, marker=tv.G_READY_MARKER, fail_markers=tv.G_ERROR_MARKERS)
        else:
            mola.wait_ready(check=node.drain)
    except IsaacStall:
        raise
    except Exception as e:
        isaac.pause()
        mola.stop()
        return None, None, float("inf"), f"mola_not_ready: {e}", 0.0, 0

    # The evidence that the new MOLA publishes is a pose received after the spawn, not the publisher
    # count: the DDS entry of the previous MOLA stays in the graph for some seconds, and waiting for
    # its removal with the robot still makes the adaptive sigma decay to the floor. The count stays
    # as a diagnostic.
    node.est = None
    node.est_t = 0.0
    deadline = time.time() + 15.0
    while time.time() < deadline and node.est_t < t_spawn:
        node.spin(0.1)
    if node.est_t < t_spawn:
        n_pub = node.count_mola_publishers()
        n_proc = MolaRunner._count_processes()
        isaac.pause()
        mola.stop()
        return (None, None, float("inf"),
                f"no_pose_from_new_mola (publisher nel grafo: {n_pub}, "
                f"processi vivi: {n_proc})", 0.0, 0)
    n_pub = node.count_mola_publishers()
    if n_pub != 1:
        print(f"    [nota] {n_pub} publisher su /lidar_odometry/pose nel grafo "
              f"({MolaRunner._count_processes()} processi mola-cli): voce "
              f"stantia, si prosegue perche' la posa arriva")

    # Map restore: the estimate from the first scan after play (on the loaded map + relocalisation)
    # is the stretch origin, like the end of the replay for --twin-state victim. Fidelity = offset
    # between this estimate and the pose believed by the victim.
    if TWIN_STATE.get("mode") == "maprestore" and node.last_replay is not None \
            and node.last_replay.get("map_load_ok"):
        rec = node.last_replay
        vend = rec.get("victim_est_end")
        if vend is not None:
            # Only NOW (MOLA already has the initial estimate on the first scan after play) the
            # victim pose is published on /initialpose and the twin is left to converge: before play
            # the relocalisation would be overwritten by the initial MOLA estimate at (0,0,0).
            rec["relocalize_published"] = node.twin_relocalize(vend)
            node.spin(2.5)
        if node.est_raw is not None:
            rec["twin_est_raw"] = list(map(float, node.est_raw))
            if vend is not None:
                rec["diff_cm"] = round(100 * math.hypot(node.est_raw[0] - vend[0],
                                                        node.est_raw[1] - vend[1]), 3)
                rec["diff_deg"] = round(math.degrees(angle_diff(node.est_raw[2], vend[2])), 3)
            rec["twin_quality"] = node.pose_quality if node.pose_quality == node.pose_quality else None
            rec["status"] = "ok"
            node.est_origin = node.est_raw
            fp.update(source="dopo_rilocalizzazione", est_raw=list(map(float, node.est_raw)),
                      est_stamp=node.est_stamp_raw if node.est_stamp_raw == node.est_stamp_raw else None)
    if fp["source"] is None and node.est_raw is not None:
        # Clean twin (or failed replay): first pose after the start.
        fp.update(source="prima_posa_dopo_avvio", est_raw=list(map(float, node.est_raw)),
                  est_stamp=node.est_stamp_raw if node.est_stamp_raw == node.est_stamp_raw else None)
    elif fp["source"] in tv.SHADOW_MODES and node.est_raw is not None:
        # first pose of twin E or G after the start, in the shadow frame
        fp.update(est_raw=list(map(float, node.est_raw)),
                  est_stamp=node.est_stamp_raw if node.est_stamp_raw == node.est_stamp_raw else None)

    track_ratio = float("nan")

    # Settling. With the attack on from the start the attacked estimate never converges and would
    # always hit the cap, paying more warm-up and more temporal drift than the nominal: the
    # comparison would not be under equal conditions. So convergence is measured only on the window
    # nominal (settle_fixed None) and that same fixed duration is applied to candidates,
    # re-evaluation and applied genome (settle_fixed = that time). With --settle-clouds a fixed
    # number of clouds for all (nominal included).
    if target > 0:
        first_pose_clouds = int((c_count() or 0) - c0)
        node.last_settle_clouds = wait_clouds(node, c_count, c0, target, t_play)
        node.last_settle_clouds["first_pose_clouds"] = first_pose_clouds
        node.last_settle_clouds["subscription"] = sub_wait
        node.last_settle_waited = node.last_settle_clouds["wall_s"]
        node.last_settle_ok = node.last_settle_clouds["ok"]
    elif assestamento_nuvole is not None:
        # realigned twin: driving starts from the first pose
        node.last_settle_waited, node.last_settle_ok = 0.0, True
    elif settle_fixed is not None:
        if settle_fixed > 0:
            node.spin(settle_fixed)
        node.last_settle_waited = settle_fixed
        node.last_settle_ok = True
    elif settle_sec > 0:
        node.last_settle_waited, node.last_settle_ok = node.wait_converged(isaac, settle_sec)
    else:
        node.last_settle_waited, node.last_settle_ok = 0.0, True

    # Start pose in the working frame, read with the simulation running and the robot still: after
    # the restore the topic publishes only at the first tick.
    true_start = node.settled_pose(isaac)
    close_settle_record(node.last_settle_clouds, c_count, c0, t_play)
    leggi_assetto(isaac, "fine_assestamento", node)
    if dopo_assestamento is not None:
        # bench realignment: pause, twin state and named snapshot, resume
        dopo_assestamento(isaac, node, {"tag": tag, "ruolo": "gemello", "mola": mola,
                                        "true_start": true_start})

    # Twin control frame: the true pose (realignment to the truth) or the pose estimated by the
    # attacker MOLA (--twin-align attacker): the local waypoints start from there. The measurement
    # stays on the true pose.
    if frame_start is None:
        frame_start = true_start
    node.last_align_err_cm = 100 * math.hypot(frame_start[0] - true_start[0], frame_start[1] - true_start[1])
    node.last_align_err_deg = math.degrees(angle_diff(frame_start[2], true_start[2]))
    node.last_resume_jump_cm = None
    node.last_twin_align = _twin_alignment_record(node, true_start, fp)
    goals = goal if isinstance(goal, list) else [goal]
    local_wps = [to_local(g, frame_start) for g in goals]

    if warmup > 0.0:
        st_w, _ = node.drive(local_wps, warmup, isaac, mola, timeout=30.0)
        if st_w not in ("ok", "arrived"):
            isaac.pause()
            mola.stop()
            return None, None, float("inf"), f"warmup_{st_w}", 0.0, 0
        # From here on it is measured: new stretch origin and attack active.
        true_start = node.settled_pose(isaac)
        local_wps = [to_local(g, true_start) for g in goals]
        node.set_attack_enabled(genome is not None)
        node.chamfer = []; node.perc_samples = []; node.proc_ms_roll = []

    if attack_from == "motion" and warmup <= 0.0 and genome is not None:
        node.set_attack_enabled(True)
        node.chamfer = []; node.perc_samples = []; node.proc_ms_roll = []

    # With drift_reset_on="drive" the temporal drift starts with motion, not at genome load: it
    # separates deception in motion from poisoning of the initialisation during settling. Also on
    # the nominal, so timings stay equal.
    if drift_reset_on == "drive":
        node.reset_drift()

    def finish_twin():
        mola.stop()
        node.est_origin = None

    return _drive_and_measure(node, isaac, mola, local_wps, horizon, true_start, trace_path,
                              track_min, track_max, track_ref, min_travel_frac,
                              catchup_sec, stop_on, finish=finish_twin, frame_pose=frame_start)


def _drive_and_measure(node, isaac, mola, local_wps, horizon, true_start, trace_path,
                       track_min, track_max, track_ref, min_travel_frac,
                       catchup_sec, stop_on, finish=None, frame_pose=None):
    """Common part of a rollout: driving, stop, tracking, localisation error, drift, motion class,
    trace to file. `finish` is called at the end (twin stop, or victim stream closing).
    """
    frames_before = node.frames_out
    track_ratio = float("nan")
    # The trace is always needed: the deviation along the stretch is computed on it. Written to file
    # only on request.
    node.trace_rows = []
    est_before = node.est
    stamp_before = node.est_stamp_raw
    # Catch-up happens inside drive(): after the stop ticks and estimate are logged while the robot
    # brakes and MOLA catches up, until convergence.
    t_guida0 = time.time()
    status, travelled = node.drive(local_wps, horizon, isaac, mola, timeout=DRIVE_TIMEOUT[0],
                                   catchup_sec=catchup_sec, stop_on=stop_on,
                                   frame_pose=frame_pose)
    est_after = node.est
    isaac.pause()
    # wall-clock window of the stretch (--controllo-eta)
    node.last_drive_wall = (t_guida0, time.time())
    node.spin(0.05)
    node.last_drive_stamps = (stamp_before, node.est_stamp_raw)
    true_end_now = node.true_pose(isaac)
    # Final estimate and truth at the same instant (last coast tick).
    true_disp = math.hypot(true_end_now[0] - true_start[0],
                           true_end_now[1] - true_start[1])

    # Tracking check with an asymmetric threshold: an estimate/truth ratio far below 1 is MOLA
    # losing tracking by itself, while a successful attack pushes the ratio above 1. An upper bound
    # would discard exactly the most effective candidates.
    if (status in ("ok", "arrived") and est_before is not None
            and est_after is not None and true_disp > 0.10):
        est_d = math.hypot(est_after[0] - est_before[0],
                           est_after[1] - est_before[1])
        ratio = est_d / max(true_disp, 1e-6)
        # Absolute threshold: with estimate and truth read at the same final instant (coast logged)
        # and stamp-aligned, the nominal ratio is ~1, so 0.60 isolates a true tracking loss. The
        # nominal (track_ref None) is never labelled untracked.
        track_ratio = ratio
        node.last_track_thr = track_min
        if track_ref is not None and (ratio < track_min
                                      or (track_max is not None and ratio > track_max)):
            status = f"untracked({ratio:.2f})"
    else:
        node.last_track_thr = track_min
    # Exposed for valid rollouts too, to report next to the deviation.
    node.last_track_ratio = track_ratio

    # Stretch too short to measure a course deviation: the attack paralysed the robot (typically
    # `arrived` at zero metres). In both stop modes: with the stop on the estimate a paralysis with
    # a moving estimate closed as `arrived` (0.00 true m, estimate 0.08 m) and disappeared from the
    # counts while staying in the damage.
    # In the last window of a path the remaining stretch can be shorter than the horizon: an
    # 'arrived' after half of the remaining distance is a legitimate arrival, not an early stop.
    residuo = float(math.hypot(local_wps[-1][0], local_wps[-1][1])) if local_wps else horizon
    soglia_breve = min(min_travel_frac * horizon, 0.5 * residuo) if status == "arrived" else min_travel_frac * horizon
    if status in ("ok", "arrived") and travelled < soglia_breve:
        status = f"stopped_early({travelled:.2f}m)"

    true_end = true_end_now
    # The last trace row is already the last coast tick (estimate and true pose at the same instant,
    # after the robot stopped and MOLA caught up): no artificial row.
    node.last_trace = node.trace_rows or []
    node.last_tau = _tau_from_trace(node.last_trace)
    # Localisation error: estimate-truth gap along the stretch (m). Aligned (estimate against true
    # pose at the scan stamp): the actual error. Not aligned (against the current true pose):
    # includes the latency.
    def _finite(key):
        return [r[key] for r in node.last_trace
                if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
    ga = _finite("est_true_gap_al")
    gr = _finite("est_true_gap")
    node.last_loc_err_mean = float(np.mean(ga)) if ga else float("nan")
    node.last_loc_err_end = ga[-1] if ga else float("nan")
    node.last_loc_err_mean_raw = float(np.mean(gr)) if gr else float("nan")
    node.last_loc_err_end_raw = gr[-1] if gr else float("nan")
    # Temporal drift at the first and last driving tick, and motion class.
    dr = _finite("drift_xy_cm")
    node.last_drift_start_cm = dr[0] if dr else float("nan")
    node.last_drift_end_cm = dr[-1] if dr else float("nan")
    node.last_motion = node.classify_motion(node.last_trace) if node.last_trace else {}
    node.last_quality_mean = _quality_mean(node.last_trace)
    node.last_quality_min = _quality_min(node.last_trace)
    # Geometric perceptibility metrics, mean over the rollout samples.
    if node.perc_samples:
        keys = sorted({k for p in node.perc_samples for k in p})
        node.last_perc = {k: float(np.mean([p[k] for p in node.perc_samples if k in p])) for k in keys}
    else:
        node.last_perc = {}
    if trace_path is not None and node.trace_rows:
        import csv
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(node.trace_rows[0].keys()))
            wr.writeheader()
            wr.writerows(node.trace_rows)
        if ASSETTO["registra"]:
            # --registra-assetto: rollout readings next to the trace
            tp_ = Path(trace_path)
            with open(tp_.with_name(tp_.stem + "_assetto.json"), "w") as f:
                json.dump(getattr(node, "last_assetto", None), f, indent=1, default=str)
    node.trace_rows = None
    ch = float(np.mean(node.chamfer)) if node.chamfer else float("nan")
    frames = max(0, node.frames_out - frames_before)
    if finish is not None:
        finish()
    return true_start, true_end, ch, status, travelled, frames


# ---------------------------------------------------------------------------

def build_nsga3(pop_size: int, init_X=None):
    """NSGA-III with Das-Dennis, SBX and polynomial mutation.

    Only the fitness changes: physical rollout instead of offline replay. With two objectives the
    reference directions are n_partitions + 1: pop_size - 1 partitions are used, so every individual
    has its own direction. Mutation uses the pymoo default per-individual probability (0.9) with
    per-gene probability 1/n_var: with prob=0.2 about one gene every five children mutated, too
    little to explore in 8 generations.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("das-dennis", 2,
                                        n_partitions=max(1, pop_size - 1))
    # Warm start: if init_X is given (half from the previous window front, half random), generation
    # 1 starts from there instead of fully random.
    sampling = init_X if init_X is not None else FloatRandomSampling()
    return NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.9, eta=20),
        eliminate_duplicates=True,
    )


# Genome groups by information required from the attacker. Indices of the active genes; the others
# stay at -1 (operator off).
GENOME_GROUPS = {
    1: [0, 1, 2, 3, 5, 11],                       # per-point noise, dropout
    2: [0, 1, 2, 3, 5, 11, 4, 6, 7, 8, 9, 10, 13, 15, 16],   # + scan geometry
    3: list(range(17)),                            # + distortion and drift (global)
}


def expand_genome(g_active, active, n_genes=17):
    """Full vector from the one restricted to the group: inactive genes at -1."""
    full = -np.ones(n_genes)
    full[active] = np.asarray(g_active, dtype=np.float64)
    return full


def build_problem(n_genes: int, eval_fn, has_constraint: bool = False):
    """Same structure as MOLAPerturbationProblem in run_nsga3.py.

    With has_constraint the perceptibility budget is a constraint (G = Chamfer - EPS <= 0): NSGA-III
    constrained dominance ranks feasible points by the two objectives and infeasible ones by
    constraint violation, so the population converges towards the admissible region even when no
    candidate is admissible. eval_fn returns (f1, f2, g).
    """
    from pymoo.core.problem import Problem

    class WindowProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_genes, n_obj=2,
                            n_ieq_constr=1 if has_constraint else 0,
                            xl=-1.0 * np.ones(n_genes),
                            xu=1.0 * np.ones(n_genes))

        def _evaluate(self, X, out, *a, **kw):
            res = [eval_fn(g) for g in X]
            out["F"] = np.array([(r[0], r[1]) for r in res])
            if has_constraint:
                out["G"] = np.array([[r[2]] for r in res])

    return WindowProblem()


def stealth_check(node, status, travelled, horizon, min_quality=STEALTH_MIN_QUALITY):
    """Admissibility of a candidate. Returns (admissible, category, detail, is_dos).

    DoS (the robot does not move): pose missing for the whole stretch, or net displacement below
    DOS_TRAVEL_M (rotation in place). Infeasible. Detectable (not stealthy but moving): mean ICP
    quality below the threshold MOLA acts on. Infeasible but not DoS. Everything else (moved and
    ended up in the wrong place, whatever the estimate/true ratio) is admissible: it is the
    diversion we are looking for.
    """
    base = str(status).split("(")[0]
    if base in ("no_pose", "pose_stale", "victim_no_pose", "mola_crash"):
        return False, "posa assente", base, True
    if travelled < DOS_TRAVEL_M:
        return False, "robot fermo", f"spostamento {travelled:.2f}<{DOS_TRAVEL_M:.2f} m", True
    q = node.last_quality_mean
    if q is not None and q < min_quality:
        return False, "rilevabile (qualita' ICP bassa)", f"qualita' {q:.2f}<{min_quality:.2f}", False
    return True, "", "", False


def stealth_robot_fermo_valido(node, min_quality=STEALTH_MIN_QUALITY):
    """--incompleti-regola: a (c) excluded by stealth_check as "robot fermo" becomes stealthy again
    only if it also passes the next stealth_check test (mean ICP quality below min_quality =
    detectable, excluded), which stealth_check does not reach for a still robot. Same output as
    stealth_check.
    """
    q = node.last_quality_mean
    if q is not None and q < min_quality:
        return False, "rilevabile (qualita' ICP bassa)", f"qualita' {q:.2f}<{min_quality:.2f}", False
    return True, "", "", False


def _tau_from_trace(trace):
    """Estimate delay (s): the tau that best overlaps the estimated distance with the true distance
    travelled, on moving ticks (same estimate used in the analyses).
    """
    rows = [r for r in (trace or []) if isinstance(r.get("true_x"), (int, float)) and r["true_x"] == r["true_x"]]
    if len(rows) < 12:
        return float("nan")
    t = np.array([r["t"] for r in rows]); v = np.array([r["cmd_v"] for r in rows])
    e = np.hypot(np.array([r["est_x"] for r in rows]) - rows[0]["est_x"], np.array([r["est_y"] for r in rows]) - rows[0]["est_y"])
    u = np.hypot(np.array([r["true_x"] for r in rows]) - rows[0]["true_x"], np.array([r["true_y"] for r in rows]) - rows[0]["true_y"])
    mv = np.where(v > 0)[0]
    if len(mv) < 6:
        return float("nan")
    best = (float("inf"), 0.0)
    for tau in np.arange(0.0, 2.0, 0.025):
        ui = np.interp(t[mv] - tau, t, u, left=0.0)
        err = float(np.mean((e[mv] - ui) ** 2))
        if err < best[0]:
            best = (err, float(tau))
    return best[1]


def _gap0_from_trace(trace):
    """Aligned estimate-truth offset at the first trace row (cm; None without samples).

    A twin that starts already far from the attacker pose did not lock onto the map at restart: its
    damage does not come from the attack.
    """
    for r in (trace or []):
        v = r.get("est_true_gap_al")
        if isinstance(v, (int, float)) and v == v:
            return float(v) * 100.0
    return None


def _quality_min(trace):
    """Minimum ICP quality along a trace (None without samples), start-up frames (q=0) excluded."""
    q = [r["pose_quality"] for r in (trace or [])
         if isinstance(r.get("pose_quality"), (int, float)) and r["pose_quality"] == r["pose_quality"] and r["pose_quality"] > 0]
    return round(float(min(q)), 4) if q else None


def _quality_mean(trace):
    """Mean pose_quality along a trace (None without samples)."""
    q = [r["pose_quality"] for r in (trace or [])
         if isinstance(r.get("pose_quality"), (int, float)) and r["pose_quality"] == r["pose_quality"]]
    return round(float(np.mean(q)), 3) if q else None


def _mola_csv_stats(path, t0, t1):
    """Per-scan statistics from the MOLA debug CSV (`traces_<tag>.csv`) between stamps t0 and t1
    (twin clock): mean and max ICP iterations, mean quality, time per scan, rows and coverage
    (rows over expected scans at 10 Hz). MOLA writes the file in 8 KiB blocks and the last block
    is lost on close: the coverage says how much is missing.
    """
    import csv
    out = {"it_mean": None, "it_max": None, "q_mean": None, "t_on_ms": None, "n": 0, "coverage": None}
    if path is None:
        return out
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception:
        return out
    sel = []
    for r in rows:
        try:
            ts = float(r["timestamp"]); it = float(r["icp_iterations"]); q = float(r["icp_quality"])
            ton = float(r.get("time_onLidar") or "nan")
        except (KeyError, ValueError, TypeError):
            continue
        if t0 == t0 and t1 == t1 and (ts < t0 - 0.05 or ts > t1 + 0.05):
            continue
        if it == 0 and q == 0:   # initial row
            continue
        sel.append((ts, it, q, ton))
    if not sel:
        return out
    its = [s[1] for s in sel]; qs = [s[2] for s in sel]; tons = [s[3] for s in sel if s[3] == s[3]]
    out.update(it_mean=round(float(np.mean(its)), 3), it_max=float(max(its)), q_mean=round(float(np.mean(qs)), 4),
               t_on_ms=round(1000 * float(np.mean(tons)), 2) if tons else None, n=len(sel))
    if t0 == t0 and t1 == t1 and t1 > t0:
        out["coverage"] = round(len(sel) / max(1.0, (t1 - t0) * 10.0), 2)
    return out


def _f2_internal(q_att, q_nom, it_att, it_nom, q_scale, it_scale):
    """Internal perceptibility: ICP quality drop and excess iterations with respect to the window
    nominal, each normalised on the p95 of its own null distribution. A term that cannot be
    computed is 0 and is returned as None.
    """
    tq = max(0.0, q_nom - q_att) / q_scale if (q_att is not None and q_nom is not None) else None
    ti = max(0.0, it_att - it_nom) / it_scale if (it_att is not None and it_nom is not None) else None
    return round((tq or 0.0) + (ti or 0.0), 4), (round(tq, 4) if tq is not None else None), (round(ti, 4) if ti is not None else None)


def _twin_flag_fields(node, mode):
    """First twin ICP flags (always recorded) and exclusion from selection with --twin-filter
    (twin_filtered, False with 'none').
    """
    flags = dict(getattr(node, "last_twin_flags", None) or {})
    flags["twin_filtered"] = wrec.twin_filtered(mode, flags)
    return flags


def _victim_window_record(node, victim, baseline_poly):
    """Per-window victim record: distance from the baseline and added deviation along the stretch
    (true poses), scans from the victim MOLA log within the stamps of the stretch (drive) and
    from the first pose after the stream reopens (open), transport state from the environment.
    """
    rec = {"baseline": None, "scans_drive": None, "scans_open": None,
           "scans_definition": wrec.SCAN_DEFINITIONS,
           "transport": wrec.transport_state()}
    try:
        if baseline_poly is not None and node.last_trace:
            rec["baseline"] = wrec.baseline_window_fields(true_path(node.last_trace), baseline_poly)
        s0, s1 = getattr(node, "last_drive_stamps", (None, None))
        rec["drive_stamps"] = [s0, s1]
        rec["open_stamp"] = getattr(node, "last_victim_open_stamp", None)
        log = getattr(victim, "log_path", None) if victim is not None else None
        rec["log_path"] = str(log) if log else None
        rows = wrec.read_printf(log)
        rec["printf_rows"] = len(rows)
        rec["prima_icp"] = wrec.prima_icp(rows, rec["open_stamp"])
        if rows:
            disc = wrec.read_discards(log)
            rec["scans_drive"] = wrec.scan_counts(rows, s0, s1, discards=disc)
            rec["scans_open"] = wrec.scan_counts(rows, rec["open_stamp"], s1, discards=disc)
        t0 = (round(rec["open_stamp"] + wrec.PERIODO_S, 1) if rec["open_stamp"] is not None else s0)
        if victim is not None and t0 is not None:
            # victim MOLA timing, attack outcome: scans after the stream reopens; the end (reopening
            # of the next window) is fixed by _refresh_victim_scans
            try:
                rec["tempi_mola"] = wrec.tempi_mola_finestra(
                    wrec.leggi_csv(getattr(victim, "traces_path", None)), t0, None, wrec.ripartenze_icp(log))
            except Exception as e:
                rec["tempi_mola"] = {"t0": t0, "t1": None, "completo": False, "errore": str(e)}
    except Exception as e:
        rec["error"] = str(e)
    return rec


def _refresh_victim_scans(history, victim, fine_run=False):
    """Recompute the counts of previous windows left incomplete: the victim MOLA writes stdout in
    blocks and the last lines of a window reach the file later. With fine_run (MOLA instances
    stopped) it recomputes only the MOLA timings of all windows, the last one up to the end of
    the run.
    """
    log = getattr(victim, "log_path", None) if victim is not None else None
    if not log:
        return
    def scans_da_rifare(vw):
        return any(isinstance(vw.get(k), dict) and not vw[k].get("log_complete")
                   for k in ("scans_drive", "scans_open"))

    # MOLA timings: the window ends at the stream reopening in the next window
    con_vw = [h for h in history if isinstance(h.get("victim_window"), dict)]
    fine = {id(a): b["victim_window"].get("open_stamp") for a, b in zip(con_vw[:-1], con_vw[1:])}

    def tempi_da_rifare(h):
        tm = h["victim_window"].get("tempi_mola")
        if not isinstance(tm, dict) or tm.get("t0") is None:
            return False
        return fine_run or (not tm.get("completo") and fine.get(id(h)) is not None)

    def scans_qui(h):
        return not fine_run and scans_da_rifare(h["victim_window"])
    todo = [h for h in con_vw if scans_qui(h) or tempi_da_rifare(h)]
    if not todo:
        return
    rows = disc = tracce = rip = None
    for h in todo:
        vw = h["victim_window"]
        if scans_qui(h):
            try:
                if rows is None:
                    rows = wrec.read_printf(log)
                    disc = wrec.read_discards(log)
                s0, s1 = (vw.get("drive_stamps") or [None, None])
                vw["scans_drive"] = wrec.scan_counts(rows, s0, s1, discards=disc)
                vw["scans_open"] = wrec.scan_counts(rows, vw.get("open_stamp"), s1, discards=disc)
                if (vw.get("prima_icp") or {}).get("q") is None:
                    vw["prima_icp"] = wrec.prima_icp(rows, vw.get("open_stamp"))
                vw["refreshed"] = True
            except Exception:
                pass
        if tempi_da_rifare(h):
            # an error in the timings stays in the window field and does not stop the others
            try:
                if tracce is None:
                    tracce = wrec.leggi_csv(getattr(victim, "traces_path", None))
                    rip = wrec.ripartenze_icp(log)
                t1 = fine.get(id(h))
                vw["tempi_mola"] = wrec.tempi_mola_finestra(tracce, vw["tempi_mola"]["t0"], t1, rip)
                if fine_run and t1 is None:
                    vw["tempi_mola"]["fino_a_fine_run"] = True
            except Exception as e:
                vw["tempi_mola"]["errore"] = str(e)


def motion_vittima(node):
    """drift_fields of the victim stretch, plus frame_coerenza with --origine-coerente-vittima. Only
    for victim_motion and applied_motion: the value is reset at every victim stretch and does not
    enter the twin records.
    """
    d = drift_fields(node)
    if ORIGINE_VITTIMA["coerente"] and getattr(node, "last_frame_coerenza", None) is not None:
        d["frame_coerenza"] = dict(node.last_frame_coerenza)
    return d


def drift_fields(node):
    """Temporal drift and motion class fields of the last rollout."""
    f = lambda v: round(v, 2) if v == v else None
    d = {"drift_start_cm": f(node.last_drift_start_cm),
         "drift_end_cm": f(node.last_drift_end_cm)}
    d.update(node.last_motion or {})
    d["align_err_cm"] = f(getattr(node, "last_align_err_cm", float("nan")))
    d["align_err_deg"] = f(getattr(node, "last_align_err_deg", float("nan")))
    if getattr(node, "last_resume_jump_cm", None) is not None:
        d["resume_jump_cm"] = round(node.last_resume_jump_cm, 2)
    # End-of-window stop (window_stop.STOP_INFO_KEYS), with both rules.
    d.update(getattr(node, "last_stop_info", None) or {})
    return d


def _damage_vs_nominal(end, nom_end, goal, nom_path, cand_path, mode, dir_weight):
    """Damage against the nominal of the same window (reference 'window')."""
    dev_end = float(math.hypot(end[0] - nom_end[0], end[1] - nom_end[1]))
    pd = path_deviation(nom_path, cand_path) if len(cand_path) >= 2 else None
    dev_mean = pd["dev_mean"] if pd else dev_end
    dev_max = pd["dev_max"] if pd else dev_end
    d_nom_goal = math.hypot(goal[0] - nom_end[0], goal[1] - nom_end[1])
    d_att_goal = math.hypot(goal[0] - end[0], goal[1] - end[1])
    directional = d_att_goal - d_nom_goal
    dev = {"end": dev_end, "mean": dev_mean, "max": dev_max}[mode]
    return {"dev_end": dev_end, "dev_mean": dev_mean, "dev_max": dev_max,
            "directional": directional,
            "damage": dev + dir_weight * max(0.0, directional),
            "path_points": int(len(cand_path))}


def window_damage(end, nom_end, goal, nom_path, cand_path, mode, dir_weight,
                  baseline=None):
    """Damage of a rollout.

    Reference 'window' (baseline=None, default): the nominal of the same window. The deviation along
    the stretch compares the two true trajectories at equal travelled distance
    (src/analysis/path_deviation.py): mode chooses whether the damage uses the mean along the
    stretch, the maximum or the final pose only. The directional term is the increase of the
    distance to the goal with respect to the nominal, counted only if positive: a drift that circles
    the goal is worth less than one that makes it miss.

    Reference 'baseline' (baseline = Mx2 polyline in the odom frame, --damage-ref baseline):
    dev_end, dev_mean and dev_max are the distances of the rollout true poses from the closest point
    of the polyline (src/analysis/baseline_distance.py; beyond the ends the distance from the end
    counts, so overshooting the goal is damage). The directional term does NOT enter the damage and
    is recorded as 0: damage is the deviation selected by mode only. Non-finite poses of cand_path
    are discarded; without true poses (cand_path empty or all non-finite) the final pose end is
    used. The comparison with the window nominal stays available, when nom_path is known, in
    dev_end_window, dev_mean_window, dev_max_window and damage_window.

    Returns a dict with dev_end, dev_mean, dev_max, directional and damage, all in metres.
    """
    if baseline is None:
        return _damage_vs_nominal(end, nom_end, goal, nom_path, cand_path, mode, dir_weight)
    cand = np.asarray(cand_path, dtype=np.float64).reshape(-1, 2)
    cand = cand[np.all(np.isfinite(cand), axis=1)]
    if len(cand) == 0:
        cand = np.array([[float(end[0]), float(end[1])]])
    bd = baseline_damage(cand, baseline)
    out = {"dev_end": bd["dev_end"], "dev_mean": bd["dev_mean"], "dev_max": bd["dev_max"],
           "directional": 0.0,
           "damage": bd[f"dev_{mode}"],
           "path_points": int(len(cand_path))}
    if nom_end is not None and nom_path is not None:
        w = _damage_vs_nominal(end, nom_end, goal, nom_path, cand_path, mode, dir_weight)
        out.update({"dev_end_window": w["dev_end"], "dev_mean_window": w["dev_mean"],
                    "dev_max_window": w["dev_max"], "damage_window": w["damage"]})
    return out


def _window_fields(dm, scale=1.0, suffix=""):
    """Per-window damage fields (reference 'window') if present in the dict."""
    keys = ("dev_end_window", "dev_mean_window", "dev_max_window", "damage_window")
    return {k + suffix: dm[k] * scale for k in keys if k in dm}


def pareto_idx_maxmin(dev, ch):
    """Non-dominated indices maximising dev and minimising ch."""
    idx = []
    for i in range(len(dev)):
        if not any(j != i and dev[j] >= dev[i] and ch[j] <= ch[i]
                   and (dev[j] > dev[i] or ch[j] < ch[i]) for j in range(len(dev))):
            idx.append(i)
    return idx


def pareto_front(F, X):
    """Non-dominated solutions (minimisation on both columns)."""
    F = np.atleast_2d(F)
    X = np.atleast_2d(X)
    if len(F) == 0:
        return F, X
    keep = []
    for i, f in enumerate(F):
        dominated = any(np.all(F[j] <= f) and np.any(F[j] < f) for j in range(len(F)) if j != i)
        if not dominated:
            keep.append(i)
    return F[keep], X[keep]


def pick_best(F, X, budget=None):
    """From the front, the solution with the highest damage/perceptibility ratio.

    F[:,0] is the negated damage in metres (pymoo minimises), F[:,1] the perceptibility in cm. With
    budget only admissible ones (F[:,1] <= budget) are considered: a genome over budget is never
    applied. Solutions with infinite fitness are discarded rollouts.

    Returns (genome, F, ratio) or (None, None, None).
    """
    if F is None or len(F) == 0:
        return None, None, None
    F = np.atleast_2d(F)
    X = np.atleast_2d(X)
    ok = np.all(np.isfinite(F), axis=1)
    if budget is not None:
        ok = ok & (F[:, 1] <= budget)
    if not ok.any():
        return None, None, None
    F, X = F[ok], X[ok]
    ratios = [(-f[0] * 100.0) / max(f[1], 0.001) for f in F]
    i = int(np.argmax(ratios))
    return X[i], F[i], ratios[i]


# ---------------------------------------------------------------------------

def exit_code(stalled, lost_state, fail_on_no_pose):
    """Exit code of main: 3 Isaac stall, 4 victim without pose with --fail-rc-on-no-pose, 0
    otherwise.
    """
    if stalled:
        return 3
    if fail_on_no_pose and lost_state.get("any"):
        return 4
    return 0


# Default --dropout-rate of perturbation_node: the generator caps it at MAX_DROPOUT_RATE (0.05).
# Used only if the node status does not report the limits (older node).
NODE_DROPOUT_DEFAULT = 0.15


def node_gen_limits(node):
    """Node generator limits for genome decoding: those published in the status (gen_limits) or, if
    missing, the node defaults.
    """
    lim = getattr(node, "node_gen_limits", None)
    if isinstance(lim, dict) and lim.get("max_dropout_rate") is not None:
        return dict(lim, source="status del nodo")
    return {"max_point_shift": 0.05, "noise_std": 0.02,
            "max_dropout_rate": min(NODE_DROPOUT_DEFAULT, MAX_DROPOUT_RATE),
            "max_ghost_points_ratio": 0.02, "source": "default del nodo (status senza gen_limits)"}


def structural_fields(node, args):
    """Shape of the structural term reported by the node (None = node without the key, i.e. net),
    expected shape (--structural) and comparison outcome.
    """
    rep = getattr(node, "node_structural", None)
    exp = getattr(args, "structural", None)
    return {"node_structural": rep, "structural_expected": exp,
            "structural_ok": exp is None or (rep or "net") == exp}


def print_generator(node):
    """Generator used only to print decoded parameters, with the actual node limits (the default
    orchestrator generator caps dropout at 0.03 instead of 0.05) and the orchestrator limit
    multiplier, if on.
    """
    lim = node_gen_limits(node)
    mol = {"moltiplicatori": dict(LIMITI["mol"])} if LIMITI["mol"] else {}
    return PerturbationGenerator(max_point_shift=lim["max_point_shift"], noise_std=lim["noise_std"],
                                 max_dropout_rate=lim["max_dropout_rate"],
                                 max_ghost_points_ratio=lim["max_ghost_points_ratio"], **mol)


def controlla_moltiplicatori(node, contesto):
    """Limit multiplier reported by the node (status moltiplicatori_limiti, absent = off) against
    the orchestrator one (LIMITI["mol"]). Different: ErroreMoltiplicatori. Off on both sides: no
    effect.
    """
    rep = getattr(node, "node_moltiplicatori", None) or {}
    mio = LIMITI["mol"] or {}
    if {k: float(v) for k, v in rep.items()} != {k: float(v) for k, v in mio.items()}:
        raise ErroreMoltiplicatori(f"{contesto}: moltiplicatore dei limiti del nodo {rep or 'spento'}, "
                                   f"dell'orchestratore {mio or 'spento'}")


def limiti_zone(node, contesto):
    """Limits for the attacker zones: the node ones (node_gen_limits) and, with the multiplier on,
    the multipliers (equal to the node ones, checked here). Off: node_gen_limits.
    """
    controlla_moltiplicatori(node, contesto)
    lim = node_gen_limits(node)
    if LIMITI["mol"]:
        lim = dict(lim, moltiplicatori=dict(LIMITI["mol"]))
    return lim


def limiti_fields(node):
    """History fields for the limits study, only with the options on: orchestrator and node
    multiplier, locked genes.
    """
    out = {}
    if LIMITI["mol"]:
        out["moltiplicatori_limiti"] = dict(LIMITI["mol"])
        out["moltiplicatori_limiti_nodo"] = getattr(node, "node_moltiplicatori", None)
    if LIMITI["bloccati"]:
        out["geni_bloccati"] = {str(g): -1.0 for g in LIMITI["bloccati"]}
    return out


def geni_attivi(gruppo, bloccati=()):
    """Indices of the genes in the search space: those of the group minus the locked genes
    (--blocca-gene), which expand_genome sets to -1 like the inactive genes. Applies to initial
    sampling, warm start, NSGA-III operators, random search and single random: none of them sees
    the locked genes.
    """
    return [g for g in GENOME_GROUPS[gruppo] if g not in set(bloccati)]


def leggi_blocca_gene(voci, n_genes=17):
    """--blocca-gene I=V (repeatable, also comma-separated) -> sorted indices. Only V = -1 (operator
    off: the gene leaves the search space and the expansion sets it to -1); error otherwise.
    """
    out = []
    for voce in voci or []:
        for parte in str(voce).split(","):
            parte = parte.strip()
            if not parte:
                continue
            try:
                i, v = parte.split("=")
                i, v = int(i), float(v)
            except ValueError:
                raise ValueError(f"--blocca-gene: voce {parte!r} non nella forma I=V (es. 14=-1)")
            if not 0 <= i < n_genes:
                raise ValueError(f"--blocca-gene: gene {i} fuori da 0..{n_genes - 1}")
            if v != -1.0:
                raise ValueError(f"--blocca-gene {parte}: solo il valore -1 (gene spento)")
            if i in out:
                raise ValueError(f"--blocca-gene: gene {i} ripetuto")
            out.append(i)
    return sorted(out)


def problema_limiti(args):
    """Checks of --moltiplicatore-limiti, --famiglie-limiti and --blocca-gene; sets LIMITI. None if
    fine (or off).
    """
    try:
        LIMITI["mol"] = moltiplicatori_da_opzioni(getattr(args, "moltiplicatore_limiti", 1.0),
                                                  getattr(args, "famiglie_limiti", None))
        LIMITI["bloccati"] = leggi_blocca_gene(getattr(args, "blocca_gene", None))
    except ValueError as e:
        return str(e)
    bl = LIMITI["bloccati"]
    if bl and getattr(args, "fixed_genome", None):
        try:
            g = [float(v) for v in args.fixed_genome.split(",")]
        except ValueError:
            return None    # a malformed genome is reported by the usual parsing
        diversi = [i for i in bl if i < len(g) and g[i] != -1.0]
        if diversi:
            return (f"--blocca-gene e --fixed-genome in conflitto: geni {diversi} del genoma fisso "
                    f"diversi da -1")
    if bl and getattr(args, "fixed_genome_seq", None):
        return "--blocca-gene non e' previsto con --fixed-genome-seq"
    return None


def processo_mola_vittima(victim):
    """Victim mola-cli process when the run closes for victim without pose (the ROS 2 bridge is a
    module of the same process: config/mola/lidar_odometry_ros2_local.yaml, mola::BridgeROS2;
    launch_mola_attack.sh execs mola-cli). esito: vivo, terminato (process crash; rc = exit code,
    negative = signal) or non_trovato (no process: victim not started or already closed; vivo
    None, not a crash).
    """
    p = getattr(victim, "proc", None)
    if p is None:
        return {"vivo": None, "rc": None, "pid": None, "esito": "non_trovato"}
    rc = p.poll()
    return {"vivo": rc is None, "rc": rc, "pid": getattr(p, "pid", None),
            "esito": "vivo" if rc is None else "terminato"}


def campi_vittima_persa(victim):
    """run_end fields for the victim without pose (always, even with the crash rule off): process,
    rejection signs in the printf lines read online (the last ones may be missing: block-buffered
    stdout) and marker (window_record.ESITI_VITTIMA_PERSA). The end of the run (queue, history)
    rereads the full log and decides the marker.
    """
    proc = processo_mola_vittima(victim)
    seg = wrec.segnali_rifiuto(wrec.read_printf(getattr(victim, "log_path", None)))
    return {"processo_mola_vittima": proc, "rifiuto_in_linea": seg,
            "esito_vittima": wrec.esito_vittima_persa(proc, seg)}


def check_spawn_b_args(args):
    """Options --victim-never-clean, --victim-start and --settle-clouds: fills in the defaults
    (modifies args) and returns an error message or None.
    """
    nc = bool(getattr(args, "victim_never_clean", False))
    if nc:
        bad = [o for o, on in (("--attack-from motion", args.attack_from == "motion"),
                               ("--warmup > 0", args.warmup > 0),
                               ("--passthrough", args.passthrough),
                               ("--map-gate-test", args.map_gate_test),
                               ("--victim restart", args.victim != "continuous"),
                               # without search every window would be held: the victim would never
                               # start
                               ("--search none", getattr(args, "search", None) == "none")) if on]
        if bad:
            return f"--victim-never-clean non va con {', '.join(bad)}"
        args.victim_start = "w0"
        if args.settle_clouds is None:
            args.settle_clouds = 20
        if args.reopen_clouds is None:
            args.reopen_clouds = 8
    args.settle_clouds = int(args.settle_clouds or 0)
    args.reopen_clouds = int(args.reopen_clouds or 0)
    if args.settle_clouds < 0 or args.reopen_clouds < 0:
        return "--settle-clouds e --reopen-clouds vanno >= 0"
    if args.reopen_clouds > 0 and args.settle_clouds == 0:
        return "--reopen-clouds richiede --settle-clouds > 0"
    if args.victim_start == "w0":
        if args.victim != "continuous":
            return "--victim-start w0 richiede --victim continuous"
        if args.twin_state in ("victim", "replay", "maprestore"):
            return (f"--victim-start w0 non va con --twin-state {args.twin_state}: prima dell'avvio "
                    "la vittima non ha storia ne' mappa")
    return None


def check_random_first_args(args):
    """Allowed combinations of --random-first-admissible: error message or None.

    The rule "apply the first valid one" matches the selection only with pareto_maxdamage (same
    candidato_valido); maxdamage and ratio choose among the admissible ones with other criteria and
    re-evaluate, so they are rejected.
    """
    k = args.random_first_admissible
    if k < 0:
        return f"--random-first-admissible deve essere >= 0 (dato {k})"
    if k > 0 and args.search != "random":
        return f"--random-first-admissible vale solo con --search random (dato --search {args.search})"
    if k > 0 and args.select != "pareto_maxdamage":
        return (f"--random-first-admissible {k} richiede --select pareto_maxdamage (dato --select "
                f"{args.select}): con le altre selezioni il genoma applicato non sarebbe il primo valido")
    return None


def warm_front_rows(warm_pool, half, rng_w, n_search):
    """Warm half of the initial NSGA-III population (--warm-start).

    N/2 from the Pareto front of the previous window (warm_pool: tuples genome, damage, Chamfer),
    spread by increasing Chamfer (uniform subsample if the front is large); if the front has fewer
    than N/2 members it is filled with their perturbations (1-3 genes, noise 0.3, also of already
    perturbed rows). Keeps the same rng_w draw sequence as the original inline code. Returns (rows,
    counts from window_record.warm_start_counts).
    """
    def _mutate(base):
        m = int(rng_w.integers(1, min(3, n_search) + 1))
        idx_m = rng_w.choice(n_search, size=m, replace=False)
        child = np.array(base, dtype=float).copy()
        child[idx_m] = np.clip(child[idx_m] + rng_w.normal(0.0, 0.3, size=m), -1.0, 1.0)
        return child
    front_sorted = sorted(warm_pool, key=lambda t: t[2])  # by Chamfer
    if len(front_sorted) >= half:
        sel = np.linspace(0, len(front_sorted) - 1, half).round().astype(int)
        rows = [np.array(front_sorted[i][0], dtype=float) for i in sel]
    else:
        rows = [np.array(t[0], dtype=float) for t in front_sorted]
        while rows and len(rows) < half:
            rows.append(_mutate(rows[int(rng_w.integers(len(rows)))]))
    n_copy = min(len(front_sorted), half)
    counts = wrec.warm_start_counts(rows[:n_copy], rows[n_copy:], [t[0] for t in front_sorted])
    if front_sorted and len(front_sorted[0]) > 3:
        # indices in the previous window eval_log of the copied members
        copied = ([front_sorted[i] for i in sel] if len(front_sorted) >= half else front_sorted)
        counts["warm_front_copied_idx"] = [int(t[3]) for t in copied]
    return rows, counts


def mola_config_env(args):
    """MOLA variables given on the command line (--lidar-pose-x, --initial-sigma), for all roles.
    Empty if the options are not given.
    """
    env = {}
    if args.lidar_pose_x is not None:
        env["LIDAR_POSE_X"] = args.lidar_pose_x
    if args.initial_sigma is not None:
        env["MOLA_INITIAL_SIGMA"] = args.initial_sigma
    return env


# Variables the orchestrator sets by itself for every instance.
ENV_RESERVED = ("ROS_ARGS", "TRACES")


def parse_env_pairs(items):
    """List of 'K=V' (--victim-mola-env) into a dict; ValueError for an entry without '=', with an
    empty or reserved key.
    """
    env = {}
    for it in items or []:
        k, sep, v = str(it).partition("=")
        k = k.strip()
        if not sep or not k:
            raise ValueError(f"--victim-mola-env '{it}': serve K=V")
        if k in ENV_RESERVED:
            raise ValueError(f"--victim-mola-env: {k} e' impostata dall'orchestratore")
        env[k] = v
    return env


def load_genome_seq(spec, n_genes=17):
    """Per-window genome sequence (--fixed-genome-seq): [(genome, source)].

    spec: .json file with a list, or entries separated by ';'. Entry: list of n_genes values, string
    of comma-separated values, or 'history.json:window[:candidate]' (index in the history list, as
    --repeat-genome: applied genome, or of one evaluation).
    """
    if str(spec).endswith(".json") and os.path.isfile(spec):
        with open(spec) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{spec}: serve una lista JSON")
    else:
        entries = [e.strip() for e in str(spec).split(";") if e.strip()]
    if not entries:
        raise ValueError("--fixed-genome-seq vuota")
    hist_cache, out = {}, []
    for e in entries:
        src = {"entry": e}
        if isinstance(e, str) and ".json:" in e:
            parts = e.split(":")
            path, widx = parts[0], int(parts[1])
            eidx = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            if path not in hist_cache:
                with open(path) as f:
                    hist_cache[path] = json.load(f)
            w = hist_cache[path][widx]
            g = w["genome"] if eidx is None else w["evaluations"][eidx]["genome"]
            src.update(history=path, source_index=widx, source_window=w.get("window"),
                       source_evaluation=eidx)
        elif isinstance(e, str):
            g = [float(v) for v in e.split(",")]
        else:
            g = e
        g = np.array(g, dtype=float)
        if g.shape != (n_genes,):
            raise ValueError(f"--fixed-genome-seq: voce {e!r} con {g.size} valori invece di {n_genes}")
        out.append((g, src))
    return out


def skipped_window_record(k, st, goal, wp_idx, v_st, window_time_s, shadow_state=None):
    """Record of a window skipped because the nominal failed twice.

    With --twin-state ombra_e and missing shadow state the nominal fails before starting MOLA, like
    every candidate of the window (same state): no genome evaluated or applied, the victim drives
    the stretch without attack. The window stays an infrastructure failure, with the shadow state
    that caused it.
    """
    rec = {"window": k, "status": f"nominal_failed({st})",
           "goal": list(map(float, goal)), "waypoint_index": wp_idx,
           "victim_status": v_st, "window_time_s": window_time_s}
    for miss in ("ombra_e_state_missing", "ombra_g_state_missing"):
        if str(st).startswith(miss):
            rec.update({"infra_failure": miss, "applied": False, "shadow_state": shadow_state})
    return rec


def ombra_infra_fields(eval_log):
    """Candidates discarded for missing shadow state; if all of them are, the window without genome
    is marked as an infrastructure failure (case of the .mm vanished after the nominal). Empty if
    there are none.
    """
    rec = {}
    for miss in ("ombra_e_state_missing", "ombra_g_state_missing"):
        n = sum(1 for e in eval_log if str(e.get("status", "")).startswith(miss))
        if not n:
            continue
        rec[f"n_{miss}"] = n
        if n == len(eval_log):
            rec.update({"infra_failure": miss, "applied": False})
    return rec


def genome_seq_entry(genome_seq, k):
    """Genome of window k (entry k, the last one for the following windows) and history record:
    seq_pos = position in the sequence, source_* = source entry.
    """
    seq_i = min(k, len(genome_seq) - 1)
    g, src = genome_seq[seq_i]
    return g, {**src, "seq_pos": seq_i, "n": len(genome_seq)}


def warm_pool_from_log(eval_log, select, budget, active_genes):
    """Pareto front of a window for the warm start of the next one.

    With pareto_maxdamage the front of fronte_pareto (valid candidates), otherwise the non-dominated
    (dev_mean, Chamfer) among the admissible ones. Returns (warm_pool, warm_best, idx): warm_pool is
    a list of tuples (genome in search space, damage, Chamfer, index in eval_log) or None; idx the
    eval_log indices of the front.
    """
    if select == "pareto_maxdamage":
        fr_w, _ = fronte_pareto(eval_log, budget=budget)
        pool = [(np.array(p["genome"], dtype=float)[active_genes], p["damage"] * 100, p["chamfer"], p["idx"])
                for p in fr_w]
        return (pool or None), None, [int(t[3]) for t in pool]
    adm = [(i, e) for i, e in enumerate(eval_log)
           if e.get("admissible") and e.get("dev_mean") is not None and e.get("genome")]
    if not adm:
        return None, None, []
    pts = [((e["dev_mean"] * 100), e["chamfer"]) for _, e in adm]
    fidx = pareto_idx_maxmin([p[0] for p in pts], [p[1] for p in pts])
    pool = [(np.array(adm[j][1]["genome"], dtype=float)[active_genes],
             adm[j][1]["dev_mean"] * 100, adm[j][1]["chamfer"], adm[j][0]) for j in fidx]
    best = sorted([(np.array(e["genome"], dtype=float)[active_genes], e["dev_mean"] * 100)
                   for _, e in adm], key=lambda t: -t[1])
    return pool, best, [int(t[3]) for t in pool]


def top_maxdamage(eval_log, args):
    """Candidates to re-evaluate with --select maxdamage: the first --topk distinct genomes by
    decreasing search damage (ties: evaluation order).

    Filters: admissible with damage, not slow (unless --search fixed or --admissibility none), twin
    aligned and not filtered. With --select-only-valid only valid candidates enter
    (motivi_non_valido).
    """
    cands = [e for e in eval_log if e.get("admissible") and e.get("damage") is not None
             and (not e.get("slow") or args.search == "fixed" or args.admissibility == "none") and not e.get("twin_misaligned")
             and not e.get("twin_filtered")]
    if getattr(args, "select_only_valid", False):
        cands = [e for e in cands if not motivi_non_valido(e, args.chamfer_budget)]
    cands.sort(key=lambda e: -e["damage"])
    seen, top = set(), []
    for e in cands:
        key = tuple(round(v, 4) for v in e["genome"])
        if key in seen:
            continue
        seen.add(key); top.append(e)
        if len(top) >= args.topk:
            break
    return top


def legacy_random_genomes(rng, n, n_search, sparse, off):
    """Random genomes with --sampling legacy: with sparse (--random-sparse, and always for the
    sparse population and the random half of the NSGA-III warm start) a random number of active
    genes with uniform values and the others at the off value `off`; without, uniform in [-1, 1].
    """
    if not sparse:
        return rng.uniform(-1.0, 1.0, size=(n, n_search))
    X = np.tile(off, (n, 1))
    for i in range(n):
        kk = int(rng.integers(1, n_search + 1))
        idx = rng.choice(n_search, size=kk, replace=False)
        X[i, idx] = rng.uniform(-1.0, 1.0, size=kk)
    return X


def random_first_run(draw, evaluate, eval_log, is_valid, k_max, rf_rec=None):
    """Single random with budget (--random-first-admissible).

    First call (rf_rec None): draw and evaluate one genome at a time until the first valid eval_log
    record, at most k_max. Later calls (after re-evaluations for an adapter restart): if no eval_log
    record is valid any more, resume with the remaining attempts. evaluate must append the record to
    eval_log. Returns (rf_rec, new genomes, new objectives); rf_rec["eval_idx"] is the first valid
    record in evaluation order.
    """
    first_call = rf_rec is None
    rec = {"k_max": int(k_max), "n_tries": 0} if first_call else rf_rec
    first = None if first_call else wrec.first_valid(eval_log, is_valid)
    left = int(k_max) - rec["n_tries"]
    xs, fs = [], []
    if first is None and left > 0:
        xs, fs, _ = wrec.first_admissible(draw, evaluate, lambda: is_valid(eval_log[-1]), left)
        first = wrec.first_valid(eval_log, is_valid)
        if not first_call:
            rec["n_tries_resumed"] = rec.get("n_tries_resumed", 0) + len(xs)
    rec["n_tries"] += len(xs)
    rec["found"] = first is not None
    rec["eval_idx"] = first
    return rec, xs, fs


def random_first_choice(fronte, eval_log, rf_rec):
    """Point applied by the single random: record rf_rec["eval_idx"], taken from the front if
    present (same dict as the selection), otherwise built with the same keys. None if no valid
    one.
    """
    idx = rf_rec.get("eval_idx") if rf_rec else None
    if idx is None:
        return None
    for p in fronte:
        if p["idx"] == idx:
            return p
    r = eval_log[idx]
    return {"idx": idx, "genome": list(r.get("genome") or []), "damage": float(r["damage"]),
            "chamfer": float(r["chamfer"]), "tau_s": r.get("tau_s"), "slow": bool(r.get("slow")),
            "status": r.get("status"), "in_pymoo_front": None, "in_final_pop": None,
            "outside_front": True}


def random_first_search(rng, n_search, mode_s, off, eval_genome, eval_log, budget, k_max):
    """Single random search in main: one genome at a time from the same generator as the random
    search, evaluated with eval_genome (which appends the record to eval_log), until the first
    valid one for the selection, at most k_max. Returns (rf_rec, X_all, F_all, draw); draw is
    used for the resume.
    """
    def rf_draw():
        return wrec.sample_genomes(rng, 1, n_search, mode_s, off)[0]
    rf_rec, rows_x, rows_f = random_first_run(
        rf_draw, lambda x: eval_genome(x)[:2], eval_log,
        lambda e: candidato_valido(e, budget), k_max)
    rf_rec["sampling"] = mode_s
    print(f"  random singolo con budget: {'valido' if rf_rec['found'] else 'nessun valido'} "
          f"dopo {rf_rec['n_tries']} tentativi su {k_max}")
    return rf_rec, np.array(rows_x), np.array(rows_f, dtype=float), rf_draw


def random_first_resume(rf_rec, rf_draw, eval_genome, eval_log, budget, k_max, X_all, F_all):
    """Resume of the single random after re-evaluations for an adapter restart: if no record is
    valid any more, search with the remaining attempts. Returns (rf_rec, X_all, F_all, number of
    new genomes).
    """
    rf_rec, more_x, more_f = random_first_run(
        rf_draw, lambda x: eval_genome(x)[:2], eval_log,
        lambda e: candidato_valido(e, budget), k_max, rf_rec)
    if more_x:
        print(f"  random singolo: ripresa dopo il riavvio dell'adattatore, {len(more_x)} "
              f"tentativi in piu' ({'valido' if rf_rec['found'] else 'nessun valido'})")
        X_all = np.vstack([X_all, np.array(more_x)])
        F_all = np.vstack([F_all, np.array(more_f, dtype=float)])
    return rf_rec, X_all, F_all, len(more_x)


def reevaluate_after_adapter_restart(eval_log, events, t_win, eval_genome, active_genes):
    """Re-evaluation of the candidates measured during an adapter degradation.

    events: rows of the --adapter-events file. For every restart since the window start (t_win) the
    records with t_wall between the last clean check and the restart become non-admissible and not
    counted, and their genome is re-evaluated (new record appended, with reevaluation_of).
    """
    restarts = [ev for ev in events if ev.get("t_restart") and ev["t_restart"] >= t_win]
    for ev in restarts:
        lo, hi = ev.get("t_prev_clean", t_win), ev["t_restart"]
        bad = [e for e in eval_log if e.get("t_wall") and lo <= e["t_wall"] <= hi and not e.get("measured_during_adapter_degradation") and not e.get("superseded_by")]
        print(f"  ADATTATORE riavviato alle {time.strftime('%H:%M:%S', time.localtime(hi))}: rivaluto {len(bad)} candidati misurati dalle {time.strftime('%H:%M:%S', time.localtime(lo))}")
        for e in list(bad):
            e["measured_during_adapter_degradation"] = True
            e["admissible"] = False; e["counted"] = False
            try:
                eval_genome(np.array(e["genome"], dtype=float)[active_genes])
                e["superseded_by"] = len(eval_log) - 1
                eval_log[-1]["reevaluation_of"] = eval_log.index(e)
            except WindowTimeout:
                print("  WATCHDOG durante la rivalutazione: resto con cio' che ho")
                break


def describe_params(params: dict) -> str:
    """All parameters decoded from the genome, one per line."""
    lines = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, np.ndarray):
            v = "[" + ", ".join(f"{x:+.3f}" for x in np.atleast_1d(v)) + "]"
            lines.append(f"      {k:26s} {v}")
        elif isinstance(v, (int, float, np.floating)):
            lines.append(f"      {k:26s} {float(v):+.4f}")
        else:
            lines.append(f"      {k:26s} {v}")
    return "\n".join(lines)


def problema_opzioni_d31(args):
    """Checks of the winner-trial, realignment and attitude options against the other arguments:
    reason or None. Off: None.
    """
    n = getattr(args, "prove_vincitore", 0) or 0
    ria = bool(getattr(args, "riallinea_dopo_assestamento", False))
    conf = bool(getattr(args, "confronto_vincitore", False)) or ria
    tia = getattr(args, "twin_init_assetto", "azzerato") or "azzerato"
    if n < 0:
        return "--prove-vincitore deve essere >= 0"
    if getattr(args, "g_banco", None) and (n or ria or conf or tia != "azzerato"):
        return ("--g-banco non va con --prove-vincitore, --riallinea-dopo-assestamento, --confronto-vincitore "
                "o --twin-init-assetto (il banco ha le sue chiavi nella specifica)")
    if tia != "azzerato" and getattr(args, "twin_init", "true") not in ("attacker_est", "ombra_est"):
        return f"--twin-init-assetto {tia} richiede --twin-init attacker_est o ombra_est (set_state del gemello)"
    if tia in ("attaccante", "attaccante_relativo") and getattr(args, "twin_init", "true") != "attacker_est":
        return f"--twin-init-assetto {tia} richiede --twin-init attacker_est (MOLA dell'attaccante)"
    if ria:
        if getattr(args, "victim", None) != "continuous":
            return "--riallinea-dopo-assestamento richiede --victim continuous"
        if getattr(args, "twin_state", None) != "ombra_g" or getattr(args, "twin_init", None) != "attacker_est":
            return "--riallinea-dopo-assestamento richiede --twin-state ombra_g e --twin-init attacker_est"
        if not getattr(args, "fresh_cloud_after_restore", False):
            # on resume the first scan is partly taken in the end pose of the realigned twin
            return "--riallinea-dopo-assestamento richiede --fresh-cloud-after-restore (nodo e orchestratore)"
    if conf and not (getattr(args, "trace", False) and getattr(args, "damage_ref", None) == "baseline"):
        return ("--confronto-vincitore e --riallinea-dopo-assestamento richiedono --trace (traccia del gemello che "
                "ha scelto) e --damage-ref baseline (laterale e danno del criterio A)")
    return None


def problema_rampa(args):
    """Checks of --rampa-genoma against the other arguments (None if fine or off)."""
    R = getattr(args, "rampa_genoma", 1)
    if R is None or R == 1:
        return None
    if R < 1:
        return "--rampa-genoma deve essere >= 1 (1 = spenta)"
    if not (getattr(args, "zone_attaccante", False) and getattr(args, "metro_al_nodo", False)):
        return "--rampa-genoma richiede --zone-attaccante e --metro-al-nodo (zone di P e di N dalla fotografia)"
    if getattr(args, "attack_from", "spawn") != "spawn":
        return "--rampa-genoma richiede --attack-from spawn (k della rampa dalla prima nuvola del genoma)"
    n = getattr(args, "settle_clouds", 0) or 0
    m = getattr(args, "reopen_clouds", 0) or 0
    ass = min(n, m) if m > 0 else n
    if n <= 0 or R > ass - 8:
        return (f"--rampa-genoma {R}: serve R <= assestamento - 8 (--settle-clouds {n}, --reopen-clouds {m}: "
                f"assestamento {ass})")
    return None


def problema_genoma_in_moto(args):
    """Checks of --genoma-in-moto against the other arguments (None if fine or off)."""
    S = getattr(args, "genoma_in_moto", 0.0)
    if S is None or S == 0:
        return None
    if not S > 0:
        return "--genoma-in-moto deve essere >= 0 (0 = spento)"
    kmin = getattr(args, "genoma_in_moto_kmin", 20)
    kmax = getattr(args, "genoma_in_moto_kmax", 40)
    if not (kmin is not None and kmax is not None and 0 <= kmin <= kmax):
        return f"--genoma-in-moto: serve 0 <= --genoma-in-moto-kmin <= --genoma-in-moto-kmax ({kmin}, {kmax})"
    if not (getattr(args, "zone_attaccante", False) and getattr(args, "metro_al_nodo", False)):
        return "--genoma-in-moto richiede --zone-attaccante e --metro-al-nodo (zone di P e di N dalla fotografia)"
    if getattr(args, "attack_from", "spawn") != "spawn":
        return "--genoma-in-moto richiede --attack-from spawn (k dell'innesco dalla prima nuvola del genoma)"
    if (getattr(args, "rampa_genoma", 1) or 1) > 1:
        return "--genoma-in-moto e --rampa-genoma R > 1 usano lo stesso canale: una sola delle due"
    if getattr(args, "g_banco", None):
        return "--genoma-in-moto non e' previsto con --g-banco"
    return None


def problema_capacita_d31(args, caps):
    """Isaac server commands required by the winner-trial, realignment and attitude options."""
    caps = set(caps or ())
    if getattr(args, "registra_assetto", False) and "assetto" not in caps:
        return "--registra-assetto: il server di Isaac non ha il comando assetto"
    tia = getattr(args, "twin_init_assetto", "azzerato") or "azzerato"
    if tia != "azzerato" and "set_state_assetto" not in caps:
        return f"--twin-init-assetto {tia}: il server di Isaac non ha set_state_assetto"
    if tia == "attaccante" and "assetto" not in caps:
        return "--twin-init-assetto attaccante: il server di Isaac non ha il comando assetto (assetto allo spawn)"
    if getattr(args, "riallinea_dopo_assestamento", False) and "save_nome" not in caps:
        return "--riallinea-dopo-assestamento: il server di Isaac non ha save_nome (istantanea con nome)"
    return None


def _g_banco_run(args, isaac, mola, node, out, wps):
    """--g-banco: rollouts of twin G from already saved states, without victim or search (banco_g).
    One rollout per (repetition, case, arm), in alternating order. Before each one the robot goes
    to the saved true pose of the window and the Isaac state is saved there, as at window start;
    then the rollout does restore and set_state on the arm target.

    With the extension keys in the spec (banco_g.usa_estensioni; without them, the base behaviour):
    metre and saved zones of the run sent to the node before every rollout with a genome (the spec
    file, not recomputed) and checked afterwards; arm attitude in set_state, attitude read after the
    restore, after set_state and at rollout end; branch outcome from the MOLA log; damage from the
    baseline and victim-twin comparisons per repetition. With the node launched with --salva-nuvole
    the rollout tag goes to the node before every rollout (also with the base spec).
    """
    import shutil

    class _IsaacBanco:
        """Isaac client of a rollout with the extensions: set_state with the arm attitude
        parameters; attitude read after the restore, after set_state and at rollout end if the
        server has the command. Everything else goes to the unchanged client.
        """

        def __init__(self, cli, params, leggi_assetto, posto=None):
            self._cli, self._params, self._leggi = cli, dict(params), bool(leggi_assetto)
            self.registro = []
            self._posto = posto      # restore of the named snapshot (realigned twin)

        def __getattr__(self, nome):
            return getattr(self._cli, nome)

        def leggi(self, fase):
            if self._leggi:
                r = self._cli.call("assetto")
                self.registro.append({"fase": fase, **{k: r.get(k) for k in (
                    "pos", "quat_wxyz", "roll", "pitch", "yaw", "lin", "ang", "playing")}})

        def restore(self):
            p = self._cli.restore(nome=self._posto) if self._posto else self._cli.restore()
            self.leggi("dopo_restore")
            return p

        def set_state(self, x, y, yaw, vx=0.0, vy=0.0, wz=0.0, zero_joints=True):
            p = self._cli.call("set_state", x=float(x), y=float(y), yaw=float(yaw), vx=float(vx), vy=float(vy),
                               wz=float(wz), zero_joints=bool(zero_joints), **self._params)["pose"]
            self.leggi("dopo_set_state")
            return p

    spec = banco_g.load_spec(args.g_banco)
    caps = isaac.caps()
    if "set_state" not in caps:
        print("ERRORE: --g-banco richiede il comando set_state del server (isaac/run_isaac_standalone.py)")
        return 2
    nuovo = banco_g.usa_estensioni(spec)
    why = banco_g.problema_capacita(spec, caps)
    if why:
        print(f"ERRORE: --g-banco: {why}")
        return 2
    stati = out / "stati"
    stati.mkdir(parents=True, exist_ok=True)
    casi = {}
    for c in spec["casi"]:
        # local copy: the per-rollout re-save is written next to the loaded file and the source run
        # directory stays intact
        dst = stati / f"{c['nome']}.molastate"
        shutil.copy2(c["state_path"], dst)
        sha = tv.file_sha256(str(dst))
        why = tv.g_state_problem(str(dst))
        if why or sha != c["state_sha256"]:
            print(f"ERRORE: caso {c['nome']}: {why or 'sha256 ' + sha[:12] + ' invece di ' + c['state_sha256'][:12]}")
            return 2
        casi[c["nome"]] = banco_g.shadow_state(c, dst, sha, os.path.getsize(dst),
                                               tv.molastate_header(str(dst))["stamp_tick"])
        print(f"  banco G, caso {c['nome']}: {c['state_path']} -> {dst} (sha256 {sha[:12]}, "
              f"stamp {casi[c['nome']]['state_stamp_tick']}), bracci "
              f"{', '.join(b['nome'] for b in c['bracci'])}")
    TWIN_STATE.update({"mode": "ombra_g", "g_root": args.g_overlay, "g_resave": args.g_resave,
                       "victim": None, "attacker_belief": None, "attacker_origin": None})
    # arm B: starting drift fixed for every rollout (None = reset)
    usa_deriva = banco_g.usa_deriva(spec)
    if usa_deriva and not node.set_twin_drift_init(None):
        print("ERRORE: bracci con drift_init: il nodo non conferma /attack/twin_drift_init "
              "(lanciato senza --banco-twin-drift?)")
        return 2
    # spec zones: copies in <out>/zone with the snapshot next to them (the node looks for it in the
    # zone file directory), reread with sha256 and hash
    zone_file = {}
    if banco_g.usa_zone(spec):
        dz = out / "zone"
        dz.mkdir(parents=True, exist_ok=True)
        for c in spec["casi"]:
            for b in c["bracci"]:
                z = b.get("zone")
                if not z:
                    continue
                try:
                    for src, sha in ((z["fotografia_path"], z["fotografia_sha256"]), (z["path"], z["sha256"])):
                        dst = dz / Path(src).name
                        if not dst.exists():
                            shutil.copy2(src, dst)
                        if tv.file_sha256(str(dst)) != sha:
                            raise za.ZoneNonValide("sha256_diverso", f"{dst} diverso da {src}")
                    za.carica_zone(dz / Path(z["path"]).name, z["sha256"], z["hash"])
                except (OSError, za.ZoneNonValide) as e:
                    print(f"ERRORE: caso {c['nome']}, braccio {b['nome']}: zone della specifica: {e}")
                    return 2
                zone_file[(c["nome"], b["nome"])] = dz / Path(z["path"]).name
        t0 = time.time()
        while node.node_zone_on is not True and time.time() - t0 < 10.0:
            node.spin(0.1)
        if node.node_zone_on is not True or not METRO["on"]:
            print("ERRORE: zone nella specifica: nodo senza --zone-attaccante o orchestratore senza --metro-al-nodo")
            return 2
    # spec ramp: zone file of the previous genome copied into <out>/zone and reread with sha256 and
    # hash (the node looks for it next to the snapshot)
    rampa_file = {}
    if banco_g.usa_rampa(spec):
        dz = out / "zone"
        dz.mkdir(parents=True, exist_ok=True)
        for c in spec["casi"]:
            for b in c["bracci"]:
                zp = (b.get("rampa") or {}).get("zone_prec")
                if not zp:
                    continue
                try:
                    for src, sha in ((zp["fotografia_path"], zp["fotografia_sha256"]), (zp["path"], zp["sha256"])):
                        dst = dz / Path(src).name
                        if not dst.exists():
                            shutil.copy2(src, dst)
                        if tv.file_sha256(str(dst)) != sha:
                            raise za.ZoneNonValide("sha256_diverso", f"{dst} diverso da {src}")
                    za.carica_zone(dz / Path(zp["path"]).name, zp["sha256"], zp["hash"])
                except (OSError, za.ZoneNonValide) as e:
                    print(f"ERRORE: caso {c['nome']}, braccio {b['nome']}: zone del genoma precedente: {e}")
                    return 2
                rampa_file[(c["nome"], b["nome"])] = {"path": str(dz / Path(zp["path"]).name),
                                                      "sha256": zp["sha256"], "hash": zp["hash"]}
        RAMPA["R"] = int(args.rampa_genoma)
        why = rampa_mismatch(node, RAMPA["R"])
        if why:
            print(f"ERRORE: --g-banco: {why}")
            return 2
    # on the bench the zones come from the saved spec file: the computation from the metre snapshot
    # (set_zone inside set_genome) and the check inside the rollout stay off; publishing and
    # checking are below
    zone_on = ZONE["on"]
    if zone_file:
        ZONE["on"] = False

    def esci(rc):
        ZONE["on"] = zone_on
        return rc

    # node --salva-nuvole: the state is read from the status (the orchestrator does not keep it);
    # without the option in the node no tag and no field in the record
    sn = {"visto": False, "stato": None}

    def _sn_cb(msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        sn["visto"] = True
        if isinstance(d.get("salva_nuvole"), dict):
            sn["stato"] = d["salva_nuvole"]
    _Str = None
    if getattr(node, "node", None) is not None:      # without an rclpy node (tests) no reading
        from std_msgs.msg import String as _Str
        node.node.create_subscription(_Str, "/attack/status", _sn_cb, 10)
        t0 = time.time()
        while not sn["visto"] and time.time() - t0 < 3.0:
            node.spin(0.1)
    sn_pub = None
    if sn["stato"] is not None:
        sn_pub = node.node.create_publisher(_Str, "/attack/salva_nuvole_tag", 10)
        print(f"  banco G: nuvole salvate dal nodo in {sn['stato'].get('dir')} (k < {sn['stato'].get('k')}), "
              "tag = rollout")
    elif not sn["visto"] and nuovo:
        print("  [nota] banco G: nessuno status del nodo in 3 s: tag delle nuvole salvate non mandati")
    # attitude at spawn (assetto esplicito_relativo): roll and pitch of the attacker MOLA are
    # relative to the robot attitude when its map was born, at the run start pose; measured here
    # with the same procedure as main (set_pose, play, 1 s), then another 4 s with the robot still
    assetto_spawn = None
    if banco_g.usa_assetto(spec, ("esplicito_relativo",)):
        sx, sy, syaw = (float(v) for v in args.start_pose.split(","))
        letture = []
        isaac.pause()
        isaac.set_pose(sx, sy, math.radians(syaw))
        isaac.play()
        for attesa, dopo in ((1.0, 1.0), (4.0, 5.0)):
            node.spin(attesa)
            isaac.pause()
            a_ = isaac.call("assetto")
            letture.append({"dopo_play_s": dopo, **{k: a_.get(k) for k in ("pos", "quat_wxyz", "roll", "pitch",
                                                                             "yaw", "lin", "ang")}})
            isaac.play()
        isaac.pause()
        u_ = letture[-1]
        assetto_spawn = {"posa_partenza": [sx, sy, syaw], "letture": letture, "usata": "dopo_play_s 5.0",
                         "roll": u_["roll"], "pitch": u_["pitch"], "yaw": u_["yaw"]}
        print(f"  banco G: assetto allo spawn in {sx},{sy},{syaw}: rollio {math.degrees(u_['roll']):+.4f} gradi, "
              f"beccheggio {math.degrees(u_['pitch']):+.4f} gradi (dopo 1 s: "
              f"{math.degrees(letture[0]['roll']):+.4f}, {math.degrees(letture[0]['pitch']):+.4f})")
    # baseline: damage per rollout and comparisons
    poly = p0 = u0 = None
    if nuovo and getattr(args, "damage_ref", None) == "baseline" and getattr(args, "baseline_path", None):
        try:
            poly = load_baseline(args.baseline_path)
        except ValueError as e:
            print(f"ERRORE: --g-banco: {e}")
            return esci(2)
        p0, u0 = banco_g.asse_baseline(poly)
    path = out / "banco_G.json"
    esiti = []
    confronti = []
    fatti = {}
    extra = {"assetto_spawn": assetto_spawn} if assetto_spawn is not None else {}

    def scrivi():
        with open(path, "w") as f:
            json.dump({"spec": str(args.g_banco), "esiti": esiti, **extra,
                       **({"confronti": confronti} if banco_g.usa_confronti(spec) else {})},
                      f, indent=1, default=str)
    # realignment: end-of-settling state of the arms a realigned arm restarts from, per (repetition,
    # case, arm)
    riallinei = {}
    for r, c, b in banco_g.schedule(spec):
        t = banco_g.tag(r, c, b)
        genome = b.get("genome")
        rial = b["init"] == "riallineato"
        mid = riallinei.get((r, c["nome"], b.get("riallinea_da"))) if rial else None
        if rial and ((mid or {}).get("stato") or {}).get("status") != "ok":
            # no end-of-settling state for the source arm: rollout not run
            rec = {"r": r, "caso": c["nome"], "braccio": b["nome"], "tag": t, "init": b["init"],
                   "status": "riallineo_mancante", "riallinea_da": b.get("riallinea_da"),
                   "riallineo": mid, "esito": banco_g.esito([]), "ramo": banco_g.esito_ramo("")}
            esiti.append(rec)
            scrivi()
            print(f"  {t}: riallineato non eseguito (stato di {b.get('riallinea_da')} assente o non salvato)")
            continue
        if usa_deriva and not node.set_twin_drift_init(b.get("drift_init")):
            print(f"ERRORE: {t}: il nodo non conferma la deriva iniziale {b.get('drift_init')}")
            return esci(2)
        pausa = banco_g.pausa_s(spec, r)
        if pausa > 0:
            # different pace: wait with Isaac paused and the chain on
            isaac.pause()
            node.spin(pausa)
        if rial:
            # pose and end-of-settling snapshot of the source arm (named slot): no set_pose or save
            # here
            isaac.pause()
            saved, saved_scene = tuple(mid["saved_odom"]), list(mid["saved_scene"])
        else:
            isaac.pause()
            isaac.set_pose(*c["saved_scene"])
            isaac.play()
            node.spin(1.0)
            saved = node.settled_pose(isaac)
            isaac.pause()
            saved_scene = isaac.save()
        tgt = banco_g.init_target(c, b, saved_scene)
        # vera and salvata_banco go through the rollout set_state like attacker_est (only the target
        # matters); ripristino without set_state
        init_r = b["init"] if b["init"] in ("attacker_est", "ombra_est") else (
            "attacker_est" if tgt["target_scene"] is not None else "true")
        TWIN_STATE.update({"shadow_state": mid["stato"] if rial else casi[c["nome"]], "saved": saved,
                           "init": init_r, "init_target": tgt, "g_env_extra": dict(b["mola_env"])})
        rec = {"r": r, "caso": c["nome"], "braccio": b["nome"], "tag": t, "init": b["init"],
               "target_scene": tgt["target_scene"], "mola_env": b["mola_env"],
               "genome": genome, "drift_init": b.get("drift_init"),
               "saved_bench": [float(v) for v in saved],
               "saved_scene_bench": [float(v) for v in saved_scene],
               "saved_err_cm": round(100 * math.hypot(saved[0] - c["saved_odom"][0],
                                                      saved[1] - c["saved_odom"][1]), 3),
               "saved_scene_err_cm": round(100 * math.hypot(saved_scene[0] - c["saved_scene"][0],
                                                            saved_scene[1] - c["saved_scene"][1]), 3)}
        if "pausa" in spec:
            rec["pausa_s"] = pausa
        isa = isaac
        if nuovo:
            try:
                params = banco_g.parametri_set_state(b.get("assetto"), None if assetto_spawn is None else (
                    assetto_spawn["roll"], assetto_spawn["pitch"], assetto_spawn["yaw"]))
            except ValueError as e:
                print(f"ERRORE: {t}: {e}")
                return esci(2)
            isa = _IsaacBanco(isaac, params, "assetto" in caps, posto=POSTO_RIALLINEO if rial else None)
            rec.update({"target_rule": tgt["rule"], "scarti": tgt.get("scarti"),
                        "assetto_braccio": b.get("assetto"), "assetto_set_state": params,
                        **({"assetto_variante": banco_g.VARIANTE_ASSETTO[(b.get("assetto") or {}).get("modo")]}
                           if (b.get("assetto") or {}).get("modo") in banco_g.VARIANTE_ASSETTO else {}),
                        "ripetizioni_braccio": banco_g.ripetizioni_braccio(spec, b)})
        # spec ramp (previous genome and its saved zones) to the node before metre, zones and
        # genome; not for the realigned arm (it starts after settling): message that applies to no
        # genome, as in the main flow
        if banco_g.usa_rampa(spec) and genome is not None and "rampa" in b:
            rp = b["rampa"]
            TWIN_STATE["window"] = int(c["metro"])
            RAMPA.update({"R": int(args.rampa_genoma), "metro": int(c["metro"]), "prec": rp["genoma_prec"],
                          "prec_sha": za.genoma_sha(np.asarray(rp["genoma_prec"], dtype=float)),
                          "zone_prec": {**rampa_file[(c["nome"], b["nome"])], "genoma_sha": None},
                          "finestra": {"messaggi": []}, "ultimo": {}, "sospesa": rial})
            try:
                node.manda_rampa("gemello", genome)
            except ErroreConfigurazioneZone as e:
                rec["rampa"] = {"messaggi": RAMPA["finestra"]["messaggi"]}
                esiti.append(rec)
                scrivi()
                print(f"ERRORE: {t}: {e}")
                return esci(2)
            finally:
                RAMPA["sospesa"] = False
            RAMPA["preinviata"] = {"gemello": za.genoma_sha(genome)}
            rec["rampa"] = {"genoma_prec_sha": RAMPA["prec_sha"], "zone_prec": RAMPA["zone_prec"],
                            "messaggio": RAMPA["ultimo"].get("gemello")}
        elif banco_g.usa_rampa(spec):
            # arm without ramp: no previous genome; if the node holds data that would apply to its
            # genome, set_genome cancels it (manda_rampa)
            RAMPA.update({"prec": None, "prec_sha": None, "sospesa": False, "preinviata": {},
                          "finestra": {"messaggi": []}})
        # metre and saved zones of the run to the node, confirmed by the status
        zrec = None
        if genome is not None and (c["nome"], b["nome"]) in zone_file:
            z = b["zone"]
            TWIN_STATE["window"] = int(c["metro"])
            ok_m = node.set_metro("gemello")
            pub, String = node._zone_pub("/attack/zone")
            ZONE["seq"] += 1
            seq = ZONE["seq"]
            m = String()
            m.data = json.dumps({"seq": seq, "flusso": "gemello", "path": str(zone_file[(c["nome"], b["nome"])]),
                                 "sha256": z["sha256"], "hash": z["hash"], "genoma_sha": z["genoma_sha"],
                                 "seme": z["seme"], "metro": z["metro"]})
            ok_z = node._publish_confirmed(
                pub, m, lambda: ((node.node_zone_ricevute or {}).get("gemello") or {}).get("seq") == seq,
                f"zone gemello w{c['metro']} (file della specifica)")
            ric = ((node.node_zone_ricevute or {}).get("gemello") or {}) if ok_z else {}
            app = (node.node_zone_applicate or {}).get("gemello") or {}
            zrec = {"file": str(zone_file[(c["nome"], b["nome"])]), "hash": z["hash"],
                    "genoma_sha": z["genoma_sha"], "seme": z["seme"], "metro": z["metro"],
                    "metro_confermato": bool(ok_m), "confermate": bool(ok_z), "stato_nodo": ric.get("stato"),
                    "n_ok0": int(app.get("n_ok") or 0), "n_err0": int(app.get("n_err") or 0)}
            if not ok_m or ric.get("stato") != "ok" or ric.get("hash") != z["hash"]:
                rec["zone"] = zrec
                esiti.append(rec)
                scrivi()
                print(f"ERRORE: {t}: metro o zone non confermati dal nodo (metro {bool(ok_m)}, zone "
                      f"{ric.get('stato') or 'nessuna conferma'})")
                return esci(2)
        snr = s_prima = None
        if sn_pub is not None:
            ms = _Str()
            ms.data = t
            ok_t = node._publish_confirmed(sn_pub, ms, lambda: (sn["stato"] or {}).get("tag") == t,
                                           f"tag delle nuvole salvate ({t})")
            s_prima = dict(sn["stato"] or {})
            snr = {"tag": t, "confermato": bool(ok_t), "dir": s_prima.get("dir"), "k": s_prima.get("k")}
        # realignment: the source arm saves at the end of settling; the realigned arm restarts from
        # there (twin state, believed frame, generator if the node offers it)
        kw_d31 = {}
        hook = None
        if b["nome"] in banco_g.bracci_riallineati(c):
            hook = RiallineoBanco(stati / f"{t}_riallineo", c["origin"])
            kw_d31["dopo_assestamento"] = hook
        if rial:
            kw_d31["assestamento_nuvole"] = 0
            gs = mid.get("generatore_salvato")
            rec.update({"riallinea_da": b["riallinea_da"], "tag_partenza": mid.get("tag_vittima"),
                        "riallinea_posa": banco_g.riallinea_posa(b),
                        "stato_caricato": {k_: mid["stato"].get(k_) for k_ in (
                            "state_path", "state_sha256", "state_stamp_tick", "pose6", "believed")},
                        "generatore_caricato": (node.stato_generatore("carica", POSTO_RIALLINEO)
                                                if gs is not None and gs.get("confermato") else None)})
            rec["generatore"] = ("trasferito" if (rec["generatore_caricato"] or {}).get("confermato") else
                                 "nodo_senza_stato_generatore" if gs is None else "non_confermato")
        frame_r = tuple(mid["stato"]["believed"]) if rial else tuple(c["believed"])
        # A, C, D: nominal, attack off, clean cloud like the window nominal; B: arm genome from the
        # first cloud (attack_from spawn)
        _, end, _, st, trav, frames = rollout(
            genome, isa, mola, node, list(wps), args.horizon, t,
            settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from,
            stop_on=args.stop_on, drift_reset_on=args.drift_reset_on, frame_start=frame_r,
            trace_path=(out / "traces" / f"{t}.csv") if args.trace else None, **kw_d31)
        if hook is not None:
            rec["riallineo"] = hook.rec
            riallinei[(r, c["nome"], b["nome"])] = hook.rec
        RAMPA["preinviata"] = {}
        if banco_g.usa_rampa(spec):
            rec.setdefault("rampa", {"messaggi": (RAMPA.get("finestra") or {}).get("messaggi")})
            rec["rampa"]["nodo"] = dict(((getattr(node, "node_rampa", None) or {}).get("gemello")) or {}) or None
        lp = getattr(mola, "log_path", None)
        text = Path(lp).read_text(errors="replace") if lp and Path(lp).is_file() else ""
        rec.update({"status": st, "travelled": trav, "frames": frames,
                    "end": [float(v) for v in end] if end is not None else None,
                    "mola_log": str(lp) if lp else None,
                    "esito": banco_g.esito(banco_g.icp_lines(text)), "fp": node.last_twin_fp})
        if nuovo:
            isa.leggi("fine_rollout")
            rec["assetto_letture"] = isa.registro
            rec["ramo"] = banco_g.esito_ramo(text)
            if hook is not None and hook.rec is not None:
                # victim branch from the first cloud after the save (comparison b)
                rec["ramo_dopo_riallineo"] = banco_g.esito_ramo(
                    text, stamp0=stamp_dopo((hook.rec.get("stato") or {}).get("pose_stamp")))
            if poly is not None:
                tr = list(getattr(node, "last_trace", None) or [])
                rec["danno_metro"] = wrec.danno_metro(true_path(tr), poly) if tr else None
                fatti[(r, c["nome"], b["nome"])] = (rec, banco_g.grandezze_traccia(tr, p0, u0))
        zona_ko = None
        if zrec is not None:
            # as zone_controllo: status counters up to 2 s if not moving
            def conta(zr=zrec):
                a_ = (node.node_zone_applicate or {}).get("gemello") or {}
                return a_, int(a_.get("n_ok") or 0) - zr["n_ok0"], int(a_.get("n_err") or 0) - zr["n_err0"]
            node.spin(0.3)
            app, n_ok, n_err = conta()
            t1 = time.time()
            while n_ok + n_err == 0 and (frames is None or frames > 0) and time.time() - t1 < 2.0:
                node.spin(0.1)
                app, n_ok, n_err = conta()
            zrec.update({"nuvole_ok": n_ok, "nuvole_err": n_err, "frames": frames, "stato": app.get("stato"),
                         "hash_applicato": app.get("hash") if n_ok > 0 else None,
                         "esito": banco_g.esito_zone(n_ok, n_err, app.get("hash"), zrec["hash"], frames)})
            rec["zone"] = zrec
            if zrec["esito"] in ("errore", "nessuna_zona_applicata"):
                zona_ko = (f"{t}: zone {zrec['esito']}: nuvole con zone {n_ok}, senza zone valide {n_err} "
                           f"({app.get('stato')}), nuvole uscite {frames}")
        if snr is not None:
            node.spin(0.2)
            s_dopo = dict(sn["stato"] or {})
            snr.update({k: int(s_dopo.get(k) or 0) - int(s_prima.get(k) or 0)
                        for k in ("salvati", "scartati_coda", "errori")})
            rec["salva_nuvole"] = snr
        esiti.append(rec)
        for cf in c.get("confronti") or []:
            kv, kg = (r, c["nome"], cf["vittima"]), (r, c["nome"], cf["gemello"])
            if b["nome"] in (cf["vittima"], cf["gemello"]) and kv in fatti and kg in fatti:
                (rv, gv), (rw, gw) = fatti[kv], fatti[kg]
                # (b): against the realigned arm the victim branch after the save applies
                ramo_v = (rv.get("ramo_dopo_riallineo") if rw.get("init") == "riallineato"
                          else rv.get("ramo"))
                # full comparison: incremental error (both on the twin stream) and damage only with
                # the victim arm damage measured
                tr_ = (lambda tg_: str(out / "traces" / f"{tg_}.csv") if args.trace else None)
                e1 = banco_g.e1_incrementale(tr_(rv["tag"]), rv.get("mola_log"), tr_(rw["tag"]), rw.get("mola_log"),
                                             CONTROLLO_ETA.get("frame_log") or getattr(args, "frame_log_nodo", None),
                                             False, False)
                cc = banco_g.confronto_coppia(gv, gw, ramo_v, rw.get("ramo"),
                                              rv.get("danno_metro"), rw.get("danno_metro"),
                                              e1=e1, dm_real=rv.get("danno_metro") is not None)
                confronti.append({"r": r, "caso": c["nome"], "confronto": cf["nome"], "vittima": cf["vittima"],
                                  "gemello": cf["gemello"], "tag_vittima": rv["tag"], "tag_gemello": rw["tag"], **cc})
                print(f"  confronto {cf['nome']} {c['nome']} r{r}: rami {cc['ramo_vittima']}/{cc['ramo_gemello']}, "
                      f"R1 {cc['R1_cm']} cm, laterale {cc['laterale_cm']} cm, danno {cc['danno_cm']} cm -> "
                      f"{'segue' if cc['segue'] else 'non segue'}")
        scrivi()
        e = rec["esito"]
        print(f"  {t}: {st}, posa salvata a {rec['saved_err_cm']} cm da quella della run, prime "
              f"{e['prime_n']} ICP buone {e['buone_prime_n']}, q1 {e['q1']} -> "
              f"{'passa' if e['passa'] else 'non passa'}")
        if nuovo:
            rm = rec["ramo"]
            print(f"    ramo {rm['classe']} (dy {rm['dy_cm']} cm da {rm['t0']}), nVoxels "
                  f"{[x['nvoxels'] for x in rm['prime'][:3]]}")
        if zona_ko:
            print(f"ERRORE: {zona_ko}")
            return esci(2)
    for c in spec["casi"]:
        for b in c["bracci"]:
            rr = [x["esito"]["passa"] for x in esiti if x["caso"] == c["nome"] and x["braccio"] == b["nome"]]
            print(f"  banco G {c['nome']} {b['nome']}: passa {sum(rr)}/{len(rr)}")
            if nuovo:
                cl = [x["ramo"]["classe"] for x in esiti if x["caso"] == c["nome"] and x["braccio"] == b["nome"]]
                print(f"    rami: {', '.join(f'{k} {cl.count(k)}' for k in sorted(set(cl)))}")
        for cf in c.get("confronti") or []:
            xs = [x for x in confronti if x["caso"] == c["nome"] and x["confronto"] == cf["nome"]]
            print(f"  banco G {c['nome']} confronto {cf['nome']} ({cf['vittima']} contro {cf['gemello']}): "
                  f"segue {sum(x['segue'] for x in xs)}/{len(xs)}, stesso ramo {sum(x['stesso_ramo'] for x in xs)}/{len(xs)}")
    if usa_deriva:
        node.set_twin_drift_init(None)
    return esci(0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", type=str, default="3,0",
                    help="bersaglio 'x,y' in metri relativi alla posa iniziale "
                         "(x in avanti, y a sinistra). Default '3,0'.")
    ap.add_argument("--horizon", type=float, default=1.0,
                    help="metri percorsi e misurati per finestra (default 1.0).")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="metri percorsi senza perturbazione prima di misurare "
                         "(default 0: assestamento a robot fermo con --settle-sec).")
    ap.add_argument("--settle-sec", type=float, default=5.0,
                    help="attesa massima (s) perche' la stima di MOLA si stabilizzi "
                         "prima di partire; si parte appena converge (default 5.0).")
    ap.add_argument("--initial-sigma", type=float, default=None,
                    help="valore iniziale della soglia adattiva di ICP (default "
                         "della pipeline 0.20). Con valori bassi sigma cade subito "
                         "al pavimento e ICP smette di accoppiare.")
    ap.add_argument("--catchup-sec", type=float, default=0.8,
                    help="secondi di attesa a robot fermo, a fine rollout, prima "
                         "di leggere la stima finale di MOLA (default 0.8): "
                         "compensa la latenza della stima rispetto al moto.")
    ap.add_argument("--passthrough", action="store_true",
                    help="nessun attacco: esegue solo i rollout nominali, per "
                         "verificare che il robot percorra il tratto dritto.")
    ap.add_argument("--nominal-reps", type=int, default=1,
                    help="in passthrough: quanti nominali per finestra dallo stesso "
                         "stato salvato (default 1). Con 2 o piu' lo scarto fra "
                         "nominali identici misura il pavimento di rumore per finestra.")
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--chamfer-max", type=float, default=5.0,
                    help="non usato nella selezione (si usa il rapporto "
                         "danno/percettibilita'); tenuto per compatibilita'.")
    ap.add_argument("--admissibility", choices=["stealth", "none"], default="stealth",
                    help="stealth (default): un candidato e' valido solo se sviamento "
                         "furtivo (robot in moto, rapporto stima/vero plausibile, "
                         "qualita' ICP sopra 0.20); robot fermo e perdita di posa sono "
                         "infattibili e riportati come DoS a parte. none: come prima.")
    ap.add_argument("--chamfer-budget", type=float, default=None,
                    help="budget di percettibilita' (cm): i candidati con Chamfer "
                         "media oltre questo valore sono inammissibili (esclusi dal "
                         "fronte e mai scelti, ma valutati e registrati). Default: "
                         "nessun budget.")
    ap.add_argument("--structural", choices=["net", "gross"], default=None,
                    help="forma attesa del termine strutturale di P nel nodo (perturbation_node.py "
                         "--structural): solo controllo e registro, il calcolo e' del nodo. Default: "
                         "nessun controllo; la forma riportata dal nodo va comunque in history.")
    ap.add_argument("--select", choices=["maxdamage", "ratio", "pareto_maxdamage"], default="maxdamage",
                    help="regola di selezione del genoma da applicare. maxdamage "
                         "(default): massimo danno fra gli ammissibili (entro il "
                         "budget di Chamfer), con rivalutazione dei top-K e scelta "
                         "sulla media (difesa dal rumore del rollout singolo). "
                         "ratio: argmax(danno/Chamfer) come nelle prime run "
                         "(vulnerabile ai candidati a Chamfer ~0). "
                         "pareto_maxdamage: fronte di Pareto (max danno, "
                         "min Chamfer) fra i candidati validi di tutta la finestra, applicato "
                         "il punto con il danno massimo (pareggi: Chamfer minore); nessuna "
                         "rivalutazione (--topk e --reeval ignorati) e nessun filtro sui "
                         "candidati lenti.")
    ap.add_argument("--select-only-valid", action="store_true",
                    help="con maxdamage o pareto_maxdamage vince solo un candidato valido: "
                         "Chamfer entro --chamfer-budget (senza budget la condizione "
                         "e' sempre vera), non untracked, non est_capped, senza guasto del gemello "
                         "(guasto_gemello, no_pose_from_new_mola, pose_stale), oltre ai filtri gia' in "
                         "vigore. Ogni candidato registra i motivi in selection_invalid_reasons. Se "
                         "nessuno e' valido la finestra segue la regola di nessun ammissibile. "
                         "Spento di default.")
    ap.add_argument("--topk", type=int, default=3,
                    help="con --select maxdamage, quanti candidati rivalutare (ignorato con pareto_maxdamage)")
    ap.add_argument("--topk-agg", choices=["mean", "min"], default="mean",
                    help="con --select maxdamage, come combinare danno di ricerca e di rivalutazione "
                         "di ciascun top-K: mean (media, come finora) oppure min (un candidato vale "
                         "quanto riesce a ripetere: un singolo valore spurio non lo fa vincere)")
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                    help="popolazione iniziale di ogni finestra (dopo la prima): meta' dal "
                         "fronte di Pareto della finestra precedente (sparsi sulla Chamfer; se "
                         "meno di N/2, con loro perturbazioni), meta' casuali. --no-warm-start "
                         "per la ricerca fredda di prima. Salva warm_start_from_prev in history.")
    ap.add_argument("--warm-front-after-reeval", action="store_true",
                    help="NSGA-III con --warm-start: il fronte passato alla finestra successiva si "
                         "calcola dopo le rivalutazioni per un riavvio dell'adattatore, sui soli "
                         "record validi (i misurati durante il degrado restano fuori). Spento "
                         "(default): calcolato subito dopo la ricerca, come in origine.")
    ap.add_argument("--reeval", action="store_true",
                    help="(solo --select ratio) rivaluta il genoma scelto con un rollout in piu'")
    ap.add_argument("--pause-sec", type=float, default=0.0,
                    help="senza attacco, con vittima continua: sosta a robot fermo (simulazione in pausa, flusso "
                         "vittima chiuso) fra il nominale del twin e il tratto della vittima, per riprodurre le soste "
                         "della ricerca a parita' di orizzonte.")
    ap.add_argument("--twin-gap0-max", type=float, default=0.0,
                    help="guardia sui twin non allineati alla partenza (cm): un candidato il cui twin parte con "
                         "scarto allineato oltre questo valore e' contato ed escluso dalla selezione (come i lenti); "
                         "la rivalutazione con twin non allineato si ripete una volta. 0 = disattivata")
    ap.add_argument("--stall-policy", choices=["abort", "tolerate"], default="abort",
                    help="stallo di Isaac: 'abort' chiude la run (come finora); 'tolerate' attende che il ciclo "
                         "riparta (max 60 s) e rifa' il rollout del twin o il rigioco colpito, registrando l'evento; "
                         "stalli nel tratto applicato, Isaac che non riparte o piu' di 3 stalli in una finestra chiudono comunque la run")
    ap.add_argument("--replay-guard-cm", type=float, default=5.0,
                    help="guardia per consenso sul rigioco al twin (--twin-state victim): stima a fine rigioco oltre "
                         "questo scarto dalla mediana dei rigiochi della finestra = rigioco rifatto una volta, poi candidato "
                         "escluso dalla selezione; il primo rigioco di ogni finestra e' confermato da un secondo. 0 = disattivata")
    ap.add_argument("--random-sparse", action=argparse.BooleanOptionalAction, default=False,
                    help="ricerca random: ogni candidato ha un numero casuale di geni attivi (1..n) con valori "
                         "casuali, gli altri spenti (-1 per le intensita', 0 per le direzioni).")
    ap.add_argument("--window-rng", choices=["sum", "seedseq"], default="sum",
                    help="ricerca random: generatore di ogni finestra. 'sum' = default_rng(seed + k), come le prime "
                         "run (con il seme 1000 + ripetizione le ripetizioni condividono i genomi sfasati di una "
                         "finestra); 'seedseq' = SeedSequence([seed, k]), flussi indipendenti per ripetizione e finestra.")
    ap.add_argument("--sampling", choices=list(wrec.SAMPLING_MODES), default="legacy",
                    help="distribuzione dei genomi casuali. legacy (default): come in origine "
                         "(random e popolazione iniziale di NSGA-III sparsi solo con --random-sparse; "
                         "meta' casuale del warm start sempre sparsa). uniform: ogni gene uniforme in "
                         "[-1, 1] per ricerca casuale, popolazione iniziale e meta' casuale del warm start. "
                         "sparse: lo stesso campionamento sparso per tutti e tre.")
    ap.add_argument("--random-first-admissible", type=int, default=0, metavar="K",
                    help="--search random: random singolo con budget. Genomi estratti in sequenza "
                         "(stesso generatore e distribuzione della ricerca casuale), ognuno valutato nel "
                         "gemello; si applica il primo valido per la selezione (candidato_valido: esito "
                         "ok/arrived, ammissibile, entro --chamfer-budget, gemello allineato e non filtrato), "
                         "al piu' K tentativi. Nessuno valido: nessun genoma applicato (no_admissible). "
                         "Se un riavvio dell'adattatore invalida il valido, la ricerca riprende con i "
                         "tentativi rimasti. Richiede --select pareto_maxdamage. "
                         "0 (default) = ricerca casuale pop x gen.")
    ap.add_argument("--origin-from-estimate", action="store_true",
                    help="origine del frame della vittima = posa vera composta con l'inversa della "
                         "stima di MOLA nello stesso istante, e origine propria per la MOLA "
                         "dell'attaccante. Spento (default): origine = posa vera a fine "
                         "assestamento, uguale per vittima e attaccante, come in origine. "
                         "L'offset delle due regole e' registrato in history (run_origin).")
    ap.add_argument("--fail-rc-on-no-pose", action="store_true",
                    help="se in qualche finestra la vittima resta senza posa (victim_no_pose, "
                         "victim_crash) l'orchestratore esce con codice 4 invece di 0 (lo stallo "
                         "di Isaac resta 3). Spento di default.")
    ap.add_argument("--chiudi-attacco-scoperto", action="store_true",
                    help="se la vittima perde la posa (victim_no_pose: "
                         "nessuna posa entro 15 s dalla riapertura; victim_crash; pose_stale: "
                         "nessuna posa nuova per piu' di 3 s durante il tratto) la run si chiude con "
                         "run_end attacco_scoperto (finestra, istante, ultima posa), separata dai "
                         "successi. Con pose_stale nel tratto applicato la finestra non conta come "
                         "danno; le finestre precedenti restano come sono. Spento di default.")
    ap.add_argument("--controllo-catena", action="store_true",
                    help="con --chiudi-attacco-scoperto: prima di chiudere "
                         "la run come attacco scoperto si controlla la catena della finestra (Isaac, "
                         "adattatore, nuvole consegnate alla vittima, pose della vittima ricevute, tempi "
                         "di MOLA della vittima, perturbation_node). Con un guasto la run si chiude con "
                         "run_end guasto_catena, file da_rifare.json nella cartella e codice 2. "
                         "Spento di default.")
    ap.add_argument("--rallentamento-mola", choices=wrec.RALLENTAMENTO_MOLA, default="guasto",
                    help="con --controllo-catena, se l'unico segnale e' il p99 di time_onLidar della "
                         "vittima oltre 200 ms e Isaac, nuvole alla vittima e nodo sono ok "
                         "(rallentamento_mola nel record): guasto = guasto della catena (v3, default); "
                         "scoperto = non e' guasto, decide la regola dell'attacco scoperto; "
                         "risultato = il tempo di MOLA della "
                         "vittima non e' mai guasto, qualunque sia l'esito degli altri controlli: "
                         "resta fra i segnali e nel record.")
    ap.add_argument("--dds-profile", type=str, default=None, metavar="XML",
                    help="profilo XML di Fast DDS: "
                         "FASTRTPS_DEFAULT_PROFILES_FILE con il percorso assoluto per "
                         "l'orchestratore e tutte le MOLA (vittima, ombra, attaccante, gemelli). "
                         "Errore se il file manca o se nella cartella di lavoro c'e' "
                         "DEFAULT_FASTRTPS_PROFILES.xml; percorso e md5 in dds_profilo.json. "
                         "Spento di default: le MOLA partono senza la variabile, come prima.")
    ap.add_argument("--registri-dal-vivo", action="store_true",
                    help="registri dal vivo in <out>/registri_dal_vivo: pose.csv (ogni posa di "
                         "vittima e gemello: stamp, source_timestamp DDS, arrivo) e "
                         "controllore.csv (ogni ciclo: posa usata e sua eta', stima, comando, "
                         "pubblicazione, motivo). Spento di default: nessun file.")
    ap.add_argument("--origine-coerente-vittima", action="store_true",
                    help="nel tratto della vittima l'origine della stima "
                         "(est_origin) e il frame di controllo si prendono dalla stessa lettura di est_raw, "
                         "subito prima della guida, come fa il gemello. Senza l'opzione l'origine e' letta "
                         "prima dell'attesa della posa vera e del riallineo, il frame dopo: se MOLA salta in "
                         "mezzo la vittima si crede spostata (s1001 nsga3_senzabudget w3: 30 cm). Nel record "
                         "della vittima: frame_coerenza con lo spostamento dell'origine. Spento di default.")
    ap.add_argument("--controllo-ombra-viva", action="store_true",
                    help="durante tutta la run (ricerca compresa) "
                         "si controlla ogni 0,5 s che il processo MOLA dell'ombra, una volta avviato, "
                         "sia vivo. Se muore la run si ferma subito: history con status ombra_persa, "
                         "da_rifare.json (guasto_catena, causa ombra_persa) e la coda la ripete. "
                         "Spento di default.")
    ap.add_argument("--genoma-nullo-se-nessun-ammissibile", action="store_true",
                    help="quando in una finestra non c'e' un genoma "
                         "ammissibile (random singolo dopo K tentativi senza validi, NSGA-III e ricerca "
                         "casuale senza candidati ammissibili) si applica il genoma nullo (geni di direzione "
                         "a 0, gli altri a -1) per la strada normale del vincitore: la vittima percorre il "
                         "metro, che conta fra le finestre guidate, e la finestra dopo riprova. Con "
                         "--victim-never-clean sostituisce la trattenuta. La deriva accumulata (drift keep) "
                         "resta applicata dal generatore. Nella history: genoma_nullo e genoma_nullo_dettagli "
                         "(motivo, tentativi, |A| della vittima dallo status del nodo, P misurato). "
                         "Spento di default.")
    ap.add_argument("--controllo-eta", type=float, default=None, metavar="S",
                    help="eta' della nuvola alla presa di MOLA per ogni rollout (definizione della "
                         "prova decisiva): oltre S secondi, "
                         "o con nuvole del tratto mai prese, guasto della catena. Il rollout del "
                         "gemello si rifa' una volta (al secondo guasto la run si ferma come per uno "
                         "stallo di Isaac), il tratto della vittima ferma la run. Misure in "
                         "logs/controllo_eta.jsonl e nel campo eta_nuvola dei candidati. Richiede "
                         "--frame-log-nodo e, con la vittima continua, --victim-shadow-env (log a "
                         "righe). Spento di default.")
    ap.add_argument("--frame-log-nodo", type=str, default=None, metavar="CSV",
                    help="registro per frame del perturbation_node (--frame-log del nodo): con "
                         "--controllo-catena vi si cercano i buchi oltre 1 s nelle nuvole "
                         "consegnate alla vittima, anche in coda, prima di chiudere la run. Senza, "
                         "quel controllo resta non determinato.")
    ap.add_argument("--mola-se-ne-accorge", action="store_true",
                    help="e' non ammissibile il candidato con cui MOLA del "
                         "gemello crasha (mola_crash) o non pubblica una posa nuova per piu' di "
                         "--accorge-senza-posa-s (righe printf del suo log, serve "
                         "MOLA_LO_DEBUG_ICP_QUALITY=1; senza righe e senza crash l'esito resta non "
                         "determinato e il candidato non e' escluso). Le ICP rifiutate di fila si "
                         "registrano soltanto. Motivo mola_se_ne_accorge. Spento di default "
                         "(acceso da run_campaign.sh con SERIE_B=1).")
    ap.add_argument("--incompleti-regola", action="store_true",
                    help="i candidati con esito timeout o "
                         "stopped_early si classificano per causa (campi incompleto e "
                         "incompleto_classe nella history). (a) guasto_catena, solo segnali "
                         "indipendenti dal genoma (stallo di Isaac, eventi dell'adattatore, buco "
                         ">= 1 s nelle nuvole in ingresso al nodo da --frame-log-nodo): misura non "
                         "contata e candidato rifatto una volta; (b) mola_perde_posa (regola 8): non "
                         "ammissibile; calcolo (calc_ok falso): non ammissibile, non rifatto; (c) "
                         "attacco_valido: valido, danno = distanza della posa vera a fine rollout "
                         "dalla posa della baseline a fine finestra (punti_fine_finestra_medi nel "
                         "JSON accanto a --baseline-path). Richiede --mola-se-ne-accorge. Spento di "
                         "default (acceso da run_campaign.sh con SERIE_B=1 nei bracci con attacco).")
    ap.add_argument("--metro-al-nodo", action="store_true",
                    help="versione I degli operatori (nodo con --operator-mode insieme): prima di ogni "
                         "genoma del gemello e della vittima manda al nodo il numero del metro (finestra "
                         "corrente, da 0) su /attack/metro e ne attende la conferma dallo status; le "
                         "zone si scelgono con il seme del genoma e questo numero. I metri non "
                         "confermati vanno nella history (metro_non_confermati). Spento di default "
                         "(acceso da run_campaign.sh con OPERATOR_MODE=insieme).")
    ap.add_argument("--rampa-genoma", type=int, default=1, metavar="R",
                    help="rampa A3 al genoma nuovo (nodo con lo stesso --rampa-genoma): a inizio metro il genoma "
                         "precedente P (l'ultimo applicato alla vittima) e le sue zone sulla fotografia del metro; "
                         "prima di ogni genoma nuovo (vittima e ogni candidato del gemello: ricerca, rivalutazioni, "
                         "prove D-31b, braccio fisso) un messaggio su /attack/rampa con conferma. Nessun messaggio "
                         "al metro 0, nel nominale e al gemello riallineato di D-31c. Richiede --zone-attaccante, "
                         "--metro-al-nodo, --attack-from spawn e R <= assestamento - 8 (--settle-clouds e "
                         "--reopen-clouds). 1 (default) = spenta")
    ap.add_argument("--genoma-in-moto", type=float, default=0.0, metavar="S_CM",
                    help="variante E1, genoma vecchio fino al moto (nodo con gli stessi --genoma-in-moto, "
                         "--genoma-in-moto-kmin e --genoma-in-moto-kmax): a inizio metro il genoma precedente P "
                         "(l'ultimo applicato alla vittima) e le sue zone sulla fotografia del metro, mandati prima "
                         "di ogni genoma nuovo su /attack/rampa come per la rampa; il nodo applica P da fermo e il "
                         "genoma nuovo dalla prima nuvola in moto dell'inseguitore dell'attaccante (spostamento "
                         ">= S_CM cm). Al metro 0 il solo genoma nuovo. Record genoma_in_moto per finestra in "
                         "history. Richiede --zone-attaccante, --metro-al-nodo, --attack-from spawn; non con "
                         "--rampa-genoma R > 1. 0 (default) = spento")
    ap.add_argument("--genoma-in-moto-kmin", type=int, default=20, metavar="K",
                    help="con --genoma-in-moto: prima nuvola (k) in cui il genoma nuovo puo' entrare (default 20)")
    ap.add_argument("--genoma-in-moto-kmax", type=int, default=40, metavar="K",
                    help="con --genoma-in-moto: nuvola (k) in cui il genoma nuovo entra comunque (default 40)")
    ap.add_argument("--chamfer-per-fase", action="store_true",
                    help="Chamfer per flusso e per fase P/N (nodo con --chamfer-per-fase o --genoma-in-moto): a fine "
                         "di ogni rollout con genoma media e numero dei valori calcolati dal nodo per fase, nel record "
                         "chamfer_fasi della finestra in history (vittima, w<k>_e<i>, w<k>_riallineato, ...); senza "
                         "gradino tutte le nuvole sono di fase N. Acceso da --genoma-in-moto. Spento di default")
    ap.add_argument("--zone-attaccante", action="store_true",
                    help="serie B: a inizio di ogni metro fotografia dell'attaccante (ultima nuvola pulita "
                         "e posa della sua MOLA, <out>/zone/w<k>_fotografia.npz); per ogni genoma zone "
                         "calcolate solo da quella fotografia e dal seme, salvate in <out>/zone/w<k>_<sha>.npz "
                         "e mandate al nodo su /attack/zone prima del genoma. Fotografia o zone non valide, "
                         "non confermate o non applicate: finestra errore_configurazione e run ferma "
                         "(codice 2). Richiede --metro-al-nodo, la MOLA dell'attaccante e il nodo con "
                         "--zone-attaccante. Spento di default.")
    ap.add_argument("--danno-metro", action="store_true",
                    help="registra per ogni metro (finestra) le grandezze del danno rispetto alla "
                         "baseline per ogni candidato (danno_metro), per il vincitore e per la "
                         "vittima, la Chamfer del tratto applicato e a fine run il riepilogo "
                         "danno_run (definizioni in window_record.DANNO_METRO_DEFINITIONS e "
                         "DANNO_RUN_DEFINITIONS). Solo registrazione: fitness e selezione invariate. "
                         "Richiede --damage-ref baseline. Spento di default (acceso da "
                         "run_campaign.sh con SERIE_B=1).")
    ap.add_argument("--accorge-icp-rifiutate", type=int, default=wrec.ACCORGE_ICP_RIFIUTATE,
                    help="con --mola-se-ne-accorge: soglia delle ICP rifiutate di fila, solo "
                         "registrata nel record (rifiutate_oltre_soglia), non esclude (default %(default)s)")
    ap.add_argument("--accorge-senza-posa-s", type=float, default=wrec.ACCORGE_SENZA_POSA_S,
                    help="con --mola-se-ne-accorge: secondi senza posa oltre cui il candidato e' "
                         "non ammissibile (default %(default)s)")
    ap.add_argument("--twin-filter", choices=list(wrec.TWIN_FILTERS), default="none",
                    help="esclusione dalla selezione dei gemelli anomali, dal log di MOLA del gemello "
                         "(serve MOLA_LO_DEBUG_ICP_QUALITY=1). am2: prima ICP entro 0.25 s dall'ultimo "
                         "scan del gemello precedente; al: MaxIterations o q < 0.9 nei primi 0.2 s; "
                         "am2_al: l'una o l'altra. none (default): nessuna esclusione. I flag sono "
                         "registrati per ogni candidato in ogni caso.")
    ap.add_argument("--moltiplicatore-limiti", type=float, default=1.0, metavar="K",
                    help="studio dei limiti: moltiplica per K >= 1 i limiti delle famiglie di "
                         "--famiglie-limiti nelle zone dell'attaccante e nelle stampe; deve essere uguale a "
                         "quello del perturbation_node (status moltiplicatori_limiti), altrimenti "
                         "errore_configurazione. History: moltiplicatori_limiti. 1 (default) = spento.")
    ap.add_argument("--famiglie-limiti", type=str, default=None, metavar="LISTA",
                    help="famiglie di --moltiplicatore-limiti separate da virgole: deriva, rumore, geometria, "
                         "strutturale, dropout (default tutte)")
    ap.add_argument("--blocca-gene", action="append", default=None, metavar="I=-1",
                    help="studio dei limiti: gene I fissato a -1 in tutti i candidati (campionamento iniziale, "
                         "partenza calda, operatori di NSGA-III, ricerca casuale e random singolo): il gene esce "
                         "dallo spazio di ricerca e l'espansione del genoma lo mette a -1, come i geni inattivi "
                         "dei gruppi. Solo il valore -1; ripetibile (es. --blocca-gene 14=-1). History: "
                         "geni_bloccati e active_genes. Spento di default.")
    ap.add_argument("--fixed-genome", type=str, default=None,
                    help="--search fixed: genoma applicato a ogni orizzonte senza ricerca (17 valori separati da virgola).")
    ap.add_argument("--fixed-genome-seq", type=str, default=None,
                    help="--search fixed: un genoma per finestra (prova 3.7c). File JSON con una lista, oppure voci "
                         "separate da ';'. Ogni voce: lista di 17 valori, 17 valori separati da virgola, o "
                         "'history.json:finestra[:candidato]' come --repeat-genome. Nella finestra k la voce k, "
                         "l'ultima per le successive. Alternativa a --fixed-genome. Default: spenta.")
    ap.add_argument("--mola-stop-signal", choices=["term", "int"], default="term",
                    help="segnale per chiudere MOLA: term (default) o int (il solo con un "
                         "gestore in mola-cli). Stessa attesa massima di 12 s; il tempo d'uscita misurato va in "
                         "history per ogni rollout (mola_stop).")
    ap.add_argument("--victim-mola-env", action="append", default=[], metavar="K=V",
                    help="variabile d'ambiente per la sola MOLA della vittima (ripetibile), per esempio "
                         "MOLA_VERBOSITY_MOLA_LO=DEBUG. Default: nessuna.")
    ap.add_argument("--victim-shadow-env", action="store_true",
                    help="da' alla MOLA della vittima lo stesso ambiente dell'ombra (shadow_extra_env: "
                         "MOLA_LO_DEBUG_ICP_QUALITY=1, libstdbuf accodata a LD_PRELOAD ereditato, _STDBUF_O=L); "
                         "con --shadow-check l'ambiente risulta identico. Spento di default: ambiente della "
                         "vittima come prima.")
    ap.add_argument("--mola-binario-g1", action="store_true",
                    help="tutte le MOLA (vittima, ombra, attaccante, gemelli) con il binario dell'overlay "
                         "--g-overlay: le stesse LD_LIBRARY_PATH, MOLA_MODULES_LIB_PATH e MOLA_FULL_STATE=1 "
                         "dell'ombra G (solo variabili d'ambiente, YAML e launch invariati). Spento di "
                         "default: vittima e attaccante su /opt come prima.")
    ap.add_argument("--victim-never-clean", action="store_true",
                    help="vittima e ombra non ricevono mai una nuvola pulita (nodo con la stessa opzione): "
                         "avvio rimandato all'applicazione del primo genoma (--victim-start w0) con l'attacco "
                         "acceso dalla prima nuvola, code di finestra pulite scartate dal nodo, nessuna guida "
                         "della vittima senza genoma (finestra marcata victim_held, robot fermo nello stato "
                         "salvato). Implica --settle-clouds 20 e --reopen-clouds 8 se non dati; rifiuta "
                         "--attack-from motion, --warmup, --passthrough, --map-gate-test. Spento di default.")
    ap.add_argument("--victim-start", choices=["run", "w0"], default="run",
                    help="avvio della MOLA vittima (e dell'ombra): run (default, a inizio run con l'attacco "
                         "spento, come prima) o w0 (alla prima applicazione, a flusso chiuso, poi apertura con "
                         "genoma e attacco gia' confermati; a inizio run parte solo la MOLA dell'attaccante). "
                         "Nelle finestre prima dell'avvio i gemelli partono da zero (stato dell'ombra assente).")
    ap.add_argument("--settle-clouds", type=int, default=None,
                    help="assestamento da fermo contato in nuvole consegnate dal nodo, dal play o "
                         "dall'apertura del flusso alla guida, uguale per gemelli, nominale e vittima al suo "
                         "avvio (20 = 2 s a 10 Hz). Sostituisce --settle-sec e la durata copiata dal nominale "
                         "nei rollout. Default: spento (0), salvo --victim-never-clean (20).")
    ap.add_argument("--reopen-clouds", type=int, default=None,
                    help="con --settle-clouds: nuvole da fermo alla riapertura della vittima gia' avviata e "
                         "per i gemelli delle stesse finestre (0 = come --settle-clouds). Default: 0, salvo "
                         "--victim-never-clean (8, valore provvisorio da fissare al banco).")
    ap.add_argument("--adapter-events", type=str, default=None,
                    help="file JSONL degli eventi dell'adattatore (t_prev_clean, t_detect, t_restart): i candidati "
                         "misurati fra l'ultimo controllo pulito e il riavvio vengono rivalutati e non possono vincere.")
    ap.add_argument("--search", choices=["nsga3", "random", "gaussian", "fixed"], default="nsga3",
                    help="nsga3 (default); random = pop x gen genomi casuali a pari budget; "
                         "gaussian = nessuna ricerca, un rollout col rumore isotropo del "
                         "nodo (lanciare perturbation_node con --gaussian-sigma)")
    ap.add_argument("--goals", type=str, default=None,
                    help='waypoint multipli relativi alla partenza, es. "5,0; 5,2.5; 0,2.5"; '
                         'ha precedenza su --goal')
    ap.add_argument("--wp-tol", type=float, default=0.5,
                    help="distanza vera entro cui un waypoint e' considerato raggiunto (m)")
    ap.add_argument("--extra-windows", type=int, default=3,
                    help="finestre concesse oltre la stima prima di fermare il task")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="numero massimo di finestre (default: nessun limite); per le "
                         "run di caratterizzazione su una sola finestra.")
    ap.add_argument("--goal-mode", choices=["waypoints", "direction"], default="waypoints",
                    help="waypoints (default): bersagli con tolleranza d'arrivo (arrived) e fine "
                         "della run a --wp-tol dalla posa vera. direction: il bersaglio da' solo la "
                         "direzione; il follower insegue la carota a --direction-lookahead metri "
                         "sulla retta partenza-bersaglio, niente arrived ne' --wp-tol, ogni finestra "
                         "si chiude con la rampa sulla stima e la run dura --max-windows finestre "
                         "guidate (obbligatorio; le finestre victim_held non contano, al piu' "
                         "--extra-windows). Richiede --stop-on estimate, --window-stop ramp e un "
                         "solo bersaglio.")
    ap.add_argument("--direction-lookahead", type=float, default=DIRECTION_LOOKAHEAD,
                    help="con --goal-mode direction: distanza della carota oltre la proiezione "
                         "della stima sulla retta (m, default 5.0; almeno max_speed/k_lin del "
                         "follower, 1.0 m, perche' non rallenti).")
    ap.add_argument("--max-proc-ms", type=float, default=100.0,
                    help="vincolo del modello di minaccia: un candidato il cui tempo di "
                         "calcolo medio per nuvola supera questo valore (ms; default 100 = "
                         "periodo fra due scansioni a 10 Hz) e' inammissibile, perche' "
                         "l'attaccante in linea non puo' applicarlo senza ritardare il flusso.")
    ap.add_argument("--window-cap-factor", type=float, default=None,
                    help="watchdog: se una finestra dura oltre questo multiplo del tempo "
                         "della prima finestra completata, la ricerca si interrompe e la "
                         "finestra prosegue coi candidati gia' valutati (o avanza col "
                         "nominale se nessuno e' ammissibile). Default: nessun tetto.")
    ap.add_argument("--start-pose", type=str, default=None,
                    help='posa iniziale assoluta "x,y,yaw_deg": il robot viene riportato '
                         'li\' prima di partire (server isaac_rollout_server.py aggiornato)')
    ap.add_argument("--genome-group", type=int, choices=[1, 2, 3], default=3,
                    help="gruppo di geni attivi: 1 = rumore per punto e dropout (6), "
                         "2 = + operatori sulla geometria dello scan (15), "
                         "3 = tutti (17, default)")
    ap.add_argument("--discard-policy", choices=["infra", "legacy"], default="infra",
                    help="regola degli scarti. infra (default): si scarta solo per "
                         "guasti d'infrastruttura (isaac_stall, mola_not_ready, no_pose, "
                         "mola_crash) e quando il nominale della finestra e' anomalo; gli "
                         "altri esiti (timeout, percorso corto, tracking perso) contano "
                         "come danno con path_deviation. legacy: ogni esito diverso da "
                         "ok/arrived e' scartato (regola fino al 30/09).")
    ap.add_argument("--untracked", choices=["discard", "nominal", "damage"], default="discard",
                    help="(solo con --discard-policy legacy) rollout con tracking perso: "
                         "discard, nominal o damage")
    ap.add_argument("--attack-from", choices=["spawn", "motion"], default="spawn",
                    help="quando accendere la perturbazione: dall'avvio di MOLA "
                         "(default, assestamento incluso: il nodo compromesso e' sempre "
                         "attivo nel threat model) o solo all'inizio del tratto misurato "
                         "(motion)")
    ap.add_argument("--true-pose", choices=["ros", "file"], default="ros",
                    help="sorgente della posa vera: /chassis/odom (default) o il "
                         "server via file; la fisica e' la stessa")
    ap.add_argument("--objective", choices=["damage", "minq"], default="damage",
                    help="V4: 'damage' massimizza la deviazione (default); 'minq' cerca il genoma che abbassa di piu' la "
                         "qualita' ICP minima del rollout entro il budget (verifica se il gate q<0.20 e' raggiungibile da un disturbo ottimizzato)")
    # Default 'internal': f2'' (internal MOLA perceptibility) is the second objective fixed for the
    # continuous-victim campaign. Chamfer stays only a logged metric, unless --chamfer-budget is
    # explicitly set as a constraint.
    ap.add_argument("--perceptibility", choices=["chamfer", "internal"], default="internal",
                    help="secondo obiettivo di NSGA-III: 'internal' (percettibilita' interna: calo di qualita' ICP e "
                         "iterazioni in eccesso del twin rispetto al nominale della finestra, normalizzati; default, "
                         "obiettivo bloccato) oppure 'chamfer' (distanza geometrica); un eventuale --chamfer-budget "
                         "resta un vincolo in entrambi i casi")
    ap.add_argument("--f2-budget", type=float, default=None,
                    help="passata 4: budget della percettibilita' interna (f2''): un candidato con f2'' oltre questo valore "
                         "e' contato ma escluso dalla selezione (ammissibilita'), come il budget di Chamfer; la selezione "
                         "sceglie il massimo danno fra i candidati con Chamfer <= budget e f2'' <= f2-budget")
    ap.add_argument("--f2-q-scale", type=float, default=0.0706,
                    help="scala del termine di qualita' della percettibilita' interna (p95 della parte positiva "
                         "sulle nulle dei twin, rettilineo rip. 1-2)")
    ap.add_argument("--f2-it-scale", type=float, default=0.840,
                    help="scala del termine delle iterazioni (p95 della nulla, iterazioni per scansione)")
    ap.add_argument("--damage", choices=["mean", "max", "end"], default="mean",
                    help="deviazione usata nel danno: media lungo il tratto a "
                         "parita' di distanza percorsa (default), massimo lungo "
                         "il tratto, o sola posa finale (come nelle run fino a "
                         "attack_v5)")
    ap.add_argument("--dir-weight", type=float, default=0.0,
                    help="peso della componente direzionale nel danno, cioe' "
                         "quanto conta allontanare il robot dal bersaglio finale "
                         "oltre alla deviazione dal nominale. Default 0 (solo "
                         "deviazione): sul rettilineo i due termini coincidono e "
                         "la paralisi veniva contata due volte (prime run: "
                         "1.0; i termini sono salvati separatamente).")
    ap.add_argument("--damage-ref", choices=["window", "baseline"], default="window",
                    help="riferimento del danno. window (default): il nominale "
                         "della stessa finestra, a parita' di distanza percorsa. "
                         "baseline: distanza di ogni posa vera del rollout dal "
                         "punto piu' vicino della polilinea di --baseline-path "
                         "(oltre gli estremi conta la distanza dall'estremo); il "
                         "termine direzionale non entra nel danno (registrato 0) e "
                         "il danno per finestra resta nei campi *_window.")
    ap.add_argument("--baseline-path", type=str, default=None,
                    help="CSV della polilinea di riferimento per --damage-ref "
                         "baseline: intestazione s_m,x_m,y_m, righe ordinate per s "
                         "crescente, x e y nel frame odom della run (le run partono "
                         "dalla stessa posa di spawn); righe che iniziano con '#' "
                         "ignorate.")
    ap.add_argument("--victim", choices=["continuous", "restart"], default="continuous",
                    help="SLAM della vittima sul percorso applicato: continuous "
                         "(default) una sola MOLA mai riavviata, con mappa e soglia "
                         "adattiva a regime, alimentata solo nei tratti applicati; "
                         "restart un twin riavviato a ogni tratto, come nelle prime "
                         "run. La ricerca usa sempre twin riavviati.")
    ap.add_argument("--twin-align", choices=["attacker", "true"], default="attacker",
                    help="da dove partono i twin della ricerca a ogni finestra: dalla "
                         "posa stimata dalla MOLA dell'attaccante (default; una MOLA "
                         "continua sulla nuvola pulita, namespace /attacker) o dalla "
                         "posa vera del simulatore (prime run). La fitness "
                         "usa sempre la traiettoria vera.")
    ap.add_argument("--reloc-frame", default=None,
                    help="frame_id della posa su /initialpose nel maprestore (default: frame osservato sulla posa "
                         "della vittima, cioe' odom; con mappa caricata il riferimento di MOLA e' 'map').")
    ap.add_argument("--twin-state", choices=list(tv.TWIN_STATES), default="clean",
                    help="stato di partenza dei twin di valutazione: clean (riavviati con mappa vuota, "
                         "come nelle prime run); victim (prima di ogni rollout ricevono "
                         "le nuvole attaccate gia' inviate alla vittima e partono dal suo stato, via rigioco; "
                         "frame di controllo dalla posa pubblicata dalla vittima, non coerente con il threat "
                         "model); replay (stesso rigioco, ma frame dalla stima del gemello o dell'attaccante, "
                         "--replay-frame, e nessuna informazione della vittima nelle decisioni); "
                         "maprestore (ripristino O(1): map_save della vittima a inizio finestra, poi map_load + "
                         "relocalize nel twin a ogni candidato; solo confronto, non coerente); "
                         "ombra_e (mappa, posa e sigma dall'ombra); ombra_g (stato completo dall'ombra: "
                         "ombra e gemello sull'overlay di MOLA, --g-overlay, una MOLA nuova per rollout che "
                         "carica il .molastate; il nodo va lanciato con --twin-clock-base)")
    ap.add_argument("--g-overlay", default=tv.G_OVERLAY_ROOT,
                    help="con --twin-state ombra_g: radice dell'overlay di MOLA (default %(default)s); "
                         "la vittima resta sulle librerie di /opt")
    ap.add_argument("--g-resave", choices=["ogni", "off"], default="ogni",
                    help="con --twin-state ombra_g: risalvataggio dello stato subito dopo il caricamento, "
                         "a ogni rollout (catena degli hash, un file per rollout accanto allo stato "
                         "dell'ombra) o spento")
    ap.add_argument("--g-banco", type=str, default=None,
                    help="prova al banco della variante G (spenta di default): file JSON con uno o piu' "
                         ".molastate gia' salvati (sha256 completo), le pose delle loro finestre e i bracci "
                         "(posa fisica attacker_est o ombra_est, variabili di MOLA ammesse). Nessuna vittima, "
                         "ombra, attaccante o ricerca: robot nella posa vera salvata, poi per ogni braccio i "
                         "rollout nominali del gemello che carica lo stato (copia in <out>/stati). Esiti in "
                         "banco_G.json. Richiede --victim restart, --twin-state clean, --twin-init true, "
                         "--uniform-sim-time true o false; nodo con --twin-clock-base "
                         "(src/optimization/banco_g.py). Bracci con genome e drift_init (braccio B): "
                         "genoma dalla prima nuvola e deriva di partenza fissata, nodo con "
                         "--banco-twin-drift, --attack-from spawn e --drift-reset-on genome. Chiavi facoltative "
                         "(prova della biforcazione): init vera, salvata_banco, ripristino; scarti "
                         "avanti_mm, lato_mm, yaw_deg; assetto azzerato, conservato, esplicito o "
                         "esplicito_relativo (server con set_state_assetto); zone salvate della run e metro "
                         "(con --zone-attaccante e --metro-al-nodo); ripetizioni per braccio; confronti "
                         "vittima-gemello (con --damage-ref baseline); esito del ramo, assetto letto e danno nel "
                         "record. init riallineato (D-31c, con riallinea_da e riallinea_posa attacker_est o "
                         "ripristino): riparte dallo stato del braccio di partenza salvato a fine assestamento "
                         "(map_save del gemello, istantanea di Isaac con nome), senza assestamento; richiede "
                         "--settle-clouds > 0 e save_nome nel server. La rampa e' riservata e rifiutata")
    ap.add_argument("--g-twin-lost-ends-run", action="store_true",
                    help="con --twin-state ombra_g: se il nominale fallisce per mancanza di posa dopo che "
                         "il gemello ha caricato lo stato dell'ombra, la run si chiude con l'esito "
                         "victim_lost_g invece di trattenere la finestra (spento di default)")
    ap.add_argument("--g-nominale-senza-posa", choices=["trattieni", "continua"], default="trattieni",
                    help="con --twin-state ombra_g e --damage-ref baseline: nominale del gemello senza posa "
                         "dopo il caricamento dello stato. trattieni (default): finestra saltata come "
                         "guasto d'infrastruttura. continua: l'esito si registra come "
                         "nominale_non_localizza e si valutano i candidati; quelli senza posa restano "
                         "esclusi (guasto_gemello). Senza nominale la percettibilita' interna non e' "
                         "definita (None nella history, obiettivo 0)")
    ap.add_argument("--no-nominal-rollout", action="store_true",
                    help="serie B: nessun rollout nominale del gemello nelle finestre; il danno si misura "
                         "solo contro la baseline. Richiede --damage-ref baseline, --perceptibility chamfer, "
                         "--settle-clouds > 0 e --victim continuous; rifiuta --untracked nominal, --f2-budget, "
                         "--g-twin-lost-ends-run, --passthrough, --repeat-*, --map-gate-test. Riferimento "
                         "untracked costante (soglia assoluta 0,60 sul candidato), controllo del frame sulla "
                         "posa vera salvata a inizio w0. Con --twin-state ombra_g ogni candidato ha "
                         "esito_gemello dai controlli del suo rollout (g_check): guasto_gemello (controlli "
                         "falliti, escluso), gemello_perde_posa (controlli giusti, nessuna posa: escluso, "
                         "registrato a parte) o controlli_ok. Spenta di default")
    ap.add_argument("--replay-frame", choices=["twin", "attacker"], default="twin",
                    help="con --twin-state replay: frame di controllo del gemello dalla sua stima a fine "
                         "rigioco composta con l'origine dell'attaccante (twin, default; se la storia e' "
                         "troncata si ricade su attacker) o dalla posa creduta dalla MOLA dell'attaccante")
    ap.add_argument("--twin-init", choices=["true", "attacker_est", "ombra_est"], default="true",
                    help="stato fisico del gemello in Isaac a ogni rollout: true (default, posa e velocita' "
                         "vere della vittima dallo stato salvato), attacker_est (posa stimata dalla MOLA "
                         "dell'attaccante, velocita' dalle sue pose consecutive o zero a robot fermo; "
                         "richiede il comando set_state del server standalone) o ombra_est (lo stesso con "
                         "la posa creduta dall'ombra; richiede --twin-state ombra_e o ombra_g)")
    ap.add_argument("--twin-init-vel", choices=["zero", "poses"], default="zero",
                    help="con --twin-init attacker_est o ombra_est: velocita' del gemello sempre a zero (default: il "
                         "robot e' fermo a inizio finestra in 49 finestre su 50, P13) o dalle pose "
                         "consecutive della MOLA dell'attaccante (sotto 0,05 m/s a zero)")
    ap.add_argument("--twin-init-assetto", choices=["azzerato", "conservato", "attaccante_relativo", "attaccante"],
                    default="azzerato",
                    help="con --twin-init attacker_est o ombra_est, rollio e beccheggio del gemello nel set_state: "
                         "azzerato (default, come sempre: quaternione dal solo yaw), conservato (keep_tilt: quelli "
                         "del restore, cioe' l'assetto vero salvato; solo confronto, fuori da D6), "
                         "attaccante_relativo (variante 3a, adottabile: quelli stimati dalla MOLA dell'attaccante, "
                         "senza altro) o attaccante (variante 3b, diagnostica, non adottabile: gli stessi composti "
                         "con l'assetto della base letto da Isaac alla posa di partenza della run). Richiede "
                         "set_state_assetto nel server di Isaac")
    ap.add_argument("--registra-assetto", action="store_true",
                    help="comando assetto di Isaac dopo il restore, dopo il set_state e a fine assestamento di "
                         "ogni rollout (vittima e gemelli), in history (per valutazione e per finestra) e accanto "
                         "alle tracce (<traccia>_assetto.json). Richiede il comando assetto nel server; spento di default")
    ap.add_argument("--prove-vincitore", type=int, default=0,
                    help="D-31b: dopo la scelta e prima dell'applicazione, N rollout identici del gemello col "
                         "genoma del vincitore dallo stesso stato (tag w<k>_prova<j>), in history[k].prove_vincitore "
                         "(danno, ramo nell'assestamento, qualita' ICP delle prime 20 nuvole, esito, tau, lento); il "
                         "vincitore si applica comunque. 0 = nessuna prova (default)")
    ap.add_argument("--riallinea-dopo-assestamento", action="store_true",
                    help="D-31c: a fine assestamento della vittima pausa, stato dell'ombra (variante G) e istantanea "
                         "di Isaac nel posto 'riallineo', gemello del vincitore che carica quello stato (posa dalla "
                         "stima dell'attaccante, nessun assestamento, stato del generatore della vittima se il nodo "
                         "lo offre), poi ripresa della vittima dall'istantanea; history[k].riallineato e confronti (a) "
                         "e (b). Richiede --victim continuous, --twin-state ombra_g, --twin-init attacker_est, --trace "
                         "e save_nome nel server di Isaac; spento di default")
    ap.add_argument("--confronto-vincitore", action="store_true",
                    help="per ogni metro applicato, in history[k]: gemello che ha scelto (tag, traccia, log di MOLA) "
                         "e confronto (a) vittima-gemello con le definizioni del criterio A. Richiede --trace e "
                         "--damage-ref baseline; acceso da --riallinea-dopo-assestamento; spento di default")
    ap.add_argument("--uniform-sim-time", choices=["off", "true", "false"], default="off",
                    help="use_sim_time delle MOLA: off (default, come prima: gemello true dalla riga di "
                         "comando del launch, vittima e attaccante false perche' ROS_ARGS la sostituisce), "
                         "true (tutte con use_sim_time:=true, come dice il launch) o false (tutte senza)")
    ap.add_argument("--shadow-check", action="store_true",
                    help="confronto ombra-vittima (strumento del banco, fuori dal ciclo di decisione): avvia "
                         "l'ombra anche senza --twin-state ombra_e, registra le pose delle due MOLA "
                         "(shadow_check_poses.csv, 17 cifre), a ogni inizio finestra livelli A/B/B'/C/C+, scan di "
                         "una sola ed episodi (campo shadow_check della history, shadow_check.json), a fine run "
                         "shadow_check_scans.csv. Vittima e ombra partono insieme a flusso chiuso, che si apre "
                         "quando tutte e due risultano iscritte (al piu' SYNC_SUB_TIMEOUT_S). L'ambiente della "
                         "vittima non cambia. Richiede --victim continuous. Spento di default.")
    ap.add_argument("--sync-wait", choices=["graph", "publisher"], default="graph",
                    help="condizione dell'avvio sincrono di vittima e ombra: graph (default, come prima) "
                         "aspetta che le iscrizioni compaiano nel grafo visto dall'orchestratore; "
                         "publisher aspetta anche che i lettori agganciati ai publisher del nodo salgano "
                         "di quanti sono vittima e ombra rispetto a prima dello spawn (status "
                         "victim_pub_matched, nodo con --report-matched), piu' SYNC_PUB_MARGIN_S, al piu' "
                         "SYNC_SUB_TIMEOUT_S; accende anche la guardia del primo stamp. Con "
                         "--twin-state ombra_g, prima del play di ogni gemello, aspetta anche un "
                         "lettore in piu' sul publisher principale (main_pub_matched)")
    ap.add_argument("--sync-guard", action="store_true",
                    help="guardia dell'avvio sincrono anche con --sync-wait graph: primo stamp elaborato "
                         "diverso fra vittima e ombra -> run chiusa come shadow_sync_fail. Spento di default "
                         "(acceso comunque con --sync-wait publisher)")
    ap.add_argument("--shadow-check-tol", type=str, default="",
                    help="soglie del confronto, chiave=valore separate da virgola (chiavi e default in "
                         "shadow_check.DEFAULT_TOL, da tolleranze_ombra_vittima_2026-10-24.md; 'none' toglie)")
    ap.add_argument("--shadow-topic", type=str, default=None,
                    help="topic dell'ombra (default: quello della vittima); da usare con il nodo lanciato con "
                         "lo stesso --shadow-topic")
    ap.add_argument("--shadow-crc-log", type=str, default=None,
                    help="registro dei crc scritto dal nodo (--shadow-crc-log del nodo), letto dal confronto")
    ap.add_argument("--fresh-cloud-after-restore", action="store_true",
                    help="dopo ogni restore di Isaac arma il filtro del nodo di perturbazione (che va lanciato "
                         "con la stessa opzione): vittima, attaccante e gemello ricevono solo nuvole con stamp "
                         "> tempo simulato della pausa + margine, non la scansione a cavallo del ripristino. "
                         "Spento di default.")
    ap.add_argument("--map-gate-test", action="store_true",
                    help="collaudo isolato del ripristino via mappa: guida la vittima ~3 m, map_save, poi su un "
                         "twin nuovo map_load + relocalize e riporta la fedelta' (scarto twin-vittima) e gli esiti "
                         "dei servizi, poi esce. Non esegue la ricerca.")
    ap.add_argument("--replay-last-s", type=float, default=0.0,
                    help="con --twin-state victim, secondi di storia rigiocati (0 = tutta)")
    ap.add_argument("--replay-rate-hz", type=float, default=10.0,
                    help="cadenza del rigioco (nuvole al secondo)")
    ap.add_argument("--replay-gate", type=int, default=0,
                    help="verifica del rigioco: dopo ogni tratto applicato, N twin rigiocati e confrontati "
                         "con la vittima (stima, scarto dal vero, qualita'), per ogni valore di --replay-gate-last-s")
    ap.add_argument("--replay-gate-last-s", type=str, default="0,20,10,5",
                    help="valori di storia rigiocata (s) provati dal gate, separati da virgola (0 = tutta)")
    ap.add_argument("--drift-reset-on", choices=["genome", "drive", "keep"], default="genome",
                    help="quando azzerare la deriva temporale accumulata: al "
                         "caricamento del genoma (default: con --attack-from spawn "
                         "accumula gia' durante l'assestamento a robot fermo), "
                         "all'inizio della guida (la deriva parte con il moto) o mai "
                         "(keep: la deriva della vittima prosegue fra genomi e finestre, "
                         "ogni gemello parte dalla deriva della vittima; richiede "
                         "--attack-from spawn)")
    ap.add_argument("--stop-on", choices=["estimate", "true"], default="estimate",
                    help="dove si ferma il rollout nella finestra: quando la distanza "
                         "STIMATA da MOLA raggiunge l'orizzonte (default: il robot si "
                         "ferma dove crede di essere arrivato), con tetto sulla distanza "
                         "vera a 1.5x l'orizzonte se la stima si congela (esito "
                         "est_capped); oppure alla distanza vera percorsa (true, "
                         "strumentazione nostra, come nelle prime run).")
    ap.add_argument("--window-stop", choices=list(wstop.MODES), default="legacy",
                    help="chiusura della finestra con --stop-on estimate: legacy "
                         "(velocita' del follower fino a stima >= H, poi stop; il robot "
                         "va ~37-39 cm oltre H) o ramp (negli ultimi metri "
                         "v = min(v_follower, max(vmin, k (H - d_stima))); coast chiuso "
                         "a robot fermo e stima successiva all'arresto, tetto 2 s).")
    ap.add_argument("--stop-ramp-k", type=float, default=wstop.DEFAULT_K,
                    help="guadagno della rampa (1/s, default 1.5)")
    ap.add_argument("--stop-ramp-vmin", type=float, default=wstop.DEFAULT_VMIN,
                    help="velocita' minima della rampa prima di H (m/s, default 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true",
                    help="stampa i parametri decodificati per ogni candidato, "
                         "non solo per quello scelto.")
    ap.add_argument("--trace", action="store_true",
                    help="salva per ogni rollout la traccia per-tick (stima MOLA, "
                         "posa vera, comandi) e le tracce per-scan di MOLA "
                         "(qualita' ICP, sigma).")
    ap.add_argument("--lidar-pose-x", type=float, default=None,
                    help="sovrascrive LIDAR_POSE_X (valore dal TF: -0.2317); "
                         "utile per verificare la convenzione di segno di MOLA.")
    ap.add_argument("--repeat-genome", type=str, default=None,
                    help="test di ripetibilita': applica --repeat-n volte lo "
                         "stesso genoma sulla prima finestra e riporta media e "
                         "dispersione del danno. Formato "
                         "'percorso/history.json:finestra' (genoma applicato) o "
                         "'percorso/history.json:finestra:candidato' (genoma di una "
                         "valutazione). La posa di partenza va data con --start-pose.")
    ap.add_argument("--repeat-n", type=int, default=3)
    ap.add_argument("--repeat-spec", type=str, default=None,
                    help="analisi degli operatori: file JSON con una lista di {label, genome (17 valori), n}; "
                         "dallo stesso stato salvato esegue il nominale e poi n rollout per voce, con esiti in "
                         "sweep.json e tracce sw_<label>_r<k>.csv.")
    ap.add_argument("--drive-timeout", type=float, default=45.0,
                    help="tempo massimo di guida di un rollout (s, default 45); per le traversate intere "
                         "senza finestre servono ~90 s.")
    ap.add_argument("--out", type=str, default="data/attack/run")
    args = ap.parse_args()
    err = check_random_first_args(args) or tv.check_twin_args(args) or banco_g.check_args(args)
    if not err and args.shadow_check and args.victim != "continuous":
        err = "--shadow-check richiede --victim continuous (l'ombra ascolta le nuvole della vittima)"
    if not err and args.shadow_check and not args.trace:
        # without traces sigma, twist and discrete quantities do not enter the levels
        err = "--shadow-check richiede --trace (livelli sulle tracce CSV di MOLA)"
    if not err:
        err = check_spawn_b_args(args)
    if not err:
        err = check_goal_mode_args(args)
    if not err and args.select_only_valid and args.select == "ratio":
        # ratio chooses on the pymoo front, without the candidate records
        err = "--select-only-valid richiede --select maxdamage o pareto_maxdamage"
    if not err and args.mola_se_ne_accorge and (args.accorge_icp_rifiutate < 1
                                                 or not args.accorge_senza_posa_s > 0):
        err = "--accorge-icp-rifiutate deve essere >= 1 e --accorge-senza-posa-s > 0"
    if not err and args.danno_metro and args.damage_ref != "baseline":
        err = "--danno-metro richiede --damage-ref baseline"
    if not err and args.incompleti_regola and not args.mola_se_ne_accorge:
        err = "--incompleti-regola richiede --mola-se-ne-accorge (classe b)"
    if not err:
        err = problema_opzioni_d31(args)
    if not err:
        err = problema_rampa(args)
    if not err:
        err = problema_genoma_in_moto(args)
    if not err:
        err = problema_limiti(args)
    shadow_tol = None
    if not err and args.shadow_check:
        try:
            shadow_tol = shc.parse_tol(args.shadow_check_tol)
        except ValueError as e:
            err = f"--shadow-check-tol: {e}"
    if not err and args.controllo_eta is not None and not args.controllo_eta > 0:
        err = "--controllo-eta deve essere > 0"
    if not err and args.controllo_eta is not None and not args.frame_log_nodo:
        err = "--controllo-eta richiede --frame-log-nodo (registro per frame del nodo)"
    if not err and getattr(args, "mola_binario_g1", False):
        err = tv.g_overlay_problem(args.g_overlay)
        err = f"--mola-binario-g1: {err}" if err else None
    if not err and args.dds_profile:
        try:
            DDS_PROFILO.update(prepara_dds_profilo(args.dds_profile))
        except ValueError as e:
            err = str(e)
    if err:
        print(f"ERRORE: {err}")
        return 2
    if DDS_PROFILO["path"]:
        # before rclpy.init: the orchestrator participant reads the profile at start-up
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = DDS_PROFILO["path"]
        print(f"  profilo DDS: {DDS_PROFILO['path']} (md5 {DDS_PROFILO['md5']})")
    genome_seq = None
    try:
        victim_env = parse_env_pairs(args.victim_mola_env)
        if args.fixed_genome_seq is not None:
            if args.search != "fixed" or args.fixed_genome is not None:
                raise ValueError("--fixed-genome-seq richiede --search fixed e non va con --fixed-genome")
            genome_seq = load_genome_seq(args.fixed_genome_seq)
    except (ValueError, OSError, KeyError, IndexError) as e:
        print(f"ERRORE: {e}")
        return 2
    MOLA_STOP["signal"] = args.mola_stop_signal
    TWIN_STATE["coherence"] = tv.coherence_record(args.twin_state, args.twin_init, args.twin_align,
                                                  args.replay_frame)
    if args.twin_state == "maprestore" or args.map_gate_test:
        print("ATTENZIONE: --twin-state maprestore legge la mappa e la posa creduta dalla MOLA della "
              "vittima: NON coerente con il threat model, solo confronto. /initialpose del gemello "
              f"rimappato su {tv.TWIN_INITIALPOSE}.")
    STALL_POLICY["tolerate"] = args.stall_policy == "tolerate"
    METRO["on"] = bool(args.metro_al_nodo)
    ZONE["on"] = bool(getattr(args, "zone_attaccante", False))
    if ZONE["on"] and not METRO["on"]:
        print("ERRORE: --zone-attaccante richiede --metro-al-nodo")
        return 2
    CLOUD_SETTLE.update({"n": args.settle_clouds, "reopen": args.reopen_clouds,
                         "never_clean": bool(args.victim_never_clean)})
    TWIN_STATE["guard"] = args.replay_guard_cm > 0
    TWIN_STATE["guard_cm"] = args.replay_guard_cm
    if args.select == "pareto_maxdamage":
        # pareto_maxdamage: no re-evaluation and no filter on slow rollouts. --topk and --reeval
        # still come from the campaign script.
        print(f"selezione pareto_maxdamage: --topk {args.topk} e --reeval {args.reeval} ignorati "
              f"(nessun rollout di rivalutazione); candidati lenti (tau > {SLOW_TAU} s) non filtrati")

    # Damage reference polyline (--damage-ref baseline), loaded once; with the per-window reference
    # it stays None.
    baseline_poly = None
    if args.damage_ref == "baseline":
        if not args.baseline_path:
            print("ERRORE: --damage-ref baseline richiede --baseline-path <csv s_m,x_m,y_m>")
            return 2
        try:
            baseline_poly = load_baseline(args.baseline_path)
        except ValueError as e:
            print(f"ERRORE: {e}")
            return 2
        print(f"riferimento del danno: baseline {args.baseline_path} "
              f"({len(baseline_poly)} punti, da {np.round(baseline_poly[0], 3)} "
              f"a {np.round(baseline_poly[-1], 3)})")
    ref_lbl = "dalla baseline" if baseline_poly is not None else "dal nominale"
    # --incompleti-regola: baseline pose at window end for the damage of (c)
    fine_finestra_base = None
    if args.incompleti_regola:
        if baseline_poly is not None:
            try:
                fine_finestra_base = wrec.punti_fine_finestra(args.baseline_path)
            except ValueError as e:
                print(f"ERRORE: --incompleti-regola: {e}")
                return 2
            print(f"incompleti: posa della baseline a fine finestra per {len(fine_finestra_base)} finestre")
        else:
            print("attenzione: --incompleti-regola senza --damage-ref baseline: i (c) tengono il danno "
                  "della regola in uso (danno_fine_finestra None)")

    goal = tuple(float(v) for v in args.goal.split(","))
    goals_rel = ([tuple(float(v) for v in g.split(",")) for g in args.goals.split(";")]
                 if args.goals else [goal])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ZONE["out"] = str(out)
    if DDS_PROFILO["path"]:
        with open(out / "dds_profilo.json", "w") as f:
            json.dump({**DDS_PROFILO, "variabile": "FASTRTPS_DEFAULT_PROFILES_FILE"}, f, indent=1)
    registri_nodo_correnti(args)
    if args.controllo_eta is not None:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        CONTROLLO_ETA.update({"soglia": float(args.controllo_eta),
                              "frame_log": frame_log_corrente(args.frame_log_nodo),
                              "registro": str(out / "logs" / "controllo_eta.jsonl")})

    script = str(project_root / "src" / "nodes" / "launch_mola_attack.sh")

    isaac = IsaacClient()
    # MOLA instances left over from previous runs would publish on the same topic and the poses
    # would mix.
    left = MolaRunner.kill_orphans()
    if left:
        print(f"\nATTENZIONE: {left} processi mola-cli non si lasciano "
              f"terminare.\nControlla con: ps aux | grep '[m]ola-cli'\n")
        return 1
    # MOLA configuration from the command line, the same for twin, victim and attacker; empty by
    # default.
    cfg_env = mola_config_env(args)
    g1_env = ambiente_binario_g1(args)
    if g1_env:
        # --mola-binario-g1: same binary for all MOLA instances (goes from cfg_env to twin, victim,
        # shadow and attacker; shadow and twin G recompute it identically)
        cfg_env.update(g1_env)
        TWIN_STATE["binario_g1"] = {"overlay": args.g_overlay, "env": dict(g1_env)}
        print(f"  binario di MOLA unico (--mola-binario-g1): overlay {args.g_overlay}")
    extra_env = dict(cfg_env)
    # MOLA ROS arguments, only from the environment variable (no YAML): twin /initialpose isolated
    # in maprestore, uniform use_sim_time.
    ros_args = {"twin": tv.mola_ros_args("twin", args.uniform_sim_time,
                                         isolate_initialpose=(args.twin_state == "maprestore"
                                                              or args.map_gate_test)),
                "victim": tv.mola_ros_args("victim", args.uniform_sim_time),
                "attacker": tv.mola_ros_args("attacker", args.uniform_sim_time)}
    if args.twin_state in tv.SHADOW_MODES or args.shadow_check:
        ros_args["shadow"] = tv.mola_ros_args("shadow", args.uniform_sim_time)
    TWIN_STATE["ros_args"] = ros_args
    if ros_args["twin"] is not None:
        extra_env["ROS_ARGS"] = ros_args["twin"]
    # twin environment without the shadow state (ombra_e completes it per window)
    TWIN_STATE["twin_env_base"] = dict(extra_env)
    if args.twin_state == "ombra_g":
        TWIN_STATE.update({"g_root": args.g_overlay, "g_resave": args.g_resave})
    mola = MolaRunner(script, out / "logs", extra_env=extra_env,
                      traces=args.trace)
    # Continuous victim (default): a single MOLA for the applied path. With --victim restart the
    # applied stretch uses a restarted twin. --victim-mola-env: extra variables for the victim only
    # (empty by default).
    victim = (VictimRunner(script, out / "logs", traces=args.trace, ros_args=ros_args["victim"],
                           env={**cfg_env, **victim_env})
              if args.victim == "continuous" else None)
    if victim is not None and args.victim_shadow_env:
        # --victim-shadow-env: the victim gets the same environment as the shadow (off by default)
        victim.extra_env = victim_extra_env(cfg_env, victim_env, ros_args["victim"], shadow_env=True)
    if CONTROLLO_ETA["soglia"] is not None and victim is not None \
            and (victim.extra_env or {}).get("_STDBUF_O") != "L":
        # with block-buffered stdout the last takes of the stretch are not in the log yet
        print("ERRORE: --controllo-eta con la vittima continua richiede --victim-shadow-env "
              f"(stdout a righe con {tv.STDBUF_LIB})")
        return 2
    # Variant E shadow: same configuration as the victim, without --victim-mola-env. With
    # --shadow-check it also starts outside variant E, as a measurement only.
    shadow = (ShadowRunner(script, out / "logs", traces=args.trace, ros_args=ros_args["shadow"],
                           env=cfg_env, topic=args.shadow_topic or VICTIM_TOPIC,
                           g_root=args.g_overlay if args.twin_state == "ombra_g" else None)
              if (args.twin_state in tv.SHADOW_MODES or args.shadow_check) and victim is not None else None)
    shadow_rec = None
    SYNC_WAIT.update({"mode": args.sync_wait, "guard": bool(args.sync_guard)})
    ORIGINE_VITTIMA["coerente"] = bool(getattr(args, "origine_coerente_vittima", False))
    if args.shadow_check and shadow is not None:
        # effective environment: MOLA and preload variables inherited (command file) plus those
        # given to each runner
        inh = {k: v for k, v in os.environ.items() if k.startswith(("MOLA_", "LD_PRELOAD", "_STDBUF"))}
        shadow_rec = shc.ShadowCheck(
            out, victim, shadow, tol=shadow_tol, crc_log=args.shadow_crc_log,
            config={"ambiente": shc.env_config_diff({**inh, **victim.extra_env}, {**inh, **shadow.extra_env}),
                    "topic_vittima": VICTIM_TOPIC, "topic_ombra": shadow.args[0],
                    "trace": bool(args.trace), "crc_log": args.shadow_crc_log})
        SYNC_WAIT["recorder"] = shadow_rec
        if args.sync_wait != "graph" or args.sync_guard:
            shadow_rec.config["sync_wait"] = {"modo": args.sync_wait, "guardia": sync_guard_active()}
        d = shadow_rec.config["ambiente"]["diverse"]
        print(f"  confronto ombra-vittima acceso: topic ombra {shadow.args[0]}, ambiente "
              f"{'identico' if not d else 'diverso in ' + ', '.join(d)}")
    # Attacker MOLA: requires the gated stream of the continuous victim.
    attacker = None
    if args.twin_align == "attacker":
        if victim is None:
            print("  [nota] --twin-align attacker richiede --victim continuous: twin allineati alla posa vera")
        else:
            attacker = AttackerRunner(script, out / "logs", traces=args.trace,
                                      ros_args=ros_args["attacker"], env=cfg_env)
    print(f"  ROS_ARGS delle MOLA: gemello {ros_args['twin'] or '(riga di comando del launch)'}, "
          f"vittima {ros_args['victim']}, attaccante {ros_args['attacker']}")
    node = AttackNode(FollowerModel())
    if args.registri_dal_vivo:
        try:
            node.abilita_registri(out / "registri_dal_vivo")
            print(f"  registri dal vivo: {out / 'registri_dal_vivo'}")
        except Exception as e:
            node.registri = None
            print(f"  [nota] registri dal vivo non avviati: {type(e).__name__}: {e}; la run continua senza")
    if shadow is not None:
        e_err = node.enable_shadow()
        if shadow_rec is not None:
            node.enable_shadow_check(shadow_rec)
        if e_err and args.twin_state in tv.SHADOW_MODES:
            print(f"ERRORE: --twin-state {args.twin_state} richiede il servizio map_save (mola_msgs): {e_err}")
            node.close()
            return 2
        print(f"  ombra (variante E): {tv.SHADOW_NS}, ROS_ARGS {ros_args['shadow']}, stdout a righe "
              f"{'si' if '_STDBUF_O' in shadow.extra_env else 'no (libstdbuf assente)'}")
    node.true_source = args.true_pose
    node.window_stop = args.window_stop
    node.stop_ramp_k = args.stop_ramp_k
    node.stop_ramp_vmin = args.stop_ramp_vmin
    node.goal_mode = args.goal_mode
    node.direction_lookahead = args.direction_lookahead
    if args.true_pose == "ros":
        # DDS discovery of a just-created node can take more than a second: a short fixed wait
        # randomly falls back to the file pose.
        t_wait = time.time()
        while node.true is None and time.time() - t_wait < 5.0:
            node.spin(0.1)
        if node.true is None:
            print("  [nota] nessun messaggio su /chassis/odom: posa vera dal server via file")
            node.true_source = "file"
    gen_obj = PerturbationGenerator()
    n_genes = gen_obj.get_genome_size()
    active_genes = geni_attivi(args.genome_group, LIMITI["bloccati"])
    n_search = len(active_genes)

    history = []
    lost_state = {"since": None, "any": False}   # victim pose loss (--fail-rc-on-no-pose)
    k = None                       # current window, to mark a stall
    stalled = None
    foreign = None                 # foreign publisher on the victim topic after start
    sync_fail = None               # synchronous start guard (shadow_sync_fail)
    guasto_catena = None           # run_end guasto_catena (--controllo-catena)
    errore_conf = None             # errore_configurazione (--zone-attaccante)
    print("=" * 70)
    print(f"  ATTACCO RECEDING-HORIZON")
    print(f"  bersaglio {goals_rel[-1]}   orizzonte {args.horizon} m   "
          f"{args.pop}x{args.gen} = {args.pop*args.gen} candidati/finestra")
    print("=" * 70)

    try:
        node.watch = isaac
        if getattr(args, "controllo_ombra_viva", False):
            if shadow is not None:
                node.ombra = shadow
                print("  controllo ombra viva: acceso (ogni 0,5 s, anche durante la ricerca)")
            else:
                print("  controllo ombra viva: chiesto ma senza ombra in questa run, quindi spento")
        if args.genoma_nullo_se_nessun_ammissibile:
            print("  genoma nullo se nessun ammissibile: acceso (la vittima percorre il metro col genoma nullo)")
        # A command left over from an interrupted run would move the robot during the state save and
        # after set_pose.
        node.halt()
        # Kept drift (keep): the node must confirm it, otherwise the run would reset the drift
        # silently. A node left in keep by a previous launch is put back to the default.
        if args.drift_reset_on == "keep":
            if not node.set_drift_keep(True):
                print("ERRORE: il nodo di perturbazione non conferma drift_keep (versione senza "
                      "keep?): --drift-reset-on keep non applicabile")
                return 2
        elif getattr(node, "st_drift_keep", None):
            node.set_drift_keep(False)
        # --fresh-cloud-after-restore: node and orchestrator must match, both ways (a node with the
        # option but never armed does not filter and does not report it).
        err = fresh_cloud_mismatch(node, args.fresh_cloud_after_restore)
        if not err and (ZONE["on"] or node.node_zone_on is True):
            err = zone_mismatch(node, ZONE["on"])
        if not err:
            RAMPA["R"] = int(args.rampa_genoma or 1)
            err = rampa_mismatch(node, RAMPA["R"])
        if not err:
            GIM.update(S=float(args.genoma_in_moto or 0.0), kmin=int(args.genoma_in_moto_kmin),
                       kmax=int(args.genoma_in_moto_kmax))
            GIM["cpf"] = bool(args.chamfer_per_fase) or GIM["S"] > 0
            err = gim_mismatch(node, GIM["S"], GIM["kmin"], GIM["kmax"])
        if not err:
            err = cpf_mismatch(node, GIM["cpf"])
        if not err and args.riallinea_dopo_assestamento:
            err = stato_generatore_mancante(node)
        # --g-banco: saved zones of the run in the spec, no live snapshot
        # (banco_g.problema_estensioni requires them with --zone-attaccante)
        if not err and ZONE["on"] and attacker is None and not args.g_banco:
            err = "--zone-attaccante richiede la MOLA dell'attaccante (--twin-align attacker, --victim continuous)"
        if err:
            print(f"ERRORE: {err}")
            return 2
        if args.fresh_cloud_after_restore:
            node.fresh_cloud = True
            print("  prima nuvola  : dopo ogni restore solo nuvole acquisite dopo la pausa (filtro del nodo)")
        if args.twin_init in ("attacker_est", "ombra_est") and "set_state" not in isaac.caps():
            print(f"ERRORE: --twin-init {args.twin_init} richiede il comando set_state del server "
                  "(isaac/run_isaac_standalone.py di questa versione); il server attivo non lo ha")
            return 2
        d31_on = bool(args.registra_assetto or args.twin_init_assetto != "azzerato"
                      or args.riallinea_dopo_assestamento)
        if d31_on:
            err = problema_capacita_d31(args, isaac.caps())
            if err:
                print(f"ERRORE: {err}")
                return 2
        ASSETTO.update({"registra": bool(args.registra_assetto), "gemello": args.twin_init_assetto})
        if args.start_pose:
            sx, sy, syaw = (float(v) for v in args.start_pose.split(","))
            isaac.pause()
            isaac.set_pose(sx, sy, math.radians(syaw))
            isaac.play()
            node.spin(1.0)
        start_pose = node.settled_pose(isaac)
        if args.twin_init_assetto == "attaccante" or args.registra_assetto:
            # starting frame of the attacker map: base attitude now, robot still, before its MOLA
            # starts
            ASSETTO["spawn"] = misura_assetto_spawn(isaac, start_pose)
            with open(out / "assetto_spawn.json", "w") as f:
                json.dump(ASSETTO["spawn"], f, indent=1)
            print(f"  assetto allo spawn: rollio {math.degrees(ASSETTO['spawn']['roll']):+.4f} gradi, "
                  f"beccheggio {math.degrees(ASSETTO['spawn']['pitch']):+.4f} gradi")
        if node.true_source == "ros":
            print(f"  frame di lavoro: odom di Isaac (/chassis/odom), non la scena; "
                  f"posa nella scena {np.round(isaac.pose(), 3)}")
        err = never_clean_mismatch(node, args.victim_never_clean)
        if not err and args.victim_never_clean:
            err = victim_topic_problem(node, timeout=10.0)
        if err:
            print(f"ERRORE: {err}")
            return 2
        if args.settle_clouds > 0:
            print(f"  assestamento : {args.settle_clouds} nuvole da fermo prima della guida"
                  + (f", {args.reopen_clouds} alla riapertura" if args.reopen_clouds > 0 else "")
                  + " (contate dal nodo; --settle-sec non usato nei rollout)")
        if victim is not None and args.victim_start == "w0":
            # Deferred start: victim and shadow are born at the first application (_victim_rollout);
            # here only the attacker MOLA, on clean clouds.
            om = "estimate" if args.origin_from_estimate else "true"
            victim.deferred = {"shadow": shadow, "origin_mode": om}
            if attacker is not None:
                victim.origin_info = attacker_phase_start(isaac, node, attacker, om,
                                                          max(args.settle_clouds, 1) / 10.0)
            if args.victim_never_clean:
                node.spin(0.3)
                node.never_clean_start = {"dropped": getattr(node, "victim_clean_dropped", None),
                                          "clean_out": getattr(node, "victim_clean_out", None)}
            print(f"  vittima{' e ombra' if shadow is not None else ''}: avvio alla prima applicazione"
                  + (", attacco acceso dalla prima nuvola" if args.victim_never_clean else "")
                  + ("; MOLA dell'attaccante avviata" if attacker is not None else ""))
        elif victim is not None:
            # Single initialisation of victim and attacker MOLA, robot still at the start pose: from
            # here on they are not restarted.
            victim.start(isaac, node, args.settle_sec, attacker=attacker,
                         origin_mode="estimate" if args.origin_from_estimate else "true",
                         shadow=shadow, sync_shadow=shadow_rec is not None)
            if shadow_rec is not None:
                shadow_rec.config["avvio_sincrono"] = victim.sync_info
                shadow_rec.save()
            # same check, no wait: with the victim started the status has arrived
            err = fresh_cloud_mismatch(node, args.fresh_cloud_after_restore)
            if err:
                print(f"ERRORE: {err}")
                return 2

        # Goals are relative to the initial pose (x forward, y left), rotated into the start
        # orientation.
        c, s = math.cos(start_pose[2]), math.sin(start_pose[2])
        goals_abs = [(start_pose[0] + gx * c - gy * s, start_pose[1] + gx * s + gy * c)
                     for gx, gy in goals_rel]
        goal_abs = goals_abs[-1]

        # First goal closer than horizon + arrival tolerance: the rollout reaches it before covering
        # the horizon. With a single waypoint it stops on `arrived` at about d - ARRIVAL_TOL metres,
        # close to or below the stopped_early threshold.
        d_first = math.hypot(goals_abs[0][0] - start_pose[0], goals_abs[0][1] - start_pose[1])
        direction = args.goal_mode == "direction"
        dir_line = None
        if direction:
            # Fixed line from the true run start pose to the goal: in the rollouts it is mapped into
            # the control frame with to_local.
            dir_line = (tuple(map(float, start_pose[:2])),
                        direction_line([start_pose[:2], goals_abs[-1]], None)[1])
        if d_first < args.horizon + ARRIVAL_TOL and not direction:
            if len(goals_abs) == 1:
                print(f"  ATTENZIONE: il bersaglio dista {d_first:.2f} m, meno di orizzonte + "
                      f"tolleranza d'arrivo ({args.horizon} + {ARRIVAL_TOL}): il rollout si "
                      f"ferma per 'arrived' a ~{max(0.0, d_first - ARRIVAL_TOL):.2f} m "
                      f"(stopped_early sotto {MIN_TRAVEL_FRAC * args.horizon:.2f} m)")
            else:
                print(f"  ATTENZIONE: il primo waypoint dista {d_first:.2f} m, meno di orizzonte + "
                      f"tolleranza d'arrivo ({args.horizon} + {ARRIVAL_TOL}): la prima finestra "
                      f"cambia waypoint durante il tratto misurato")

        total = 0.0
        prev = start_pose[:2]
        for g in goals_abs:
            total += math.hypot(g[0] - prev[0], g[1] - prev[1])
            prev = g
        # Every window uses warmup + horizon metres: counting it on the horizon only creates extra
        # windows, which find the goal already reached.
        per_window_m = args.horizon + max(0.0, args.warmup)
        n_windows = max(1, int(total / per_window_m))
        if direction:
            n_windows = args.max_windows
            print(f"  bersaglio     : solo direzione {np.round(dir_line[1], 4)}, carota a "
                  f"{args.direction_lookahead} m, {n_windows} finestre guidate "
                  f"(al piu' {args.extra_windows} trattenute), chiusura con la rampa")

        frames_per_window = args.horizon / 0.8 * 10.0
        evals = 0 if args.passthrough else args.pop * args.gen
        per_window = (evals + 2) * (10.0 + args.warmup / 0.8)

        print(f"  partenza      : {np.round(start_pose,3)}")
        print(f"  waypoint rel  : {goals_rel}")
        print(f"  waypoint abs  : {[tuple(np.round(g, 3)) for g in goals_abs]}")
        print(f"  distanza      : {total:.2f} m  ->  {n_windows} finestre "
              f"da {per_window_m:.2f} m "
              f"({args.horizon} misurati"
              f"{f' + {args.warmup} riscaldamento' if args.warmup > 0 else ''})")
        if args.settle_sec > 0:
            print(f"  assestamento  : {args.settle_sec}s a robot fermo dopo "
                  f"l'avvio di MOLA")
        if args.warmup > 0:
            print(f"  riscaldamento : {args.warmup} m senza attacco prima di "
                  f"ogni misura")
        print(f"  frame/finestra: ~{frames_per_window:.0f} a regime "
              f"(10 Hz a 0.8 m/s = 8 cm fra uno scan e l'altro)")
        if args.passthrough:
            print(f"  MODALITA' PASSTHROUGH: nessun attacco, solo i nominali")
        else:
            print(f"  valutazioni   : {args.pop} individui x {args.gen} "
                  f"generazioni = {evals}, piu' nominale e applicazione")
        print(f"  stima tempo   : ~{per_window/60:.1f} min/finestra, "
              f"~{per_window*n_windows/60:.0f} min totali")
        print(f"  genoma        : gruppo {args.genome_group}, {n_search} geni attivi su "
              f"{n_genes} (inattivi a -1)")
        lim = node_gen_limits(node)
        print(f"  generatore    : limiti della decodifica stampata da {lim['source']}: dropout max "
              f"{lim['max_dropout_rate']:.3f}, spostamento max {lim['max_point_shift']:.3f} m")
        if LIMITI["mol"]:
            print("  limiti        : moltiplicatore " + ", ".join(f"{f} x{v:g}" for f, v in LIMITI["mol"].items())
                  + f" (nodo: {getattr(node, 'node_moltiplicatori', None) or 'non ancora riportato'})")
        if LIMITI["bloccati"]:
            print(f"  geni bloccati : {', '.join(f'g{g} = -1' for g in LIMITI['bloccati'])} in tutti i "
                  f"candidati (fuori dallo spazio di ricerca)")
        if args.structural is not None or node.node_structural not in (None, "net"):
            print(f"  strutturale   : nodo {node.node_structural or 'non riportato (net)'}"
                  + (f", atteso {args.structural}" if args.structural is not None else ""))
        if args.initial_sigma is not None:
            print(f"  sigma iniziale: {args.initial_sigma} "
                  f"(default pipeline 0.20)")
        if args.lidar_pose_x is not None:
            print(f"  LIDAR_POSE_X  : {args.lidar_pose_x} (sovrascritto)")
        if args.trace:
            print(f"  tracce        : {out/'traces'} e {out/'logs'}")
        if args.window_stop == "ramp":
            print(f"  arresto       : rampa k {args.stop_ramp_k} 1/s, vmin {args.stop_ramp_vmin} m/s"
                  + ("" if args.stop_on == "estimate" else " (senza effetto: --stop-on true)"))
        if args.sampling != "legacy" or args.random_first_admissible > 0 or args.twin_filter != "none":
            print(f"  campionamento : {args.sampling}; random singolo con budget: "
                  f"{args.random_first_admissible or 'no'}; filtro gemelli: {args.twin_filter}")
        print(f"  seed          : {args.seed}"
              + (f" (generatore per finestra: {args.window_rng})" if args.search == "random" else "") + "\n")

        goal = goal_abs
        DRIVE_TIMEOUT[0] = args.drive_timeout

        # Repeatability test (one genome) or operator analysis (--repeat-spec, several genomes):
        # from the same saved state, nominal then N rollouts per genome, following the WHOLE
        # waypoint list (goals_abs), not just the last one.
        if args.repeat_genome or args.repeat_spec:
            entries = []
            if args.repeat_spec:
                with open(args.repeat_spec) as f:
                    for e in json.load(f):
                        entries.append((str(e["label"]), np.array(e["genome"], dtype=np.float64), int(e.get("n", args.repeat_n))))
                print(f"  ANALISI OPERATORI: {len(entries)} genomi da {args.repeat_spec}")
            else:
                parts = args.repeat_genome.split(":")
                src = parts[0]
                widx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                eidx = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
                with open(src) as f:
                    hist_src = json.load(f)
                if eidx is None:
                    g_rep = np.array(hist_src[widx]["genome"], dtype=np.float64)
                    print(f"  TEST RIPETIBILITA': genoma applicato della finestra {widx} di {src}")
                else:
                    g_rep = np.array(hist_src[widx]["evaluations"][eidx]["genome"], dtype=np.float64)
                    print(f"  TEST RIPETIBILITA': genoma del candidato {eidx} della finestra {widx} di {src}")
                entries.append(("rep", g_rep, args.repeat_n))
            print(f"  waypoint seguiti: {[tuple(np.round(g, 3)) for g in goals_abs]}\n")
            wps_all = [tuple(g) for g in goals_abs]

            saved = node.true_pose(isaac)
            isaac.pause()
            isaac.save()
            _, nom_end, _, st, nom_trav, _ = rollout(
                None, isaac, mola, node, list(wps_all), args.horizon, "rep_nominal",
                settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from, stop_on=args.stop_on,
                trace_path=(out / "traces" / "rep_nominal.csv") if args.trace else None)
            rep_settle = node.last_settle_waited
            if nom_end is None or st not in ("ok", "arrived"):
                print(f"  nominale FALLITO ({st})")
                return 1
            print(f"  nominale: {np.round(nom_end,3)}  percorsi {nom_trav:.2f} m\n")
            nom_path = true_path(node.last_trace)

            devs, dirs, dmgs, perts, dev_means, motions = [], [], [], [], [], []
            sweep = []
            for label, g_rep, n_rep in entries:
              if len(entries) > 1:
                  print(f"\n  --- {label} ({n_rep} rollout) ---")
              for r in range(n_rep):
                tag = f"rep_{r}" if label == "rep" else f"sw_{label}_r{r}"
                node.proc_ms_roll = []
                _, end, ch, st, trav, frames = rollout(
                    g_rep, isaac, mola, node, list(wps_all), args.horizon,
                    tag, settle_sec=args.settle_sec, settle_fixed=rep_settle,
                    catchup_sec=args.catchup_sec, attack_from=args.attack_from, stop_on=args.stop_on,
                    trace_path=(out / "traces" / f"{tag}.csv") if args.trace else None)
                rec = {"label": label, "r": r, "status": st, "travelled": trav, "frames": frames, "chamfer": ch,
                       "end": list(map(float, end)) if end is not None else None,
                       "proc_ms_mean": float(np.mean(node.proc_ms_roll)) if node.proc_ms_roll else None,
                       "track_ratio": node.last_track_ratio, "perc": node.last_perc or None,
                       "loc_err_mean_cm": node.last_loc_err_mean * 100 if node.last_loc_err_mean == node.last_loc_err_mean else None,
                       "loc_err_end_cm": node.last_loc_err_end * 100 if node.last_loc_err_end == node.last_loc_err_end else None,
                       "quality_mean": _quality_mean(node.last_trace), "stop_reason": node.last_stop_reason,
                       "wps_reached": getattr(node, "last_wps_reached", None)}
                if end is None or st not in ("ok", "arrived"):
                    print(f"    {tag}: SCARTATO [{st}]  {trav:.2f} m")
                    sweep.append(rec)
                    with open(out / "sweep.json", "w") as f:
                        json.dump(sweep, f, indent=1)
                    continue
                dm = window_damage(end, nom_end, goal, nom_path,
                                   true_path(node.last_trace),
                                   args.damage, args.dir_weight, baseline=baseline_poly)
                dev, dr, dmg = dm["dev_end"], dm["directional"], dm["damage"]
                devs.append(dev); dirs.append(dr); dmgs.append(dmg); perts.append(ch)
                dev_means.append(dm["dev_mean"]); motions.append(drift_fields(node))
                rec.update({"dev_end_cm": dev * 100, "dev_mean_cm": dm["dev_mean"] * 100, "dev_max_cm": dm["dev_max"] * 100,
                            "directional_cm": dr * 100, "damage_cm": dmg * 100, "motion": drift_fields(node),
                            **_window_fields(dm, 100.0, "_cm")})
                sweep.append(rec)
                with open(out / "sweep.json", "w") as f:
                    json.dump(sweep, f, indent=1)
                print(f"    rollout {r}: end {dev*100:5.1f} mean {dm['dev_mean']*100:5.1f} "
                      f"max {dm['dev_max']*100:5.1f}  dir {dr*100:+6.1f}  "
                      f"dmg {dmg*100:6.1f} cm   pert {ch:7.2f}   "
                      f"{trav:.2f} m  {frames:3d}f  [{st}]")

            if len(dmgs) >= 2:
                m, s = np.mean(dmgs) * 100, np.std(dmgs, ddof=1) * 100
                cv = 100 * s / max(m, 1e-6)
                print(f"\n  danno: media {m:.1f} cm   dev.std {s:.1f} cm   "
                      f"CV {cv:.0f}%")
                print(f"  deviazione: {np.mean(devs)*100:.1f} +/- "
                      f"{np.std(devs, ddof=1)*100:.1f} cm")
                print(f"  percettibilita': {np.mean(perts):.2f} +/- "
                      f"{np.std(perts, ddof=1):.2f}")
                print()
                if cv < 20:
                    print("  Ripetibile: un rollout per candidato e' sufficiente.")
                elif cv < 40:
                    print("  Rumore moderato: considerare 2-3 rollout per "
                          "candidato, o accettare e riportare il valore applicato.")
                else:
                    print("  Rumore alto: servono piu' rollout per candidato, "
                          "altrimenti NSGA-III ottimizza le fluttuazioni.")
            with open(out / "repeatability.json", "w") as f:
                json.dump({"genome": g_rep.tolist(), "source": args.repeat_genome,
                           "damage_cm": [d * 100 for d in dmgs],
                           "dev_cm": [d * 100 for d in devs],
                           "dev_mean_cm": [d * 100 for d in dev_means],
                           "motion": motions,
                           "directional_cm": [d * 100 for d in dirs],
                           "perturbation": perts}, f, indent=2)
            return 0

        if args.g_banco:
            # same waypoints as the window rollouts (with direction: the run line)
            return _g_banco_run(args, isaac, mola, node, out,
                                [dir_line[0], goals_abs[-1]] if direction else list(goals_abs))

        if args.map_gate_test:
            print("── COLLAUDO ISOLATO map-restore " + "─" * 30)
            if node._map_restore_import_error:
                print(f"  SERVIZI MAP NON DISPONIBILI (import): {node._map_restore_import_error}")
                return
            import glob as _glob
            node.true_pose(isaac); isaac.pause(); isaac.save()
            gh = min(3.0, max(1.0, args.horizon * 3))
            try:
                rollout(None, isaac, mola, node, [goals_abs[0]], gh, "gate_drive",
                        settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                        attack_from=args.attack_from, stop_on=args.stop_on, victim=victim)
            except Exception as e:
                print(f"  guida vittima fallita: {e}"); isaac.pause(); mola.stop(); return
            vend = victim.last_est_end if victim is not None else None
            print(f"  vittima guidata ~{gh:.1f} m; posa creduta vittima = "
                  f"{np.round(vend, 3) if vend is not None else None}; frame posa = {node.victim_est_frame_id}")
            # New savepoint where the victim left the robot: the gate twin will be restored HERE
            # (same physical pose as the saved map), so the twin scan against the loaded map is
            # consistent.
            node.true_pose(isaac); isaac.pause(); isaac.save()
            mp = out / "gate_map"; mp.parent.mkdir(parents=True, exist_ok=True)
            ok_s, err_s = node.victim_map_save(mp)
            print(f"  /victim/map_save -> {'OK' if ok_s else 'FALLITO'} ({mp if ok_s else err_s}); "
                  f"file: {_glob.glob(str(mp) + '*')}")
            if not ok_s:
                print("  GATE: map_save fallito -> map-restore non utilizzabile su questa config.")
                isaac.pause(); mola.stop(); return
            TWIN_STATE.update({"mode": "maprestore", "victim": victim,
                               "victim_map_path": str(mp), "window": 0,
                               "reloc_frame": args.reloc_frame})
            try:
                rollout(None, isaac, mola, node, [goals_abs[0]], 0.5, "gate_twin",
                        settle_sec=0.0, catchup_sec=args.catchup_sec,
                        attack_from="spawn", stop_on=args.stop_on, victim=None)
            except Exception as e:
                print(f"  rollout twin gate (non critico): {e}")
            rep = node.last_replay or {}
            print(f"  GATE map-restore: status={rep.get('status')} map_load_ok={rep.get('map_load_ok')} "
                  f"relocalize_accepted={rep.get('relocalize_accepted')}")
            print(f"  >>> FEDELTA' ripristino: diff_cm={rep.get('diff_cm')} diff_deg={rep.get('diff_deg')} "
                  f"twin_quality={rep.get('twin_quality')} (scarto twin-vittima; piccolo = ripristino fedele)")
            isaac.pause(); mola.stop(); return
        wp_idx = 0
        k = -1
        frame_verificato = False       # frame consistency with the baseline: once only
        nom_quality, nom_it, nom_stats, nom_perc = None, None, {}, {}
        nom_assente = None  # --g-nominale-senza-posa continua: entry of the current window
        victim_lost = False
        scoperto = None                # --chiudi-attacco-scoperto: (window record, run_end)
        catena_t0 = None               # window start for the chain check

        def _esito_finestra(rec, v_status):
            # Per-window record: victim pose loss (also in skipped windows, where victim_lost does
            # not stop the run) and frame origin.
            nonlocal scoperto
            wrec.mark_victim_lost(rec, v_status, k, lost_state)
            if CONTROLLO_ETA["soglia"] is not None:
                # --controllo-eta: attempts, repeats and repeated failures of the window
                rec["controllo_eta"] = CONTROLLO_ETA["finestre"].get(str(STALL_POLICY.get("window")))
            if (args.chiudi_attacco_scoperto and victim is not None and scoperto is None
                    and wrec.vittima_persa(v_status)):
                # victim without pose: the run closes at the next iteration, as attack detected if
                # the window chain is healthy (or --controllo-catena off), otherwise as a chain
                # failure (to repeat with the same seed)
                tb = getattr(node, "true_buf", None)
                catena = (controlla_catena(args, node, victim, k, catena_t0)
                          if getattr(args, "controllo_catena", False) else None)
                # victim MOLA process (alive, terminated, not found) and rejection signs in the
                # printf lines read now; recorded only, the outcome does not change
                proc_v = campi_vittima_persa(victim)
                if catena is None:
                    scoperto = (rec, {**wrec.attacco_scoperto_record(
                        k, v_status, wrec.read_printf(getattr(victim, "log_path", None)),
                        getattr(victim, "last_est_end", None), getattr(node, "true", None),
                        tb[-1][0] if tb else None, time.time()), **proc_v})
                elif catena["guasto"]:
                    scoperto = (rec, {**wrec.guasto_catena_record(
                        k, v_status, catena, (catena_t0 or {}).get("wall"),
                        tb[-1][0] if tb else None, time.time(), args.seed),
                        "stamp_ultima_posa_ricevuta": getattr(node, "victim_est_stamp", None),
                        **proc_v})
                    if rec.get("danno_escluso"):
                        rec["danno_escluso"] = "guasto_catena"
                else:
                    scoperto = (rec, {**wrec.attacco_scoperto_record(
                        k, v_status, wrec.read_printf(getattr(victim, "log_path", None)),
                        getattr(victim, "last_est_end", None), getattr(node, "true", None),
                        tb[-1][0] if tb else None, time.time()), "controllo_catena": catena,
                        "rallentamento_mola": catena["rallentamento_mola"],
                        "stamp_ultima_posa_ricevuta": getattr(node, "victim_est_stamp", None),
                        **proc_v})
            if nom_assente is not None and nom_assente.get("window") == k:
                if nom_assente.get("senza_nominale"):
                    # --no-nominal-rollout: window entry and twin G outcomes
                    rec["nominale_assente"] = nom_assente
                    if "evaluations" in rec:
                        rec["esiti_gemello"] = tv.conta_esiti_gemello(rec["evaluations"])
                else:
                    rec["nominale_non_localizza"] = nom_assente
            if victim is not None:
                rec["run_origin"] = victim.origin_info
            if victim is not None and args.victim_start == "w0":
                rec["victim_start"] = {"mode": "w0", "started": victim.started, **(victim.start_info or {})}
                if shadow_rec is not None and victim.sync_info and "avvio_sincrono" not in shadow_rec.config:
                    shadow_rec.config["avvio_sincrono"] = victim.sync_info
                    shadow_rec.save()
            if args.settle_clouds > 0:
                rec["settle_clouds"] = {"n": args.settle_clouds, "reopen": args.reopen_clouds,
                                        "victim": getattr(node, "last_victim_settle_clouds", None)}
            if args.victim_never_clean:
                rec["never_clean"] = never_clean_fields(node)
            if direction:
                rec["goal_mode"] = _goal_mode_rec()
            return rec

        def _goal_mode_rec():
            # Driven or held window: victim_held in the record says which.
            return {"mode": "direction", "lookahead_m": args.direction_lookahead,
                    "line_start": list(dir_line[0]), "line_dir": [float(v) for v in dir_line[1]],
                    "max_windows": args.max_windows, "max_held": args.extra_windows}
        warm_pool = None  # Pareto front (search-space, damage, Chamfer, idx) of the previous window
        warm_src = None    # warm_pool source: window, eval_log indices, time of computation
        warm_best = None   # admissible, by damage, previous window (best N/2)
        first_win_time = None  # time of the first completed window (watchdog reference)
        win_cap = None
        while True:
            if scoperto is not None:
                # --chiudi-attacco-scoperto: run closed with the final record in the window where
                # the victim lost the pose
                rec_s, fine_s = scoperto
                fine_s.update(zip(("windows_driven", "windows_held"),
                                  direction_window_count(history, k + 1) if direction else (k + 1, 0)))
                rec_s["run_end"] = fine_s
                if fine_s["reason"] == "guasto_catena":
                    guasto_catena = fine_s
                    # marker for run_campaign.sh and the queue: repetition to redo with the same
                    # seed
                    with open(out / "da_rifare.json", "w") as f:
                        json.dump({k_: fine_s.get(k_) for k_ in ("reason", "window", "status", "motivi",
                                                                 "seme", "wall_iso", "rallentamento_mola")},
                                  f, indent=1, ensure_ascii=False)
                    print(f"  GUASTO DELLA CATENA nella finestra {k + 1} [{fine_s['status']}]: "
                          f"{'; '.join(fine_s['motivi'])}. Non e' un attacco scoperto: run da rifare "
                          f"con lo stesso seme ({args.seed}); la run si ferma qui")
                    break
                print(f"  ATTACCO SCOPERTO: vittima senza posa nella finestra {k + 1} [{fine_s['status']}], "
                      f"ultima posa allo stamp {fine_s['stamp_ultima_posa']}; la run si ferma qui")
                break
            if victim_lost:
                # The victim no longer publishes a pose: without localisation the robot cannot go
                # on. It is the maximum outcome of the attack on a continuous SLAM and closes the
                # run; the remaining distance to the goal stays in the history (final_dist of the
                # aggregator).
                print(f"  vittima senza posa: la run si ferma qui, bersaglio a "
                      f"{math.hypot(goals_abs[-1][0] - saved[0], goals_abs[-1][1] - saved[1]):.2f} m")
                break
            k += 1
            STALL_POLICY["window"] = k
            TWIN_STATE["window"] = k
            TWIN_STATE["ref"] = {}
            # Window start with the robot still: a transient stall here touches no measurement and
            # is tolerated as in the twin rollouts.
            for _attempt in range(2):
                try:
                    saved = node.true_pose(isaac)
                    isaac.pause()
                    saved_scene = isaac.save()
                    break
                except IsaacStall as e:
                    if _attempt == 1:
                        raise
                    _isaac_recovery(isaac, node, e, f"inizio finestra {k + 1}")
            # --genoma-nullo-se-nessun-ammissibile: victim |A| at metre start (victim still since
            # the end of the previous stretch)
            deriva_inizio = deriva_vittima_status(node) if args.genoma_nullo_se_nessun_ammissibile else None
            if direction:
                # Fixed number of driven windows; where the robot stops is a result. No --wp-tol and
                # no window estimate.
                fine = direction_run_end(history, k, args.max_windows, args.extra_windows)
                d_last = math.hypot(goals_abs[-1][0] - saved[0], goals_abs[-1][1] - saved[1])
                if fine is not None:
                    driven, held = direction_window_count(history, k)
                    print(f"  fine della run ({fine}): {driven} finestre guidate, {held} trattenute")
                    if history:
                        history[-1]["run_end"] = {"reason": fine, "windows_driven": driven,
                                                  "windows_held": held}
                    break
                wps = [dir_line[0], goals_abs[-1]]
                goal = direction_window_goal(dir_line[0], dir_line[1], saved[:2],
                                             args.direction_lookahead)
            else:
                # Waypoint advance on the true pose: the task stop criterion does not depend on the
                # attacked estimate.
                while wp_idx < len(goals_abs) - 1 and math.hypot(
                        goals_abs[wp_idx][0] - saved[0], goals_abs[wp_idx][1] - saved[1]) < args.wp_tol:
                    wp_idx += 1
                d_last = math.hypot(goals_abs[-1][0] - saved[0], goals_abs[-1][1] - saved[1])
                if d_last < args.wp_tol:
                    print(f"  bersaglio finale raggiunto ({d_last:.2f} m): fine del task")
                    break
                if k >= n_windows + args.extra_windows:
                    print(f"  limite di finestre raggiunto ({k}), bersaglio a {d_last:.2f} m: fine")
                    break
                if args.max_windows is not None and k >= args.max_windows:
                    print(f"  --max-windows {args.max_windows} raggiunto: fine")
                    break
                wps = goals_abs[wp_idx:]
                goal = wps[0]
            print(f"── finestra {k+1} (stima {n_windows}) " + "─" * 40)
            print(f"  waypoint corrente {wp_idx+1}/{len(goals_abs)} {np.round(goal,3)}  "
                  f"distanza al bersaglio finale {d_last:.2f} m")
            t_win = time.time()
            if (LIMITI["mol"] or getattr(node, "node_moltiplicatori", None)) and node.node_gen_limits is not None:
                # limit multiplier: the node must have the same one (even without zones)
                controlla_moltiplicatori(node, f"finestra {k}")
            if args.chiudi_attacco_scoperto and args.controllo_catena and victim is not None:
                catena_t0 = {"wall": t_win, "stamp_tracce": wrec.ultimo_stamp_tracce(
                    wrec.leggi_csv(getattr(victim, "traces_path", None)))}
            print(f"  stato salvato in {np.round(saved,3)}")
            node.last_victim_settle_clouds = None
            # --victim-start w0: before the victim starts the shadow does not exist and the twins
            # start from zero, as the victim will
            pre_start = victim is not None and victim.deferred is not None and not victim.started
            if shadow_rec is not None and not pre_start:
                # Shadow-victim comparison with the robot still: bench measurement, no field enters
                # TWIN_STATE or the decisions.
                fp = shadow_rec.snapshot(window=k).get("finestra_precedente") or {}
                shadow_rec.fill_history(history)
                with open(out / "history.json", "w") as f:
                    json.dump(history, f, indent=2)
                print(f"  ombra-vittima a inizio finestra {k}: livello {fp.get('livello_peggiore')} "
                      f"(ultimo scan {fp.get('livello_ultimo_scan')}), comuni {fp.get('scan_comuni')}, "
                      f"marcata {fp.get('marcata')}")
            # Twin start frame: pose believed by the attacker, or the true one; with --twin-state
            # victim, the pose the victim believes it has.
            frame_start = attacker.believed_pose(node) if attacker is not None else None
            if ZONE["on"]:
                node.fotografia_metro(k, frame_start)
            if canale_rampa():
                prepara_rampa_metro(node, k)
            if GIM["S"] > 0 or GIM["cpf"]:
                gim_inizio_finestra()
            TWIN_STATE.update({"mode": args.twin_state, "last_s": args.replay_last_s,
                               "rate_hz": args.replay_rate_hz, "victim": victim,
                               "reloc_frame": args.reloc_frame,
                               "replay_frame": args.replay_frame,
                               "attacker_origin": attacker.run_origin if attacker is not None else None,
                               "attacker_belief": frame_start, "saved": saved,
                               "init": args.twin_init, "init_target": None})
            if args.twin_init == "attacker_est" and frame_start is not None:
                TWIN_STATE["init_target"] = attacker_init_target(node, attacker, saved, saved_scene,
                                                                 frame_start, args.twin_init_vel)
                ti = TWIN_STATE["init_target"]
                print(f"  gemello dalla stima dell'attaccante: scena {np.round(ti['target_scene'], 3)}, "
                      f"velocita' {ti['rule']} (v {ti['v_att']}, w {ti['w_att']})")
            if args.twin_state in ("victim", "maprestore") and victim is not None \
                    and victim.last_est_end is not None and victim.run_origin is not None:
                frame_start = node.compose(victim.run_origin, victim.last_est_end)
            if args.twin_state == "maprestore" and victim is not None and victim.last_est_end is not None:
                # Map restore: save NOW the map accumulated by the victim; the candidates of this
                # window will reload it into the twin (map_load + relocalize).
                mp = out / "maps" / f"victim_w{k}"
                mp.parent.mkdir(parents=True, exist_ok=True)
                ok_s, err_s = node.victim_map_save(mp)
                TWIN_STATE["victim_map_path"] = str(mp) if ok_s else None
                print(f"  map_save vittima finestra {k}: {'ok' if ok_s else 'FALLITO'} "
                      f"({mp if ok_s else err_s})")
            if pre_start and args.twin_state in tv.SHADOW_MODES:
                TWIN_STATE["mode"] = "clean"
                TWIN_STATE["shadow_state"] = {"status": "ombra_non_avviata", "window": k}
                print(f"  ombra non ancora avviata (--victim-start w0): gemelli da zero in questa finestra")
            elif shadow is not None and args.twin_state in tv.SHADOW_MODES:
                # Variant E: with the robot still and the stream closed, shadow map_save and its
                # pose and sigma; the twin frame is the pose believed by the shadow. True pose and
                # victim state only in the lab_* fields.
                mp = out / "maps" / f"ombra_w{k}"
                mp.parent.mkdir(parents=True, exist_ok=True)
                sh = shadow.save_state(node, mp)
                if sh.get("believed") is not None:
                    frame_start = tuple(sh["believed"])
                    sh["lab_true_saved"] = [float(v) for v in saved]
                    sh["lab_shadow_err_cm"] = round(100 * math.hypot(frame_start[0] - saved[0],
                                                                     frame_start[1] - saved[1]), 3)
                    if victim.last_est_end is not None and victim.run_origin is not None:
                        vb = node.compose(victim.run_origin, victim.last_est_end)
                        sh["lab_victim_believed"] = [float(v) for v in vb]
                        sh["lab_shadow_vs_victim_cm"] = round(100 * math.hypot(frame_start[0] - vb[0],
                                                                               frame_start[1] - vb[1]), 3)
                TWIN_STATE["shadow_state"] = sh
                if args.twin_state == "ombra_g":
                    print(f"  ombra G finestra {k}: .molastate {sh.get('state_bytes')} B, sha256 "
                          f"{str(sh.get('state_sha256'))[:12]}, stamp {sh.get('state_stamp_tick')} "
                          f"(posa {sh.get('pose_stamp_tick')}, diff {sh.get('stamp_diff_ticks')} tick)")
                print(f"  ombra finestra {k}: {sh.get('status')} (.mm {sh.get('mm_bytes')} B, sigma "
                      f"{sh.get('sigma')}{'' if sh.get('sigma_aligned') else ' NON allineata'}, posa creduta "
                      f"{np.round(frame_start, 3) if frame_start is not None else None})")
                if args.twin_init == "ombra_est":
                    # twin physical pose from the pose believed by the shadow: physical and internal
                    # pose from the same estimator; without a believed pose it stays None and the
                    # twin starts from the saved state, as with --twin-init true
                    TWIN_STATE["init_target"] = (attacker_init_target(
                        node, shadow, saved, saved_scene, tuple(sh["believed"]), args.twin_init_vel,
                        buf_name="shadow_buf") if sh.get("believed") is not None else None)
                    if TWIN_STATE["init_target"] is not None:
                        TWIN_STATE["init_target"]["stimatore"] = "ombra"
            if args.twin_init_assetto != "azzerato" and TWIN_STATE.get("init_target") is not None:
                # twin roll and pitch in set_state, with their source (ErroreAssetto stops the run)
                kw_a, rec_a = assetto_gemello(args.twin_init_assetto, node)
                TWIN_STATE["init_target"]["assetto_set_state"] = kw_a
                TWIN_STATE["init_target"]["assetto"] = rec_a
                print(f"  assetto del gemello ({args.twin_init_assetto}): {kw_a}")
            if args.replay_gate > 0 and victim is not None and victim.last_est_end is not None \
                    and node.victim_rec_n > 0:
                gate = []
                for ls in [float(x) for x in args.replay_gate_last_s.split(",") if x.strip()]:
                    for i in range(args.replay_gate):
                        try:
                            rec = _gate_replay_once(k, ls, i, isaac, mola, node, victim, saved, args)
                        except IsaacStall as e:
                            _isaac_recovery(isaac, node, e, f"gate w{k} {ls:g} s #{i}")
                            rec = {"last_s": ls, "i": i, "status": "isaac_stall_transient"}
                        gate.append(rec)
                        print(f"  gate rigioco {ls:g} s #{i}: {rec.get('status')} nuvole {rec.get('n')} "
                              f"in {rec.get('replay_s')} s, scarto stima twin-vittima {rec.get('diff_cm')} cm "
                              f"{rec.get('diff_deg')}°, gap vero twin {rec.get('twin_gap_cm')} / vittima {rec.get('victim_gap_cm')} cm, "
                              f"qualita' {rec.get('twin_quality')} / {rec.get('victim_quality_end')}")
                if history:
                    history[-1]["replay_gate"] = gate
                    with open(out / "history.json", "w") as f:
                        json.dump(history, f, indent=2)
            if frame_start is not None:
                att_err_cm = 100 * math.hypot(frame_start[0] - saved[0], frame_start[1] - saved[1])
                att_err_deg = math.degrees(angle_diff(frame_start[2], saved[2]))
                print(f"  attaccante crede {np.round(frame_start,3)}: scarto dalla posa vera "
                      f"{att_err_cm:.1f} cm, {att_err_deg:+.1f}°")
            else:
                att_err_cm = att_err_deg = None

            # Nominal: same rollout without perturbation, reference for the deviation. Recomputed
            # per window because the geometry changes. A failed nominal is retried once from the
            # same state; if it fails again the window is skipped, but with the continuous victim
            # the stretch must still be driven by it (without attack): leaving the robot where the
            # failed twin took it would make the victim see a teleport on resume. With
            # --no-nominal-rollout the nominal is not run (no attempt).
            nom_start = nom_end = None
            st, nom_trav, nom_frames, attempt = "senza_nominale", 0.0, 0, 0
            for attempt in (() if args.no_nominal_rollout else (1, 2)):
                nom_start, nom_end, _, st, nom_trav, nom_frames = rollout(
                    None, isaac, mola, node, wps, args.horizon, f"w{k}_nominal",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from, stop_on=args.stop_on,
                    drift_reset_on=args.drift_reset_on, frame_start=frame_start,
                    trace_path=(out / "traces" / f"w{k}_nominal.csv")
                    if args.trace else None)
                if nom_end is not None and st in ("ok", "arrived"):
                    break
                print(f"  nominale FALLITO ({st}) al tentativo {attempt}")
            # --g-nominale-senza-posa continua: loaded twin G that does not localise on the clean
            # cloud; the window goes on without nominal (damage from the baseline)
            nom_assente = tv.g_nominale_senza_posa(args, st, nom_end, getattr(mola, "log_path", None),
                                                   k, attempt, time.time())
            if nom_assente is not None and nom_assente.get("senza_nominale"):
                print("  nessun nominale del gemello (--no-nominal-rollout): danno dalla baseline")
            elif nom_assente is not None:
                print(f"  nominale non localizza (gemello G caricato, q max {nom_assente['q_max']} "
                      f"su {nom_assente['q_n']} ICP): si valutano i candidati")
            if nom_assente is None and (nom_end is None or st not in ("ok", "arrived")):
                g_lines = None
                if args.g_twin_lost_ends_run:
                    lp = getattr(mola, "log_path", None)
                    try:
                        g_lines = tv.g_twin_lost_lines(Path(lp).read_text(errors="replace") if lp else "")
                    except OSError:
                        g_lines = None
                if args.g_twin_lost_ends_run and tv.g_twin_lost(st, args.twin_state, g_lines):
                    # Variant G: state loaded but no pose. In the threat model this is the lost
                    # victim: the run closes, the window is not held.
                    rec = _esito_finestra(
                        skipped_window_record(k, st, goal, wp_idx, None, time.time() - t_win,
                                              TWIN_STATE.get("shadow_state")), None)
                    rec["victim_lost_g"] = tv.g_twin_lost_record(st, g_lines, getattr(mola, "log_path", None))
                    driven, held = direction_window_count(history, k) if direction else (k, 0)
                    rec["run_end"] = {"reason": "victim_lost_g", "windows_driven": driven,
                                      "windows_held": held}
                    history.append(rec)
                    with open(out / "history.json", "w") as f:
                        json.dump(history, f, indent=2)
                    print(f"  fine della run (victim_lost_g): gemello G caricato senza posa, "
                          f"icp_rejected {g_lines.get('icp_rejected')}, consecutive_bad "
                          f"{g_lines.get('consecutive_bad_last')}; {driven} finestre guidate, {held} trattenute")
                    break
                print("  finestra saltata")
                hold = None
                if victim is not None and args.victim_never_clean:
                    hold = victim_hold_fields("nominale_fallito", victim)
                    v_st = None
                    print("  vittima ferma nello stato salvato (--victim-never-clean: nessuna guida senza genoma)")
                elif victim is not None:
                    _, v_end, _, v_st, v_trav, _ = rollout(
                        None, isaac, mola, node, wps, args.horizon, f"w{k}_victim",
                        settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                        attack_from=args.attack_from, stop_on=args.stop_on,
                        settle_fixed=node.last_settle_waited, victim=victim)
                    print(f"  tratto percorso dalla vittima senza attacco: {v_trav:.2f} m [{v_st}]")
                history.append(_esito_finestra(
                               skipped_window_record(k, st, goal, wp_idx,
                                                     v_st if victim is not None else None,
                                                     time.time() - t_win, TWIN_STATE.get("shadow_state")),
                               v_st if victim is not None else None))
                if hold is not None:
                    history[-1].update(hold)
                    history[-1]["hold_restore"] = hold_restore(isaac, mola, node, k, saved)
                with open(out / "history.json", "w") as f:
                    json.dump(history, f, indent=2)
                continue
            nom_path = true_path(node.last_trace) if nom_assente is None else []
            # With --damage-ref baseline the nominal does not enter the damage; its distance from
            # the polyline is recorded for comparison.
            nom_base_fields = {}
            if baseline_poly is not None and len(nom_path) > 0:
                nb = baseline_damage(nom_path, baseline_poly)
                nom_base_fields = {"nominal_baseline_dev_mean_cm": nb["dev_mean"] * 100,
                                   "nominal_baseline_dev_end_cm": nb["dev_end"] * 100,
                                   "nominal_baseline_dev_max_cm": nb["dev_max"] * 100,
                                   "nominal_window": wrec.baseline_window_fields(nom_path, baseline_poly)}
                print(f"  nominale rispetto alla baseline: media {nb['dev_mean']*100:.1f} cm, "
                      f"fine {nb['dev_end']*100:.1f} cm, max {nb['dev_max']*100:.1f} cm")
                if not frame_verificato:
                    # Frame consistency, once only: in the first window with a valid nominal (if
                    # window 0 fails or has no true poses, the next one is used).
                    frame_verificato = True
                    chk_cm = nb["dev_mean"] * 100
                    frame_ok = bool(chk_cm <= BASELINE_FRAME_MAX_CM)   # NaN -> False
                    nom_base_fields.update({"baseline_frame_check_cm": chk_cm,
                                            "baseline_frame_max_cm": BASELINE_FRAME_MAX_CM,
                                            "baseline_frame_ok": frame_ok})
                    if not frame_ok:
                        print(f"ERRORE: il nominale della finestra {k} dista {chk_cm:.1f} cm dalla "
                              f"baseline: frame non coincidente o baseline sbagliata")
                        history.append({"window": k, "status": "baseline_frame_mismatch",
                                        "goal": list(map(float, goal)), "waypoint_index": wp_idx,
                                        "damage_ref": args.damage_ref, "baseline_path": args.baseline_path,
                                        "nominal_end": list(map(float, nom_end)),
                                        **nom_base_fields, "window_time_s": time.time() - t_win})
                        # Clean exit: the finally of main closes node, MOLA, victim and Isaac and
                        # saves the history.
                        return 2
            senza_nom = nom_assente is not None and bool(nom_assente.get("senza_nominale"))
            if senza_nom and baseline_poly is not None and not frame_verificato:
                # --no-nominal-rollout: frame consistency of the true pose saved at the start of w0
                # (/chassis/odom) against the polyline. A single point, without direction: a weaker
                # check than the one on the nominal.
                frame_verificato = True
                chk_cm = float(dist_to_polyline(np.array([saved[:2]], dtype=float), baseline_poly)[0]) * 100
                frame_ok = bool(chk_cm <= BASELINE_FRAME_MAX_CM)   # NaN -> False
                nom_base_fields.update({"baseline_frame_check_cm": chk_cm,
                                        "baseline_frame_max_cm": BASELINE_FRAME_MAX_CM,
                                        "baseline_frame_ok": frame_ok,
                                        "baseline_frame_check_from": "posa_vera_salvata"})
                if not frame_ok:
                    print(f"ERRORE: la posa vera salvata della finestra {k} dista {chk_cm:.1f} cm dalla "
                          f"baseline: frame non coincidente o baseline sbagliata")
                    history.append({"window": k, "status": "baseline_frame_mismatch",
                                    "goal": list(map(float, goal)), "waypoint_index": wp_idx,
                                    "damage_ref": args.damage_ref, "baseline_path": args.baseline_path,
                                    "nominal_end": None, "saved": [float(v) for v in saved],
                                    **nom_base_fields, "window_time_s": time.time() - t_win})
                    return 2
            if nom_assente is not None:
                # No nominal: no reference for the internal perceptibility (internal f2 undefined),
                # for the fixed settling or for the estimate/truth ratio. Candidates stay subject to
                # the untracked check (track_ref not None) and count as in a window with a regular
                # nominal (nominal_ok True).
                nom_quality, nom_perc, nom_stats, nom_it = None, {}, {}, None
                nominal_ok = True
                nom_motion = nom_twin_start = None
                nom_loc_mean = nom_loc_end = nom_loc_mean_raw = nom_loc_end_raw = None
                # --no-nominal-rollout: constant non-None reference (the threshold is absolute)
                nom_ratio = tv.TRACK_REF_SENZA_NOMINALE if senza_nom else float("nan")
                nom_settle_s = nom_settle_ok = None
            else:
                # Internal signals of the nominal (reference for the internal perceptibility): mean
                # quality from the topic, iterations from the MOLA CSV.
                nom_quality = node.last_quality_mean
                nom_perc = dict(node.last_perc or {})
                nom_stats = _mola_csv_stats(getattr(mola, "traces_path", None),
                                            *getattr(node, "last_drive_stamps", (float("nan"), float("nan"))))
                nom_it = nom_stats.get("it_mean")
                if args.perceptibility == "internal":
                    print(f"  nominale: qualita' {nom_quality}, iterazioni {nom_it} (copertura CSV {nom_stats.get('coverage')})")
                # Anomalous nominal: estimate/truth ratio missing or below the tracking threshold
                # (MOLA lost or froze the estimate without attack, e.g. an estimate frozen for 2.2 s
                # with a 49 cm gap). An initial rotation towards the goal is not anomalous.
                nominal_ok = (node.last_track_ratio == node.last_track_ratio
                              and node.last_track_ratio >= 0.60)
                nom_motion = drift_fields(node)
                nom_twin_start = _twin_start_fields(node)
                if not nominal_ok:
                    print(f"  ATTENZIONE: nominale anomalo (rapporto stima/verita' "
                          f"{node.last_track_ratio:.2f} < 0.60): i candidati di questa "
                          f"finestra vengono scartati")
                nom_loc_mean = node.last_loc_err_mean * 100 if node.last_loc_err_mean == node.last_loc_err_mean else None
                nom_loc_end = node.last_loc_err_end * 100 if node.last_loc_err_end == node.last_loc_err_end else None
                nom_loc_mean_raw = node.last_loc_err_mean_raw * 100 if node.last_loc_err_mean_raw == node.last_loc_err_mean_raw else None
                nom_loc_end_raw = node.last_loc_err_end_raw * 100 if node.last_loc_err_end_raw == node.last_loc_err_end_raw else None
                nom_ratio = node.last_track_ratio
                nom_settle_s = node.last_settle_waited
                nom_settle_ok = node.last_settle_ok
                nom_yaw_deg = math.degrees(angle_diff(nom_end[2], saved[2]))
                # Expected rotation: towards the goal from the saved state. After an attacked window
                # the robot may be skewed and the nominal must turn to get back on course: not an
                # anomaly.
                bearing_deg = math.degrees(angle_diff(
                    math.atan2(goal[1] - saved[1], goal[0] - saved[0]), saved[2]))
                rst = math.hypot(nom_start[0] - saved[0], nom_start[1] - saved[1])
                print(f"  nominale: parte da {np.round(nom_start,3)} (ripristino {rst*100:.1f} cm "
                      f"dallo stato salvato)  arriva in {np.round(nom_end,3)}  "
                      f"percorsi {nom_trav:.2f} m  {nom_frames} frame  "
                      f"{len(nom_path)} pose vere  rotazione {nom_yaw_deg:+.0f}° "
                      f"(bersaglio a {bearing_deg:+.0f}°)  "
                      f"rapporto stima/verita' {node.last_track_ratio:.2f}  [{st}]")
                if abs(nom_yaw_deg) > 30 and abs(bearing_deg) < 15:
                    print("  ATTENZIONE: il nominale ruota molto con il bersaglio "
                          "davanti. MOLA non e' a regime: alza --warmup o --horizon.")

            if args.passthrough:
                # Per-window noise floor: more nominals from the same saved state, with the same
                # settling as the candidates. The spread among identical nominals is the noise below
                # which the damage of a window cannot be distinguished. The robot stays where the
                # last nominal takes it, as in the single-nominal flow.
                nom_reps = []
                for j in range(2, args.nominal_reps + 1):
                    _, r_end, _, r_st, r_trav, r_frames = rollout(
                        None, isaac, mola, node, wps, args.horizon, f"w{k}_nominal_r{j}",
                        warmup=args.warmup, settle_sec=args.settle_sec, settle_fixed=nom_settle_s,
                        catchup_sec=args.catchup_sec, attack_from=args.attack_from, stop_on=args.stop_on,
                        drift_reset_on=args.drift_reset_on, frame_start=frame_start,
                        trace_path=(out / "traces" / f"w{k}_nominal_r{j}.csv")
                        if args.trace else None)
                    rec_r = {"rep": j, "status": r_st, "travelled": r_trav, "frames": r_frames,
                             "end": list(map(float, r_end)) if r_end is not None else None,
                             "track_ratio": node.last_track_ratio}
                    if r_end is not None and r_st in ("ok", "arrived"):
                        dm_r = window_damage(r_end, nom_end, goal, nom_path,
                                             true_path(node.last_trace), args.damage, args.dir_weight)
                        rec_r["dev_mean_cm"] = dm_r["dev_mean"] * 100
                        rec_r["dev_end_cm"] = dm_r["dev_end"] * 100
                        print(f"  nominale ripetuto {j}: {r_trav:.2f} m  scarto dal primo nominale: "
                              f"media {rec_r['dev_mean_cm']:.1f} cm, fine {rec_r['dev_end_cm']:.1f} cm  [{r_st}]")
                    else:
                        print(f"  nominale ripetuto {j}: FALLITO [{r_st}]")
                    nom_reps.append(rec_r)
                # Nominals only: checks that the robot drives the stretch straight without any
                # perturbation. Goal and time feed the "none" arm of the campaign (final distance,
                # duration).
                history.append({"window": k,
                                "damage_ref": args.damage_ref,
                                "baseline_path": args.baseline_path if baseline_poly is not None else None,
                                **nom_base_fields,
                                "nominal_end": list(map(float, nom_end)),
                                "travelled": nom_trav, "frames": nom_frames,
                                "status": st,
                                "goal": list(map(float, goal)),
                                "waypoint_index": wp_idx,
                                "search": "none",
                                "nominal_twin_start": nom_twin_start, **twin_config_fields(args),
                                "nominal_reps": nom_reps,
                                "loc_err_mean_cm": nom_loc_mean,
                                "loc_err_end_cm": nom_loc_end,
                                "loc_err_mean_raw_cm": nom_loc_mean_raw,
                                "nominal_track_ratio": nom_ratio,
                                "nominal_settle_s": node.last_settle_waited,
                                "nominal_settle_ok": node.last_settle_ok,
                                **drift_fields(node),
                                "window_time_s": time.time() - t_win})
                if victim is not None:
                    # The same stretch driven by the continuous victim from the same saved state:
                    # the actual traversal of the "none" arm, and the comparison with the twin
                    # nominal measures how much the restarted SLAM differs from the steady-state
                    # one.
                    if args.pause_sec > 0:
                        # Pause as during a search: robot still, victim stream closed, simulation
                        # paused.
                        print(f"  sosta di {args.pause_sec:.0f} s a robot fermo (baseline con le soste dell'orizzonte)")
                        t_ps = time.time()
                        while time.time() - t_ps < args.pause_sec:
                            node.spin(min(1.0, args.pause_sec - (time.time() - t_ps)))
                    _, v_end, _, v_st, v_trav, v_frames = rollout(
                        None, isaac, mola, node, wps, args.horizon, f"w{k}_victim",
                        settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                        attack_from=args.attack_from, stop_on=args.stop_on,
                        settle_fixed=node.last_settle_waited, victim=victim,
                        trace_path=(out / "traces" / f"w{k}_victim.csv") if args.trace else None)
                    rec = history[-1]
                    rec.update({"victim": "continuous", "victim_status": v_st,
                                "twin_align": args.twin_align if attacker is not None else "true",
                                "attacker_align_err_cm": att_err_cm, "attacker_align_err_deg": att_err_deg,
                                "victim_end": list(map(float, v_end)) if v_end else None,
                                "victim_travelled": v_trav, "victim_frames": v_frames,
                                "victim_track_ratio": node.last_track_ratio,
                                "victim_loc_err_mean_cm": node.last_loc_err_mean * 100 if node.last_loc_err_mean == node.last_loc_err_mean else None,
                                "victim_quality_mean": _quality_mean(node.last_trace),
                                "victim_settle_s": victim.settle_s,
                                "victim_motion": motion_vittima(node),
                                "victim_window": _victim_window_record(node, victim, baseline_poly)})
                    _esito_finestra(rec, v_st)
                    _refresh_victim_scans(history[:-1], victim)
                    if v_end is not None and v_st in VALID:
                        dm_v = window_damage(v_end, nom_end, goal, nom_path,
                                             true_path(node.last_trace), args.damage, args.dir_weight,
                                             baseline=baseline_poly)
                        # The offset from the twin stays the per-window comparison; with the
                        # baseline the distance from the polyline is also recorded.
                        rec["victim_vs_twin_dev_mean_cm"] = dm_v.get("dev_mean_window", dm_v["dev_mean"]) * 100
                        rec["victim_vs_twin_dev_end_cm"] = dm_v.get("dev_end_window", dm_v["dev_end"]) * 100
                        if baseline_poly is not None:
                            rec["victim_baseline_dev_mean_cm"] = dm_v["dev_mean"] * 100
                            rec["victim_baseline_dev_end_cm"] = dm_v["dev_end"] * 100
                        print(f"  vittima continua: {v_trav:.2f} m  rapporto {node.last_track_ratio:.2f}  "
                              f"qualita' {rec['victim_quality_mean']}  scarto dal nominale twin: "
                              f"media {rec['victim_vs_twin_dev_mean_cm']:.1f} cm, fine {rec['victim_vs_twin_dev_end_cm']:.1f} cm  [{v_st}]")
                    else:
                        print(f"  vittima continua: FALLITA [{v_st}]")
                    rec["window_time_s"] = time.time() - t_win
                elif direction:
                    history[-1]["goal_mode"] = _goal_mode_rec()
                with open(out / "history.json", "w") as f:
                    json.dump(history, f, indent=2)
                # Waypoint advance here too: without attack the state is moved by the victim rollout
                # (or by the nominal) and the window waypoint advances by the waypoints that rollout
                # passed.
                wp_idx = min(len(goals_abs) - 1, wp_idx + getattr(node, "last_wps_reached", 0))
                print(f"  tempo finestra: {(time.time()-t_win)/60:.1f} min\n")
                continue

            print("  — le deviazioni qui sotto sono misurate rispetto a questo punto")

            # NSGA-III search, configuration of run_nsga3.py.
            from pymoo.optimize import minimize

            eval_log = []

            def eval_genome(g_active):
                """Fitness of a candidate: (negated damage, perceptibility).

                The first objective is negated because pymoo minimises. The damage sums the
                deviation from the nominal and a directional term, i.e. how far the attack pushes
                away from the final goal: a drift that circles the goal is worth less than one that
                makes it miss. With --incompleti-regola eval_genome.rifatto is True during the
                repeat for a chain failure, which is not repeated again.
                """
                if win_cap is not None and time.time() - t_win > win_cap:
                    raise WindowTimeout()
                idx = len(eval_log)
                g = expand_genome(g_active, active_genes, n_genes)
                t_roll0 = time.time()
                c_start, end, ch, st, trav, frames = rollout(
                    g, isaac, mola, node, wps, args.horizon, f"w{k}_e{idx}",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from, stop_on=args.stop_on,
                    untracked=args.untracked, nominal_ok=nominal_ok, track_ref=nom_ratio,
                    settle_fixed=nom_settle_s, drift_reset_on=args.drift_reset_on,
                    frame_start=frame_start,
                    trace_path=(out / "traces" / f"w{k}_e{idx:02d}.csv")
                    if args.trace else None)
                t_roll1 = time.time()

                # Metrics common to every outcome, recorded for discarded ones too.
                loc_mean = node.last_loc_err_mean * 100 if node.last_loc_err_mean == node.last_loc_err_mean else None
                loc_end = node.last_loc_err_end * 100 if node.last_loc_err_end == node.last_loc_err_end else None
                loc_mean_raw = node.last_loc_err_mean_raw * 100 if node.last_loc_err_mean_raw == node.last_loc_err_mean_raw else None
                kind = outcome_kind(st)
                if end is None or kind == "infra":
                    discard = True
                elif kind == "valid":
                    discard = False
                elif args.discard_policy == "infra":
                    # Measurable but incomplete outcome: counts as damage, except when the window
                    # nominal was anomalous.
                    discard = not nominal_ok
                else:  # legacy: only ok/arrived count (untracked as per --untracked)
                    base = st.split("(")[0]
                    discard = (base != "untracked"
                               or args.untracked == "discard"
                               or (args.untracked == "nominal" and not nominal_ok))
                # --no-nominal-rollout: twin G outcome from the checks of its rollout (g_check);
                # with failed checks the candidate is excluded
                esito_g = (tv.esito_gemello_g(TWIN_STATE.get("mode"), st, getattr(node, "last_twin_fp", None))
                           if args.no_nominal_rollout else None)
                if esito_g == "guasto_gemello":
                    discard = True

                if discard:
                    dos = str(st).startswith("victim_")   # victim without pose = DoS, not infra
                    # --mola-se-ne-accorge: a discarded candidate with which the twin MOLA crashes
                    # or stays without pose beyond the threshold is also non-admissible
                    acc_d = None
                    if getattr(args, "mola_se_ne_accorge", False):
                        acc_d = wrec.mola_accorge(wrec.read_printf(getattr(mola, "log_path", None)),
                                                  args.accorge_icp_rifiutate, args.accorge_senza_posa_s,
                                                  crash=str(st).split("(")[0].strip() == "mola_crash")
                    # twin G without pose: excluded from damage and selection, labelled
                    guasto_g = (args.g_nominale_senza_posa == "continua"
                                and str(st).startswith("no_pose_from_new_mola"))
                    if esito_g is not None:
                        guasto_g = esito_g == "guasto_gemello"
                    eval_log.append({"genome": [float(v) for v in g],
                                     **({"guasto_gemello": True} if guasto_g else {}),
                                     **({"esito_gemello": esito_g} if esito_g is not None else {}),
                                     "dev": None, "chamfer": ch, "status": st,
                                     "track_ratio": node.last_track_ratio,
                                     "loc_err_mean_cm": loc_mean, "loc_err_end_cm": loc_end,
                                     "loc_err_mean_raw_cm": loc_mean_raw,
                                     "perc": node.last_perc or None, **drift_fields(node),
                                     "travelled": trav, "frames": frames, "counted": False,
                                     "dos": dos, "dos_reason": ("posa assente" if dos else None),
                                     **_twin_flag_fields(node, args.twin_filter),
                                     **_twin_start_fields(node),
                                     **({"accorge_ok": acc_d["accorge"] is not True, "mola_accorge": acc_d}
                                        if acc_d is not None else {}),
                                     **({"admissible": False, "inadmissible_reason": "mola_se_ne_accorge"}
                                        if acc_d is not None and acc_d["accorge"] is True else {}),
                                     "stop_reason": node.last_stop_reason})
                    print(f"    #{idx:02d}  SCARTATA  {trav:.2f} m  "
                          f"{frames:3d} frame  [{st}]" + ("  (DoS: vittima senza posa)" if dos else "")
                          + ("  (guasto gemello)" if guasto_g else "")
                          + ("  (il gemello perde la posa)" if esito_g == "gemello_perde_posa" else "")
                          + (f"  (MOLA se ne accorge: {acc_d['motivo']})"
                             if acc_d is not None and acc_d["accorge"] is True else ""))
                    eval_log[idx]["fitness_riserva"] = causa_riserva(
                        st, kind, guasto_g=guasto_g, dos=dos,
                        accorge=acc_d is not None and acc_d["accorge"] is True)
                    return (np.inf, np.inf, 1e6)

                dm = window_damage(end, nom_end, goal, nom_path,
                                   true_path(node.last_trace),
                                   args.damage, args.dir_weight, baseline=baseline_poly)
                damage = dm["damage"]

                pert = ch if ch == ch else 1e3
                # Stealth as an admissibility constraint: a diversion is valid, a denial of service
                # (robot still, ratio out of band, collapsed ICP quality) is infeasible. The DoS
                # stays recorded and counted but does not compete as damage.
                if args.admissibility == "stealth":
                    stealth_ok, dos_reason, dos_detail, is_dos = stealth_check(
                        node, st, trav, args.horizon)
                else:
                    stealth_ok, dos_reason, dos_detail, is_dos = True, "", "", False
                # Perceptibility budget: like stealth, it excludes from the front without discarding
                # the measurement.
                budget_ok = args.chamfer_budget is None or pert <= args.chamfer_budget
                # Threat model constraint: the online attacker must process every cloud within the
                # period between two scans; a more expensive genome delays the stream and the delay
                # itself becomes "damage" (slow estimate regime).
                proc_mean = float(np.mean(node.proc_ms_roll)) if node.proc_ms_roll else float("nan")
                proc_max = float(np.max(node.proc_ms_roll)) if node.proc_ms_roll else float("nan")
                calc_ok = (args.max_proc_ms is None or args.max_proc_ms <= 0 or proc_mean != proc_mean
                           or proc_mean <= args.max_proc_ms)
                if not calc_ok and not dos_reason:
                    dos_reason = f"calcolo {proc_mean:.0f} ms oltre il periodo di scansione ({args.max_proc_ms:.0f} ms)"
                # Internal perceptibility: computed before admissibility because with --f2-budget it
                # becomes a constraint itself (maximum damage among the candidates that stay
                # "invisible to MOLA").
                stats = _mola_csv_stats(getattr(mola, "traces_path", None),
                                        *getattr(node, "last_drive_stamps", (float("nan"), float("nan"))))
                q_att = stats.get("q_mean") if stats.get("q_mean") is not None else node.last_quality_mean
                q_ref = nom_stats.get("q_mean") if nom_stats.get("q_mean") is not None else nom_quality
                f2_int, f2_q, f2_it = _f2_internal(q_att, q_ref, stats.get("it_mean"), nom_it,
                                                   args.f2_q_scale, args.f2_it_scale)
                if nom_assente is not None:
                    # without nominal the internal f2 is undefined: recorded as None, objective 0
                    f2_int = f2_q = f2_it = None
                pert_obj = (f2_int if f2_int is not None else 0.0) if args.perceptibility == "internal" else pert
                # Internal perceptibility budget: admissibility constraint only if requested
                # (--f2-budget) and computable (f2_int not None).
                f2_ok = (args.f2_budget is None or args.f2_budget <= 0 or f2_int is None or f2_int <= args.f2_budget)
                if not f2_ok and not dos_reason:
                    dos_reason = f"percettibilita' interna {f2_int:.2f} oltre il budget ({args.f2_budget:.2f})"
                # --mola-se-ne-accorge: a pose missing beyond the threshold in the candidate twin
                # log makes it non-admissible (consecutive rejected scans are recorded)
                accorge, accorge_ok = None, True
                if getattr(args, "mola_se_ne_accorge", False):
                    accorge = wrec.mola_accorge(wrec.read_printf(getattr(mola, "log_path", None)),
                                                args.accorge_icp_rifiutate, args.accorge_senza_posa_s)
                    accorge_ok = accorge["accorge"] is not True
                    if not accorge_ok and not dos_reason:
                        dos_reason = "mola_se_ne_accorge"
                # --incompleti-regola: timeout and stopped_early classified by cause
                # (window_record.CLASSI_INCOMPLETO)
                inc = None
                if getattr(args, "incompleti_regola", False) and wrec.stato_incompleto(st):
                    inc = wrec.classe_incompleto(st, controlli_incompleto(args, t_roll0, t_roll1),
                                                 accorge, calc_ok)
                    if inc["classe"] == "attacco_valido" and dos_reason == "robot fermo":
                        # (c): a robot that gets stuck with a healthy chain and a publishing MOLA is
                        # valid
                        inc["stealth_originale"] = {"categoria": dos_reason, "dettaglio": dos_detail}
                        # the q check of stealth_check still applies: detectable stays excluded
                        stealth_ok, dos_reason, dos_detail, is_dos = stealth_robot_fermo_valido(node)
                        if stealth_ok and not f2_ok:
                            dos_reason = f"percettibilita' interna {f2_int:.2f} oltre il budget ({args.f2_budget:.2f})"
                admissible = stealth_ok and budget_ok and calc_ok and f2_ok and accorge_ok
                gap0 = _gap0_from_trace(node.last_trace)
                twin_mis, twin_mis_why = _twin_misaligned(node, args, gap0)
                eval_log.append({"genome": [float(v) for v in g],
                                 "dev": dm["dev_end"], "dev_mean": dm["dev_mean"],
                                 "dev_max": dm["dev_max"], "damage_mode": args.damage,
                                 "directional": dm["directional"],
                                 "damage": damage, "chamfer": ch, "status": st,
                                 "track_ratio": node.last_track_ratio,
                                 "quality_mean": node.last_quality_mean,
                                 "quality_min": node.last_quality_min,
                                 "loc_err_mean_cm": loc_mean, "loc_err_end_cm": loc_end,
                                 "loc_err_mean_raw_cm": loc_mean_raw,
                                 "perc": node.last_perc or None, **drift_fields(node),
                                 "travelled": trav, "frames": frames,
                                 "path_points": dm["path_points"], "counted": True,
                                 **_window_fields(dm),
                                 "admissible": admissible, "stealth_ok": stealth_ok,
                                 "budget_ok": budget_ok, "calc_ok": calc_ok, "f2_ok": f2_ok,
                                 **({"accorge_ok": accorge_ok, "mola_accorge": accorge}
                                    if getattr(args, "mola_se_ne_accorge", False) else {}),
                                 "proc_ms_mean": proc_mean, "proc_ms_max": proc_max,
                                 "inadmissible_reason": dos_reason or None,
                                 "dos": is_dos, "dos_reason": (dos_reason if is_dos else None),
                                 "stop_reason": node.last_stop_reason,
                                 "tau_s": node.last_tau,
                                 "slow": bool(node.last_tau == node.last_tau and node.last_tau > SLOW_TAU),
                                 "twin_gap0_cm": gap0,
                                 "twin_misaligned": twin_mis,
                                 "twin_misaligned_why": twin_mis_why or None,
                                 "f2_int": f2_int, "f2_q_term": f2_q, "f2_it_term": f2_it,
                                 "it_mean": stats.get("it_mean"), "it_max": stats.get("it_max"),
                                 "it_cov": stats.get("coverage"), "csv_q_mean": stats.get("q_mean"),
                                 "t_on_ms": stats.get("t_on_ms"),
                                 "t_wall": time.time(),
                                 "replay": getattr(node, "last_replay", None),
                                 **_twin_flag_fields(node, args.twin_filter),
                                 **_twin_start_fields(node),
                                 "baseline_window": wrec.baseline_window_fields(
                                     true_path(node.last_trace), baseline_poly),
                                 **({"danno_metro": wrec.danno_metro(true_path(node.last_trace), baseline_poly)}
                                    if getattr(args, "danno_metro", False) else {}),
                                 **({"esito_gemello": esito_g} if esito_g is not None else {}),
                                 "end": list(map(float, end))})
                if inc is not None:
                    rec_i = eval_log[-1]
                    rec_i["incompleto"] = inc
                    rec_i["incompleto_classe"] = inc["classe"]
                    if inc["classe"] == "attacco_valido":
                        # (c): damage from the baseline pose at window end; the measurement on the
                        # polyline stays in the record
                        d_fine = wrec.danno_fine_finestra(end, (fine_finestra_base or {}).get(k))
                        rec_i["danno_fine_finestra"] = d_fine
                        if d_fine is not None:
                            rec_i["damage_polilinea"] = rec_i["damage"]
                            rec_i["dev_mean_polilinea"] = rec_i["dev_mean"]
                            rec_i["damage"] = rec_i["dev_mean"] = damage = d_fine
                    elif inc["classe"] == "guasto_catena":
                        rec_i["counted"] = False
                        rec_i["admissible"] = False
                        rec_i["inadmissible_reason"] = "guasto_catena"
                gen_no = idx // args.pop + 1
                # Ratio in consistent units: damage in cm over perceptibility in cm.
                rst = math.hypot(c_start[0] - saved[0], c_start[1] - saved[1]) if c_start else float("nan")
                print(f"    gen{gen_no} #{idx:02d}  rst {rst*100:4.1f}  end {dm['dev_end']*100:5.1f} "
                      f"mean {dm['dev_mean']*100:5.1f} max {dm['dev_max']*100:5.1f}  "
                      f"dir {dm['directional']*100:+6.1f}  dmg {damage*100:6.1f} cm   "
                      f"pert {pert:7.2f} cm   {trav:.2f} m  {frames:3d}f  "
                      f"rapp {node.last_track_ratio:.2f}  [{st}/{node.last_stop_reason}]")
                if args.perceptibility == "internal" and f2_int is not None:
                    print(f"      f2 interna {f2_int:.2f} (qualita' {f2_q}, iterazioni {f2_it}; "
                          f"it {stats.get('it_mean')} contro {nom_it}, copertura CSV {stats.get('coverage')})")
                if eval_log[-1].get("twin_filtered"):
                    print(f"      -> gemello anomalo ({args.twin_filter}: intervallo "
                          f"{eval_log[-1].get('twin_icp_gap_s')} s, prima q {eval_log[-1].get('twin_icp_first_q')}): "
                          "misura tenuta, escluso dalla selezione")
                if twin_mis:
                    print(f"      -> twin non allineato alla partenza ({twin_mis_why}): "
                          "misura tenuta, escluso dalla selezione")
                if not stealth_ok:
                    print(f"      -> NON FURTIVO: {dos_reason} [{dos_detail}] (DoS, non scelto)")
                elif not budget_ok:
                    print(f"      -> oltre il budget: Chamfer {pert:.1f} > {args.chamfer_budget:.0f} cm "
                          f"(vincolo, non scelto)")
                if not accorge_ok:
                    print(f"      -> MOLA SE NE ACCORGE: {accorge['motivo']} (non ammissibile)")
                if args.verbose:
                    print(describe_params(print_generator(node).encode_perturbation(g)))
                if inc is not None:
                    print(f"      -> incompleto: {inc['classe']} ({inc['motivo']})")
                if inc is not None and inc["classe"] == "guasto_catena":
                    # (a): the candidate is repeated once (like the rollout after an Isaac stall);
                    # the old measurement does not count and cannot win
                    if getattr(eval_genome, "rifatto", False):
                        eval_log[idx]["fitness_riserva"] = "guasto_catena_incompleto_ripetuto"
                        return (np.inf, pert_obj, 1e6)
                    idx_r = len(eval_log)
                    eval_genome.rifatto = True
                    try:
                        res_r = eval_genome(g_active)
                    finally:
                        eval_genome.rifatto = False
                    eval_log[idx]["superseded_by"] = idx_r
                    eval_log[idx_r]["reevaluation_of"] = idx
                    return res_r
                # A non-stealthy candidate is infeasible: damage +inf, like a failure, so NSGA-III
                # never picks it and maximises the deviation only within the admissible set. Chamfer
                # over budget stays a soft constraint (ranked violation).
                if not stealth_ok or not calc_ok or not accorge_ok:
                    eval_log[idx]["fitness_riserva"] = ("non_furtivo" if not stealth_ok else
                                                        "calcolo_oltre_limite" if not calc_ok else
                                                        "mola_se_ne_accorge")
                    return (np.inf, pert_obj, 1e6)
                g_viol = (pert - args.chamfer_budget) if args.chamfer_budget is not None else -1.0
                if args.objective == "minq":
                    # f1 = minimum quality (NSGA-III minimises): look for the genome that lowers q
                    # the most within the budget.
                    qmin = node.last_quality_min if node.last_quality_min is not None else 1.0
                    return (qmin, pert_obj, g_viol)
                return (-damage, pert_obj, g_viol)

            n_warm = 0
            n_mut = 0
            warm_counts = None   # warm start composition (history)
            warm_used = None     # front used by the warm start of this window (history)
            win_timeout = False
            res_pop_F = None   # objectives of the final NSGA-III population (note in history)
            rf_rec = None      # single random with budget (--random-first-admissible)
            seq_rec = None     # --fixed-genome-seq entry used in this window

            def campiona_sparso(rng_s, n):
                """n random genomes with a random number of active genes (others off), as in the
                random arm.
                """
                off = np.array([0.0 if ag in DIRECTION_GENES else -1.0 for ag in active_genes])
                return legacy_random_genomes(rng_s, n, n_search, True, off)
            if args.search == "nsga3":
                problem = build_problem(n_search, eval_genome,
                                        has_constraint=args.chamfer_budget is not None)
                init_X = None
                if args.warm_start and warm_pool:
                    # Warm start (continuous-victim campaign): the population of this window = N/2
                    # from the previous PARETO FRONT (spread over Chamfer to keep the
                    # damage/perceptibility trade-off; if the front has fewer than N/2 solutions it
                    # is filled with their perturbations) + N/2 RANDOM (same sparse sampling as the
                    # random arm, to keep diversity). The first window stays cold-start (warm_pool
                    # is None).
                    half = args.pop // 2
                    rng_w = np.random.default_rng(args.seed + k + 7919)
                    pareto_rows, warm_counts = warm_front_rows(warm_pool, half, rng_w, n_search)
                    n_warm, n_mut = warm_counts["warm_front_n"], warm_counts["warm_mut_n"]
                    warm_used = dict(warm_src or {}, copied_idx=warm_counts.get("warm_front_copied_idx"))
                    # N/2 random (same sparse sampling as the random arm)
                    n_rand = max(0, args.pop - len(pareto_rows))
                    if args.sampling == "uniform":
                        rand_rows = list(wrec.sample_genomes(rng_w, n_rand, n_search, "uniform"))
                    else:
                        rand_rows = [r for r in campiona_sparso(rng_w, n_rand)]
                    rows = (pareto_rows + rand_rows)[:args.pop]
                    init_X = np.array(rows)
                    print(f"  warm start: {n_warm} copie dal fronte di Pareto + {n_mut} perturbazioni "
                          f"({warm_counts['warm_mut_identical_n']} uguali a un membro del fronte) "
                          f"+ {len(rand_rows)} casuali")
                if init_X is None and (args.sampling == "sparse"
                                       or (args.sampling == "legacy" and args.random_sparse)):
                    # First horizon (or no memory): sparse initial population, same generator as the
                    # random arm.
                    init_X = campiona_sparso(np.random.default_rng(args.seed + k + 4243), args.pop)
                    print("  popolazione iniziale sparsa (stesso generatore del braccio random)")
                algorithm = build_nsga3(args.pop, init_X=init_X)
                try:
                    res = minimize(problem, algorithm, ("n_gen", args.gen),
                                   seed=args.seed, verbose=False)
                    res_F, res_X = res.F, res.X
                    res_pop_F = res.pop.get("F") if getattr(res, "pop", None) is not None else None
                except WindowTimeout:
                    # Watchdog: the window failed. No genome applied; the stretch is driven with the
                    # nominal, otherwise the next window would restart from the same state and time
                    # out again.
                    win_timeout = True
                    reason = (f"oltre {args.window_cap_factor}x il tempo della prima finestra "
                              f"completata ({first_win_time/60:.1f} min): ricerca interrotta dopo "
                              f"{len(eval_log)} valutazioni")
                    print(f"  WATCHDOG: finestra fallita, {reason}")
                    history.append({"window": k, "status": "watchdog_timeout", "reason": reason,
                                    "goal": list(map(float, goal)), "waypoint_index": wp_idx,
                                    "n_evaluations": len(eval_log), "evaluations": eval_log,
                                    "nominal_end": (list(map(float, nom_end)) if nom_end is not None else None),
                                    "watchdog_timeout": True, "window_cap_s": win_cap,
                                    "window_time_s": time.time() - t_win})
                    with open(out / "history.json", "w") as f:
                        json.dump(history, f, indent=2)
                    if args.victim_never_clean:
                        # no driving without genome: robot still in the saved state
                        print("  vittima ferma nello stato salvato (--victim-never-clean), poi finestra successiva")
                        history[-1].update(victim_hold_fields("watchdog", victim))
                        history[-1]["hold_restore"] = hold_restore(isaac, mola, node, k, saved)
                        _esito_finestra(history[-1], None)
                        with open(out / "history.json", "w") as f:
                            json.dump(history, f, indent=2)
                        print(f"  tempo finestra: {(time.time()-t_win)/60:.1f} min\n")
                        continue
                    print("  avanzamento col nominale, poi finestra successiva")
                    _, _, _, wd_st, _, _ = rollout(
                            None, isaac, mola, node, wps, args.horizon, f"w{k}_advance",
                            warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                            attack_from=args.attack_from, stop_on=args.stop_on,
                            drift_reset_on=args.drift_reset_on, frame_start=frame_start, victim=victim,
                            trace_path=(out / "traces" / f"w{k}_advance.csv") if args.trace else None)
                    history[-1]["advance_status"] = wd_st
                    _esito_finestra(history[-1], wd_st if victim is not None else None)
                    with open(out / "history.json", "w") as f:
                        json.dump(history, f, indent=2)
                    wp_idx = min(len(goals_abs) - 1, wp_idx + getattr(node, "last_wps_reached", 0))
                    print(f"  tempo finestra: {(time.time()-t_win)/60:.1f} min\n")
                    continue
                # Front of this window for the next one (warm_pool_from_log): here, before the
                # re-evaluations; with --warm-front-after-reeval after the re-evaluations (below).
                if not args.warm_front_after_reeval:
                    warm_pool, warm_best, w_idx = warm_pool_from_log(
                        eval_log, args.select, args.chamfer_budget, active_genes)
                    warm_src = {"window": k, "idx": w_idx, "computed": "before_reeval"}
            elif args.search == "random":
                # Random search with the same budget: pop x gen genomes drawn at random in the
                # group, chosen with the same criterion. The front is computed on the valid
                # evaluations.
                if args.window_rng == "seedseq":
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, k]))
                else:
                    rng = np.random.default_rng(args.seed + k)
                off = np.array([0.0 if ag in DIRECTION_GENES else -1.0 for ag in active_genes])
                mode_s = args.sampling if args.sampling != "legacy" else (
                    "sparse" if args.random_sparse else "uniform")
                if args.random_first_admissible > 0:
                    # Single random with budget: one genome at a time from the same generator,
                    # evaluated in the twin; stop at the first one valid for the selection (same
                    # rule as fronte_pareto), at most K.
                    rf_rec, X_all, F_all, rf_draw = random_first_search(
                        rng, n_search, mode_s, off, eval_genome, eval_log,
                        args.chamfer_budget, args.random_first_admissible)
                elif args.sampling != "legacy":
                    X_all = wrec.sample_genomes(rng, args.pop * args.gen, n_search, mode_s, off)
                else:
                    X_all = legacy_random_genomes(rng, args.pop * args.gen, n_search,
                                                  args.random_sparse, off)
                if args.random_first_admissible <= 0:
                    F_all = np.array([eval_genome(x) for x in X_all])[:, :2]
                ok = np.all(np.isfinite(F_all), axis=1)
                res_F, res_X = pareto_front(F_all[ok], X_all[ok])
            elif args.search == "fixed":
                # "massimo" arm: a single given genome, applied at every horizon without search (one
                # twin evaluation, then the selection re-evaluates and applies it).
                if genome_seq is not None:
                    g_fix, seq_rec = genome_seq_entry(genome_seq, k)
                    print(f"  genoma fisso della sequenza: voce {seq_rec['seq_pos'] + 1}/{len(genome_seq)} "
                          f"({seq_rec['entry']})")
                else:
                    g_fix = np.array([float(v) for v in args.fixed_genome.split(",")], dtype=float)
                X_all = np.array([g_fix[active_genes]])
                F_all = np.array([eval_genome(x) for x in X_all])[:, :2]
                ok = np.all(np.isfinite(F_all), axis=1)
                res_F, res_X = F_all[ok], X_all[ok]
            else:
                # Gaussian: no search, a single rollout with the isotropic noise of the node
                # (--gaussian-sigma on perturbation_node). The published genome is ignored by the
                # node in that mode.
                F_all = np.array([eval_genome(np.zeros(n_search))])[:, :2]
                X_all = np.zeros((1, n_search))
                ok = np.all(np.isfinite(F_all), axis=1)
                res_F, res_X = F_all[ok], X_all[ok]

            # --- Candidates measured during an adapter degradation ---
            # An external check logs the restarts; everything measured between the last clean check
            # and the restart is re-evaluated (new eval_log entry) and the old measurement cannot
            # win.
            if args.adapter_events and os.path.exists(args.adapter_events) and args.search in ("nsga3", "random"):
                try:
                    with open(args.adapter_events) as f_ev:
                        events = [json.loads(l) for l in f_ev if l.strip()]
                except Exception:
                    events = []
                reevaluate_after_adapter_restart(eval_log, events, t_win, eval_genome, active_genes)
            if args.search == "nsga3" and args.warm_front_after_reeval:
                # Front for the warm start after the re-evaluations: the records measured during the
                # degradation have admissible and counted False and stay out; their valid
                # re-evaluations enter.
                warm_pool, warm_best, w_idx = warm_pool_from_log(
                    eval_log, args.select, args.chamfer_budget, active_genes)
                warm_src = {"window": k, "idx": w_idx, "computed": "after_reeval"}
            if rf_rec is not None:
                # Single random: if the re-evaluations removed the valid one, the search resumes
                # with the remaining attempts; eval_idx = first valid record after the
                # re-evaluations.
                rf_rec, X_all, F_all, n_more = random_first_resume(
                    rf_rec, rf_draw, eval_genome, eval_log, args.chamfer_budget,
                    args.random_first_admissible, X_all, F_all)
                if n_more:
                    ok = np.all(np.isfinite(F_all), axis=1)
                    res_F, res_X = pareto_front(F_all[ok], X_all[ok])
            # --- Selection of the genome to apply ---
            # 'ratio': argmax(damage/Chamfer), vulnerable to candidates with Chamfer ~0 (noise).
            # 'maxdamage': maximum damage among the admissible ones (already within budget), with a
            # defence against noise: the top-K by search damage are re-evaluated with one more
            # rollout and the choice is on the MEAN of the two measurements (against the winner's
            # curse of the single rollout). A candidate not reproducible in re-evaluation (invalid
            # outcome) counts 0 in that measurement.
            reeval = None
            sel_rec = None   # pareto_maxdamage selection record (in history)
            n_valid_sel = None
            if args.select_only_valid:
                # --select-only-valid: every candidate records in history its invalidity reasons
                # (empty list = valid); only valid ones can win. Slow-rollout filter of
                # top_maxdamage (only with maxdamage, unless --search fixed and --admissibility
                # none): the reason is "lento".
                lenti_fuori = (args.select == "maxdamage" and args.search != "fixed"
                               and args.admissibility != "none")
                for e in eval_log:
                    e["selection_invalid_reasons"] = motivi_non_valido(
                        e, args.chamfer_budget, da_esito_gemello=args.no_nominal_rollout)
                    if lenti_fuori and e.get("slow"):
                        e["selection_invalid_reasons"].append("lento")
                n_valid_sel = sum(1 for e in eval_log if not e["selection_invalid_reasons"])
                print(f"  solo candidati validi (--select-only-valid): {n_valid_sel} su {len(eval_log)}")
            if args.select == "ratio":
                best_g, best_F, ratio = pick_best(res_F, res_X, budget=args.chamfer_budget)
                if best_g is not None:
                    best_g = expand_genome(best_g, active_genes, n_genes)
            elif args.select == "pareto_maxdamage":
                # pareto_maxdamage: front (max damage, min Chamfer) among the valid candidates of
                # the whole window (src/analysis/fronte_pareto.py) and application of the point with
                # maximum damage. No re-evaluation, no extra rollout, no filter on slow rollouts:
                # tau and slow are only recorded.
                fronte, scelto = fronte_pareto(eval_log, budget=args.chamfer_budget,
                                               solo_validi=args.select_only_valid)
                n_valid = sum(1 for e in eval_log
                              if candidato_valido(e, args.chamfer_budget, args.select_only_valid))
                n_slow = sum(1 for e in eval_log if e.get("slow"))
                n_mis = sum(1 for e in eval_log if e.get("twin_misaligned"))
                if n_slow:
                    print(f"  {n_slow} candidati nel regime lento (tau > {SLOW_TAU} s): tenuti (nessun filtro)")
                if n_mis:
                    print(f"  {n_mis} candidati con twin non allineato alla partenza (> {args.twin_gap0_max:.0f} cm): esclusi dalla selezione")

                def _righe_F(a):
                    # (n, 2) of finite objectives, empty if missing
                    if a is None:
                        return np.zeros((0, 2))
                    a = np.atleast_2d(np.asarray(a, dtype=float))
                    if a.ndim != 2 or a.shape[1] != 2:
                        return np.zeros((0, 2))
                    return a[np.all(np.isfinite(a), axis=1)]

                def _presente(rows, danno, cham):
                    # pymoo records (-damage, Chamfer); comparison up to numerical noise
                    return bool(len(rows)) and bool(np.any(np.all(np.isclose(rows, [-danno, cham], rtol=0.0, atol=1e-9), axis=1)))

                pf_rows = _righe_F(res_F)
                pop_rows = _righe_F(res_pop_F)
                for p in fronte:
                    p["in_pymoo_front"] = _presente(pf_rows, p["damage"], p["chamfer"])
                    p["in_final_pop"] = _presente(pop_rows, p["damage"], p["chamfer"]) if res_pop_F is not None else None
                front_rows = _righe_F([[-p["damage"], p["chamfer"]] for p in fronte]) if fronte else np.zeros((0, 2))
                pymoo_eq = (len(pf_rows) == len(front_rows)
                            and all(_presente(front_rows, -r[0], r[1]) for r in pf_rows)
                            and all(p["in_pymoo_front"] for p in fronte))
                nota = None
                scelto_fronte = scelto
                if rf_rec is not None:
                    # Single random: the first valid one is applied (normally the only one, hence
                    # the front point); different only after re-evaluations.
                    scelto = random_first_choice(fronte, eval_log, rf_rec)
                    if (args.select_only_valid and scelto is not None
                            and eval_log[scelto["idx"]]["selection_invalid_reasons"]):
                        scelto = None
                if scelto is None:
                    best_g, best_F, ratio = None, None, None
                else:
                    best_g = np.array(scelto["genome"], dtype=float)
                    best_F = np.array([-scelto["damage"], scelto["chamfer"]])
                    ratio = scelto["damage"] * 100.0 / max(scelto["chamfer"], 1.0)
                    if res_pop_F is not None and not scelto["in_final_pop"]:
                        nota = "il punto di danno massimo non e' nella popolazione finale di NSGA-III"
                    elif scelto["in_pymoo_front"] is False:
                        nota = "il punto di danno massimo non e' nel fronte salvato da pymoo (pareto_F)"
                sel_rec = {"rule": "pareto_maxdamage", "budget": args.chamfer_budget,
                           "n_evaluated": len(eval_log), "n_valid": n_valid, "n_front": len(fronte),
                           "front": fronte,
                           "chosen_idx": scelto["idx"] if scelto is not None else None,
                           "chosen": scelto,
                           **({"random_first": True,
                               "front_choice_idx": scelto_fronte["idx"] if scelto_fronte is not None else None}
                              if rf_rec is not None else {}),
                           "pymoo_front_n": int(len(pf_rows)), "pymoo_front_equals_all": bool(pymoo_eq),
                           "final_pop_n": int(len(pop_rows)) if res_pop_F is not None else None,
                           "n_slow": n_slow, "n_slow_in_front": sum(1 for p in fronte if p["slow"]),
                           "slow_filter": False, "n_twin_misaligned": n_mis,
                           "twin_filter": args.twin_filter,
                           "n_twin_filtered": sum(1 for e in eval_log if e.get("twin_filtered")),
                           "topk_ignored": args.topk, "reeval_ignored": bool(args.reeval),
                           "n_reeval_rollouts": 0, "note": nota,
                           **({"only_valid": True} if args.select_only_valid else {})}
                print(f"\n  SELEZIONE pareto_maxdamage: fronte di {len(fronte)} punti su {n_valid} candidati validi "
                      f"({len(eval_log)} valutati); fronte pymoo {len(pf_rows)} punti, "
                      f"{'coincide' if pymoo_eq else 'non coincide'} con quello su tutti i candidati")
                for p in fronte:
                    mark = " <- scelto" if scelto is not None and p["idx"] == scelto["idx"] else ""
                    print(f"    #{p['idx']:03d}  danno {p['damage']*100:7.2f} cm   Chamfer {p['chamfer']:8.3f} cm   "
                          f"tau {p['tau_s']}  lento {p['slow']}  pymoo {p['in_pymoo_front']}  "
                          f"pop.finale {p['in_final_pop']}{mark}")
                if nota:
                    print(f"  nota: {nota}")
            else:
                n_slow = sum(1 for e in eval_log if e.get("slow"))
                n_mis = sum(1 for e in eval_log if e.get("twin_misaligned"))
                if n_slow:
                    _lenti_tenuti = args.search == "fixed" or args.admissibility == "none"
                    print(f"  {n_slow} candidati nel regime lento (tau > {SLOW_TAU} s): "
                          f"{'tenuti (applicazione diretta)' if _lenti_tenuti else 'esclusi dalla selezione'}")
                if n_mis:
                    print(f"  {n_mis} candidati con twin non allineato alla partenza (> {args.twin_gap0_max:.0f} cm): esclusi dalla selezione")
                top = top_maxdamage(eval_log, args)
                if not top:
                    best_g, best_F, ratio = None, None, None
                else:
                    print(f"\n  SELEZIONE max-danno: rivalutazione dei top-{len(top)} per difesa dal rumore")
                    topk_rec = []
                    for j, e in enumerate(top):
                        gfull = np.array(e["genome"], dtype=float)
                        t_re0 = time.time()
                        _, re_end, re_ch, re_st, _, _ = rollout(
                            gfull, isaac, mola, node, wps, args.horizon, f"w{k}_sel{j}",
                            warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                            attack_from=args.attack_from, stop_on=args.stop_on, untracked=args.untracked,
                            nominal_ok=nominal_ok, track_ref=nom_ratio, settle_fixed=nom_settle_s,
                            drift_reset_on=args.drift_reset_on, frame_start=frame_start)
                        re_slow = bool(node.last_tau == node.last_tau and node.last_tau > SLOW_TAU)
                        re_gap0 = _gap0_from_trace(node.last_trace)
                        re_mis, re_why = _twin_misaligned(node, args, re_gap0)
                        if re_mis:
                            print(f"    top{j}: twin della rivalutazione non allineato ({re_why}): ripeto una volta")
                            t_re0 = time.time()
                            _, re_end, re_ch, re_st, _, _ = rollout(
                                gfull, isaac, mola, node, wps, args.horizon, f"w{k}_sel{j}b",
                                warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                                attack_from=args.attack_from, stop_on=args.stop_on, untracked=args.untracked,
                                nominal_ok=nominal_ok, track_ref=nom_ratio, settle_fixed=nom_settle_s,
                                drift_reset_on=args.drift_reset_on, frame_start=frame_start)
                            re_slow = bool(node.last_tau == node.last_tau and node.last_tau > SLOW_TAU)
                            re_gap0 = _gap0_from_trace(node.last_trace)
                            re_mis, re_why = _twin_misaligned(node, args, re_gap0)
                        # --incompleti-regola: incomplete re-evaluation classified like the
                        # candidates; with (a) it is repeated once, (c) counts with the window-end
                        # damage
                        re_inc = None
                        if getattr(args, "incompleti_regola", False) and wrec.stato_incompleto(re_st):
                            re_inc = classe_rollout_incompleto(args, node, mola, re_st, t_re0, time.time())
                            if re_inc["classe"] == "guasto_catena":
                                print(f"    top{j}: rivalutazione incompleta per guasto della catena "
                                      f"({re_inc['motivo']}): ripeto una volta")
                                t_re0 = time.time()
                                _, re_end, re_ch, re_st, _, _ = rollout(
                                    gfull, isaac, mola, node, wps, args.horizon, f"w{k}_sel{j}c",
                                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                                    attack_from=args.attack_from, stop_on=args.stop_on, untracked=args.untracked,
                                    nominal_ok=nominal_ok, track_ref=nom_ratio, settle_fixed=nom_settle_s,
                                    drift_reset_on=args.drift_reset_on, frame_start=frame_start)
                                re_slow = bool(node.last_tau == node.last_tau and node.last_tau > SLOW_TAU)
                                re_gap0 = _gap0_from_trace(node.last_trace)
                                re_mis, re_why = _twin_misaligned(node, args, re_gap0)
                                re_inc = dict(re_inc, ripetuta=True, esito_ripetizione=(
                                    classe_rollout_incompleto(args, node, mola, re_st, t_re0, time.time())
                                    if wrec.stato_incompleto(re_st) else None))
                                if wrec.stato_incompleto(re_st):
                                    re_inc["classe"] = re_inc["esito_ripetizione"]["classe"]
                                else:
                                    re_inc["classe"] = None
                        re_valido_inc = re_inc is not None and re_inc["classe"] == "attacco_valido"
                        if re_end is not None and (re_st in VALID or re_valido_inc) and (not re_slow or args.search == "fixed" or args.admissibility == "none") and not re_mis:
                            dm_re = window_damage(re_end, nom_end, goal, nom_path,
                                                  true_path(node.last_trace), args.damage, args.dir_weight,
                                                  baseline=baseline_poly)
                            rev = dm_re["dev_mean"] * 100
                            if re_valido_inc:
                                d_fine = wrec.danno_fine_finestra(re_end, (fine_finestra_base or {}).get(k))
                                if d_fine is not None:
                                    rev = d_fine * 100
                        else:
                            rev = 0.0
                        search_cm = e["dev_mean"] * 100
                        v_top = valori_topk(search_cm, rev, re_st, args.topk_agg)
                        topk_rec.append({"genome": e["genome"], "search_dev_mean_cm": search_cm,
                                         **{c: v for c, v in v_top.items() if c not in ("reeval_escluso", "valutato_su")},
                                         "chamfer": e["chamfer"], "reeval_status": re_st, "reeval_tau_s": node.last_tau, "reeval_slow": re_slow,
                                         "search_gap0_cm": e.get("twin_gap0_cm"), "reeval_gap0_cm": re_gap0, "reeval_twin_misaligned": re_mis,
                                         **({"reeval_incompleto": re_inc} if re_inc is not None else {}),
                                         **{c: v_top[c] for c in ("reeval_escluso", "valutato_su") if c in v_top}})
                        if "reeval_escluso" in v_top:
                            print(f"    top{j}: ricerca {search_cm:5.1f} -> rivalutazione esclusa ({re_st}): "
                                  f"vale il solo danno di ricerca  ch {e['chamfer']:5.1f}")
                        else:
                            print(f"    top{j}: ricerca {search_cm:5.1f} -> rivalut {rev:5.1f} "
                                  f"(media {(search_cm+rev)/2:5.1f}, minimo {min(search_cm, rev):5.1f})  ch {e['chamfer']:5.1f}  [{re_st}]")
                    best_rec = scegli_topk(topk_rec)
                    best_g = np.array(best_rec["genome"], dtype=float)
                    best_F = np.array([-best_rec["search_dev_mean_cm"] / 100.0, best_rec["chamfer"]])
                    ratio = best_rec["mean_cm"] / max(best_rec["chamfer"], 1.0)
                    reeval = {"rule": f"maxdamage_topk_{args.topk_agg}", "budget": args.chamfer_budget,
                              "topk": topk_rec, "selected_mean_cm": best_rec["mean_cm"],
                              "selected_agg_cm": best_rec["agg_cm"],
                              "selected_search_cm": best_rec["search_dev_mean_cm"],
                              "selected_reeval_cm": best_rec["reeval_dev_mean_cm"],
                              "winners_curse_cm": (None if best_rec["reeval_dev_mean_cm"] is None else
                                                   best_rec["search_dev_mean_cm"] - best_rec["reeval_dev_mean_cm"]),
                              **({"scelta_fra_esclusi": True} if best_rec.get("scelta_fra_esclusi") else {}),
                              **({"only_valid": True} if args.select_only_valid else {})}
                    print(f"  scelto per {'minimo' if args.topk_agg == 'min' else 'media'} {best_rec['agg_cm']:.1f} cm "
                          f"(ricerca {best_rec['selected_search_cm'] if 'selected_search_cm' in best_rec else best_rec['search_dev_mean_cm']:.1f}, "
                          + (f"rivalut {best_rec['reeval_dev_mean_cm']:.1f}; winner's curse "
                             f"{best_rec['search_dev_mean_cm']-best_rec['reeval_dev_mean_cm']:+.1f})"
                             if best_rec["reeval_dev_mean_cm"] is not None else
                             "rivalutazioni tutte guaste: scelto sulla sola ricerca)"))
            n_dos_win = sum(1 for e in eval_log if e.get("dos"))
            genoma_nullo_rec = None
            if args.genoma_nullo_se_nessun_ammissibile and (best_g is None or not np.all(np.isfinite(best_F))):
                # No admissible genome: null genome through the normal winner path (winner trials,
                # realigned twin, comparisons, record, shadow) instead of holding or advancing with
                # the nominal
                best_g, best_F, ratio, res_F, genoma_nullo_rec = scelta_genoma_nullo(
                    args, rf_rec, eval_log, n_valid_sel, active_genes, n_genes, res_F)
                genoma_nullo_rec["deriva_vittima_inizio_finestra"] = deriva_inizio
                print(f"  {genoma_nullo_rec['motivo']} ({n_dos_win} DoS, "
                      f"{sum(1 for e in eval_log if not e.get('counted'))} scartati) "
                      "— si applica il GENOMA NULLO (direzioni 0, altri geni -1): la vittima percorre il metro")
            if genoma_nullo_rec is None and (best_g is None or not np.all(np.isfinite(best_F))):
                print(f"  nessun candidato ammissibile ({n_dos_win} DoS, "
                      f"{sum(1 for e in eval_log if not e.get('counted'))} scartati) "
                      "— nessun genoma applicato, finestra saltata")
                history.append({"window": k, "status": "no_admissible",
                                "goal": list(map(float, goal)), "waypoint_index": wp_idx,
                                "n_dos": n_dos_win, "n_evaluations": len(eval_log),
                                "evaluations": eval_log, "nominal_end": (list(map(float, nom_end)) if nom_end is not None else None),
                                **ombra_infra_fields(eval_log),
                                **({"select_only_valid": True, "n_valid_selection": n_valid_sel}
                                   if args.select_only_valid else {}),
                                "window_time_s": time.time() - t_win})
                # The stretch is still driven without attack (by the continuous victim, if present):
                # the state stays consistent and the next window restarts from where the robot
                # actually arrived, not from where the last evaluation left it.
                if args.victim_never_clean:
                    # no driving without genome: robot still in the saved state
                    print("  vittima ferma nello stato salvato (--victim-never-clean), poi finestra successiva")
                    if rf_rec is not None:
                        history[-1]["random_first"] = rf_rec
                    history[-1].update(victim_hold_fields("nessun_ammissibile", victim))
                    history[-1]["hold_restore"] = hold_restore(isaac, mola, node, k, saved)
                    _esito_finestra(history[-1], None)
                    with open(out / "history.json", "w") as f:
                        json.dump(history, f, indent=2)
                    continue
                print("  avanzamento col nominale (nessun genoma applicato), poi finestra successiva")
                _, adv_end, _, adv_st, adv_trav, _ = rollout(
                    None, isaac, mola, node, wps, args.horizon, f"w{k}_advance",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                    attack_from=args.attack_from, stop_on=args.stop_on, settle_fixed=nom_settle_s,
                    drift_reset_on=args.drift_reset_on, frame_start=frame_start, victim=victim,
                    trace_path=(out / "traces" / f"w{k}_advance.csv") if args.trace else None)
                if rf_rec is not None:
                    history[-1]["random_first"] = rf_rec
                history[-1]["victim_window"] = _victim_window_record(node, victim, baseline_poly)
                _refresh_victim_scans(history[:-1], victim)
                history[-1].update({"advance_status": adv_st, "advance_travelled": adv_trav,
                                    "advance_end": list(map(float, adv_end)) if adv_end is not None else None,
                                    "advanced_with_nominal": True})
                _esito_finestra(history[-1], adv_st if victim is not None else None)
                with open(out / "history.json", "w") as f:
                    json.dump(history, f, indent=2)
                wp_idx = min(len(goals_abs) - 1, wp_idx + getattr(node, "last_wps_reached", 0))
                continue

            F = np.atleast_2d(res_F)
            print(f"\n  FRONTE DI PARETO: {len(F)} soluzioni non dominate "
                  f"su {len(eval_log)} valutate "
                  f"({sum(1 for e in eval_log if e['status'] in ('ok','arrived'))} valide)")
            for i, f in enumerate(F):
                mark = " <- scelto" if np.allclose(f, best_F) else ""
                print(f"    danno {-f[0]*100:7.2f} cm   pert {f[1]:8.3f} cm   "
                      f"rapporto {(-f[0]*100)/max(f[1],1e-3):6.2f}{mark}")
            print(f"\n  Genoma scelto (tutti i {n_genes} geni decodificati):")
            print(describe_params(print_generator(node).encode_perturbation(best_g)))
            if genoma_nullo_rec is not None:
                # victim |A| right before the trials and the application: the search moves only the
                # twin generator, the victim is still
                genoma_nullo_rec["deriva_vittima_prima_applicazione"] = deriva_vittima_status(node)

            # Re-evaluation of the winner: one more rollout with the same genome, to measure how
            # much the search value is inflated by noise (selection of the maximum among noisy
            # evaluations).
            if args.select == "ratio" and args.reeval:
                _, re_end, re_ch, re_st, re_trav, re_frames = rollout(
                    best_g, isaac, mola, node, wps, args.horizon, f"w{k}_reeval",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                    attack_from=args.attack_from, stop_on=args.stop_on, untracked=args.untracked, nominal_ok=nominal_ok,
                    track_ref=nom_ratio, settle_fixed=nom_settle_s, drift_reset_on=args.drift_reset_on,
                    frame_start=frame_start)
                if re_end is not None and re_st in ("ok", "arrived"):
                    dm_re = window_damage(re_end, nom_end, goal, nom_path,
                                          true_path(node.last_trace), args.damage, args.dir_weight,
                                          baseline=baseline_poly)
                    reeval = {"damage_cm": dm_re["damage"] * 100, "dev_mean_cm": dm_re["dev_mean"] * 100,
                              **_window_fields(dm_re, 100.0, "_cm"),
                              "chamfer": re_ch, "status": re_st, "frames": re_frames,
                              "loc_err_mean_cm": node.last_loc_err_mean * 100 if node.last_loc_err_mean == node.last_loc_err_mean else None,
                              "loc_err_end_cm": node.last_loc_err_end * 100 if node.last_loc_err_end == node.last_loc_err_end else None,
                              "loc_err_mean_raw_cm": node.last_loc_err_mean_raw * 100 if node.last_loc_err_mean_raw == node.last_loc_err_mean_raw else None,
                              "perc": node.last_perc or None, **drift_fields(node)}
                    print(f"  rivalutazione: danno {dm_re['damage']*100:.1f} cm "
                          f"(in ricerca {-best_F[0]*100:.1f})  pert {re_ch:.2f}  [{re_st}]")
                else:
                    reeval = {"damage_cm": None, "status": re_st,
                              **({"escluso": ESITO_ETA_RIPETUTO} if re_st == ESITO_ETA_RIPETUTO else {})}
                    print(f"  rivalutazione: SCARTATA [{re_st}]")

            # --prove-vincitore N: N identical twin rollouts with the winner genome from the same
            # state and with the same frame as the candidates, before applying it; the winner is
            # applied anyway
            prove_rec = None
            if args.prove_vincitore > 0:
                print(f"\n  PROVE DEL VINCITORE (D-31b): {args.prove_vincitore} rollout identici")

                def _esegui_prova(tag):
                    return rollout(best_g, isaac, mola, node, wps, args.horizon, tag,
                                   warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                                   attack_from=args.attack_from, stop_on=args.stop_on, untracked=args.untracked,
                                   nominal_ok=nominal_ok, track_ref=nom_ratio, settle_fixed=nom_settle_s,
                                   drift_reset_on=args.drift_reset_on, frame_start=frame_start,
                                   trace_path=(out / "traces" / f"{tag}.csv") if args.trace else None)

                def _misura_prova(res):
                    tp = true_path(node.last_trace or [])
                    try:
                        dm_p = window_damage(res[1], nom_end, goal, nom_path, tp, args.damage, args.dir_weight,
                                             baseline=baseline_poly)
                    except (ValueError, IndexError, TypeError):
                        dm_p = None
                    return dm_p, (wrec.danno_metro(tp, baseline_poly) if baseline_poly is not None else None)
                prove_rec = prove_del_vincitore(args.prove_vincitore, k, _esegui_prova, _misura_prova, node, mola,
                                                danno_ricerca_cm=-float(best_F[0]) * 100)
            # --riallinea-dopo-assestamento: twin realigned at the end of victim settling, inside
            # its rollout (RiallineoVittima)
            rial = None
            if args.riallinea_dopo_assestamento and victim is not None:
                def _esegui_rial(isa_p, tag, fs):
                    return rollout(best_g, isa_p, mola, node, wps, args.horizon, tag,
                                   warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                                   attack_from=args.attack_from, stop_on=args.stop_on, untracked=args.untracked,
                                   nominal_ok=nominal_ok, track_ref=nom_ratio, settle_fixed=nom_settle_s,
                                   drift_reset_on=args.drift_reset_on, frame_start=fs,
                                   trace_path=(out / "traces" / f"{tag}.csv") if args.trace else None,
                                   assestamento_nuvole=0)
                rial = RiallineoVittima(k, _esegui_rial, shadow, attacker, out, args.twin_init_vel, mola=mola)

            # Actual application; with --trace the per-tick trace of this rollout is also saved, the
            # attacked trajectory to compare with the nominal.
            _, real_end, ch, st, trav, frames = rollout(
                best_g, isaac, mola, node, wps, args.horizon, f"w{k}_applied",
                warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from, stop_on=args.stop_on,
                untracked=args.untracked, nominal_ok=nominal_ok, track_ref=nom_ratio,
                settle_fixed=nom_settle_s, drift_reset_on=args.drift_reset_on,
                victim=victim,
                trace_path=(out / "traces" / f"w{k}_applied.csv")
                if args.trace else None,
                **({"dopo_assestamento": rial} if rial is not None else {}))
            # Winner trials, twin that chose, realigned twin and comparisons (a) and (b); empty
            # without the options
            d31_rec = {}
            d31_errore = None
            if prove_rec is not None:
                d31_rec["prove_vincitore"] = prove_rec
            if rial is not None:
                rial.alla_fine(node)
                # Realignment requested and not done (or without generator state): the run stops as
                # a configuration error, after writing the window
                d31_errore = errore_riallineo(rial.rec) if real_end is not None else None
                g_r = (rial.rec or {}).get("gemello")
                if g_r is not None and rial.trace_rows:
                    g_r["traccia"] = str(out / "traces" / f"{rial.tag}.csv") if args.trace else None
                    g_r["danno_metro"] = (wrec.danno_metro(true_path(rial.trace_rows), baseline_poly)
                                          if baseline_poly is not None else None)
                d31_rec["riallineato"] = rial.rec
            if args.confronto_vincitore or rial is not None:
                i_sc = wrec.winner_index(eval_log, [float(v) for v in best_g], -float(best_F[0]),
                                         (sel_rec or {}).get("chosen_idx"))
                sc = gemello_scelto(eval_log, k, i_sc, out)
                d31_rec["gemello_scelto"] = sc
                rr = rial.rec if rial is not None else None
                d31_rec["confronti_d31"] = confronti_d31(
                    baseline_poly, list(node.last_trace or []) if real_end is not None else [],
                    getattr(victim, "log_path", None) if victim is not None else None,
                    getattr(node, "last_victim_open_stamp", None), sc,
                    riallineato=(rr or {}).get("gemello"), riallineato_rows=rial.trace_rows if rial else None,
                    stamp_ripresa=((rr or {}).get("vittima_alla_pausa") or {}).get("est_stamp_raw"),
                    # dm_real as below: victim damage measured (attack not detected)
                    dm_real=bool(real_end is not None and not (args.chiudi_attacco_scoperto and victim is not None
                                                               and wrec.vittima_persa(st))),
                    frame_log=CONTROLLO_ETA.get("frame_log") or getattr(args, "frame_log_nodo", None),
                    traccia_vittima=str(out / "traces" / f"w{k}_applied.csv") if args.trace else None,
                    attendi_fino=((getattr(node, "last_drive_wall", None) or (None, time.time()))[1]
                                  + FRAME_LOG_FLUSH_S + 0.5))
                for c_, x_ in (d31_rec["confronti_d31"] or {}).items():
                    if c_ in ("a", "b"):
                        print(f"  confronto ({c_}) vittima-{x_['gemello']}: rami {x_['ramo_vittima']}/{x_['ramo_gemello']}, "
                              f"R1 {x_['R1_cm']} cm, laterale {x_['laterale_cm']} cm, danno {x_['danno_cm']} cm -> "
                              f"{'segue' if x_['segue'] else 'non segue'}")
            if ASSETTO["registra"]:
                d31_rec["assetto_applicato"] = getattr(node, "last_assetto", None)
            if RAMPA["R"] > 1:
                # previous genome and zones of the metre, messages, victim ramp at the end of the
                # stretch
                d31_rec["rampa"] = {**(RAMPA.get("finestra") or {}), "vittima": node.stato_rampa("vittima")}
            if GIM["S"] > 0 and GIM["finestra"] is not None:
                # --genoma-in-moto: per stream step state, entry of N, phase at the realignment save
                # (victim and realigned twin), per-phase Chamfer
                fl = GIM["finestra"]["flussi"]
                if rial is not None and rial.rec is not None:
                    fs = gim_fase_salvataggio(((rial.rec.get("generatore_salvato") or {}).get("posto")))
                    for nome in ("vittima", rial.tag):
                        if nome in fl:
                            fl[nome]["fase_al_salvataggio_D31c"] = fs
                d31_rec["genoma_in_moto"] = GIM["finestra"]
            if GIM["cpf"] and GIM["chamfer_fasi"] is not None:
                # --chamfer-per-fase: Chamfer per stream and phase (without step everything in N)
                d31_rec["chamfer_fasi"] = GIM["chamfer_fasi"]
            # Waypoints passed by the applied rollout advance the window waypoint: with only the
            # distance from the saved point (below) a window that passes the corner by more than
            # wp_tol would send the robot back in the next window.
            wp_idx = min(len(goals_abs) - 1, wp_idx + getattr(node, "last_wps_reached", 0))
            applied_quality = _quality_mean(node.last_trace)
            applied_loc_mean = node.last_loc_err_mean * 100 if node.last_loc_err_mean == node.last_loc_err_mean else None
            applied_loc_end = node.last_loc_err_end * 100 if node.last_loc_err_end == node.last_loc_err_end else None
            applied_loc_mean_raw = node.last_loc_err_mean_raw * 100 if node.last_loc_err_mean_raw == node.last_loc_err_mean_raw else None
            applied_perc = node.last_perc or None
            victim_win = _victim_window_record(node, victim, baseline_poly)
            _refresh_victim_scans(history, victim)
            # --chiudi-attacco-scoperto: a victim that stops publishing in the stretch (pose_stale)
            # closes the run and the window does not count as damage
            danno_escluso = bool(args.chiudi_attacco_scoperto and victim is not None
                                 and real_end is not None and wrec.vittima_persa(st))
            if danno_escluso:
                drift = None
                dm_real = None
                print(f"\n  VITTIMA SENZA POSA NEL TRATTO [{st}]: attacco scoperto, finestra fuori dal danno")
            elif real_end is not None:
                dm_real = window_damage(real_end, nom_end, goal, nom_path,
                                        true_path(node.last_trace),
                                        args.damage, args.dir_weight, baseline=baseline_poly)
                drift = dm_real["dev_end"]
                print(f"\n  APPLICATO: il robot e' in {np.round(real_end,3)}")
                print(f"  percorsi {trav:.2f} m su {frames} frame perturbati")
                print(f"  deviazione reale {ref_lbl}: fine {drift*100:.1f}  "
                      f"media {dm_real['dev_mean']*100:.1f}  "
                      f"max {dm_real['dev_max']*100:.1f} cm   "
                      f"danno {dm_real['damage']*100:.1f} cm "
                      f"(in valutazione era {-best_F[0]*100:.1f} cm)   [{st}]")
            else:
                drift = None
                dm_real = None
                applied_quality = None
                applied_loc_mean = applied_loc_end = applied_loc_mean_raw = None
                applied_perc = None
                print(f"\n  APPLICAZIONE FALLITA [{st}]")
                if str(st).startswith("victim_"):
                    victim_lost = True
            # --danno-metro: metre quantities for the victim (only if the damage counts), for the
            # winner in the search, and Chamfer of the two stretches
            danno_metro_rec = {}
            if getattr(args, "danno_metro", False):
                i_win = wrec.winner_index(eval_log, [float(v) for v in best_g], -float(best_F[0]),
                                          (sel_rec or {}).get("chosen_idx"))
                e_win = eval_log[i_win] if i_win is not None else {}
                danno_metro_rec = {
                    "danno_metro_vittima": (wrec.danno_metro(true_path(node.last_trace), baseline_poly)
                                            if dm_real else None),
                    "danno_metro_vincitore": e_win.get("danno_metro"),
                    "vincitore_idx": i_win,
                    "prima_icp_vincitore": wrec.prima_icp_gemello(e_win),
                    "chamfer_vincitore_cm": e_win.get("chamfer"),
                    "chamfer_applicato_cm": (float(ch) if real_end is not None and ch == ch else None),
                    "danno_metro_definizioni": wrec.DANNO_METRO_DEFINITIONS,
                }
            print(f"  tempo finestra: {(time.time()-t_win)/60:.1f} min\n")
            if genoma_nullo_rec is not None:
                # null genome: no search winner (not evaluated in the twin); P measured on the
                # victim in the applied stretch, as for winners
                genoma_nullo_rec["P_misurato_cm"] = (float(ch) if real_end is not None and ch == ch else None)
                genoma_nullo_rec["P_campo"] = "chamfer_applicato_cm (Chamfer della vittima nel tratto applicato)"
                genoma_nullo_rec["deriva_vittima_dopo_applicazione"] = deriva_vittima_status(node)
                if danno_metro_rec:
                    danno_metro_rec.update(vincitore_idx=None, danno_metro_vincitore=None, chamfer_vincitore_cm=None)
                if prove_rec is not None:
                    prove_rec["danno_ricerca_cm"] = None
                if "gemello_scelto" in d31_rec:
                    d31_rec["gemello_scelto"] = None

            # Window computational cost: time and peak VRAM via nvidia-smi. NSGA-III runs on CPU, so
            # VRAM depends on Isaac Sim rendering.
            t_window_s = time.time() - t_win
            if first_win_time is None and not win_timeout:
                first_win_time = t_window_s
                if args.window_cap_factor:
                    win_cap = args.window_cap_factor * first_win_time
                    print(f"  watchdog: tetto per le finestre successive {win_cap/60:.1f} min "
                          f"({args.window_cap_factor}x {first_win_time/60:.1f} min)")
            vram_mib = None
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                vram_mib = int(r.stdout.strip().splitlines()[0])
            except Exception:
                pass

            # Window metrics besides damage: heading error, frames per metre (time lost), candidates
            # that paralyse or lose tracking.
            statuses = [e["status"] for e in eval_log]
            heading_err_deg = (math.degrees(angle_diff(real_end[2], nom_end[2]))
                               if real_end is not None and nom_end is not None else None)
            if METRO["on"]:
                metro_nc = [e for e in METRO["non_confermati"]]
                METRO["non_confermati"].clear()
            zone_fin = ZONE.get("finestra") if ZONE["on"] else None
            history.append({
                "window": k,
                **({"metro_non_confermati": metro_nc} if METRO["on"] else {}),
                **({"zone_attaccante": zone_fin} if ZONE["on"] else {}),
                "heading_error_deg": heading_err_deg,
                "frames_per_m_nominal": nom_frames / max(nom_trav, 1e-6) if nom_assente is None else None,
                "frames_per_m_applied": (frames / max(trav, 1e-6)) if real_end is not None else None,
                "n_stopped_early": sum(1 for st_ in statuses if str(st_).startswith("stopped_early")),
                "n_untracked": sum(1 for st_ in statuses if str(st_).startswith("untracked")),
                "n_timeout": sum(1 for st_ in statuses if str(st_).startswith("timeout")),
                "n_failed": sum(1 for e in eval_log if outcome_kind(str(e["status"])) == "infra"),
                "n_counted": sum(1 for e in eval_log if e.get("counted")),
                "n_discarded": sum(1 for e in eval_log if not e.get("counted")),
                "fitness_riserva": fitness_riserva_finestra(eval_log),
                # with pareto_maxdamage there is no re-evaluation: no reeval field
                **({} if args.select == "pareto_maxdamage" else {"reeval": reeval}),
                "select": args.select,
                **({"select_only_valid": True, "n_valid_selection": n_valid_sel}
                   if args.select_only_valid else {}),
                "selection": sel_rec,
                "nominal_end": (list(map(float, nom_end)) if nom_end is not None else None),
                "applied_end": list(map(float, real_end)) if real_end else None,
                "goal": list(map(float, goal)),
                "waypoint_index": wp_idx,
                "genome": [float(v) for v in best_g],
                "damage_cm": float(-best_F[0]) * 100,
                "damage_mode": args.damage,
                "dev_real_cm": (drift * 100) if drift is not None else None,
                "dev_real_mean_cm": (dm_real["dev_mean"] * 100) if dm_real else None,
                "dev_real_max_cm": (dm_real["dev_max"] * 100) if dm_real else None,
                "damage_real_cm": (dm_real["damage"] * 100) if dm_real else None,
                **({"danno_escluso": "attacco_scoperto"} if danno_escluso else {}),
                **({"dev_real_mean_window_cm": dm_real["dev_mean_window"] * 100,
                    "dev_real_end_window_cm": dm_real["dev_end_window"] * 100,
                    "damage_real_window_cm": dm_real["damage_window"] * 100}
                   if dm_real and "damage_window" in dm_real else {}),
                "applied_status": st,
                "applied_travelled": trav,
                "nominal_loc_err_mean_cm": nom_loc_mean,
                "nominal_loc_err_end_cm": nom_loc_end,
                "nominal_loc_err_mean_raw_cm": nom_loc_mean_raw,
                "nominal_loc_err_end_raw_cm": nom_loc_end_raw,
                "applied_loc_err_mean_cm": applied_loc_mean,
                "applied_loc_err_end_cm": applied_loc_end,
                "applied_loc_err_mean_raw_cm": applied_loc_mean_raw,
                # without nominal the constant reference is not a measured ratio
                "nominal_track_ratio": None if args.no_nominal_rollout else nom_ratio,
                "track_threshold": 0.60,
                "nominal_settle_s": nom_settle_s,
                "twin_state": args.twin_state, "replay_last_s": args.replay_last_s,
                "nominal_settle_ok": nom_settle_ok,
                "perceptibility": args.perceptibility,
                "nominal_quality_mean": nom_quality, "nominal_it_mean": nom_it,
                "nominal_perc": nom_perc or None,
                "nominal_it_cov": nom_stats.get("coverage"),
                "settle_used_s": nom_settle_s,
                "applied_perc": applied_perc,
                "perturbation": float(best_F[1]),
                "ratio": float(ratio),
                "pareto_F": np.atleast_2d(res_F).tolist(),
                "search": args.search,
                **({"fixed_genome_seq": seq_rec} if seq_rec is not None else {}),
                "window_rng": args.window_rng,
                "evaluations": eval_log,
                "window_time_s": t_window_s,
                "watchdog_timeout": win_timeout, "window_cap_s": win_cap,
                "n_evaluations": len(eval_log),
                "vram_mib": vram_mib,
                "pop": args.pop, "gen": args.gen,
                "horizon_m": args.horizon,
                "genome_group": args.genome_group,
                "active_genes": active_genes,
                "attack_from": args.attack_from,
                "untracked_policy": args.untracked,
                "discard_policy": args.discard_policy,
                "chamfer_budget": args.chamfer_budget,
                "damage_ref": args.damage_ref,
                "baseline_path": args.baseline_path if baseline_poly is not None else None,
                **nom_base_fields,
                "n_inadmissible": sum(1 for e in eval_log if e.get("admissible") is False),
                "admissibility": args.admissibility,
                "n_dos": sum(1 for e in eval_log if e.get("dos")),
                "n_stealth_ok": sum(1 for e in eval_log if e.get("stealth_ok")),
                "n_inadmissible_stealth": sum(1 for e in eval_log if e.get("stealth_ok") is False),
                "dos_reasons": {r: sum(1 for e in eval_log if e.get("dos_reason") == r)
                                for r in sorted({e.get("dos_reason") for e in eval_log if e.get("dos_reason")})},
                "inadmissible_reasons": {r: sum(1 for e in eval_log if e.get("inadmissible_reason") == r)
                                for r in sorted({e.get("inadmissible_reason") for e in eval_log if e.get("inadmissible_reason")})},
                **({"incompleti_classi": {c: sum(1 for e in eval_log if e.get("incompleto_classe") == c)
                                          for c in wrec.CLASSI_INCOMPLETO},
                    "incompleti_definizioni": wrec.CLASSI_INCOMPLETO}
                   if getattr(args, "incompleti_regola", False) else {}),
                "stop_reasons": {r: sum(1 for e in eval_log if e.get("stop_reason") == r)
                                for r in sorted({e.get("stop_reason") for e in eval_log if e.get("stop_reason")})},
                "stop_on": args.stop_on,
                "window_stop": args.window_stop,
                "drift_reset_on": args.drift_reset_on,
                "max_drift_lag_s": node.max_drift_lag_s,
                "nominal_motion": nom_motion,
                "applied_motion": motion_vittima(node) if real_end is not None else None,
                "victim_lost": victim_lost,
                "victim": args.victim,
                "twin_align": args.twin_align if attacker is not None else "true",
                "attacker_pose": list(map(float, frame_start)) if frame_start is not None else None,
                "attacker_align_err_cm": att_err_cm,
                "attacker_align_err_deg": att_err_deg,
                "applied_quality_mean": applied_quality,
                "applied_track_ratio": node.last_track_ratio if real_end is not None else None,
                "victim_settle_s": victim.settle_s if victim is not None else None,
                "n_paralisi_deriva": sum(1 for e in eval_log if e.get("motion_class") == "paralisi_deriva"),
                "n_rotazione_assestamento": sum(1 for e in eval_log if e.get("motion_class") == "rotazione_assestamento"),
                "n_arresto_altro": sum(1 for e in eval_log if e.get("motion_class") == "arresto_altro"),
                "warm_start": args.warm_start,
                # warm_start_from_prev = copies + perturbations
                "warm_start_from_prev": (warm_counts or {}).get("warm_start_from_prev", 0),
                "warm_front_n": n_warm, "warm_mut_n": n_mut,
                "warm_mut_identical_n": (warm_counts or {}).get("warm_mut_identical_n"),
                # front used by the warm start (indices in the eval_log of the given window) and
                # front computed here for the next one
                "warm_front_used": warm_used,
                "warm_front_next": (warm_src if args.search == "nsga3" and warm_src
                                    and warm_src.get("window") == k else None),
                "warm_front_after_reeval": args.warm_front_after_reeval,
                "pause_start_wall": node.pause_start_wall, "pause_end_wall": node.last_victim_open_wall,
                "pause_s": (node.last_victim_open_wall - node.pause_start_wall) if (node.pause_start_wall and node.last_victim_open_wall) else None,
                "victim_stamp_offset_s": node.victim_stamp_offset,
                # --fresh-cloud-after-restore: arm of the applied stretch and counts since start of
                # restores sent to Isaac (by kind, window and total) and window arms
                **({"fresh_arm_victim": getattr(node, "last_victim_fresh_arm", None),
                    "fresh_arm_n": node.fresh_arm_n, "fresh_arm_ok": node.fresh_arm_ok,
                    "restore_n": restore_counts(node, k),
                    "fresh_arms": [a for a in (getattr(node, "fresh_arm_log", None) or [])
                                   if a.get("window") == k]}
                   if getattr(node, "fresh_cloud", False) else {}),
                "applied_tau_s": node.last_tau,
                "n_slow": sum(1 for e in eval_log if e.get("slow")),
                "n_twin_misaligned": sum(1 for e in eval_log if e.get("twin_misaligned")),
                "twin_filter": args.twin_filter,
                "n_twin_filtered": sum(1 for e in eval_log if e.get("twin_filtered")),
                "victim_window": victim_win,
                **danno_metro_rec,
                **d31_rec,
                **({"random_first": rf_rec} if rf_rec is not None else {}),
                "sampling": args.sampling,
                "isaac_stall_transient": [e for e in STALL_POLICY["events"] if e.get("window") == k],
                "stall_policy": "tolerate" if STALL_POLICY["tolerate"] else "abort",
                "replay_guard_cm": TWIN_STATE.get("guard_cm") if TWIN_STATE.get("guard") else 0.0,
                "twin_gap0_max_cm": args.twin_gap0_max,
                "n_reevaluated_after_adapter_restart": sum(1 for e in eval_log if e.get("measured_during_adapter_degradation")),
                "node_gen_limits": node_gen_limits(node),
                **structural_fields(node, args),
                **limiti_fields(node),
                "nominal_twin_start": nom_twin_start, **twin_config_fields(args),
            })
            if genoma_nullo_rec is not None:
                # search objectives not measured for the null genome: None, not NaN
                history[-1].update({"damage_cm": None, "perturbation": None, "ratio": None,
                                    "genoma_nullo": True, "genoma_nullo_dettagli": genoma_nullo_rec})
            _esito_finestra(history[-1], st if victim is not None else None)
            with open(out / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            if d31_errore:
                raise ErroreConfigurazioneZone(f"D-31c, finestra {k}: {d31_errore}")

    except KeyboardInterrupt:
        print("\nInterrotto.")
    except ShadowSyncFail as e:
        sync_fail = str(e) or "shadow_sync_fail"
        node.stop_robot()
        history.append({"window": k, "status": "shadow_sync_fail", "error": sync_fail,
                        "sync": victim.sync_info if victim is not None else None})
        print(f"\nERRORE D'INFRASTRUTTURA (shadow_sync_fail): {e}\n  la run si ferma qui. History parziale salvata.")
    except ErroreAssetto as e:
        errore_conf = str(e) or "errore_assetto"
        node.stop_robot()
        history.append({"window": k, "status": "errore_configurazione", "error": errore_conf,
                        "twin_init_assetto": args.twin_init_assetto})
        print(f"\nERRORE DI CONFIGURAZIONE (assetto del gemello): {e}\n  la run si ferma qui. History parziale salvata.")
    except ErroreConfigurazioneZone as e:
        errore_conf = str(e) or "errore_configurazione"
        node.stop_robot()
        rec_e = {"window": k, "status": "errore_configurazione", "error": errore_conf,
                 "zone_attaccante": ZONE.get("finestra")}
        history.append(rec_e)
        with open(out / "errore_configurazione.json", "w") as f:
            json.dump(rec_e, f, indent=1, ensure_ascii=False)
        print(f"\nERRORE DI CONFIGURAZIONE (zone dell'attaccante): {e}\n  finestra {k} registrata come "
              f"errore_configurazione, non come attacco; la run si ferma qui. History parziale salvata.")
    except VictimTopicForeign as e:
        foreign = str(e) or "victim_topic_foreign_publisher"
        node.stop_robot()
        history.append({"window": k, "status": "victim_topic_foreign_publisher", "error": str(e)})
        print(f"\nERRORE D'INFRASTRUTTURA: {e}\n  la run si ferma qui. History parziale salvata.")
    except IsaacStall as e:
        stalled = str(e)
        node.watch = None
        node.stop_robot()
        # Stall context: temporal drift at the stall and duration of the still phase (v < STOP_V)
        # that closes the last trace, ongoing or just ended. Used to check whether Isaac stalls
        # follow long paralyses (robot rotating in place).
        tr = node.trace_rows if node.trace_rows else node.last_trace
        still_s = 0.0
        if tr:
            j = len(tr)
            while j > 0 and abs(tr[j - 1]["cmd_v"]) < STOP_V:
                j -= 1
            still_s = tr[-1]["t"] - (tr[j]["t"] if j < len(tr) else tr[-1]["t"])
        stall_info = {"stall_drift_xy_cm": round(node.drift_xy_cm, 2),
                      "stall_still_s": round(still_s, 2),
                      "stall_trace_in_progress": bool(node.trace_rows),
                      "stall_trace_len_s": round(tr[-1]["t"], 2) if tr else None,
                      "stall_last_motion": node.last_motion or None}
        print(f"  contesto: deriva {stall_info['stall_drift_xy_cm']:.1f} cm, fase ferma "
              f"finale {still_s:.1f} s (traccia {'in corso' if node.trace_rows else 'conclusa'})")
        history.append({"window": k, "status": "ombra_persa" if isinstance(e, OmbraPersa) else "isaac_stall",
                        "error": stalled, **stall_info,
                        **({"causa": "eta_nuvola"} if isinstance(e, GuastoEtaNuvola) else {}),
                        **({"causa": "ombra_persa"} if isinstance(e, OmbraPersa) else {})})
        if isinstance(e, OmbraPersa):
            # shadow process terminated: chain failure, the run is repeated entirely
            scrivi_da_rifare_ombra(out, k, stalled, args.seed)
        if isinstance(e, GuastoEtaNuvola):
            # victim cloud age over threshold: chain failure, the run is repeated entirely
            # (run_campaign.sh does not repeat it as a stall)
            with open(out / "da_rifare.json", "w") as f:
                json.dump({"reason": "guasto_catena", "window": k, "status": "isaac_stall",
                           "causa": "eta_nuvola", "motivi": [stalled], "seme": args.seed,
                           "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S")},
                          f, indent=1, ensure_ascii=False)
        print(f"\n{'OMBRA PERSA' if isinstance(e, OmbraPersa) else 'STALLO DI ISAAC'}: {stalled}")
        print(f"  finestra {k + 1 if k is not None else '-'} marcata fallita ({'ombra_persa' if isinstance(e, OmbraPersa) else 'isaac_stall'}); "
              f"la run si ferma qui. History parziale salvata.")
    finally:
        node.watch = None          # shutdown must not check the stall again
        spegni_attacco_fine_run(node)
        node.close()
        mola.stop()
        if victim is not None:
            victim.stop()
        if attacker is not None:
            attacker.stop()
        if shadow is not None:
            shadow.stop()
        if shadow_rec is not None:
            # with MOLA stopped the shadow printf is complete; the victim one and the CSV traces may
            # lose the tail (block-buffered stdout, ofstream without flush,
            # LidarOdometry_Main.cpp:752-755): the comparison stops at the last stamp present in
            # both traces and marks the rest as not comparable
            shadow_rec.close()
            shadow_rec.snapshot(final=True)
            shadow_rec.fill_history(history)
        if victim is not None:
            # MOLA stopped: final MOLA timings for all windows
            _refresh_victim_scans(history, victim, fine_run=True)
        # Isaac Sim stays paused after the last rollout: it is always restarted, even when the
        # orchestrator exits on an error, except after a stall (the command would wait for the
        # timeout for nothing).
        if not stalled:
            try:
                isaac.play()
            except Exception:
                pass
        if getattr(args, "danno_metro", False) and history:
            # end-of-run summary in the last history entry
            history[-1]["danno_run"] = wrec.danno_run(history)
            dr = history[-1]["danno_run"]
            if dr:
                print(f"  danno a fine run (vittima): fine {dr['victim_end_cm']:.1f} cm, "
                      f"media {dr['victim_mean_cm']:.1f} cm su {dr['n_windows']} metri")
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nStorico: {out/'history.json'}")

    # Code 3: stopped for an Isaac stall (run_campaign.sh relaunches the repetition once). Code 4
    # (only with --fail-rc-on-no-pose): the victim stayed without pose in at least one window.
    if lost_state["any"]:
        print(f"  vittima senza posa in almeno una finestra (ultima serie dalla finestra "
              f"{lost_state['since']}); --fail-rc-on-no-pose {'acceso' if args.fail_rc_on_no_pose else 'spento'}")
    # Code 2, as at start-up: run truncated by an infrastructure error, which run_campaign.sh does
    # not mark as completed.
    if foreign or sync_fail or guasto_catena or errore_conf:
        return 2
    return exit_code(stalled, lost_state, args.fail_rc_on_no_pose)


if __name__ == "__main__":
    sys.exit(main())
