"""Live CSV logs of the chain (--registri-dal-vivo, REGISTRI_DAL_VIVO=1).

Three CSV files in <rep>/registri_dal_vivo, header on the first line, comma
separated, each row written and flushed on its own: pose.csv (every MOLA pose
received by the orchestrator, victim and twin), controllore.csv (every
controller cycle: drive, stop, coast) and isaac_cmd.csv (every /cmd_vel received
by Isaac when run_isaac_standalone.py has REGISTRI_DAL_VIVO_DIR, with the wheel
speeds and the true velocities of the next step).

Clocks: wall time is time.time() (same host for all processes); orchestrator
sim time is the stamp of the last /chassis/odom received (the node does not use
use_sim_time nor /clock); Isaac sim time is
SimulationManager.get_simulation_time().

A logging error never stops the chain: the log disables itself with a single
warning.
"""
import math
import sys
import time
from pathlib import Path

POSE_COLONNE = ("ruolo", "tag", "stamp_nuvola_s", "t_pub_src_ns", "t_ric_dds_ns",
                "t_ric_wall_s", "t_ric_sim_s", "x", "y", "yaw")
CONTROLLORE_COLONNE = ("ruolo", "tag", "t_wall_s", "t_sim_s", "stamp_posa_usata_s",
                       "eta_posa_s", "eta_posa_wall_s", "est_x", "est_y", "est_yaw",
                       "est_dist", "cmd_v", "cmd_w", "t_pub_cmd_wall_s", "motivo",
                       "stamp_posa_grezzo_s", "offset_stamp_s")
ISAAC_COLONNE = ("t_ric_wall_s", "t_ric_sim_s", "cmd_v", "cmd_w", "ruota_sx_rad_s",
                 "ruota_dx_rad_s", "step", "t_step_succ_sim_s", "v_vera_step_succ",
                 "w_vera_step_succ")
MOTIVI = ("crociera", "rampa", "coast", "stop", "altro")
NOMI_FILE = {"pose": "pose.csv", "controllore": "controllore.csv", "isaac": "isaac_cmd.csv"}


def _num(v, nd=6):
    """Number with nd decimals; empty for None, NaN, inf or non-numbers."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{f:.{nd}f}" if f == f and abs(f) != math.inf else ""


def motivo_da_fase(fase):
    """Command reason from the trace phase; the coast phase 'fermo' maps to stop."""
    if fase in ("crociera", "rampa", "coast", "stop"):
        return fase
    if fase == "fermo":
        return "stop"
    return "altro"


def eta(t, stamp):
    """Age t - stamp (s); NaN if either is missing."""
    try:
        a, b = float(t), float(stamp)
    except (TypeError, ValueError):
        return float("nan")
    return a - b if a == a and b == b else float("nan")


def _info_ns(info, chiave):
    try:
        v = int(info[chiave])
    except (TypeError, KeyError, ValueError, IndexError):
        return ""
    return str(v) if v > 0 else ""


def source_ns(info):
    """DDS source_timestamp (ns) from the rclpy MessageInfo; empty if missing or zero."""
    return _info_ns(info, "source_timestamp")


def received_ns(info):
    """DDS received_timestamp (ns) from the rclpy MessageInfo; empty if missing or zero."""
    return _info_ns(info, "received_timestamp")


def stamp_s(msg):
    st = msg.header.stamp
    return st.sec + st.nanosec * 1e-9


def yaw_di(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def apri_csv(path, colonne):
    """Open in append mode, line buffered (flushed at every row); header only if
    the file is new or empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", buffering=1)
    if f.tell() == 0:
        f.write(",".join(colonne) + "\n")
    return f


class _Registro:
    """Disables itself at the first error, with a single warning on stderr."""

    attivo = True

    def _disattiva(self, dove, e):
        if self.attivo:
            self.attivo = False
            print(f"[registri_dal_vivo] {dove}: {type(e).__name__}: {e}; registro disattivato, "
                  "la catena continua", file=sys.stderr, flush=True)
            self.close()


class RegistriDalVivo(_Registro):
    """Orchestrator logs pose.csv and controllore.csv."""

    def __init__(self, cartella, orologio=time.time):
        self.cartella = Path(cartella)
        self.tag = ""
        self._orologio = orologio
        self._t0_rollout = -math.inf     # poses received earlier do not belong to the rollout
        self._pose = apri_csv(self.cartella / NOMI_FILE["pose"], POSE_COLONNE)
        self._ctrl = apri_csv(self.cartella / NOMI_FILE["controllore"], CONTROLLORE_COLONNE)

    def inizio_rollout(self, tag):
        """Set rollout tag and start time: a pose received earlier is not used by
        this rollout (rows without an estimate get empty fields)."""
        self.tag = tag
        self._t0_rollout = self._orologio()

    def posa(self, ruolo, msg, info, t_ric_wall, t_ric_sim):
        if not self.attivo:
            return
        try:
            p = msg.pose.pose
            self._pose.write(",".join((
                ruolo, self.tag, _num(stamp_s(msg), 9), source_ns(info), received_ns(info),
                _num(t_ric_wall), _num(t_ric_sim), _num(p.position.x), _num(p.position.y),
                _num(yaw_di(p.orientation)))) + "\n")
        except Exception as e:
            self._disattiva("pose.csv", e)

    def callback_posa(self, ruolo, cb, t_ric_wall_fn, t_sim_fn):
        """Two-argument callback (msg, MessageInfo), so rclpy passes the
        source_timestamp and received_timestamp. Runs the original callback first,
        then writes the row; a logging error does not leave the callback."""
        def _cb(msg, info):
            cb(msg)
            if not self.attivo:
                return
            try:
                t_w, t_s = t_ric_wall_fn(), t_sim_fn()
            except Exception as e:
                self._disattiva("pose.csv", e)
                return
            self.posa(ruolo, msg, info, t_w, t_s)
        return _cb

    def ciclo(self, ruolo, t_wall, t_sim, stamp, t_ric_wall, est, est_dist, v, w,
              t_pub, fase, stamp_grezzo=None, offset=None):
        """One controller row. Without an estimate of the current rollout (est
        None or received before inizio_rollout) estimate, stamp and age are empty.

        eta_posa_s uses the clock offset known when the pose was received.
        stamp_grezzo (pose header stamp) and offset (stamp - raw stamp) are
        appended, empty if not given: with frame_log.csv they give the cloud age
        t_sim - stamp_in without matching pose.csv. The offset is the one applied
        at reception: for the first victim pose after a pause/resume it is still
        the pre-pause offset, so raw stamp + offset is not the cloud stamp there."""
        if not self.attivo:
            return
        try:
            valida = est is not None and t_ric_wall is not None and t_ric_wall >= self._t0_rollout
            if valida:
                ex, ey, eyaw = est
            else:
                ex = ey = eyaw = stamp = t_ric_wall = est_dist = stamp_grezzo = offset = None
            self._ctrl.write(",".join((
                ruolo, self.tag, _num(t_wall), _num(t_sim), _num(stamp, 9),
                _num(eta(t_sim, stamp)), _num(eta(t_wall, t_ric_wall)),
                _num(ex), _num(ey), _num(eyaw), _num(est_dist), _num(v, 4), _num(w, 4),
                _num(t_pub), motivo_da_fase(fase), _num(stamp_grezzo, 9), _num(offset, 9))) + "\n")
        except Exception as e:
            self._disattiva("controllore.csv", e)

    def close(self):
        for f in (getattr(self, "_pose", None), getattr(self, "_ctrl", None)):
            try:
                f.close()
            except Exception:
                pass


class RegistroIsaacCmd(_Registro):
    """isaac_cmd.csv. Commands received on the rclpy thread are queued and
    written after the next step, with the wheel speeds and the robot's true
    velocities at that step."""

    def __init__(self, cartella, orologio=time.time):
        import collections
        import threading
        self._f = apri_csv(Path(cartella) / NOMI_FILE["isaac"], ISAAC_COLONNE)
        self._attesa = collections.deque()
        self._lock = threading.Lock()
        self._orologio = orologio
        self.t_sim = float("nan")        # sim time of the last completed step
        self.step = None                 # step counter at the last update

    def comando(self, v, w, t_ric_wall=None):
        """Queue a received command with wall time and sim time of the last step."""
        if not self.attivo:
            return
        try:
            t = self._orologio() if t_ric_wall is None else t_ric_wall
            with self._lock:
                self._attesa.append((t, self.t_sim, float(v), float(w)))
        except Exception as e:
            self._disattiva("isaac_cmd.csv (ricezione)", e)

    def prima_dello_step(self):
        """Number of commands queued before the step about to start."""
        with self._lock:
            return len(self._attesa)

    def dopo_lo_step(self, n, step, t_sim, ruote_fn, vel_fn):
        """Write the n commands queued before the update just completed, if the
        step counter advanced (while paused they wait for the first real step);
        ruote_fn and vel_fn are called only when there is a row to write."""
        avanzato = step != self.step
        self.step = step
        self.t_sim = t_sim
        if n <= 0 or not avanzato or not self.attivo:
            return 0
        with self._lock:
            righe = [self._attesa.popleft() for _ in range(min(n, len(self._attesa)))]
        try:
            sx, dx = ruote_fn()
        except Exception:
            sx = dx = None
        try:
            vv, ww = vel_fn()
        except Exception:
            vv = ww = None
        try:
            for t_w, t_s, v, w in righe:
                self._f.write(",".join((
                    _num(t_w), _num(t_s), _num(v, 4), _num(w, 4), _num(sx), _num(dx),
                    str(int(step)), _num(t_sim), _num(vv), _num(ww))) + "\n")
        except Exception as e:
            self._disattiva("isaac_cmd.csv", e)
            return 0
        return len(righe)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
