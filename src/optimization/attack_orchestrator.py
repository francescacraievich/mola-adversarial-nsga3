#!/usr/bin/env python3
"""
Orchestratore dell'attacco adversarial receding-horizon.

Ad ogni finestra di H metri: salva lo stato di Isaac Sim, valuta N candidati con
un rollout fisico (ripristino dello stato, riavvio di MOLA, carico del genoma,
H metri di guida, misura della posa finale e della percettibilita'), ordina con
NSGA-III, sceglie dal fronte di Pareto e applica il vincente per un tratto vero.

La fitness e' misurata con il rollout e non stimata con un modello cinematico:
il ripristino dello stato e' fedele e non servono assunzioni sul moto futuro.
MOLA non espone un servizio di reset e viene riavviata a ogni candidato, altrimenti
mappa locale e sigma adattivo del candidato precedente falserebbero il confronto.
A ogni riavvio MOLA rilocalizza in (0,0,0) nella posa corrente del robot, quindi
il waypoint va riespresso in quel frame a partire dalla posa vera.

Uso:
    python3 -u src/optimization/attack_orchestrator.py --goal "5,0" --horizon 1.0 \
        --settle-sec 2.0 --pop 4 --gen 2 --trace --out data/attack/<nome>

Prerequisiti: Isaac Sim in Play con isaac/isaac_rollout_server.py caricato nello
Script Editor, `python3 src/nodes/add_intensity_node.py ...` e
`python3 src/nodes/perturbation_node.py` in esecuzione.

Legge /lidar_odometry/pose, /lidar_odometry/pose_quality e /attack/status;
pubblica /cmd_vel, /attack/genome e /attack/enabled. Dialoga con Isaac Sim via
/tmp/isaac_cmd.json e /tmp/isaac_reply.json. Scrive history.json, logs/ e
traces/ nella cartella --out.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402
from src.optimization.trajectory_predictor import FollowerModel, angle_diff  # noqa: E402

CMD_FILE = "/tmp/isaac_cmd.json"
REPLY_FILE = "/tmp/isaac_reply.json"


# ---------------------------------------------------------------------------
# Lato Isaac Sim
# ---------------------------------------------------------------------------

class IsaacClient:
    """Comunicazione con isaac/isaac_rollout_server.py via file.

    Scambio a file e non via ROS perche' rclpy non e' disponibile dentro
    l'interprete di Isaac Sim. Il campo id distingue una risposta nuova da una
    vecchia rimasta su disco.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._id = 0
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

        deadline = time.time() + self.timeout
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
                    return r
            time.sleep(0.02)
        raise TimeoutError(
            f"Isaac non risponde a '{cmd}'. isaac_rollout_server.py e' in "
            f"esecuzione nello Script Editor?")

    def save(self):    return self.call("save")["pose"]
    def restore(self): return self.call("restore")["pose"]
    def pose(self):    return self.call("pose")["pose"]
    def play(self):    self.call("play")
    def pause(self):   self.call("pause")


# ---------------------------------------------------------------------------
# Lato MOLA
# ---------------------------------------------------------------------------

class MolaRunner:
    """Avvia e ferma l'istanza di MOLA sotto attacco."""

    def __init__(self, script: str, log_dir: Path, ready_timeout: float = 40.0,
                 extra_env: dict | None = None, traces: bool = False):
        self.script = script
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ready_timeout = ready_timeout
        self.extra_env = extra_env or {}
        self.traces = traces
        self.proc = None
        self.log_path = None

    def spawn(self, tag: str):
        """Avvia il processo senza attendere che sia pronto.

        MOLA si dichiara pronta solo dopo la prima nuvola, che a simulazione
        ferma non arriva: avviare e attendere in un colpo solo con Isaac Sim in
        pausa porta a un timeout.
        """
        self.stop()
        # Processi sopravvissuti a stop() pubblicherebbero sullo stesso topic
        # e le pose si mescolerebbero.
        left = self._count_processes()
        if left:
            raise RuntimeError(
                f"{left} processi mola-cli ancora vivi prima dell'avvio: "
                f"pubblicherebbero sullo stesso topic e le pose si "
                f"mescolerebbero. Controlla con: ps aux | grep '[m]ola-cli'")
        self.log_path = self.log_dir / f"mola_{tag}.log"
        f = open(self.log_path, "w")
        self._f = f
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.extra_env.items()})
        if self.traces:
            # Tracce per-scan di MOLA: qualita' ICP, sigma adattivo, twist stimato.
            env["TRACES"] = str(self.log_dir / f"traces_{tag}.csv")
        self.proc = subprocess.Popen(
            ["bash", self.script], stdout=f, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, env=env)

    def wait_ready(self):
        """Attende la rilocalizzazione iniziale. Richiede la simulazione in play."""
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(f"mola-cli terminato subito ({self.log_path})")
            try:
                if "Initial re-localization done" in self.log_path.read_text():
                    return True
            except OSError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"MOLA non pronto ({self.log_path})")

    def start(self, tag: str):
        """Avvia e attende. Usare solo a simulazione in play."""
        self.spawn(tag)
        return self.wait_ready()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def crashed(self) -> bool:
        """Vero se il processo e' terminato da solo.

        Un genoma aggressivo puo' produrre nuvole degeneri e far cadere MOLA:
        si tratta come valutazione fallita, non come errore fatale.
        """
        return self.proc is not None and self.proc.poll() is not None

    @staticmethod
    def _count_processes() -> int:
        """Numero di mola-cli vivi secondo la tabella dei processi.

        Non si usa il grafo ROS: la discovery impiega secondi a rimuovere un
        publisher morto e darebbe falsi positivi subito dopo una terminazione.
        """
        r = subprocess.run(["pgrep", "-f", "mola-cli"], capture_output=True,
                           text=True)
        return len([ln for ln in r.stdout.splitlines() if ln.strip()])

    @classmethod
    def kill_orphans(cls, timeout: float = 15.0) -> int:
        """Termina ogni mola-cli e attende che sia sparito dalla tabella dei processi.

        Prima SIGTERM, che mola-cli gestisce chiudendo le connessioni DDS; SIGKILL
        solo come ultima risorsa. Il publisher di /lidar_odometry/pose ha liveliness
        infinita: senza il messaggio di deregistrazione la sua voce resta nel grafo
        ROS anche a processo morto.

        Restituisce il numero di processi ancora vivi.
        """
        if cls._count_processes() == 0:
            return 0
        subprocess.run(["pkill", "-TERM", "-f", "mola-cli"], capture_output=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cls._count_processes() == 0:
                return 0
            time.sleep(0.2)
        subprocess.run(["pkill", "-9", "-f", "mola-cli"], capture_output=True)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if cls._count_processes() == 0:
                return 0
            time.sleep(0.2)
        return cls._count_processes()

    def stop(self):
        """Ferma il processo lasciandogli il tempo di deregistrarsi da DDS."""
        if self.proc is None:
            try:
                self._f.close()
            except Exception:
                pass
            return

        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass

        deadline = time.time() + 12.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        else:
            # SIGKILL lascia una voce stantia nel grafo ROS per alcuni secondi.
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=3.0)
            except Exception:
                pass

        self.proc = None
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Nodo ROS dell'orchestratore
# ---------------------------------------------------------------------------

class AttackNode:
    """Guida il robot durante i rollout e dialoga con il nodo di perturbazione.

    Il controllo e' interno e non delegato a waypoint_follower_node: serve
    fermarsi con precisione a H metri veri e riesprimere il waypoint nel frame
    di MOLA a ogni riavvio.
    """

    def __init__(self, follower: FollowerModel):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32MultiArray, String

        self._rclpy = rclpy
        self._Twist = Twist
        self._F32 = Float32MultiArray
        rclpy.init()
        self.node = Node("attack_orchestrator")

        self.est = None            # posa stimata da MOLA
        self.est_t = 0.0
        self.chamfer = []          # percettibilita' riportata dal nodo
        self.frames_out = 0        # nuvole perturbate, cumulativo
        self.trace_rows = None     # se non None, drive() vi accumula la traccia
        self._last_true = None
        # Qualita' di registrazione riportata da MOLA: canale di rilevabilita'
        # indipendente dalla Chamfer, perche' un difensore non dispone della
        # nuvola originale ma vede se lo SLAM registra bene.
        self.pose_quality = float("nan")
        self.last_track_ratio = float("nan")   # rapporto stima/verita' dell'ultimo rollout

        self.node.create_subscription(Odometry, "/lidar_odometry/pose",
                                      self._est_cb, 100)
        self.node.create_subscription(String, "/attack/status",
                                      self._status_cb, 10)
        from std_msgs.msg import Float32
        self.node.create_subscription(Float32, "/lidar_odometry/pose_quality",
                                      self._quality_cb, 50)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.genome_pub = self.node.create_publisher(
            Float32MultiArray, "/attack/genome", 10)
        self.follower = follower

    def _est_cb(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.est = (p.position.x, p.position.y, yaw)
        self.est_t = time.time()

    def _status_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        # frames_out e' cumulativo: il conteggio per finestra e' la differenza
        # fra inizio e fine del rollout.
        if d.get("frames_out") is not None:
            self.frames_out = int(d["frames_out"])
        # chamfer_cm e' None in passthrough o prima del primo campione:
        # float(None) nel callback farebbe cadere lo spin dell'executor.
        v = d.get("chamfer_cm")
        if v is None:
            return
        try:
            self.chamfer.append(float(v))
        except (TypeError, ValueError):
            pass

    def _quality_cb(self, msg):
        self.pose_quality = float(msg.data)

    def count_mola_publishers(self) -> int:
        """Publisher su /lidar_odometry/pose nel grafo ROS (diagnostica, atteso 1)."""
        return self.node.count_publishers("/lidar_odometry/pose")

    def spin(self, sec: float):
        t0 = time.time()
        while time.time() - t0 < sec:
            self._rclpy.spin_once(self.node, timeout_sec=0.01)

    def set_genome(self, genome):
        m = self._F32()
        m.data = [float(v) for v in genome]
        self.genome_pub.publish(m)
        self.spin(0.2)

    def set_attack_enabled(self, on: bool):
        """Accende o spegne la perturbazione senza riavviare il nodo."""
        from std_msgs.msg import Bool
        if not hasattr(self, "_enable_pub"):
            self._enable_pub = self.node.create_publisher(
                Bool, "/attack/enabled", 10)
        m = Bool()
        m.data = bool(on)
        self._enable_pub.publish(m)
        self.spin(0.2)

    def stop_robot(self):
        self.cmd_pub.publish(self._Twist())

    def drive(self, local_wp, distance, isaac: IsaacClient,
              mola: MolaRunner, timeout=45.0, rate=20.0,
              check_every=0.25, over_run=1.30, arrival_tol=0.30):
        """Guida il robot fino a `distance` metri veri dal punto di partenza.

        L'arresto usa la posa vera di Isaac Sim e non la stima di MOLA: fermarsi
        quando MOLA crede di aver percorso il tratto rende la fitness illimitata,
        perche' un attacco che fa sottostimare la traslazione tiene il robot in
        moto e la deviazione cresce da sola.

        Esiti: ok, arrived (waypoint entro arrival_tol), overrun (tratto superato
        del fattore over_run senza arresto), timeout, mola_crash, pose_stale,
        no_pose. Restituisce (esito, metri percorsi).
        """
        self.est = None
        t_wait = time.time()
        while self.est is None and time.time() - t_wait < 15.0:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            if mola.crashed():
                return "mola_crash", 0.0
        if self.est is None:
            return "no_pose", 0.0

        true_start = np.array(isaac.pose()[:2])
        self._last_true = isaac.pose()
        dt = 1.0 / rate
        t0 = time.time()
        t_last_check = 0.0
        travelled = 0.0

        while time.time() - t0 < timeout:
            self._rclpy.spin_once(self.node, timeout_sec=0.005)
            if mola.crashed():
                self.stop_robot()
                return "mola_crash", travelled
            if self.est is None or time.time() - self.est_t > 3.0:
                self.stop_robot()
                return "pose_stale", travelled

            x, y, yaw = self.est

            # Waypoint entro la tolleranza: il follower comanda velocita' nulla
            # e il tratto non verrebbe mai completato.
            if math.hypot(local_wp[0] - x, local_wp[1] - y) < arrival_tol:
                self.stop_robot()
                return "arrived", travelled

            # La posa vera passa dall'IPC con Isaac Sim: interrogarla a ogni
            # ciclo rallenterebbe il controllo. A 0.8 m/s, 250 ms sono 20 cm.
            now = time.time() - t0
            if now - t_last_check >= check_every:
                t_last_check = now
                try:
                    p = isaac.pose()
                    self._last_true = p
                    travelled = float(np.linalg.norm(
                        np.array(p[:2]) - true_start))
                except Exception:
                    pass
                if travelled >= distance:
                    self.stop_robot()
                    return "ok", travelled
                if travelled >= distance * over_run:
                    self.stop_robot()
                    return "overrun", travelled

            v, w, _ = self.follower.command(x, y, yaw, local_wp[0], local_wp[1])

            # Traccia per-tick: stima di MOLA, posa vera e comandi sullo stesso
            # asse dei tempi.
            if self.trace_rows is not None:
                tp = self._last_true if self._last_true else (float("nan"),) * 3
                self.trace_rows.append({
                    "t": round(time.time() - t0, 3),
                    "est_x": round(x, 4), "est_y": round(y, 4),
                    "est_yaw_deg": round(math.degrees(yaw), 2),
                    "true_x": round(tp[0], 4), "true_y": round(tp[1], 4),
                    "true_yaw_deg": round(math.degrees(tp[2]), 2),
                    "wp_x": round(local_wp[0], 3), "wp_y": round(local_wp[1], 3),
                    "cmd_v": round(v, 3), "cmd_w": round(w, 3),
                    "travelled": round(travelled, 4),
                    # Divario stima/verita' e qualita' di registrazione: legano
                    # perturbazione, errore SLAM e deviazione.
                    "est_true_gap": round(math.hypot(x - tp[0], y - tp[1]), 4)
                    if tp[0] == tp[0] else float("nan"),
                    "pose_quality": (round(self.pose_quality, 4)
                                     if self.pose_quality == self.pose_quality
                                     else float("nan")),
                    "chamfer_cm": (round(self.chamfer[-1], 3)
                                   if self.chamfer else float("nan")),
                })

            m = self._Twist()
            m.linear.x = v
            m.angular.z = w
            self.cmd_pub.publish(m)
            time.sleep(dt)

        self.stop_robot()
        return "timeout", travelled

    def close(self):
        self.stop_robot()
        try:
            self.node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Un rollout
# ---------------------------------------------------------------------------

def to_local(goal, true_pose):
    """Waypoint globale espresso nel frame di MOLA appena rilocalizzata.

    A ogni riavvio MOLA pone l'origine nella posa corrente del robot, quindi le
    coordinate del bersaglio cambiano a ogni rollout.
    """
    gx, gy = goal
    x, y, yaw = true_pose
    dx, dy = gx - x, gy - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return (dx * c - dy * s, dx * s + dy * c)


def rollout(genome, isaac, mola, node, goal, horizon, tag, warmup=0.0,
            trace_path=None, track_min=0.60, track_max=None,
            settle_sec=2.0, min_travel_frac=0.70, catchup_sec=0.8):
    """Valuta un candidato: ripristina, riavvia MOLA, percorre H metri, misura.

    Con warmup > 0 il robot percorre prima quel tratto senza perturbazione e la
    misura parte dopo: MOLA riparte senza mappa locale e con il modello di
    velocita' da riconvergere, e la convergenza avviene col movimento, non col
    tempo.

    min_travel_frac e' la frazione minima del tratto da percorrere perche' la
    deviazione sia misurabile: sotto soglia l'esito e' `stopped_early` (l'attacco
    ha paralizzato il robot). Va escluso dal fronte, altrimenti la soluzione
    degenere "robot fermo" ottiene la deviazione massima dal nominale in moto.

    Restituisce (posa_iniziale, posa_finale_vera, chamfer_medio, esito,
    metri_percorsi, frame_perturbati).
    """
    isaac.pause()
    true_start = isaac.restore()
    mola.stop()

    # Avvio senza attesa: la rilocalizzazione iniziale richiede una nuvola, che
    # a simulazione ferma non arriva. Prima lo spawn, poi il play, poi l'attesa.
    t_spawn = time.time()
    try:
        mola.spawn(tag)
    except Exception as e:
        return None, None, float("inf"), f"mola_spawn_fail: {e}", 0.0, 0

    # Durante il riscaldamento l'attacco resta spento, altrimenti il punto di
    # partenza della misura cambierebbe da candidato a candidato.
    node.set_attack_enabled(warmup <= 0.0 and genome is not None)
    if genome is not None:
        node.set_genome(genome)
    node.chamfer = []

    isaac.play()
    try:
        mola.wait_ready()
    except Exception as e:
        isaac.pause()
        mola.stop()
        return None, None, float("inf"), f"mola_not_ready: {e}", 0.0, 0

    # La prova che la nuova MOLA pubblica e' una posa ricevuta dopo lo spawn,
    # non il conteggio dei publisher: la voce DDS della MOLA precedente resta
    # nel grafo per alcuni secondi, e attenderne la rimozione a robot fermo fa
    # decadere il sigma adattivo al pavimento. Il conteggio resta come
    # diagnostica.
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

    track_ratio = float("nan")

    # Assestamento a robot fermo: wait_ready ritorna dopo la prima scansione,
    # con mappa locale vuota e sigma al valore iniziale. Non deve essere lungo:
    # a robot fermo il sigma adattivo decade verso il pavimento e la soglia di
    # accoppiamento ICP diventa confrontabile con il moto fra due scan. E' tempo,
    # non distanza: non consuma frame senza attacco.
    if settle_sec > 0:
        node.spin(settle_sec)

    local_wp = to_local(goal, true_start)

    if warmup > 0.0:
        st_w, _ = node.drive(local_wp, warmup, isaac, mola, timeout=30.0)
        if st_w not in ("ok", "arrived"):
            isaac.pause()
            mola.stop()
            return None, None, float("inf"), f"warmup_{st_w}", 0.0, 0
        # Da qui in poi si misura: nuova origine del tratto e attacco attivo.
        true_start = isaac.pose()
        local_wp = to_local(goal, true_start)
        node.set_attack_enabled(genome is not None)
        node.chamfer = []

    frames_before = node.frames_out
    if trace_path is not None:
        node.trace_rows = []
    est_before = node.est
    status, travelled = node.drive(local_wp, horizon, isaac, mola)

    # La stima di MOLA segue il moto vero con 0.3-0.5 s di latenza (LiDAR a
    # 10 Hz, ICP, stimatore): letta all'istante dell'arresto sottostima il
    # tratto anche a tracking perfetto. Il robot ha gia' ricevuto lo stop; si
    # lascia alla stima il tempo di raggiungerlo.
    if status in ("ok", "arrived") and catchup_sec > 0:
        node.spin(catchup_sec)
    est_after = node.est
    isaac.pause()
    true_end_now = isaac.pose()
    # Spostamento vero letto nello stesso istante della stima finale, cosi' il
    # rapporto resta coerente anche con il coasting dopo lo stop.
    true_disp = math.hypot(true_end_now[0] - true_start[0],
                           true_end_now[1] - true_start[1])

    # Controllo di tracking con soglia asimmetrica: un rapporto stima/verita'
    # molto sotto 1 e' MOLA che perde il tracking da sola, mentre un attacco
    # riuscito porta il rapporto sopra 1. Un limite superiore scarterebbe
    # proprio i candidati piu' efficaci.
    if (status in ("ok", "arrived") and est_before is not None
            and est_after is not None and true_disp > 0.10):
        est_d = math.hypot(est_after[0] - est_before[0],
                           est_after[1] - est_before[1])
        ratio = est_d / max(true_disp, 1e-6)
        if ratio < track_min or (track_max is not None and ratio > track_max):
            status = f"untracked({ratio:.2f})"
        else:
            track_ratio = ratio
    # Esposto anche per i rollout validi, da riportare accanto alla deviazione.
    node.last_track_ratio = track_ratio

    # Tratto troppo breve per misurare una deviazione di rotta: l'attacco ha
    # paralizzato il robot (tipicamente `arrived` a zero metri). Senza questa
    # marcatura NSGA-III sceglie il genoma che ferma il robot.
    if status in ("ok", "arrived") and travelled < min_travel_frac * horizon:
        status = f"stopped_early({travelled:.2f}m)"

    if trace_path is not None and node.trace_rows:
        import csv
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(node.trace_rows[0].keys()))
            wr.writeheader()
            wr.writerows(node.trace_rows)
        node.trace_rows = None

    true_end = true_end_now
    ch = float(np.mean(node.chamfer)) if node.chamfer else float("nan")
    frames = max(0, node.frames_out - frames_before)
    mola.stop()
    return true_start, true_end, ch, status, travelled, frames


# ---------------------------------------------------------------------------

def build_nsga3(pop_size: int):
    """NSGA-III con la configurazione di run_nsga3.py (Das-Dennis, SBX, PM).

    Cambia solo la fitness: rollout fisico al posto del replay offline. Il
    numero di direzioni di riferimento deve restare sotto la popolazione,
    altrimenti pymoo avvisa di comportamento imprevedibile: le partizioni si
    adattano a pop_size e tornano al valore originale 12 da 13 individui in su.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("das-dennis", 2,
                                        n_partitions=max(1, min(12, pop_size - 1)))
    return NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )


def build_problem(n_genes: int, eval_fn):
    """Stessa struttura di MOLAPerturbationProblem in run_nsga3.py."""
    from pymoo.core.problem import Problem

    class WindowProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_genes, n_obj=2,
                            xl=-1.0 * np.ones(n_genes),
                            xu=1.0 * np.ones(n_genes))

        def _evaluate(self, X, out, *a, **kw):
            out["F"] = np.array([eval_fn(g) for g in X])

    return WindowProblem()


def pick_best(F, X):
    """Dal fronte, la soluzione con il rapporto danno/percettibilita' piu' alto.

    F[:,0] e' il danno negato in metri (pymoo minimizza), F[:,1] la
    percettibilita' in cm: il rapporto si calcola in cm su cm perche' il valore
    riportato abbia significato fisico (il fattore 100 e' monotono e non cambia
    la scelta). Le soluzioni con fitness infinita sono rollout scartati.

    Restituisce (genoma, F, rapporto) oppure (None, None, None).
    """
    if F is None or len(F) == 0:
        return None, None, None
    F = np.atleast_2d(F)
    X = np.atleast_2d(X)
    ok = np.all(np.isfinite(F), axis=1)
    if not ok.any():
        return None, None, None
    F, X = F[ok], X[ok]
    ratios = [(-f[0] * 100.0) / max(f[1], 0.001) for f in F]
    i = int(np.argmax(ratios))
    return X[i], F[i], ratios[i]


# ---------------------------------------------------------------------------

def describe_params(params: dict) -> str:
    """Tutti i parametri decodificati dal genoma, uno per riga."""
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
    ap.add_argument("--settle-sec", type=float, default=2.0,
                    help="secondi di scansioni a robot fermo fra l'avvio di MOLA "
                         "e la partenza (default 2.0).")
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
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--chamfer-max", type=float, default=5.0,
                    help="non usato nella selezione (si usa il rapporto "
                         "danno/percettibilita'); tenuto per compatibilita'.")
    ap.add_argument("--dir-weight", type=float, default=1.0,
                    help="peso della componente direzionale nel danno, cioe' "
                         "quanto conta allontanare il robot dal bersaglio finale "
                         "oltre alla deviazione dal nominale (default 1.0; 0 = "
                         "solo deviazione).")
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
                         "'percorso/history.json:finestra'.")
    ap.add_argument("--repeat-n", type=int, default=3)
    ap.add_argument("--out", type=str, default="data/attack/run")
    args = ap.parse_args()

    goal = tuple(float(v) for v in args.goal.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    script = str(project_root / "src" / "nodes" / "launch_mola_attack.sh")

    isaac = IsaacClient()
    # Istanze di MOLA rimaste da run precedenti pubblicherebbero sullo stesso
    # topic e le pose si mescolerebbero.
    left = MolaRunner.kill_orphans()
    if left:
        print(f"\nATTENZIONE: {left} processi mola-cli non si lasciano "
              f"terminare.\nControlla con: ps aux | grep '[m]ola-cli'\n")
        return 1
    extra_env = {}
    if args.lidar_pose_x is not None:
        extra_env["LIDAR_POSE_X"] = args.lidar_pose_x
    if args.initial_sigma is not None:
        extra_env["MOLA_INITIAL_SIGMA"] = args.initial_sigma
    mola = MolaRunner(script, out / "logs", extra_env=extra_env,
                      traces=args.trace)
    node = AttackNode(FollowerModel())
    gen_obj = PerturbationGenerator()
    n_genes = gen_obj.get_genome_size()

    history = []
    print("=" * 70)
    print(f"  ATTACCO RECEDING-HORIZON")
    print(f"  bersaglio {goal}   orizzonte {args.horizon} m   "
          f"{args.pop}x{args.gen} = {args.pop*args.gen} candidati/finestra")
    print("=" * 70)

    try:
        start_pose = isaac.pose()

        # Il bersaglio e' relativo alla posa iniziale (x in avanti, y a
        # sinistra), ruotato nell'orientamento di partenza.
        gx_rel, gy_rel = goal
        c, s = math.cos(start_pose[2]), math.sin(start_pose[2])
        goal_abs = (start_pose[0] + gx_rel * c - gy_rel * s,
                    start_pose[1] + gx_rel * s + gy_rel * c)

        total = math.hypot(goal_abs[0] - start_pose[0],
                           goal_abs[1] - start_pose[1])
        # Ogni finestra consuma warmup + horizon metri: contarla sul solo
        # orizzonte crea finestre in eccesso, che trovano il bersaglio gia'
        # raggiunto.
        per_window_m = args.horizon + max(0.0, args.warmup)
        n_windows = max(1, int(total / per_window_m))

        frames_per_window = args.horizon / 0.8 * 10.0
        evals = 0 if args.passthrough else args.pop * args.gen
        per_window = (evals + 2) * (10.0 + args.warmup / 0.8)

        print(f"  partenza      : {np.round(start_pose,3)}")
        print(f"  bersaglio rel : {goal}  ->  assoluto {np.round(goal_abs,3)}")
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
        print(f"  genoma        : {n_genes} geni in [-1, 1]")
        if args.initial_sigma is not None:
            print(f"  sigma iniziale: {args.initial_sigma} "
                  f"(default pipeline 0.20)")
        if args.lidar_pose_x is not None:
            print(f"  LIDAR_POSE_X  : {args.lidar_pose_x} (sovrascritto)")
        if args.trace:
            print(f"  tracce        : {out/'traces'} e {out/'logs'}")
        print(f"  seed          : {args.seed}\n")

        goal = goal_abs

        # Test di ripetibilita': stesso genoma, N rollout sulla stessa finestra.
        if args.repeat_genome:
            src, _, widx = args.repeat_genome.partition(":")
            widx = int(widx) if widx else 0
            with open(src) as f:
                hist_src = json.load(f)
            g_rep = np.array(hist_src[widx]["genome"], dtype=np.float64)
            print(f"  TEST RIPETIBILITA': genoma dalla finestra {widx} di {src}")
            print(f"  {args.repeat_n} rollout identici sulla prima finestra\n")

            isaac.pause()
            saved = isaac.save()
            _, nom_end, _, st, nom_trav, _ = rollout(
                None, isaac, mola, node, goal, args.horizon, "rep_nominal",
                settle_sec=args.settle_sec, catchup_sec=args.catchup_sec)
            if nom_end is None or st not in ("ok", "arrived"):
                print(f"  nominale FALLITO ({st})")
                return 1
            print(f"  nominale: {np.round(nom_end,3)}  percorsi {nom_trav:.2f} m\n")

            devs, dirs, dmgs, perts = [], [], [], []
            for r in range(args.repeat_n):
                _, end, ch, st, trav, frames = rollout(
                    g_rep, isaac, mola, node, goal, args.horizon,
                    f"rep_{r}", settle_sec=args.settle_sec, catchup_sec=args.catchup_sec)
                if end is None or st not in ("ok", "arrived"):
                    print(f"    rollout {r}: SCARTATO [{st}]  {trav:.2f} m")
                    continue
                dev = math.hypot(end[0] - nom_end[0], end[1] - nom_end[1])
                d_nom = math.hypot(goal[0] - nom_end[0], goal[1] - nom_end[1])
                d_att = math.hypot(goal[0] - end[0], goal[1] - end[1])
                dr = d_att - d_nom
                dmg = dev + args.dir_weight * max(0.0, dr)
                devs.append(dev); dirs.append(dr); dmgs.append(dmg); perts.append(ch)
                print(f"    rollout {r}: dev {dev*100:6.1f}  dir {dr*100:+6.1f}  "
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
                           "directional_cm": [d * 100 for d in dirs],
                           "perturbation": perts}, f, indent=2)
            return 0

        for k in range(n_windows):
            print(f"── finestra {k+1}/{n_windows} " + "─" * 46)
            t_win = time.time()
            isaac.pause()
            saved = isaac.save()
            print(f"  stato salvato in {np.round(saved,3)}")

            # Nominale: stesso rollout senza perturbazione, riferimento della
            # deviazione. Ricalcolato per finestra perche' la geometria cambia.
            _, nom_end, _, st, nom_trav, nom_frames = rollout(
                None, isaac, mola, node, goal, args.horizon, f"w{k}_nominal",
                warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                trace_path=(out / "traces" / f"w{k}_nominal.csv")
                if args.trace else None)
            if nom_end is None or st not in ("ok", "arrived"):
                print(f"  nominale FALLITO ({st}) — finestra saltata")
                continue
            nom_yaw_deg = math.degrees(nom_end[2] - saved[2])
            print(f"  nominale: arriva in {np.round(nom_end,3)}  "
                  f"percorsi {nom_trav:.2f} m  {nom_frames} frame  "
                  f"rotazione {nom_yaw_deg:+.0f}°  "
                  f"rapporto stima/verita' {node.last_track_ratio:.2f}  [{st}]")
            if abs(nom_yaw_deg) > 30:
                print("  ATTENZIONE: il nominale ruota molto per andare dritto. "
                      "MOLA non e' a regime: alza --warmup o --horizon.")

            if args.passthrough:
                # Solo i nominali: verifica che il robot percorra il tratto
                # dritto senza alcuna perturbazione.
                history.append({"window": k,
                                "nominal_end": list(map(float, nom_end)),
                                "travelled": nom_trav, "frames": nom_frames,
                                "status": st})
                with open(out / "history.json", "w") as f:
                    json.dump(history, f, indent=2)
                print(f"  tempo finestra: {(time.time()-t_win)/60:.1f} min\n")
                continue

            print("  — le deviazioni qui sotto sono misurate rispetto a questo punto")

            # Ricerca con NSGA-III, configurazione di run_nsga3.py.
            from pymoo.optimize import minimize

            eval_log = []

            def eval_genome(g):
                """Fitness di un candidato: (danno negato, percettibilita').

                Il primo obiettivo e' negato perche' pymoo minimizza. Il danno
                somma la deviazione dal nominale e un termine direzionale, cioe'
                quanto l'attacco allontana dal bersaglio finale: un drift che
                gira attorno al bersaglio vale meno di uno che lo fa mancare.
                """
                idx = len(eval_log)
                _, end, ch, st, trav, frames = rollout(
                    g, isaac, mola, node, goal, args.horizon, f"w{k}_e{idx}",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec)

                # Un rollout che non completa il tratto (tracking perso, crash,
                # stopped_early) non e' confrontabile con uno che l'ha percorso:
                # escluso con fitness infinita.
                if end is None or st not in ("ok", "arrived"):
                    eval_log.append({"dev": None, "chamfer": ch, "status": st,
                                     "travelled": trav, "frames": frames})
                    print(f"    #{idx:02d}  SCARTATA  {trav:.2f} m  "
                          f"{frames:3d} frame  [{st}]")
                    return (np.inf, np.inf)

                dev = float(np.hypot(end[0] - nom_end[0], end[1] - nom_end[1]))

                # Termine direzionale: aumento della distanza dal bersaglio
                # rispetto al nominale; conta solo se positivo.
                d_nom_goal = math.hypot(goal[0] - nom_end[0], goal[1] - nom_end[1])
                d_att_goal = math.hypot(goal[0] - end[0], goal[1] - end[1])
                directional = d_att_goal - d_nom_goal
                damage = dev + args.dir_weight * max(0.0, directional)

                pert = ch if ch == ch else 1e3
                eval_log.append({"dev": dev, "directional": directional,
                                 "damage": damage, "chamfer": ch, "status": st,
                                 "track_ratio": node.last_track_ratio,
                                 "travelled": trav, "frames": frames,
                                 "end": list(map(float, end))})
                gen_no = idx // args.pop + 1
                # Rapporto in unita' coerenti: danno in cm su percettibilita'
                # in cm.
                print(f"    gen{gen_no} #{idx:02d}  dev {dev*100:6.1f}  "
                      f"dir {directional*100:+6.1f}  dmg {damage*100:6.1f} cm   "
                      f"pert {pert:7.2f} cm   {trav:.2f} m  {frames:3d}f  "
                      f"rapp {node.last_track_ratio:.2f}  [{st}]")
                if args.verbose:
                    print(describe_params(gen_obj.encode_perturbation(g)))
                return (-damage, pert)

            problem = build_problem(n_genes, eval_genome)
            algorithm = build_nsga3(args.pop)
            res = minimize(problem, algorithm, ("n_gen", args.gen),
                           seed=args.seed, verbose=False)

            best_g, best_F, ratio = pick_best(res.F, res.X)
            if best_g is None or not np.all(np.isfinite(best_F)):
                print("  nessuna soluzione valida (tutti i rollout scartati) "
                      "— finestra saltata")
                continue

            F = np.atleast_2d(res.F)
            print(f"\n  FRONTE DI PARETO: {len(F)} soluzioni non dominate "
                  f"su {len(eval_log)} valutate "
                  f"({sum(1 for e in eval_log if e['status'] in ('ok','arrived'))} valide)")
            for i, f in enumerate(F):
                mark = " <- scelto" if np.allclose(f, best_F) else ""
                print(f"    danno {-f[0]*100:7.2f} cm   pert {f[1]:8.3f} cm   "
                      f"rapporto {(-f[0]*100)/max(f[1],1e-3):6.2f}{mark}")
            print(f"\n  Genoma scelto (tutti i {n_genes} geni decodificati):")
            print(describe_params(gen_obj.encode_perturbation(best_g)))

            # Applicazione vera; con --trace si salva anche la traccia per-tick
            # di questo rollout, la traiettoria attaccata da confrontare con la
            # nominale.
            _, real_end, ch, st, trav, frames = rollout(
                best_g, isaac, mola, node, goal, args.horizon, f"w{k}_applied",
                warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                trace_path=(out / "traces" / f"w{k}_applied.csv")
                if args.trace else None)
            if real_end is not None:
                drift = math.hypot(real_end[0] - nom_end[0],
                                   real_end[1] - nom_end[1])
                print(f"\n  APPLICATO: il robot e' in {np.round(real_end,3)}")
                print(f"  percorsi {trav:.2f} m su {frames} frame perturbati")
                print(f"  deviazione reale dal nominale: {drift*100:.2f} cm   "
                      f"(in valutazione era {-best_F[0]*100:.2f} cm)   [{st}]")
            else:
                drift = None
                print(f"\n  APPLICAZIONE FALLITA [{st}]")
            print(f"  tempo finestra: {(time.time()-t_win)/60:.1f} min\n")

            # Costo computazionale della finestra: tempo e VRAM di picco via
            # nvidia-smi. NSGA-III gira su CPU, quindi la VRAM dipende dal
            # rendering di Isaac Sim.
            t_window_s = time.time() - t_win
            vram_mib = None
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                vram_mib = int(r.stdout.strip().splitlines()[0])
            except Exception:
                pass

            history.append({
                "window": k,
                "nominal_end": list(map(float, nom_end)),
                "applied_end": list(map(float, real_end)) if real_end else None,
                "goal": list(map(float, goal)),
                "genome": [float(v) for v in best_g],
                "damage_cm": float(-best_F[0]) * 100,
                "dev_real_cm": (drift * 100) if drift is not None else None,
                "applied_status": st,
                "applied_travelled": trav,
                "perturbation": float(best_F[1]),
                "ratio": float(ratio),
                "pareto_F": np.atleast_2d(res.F).tolist(),
                "evaluations": eval_log,
                "window_time_s": t_window_s,
                "n_evaluations": len(eval_log),
                "vram_mib": vram_mib,
                "pop": args.pop, "gen": args.gen,
                "horizon_m": args.horizon,
            })
            with open(out / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        node.close()
        mola.stop()
        # Isaac Sim resta in pausa dopo l'ultimo rollout: si fa ripartire
        # sempre, anche quando l'orchestratore esce per un errore.
        try:
            isaac.play()
        except Exception:
            pass
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nStorico: {out/'history.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
