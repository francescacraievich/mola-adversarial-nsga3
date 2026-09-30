#!/usr/bin/env python3
"""
Orchestratore dell'attacco adversarial receding-horizon.

Ad ogni finestra di H metri: salva lo stato di Isaac Sim, valuta N candidati con
un rollout fisico (ripristino dello stato, riavvio di MOLA, carico del genoma,
H metri di guida, misura della posa finale e della percettibilita'), ordina con
NSGA-III, sceglie dal fronte di Pareto e applica il vincente per un tratto vero.

La fitness e' misurata con il rollout e non stimata con un modello cinematico:
il ripristino dello stato e' fedele e non servono assunzioni sul moto futuro.
Il danno di un candidato e' la deviazione della sua traiettoria vera da quella
del rollout nominale della stessa finestra, confrontate a parita' di distanza
percorsa (--damage mean|max|end), piu' l'aumento della distanza dal bersaglio.
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
from src.analysis.path_deviation import path_deviation, true_path  # noqa: E402

CMD_FILE = "/tmp/isaac_cmd.json"
REPLY_FILE = "/tmp/isaac_reply.json"
POSE_FILE = "/tmp/isaac_pose.json"

# Distanza stimata da MOLA entro cui un waypoint e' raggiunto (m).
ARRIVAL_TOL = 0.30
# Frazione minima dell'orizzonte percorsa perche' un rollout sia valido.
MIN_TRAVEL_FRAC = 0.70
# Secondi senza posa dal ciclo di Isaac o senza nuvole perturbate (a
# simulazione in corso) oltre i quali Isaac e' considerato fermo.
STALL_SEC = 5.0


class IsaacStall(RuntimeError):
    """Ciclo di Isaac fermo o nuvole assenti: la run non puo' proseguire."""


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
        self.playing = True
        self.t_play = None         # ultimo play dato da questo client
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

        # Il server risponde entro un frame (step: n frame). Un'attesa oltre
        # STALL_SEC, o la posa del ciclo ferma da STALL_SEC durante l'attesa,
        # e' uno stallo anche se poi il ciclo riparte: un blocco transitorio
        # mentre l'orchestratore aspetta qui non sarebbe visto altrove.
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

    def save(self):    return self.call("save")["pose"]
    def restore(self): return self.call("restore")["pose"]
    def pose(self):    return self.call("pose")["pose"]
    def set_pose(self, x, y, yaw):
        return self.call("set_pose", x=float(x), y=float(y), yaw=float(yaw))["pose"]
    def play(self):
        self.call("play"); self.playing = True; self.t_play = time.time()
    def pause(self):   self.call("pause"); self.playing = False

    def pose_stream(self, max_age=0.5):
        """Posa vera dal file che il server riscrive a ogni frame di Isaac Sim.

        Una lettura e non una richiesta: non blocca il ciclo di controllo e
        permette di registrare la posa vera a ogni tick. Se il file manca o e'
        vecchio (server precedente a questa versione, o simulazione bloccata)
        si ricade sulla richiesta.
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

    def wait_ready(self, check=None):
        """Attende la rilocalizzazione iniziale. Richiede la simulazione in play.

        check, se dato, e' chiamato a ogni iterazione e puo' sollevare
        IsaacStall: senza nuvole MOLA non si dichiara mai pronta e lo stallo
        diventerebbe un semplice timeout.
        """
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if check is not None:
                check()
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
        Nome esatto del processo (-x), non la riga di comando (-f): con -f
        conterebbe, e kill_orphans ucciderebbe, qualunque shell, grep o editor
        che nomini mola-cli.
        """
        r = subprocess.run(["pgrep", "-x", "mola-cli"], capture_output=True,
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
        from std_msgs.msg import Bool, Float32MultiArray, String

        self._rclpy = rclpy
        self._Twist = Twist
        self._F32 = Float32MultiArray
        from rclpy.executors import SingleThreadedExecutor
        rclpy.init()
        self.node = Node("attack_orchestrator")
        # Executor proprio con il nodo aggiunto una volta: rclpy.spin_once(node)
        # aggiunge e rimuove il nodo dall'executor globale a ogni chiamata, e il
        # risveglio che ne segue consuma iterazioni senza eseguire callback.
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)

        self.est = None            # posa stimata da MOLA
        self.est_t = 0.0
        self.chamfer = []          # percettibilita' riportata dal nodo
        self.frames_out = 0        # nuvole perturbate, cumulativo
        self.trace_rows = None     # se non None, drive() vi accumula la traccia
        self.last_trace = []       # traccia dell'ultimo rollout, per la deviazione lungo il tratto
        self._last_true = None
        self._warned_no_stream = False
        self._n_cb = 0             # callback eseguiti, per svuotare la coda in drain()
        # Qualita' di registrazione riportata da MOLA: canale di rilevabilita'
        # indipendente dalla Chamfer, perche' un difensore non dispone della
        # nuvola originale ma vede se lo SLAM registra bene.
        self.pose_quality = float("nan")
        self.last_track_ratio = float("nan")   # rapporto stima/verita' dell'ultimo rollout

        self.node.create_subscription(Odometry, "/lidar_odometry/pose",
                                      self._est_cb, 100)
        # Posa vera dal topic di Isaac Sim: stessa sorgente fisica del server
        # nello Script Editor, ma su ROS. Il canale a file resta per pausa,
        # play, salvataggio e ripristino, che non hanno equivalente ROS.
        self.true = None
        self.true_t = 0.0
        self.true_source = "ros"
        # Coda corta: serve solo l'ultimo messaggio, e a ~50 messaggi al
        # secondo una coda lunga non svuotata diventa ritardo.
        self.node.create_subscription(Odometry, "/chassis/odom",
                                      self._true_cb, 5)
        self.node.create_subscription(String, "/attack/status",
                                      self._status_cb, 10)
        from std_msgs.msg import Float32
        self.node.create_subscription(Float32, "/lidar_odometry/pose_quality",
                                      self._quality_cb, 50)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self._enable_pub = self.node.create_publisher(Bool, "/attack/enabled", 10)
        self.genome_pub = self.node.create_publisher(
            Float32MultiArray, "/attack/genome", 10)
        # Arrivo delle nuvole all'ingresso di MOLA, per riconoscere uno stallo
        # di Isaac. raw=True: si registra solo l'istante, senza deserializzare
        # ~700 kB per messaggio.
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        self.cloud_t = 0.0
        self.node.create_subscription(PointCloud2, "/carter/lidar_perturbed",
                                      self._cloud_cb, qos_profile_sensor_data,
                                      raw=True)
        self.watch = None          # IsaacClient da sorvegliare, None = controllo spento
        self._t_check = 0.0
        self.follower = follower

    def _est_cb(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.est = (p.position.x, p.position.y, yaw)
        self.est_t = time.time()
        self._n_cb += 1

    def _true_cb(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.true = (p.position.x, p.position.y, yaw)
        self.true_t = time.time()
        self._n_cb += 1

    def true_pose(self, isaac, max_age=0.5):
        """Posa vera nel frame di lavoro.

        Con true_source "ros" e' l'ultimo messaggio di /chassis/odom, che Isaac
        Sim esprime nel frame odom (origine dove il robot era al Play), non
        nella scena: per questo tutta la logica usa questa funzione e mai le
        pose restituite dal server, che sono nel frame della scena. A
        simulazione in pausa il topic tace e l'ultimo messaggio e' la posa
        ferma; max_age vale solo a simulazione in corso. Con "file" e' la posa
        del server, nel frame della scena.
        """
        if self.true_source == "ros" and self.true is not None and (
                time.time() - self.true_t <= max_age or not isaac.playing):
            return self.true
        return isaac.pose_stream()

    def settled_pose(self, isaac):
        """Posa vera a robot fermo e simulazione in corso: aspetta un messaggio nuovo."""
        if self.true_source == "ros":
            t0 = self.true_t
            deadline = time.time() + 2.0
            while self.true_t <= t0 and time.time() < deadline:
                self.spin(0.05)
            if self.true is not None:
                return self.true
        return isaac.pose()

    def _status_cb(self, msg):
        self._n_cb += 1
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
        if v is None or d.get("enabled") is False:
            return
        try:
            self.chamfer.append(float(v))
        except (TypeError, ValueError):
            pass

    def _cloud_cb(self, _msg):
        self._n_cb += 1
        self.cloud_t = time.time()

    def check_stall(self):
        """Solleva IsaacStall se il ciclo di Isaac o le nuvole sono fermi da STALL_SEC.

        L'eta' della posa nel file del server vale sempre (il ciclo la scrive
        anche in pausa); le nuvole si contano solo a simulazione in corso e
        dopo un play dato da questo orchestratore. Al piu' un controllo ogni
        0.5 s.
        """
        isaac = self.watch
        now = time.time()
        if isaac is None or now - self._t_check < 0.5:
            return
        self._t_check = now
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

    def _quality_cb(self, msg):
        self._n_cb += 1
        self.pose_quality = float(msg.data)

    def count_mola_publishers(self) -> int:
        """Publisher su /lidar_odometry/pose nel grafo ROS (diagnostica, atteso 1)."""
        return self.node.count_publishers("/lidar_odometry/pose")

    def drain(self, budget: float = 0.02):
        """Esegue tutti i callback in coda, entro budget secondi.

        spin_once ne esegue uno solo: con /chassis/odom a ~50 messaggi al
        secondo, uno per tick di controllo (20 Hz) lascia crescere la coda e
        posa vera e stima arrivano con 1-2 s di ritardo. Un'iterazione vuota
        non basta a dire che la coda e' vuota (risvegli senza callback): ci
        si ferma dopo tre di fila.
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
        m = self._F32()
        m.data = [float(v) for v in genome]
        self.genome_pub.publish(m)
        self.spin(0.2)

    def set_attack_enabled(self, on: bool):
        """Accende o spegne la perturbazione senza riavviare il nodo."""
        from std_msgs.msg import Bool
        m = Bool()
        m.data = bool(on)
        self._enable_pub.publish(m)
        self.spin(0.2)

    def stop_robot(self):
        self.cmd_pub.publish(self._Twist())

    def halt(self, sec: float = 0.5):
        """Comando nullo ripetuto e consegnato.

        Il differential_drive di Isaac Sim applica l'ultimo /cmd_vel ricevuto
        senza scadenza: un solo messaggio pubblicato subito prima di chiudere
        il nodo puo' non partire, e il robot continua a muoversi anche dopo
        set_pose e nella run successiva.
        """
        t_end = time.time() + sec
        while time.time() < t_end:
            self.stop_robot()
            self.spin(0.05)

    def drive(self, local_wps, distance, isaac: IsaacClient,
              mola: MolaRunner, timeout=45.0, rate=20.0,
              check_every=None, over_run=1.30, arrival_tol=ARRIVAL_TOL):
        """Guida il robot fino a `distance` metri veri dal punto di partenza.

        L'arresto usa la posa vera di Isaac Sim e non la stima di MOLA: fermarsi
        quando MOLA crede di aver percorso il tratto rende la fitness illimitata,
        perche' un attacco che fa sottostimare la traslazione tiene il robot in
        moto e la deviazione cresce da sola.

        La posa vera viene letta dal file che il server riscrive a ogni frame,
        quindi a ogni tick di controllo; se il server non lo pubblica si ricade
        sulla richiesta bloccante ogni check_every secondi (0.25 s: a 0.8 m/s
        sono 20 cm fra due letture).

        local_wps e' la lista dei waypoint rimanenti nel frame di MOLA: quando
        il primo e' entro arrival_tol si passa al successivo senza fermarsi;
        `arrived` solo sull'ultimo.

        Esiti: ok, arrived (ultimo waypoint entro arrival_tol), overrun (tratto superato
        del fattore over_run senza arresto), timeout, mola_crash, pose_stale,
        no_pose. Restituisce (esito, metri percorsi).
        """
        self.est = None
        t_wait = time.time()
        while self.est is None and time.time() - t_wait < 15.0:
            self._exec.spin_once(timeout_sec=0.05)
            self.check_stall()
            if mola.crashed():
                return "mola_crash", 0.0
        if self.est is None:
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

        start_pose = self.true_pose(isaac)
        true_start = np.array(start_pose[:2])
        self._last_true = start_pose
        # La stima di MOLA parte da (0,0,0) nella posa vera iniziale: per il
        # confronto stima/verita' nella traccia va riportata nel frame del mondo.
        c0, s0 = math.cos(start_pose[2]), math.sin(start_pose[2])
        dt = 1.0 / rate
        t0 = time.time()
        t_last_check = -1.0
        travelled = 0.0
        wps = [tuple(w) for w in local_wps]
        local_wp = wps[0]

        while time.time() - t0 < timeout:
            self.drain()
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
                if len(wps) > 1:
                    wps.pop(0)
                    local_wp = wps[0]
                else:
                    self.stop_robot()
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
                est_wx = start_pose[0] + c0 * x - s0 * y
                est_wy = start_pose[1] + s0 * x + c0 * y
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
                    # Divario stima/verita' (stima nel frame del mondo) e
                    # qualita' di registrazione: legano perturbazione, errore
                    # SLAM e deviazione.
                    "est_true_gap": round(math.hypot(est_wx - tp[0], est_wy - tp[1]), 4)
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
        try:
            self.halt()
        except Exception:
            pass
        try:
            self._exec.shutdown()
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
            settle_sec=2.0, min_travel_frac=MIN_TRAVEL_FRAC, catchup_sec=0.8,
            attack_from="motion", untracked="discard", nominal_ok=True):
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
    isaac.restore()
    mola.stop()

    # Avvio senza attesa: la rilocalizzazione iniziale richiede una nuvola, che
    # a simulazione ferma non arriva. Prima lo spawn, poi il play, poi l'attesa.
    t_spawn = time.time()
    try:
        mola.spawn(tag)
    except Exception as e:
        return None, None, float("inf"), f"mola_spawn_fail: {e}", 0.0, 0

    # Il genoma viene caricato subito, l'attacco si accende solo all'inizio
    # del tratto misurato: durante l'assestamento (e il riscaldamento) MOLA
    # vede nuvole pulite, il drift temporale non accumula a robot fermo e la
    # Chamfer media copre solo il moto. Con attack_from="spawn" si accende
    # da subito, come nelle run fino ad attack_v5.
    if genome is not None:
        node.set_genome(genome)
    node.set_attack_enabled(attack_from == "spawn" and warmup <= 0.0
                            and genome is not None)
    node.chamfer = []

    isaac.play()
    try:
        # drain e non il solo controllo: l'attesa non fa girare l'executor e
        # senza callback l'arrivo delle nuvole non verrebbe registrato.
        mola.wait_ready(check=node.drain)
    except IsaacStall:
        raise
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

    # Posa di partenza nel frame di lavoro, letta a simulazione in corso e
    # robot fermo: dopo il ripristino il topic pubblica solo al primo tick.
    true_start = node.settled_pose(isaac)

    goals = goal if isinstance(goal, list) else [goal]
    local_wps = [to_local(g, true_start) for g in goals]

    if warmup > 0.0:
        st_w, _ = node.drive(local_wps, warmup, isaac, mola, timeout=30.0)
        if st_w not in ("ok", "arrived"):
            isaac.pause()
            mola.stop()
            return None, None, float("inf"), f"warmup_{st_w}", 0.0, 0
        # Da qui in poi si misura: nuova origine del tratto e attacco attivo.
        true_start = node.settled_pose(isaac)
        local_wps = [to_local(g, true_start) for g in goals]
        node.set_attack_enabled(genome is not None)
        node.chamfer = []

    if attack_from == "motion" and warmup <= 0.0 and genome is not None:
        node.set_attack_enabled(True)
        node.chamfer = []

    frames_before = node.frames_out
    # La traccia serve sempre: la deviazione lungo il tratto si calcola su di
    # essa. Su file solo se richiesto.
    node.trace_rows = []
    est_before = node.est
    status, travelled = node.drive(local_wps, horizon, isaac, mola)

    # La stima di MOLA segue il moto vero con 0.3-0.5 s di latenza (LiDAR a
    # 10 Hz, ICP, stimatore): letta all'istante dell'arresto sottostima il
    # tratto anche a tracking perfetto. Il robot ha gia' ricevuto lo stop; si
    # lascia alla stima il tempo di raggiungerlo.
    if status in ("ok", "arrived") and catchup_sec > 0:
        node.spin(catchup_sec)
    est_after = node.est
    isaac.pause()
    node.spin(0.05)
    true_end_now = node.true_pose(isaac)
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
        lost = ratio < track_min or (track_max is not None and ratio > track_max)
        # Tracking perso: scartato sempre (discard), solo se anche il nominale
        # della finestra era anomalo (nominal), oppure mai (damage): negli
        # ultimi due casi la deviazione resta e conta come danno dell'attacco.
        if lost and (untracked == "discard"
                     or (untracked == "nominal" and not nominal_ok)):
            status = f"untracked({ratio:.2f})"
        else:
            track_ratio = ratio
            if lost:
                print(f"    [nota] tracking perso (rapporto {ratio:.2f}) contato come danno")
    # Esposto anche per i rollout validi, da riportare accanto alla deviazione.
    node.last_track_ratio = track_ratio

    # Tratto troppo breve per misurare una deviazione di rotta: l'attacco ha
    # paralizzato il robot (tipicamente `arrived` a zero metri). Senza questa
    # marcatura NSGA-III sceglie il genoma che ferma il robot.
    if status in ("ok", "arrived") and travelled < min_travel_frac * horizon:
        status = f"stopped_early({travelled:.2f}m)"

    true_end = true_end_now
    # L'ultima riga della traccia e' la posa finale letta a simulazione ferma,
    # dopo il coasting: cosi' la deviazione al termine del tratto coincide con
    # quella calcolata sulle pose finali.
    if node.trace_rows:
        last = dict(node.trace_rows[-1])
        last.update({"t": round(last["t"] + catchup_sec, 3),
                     "true_x": round(true_end[0], 4), "true_y": round(true_end[1], 4),
                     "true_yaw_deg": round(math.degrees(true_end[2]), 2),
                     "cmd_v": 0.0, "cmd_w": 0.0})
        node.trace_rows.append(last)
    node.last_trace = node.trace_rows or []
    if trace_path is not None and node.trace_rows:
        import csv
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(node.trace_rows[0].keys()))
            wr.writeheader()
            wr.writerows(node.trace_rows)
    node.trace_rows = None
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


# Gruppi di genoma per informazione richiesta all'attaccante. Indici dei geni
# attivi; gli altri restano a -1 (operatore spento).
GENOME_GROUPS = {
    1: [0, 1, 2, 3, 5, 11],                       # rumore per punto, dropout
    2: [0, 1, 2, 3, 5, 11, 4, 6, 7, 8, 9, 10, 13, 15, 16],   # + geometria dello scan
    3: list(range(17)),                            # + distorsione e drift (globali)
}


def expand_genome(g_active, active, n_genes=17):
    """Vettore completo da quello ristretto al gruppo: geni inattivi a -1."""
    full = -np.ones(n_genes)
    full[active] = np.asarray(g_active, dtype=np.float64)
    return full


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


def window_damage(end, nom_end, goal, nom_path, cand_path, mode, dir_weight):
    """Danno di un rollout rispetto al nominale della stessa finestra.

    La deviazione lungo il tratto confronta le due traiettorie vere a parita'
    di distanza percorsa (src/analysis/path_deviation.py): mode sceglie se il
    danno usa la media lungo il tratto, il massimo o la sola posa finale. Il
    termine direzionale e' l'aumento della distanza dal bersaglio rispetto al
    nominale, conta solo se positivo: un drift che gira attorno al bersaglio
    vale meno di uno che lo fa mancare.

    Restituisce un dict con dev_end, dev_mean, dev_max, directional e damage,
    tutti in metri.
    """
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


def pareto_front(F, X):
    """Soluzioni non dominate (minimizzazione su entrambe le colonne)."""
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
    ap.add_argument("--reeval", action="store_true",
                    help="rivaluta il genoma scelto con un rollout in piu' prima di applicarlo")
    ap.add_argument("--search", choices=["nsga3", "random", "gaussian"], default="nsga3",
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
    ap.add_argument("--start-pose", type=str, default=None,
                    help='posa iniziale assoluta "x,y,yaw_deg": il robot viene riportato '
                         'li\' prima di partire (server isaac_rollout_server.py aggiornato)')
    ap.add_argument("--genome-group", type=int, choices=[1, 2, 3], default=3,
                    help="gruppo di geni attivi: 1 = rumore per punto e dropout (6), "
                         "2 = + operatori sulla geometria dello scan (15), "
                         "3 = tutti (17, default)")
    ap.add_argument("--untracked", choices=["discard", "nominal", "damage"], default="discard",
                    help="rollout con tracking perso (rapporto stima/verita' < 0.6): "
                         "discard = scartato (default); nominal = scartato solo se anche "
                         "il nominale della finestra era anomalo; damage = mai scartato, "
                         "la deviazione conta come danno")
    ap.add_argument("--attack-from", choices=["motion", "spawn"], default="motion",
                    help="quando accendere la perturbazione: all'inizio del tratto "
                         "misurato (default) o gia' dall'avvio di MOLA (assestamento "
                         "incluso, come nelle run fino ad attack_v5)")
    ap.add_argument("--true-pose", choices=["ros", "file"], default="ros",
                    help="sorgente della posa vera: /chassis/odom (default) o il "
                         "server via file; la fisica e' la stessa")
    ap.add_argument("--damage", choices=["mean", "max", "end"], default="mean",
                    help="deviazione usata nel danno: media lungo il tratto a "
                         "parita' di distanza percorsa (default), massimo lungo "
                         "il tratto, o sola posa finale (come nelle run fino a "
                         "attack_v5)")
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
    goals_rel = ([tuple(float(v) for v in g.split(",")) for g in args.goals.split(";")]
                 if args.goals else [goal])
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
    node.true_source = args.true_pose
    if args.true_pose == "ros":
        # La discovery DDS di un nodo appena creato puo' superare il secondo:
        # un'attesa fissa breve fa ricadere a caso sulla posa via file.
        t_wait = time.time()
        while node.true is None and time.time() - t_wait < 5.0:
            node.spin(0.1)
        if node.true is None:
            print("  [nota] nessun messaggio su /chassis/odom: posa vera dal server via file")
            node.true_source = "file"
    gen_obj = PerturbationGenerator()
    n_genes = gen_obj.get_genome_size()
    active_genes = GENOME_GROUPS[args.genome_group]
    n_search = len(active_genes)

    history = []
    k = None                       # finestra corrente, per marcare uno stallo
    stalled = None
    print("=" * 70)
    print(f"  ATTACCO RECEDING-HORIZON")
    print(f"  bersaglio {goals_rel[-1]}   orizzonte {args.horizon} m   "
          f"{args.pop}x{args.gen} = {args.pop*args.gen} candidati/finestra")
    print("=" * 70)

    try:
        node.watch = isaac
        # Un comando rimasto da una run interrotta muoverebbe il robot durante
        # il salvataggio dello stato e dopo set_pose.
        node.halt()
        if args.start_pose:
            sx, sy, syaw = (float(v) for v in args.start_pose.split(","))
            isaac.pause()
            isaac.set_pose(sx, sy, math.radians(syaw))
            isaac.play()
            node.spin(1.0)
        start_pose = node.settled_pose(isaac)
        if node.true_source == "ros":
            print(f"  frame di lavoro: odom di Isaac (/chassis/odom), non la scena; "
                  f"posa nella scena {np.round(isaac.pose(), 3)}")

        # I bersagli sono relativi alla posa iniziale (x in avanti, y a
        # sinistra), ruotati nell'orientamento di partenza.
        c, s = math.cos(start_pose[2]), math.sin(start_pose[2])
        goals_abs = [(start_pose[0] + gx * c - gy * s, start_pose[1] + gx * s + gy * c)
                     for gx, gy in goals_rel]
        goal_abs = goals_abs[-1]

        # Primo bersaglio piu' vicino di orizzonte + tolleranza d'arrivo: il
        # rollout lo raggiunge prima di aver percorso l'orizzonte. Con un solo
        # waypoint si ferma per `arrived` a circa d - ARRIVAL_TOL metri, vicino
        # o sotto la soglia di stopped_early.
        d_first = math.hypot(goals_abs[0][0] - start_pose[0], goals_abs[0][1] - start_pose[1])
        if d_first < args.horizon + ARRIVAL_TOL:
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
        # Ogni finestra consuma warmup + horizon metri: contarla sul solo
        # orizzonte crea finestre in eccesso, che trovano il bersaglio gia'
        # raggiunto.
        per_window_m = args.horizon + max(0.0, args.warmup)
        n_windows = max(1, int(total / per_window_m))

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

            saved = node.true_pose(isaac)
            isaac.pause()
            isaac.save()
            _, nom_end, _, st, nom_trav, _ = rollout(
                None, isaac, mola, node, goal, args.horizon, "rep_nominal",
                settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from)
            if nom_end is None or st not in ("ok", "arrived"):
                print(f"  nominale FALLITO ({st})")
                return 1
            print(f"  nominale: {np.round(nom_end,3)}  percorsi {nom_trav:.2f} m\n")
            nom_path = true_path(node.last_trace)

            devs, dirs, dmgs, perts = [], [], [], []
            for r in range(args.repeat_n):
                _, end, ch, st, trav, frames = rollout(
                    g_rep, isaac, mola, node, goal, args.horizon,
                    f"rep_{r}", settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from)
                if end is None or st not in ("ok", "arrived"):
                    print(f"    rollout {r}: SCARTATO [{st}]  {trav:.2f} m")
                    continue
                dm = window_damage(end, nom_end, goal, nom_path,
                                   true_path(node.last_trace),
                                   args.damage, args.dir_weight)
                dev, dr, dmg = dm["dev_end"], dm["directional"], dm["damage"]
                devs.append(dev); dirs.append(dr); dmgs.append(dmg); perts.append(ch)
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
                           "directional_cm": [d * 100 for d in dirs],
                           "perturbation": perts}, f, indent=2)
            return 0

        wp_idx = 0
        k = -1
        while True:
            k += 1
            saved = node.true_pose(isaac)
            isaac.pause()
            isaac.save()
            # Avanzamento dei waypoint sulla posa vera: il criterio di arresto
            # del task non dipende dalla stima attaccata.
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
            wps = goals_abs[wp_idx:]
            goal = wps[0]
            print(f"── finestra {k+1} (stima {n_windows}) " + "─" * 40)
            print(f"  waypoint corrente {wp_idx+1}/{len(goals_abs)} {np.round(goal,3)}  "
                  f"distanza al bersaglio finale {d_last:.2f} m")
            t_win = time.time()
            print(f"  stato salvato in {np.round(saved,3)}")

            # Nominale: stesso rollout senza perturbazione, riferimento della
            # deviazione. Ricalcolato per finestra perche' la geometria cambia.
            nom_start, nom_end, _, st, nom_trav, nom_frames = rollout(
                None, isaac, mola, node, wps, args.horizon, f"w{k}_nominal",
                warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from,
                trace_path=(out / "traces" / f"w{k}_nominal.csv")
                if args.trace else None)
            if nom_end is None or st not in ("ok", "arrived"):
                print(f"  nominale FALLITO ({st}) — finestra saltata")
                continue
            nom_path = true_path(node.last_trace)
            nominal_ok = node.last_track_ratio == node.last_track_ratio   # non NaN
            nom_yaw_deg = math.degrees(angle_diff(nom_end[2], saved[2]))
            # Rotazione attesa: quella verso il bersaglio dallo stato salvato.
            # Dopo una finestra attaccata il robot puo' trovarsi storto e il
            # nominale deve girare per tornare in rotta: non e' un'anomalia.
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

            def eval_genome(g_active):
                """Fitness di un candidato: (danno negato, percettibilita').

                Il primo obiettivo e' negato perche' pymoo minimizza. Il danno
                somma la deviazione dal nominale e un termine direzionale, cioe'
                quanto l'attacco allontana dal bersaglio finale: un drift che
                gira attorno al bersaglio vale meno di uno che lo fa mancare.
                """
                idx = len(eval_log)
                g = expand_genome(g_active, active_genes, n_genes)
                c_start, end, ch, st, trav, frames = rollout(
                    g, isaac, mola, node, wps, args.horizon, f"w{k}_e{idx}",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from,
                    untracked=args.untracked, nominal_ok=nominal_ok,
                    trace_path=(out / "traces" / f"w{k}_e{idx:02d}.csv")
                    if args.trace else None)

                # Un rollout che non completa il tratto (tracking perso, crash,
                # stopped_early) non e' confrontabile con uno che l'ha percorso:
                # escluso con fitness infinita.
                if end is None or st not in ("ok", "arrived"):
                    eval_log.append({"genome": [float(v) for v in g],
                                     "dev": None, "chamfer": ch, "status": st,
                                     "travelled": trav, "frames": frames})
                    print(f"    #{idx:02d}  SCARTATA  {trav:.2f} m  "
                          f"{frames:3d} frame  [{st}]")
                    return (np.inf, np.inf)

                dm = window_damage(end, nom_end, goal, nom_path,
                                   true_path(node.last_trace),
                                   args.damage, args.dir_weight)
                damage = dm["damage"]

                pert = ch if ch == ch else 1e3
                eval_log.append({"genome": [float(v) for v in g],
                                 "dev": dm["dev_end"], "dev_mean": dm["dev_mean"],
                                 "dev_max": dm["dev_max"], "damage_mode": args.damage,
                                 "directional": dm["directional"],
                                 "damage": damage, "chamfer": ch, "status": st,
                                 "track_ratio": node.last_track_ratio,
                                 "travelled": trav, "frames": frames,
                                 "path_points": dm["path_points"],
                                 "end": list(map(float, end))})
                gen_no = idx // args.pop + 1
                # Rapporto in unita' coerenti: danno in cm su percettibilita'
                # in cm.
                rst = math.hypot(c_start[0] - saved[0], c_start[1] - saved[1]) if c_start else float("nan")
                print(f"    gen{gen_no} #{idx:02d}  rst {rst*100:4.1f}  end {dm['dev_end']*100:5.1f} "
                      f"mean {dm['dev_mean']*100:5.1f} max {dm['dev_max']*100:5.1f}  "
                      f"dir {dm['directional']*100:+6.1f}  dmg {damage*100:6.1f} cm   "
                      f"pert {pert:7.2f} cm   {trav:.2f} m  {frames:3d}f  "
                      f"rapp {node.last_track_ratio:.2f}  [{st}]")
                if args.verbose:
                    print(describe_params(gen_obj.encode_perturbation(g)))
                return (-damage, pert)

            if args.search == "nsga3":
                problem = build_problem(n_search, eval_genome)
                algorithm = build_nsga3(args.pop)
                res = minimize(problem, algorithm, ("n_gen", args.gen),
                               seed=args.seed, verbose=False)
                res_F, res_X = res.F, res.X
            elif args.search == "random":
                # Ricerca casuale a pari budget: pop x gen genomi estratti a
                # caso nel gruppo, scelta con lo stesso criterio. Il fronte e'
                # calcolato sulle valutazioni valide.
                rng = np.random.default_rng(args.seed + k)
                X_all = rng.uniform(-1.0, 1.0, size=(args.pop * args.gen, n_search))
                F_all = np.array([eval_genome(x) for x in X_all])
                ok = np.all(np.isfinite(F_all), axis=1)
                res_F, res_X = pareto_front(F_all[ok], X_all[ok])
            else:
                # Gaussiano: nessuna ricerca, un solo rollout con il rumore
                # isotropo del nodo (--gaussian-sigma sul perturbation_node).
                # Il genoma pubblicato e' ignorato dal nodo in quella modalita'.
                F_all = np.array([eval_genome(np.zeros(n_search))])
                X_all = np.zeros((1, n_search))
                ok = np.all(np.isfinite(F_all), axis=1)
                res_F, res_X = F_all[ok], X_all[ok]

            best_g, best_F, ratio = pick_best(res_F, res_X)
            if best_g is not None:
                best_g = expand_genome(best_g, active_genes, n_genes)
            if best_g is None or not np.all(np.isfinite(best_F)):
                print("  nessuna soluzione valida (tutti i rollout scartati) "
                      "— finestra saltata")
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
            print(describe_params(gen_obj.encode_perturbation(best_g)))

            # Rivalutazione del vincente: un rollout in piu' con lo stesso
            # genoma, per misurare quanto il valore di ricerca e' gonfiato dal
            # rumore (selezione del massimo fra valutazioni rumorose).
            reeval = None
            if args.reeval:
                _, re_end, re_ch, re_st, re_trav, re_frames = rollout(
                    best_g, isaac, mola, node, wps, args.horizon, f"w{k}_reeval",
                    warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec,
                    attack_from=args.attack_from, untracked=args.untracked, nominal_ok=nominal_ok)
                if re_end is not None and re_st in ("ok", "arrived"):
                    dm_re = window_damage(re_end, nom_end, goal, nom_path,
                                          true_path(node.last_trace), args.damage, args.dir_weight)
                    reeval = {"damage_cm": dm_re["damage"] * 100, "dev_mean_cm": dm_re["dev_mean"] * 100,
                              "chamfer": re_ch, "status": re_st, "frames": re_frames}
                    print(f"  rivalutazione: danno {dm_re['damage']*100:.1f} cm "
                          f"(in ricerca {-best_F[0]*100:.1f})  pert {re_ch:.2f}  [{re_st}]")
                else:
                    reeval = {"damage_cm": None, "status": re_st}
                    print(f"  rivalutazione: SCARTATA [{re_st}]")

            # Applicazione vera; con --trace si salva anche la traccia per-tick
            # di questo rollout, la traiettoria attaccata da confrontare con la
            # nominale.
            _, real_end, ch, st, trav, frames = rollout(
                best_g, isaac, mola, node, wps, args.horizon, f"w{k}_applied",
                warmup=args.warmup, settle_sec=args.settle_sec, catchup_sec=args.catchup_sec, attack_from=args.attack_from,
                untracked=args.untracked, nominal_ok=nominal_ok,
                trace_path=(out / "traces" / f"w{k}_applied.csv")
                if args.trace else None)
            if real_end is not None:
                dm_real = window_damage(real_end, nom_end, goal, nom_path,
                                        true_path(node.last_trace),
                                        args.damage, args.dir_weight)
                drift = dm_real["dev_end"]
                print(f"\n  APPLICATO: il robot e' in {np.round(real_end,3)}")
                print(f"  percorsi {trav:.2f} m su {frames} frame perturbati")
                print(f"  deviazione reale dal nominale: fine {drift*100:.1f}  "
                      f"media {dm_real['dev_mean']*100:.1f}  "
                      f"max {dm_real['dev_max']*100:.1f} cm   "
                      f"danno {dm_real['damage']*100:.1f} cm "
                      f"(in valutazione era {-best_F[0]*100:.1f} cm)   [{st}]")
            else:
                drift = None
                dm_real = None
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

            # Metriche di finestra oltre al danno: errore di prua, frame per
            # metro (tempo perso), candidati che paralizzano o perdono il
            # tracking.
            statuses = [e["status"] for e in eval_log]
            heading_err_deg = (math.degrees(angle_diff(real_end[2], nom_end[2]))
                               if real_end is not None else None)
            history.append({
                "window": k,
                "heading_error_deg": heading_err_deg,
                "frames_per_m_nominal": nom_frames / max(nom_trav, 1e-6),
                "frames_per_m_applied": (frames / max(trav, 1e-6)) if real_end is not None else None,
                "n_stopped_early": sum(1 for st_ in statuses if str(st_).startswith("stopped_early")),
                "n_untracked": sum(1 for st_ in statuses if str(st_).startswith("untracked")),
                "n_failed": sum(1 for st_ in statuses if st_ not in ("ok", "arrived")
                                and not str(st_).startswith(("stopped_early", "untracked"))),
                "reeval": reeval,
                "nominal_end": list(map(float, nom_end)),
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
                "applied_status": st,
                "applied_travelled": trav,
                "perturbation": float(best_F[1]),
                "ratio": float(ratio),
                "pareto_F": np.atleast_2d(res_F).tolist(),
                "search": args.search,
                "evaluations": eval_log,
                "window_time_s": t_window_s,
                "n_evaluations": len(eval_log),
                "vram_mib": vram_mib,
                "pop": args.pop, "gen": args.gen,
                "horizon_m": args.horizon,
                "genome_group": args.genome_group,
                "active_genes": active_genes,
                "attack_from": args.attack_from,
                "untracked_policy": args.untracked,
            })
            with open(out / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    except KeyboardInterrupt:
        print("\nInterrotto.")
    except IsaacStall as e:
        stalled = str(e)
        node.watch = None
        node.stop_robot()
        history.append({"window": k, "status": "isaac_stall", "error": stalled})
        print(f"\nSTALLO DI ISAAC: {stalled}")
        print(f"  finestra {k + 1 if k is not None else '-'} marcata fallita (isaac_stall); "
              f"la run si ferma qui. History parziale salvata.")
    finally:
        node.watch = None          # la chiusura non deve ricontrollare lo stallo
        node.close()
        mola.stop()
        # Isaac Sim resta in pausa dopo l'ultimo rollout: si fa ripartire
        # sempre, anche quando l'orchestratore esce per un errore, tranne
        # dopo uno stallo (il comando aspetterebbe il timeout per nulla).
        if not stalled:
            try:
                isaac.play()
            except Exception:
                pass
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nStorico: {out/'history.json'}")

    # Codice 3: fermata per stallo di Isaac (run_campaign.sh rilancia la
    # ripetizione una volta).
    return 3 if stalled else 0


if __name__ == "__main__":
    sys.exit(main())
