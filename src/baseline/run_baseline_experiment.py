#!/usr/bin/env python3
"""
Esegue una run di baseline di MOLA LiDAR odometry in Isaac Sim (scena carter_warehouse.usd,
robot Nova Carter, LiDAR Hesai XT-32 su /front_3d_lidar/lidar_points).

La baseline e' in anello aperto: il robot segue un percorso predefinito comandato su /cmd_vel e
chiuso sull'odometria /chassis/odom; MOLA e' solo un osservatore, la sua stima non influenza il
moto. Percorsi: stationary (robot fermo), loop (rettangolo chiuso), serpentine.

Passi di una run: reset di Isaac Sim via /isaac/reset_world (saltabile con --skip-reset); avvio di
add_intensity_node e di mola-cli con i parametri MOLA passati come variabili d'ambiente (con
--deterministic quelli che dipendono dallo stato stimato vengono pinnati); esecuzione del percorso
con un cmd_vel per ogni messaggio odom; stop dei processi; scrittura delle traiettorie TUM di
MOLA e del riferimento (anche allineato ai timestamp MOLA); statistiche finali e copertura scan.

Uso:
    python3 src/baseline/run_baseline_experiment.py --run-id 1 --path loop --loop-size 4,2.5 \\
        --loop-dir ccw --deterministic --local-map-size 25 \\
        --lidar-topic /front_3d_lidar/lidar_points --sub-scans-per-cycle 1 \\
        --lidar-pose="-0.2317,0,0.526" --skip-reset

Output sotto --output-dir (default data/trajectories/): mola/run_N.tum, gt/run_N.tum,
gt/run_N_aligned.tum, logs/mola_run_N.log, logs/add_intensity_run_N.log,
logs/scan_count_run_N.txt e, con --deterministic, logs/traces_run_N.csv.
Analisi su piu' run: src/baseline/loop_baseline_stats.py --runs 2 3 4 5 6 e
src/plots/plot_loop_baseline.py --runs 2 3 4 5 6.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Empty


# ---------------------------------------------------------------------------
# Segmenti di percorso basati su odometria, indipendenti dall'RTF.
#
# Ogni segmento e' una tupla (type, linear_x, angular_z, target):
#   "straight" -> target = distanza in metri (fine quando la distanza odom >= target)
#   "turn"     -> target = angolo in radianti con segno (fine quando il delta yaw e' raggiunto)
#   "stop"     -> ferma il robot e termina
#
# La fine di ogni segmento e' decisa dall'odometria di /chassis/odom, non dal wall-clock,
# cosi' il percorso e' riproducibile a qualunque RTF di Isaac Sim.
#
# Velocita' di svolta 0.30 rad/s: a velocita' maggiori l'inerzia rotazionale residua dopo lo
# stop del comando dipende dall'RTF (alcuni gradi di overshoot) e la lunghezza del percorso
# cambia fra run; a 0.30 rad/s l'overshoot resta sotto il grado.
#
# angular_z < 0 = orario (-Y in odom).
# ---------------------------------------------------------------------------
PATH_SEGMENTS = [
    ("straight", 0.80,  0.00,  9.0),                     #  1: +9 m in direzione corrente
    ("turn",     0.00, -0.30, -math.radians(80)),        #  2: -80 gradi (orario)
    ("straight", 0.80,  0.00,  5.0),                    #  3: +5 m
    ("turn",     0.00, -0.30, -math.radians(80)),        #  4: -80 gradi
    ("straight", 0.80,  0.00,  9.0),                    #  5: +9 m
    ("turn",     0.00, +0.30, +math.radians(80)),        #  6: +80 gradi (antiorario)
    ("straight", 0.80,  0.00,  3.0),                    #  7: +3 m traversata
    ("turn",     0.00, +0.30, +math.radians(80)),        #  8: +80 gradi
    ("straight", 0.80,  0.00,  9.0),                    #  9: +9 m (ritorno)
    ("stop",     0.00,  0.00,  0.0),                    # 10: ferma
]


# ---------------------------------------------------------------------------
# Percorso chiuso per la baseline (--path loop).
#
# Rettangolo con quattro svolte a 90 gradi: il robot torna al punto di partenza con lo stesso
# heading. mola-sm-loopclosure non e' attivo in questa pipeline, quindi MOLA non aggancia il
# loop e la distanza fra posa finale stimata e posa iniziale e' drift accumulato puro: una
# metrica scalare, calcolabile senza allineare le traiettorie.
#
# L'inerzia residua dopo le svolte e' gestita dalla rampa di decelerazione in _execute_motion
# e dall'attesa in _wait_for_stop.
# ---------------------------------------------------------------------------
def make_loop(long_side: float = 4.0, short_side: float = 3.0,
              clockwise: bool = True, speed: float = 0.80,
              turn_rate: float = 0.30):
    """Costruisce un percorso rettangolare chiuso (lati in metri).

    Il primo segmento e' sempre dritto nella direzione in cui il robot e' girato; le svolte
    successive vanno a destra (clockwise=True) o a sinistra, da scegliere in base allo spazio
    libero attorno al punto di partenza.
    """
    s = -1.0 if clockwise else 1.0
    ang = s * turn_rate
    tgt = s * math.radians(90)
    return [
        ("straight", speed, 0.00, long_side),
        ("turn",     0.00,  ang,  tgt),
        ("straight", speed, 0.00, short_side),
        ("turn",     0.00,  ang,  tgt),
        ("straight", speed, 0.00, long_side),
        ("turn",     0.00,  ang,  tgt),
        ("straight", speed, 0.00, short_side),
        ("turn",     0.00,  ang,  tgt),
        ("stop",     0.00,  0.00, 0.0),
    ]


PATH_SEGMENTS_LOOP = make_loop()


# Modalita' T0: robot fermo, nessun comando di moto. Isola la deriva di MOLA a robot immobile:
# se qui deriva, il problema e' nella catena LiDAR (sub-scan parziali, aggregazione) e non ha
# senso valutare alcun percorso.
PATH_SEGMENTS_STATIONARY = [
    ("stop", 0.00, 0.00, 0.0),
]

PATHS = {
    "serpentine": PATH_SEGMENTS,
    "loop": PATH_SEGMENTS_LOOP,
    "stationary": PATH_SEGMENTS_STATIONARY,
}

# Timeout wall-clock per segmento (rete di sicurezza se il robot resta bloccato contro un
# ostacolo), dimensionati per RTF molto bassi (~0.05).
_TIMEOUT_STRAIGHT_S = 300.0
_TIMEOUT_TURN_S     = 120.0


# ---------------------------------------------------------------------------
# Funzioni geometriche di modulo
# ---------------------------------------------------------------------------

def _yaw_from_quaternion(q) -> float:
    """Estrae lo yaw (rotazione attorno a Z) da un geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_diff(a: float, b: float) -> float:
    """Differenza angolare a-b normalizzata in (-pi, pi]."""
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


def _iter_procs():
    """Itera sui processi attivi leggendo /proc (Linux); produce dict con pid e cmdline."""
    proc_dir = Path("/proc")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            yield {"pid": int(entry.name), "cmdline": cmdline}
        except (FileNotFoundError, PermissionError):
            continue


class OdomRemapper(Node):
    """Legge /chassis/odom e pubblica /chassis/odom_remapped relativo all'inizio della run.

    Isaac Sim pubblica pose assolute nella scena e timestamp cumulativi, mentre MOLA parte da
    (0,0,0) con clock a 0 dopo il reset. Il nodo riporta a zero timestamp, posizione e
    orientamento (q0_inv * q_curr) rispetto al primo messaggio ricevuto.
    """

    def __init__(self):
        super().__init__("odom_remapper")
        self._pub = self.create_publisher(Odometry, "/chassis/odom_remapped", 10)
        self._origin: Odometry | None = None
        self._t0_ns: int | None = None
        self.create_subscription(Odometry, "/chassis/odom", self._odom_cb, 10)

    @staticmethod
    def _quat_mult(aw, ax, ay, az, bw, bx, by, bz):
        """Prodotto di quaternioni a * b, restituisce (w, x, y, z)."""
        return (
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self._origin is None:
            self._origin = msg
            self._t0_ns = t_ns

        dt_ns = t_ns - self._t0_ns

        # Posizione delta in world frame (valida se q0 e' circa l'identita').
        ox = self._origin.pose.pose.position.x
        oy = self._origin.pose.pose.position.y
        oz = self._origin.pose.pose.position.z

        # Orientamento delta: q0_inv * q_curr, con q0_inv = coniugato di q0.
        q0 = self._origin.pose.pose.orientation
        q = msg.pose.pose.orientation
        dw, dx, dy, dz = self._quat_mult(
            q0.w, -q0.x, -q0.y, -q0.z,   # q0_inv
            q.w,   q.x,   q.y,   q.z,     # q_curr
        )

        out = Odometry()
        out.header.stamp = RosTime(
            sec=int(dt_ns // 1_000_000_000),
            nanosec=int(dt_ns % 1_000_000_000),
        )
        out.header.frame_id = "odom"
        out.child_frame_id = "base_link"
        out.pose.pose.position.x = msg.pose.pose.position.x - ox
        out.pose.pose.position.y = msg.pose.pose.position.y - oy
        out.pose.pose.position.z = msg.pose.pose.position.z - oz
        out.pose.pose.orientation.w = dw
        out.pose.pose.orientation.x = dx
        out.pose.pose.orientation.y = dy
        out.pose.pose.orientation.z = dz
        out.twist = msg.twist  # velocita' gia' in body frame, nessuna trasformazione
        self._pub.publish(out)


class ExperimentOrchestrator(Node):
    """Nodo che coordina una run: processi MOLA, controllo del robot, raccolta pose, statistiche."""

    def __init__(
        self,
        run_id: int,
        output_dir: Path,
        mola_binary: str,
        mola_config: str,
        skip_reset: bool,
        deterministic: bool = False,
        path_name: str = "serpentine",
        gt_topic: str = "/chassis/odom",
        stationary_sec: float = 60.0,
        taskset_cpus: str = "",
        sub_scans_per_cycle: int = 0,
        lidar_topic: str = "",
        lidar_pose: tuple = (0.0, 0.0, 0.0),
        loop_segments: list | None = None,
        local_map_size: float = 25.0,
    ):
        super().__init__("baseline_experiment_orchestrator")
        self.run_id = run_id
        self.output_dir = output_dir
        self.mola_binary = mola_binary
        self.mola_config = mola_config
        self.skip_reset = skip_reset
        self.deterministic = deterministic
        self.path_name = path_name
        self.segments = PATHS[path_name]
        if path_name == "loop" and loop_segments is not None:
            self.segments = loop_segments
        self.gt_topic = gt_topic
        self.stationary_sec = stationary_sec
        self.taskset_cpus = taskset_cpus
        self.sub_scans_per_cycle = sub_scans_per_cycle
        self.lidar_topic = lidar_topic
        self.lidar_pose = lidar_pose
        self.local_map_size = local_map_size

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._current_odom: Odometry | None = None
        self._odom_seq = 0          # contatore messaggi odom, usato per il gating di cmd_vel
        self._odom_sub = self.create_subscription(
            Odometry, "/chassis/odom", self._odom_cb, 50
        )
        self.mola_proc = None
        self.intensity_proc = None
        self._scan_count_end = None

        # Le pose vengono accumulate in memoria dai subscriber e scritte in TUM a fine run,
        # senza passare da una bag.
        self._mola_poses: list[tuple] = []   # (t, x, y, z, qx, qy, qz, qw)
        self._gt_poses: list[tuple] = []
        # Profondita' coda 2000: i callback girano nello stesso executor del loop di controllo,
        # che fa spin_once(timeout_sec=0.01) fra un cmd_vel e l'altro. Con una coda corta le
        # pose vengono scartate sotto carico prima di arrivare all'accumulatore e il file TUM
        # ha buchi dovuti al backpressure ROS2, non a MOLA.
        self._mola_sub = self.create_subscription(
            Odometry, "/lidar_odometry/pose", self._mola_pose_cb, 2000
        )
        self._gt_sub = self.create_subscription(
            Odometry, self.gt_topic, self._gt_pose_cb, 2000
        )

        self.tum_mola = output_dir / "mola" / f"run_{run_id}.tum"
        self.tum_gt = output_dir / "gt" / f"run_{run_id}.tum"

        self.tum_mola.parent.mkdir(parents=True, exist_ok=True)
        self.tum_gt.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pulizia: processi orfani di run precedenti
    # ------------------------------------------------------------------

    def _kill_orphans(self):
        """Termina istanze orfane di add_intensity_node e mola-cli di run precedenti.

        Piu' istanze che pubblicano sullo stesso topic LiDAR producono uno stream interleaved
        e MOLA deriva.
        """
        import signal as _sig
        patterns = ["add_intensity_node.py", "mola-cli"]
        my_pid = os.getpid()
        killed = []
        for proc_entry in _iter_procs():
            if proc_entry["pid"] == my_pid:
                continue
            cmdline = proc_entry.get("cmdline", "")
            if any(p in cmdline for p in patterns):
                pid = proc_entry["pid"]
                try:
                    os.kill(pid, _sig.SIGTERM)
                    killed.append((pid, cmdline[:60]))
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
        if killed:
            time.sleep(1.5)
            for pid, cmd in killed:
                try:
                    os.kill(pid, _sig.SIGKILL)
                except ProcessLookupError:
                    pass
            self.get_logger().info(
                f"  Orfani terminati: {[p for p, _ in killed]}"
            )
        else:
            self.get_logger().info("  Nessun processo orfano trovato.")

    def _read_scan_count(self) -> int | None:
        """Legge il contatore di scan pubblicati scritto da add_intensity_node."""
        try:
            return int(Path(self._count_file).read_text().strip())
        except (OSError, ValueError, AttributeError):
            return None

    def run(self):
        """Esegue la run completa: reset, avvio MOLA, percorso, stop, traiettorie, statistiche."""
        det_label = " [DETERMINISTIC]" if self.deterministic else ""
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"  BASELINE RUN {self.run_id}  (odom-based control){det_label}")
        self.get_logger().info(
            f"  Percorso: {self.path_name}  |  segmenti: {len(self.segments)}  "
            f"|  controllo via /chassis/odom  |  riferimento: {self.gt_topic}"
        )
        self.get_logger().info(f"{'='*60}")

        try:
            self._kill_orphans()

            if not self.skip_reset:
                self._reset_isaac_sim()

            self._start_mola()
            # Contatore scan all'inizio della finestra in cui MOLA e' attivo: add_intensity_node
            # parte prima e termina dopo, quindi il totale cumulativo non e' confrontabile con
            # le pose MOLA.
            self._scan_count_start = self._read_scan_count()
            self._execute_motion()
            # La finestra di misura si chiude qui e non in _stop_all: fra la fine del percorso e
            # la terminazione di mola-cli (SIGTERM + wait fino a 5 s) add_intensity_node continua
            # a pubblicare a ~10 Hz verso un processo in chiusura, e quegli scan non vanno contati.
            self._scan_count_end = self._read_scan_count()
            self._stop_all()
            self._write_trajectories()
            self._print_stats()
            self._check_scan_coverage()

        except KeyboardInterrupt:
            self.get_logger().warn("Interrotto. Pulizia in corso...")
            self._stop_all()
            raise
        except Exception as e:
            self.get_logger().error(f"Errore: {e}")
            self._stop_all()
            raise

    # ------------------------------------------------------------------
    # Passo 1: reset di Isaac Sim
    # ------------------------------------------------------------------

    def _reset_isaac_sim(self):
        """Chiama il servizio /isaac/reset_world e attende la stabilizzazione della fisica."""
        self.get_logger().info("Step 1: Reset Isaac Sim...")
        client = self.create_client(Empty, "/isaac/reset_world")

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Servizio /isaac/reset_world non trovato!\n"
                "Esegui isaac_reset_service.py nello Script Editor di Isaac Sim prima di continuare.\n"
                "Oppure usa --skip-reset e fai world.reset() manualmente."
            )
            raise RuntimeError("Isaac Sim reset service not available")

        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            raise RuntimeError("Reset Isaac Sim fallito")

        self.get_logger().info("  Isaac Sim reset OK. Attendo stabilizzazione fisica...")
        time.sleep(2.0)

    # ------------------------------------------------------------------
    # Passo 2: avvio di add_intensity_node e MOLA
    # ------------------------------------------------------------------

    def _start_mola(self):
        """Avvia add_intensity_node e mola-cli, attende MOLA pronto e fa partire la simulazione."""
        self.get_logger().info("Step 2: Avvio add_intensity_node + MOLA...")

        # add_intensity_node: topic LiDAR grezzo -> /carter/lidar_with_intensity
        intensity_script = (
            Path(__file__).resolve().parents[1] / "nodes" / "add_intensity_node.py"
        )
        intensity_log_path = self.output_dir / "logs" / f"add_intensity_run_{self.run_id}.log"
        intensity_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._intensity_log_path = intensity_log_path
        self.intensity_log_file = open(intensity_log_path, "w")

        self._count_file = self.output_dir / "logs" / f"scan_count_run_{self.run_id}.txt"
        intensity_cmd = [sys.executable, "-u", str(intensity_script),
                         "--count-file", str(self._count_file.resolve())]
        if self.sub_scans_per_cycle > 0:
            intensity_cmd += ["--sub-scans-per-cycle", str(self.sub_scans_per_cycle)]
        if self.lidar_topic:
            intensity_cmd += ["--input-topic", self.lidar_topic]

        self.intensity_proc = subprocess.Popen(
            intensity_cmd,
            stdout=self.intensity_log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(1.5)
        if self.intensity_proc.poll() is not None:
            self.intensity_log_file.close()
            out = intensity_log_path.read_text()[:500]
            raise RuntimeError(f"add_intensity_node terminato subito: {out}")
        self.get_logger().info(
            f"  add_intensity_node avviato (pid={self.intensity_proc.pid}), "
            f"log: {intensity_log_path}"
        )

        # MOLA: legge /carter/lidar_with_intensity. I parametri della pipeline sono passati
        # come variabili d'ambiente lette dai YAML in config/mola/.
        env = os.environ.copy()

        env.update({
            # Sensore LiDAR
            "MOLA_LIDAR_TOPIC": "/carter/lidar_with_intensity",
            "MOLA_LIDAR_NAME": "lidar",
            "MOLA_WITH_GUI": "false",
            "MOLA_USE_FIXED_LIDAR_POSE": "true",
            # Posa del LiDAR rispetto a base_link, letta dal /tf della scena
            # (base_link -> XT_32 = (-0.2317, 0, 0.5260), rotazione identita'). Lasciarla a
            # zero fa credere a MOLA che il sensore sia nell'origine del robot: a robot fermo
            # e' ininfluente, ma in curva lo sbalzo longitudinale diventa un braccio di leva
            # e la traiettoria stimata ruota attorno al sensore invece che attorno al robot.
            "LIDAR_POSE_X": str(self.lidar_pose[0]),
            "LIDAR_POSE_Y": str(self.lidar_pose[1]),
            "LIDAR_POSE_Z": str(self.lidar_pose[2]),
            "LIDAR_POSE_YAW": "0", "LIDAR_POSE_PITCH": "0", "LIDAR_POSE_ROLL": "0",
            # IMU non usata: optimize_twist e' pensato per LiDAR-only e stima la velocita'
            # iterativamente dentro ICP. Con l'IMU attiva i due motion prior (optimize_twist e
            # StateEstimationSimple+IMU) entrano in conflitto, ICP parte da una posa sbagliata
            # con alta confidenza e gli scan vengono rifiutati a cascata. Per usare l'IMU
            # servirebbe una pipeline YAML separata senza optimize_twist.
            #
            # Pipeline con optimize_twist: la velocita' lineare e angolare viene stimata durante
            # ICP stesso, evitando i salti quando il velocity model esterno fallisce.
            "MOLA_ODOMETRY_PIPELINE_YAML": str(
                Path(__file__).resolve().parents[2] / "config" / "mola"
                / "lidar3d-gicp-optimize-twist-local.yaml"
            ),
            "MOLA_OPTIMIZE_TWIST": "true",
            # Numero massimo di correzioni della velocita' per scan. Con enforce_planar_motion
            # (3DOF) e sigma_max=0.25 il solver converge in 2-3 iterazioni; 4 lascia margine e
            # contiene il costo per scan.
            "MOLA_OPTIMIZE_TWIST_MAX_CORRECTIONS": "4",
            # Decimazione: con il vincolo planare e sigma_max=0.25 ICP e' piu' vincolato e
            # richiede meno punti per convergere; 2500 per ICP e 4000 per la mappa bilanciano
            # velocita' e accuratezza.
            "MOLA_CLOUD_DECIMATION_VOXEL_SIZE": "0.15",
            "MOLA_DECIMATED_POINTS_ICP": "2500",
            "MOLA_DECIMATED_POINTS_MAP": "4000",
            "MOLA_LOCALIZ_USE_REP105": "false",
            "MOLA_FORWARD_ROS_TF_ODOM_TO_MOLA": "false",
            # Sigma adattivo: la soglia di matching Cov2Cov arriva a 2*sigma_max. Con il default
            # 0.50 m GICP accoppia punti a 1 m di distanza e scivola lungo strutture ripetitive
            # (scaffalature, corridoi). Con 0.25 m la soglia massima e' 0.50 m: sufficiente per
            # il moto fra due scan (~0.13 m a 0.80 m/s), non per accoppiare copie della stessa
            # struttura.
            "MOLA_SIGMA_MAX_MOTION": "0.25",
            # Vincolo planare: il Carter si muove su un pavimento piatto, stimare Z, pitch e
            # roll aggiunge tre gradi di liberta' in cui ICP accumula errore spurio. MOLA stima
            # solo (x, y, yaw).
            "MOLA_NAVSTATE_ENFORCE_PLANAR_MOTION": "true",
        })

        if self.deterministic:
            env.update({
                # Ordine di iterazione stabile per dict/set in Python.
                "PYTHONHASHSEED": "0",

                # Soglia keyframe costante invece dell'espressione di default
                #   (0.1e-2 + sqrt(wx^2+wy^2+wz^2)*0.1) * ESTIMATED_SENSOR_MAX_RANGE
                # che dipende dalla velocita' angolare stimata e da un range ricalcolato scan per
                # scan. In rettilineo w~0 e la soglia collassa a ~0.001*range (circa 1.5 cm con
                # range 15 m): quasi ogni scan diventa keyframe, la local map cresce e il costo
                # per scan sale. Valori costanti rendono gli aggiornamenti di mappa indipendenti
                # dalla stima corrente.
                "MOLA_MIN_XYZ_BETWEEN_MAP_UPDATES": "0.20",
                "MOLA_MIN_ROT_BETWEEN_MAP_UPDATES": "10",

                # Dimensione della local map. Il default del YAML e'
                # max(100.0, 1.50*ESTIMATED_SENSOR_MAX_RANGE): al chiuso vince sempre il ramo
                # 100.0 e i keyframe lontani non vengono mai rimossi.
                #
                # Questo parametro decide il regime in cui gira MOLA, e quindi cosa un attacco
                # adversarial puo' fare:
                #  - valore molto maggiore dell'estensione del percorso (es. 25 m su un anello
                #    con diagonale di pochi metri): il punto di partenza resta nella mappa
                #    locale, la stima e' continuamente riancorata e il drift non si accumula
                #    (robustezza istantanea);
                #  - valore minore dell'estensione del percorso (es. 8 m): la finestra scorre,
                #    MOLA fa odometria incrementale pura e gli errori si accumulano segmento
                #    dopo segmento (regime dello schema receding-horizon).
                # 8 m e' dell'ordine della portata utile del sensore al chiuso e corrisponde al
                # regime in cui MOLA lavora all'aperto, dove la mappa locale e' una frazione
                # minima della traiettoria.
                "MOLA_LOCAL_MAP_MAX_SIZE": str(self.local_map_size),

                # Tracce per-scan (qualita' ICP e sigma adattivo) in mola-lo-traces.csv, per
                # localizzare lo scan in cui due run divergono.
                "MOLA_SAVE_DEBUG_TRACES": "true",
                "MOLA_DEBUG_TRACES_FILE": str(
                    (self.output_dir / "logs" / f"traces_run_{self.run_id}.csv").resolve()
                ),
            })

        # CPU pinning opzionale (--taskset). Default: nessun pinning, MOLA usa tutte le CPU.
        # Pinnare su un solo core serializza il parallelismo TBB residuo ed elimina le
        # variazioni d'ordine fra thread, al costo di throughput.
        cmd = []
        if self.taskset_cpus:
            cmd += ["taskset", "-c", self.taskset_cpus]
        cmd.append(self.mola_binary)
        if self.mola_config:
            cmd.append(self.mola_config)
        # use_sim_time=true allinea l'orologio di MOLA a quello di Isaac Sim; senza, gli scan
        # con timestamp precedenti al reset vengono scartati come "timestamps backwards".
        cmd += ["--ros-args", "-p", "use_sim_time:=true"]

        mola_log_path = self.output_dir / "logs" / f"mola_run_{self.run_id}.log"
        mola_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.mola_log_file = open(mola_log_path, "w")

        self.mola_proc = subprocess.Popen(
            cmd,
            stdout=self.mola_log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        if self.mola_proc.poll() is not None:
            self.mola_log_file.close()
            raise RuntimeError(f"MOLA terminato subito. Vedi log: {mola_log_path}")

        # Il robot parte solo dopo la prima re-localizzazione di MOLA. Un sleep fisso non
        # basta: a seconda del carico MOLA impiega da 2 a 8 s, e se il robot si muove mentre la
        # coda scan iniziale non e' ancora svuotata i delta t risultano enormi e il velocity
        # model fallisce fin dall'inizio.
        self._wait_for_mola_ready(mola_log_path, timeout=30.0)
        self._play_isaac_sim()

        self.get_logger().info(f"  MOLA avviato (pid={self.mola_proc.pid}), log → {mola_log_path}")

    def _wait_for_mola_ready(self, log_path: Path, timeout: float = 30.0) -> None:
        """Attende 'Initial re-localization done' nel log MOLA, poi 2 s per il velocity model."""
        deadline = time.time() + timeout
        self.get_logger().info("  Attendo MOLA ready (Initial re-localization done)...")
        while time.time() < deadline:
            if self.mola_proc.poll() is not None:
                raise RuntimeError("MOLA terminato durante l'attesa di ready.")
            try:
                text = log_path.read_text()
                if "Initial re-localization done" in text:
                    self.get_logger().info("  MOLA ready. Attendo 2s per stabilizzare il velocity model...")
                    time.sleep(2.0)
                    return
            except OSError:
                pass
            time.sleep(0.2)
        self.get_logger().warn(f"  MOLA non ha raggiunto ready entro {timeout}s, procedo comunque.")

    def _play_isaac_sim(self) -> None:
        """Avvia la simulazione creando il file sentinel /tmp/isaac_play.trigger.

        Lo script di Isaac Sim (run_isaac_headless.py) controlla il file ogni frame e chiama
        timeline.play() quando lo trova. Senza watcher attivo il Play va premuto a mano.
        """
        trigger = Path("/tmp/isaac_play.trigger")
        trigger.touch()
        self.get_logger().info("  Isaac Sim: trigger Play scritto → simulazione avviata.")

    # ------------------------------------------------------------------
    # Passo 3: accumulo pose (callback registrati nel costruttore)
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry) -> None:
        self._current_odom = msg
        self._odom_seq += 1

    def _mola_pose_cb(self, msg: Odometry) -> None:
        """Accumula le pose MOLA da /lidar_odometry/pose."""
        if not self._mola_poses:
            # Contatore scan alla prima posa pubblicata da MOLA. Contare dall'avvio del
            # processo sovrastima: fra lo spawn di mola-cli e l'aggancio del suo subscriber
            # passano alcuni secondi in cui add_intensity_node pubblica gia', e quegli scan
            # non erano ricevibili da nessuno.
            self._scan_count_at_first_pose = self._read_scan_count()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._mola_poses.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def _gt_pose_cb(self, msg: Odometry) -> None:
        """Accumula le pose di riferimento dal topic scelto con --gt-topic."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._gt_poses.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def _wait_for_stop(self, timeout: float = 15.0, thresh: float = 0.002) -> None:
        """Attende che la velocita' angolare residua dopo una svolta scenda sotto la soglia.

        La fisica di Isaac Sim mantiene inerzia rotazionale per circa 1 s di sim-time dopo
        cmd_vel=0. La soglia e' stretta perche' la velocita' residua all'uscita si integra per
        tutto il rettilineo successivo: 0.02 rad/s su 5 s fanno 0.1 rad di deriva angolare,
        0.002 rad/s la riducono a circa 0.6 gradi.

        Pubblica cmd_vel=0 a ogni nuovo messaggio odom: senza comando le ruote restano libere e
        il controller non frena.
        """
        t0 = time.time()
        last_seq = -1
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom_seq != last_seq:
                last_seq = self._odom_seq
                self.cmd_vel_pub.publish(Twist())
            if self._current_odom is not None:
                wz = abs(self._current_odom.twist.twist.angular.z)
                if wz < thresh:
                    return
        self.get_logger().warn(
            f"_wait_for_stop: timeout {timeout:.0f}s con soglia {thresh} rad/s, procedo."
        )

    # ------------------------------------------------------------------
    # Passo 4: esecuzione del percorso (basata su odom, indipendente dall'RTF)
    # ------------------------------------------------------------------

    def _execute_stationary(self):
        """Modalita' T0: robot fermo per stationary_sec, si osserva solo la deriva di MOLA.

        Pubblica Twist() a zero a ogni messaggio odom. Non equivale a non pubblicare nulla: un
        comando di velocita' nulla fa applicare al controller differenziale la coppia che tiene
        ferme le ruote, mentre senza comandi le ruote restano libere e il Carter scivola sulle
        ruote pivottanti (frazioni di mm/s, alcuni cm al minuto).

        La cadenza e' agganciata a odom come nel resto del percorso, cosi' il comportamento non
        dipende dal wall-clock. Il criterio di successo lo valuta _print_stats: la deriva di
        MOLA va confrontata con quella del riferimento, non con lo zero assoluto.
        """
        self.get_logger().info(
            f"Step 4: modalita' STAZIONARIA — robot fermo per {self.stationary_sec:.0f}s "
            f"(freno attivo: cmd_vel=0)..."
        )
        t0 = time.time()
        next_report = 10.0
        last_cmd_seq = -1
        while time.time() - t0 < self.stationary_sec:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self._odom_seq != last_cmd_seq:
                last_cmd_seq = self._odom_seq
                self.cmd_vel_pub.publish(Twist())

            elapsed = time.time() - t0
            if elapsed >= next_report:
                v = None
                if self._current_odom is not None:
                    lin = self._current_odom.twist.twist.linear
                    v = math.sqrt(lin.x**2 + lin.y**2)
                self.get_logger().info(
                    f"  t={elapsed:.0f}s  pose MOLA={len(self._mola_poses)}"
                    + (f"  |v|={v*1000:.2f} mm/s" if v is not None else "")
                )
                next_report += 10.0
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info("  Fase stazionaria completata.")

    def _execute_motion(self):
        """Esegue i segmenti del percorso selezionato con controllo chiuso su /chassis/odom."""
        if self.path_name == "stationary":
            self._execute_stationary()
            return

        self.get_logger().info(
            f"Step 4: Esecuzione percorso '{self.path_name}' (odom-based)..."
        )

        # Attesa del primo messaggio odom.
        t_wait = time.time()
        while self._current_odom is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - t_wait > 10.0:
                raise RuntimeError("/chassis/odom non ricevuto entro 10s — Isaac Sim in play?")

        wall_start = time.time()

        for seg_idx, seg in enumerate(self.segments):
            seg_type, linear_x, angular_z, target = seg

            if seg_type == "stop":
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info("  Stop.")
                break

            # Posa iniziale del segmento.
            rclpy.spin_once(self, timeout_sec=0.0)
            pose0 = self._current_odom.pose.pose
            x0 = pose0.position.x
            y0 = pose0.position.y
            yaw0 = _yaw_from_quaternion(pose0.orientation)

            timeout = _TIMEOUT_STRAIGHT_S if seg_type == "straight" else _TIMEOUT_TURN_S
            seg_wall_start = time.time()

            if seg_type == "straight":
                label = f"{target:.1f}m"
            else:
                label = f"{math.degrees(target):.0f}°"
            self.get_logger().info(
                f"  Seg {seg_idx + 1}/{len(self.segments)}: {seg_type} {label}"
                f"  start=({x0:.2f},{y0:.2f}) yaw={math.degrees(yaw0):.1f}°"
            )

            last_cmd_seq = -1

            while True:
                rclpy.spin_once(self, timeout_sec=0.01)

                if time.time() - seg_wall_start > timeout:
                    self.get_logger().warn(
                        f"  Timeout segmento {seg_idx + 1} ({timeout:.0f}s) — robot bloccato?"
                    )
                    break

                pose = self._current_odom.pose.pose
                cur_x = pose.position.x
                cur_y = pose.position.y
                cur_yaw = _yaw_from_quaternion(pose.orientation)

                if seg_type == "straight":
                    dist = math.sqrt((cur_x - x0) ** 2 + (cur_y - y0) ** 2)
                    done = dist >= target
                else:  # turn
                    diff = _angle_diff(cur_yaw, yaw0)
                    done = (diff <= target) if angular_z < 0 else (diff >= target)

                if done:
                    break

                # Gating su odom: un cmd_vel per ogni nuovo messaggio /chassis/odom, non alla
                # velocita' libera del loop Python. Altrimenti la cadenza dei comandi dipende dal
                # wall-clock e dal carico, e a RTF diversi il profilo di velocita' dentro il
                # segmento cambia (stesso punto d'arrivo, traiettoria diversa nel mezzo).
                # Agganciata a odom la cadenza e' in sim-time per costruzione.
                if self._odom_seq == last_cmd_seq:
                    continue
                last_cmd_seq = self._odom_seq

                msg = Twist()
                msg.linear.x = linear_x
                if seg_type == "turn":
                    # Rampa di decelerazione negli ultimi 20 gradi, fino al 20% della velocita'
                    # di svolta, per ridurre l'inerzia angolare residua a fine svolta.
                    remaining = abs(_angle_diff(cur_yaw, yaw0) - target)
                    slow_zone = math.radians(20)
                    speed_factor = min(1.0, max(0.20, remaining / slow_zone))
                    msg.angular.z = angular_z * speed_factor
                else:
                    # Mantenimento dell'assetto nel rettilineo: con angular.z = 0 qualunque
                    # velocita' angolare residua o asimmetria di attrito fa curvare il robot e
                    # l'errore si integra, sempre nello stesso verso, e il rettangolo non si
                    # chiude. Correzione proporzionale sullo yaw iniziale del segmento, con
                    # guadagno basso e saturato per non introdurre oscillazioni nei dati che
                    # MOLA osserva.
                    yaw_err = _angle_diff(yaw0, cur_yaw)
                    msg.angular.z = max(-0.15, min(0.15, 1.0 * yaw_err))
                self.cmd_vel_pub.publish(msg)

            # Arresto del robot a fine segmento.
            self.cmd_vel_pub.publish(Twist())

            rclpy.spin_once(self, timeout_sec=0.0)
            end_pose = self._current_odom.pose.pose
            end_yaw = _yaw_from_quaternion(end_pose.orientation)
            self.get_logger().info(
                f"  Seg {seg_idx + 1} fine: "
                f"pos=({end_pose.position.x:.2f},{end_pose.position.y:.2f})"
                f"  yaw={math.degrees(end_yaw):.1f}°"
            )

            # Dopo una svolta l'inerzia angolare va smaltita prima del rettilineo successivo;
            # a RTF bassi servono diversi secondi di wall-clock, un sleep breve non basta.
            if seg_type == "turn":
                self._wait_for_stop()
            else:
                time.sleep(0.15)

        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info(
            f"  Percorso completato in {time.time() - wall_start:.1f}s (wall-clock)."
        )

    # ------------------------------------------------------------------
    # Passo 5: arresto dei processi
    # ------------------------------------------------------------------

    def _stop_all(self):
        """Termina mola-cli e add_intensity_node e chiude i relativi log."""
        self.get_logger().info("Step 5: Stop MOLA e registrazione...")

        if self.mola_proc:
            try:
                os.killpg(os.getpgid(self.mola_proc.pid), signal.SIGTERM)
                self.mola_proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.mola_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.mola_proc = None

        # Contatore scan di riserva, nel caso run() non l'abbia gia' chiuso (es. interruzione
        # da tastiera a meta' percorso).
        if getattr(self, "_scan_count_end", None) is None:
            self._scan_count_end = self._read_scan_count()

        if hasattr(self, "mola_log_file") and self.mola_log_file:
            self.mola_log_file.close()
            self.mola_log_file = None

        if self.intensity_proc:
            try:
                os.killpg(os.getpgid(self.intensity_proc.pid), signal.SIGTERM)
                self.intensity_proc.wait(timeout=3.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.intensity_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.intensity_proc = None

        if hasattr(self, "intensity_log_file") and self.intensity_log_file:
            self.intensity_log_file.close()
            self.intensity_log_file = None

        time.sleep(1.0)
        self.get_logger().info("  Stop completato.")

    # ------------------------------------------------------------------
    # Passo 6: scrittura delle traiettorie TUM dalle pose in memoria
    # ------------------------------------------------------------------

    def _write_trajectories(self):
        """Scrive i TUM di MOLA, del riferimento e del riferimento allineato ai tempi MOLA."""
        self.get_logger().info("Step 6: Scrittura traiettorie TUM da dati in memoria...")

        mola_poses = self._mola_poses
        gt_poses = self._gt_poses

        self._save_tum(mola_poses, self.tum_mola, "MOLA")
        self._save_tum(gt_poses, self.tum_gt, "Ground Truth")

        # Riferimento allineato ai timestamp MOLA: per ogni posa MOLA si interpola la posizione
        # di riferimento a quel timestamp. I due file hanno lo stesso numero di righe e gli
        # stessi timestamp, quindi l'ATE si calcola direttamente senza associazione esterna.
        if mola_poses and len(gt_poses) >= 2:
            import numpy as np

            gt_arr = np.array(gt_poses)          # (N, 8): t,x,y,z,qx,qy,qz,qw
            gt_t = gt_arr[:, 0]

            aligned_gt = []
            for mp in mola_poses:
                mt = mp[0]  # timestamp MOLA
                # Solo se il timestamp MOLA cade dentro l'intervallo del riferimento.
                if mt < gt_t[0] or mt > gt_t[-1]:
                    continue
                # Interpolazione lineare per x, y, z.
                gx = float(np.interp(mt, gt_t, gt_arr[:, 1]))
                gy = float(np.interp(mt, gt_t, gt_arr[:, 2]))
                gz = float(np.interp(mt, gt_t, gt_arr[:, 3]))
                # Per il quaternione si prende il campione piu' vicino: lo SLERP non serve per
                # un confronto posizionale (ATE).
                idx = int(np.searchsorted(gt_t, mt))
                idx = min(idx, len(gt_arr) - 1)
                qx, qy, qz, qw = gt_arr[idx, 4], gt_arr[idx, 5], gt_arr[idx, 6], gt_arr[idx, 7]
                aligned_gt.append((mt, gx, gy, gz, qx, qy, qz, qw))

            gt_aligned_path = self.tum_gt.parent / f"run_{self.run_id}_aligned.tum"
            self._save_tum(aligned_gt, gt_aligned_path, "GT allineato")

    def _save_tum(self, poses: list, path: Path, label: str):
        """Scrive una lista di pose (t, x, y, z, qx, qy, qz, qw) in formato TUM."""
        if not poses:
            self.get_logger().warn(f"  Nessuna posa {label} registrata.")
            return

        with open(path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for t, x, y, z, qx, qy, qz, qw in poses:
                f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} "
                        f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

        self.get_logger().info(f"  {label}: {len(poses)} pose → {path}")

    # ------------------------------------------------------------------
    # Statistiche
    # ------------------------------------------------------------------

    def _print_stats(self):
        """Stampa spostamento netto, lunghezza percorso e le metriche specifiche del percorso."""
        import numpy as np

        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"  RISULTATI RUN {self.run_id}")
        self.get_logger().info(f"{'='*60}")

        for label, tum_path in [("MOLA", self.tum_mola), ("RIFER", self.tum_gt)]:
            if not tum_path.exists():
                continue
            data = np.loadtxt(tum_path, comments="#")
            if data.ndim == 1 or len(data) < 2:
                continue
            positions = data[:, 1:4]
            # Due grandezze distinte: lo spostamento netto |p_finale - p_iniziale| misura
            # l'accuratezza; la lunghezza del percorso (somma di |delta p|) accumula anche il
            # tremolio della stima e a robot fermo cresce con il numero di pose, quindi misura
            # il jitter, non l'errore.
            net = float(np.linalg.norm(positions[-1] - positions[0]))
            path_len = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            duration = data[-1, 0] - data[0, 0]
            line = (
                f"  {label}: {len(data)} pose in {duration:.1f}s\n"
                f"      spostamento netto  : {net*100:.2f} cm\n"
                f"      lunghezza percorso : {path_len*100:.2f} cm"
            )
            if self.path_name == "stationary":
                # Solo a robot fermo la differenza fra percorso e spostamento e' attribuibile
                # al tremolio della stima; in movimento e' quasi tutta avanzamento reale e
                # divisa per il numero di pose darebbe la distanza percorsa fra due scan.
                wobble_mm = (path_len - net) / max(1, len(positions) - 1) * 1000.0
                line += f"\n      oscillazione/posa  : {wobble_mm:.3f} mm"
            self.get_logger().info(line)

        # Metriche del percorso chiuso: senza mola-sm-loopclosure la distanza fra posa finale
        # e iniziale e' drift accumulato puro, senza allineamento di traiettorie.
        if self.path_name == "loop" and len(self._mola_poses) >= 2:
            p0 = np.array(self._mola_poses[0][1:4])
            pN = np.array(self._mola_poses[-1][1:4])
            drift = float(np.linalg.norm(pN - p0))
            self.get_logger().info(
                f"\n  CHIUSURA DEL LOOP\n"
                f"    MOLA : {drift*100:.1f} cm dalla partenza"
            )
            if self._gt_poses:
                g0 = np.array(self._gt_poses[0][1:4])
                gN = np.array(self._gt_poses[-1][1:4])
                gdrift = float(np.linalg.norm(gN - g0))
                self.get_logger().info(
                    f"    RIFER: {gdrift*100:.1f} cm dalla partenza"
                )
                # La misura di MOLA e' la differenza fra i due valori: se il riferimento non
                # torna al punto di partenza il rettangolo non si e' chiuso fisicamente (deriva
                # di yaw nei rettilinei, overshoot nelle svolte), che e' un problema di
                # controllo e non di SLAM.
                self.get_logger().info(
                    f"    errore MOLA vs riferimento: {abs(drift-gdrift)*100:.2f} cm"
                )
                if gdrift > 0.15:
                    self.get_logger().warn(
                        "    Il riferimento non chiude il loop: il robot non ha\n"
                        "    percorso davvero un rettangolo. Controlla la deriva di\n"
                        "    yaw nei rettilinei nei log dei segmenti."
                    )

        # Metrica T0: deriva a robot fermo. La soglia e' relativa al riferimento e non assoluta:
        # il robot non e' perfettamente immobile nemmeno senza comandi (assestamento delle
        # sospensioni, solver PhysX) e confrontare MOLA con lo zero conterebbe come errore anche
        # il movimento fisico reale.
        if self.path_name == "stationary" and len(self._mola_poses) >= 2:
            arr = np.array([p[1:4] for p in self._mola_poses])
            mola_disp = np.linalg.norm(arr - arr[0], axis=1)
            mola_max = float(mola_disp.max())
            mola_final = float(mola_disp[-1])

            gt_max = gt_final = None
            if len(self._gt_poses) >= 2:
                g = np.array([p[1:4] for p in self._gt_poses])
                gt_disp = np.linalg.norm(g - g[0], axis=1)
                gt_max = float(gt_disp.max())
                gt_final = float(gt_disp[-1])

            self.get_logger().info(
                f"\n  T0 — ROBOT FERMO:\n"
                f"    MOLA  spostamento max: {mola_max*100:.2f} cm   "
                f"finale: {mola_final*100:.2f} cm"
            )
            if gt_max is not None:
                self.get_logger().info(
                    f"    RIFER spostamento max: {gt_max*100:.2f} cm   "
                    f"finale: {gt_final*100:.2f} cm"
                )
                excess = mola_final - gt_final
                self.get_logger().info(
                    f"    ECCESSO MOLA su riferimento (finale): {excess*100:.2f} cm"
                )
                if abs(excess) > 0.03:
                    self.get_logger().error(
                        "    ESITO: FALLITO (soglia 3 cm di eccesso).\n"
                        "    MOLA si muove piu' del riferimento a robot fermo: il\n"
                        "    problema e' nella catena LiDAR. Ispeziona\n"
                        f"    {self._intensity_log_path} e confronta i centroid per scan."
                    )
                else:
                    self.get_logger().info("    ESITO: OK (eccesso < 3 cm).")
            else:
                self.get_logger().warn(
                    "    Nessuna posa di riferimento: impossibile valutare "
                    "l'eccesso, il valore assoluto da solo non e' conclusivo."
                )

            # Il jitter dipende dal numero di pose, quindi non e' confrontabile fra serie
            # campionate a frequenze diverse (MOLA ~10 Hz, riferimento ~60 Hz); il valore
            # robusto e' l'oscillazione per posa.
            mola_len = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
            wobble = (mola_len - mola_final) / max(1, len(arr) - 1) * 1000.0
            self.get_logger().info(
                f"    Jitter MOLA: {wobble:.3f} mm per posa "
                f"({(mola_len-mola_final)*100:.1f} cm accumulati su {len(arr)} pose)"
            )

        self.get_logger().info(
            f"\n  File salvati:\n"
            f"    {self.tum_mola}\n"
            f"    {self.tum_gt}"
        )

    # ------------------------------------------------------------------
    # Verifica copertura scan
    # ------------------------------------------------------------------

    def _check_scan_coverage(self):
        """Confronta gli scan aggregati pubblicati con le pose prodotte da MOLA.

        Se MOLA non processa tutti gli scan ricevuti, quali vengono scartati dipende dallo
        scheduling del sistema operativo e cambia a ogni run; i controller ricorsivi della
        pipeline (adaptive_threshold, icp_quality_controller) portano stato da uno scan al
        successivo, quindi un singolo scan perso fa divergere il resto della run. Una run con
        copertura sotto il 98% va scartata, non interpretata.

        Il conteggio degli scan pubblicati viene dal file --count-file di add_intensity_node.
        """
        n_poses = len(self._mola_poses)

        # Delta del contatore sulla finestra in cui MOLA era attivo: add_intensity_node parte
        # prima e termina dopo, il totale cumulativo sovrastima gli scan ricevibili.
        c0 = getattr(self, "_scan_count_at_first_pose", None)
        c1 = getattr(self, "_scan_count_end", None)
        label = "dalla prima posa MOLA"

        if c0 is None:
            # Nessuna posa ricevuta: si ripiega sul contatore all'avvio del processo.
            c0 = getattr(self, "_scan_count_start", None)
            label = "dall'avvio del processo MOLA"

        if c0 is not None and c1 is not None and c1 >= c0:
            n_published = c1 - c0
            source = f"{label} (contatore {c0} -> {c1})"
        else:
            n_published = None
            source = None

        if n_published is None:
            self.get_logger().warn(
                "  Copertura scan: contatore non disponibile, controllo saltato."
            )
            return

        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("  COPERTURA SCAN")
        self.get_logger().info(f"{'='*60}")
        self.get_logger().info(f"  Misurata su              : {source}")
        self.get_logger().info(f"  Scan pubblicati          : {n_published}")
        self.get_logger().info(f"  Pose MOLA registrate     : {n_poses}")

        if n_published == 0:
            self.get_logger().error(
                "  Nessuno scan pubblicato nella finestra MOLA: verifica che "
                "add_intensity_node riceva dati dal topic di ingresso corretto."
            )
            return

        coverage = n_poses / n_published
        self.get_logger().info(f"  Copertura                : {coverage*100:.1f}%")

        if coverage < 0.98:
            self.get_logger().error(
                f"\n  RUN DA SCARTARE — copertura {coverage*100:.1f}% < 98%.\n"
                f"  {n_published - n_poses} scan non hanno prodotto una posa.\n"
                "  Cause possibili, in ordine di probabilita':\n"
                "    1. MOLA non tiene il passo (riduci scanRateBaseHz del LiDAR\n"
                "       in Isaac Sim, oppure abbassa MOLA_DECIMATED_POINTS_*).\n"
                "    2. min_time_between_scans nel pipeline YAML scarta scan\n"
                "       troppo ravvicinati (valore attuale: 0.080 s).\n"
                "    3. Scan rifiutati per qualita' ICP sotto min_icp_goodness\n"
                f"       (cerca 'quality' in {self.output_dir}/logs/mola_run_{self.run_id}.log)."
            )
        else:
            self.get_logger().info("  ESITO: OK — nessuno scan perso in modo significativo.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Esperimento baseline singolo — muove Carter e registra traiettoria MOLA"
    )
    parser.add_argument("--run-id", type=int, required=True, help="ID della run (1-10)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/trajectories",
        help="Cartella output per file TUM (default: data/trajectories/)",
    )
    parser.add_argument(
        "--mola-binary",
        type=str,
        default="/opt/ros/jazzy/bin/mola-cli",
        help="Path al binario mola-cli",
    )
    parser.add_argument(
        "--mola-config",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "config" / "mola" / "lidar_odometry_ros2_local.yaml"
        ),
        help="Path alla config MOLA (launch YAML con subscribe a /chassis/odom)",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Salta il reset Isaac Sim (fallo manualmente con world.reset() prima di avviare)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Pinna i parametri MOLA che altrimenti dipendono dallo stato stimato: "
            "soglie keyframe costanti, dimensione local map fissa, PYTHONHASHSEED=0, "
            "e abilita le tracce per-scan (traces_run_N.csv)."
        ),
    )
    parser.add_argument(
        "--path",
        choices=sorted(PATHS.keys()),
        default="serpentine",
        help=(
            "Percorso da eseguire. 'stationary' = test T0 (robot fermo, misura la "
            "deriva di MOLA). 'loop' = rettangolo chiuso 4x3 m, il drift di "
            "chiusura e' la metrica di baseline. 'serpentine' = percorso storico."
        ),
    )
    parser.add_argument(
        "--stationary-sec",
        type=float,
        default=60.0,
        help="Durata in secondi wall-clock del test T0 (default: 60).",
    )
    parser.add_argument(
        "--gt-topic",
        type=str,
        default="/chassis/odom",
        help=(
            "Topic usato come riferimento per il confronto. Default /chassis/odom. "
            "Se hai un publisher della posa vera del prim, puntalo qui "
            "(es. /carter/ground_truth): /chassis/odom potrebbe essere odometria "
            "ruote e non verita' di simulazione."
        ),
    )
    parser.add_argument(
        "--sub-scans-per-cycle",
        type=int,
        default=0,
        help=(
            "Passato ad add_intensity_node: aggrega esattamente N sub-scan per "
            "nuvola invece di usare la finestra temporale da 120ms. 0 = finestra "
            "temporale. Per scoprire N, fai una run senza il flag e leggi "
            "l'istogramma '[DIAG] SUB-SCAN per ciclo' nel log di add_intensity."
        ),
    )
    parser.add_argument(
        "--lidar-topic",
        type=str,
        default="",
        help=(
            "Topic PointCloud2 grezzo da passare ad add_intensity_node. "
            "Vuoto = usa il default dello script (/carter/lidar_fix). "
            "Per la scena carter_warehouse: /front_3d_lidar/lidar_points."
        ),
    )
    parser.add_argument(
        "--lidar-pose",
        type=str,
        default="0,0,0",
        help=(
            "Posa del LiDAR rispetto a base_link come 'x,y,z' in metri. "
            "Per la scena carter_warehouse, letta dal /tf: "
            "'-0.2317,0,0.526'. Lasciarla a zero fa ruotare la traiettoria "
            "stimata attorno al sensore invece che attorno al robot."
        ),
    )
    parser.add_argument(
        "--loop-size",
        type=str,
        default="4,3",
        help=(
            "Dimensioni del rettangolo per --path loop, come 'lato_lungo,lato_corto' "
            "in metri (default 4,3). Riducilo se lo spazio libero e' poco."
        ),
    )
    parser.add_argument(
        "--loop-dir",
        choices=["cw", "ccw"],
        default="cw",
        help=(
            "Verso delle svolte: 'cw' spazza a destra del robot, 'ccw' a "
            "sinistra. Da scegliere in base a dove c'e' spazio: il primo "
            "segmento va sempre dritto nella direzione in cui il robot guarda."
        ),
    )
    parser.add_argument(
        "--local-map-size",
        type=float,
        default=25.0,
        help=(
            "MOLA_LOCAL_MAP_MAX_SIZE in metri (attivo solo con --deterministic). "
            "Decide il regime: un valore molto maggiore dell'estensione del "
            "percorso tiene il punto di partenza sempre in mappa e impedisce "
            "l'accumulo di drift (robustezza istantanea); un valore minore fa "
            "scorrere la finestra e rende MOLA odometria pura (il drift "
            "accumula). Default 25, usare 8 per il regime scorrevole."
        ),
    )
    parser.add_argument(
        "--taskset",
        type=str,
        default="",
        dest="taskset_cpus",
        help=(
            "Lista CPU su cui confinare MOLA (formato di taskset, es. '0' oppure "
            "'8-15'). Default: nessun pinning."
        ),
    )
    args = parser.parse_args()

    print(f"\nAttenzione: assicurati di aver fatto world.reset() in Isaac Sim")
    print(f"prima di avviare questa run, oppure usa --skip-reset.\n")

    rclpy.init()

    if args.deterministic:
        print("  Modalita' DETERMINISTICA attiva:")
        print("    - MOLA_MIN_XYZ_BETWEEN_MAP_UPDATES=0.20 (costante, non espressione)")
        print("    - MOLA_MIN_ROT_BETWEEN_MAP_UPDATES=10")
        print(f"    - MOLA_LOCAL_MAP_MAX_SIZE={args.local_map_size} "
              f"({'ancorato' if args.local_map_size >= 20 else 'scorrevole'})")
        print("    - PYTHONHASHSEED=0")
        print("    - tracce per-scan in logs/traces_run_N.csv")
    print(f"  Percorso: {args.path}   |   riferimento: {args.gt_topic}")
    if args.path == "loop":
        print(f"  Loop: {args.loop_size} m, svolte {args.loop_dir}")
    if args.taskset_cpus:
        print(f"  CPU pinning MOLA: {args.taskset_cpus}")

    node = ExperimentOrchestrator(
        run_id=args.run_id,
        output_dir=Path(args.output_dir),
        mola_binary=args.mola_binary,
        mola_config=args.mola_config,
        skip_reset=args.skip_reset,
        deterministic=args.deterministic,
        path_name=args.path,
        gt_topic=args.gt_topic,
        stationary_sec=args.stationary_sec,
        taskset_cpus=args.taskset_cpus,
        sub_scans_per_cycle=args.sub_scans_per_cycle,
        lidar_topic=args.lidar_topic,
        lidar_pose=tuple(float(v) for v in args.lidar_pose.split(",")),
        loop_segments=make_loop(
            long_side=float(args.loop_size.split(",")[0]),
            short_side=float(args.loop_size.split(",")[1]),
            clockwise=(args.loop_dir == "cw"),
        ),
        local_map_size=args.local_map_size,
    )

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"\nRun {args.run_id} completata.")
    print(f"Prossimo step: fai world.reset() in Isaac Sim, poi esegui con --run-id {args.run_id + 1}")


if __name__ == "__main__":
    sys.exit(main())
