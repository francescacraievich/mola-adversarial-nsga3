#!/usr/bin/env python3
"""
Nodo di perturbazione adversarial in linea sul flusso LiDAR.

Sta fra il sensore e MOLA: ogni nuvola viene perturbata con il genoma corrente
prima di raggiungere lo SLAM.

    Isaac Sim -> add_intensity -> [questo nodo] -> MOLA -> waypoint_follower

Il genoma si sostituisce a caldo su /attack/genome, senza riavviare il nodo ne'
interrompere il flusso verso MOLA; l'attacco si abilita e disabilita su
/attack/enabled e lo stato e' pubblicato su /attack/status. Il seed di ogni
nuvola e' hash(genoma) + indice di frame, con l'indice azzerato a ogni cambio di
genoma: la sequenza di nuvole perturbate e' riproducibile a parita' di genoma e
di ingresso. La percettibilita' (Chamfer) e' campionata ogni N frame. Lettura e
scrittura delle nuvole sono vettorizzate per restare nel budget di 100 ms a 10 Hz.

Uso:
    python3 src/nodes/perturbation_node.py                # attacco attivo
    python3 src/nodes/perturbation_node.py --passthrough  # baseline, nuvole invariate

Topic
-----
  in      /carter/lidar_with_intensity   (configurabile)
  out     /carter/lidar_perturbed        (configurabile)
  genoma  /attack/genome        Float32MultiArray, 17 valori in [-1, 1]
  on/off  /attack/enabled       Bool
  stato   /attack/status        String (JSON), per l'orchestratore
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32MultiArray, String

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import (  # noqa: E402
    PerturbationGenerator,
)

# Tutti e quattro i campi sono float32 contigui: si puo' impacchettare il
# messaggio con un solo tobytes() invece di passare per create_cloud().
FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]
POINT_STEP = 16


def genome_seed(genome: np.ndarray) -> int:
    """Seed deterministico derivato dal genoma.

    Lo stesso genoma deve produrre la stessa perturbazione, altrimenti la
    selezione di NSGA-III confronta valori di fitness rumorosi.
    """
    h = hashlib.sha1(np.asarray(genome, dtype=np.float64).tobytes()).digest()
    return int.from_bytes(h[:4], "little")


class PerturbationNode(Node):

    def __init__(self, args):
        super().__init__("perturbation_node")

        self.generator = PerturbationGenerator(
            max_point_shift=args.max_point_shift,
            noise_std=args.noise_std,
            max_dropout_rate=args.dropout_rate,
        )

        self.enabled = not (args.start_disabled or args.passthrough)
        self.frame_idx = 0
        self.genome_id = 0
        self.n_in = 0
        self.n_out = 0
        self.proc_times = []       # calcolo della perturbazione
        self.lat_times = []        # dalla ricezione alla pubblicazione, attese incluse
        # Percettibilita' dell'ultima nuvola misurata: secondo obiettivo di
        # NSGA-III, letto dall'orchestratore su /attack/status.
        self._chamfer_every = args.chamfer_every
        self._passthrough_delay = args.passthrough_delay_ms / 1000.0
        self._min_latency_ms = args.min_latency_ms
        self._gaussian_sigma = args.gaussian_sigma
        self._chamfer = float("nan")

        if args.genome_file:
            g = np.load(args.genome_file)
            self.get_logger().info(f"Genoma caricato da {args.genome_file}")
        else:
            # Un genoma di zeri non e' assenza di perturbazione: ogni gene e'
            # mappato da [-1, 1] al range del suo parametro, quindi lo zero cade
            # a meta' scala ed e' un attacco di media intensita'. Per nuvole
            # invariate serve --passthrough, che scavalca apply_perturbation.
            g = self.generator.random_genome() * args.perturbation_level
        self._set_genome(g, source="avvio")

        self.sub = self.create_subscription(
            PointCloud2, args.input_topic, self._cloud_cb, 10)
        self.pub = self.create_publisher(
            PointCloud2, args.output_topic, 10)

        self.create_subscription(
            Float32MultiArray, "/attack/genome", self._genome_cb, 10)
        self.create_subscription(
            Bool, "/attack/enabled", self._enabled_cb, 10)
        self.status_pub = self.create_publisher(String, "/attack/status", 10)

        self.create_timer(0.2, self._publish_status)

        self.get_logger().info("=" * 64)
        self.get_logger().info("  PERTURBATION NODE")
        self.get_logger().info("=" * 64)
        self.get_logger().info(f"  in  : {args.input_topic}")
        self.get_logger().info(f"  out : {args.output_topic}")
        self.get_logger().info(
            f"  attacco {'ATTIVO' if self.enabled else 'PASSTHROUGH (nuvole invariate)'}")
        if self._min_latency_ms > 0:
            self.get_logger().info(f"  latenza minima per scan: {self._min_latency_ms:.0f} ms")

    # ------------------------------------------------------------------
    # Gestione del genoma
    # ------------------------------------------------------------------

    def _set_genome(self, genome, source: str):
        self.genome = np.asarray(genome, dtype=np.float64)
        self.params = self.generator.encode_perturbation(self.genome)
        self.base_seed = genome_seed(self.genome)
        self.genome_id += 1
        self._chamfer = float("nan")
        # Il seed di ogni nuvola e' base_seed + frame_idx: con un contatore
        # cumulativo lo stesso genoma applicato a frame diversi produrrebbe
        # nuvole diverse (altri punti eliminati, altro rumore, altri ghost) e la
        # deviazione misurata in valutazione non si ripresenterebbe in
        # applicazione. Azzerando il contatore, valutazione e applicazione dello
        # stesso genoma producono la stessa sequenza di nuvole a parita' di
        # ingresso e la fitness e' una funzione del genoma.
        self.frame_idx = 0
        # Il drift temporale accumula un bias frame dopo frame: e' stato
        # per-genoma e non va trascinato nella perturbazione successiva.
        if hasattr(self.generator, "reset_temporal_state"):
            self.generator.reset_temporal_state()
        p = self.params
        self.get_logger().info(
            f"[genoma #{self.genome_id} da {source}]  "
            f"noise {p['noise_intensity']*100:.2f} cm  "
            f"dropout {p['dropout_rate']*100:.1f}%  "
            f"ghost {p['ghost_ratio']*100:.1f}%  "
            f"geom {p.get('geometric_distortion', 0):.3f}  "
            f"edge {p.get('edge_attack_strength', 0):.2f}  "
            f"drift {p.get('temporal_drift_strength', 0):.2f}"
        )

    def _genome_cb(self, msg: Float32MultiArray):
        g = np.array(msg.data, dtype=np.float64)
        expected = self.generator.get_genome_size()
        if len(g) != expected:
            self.get_logger().error(
                f"Genoma di lunghezza {len(g)}, attesi {expected}. Ignorato.")
            return
        self._set_genome(g, source="/attack/genome")

    def _enabled_cb(self, msg: Bool):
        if msg.data != self.enabled:
            self.enabled = msg.data
            self.get_logger().info(
                f"Attacco {'ATTIVATO' if self.enabled else 'DISATTIVATO'}")

    # ------------------------------------------------------------------
    # Elaborazione nuvole
    # ------------------------------------------------------------------

    def _cloud_cb(self, msg: PointCloud2):
        t0 = time.perf_counter()
        self.n_in += 1

        arr = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        if arr is None or len(arr) == 0:
            return

        # Array strutturato: colonne estratte per nome, senza iterare sui punti.
        cloud = np.empty((len(arr), 4), dtype=np.float64)
        cloud[:, 0] = arr["x"]
        cloud[:, 1] = arr["y"]
        cloud[:, 2] = arr["z"]
        cloud[:, 3] = arr["intensity"]

        if self.enabled:
            if self._gaussian_sigma > 0.0:
                # Braccio di confronto: rumore gaussiano isotropo per punto,
                # senza genoma ne' analisi della scena. Stessa Chamfer, stesso
                # seme per frame.
                rng = np.random.default_rng((self.base_seed + self.frame_idx) % (2**31 - 1))
                out = cloud.copy()
                out[:, :3] += rng.normal(0.0, self._gaussian_sigma, size=(len(cloud), 3))
            else:
                out = self.generator.apply_perturbation(
                    cloud, self.params,
                    # Riproducibile a parita' di (genoma, frame), diverso fra frame.
                    seed=(self.base_seed + self.frame_idx) % (2**31 - 1),
                )
            # Percettibilita' campionata ogni chamfer_every frame: due KD-tree
            # per nuvola sforerebbero il budget di 100 ms, e all'orchestratore
            # basta un valore medio sulla finestra.
            if (self._chamfer_every > 0
                    and self.frame_idx % self._chamfer_every == 0):
                try:
                    self._chamfer = float(
                        self.generator.compute_perturbation_magnitude(
                            cloud, out, self.params))
                except Exception:
                    pass
        else:
            out = cloud
            self._chamfer = 0.0
            # Ritardo artificiale a attacco spento: il nominale paga la stessa
            # latenza dei candidati, cosi' il ritardo dell'elaborazione non
            # entra nel danno. Il valore va preso da proc_ms_mean dello status.
            if self._passthrough_delay > 0.0:
                time.sleep(self._passthrough_delay)

        t_proc = time.perf_counter() - t0
        # Latenza minima uguale per nominale e candidati, qualunque sia il
        # braccio: la differenza di tempo di calcolo non entra nel danno. Gli
        # scan che la superano gia' non vengono accorciati.
        if self._min_latency_ms > 0:
            wait = self._min_latency_ms / 1000.0 - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        self._publish_cloud(out, msg.header)
        self.frame_idx += 1
        self.n_out += 1
        self.proc_times.append(t_proc)
        self.lat_times.append(time.perf_counter() - t0)
        if len(self.proc_times) > 200:
            self.proc_times.pop(0)
            self.lat_times.pop(0)

    def _publish_cloud(self, cloud: np.ndarray, header):
        if cloud.shape[1] == 3:
            cloud = np.hstack([cloud, np.full((len(cloud), 1), 100.0)])
        data = np.ascontiguousarray(cloud[:, :4], dtype=np.float32)

        out = PointCloud2()
        out.header = header
        out.height = 1
        out.width = len(data)
        out.fields = FIELDS
        out.is_bigendian = False
        out.point_step = POINT_STEP
        out.row_step = POINT_STEP * len(data)
        out.data = data.tobytes()
        out.is_dense = True
        self.pub.publish(out)

    # ------------------------------------------------------------------
    # Stato
    # ------------------------------------------------------------------

    def _publish_status(self):
        if not self.proc_times:
            return
        mean_ms = float(np.mean(self.proc_times)) * 1000.0
        max_ms = float(np.max(self.proc_times)) * 1000.0
        st = {
            "enabled": self.enabled,
            "genome_id": self.genome_id,
            "frames_in": self.n_in,
            "frames_out": self.n_out,
            "proc_ms_mean": round(mean_ms, 2),
            "proc_ms_max": round(max_ms, 2),
            "latency_ms_mean": round(float(np.mean(self.lat_times)) * 1000.0, 2),
            "latency_ms_max": round(float(np.max(self.lat_times)) * 1000.0, 2),
            "min_latency_ms": self._min_latency_ms,
            "chamfer_cm": (None if self._chamfer != self._chamfer
                           else round(self._chamfer, 3)),
        }
        self.status_pub.publish(String(data=json.dumps(st)))
        # A 10 Hz il budget per nuvola e' 100 ms: oltre, il nodo stesso causa
        # scan persi e l'effetto dell'attacco non e' separabile dal ritardo.
        if mean_ms > 80.0:
            self.get_logger().warn(
                f"Elaborazione {mean_ms:.0f} ms per nuvola: vicino al budget di "
                f"100 ms a 10 Hz. Rischio di scartare frame."
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-topic", type=str, default="/carter/lidar_with_intensity")
    ap.add_argument("--output-topic", type=str, default="/carter/lidar_perturbed")
    ap.add_argument("--genome-file", type=str, default=None)
    ap.add_argument("--passthrough", action="store_true",
                    help="non perturba: le nuvole passano invariate. Serve a "
                         "misurare la baseline in anello chiuso e il costo del "
                         "nodo. NON usare un genoma di zeri per questo: lo zero "
                         "cade a meta' scala di ogni parametro ed e' un attacco "
                         "di media intensita'.")
    ap.add_argument("--perturbation-level", type=float, default=0.5)
    ap.add_argument("--max-point-shift", type=float, default=0.05)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--dropout-rate", type=float, default=0.15)
    ap.add_argument("--gaussian-sigma", type=float, default=0.0,
                    help="se > 0, con attacco acceso applica solo rumore gaussiano "
                         "isotropo per punto con questa deviazione standard (m), "
                         "ignorando il genoma: braccio di confronto")
    ap.add_argument("--passthrough-delay-ms", type=float, default=0.0,
                    help="ritardo aggiunto a ogni nuvola quando l'attacco e' spento, "
                         "per dare al rollout nominale la latenza dei candidati "
                         "(tipicamente proc_ms_mean dello status, ~85 ms). Tenuto per "
                         "compatibilita': per la campagna --min-latency-ms")
    ap.add_argument("--min-latency-ms", type=float, default=0.0,
                    help="durata minima di ogni scan dalla ricezione alla pubblicazione, "
                         "ad attacco acceso o spento e per ogni braccio; gli scan piu' "
                         "lenti non vengono accorciati (campagna: 85)")
    ap.add_argument("--chamfer-every", type=int, default=5,
                    help="calcola la percettibilita' ogni N frame (0 = mai). "
                         "Costa due KD-tree su ~45k punti: ad ogni nuvola "
                         "sforerebbe il budget di 100 ms.")
    ap.add_argument("--start-disabled", action="store_true",
                    help="parte in passthrough, si attiva via /attack/enabled")
    cli, ros_args = ap.parse_known_args(sys.argv[1:])

    rclpy.init(args=ros_args)
    node = PerturbationNode(cli)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
