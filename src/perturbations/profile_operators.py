#!/usr/bin/env python3
"""
Profilazione degli operatori di perturbazione su una nuvola reale.

Cattura una nuvola dal topic live (o dal primo frame di un .npy) e cronometra
apply_perturbation nel suo insieme e ogni operatore separatamente, riportando
la mediana su piu' ripetizioni e la quota di ciascuno sul totale.

Il budget in anello chiuso e' 100 ms per nuvola a 10 Hz: oltre, il nodo di
perturbazione introduce un ritardo proprio e l'effetto dell'attacco non e' piu'
separabile da quello della latenza.

Uso:
    python3 src/perturbations/profile_operators.py
    python3 src/perturbations/profile_operators.py --topic /carter/lidar_with_intensity
    python3 src/perturbations/profile_operators.py --npy data/frame_sequence.npy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402


def grab_cloud_from_topic(topic: str, timeout: float = 15.0):
    """Cattura una nuvola dal topic ROS 2."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    holder = {}

    class Grab(Node):
        def __init__(self):
            super().__init__("profile_grabber")
            self.create_subscription(PointCloud2, topic, self.cb, 1)

        def cb(self, msg):
            if "cloud" in holder:
                return
            a = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            c = np.empty((len(a), 4))
            c[:, 0], c[:, 1], c[:, 2] = a["x"], a["y"], a["z"]
            c[:, 3] = 100.0
            holder["cloud"] = c

    rclpy.init()
    n = Grab()
    t0 = time.time()
    while "cloud" not in holder and time.time() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
    n.destroy_node()
    rclpy.shutdown()
    return holder.get("cloud")


def timeit(fn, repeats: int = 3):
    """Mediana in ms di `repeats` esecuzioni, robusta a un singolo outlier."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=str, default="/carter/lidar_with_intensity")
    ap.add_argument("--npy", type=str, default=None,
                    help="usa il primo frame di questo file invece del topic")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if args.npy:
        seq = np.load(args.npy, allow_pickle=True)
        cloud = np.asarray(seq[0], dtype=np.float64)
        if cloud.shape[1] == 3:
            cloud = np.hstack([cloud, np.full((len(cloud), 1), 100.0)])
        print(f"Nuvola dal file: {args.npy}")
    else:
        print(f"In attesa di una nuvola su {args.topic} ...")
        cloud = grab_cloud_from_topic(args.topic)
        if cloud is None:
            print("Nessuna nuvola ricevuta. Isaac Sim e' in Play? Il nodo pubblica?")
            return 1

    n = len(cloud)
    print(f"Nuvola: {n} punti\n")

    gen = PerturbationGenerator(max_point_shift=0.05, noise_std=0.02,
                                max_dropout_rate=0.15)
    genome = gen.random_genome() * 0.5
    params = gen.encode_perturbation(genome)

    weights = np.ones(n)

    # Ogni voce: (etichetta, callable) con la firma dell'operatore corrispondente
    # in PerturbationGenerator.
    cases = [
        ("apply_perturbation (TOTALE)",
         lambda: gen.apply_perturbation(cloud, params, seed=1)),
        ("_compute_perturbation_weights",
         lambda: gen._compute_perturbation_weights(cloud.copy(), n, params)),
        ("compute_curvature",
         lambda: gen.compute_curvature(cloud[:, :3])),
        ("detect_edges_and_corners",
         lambda: gen.detect_edges_and_corners(cloud[:, :3])),
        ("_apply_noise",
         lambda: gen._apply_noise(cloud.copy(), n, weights, params)),
        ("_apply_dropout",
         lambda: gen._apply_dropout(cloud.copy(), n, weights, params)),
        ("_apply_cluster_perturbation",
         lambda: gen._apply_cluster_perturbation(
             cloud.copy(), params.get("cluster_direction", np.array([1.0, 0, 0])),
             params.get("cluster_strength", 0.5))),
        ("_add_ghost_points",
         lambda: gen._add_ghost_points(cloud.copy(), params)),
        ("_apply_geometric_distortion",
         lambda: gen._apply_geometric_distortion(cloud.copy(), params)),
        ("_apply_edge_attack",
         lambda: gen._apply_edge_attack(cloud.copy(), params)),
        ("_apply_temporal_drift",
         lambda: gen._apply_temporal_drift(cloud.copy(), params)),
        ("_apply_scanline_perturbation",
         lambda: gen._apply_scanline_perturbation(cloud.copy(), params)),
    ]

    print(f"{'operatore':<34} {'ms':>9}   {'% budget 100ms':>15}")
    print("-" * 64)

    total = None
    rows = []
    for label, fn in cases:
        if not hasattr(gen, fn.__code__.co_names[1] if False else "apply_perturbation"):
            pass
        ms, err = timeit(fn, args.repeats)
        if err:
            print(f"{label:<34} {'n/d':>9}   ({err[:40]})")
            continue
        if "TOTALE" in label:
            total = ms
            print(f"{label:<34} {ms:>8.1f}   {ms:>14.0f}%")
            print("-" * 64)
        else:
            rows.append((label, ms))

    for label, ms in sorted(rows, key=lambda r: -r[1]):
        share = f"{100.0*ms/total:.0f}% del totale" if total else ""
        print(f"{label:<34} {ms:>8.1f}   {share:>15}")

    print()
    if total:
        print(f"Totale misurato: {total:.0f} ms  (budget 100 ms a 10 Hz)")
        if total > 100:
            print(f"Sforo di {total-100:.0f} ms. Gli operatori in cima alla lista")
            print("sono quelli da affrontare per primi: tipicamente quelli che")
            print("costruiscono un KD-tree o calcolano vicinati sull'intera nuvola.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
