#!/usr/bin/env python3
"""
NSGA-II optimization for adversarial perturbations against MOLA SLAM.

Perturbation approach:
- Per-point shifts (not rigid transforms)
- Targets high-curvature regions
- Chamfer distance as imperceptibility metric
- Bounds in centimeter scale

References:
- FLAT (ECCV 2024)
- Adversarial Point Cloud Perturbations (Neurocomputing 2021)
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Go to project root

from src.evaluation.metrics import compute_localization_error  # noqa: E402
from src.perturbations.perturbation_generator import (  # noqa: E402
    PerturbationGenerator,
)
from src.utils.data_loaders import (  # noqa: E402
    load_point_clouds_from_npy,
    load_timestamps_from_npy,
    load_trajectory_from_tum,
)


class MOLAEvaluator(Node):
    """ROS2 node that evaluates perturbations by running MOLA and measuring ATE."""

    def __init__(
        self,
        perturbation_generator: PerturbationGenerator,
        ground_truth_trajectory: np.ndarray,
        point_cloud_sequence: list,
        timestamps: np.ndarray,
        mola_binary_path: str,
        mola_config_path: str,
        bag_path: str = None,
        lidar_topic: str = "/mola_nsga2/lidar",
        odom_topic: str = "/lidar_odometry/pose",
    ):
        super().__init__("mola_evaluator")

        self.perturbation_generator = perturbation_generator
        self.ground_truth_trajectory = ground_truth_trajectory
        self.point_cloud_sequence = point_cloud_sequence
        self.timestamps = timestamps
        self.mola_binary_path = mola_binary_path
        self.mola_config_path = mola_config_path
        self.bag_path = bag_path
        self.lidar_topic = lidar_topic
        self.odom_topic = odom_topic

        self.pc_publisher = self.create_publisher(PointCloud2, lidar_topic, 10)
        self.tf_publisher = self.create_publisher(TFMessage, "/tf", 10)

        self.chassis_odom_subscriber = self.create_subscription(
            Odometry, "/chassis/odom", self._chassis_odom_callback, 10
        )

        self.collected_trajectory = []
        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, 10
        )

        self.evaluation_count = 0
        self.mola_process = None
        self.bag_process = None
        self.original_clouds = point_cloud_sequence

        self.get_logger().info(
            f"Evaluator ready - max shift: {perturbation_generator.max_point_shift * 100:.1f} cm"
        )

    def _chassis_odom_callback(self, msg):
        """Forward chassis odom as TF."""
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = msg.child_frame_id
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        tf_msg = TFMessage()
        tf_msg.transforms.append(t)
        self.tf_publisher.publish(tf_msg)

    def _odom_callback(self, msg):
        """Store MOLA's estimated poses."""
        pos = msg.pose.pose.position
        self.collected_trajectory.append([pos.x, pos.y, pos.z])

    def _create_pointcloud2_msg(self, point_cloud, timestamp_ns):
        """Convert numpy array to PointCloud2."""
        from builtin_interfaces.msg import Time
        from std_msgs.msg import Header

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = Time(sec=int(timestamp_ns // 1_000_000_000), nanosec=int(timestamp_ns % 1_000_000_000))
        header.frame_id = "base_link"

        msg = point_cloud2.create_cloud(header, fields, point_cloud)
        return msg

    def _start_bag_play(self):
        """Play back TF and odom from bag."""
        if not self.bag_path:
            return

        cmd = [
            "ros2",
            "bag",
            "play",
            self.bag_path,
            "--topics",
            "/tf",
            "/tf_static",
            "/chassis/odom",
            "--loop",
            "--rate",
            "1.0",
        ]

        self.bag_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

    def _stop_bag_play(self):
        """Kill bag playback."""
        if self.bag_process:
            try:
                os.killpg(os.getpgid(self.bag_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.bag_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.bag_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.bag_process = None

    def _start_mola(self):
        """Launch MOLA in background."""
        env = os.environ.copy()
        env["MOLA_LIDAR_TOPIC"] = self.lidar_topic
        env["MOLA_LIDAR_NAME"] = "lidar"
        env["MOLA_WITH_GUI"] = "false"
        env["MOLA_USE_FIXED_LIDAR_POSE"] = "true"
        env["LIDAR_POSE_X"] = "0"
        env["LIDAR_POSE_Y"] = "0"
        env["LIDAR_POSE_Z"] = "0"
        env["LIDAR_POSE_YAW"] = "0"
        env["LIDAR_POSE_PITCH"] = "0"
        env["LIDAR_POSE_ROLL"] = "0"

        cmd = [self.mola_binary_path]
        if self.mola_config_path:
            cmd.append(self.mola_config_path)

        self.mola_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid,
        )

    def _stop_mola(self):
        """Kill MOLA and cleanup."""
        if self.mola_process:
            try:
                os.killpg(os.getpgid(self.mola_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.mola_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.mola_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.mola_process = None

        try:
            subprocess.run(["pkill", "-9", "-f", "mola-cli"], capture_output=True, timeout=2.0)
        except Exception:
            pass
        time.sleep(0.5)

    def _generate_perturbed_sequence(self, params):
        """Generate perturbed point cloud sequence and compute avg Chamfer."""
        perturbed_sequence = []
        total_chamfer = 0

        for i, cloud in enumerate(self.point_cloud_sequence):
            perturbed = self.perturbation_generator.apply_perturbation(
                cloud, params, seed=self.evaluation_count * 1000 + i
            )
            perturbed_sequence.append(perturbed)

            if i < 10:
                chamfer = self.perturbation_generator.compute_chamfer_distance(cloud, perturbed)
                total_chamfer += chamfer

        avg_chamfer = total_chamfer / min(10, len(self.point_cloud_sequence))
        return perturbed_sequence, avg_chamfer

    def _wait_for_trajectory(self, min_points_expected, max_wait):
        """Wait for MOLA to collect enough trajectory points."""
        start_wait = time.time()
        last_count = 0
        stable_count = 0

        while (time.time() - start_wait) < max_wait:
            rclpy.spin_once(self, timeout_sec=0.1)
            current_count = len(self.collected_trajectory)
            if current_count > last_count:
                last_count = current_count
                stable_count = 0
            else:
                stable_count += 1
            if current_count >= min_points_expected and stable_count > 20:
                break
            time.sleep(0.1)

    def _publish_clouds(self, perturbed_sequence, rate_hz=10.0):
        """Publish perturbed point clouds to MOLA."""
        dt = 1.0 / rate_hz
        for i, cloud in enumerate(perturbed_sequence):
            msg = self._create_pointcloud2_msg(cloud, self.timestamps[i])
            self.pc_publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=dt)
            time.sleep(dt)

    def _retry_evaluation(self, perturbed_sequence, min_valid_points):
        """Retry MOLA evaluation after failure."""
        subprocess.run(["pkill", "-9", "-f", "mola"], capture_output=True, timeout=2.0)
        subprocess.run(["pkill", "-9", "-f", "ros2.bag"], capture_output=True, timeout=2.0)
        time.sleep(5.0)

        self._start_mola()
        self._start_bag_play()
        time.sleep(5.0)

        self.collected_trajectory = []
        self._publish_clouds(perturbed_sequence, rate_hz=10.0)

        time.sleep(5.0)
        self._wait_for_trajectory(min_valid_points, max_wait=15.0)

        self._stop_mola()
        self._stop_bag_play()
        return np.array(self.collected_trajectory) if self.collected_trajectory else np.array([])

    def evaluate(self, genome: np.ndarray) -> tuple:
        """Run MOLA with perturbed clouds and return (neg_ate, chamfer_distance)."""
        self.evaluation_count += 1
        self.get_logger().info(f"\n{'=' * 60}")
        self.get_logger().info(f"Evaluation #{self.evaluation_count}")
        self.get_logger().info(f"{'=' * 60}")

        params = self.perturbation_generator.encode_perturbation(genome)

        self.get_logger().info(f"  Noise intensity: {params['noise_intensity'] * 100:.2f} cm")
        self.get_logger().info(f"  Curvature targeting: {params['curvature_strength']:.2f}")
        self.get_logger().info(f"  Dropout rate: {params['dropout_rate'] * 100:.1f}%")
        self.get_logger().info(f"  Ghost ratio: {params['ghost_ratio'] * 100:.1f}%")

        perturbed_sequence, avg_chamfer = self._generate_perturbed_sequence(params)
        self.get_logger().info(f"  Avg Chamfer distance: {avg_chamfer * 100:.3f} cm")

        self._start_mola()
        self._start_bag_play()
        time.sleep(5.0)

        self.collected_trajectory = []
        self.get_logger().info(f"Publishing {len(perturbed_sequence)} perturbed clouds...")
        self._publish_clouds(perturbed_sequence)

        self.get_logger().info("Waiting for MOLA to process...")
        min_points_expected = 40
        self._wait_for_trajectory(min_points_expected, max_wait=15.0)

        self._stop_mola()
        self._stop_bag_play()

        estimated_traj = (
            np.array(self.collected_trajectory) if self.collected_trajectory else np.array([])
        )
        self.get_logger().info(f"Collected {len(estimated_traj)} trajectory points")

        # Retry logic
        min_valid_points = 40
        max_retries = 2
        for retry_count in range(max_retries):
            if len(estimated_traj) >= min_valid_points:
                break
            self.get_logger().warn(
                f"Not enough points ({len(estimated_traj)}<{min_valid_points}) "
                f"- retry {retry_count + 1}/{max_retries}..."
            )
            estimated_traj = self._retry_evaluation(perturbed_sequence, min_valid_points)
            self.get_logger().info(f"Retry collected {len(estimated_traj)} trajectory points")

        if len(estimated_traj) < min_valid_points:
            self.get_logger().error(f"Failed after {max_retries} retries - returning invalid fitness")
            return (np.inf, np.inf)

        min_len = min(len(self.ground_truth_trajectory), len(estimated_traj))
        ate = compute_localization_error(
            self.ground_truth_trajectory[:min_len], estimated_traj[:min_len], method="ate"
        )

        pert_mag = self.perturbation_generator.compute_perturbation_magnitude(
            self.original_clouds[0], perturbed_sequence[0], params
        )

        self.get_logger().info(f"Fitness: ATE={ate:.4f}m, Pert={pert_mag:.4f}")

        return (-ate, pert_mag)

    def get_fitness_function(self):
        """Return fitness function for optimizer."""
        return lambda genome: self.evaluate(genome)


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NSGA-II optimization for adversarial perturbations")
    parser.add_argument("--gt-traj", type=str, default="maps/ground_truth_trajectory.tum")
    parser.add_argument("--frames", type=str, default="data/frame_sequence.npy")
    parser.add_argument("--mola-binary", type=str, default="/opt/ros/jazzy/bin/mola-cli")
    parser.add_argument(
        "--mola-config",
        type=str,
        default="/opt/ros/jazzy/share/mola_lidar_odometry/mola-cli-launchs/lidar_odometry_ros2.yaml",
    )
    parser.add_argument("--bag", type=str, default="bags/lidar_sequence_with_odom")
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--output", type=str, default="src/results/optimized_genome.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-point-shift",
        type=float,
        default=0.05,
        help="Max per-point displacement in meters (default: 5cm)",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.02, help="Gaussian noise std in meters (default: 2cm)"
    )
    parser.add_argument(
        "--max-dropout", type=float, default=0.15, help="Max dropout rate (default: 15%)"
    )
    return parser.parse_args()


def _get_next_run_number(base_path: Path) -> int:
    """Find the next available run number."""
    base_dir = base_path.parent
    base_name = base_path.stem  # e.g., "optimized_genome"

    if not base_dir.exists():
        return 1

    existing_numbers = []
    for file in base_dir.glob(f"{base_name}*.npy"):
        # Extract number from filenames like optimized_genome1.npy, optimized_genome2.npy
        name = file.stem
        if name == base_name:
            # No number, treat as run 0
            existing_numbers.append(0)
        elif name.startswith(base_name):
            suffix = name[len(base_name):]
            if suffix.isdigit():
                existing_numbers.append(int(suffix))

    return max(existing_numbers, default=0) + 1


def _print_results(result, elapsed, pareto_front, pareto_set, history_callback, output_path):
    """Print optimization results and save files."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"\nFound {len(pareto_front)} Pareto-optimal solutions")
    print(f"  Time: {elapsed / 60:.1f} minutes")

    print("\nPareto Front (ATE vs Perturbation Magnitude):")
    print(f"{'ATE (m)':<12} {'Pert Mag':<12} {'Attack Ratio':<12}")
    print("-" * 40)

    for f in pareto_front:
        ate = -f[0]
        pert = f[1]
        ratio = ate / max(pert, 0.001)
        print(f"{ate:<12.4f} {pert:<12.4f} {ratio:<12.2f}")

    ratios = [-f[0] / max(f[1], 0.001) for f in pareto_front]
    best_idx = np.argmax(ratios)

    print("\nBest stealth attack (highest ATE/perturbation ratio):")
    print(f"   ATE: {-pareto_front[best_idx][0]:.4f}m")
    print(f"   Perturbation: {pareto_front[best_idx][1]:.4f}")
    print(f"   Ratio: {ratios[best_idx]:.2f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, pareto_set[best_idx])
    np.save(output_path.with_suffix(".pareto_front.npy"), pareto_front)
    np.save(output_path.with_suffix(".pareto_set.npy"), pareto_set)

    all_points = np.array(history_callback.all_fitness)
    valid_points = (
        np.array(history_callback.valid_fitness) if history_callback.valid_fitness else np.array([])
    )
    np.save(output_path.with_suffix(".all_points.npy"), all_points)
    np.save(output_path.with_suffix(".valid_points.npy"), valid_points)

    print(f"\nSaved to: {output_path}")
    print(f"   - Best genome: {output_path}")
    print(f"   - Pareto front: {output_path.with_suffix('.pareto_front.npy')}")
    print(
        f"   - All {len(all_points)} evaluated points: {output_path.with_suffix('.all_points.npy')}"
    )
    print(
        f"   - Valid {len(valid_points)} points (ATE<10m): "
        f"{output_path.with_suffix('.valid_points.npy')}"
    )


def _load_data(args):
    """Load ground truth trajectory, point clouds, and timestamps."""
    print("Loading data...")
    gt_traj = load_trajectory_from_tum(args.gt_traj)
    if gt_traj is None:
        print("Failed to load ground truth")
        return None, None, None

    clouds = load_point_clouds_from_npy(args.frames)
    if clouds is None:
        return None, None, None

    timestamps = load_timestamps_from_npy(args.frames.replace(".npy", ".timestamps.npy"))
    if timestamps is None:
        print("Failed to load timestamps")
        return None, None, None

    print(f"  Ground truth: {len(gt_traj)} poses")
    print(f"  Point clouds: {len(clouds)} frames")
    print(f"  Timestamps: {len(timestamps)}")
    return gt_traj, clouds, timestamps


def _create_evaluator(args, gt_traj, clouds, timestamps):
    """Create perturbation generator and evaluator."""
    generator = PerturbationGenerator(
        max_point_shift=args.max_point_shift,
        noise_std=args.noise_std,
        target_high_curvature=True,
        curvature_percentile=90.0,
        max_dropout_rate=args.max_dropout,
        max_ghost_points_ratio=0.05,
    )

    evaluator = MOLAEvaluator(
        perturbation_generator=generator,
        ground_truth_trajectory=gt_traj,
        point_cloud_sequence=clouds,
        timestamps=timestamps,
        mola_binary_path=args.mola_binary,
        mola_config_path=args.mola_config,
        bag_path=args.bag,
        lidar_topic="/mola_nsga2/lidar",
        odom_topic="/lidar_odometry/pose",
    )
    return generator, evaluator


def _run_optimization(args, problem, algorithm, history_callback):
    """Run the NSGA-II optimization."""
    from pymoo.optimize import minimize

    print("\nStarting NSGA-II optimization")
    print(f"   Population: {args.pop_size}, Generations: {args.n_gen}")
    total_evals = args.pop_size * args.n_gen
    print(f"   Total evaluations: {total_evals}")
    print(f"   Estimated time: ~{(total_evals * 50) / 60:.0f} minutes")
    print("\nStarting optimization...")

    start_time = time.time()
    result = minimize(
        problem,
        algorithm,
        ("n_gen", args.n_gen),
        seed=args.seed,
        verbose=True,
        callback=history_callback,
    )
    elapsed = time.time() - start_time
    return result, elapsed


def _print_header(args):
    """Print optimization header with settings."""
    print("\n" + "=" * 80)
    print(" NSGA-II ADVERSARIAL PERTURBATION OPTIMIZATION")
    print("=" * 80)
    print("\nPerturbation settings (based on research papers):")
    print(f"  Max point shift: {args.max_point_shift * 100:.1f} cm")
    print(f"  Noise std: {args.noise_std * 100:.1f} cm")
    print(f"  Max dropout: {args.max_dropout * 100:.0f}%")
    print("  Feature targeting: Enabled (high curvature regions)")
    print()


def _setup_optimizer(args, evaluator, generator):
    """Set up NSGA-II optimizer components."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    class MOLAPerturbationProblem(Problem):
        def __init__(self, evaluator_func, genome_size):
            super().__init__(
                n_var=genome_size,
                n_obj=2,
                xl=-1.0 * np.ones(genome_size),
                xu=1.0 * np.ones(genome_size),
            )
            self.evaluator_func = evaluator_func

        def _evaluate(self, X, out, *_args, **_kwargs):
            out["F"] = np.array([self.evaluator_func(g) for g in X])

    class SaveHistoryCallback(Callback):
        def __init__(self):
            super().__init__()
            self.all_fitness = []
            self.valid_fitness = []

        def notify(self, algorithm):
            for ind in algorithm.pop:
                if ind.F is not None:
                    self.all_fitness.append(ind.F.copy())
                    if -ind.F[0] < 10.0:
                        self.valid_fitness.append(ind.F.copy())

    problem = MOLAPerturbationProblem(evaluator.get_fitness_function(), generator.get_genome_size())
    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    history_callback = SaveHistoryCallback()
    return problem, algorithm, history_callback


def main():
    """Main function for NSGA-II optimization."""
    args = _parse_args()
    _print_header(args)

    # Generate numbered output filename
    base_output_path = Path(args.output)
    run_number = _get_next_run_number(base_output_path)
    numbered_output = base_output_path.parent / f"{base_output_path.stem}{run_number}{base_output_path.suffix}"

    # Create results directory if it doesn't exist
    numbered_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {numbered_output}\n")

    gt_traj, clouds, timestamps = _load_data(args)
    if gt_traj is None:
        return 1

    rclpy.init()
    generator, evaluator = _create_evaluator(args, gt_traj, clouds, timestamps)
    problem, algorithm, history_callback = _setup_optimizer(args, evaluator, generator)

    try:
        result, elapsed = _run_optimization(args, problem, algorithm, history_callback)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        evaluator._stop_mola()
        evaluator._stop_bag_play()
        evaluator.destroy_node()
        rclpy.shutdown()
        return 1

    _print_results(result, elapsed, result.F, result.X, history_callback, numbered_output)

    evaluator.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
