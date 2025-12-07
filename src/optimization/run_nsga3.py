#!/usr/bin/env python3
"""
NSGA-III optimization for adversarial perturbations against MOLA SLAM.

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
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402
from src.utils.data_loaders import (  # noqa: E402
    load_point_clouds_from_npy,
    load_timestamps_from_npy,
    load_trajectory_from_tum,
)


# Load TF data from saved files
def load_tf_from_npy(tf_path: str) -> list:
    """Load TF sequence from numpy file."""
    import numpy as np

    tf_data = np.load(tf_path, allow_pickle=True)
    return list(tf_data)


def load_tf_static_from_npy(tf_static_path: str) -> list:
    """Load TF static from numpy file."""
    import numpy as np

    tf_static_data = np.load(tf_static_path, allow_pickle=True)
    return list(tf_static_data)


class MOLAEvaluator(Node):
    """ROS2 node that evaluates perturbations by running MOLA and measuring ATE."""

    def __init__(
        self,
        perturbation_generator: PerturbationGenerator,
        ground_truth_trajectory: np.ndarray,
        point_cloud_sequence: list,
        timestamps: np.ndarray,
        tf_sequence: list,
        tf_static: list,
        mola_binary_path: str,
        mola_config_path: str,
        lidar_topic: str = "/mola_nsga3/lidar",
        odom_topic: str = "/lidar_odometry/pose",
    ):
        super().__init__("mola_evaluator")

        self.perturbation_generator = perturbation_generator
        self.ground_truth_trajectory = ground_truth_trajectory
        self.point_cloud_sequence = point_cloud_sequence
        self.timestamps = timestamps
        self.tf_sequence = tf_sequence
        self.tf_static = tf_static
        self.mola_binary_path = mola_binary_path
        self.mola_config_path = mola_config_path
        self.lidar_topic = lidar_topic
        self.odom_topic = odom_topic

        self.pc_publisher = self.create_publisher(PointCloud2, lidar_topic, 10)
        self.tf_publisher = self.create_publisher(TFMessage, "/tf", 10)

        # TF static requires TRANSIENT_LOCAL durability for late subscribers
        from rclpy.qos import DurabilityPolicy, QoSProfile

        tf_static_qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_static_publisher = self.create_publisher(TFMessage, "/tf_static", tf_static_qos)

        self.collected_trajectory = []
        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, 10
        )

        self.evaluation_count = 0
        self.mola_process = None
        self.original_clouds = point_cloud_sequence

        # Precompute TF indices for each cloud timestamp for faster lookup
        self._precompute_tf_indices()

        self.get_logger().info(
            f"Evaluator ready - max shift: {perturbation_generator.max_point_shift * 100:.1f} cm"
        )
        self.get_logger().info(f"  Loaded {len(tf_sequence)} TF, {len(tf_static)} TF static")

    def _precompute_tf_indices(self):
        """Precompute which TF messages to publish for each cloud timestamp."""
        tf_timestamps = np.array([t["timestamp"] for t in self.tf_sequence])
        self.tf_indices_per_cloud = []

        for i, cloud_ts in enumerate(self.timestamps):
            if i == 0:
                # First cloud: get all TF up to this timestamp
                mask = tf_timestamps <= cloud_ts
            else:
                # Get TF between previous and current cloud
                prev_ts = self.timestamps[i - 1]
                mask = (tf_timestamps > prev_ts) & (tf_timestamps <= cloud_ts)

            indices = np.where(mask)[0].tolist()
            self.tf_indices_per_cloud.append(indices)

    def _odom_callback(self, msg):
        """Store MOLA's estimated poses."""
        pos = msg.pose.pose.position
        self.collected_trajectory.append([pos.x, pos.y, pos.z])
        if len(self.collected_trajectory) == 1:
            self.get_logger().info(
                f"First odometry received! pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})"
            )

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
        header.stamp = Time(
            sec=int(timestamp_ns // 1_000_000_000), nanosec=int(timestamp_ns % 1_000_000_000)
        )
        header.frame_id = "base_link"

        msg = point_cloud2.create_cloud(header, fields, point_cloud)
        return msg

    def _publish_tf_static(self):
        """Publish all static TF transforms."""
        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import Quaternion, Vector3

        tf_msg = TFMessage()
        for tf_data in self.tf_static:
            t = TransformStamped()
            t.header.stamp = Time(
                sec=int(tf_data["timestamp"] // 1_000_000_000),
                nanosec=int(tf_data["timestamp"] % 1_000_000_000),
            )
            t.header.frame_id = tf_data["frame_id"]
            t.child_frame_id = tf_data["child_frame_id"]
            t.transform.translation = Vector3(
                x=tf_data["translation"][0],
                y=tf_data["translation"][1],
                z=tf_data["translation"][2],
            )
            t.transform.rotation = Quaternion(
                x=tf_data["rotation"][0],
                y=tf_data["rotation"][1],
                z=tf_data["rotation"][2],
                w=tf_data["rotation"][3],
            )
            tf_msg.transforms.append(t)

        self.tf_static_publisher.publish(tf_msg)

    def _publish_tf_for_cloud(self, cloud_index: int):
        """Publish TF transforms associated with a specific cloud frame."""
        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import Quaternion, Vector3

        indices = self.tf_indices_per_cloud[cloud_index]
        if not indices:
            return

        tf_msg = TFMessage()
        for idx in indices:
            tf_data = self.tf_sequence[idx]
            t = TransformStamped()
            t.header.stamp = Time(
                sec=int(tf_data["timestamp"] // 1_000_000_000),
                nanosec=int(tf_data["timestamp"] % 1_000_000_000),
            )
            t.header.frame_id = tf_data["frame_id"]
            t.child_frame_id = tf_data["child_frame_id"]
            t.transform.translation = Vector3(
                x=tf_data["translation"][0],
                y=tf_data["translation"][1],
                z=tf_data["translation"][2],
            )
            t.transform.rotation = Quaternion(
                x=tf_data["rotation"][0],
                y=tf_data["rotation"][1],
                z=tf_data["rotation"][2],
                w=tf_data["rotation"][3],
            )
            tf_msg.transforms.append(t)

        self.tf_publisher.publish(tf_msg)

    def _publish_tf_static_current(self):
        """Publish TF static with current timestamp (identity transform base_link->lidar)."""
        from geometry_msgs.msg import Quaternion, Vector3

        tf_msg = TFMessage()
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "lidar"
        t.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        tf_msg.transforms.append(t)
        self.tf_static_publisher.publish(tf_msg)

    def _publish_tf_current(self, current_time):
        """Publish TF with current timestamp."""
        from geometry_msgs.msg import Quaternion, Vector3

        tf_msg = TFMessage()
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "lidar"
        t.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        tf_msg.transforms.append(t)
        self.tf_publisher.publish(tf_msg)

    def _voxel_downsample(self, points, voxel_size=0.15):
        """Voxel grid downsampling - keeps one point per voxel.

        More uniform than random sampling, preserves spatial structure.
        """
        # Handle empty input
        if len(points) == 0:
            return points

        # Compute voxel indices
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # Create unique key for each voxel
        # Shift to handle negative coordinates
        min_idx = voxel_indices.min(axis=0)
        voxel_indices = voxel_indices - min_idx

        # Hash voxels to single value
        max_idx = voxel_indices.max(axis=0) + 1
        voxel_hash = (
            voxel_indices[:, 0] * max_idx[1] * max_idx[2]
            + voxel_indices[:, 1] * max_idx[2]
            + voxel_indices[:, 2]
        )

        # Get unique voxels (first point in each)
        _, unique_indices = np.unique(voxel_hash, return_index=True)

        return points[unique_indices]

    def _create_pointcloud2_msg_current(self, point_cloud, current_time, max_points=25000):
        """Convert numpy array to PointCloud2 with current timestamp.

        Uses intelligent voxel-based downsampling for uniform spatial coverage.

        Args:
            point_cloud: Input point cloud (N, 3 or 4)
            current_time: ROS time to use for timestamp
            max_points: Target number of points (default 25k for fast MOLA processing)
        """
        from std_msgs.msg import Header

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Intelligent downsampling using voxel grid
        if len(point_cloud) > max_points:
            # Start with coarse voxel size, refine if needed
            voxel_size = 0.10  # 10cm voxels initially

            for _ in range(5):  # Max 5 iterations to find right voxel size
                downsampled = self._voxel_downsample(point_cloud, voxel_size)

                if len(downsampled) <= max_points:
                    point_cloud = downsampled
                    break
                elif len(downsampled) > max_points * 1.5:
                    # Too many points, increase voxel size
                    voxel_size *= 1.3
                else:
                    # Close enough, use random subsampling for final adjustment
                    if len(downsampled) > max_points:
                        indices = np.random.choice(len(downsampled), max_points, replace=False)
                        point_cloud = downsampled[indices]
                    else:
                        point_cloud = downsampled
                    break

        # Ensure point cloud has intensity column
        if point_cloud.shape[1] == 3:
            intensity = np.ones((len(point_cloud), 1), dtype=np.float32) * 100
            point_cloud = np.hstack([point_cloud, intensity])

        point_cloud = point_cloud.astype(np.float32)

        header = Header()
        header.stamp = current_time.to_msg()
        header.frame_id = "base_link"

        msg = point_cloud2.create_cloud(header, fields, point_cloud)
        return msg

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

        # Force MOLA to publish odometry more frequently
        # Lower thresholds = publish on smaller motion
        env["MOLA_MIN_XYZ_BETWEEN_MAP_UPDATES"] = "0.001"  # 1mm (default is ~10cm)
        env["MOLA_MIN_ROT_BETWEEN_MAP_UPDATES"] = "0.1"  # 0.1 deg (default is ~15deg)
        env["MOLA_MINIMUM_ICP_QUALITY"] = "0.05"  # Lower quality threshold

        cmd = [self.mola_binary_path]
        if self.mola_config_path:
            cmd.append(self.mola_config_path)

        self.get_logger().info(f"Starting MOLA: {' '.join(cmd)}")
        self.get_logger().info(f"  MOLA_LIDAR_TOPIC={self.lidar_topic}")

        self.mola_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            env=env,
            preexec_fn=os.setsid,
        )

        # Check if MOLA started successfully
        time.sleep(1.0)
        if self.mola_process.poll() is not None:
            # MOLA exited immediately - read output for error
            output, _ = self.mola_process.communicate()
            self.get_logger().error(f"MOLA failed to start! Output: {output.decode()[:500]}")
        else:
            self.get_logger().info("MOLA process started (pid={})".format(self.mola_process.pid))

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

    def _downsample_sequence(self, sequence, max_points=20000):
        """Pre-downsample all clouds in sequence for faster MOLA processing."""
        downsampled = []
        for cloud in sequence:
            if len(cloud) > max_points:
                # Use voxel downsampling
                ds = self._voxel_downsample(cloud, voxel_size=0.15)
                if len(ds) > max_points:
                    # Still too many, random subsample
                    indices = np.random.choice(len(ds), max_points, replace=False)
                    ds = ds[indices]
                downsampled.append(ds)
            else:
                downsampled.append(cloud)
        return downsampled

    def _run_synchronized_playback(self, perturbed_sequence, min_points_expected):
        """Run synchronized playback of TF and perturbed point clouds.

        Uses CURRENT timestamps (not original bag timestamps) because MOLA
        ignores messages with timestamps too far from the ROS clock.

        Args:
            perturbed_sequence: List of perturbed point clouds
            min_points_expected: Minimum trajectory points to collect

        Returns:
            Collected trajectory as numpy array
        """
        self.collected_trajectory = []

        # Pre-downsample all clouds for faster processing
        perturbed_sequence = self._downsample_sequence(perturbed_sequence, max_points=20000)

        # Log cloud size after downsampling
        cloud_sizes = [len(c) for c in perturbed_sequence]
        self.get_logger().info(
            f"Cloud sizes (after downsampling): min={min(cloud_sizes)}, "
            f"max={max(cloud_sizes)}, avg={sum(cloud_sizes) // len(cloud_sizes)}"
        )

        # Start MOLA
        self._start_mola()
        time.sleep(3.0)  # Let MOLA fully initialize

        # Publish TF static with current timestamp - do it multiple times for reliability
        for _ in range(3):
            self._publish_tf_static_current()
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        # Publish clouds with current timestamps
        for i, cloud in enumerate(perturbed_sequence):
            # Get current ROS time
            current_time = self.get_clock().now()

            # Publish TF with current time
            self._publish_tf_current(current_time)

            # Publish the perturbed cloud with current timestamp (no downsampling needed, already done)
            msg = self._create_pointcloud2_msg_current(cloud, current_time, max_points=25000)
            self.pc_publisher.publish(msg)

            # Spin to receive MOLA's response
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.05)

            # Wait between frames
            if i < len(perturbed_sequence) - 1:
                time.sleep(0.3)
                rclpy.spin_once(self, timeout_sec=0.1)

        # Wait for final trajectory points
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)

        self._stop_mola()

        return np.array(self.collected_trajectory) if self.collected_trajectory else np.array([])

    def _retry_evaluation(self, perturbed_sequence, min_valid_points):
        """Retry MOLA evaluation after failure."""
        subprocess.run(["pkill", "-9", "-f", "mola"], capture_output=True, timeout=2.0)
        time.sleep(2.0)

        return self._run_synchronized_playback(perturbed_sequence, min_valid_points)

    def evaluate(self, genome: np.ndarray) -> tuple:
        """Run MOLA with perturbed clouds and return (neg_ate, chamfer_distance)."""
        self.evaluation_count += 1
        self.get_logger().info(f"\n{'=' * 60}")
        self.get_logger().info(f"Evaluation #{self.evaluation_count}")
        self.get_logger().info(f"{'=' * 60}")

        # Reset temporal state for new evaluation (important for drift attack)
        self.perturbation_generator.reset_temporal_state()

        params = self.perturbation_generator.encode_perturbation(genome)

        self.get_logger().info(f"  Noise intensity: {params['noise_intensity'] * 100:.2f} cm")
        self.get_logger().info(f"  Curvature targeting: {params['curvature_strength']:.2f}")
        self.get_logger().info(f"  Dropout rate: {params['dropout_rate'] * 100:.1f}%")
        self.get_logger().info(f"  Ghost ratio: {params['ghost_ratio'] * 100:.1f}%")
        self.get_logger().info(f"  Geometric distortion: {params['geometric_distortion']:.3f}")
        # Log new attack parameters
        self.get_logger().info(
            f"  Edge attack: {params.get('edge_attack_strength', 0):.2f}, "
            f"Temporal drift: {params.get('temporal_drift_strength', 0):.2f}"
        )
        self.get_logger().info(
            f"  Scanline: {params.get('scanline_strength', 0):.2f}, "
            f"Strategic ghost: {params.get('strategic_ghost', 0):.2f}"
        )

        perturbed_sequence, avg_chamfer = self._generate_perturbed_sequence(params)
        # avg_chamfer is in m² (sum of squared distances), convert to cm
        avg_chamfer_cm = np.sqrt(avg_chamfer * 10000) if avg_chamfer > 0 else 0
        self.get_logger().info(f"  Avg Chamfer distance: {avg_chamfer_cm:.3f} cm")

        # Run synchronized playback
        min_points_expected = 40
        self.get_logger().info(f"Publishing {len(perturbed_sequence)} clouds with TF...")

        estimated_traj = self._run_synchronized_playback(perturbed_sequence, min_points_expected)
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
            self.get_logger().error(
                f"Failed after {max_retries} retries - returning invalid fitness"
            )
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
    parser = argparse.ArgumentParser(
        description="NSGA-III optimization for adversarial perturbations"
    )
    parser.add_argument("--gt-traj", type=str, default="maps/ground_truth_interpolated.npy")
    parser.add_argument("--frames", type=str, default="data/frame_sequence.npy")
    parser.add_argument("--mola-binary", type=str, default="/opt/ros/jazzy/bin/mola-cli")
    parser.add_argument(
        "--mola-config",
        type=str,
        default="/opt/ros/jazzy/share/mola_lidar_odometry/mola-cli-launchs/lidar_odometry_ros2.yaml",
    )
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
        "--noise-std",
        type=float,
        default=0.015,
        help="Gaussian noise std in meters (default: 1.5cm)",
    )
    parser.add_argument(
        "--max-dropout", type=float, default=0.03, help="Max dropout rate (default: 0.03)"
    )
    parser.add_argument(
        "--max-ghost", type=float, default=0.02, help="Max ghost points ratio (default: 0.02)"
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
            suffix = name[len(base_name) :]
            if suffix.isdigit():
                existing_numbers.append(int(suffix))

    return max(existing_numbers, default=0) + 1


def _compute_pareto_front(fitness: np.ndarray, genomes: np.ndarray):
    """
    Compute Pareto front from fitness values and corresponding genomes.

    For adversarial attacks:
    - Objective 1: -ATE (minimize, i.e., maximize ATE)
    - Objective 2: Perturbation (minimize)

    Returns:
        Tuple of (pareto_front_fitness, pareto_set_genomes)
    """
    if len(fitness) == 0:
        return np.array([]), np.array([])

    n_points = len(fitness)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            # Point j dominates point i if:
            # j is better or equal in all objectives AND strictly better in at least one
            # For minimization: lower is better
            if (fitness[j] <= fitness[i]).all() and (fitness[j] < fitness[i]).any():
                is_pareto[i] = False
                break

    pareto_front = fitness[is_pareto]
    pareto_set = genomes[is_pareto]

    # Sort by perturbation (second objective)
    sort_idx = np.argsort(pareto_front[:, 1])
    return pareto_front[sort_idx], pareto_set[sort_idx]


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
    """Load ground truth trajectory, point clouds, timestamps, and TF data."""
    print("Loading data...")

    clouds = load_point_clouds_from_npy(args.frames)
    if clouds is None:
        return None, None, None, None, None

    timestamps = load_timestamps_from_npy(args.frames.replace(".npy", ".timestamps.npy"))
    if timestamps is None:
        print("Failed to load timestamps")
        return None, None, None, None, None

    # Load GT with interpolation to match point cloud frame count
    gt_traj = load_trajectory_from_tum(
        args.gt_traj, interpolate_to_frames=len(clouds), pc_timestamps=timestamps
    )
    if gt_traj is None:
        print("Failed to load ground truth")
        return None, None, None, None, None

    # Load TF data from numpy files (extracted from bag)
    tf_path = args.frames.replace("frame_sequence.npy", "tf_sequence.npy")
    tf_static_path = args.frames.replace("frame_sequence.npy", "tf_static.npy")

    tf_sequence = load_tf_from_npy(tf_path)
    tf_static = load_tf_static_from_npy(tf_static_path)

    print(f"  Ground truth: {len(gt_traj)} poses")
    print(f"  Point clouds: {len(clouds)} frames")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  TF sequence: {len(tf_sequence)} transforms")
    print(f"  TF static: {len(tf_static)} transforms")
    return gt_traj, clouds, timestamps, tf_sequence, tf_static


def _create_evaluator(args, gt_traj, clouds, timestamps, tf_sequence, tf_static):
    """Create perturbation generator and evaluator."""
    generator = PerturbationGenerator(
        max_point_shift=args.max_point_shift,
        noise_std=args.noise_std,
        target_high_curvature=True,
        curvature_percentile=90.0,
        max_dropout_rate=args.max_dropout,
        max_ghost_points_ratio=args.max_ghost,
    )

    evaluator = MOLAEvaluator(
        perturbation_generator=generator,
        ground_truth_trajectory=gt_traj,
        point_cloud_sequence=clouds,
        timestamps=timestamps,
        tf_sequence=tf_sequence,
        tf_static=tf_static,
        mola_binary_path=args.mola_binary,
        mola_config_path=args.mola_config,
        lidar_topic="/mola_nsga3/lidar",
        odom_topic="/lidar_odometry/pose",
    )
    return generator, evaluator


def _run_optimization(args, problem, algorithm, history_callback):
    """Run the NSGA-III optimization."""
    from pymoo.optimize import minimize

    print("\nStarting NSGA-III optimization")
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
    print(" NSGA-III ADVERSARIAL PERTURBATION OPTIMIZATION")
    print("=" * 80)
    print("\nAlgorithm: NSGA-III (Reference-point based)")
    print("  - Better Pareto front distribution via reference directions")
    print("  - Das-Dennis reference directions (12 partitions)")
    print("\nGenome: 17 parameters (ENHANCED)")
    print("  - Basic: Noise, dropout, ghost, cluster perturbations")
    print("  - Geometric distortion (ICP attack)")
    print("  - Edge attack (SLACK-inspired)")
    print("  - Temporal drift (accumulating bias)")
    print("  - Scanline perturbation (ASP-inspired)")
    print("  - Strategic ghost placement")
    print("\nPerturbation settings:")
    print(f"  Max point shift: {args.max_point_shift * 100:.1f} cm")
    print(f"  Noise std: {args.noise_std * 100:.1f} cm")
    print(f"  Max dropout: {args.max_dropout * 100:.0f}%")
    print("\nFitness evaluation:")
    print("  - ATE: Umeyama alignment (R+t, no scale) + RMSE")
    print("  - Perturbation: Chamfer distance + structural penalty")
    print()


def _setup_optimizer(args, evaluator, generator):
    """Set up NSGA-III optimizer components."""
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.util.ref_dirs import get_reference_directions

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
            self.all_genomes = []
            self.valid_fitness = []
            self.valid_genomes = []

        def notify(self, algorithm):
            for ind in algorithm.pop:
                if ind.F is not None and ind.X is not None:
                    self.all_fitness.append(ind.F.copy())
                    self.all_genomes.append(ind.X.copy())
                    if -ind.F[0] < 10.0:
                        self.valid_fitness.append(ind.F.copy())
                        self.valid_genomes.append(ind.X.copy())

    problem = MOLAPerturbationProblem(evaluator.get_fitness_function(), generator.get_genome_size())

    # NSGA-III requires reference directions for diversity
    # For 2 objectives, use Das-Dennis with ~12-15 divisions for good Pareto spread
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=args.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )
    history_callback = SaveHistoryCallback()
    return problem, algorithm, history_callback


def _save_partial_results(history_callback, numbered_output):
    """Save partial results after interruption."""
    n_evals = len(history_callback.all_fitness) if history_callback.all_fitness else 0
    n_valid = len(history_callback.valid_fitness) if history_callback.valid_fitness else 0

    if n_evals < 5:
        print(f"\nNot enough data to save ({n_evals} evaluations, need >= 5)")
        print("No partial results saved.")
        return

    all_fitness = np.array(history_callback.all_fitness)
    all_genomes = np.array(history_callback.all_genomes)
    valid_fitness = np.array(history_callback.valid_fitness) if n_valid > 0 else np.array([])
    valid_genomes = np.array(history_callback.valid_genomes) if n_valid > 0 else np.array([])

    # Compute Pareto front from valid points
    if n_valid >= 2:
        pareto_front, pareto_set = _compute_pareto_front(valid_fitness, valid_genomes)
    else:
        pareto_front = valid_fitness
        pareto_set = valid_genomes

    # Save all files
    np.save(numbered_output.with_suffix(".all_points.npy"), all_fitness)
    np.save(numbered_output.with_suffix(".valid_points.npy"), valid_fitness)
    np.save(numbered_output.with_suffix(".all_genomes.npy"), all_genomes)
    np.save(numbered_output.with_suffix(".valid_genomes.npy"), valid_genomes)

    if len(pareto_front) > 0:
        np.save(numbered_output.with_suffix(".pareto_front.npy"), pareto_front)
        np.save(numbered_output.with_suffix(".pareto_set.npy"), pareto_set)
        # Save best genome (highest ATE/pert ratio)
        ratios = [-f[0] / max(f[1], 0.001) for f in pareto_front]
        best_idx = np.argmax(ratios)
        np.save(numbered_output, pareto_set[best_idx])

    print(f"\nPartial results saved ({n_evals} evaluations):")
    print(f"  - All points: {numbered_output.with_suffix('.all_points.npy')}")
    print(f"  - All genomes: {numbered_output.with_suffix('.all_genomes.npy')}")
    print(f"  - Valid points: {n_valid} (ATE < 10m)")
    if len(pareto_front) > 0:
        print(f"  - Pareto front: {len(pareto_front)} solutions")
        print(f"  - Best genome: {numbered_output}")


def _cleanup_rclpy(evaluator):
    """Clean up ROS2 resources."""
    evaluator.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


def main():
    """Main function for NSGA-III optimization."""
    args = _parse_args()
    _print_header(args)

    # Generate numbered output filename
    base_output_path = Path(args.output)
    run_number = _get_next_run_number(base_output_path)
    numbered_output = (
        base_output_path.parent / f"{base_output_path.stem}{run_number}{base_output_path.suffix}"
    )

    # Create results directory if it doesn't exist
    numbered_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {numbered_output}\n")

    gt_traj, clouds, timestamps, tf_sequence, tf_static = _load_data(args)
    if gt_traj is None:
        return 1

    rclpy.init()
    generator, evaluator = _create_evaluator(
        args, gt_traj, clouds, timestamps, tf_sequence, tf_static
    )
    problem, algorithm, history_callback = _setup_optimizer(args, evaluator, generator)

    try:
        result, elapsed = _run_optimization(args, problem, algorithm, history_callback)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("INTERRUPTED BY USER")
        print("=" * 60)
        evaluator._stop_mola()
        _save_partial_results(history_callback, numbered_output)
        _cleanup_rclpy(evaluator)
        return 1

    _print_results(result, elapsed, result.F, result.X, history_callback, numbered_output)
    _cleanup_rclpy(evaluator)
    return 0


if __name__ == "__main__":
    sys.exit(main())
