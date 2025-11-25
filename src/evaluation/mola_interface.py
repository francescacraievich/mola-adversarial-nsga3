"""
Interface for communicating with MOLA SLAM system via ROS 2.

Provides:
1. Point cloud publisher (publishes perturbed point clouds)
2. Trajectory subscriber (collects SLAM odometry)
3. Ground truth trajectory loading
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class MOLAInterface(Node):
    """
    ROS 2 interface for MOLA SLAM system.

    Publishes perturbed point clouds and collects SLAM trajectory.
    """

    def __init__(
        self,
        lidar_topic: str = "/carter/lidar_with_intensity",
        odom_topic: str = "/lidar_odometry/odom",
        frame_id: str = "chassis_link",
    ):
        """
        Initialize MOLA interface.

        Args:
            lidar_topic: Topic to publish point clouds to
            odom_topic: Topic to subscribe for SLAM odometry
            frame_id: Frame ID for point clouds
        """
        super().__init__("mola_interface")

        # Configuration
        self.lidar_topic = lidar_topic
        self.odom_topic = odom_topic
        self.frame_id = frame_id

        # Publisher for point clouds
        self.pointcloud_publisher = self.create_publisher(PointCloud2, lidar_topic, 10)

        # Subscriber for SLAM odometry
        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, 10
        )

        # Storage for collected trajectory
        self.trajectory: List[np.ndarray] = []
        self.timestamps: List[float] = []

        self.get_logger().info(f"MOLA Interface initialized")
        self.get_logger().info(f"Publishing point clouds to: {lidar_topic}")
        self.get_logger().info(f"Subscribing to odometry: {odom_topic}")

    def publish_point_cloud(
        self, point_cloud: np.ndarray, timestamp: Optional[float] = None
    ) -> None:
        """
        Publish point cloud to MOLA.

        Args:
            point_cloud: Point cloud array (N, 4) with [x, y, z, intensity]
            timestamp: Optional timestamp (seconds). If None, uses current time.
        """
        if point_cloud.shape[1] != 4:
            raise ValueError(f"Point cloud must have 4 columns, got {point_cloud.shape[1]}")

        # Create PointCloud2 message
        msg = self._create_pointcloud2_msg(point_cloud, timestamp)

        # Publish
        self.pointcloud_publisher.publish(msg)
        self.get_logger().debug(f"Published point cloud with {len(point_cloud)} points")

    def _create_pointcloud2_msg(
        self, point_cloud: np.ndarray, timestamp: Optional[float] = None
    ) -> PointCloud2:
        """
        Create PointCloud2 message from numpy array.

        Args:
            point_cloud: Point cloud array (N, 4) [x, y, z, intensity]
            timestamp: Optional timestamp in seconds

        Returns:
            PointCloud2 message
        """
        # Define fields
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Create header
        header = self.get_clock().now().to_msg()
        if timestamp is not None:
            header.stamp.sec = int(timestamp)
            header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        header.frame_id = self.frame_id

        # Convert to list of [x, y, z, intensity]
        points = point_cloud.tolist()

        # Create message
        msg = point_cloud2.create_cloud(header, fields, points)

        return msg

    def _odom_callback(self, msg: Odometry) -> None:
        """
        Callback for SLAM odometry messages.

        Collects pose and timestamp for trajectory reconstruction.
        """
        # Extract position
        position = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        )

        # Extract timestamp
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Store
        self.trajectory.append(position)
        self.timestamps.append(timestamp)

        self.get_logger().debug(
            f"Received odometry: pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        )

    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get collected SLAM trajectory.

        Returns:
            Tuple of (trajectory, timestamps)
            - trajectory: Array of shape (N, 3) with [x, y, z] positions
            - timestamps: Array of shape (N,) with timestamps
        """
        if len(self.trajectory) == 0:
            return np.array([]), np.array([])

        return np.array(self.trajectory), np.array(self.timestamps)

    def reset_trajectory(self) -> None:
        """Clear collected trajectory data."""
        self.trajectory.clear()
        self.timestamps.clear()
        self.get_logger().info("Trajectory reset")

    def wait_for_trajectory(self, min_points: int = 10, timeout_sec: float = 30.0) -> bool:
        """
        Wait until trajectory has enough points or timeout.

        Args:
            min_points: Minimum number of trajectory points required
            timeout_sec: Maximum time to wait in seconds

        Returns:
            True if enough points collected, False if timeout
        """
        rate = self.create_rate(10)  # 10 Hz
        elapsed = 0.0
        dt = 0.1

        while elapsed < timeout_sec:
            if len(self.trajectory) >= min_points:
                self.get_logger().info(f"Trajectory ready with {len(self.trajectory)} points")
                return True

            rclpy.spin_once(self, timeout_sec=dt)
            elapsed += dt

        self.get_logger().warn(
            f"Timeout waiting for trajectory (got {len(self.trajectory)} points)"
        )
        return False


def load_ground_truth_trajectory(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth trajectory from file.

    Supports formats:
    - .txt, .csv: Space/comma-separated [timestamp, x, y, z, ...]
    - .npy: Numpy array with shape (N, 4+) [timestamp, x, y, z, ...]

    Args:
        filepath: Path to trajectory file

    Returns:
        Tuple of (trajectory, timestamps)
        - trajectory: Array of shape (N, 3) with [x, y, z] positions
        - timestamps: Array of shape (N,) with timestamps
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth trajectory file not found: {filepath}")

    if filepath.suffix == ".npy":
        data = np.load(filepath)
    elif filepath.suffix in [".txt", ".csv"]:
        data = np.loadtxt(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Expected format: [timestamp, x, y, z, ...]
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Invalid trajectory format. Expected (N, 4+), got {data.shape}")

    timestamps = data[:, 0]
    trajectory = data[:, 1:4]  # Extract x, y, z

    return trajectory, timestamps


def save_trajectory(trajectory: np.ndarray, timestamps: np.ndarray, filepath: str) -> None:
    """
    Save trajectory to file.

    Args:
        trajectory: Array of shape (N, 3) with [x, y, z] positions
        timestamps: Array of shape (N,) with timestamps
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Combine into [timestamp, x, y, z]
    data = np.column_stack([timestamps, trajectory])

    if filepath.suffix == ".npy":
        np.save(filepath, data)
    elif filepath.suffix in [".txt", ".csv"]:
        np.savetxt(filepath, data, fmt="%.6f", header="timestamp x y z")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
