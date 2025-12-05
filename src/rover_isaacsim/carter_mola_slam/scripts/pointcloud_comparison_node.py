#!/usr/bin/env python3
"""
ROS 2 Node that publishes original and perturbed point clouds side-by-side for RViz visualization.

Publishes:
- /comparison/original (blue points)
- /comparison/perturbed (red points)
- /comparison/difference (green points showing displacement vectors)

This allows visualizing the attack effect in real-time in RViz.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import PerturbationGenerator


def create_colored_pointcloud2(points, header, rgb):
    """
    Create a PointCloud2 message with colored points.

    Args:
        points: Nx3 or Nx4 numpy array of points
        header: ROS Header
        rgb: Tuple (r, g, b) with values 0-255
    """
    # Pack RGB into float
    r, g, b = rgb
    rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

    # Create point data with XYZRGB
    cloud_data = []
    for p in points[:, :3]:
        cloud_data.append(struct.pack('ffffi', p[0], p[1], p[2], rgb_packed, 0))

    # Define fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 20  # 5 floats * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.data = b''.join(cloud_data)
    msg.is_dense = True

    return msg


def create_displacement_markers(original, perturbed, header, max_markers=1000):
    """
    Create arrow markers showing point displacement.

    Args:
        original: Original point cloud
        perturbed: Perturbed point cloud
        header: ROS Header
        max_markers: Maximum number of arrows to show
    """
    marker_array = MarkerArray()

    # Sample points if too many
    n_points = min(len(original), len(perturbed))
    if n_points > max_markers:
        indices = np.random.choice(n_points, max_markers, replace=False)
    else:
        indices = np.arange(n_points)

    for i, idx in enumerate(indices):
        if idx >= len(original) or idx >= len(perturbed):
            continue

        orig_pt = original[idx, :3]
        pert_pt = perturbed[idx, :3]
        displacement = np.linalg.norm(pert_pt - orig_pt)

        # Only show arrows for significant displacement (> 1cm)
        if displacement < 0.01:
            continue

        marker = Marker()
        marker.header = header
        marker.ns = "displacement"
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow from original to perturbed
        marker.points = []
        from geometry_msgs.msg import Point
        start = Point(x=float(orig_pt[0]), y=float(orig_pt[1]), z=float(orig_pt[2]))
        end = Point(x=float(pert_pt[0]), y=float(pert_pt[1]), z=float(pert_pt[2]))
        marker.points.append(start)
        marker.points.append(end)

        # Arrow size
        marker.scale.x = 0.005  # shaft diameter
        marker.scale.y = 0.01   # head diameter
        marker.scale.z = 0.01   # head length

        # Color based on displacement magnitude (green to yellow to red)
        normalized_disp = min(displacement / 0.1, 1.0)  # Normalize to 10cm max
        marker.color.r = float(normalized_disp)
        marker.color.g = float(1.0 - normalized_disp * 0.5)
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 100000000  # 100ms

        marker_array.markers.append(marker)

    return marker_array


class PointCloudComparisonNode(Node):
    """ROS 2 node that shows original vs perturbed point clouds."""

    def __init__(self, args):
        super().__init__("pointcloud_comparison_node")

        # Initialize perturbation generator
        self.generator = PerturbationGenerator(
            max_point_shift=args.max_point_shift,
            noise_std=args.noise_std,
            max_dropout_rate=args.dropout_rate,
        )

        # Load genome
        if args.genome_file:
            self.genome = np.load(args.genome_file)
            self.get_logger().info(f"Loaded genome from: {args.genome_file}")
        else:
            self.genome = self.generator.random_genome() * args.perturbation_level
            self.get_logger().info("Using random genome")

        self.params = self.generator.encode_perturbation(self.genome)

        # Publishers for colored point clouds
        self.pub_original = self.create_publisher(
            PointCloud2, "/comparison/original", 10
        )
        self.pub_perturbed = self.create_publisher(
            PointCloud2, "/comparison/perturbed", 10
        )
        self.pub_displacement = self.create_publisher(
            MarkerArray, "/comparison/displacement_arrows", 10
        )

        # Subscriber to input point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            args.input_topic,
            self.pointcloud_callback,
            10
        )

        self.frame_count = 0
        self.show_arrows = args.show_arrows

        # Log configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("Point Cloud Comparison Node Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Input topic: {args.input_topic}")
        self.get_logger().info(f"Publishing:")
        self.get_logger().info(f"  /comparison/original (BLUE)")
        self.get_logger().info(f"  /comparison/perturbed (RED)")
        if self.show_arrows:
            self.get_logger().info(f"  /comparison/displacement_arrows (GREEN->RED)")
        self.get_logger().info("=" * 60)

    def pointcloud_callback(self, msg):
        """Process incoming point cloud."""
        from sensor_msgs_py import point_cloud2

        # Read points
        points = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append([p[0], p[1], p[2], p[3] if len(p) > 3 else 1.0])

        if len(points) == 0:
            return

        original = np.array(points)

        # Apply perturbation
        perturbed = self.generator.apply_perturbation(original.copy(), self.params, seed=None)

        # Create header
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        # Publish original (BLUE)
        original_msg = create_colored_pointcloud2(original, header, (0, 100, 255))
        self.pub_original.publish(original_msg)

        # Publish perturbed (RED)
        perturbed_msg = create_colored_pointcloud2(perturbed, header, (255, 50, 50))
        self.pub_perturbed.publish(perturbed_msg)

        # Publish displacement arrows
        if self.show_arrows:
            arrows = create_displacement_markers(original, perturbed, header, max_markers=500)
            self.pub_displacement.publish(arrows)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            # Compute statistics
            n_orig = len(original)
            n_pert = len(perturbed)

            # Chamfer-like average displacement (for matching points)
            min_len = min(n_orig, n_pert)
            displacements = np.linalg.norm(original[:min_len, :3] - perturbed[:min_len, :3], axis=1)
            avg_disp = np.mean(displacements) * 100  # cm
            max_disp = np.max(displacements) * 100  # cm

            self.get_logger().info(
                f"Frame {self.frame_count}: {n_orig} orig -> {n_pert} pert | "
                f"Avg disp: {avg_disp:.2f}cm, Max: {max_disp:.2f}cm"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original vs perturbed point clouds in RViz"
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default="/carter/lidar_with_intensity",
        help="Input point cloud topic",
    )
    parser.add_argument(
        "--genome-file",
        type=str,
        help="Path to genome .npy file",
    )
    parser.add_argument(
        "--perturbation-level",
        type=float,
        default=0.5,
        help="Perturbation level if no genome file (0.0-1.0)",
    )
    parser.add_argument(
        "--max-point-shift",
        type=float,
        default=0.05,
        help="Max point shift in meters",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Noise standard deviation in meters",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.15,
        help="Max dropout rate",
    )
    parser.add_argument(
        "--show-arrows",
        action="store_true",
        default=True,
        help="Show displacement arrows",
    )
    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="Disable displacement arrows",
    )

    args = parser.parse_args()
    if args.no_arrows:
        args.show_arrows = False

    rclpy.init()
    node = PointCloudComparisonNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
