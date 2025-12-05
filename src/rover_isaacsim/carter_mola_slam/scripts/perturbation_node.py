#!/usr/bin/env python3
"""
ROS 2 Node that applies adversarial perturbations to LiDAR point clouds in real-time.

Subscribes to: /carter/lidar_with_intensity
Publishes to: /carter/lidar_perturbed

This allows you to see the effect of perturbations on MOLA SLAM in real-time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.perturbations.perturbation_generator import (  # noqa: E402
    PerturbationGenerator,
)


class PerturbationNode(Node):
    """ROS 2 node that applies perturbations to point clouds."""

    def __init__(self, args=None):
        super().__init__("perturbation_node")

        # Declare parameters with CLI argument overrides
        self.declare_parameter("max_point_shift", args.max_point_shift if args else 0.05)
        self.declare_parameter("noise_std", args.noise_std if args else 0.02)
        self.declare_parameter("max_dropout_rate", args.dropout_rate if args else 0.15)
        self.declare_parameter("perturbation_level", args.perturbation_level if args else 0.5)

        # Get parameters
        max_point_shift = self.get_parameter("max_point_shift").value
        noise_std = self.get_parameter("noise_std").value
        max_dropout_rate = self.get_parameter("max_dropout_rate").value
        perturbation_level = self.get_parameter("perturbation_level").value

        # Initialize perturbation generator
        self.generator = PerturbationGenerator(
            max_point_shift=max_point_shift,
            noise_std=noise_std,
            max_dropout_rate=max_dropout_rate,
        )

        # Load or generate perturbation genome
        if args and args.genome_file:
            # Load pre-optimized genome from NSGA-III
            self.genome = np.load(args.genome_file)
            self.get_logger().info(f"Loaded optimized genome from: {args.genome_file}")
            self.params = self.generator.encode_perturbation(self.genome)
        else:
            # Generate random perturbation genome scaled by perturbation_level
            self.genome = self.generator.random_genome() * perturbation_level
            self.params = self.generator.encode_perturbation(self.genome)
            self.get_logger().info("Using randomly generated genome")

        # Log configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("Perturbation Node Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Perturbation Level: {perturbation_level:.1%}")
        self.get_logger().info(f'Noise Intensity: {self.params["noise_intensity"]:.4f}')
        self.get_logger().info(f'Dropout Rate: {self.params["dropout_rate"]:.1%}')
        self.get_logger().info(f'Ghost Ratio: {self.params["ghost_ratio"]:.1%}')
        self.get_logger().info(f'Cluster Strength: {self.params["cluster_strength"]:.2f}')
        self.get_logger().info("=" * 60)
        self.get_logger().info("")

        # Subscriber to original point clouds
        self.subscription = self.create_subscription(
            PointCloud2, "/carter/lidar_with_intensity", self.pointcloud_callback, 10
        )

        # Publisher for perturbed point clouds
        self.publisher = self.create_publisher(PointCloud2, "/carter/lidar_perturbed", 10)

        self.count = 0

    def pointcloud_callback(self, msg):
        """Process incoming point cloud and apply perturbation."""
        # Read point cloud
        points = []
        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        ):
            points.append([p[0], p[1], p[2], p[3]])

        if len(points) == 0:
            self.get_logger().warn("Received empty point cloud")
            return

        cloud = np.array(points)

        # Apply perturbation with fixed seed for reproducibility
        perturbed = self.generator.apply_perturbation(cloud, self.params, seed=42)

        # Create new PointCloud2 message
        new_msg = point_cloud2.create_cloud(msg.header, msg.fields, perturbed.tolist())

        # Publish
        self.publisher.publish(new_msg)

        self.count += 1
        if self.count % 20 == 0:
            dropped = len(cloud) - len(perturbed)
            self.get_logger().info(
                f"Processed {self.count} clouds: {len(cloud)} → {len(perturbed)} points "
                f"(dropped {dropped})"
            )


def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="ROS 2 node that applies adversarial perturbations to LiDAR"
    )
    parser.add_argument(
        "--max-point-shift",
        type=float,
        default=0.05,
        help="Maximum per-point shift in meters (default: 0.05)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise standard deviation in meters (default: 0.02)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.15,
        help="Maximum point dropout rate (default: 0.15)",
    )
    parser.add_argument(
        "--perturbation-level",
        type=float,
        default=0.5,
        help="Perturbation level scaling factor 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--genome-file",
        type=str,
        help="Path to pre-optimized genome file (.npy) from NSGA-III optimization",
    )

    cli_args = parser.parse_args()

    rclpy.init(args=args)
    node = PerturbationNode(cli_args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Total clouds processed: {node.count}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
