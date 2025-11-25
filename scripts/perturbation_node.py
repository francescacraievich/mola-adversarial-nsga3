#!/usr/bin/env python3
"""
ROS 2 Node that applies adversarial perturbations to LiDAR point clouds in real-time.

Subscribes to: /carter/lidar_with_intensity
Publishes to: /carter/lidar_perturbed

This allows you to see the effect of perturbations on MOLA SLAM in real-time.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from src.perturbations.perturbation_generator import PerturbationGenerator


class PerturbationNode(Node):
    """ROS 2 node that applies perturbations to point clouds."""

    def __init__(self):
        super().__init__("perturbation_node")

        # Declare parameters
        self.declare_parameter("max_translation", 0.5)
        self.declare_parameter("max_rotation", 0.1)
        self.declare_parameter("max_intensity_change", 50.0)
        self.declare_parameter("point_dropout_rate", 0.1)
        self.declare_parameter("perturbation_level", 0.5)  # 0.0 to 1.0

        # Get parameters
        max_translation = self.get_parameter("max_translation").value
        max_rotation = self.get_parameter("max_rotation").value
        max_intensity_change = self.get_parameter("max_intensity_change").value
        point_dropout_rate = self.get_parameter("point_dropout_rate").value
        perturbation_level = self.get_parameter("perturbation_level").value

        # Initialize perturbation generator
        self.generator = PerturbationGenerator(
            max_translation=max_translation,
            max_rotation=max_rotation,
            max_intensity_change=max_intensity_change,
            point_dropout_rate=point_dropout_rate,
        )

        # Generate perturbation genome scaled by perturbation_level
        self.genome = self.generator.random_genome() * perturbation_level
        self.params = self.generator.encode_perturbation(self.genome)

        # Log configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("🔧 Perturbation Node Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Perturbation Level: {perturbation_level:.1%}")
        self.get_logger().info(
            f'Translation: [{self.params["translation"][0]:.3f}, '
            f'{self.params["translation"][1]:.3f}, '
            f'{self.params["translation"][2]:.3f}]'
        )
        self.get_logger().info(
            f'Rotation:    [{self.params["rotation"][0]:.3f}, '
            f'{self.params["rotation"][1]:.3f}, '
            f'{self.params["rotation"][2]:.3f}]'
        )
        self.get_logger().info(f'Intensity:   {self.params["intensity_scale"]:+.2f}')
        self.get_logger().info(f'Dropout:     {self.params["dropout_rate"]:.1%}')
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
    rclpy.init(args=args)
    node = PerturbationNode()

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
