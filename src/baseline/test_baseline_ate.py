#!/usr/bin/env python3
"""
Test baseline ATE with zero perturbation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Go to project root

import rclpy  # noqa: E402

from src.optimization.run_nsga2 import MOLAEvaluator  # noqa: E402
from src.perturbations.perturbation_generator import (  # noqa: E402
    PerturbationGenerator,
)
from src.utils.data_loaders import (  # noqa: E402
    load_point_clouds_from_npy,
    load_timestamps_from_npy,
    load_trajectory_from_tum,
)


def main():
    print("\n" + "=" * 60)
    print(" BASELINE ATE TEST (Zero Perturbation)")
    print("=" * 60 + "\n")

    # Load data
    gt_traj = load_trajectory_from_tum("maps/ground_truth_trajectory.tum")
    clouds = load_point_clouds_from_npy("data/frame_sequence.npy")
    timestamps = load_timestamps_from_npy("data/frame_sequence.timestamps.npy")

    if gt_traj is None or clouds is None or timestamps is None:
        print("Failed to load data")
        return 1

    print(f"Ground truth: {len(gt_traj)} poses")
    print(f"Point clouds: {len(clouds)} frames")
    print(f"Timestamps: {len(timestamps)}")

    # Initialize ROS2
    rclpy.init()

    generator = PerturbationGenerator(
        max_point_shift=0.05,
        noise_std=0.02,
        max_dropout_rate=0.15,
    )

    evaluator = MOLAEvaluator(
        perturbation_generator=generator,
        ground_truth_trajectory=gt_traj,
        point_cloud_sequence=clouds,
        timestamps=timestamps,
        mola_binary_path="/opt/ros/jazzy/bin/mola-cli",
        mola_config_path="/opt/ros/jazzy/share/mola_lidar_odometry/mola-cli-launchs/lidar_odometry_ros2.yaml",
        bag_path="bags/lidar_sequence_with_odom",
        lidar_topic="/mola_nsga2/lidar",
        odom_topic="/lidar_odometry/pose",
    )

    # Test with ZERO genome (no perturbation)
    print("\n" + "=" * 60)
    print(" Testing with ZERO perturbation")
    print("=" * 60)

    zero_genome = np.zeros(12)  # 12 parameters for PerturbationGenerator
    neg_ate_zero, pert_mag_zero = evaluator.evaluate(zero_genome)

    print("\n" + "=" * 60)
    print(" BASELINE RESULT")
    print("=" * 60)
    print(f"  ATE (baseline): {-neg_ate_zero:.4f} m")
    print(f"  Perturbation magnitude: {pert_mag_zero:.4f}")
    print("=" * 60 + "\n")

    # Test with small perturbation
    print("\n" + "=" * 60)
    print(" Testing with SMALL perturbation")
    print("=" * 60)

    small_genome = np.array(
        [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2]
    )  # 12 params
    neg_ate_small, pert_mag_small = evaluator.evaluate(small_genome)

    print("\n" + "=" * 60)
    print(" SMALL PERTURBATION RESULT")
    print("=" * 60)
    print(f"  ATE (perturbed): {-neg_ate_small:.4f} m")
    print(f"  Perturbation magnitude: {pert_mag_small:.4f}")
    print("=" * 60 + "\n")

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Baseline ATE:  {-neg_ate_zero:.4f} m")
    print(f"  Perturbed ATE: {-neg_ate_small:.4f} m")
    print(f"  Difference:    {-neg_ate_small - (-neg_ate_zero):.4f} m")
    print("=" * 60 + "\n")

    # Cleanup
    evaluator.destroy_node()
    rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
