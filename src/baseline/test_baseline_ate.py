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
    import argparse

    parser = argparse.ArgumentParser(description="Measure baseline ATE")
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of runs to average (default: 5)"
    )
    args = parser.parse_args()

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

    # Run multiple times with ZERO genome (no perturbation)
    print("\n" + "=" * 60)
    print(f" Running {args.num_runs} evaluations with ZERO perturbation")
    print("=" * 60)

    zero_genome = np.zeros(12)  # 12 parameters for PerturbationGenerator
    baseline_ates = []

    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}...")
        neg_ate_zero, pert_mag_zero = evaluator.evaluate(zero_genome)
        ate = -neg_ate_zero
        baseline_ates.append(ate)
        print(f"  ATE: {ate:.4f} m ({ate * 100:.2f} cm)")

    # Compute statistics
    baseline_ates = np.array(baseline_ates)
    mean_ate = np.mean(baseline_ates)
    std_ate = np.std(baseline_ates)
    min_ate = np.min(baseline_ates)
    max_ate = np.max(baseline_ates)

    print("\n" + "=" * 60)
    print(" BASELINE STATISTICS")
    print("=" * 60)
    print(f"  Runs: {args.num_runs}")
    print(f"  Mean ATE:   {mean_ate:.4f} m ({mean_ate * 100:.2f} cm)")
    print(f"  Std dev:    {std_ate:.4f} m ({std_ate * 100:.2f} cm)")
    print(f"  Min ATE:    {min_ate:.4f} m ({min_ate * 100:.2f} cm)")
    print(f"  Max ATE:    {max_ate:.4f} m ({max_ate * 100:.2f} cm)")
    print(f"  Range:      {(max_ate - min_ate) * 100:.2f} cm")
    print("\nIndividual runs:")
    for i, ate in enumerate(baseline_ates):
        print(f"  Run {i + 1}: {ate:.4f} m ({ate * 100:.2f} cm)")
    print("=" * 60 + "\n")

    # Test with small perturbation (single run for comparison)
    print("\n" + "=" * 60)
    print(" Testing with SMALL perturbation (single run)")
    print("=" * 60)

    small_genome = np.array(
        [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2]
    )  # 12 params
    neg_ate_small, pert_mag_small = evaluator.evaluate(small_genome)
    ate_small = -neg_ate_small

    print("\n" + "=" * 60)
    print(" SMALL PERTURBATION RESULT")
    print("=" * 60)
    print(f"  ATE (perturbed): {ate_small:.4f} m ({ate_small * 100:.2f} cm)")
    print(f"  Perturbation magnitude: {pert_mag_small:.4f}")
    print("=" * 60 + "\n")

    # Summary
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"  Baseline ATE (mean):  {mean_ate:.4f} m ({mean_ate * 100:.2f} cm)")
    print(f"  Baseline ATE (±std):  ± {std_ate:.4f} m (± {std_ate * 100:.2f} cm)")
    print(f"  Perturbed ATE:        {ate_small:.4f} m ({ate_small * 100:.2f} cm)")
    print(f"  Improvement:          {(ate_small - mean_ate):.4f} m ({(ate_small - mean_ate) * 100:.2f} cm)")
    print(f"  Improvement %:        {((ate_small - mean_ate) / mean_ate * 100):.1f}%")
    print("=" * 60 + "\n")

    print("RECOMMENDED BASELINE for NSGA-II:")
    print(f"  Use: {mean_ate:.4f} m ({mean_ate * 100:.2f} cm)")
    print()

    # Cleanup
    evaluator.destroy_node()
    rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
