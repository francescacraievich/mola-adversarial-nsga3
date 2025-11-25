#!/usr/bin/env python3
"""
Compare and visualize trajectories (clean vs perturbed).

Loads trajectory files and creates overlay visualization.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from src.evaluation.metrics import compute_localization_error


def load_trajectory(filepath):
    """Load trajectory from text file."""
    data = np.loadtxt(filepath)
    timestamps = data[:, 0]
    trajectory = data[:, 1:4]
    return timestamps, trajectory


def plot_trajectory_comparison(trajectories, labels, output_file=None):
    """Plot multiple trajectories overlaid."""
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: 2D top-down view (XY)
    ax1 = fig.add_subplot(131)
    colors = ["blue", "red", "green", "orange", "purple"]

    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        color = colors[idx % len(colors)]
        ax1.plot(
            traj[:, 0],
            traj[:, 1],
            "-o",
            label=label,
            color=color,
            markersize=3,
            linewidth=2,
            alpha=0.7,
        )

    ax1.set_xlabel("X (m)", fontsize=12)
    ax1.set_ylabel("Y (m)", fontsize=12)
    ax1.set_title("Top-Down View (XY)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Plot 2: Side view (XZ)
    ax2 = fig.add_subplot(132)
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        color = colors[idx % len(colors)]
        ax2.plot(
            traj[:, 0],
            traj[:, 2],
            "-o",
            label=label,
            color=color,
            markersize=3,
            linewidth=2,
            alpha=0.7,
        )

    ax2.set_xlabel("X (m)", fontsize=12)
    ax2.set_ylabel("Z (m)", fontsize=12)
    ax2.set_title("Side View (XZ)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: 3D view
    ax3 = fig.add_subplot(133, projection="3d")
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        color = colors[idx % len(colors)]
        ax3.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            "-o",
            label=label,
            color=color,
            markersize=2,
            linewidth=2,
            alpha=0.7,
        )

    ax3.set_xlabel("X (m)", fontsize=10)
    ax3.set_ylabel("Y (m)", fontsize=10)
    ax3.set_zlabel("Z (m)", fontsize=10)
    ax3.set_title("3D View", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {output_file}")

    plt.show()


def plot_error_over_time(trajectories, labels, ground_truth_idx=0, output_file=None):
    """Plot localization error over time."""
    if len(trajectories) < 2:
        print("Need at least 2 trajectories for error comparison")
        return

    gt_traj = trajectories[ground_truth_idx]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    colors = ["red", "green", "orange", "purple"]

    # Plot 1: Absolute error over time
    ax1 = axes[0]
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        if idx == ground_truth_idx:
            continue

        # Compute error at each pose
        min_len = min(len(gt_traj), len(traj))
        errors = np.linalg.norm(gt_traj[:min_len] - traj[:min_len], axis=1)

        color = colors[(idx - 1) % len(colors)]
        ax1.plot(errors, label=label, color=color, linewidth=2)

    ax1.set_xlabel("Pose Index", fontsize=12)
    ax1.set_ylabel("Position Error (m)", fontsize=12)
    ax1.set_title("Localization Error Over Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative error
    ax2 = axes[1]
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        if idx == ground_truth_idx:
            continue

        min_len = min(len(gt_traj), len(traj))
        errors = np.linalg.norm(gt_traj[:min_len] - traj[:min_len], axis=1)
        cumulative = np.cumsum(errors)

        color = colors[(idx - 1) % len(colors)]
        ax2.plot(cumulative, label=label, color=color, linewidth=2)

    ax2.set_xlabel("Pose Index", fontsize=12)
    ax2.set_ylabel("Cumulative Error (m)", fontsize=12)
    ax2.set_title("Cumulative Localization Error", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {output_file}")

    plt.show()


def compute_metrics(trajectories, labels, ground_truth_idx=0):
    """Compute and display metrics."""
    gt_traj = trajectories[ground_truth_idx]

    print("\n" + "=" * 70)
    print("📊 TRAJECTORY COMPARISON METRICS")
    print("=" * 70 + "\n")

    print(f"Ground truth: {labels[ground_truth_idx]}")
    print(f"  Length: {len(gt_traj)} poses\n")

    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        if idx == ground_truth_idx:
            continue

        print(f"{label}:")
        print(f"  Length: {len(traj)} poses")

        # Compute errors
        ate = compute_localization_error(gt_traj, traj, method="ate")
        rpe = compute_localization_error(gt_traj, traj, method="rpe")
        final = compute_localization_error(gt_traj, traj, method="final")

        print(f"  ATE (Absolute Trajectory Error): {ate:.4f} m")
        print(f"  RPE (Relative Pose Error): {rpe:.4f} m")
        print(f"  Final Position Error: {final:.4f} m")

        # Drift
        min_len = min(len(gt_traj), len(traj))
        drift = np.linalg.norm(gt_traj[min_len - 1] - traj[min_len - 1])
        print(f"  Final Drift: {drift:.4f} m\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trajectory files")
    parser.add_argument("trajectories", nargs="+", help="Trajectory files to compare")
    parser.add_argument("--labels", nargs="+", help="Labels for trajectories")
    parser.add_argument(
        "--ground-truth", type=int, default=0, help="Index of ground truth trajectory (default: 0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/comparison", help="Output directory for plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("📊 TRAJECTORY COMPARISON")
    print("=" * 70 + "\n")

    # Load trajectories
    trajectories = []
    labels = []

    for idx, filepath in enumerate(args.trajectories):
        print(f"Loading: {filepath}")
        timestamps, traj = load_trajectory(filepath)
        trajectories.append(traj)

        # Generate label
        if args.labels and idx < len(args.labels):
            label = args.labels[idx]
        else:
            label = Path(filepath).stem

        labels.append(label)
        print(f"  ✓ Loaded {len(traj)} poses as '{label}'")

    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    compute_metrics(trajectories, labels, args.ground_truth)

    # Plot comparison
    print("\n📈 Generating plots...")
    plot_trajectory_comparison(
        trajectories, labels, output_file=output_dir / "trajectory_comparison.png"
    )

    plot_error_over_time(
        trajectories, labels, args.ground_truth, output_file=output_dir / "error_over_time.png"
    )

    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}\n")


if __name__ == "__main__":
    main()
