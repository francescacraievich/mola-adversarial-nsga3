#!/usr/bin/env python3
"""
Compare multiple trajectory recordings to visualize attack effectiveness.

Creates publication-quality figures showing:
1. Side-by-side trajectory comparison
2. Error over time
3. Final position drift
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_trajectory(filepath):
    """Load trajectory from numpy or TUM file."""
    filepath = str(filepath)
    if filepath.endswith(".tum"):
        # TUM format: timestamp tx ty tz qx qy qz qw
        data = np.loadtxt(filepath)
        return data[:, 1:4]  # Return only x, y, z
    else:
        return np.load(filepath)


def plot_trajectory_3d(
    baseline, perturbed, output_path, title="Trajectory Comparison: Blue=Map1 | Red=Map2"
):
    """
    Create single-panel X-Y trajectory comparison plot.

    Args:
        baseline: Baseline trajectory (Nx3)
        perturbed: Perturbed trajectory (Mx3)
        output_path: Output file path
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10), facecolor="white")

    # ============ Single X-Y view (top down) ============
    ax1 = fig.add_subplot(111)
    ax1.plot(baseline[:, 0], baseline[:, 1], "b-", linewidth=2.5, label="Baseline", alpha=0.8)
    ax1.scatter(baseline[:, 0], baseline[:, 1], c="blue", s=20, marker="o", alpha=0.6, zorder=5)
    ax1.plot(perturbed[:, 0], perturbed[:, 1], "r-", linewidth=2.5, label="Perturbed", alpha=0.8)
    ax1.scatter(perturbed[:, 0], perturbed[:, 1], c="red", s=20, marker="s", alpha=0.6, zorder=5)

    # Mark start and end
    ax1.scatter(
        baseline[0, 0],
        baseline[0, 1],
        c="green",
        s=150,
        marker="*",
        zorder=10,
        label="Start",
        edgecolors="darkgreen",
        linewidths=2,
    )
    ax1.scatter(
        baseline[-1, 0],
        baseline[-1, 1],
        c="blue",
        s=100,
        marker="X",
        zorder=10,
        label="Baseline End",
        edgecolors="darkblue",
        linewidths=2,
    )
    ax1.scatter(
        perturbed[-1, 0],
        perturbed[-1, 1],
        c="red",
        s=100,
        marker="X",
        zorder=10,
        label="Perturbed End",
        edgecolors="darkred",
        linewidths=2,
    )

    ax1.set_xlabel("X [m]", fontsize=14)
    ax1.set_ylabel("Y [m]", fontsize=14)
    ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()

    # Print statistics to console
    min_len = min(len(baseline), len(perturbed))
    errors = np.linalg.norm(baseline[:min_len] - perturbed[:min_len], axis=1)
    mean_err = np.mean(errors) * 100
    max_err = np.max(errors) * 100
    final_err = errors[-1] * 100

    print("\n" + "=" * 50)
    print("TRAJECTORY STATISTICS")
    print("=" * 50)
    print(f"Baseline:  {len(baseline)} points")
    print(f"Perturbed: {len(perturbed)} points")
    print(f"\nDrift (first {min_len} points):")
    print(f"  Mean:  {mean_err:.2f} cm")
    print(f"  Max:   {max_err:.2f} cm")
    print(f"  Final: {final_err:.2f} cm")
    print("=" * 50)


def compute_ate(gt, est):
    """Compute Absolute Trajectory Error after alignment."""
    min_len = min(len(gt), len(est))
    gt = gt[:min_len]
    est = est[:min_len]

    # Simple alignment: translate to start at same point
    offset = gt[0] - est[0]
    est_aligned = est + offset

    errors = np.linalg.norm(gt - est_aligned, axis=1)
    return errors, np.sqrt(np.mean(errors**2))


def _plot_trajectories_2d(ax, trajectories, names, colors):
    """Plot 2D trajectory overlay on given axis."""
    for i, ((est, gt), name, color) in enumerate(zip(trajectories, names, colors)):
        if gt is not None and len(gt) > 0:
            offset = gt[0] - est[0]
            est_plot = est + offset
            if i == 0:
                ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=2, label="Ground Truth", alpha=0.7)
        else:
            est_plot = est

        ax.plot(est_plot[:, 0], est_plot[:, 1], color=color, linewidth=2, label=name)
        ax.plot(est_plot[-1, 0], est_plot[-1, 1], "o", color=color, markersize=10)

    ax.plot(trajectories[0][0][0, 0], trajectories[0][0][0, 1], "g*", markersize=15, label="Start")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Trajectory Comparison (2D)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")


def _plot_error_over_time(ax, trajectories, names, colors):
    """Plot localization error over time on given axis."""
    for (est, gt), name, color in zip(trajectories, names, colors):
        if gt is not None and len(gt) > 0:
            errors, rmse = compute_ate(gt, est)
            frames = np.arange(len(errors))
            ax.plot(
                frames,
                errors * 100,
                color=color,
                linewidth=2,
                label=f"{name} (RMSE: {rmse * 100:.1f}cm)",
            )

    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Position Error (cm)", fontsize=11)
    ax.set_title("Localization Error Over Time", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_final_drift_bars(ax, trajectories, names, colors):
    """Plot bar chart of final drift by axis."""
    n_traj = len(trajectories)
    bar_width = 0.8 / n_traj
    x = np.arange(3)

    for i, ((est, gt), name, color) in enumerate(zip(trajectories, names, colors)):
        if gt is not None and len(gt) > 0:
            final_error = est[-1] - gt[-1]
        else:
            final_error = est[-1] - est[0]

        offset = (i - n_traj / 2 + 0.5) * bar_width
        ax.bar(x + offset, np.abs(final_error) * 100, bar_width, label=name, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["X", "Y", "Z"])
    ax.set_ylabel("Final Position Error (cm)", fontsize=11)
    ax.set_title("Final Drift by Axis", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def _create_summary_table(ax, trajectories, names):
    """Create summary statistics table on given axis."""
    ax.axis("off")

    table_data = []
    headers = ["Experiment", "RMSE (cm)", "Max Error (cm)", "Final Drift (cm)"]

    for (est, gt), name in zip(trajectories, names):
        if gt is not None and len(gt) > 0:
            errors, rmse = compute_ate(gt, est)
            max_err = np.max(errors)
            final_drift = np.linalg.norm(est[-1] - gt[-1])
        else:
            rmse = 0
            max_err = 0
            final_drift = np.linalg.norm(est[-1] - est[0])

        table_data.append(
            [name, f"{rmse * 100:.1f}", f"{max_err * 100:.1f}", f"{final_drift * 100:.1f}"]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.3, 0.2, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#D6DCE4")

    ax.set_title("Summary Statistics", fontsize=13, fontweight="bold", pad=20)


def plot_trajectory_comparison(
    trajectories, names, colors, output_path, title="Trajectory Comparison"
):
    """
    Create a comprehensive trajectory comparison figure.

    Args:
        trajectories: List of (estimated, ground_truth) tuples
        names: List of experiment names
        colors: List of colors for each trajectory
        output_path: Where to save the figure
        title: Figure title
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Create each subplot using helper functions
    _plot_trajectories_2d(fig.add_subplot(gs[0, 0]), trajectories, names, colors)
    _plot_error_over_time(fig.add_subplot(gs[0, 1]), trajectories, names, colors)
    _plot_final_drift_bars(fig.add_subplot(gs[1, 0]), trajectories, names, colors)
    _create_summary_table(fig.add_subplot(gs[1, 1]), trajectories, names)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare trajectory recordings")
    parser.add_argument("--baseline", type=str, help="Baseline trajectory file (.tum or .npy)")
    parser.add_argument("--perturbed", type=str, help="Perturbed trajectory file (.tum or .npy)")
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default="trajectory_recordings",
        help="Directory with trajectory recordings",
    )
    parser.add_argument(
        "--output", type=str, default="trajectory_comparison.png", help="Output figure path"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["baseline", "efficient", "aggressive"],
        help="Experiment names to compare",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Trajectory Comparison: Blue=Baseline | Red=Perturbed",
        help="Plot title",
    )
    args = parser.parse_args()

    # If --baseline and --perturbed are provided, use 3D plot
    if args.baseline and args.perturbed:
        baseline = load_trajectory(args.baseline)
        perturbed = load_trajectory(args.perturbed)
        print(f"Loaded baseline: {len(baseline)} points from {args.baseline}")
        print(f"Loaded perturbed: {len(perturbed)} points from {args.perturbed}")
        plot_trajectory_3d(baseline, perturbed, args.output, args.title)
        return

    # Otherwise use the old method with recordings directory
    recordings_dir = Path(args.recordings_dir)

    # Load trajectories
    trajectories = []
    names = []
    colors = ["green", "orange", "red", "purple", "blue"]

    for exp_name in args.experiments:
        est_file = recordings_dir / f"{exp_name}_estimated.npy"
        gt_file = recordings_dir / f"{exp_name}_ground_truth.npy"

        if not est_file.exists():
            print(f"Warning: {est_file} not found, skipping")
            continue

        est = load_trajectory(est_file)
        gt = load_trajectory(gt_file) if gt_file.exists() else None

        trajectories.append((est, gt))
        names.append(exp_name.replace("_", " ").title())

    if len(trajectories) < 2:
        print("Need at least 2 trajectories to compare")
        return

    # Generate comparison plot
    plot_trajectory_comparison(
        trajectories,
        names,
        colors[: len(trajectories)],
        args.output,
        title="Adversarial Attack on MOLA SLAM: Trajectory Comparison",
    )


if __name__ == "__main__":
    main()
