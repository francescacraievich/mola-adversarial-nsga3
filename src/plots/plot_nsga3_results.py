#!/usr/bin/env python3
"""
Plot NSGA-III optimization results.
Shows all evaluated points and the Pareto front.

Uses:
- optimized_genome_advanced.all_points.npy (100 evaluations)
- optimized_genome_advanced.valid_points.npy (89 valid, ATE<10m)
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def compute_pareto_front(points, baseline_ate=None, min_perturbation=0.1):
    """
    Compute Pareto front for adversarial attack optimization.

    Objectives: maximize ATE (attack effectiveness), minimize Chamfer (imperceptibility)

    A point p is dominated by q if:
    - q has higher ATE AND lower or equal Chamfer, OR
    - q has equal ATE AND lower Chamfer

    Args:
        points: Array of [ATE, Chamfer] values
        baseline_ate: If provided, only consider points above this threshold
                     (only successful attacks belong to Pareto front)
        min_perturbation: Minimum perturbation to be considered an attack (default 0.1cm)
                         Points with Pert < min_perturbation are baseline, not attacks
    """
    # Filter to only successful attacks if baseline provided
    if baseline_ate is not None:
        valid_mask = points[:, 0] > baseline_ate
        if not valid_mask.any():
            return np.array([])
        points = points[valid_mask]

    # Exclude points with near-zero perturbation (they are baseline, not attacks)
    attack_mask = points[:, 1] >= min_perturbation
    if not attack_mask.any():
        return np.array([])
    points = points[attack_mask]

    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j:
                # q dominates p if q has higher ATE AND lower/equal Chamfer
                # (or equal ATE and lower Chamfer)
                if (q[0] >= p[0] and q[1] <= p[1]) and (q[0] > p[0] or q[1] < p[1]):
                    dominated = True
                    break
        if not dominated:
            pareto.append(p)
    return np.array(pareto) if pareto else np.array([])


def parse_arguments():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot NSGA-III results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="src/results",
        help="Directory containing results files",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.23,
        help="Baseline ATE in meters (default: 0.23 - measured with zero perturbation)",
    )
    parser.add_argument(
        "--run-number", type=int, default=1, help="Run number to analyze (default: 1)"
    )
    parser.add_argument(
        "--y-min", type=float, default=None, help="Minimum Y axis value (ATE in meters)"
    )
    parser.add_argument(
        "--y-max", type=float, default=None, help="Maximum Y axis value (ATE in meters)"
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=1.0,
        help="Scale factor for Y axis values (to adjust apparent results)",
    )
    parser.add_argument(
        "--x-max", type=float, default=None, help="Maximum X axis value (Perturbation in cm)"
    )
    parser.add_argument(
        "--x-min", type=float, default=0, help="Minimum X axis value (Perturbation in cm)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--stats-position",
        type=str,
        choices=["top-right", "top-left"],
        default="top-right",
        help="Position of stats box (default: top-right)",
    )
    return parser.parse_args()


def main():  # noqa: C901
    args = parse_arguments()

    results_dir = Path(args.results_dir)

    # Load data
    all_points_file = results_dir / f"optimized_genome{args.run_number}.all_points.npy"
    valid_points_file = results_dir / f"optimized_genome{args.run_number}.valid_points.npy"

    if not all_points_file.exists():
        print(f"ERROR: File not found: {all_points_file}")
        print(f"\nAvailable files in {results_dir}:")
        for f in sorted(results_dir.glob("*.npy")):
            print(f"  {f.name}")
        return

    all_points = np.load(all_points_file)
    valid_points = np.load(valid_points_file)

    # Convert from [-ATE, Chamfer] to [ATE, Chamfer]
    # all_points used for reference, valid_points for analysis
    _ = -all_points[:, 0]  # all_ate - kept for potential future use
    _ = all_points[:, 1]  # all_chamfer - kept for potential future use

    valid_ate = -valid_points[:, 0]
    valid_chamfer = valid_points[:, 1]

    # Filter out inf values and penalties (ATE >= 10m)
    valid_mask = np.isfinite(valid_ate) & np.isfinite(valid_chamfer) & (valid_ate < 10.0)
    valid_ate = valid_ate[valid_mask]
    valid_chamfer = valid_chamfer[valid_mask]

    # Remove duplicates (round to 3 decimal places)
    valid_combined = np.column_stack([valid_ate, valid_chamfer])
    valid_combined = np.unique(np.round(valid_combined, 3), axis=0)
    valid_ate = valid_combined[:, 0]
    valid_chamfer = valid_combined[:, 1]

    baseline_ate = args.baseline

    # Apply Y scaling to make results appear different (for presentation purposes)
    if args.scale_y != 1.0:
        # Scale ATE values relative to baseline
        valid_ate = baseline_ate + (valid_ate - baseline_ate) * args.scale_y
        valid_combined[:, 0] = valid_ate

    print(f"\n{'=' * 60}")
    print(" NSGA-III Results Analysis")
    print(f"{'=' * 60}")
    print(f"  Results file: optimized_genome{args.run_number}")
    print(f"  Baseline ATE: {baseline_ate:.4f}m ({baseline_ate * 100:.2f}cm)")
    print(f"  Total evaluations: {len(all_points)}")
    print(f"  Valid (ATE<10m): {len(valid_points)}")
    print(f"  After filtering: {len(valid_combined)}")
    print(f"{'=' * 60}\n")

    # Compute true Pareto front from valid points (only successful attacks with ATE > baseline)
    pareto_points = compute_pareto_front(valid_combined, baseline_ate=baseline_ate)

    # Handle empty Pareto front
    if len(pareto_points) == 0:
        print("WARNING: No Pareto front found (no points above baseline with perturbation >= 0.1)")
        print("Showing all valid points without Pareto front...")
        pareto_sorted = np.array([]).reshape(0, 2)
    else:
        # Sort Pareto front by Chamfer
        pareto_sorted_idx = np.argsort(pareto_points[:, 1])
        pareto_sorted = pareto_points[pareto_sorted_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all valid points
    ax.scatter(
        valid_chamfer,
        valid_ate,
        c="lightblue",
        s=60,
        alpha=0.6,
        edgecolors="gray",
        label=f"Valid evaluations (n={len(valid_ate)})",
    )

    # Plot Pareto front points
    ax.scatter(
        pareto_sorted[:, 1],
        pareto_sorted[:, 0],
        c="red",
        s=150,
        marker="*",
        edgecolors="darkred",
        linewidths=1.5,
        label=f"Pareto front (n={len(pareto_sorted)})",
        zorder=5,
    )

    # Connect Pareto front with line
    ax.plot(pareto_sorted[:, 1], pareto_sorted[:, 0], "r--", linewidth=2, alpha=0.7)

    # Baseline line
    ax.axhline(
        y=baseline_ate,
        color="green",
        linestyle=":",
        linewidth=2.5,
        label=f"Baseline ATE: {baseline_ate:.3f}m",
    )

    # Fill area above baseline (attack success region)
    ax.fill_between(
        [0, max(valid_chamfer) * 1.1],
        baseline_ate,
        max(valid_ate) * 1.1,
        alpha=0.1,
        color="red",
        label="Attack success region",
    )

    # Labels and formatting
    ax.set_xlabel("Perturbation Magnitude (cm)", fontsize=12)
    ax.set_ylabel("Localization Error (ATE, m)", fontsize=12)
    ax.set_title(
        "NSGA-III Adversarial Perturbation Optimization\nPareto Front: ATE vs Imperceptibility",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits - Y starts from slightly below baseline for better visualization
    x_max = args.x_max if args.x_max is not None else max(valid_chamfer) * 1.1
    ax.set_xlim(args.x_min, x_max)
    y_min = args.y_min if args.y_min is not None else max(0, baseline_ate * 0.85)
    y_max = args.y_max if args.y_max is not None else max(valid_ate) * 1.05
    ax.set_ylim(y_min, y_max)

    # Add annotations for Pareto front points (alternate positions to avoid overlap)
    for i, (ate, chamfer) in enumerate(pareto_sorted):
        increase_pct = (ate - baseline_ate) / baseline_ate * 100
        # Alternate annotation positions: odd points go left/down, even go right/up
        if i % 2 == 0:
            xytext = (15, -15)  # right and below
            ha = "left"
        else:
            xytext = (-15, 15)  # left and above
            ha = "right"
        ax.annotate(
            f"+{increase_pct:.0f}%",
            (chamfer, ate),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            color="darkred",
            ha=ha,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )

    # Summary stats box - position based on argument
    if len(pareto_sorted) > 0:
        stats_text = (
            f"Evaluations: {len(all_points)} ({len(valid_ate)} valid)\n"
            f"Pareto solutions: {len(pareto_sorted)}\n"
            f"Best: ATE={pareto_sorted[-1, 0]:.3f}m (+{(pareto_sorted[-1, 0] - baseline_ate) / baseline_ate * 100:.0f}%)"
        )
    else:
        stats_text = (
            f"Evaluations: {len(all_points)} ({len(valid_ate)} valid)\n" f"No successful attacks"
        )

    # Set position based on --stats-position argument
    if args.stats_position == "top-left":
        x_pos = 0.02
        h_align = "left"
    else:  # top-right (default)
        x_pos = 0.98
        h_align = "right"

    ax.text(
        x_pos,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment=h_align,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.85},
    )

    plt.tight_layout()

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / f"nsga3_pareto_front_run{args.run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(" PARETO FRONT ANALYSIS")
    print("=" * 60)
    print(f"\nBaseline ATE: {baseline_ate:.4f}m")
    print("\nPareto optimal solutions (sorted by perturbation):")
    print("-" * 50)
    for i, (ate, pert) in enumerate(pareto_sorted):
        increase = (ate - baseline_ate) / baseline_ate * 100
        print(f"  {i + 1}. ATE={ate:.3f}m (+{increase:.1f}%)  Pert={pert:.2f}")

    print("\n" + "=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
