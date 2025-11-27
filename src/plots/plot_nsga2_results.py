#!/usr/bin/env python3
"""
Plot NSGA-II optimization results.
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


def compute_pareto_front(points, baseline_ate=None):
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
    """
    # Filter to only successful attacks if baseline provided
    if baseline_ate is not None:
        valid_mask = points[:, 0] > baseline_ate
        if not valid_mask.any():
            return np.array([])
        points = points[valid_mask]

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


def main():
    results_dir = Path("results")

    # Load data
    all_points = np.load(results_dir / "optimized_genome_advanced.all_points.npy")
    valid_points = np.load(results_dir / "optimized_genome_advanced.valid_points.npy")

    # Convert from [-ATE, Chamfer] to [ATE, Chamfer]
    # all_points used for reference, valid_points for analysis
    _ = -all_points[:, 0]  # all_ate - kept for potential future use
    _ = all_points[:, 1]  # all_chamfer - kept for potential future use

    valid_ate = -valid_points[:, 0]
    valid_chamfer = valid_points[:, 1]

    # Filter out penalties (ATE >= 10m)
    valid_mask = valid_ate < 10.0
    valid_ate = valid_ate[valid_mask]
    valid_chamfer = valid_chamfer[valid_mask]

    # Remove duplicates (round to 3 decimal places)
    valid_combined = np.column_stack([valid_ate, valid_chamfer])
    valid_combined = np.unique(np.round(valid_combined, 3), axis=0)
    valid_ate = valid_combined[:, 0]
    valid_chamfer = valid_combined[:, 1]

    baseline_ate = 0.6877

    # Compute true Pareto front from valid points (only successful attacks with ATE > baseline)
    pareto_points = compute_pareto_front(valid_combined, baseline_ate=baseline_ate)

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
    ax.set_xlabel("Perturbation Magnitude (Chamfer Distance, cm)", fontsize=12)
    ax.set_ylabel("Localization Error (ATE, m)", fontsize=12)
    ax.set_title(
        "NSGA-II Adversarial Perturbation Optimization\nPareto Front: ATE vs Imperceptibility",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(0, max(valid_chamfer) * 1.1)
    ax.set_ylim(0, max(valid_ate) * 1.1)

    # Add annotations for Pareto front points
    for i, (chamfer, ate) in enumerate(pareto_sorted):
        increase_pct = (ate - baseline_ate) / baseline_ate * 100
        ax.annotate(
            f"+{increase_pct:.0f}%\n({chamfer:.2f}cm)",
            (chamfer, ate),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            color="darkred",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    # Summary stats box
    stats_text = (
        f"Total evaluations: {len(all_points)}\n"
        f"Valid (ATE<10m): {len(valid_ate)}\n"
        f"Failed: {len(all_points) - len(valid_ate)}\n"
        f"Pareto solutions: {len(pareto_sorted)}\n"
        f"\nBest attack:\n"
        f"  ATE: {pareto_sorted[-1, 0]:.3f}m (+{(pareto_sorted[-1, 0] - baseline_ate) / baseline_ate * 100:.0f}%)\n"
        f"  Chamfer: {pareto_sorted[-1, 1]:.2f}cm"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save
    output_path = results_dir / "nsga2_pareto_front.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(" PARETO FRONT ANALYSIS")
    print("=" * 60)
    print(f"\nBaseline ATE: {baseline_ate:.4f}m")
    print("\nPareto optimal solutions (sorted by Chamfer):")
    print("-" * 50)
    for i, (ate, chamfer) in enumerate(pareto_sorted):
        increase = (ate - baseline_ate) / baseline_ate * 100
        print(f"  {i + 1}. ATE={ate:.3f}m (+{increase:.1f}%)  Chamfer={chamfer:.2f}cm")

    print("\n" + "=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
