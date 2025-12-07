#!/usr/bin/env python3
"""
Analyze correlation of genome parameters with BOTH objectives:
1. ATE (attack effectiveness)
2. Perturbation magnitude (imperceptibility)

This helps identify which parameters should be removed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GENOME_PARAMS = [
    "Noise Dir X",
    "Noise Dir Y",
    "Noise Dir Z",
    "Noise Intensity",
    "Curvature Targeting",
    "Dropout Rate",
    "Ghost Ratio",
    "Cluster Dir X",
    "Cluster Dir Y",
    "Cluster Dir Z",
    "Cluster Strength",
    "Spatial Correlation",
    "Geometric Distortion",
    "Edge Attack",
    "Temporal Drift",
    "Scanline Perturbation",
    "Strategic Ghost",
]


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Analyze dual correlation: parameters vs ATE and Perturbation"
    )
    parser.add_argument("--run", type=int, default=11, help="Run number")
    parser.add_argument(
        "--results-dir", type=str, default="src/results/runs", help="Results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/results/parameter_dual_correlation.png",
        help="Output file",
    )
    args = parser.parse_args()

    # Load data
    base_path = Path(args.results_dir) / f"optimized_genome{args.run}"
    valid_points_path = base_path.with_suffix(".valid_points.npy")
    valid_genomes_path = base_path.with_suffix(".valid_genomes.npy")

    if not valid_points_path.exists() or not valid_genomes_path.exists():
        print("ERROR: Required files not found")
        return 1

    fitness = np.load(valid_points_path)
    genomes = np.load(valid_genomes_path)

    # Convert negative ATE
    if fitness[:, 0].mean() < 0:
        fitness[:, 0] = -fitness[:, 0]

    ate_values = fitness[:, 0]
    pert_values = fitness[:, 1]

    print(f"\n{'=' * 70}")
    print(f" DUAL CORRELATION ANALYSIS (Run {args.run})")
    print(f"{'=' * 70}")
    print(f"  Evaluations: {len(genomes)}")
    print(f"  Parameters: {len(GENOME_PARAMS)}")
    print("\n  Analyzing correlation with:")
    print("    1. ATE (higher = worse SLAM = good for attack)")
    print("    2. Perturbation (higher = more noticeable = bad)")
    print(f"{'=' * 70}\n")

    # Compute correlations
    correlations_ate = []
    correlations_pert = []

    for i in range(genomes.shape[1]):
        param_values = genomes[:, i]

        # Correlation with ATE
        corr_ate = np.corrcoef(param_values, ate_values)[0, 1]
        correlations_ate.append(corr_ate if not np.isnan(corr_ate) else 0)

        # Correlation with Perturbation
        corr_pert = np.corrcoef(param_values, pert_values)[0, 1]
        correlations_pert.append(corr_pert if not np.isnan(corr_pert) else 0)

    correlations_ate = np.array(correlations_ate)
    correlations_pert = np.array(correlations_pert)

    # Categorize parameters
    categories = []
    for i, param in enumerate(GENOME_PARAMS):
        corr_ate = correlations_ate[i]
        corr_pert = correlations_pert[i]

        # Determine category
        if corr_ate > 0.2 and abs(corr_pert) < 0.3:
            category = "GOOD: Increases ATE, low perturbation impact"
        elif corr_ate > 0.2 and corr_pert > 0.3:
            category = "TRADE-OFF: Increases ATE but also perturbation"
        elif corr_ate < -0.15 and corr_pert > 0.3:
            category = "BAD: Decreases ATE and increases perturbation"
        elif corr_ate < -0.15 and corr_pert < -0.2:
            category = "USEFUL: Reduces both ATE and perturbation (for low-pert solutions)"
        elif abs(corr_ate) < 0.15 and abs(corr_pert) < 0.15:
            category = "USELESS: No significant correlation with either objective"
        else:
            category = "UNCLEAR: Complex interaction"

        categories.append((i, param, corr_ate, corr_pert, category))

    # Sort by ATE correlation (most important first)
    categories.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print analysis
    print("\nPARAMETER RECOMMENDATIONS:\n")

    keep = []
    consider_remove = []
    remove = []

    for i, param, corr_ate, corr_pert, cat in categories:
        status = "?"
        if "GOOD" in cat or "TRADE-OFF" in cat:
            status = "✓ KEEP"
            keep.append(param)
        elif "USELESS" in cat or "BAD" in cat:
            status = "✗ REMOVE"
            remove.append(param)
        else:
            status = "? REVIEW"
            consider_remove.append(param)

        print(f"{status:12s} | {param:20s} | ATE: {corr_ate:+.3f} | Pert: {corr_pert:+.3f}")
        print(f"             | {cat}")
        print()

    # Summary
    print(f"{'=' * 70}")
    print(" SUMMARY")
    print(f"{'=' * 70}")
    print(f"  ✓ Keep ({len(keep)} params): {', '.join(keep)}")
    print(f"  ? Review ({len(consider_remove)} params): {', '.join(consider_remove)}")
    print(f"  ✗ Consider removing ({len(remove)} params): {', '.join(remove)}")
    print(f"{'=' * 70}\n")

    # Create TWO separate bar plots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))

    # ============ PLOT 1: Correlation with ATE ============
    # Sort by ATE correlation
    sorted_idx_ate = np.argsort(correlations_ate)
    sorted_params_ate = [GENOME_PARAMS[i] for i in sorted_idx_ate]
    sorted_corrs_ate = correlations_ate[sorted_idx_ate]

    y_pos = np.arange(len(sorted_params_ate))
    colors_ate = ["#2ecc71" if c > 0 else "#e74c3c" for c in sorted_corrs_ate]

    bars1 = ax1.barh(
        y_pos, sorted_corrs_ate, color=colors_ate, edgecolor="black", linewidth=1.5, alpha=0.85
    )

    # Highlight top 3 by absolute value
    top3_abs_idx = np.argsort(np.abs(correlations_ate))[-3:]
    for idx in top3_abs_idx:
        pos = np.where(sorted_idx_ate == idx)[0][0]
        bars1[pos].set_linewidth(3)
        bars1[pos].set_edgecolor("darkblue")
        bars1[pos].set_alpha(0.95)

    # Add value labels - all outside bars for visibility
    for i, (bar, corr) in enumerate(zip(bars1, sorted_corrs_ate)):
        width = bar.get_width()
        # More space for negative values to avoid overlap with y-labels
        label_x = width + 0.03 if width > 0 else width - 0.08
        ha = "left" if width > 0 else "right"

        fontweight = "bold" if i in [len(bars1) - 1, len(bars1) - 2, len(bars1) - 3] else "normal"
        fontsize = 11
        ax1.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{corr:+.3f}",
            ha=ha,
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color="black",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_params_ate, fontsize=11)
    ax1.set_xlabel("Correlation with ATE", fontsize=13, fontweight="bold")
    ax1.set_title("Parameters vs ATE", fontsize=14, fontweight="bold", pad=15)
    ax1.axvline(x=0, color="black", linewidth=2)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    ax1.set_xlim(left=-0.8, right=0.8)  # Add more space on both sides
    ax1.invert_yaxis()

    # ============ PLOT 2: Correlation with Perturbation ============
    # Sort by Perturbation correlation
    sorted_idx_pert = np.argsort(correlations_pert)
    sorted_params_pert = [GENOME_PARAMS[i] for i in sorted_idx_pert]
    sorted_corrs_pert = correlations_pert[sorted_idx_pert]

    colors_pert = [
        "#e74c3c" if c > 0 else "#2ecc71" for c in sorted_corrs_pert
    ]  # Inverse: high pert is bad

    bars2 = ax2.barh(
        y_pos, sorted_corrs_pert, color=colors_pert, edgecolor="black", linewidth=1.5, alpha=0.85
    )

    # Highlight top 3 by absolute value
    top3_abs_idx_pert = np.argsort(np.abs(correlations_pert))[-3:]
    for idx in top3_abs_idx_pert:
        pos = np.where(sorted_idx_pert == idx)[0][0]
        bars2[pos].set_linewidth(3)
        bars2[pos].set_edgecolor("darkblue")
        bars2[pos].set_alpha(0.95)

    # Add value labels - all outside bars for visibility
    for i, (bar, corr) in enumerate(zip(bars2, sorted_corrs_pert)):
        width = bar.get_width()
        # More space for negative values to avoid overlap with y-labels
        label_x = width + 0.03 if width > 0 else width - 0.08
        ha = "left" if width > 0 else "right"

        fontweight = (
            "bold" if i in [len(bars2) - 1, len(bars2) - 2, len(bars2) - 3, 0, 1, 2] else "normal"
        )
        fontsize = 11
        ax2.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{corr:+.3f}",
            ha=ha,
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color="black",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_params_pert, fontsize=11)
    ax2.set_xlabel("Correlation with Perturbation", fontsize=13, fontweight="bold")
    ax2.set_title("Parameters vs Perturbation", fontsize=14, fontweight="bold", pad=15)
    ax2.axvline(x=0, color="black", linewidth=2)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.set_xlim(
        left=-0.8, right=1.2
    )  # Add more space on both sides (more on right for large values)
    ax2.invert_yaxis()

    # Main title
    fig.suptitle(
        f"Dual Correlation Analysis - Run {args.run}", fontsize=15, fontweight="bold", y=0.96
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
