#!/usr/bin/env python3
"""
Optimize adversarial perturbations using NSGA-II.

This script runs NSGA-II optimization to find perturbations that:
1. Maximize localization error (make robot get lost)
2. Minimize perturbation magnitude (minimum effort)

Can run in two modes:
- offline: Fast, uses simulated fitness (for testing)
- online: Slow, uses real MOLA evaluation (accurate)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.fitness_evaluator import OfflineFitnessEvaluator  # noqa: E402
from src.optimization.pymoo_wrapper import PerturbationOptimizer  # noqa: E402
from src.perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize adversarial perturbations with NSGA-II")
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="Evaluation mode (offline=fast/simulated, online=real MOLA)",
    )
    parser.add_argument("--population", type=int, default=50, help="Population size for NSGA-II")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def create_synthetic_data():
    """Create synthetic data for offline mode."""
    print("📊 Creating synthetic data...")

    # Ground truth trajectory
    n_poses = 50
    gt_trajectory = np.zeros((n_poses, 3))
    gt_trajectory[:, 0] = np.linspace(0, 10, n_poses)
    gt_trajectory[:, 1] = np.sin(gt_trajectory[:, 0] * 0.5) * 2

    # Baseline SLAM trajectory (with small noise)
    baseline_trajectory = gt_trajectory + np.random.normal(0, 0.05, gt_trajectory.shape)

    # Reference point cloud
    n_points = 1000
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = 5 * np.cos(theta)
    y = 5 * np.sin(theta)
    z = np.random.uniform(-0.5, 0.5, n_points)
    intensity = 100 * np.ones(n_points)
    point_cloud = np.column_stack([x, y, z, intensity])

    print(f"  ✓ Ground truth: {gt_trajectory.shape}")
    print(f"  ✓ Baseline: {baseline_trajectory.shape}")
    print(f"  ✓ Point cloud: {point_cloud.shape}")

    return gt_trajectory, baseline_trajectory, point_cloud


def plot_results(result, generator, output_dir):
    """Plot and save optimization results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_genomes = result["pareto_front"]
    pareto_objectives = result["pareto_objectives"]
    all_objectives = result["objectives"]

    # Plot 1: Pareto front
    plt.figure(figsize=(10, 6))

    # All solutions
    plt.scatter(
        all_objectives[:, 1],
        -all_objectives[:, 0],
        alpha=0.3,
        s=30,
        label="All solutions",
        color="lightblue",
    )

    # Pareto front
    plt.scatter(
        pareto_objectives[:, 1],
        -pareto_objectives[:, 0],
        alpha=0.8,
        s=100,
        label="Pareto front",
        color="red",
        edgecolors="black",
        linewidths=1.5,
    )

    plt.xlabel("Imperceptibility (lower = more stealthy)", fontsize=12)
    plt.ylabel("Attack Effectiveness (higher = more drift)", fontsize=12)
    plt.title("NSGA-II Pareto Front: Attack vs Stealth Trade-off", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pareto_plot = output_dir / "pareto_front.png"
    plt.savefig(pareto_plot, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {pareto_plot}")
    plt.close()

    # Plot 2: Top 3 solutions analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Find best solutions for different weights
    weights_configs = [
        ("High Attack\n(90% weight)", np.array([0.9, 0.1]), "red"),
        ("Balanced\n(50/50)", np.array([0.5, 0.5]), "orange"),
        ("High Stealth\n(90% weight)", np.array([0.1, 0.9]), "green"),
    ]

    for idx, (title, weights, color) in enumerate(weights_configs):
        ax = axes[idx]

        # Find best solution
        weighted_scores = pareto_objectives @ weights
        best_idx = np.argmin(weighted_scores)
        best_genome = pareto_genomes[best_idx]
        best_obj = pareto_objectives[best_idx]

        # Decode perturbation
        params = generator.encode_perturbation(best_genome)

        # Plot perturbation components
        components = {
            "Translation\n(m)": np.linalg.norm(params["translation"]),
            "Rotation\n(rad)": np.linalg.norm(params["rotation"]),
            "Intensity": abs(params["intensity_scale"]) / 50.0,
            "Dropout\n(rate)": params["dropout_rate"] * 10,
        }

        ax.bar(range(len(components)), list(components.values()), color=color, alpha=0.7)
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(list(components.keys()), fontsize=9)
        ax.set_ylabel("Magnitude", fontsize=10)
        ax.set_title(
            f"{title}\nAttack: {-best_obj[0]:.3f} | Stealth: {best_obj[1]:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    solutions_plot = output_dir / "top_solutions.png"
    plt.savefig(solutions_plot, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {solutions_plot}")
    plt.close()


def save_best_perturbations(result, generator, output_dir):
    """Save best perturbation genomes for later use."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_genomes = result["pareto_front"]
    pareto_objectives = result["pareto_objectives"]

    # Find best solutions
    best_attack = pareto_genomes[np.argmin(pareto_objectives @ np.array([0.9, 0.1]))]
    best_balanced = pareto_genomes[np.argmin(pareto_objectives @ np.array([0.5, 0.5]))]
    best_stealth = pareto_genomes[np.argmin(pareto_objectives @ np.array([0.1, 0.9]))]

    # Save
    output_file = output_dir / "best_perturbations.npz"
    np.savez(
        output_file,
        best_attack=best_attack,
        best_balanced=best_balanced,
        best_stealth=best_stealth,
        pareto_front=pareto_genomes,
        pareto_objectives=pareto_objectives,
    )
    print(f"  ✓ Saved: {output_file}")

    # Save human-readable text file
    text_file = output_dir / "best_perturbations.txt"
    with open(text_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("BEST PERTURBATIONS FROM NSGA-II OPTIMIZATION\n")
        f.write("=" * 70 + "\n\n")

        for name, genome in [
            ("High Attack", best_attack),
            ("Balanced", best_balanced),
            ("High Stealth", best_stealth),
        ]:
            params = generator.encode_perturbation(genome)
            f.write(f"{name}:\n")
            f.write(f"  Translation: {params['translation']}\n")
            f.write(f"  Rotation: {params['rotation']}\n")
            f.write(f"  Intensity: {params['intensity_scale']:.2f}\n")
            f.write(f"  Dropout: {params['dropout_rate']:.1%}\n")
            f.write(f"  Genome: {genome}\n\n")

    print(f"  ✓ Saved: {text_file}")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("🧬 NSGA-II ADVERSARIAL PERTURBATION OPTIMIZATION")
    print("=" * 70 + "\n")

    print("Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed: {args.seed}\n")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"optimization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Results will be saved to: {output_dir}\n")

    # Initialize perturbation generator
    print("🔧 Initializing perturbation generator...")
    generator = PerturbationGenerator(
        max_translation=0.5,
        max_rotation=0.1,
        max_intensity_change=50.0,
        point_dropout_rate=0.1,
    )
    print(f"  ✓ Genome size: {generator.get_genome_size()}")

    # Initialize fitness evaluator
    print(f"\n📊 Initializing fitness evaluator ({args.mode} mode)...")

    if args.mode == "offline":
        # Offline mode: fast, simulated
        gt_trajectory, baseline_trajectory, point_cloud = create_synthetic_data()

        evaluator = OfflineFitnessEvaluator(
            perturbation_generator=generator,
            ground_truth_trajectory=gt_trajectory,
            baseline_trajectory=baseline_trajectory,
            reference_point_cloud=point_cloud,
        )
        print("  ✓ Offline evaluator ready (simulated fitness)")
    else:
        # Online mode: real MOLA
        print("  ⚠️  Online mode not yet implemented!")
        print("  Use offline mode for now")
        return

    # Initialize optimizer
    print("\n🧬 Initializing NSGA-II optimizer...")
    optimizer = PerturbationOptimizer(
        genome_size=generator.get_genome_size(),
        fitness_function=evaluator.get_fitness_function(),
        n_objectives=2,
        population_size=args.population,
        seed=args.seed,
    )
    print("  ✓ Optimizer ready")

    # Run optimization
    print("\n⚡ Running optimization...")
    print(f"  This will evaluate {args.population * args.generations} solutions")
    print(f"  Estimated time: ~{args.population * args.generations * 0.01:.1f} seconds\n")

    result = optimizer.optimize(n_generations=args.generations, verbose=True)

    print("\n✅ Optimization complete!")
    print(f"  Final population: {len(result['population'])}")
    print(f"  Pareto front size: {len(result['pareto_front'])}")

    # Analyze results
    print("\n📊 Analyzing results...")
    pareto_objectives = result["pareto_objectives"]

    print("\nPareto front statistics:")
    print("  Attack effectiveness (neg. error):")
    print(f"    Range: [{pareto_objectives[:, 0].min():.4f}, {pareto_objectives[:, 0].max():.4f}]")
    print(f"    Mean: {pareto_objectives[:, 0].mean():.4f}")
    print("  Imperceptibility (perturbation mag):")
    print(f"    Range: [{pareto_objectives[:, 1].min():.4f}, {pareto_objectives[:, 1].max():.4f}]")
    print(f"    Mean: {pareto_objectives[:, 1].mean():.4f}")

    # Get best solutions
    print("\n🎯 Best solutions:")

    best_attack = optimizer.get_best_solution(np.array([0.9, 0.1]))
    print("\n  1. Most effective attack (90% weight):")
    print(f"     Attack: {-best_attack[1][0]:.4f} | Stealth: {best_attack[1][1]:.4f}")
    params = generator.encode_perturbation(best_attack[0])
    print(f"     Translation: {np.linalg.norm(params['translation']):.3f} m")
    print(f"     Rotation: {np.linalg.norm(params['rotation']):.3f} rad")

    best_balanced = optimizer.get_best_solution(np.array([0.5, 0.5]))
    print("\n  2. Balanced trade-off (50/50):")
    print(f"     Attack: {-best_balanced[1][0]:.4f} | Stealth: {best_balanced[1][1]:.4f}")
    params = generator.encode_perturbation(best_balanced[0])
    print(f"     Translation: {np.linalg.norm(params['translation']):.3f} m")
    print(f"     Rotation: {np.linalg.norm(params['rotation']):.3f} rad")

    best_stealth = optimizer.get_best_solution(np.array([0.1, 0.9]))
    print("\n  3. Most stealthy (90% weight):")
    print(f"     Attack: {-best_stealth[1][0]:.4f} | Stealth: {best_stealth[1][1]:.4f}")
    params = generator.encode_perturbation(best_stealth[0])
    print(f"     Translation: {np.linalg.norm(params['translation']):.3f} m")
    print(f"     Rotation: {np.linalg.norm(params['rotation']):.3f} rad")

    # Save results
    print("\n💾 Saving results...")
    plot_results(result, generator, output_dir)
    save_best_perturbations(result, generator, output_dir)

    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}")
    print("\nNext steps:")
    print(f"  1. View plots: {output_dir}/pareto_front.png")
    print(f"  2. Load best perturbations: {output_dir}/best_perturbations.npz")
    print("  3. Test in MOLA using scripts/test_perturbation.py")
    print()


if __name__ == "__main__":
    main()
