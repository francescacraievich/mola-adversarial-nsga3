#!/usr/bin/env python3
"""
Plot perturbation efficiency - simple bar chart.
Shows ATE increase from baseline for each perturbation type.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from the test
results = [
    {"name": "Baseline", "ate": 0.6877, "chamfer_cm": 1.079, "ate_increase": 0.0},
    {"name": "Small noise (1cm)", "ate": 0.7470, "chamfer_cm": 1.126, "ate_increase": 8.6},
    {"name": "Medium noise (3cm)", "ate": 0.7625, "chamfer_cm": 1.053, "ate_increase": 10.9},
    {"name": "Large noise (5cm)", "ate": 0.7681, "chamfer_cm": 1.056, "ate_increase": 11.7},
    {"name": "Feature mild", "ate": 0.7343, "chamfer_cm": 1.258, "ate_increase": 6.8},
    {"name": "Feature strong", "ate": 0.7319, "chamfer_cm": 1.420, "ate_increase": 6.4},
    {"name": "Dropout 5%", "ate": 0.8297, "chamfer_cm": 1.098, "ate_increase": 20.7},
    {"name": "Dropout 10%", "ate": 0.7544, "chamfer_cm": 1.056, "ate_increase": 9.7},
    {"name": "Targeted dropout", "ate": 0.7836, "chamfer_cm": 1.447, "ate_increase": 14.0},
    {"name": "Ghost points 3%", "ate": 0.7249, "chamfer_cm": 1.093, "ate_increase": 5.4},
    {"name": "Cluster", "ate": 0.7342, "chamfer_cm": 1.629, "ate_increase": 6.8},
    {"name": "Noise + Feature", "ate": 0.8208, "chamfer_cm": 1.999, "ate_increase": 19.4},
    {"name": "All mild", "ate": 0.7144, "chamfer_cm": 2.133, "ate_increase": 3.9},
    {"name": "Aggressive stealth", "ate": 0.8049, "chamfer_cm": 3.207, "ate_increase": 17.0},
]

# Sort by efficiency (ATE increase / Chamfer distance) - skip baseline
perturbations = [r for r in results if r["name"] != "Baseline"]
efficiency = [r["ate_increase"] / r["chamfer_cm"] for r in perturbations]
sorted_indices = np.argsort(efficiency)[::-1]

# Get sorted data
sorted_names = [perturbations[i]["name"] for i in sorted_indices]
sorted_ate = [perturbations[i]["ate"] for i in sorted_indices]
sorted_efficiency = [efficiency[i] for i in sorted_indices]
sorted_chamfer = [perturbations[i]["chamfer_cm"] for i in sorted_indices]

baseline_ate = 0.6877

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create horizontal bar chart
y_pos = np.arange(len(sorted_names))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))

bars = ax.barh(y_pos, sorted_ate, color=colors, edgecolor="black", alpha=0.85, height=0.7)

# Add baseline line
ax.axvline(
    x=baseline_ate,
    color="blue",
    linestyle="--",
    linewidth=2.5,
    label=f"Baseline ATE: {baseline_ate:.3f}m",
)

# Labels and formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=10)
ax.set_xlabel("ATE (m) - Localization Error", fontsize=12)
ax.set_title(
    "Perturbation Efficiency Ranking\n(Sorted by ATE%/Chamfer - higher = more efficient attack)",
    fontsize=13,
    fontweight="bold",
)

# Set x-axis to start from baseline
ax.set_xlim(baseline_ate - 0.02, max(sorted_ate) + 0.05)

# Add efficiency labels on bars
for i, (bar, eff, chamfer) in enumerate(zip(bars, sorted_efficiency, sorted_chamfer)):
    width = bar.get_width()
    ax.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{eff:.1f}%/cm",
        va="center",
        fontsize=9,
        color="darkred",
        fontweight="bold",
    )
    ax.text(
        baseline_ate + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"Δ={chamfer:.2f}cm",
        va="center",
        fontsize=8,
        color="gray",
        alpha=0.8,
    )

ax.legend(loc="lower right", fontsize=11)
ax.grid(axis="x", alpha=0.3)

# Invert y-axis so best is at top
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("results/perturbation_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: results/perturbation_analysis.png")

# Summary
print("\n" + "=" * 60)
print(" TOP 5 MOST EFFICIENT ATTACKS")
print("=" * 60)
for i in range(min(5, len(sorted_names))):
    print(f"  {i + 1}. {sorted_names[i]}: {sorted_efficiency[i]:.1f} ATE%/cm")

plt.show()
