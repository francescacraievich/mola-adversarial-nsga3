# Fitness Evaluation

## Overview

Fitness evaluation is the core of the NSGA-III optimization process. For each candidate solution (genome), we measure how well it achieves our two objectives: maximizing localization error while minimizing perturbation magnitude.

This evaluation happens hundreds of times during optimization (750 evaluations in genome12), making it the most computationally expensive component of the system.

## Two Objectives

### Objective 1: Maximize Localization Error (ATE)

The primary goal is to degrade MOLA's localization accuracy as much as possible. We measure this using Absolute Trajectory Error (ATE).

**ATE Definition:**
ATE is the root mean square error (RMSE) between the estimated trajectory and ground truth trajectory after optimal alignment.

**Interpretation:**
- Higher ATE = More trajectory error = Better attack
- Baseline ATE ≈ 23cm (MOLA's natural error on unperturbed data)
- Target ATE ≈ 65-85cm (after perturbations, +183% to +269%)

NSGA-III tries to **maximize** this objective (internally stored as negative value).

### Objective 2: Minimize Perturbation Magnitude

The secondary goal is to keep perturbations as small as possible to avoid detection.

**Perturbation Magnitude:**
We use the Chamfer distance between original and perturbed point clouds as the imperceptibility metric. This measures how much the point cloud has changed in a way that accounts for both point displacement and structural changes.

**Interpretation:**
- Lower Chamfer distance = More stealthy = Better
- Typical perturbations: 1-5cm Chamfer distance
- Physical constraint: perturbations should remain below sensor noise levels

NSGA-III tries to **minimize** this objective.

## Evaluation Pipeline

For each genome in the population, the `MOLAEvaluator` class performs these steps:

### Step 1: Decode Genome

The 17-parameter genome encodes multiple attack strategies simultaneously. The genome is a continuous array in range [-1, 1] that encodes:

- Parameters 0-2: Noise direction (3D vector)
- Parameter 3: Noise intensity
- Parameter 4: Curvature targeting strength
- Parameter 5: Dropout rate
- Parameter 6: Ghost point ratio
- Parameters 7-10: Cluster perturbation
- Parameter 11: Spatial correlation
- Parameter 12: Geometric distortion (ICP attack)
- Parameter 13: Edge attack strength
- Parameter 14: Temporal drift
- Parameter 15: Scanline perturbation
- Parameter 16: Strategic ghost placement

### Step 2: Generate Perturbed Sequence

Apply perturbations to all point cloud frames:

1. Apply the perturbation using `PerturbationGenerator.apply_perturbation()`
2. Compute Chamfer distance between original and perturbed cloud
3. Accumulate for average Chamfer distance

### Step 3: Run MOLA SLAM

Publish perturbed point clouds to ROS2 and run MOLA:

1. Start MOLA process in background
2. Publish clouds with synchronized timestamps and TF at 10Hz
3. Collect trajectory from MOLA odometry output

### Step 4: Compute ATE

Compare estimated trajectory with ground truth using the `compute_localization_error` function:

1. Align trajectories using Umeyama algorithm
2. Compute RMSE of position errors

### Step 5: Return Fitness

Return both objectives:
- First objective: negative ATE (pymoo minimizes, so we negate to maximize)
- Second objective: perturbation magnitude (Chamfer distance)

## Handling Invalid Solutions

Some perturbations cause MOLA to fail completely:

**Failure cases:**
- Too much dropout: Not enough points for ICP
- Extreme noise: Feature matching fails
- MOLA process crashes

**Detection:**
If MOLA collects fewer than 40 trajectory points, the evaluation retries up to 2 times. If it still fails, the solution returns infinite fitness and is excluded from the Pareto front.

## Baseline Performance

Before optimization, we measure baseline ATE on unperturbed data:

**Baseline ATE: ~23cm**

This represents MOLA's natural localization error on clean Isaac Sim data.

For reference:

| ATE | Interpretation | Quality |
|-----|---------------|---------|
| 0-30cm | Excellent SLAM performance | Professional-grade |
| 30-60cm | Good performance | Acceptable for robotics |
| 60-100cm | Moderate drift | Usable but degraded |
| 100-150cm | Significant drift | Unreliable localization |
| 150cm+ | Severe failure | Unusable |

## Results (Genome 12 - 750 evaluations)

**Optimization Results:**
- Total evaluations: 750
- Valid evaluations: 347 (46%)
- Pareto-optimal solutions: 32

**Best Solutions:**

| Strategy | ATE | Perturbation | Degradation |
|----------|-----|--------------|-------------|
| Aggressive | 85cm | 4.6cm | +269% |
| Balanced | 65cm | 3.5cm | +183% |
| Moderate | 44cm | 2.0cm | +91% |
| Stealthy | 27cm | 1.0cm | +17% |

**Most Effective Parameters:**
- Temporal drift: Strongest correlation with ATE increase
- Scanline perturbation: Second most effective
- Geometric distortion: High impact on ICP alignment

## Efficiency Metric

Beyond absolute ATE, we care about efficiency: how much ATE per unit of perturbation?

**Efficiency = (ATE - baseline_ATE) / perturbation_magnitude**

Example from genome12:
- Solution A: ATE=85cm, pert=4.6cm → efficiency = (85-23)/4.6 = 13.5 cm/cm
- Solution B: ATE=65cm, pert=3.5cm → efficiency = (65-23)/3.5 = 12.0 cm/cm

## Implementation

The fitness evaluator is implemented in `src/optimization/run_nsga3.py` as the `MOLAEvaluator` class, a ROS2 node that:

1. Receives genome from NSGA-III optimizer
2. Decodes genome to perturbation parameters
3. Generates perturbed point cloud sequence
4. Publishes to ROS2 topics
5. Runs MOLA SLAM
6. Collects trajectory and computes ATE
7. Returns fitness tuple (negative ATE, Chamfer distance)

The optimizer uses pymoo's NSGA-III with Das-Dennis reference directions.
