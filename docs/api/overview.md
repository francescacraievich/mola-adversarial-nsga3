# API Reference

## Overview

This document provides an API reference for the main modules in the MOLA Adversarial NSGA-III project.

## Modules

### Perturbation Generator

**Location:** `src/perturbations/perturbation_generator.py`

The `PerturbationGenerator` class implements state-of-the-art adversarial perturbation techniques for LiDAR point clouds.

#### Key Methods

| Method | Description |
|--------|-------------|
| `get_genome_size()` | Returns genome size (17 parameters) |
| `encode_perturbation(genome)` | Converts genome to perturbation parameters |
| `apply_perturbation(cloud, params, seed)` | Applies perturbation to point cloud |
| `compute_chamfer_distance(cloud1, cloud2)` | Computes Chamfer distance between clouds |
| `compute_curvature(points)` | Computes local curvature for targeting |
| `detect_edges_and_corners(points)` | Detects edge/corner points (SLACK-inspired) |
| `reset_temporal_state()` | Resets temporal drift state |

#### Genome Structure (17 parameters)

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| 0-2 | Noise direction | [-1, 1] | Directional bias for noise |
| 3 | Noise intensity | [0, 1] | Scaled by `noise_std` |
| 4 | Curvature strength | [0, 1] | High-curvature targeting |
| 5 | Dropout rate | [0, 1] | Scaled by `max_dropout_rate` |
| 6 | Ghost ratio | [0, 1] | Scaled by `max_ghost_points_ratio` |
| 7-9 | Cluster direction | [-1, 1] | Direction for cluster perturbation |
| 10 | Cluster strength | [0, 1] | Cluster perturbation intensity |
| 11 | Spatial correlation | [0, 1] | Correlation of nearby perturbations |
| 12 | Geometric distortion | [0, 1] | ICP attack strength |
| 13 | Edge attack | [0, 1] | SLACK-inspired edge targeting |
| 14 | Temporal drift | [0, 1] | Accumulating drift strength |
| 15 | Scanline perturbation | [0, 1] | ASP-inspired attack |
| 16 | Strategic ghost | [0, 1] | Feature-based ghost placement |

---

### Metrics

**Location:** `src/evaluation/metrics.py`

Functions for computing fitness metrics.

#### Functions

| Function | Description |
|----------|-------------|
| `compute_localization_error(gt, est, method)` | Compute ATE/RPE between trajectories |
| `compute_imperceptibility(orig, pert, method)` | Compute perturbation magnitude |
| `compute_multi_objective_fitness(...)` | Combined fitness for NSGA-III |
| `normalize_fitness(values, ref_point)` | Normalize fitness values |

#### ATE Computation

The `_compute_ate` function implements standard Absolute Trajectory Error:

1. **Umeyama alignment**: Rigid alignment (R + t, no scale) of estimated to ground truth
2. **RMSE**: Root mean squared error of per-pose distances

---

### Data Loaders

**Location:** `src/utils/data_loaders.py`

Functions for loading point clouds and trajectories.

#### Functions

| Function | Description |
|----------|-------------|
| `load_point_clouds_from_npy(path)` | Load point cloud sequence |
| `load_timestamps_from_npy(path)` | Load frame timestamps |
| `load_trajectory_from_tum(path, ...)` | Load trajectory (TUM or NPY format) |

---

### NSGA-III Optimizer

**Location:** `src/optimization/run_nsga3.py`

Main optimization script using pymoo's NSGA-III algorithm.

#### Key Classes

**MOLAEvaluator**: ROS2 node that evaluates genomes by running MOLA SLAM.

#### Command Line Arguments

```bash
python src/optimization/run_nsga3.py \
    --gt-traj maps/ground_truth_interpolated.npy \
    --frames data/frame_sequence.npy \
    --pop-size 10 \
    --n-gen 20 \
    --max-point-shift 0.05 \
    --noise-std 0.015 \
    --max-dropout 0.15 \
    --output src/results/optimized_genome.npy
```

---

### Baseline ATE

**Location:** `src/baseline/baseline_ate.py`

Script for measuring baseline ATE with zero perturbation.

```bash
python src/baseline/baseline_ate.py --num-runs 3
```

Runs MOLA SLAM on unperturbed data to establish the baseline ATE (~23cm).

---

### Preprocessing

#### Extract Frames from Bag

**Location:** `src/preprocessing_data/extract_frames_and_tf_from_bag.py`

Extracts point cloud frames and TF transforms from ROS2 bag files.

```bash
python src/preprocessing_data/extract_frames_and_tf_from_bag.py \
    --bag bags/carter_lidar \
    --lidar-topic /carter/lidar \
    --output-frames data/frame_sequence.npy \
    --output-tf data/tf_sequence.npy \
    --output-tf-static data/tf_static.npy
```

#### Create Frame Sequence

**Location:** `src/preprocessing_data/create_frame_sequence.py`

Creates frame sequences from extracted data.

---

### Plotting

#### Pareto Front Visualization

**Location:** `src/plots/plot_nsga3_results.py`

Visualization of optimization results showing Pareto front.

```bash
python src/plots/plot_nsga3_results.py src/results/optimized_genome12
```

Generates:
- Pareto front plot (ATE vs Chamfer distance)
- Valid evaluations scatter plot
- Baseline reference line
- Best solution statistics

#### Parameter Analysis

**Location:** `src/plots/plot_parameter_dual_correlation.py`

Analysis of genome parameter importance and correlation with fitness.

Generates:
- Parameter importance ranking
- Dual correlation analysis (ATE and Chamfer distance)

#### Trajectory Comparison

**Location:** `src/plots/compare_trajectories.py`

Compares ground truth and estimated trajectories visually.

---

### ROS2 Nodes

#### Add Intensity Node

**Location:** `src/rover_isaacsim/carter_mola_slam/scripts/add_intensity_node.py`

ROS2 node that adds intensity field to point clouds from Isaac Sim.

#### Perturbation Node

**Location:** `src/rover_isaacsim/carter_mola_slam/scripts/perturbation_node.py`

ROS2 node for real-time perturbation injection.

#### Point Cloud Comparison Node

**Location:** `src/rover_isaacsim/carter_mola_slam/scripts/pointcloud_comparison_node.py`

ROS2 node for comparing original and perturbed point clouds.

---

## References

- **NSGA-III**: Deb & Jain (2014) - Reference-point based NSGA
- **SLACK**: arXiv 2024 - Attacking LiDAR-based SLAM
- **ICP Attack**: arXiv 2403.05666 - ICP adversarial perturbations
- **ASP**: IEEE 2024 - Attribution-based Scanline Perturbation
- **FLAT**: ECCV 2024 - Flux-Aware Imperceptible Adversarial Attacks
