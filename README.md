# Multi-Objective Adversarial Perturbations for SLAM Systems using NSGA-III

[![CI](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/ci.yml/badge.svg)](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/ci.yml)
[![Documentation](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/docs.yml/badge.svg)](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://francescacraievich.github.io/mola-adversarial-nsga3/)
[![codecov](https://codecov.io/github/francescacraievich/mola-adversarial-nsga3/graph/badge.svg?token=BQX8LWJMSJ)](https://codecov.io/github/francescacraievich/mola-adversarial-nsga3)

Documentation available at: [documentation](https://francescacraievich.github.io/mola-adversarial-nsga3/)

Evolutionary multi-objective optimization of adversarial perturbations on LiDAR point clouds to evaluate the robustness of SLAM systems.

## Overview

This project uses **NSGA-III** (Non-dominated Sorting Genetic Algorithm III) to generate adversarial perturbations on LiDAR point clouds that compromise SLAM systems. The algorithm optimizes a trade-off between:

- **Attack Effectiveness**: Maximize localization error (ATE - Absolute Trajectory Error)
- **Imperceptibility**: Minimize the magnitude of perturbations

### Key Findings

NSGA-III optimization (750 evaluations) reveals a Pareto front with multiple attack strategies:
- **Baseline ATE**: 23cm (unperturbed SLAM accuracy)
- **Best attack**: 85cm drift with 4.6cm perturbation (**+269% degradation**)
- **Balanced sweet spot**: 65cm drift with 3.5cm perturbation (+183% degradation)
- **Stealthy attack**: 32cm drift with 1.5cm perturbation (+39% degradation)

Perturbations remain below sensor noise levels while causing significant SLAM degradation.

### System Integration

The system integrates **MOLA SLAM** with **NVIDIA Isaac Sim** using ROS2, enabling:
- Real-time perturbation injection into LiDAR point clouds
- Automated fitness evaluation by comparing perturbed vs baseline trajectories
- Pareto front exploration for optimal attack/imperceptibility trade-offs

## Features

- Multi-objective optimization using NSGA-III with reference points
- Real-time LiDAR perturbation via ROS2 nodes
- Automated trajectory comparison and ATE computation
- Comprehensive test suite 
- Visualization tools for trajectory comparison and Pareto fronts
- Comparison with baseline approaches (random perturbations, grid search)

## Installation

### Prerequisites

- Python 3.10+
- ROS2 Jazzy 
- NVIDIA Isaac Sim 5.0
- MOLA SLAM binaries

### Setup

```bash
# Clone repository
git clone https://github.com/francescacraievich/mola-adversarial-nsga3.git
cd mola-adversarial-nsga3

# Install Python dependencies
pip install -r requirements.txt

```

## Understanding the Optimization

### Fitness Functions

The NSGA-III algorithm optimizes two competing objectives:

1. **f1 = -ATE (Absolute Trajectory Error)**
   - Measures localization accuracy degradation
   - Computed as RMSE between MOLA trajectory and ground truth
   - Negated because we want to maximize error (minimize fitness)
   - Formula: `ATE = sqrt(mean((x_est - x_gt)^2 + (y_est - y_gt)^2 + (z_est - z_gt)^2))`

2. **f2 = perturbation_magnitude**
   - Measures imperceptibility of the attack
   - Computed using Chamfer distance between original and perturbed clouds
   - Want to minimize this to keep perturbations stealthy
   - Formula: `Chamfer = mean(min_distance(perturbed → original)) + mean(min_distance(original → perturbed))`

### NSGA-III vs NSGA-II

This project uses **NSGA-III** (not NSGA-II) because:

| Feature | NSGA-II | NSGA-III |
|---------|---------|----------|
| Objectives | 2-3 | Many (3+) |
| Selection | Crowding distance | Reference points |
| Diversity | Spread along front | Structured exploration |
| Our use case | Could work | Better for future extension to 3+ objectives |

We currently optimize 2 objectives (ATE, imperceptibility), but NSGA-III allows future expansion to optimize additional objectives like:
- Computational cost
- Physical realizability
- Detection probability

### Perturbation Types

The genome encodes **17 parameters** combining multiple attack strategies:

**Basic perturbations (7 params):**
1-3. **Directional bias** (3D vector for systematic drift)
4. **Noise intensity** (Gaussian noise scale [0, 5cm])
5. **Curvature targeting** (weight high-curvature regions)
6. **Dropout rate** (point removal [0, 3%])
7. **Ghost ratio** (fake points [0, 2%])

**Cluster attack (4 params):**
8-10. **Cluster direction** (3D vector for localized errors)
11. **Cluster strength** (magnitude of cluster displacement)

**Advanced attacks (6 params):**
12. **Spatial correlation** (coherent noise patterns)
13. **Geometric distortion** (range/angle/scale distortion - **KEY for ICP**)
14. **Edge attack** (target critical features - **SLACK-inspired**)
15. **Temporal drift** (accumulating bias - **breaks loop closure**)
16. **Scanline perturbation** (along-beam shifts - **ASP-inspired**)
17. **Strategic ghost** (place ghosts near features - **SLACK-inspired**)



## Project Structure

```
mola-adversarial-nsga3/
├── src/
│   ├── optimization/
│   │   └── run_nsga3.py              # NSGA-III optimizer with MOLAEvaluator
│   ├── perturbations/
│   │   └── perturbation_generator.py # LiDAR perturbation generation
│   ├── evaluation/
│   │   └── metrics.py                # ATE and Chamfer distance computation
│   ├── utils/
│   │   └── data_loaders.py           # Load point clouds, trajectories, timestamps
│   ├── baseline/
│   │   └── baseline_ate.py           # Baseline ATE measurement
│   ├── preprocessing_data/
│   │   ├── extract_frames_and_tf_from_bag.py  # Extract data from ROS bags
│   │   └── create_frame_sequence.py  # Create frame sequences
│   ├── plots/
│   │   ├── compare_trajectories.py   # Trajectory visualization
│   │   ├── plot_nsga3_results.py     # Pareto front visualization
│   │   └── plot_parameter_dual_correlation.py  # Parameter analysis
│   ├── rover_isaacsim/
│   │   └── carter_mola_slam/
│   │       └── scripts/
│   │           ├── add_intensity_node.py      # ROS2 node: add intensity
│   │           ├── perturbation_node.py       # ROS2 node: perturbation
│   │           └── pointcloud_comparison_node.py  # Compare point clouds
│   ├── results/                      # Optimization outputs and plots
│   └── tests/
│       ├── test_mola_evaluator.py    # MOLAEvaluator tests
│       ├── test_perturbation_generator.py
│       ├── test_data_loaders.py
│       ├── test_metrics.py
│       └── test_nsga3_optimization.py
├── data/
│   ├── maps/                         # Ground truth trajectories
│   └── trajectory_recordings/        # Saved trajectories (baseline, perturbed)
├── bags/                             # ROS bag recordings (not tracked)
└── docs/                             # MkDocs documentation
```

## Testing

The project includes a comprehensive test suite with 31 tests covering critical components:

### Test Coverage

- **Voxel downsampling** (7 tests): Pure numpy logic, performance-critical
- **Trajectory collection** (4 tests): State management and odometry processing
- **MOLA lifecycle** (8 tests): Subprocess management and signal handling
- **Edge cases** (8 tests): Empty arrays, malformed messages, process crashes
- **Fixtures** (4): Reusable mock objects for ROS2 components

### Running Tests

```bash
# Run all tests
pytest src/tests/

# Run with coverage report
pytest src/tests/ --cov=src --cov-report=html

# Run only MOLAEvaluator tests
pytest src/tests/test_mola_evaluator.py -v
```

## Results

### Optimization Results (Genome 12 - 750 evaluations)

NSGA-III evolved **32 Pareto-optimal solutions** from 750 evaluations (347 valid):

| Strategy | ATE (cm) | Perturbation (cm) | Effectiveness | Use Case |
|----------|----------|-------------------|---------------|----------|
| **Aggressive** | **85** | **4.6** | **+269%** | **Maximum degradation** ✓ |
| Balanced | 65 | 3.5 | +183% | Good trade-off |
| Moderate | 44 | 2.0 | +91% | Lower imperceptibility |
| Stealthy | 27 | 1.0 | +17% | Minimal perturbation |
| *Baseline* | *23* | *0* | *0%* | *No attack* |

**Key findings:**
- **Baseline ATE**: 23cm (unperturbed MOLA SLAM)
- **Best attack**: 85cm with 4.6cm perturbation (+269% degradation)
- **Most effective parameters**: Temporal drift, Scanline perturbation
- **32 Pareto-optimal solutions** spanning the trade-off space

### Experimental Workflow

The complete experimental pipeline consists of:

1. **Offline optimization**: NSGA-III evolves perturbation genomes (10-50 generations)
2. **Real-time testing**: Best genomes tested in Isaac Sim with ROS2 integration
3. **Trajectory comparison**: ATE computed between baseline and perturbed runs
4. **Visualization**: Pareto fronts and trajectory plots for analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Francesca Craievich
