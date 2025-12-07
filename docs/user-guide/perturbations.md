# Perturbation Strategies

## Overview

Adversarial perturbations are small, carefully crafted modifications to LiDAR point clouds that degrade SLAM performance while remaining imperceptible. This document describes the 17-parameter genome encoding and the effectiveness of each perturbation strategy against MOLA SLAM.

The perturbation generator implements state-of-the-art adversarial techniques based on recent research:

- **FLAT**: Flux-Aware Imperceptible Adversarial Attacks (ECCV 2024)
- **SLACK**: Attacking LiDAR-based SLAM (arXiv 2024)
- **ICP Attack**: Adversarial attacks on ICP registration (arXiv 2403.05666)
- **ASP**: Attribution-based Scanline Perturbation (IEEE 2024)

## Attack Objectives

An effective adversarial perturbation must balance two goals:

1. **Maximize localization error** - Cause MOLA to produce inaccurate trajectory estimates
2. **Minimize detectability** - Keep perturbations small enough to avoid detection

The NSGA-III optimization finds perturbations that achieve the best trade-off between these objectives.

## 17-Parameter Genome

The genome encodes 17 continuous parameters in the range [-1, 1], which are then scaled to their respective physical ranges. Each parameter controls a different aspect of the perturbation.

### Basic Perturbations (Parameters 0-6)

#### Parameters 0-2: Noise Direction (3D Vector)

A directional bias applied to all points.

- **Parameter 0**: X component of noise direction
- **Parameter 1**: Y component of noise direction
- **Parameter 2**: Z component of noise direction

The direction vector is normalized and scaled by noise intensity. This creates a systematic drift in a specific direction rather than random noise.

**Why it works:** Directional bias accumulates over time, causing consistent odometry drift that SLAM cannot correct without loop closures.

#### Parameter 3: Noise Intensity

Controls the magnitude of Gaussian noise added to point coordinates.

- **Range**: 0-5cm standard deviation
- **ATE correlation**: +0.370 (strong positive effect)
- **Perturbation correlation**: +0.366 (increases detectability)

**Why it works:** Random noise degrades ICP point-to-plane alignment accuracy. Higher noise means less precise feature matching between consecutive frames.

#### Parameter 4: Curvature Targeting

Targets high-curvature regions (edges, corners) with stronger perturbations.

- **Range**: 0-100% strength
- **ATE correlation**: +0.227 (moderate positive effect)
- **Perturbation correlation**: -0.025 (minimal impact on detectability)

**Why it works:** SLAM systems rely on geometric features (edges, corners, planes) for alignment. Corrupting these features specifically degrades matching quality.

#### Parameter 5: Dropout Rate

Percentage of points randomly removed from each frame.

- **Range**: 0-30%
- **ATE correlation**: +0.055 (weak positive effect)
- **Perturbation correlation**: +0.105 (low detectability impact)

**Why it works:** Removing points reduces feature density, degrading ICP convergence and loop closure detection. However, MOLA is relatively robust to moderate dropout.

#### Parameter 6: Ghost Point Ratio

Adds synthetic points near real measurements.

- **Range**: 0-10% additional points
- **ATE correlation**: -0.086 (slightly negative effect)
- **Perturbation correlation**: +0.110 (increases detectability)

**Why it works:** Ghost points create false correspondences in ICP. However, MOLA's outlier rejection often filters these, making this less effective.

### Cluster Perturbations (Parameters 7-10)

#### Parameters 7-9: Cluster Direction (3D Vector)

Direction for cluster-based perturbations.

- **Parameter 7**: X component (ATE correlation: -0.261)
- **Parameter 8**: Y component (ATE correlation: +0.003)
- **Parameter 9**: Z component (ATE correlation: +0.036)

#### Parameter 10: Cluster Strength

Intensity of cluster-based perturbations.

- **Range**: 0-100%
- **ATE correlation**: +0.071 (weak positive effect)
- **Perturbation correlation**: -0.037 (minimal impact)

**Why it works:** Clusters of points are shifted together, maintaining local structure while introducing global distortion.

### Advanced Perturbations (Parameters 11-16)

#### Parameter 11: Spatial Correlation

Controls how perturbations are correlated spatially.

- **Range**: 0-100%
- **ATE correlation**: -0.167 (negative effect)
- **Perturbation correlation**: -0.070 (reduces detectability)

**Why it works:** Spatially correlated perturbations appear more natural than random noise. However, high correlation may actually help SLAM by preserving local structure.

#### Parameter 12: Geometric Distortion (ICP Attack)

Applies systematic geometric distortions (scaling, shearing).

- **Range**: 0-100%
- **ATE correlation**: +0.357 (strong positive effect)
- **Perturbation correlation**: +0.970 (very high detectability!)

**Why it works:** ICP assumes rigid transformations. Systematic distortions violate this assumption, causing alignment failures. However, this is highly detectable due to large point displacements.

#### Parameter 13: Edge Attack (SLACK-inspired)

Targets edge and corner points with perpendicular shifts.

- **Range**: 0-100%
- **ATE correlation**: +0.033 (weak positive effect)
- **Perturbation correlation**: -0.149 (reduces detectability)

**Why it works:** Shifting edge points perpendicular to their principal direction maximizes ICP confusion while minimizing visible distortion.

#### Parameter 14: Temporal Drift

Accumulating bias across frames over time.

- **Range**: 0-100%
- **ATE correlation**: +0.627 (strongest positive effect!)
- **Perturbation correlation**: +0.237 (moderate detectability)

**Why it works:** Temporal drift is the most effective attack because:
- Bias accumulates consistently over the trajectory
- Prevents loop closure detection (locations appear different over time)
- SLAM cannot correct accumulated drift without recognizing revisited places

#### Parameter 15: Scanline Perturbation (ASP-inspired)

Perturbs points along their laser beam directions.

- **Range**: 0-100%
- **ATE correlation**: +0.595 (second strongest effect!)
- **Perturbation correlation**: +0.392 (moderate-high detectability)

**Why it works:** Perturbations along scanlines simulate realistic sensor interference (dust, particles). This systematically affects range measurements in a physically plausible way.

#### Parameter 16: Strategic Ghost Placement

Places ghost points near geometric features.

- **Range**: 0-100% (activates when > 50%)
- **ATE correlation**: +0.344 (strong positive effect)
- **Perturbation correlation**: -0.026 (minimal detectability impact)

**Why it works:** Ghost points placed near features create ambiguous correspondences that ICP cannot easily reject, unlike random ghost points.

## Effectiveness Rankings

Based on correlation analysis from genome12 results (347 valid evaluations):

### Most Effective for Increasing ATE

| Rank | Parameter | ATE Correlation | Why Effective |
|------|-----------|-----------------|---------------|
| 1 | Temporal Drift | +0.627 | Accumulates over time, breaks loop closure |
| 2 | Scanline Perturbation | +0.595 | Systematic range errors, physically realistic |
| 3 | Noise Intensity | +0.370 | Degrades ICP alignment precision |
| 4 | Geometric Distortion | +0.357 | Violates ICP rigid transformation assumption |
| 5 | Strategic Ghost | +0.344 | Creates ambiguous feature correspondences |


## Defending Against Perturbations

Understanding these attacks informs defense strategies:

1. **Temporal consistency checking**: Detect sudden changes in point cloud statistics between frames
2. **Multi-sensor fusion**: Combine LiDAR with camera or IMU for redundancy
3. **Learned anomaly detection**: Train classifiers to detect adversarial patterns
4. **Robust loop closure**: Use multiple verification methods for place recognition
5. **Range verification**: Cross-check LiDAR ranges with expected environment geometry
