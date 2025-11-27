# Perturbation Strategies

## Overview

Adversarial perturbations are small, carefully crafted modifications to LiDAR point clouds that degrade SLAM performance while remaining imperceptible. This document describes the different perturbation strategies implemented and their effectiveness against MOLA SLAM.

## Attack Objectives

An effective adversarial perturbation must balance two goals:

1. **Maximize localization error** - Cause MOLA to produce inaccurate trajectory estimates
2. **Minimize detectability** - Keep perturbations small enough to avoid detection

The NSGA-II optimization finds perturbations that achieve the best trade-off between these objectives.

## Perturbation Types

### 1. Dropout Attack

The dropout attack removes a percentage of points from each point cloud frame.

**How it works:**
- Randomly select N% of points in each frame
- Remove these points from the point cloud
- MOLA processes the remaining points

**Parameters:**
- Dropout rate: 0-100% (typically 5-20%)

**Effectiveness:**
- Most efficient attack strategy
- 18.9% ATE increase per cm of perturbation budget
- At 5% dropout: achieves 120cm ATE vs 68cm baseline

**Why it works:**
Dropout breaks SLAM in multiple ways:
- Reduces number of feature correspondences between frames
- Degrades loop closure detection (fewer points to match)
- Increases uncertainty in ICP alignment
- Causes accumulated odometry drift

Removing points is particularly effective because SLAM systems rely on dense, consistent measurements. Missing data creates ambiguity that propagates through the entire mapping process.

**Example:**
```
Original frame: 50,000 points
5% dropout: 47,500 points (2,500 removed)
10% dropout: 45,000 points (5,000 removed)
```

### 2. Gaussian Noise Attack

The Gaussian noise attack adds random displacement to point coordinates.

**How it works:**
- For each point (x, y, z), add random noise: (x + Nx, y + Ny, z + Nz)
- Noise is sampled from Gaussian distribution N(0, σ²)
- Standard deviation σ controls noise magnitude

**Parameters:**
- Standard deviation σ: 0-10cm (typically 1-5cm)

**Effectiveness:**
- Second most efficient strategy
- 15.2% ATE increase per cm of perturbation budget
- At σ=3cm: achieves 105cm ATE vs 68cm baseline

**Why it works:**
Noise degrades SLAM by:
- Reducing precision of feature matching
- Introducing errors in ICP point-to-plane alignment
- Creating inconsistencies between overlapping scans
- Degrading loop closure reliability

Unlike dropout, noise preserves point density but corrupts position accuracy. This is particularly effective against ICP-based odometry, which assumes point measurements are accurate.

**Example:**
```python
# Add Gaussian noise with σ=2cm
noise = np.random.normal(0, 0.02, size=points.shape)
perturbed_points = points + noise
```

### 3. Feature Targeting Attack

The feature targeting attack identifies and perturbs high-gradient regions (edges, corners) that SLAM systems rely on for feature matching.

**How it works:**
1. Compute local point density or gradient for each point
2. Identify high-gradient points (features)
3. Apply larger perturbations to these feature points
4. Apply smaller perturbations to flat regions

**Parameters:**
- Feature threshold: Points above this gradient are considered features
- Feature perturbation: 5-15cm displacement
- Background perturbation: 1-3cm displacement

**Effectiveness:**
- Third most efficient strategy
- 12.7% ATE increase per cm of perturbation budget
- More targeted than uniform noise

**Why it works:**
SLAM systems extract features (edges, corners, planes) for matching and alignment. By corrupting these features specifically, the attack:
- Prevents correct feature correspondence
- Degrades place recognition for loop closure
- Increases ICP alignment errors
- Creates false matches that corrupt the map

This strategy is more sophisticated than uniform noise because it focuses perturbation budget on the most critical points for SLAM.

**Example:**
```python
# Identify features by local density
density = compute_local_density(points, radius=0.5)
is_feature = density > threshold

# Apply larger perturbations to features
perturbations = np.where(is_feature[:, None],
                         large_noise,
                         small_noise)
perturbed_points = points + perturbations
```

### 4. Ghost Points Attack

The ghost points attack adds false points that don't correspond to real objects.

**How it works:**
- Generate synthetic points near real sensor readings
- Add these points to the point cloud
- MOLA processes both real and fake points

**Parameters:**
- Number of ghost points: 100-5000 per frame
- Ghost point distribution: Near real points or in empty space

**Effectiveness:**
- Least efficient strategy
- 8.3% ATE increase per cm of perturbation budget
- Requires more perturbation budget for same ATE

**Why it works:**
Ghost points confuse SLAM by:
- Creating false surfaces and structures
- Introducing spurious feature matches
- Degrading scan matching quality
- Corrupting map consistency

However, this is less efficient than dropout because:
- SLAM systems have outlier rejection mechanisms
- Ghost points in empty space are easily filtered
- Adding points is more detectable than removing or shifting them

**Example:**
```python
# Add ghost points near real points
n_ghosts = 1000
base_indices = np.random.choice(len(points), n_ghosts)
ghost_offset = np.random.normal(0, 0.05, size=(n_ghosts, 3))
ghost_points = points[base_indices] + ghost_offset
perturbed_cloud = np.vstack([points, ghost_points])
```

## Comparative Analysis

Based on experimental results:

| Strategy | ATE/cm Efficiency | Best Use Case | Detectability |
|----------|------------------|---------------|---------------|
| Dropout (5%) | 18.9% | Most efficient general attack | Low (missing data is common) |
| Gaussian noise | 15.2% | When dropout is detectable | Medium (depends on σ) |
| Feature targeting | 12.7% | Targeted attack on features | Medium-High |
| Ghost points | 8.3% | When modifying points is detectable | High (easier to detect additions) |

### Key Insights

1. **Dropout is most efficient**: Removing points causes more damage per unit of perturbation than any other strategy. This is because SLAM fundamentally relies on having complete, dense measurements.

2. **Loop closure is the weak point**: All effective attacks degrade loop closure detection. Without successful loop closures, odometry drift accumulates linearly over time.

3. **Feature-based attacks are less efficient**: While intuitively appealing, targeting features specifically is less efficient than uniform dropout or noise. This suggests MOLA's robustness mechanisms handle feature corruption better than missing data.

4. **Additions are easier to defend against**: Ghost points are the least efficient attack, likely because MOLA has outlier rejection that filters spurious measurements.

## Temporal Patterns

Perturbations can be applied with different temporal patterns:

### Uniform Perturbation
Apply same perturbation to all frames.
- Simple to implement
- Consistent effect throughout trajectory
- Easy to optimize

### Temporal Dropout
Apply perturbations only to specific frames.
- Target loop closure frames specifically
- Minimize overall perturbation budget
- More sophisticated attack

### Adaptive Perturbation
Adjust perturbation based on MOLA's state.
- Increase perturbation when MOLA is uncertain
- Reduce perturbation when MOLA is confident
- Requires online feedback (not implemented)

## Physical Constraints

To maintain realism, perturbations are constrained:

1. **Maximum displacement**: 50cm per point
   - Larger displacements create obvious artifacts
   - Real sensor noise is typically < 5cm

2. **Dropout rate**: 0-100%
   - Cannot remove more than all points
   - Typical rates: 5-20%

3. **Noise standard deviation**: 0-10cm
   - Based on realistic sensor noise characteristics
   - LiDAR accuracy is typically 1-3cm

4. **Ghost point density**: < 10% of real points
   - Too many ghost points are obviously fake
   - Must match real point cloud density

5. **Spatial coherence**: Perturbations should be locally consistent
   - Random per-point perturbations create speckle noise
   - Spatially correlated noise is more realistic

## Implementation Details

Perturbations are applied during NSGA-II fitness evaluation:

1. Load original point cloud frames
2. Decode genome to perturbation parameters
3. Apply perturbation to each frame
4. Write perturbed frames to temporary file
5. Run MOLA SLAM on perturbed data
6. Evaluate ATE and perturbation magnitude
7. Return fitness values to NSGA-II

The perturbation module is in [src/perturbations/](../../src/perturbations/).

### Dropout Implementation

```python
def apply_dropout(cloud, dropout_rate):
    """Remove random points from cloud."""
    n_points = len(cloud)
    n_keep = int(n_points * (1 - dropout_rate))
    keep_indices = np.random.choice(n_points, n_keep, replace=False)
    return cloud[keep_indices]
```

### Gaussian Noise Implementation

```python
def apply_gaussian_noise(cloud, sigma):
    """Add Gaussian noise to point coordinates."""
    noise = np.random.normal(0, sigma, size=cloud.shape)
    return cloud + noise
```

## Defending Against Perturbations

Understanding these attacks informs defense strategies:

1. **Outlier rejection**: Filter points with high residuals
   - Helps against ghost points
   - Less effective against dropout and noise

2. **Multi-sensor fusion**: Combine LiDAR with camera or IMU
   - Provides redundancy against single-sensor attacks
   - Increases attack complexity

3. **Temporal consistency**: Check for frame-to-frame consistency
   - Detects sudden dropout or noise changes
   - Requires buffering multiple frames

4. **Learned anomaly detection**: Train classifier to detect adversarial perturbations
   - Can detect statistical anomalies
   - Requires representative training data

5. **Robust estimation**: Use RANSAC or M-estimators
   - Already implemented in MOLA
   - Provides some inherent robustness

## Future Directions

Potential improvements to perturbation strategies:

1. **Spatially-correlated noise**: Make noise patterns more realistic
2. **Semantic targeting**: Perturb specific object types (walls, obstacles)
3. **Transfer attacks**: Optimize on one SLAM system, test on others
4. **Physical perturbations**: 3D-print adversarial objects to place in environment
5. **Online adaptive attacks**: Adjust perturbations based on SLAM state feedback

## Keyframes vs All Frames

An important consideration is whether to perturb all frames or only keyframes:

**All frames (113 frames at 10Hz):**
- LiDAR publishes at 10Hz during ~11.3 second trajectory
- More data to perturb
- Affects odometry estimation

**Keyframes only (49 frames):**
- MOLA selects keyframes when robot moves enough
- Less data to perturb
- Directly affects what MOLA processes
- More efficient use of perturbation budget

Current implementation perturbs all frames, but future work could target keyframes specifically.

## Summary

- **Dropout is the most efficient attack** (18.9% ATE/cm)
- **Loop closure degradation is the primary damage mechanism**
- **Physical constraints ensure perturbations remain realistic**
- **Different attacks have different detectability vs effectiveness trade-offs**
- **NSGA-II optimization discovers these trade-offs automatically**

The optimization process reveals that simpler attacks (dropout) are more effective than complex ones (feature targeting), which is a valuable insight for both adversarial robustness and SLAM system design.
