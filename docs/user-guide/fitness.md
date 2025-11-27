# Fitness Evaluation

## Overview

Fitness evaluation is the core of the NSGA-II optimization process. For each candidate solution (genome), we need to measure how well it achieves our two objectives: maximizing localization error while minimizing perturbation magnitude.

This evaluation happens hundreds of times during optimization (400 evaluations = 20 generations × 20 population), making it the most computationally expensive component of the system.

## Two Objectives

### Objective 1: Maximize Localization Error (ATE)

The primary goal is to degrade MOLA's localization accuracy as much as possible. We measure this using Absolute Trajectory Error (ATE).

**ATE Definition:**
ATE is the root mean square error (RMSE) between the estimated trajectory and ground truth trajectory after optimal alignment.

**Formula:**
```
ATE = sqrt(1/N * Σ ||p_i - q_i||²)
```

Where:
- N = number of trajectory poses
- p_i = ground truth position at pose i
- q_i = estimated position at pose i
- ||·|| = Euclidean distance (L2 norm)

**Interpretation:**
- Higher ATE = More trajectory error = Better attack
- Baseline ATE ≈ 68cm (MOLA's natural error on unperturbed data)
- Target ATE ≈ 120-150cm (after perturbations)

NSGA-II tries to **maximize** this objective.

### Objective 2: Minimize Perturbation Magnitude

The secondary goal is to keep perturbations as small as possible to avoid detection.

**Perturbation Magnitude:**
We measure the average displacement applied to points across all frames.

**Formula:**
```
Perturbation = (1/M) * Σ ||δ_j||
```

Where:
- M = total number of perturbed points
- δ_j = displacement vector for point j
- ||·|| = Euclidean distance (L2 norm)

**Special case for dropout:**
When points are removed (dropout), we count the missing point as having infinite displacement for measurement purposes, but use the dropout rate as a proxy:
```
Perturbation (dropout) = dropout_rate * average_point_spacing
```

**Interpretation:**
- Lower perturbation = More stealthy = Better
- Typical perturbations: 1-15cm average displacement
- Physical constraint: < 50cm maximum displacement per point

NSGA-II tries to **minimize** this objective.

## Evaluation Pipeline

For each genome in the population, the fitness evaluation follows these steps:

### Step 1: Load Data

```python
# Load point cloud frames
frames = np.load('data/frame_sequence.npy', allow_pickle=True)  # 113 frames

# Load timestamps
timestamps = np.load('data/frame_sequence.timestamps.npy')  # nanoseconds

# Load ground truth trajectory
ground_truth = load_trajectory_from_tum('data/ground_truth.tum')
```

### Step 2: Decode Genome

The genome encodes the perturbation strategy and parameters:

```python
# Example genome: [strategy_id, param1, param2, ...]
# strategy_id: 0=dropout, 1=gaussian, 2=feature, 3=ghost
# params depend on strategy

strategy = int(genome[0])
if strategy == 0:  # Dropout
    dropout_rate = genome[1]  # 0-100%
elif strategy == 1:  # Gaussian noise
    noise_sigma = genome[1]  # 0-10cm
# ... etc
```

### Step 3: Apply Perturbations

Apply the decoded perturbations to all frames:

```python
perturbed_frames = []
total_perturbation = 0.0

for frame in frames:
    perturbed_frame, perturbation_amount = apply_perturbation(
        frame, strategy, params
    )
    perturbed_frames.append(perturbed_frame)
    total_perturbation += perturbation_amount

avg_perturbation = total_perturbation / len(frames)
```

### Step 4: Write Temporary Files

MOLA needs data in ROS2 format, so we write perturbed frames to temporary files:

```python
# Write perturbed frames to .npy
temp_frames = 'data/temp_perturbed.npy'
np.save(temp_frames, perturbed_frames)

# Copy timestamps (unchanged)
temp_timestamps = 'data/temp_perturbed.timestamps.npy'
shutil.copy('data/frame_sequence.timestamps.npy', temp_timestamps)
```

### Step 5: Run MOLA SLAM

Launch MOLA to process the perturbed point clouds:

```python
# Create ROS2 node
ros2 = subprocess.Popen(['ros2', 'launch', 'mola_lidar_odometry', ...])

# Publish perturbed frames with timestamps
for frame, timestamp in zip(perturbed_frames, timestamps):
    publish_pointcloud(frame, timestamp)
    time.sleep(0.1)  # 10Hz

# Wait for MOLA to complete
ros2.wait(timeout=60)
```

MOLA outputs an estimated trajectory.

### Step 6: Extract Estimated Trajectory

Read MOLA's output trajectory:

```python
# MOLA writes trajectory to file
estimated_traj = load_trajectory_from_tum('maps/estimated_trajectory.tum')

# Check if MOLA succeeded
if len(estimated_traj) < 10:
    # MOLA failed - return invalid fitness
    return (np.inf, np.inf)
```

If MOLA collects fewer than 10 poses, something went wrong (likely too much dropout or noise). We return infinite fitness to exclude this solution.

### Step 7: Align Trajectories

Before computing ATE, we need to align the trajectories:

```python
from scipy.spatial.transform import Rotation

def align_trajectories(estimated, ground_truth):
    """Align estimated trajectory to ground truth using Umeyama algorithm."""
    # Compute optimal rotation and translation
    R, t, scale = umeyama_alignment(estimated, ground_truth)

    # Apply transformation
    aligned = scale * (estimated @ R.T) + t

    return aligned
```

This accounts for:
- Different coordinate frames
- Different starting positions
- Scale differences (usually scale=1 for MOLA)

### Step 8: Compute ATE

Calculate the trajectory error:

```python
def compute_ate(estimated, ground_truth):
    """Compute Absolute Trajectory Error (RMSE)."""
    # Align trajectories first
    aligned = align_trajectories(estimated, ground_truth)

    # Ensure same length (interpolate if needed)
    if len(aligned) != len(ground_truth):
        aligned = interpolate_trajectory(aligned, len(ground_truth))

    # Compute RMSE
    squared_errors = np.sum((aligned - ground_truth) ** 2, axis=1)
    ate = np.sqrt(np.mean(squared_errors))

    return ate
```

### Step 9: Return Fitness

Return both objectives to NSGA-II:

```python
fitness = (
    -ate,  # Negative because pymoo minimizes (we want to maximize ATE)
    avg_perturbation  # Minimize perturbation
)
return fitness
```

Note: pymoo minimizes all objectives by default, so we negate ATE to convert maximization to minimization.

## Handling Invalid Solutions

Some perturbations cause MOLA to fail completely:

**Failure cases:**
- 100% dropout: No points to process
- Extreme noise: All points displaced far from original positions
- Too many ghost points: Memory overflow

**Handling strategy:**
Return infinite fitness for both objectives:

```python
if mola_failed or len(estimated_traj) < 10:
    return (np.inf, np.inf)
```

NSGA-II automatically excludes these solutions from the Pareto front. This is better than using a penalty value (like 50.0) because penalties can distort the Pareto front.

## Performance Optimization

Fitness evaluation is expensive (several minutes per evaluation). Optimizations:

### 1. Caching Baseline

Compute baseline ATE once and reuse:

```python
if not hasattr(self, '_baseline_ate'):
    self._baseline_ate = evaluate_unperturbed()
```

### 2. Early Termination

If MOLA shows signs of failing, terminate early:

```python
if len(collected_points) == 0:
    # MOLA collected nothing - stop early
    terminate_mola()
    return (np.inf, np.inf)
```

### 3. Parallel Evaluation

Future work: Evaluate multiple genomes in parallel using multiprocessing.

Currently not implemented because MOLA uses ROS2, which complicates parallelization.

## Metrics Tracking

During optimization, we track several metrics:

```python
# Best ATE achieved so far
best_ate = max(population, key=lambda x: -x.F[0])

# Most efficient solution (highest ATE per cm)
efficiency = [-f[0] / f[1] for f in population.F]
best_efficiency = max(efficiency)

# Pareto front size
pareto_size = len(population[population.rank == 0])
```

These metrics help monitor convergence and identify the most effective attacks.

## Baseline Performance

Before optimization, we evaluate baseline performance on unperturbed data:

```python
baseline_ate = evaluate_unperturbed()
print(f"Baseline ATE: {baseline_ate:.4f}m ({baseline_ate*100:.2f}cm)")
```

Expected baseline:
- Ideal conditions: 30-50cm
- Realistic conditions: 60-80cm
- Poor conditions: 100cm+

Our baseline is ~68cm, which suggests:
- Ground truth may have slight misalignment
- MOLA parameters may not be optimal
- Natural drift accumulation without perfect loop closure

## Understanding ATE Values

What do different ATE values mean?

| ATE | Interpretation | Quality |
|-----|---------------|---------|
| 0-30cm | Excellent SLAM performance | Professional-grade |
| 30-60cm | Good performance | Acceptable for robotics |
| 60-100cm | Moderate drift | Usable but degraded |
| 100-150cm | Significant drift | Unreliable localization |
| 150cm+ | Severe failure | Unusable |

Our goal is to push ATE from ~68cm (baseline) to 120-150cm (severely degraded).

## Efficiency Metric

Beyond absolute ATE, we care about efficiency: how much ATE per unit of perturbation?

**Efficiency = (ATE - baseline_ATE) / perturbation_magnitude**

Example:
- Solution A: ATE=120cm, pert=5cm → efficiency = (120-68)/5 = 10.4%/cm
- Solution B: ATE=140cm, pert=10cm → efficiency = (140-68)/10 = 7.2%/cm

Solution A is more efficient despite lower absolute ATE.

This metric reveals that **dropout at 5% is most efficient** (18.9%/cm).

## Debugging Failed Evaluations

If many evaluations return infinite fitness:

1. **Check MOLA logs**: Look for errors or warnings
2. **Visualize perturbations**: Ensure they're reasonable
3. **Reduce perturbation magnitude**: May be too aggressive
4. **Check timestamps**: MOLA requires proper timestamp synchronization
5. **Verify data files**: Ensure frames and timestamps are valid

Common issues:
- Missing intensity field in point clouds (use `add_intensity_node.py`)
- Timestamp misalignment (MOLA expects nanoseconds)
- Extreme perturbations that violate physical constraints

## Implementation

The fitness evaluator is implemented in [src/optimization/run_nsga2.py](../../src/optimization/run_nsga2.py) as a pymoo Problem:

```python
class AdversarialPerturbationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=genome_length,
            n_obj=2,  # ATE and perturbation
            n_constr=0,
            xl=lower_bounds,
            xu=upper_bounds
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x is the genome
        ate, perturbation = evaluate_genome(x)

        # Return objectives (both minimization)
        out["F"] = [-ate, perturbation]
```

## Summary

Fitness evaluation measures two competing objectives:

1. **ATE (maximize)**: How much MOLA's trajectory deviates from ground truth
2. **Perturbation (minimize)**: Average displacement applied to points

The evaluation pipeline:
1. Apply perturbations to point clouds
2. Run MOLA SLAM
3. Extract estimated trajectory
4. Align with ground truth
5. Compute ATE
6. Return fitness to NSGA-II

Key considerations:
- Invalid solutions return infinite fitness
- Baseline ATE is ~68cm
- Target ATE is 120-150cm
- Most efficient attack: dropout at 5% (18.9%/cm)
- Evaluation takes several minutes per genome
