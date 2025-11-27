# Quick Start

This guide walks you through collecting data in Isaac Sim, running NSGA-II optimization, and analyzing the results.

## Overview

The complete workflow consists of:
1. Data collection in Isaac Sim (simulated robot with LiDAR)
2. Extracting point clouds from ROS2 bag files
3. Running MOLA SLAM to get ground truth trajectory
4. Running NSGA-II optimization to find adversarial perturbations
5. Analyzing results and visualizing point clouds

## Prerequisites

Before starting, ensure you have completed the [Installation](installation.md) steps:
- ROS 2 Jazzy installed and sourced
- Python virtual environment activated
- MOLA SLAM installed
- Isaac Sim installed

## Step 1: Data Collection in Isaac Sim

Data collection requires running 6 terminals simultaneously. The robot navigates autonomously while collecting LiDAR data.

### Terminal 1: Isaac Sim
```bash
# Launch Isaac Sim (adjust path to your installation)
~/.local/share/ov/pkg/isaac-sim-4.2.0/isaac-sim.sh
```

In Isaac Sim:
1. Open the scene with the Carter robot
2. Enable LiDAR sensor
3. Start the simulation

### Terminal 2: ROS2 Bridge
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Launch ROS2 bridge for Isaac Sim
ros2 launch isaac_ros_carter_navigation carter_navigation.launch.py
```

### Terminal 3: ROS2 Bag Recording
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Create bags directory if it doesn't exist
mkdir -p bags

# Record LiDAR topic
ros2 bag record -o bags/carter_lidar /point_cloud
```

### Terminal 4: Add Intensity Node
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Activate virtual environment
source .venv/bin/activate

# Run intensity node to add intensity field to point clouds
python src/rover_isaacsim/carter_mola_slam/scripts/add_intensity_node.py
```

This node subscribes to `/point_cloud` and republishes to `/point_cloud_with_intensity`, adding the intensity field that MOLA expects.

### Terminal 5: MOLA SLAM
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Run MOLA LiDAR odometry
ros2 launch mola_lidar_odometry ros2-lidar-odometry.launch.py \
  input_sensor_topic:=/point_cloud_with_intensity \
  output_trajectory_topic:=/estimated_trajectory
```

MOLA will process the LiDAR scans and estimate the robot's trajectory in real-time.

### Terminal 6: Trajectory Recording
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Record estimated trajectory
ros2 bag record -o bags/mola_trajectory /estimated_trajectory
```

### Stop Recording

After the robot completes its navigation:
1. Stop all ROS2 bag recordings (Ctrl+C in terminals 3 and 6)
2. Stop MOLA (Ctrl+C in terminal 5)
3. Stop the intensity node (Ctrl+C in terminal 4)
4. Stop Isaac Sim simulation

You should now have two bag files:
- `bags/carter_lidar/` - Contains LiDAR point clouds
- `bags/mola_trajectory/` - Contains MOLA's estimated trajectory

## Step 2: Extract Point Clouds from Bag File

Convert ROS2 bag to numpy format for preprocessing:

```bash
# Activate virtual environment
source .venv/bin/activate

# Extract point clouds and transforms
python src/preprocessing_data/extract_frames_and_tf_from_bag.py \
  --bag bags/carter_lidar \
  --topic /point_cloud \
  --output data/frame_sequence.npy \
  --timestamps data/frame_sequence.timestamps.npy \
  --tf-topic /tf \
  --tf-output data/tf_transforms.npy
```

This creates:
- `data/frame_sequence.npy` - List of point cloud arrays (N, 3) with xyz coordinates
- `data/frame_sequence.timestamps.npy` - Timestamps for each frame in nanoseconds
- `data/tf_transforms.npy` - TF transforms for trajectory reconstruction

## Step 3: Export Ground Truth Trajectory

MOLA saves the trajectory in its internal format. Export it to TUM format for evaluation:

```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Export trajectory (adjust paths as needed)
mp2p_icp_log_to_tum \
  --input-log bags/mola_trajectory/trajectory.log \
  --output data/ground_truth.tum
```

The TUM format contains: `timestamp tx ty tz qx qy qz qw`

Alternatively, if MOLA saved the map:
```bash
# Convert MOLA map to trajectory
mp2p_icp_map_to_tum \
  --input-map maps/slam_output.mm \
  --output data/ground_truth.tum
```

## Step 4: Run NSGA-II Optimization

Now you can run the optimization to find adversarial perturbations:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run NSGA-II (400 evaluations: 20 generations × 20 population)
python src/optimization/run_nsga2.py --n-gen 20 --pop-size 20
```

This will:
1. Load point clouds from `data/frame_sequence.npy`
2. Load timestamps from `data/frame_sequence.timestamps.npy`
3. Load ground truth from `data/ground_truth.tum`
4. Run NSGA-II optimization to find perturbations that maximize ATE
5. Save results to `src/results/optimized_genome1.npy`

The optimization takes several hours depending on your hardware. You'll see progress like:
```
Generation 1/20
  Baseline ATE: 0.6827m (68.27cm)
  Best ATE: 1.2341m (123.41cm)
  Best perturbation: 0.0234 m/cm

Generation 2/20
  ...
```

### Quick Test Run

For testing, use fewer evaluations:
```bash
# 4 evaluations: 2 generations × 2 population
python src/optimization/run_nsga2.py --n-gen 2 --pop-size 2
```

## Step 5: Analyze Results

The optimization saves the Pareto front to `src/results/optimized_genomeN.npy`. Each subsequent run increments the number automatically.

### Understanding the Results

The optimization has two objectives:
1. **Maximize ATE** - Make MOLA's trajectory as inaccurate as possible
2. **Minimize perturbation magnitude** - Keep perturbations small and imperceptible

The Pareto front shows the trade-off: you can achieve higher ATE with larger perturbations, but the goal is to find solutions that achieve high ATE with minimal perturbation.

### Key Metrics

From the optimization output:
- **Baseline ATE**: ~68cm (MOLA's natural error on unperturbed data)
- **Best ATE**: The maximum trajectory error achieved
- **Perturbation budget**: Average displacement per point in cm

### Attack Strategy Results

Based on experiments, the most effective attack strategies are:
- **Dropout 5%**: 18.9% ATE/cm (most efficient)
- **Gaussian noise**: 15.2% ATE/cm
- **Feature targeting**: 12.7% ATE/cm
- **Ghost points**: 8.3% ATE/cm

Dropout is the most efficient because removing points:
- Breaks feature correspondences
- Degrades loop closure detection
- Causes accumulated drift over time

## Step 6: Visualize Results

### Visualize Optimization Results

Plot the Pareto front and convergence:

```bash
# Activate virtual environment
source .venv/bin/activate

# Plot NSGA-II results
python src/plots/plot_nsga2_results.py \
  --input src/results/optimized_genome1.npy \
  --output src/results/pareto_front.png
```

This creates visualizations showing:
- Pareto front (ATE vs perturbation magnitude)
- Convergence over generations
- Best solutions found

### Visualize Perturbations

Compare original and perturbed point clouds:

```bash
# Plot perturbation effects
python src/plots/plot_perturbation_results.py \
  --original data/frame_sequence.npy \
  --perturbed data/temp_perturbed.npy \
  --frame 0 \
  --output src/results/perturbation_comparison.png
```

### Open in CloudCompare (Optional)

For detailed 3D visualization:

```bash
# Install CloudCompare if needed
sudo snap install cloudcompare

# Open point clouds (if you have .ply files)
cloudcompare maps/*.ply
```

In CloudCompare:
1. Use point size 2-3 for better visibility
2. Color by intensity or height (Z coordinate)
3. Compare original vs perturbed frames side-by-side

## Understanding Loop Closure

Loop closure is critical for SLAM accuracy. When the robot revisits a previously mapped area:
- MOLA detects the loop by matching current scan with past scans
- It corrects accumulated drift from odometry errors
- The trajectory "snaps" back to align with the previously mapped location

Adversarial perturbations aim to prevent loop closure by:
- Removing distinctive features that enable matching
- Adding noise to degrade correspondence quality
- Creating ghost points that confuse the matcher

Without successful loop closure, odometry drift accumulates linearly over time, causing the trajectory error to grow.

## Common Issues

### MOLA Collects 0 Points

If MOLA fails to process point clouds:
- Check that intensity field is present (use `add_intensity_node.py`)
- Verify point cloud timestamps match MOLA's expected format (nanoseconds)
- Check MOLA logs for errors

The optimization automatically skips invalid solutions by returning infinite fitness.

### High Baseline ATE

If baseline ATE is unexpectedly high (>1m):
- Check ground truth and estimated trajectory alignment
- Verify they use the same coordinate frame
- Consider recalibrating MOLA parameters
- Visualize trajectories to identify systematic offset

### Out of Memory

If optimization runs out of memory:
- Reduce population size: `--pop-size 10`
- Reduce number of generations: `--n-gen 10`
- Process fewer frames (modify data loader)

## Next Steps

After completing the quickstart:
1. Read [NSGA-II Algorithm](../user-guide/nsga2.md) to understand the optimization
2. Learn about [Perturbation Strategies](../user-guide/perturbations.md)
3. Understand the [Fitness Function](../user-guide/fitness.md)
4. Review [Baseline Performance](../results/baseline.md) expectations

## Summary of Key Commands

```bash
# 1. Data collection (6 terminals - see above)

# 2. Extract point clouds and transforms
python src/preprocessing_data/extract_frames_and_tf_from_bag.py \
  --bag bags/carter_lidar \
  --topic /point_cloud \
  --output data/frame_sequence.npy \
  --timestamps data/frame_sequence.timestamps.npy \
  --tf-topic /tf \
  --tf-output data/tf_transforms.npy

# 3. Export ground truth
mp2p_icp_log_to_tum \
  --input-log bags/mola_trajectory/trajectory.log \
  --output data/ground_truth.tum

# 4. Run optimization (400 evaluations)
python src/optimization/run_nsga2.py --n-gen 20 --pop-size 20

# 5. Quick test (4 evaluations)
python src/optimization/run_nsga2.py --n-gen 2 --pop-size 2

# 6. Visualize results
python src/plots/plot_nsga2_results.py \
  --input src/results/optimized_genome1.npy \
  --output src/results/pareto_front.png
```
