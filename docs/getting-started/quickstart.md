# Quick Start

This guide walks you through collecting data in Isaac Sim, running NSGA-III optimization, and analyzing the results.

## Overview

The complete workflow consists of:
1. Data collection in Isaac Sim (simulated robot with LiDAR)
2. Extracting point clouds from ROS2 bag files
3. Exporting ground truth trajectory
4. Running NSGA-III optimization to find adversarial perturbations
5. Analyzing results and visualizing Pareto front

## Prerequisites

Before starting, ensure you have completed the [Installation](installation.md) steps:
- ROS 2 Jazzy installed and sourced
- Python virtual environment activated
- MOLA SLAM installed
- Isaac Sim installed

## Step 1: Data Collection in Isaac Sim

Data collection requires running 5 terminals simultaneously. The robot navigates autonomously while collecting LiDAR data.

### Terminal 1: Isaac Sim
```bash
# Launch Isaac Sim (adjust path to your installation)
~/.local/share/ov/pkg/isaac-sim-4.2.0/isaac-sim.sh
```

In Isaac Sim:
1. Open the scene with the Carter robot
2. Start the simulation

### Terminal 2: ROS2 Bag Recording
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Create bags directory if it doesn't exist
mkdir -p bags

# Record LiDAR topic
ros2 bag record -o bags/carter_lidar /carter/lidar
```

### Terminal 3: Add Intensity Node
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Activate virtual environment
source .venv/bin/activate

# Run intensity node to add intensity field to point clouds
python src/rover_isaacsim/carter_mola_slam/scripts/add_intensity_node.py
```

This node subscribes to `/carter/lidar` and republishes to `/carter/lidar_with_intensity`, adding the intensity field that MOLA expects.

### Terminal 4: MOLA SLAM
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Run MOLA LiDAR odometry
ros2 launch mola_lidar_odometry ros2-lidar-odometry.launch.py \
  lidar_topic_name:=/carter/lidar_with_intensity \
  use_mola_gui:=True
```

MOLA will process the LiDAR scans and estimate the robot's trajectory in real-time.

### Terminal 5: Trajectory Recording
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Record estimated trajectory
ros2 bag record -o bags/mola_trajectory /mola_estimated_trajectory
```

### Stop Recording

After the robot completes its navigation:
1. Stop all ROS2 bag recordings (Ctrl+C)
2. Stop MOLA (Ctrl+C)
3. Stop the intensity node (Ctrl+C)
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
  --lidar-topic /carter/lidar \
  --output-frames data/frame_sequence.npy \
  --output-tf data/tf_sequence.npy \
  --output-tf-static data/tf_static.npy
```

This creates:
- `data/frame_sequence.npy` - List of point cloud arrays (N, 4) with xyzi coordinates
- `data/frame_sequence.timestamps.npy` - Timestamps for each frame in nanoseconds
- `data/tf_sequence.npy` - TF transforms
- `data/tf_static.npy` - Static TF transforms

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

## Step 4: Run NSGA-III Optimization

Now you can run the optimization to find adversarial perturbations:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run NSGA-III (400 evaluations: 20 generations × 20 population)
python src/optimization/run_nsga3.py --n-gen 20 --pop-size 20
```

This will:
1. Load point clouds from `data/frame_sequence.npy`
2. Load timestamps from `data/frame_sequence.timestamps.npy`
3. Load ground truth from `data/ground_truth.tum`
4. Run NSGA-III optimization to find perturbations that maximize ATE
5. Save results to `src/results/optimized_genome.npy`

The optimization takes several hours depending on your hardware.

### Quick Test Run

For testing, use fewer evaluations:
```bash
# 4 evaluations: 2 generations × 2 population
python src/optimization/run_nsga3.py --n-gen 2 --pop-size 2
```

## Step 5: Analyze Results

The optimization saves the Pareto front to `src/results/optimized_genomeN.npy`. Each subsequent run increments the number automatically.

### Visualize Optimization Results

Plot the Pareto front:

```bash
# Activate virtual environment
source .venv/bin/activate

# Plot NSGA-III results
python src/plots/plot_nsga3_results.py src/results/optimized_genome1
```

This creates visualizations showing:
- Pareto front (ATE vs Chamfer distance)
- Valid evaluations scatter plot
- Baseline reference line

### Open in CloudCompare (Optional)

For detailed 3D visualization of point clouds:

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

## Next Steps

After completing the quickstart:
1. Learn about [Perturbation Strategies](../user-guide/perturbations.md)
2. Understand the [Fitness Function](../user-guide/fitness.md)
3. Review [Baseline](../user-guide/baseline.md)
