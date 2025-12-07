# Installation

## Prerequisites

- ROS 2 Jazzy
- Python 3.11 or higher
- NVIDIA Isaac Sim
- MOLA SLAM library
- pip package manager
- Git

## Clone the Repository

```bash
git clone https://github.com/francescacraievich/mola-adversarial-nsga3.git
cd mola-adversarial-nsga3
```

## Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

## Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements/requirements.txt
```

The main dependencies are:
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing (for KDTree)
- `pymoo>=0.6.0` - Multi-objective optimization (NSGA-III)
- `matplotlib>=3.7.0` - Visualization
- `rosbags>=0.9.0` - ROS bag file reading
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Code coverage

## Install ROS 2 Dependencies

```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Install MOLA SLAM
sudo apt install ros-jazzy-mola-lidar-odometry

# Install mp2p_icp for trajectory export
# (follow MOLA installation guide)
```

## Install Isaac Sim

Refer to the [NVIDIA Isaac Sim documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) for installation instructions.

Required components:
- Isaac Sim base installation
- Carter robot asset
- LiDAR sensor support

## Verify Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Test Python dependencies
python -c "import numpy; import scipy; import pymoo; print('Python deps OK')"

# Test ROS 2 setup
source /opt/ros/jazzy/setup.bash
ros2 pkg list | grep mola
```



## Next Steps

After installation, proceed to the [Quickstart Guide](quickstart.md) to:
1. Collect data in Isaac Sim
2. Extract point clouds from bag files
3. Run NSGA-III optimization
4. Analyze results
