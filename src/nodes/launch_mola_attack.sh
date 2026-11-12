#!/usr/bin/env bash
# Launch mola-cli with the configuration shared by baseline and attack runs.
#
# MOLA reads its configuration from environment variables and silently falls back
# to defaults for missing ones. This script is the single source of those
# variables, so baseline and attack run the same SLAM. Main ones:
#
#   MOLA_ODOMETRY_PIPELINE_YAML   GICP pipeline with optimize_twist; the default
#                                 pipeline does not enable it (the start-up banner
#                                 must say "with optimize_twist")
#   use_sim_time                  Isaac Sim publishes sim time; with wall clock the
#                                 timestamps are not monotonic and scans are dropped
#   MOLA_SIGMA_MAX_MOTION         Cov2Cov matching threshold; with the default
#                                 (0.50) GICP pairs symmetric copies of the corridor
#   ENFORCE_PLANAR_MOTION         3DOF on a flat floor instead of 6DOF
#
# Usage:
#   ./src/nodes/launch_mola_attack.sh                              # /carter/lidar_perturbed
#   ./src/nodes/launch_mola_attack.sh /carter/lidar_with_intensity # bypass the attack
#   LOCAL_MAP=8 ./src/nodes/launch_mola_attack.sh                  # smaller local map
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HERE/../../config/mola"

LIDAR_TOPIC="${1:-/carter/lidar_perturbed}"
LOCAL_MAP="${LOCAL_MAP:-25.0}"
TRACES="${TRACES:-}"

export MOLA_LIDAR_TOPIC="$LIDAR_TOPIC"
export MOLA_LIDAR_NAME="lidar"
export MOLA_WITH_GUI="false"

# Fixed LiDAR pose on the base. The translation can be overridden from the
# environment to check MOLA's sign convention with an A/B comparison.
export MOLA_USE_FIXED_LIDAR_POSE="true"
export LIDAR_POSE_X="${LIDAR_POSE_X:--0.2317}"
export LIDAR_POSE_Y="${LIDAR_POSE_Y:-0}"
export LIDAR_POSE_Z="${LIDAR_POSE_Z:-0.526}"
export LIDAR_POSE_YAW="0"
export LIDAR_POSE_PITCH="0"
export LIDAR_POSE_ROLL="0"

# optimize_twist pipeline: estimates linear and angular velocity inside ICP, without
# IMU, and does not jump in symmetric corridors when the external motion model fails.
export MOLA_ODOMETRY_PIPELINE_YAML="$CONFIG_DIR/lidar3d-gicp-optimize-twist-local.yaml"
export MOLA_OPTIMIZE_TWIST="true"
export MOLA_OPTIMIZE_TWIST_MAX_CORRECTIONS="4"

export MOLA_CLOUD_DECIMATION_VOXEL_SIZE="0.15"
export MOLA_DECIMATED_POINTS_ICP="2500"
export MOLA_DECIMATED_POINTS_MAP="4000"

export MOLA_LOCALIZ_USE_REP105="false"
export MOLA_FORWARD_ROS_TF_ODOM_TO_MOLA="false"

# Cov2Cov matching threshold.
export MOLA_SIGMA_MAX_MOTION="${MOLA_SIGMA_MAX_MOTION:-0.25}"

# Initial sigma.
export MOLA_INITIAL_SIGMA="${MOLA_INITIAL_SIGMA:-0.20}"

# Flat floor: estimating z, pitch and roll only adds error.
export MOLA_NAVSTATE_ENFORCE_PLANAR_MOTION="true"

# Constant keyframe thresholds (m, deg) instead of the expression based on estimated
# angular velocity and range, which collapses on a straight line and makes every
# scan a keyframe.
export MOLA_MIN_XYZ_BETWEEN_MAP_UPDATES="0.20"
export MOLA_MIN_ROT_BETWEEN_MAP_UPDATES="10"

# Local map (m) much larger than the path: the estimate is re-anchored and drift
# does not accumulate. Much smaller: pure odometry, drift accumulates.
export MOLA_LOCAL_MAP_MAX_SIZE="$LOCAL_MAP"

if [[ -n "$TRACES" ]]; then
  export MOLA_SAVE_DEBUG_TRACES="true"
  export MOLA_DEBUG_TRACES_FILE="$TRACES"
fi

echo "─────────────────────────────────────────────────────────────"
echo "  topic LiDAR : $MOLA_LIDAR_TOPIC"
echo "  local map   : $MOLA_LOCAL_MAP_MAX_SIZE m"
echo "  sigma       : iniziale $MOLA_INITIAL_SIGMA, max $MOLA_SIGMA_MAX_MOTION"
echo "  lidar pose  : ($LIDAR_POSE_X, $LIDAR_POSE_Y, $LIDAR_POSE_Z)"
echo "  pipeline    : $(basename "$MOLA_ODOMETRY_PIPELINE_YAML")"
echo "  tracce      : ${TRACES:-(disattivate)}"
echo "─────────────────────────────────────────────────────────────"
echo "  Controlla che il riquadro all'avvio dica 'with optimize_twist'."
echo ""

exec mola-cli "$CONFIG_DIR/lidar_odometry_ros2_local.yaml" \
     --ros-args -p use_sim_time:=true
