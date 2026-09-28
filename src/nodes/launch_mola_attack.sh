#!/usr/bin/env bash
# Lancia mola-cli con la configurazione usata sia dalla baseline sia dall'attacco.
#
# MOLA legge la configurazione da variabili d'ambiente e, per quelle mancanti,
# usa il default senza segnalarlo. Questo script e' l'unica sorgente di tali
# variabili, cosi' baseline e attacco girano con lo stesso SLAM. Le principali:
#
#   MOLA_ODOMETRY_PIPELINE_YAML   pipeline GICP con optimize_twist; con il
#                                 default optimize_twist non e' attivo (il
#                                 riquadro all'avvio deve dire "with optimize_twist")
#   use_sim_time                  Isaac Sim pubblica sim-time; con il wall-clock
#                                 i timestamp non sono monotoni e gli scan
#                                 vengono scartati
#   MOLA_SIGMA_MAX_MOTION         soglia di matching Cov2Cov; con il default
#                                 (0.50) GICP accoppia copie simmetriche del
#                                 corridoio
#   ENFORCE_PLANAR_MOTION         3DOF su pavimento piatto invece di 6DOF
#
# Uso:
#   ./src/nodes/launch_mola_attack.sh                              # /carter/lidar_perturbed
#   ./src/nodes/launch_mola_attack.sh /carter/lidar_with_intensity # bypassa l'attacco
#   LOCAL_MAP=8 ./src/nodes/launch_mola_attack.sh                  # mappa locale ridotta
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HERE/../../config/mola"

LIDAR_TOPIC="${1:-/carter/lidar_perturbed}"
LOCAL_MAP="${LOCAL_MAP:-25.0}"
TRACES="${TRACES:-}"

export MOLA_LIDAR_TOPIC="$LIDAR_TOPIC"
export MOLA_LIDAR_NAME="lidar"
export MOLA_WITH_GUI="false"

# Posa del LiDAR rispetto a base_link, dal /tf della scena:
#   base_link -> XT_32 = (-0.2317, 0, 0.5260), rotazione identita'
# Le componenti di traslazione sono sovrascrivibili da env per verificare la
# convenzione di segno di MOLA con un confronto A/B.
export MOLA_USE_FIXED_LIDAR_POSE="true"
export LIDAR_POSE_X="${LIDAR_POSE_X:--0.2317}"
export LIDAR_POSE_Y="${LIDAR_POSE_Y:-0}"
export LIDAR_POSE_Z="${LIDAR_POSE_Z:-0.526}"
export LIDAR_POSE_YAW="0"
export LIDAR_POSE_PITCH="0"
export LIDAR_POSE_ROLL="0"

# Pipeline con optimize_twist: stima velocita' lineare e angolare dentro ICP,
# senza IMU, e non salta nei corridoi simmetrici quando il modello di velocita'
# esterno fallisce.
export MOLA_ODOMETRY_PIPELINE_YAML="$CONFIG_DIR/lidar3d-gicp-optimize-twist-local.yaml"
export MOLA_OPTIMIZE_TWIST="true"
export MOLA_OPTIMIZE_TWIST_MAX_CORRECTIONS="4"

export MOLA_CLOUD_DECIMATION_VOXEL_SIZE="0.15"
export MOLA_DECIMATED_POINTS_ICP="2500"
export MOLA_DECIMATED_POINTS_MAP="4000"

export MOLA_LOCALIZ_USE_REP105="false"
export MOLA_FORWARD_ROS_TF_ODOM_TO_MOLA="false"

# Soglia di matching Cov2Cov: 2.0 x ADAPTIVE_THRESHOLD_SIGMA. A 0.25 il massimo
# e' 0.50 m: basta per il moto fra scan (~0.08 m a 0.8 m/s e 10 Hz) ma non per
# accoppiare copie simmetriche del corridoio.
export MOLA_SIGMA_MAX_MOTION="${MOLA_SIGMA_MAX_MOTION:-0.25}"

# Sigma iniziale. Si adatta verso il regime (~0.05) al 5% per scan, cioe' in
# circa 60 scan: nei rollout brevi dell'attacco, in cui MOLA riparte ogni volta,
# il transitorio coprirebbe l'intera misura. Sovrascrivibile da env per partire
# vicino al regime senza consumare metri di riscaldamento.
export MOLA_INITIAL_SIGMA="${MOLA_INITIAL_SIGMA:-0.20}"

# Pavimento piatto: stimare z, pitch e roll aggiunge solo errore.
export MOLA_NAVSTATE_ENFORCE_PLANAR_MOTION="true"

# Soglie keyframe costanti invece dell'espressione dipendente da velocita'
# angolare stimata e range, che in rettilineo collassa e rende ogni scan un
# keyframe.
export MOLA_MIN_XYZ_BETWEEN_MAP_UPDATES="0.20"
export MOLA_MIN_ROT_BETWEEN_MAP_UPDATES="10"

# Mappa locale molto piu' estesa del percorso: stima riancorata, il drift non
# accumula. Molto meno estesa: odometria pura, il drift accumula.
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
