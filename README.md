# mola-adversarial-nsga3

Adversarial perturbations of LiDAR point clouds against the MOLA LiDAR-odometry
pipeline, optimised with NSGA-III in a closed loop: the perturbed scans drive the
SLAM estimate, the estimate drives the robot, and the physical deviation of the
robot is the objective. Simulation in NVIDIA Isaac Sim (Nova Carter, Hesai XT-32
in a warehouse scene), SLAM with MOLA on ROS 2 Jazzy.

Master's thesis work, University of Trieste, 2026.

## How it works

```
Isaac Sim ──/front_3d_lidar/lidar_points──▶ add_intensity_node
          ──/carter/lidar_with_intensity──▶ perturbation_node   (genome → 9 operators, every scan)
          ──/carter/lidar_perturbed──────▶ MOLA (mola-cli)      ──/lidar_odometry/pose──▶ controller
Isaac Sim ◀──────────────────────────────/cmd_vel───────────────────────────────────────┘
```

The controller reads only the MOLA estimate. The true pose (`/chassis/odom`,
Isaac state) is used for measurement only.

The attack runs as a receding horizon. For every window of H metres the
orchestrator saves the simulator state, runs one nominal rollout (no attack) and
N×G candidate rollouts (restore state, restart MOLA, load genome, drive H metres),
scores each candidate by physical deviation from the nominal end point plus a
directional term, and by perturbation magnitude (Chamfer distance); NSGA-III
selects the Pareto front, the chosen genome is applied for a real H-metre stretch,
and the next window starts from the new state.

The genome has 17 values in [-1, 1], decoded into 13 parameters of nine
perturbation operators (per-point noise, cluster shifts, dropout, ghost points,
geometric distortion, edge attack, temporal drift, scanline shift, feature-anchored
ghosts). See `src/perturbations/perturbation_generator.py`.

## Layout

```
config/mola/          MOLA system file and odometry pipeline (GICP + optimize_twist)
isaac/                scene (carter_warehouse.usd) and Script Editor helpers
src/nodes/            ROS 2 nodes: add_intensity, perturbation, waypoint follower; MOLA launcher
src/perturbations/    perturbation generator and its diagnostics
src/optimization/     receding-horizon orchestrator (NSGA-III) and follower model
src/baseline/         open-loop deterministic baseline and its statistics
src/analysis/         rollout and MOLA-trace diagnostics
src/plots/            trajectory, Pareto, temporal and baseline plots
data/                 run outputs (history.json, TUM trajectories, traces)
results/              archived results
docs/mola-reference/  MOLA headers and pipelines used for reference
```

## Requirements

- Ubuntu 24.04, ROS 2 Jazzy, `ros-jazzy-mola`, `ros-jazzy-mola-lidar-odometry`,
  `ros-jazzy-mola-state-estimation`
- NVIDIA Isaac Sim 6.1 (pip install), scene `isaac/carter_warehouse.usd`
- Python: `pip install -r requirements.txt`

## Running

All commands from the repository root, with `source /opt/ros/jazzy/setup.bash`.
Isaac Sim must be in Play with the LiDAR publishing at ~10 Hz.

Open-loop baseline (robot follows the true pose, MOLA observes):

```bash
# Script Editor: isaac/run_isaac_headless.py
python3 src/baseline/run_baseline_experiment.py --run-id 1 --path loop --loop-size 4,2.5 \
    --loop-dir ccw --deterministic --local-map-size 25 \
    --lidar-topic /front_3d_lidar/lidar_points --sub-scans-per-cycle 1 \
    --lidar-pose="-0.2317,0,0.526" --skip-reset
python3 src/baseline/loop_baseline_stats.py --runs 2 3 4 5 6
python3 src/plots/plot_loop_baseline.py --runs 2 3 4 5 6
```

Closed loop without optimisation (four terminals):

```bash
python3 src/nodes/add_intensity_node.py --input-topic /front_3d_lidar/lidar_points \
    --sub-scans-per-cycle 1 --output-frame base_link
python3 src/nodes/perturbation_node.py --passthrough
./src/nodes/launch_mola_attack.sh
python3 src/nodes/waypoint_follower_node.py --waypoints "4,0; 4,2.5; 0,2.5; 0,0" \
    --warmup-poses 60 --log data/attack/closedloop.csv
```

Receding-horizon attack (the orchestrator starts MOLA and drives the robot itself):

```bash
# Script Editor: isaac/isaac_rollout_server.py
python3 src/nodes/add_intensity_node.py --input-topic /front_3d_lidar/lidar_points \
    --sub-scans-per-cycle 1 --output-frame base_link
python3 src/nodes/perturbation_node.py
python3 -u src/optimization/attack_orchestrator.py --goal "5,0" --horizon 1.0 \
    --settle-sec 2.0 --pop 4 --gen 2 --trace --out data/attack/run_01
```

Plots and diagnostics:

```bash
python3 src/plots/plot_trajectory.py data/attack/run_01/history.json
python3 src/plots/plot_pareto.py data/attack/run_01/history.json
python3 src/analysis/diagnose_rollout.py data/attack/run_01/traces/*.csv
python3 src/perturbations/chamfer_by_operator.py --history data/attack/run_01/history.json
```

## Notes

- Sensor noise of the XT-32 is set to zero in the scene so that any effect is
  attributable to the perturbation alone.
- MOLA is restarted for every rollout because it exposes no reset; the victim
  SLAM is therefore not persistent across windows.
- The state save/restore makes the optimisation a worst-case analysis with a
  digital twin; only the applied perturbation is something a real attacker
  could reproduce.
- The earlier offline pipeline (replay of recorded scans, ATE against ground
  truth) is kept in the git tag `offline-pipeline-v1`.

## References

- FLAT: Flux-Aware Imperceptible Adversarial Attacks on 3D Point Clouds (ECCV 2024)
- SLACK: Attacking LiDAR-based SLAM with Adversarial Point Injections (arXiv 2024)
- Adversarial attacks on ICP-based registration (arXiv 2403.05666)
- ASP: Attribution-based Scanline Perturbation (IEEE 2024)
