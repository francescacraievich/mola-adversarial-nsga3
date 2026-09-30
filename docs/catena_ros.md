# Catena ROS 2: nodi e topic verificati

Nodi e topic dei tre scenari, letti dal grafo ROS in esecuzione con
`ros2 node list`, `ros2 node info <nodo>`, `ros2 topic list` e
`ros2 topic info -v <topic>` (Ubuntu 24.04, ROS 2 Jazzy, Isaac Sim 6.1, scena
warehouse con Nova Carter). Le catture sono in `data/attack/verifica/`
(`02_ros_graph.txt`, `10A_*`, `10B_*`, `10C_*`, `11_*`). Gli schemi
corrispondenti sono `schema_catena_{A,B,C}.png`.

Convenzione: **scrive** = il nodo pubblica sul topic; **legge** = il nodo è
iscritto al topic. `/parameter_events` e `/rosout` sono presenti su ogni nodo
ROS e non trasportano dati della catena: omessi ovunque.

## Nodi di Isaac Sim (asset Nova Carter)

Non compaiono in `ros2 node list` (sono endpoint del bridge OmniGraph) ma sono
visibili come publisher/subscriber in `ros2 topic info -v`. Nome ROS e, fra
parentesi, il nodo OmniGraph della scena.

| Nome ROS | OmniGraph | Scrive | Legge |
|---|---|---|---|
| `/front_3d_lidar/_Render_PostProcess_SDGPipeline_Replicator_02_NodeWriterWriter` | `chassis_link/sensors/XT_32/ROS_LidarRTX/PointCloudPublish` | `/front_3d_lidar/lidar_points` (PointCloud2: x, y, z; frame `front_3d_lidar`; 10 Hz; ~45 000 punti) | — |
| `_World_ROS_Clock_ros2_publish_clock` | `/World/ROS_Clock/ros2_publish_clock` | `/clock` (Clock) | — |
| `_World_Nova_Carter_ROS_differential_drive_ros2_subscribe_twist` | `differential_drive/ros2_subscribe_twist` → `differential_controller_01` → `articulation_controller_01` | — | `/cmd_vel` (Twist) |
| `_World_Nova_Carter_ROS_transform_tree_odometry_ros2_publish_odometry` | `transform_tree_odometry/isaac_compute_odometry_node` → `ros2_publish_odometry` | `/chassis/odom` (Odometry, posa vera) | — |
| `*_ROS_TF_PublisherTF` (uno per sensore) | `transform_tree_odometry/tf_tree_*`, `*/ROS_TF/PublisherTF` | `/tf` (TFMessage) | — |
| `chassis_imu`, `*_hawk/Imu` | `ros2_publish_imu` | `/chassis/imu`, `/front_stereo_camera/imu`, `/back_stereo_camera/imu`, `/left_stereo_camera/imu`, `/right_stereo_camera/imu` (Imu) | — |
| `*/ROS_Camera` | `publish_image`, `ros2_camera_info_helper` | `/front_stereo_camera/{left,right}/image_raw`, `/front_stereo_camera/{left,right}/camera_info` | — |

Con la sola scena in Play `ros2 topic list` mostra esattamente questi 16 topic
(più `/parameter_events`, `/rosout`); `/tf_static` non è pubblicato da Isaac.

## Nodi aggiunti

| Nodo | Scrive | Legge | Ruolo |
|---|---|---|---|
| `/add_intensity_node` | `/carter/lidar_with_intensity` (PointCloud2: x, y, z, intensity; frame `base_link`; stesso stamp) | `/front_3d_lidar/lidar_points` | adattatore: fa il lavoro del driver di un LiDAR reale. Parte del sistema vittima |
| `/perturbation_node` | `/carter/lidar_perturbed` (PointCloud2, stessi campi, frame e stamp), `/attack/status` (String JSON) | `/carter/lidar_with_intensity`, `/attack/genome` (Float32MultiArray), `/attack/enabled` (Bool) | attaccante. Nessun topic di posa |
| `/waypoint_follower` | `/cmd_vel` | `/lidar_odometry/pose`, `/chassis/odom` (solo registrato nel log) | controllore del closed loop senza ottimizzazione |
| `/attack_orchestrator` | `/cmd_vel`, `/attack/genome`, `/attack/enabled` | `/lidar_odometry/pose`, `/lidar_odometry/pose_quality` (solo registrato), `/attack/status`, `/chassis/odom` (posa vera: danno, arresto a 1 m, tracce) | controllore + NSGA-III durante l'attacco |

## MOLA (`/mola_bridge_ros2`, processo `mola-cli`)

Legge: la nuvola configurata (`/front_3d_lidar/lidar_points`, oppure
`/carter/lidar_with_intensity`, oppure `/carter/lidar_perturbed`, mai più di
una), `/clock`. Sottoscrive anche `/imu`, `/gps`, `/gps_fix`, `/initialpose`
(previsti dal file di configurazione; nessuno li pubblica) e `/tf`,
`/tf_static` (non usati: la posa del LiDAR rispetto a `base_link` è passata
come costante, `MOLA_USE_FIXED_LIDAR_POSE`, valore letto dal tf della scena).

Scrive: `/lidar_odometry/pose` (Odometry: x, y, yaw stimati), letto dal
controllore; `/lidar_odometry/pose_quality` (Float32), solo registrato;
`/lidar_odometry/localmap_points`, `/lidar_odometry/metadata`,
`/diagnostics`, `/mola_diagnostics/lidar_odom/status`, `/tf`, `/tf_static`,
senza lettori.

Servizi esposti, non chiamati da nessuno: `/map_load`, `/map_save`,
`/relocalize_near_pose`, `/relocalize_from_state_estimator`,
`/mola_runtime_param_get`, `/mola_runtime_param_set`.

## Scenario A: sistema nativo, nessun nodo aggiunto

```
LiDAR ──/front_3d_lidar/lidar_points──▶ MOLA ──/lidar_odometry/pose──▶ waypoint_follower
clock ──/clock──▶ MOLA
Isaac ◀──/cmd_vel── waypoint_follower
```

Verificato (10A): MOLA iscritta a `/front_3d_lidar/lidar_points`, pubblica
`/lidar_odometry/pose` a 4.4 Hz a robot fermo (buchi fino a 0.77 s).
`/chassis/odom` è letto dal follower solo per il log.

## Scenario B: sistema vittima degli esperimenti, senza attacco

```
LiDAR ──/front_3d_lidar/lidar_points──▶ add_intensity_node ──/carter/lidar_with_intensity──▶ MOLA
MOLA ──/lidar_odometry/pose──▶ waypoint_follower ──/cmd_vel──▶ Isaac
```

Verificato (10B): `lidar_with_intensity` ha 1 publisher (add_intensity) e 1
subscriber (MOLA); MOLA non è iscritta a `lidar_points`; pose a 9.6 Hz a
robot fermo.

## Scenario C: sotto attacco

```
LiDAR ──lidar_points──▶ add_intensity_node ──lidar_with_intensity──▶ perturbation_node ──lidar_perturbed──▶ MOLA
MOLA ──/lidar_odometry/pose──▶ attack_orchestrator ──/cmd_vel──▶ Isaac
attack_orchestrator ──/attack/genome, /attack/enabled──▶ perturbation_node ──/attack/status──▶ attack_orchestrator
```

Verificato (10C, 11): `lidar_with_intensity` ha 1 publisher (add_intensity) e
1 subscriber (perturbation_node); `lidar_perturbed` ha 1 publisher
(perturbation_node) e 1 subscriber (MOLA); MOLA non è iscritta a
`lidar_with_intensity`; `perturbation_node` non è iscritto a nessun topic di
posa; `/chassis/odom` è letto solo dall'orchestratore. Messaggio perturbato: stessi campi,
frame e stamp del pulito, numero di punti diverso col dropout (42 656 contro
44 982 nella cattura). Pose a 7.9 Hz a robot fermo con la perturbazione a
80-88 ms per scan.

La posa vera arriva all'orchestratore da `/chassis/odom` (opzione
`--true-pose ros`, default). Fuori da ROS resta solo il telecomando del
simulatore: `isaac_rollout_server.py` gira nello Script Editor di Isaac Sim
(interprete senza `rclpy`) e risponde via file (`/tmp/isaac_cmd.json`,
`/tmp/isaac_reply.json`) ai comandi pausa, play, salva stato, ripristina stato
e set_pose (posa iniziale di una run), che non hanno equivalente ROS. Con
`--true-pose file` la posa viene letta dal server (`/tmp/isaac_pose.json`),
stessa sorgente fisica.

## Cosa questo dice per il threat model

Il sistema vittima è LiDAR → MOLA → controllore → robot. `add_intensity_node`
appartiene alla vittima (MOLA funziona anche senza, ma a metà frequenza).
L'attacco aggiunge un solo nodo, `perturbation_node`, fra driver e SLAM: legge
la nuvola, la modifica, la ripubblica con lo stesso formato, e MOLA legge la sua
uscita al posto di quella del driver. Non riceve informazioni sulla posa.
Tutto ciò che riguarda la posa vera (danno, arresto a 1 m, salva/ripristina)
è nell'orchestratore, fuori dalla catena LiDAR → attaccante → SLAM → controllore.
