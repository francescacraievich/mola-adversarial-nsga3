# MOLA full-state patch

Patch to two MOLA packages that adds saving and loading of the complete internal state of
the LiDAR odometry and of its state estimator to a binary file. The shadow instance saves
its state at the end of a window; the twin instance loads it before each candidate rollout, so
every candidate starts from the same state as the victim.

| package | upstream | version | patch |
|---|---|---|---|
| `mola_lidar_odometry` | https://github.com/MOLAorg/mola_lidar_odometry | tag `3.2.0` | `mola_lidar_odometry-3.2.0-full-state.patch` |
| `mola_state_estimation_simple` | https://github.com/MOLAorg/mola_state_estimation (subfolder) | tag `2.4.2` | `mola_state_estimation_simple-2.4.2-full-state.patch` |

Both packages are GPL-3.0 (`LICENSE`), so this patch is GPL-3.0 as well. Nothing else in the
packages changes; the apt installation in `/opt/ros/jazzy` is left untouched and the patched
libraries are loaded through `MOLA_MODULES_LIB_PATH` and `LD_LIBRARY_PATH`.

Build into `~/mola_ws/install_G1` (the path expected by `src/optimization/twin_variants.py`):

```bash
bash mola_overlay/build.sh            # MOLA_WS=<dir> to use another workspace
```
