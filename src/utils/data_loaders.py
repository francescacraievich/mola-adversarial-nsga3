"""
Data loading utilities for NSGA-III adversarial perturbations.

Functions for loading point clouds and trajectories from various formats.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np


def load_point_clouds_from_npy(
    data_path: str = "data/frame_sequence.npy",
) -> Optional[List[np.ndarray]]:
    """
    Load point cloud sequence from numpy file.

    Args:
        data_path: Path to .npy file containing list of point clouds

    Returns:
        List of point cloud arrays, or None if file not found
    """
    path = Path(data_path)
    if path.exists():
        frames = np.load(path, allow_pickle=True)
        clouds = list(frames)
        print(f"  Loaded {len(clouds)} frames from {path}")
        return clouds
    else:
        print(f"  Not found: {path}")
        return None


def load_timestamps_from_npy(
    timestamps_path: str = "data/frame_sequence.timestamps.npy",
) -> Optional[np.ndarray]:
    """
    Load timestamps for point cloud frames.

    Args:
        timestamps_path: Path to .npy file containing timestamps

    Returns:
        Array of timestamps in nanoseconds, or None if file not found
    """
    path = Path(timestamps_path)
    if path.exists():
        timestamps = np.load(path)
        print(f"  Loaded {len(timestamps)} timestamps from {path}")
        return timestamps
    else:
        print(f"  Not found: {path}")
        return None


def load_trajectory_from_tum(  # noqa: C901
    tum_path: str, interpolate_to_frames: int = None, pc_timestamps: np.ndarray = None
) -> Optional[np.ndarray]:
    """
    Load trajectory from TUM format file or numpy file.

    TUM format: timestamp tx ty tz qx qy qz qw
    NPY format: (N, 3) array with xyz positions

    Args:
        tum_path: Path to TUM format trajectory file or .npy file
        interpolate_to_frames: If provided, interpolate to this many frames
        pc_timestamps: Point cloud timestamps (in nanoseconds) for interpolation

    Returns:
        Array of shape (N, 3) with xyz positions, or None if error
    """
    # Support .npy files directly
    if tum_path.endswith(".npy"):
        path = Path(tum_path)
        if path.exists():
            traj = np.load(path)
            print(f"  Loaded {len(traj)} poses from {path} (numpy format)")
            return traj
        else:
            print(f"  Not found: {path}")
            return None

    traj = []
    timestamps = []
    try:
        with open(tum_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 8:
                        timestamps.append(float(parts[0]))
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        traj.append([tx, ty, tz])
        if not traj:
            print(f"  No poses found in {tum_path}")
            return None

        traj = np.array(traj)
        timestamps = np.array(timestamps)
        print(f"  Loaded {len(traj)} poses from {tum_path}")

        # Interpolate if requested
        if interpolate_to_frames is not None and pc_timestamps is not None:
            from scipy.interpolate import interp1d

            # Normalize timestamps to start at 0
            gt_ts_norm = timestamps - timestamps[0]
            pc_ts_norm = pc_timestamps / 1e9  # Convert nanoseconds to seconds
            pc_ts_norm = pc_ts_norm - pc_ts_norm[0]

            # Scale GT timestamps to match PC duration
            if gt_ts_norm[-1] > 0:
                scale_factor = pc_ts_norm[-1] / gt_ts_norm[-1]
                gt_ts_scaled = gt_ts_norm * scale_factor

                # Interpolate each axis
                interp_x = interp1d(
                    gt_ts_scaled, traj[:, 0], kind="linear", fill_value="extrapolate"
                )
                interp_y = interp1d(
                    gt_ts_scaled, traj[:, 1], kind="linear", fill_value="extrapolate"
                )
                interp_z = interp1d(
                    gt_ts_scaled, traj[:, 2], kind="linear", fill_value="extrapolate"
                )

                traj = np.column_stack(
                    [interp_x(pc_ts_norm), interp_y(pc_ts_norm), interp_z(pc_ts_norm)]
                )
                print(f"  Interpolated to {len(traj)} poses (scale factor: {scale_factor:.2f})")

        return traj
    except FileNotFoundError:
        print(f"  Not found: {tum_path}")
        return None
