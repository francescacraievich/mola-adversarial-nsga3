"""
Data loading utilities for NSGA-II adversarial perturbations.

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


def load_trajectory_from_tum(tum_path: str) -> Optional[np.ndarray]:
    """
    Load trajectory from TUM format file.

    TUM format: timestamp tx ty tz qx qy qz qw

    Args:
        tum_path: Path to TUM format trajectory file

    Returns:
        Array of shape (N, 3) with xyz positions, or None if error
    """
    traj = []
    try:
        with open(tum_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 8:
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        traj.append([tx, ty, tz])
        if traj:
            print(f"  Loaded {len(traj)} poses from {tum_path}")
            return np.array(traj)
        else:
            print(f"  No poses found in {tum_path}")
            return None
    except FileNotFoundError:
        print(f"  Not found: {tum_path}")
        return None
