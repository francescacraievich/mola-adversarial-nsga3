"""
Point cloud operations and utilities.

Helper functions for loading, saving, and validating point clouds.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_point_cloud(filepath: str) -> np.ndarray:
    """
    Load point cloud from file.

    Supports .npy, .txt formats with [x, y, z, intensity] columns.

    Args:
        filepath: Path to point cloud file

    Returns:
        Point cloud array (N, 4)
    """
    filepath = Path(filepath)

    if filepath.suffix == ".npy":
        return np.load(filepath)
    elif filepath.suffix in [".txt", ".csv"]:
        return np.loadtxt(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_point_cloud(point_cloud: np.ndarray, filepath: str):
    """
    Save point cloud to file.

    Args:
        point_cloud: Point cloud array (N, 4)
        filepath: Output path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == ".npy":
        np.save(filepath, point_cloud)
    elif filepath.suffix in [".txt", ".csv"]:
        np.savetxt(filepath, point_cloud, fmt="%.6f")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def validate_point_cloud(point_cloud: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate point cloud structure and values.

    Args:
        point_cloud: Point cloud array

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(point_cloud, np.ndarray):
        return False, "Point cloud must be a numpy array"

    if point_cloud.ndim != 2:
        return False, f"Point cloud must be 2D array, got {point_cloud.ndim}D"

    if point_cloud.shape[1] not in [3, 4]:
        return False, f"Point cloud must have 3 or 4 columns, got {point_cloud.shape[1]}"

    if len(point_cloud) == 0:
        return False, "Point cloud is empty"

    if not np.isfinite(point_cloud).all():
        return False, "Point cloud contains NaN or Inf values"

    return True, None


def apply_perturbations(point_cloud: np.ndarray, perturbation_params: dict) -> np.ndarray:
    """
    Apply perturbations to point cloud.

    This is a convenience wrapper around PerturbationGenerator.apply_perturbation().

    Args:
        point_cloud: Input point cloud (N, 4)
        perturbation_params: Perturbation parameters

    Returns:
        Perturbed point cloud
    """
    from .perturbation_generator import PerturbationGenerator

    generator = PerturbationGenerator()
    return generator.apply_perturbation(point_cloud, perturbation_params)


def compute_perturbation_magnitude(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Compute L2 magnitude of perturbation.

    Args:
        original: Original point cloud (N, 3 or 4)
        perturbed: Perturbed point cloud (M, 3 or 4)

    Returns:
        L2 norm of perturbation
    """
    # Only consider spatial coordinates for magnitude
    orig_xyz = original[:, :3]
    pert_xyz = perturbed[:, :3]

    # Handle different sizes due to dropout
    min_size = min(len(orig_xyz), len(pert_xyz))
    orig_xyz = orig_xyz[:min_size]
    pert_xyz = pert_xyz[:min_size]

    diff = pert_xyz - orig_xyz
    return np.linalg.norm(diff)


def downsample_point_cloud(point_cloud: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Downsample point cloud using voxel grid.

    Args:
        point_cloud: Input point cloud (N, 3 or 4)
        voxel_size: Size of voxel grid in meters

    Returns:
        Downsampled point cloud
    """
    # Voxelize points
    voxel_indices = np.floor(point_cloud[:, :3] / voxel_size).astype(int)

    # Get unique voxels
    unique_voxels, indices = np.unique(voxel_indices, axis=0, return_index=True)

    return point_cloud[indices]
