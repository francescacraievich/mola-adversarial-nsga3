"""
Fitness metrics for evaluating adversarial perturbations.

Provides functions to compute:
1. Localization error (attack effectiveness)
2. Imperceptibility (perturbation magnitude)
"""

from typing import Optional, Tuple

import numpy as np


def compute_localization_error(
    ground_truth_trajectory: np.ndarray,
    estimated_trajectory: np.ndarray,
    method: str = "ate",
) -> float:
    """
    Compute localization error between ground truth and estimated trajectories.

    Args:
        ground_truth_trajectory: Ground truth poses (N, 3) or (N, 7) [x, y, z, qw, qx, qy, qz]
        estimated_trajectory: SLAM estimated poses (M, 3) or (M, 7)
        method: Error metric to use:
            - "ate": Absolute Trajectory Error (default)
            - "rpe": Relative Pose Error
            - "final": Final position error

    Returns:
        Localization error (lower is better for SLAM, higher is better for attack)
    """
    if len(ground_truth_trajectory) == 0 or len(estimated_trajectory) == 0:
        return float("inf")

    # Extract positions (x, y, z)
    gt_positions = ground_truth_trajectory[:, :3]
    est_positions = estimated_trajectory[:, :3]

    if method == "ate":
        # Absolute Trajectory Error: RMSE of aligned trajectories
        return _compute_ate(gt_positions, est_positions)
    elif method == "rpe":
        # Relative Pose Error: measures drift over time
        return _compute_rpe(gt_positions, est_positions)
    elif method == "final":
        # Final position error: distance between last poses
        return np.linalg.norm(gt_positions[-1] - est_positions[-1])
    else:
        raise ValueError(f"Unknown error method: {method}")


def _rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple:
    """
    Compute rigid alignment (rotation, translation only - NO scale) from source to target.

    For adversarial attacks, we don't want scale correction as it would
    hide trajectory distortions caused by the attack.

    Args:
        source: Source points (N, 3)
        target: Target points (N, 3)

    Returns:
        Tuple of (rotation_matrix, translation)
    """
    # Center the point clouds
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Correct for reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation (NO scale)
    t = target_mean - R @ source_mean

    return R, t


def _compute_ate(
    gt_positions: np.ndarray, est_positions: np.ndarray, verbose: bool = True
) -> float:
    """
    Compute Absolute Trajectory Error (ATE) with Umeyama alignment.

    Standard ATE computation:
    1. Align estimated trajectory to ground truth using Umeyama (R, t, no scale)
    2. Compute RMSE of per-pose translational errors

    Args:
        gt_positions: Ground truth positions (N, 3)
        est_positions: Estimated positions (M, 3)
        verbose: Print debug information

    Returns:
        ATE error (meters)
    """
    # Align trajectories to same length
    min_len = min(len(gt_positions), len(est_positions))
    gt_aligned = gt_positions[:min_len]
    est_aligned = est_positions[:min_len]

    if min_len < 3:
        squared_errors = np.sum((gt_aligned - est_aligned) ** 2, axis=1)
        return np.sqrt(np.mean(squared_errors))

    # Umeyama alignment: find R, t that minimizes ||gt - (R @ est + t)||
    R, t = _rigid_alignment(est_aligned, gt_aligned)

    # Transform estimated trajectory
    est_transformed = (R @ est_aligned.T).T + t

    # Compute per-pose errors
    errors = np.linalg.norm(gt_aligned - est_transformed, axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    # Also compute final position error (drift)
    final_error = np.linalg.norm(gt_aligned[-1] - est_transformed[-1])

    if verbose:
        print("  [ATE] Umeyama alignment applied (R + t)")
        print(
            f"  [ATE] Error: min={errors.min():.3f}m, max={errors.max():.3f}m, mean={errors.mean():.3f}m"
        )
        print(f"  [ATE] Final drift: {final_error:.3f}m")
        print(f"  [ATE] RMSE: {rmse:.4f}m")

    return rmse


def _compute_rpe(gt_positions: np.ndarray, est_positions: np.ndarray, delta: int = 1) -> float:
    """
    Compute Relative Pose Error (RPE).

    RPE measures local consistency of the trajectory by comparing
    relative motions between consecutive poses.

    Args:
        gt_positions: Ground truth positions (N, 3)
        est_positions: Estimated positions (M, 3)
        delta: Frame spacing for relative pose computation

    Returns:
        RPE error (meters)
    """
    min_len = min(len(gt_positions), len(est_positions))
    if min_len < delta + 1:
        return float("inf")

    gt_positions = gt_positions[:min_len]
    est_positions = est_positions[:min_len]

    # Compute relative motions
    gt_relative = gt_positions[delta:] - gt_positions[:-delta]
    est_relative = est_positions[delta:] - est_positions[:-delta]

    # Compute RMSE of relative errors
    relative_errors = np.linalg.norm(gt_relative - est_relative, axis=1)
    rmse = np.sqrt(np.mean(relative_errors**2))

    return rmse


def compute_imperceptibility(
    original_point_cloud: np.ndarray,
    perturbed_point_cloud: np.ndarray,
    method: str = "l2",
) -> float:
    """
    Compute imperceptibility metric (perturbation magnitude).

    Args:
        original_point_cloud: Original point cloud (N, 3 or 4)
        perturbed_point_cloud: Perturbed point cloud (M, 3 or 4)
        method: Metric to use:
            - "l2": L2 norm (default)
            - "linf": L-infinity norm
            - "relative": Relative L2 norm

    Returns:
        Perturbation magnitude (lower is more imperceptible)
    """
    if len(original_point_cloud) == 0 or len(perturbed_point_cloud) == 0:
        return 0.0

    # Extract spatial coordinates
    orig_xyz = original_point_cloud[:, :3]
    pert_xyz = perturbed_point_cloud[:, :3]

    # Handle size mismatch due to dropout
    min_size = min(len(orig_xyz), len(pert_xyz))
    orig_xyz = orig_xyz[:min_size]
    pert_xyz = pert_xyz[:min_size]

    diff = pert_xyz - orig_xyz

    if method == "l2":
        # L2 norm: Euclidean distance
        return float(np.linalg.norm(diff))
    elif method == "linf":
        # L-infinity norm: maximum absolute difference
        return float(np.max(np.abs(diff)))
    elif method == "relative":
        # Relative L2 norm
        orig_norm = np.linalg.norm(orig_xyz)
        if orig_norm == 0:
            return 0.0
        return float(np.linalg.norm(diff) / orig_norm)
    else:
        raise ValueError(f"Unknown imperceptibility method: {method}")


def compute_multi_objective_fitness(
    ground_truth_trajectory: np.ndarray,
    estimated_trajectory: np.ndarray,
    original_point_cloud: np.ndarray,
    perturbed_point_cloud: np.ndarray,
    error_method: str = "ate",
    imperceptibility_method: str = "l2",
) -> Tuple[float, float]:
    """
    Compute multi-objective fitness for NSGA-III.

    Objectives:
    1. Maximize localization error (attack effectiveness)
    2. Minimize perturbation magnitude (imperceptibility)

    Args:
        ground_truth_trajectory: Ground truth trajectory
        estimated_trajectory: SLAM estimated trajectory
        original_point_cloud: Original point cloud
        perturbed_point_cloud: Perturbed point cloud
        error_method: Localization error metric
        imperceptibility_method: Imperceptibility metric

    Returns:
        Tuple of (neg_localization_error, imperceptibility)
        Note: Returns negative localization error for minimization
    """
    # Objective 1: Localization error (higher is better for attack, so negate for minimization)
    loc_error = compute_localization_error(
        ground_truth_trajectory, estimated_trajectory, method=error_method
    )

    # Objective 2: Imperceptibility (lower is better)
    imperceptibility = compute_imperceptibility(
        original_point_cloud, perturbed_point_cloud, method=imperceptibility_method
    )

    # Return for minimization: we want to minimize both objectives
    # For attack effectiveness: minimize negative error (= maximize error)
    # For imperceptibility: minimize perturbation magnitude
    return -loc_error, imperceptibility


def normalize_fitness(
    fitness_values: np.ndarray, reference_point: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize fitness values to [0, 1] range.

    Args:
        fitness_values: Array of fitness values (N, n_objectives)
        reference_point: Optional reference point for normalization

    Returns:
        Normalized fitness values
    """
    if reference_point is None:
        # Use min-max normalization
        min_vals = np.min(fitness_values, axis=0)
        max_vals = np.max(fitness_values, axis=0)

        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        return (fitness_values - min_vals) / ranges
    else:
        # Normalize by reference point
        return fitness_values / reference_point
