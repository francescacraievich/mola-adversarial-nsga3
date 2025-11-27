#!/usr/bin/env python3
"""
Create a sequence of point cloud frames from the global map and poses.

This simulates what the LiDAR would see at each pose by:
1. Taking the global point cloud
2. Transforming it to the robot's local frame at each pose
3. Filtering points by distance (LiDAR range)
"""

from pathlib import Path

import numpy as np


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q

    # Rotation matrix from quaternion
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    return R


def load_poses_from_tum(tum_path):
    """Load poses from TUM format trajectory file."""
    poses = []
    with open(tum_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 8:
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )
                    poses.append(
                        {
                            "position": np.array([tx, ty, tz]),
                            "quaternion": np.array([qx, qy, qz, qw]),
                        }
                    )
    return poses


def transform_cloud_to_local_frame(global_cloud, pose, max_range=20.0):
    """
    Transform global point cloud to robot's local frame.

    Args:
        global_cloud: (N, 4) array [x, y, z, intensity] in global frame
        pose: dict with 'position' and 'quaternion'
        max_range: maximum LiDAR range in meters

    Returns:
        (M, 4) array of points in local frame within range
    """
    # Get transformation
    t = pose["position"]
    R = quaternion_to_rotation_matrix(pose["quaternion"])

    # Transform points: local = R^T @ (global - t)
    points_xyz = global_cloud[:, :3]
    points_local = (R.T @ (points_xyz - t).T).T

    # Filter by range
    distances = np.linalg.norm(points_local, axis=1)
    mask = distances < max_range

    # Keep intensity
    local_cloud = np.zeros((mask.sum(), 4), dtype=np.float32)
    local_cloud[:, :3] = points_local[mask]
    local_cloud[:, 3] = global_cloud[mask, 3]

    return local_cloud


def add_sensor_noise(cloud, position_noise=0.01, intensity_noise=2.0):
    """Add realistic sensor noise to point cloud."""
    noisy = cloud.copy()
    noisy[:, :3] += np.random.normal(0, position_noise, (len(cloud), 3))
    noisy[:, 3] += np.random.normal(0, intensity_noise, len(cloud))
    noisy[:, 3] = np.clip(noisy[:, 3], 0, 255)
    return noisy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create frame sequence from global map")
    parser.add_argument(
        "--ref-cloud",
        type=str,
        default="data/reference_cloud.npy",
        help="Path to reference point cloud",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default="maps/ground_truth_trajectory.tum",
        help="Path to TUM trajectory file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/frame_sequence.npy",
        help="Output path for frame sequence",
    )
    parser.add_argument(
        "--max-range", type=float, default=20.0, help="Maximum LiDAR range in meters"
    )
    parser.add_argument("--add-noise", action="store_true", help="Add sensor noise to frames")
    parser.add_argument(
        "--subsample", type=int, default=None, help="Subsample to N points per frame"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" CREATE FRAME SEQUENCE FROM GLOBAL MAP")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    ref_cloud = np.load(args.ref_cloud)
    poses = load_poses_from_tum(args.poses)

    print(f"   Reference cloud: {ref_cloud.shape[0]} points")
    print(f"   Poses: {len(poses)}")
    print(f"   Max range: {args.max_range}m")

    # Create frame sequence
    print("\nCreating frame sequence...")
    frames = []

    for i, pose in enumerate(poses):
        # Transform to local frame
        local_cloud = transform_cloud_to_local_frame(ref_cloud, pose, args.max_range)

        # Add noise if requested
        if args.add_noise:
            local_cloud = add_sensor_noise(local_cloud)

        # Subsample if requested
        if args.subsample and len(local_cloud) > args.subsample:
            indices = np.random.choice(len(local_cloud), args.subsample, replace=False)
            local_cloud = local_cloud[indices]

        frames.append(local_cloud)

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(poses)} frames ({len(local_cloud)} points)")

    # Statistics
    points_per_frame = [len(f) for f in frames]
    print("\nStatistics:")
    print(f"   Frames: {len(frames)}")
    print(
        f"   Points/frame: min={min(points_per_frame)}, max={max(points_per_frame)}, mean={np.mean(points_per_frame):.0f}"
    )

    # Save as list of arrays (variable size per frame)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(frames, dtype=object), allow_pickle=True)

    print(f"\nSaved to: {output_path}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
