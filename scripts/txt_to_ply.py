#!/usr/bin/env python3
"""Convert MRPT txt point cloud to PLY format for CloudCompare."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def txt_to_ply(txt_file, ply_file, color=None):
    """Convert txt (X Y Z I) to PLY format."""
    print(f"Loading {txt_file}...")

    # Load data
    data = np.loadtxt(txt_file)

    points = data[:, :3]  # X Y Z
    intensity = data[:, 3]  # Intensity

    n_points = len(points)
    print(f"  Loaded {n_points:,} points")

    # Normalize intensity to 0-255 for color
    if color is None:
        # Use intensity as grayscale
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        colors = np.column_stack(
            [
                (intensity_norm * 255).astype(np.uint8),
                (intensity_norm * 255).astype(np.uint8),
                (intensity_norm * 255).astype(np.uint8),
            ]
        )
    else:
        # Use provided color (R, G, B in 0-255)
        colors = np.tile(color, (n_points, 1)).astype(np.uint8)

    print(f"Writing {ply_file}...")

    with open(ply_file, "w") as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write points
        for i in range(n_points):
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )

    print(f"  ✓ Saved {ply_file}")


if __name__ == "__main__":
    # Convert both maps with different colors
    print("Converting MRPT maps to PLY format for CloudCompare\n")

    # Map 1 (clean) - BLUE
    txt_to_ply("final_map_raw.txt", "final_map_blue.ply", color=[0, 0, 255])

    # Map 2 (perturbed) - RED
    txt_to_ply("final_map2_raw.txt", "final_map2_red.ply", color=[255, 0, 0])

    print("\n✅ Done! Now open in CloudCompare:")
    print("   cloudcompare.CloudCompare final_map_blue.ply final_map2_red.ply")
