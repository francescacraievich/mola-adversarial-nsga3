"""
Perturbation generator for LiDAR point clouds.

Generates and applies various types of perturbations to point clouds
while maintaining physical realism constraints.
"""

from typing import Dict, Optional

import numpy as np


class PerturbationGenerator:
    """
    Generates adversarial perturbations for LiDAR point clouds.

    The perturbations are constrained to maintain physical realism while
    maximizing impact on SLAM performance.
    """

    def __init__(
        self,
        max_translation: float = 0.5,
        max_rotation: float = 0.1,
        max_intensity_change: float = 50.0,
        point_dropout_rate: float = 0.1,
    ):
        """
        Initialize perturbation generator.

        Args:
            max_translation: Maximum translation in meters (default: 0.5m)
            max_rotation: Maximum rotation in radians (default: 0.1 rad ≈ 5.7°)
            max_intensity_change: Maximum intensity change (default: 50.0)
            point_dropout_rate: Maximum point dropout rate (default: 0.1 = 10%)
        """
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_intensity_change = max_intensity_change
        self.point_dropout_rate = point_dropout_rate

    def encode_perturbation(self, genome: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Encode genome into perturbation parameters.

        Args:
            genome: Encoded perturbation parameters (normalized [-1, 1])
                   Format: [tx, ty, tz, rx, ry, rz, intensity_scale, dropout_rate, ...]

        Returns:
            Dictionary with perturbation parameters
        """
        # Translation (first 3 genes)
        translation = genome[:3] * self.max_translation

        # Rotation (next 3 genes) - Euler angles
        rotation = genome[3:6] * self.max_rotation

        # Intensity scaling (next gene)
        intensity_scale = genome[6] * self.max_intensity_change

        # Point dropout rate (next gene)
        dropout_rate = (genome[7] + 1) / 2 * self.point_dropout_rate  # Map [-1,1] to [0, max]

        return {
            "translation": translation,
            "rotation": rotation,
            "intensity_scale": intensity_scale,
            "dropout_rate": dropout_rate,
        }

    def decode_perturbation(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Decode perturbation parameters back into genome.

        Args:
            params: Dictionary with perturbation parameters

        Returns:
            Genome array (normalized [-1, 1])
        """
        genome = np.zeros(8)
        genome[:3] = params["translation"] / self.max_translation
        genome[3:6] = params["rotation"] / self.max_rotation
        genome[6] = params["intensity_scale"] / self.max_intensity_change
        genome[7] = params["dropout_rate"] / self.point_dropout_rate * 2 - 1
        return genome

    def apply_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, np.ndarray], seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply perturbation to point cloud.

        Args:
            point_cloud: Input point cloud (N, 4) with [x, y, z, intensity]
            params: Perturbation parameters from encode_perturbation()
            seed: Random seed for reproducibility

        Returns:
            Perturbed point cloud (M, 4) where M <= N
        """
        if seed is not None:
            np.random.seed(seed)

        perturbed = point_cloud.copy()

        # Apply translation
        perturbed[:, :3] += params["translation"]

        # Apply rotation (using rotation matrix from Euler angles)
        rotation_matrix = self._euler_to_rotation_matrix(params["rotation"])
        perturbed[:, :3] = (rotation_matrix @ perturbed[:, :3].T).T

        # Apply intensity perturbation
        perturbed[:, 3] += params["intensity_scale"]
        perturbed[:, 3] = np.clip(perturbed[:, 3], 0, 255)  # Keep intensity in valid range

        # Apply point dropout
        if params["dropout_rate"] > 0:
            keep_mask = np.random.random(len(perturbed)) > params["dropout_rate"]
            perturbed = perturbed[keep_mask]

        return perturbed

    @staticmethod
    def _euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrix.

        Args:
            euler_angles: Array of [roll, pitch, yaw] in radians

        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = euler_angles

        # Rotation around X-axis (roll)
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )

        # Rotation around Y-axis (pitch)
        Ry = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )

        # Rotation around Z-axis (yaw)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx

    def get_genome_size(self) -> int:
        """Get the size of the genome encoding."""
        return 8  # tx, ty, tz, rx, ry, rz, intensity, dropout

    def random_genome(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate random genome(s).

        Args:
            size: Number of genomes to generate (None for single genome)

        Returns:
            Random genome(s) in range [-1, 1]
        """
        if size is None:
            return np.random.uniform(-1, 1, self.get_genome_size())
        return np.random.uniform(-1, 1, (size, self.get_genome_size()))
