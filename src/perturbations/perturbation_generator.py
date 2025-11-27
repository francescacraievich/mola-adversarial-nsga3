"""
Perturbation Generator for LiDAR Point Clouds.

Implements state-of-the-art adversarial perturbation techniques based on research:
- Per-point perturbations (not global transforms)
- Feature-region targeting (high curvature areas)
- Chamfer distance for imperceptibility measurement
- Realistic perturbation bounds (centimeter-scale)

References:
- FLAT: Flux-Aware Imperceptible Adversarial Attacks (ECCV 2024)
- Adversarial Point Cloud Perturbations (Neurocomputing 2021)
- Survey on Adversarial Robustness of LiDAR-based ML (2024)
"""

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


class PerturbationGenerator:
    """
    Adversarial perturbation generator for LiDAR point clouds.

    Uses per-point perturbations with realistic bounds based on research papers.
    Targets high-curvature regions that are critical for SLAM feature extraction.
    """

    def __init__(
        self,
        # Per-point perturbation bounds (in meters)
        max_point_shift: float = 0.05,  # 5 cm max per-point displacement
        # Noise parameters
        noise_std: float = 0.02,  # 2 cm Gaussian noise std
        # Feature targeting
        target_high_curvature: bool = True,
        curvature_percentile: float = 90.0,  # Target top 10% curvature points
        # Point manipulation
        max_dropout_rate: float = 0.15,  # Max 15% point removal
        max_ghost_points_ratio: float = 0.05,  # Max 5% ghost points added
        # Cluster perturbation
        cluster_shift_std: float = 0.03,  # 3 cm cluster displacement std
        n_clusters: int = 5,  # Number of perturbation clusters
    ):
        """
        Initialize perturbation generator.

        Args:
            max_point_shift: Maximum displacement per point in meters (default: 5cm)
            noise_std: Standard deviation of Gaussian noise in meters (default: 2cm)
            target_high_curvature: Whether to target high-curvature regions
            curvature_percentile: Percentile threshold for high-curvature points
            max_dropout_rate: Maximum fraction of points to remove
            max_ghost_points_ratio: Maximum ratio of ghost points to add
            cluster_shift_std: Std of cluster-based displacement
            n_clusters: Number of perturbation clusters
        """
        self.max_point_shift = max_point_shift
        self.noise_std = noise_std
        self.target_high_curvature = target_high_curvature
        self.curvature_percentile = curvature_percentile
        self.max_dropout_rate = max_dropout_rate
        self.max_ghost_points_ratio = max_ghost_points_ratio
        self.cluster_shift_std = cluster_shift_std
        self.n_clusters = n_clusters

    def get_genome_size(self) -> int:
        """
        Get the size of the genome encoding.

        Genome structure (12 parameters):
        - [0-2]: Directional bias for per-point noise (normalized direction)
        - [3]: Noise intensity scale [0, 1]
        - [4]: Curvature targeting strength [0, 1]
        - [5]: Point dropout rate [0, 1]
        - [6]: Ghost points ratio [0, 1]
        - [7-9]: Cluster perturbation direction
        - [10]: Cluster perturbation strength [0, 1]
        - [11]: Spatial correlation of perturbations [0, 1]
        """
        return 12

    def encode_perturbation(self, genome: np.ndarray) -> Dict[str, any]:
        """
        Encode genome into perturbation parameters.

        Args:
            genome: Normalized parameters in range [-1, 1]

        Returns:
            Dictionary with perturbation parameters
        """
        # Normalize genome to [0, 1] for rates, keep [-1, 1] for directions
        genome = np.clip(genome, -1, 1)

        # Directional bias for noise (keep as direction vector)
        noise_direction = genome[0:3]
        noise_direction_norm = np.linalg.norm(noise_direction)
        if noise_direction_norm > 0:
            noise_direction = noise_direction / noise_direction_norm

        # Noise intensity [0, 1] -> [0, max_point_shift]
        noise_intensity = (genome[3] + 1) / 2 * self.max_point_shift

        # Curvature targeting strength [0, 1]
        curvature_strength = (genome[4] + 1) / 2

        # Dropout rate [0, max_dropout_rate]
        dropout_rate = (genome[5] + 1) / 2 * self.max_dropout_rate

        # Ghost points ratio [0, max_ghost_points_ratio]
        ghost_ratio = (genome[6] + 1) / 2 * self.max_ghost_points_ratio

        # Cluster perturbation direction
        cluster_direction = genome[7:10]
        cluster_dir_norm = np.linalg.norm(cluster_direction)
        if cluster_dir_norm > 0:
            cluster_direction = cluster_direction / cluster_dir_norm

        # Cluster strength [0, 1]
        cluster_strength = (genome[10] + 1) / 2

        # Spatial correlation [0, 1] - how correlated nearby point perturbations are
        spatial_correlation = (genome[11] + 1) / 2

        return {
            "noise_direction": noise_direction,
            "noise_intensity": noise_intensity,
            "curvature_strength": curvature_strength,
            "dropout_rate": dropout_rate,
            "ghost_ratio": ghost_ratio,
            "cluster_direction": cluster_direction,
            "cluster_strength": cluster_strength,
            "spatial_correlation": spatial_correlation,
        }

    def compute_curvature(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute local curvature using fast approximation.

        OPTIMIZED: Uses small sample and vectorized nearest-neighbor assignment.

        Args:
            points: Point cloud (N, 3+) XYZ coordinates
            k: Number of nearest neighbors

        Returns:
            Curvature values for each point (N,)
        """
        n_points = len(points)
        if n_points < k + 1:
            return np.zeros(n_points)

        # Use very small sample for speed (1000 points max)
        sample_size = min(n_points, 1000)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_indices, :3]

        # Build KD-tree for sampled points
        tree = cKDTree(sample_points)

        # Compute curvature for sampled points
        sample_curvatures = np.zeros(sample_size)
        k_use = min(k, sample_size - 1)

        # Batch query for all sample points
        _, all_neighbors = tree.query(sample_points, k=k_use + 1)

        for i in range(sample_size):
            neighbors = sample_points[all_neighbors[i]]
            centered = neighbors - neighbors.mean(axis=0)

            if len(centered) > 3:
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                total = eigenvalues.sum()
                if total > 0:
                    sample_curvatures[i] = np.min(eigenvalues) / total

        # Assign curvature to all points from nearest sample (vectorized)
        _, nearest = tree.query(points[:, :3], k=1)
        curvatures = sample_curvatures[nearest]

        return curvatures

    def _compute_perturbation_weights(self, perturbed, n_points, params):
        """Compute curvature-based weights for perturbation targeting."""
        if not self.target_high_curvature or params["curvature_strength"] <= 0.1:
            return np.ones(n_points)

        curvatures = self.compute_curvature(perturbed[:, :3])
        if curvatures.max() > curvatures.min():
            curvature_weights = (curvatures - curvatures.min()) / (
                curvatures.max() - curvatures.min()
            )
        else:
            return np.ones(n_points)

        threshold = np.percentile(
            curvature_weights, 100 - self.curvature_percentile * params["curvature_strength"]
        )
        return np.where(curvature_weights >= threshold, 1.0, 0.3)

    def _apply_noise(self, perturbed, n_points, perturbation_weights, params):
        """Apply per-point Gaussian noise with directional bias."""
        if params["noise_intensity"] <= 0.001:
            return perturbed

        noise = np.random.randn(n_points, 3) * self.noise_std

        if params["spatial_correlation"] > 0.1:
            noise = self._apply_spatial_correlation(
                perturbed[:, :3], noise, params["spatial_correlation"]
            )

        directional_component = params["noise_direction"] * params["noise_intensity"]
        noise += directional_component
        noise *= perturbation_weights[:, np.newaxis]

        noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
        noise = np.where(
            noise_norms > self.max_point_shift,
            noise / noise_norms * self.max_point_shift,
            noise,
        )
        perturbed[:, :3] += noise
        return perturbed

    def _apply_dropout(self, perturbed, n_points, perturbation_weights, params):
        """Apply targeted point dropout."""
        if params["dropout_rate"] <= 0.01:
            return perturbed

        keep_prob = 1 - params["dropout_rate"] * perturbation_weights
        keep_mask = np.random.random(n_points) < keep_prob
        if keep_mask.sum() < n_points * 0.5:
            keep_mask = np.random.random(n_points) < 0.5
        return perturbed[keep_mask]

    def _add_ghost_points(self, perturbed, params):
        """Add ghost points to confuse feature matching."""
        if params["ghost_ratio"] <= 0.01 or len(perturbed) == 0:
            return perturbed

        n_ghost = int(len(perturbed) * params["ghost_ratio"])
        if n_ghost > 0:
            ghost_points = self._generate_ghost_points(perturbed, n_ghost)
            perturbed = np.vstack([perturbed, ghost_points])
        return perturbed

    def apply_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, any], seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply advanced adversarial perturbation to point cloud.

        Args:
            point_cloud: Input point cloud (N, 4) with [x, y, z, intensity]
            params: Perturbation parameters from encode_perturbation()
            seed: Random seed for reproducibility

        Returns:
            Perturbed point cloud (M, 4) where M may differ from N
        """
        if seed is not None:
            np.random.seed(seed)

        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        perturbation_weights = self._compute_perturbation_weights(perturbed, n_points, params)
        perturbed = self._apply_noise(perturbed, n_points, perturbation_weights, params)

        if params["cluster_strength"] > 0.1:
            perturbed = self._apply_cluster_perturbation(
                perturbed, params["cluster_direction"], params["cluster_strength"]
            )

        perturbed = self._apply_dropout(perturbed, n_points, perturbation_weights, params)
        perturbed = self._add_ghost_points(perturbed, params)

        return perturbed

    def _apply_spatial_correlation(
        self, points: np.ndarray, noise: np.ndarray, correlation: float
    ) -> np.ndarray:
        """
        Apply spatial correlation to noise - nearby points have similar perturbations.
        OPTIMIZED: Skip if correlation is low, use simple Gaussian smoothing approximation.
        """
        if correlation < 0.3 or len(points) < 100:
            return noise

        # Simple approximation: add a global smooth component
        # This is much faster than per-point neighbor averaging
        global_shift = noise.mean(axis=0) * correlation
        correlated_noise = noise * (1 - correlation * 0.5) + global_shift

        return correlated_noise

    def _apply_cluster_perturbation(
        self, point_cloud: np.ndarray, direction: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Apply perturbation to random clusters of points.
        Simulates localized sensor errors or environmental interference.
        OPTIMIZED: Uses vectorized operations instead of loops.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        if n_points < 100:
            return perturbed

        # Select random cluster centers
        n_clusters = min(self.n_clusters, n_points // 100)
        cluster_centers_idx = np.random.choice(n_points, n_clusters, replace=False)

        # Compute cluster radius
        cloud_center = perturbed[:, :3].mean(axis=0)
        distances_to_center = np.linalg.norm(perturbed[:, :3] - cloud_center, axis=1)
        cluster_radius = np.percentile(distances_to_center, 10)

        for center_idx in cluster_centers_idx:
            center = perturbed[center_idx, :3]

            # Vectorized distance computation
            distances = np.linalg.norm(perturbed[:, :3] - center, axis=1)
            mask = distances < cluster_radius

            if mask.sum() > 0:
                # Generate cluster-specific random displacement
                cluster_shift = direction * strength * self.cluster_shift_std
                cluster_shift += np.random.randn(3) * self.cluster_shift_std * strength * 0.5

                # Vectorized falloff and application
                falloff = np.exp(-distances[mask] / cluster_radius)
                perturbed[mask, :3] += cluster_shift * falloff[:, np.newaxis]

        return perturbed

    def _generate_ghost_points(self, point_cloud: np.ndarray, n_ghost: int) -> np.ndarray:
        """
        Generate ghost points that look plausible but confuse feature matching.
        Places them near existing points with slight offsets.
        """
        # Select random existing points as bases
        base_indices = np.random.choice(len(point_cloud), n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        # Add small random offsets (1-3 cm)
        offsets = np.random.randn(n_ghost, 3) * 0.02
        ghost_points[:, :3] += offsets

        # Slightly modify intensity
        ghost_points[:, 3] += np.random.randn(n_ghost) * 10
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

    def compute_chamfer_distance(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute Chamfer distance between original and perturbed point clouds.

        This is the standard metric for measuring point cloud perturbation magnitude.
        Lower values = more imperceptible perturbation.

        Args:
            original: Original point cloud (N, 3+)
            perturbed: Perturbed point cloud (M, 3+)

        Returns:
            Chamfer distance (average nearest-neighbor distance)
        """
        if len(original) == 0 or len(perturbed) == 0:
            return float("inf")

        # Build KD-trees
        tree_orig = cKDTree(original[:, :3])
        tree_pert = cKDTree(perturbed[:, :3])

        # Forward distance: for each point in perturbed, find nearest in original
        dist_forward, _ = tree_orig.query(perturbed[:, :3], k=1)

        # Backward distance: for each point in original, find nearest in perturbed
        dist_backward, _ = tree_pert.query(original[:, :3], k=1)

        # Chamfer distance = mean of forward + backward
        chamfer = (dist_forward.mean() + dist_backward.mean()) / 2

        return chamfer

    def compute_perturbation_magnitude(
        self, original: np.ndarray, perturbed: np.ndarray, params: Dict[str, any]
    ) -> float:
        """
        Compute perturbation magnitude for NSGA-II objective.

        Returns Chamfer distance in centimeters - this is the standard metric
        for point cloud perturbation imperceptibility.

        Args:
            original: Original point cloud
            perturbed: Perturbed point cloud
            params: Perturbation parameters (unused, kept for API compatibility)

        Returns:
            Chamfer distance in centimeters (lower = more imperceptible)
        """
        # Compute Chamfer distance in meters
        chamfer_m = self.compute_chamfer_distance(original, perturbed)

        # Return in centimeters for readability
        # Typical range: 0.5 - 5.0 cm for adversarial perturbations
        chamfer_cm = chamfer_m * 100

        return chamfer_cm

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
