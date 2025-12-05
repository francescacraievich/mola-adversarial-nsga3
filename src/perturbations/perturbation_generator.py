"""
Perturbation Generator for LiDAR Point Clouds.

Implements state-of-the-art adversarial perturbation techniques based on research:
- Per-point perturbations (not global transforms)
- Feature-region targeting (high curvature areas, edges, corners)
- Chamfer distance for imperceptibility measurement
- Realistic perturbation bounds (centimeter-scale)
- Coherent temporal drift (accumulating bias across frames)
- Strategic ghost point injection near geometric features
- Scanline perturbation (ASP-inspired)

References:
- FLAT: Flux-Aware Imperceptible Adversarial Attacks (ECCV 2024)
- Adversarial Point Cloud Perturbations (Neurocomputing 2021)
- Survey on Adversarial Robustness of LiDAR-based ML (2024)
- SLACK: Attacking LiDAR-based SLAM (arXiv 2024)
- ICP Adversarial Attack (arXiv 2403.05666)
- ASP: Attribution-based Scanline Perturbation (IEEE 2024)
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
        # Point manipulation (minimal for MOLA stability)
        max_dropout_rate: float = 0.03,  # Max 3% point removal
        max_ghost_points_ratio: float = 0.02,  # Max 2% ghost points added
        # Cluster perturbation
        cluster_shift_std: float = 0.03,  # 3 cm cluster displacement std
        n_clusters: int = 5,  # Number of perturbation clusters
        # Advanced attack parameters
        max_edge_shift: float = 0.08,  # 8 cm max shift for edge points
        max_temporal_drift: float = 0.05,  # 5 cm max accumulated drift per frame
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
            max_edge_shift: Maximum shift for detected edge points
            max_temporal_drift: Maximum accumulated drift per frame
        """
        self.max_point_shift = max_point_shift
        self.noise_std = noise_std
        self.target_high_curvature = target_high_curvature
        self.curvature_percentile = curvature_percentile
        self.max_dropout_rate = max_dropout_rate
        self.max_ghost_points_ratio = max_ghost_points_ratio
        self.cluster_shift_std = cluster_shift_std
        self.n_clusters = n_clusters
        self.max_edge_shift = max_edge_shift
        self.max_temporal_drift = max_temporal_drift
        # Temporal state for coherent drift attack
        self._accumulated_drift = np.zeros(3)
        self._frame_counter = 0

    def get_genome_size(self) -> int:
        """
        Get the size of the genome encoding.

        Genome structure (17 parameters):
        - [0-2]: Directional bias for per-point noise (normalized direction)
        - [3]: Noise intensity scale [0, 1]
        - [4]: Curvature targeting strength [0, 1]
        - [5]: Point dropout rate [0, 1]
        - [6]: Ghost points ratio [0, 1]
        - [7-9]: Cluster perturbation direction
        - [10]: Cluster perturbation strength [0, 1]
        - [11]: Spatial correlation of perturbations [0, 1]
        - [12]: Geometric distortion strength [0, 1] - KEY for ICP attacks
        - [13]: Edge attack strength [0, 1] - Target edges/corners (SLACK-inspired)
        - [14]: Temporal drift strength [0, 1] - Accumulating drift (ICP attack)
        - [15]: Scanline perturbation [0, 1] - ASP-inspired attack
        - [16]: Strategic ghost placement [0, 1] - Place ghosts near features
        """
        return 17

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
        # Adversarial: can be zero for stealth attacks
        noise_intensity = (genome[3] + 1) / 2 * self.max_point_shift

        # Curvature targeting strength [0, 1]
        curvature_strength = (genome[4] + 1) / 2

        # Dropout rate [0, max_dropout_rate]
        # Adversarial: can be zero for stealth attacks
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

        # Geometric distortion [0, 1] - KEY parameter for ICP attacks
        # High values create systematic distortions that break ICP convergence
        # Full range [0, 1] for maximum exploration
        geometric_distortion = (genome[12] + 1) / 2  # Maps [-1,1] -> [0, 1]

        # NEW ATTACK PARAMETERS (SLACK, ICP Attack, ASP inspired)
        # Edge attack strength [0, 1] - targets edges/corners critical for ICP
        edge_attack_strength = (genome[13] + 1) / 2 if len(genome) > 13 else 0.0

        # Temporal drift [0, 1] - accumulating bias that breaks loop closure
        temporal_drift_strength = (genome[14] + 1) / 2 if len(genome) > 14 else 0.0

        # Scanline perturbation [0, 1] - ASP-inspired attack along laser beams
        scanline_strength = (genome[15] + 1) / 2 if len(genome) > 15 else 0.0

        # Strategic ghost placement [0, 1] - place ghosts near features
        strategic_ghost = (genome[16] + 1) / 2 if len(genome) > 16 else 0.0

        return {
            "noise_direction": noise_direction,
            "noise_intensity": noise_intensity,
            "curvature_strength": curvature_strength,
            "dropout_rate": dropout_rate,
            "ghost_ratio": ghost_ratio,
            "cluster_direction": cluster_direction,
            "cluster_strength": cluster_strength,
            "spatial_correlation": spatial_correlation,
            "geometric_distortion": geometric_distortion,
            "edge_attack_strength": edge_attack_strength,
            "temporal_drift_strength": temporal_drift_strength,
            "scanline_strength": scanline_strength,
            "strategic_ghost": strategic_ghost,
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

    def detect_edges_and_corners(self, points: np.ndarray, k: int = 15) -> np.ndarray:
        """
        Detect edge and corner points using eigenvalue analysis.

        SLACK-inspired: These are the critical points for ICP matching.
        Perturbing them has maximum impact on scan registration.

        Classification based on eigenvalue ratios:
        - Planar: λ1 ≈ λ2 >> λ3 (surface points)
        - Edge: λ1 >> λ2 ≈ λ3 (line features)
        - Corner: λ1 ≈ λ2 ≈ λ3 (3D features)

        Args:
            points: Point cloud (N, 3+)
            k: Number of neighbors for local analysis

        Returns:
            Edge scores for each point (N,) - higher = more edge-like
        """
        n_points = len(points)
        if n_points < k + 1:
            return np.zeros(n_points)

        # Sample for speed
        sample_size = min(n_points, 2000)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_indices, :3]

        tree = cKDTree(sample_points)
        edge_scores = np.zeros(sample_size)

        _, all_neighbors = tree.query(sample_points, k=k + 1)

        for i in range(sample_size):
            neighbors = sample_points[all_neighbors[i]]
            centered = neighbors - neighbors.mean(axis=0)

            if len(centered) > 3:
                cov = np.cov(centered.T)
                eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]  # Descending

                # Edge score: high when λ1 >> λ2 (line-like structure)
                # Corner score: high when λ1 ≈ λ2 ≈ λ3 (3D feature)
                total = eigenvalues.sum() + 1e-10
                if total > 0:
                    # Linearity (edge): (λ1 - λ2) / λ1
                    linearity = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-10)
                    # Sphericity (corner): λ3 / λ1
                    sphericity = eigenvalues[2] / (eigenvalues[0] + 1e-10)
                    # Combined edge/corner score
                    edge_scores[i] = linearity + sphericity * 0.5

        # Assign to all points
        _, nearest = tree.query(points[:, :3], k=1)
        all_edge_scores = edge_scores[nearest]

        return all_edge_scores

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
        """
        Apply per-point Gaussian noise with directional bias.

        noise_intensity controls the overall magnitude (0-max_point_shift).
        noise_direction adds directional bias to the random noise.
        """
        if params["noise_intensity"] <= 0.001:
            return perturbed

        # Generate random noise scaled by noise_intensity (not fixed noise_std!)
        # This allows genome[3]=-1 → intensity=0 → no noise
        #           genome[3]=1  → intensity=5cm → full noise
        noise = np.random.randn(n_points, 3) * params["noise_intensity"]

        if params["spatial_correlation"] > 0.1:
            noise = self._apply_spatial_correlation(
                perturbed[:, :3], noise, params["spatial_correlation"]
            )

        # Add directional bias (30% of intensity in specified direction)
        directional_component = params["noise_direction"] * params["noise_intensity"] * 0.3
        noise += directional_component
        noise *= perturbation_weights[:, np.newaxis]

        # Clip to max_point_shift (safety limit)
        noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
        noise = np.where(
            noise_norms > self.max_point_shift,
            noise / noise_norms * self.max_point_shift,
            noise,
        )
        perturbed[:, :3] += noise
        return perturbed

    def _apply_dropout(self, perturbed, n_points, perturbation_weights, params):
        """
        Apply point dropout - DENSITY-BASED targeting for aggressive attacks.

        NEW STRATEGY: Target high-density regions that are critical for SLAM.
        SLAM relies on dense geometric features for matching, so removing
        clusters of nearby points degrades performance more than random dropout.
        """
        if params["dropout_rate"] <= 0.01:
            return perturbed

        # Compute local density using k-nearest neighbors
        k = min(20, n_points - 1)
        if k > 0:
            tree = cKDTree(perturbed[:, :3])
            distances, _ = tree.query(perturbed[:, :3], k=k + 1)
            # Local density = inverse of mean distance to k neighbors
            local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
            # Normalize to [0, 1]
            density_weights = (local_density - local_density.min()) / (
                local_density.max() - local_density.min() + 1e-6
            )
        else:
            density_weights = np.ones(n_points)

        # Target high-density points (inverse probability)
        # High density = low keep probability → more likely to drop
        # BUT keep the dropout rate close to the requested value
        keep_prob_base = 1 - params["dropout_rate"]
        # Only slight density-based variation (max 10% extra dropout for densest points)
        keep_prob_per_point = keep_prob_base * (1.0 - 0.1 * density_weights)
        keep_mask = np.random.random(n_points) < keep_prob_per_point

        # Safety: Keep at least 90% of points for adversarial imperceptibility
        min_keep_ratio = max(0.90, 1 - params["dropout_rate"] * 2)
        if keep_mask.sum() < n_points * min_keep_ratio:
            keep_mask = np.random.random(n_points) < min_keep_ratio

        return perturbed[keep_mask]

    def _add_ghost_points(self, perturbed, params):
        """Add ghost points to confuse feature matching."""
        if params["ghost_ratio"] <= 0.01 or len(perturbed) == 0:
            return perturbed

        n_ghost = int(len(perturbed) * params["ghost_ratio"])
        if n_ghost > 0:
            # Check if strategic placement is enabled
            strategic = params.get("strategic_ghost", 0)
            if strategic > 0.5:
                ghost_points = self._generate_strategic_ghost_points(perturbed, n_ghost, params)
            else:
                ghost_points = self._generate_ghost_points(perturbed, n_ghost)
            perturbed = np.vstack([perturbed, ghost_points])
        return perturbed

    def _generate_strategic_ghost_points(
        self, point_cloud: np.ndarray, n_ghost: int, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Generate ghost points strategically placed near geometric features.

        SLACK-inspired: Place ghost points where they will maximally confuse
        ICP's correspondence matching - near edges, corners, and distinctive features.
        """
        # Detect edges and high-curvature regions
        edge_scores = self.detect_edges_and_corners(point_cloud)

        # Select high-feature points as bases for ghost points
        threshold = np.percentile(edge_scores, 70)
        feature_mask = edge_scores >= threshold
        feature_indices = np.where(feature_mask)[0]

        if len(feature_indices) < n_ghost:
            feature_indices = np.arange(len(point_cloud))

        # Select bases from feature points
        base_indices = np.random.choice(feature_indices, n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        # Add small offsets to create ambiguous correspondences
        # These ghosts are close enough to real features to confuse ICP
        offsets = np.random.randn(n_ghost, 3) * 0.025  # 2.5cm std - very close
        ghost_points[:, :3] += offsets

        # Modify intensity slightly
        ghost_points[:, 3] += np.random.randn(n_ghost) * 10
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

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

        # Apply geometric distortion - KEY for ICP attacks
        perturbed = self._apply_geometric_distortion(perturbed, params)

        # NEW ATTACKS (SLACK, ICP Attack, ASP inspired)
        # Edge attack: Target edges/corners with larger perturbations
        if params.get("edge_attack_strength", 0) > 0.1:
            perturbed = self._apply_edge_attack(perturbed, params)

        # Temporal drift: Accumulating bias across frames
        if params.get("temporal_drift_strength", 0) > 0.1:
            perturbed = self._apply_temporal_drift(perturbed, params)

        # Scanline perturbation: ASP-inspired attack along laser beams
        if params.get("scanline_strength", 0) > 0.1:
            perturbed = self._apply_scanline_perturbation(perturbed, params)

        return perturbed

    def _apply_edge_attack(self, point_cloud: np.ndarray, params: Dict[str, any]) -> np.ndarray:
        """
        Apply targeted perturbation to edge and corner points.

        SLACK-inspired: "Location of injection matters more than quantity"
        Edge and corner points are critical for ICP - perturbing them
        has disproportionate impact on scan matching.

        Strategy: Shift edge points perpendicular to their principal direction
        to maximally confuse ICP correspondence matching.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)
        if n_points < 100:
            return perturbed

        strength = params.get("edge_attack_strength", 0)
        if strength < 0.1:
            return perturbed

        # Detect edges and corners
        edge_scores = self.detect_edges_and_corners(perturbed)

        # Select top edge/corner points (top 20%)
        threshold = np.percentile(edge_scores, 80)
        edge_mask = edge_scores >= threshold

        if edge_mask.sum() < 10:
            return perturbed

        # Compute local principal direction for edge points
        edge_indices = np.where(edge_mask)[0]
        tree = cKDTree(perturbed[:, :3])

        for idx in edge_indices[: min(500, len(edge_indices))]:  # Limit for speed
            point = perturbed[idx, :3]

            # Get neighbors
            _, neighbors_idx = tree.query(point, k=10)
            neighbors = perturbed[neighbors_idx, :3]

            # PCA to find principal direction
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Perturb perpendicular to principal direction (max confusion for ICP)
            # principal_dir = eigenvectors[:, -1]  # Largest eigenvalue (unused)
            perp_dir = eigenvectors[:, 0]  # Smallest eigenvalue (perpendicular)

            # Shift amount based on strength and edge score
            shift_amount = (
                strength * self.max_edge_shift * (edge_scores[idx] / (edge_scores.max() + 1e-6))
            )
            shift = perp_dir * shift_amount * np.sign(np.random.randn())

            perturbed[idx, :3] += shift

        return perturbed

    def _apply_temporal_drift(self, point_cloud: np.ndarray, params: Dict[str, any]) -> np.ndarray:
        """
        Apply accumulating temporal drift to break loop closure.

        ICP Attack inspired: Consistent bias that accumulates over frames
        prevents SLAM from recognizing previously visited locations.

        This is devastating for loop closure detection.
        """
        perturbed = point_cloud.copy()

        strength = params.get("temporal_drift_strength", 0)
        if strength < 0.1:
            return perturbed

        # Increment frame counter
        self._frame_counter += 1

        # Accumulate drift in the specified direction
        drift_direction = params["noise_direction"]
        frame_drift = drift_direction * strength * self.max_temporal_drift

        # Add to accumulated drift (with some decay to prevent explosion)
        decay = 0.98
        self._accumulated_drift = self._accumulated_drift * decay + frame_drift

        # Apply accumulated drift to all points
        perturbed[:, :3] += self._accumulated_drift

        return perturbed

    def _apply_scanline_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Apply scanline-based perturbation (ASP-inspired).

        ASP Attack: Perturb points along their laser beam directions.
        This simulates particles between sensor and objects, which is
        physically realistic and hard to detect.

        Strategy: Move points along their range direction (toward/away from sensor)
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        strength = params.get("scanline_strength", 0)
        if strength < 0.1 or n_points < 10:
            return perturbed

        # Compute range direction for each point (assuming sensor at origin)
        points_xyz = perturbed[:, :3]
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        range_directions = points_xyz / (ranges + 1e-6)

        # Generate perturbation along scanline (range direction)
        # Mix of random and systematic perturbation
        random_component = np.random.randn(n_points) * 0.03  # 3cm random
        systematic_component = np.sin(np.arange(n_points) * 0.1) * 0.02  # Wave pattern

        scanline_shift = (random_component + systematic_component) * strength
        scanline_shift = scanline_shift[:, np.newaxis] * range_directions

        # Apply shift along scanlines
        perturbed[:, :3] += scanline_shift

        return perturbed

    def reset_temporal_state(self):
        """Reset temporal state for new sequence evaluation."""
        self._accumulated_drift = np.zeros(3)
        self._frame_counter = 0

    def _apply_geometric_distortion(
        self, point_cloud: np.ndarray, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Apply systematic geometric distortion to break ICP convergence.

        ICP is robust to random noise but weak against systematic distortions
        like scaling, shearing, or range-dependent bias.

        This is the KEY adversarial attack for ICP-based SLAM.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)
        if n_points < 10:
            return perturbed

        # Use dedicated geometric_distortion parameter
        distortion_strength = params.get("geometric_distortion", 0.0)
        if distortion_strength < 0.01:
            return perturbed

        points_xyz = perturbed[:, :3]
        center = points_xyz.mean(axis=0)

        # Range from sensor (assuming sensor at origin)
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        max_range = ranges.max() + 1e-6

        # 1. Range-dependent bias: points farther away get pushed more
        # This mimics sensor calibration errors and breaks ICP badly
        # Scale: up to 5cm bias at max range with full distortion
        range_bias = (ranges / max_range) * distortion_strength * 0.05
        direction = params["noise_direction"].reshape(1, 3)
        perturbed[:, :3] += range_bias * direction

        # 2. Angular distortion: rotation that accumulates drift
        # Creates systematic misalignment ICP cannot correct
        # Scale: up to 3 degrees with full distortion
        angle = distortion_strength * 0.05  # ~3 degrees max
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rotation around Z axis (yaw) - most impactful for 2D SLAM
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        perturbed[:, :3] = (perturbed[:, :3] - center) @ rot_z.T + center

        # 3. Non-uniform scaling: stretch geometry in one direction
        # Scale: up to 3% stretch with full distortion
        scale_factor = 1.0 + distortion_strength * 0.03
        scale_direction = np.abs(params["cluster_direction"])
        scale_direction = scale_direction / (scale_direction.sum() + 1e-6)
        scale_matrix = np.eye(3) + np.outer(scale_direction, scale_direction) * (scale_factor - 1)
        perturbed[:, :3] = (perturbed[:, :3] - center) @ scale_matrix + center

        # 4. Per-scan drift: add consistent bias that accumulates over time
        # This is controlled by noise_direction to create drift in a consistent direction
        drift_bias = params["noise_direction"] * distortion_strength * 0.02  # 2cm drift per scan
        perturbed[:, :3] += drift_bias

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
        Generate ghost points to confuse feature matching - AGGRESSIVE strategy.

        NEW STRATEGY: Mix of two types:
        1. Near-duplicates (50%): Close to real points to create ambiguous matches
        2. Outliers (50%): Far from real points to add false geometric features
        """
        # Select random existing points as bases
        base_indices = np.random.choice(len(point_cloud), n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        # Split into two groups: near-duplicates and outliers
        n_near = n_ghost // 2
        n_far = n_ghost - n_near

        # Near-duplicates: Small offsets (2-5 cm) to create ambiguous matches
        if n_near > 0:
            near_offsets = np.random.randn(n_near, 3) * 0.035  # 3.5cm std
            ghost_points[:n_near, :3] += near_offsets

        # Outliers: Large offsets (10-20 cm) to add false features
        if n_far > 0:
            far_offsets = np.random.randn(n_far, 3) * 0.15  # 15cm std
            ghost_points[n_near:, :3] += far_offsets

        # Modify intensity to look plausible but different
        ghost_points[:, 3] += np.random.randn(n_ghost) * 20
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

    def compute_chamfer_distance(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute Chamfer distance between original and perturbed point clouds.

        Uses the standard bidirectional formula:
        CD(A, B) = (1/|A|) * Σ min ||a - b||² + (1/|B|) * Σ min ||b - a||²

        Lower values = more imperceptible perturbation.

        Args:
            original: Original point cloud (N, 3+)
            perturbed: Perturbed point cloud (M, 3+)

        Returns:
            Chamfer distance (sum of mean squared nearest-neighbor distances)
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

        # Chamfer distance = sum of mean squared distances (standard formula)
        # CD(A, B) = mean(dist_forward²) + mean(dist_backward²)
        chamfer = (dist_forward**2).mean() + (dist_backward**2).mean()

        return chamfer

    def compute_perturbation_magnitude(
        self, original: np.ndarray, perturbed: np.ndarray, params: Dict[str, any]
    ) -> float:
        """
        Compute perturbation magnitude for NSGA-III objective.

        Combines Chamfer distance (point displacement) with structural changes
        (dropout/ghost points) to capture both geometric and topological perturbations.

        Formula:
        - Chamfer distance captures point displacement
        - Point count change captures dropout/ghost points as % of original cloud
        - Total perturbation = sqrt(chamfer² + dropout_penalty²)

        Args:
            original: Original point cloud
            perturbed: Perturbed point cloud
            params: Perturbation parameters for decoding dropout/ghost settings

        Returns:
            Combined perturbation magnitude in cm (accounts for both displacement and structure)
        """
        # Compute Chamfer distance in meters² (squared distances)
        chamfer_m2 = self.compute_chamfer_distance(original, perturbed)

        # Convert to cm
        chamfer_cm = np.sqrt(chamfer_m2 * 10000)  # m² to cm²

        # Structural perturbation: point count change normalized by cloud size
        # Dropout removes points, ghost adds points
        n_orig = len(original)
        n_pert = len(perturbed)
        point_change_ratio = abs(n_pert - n_orig) / max(n_orig, 1)

        # Convert ratio to cm-equivalent penalty (scaled to be comparable to Chamfer)
        # A 10% dropout should contribute similarly to ~2cm Chamfer distance
        structural_penalty_cm = point_change_ratio * 20.0  # 10% dropout = 2cm penalty

        # Combined perturbation magnitude using Euclidean norm
        # This creates a smooth trade-off between displacement and structure changes
        total_perturbation_cm = np.sqrt(chamfer_cm**2 + structural_penalty_cm**2)

        return total_perturbation_cm

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
