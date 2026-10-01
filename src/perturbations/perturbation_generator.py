"""
Perturbation generator for LiDAR point clouds.

A genome of 17 values in [-1, 1] is decoded into 13 perturbation parameters
(encode_perturbation), which drive nine operators applied in sequence to every
scan (apply_perturbation): per-point noise with directional bias, cluster
displacement, density-weighted dropout, ghost point injection (random or near
geometric features), global geometric distortion, edge/corner attack, temporal
drift, and scanline shift along the beam direction.

Physical bounds are centimeter-scale (default: 5 cm per-point shift, 8 cm on
edge points, 5 cm drift per frame, at most 3% dropout and 2% ghost points).
Perceptibility is measured with the bidirectional Chamfer distance combined
with a structural penalty for the change in point count
(compute_perturbation_magnitude), expressed in cm.

References:
- FLAT: Flux-Aware Imperceptible Adversarial Attacks (ECCV 2024)
- SLACK: Attacking LiDAR-based SLAM (arXiv 2024)
- Adversarial attack on ICP-based localization (arXiv 2403.05666)
- ASP: Attribution-based Scanline Perturbation (IEEE 2024)
- Survey on adversarial robustness of LiDAR-based ML (2024)
"""

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


# Limiti di plausibilita' della perturbazione (costanti documentate): un
# attaccante che sostituisce la nuvola non e' realistico. La distanza di un
# punto fantasma dal suo punto base e la frazione di punti eliminabili sono
# limitate indipendentemente dal genoma.
MAX_GHOST_OFFSET_M = 0.10   # distanza massima di un punto fantasma (m)
MAX_DROPOUT_RATE = 0.05     # frazione massima di punti eliminabili


def _clip_offsets(offsets):
    """Limita la norma di ogni offset a MAX_GHOST_OFFSET_M."""
    import numpy as _np
    norm = _np.linalg.norm(offsets, axis=1, keepdims=True)
    scale = _np.minimum(1.0, MAX_GHOST_OFFSET_M / _np.maximum(norm, 1e-9))
    return offsets * scale


class PerturbationGenerator:
    """
    Adversarial perturbation generator for LiDAR point clouds.

    Applies per-point perturbations within centimeter-scale bounds and, when
    enabled, concentrates them on high-curvature regions used by SLAM feature
    extraction.
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
        # Point manipulation (kept small for MOLA stability)
        max_dropout_rate: float = 0.03,  # Max 3% point removal
        max_ghost_points_ratio: float = 0.02,  # Max 2% ghost points added
        # Cluster perturbation
        cluster_shift_std: float = 0.03,  # 3 cm cluster displacement std
        n_clusters: int = 5,  # Number of perturbation clusters
        # Edge attack and temporal drift
        max_edge_shift: float = 0.08,  # 8 cm max shift for edge points
        max_temporal_drift: float = 0.05,  # 5 cm max accumulated drift per frame
    ):
        """
        Initialize the perturbation generator.

        Args:
            max_point_shift: Maximum displacement per point in meters (default: 5 cm)
            noise_std: Standard deviation of Gaussian noise in meters (default: 2 cm)
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
        # Mai oltre il limite di plausibilita', qualunque sia il valore richiesto.
        self.max_dropout_rate = min(max_dropout_rate, MAX_DROPOUT_RATE)
        self.max_ghost_points_ratio = max_ghost_points_ratio
        self.cluster_shift_std = cluster_shift_std
        self.n_clusters = n_clusters
        self.max_edge_shift = max_edge_shift
        self.max_temporal_drift = max_temporal_drift
        # Temporal state for the drift operator (persists across frames)
        self._accumulated_drift = np.zeros(3)
        self._frame_counter = 0

    def get_genome_size(self) -> int:
        """
        Return the size of the genome encoding.

        Genome structure (17 parameters):
        - [0-2]: Directional bias for per-point noise (normalized direction)
        - [3]: Noise intensity scale [0, 1]
        - [4]: Curvature targeting strength [0, 1]
        - [5]: Point dropout rate [0, 1]
        - [6]: Ghost points ratio [0, 1]
        - [7-9]: Cluster perturbation direction
        - [10]: Cluster perturbation strength [0, 1]
        - [11]: Spatial correlation of perturbations [0, 1]
        - [12]: Geometric distortion strength [0, 1] (ICP attack)
        - [13]: Edge attack strength [0, 1] (edges/corners, SLACK)
        - [14]: Temporal drift strength [0, 1] (accumulating drift, ICP attack)
        - [15]: Scanline perturbation [0, 1] (ASP)
        - [16]: Strategic ghost placement [0, 1] (ghosts near features)
        """
        return 17

    def encode_perturbation(self, genome: np.ndarray) -> Dict[str, any]:
        """
        Decode a genome into perturbation parameters.

        Args:
            genome: Normalized parameters in range [-1, 1]

        Returns:
            Dictionary with perturbation parameters
        """
        # Rates are mapped to [0, 1]; directions stay in [-1, 1]
        genome = np.clip(genome, -1, 1)

        # Directional bias for noise (unit vector)
        noise_direction = genome[0:3]
        noise_direction_norm = np.linalg.norm(noise_direction)
        if noise_direction_norm > 0:
            noise_direction = noise_direction / noise_direction_norm

        # Noise intensity [0, 1] -> [0, max_point_shift]; zero disables the noise
        noise_intensity = (genome[3] + 1) / 2 * self.max_point_shift

        # Curvature targeting strength [0, 1]
        curvature_strength = (genome[4] + 1) / 2

        # Dropout rate [0, max_dropout_rate]; zero disables the dropout
        dropout_rate = (genome[5] + 1) / 2 * self.max_dropout_rate

        # Ghost points ratio [0, max_ghost_points_ratio]
        ghost_ratio = (genome[6] + 1) / 2 * self.max_ghost_points_ratio

        # Cluster perturbation direction (unit vector)
        cluster_direction = genome[7:10]
        cluster_dir_norm = np.linalg.norm(cluster_direction)
        if cluster_dir_norm > 0:
            cluster_direction = cluster_direction / cluster_dir_norm

        # Cluster strength [0, 1]
        cluster_strength = (genome[10] + 1) / 2

        # Spatial correlation [0, 1]: how correlated nearby point perturbations are
        spatial_correlation = (genome[11] + 1) / 2

        # Geometric distortion [0, 1]: systematic distortions targeting ICP convergence
        geometric_distortion = (genome[12] + 1) / 2  # Maps [-1,1] -> [0, 1]

        # Parameters of the SLACK / ICP attack / ASP inspired operators.
        # The length checks keep older 13-gene genomes decodable.
        # Edge attack strength [0, 1]: targets edges/corners used by ICP
        edge_attack_strength = (genome[13] + 1) / 2 if len(genome) > 13 else 0.0

        # Temporal drift [0, 1]: bias accumulating across frames
        temporal_drift_strength = (genome[14] + 1) / 2 if len(genome) > 14 else 0.0

        # Scanline perturbation [0, 1]: shift along the laser beam (ASP)
        scanline_strength = (genome[15] + 1) / 2 if len(genome) > 15 else 0.0

        # Strategic ghost placement [0, 1]: ghosts placed near geometric features
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
        Compute an approximate local curvature for each point.

        The curvature (smallest eigenvalue over the sum of eigenvalues of the
        local covariance) is computed on a random sample of at most 1000 points
        and propagated to every point from its nearest sampled neighbour.

        Args:
            points: Point cloud (N, 3+) XYZ coordinates
            k: Number of nearest neighbors

        Returns:
            Curvature values for each point (N,)
        """
        n_points = len(points)
        if n_points < k + 1:
            return np.zeros(n_points)

        # Small sample for speed (1000 points max)
        sample_size = min(n_points, 1000)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[sample_indices, :3]

        # KD-tree on the sampled points
        tree = cKDTree(sample_points)

        k_use = min(k, sample_size - 1)

        # Batch query for all sample points (workers=-1 uses all cores)
        _, all_neighbors = tree.query(sample_points, k=k_use + 1, workers=-1)

        # Vectorised over the sample: same covariance/eigenvalue computation as a
        # per-point loop, batched. Covariance uses ddof=1 (divide by k_use).
        sample_curvatures = np.zeros(sample_size)
        if k_use + 1 > 3:
            nb = sample_points[all_neighbors]                     # (S, k+1, 3)
            centered = nb - nb.mean(axis=1, keepdims=True)
            m = centered.shape[1]
            cov = np.einsum("sij,sik->sjk", centered, centered) / (m - 1)
            eigenvalues = np.linalg.eigvalsh(cov)                  # (S,3) ascending
            total = eigenvalues.sum(axis=1)
            ok = total > 0
            # eigenvalues[:, 0] is the smallest eigenvalue
            sample_curvatures[ok] = eigenvalues[ok, 0] / total[ok]

        # Assign curvature to all points from the nearest sampled point
        _, nearest = tree.query(points[:, :3], k=1, workers=-1)
        curvatures = sample_curvatures[nearest]

        return curvatures

    def detect_edges_and_corners(self, points: np.ndarray, k: int = 15) -> np.ndarray:
        """
        Detect edge and corner points using eigenvalue analysis (SLACK).

        Classification based on eigenvalue ratios:
        - Planar: λ1 ≈ λ2 >> λ3 (surface points)
        - Edge: λ1 >> λ2 ≈ λ3 (line features)
        - Corner: λ1 ≈ λ2 ≈ λ3 (3D features)

        Args:
            points: Point cloud (N, 3+)
            k: Number of neighbors for local analysis

        Returns:
            Edge scores for each point (N,), higher = more edge-like
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

        _, all_neighbors = tree.query(sample_points, k=k + 1, workers=-1)

        # Vectorised over the sample: same covariance/eigenvalue computation as a
        # per-point loop, batched.
        #   linearity  = (l1 - l2) / l1        with l1 >= l2 >= l3
        #   sphericity = l3 / l1
        #   score      = linearity + 0.5 * sphericity
        # eigvalsh returns ascending eigenvalues, hence the reversal to [l1, l2, l3].
        if k + 1 > 3:
            nb = sample_points[all_neighbors]                      # (S, k+1, 3)
            centered = nb - nb.mean(axis=1, keepdims=True)
            m = centered.shape[1]
            cov = np.einsum("sij,sik->sjk", centered, centered) / (m - 1)
            ev = np.linalg.eigvalsh(cov)[:, ::-1]                  # descending
            l1 = ev[:, 0] + 1e-10
            linearity = (ev[:, 0] - ev[:, 1]) / l1
            sphericity = ev[:, 2] / l1
            edge_scores = linearity + sphericity * 0.5

        # Assign to all points
        _, nearest = tree.query(points[:, :3], k=1, workers=-1)
        all_edge_scores = edge_scores[nearest]

        return all_edge_scores

    def _compute_perturbation_weights(self, perturbed, n_points, params):
        """Compute curvature-based per-point weights (1.0 on targeted points, 0.3 elsewhere)."""
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
        Apply per-point Gaussian noise with a directional bias (FLAT-style per-point shift).

        noise_intensity sets the magnitude in [0, max_point_shift]; noise_direction
        adds a bias of 30% of the intensity. Each displacement is clipped to
        max_point_shift (default 5 cm).
        """
        if params["noise_intensity"] <= 0.001:
            return perturbed

        # Noise scaled by noise_intensity rather than the fixed noise_std, so that
        # genome[3] = -1 gives no noise and genome[3] = 1 gives the full max_point_shift
        noise = np.random.randn(n_points, 3) * params["noise_intensity"]

        if params["spatial_correlation"] > 0.1:
            noise = self._apply_spatial_correlation(
                perturbed[:, :3], noise, params["spatial_correlation"]
            )

        # Directional bias (30% of intensity in the specified direction)
        directional_component = params["noise_direction"] * params["noise_intensity"] * 0.3
        noise += directional_component
        noise *= perturbation_weights[:, np.newaxis]

        # Clip to max_point_shift
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
        Remove points with a density-weighted probability.

        dropout_rate sets the base removal rate (at most max_dropout_rate, 3%);
        denser regions get up to 10% extra removal, since SLAM matching relies on
        dense geometric structure. At least 90% of the points are always kept.
        """
        if params["dropout_rate"] <= 0.01:
            return perturbed

        # Local density from k-nearest neighbours
        k = min(20, n_points - 1)
        if k > 0:
            tree = cKDTree(perturbed[:, :3])
            # Most expensive step of the generator: a full-cloud neighbour query
            # used only to estimate local density. workers=-1 parallelises it.
            distances, _ = tree.query(perturbed[:, :3], k=k + 1, workers=-1)
            # Local density = inverse of mean distance to k neighbours
            local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
            # Normalise to [0, 1]
            density_weights = (local_density - local_density.min()) / (
                local_density.max() - local_density.min() + 1e-6
            )
        else:
            density_weights = np.ones(n_points)

        # Higher density -> lower keep probability, while staying close to the
        # requested dropout rate (at most 10% extra removal for the densest points)
        keep_prob_base = 1 - params["dropout_rate"]
        keep_prob_per_point = keep_prob_base * (1.0 - 0.1 * density_weights)
        keep_mask = np.random.random(n_points) < keep_prob_per_point

        # Keep at least 90% of the points to bound perceptibility
        min_keep_ratio = max(0.90, 1 - params["dropout_rate"] * 2)
        if keep_mask.sum() < n_points * min_keep_ratio:
            keep_mask = np.random.random(n_points) < min_keep_ratio

        return perturbed[keep_mask]

    def _add_ghost_points(self, perturbed, params):
        """
        Append ghost points to the cloud.

        ghost_ratio sets how many (at most max_ghost_points_ratio, 2%);
        strategic_ghost > 0.5 places them near geometric features (SLACK),
        otherwise they are placed around random points.
        """
        if params["ghost_ratio"] <= 0.01 or len(perturbed) == 0:
            return perturbed

        n_ghost = int(len(perturbed) * params["ghost_ratio"])
        if n_ghost > 0:
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
        Generate ghost points near edges and corners (SLACK: placement matters more
        than quantity).

        Bases are drawn from the top 30% edge scores and offset by Gaussian noise
        with 2.5 cm std, so that the ghosts create ambiguous ICP correspondences.
        """
        edge_scores = self.detect_edges_and_corners(point_cloud)

        # High-feature points as bases for the ghosts
        threshold = np.percentile(edge_scores, 70)
        feature_mask = edge_scores >= threshold
        feature_indices = np.where(feature_mask)[0]

        if len(feature_indices) < n_ghost:
            feature_indices = np.arange(len(point_cloud))

        base_indices = np.random.choice(feature_indices, n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        # Small offsets: close enough to real features to create ambiguous matches
        offsets = _clip_offsets(np.random.randn(n_ghost, 3) * 0.025)  # 2.5 cm std
        ghost_points[:, :3] += offsets

        # Slight intensity change
        ghost_points[:, 3] += np.random.randn(n_ghost) * 10
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

    def apply_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, any], seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply the full perturbation pipeline to a point cloud.

        Operators run in sequence: curvature weights, per-point noise, cluster
        perturbation, dropout, ghost points, geometric distortion, edge attack,
        temporal drift, scanline perturbation.

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

        # Global geometric distortion (ICP attack)
        perturbed = self._apply_geometric_distortion(perturbed, params)

        # Edge attack: larger shifts on edge/corner points (SLACK)
        if params.get("edge_attack_strength", 0) > 0.1:
            perturbed = self._apply_edge_attack(perturbed, params)

        # Temporal drift: bias accumulating across frames (ICP attack)
        if params.get("temporal_drift_strength", 0) > 0.1:
            perturbed = self._apply_temporal_drift(perturbed, params)

        # Scanline perturbation: shift along the laser beam (ASP)
        if params.get("scanline_strength", 0) > 0.1:
            perturbed = self._apply_scanline_perturbation(perturbed, params)

        return perturbed

    def _apply_edge_attack(self, point_cloud: np.ndarray, params: Dict[str, any]) -> np.ndarray:
        """
        Shift edge and corner points perpendicular to their principal direction.

        Inspired by SLACK ("location of injection matters more than quantity"):
        edge_attack_strength scales the shift, up to max_edge_shift (8 cm), on the
        top 20% edge scores (at most 500 points). The shift direction is the
        smallest-eigenvalue axis of the local PCA, which disturbs ICP correspondences.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)
        if n_points < 100:
            return perturbed

        strength = params.get("edge_attack_strength", 0)
        if strength < 0.1:
            return perturbed

        edge_scores = self.detect_edges_and_corners(perturbed)

        # Top 20% edge/corner points
        threshold = np.percentile(edge_scores, 80)
        edge_mask = edge_scores >= threshold

        if edge_mask.sum() < 10:
            return perturbed

        edge_indices = np.where(edge_mask)[0]
        tree = cKDTree(perturbed[:, :3])

        sel = edge_indices[: min(500, len(edge_indices))]  # Limit for speed

        # Neighbour indices are queried in one batch on the unmodified cloud:
        # each selected index is visited once, so the result equals per-point queries.
        if len(sel) == 0:
            return perturbed
        _, all_nb_idx = tree.query(perturbed[sel, :3], k=10, workers=-1)

        # The PCA loop is kept sequential on purpose: neighbours may already have
        # been shifted in a previous iteration.
        edge_max = edge_scores.max() + 1e-6
        for j, idx in enumerate(sel):
            neighbors = perturbed[all_nb_idx[j], :3]

            # PCA of the local neighbourhood
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Shift along the smallest-eigenvalue axis (perpendicular to the edge)
            perp_dir = eigenvectors[:, 0]

            # Shift amount scaled by strength and normalised edge score
            shift_amount = strength * self.max_edge_shift * (edge_scores[idx] / edge_max)
            shift = perp_dir * shift_amount * np.sign(np.random.randn())

            perturbed[idx, :3] += shift

        return perturbed

    def _apply_temporal_drift(self, point_cloud: np.ndarray, params: Dict[str, any]) -> np.ndarray:
        """
        Translate the whole cloud by a bias that accumulates across frames.

        Global transformation, unlike the per-point operators. Inspired by the ICP
        adversarial attack: temporal_drift_strength scales the per-frame step along
        noise_direction, up to max_temporal_drift (5 cm); the accumulated drift
        decays by 0.98 per frame so it stays bounded. Targets loop closure.
        """
        perturbed = point_cloud.copy()

        strength = params.get("temporal_drift_strength", 0)
        if strength < 0.1:
            return perturbed

        self._frame_counter += 1

        # Per-frame drift along the noise direction
        drift_direction = params["noise_direction"]
        frame_drift = drift_direction * strength * self.max_temporal_drift

        # Accumulate with decay so the drift stays bounded
        decay = 0.98
        self._accumulated_drift = self._accumulated_drift * decay + frame_drift

        # Apply the accumulated drift to all points
        perturbed[:, :3] += self._accumulated_drift

        return perturbed

    def _apply_scanline_perturbation(
        self, point_cloud: np.ndarray, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Shift points along their range direction (toward/away from the sensor).

        Inspired by ASP: a shift along the laser beam resembles particles between
        sensor and object, which is physically plausible. scanline_strength scales
        a mix of Gaussian noise (3 cm std) and a sinusoidal pattern (2 cm amplitude).
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        strength = params.get("scanline_strength", 0)
        if strength < 0.1 or n_points < 10:
            return perturbed

        # Range direction of each point (sensor at the origin)
        points_xyz = perturbed[:, :3]
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        range_directions = points_xyz / (ranges + 1e-6)

        # Random plus systematic component along the beam
        random_component = np.random.randn(n_points) * 0.03  # 3 cm std
        systematic_component = np.sin(np.arange(n_points) * 0.1) * 0.02  # Wave pattern

        scanline_shift = (random_component + systematic_component) * strength
        scanline_shift = scanline_shift[:, np.newaxis] * range_directions

        perturbed[:, :3] += scanline_shift

        return perturbed

    def reset_temporal_state(self):
        """Reset the accumulated drift and frame counter before a new sequence."""
        self._accumulated_drift = np.zeros(3)
        self._frame_counter = 0

    def _apply_geometric_distortion(
        self, point_cloud: np.ndarray, params: Dict[str, any]
    ) -> np.ndarray:
        """
        Apply systematic distortions to the whole cloud (global transformation,
        unlike the per-point operators).

        Inspired by the ICP adversarial attack: ICP tolerates random noise but not
        systematic distortion. geometric_distortion scales four terms: range-dependent
        bias (up to 5 cm at max range), yaw rotation (up to 0.05 rad, about 3 degrees),
        non-uniform scaling (up to 3%) and a constant per-scan bias (up to 2 cm).
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)
        if n_points < 10:
            return perturbed

        distortion_strength = params.get("geometric_distortion", 0.0)
        if distortion_strength < 0.01:
            return perturbed

        points_xyz = perturbed[:, :3]
        center = points_xyz.mean(axis=0)

        # Range from the sensor (sensor at the origin)
        ranges = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        max_range = ranges.max() + 1e-6

        # 1. Range-dependent bias: farther points are pushed more, as in a
        #    calibration error. Up to 5 cm at max range with full distortion.
        range_bias = (ranges / max_range) * distortion_strength * 0.05
        direction = params["noise_direction"].reshape(1, 3)
        perturbed[:, :3] += range_bias * direction

        # 2. Yaw rotation around the cloud centre, up to 0.05 rad (about 3 degrees)
        angle = distortion_strength * 0.05
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        perturbed[:, :3] = (perturbed[:, :3] - center) @ rot_z.T + center

        # 3. Non-uniform scaling along cluster_direction, up to 3% stretch
        scale_factor = 1.0 + distortion_strength * 0.03
        scale_direction = np.abs(params["cluster_direction"])
        scale_direction = scale_direction / (scale_direction.sum() + 1e-6)
        scale_matrix = np.eye(3) + np.outer(scale_direction, scale_direction) * (scale_factor - 1)
        perturbed[:, :3] = (perturbed[:, :3] - center) @ scale_matrix + center

        # 4. Constant per-scan bias along noise_direction, up to 2 cm
        drift_bias = params["noise_direction"] * distortion_strength * 0.02
        perturbed[:, :3] += drift_bias

        return perturbed

    def _apply_spatial_correlation(
        self, points: np.ndarray, noise: np.ndarray, correlation: float
    ) -> np.ndarray:
        """
        Make the noise spatially correlated so nearby points move similarly.

        Approximation: blend the per-point noise with its global mean instead of
        averaging over neighbours. Skipped when correlation < 0.3.
        """
        if correlation < 0.3 or len(points) < 100:
            return noise

        global_shift = noise.mean(axis=0) * correlation
        correlated_noise = noise * (1 - correlation * 0.5) + global_shift

        return correlated_noise

    def _apply_cluster_perturbation(
        self, point_cloud: np.ndarray, direction: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Displace random clusters of points, as in a localised sensor error.

        cluster_strength scales a shift along cluster_direction (std cluster_shift_std,
        3 cm) with an exponential falloff from each of the n_clusters centres; the
        cluster radius is the 10th percentile of the distances to the cloud centre.
        """
        perturbed = point_cloud.copy()
        n_points = len(perturbed)

        if n_points < 100:
            return perturbed

        # Random cluster centres
        n_clusters = min(self.n_clusters, n_points // 100)
        cluster_centers_idx = np.random.choice(n_points, n_clusters, replace=False)

        # Cluster radius
        cloud_center = perturbed[:, :3].mean(axis=0)
        distances_to_center = np.linalg.norm(perturbed[:, :3] - cloud_center, axis=1)
        cluster_radius = np.percentile(distances_to_center, 10)

        for center_idx in cluster_centers_idx:
            center = perturbed[center_idx, :3]

            distances = np.linalg.norm(perturbed[:, :3] - center, axis=1)
            mask = distances < cluster_radius

            if mask.sum() > 0:
                # Cluster-specific displacement: directed part plus a random part
                cluster_shift = direction * strength * self.cluster_shift_std
                cluster_shift += np.random.randn(3) * self.cluster_shift_std * strength * 0.5

                # Exponential falloff from the cluster centre
                falloff = np.exp(-distances[mask] / cluster_radius)
                perturbed[mask, :3] += cluster_shift * falloff[:, np.newaxis]

        return perturbed

    def _generate_ghost_points(self, point_cloud: np.ndarray, n_ghost: int) -> np.ndarray:
        """
        Generate ghost points around random existing points.

        Half are near-duplicates (3.5 cm std) that create ambiguous matches, half
        are outliers (15 cm std) that add false geometric features.
        """
        base_indices = np.random.choice(len(point_cloud), n_ghost, replace=True)
        ghost_points = point_cloud[base_indices].copy()

        n_near = n_ghost // 2
        n_far = n_ghost - n_near

        # Near-duplicates: small offsets
        if n_near > 0:
            near_offsets = np.random.randn(n_near, 3) * 0.035  # 3.5 cm std
            ghost_points[:n_near, :3] += _clip_offsets(near_offsets)

        # Outliers: larger offsets, comunque entro il limite di plausibilita'
        if n_far > 0:
            far_offsets = np.random.randn(n_far, 3) * 0.08  # 8 cm std
            ghost_points[n_near:, :3] += _clip_offsets(far_offsets)

        # Plausible but different intensity
        ghost_points[:, 3] += np.random.randn(n_ghost) * 20
        ghost_points[:, 3] = np.clip(ghost_points[:, 3], 0, 255)

        return ghost_points

    def compute_chamfer_distance(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute the bidirectional Chamfer distance between two point clouds.

        CD(A, B) = (1/|A|) * Σ min ||a - b||² + (1/|B|) * Σ min ||b - a||²

        Lower values mean a less perceptible perturbation.

        Args:
            original: Original point cloud (N, 3+)
            perturbed: Perturbed point cloud (M, 3+)

        Returns:
            Chamfer distance (sum of mean squared nearest-neighbour distances, m²)
        """
        if len(original) == 0 or len(perturbed) == 0:
            return float("inf")

        tree_orig = cKDTree(original[:, :3])
        tree_pert = cKDTree(perturbed[:, :3])

        # Forward: for each perturbed point, nearest original point
        dist_forward, _ = tree_orig.query(perturbed[:, :3], k=1, workers=-1)

        # Backward: for each original point, nearest perturbed point
        dist_backward, _ = tree_pert.query(original[:, :3], k=1, workers=-1)

        chamfer = (dist_forward**2).mean() + (dist_backward**2).mean()

        return chamfer

    def compute_perturbation_magnitude(
        self, original: np.ndarray, perturbed: np.ndarray, params: Dict[str, any]
    ) -> float:
        """
        Compute the perturbation magnitude used as the NSGA-III perceptibility objective.

        Formula, with CD the Chamfer distance in m² and r = |M - N| / N the relative
        change in point count (dropout and ghost points):
            chamfer_cm    = sqrt(CD * 10000)
            structural_cm = r * 20        (10% change in point count = 2 cm)
            magnitude     = sqrt(chamfer_cm² + structural_cm²)

        Args:
            original: Original point cloud
            perturbed: Perturbed point cloud
            params: Perturbation parameters (unused, kept for interface compatibility)

        Returns:
            Combined perturbation magnitude in cm
        """
        chamfer_m2 = self.compute_chamfer_distance(original, perturbed)

        # m² -> cm
        chamfer_cm = np.sqrt(chamfer_m2 * 10000)

        # Structural term: change in point count relative to the original cloud
        n_orig = len(original)
        n_pert = len(perturbed)
        point_change_ratio = abs(n_pert - n_orig) / max(n_orig, 1)

        # Scaled so that a 10% change in point count weighs like 2 cm of Chamfer
        structural_penalty_cm = point_change_ratio * 20.0

        # Euclidean combination of displacement and structural terms
        total_perturbation_cm = np.sqrt(chamfer_cm**2 + structural_penalty_cm**2)

        return total_perturbation_cm

    def random_genome(self, size: Optional[int] = None) -> np.ndarray:
        """
        Generate random genome(s).

        Args:
            size: Number of genomes to generate (None for a single genome)

        Returns:
            Random genome(s) in range [-1, 1]
        """
        if size is None:
            return np.random.uniform(-1, 1, self.get_genome_size())
        return np.random.uniform(-1, 1, (size, self.get_genome_size()))
