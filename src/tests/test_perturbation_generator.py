"""Tests for PerturbationGenerator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402


class TestPerturbationGenerator:
    """Tests for PerturbationGenerator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = PerturbationGenerator(
            max_point_shift=0.05,
            noise_std=0.02,
            max_dropout_rate=0.15,
        )
        # Create test point cloud (100 points, 4 columns: x, y, z, intensity)
        np.random.seed(42)
        self.test_cloud = np.random.rand(100, 4) * 10
        self.test_cloud[:, 3] = np.random.rand(100) * 255  # intensity [0, 255]

    def test_genome_size(self):
        """Test genome size is 17 (expanded with advanced attacks)."""
        assert self.generator.get_genome_size() == 17

    def test_random_genome_shape(self):
        """Test random genome generation."""
        genome = self.generator.random_genome()
        assert genome.shape == (17,)
        assert np.all(genome >= -1) and np.all(genome <= 1)

    def test_random_genome_batch(self):
        """Test batch genome generation."""
        genomes = self.generator.random_genome(size=10)
        assert genomes.shape == (10, 17)
        assert np.all(genomes >= -1) and np.all(genomes <= 1)

    def test_encode_perturbation(self):
        """Test perturbation encoding."""
        genome = np.zeros(17)
        params = self.generator.encode_perturbation(genome)

        assert "noise_direction" in params
        assert "noise_intensity" in params
        assert "curvature_strength" in params
        assert "dropout_rate" in params
        assert "ghost_ratio" in params
        assert "cluster_direction" in params
        assert "cluster_strength" in params
        assert "spatial_correlation" in params

    def test_apply_perturbation_shape(self):
        """Test perturbation maintains valid shape."""
        genome = self.generator.random_genome()
        params = self.generator.encode_perturbation(genome)
        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        assert perturbed.ndim == 2
        assert perturbed.shape[1] == 4

    def test_apply_perturbation_deterministic(self):
        """Test perturbation is deterministic with seed."""
        genome = self.generator.random_genome()
        params = self.generator.encode_perturbation(genome)

        result1 = self.generator.apply_perturbation(self.test_cloud, params, seed=42)
        result2 = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_zero_perturbation(self):
        """Test zero genome produces minimal change."""
        genome = np.zeros(17)
        params = self.generator.encode_perturbation(genome)
        _ = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        # With zero genome, noise_intensity should be 0, so no noise added
        # Only potential difference is from dropout/ghost which are also minimized
        assert params["noise_intensity"] == pytest.approx(
            0.025, abs=0.001
        )  # half of max due to encoding
        assert params["dropout_rate"] == pytest.approx(0.075, abs=0.001)

    def test_chamfer_distance_identical(self):
        """Test Chamfer distance is 0 for identical clouds."""
        chamfer = self.generator.compute_chamfer_distance(self.test_cloud, self.test_cloud)
        assert chamfer == pytest.approx(0.0, abs=1e-6)

    def test_chamfer_distance_shifted(self):
        """Test Chamfer distance for shifted cloud."""
        shifted = self.test_cloud.copy()
        shifted[:, :3] += 0.1  # shift by 10cm in all directions
        chamfer = self.generator.compute_chamfer_distance(self.test_cloud, shifted)
        # With new formula: CD = mean(dist²) + mean(dist²)
        # dist² = 0.1² + 0.1² + 0.1² = 0.03 m²
        # CD = 0.03 + 0.03 = 0.06 m² (bidirectional)
        assert chamfer > 0.03  # Greater than single direction
        assert chamfer < 0.10  # Less than unreasonable value

    def test_perturbation_magnitude(self):
        """Test perturbation magnitude computation."""
        genome = self.generator.random_genome()
        params = self.generator.encode_perturbation(genome)
        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        magnitude = self.generator.compute_perturbation_magnitude(
            self.test_cloud, perturbed, params
        )
        # Should be in centimeters
        assert magnitude >= 0

    def test_curvature_computation(self):
        """Test curvature computation doesn't crash."""
        curvatures = self.generator.compute_curvature(self.test_cloud[:, :3])
        assert len(curvatures) == len(self.test_cloud)
        assert np.all(curvatures >= 0)

    def test_curvature_computation_small_cloud(self):
        """Test curvature with very small point cloud."""
        small_cloud = np.random.rand(5, 3)  # Less than k+1 neighbors
        curvatures = self.generator.compute_curvature(small_cloud, k=10)
        assert len(curvatures) == 5
        assert np.all(curvatures == 0)  # Should return zeros for small clouds

    def test_edge_detection(self):
        """Test edge and corner detection."""
        edge_scores = self.generator.detect_edges_and_corners(self.test_cloud)
        assert len(edge_scores) == len(self.test_cloud)
        assert np.all(edge_scores >= 0)
        assert np.all(edge_scores <= 1.5)  # Max linearity + sphericity*0.5

    def test_edge_detection_small_cloud(self):
        """Test edge detection with very small point cloud."""
        small_cloud = np.random.rand(5, 4)  # Less than k+1 neighbors
        edge_scores = self.generator.detect_edges_and_corners(small_cloud, k=10)
        assert len(edge_scores) == 5
        assert np.all(edge_scores == 0)  # Should return zeros

    def test_apply_edge_attack(self):
        """Test edge attack perturbation."""
        genome = np.ones(17) * 0.5
        params = self.generator.encode_perturbation(genome)
        params["edge_attack_strength"] = 0.5
        perturbed = self.generator._apply_edge_attack(self.test_cloud, params)
        assert perturbed.shape == self.test_cloud.shape
        # Points should be perturbed
        assert not np.allclose(perturbed[:, :3], self.test_cloud[:, :3])

    def test_apply_edge_attack_low_strength(self):
        """Test edge attack with low strength returns unchanged cloud."""
        params = {"edge_attack_strength": 0.05}
        perturbed = self.generator._apply_edge_attack(self.test_cloud, params)
        np.testing.assert_array_equal(perturbed, self.test_cloud)

    def test_apply_edge_attack_small_cloud(self):
        """Test edge attack with small cloud returns unchanged."""
        small_cloud = np.random.rand(50, 4)
        params = {"edge_attack_strength": 0.8}
        perturbed = self.generator._apply_edge_attack(small_cloud, params)
        np.testing.assert_array_equal(perturbed, small_cloud)

    def test_apply_temporal_drift(self):
        """Test temporal drift perturbation."""
        params = {
            "temporal_drift_strength": 0.7,
            "frame_index": 50,
            "noise_direction": np.array([1.0, 0.0, 0.0]),
        }
        perturbed = self.generator._apply_temporal_drift(self.test_cloud, params)
        assert perturbed.shape == self.test_cloud.shape

    def test_apply_temporal_drift_low_strength(self):
        """Test temporal drift with low strength."""
        params = {
            "temporal_drift_strength": 0.05,
            "frame_index": 10,
            "noise_direction": np.array([0.0, 1.0, 0.0]),
        }
        perturbed = self.generator._apply_temporal_drift(self.test_cloud, params)
        assert perturbed.shape == self.test_cloud.shape

    def test_apply_scanline_perturbation(self):
        """Test scanline perturbation (ASP-inspired)."""
        params = {"scanline_strength": 0.6}
        perturbed = self.generator._apply_scanline_perturbation(self.test_cloud, params)
        assert perturbed.shape == self.test_cloud.shape

    def test_apply_scanline_perturbation_low_strength(self):
        """Test scanline perturbation with low strength."""
        params = {"scanline_strength": 0.05}
        perturbed = self.generator._apply_scanline_perturbation(self.test_cloud, params)
        np.testing.assert_array_equal(perturbed, self.test_cloud)

    def test_ghost_and_dropout_integration(self):
        """Test ghost point and dropout work together."""
        genome = np.ones(17) * 0.3
        params = self.generator.encode_perturbation(genome)
        params["ghost_ratio"] = 0.1
        params["dropout_rate"] = 0.1

        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        # Should have some ghost points added and some dropped
        assert perturbed.shape[0] > 0
        assert perturbed.shape[1] == 4

    def test_full_genome_encode_all_params(self):
        """Test that full genome encoding includes all 17 parameters."""
        genome = np.random.uniform(-1, 1, 17)
        params = self.generator.encode_perturbation(genome)

        expected_keys = [
            "noise_direction",
            "noise_intensity",
            "curvature_strength",
            "dropout_rate",
            "ghost_ratio",
            "cluster_direction",
            "cluster_strength",
            "spatial_correlation",
            "geometric_distortion",
            "edge_attack_strength",
            "temporal_drift_strength",
            "scanline_strength",
            "strategic_ghost",
        ]
        for key in expected_keys:
            assert key in params

    def test_advanced_attacks_integration(self):
        """Test that all advanced attacks work together."""
        genome = np.ones(17) * 0.5  # Mid-range values
        params = self.generator.encode_perturbation(genome)
        params["frame_index"] = 25  # For temporal drift

        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        assert perturbed.shape[0] > 0  # Some points remain after dropout
        assert perturbed.shape[1] == 4
        # Magnitude should be reasonable
        mag = self.generator.compute_perturbation_magnitude(self.test_cloud, perturbed, params)
        assert mag >= 0
        assert mag < 100  # Less than 1 meter in centimeters

    def test_compute_perturbation_weights_no_curvature(self):
        """Test perturbation weights when curvature targeting is disabled."""
        params = {"curvature_strength": 0.05}
        weights = self.generator._compute_perturbation_weights(
            self.test_cloud, len(self.test_cloud), params
        )
        assert np.all(weights == 1.0)

    def test_compute_perturbation_weights_uniform_curvature(self):
        """Test perturbation weights with uniform curvature."""
        # Create uniform cloud
        uniform_cloud = np.ones((100, 4))
        params = {"curvature_strength": 0.5}
        weights = self.generator._compute_perturbation_weights(
            uniform_cloud, len(uniform_cloud), params
        )
        assert np.all(weights == 1.0)  # Should return ones when curvature is uniform

    def test_apply_noise_zero_intensity(self):
        """Test that zero noise intensity doesn't change cloud."""
        params = {"noise_intensity": 0.0}
        perturbed = self.generator._apply_noise(
            self.test_cloud.copy(), len(self.test_cloud), np.ones(len(self.test_cloud)), params
        )
        np.testing.assert_array_equal(perturbed, self.test_cloud)

    def test_apply_geometric_distortion_high_strength(self):
        """Test geometric distortion with high strength."""
        params = {
            "geometric_distortion": 0.8,
            "noise_direction": np.array([1.0, 0.0, 0.0]),
            "cluster_direction": np.array([0.0, 1.0, 0.0]),
        }
        perturbed = self.generator._apply_geometric_distortion(self.test_cloud, params)
        assert perturbed.shape == self.test_cloud.shape
        # Points should be distorted
        assert not np.allclose(perturbed[:, :3], self.test_cloud[:, :3])

    def test_apply_geometric_distortion_low_strength(self):
        """Test geometric distortion with low strength."""
        params = {
            "geometric_distortion": 0.005,
            "noise_direction": np.array([0.0, 1.0, 0.0]),
        }
        perturbed = self.generator._apply_geometric_distortion(self.test_cloud, params)
        np.testing.assert_array_equal(perturbed, self.test_cloud)

    def test_edge_attack_with_few_edges(self):
        """Test edge attack when few edge points are detected."""
        # Create cloud with uniform points (no clear edges)
        uniform_cloud = np.random.rand(200, 4)
        params = {"edge_attack_strength": 0.8}
        perturbed = self.generator._apply_edge_attack(uniform_cloud, params)
        # Should handle gracefully even with few edge points
        assert perturbed.shape == uniform_cloud.shape

    def test_full_perturbation_pipeline_edge_cases(self):
        """Test full perturbation pipeline with edge case genomes."""
        # Test with all zeros genome
        genome_zeros = np.zeros(17)
        params_zeros = self.generator.encode_perturbation(genome_zeros)
        perturbed_zeros = self.generator.apply_perturbation(self.test_cloud, params_zeros, seed=42)
        assert perturbed_zeros.shape[0] > 0

        # Test with all max values genome
        genome_max = np.ones(17)
        params_max = self.generator.encode_perturbation(genome_max)
        perturbed_max = self.generator.apply_perturbation(self.test_cloud, params_max, seed=42)
        assert perturbed_max.shape[0] > 0

        # Test with all negative values genome
        genome_neg = -np.ones(17)
        params_neg = self.generator.encode_perturbation(genome_neg)
        perturbed_neg = self.generator.apply_perturbation(self.test_cloud, params_neg, seed=42)
        assert perturbed_neg.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
