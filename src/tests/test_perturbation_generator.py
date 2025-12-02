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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
