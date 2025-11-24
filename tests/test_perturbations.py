"""
Tests for perturbation module.

Tests the PerturbationGenerator and point cloud operations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perturbations.perturbation_generator import PerturbationGenerator
from perturbations.point_cloud_ops import (
    compute_perturbation_magnitude,
    downsample_point_cloud,
    validate_point_cloud,
)


class TestPerturbationGenerator:
    """Tests for PerturbationGenerator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = PerturbationGenerator()
        # Create a simple test point cloud
        self.test_cloud = np.array(
            [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 150.0],
                [0.0, 0.0, 1.0, 200.0],
                [1.0, 1.0, 0.0, 125.0],
            ]
        )

    def test_genome_size(self):
        """Test genome size is correct."""
        assert self.generator.get_genome_size() == 8

    def test_random_genome_shape(self):
        """Test random genome generation."""
        # Single genome
        genome = self.generator.random_genome()
        assert genome.shape == (8,)
        assert np.all(genome >= -1) and np.all(genome <= 1)

        # Multiple genomes
        genomes = self.generator.random_genome(size=10)
        assert genomes.shape == (10, 8)
        assert np.all(genomes >= -1) and np.all(genomes <= 1)

    def test_encode_decode_consistency(self):
        """Test encoding and decoding are consistent."""
        genome = self.generator.random_genome()
        params = self.generator.encode_perturbation(genome)
        decoded = self.generator.decode_perturbation(params)

        np.testing.assert_allclose(genome, decoded, rtol=1e-5)

    def test_encode_perturbation(self):
        """Test perturbation encoding."""
        genome = np.ones(8) * 0.5
        params = self.generator.encode_perturbation(genome)

        assert "translation" in params
        assert "rotation" in params
        assert "intensity_scale" in params
        assert "dropout_rate" in params

        assert params["translation"].shape == (3,)
        assert params["rotation"].shape == (3,)

    def test_apply_perturbation_shape(self):
        """Test perturbation application maintains valid shape."""
        genome = self.generator.random_genome()
        params = self.generator.encode_perturbation(genome)

        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        assert perturbed.ndim == 2
        assert perturbed.shape[1] == 4
        # May have fewer points due to dropout
        assert perturbed.shape[0] <= self.test_cloud.shape[0]

    def test_apply_perturbation_deterministic(self):
        """Test perturbation is deterministic with seed."""
        genome = self.generator.random_genome()
        params = self.generator.encode_perturbation(genome)

        result1 = self.generator.apply_perturbation(self.test_cloud, params, seed=42)
        result2 = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_translation_applied(self):
        """Test translation is correctly applied."""
        genome = np.zeros(8)
        genome[0] = 1.0  # Max translation in X
        params = self.generator.encode_perturbation(genome)

        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        # X coordinates should be shifted
        assert np.any(perturbed[:, 0] != self.test_cloud[: len(perturbed), 0])

    def test_intensity_clipping(self):
        """Test intensity values are clipped to valid range."""
        genome = np.zeros(8)
        genome[6] = 10.0  # Large intensity change
        params = self.generator.encode_perturbation(genome)

        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        # Intensity should be clipped to [0, 255]
        assert np.all(perturbed[:, 3] >= 0)
        assert np.all(perturbed[:, 3] <= 255)

    def test_zero_perturbation(self):
        """Test zero perturbation leaves cloud mostly unchanged."""
        genome = np.zeros(8)
        params = self.generator.encode_perturbation(genome)

        perturbed = self.generator.apply_perturbation(self.test_cloud, params, seed=42)

        # Should be very close to original (except possible dropout)
        np.testing.assert_allclose(
            perturbed[:, :3], self.test_cloud[: len(perturbed), :3], rtol=1e-5
        )


class TestPointCloudOps:
    """Tests for point cloud operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_cloud = np.array(
            [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 150.0],
                [0.0, 0.0, 1.0, 200.0],
            ]
        )

    def test_validate_valid_cloud(self):
        """Test validation accepts valid point cloud."""
        is_valid, error = validate_point_cloud(self.test_cloud)
        assert is_valid
        assert error is None

    def test_validate_empty_cloud(self):
        """Test validation rejects empty cloud."""
        empty = np.array([]).reshape(0, 4)
        is_valid, error = validate_point_cloud(empty)
        assert not is_valid
        assert "empty" in error.lower()

    def test_validate_wrong_dimensions(self):
        """Test validation rejects wrong dimensions."""
        wrong = np.array([1, 2, 3, 4])  # 1D
        is_valid, error = validate_point_cloud(wrong)
        assert not is_valid
        assert "2D" in error

    def test_validate_nan_values(self):
        """Test validation rejects NaN values."""
        nan_cloud = self.test_cloud.copy()
        nan_cloud[0, 0] = np.nan
        is_valid, error = validate_point_cloud(nan_cloud)
        assert not is_valid
        assert "NaN" in error or "Inf" in error

    def test_compute_perturbation_magnitude(self):
        """Test perturbation magnitude computation."""
        original = np.array([[0, 0, 0, 100], [1, 0, 0, 100]])
        perturbed = np.array([[1, 0, 0, 100], [2, 0, 0, 100]])

        magnitude = compute_perturbation_magnitude(original, perturbed)
        expected = np.sqrt(2)  # sqrt((1-0)^2 + (2-1)^2)

        np.testing.assert_allclose(magnitude, expected, rtol=1e-5)

    def test_downsample_reduces_points(self):
        """Test downsampling reduces number of points."""
        # Create dense point cloud
        dense = np.random.rand(1000, 4)
        downsampled = downsample_point_cloud(dense, voxel_size=0.1)

        assert len(downsampled) <= len(dense)
        assert downsampled.shape[1] == 4


def test_integration_perturbation_pipeline():
    """Integration test: full perturbation pipeline."""
    generator = PerturbationGenerator()

    # Create test cloud
    cloud = np.random.rand(100, 4) * 10

    # Generate random perturbation
    genome = generator.random_genome()
    params = generator.encode_perturbation(genome)

    # Apply perturbation
    perturbed = generator.apply_perturbation(cloud, params, seed=42)

    # Validate result
    is_valid, error = validate_point_cloud(perturbed)
    assert is_valid, f"Perturbed cloud invalid: {error}"

    # Compute magnitude
    magnitude = compute_perturbation_magnitude(cloud, perturbed)
    assert magnitude >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
