"""Tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (  # noqa: E402
    compute_imperceptibility,
    compute_localization_error,
    compute_multi_objective_fitness,
    normalize_fitness,
)


class TestLocalizationError:
    """Tests for localization error computation."""

    def test_ate_identical_trajectories(self):
        """Test ATE with identical trajectories."""
        trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        error = compute_localization_error(trajectory, trajectory, method="ate")
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_ate_offset_trajectories(self):
        """Test ATE with constant offset - Umeyama alignment corrects pure translation."""
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        estimated = gt + np.array([1, 1, 0])
        error = compute_localization_error(gt, estimated, method="ate")
        # Umeyama alignment corrects constant translation offset to ~0
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_rpe_identical_trajectories(self):
        """Test RPE with identical trajectories."""
        trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        error = compute_localization_error(trajectory, trajectory, method="rpe")
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_final_position_error(self):
        """Test final position error."""
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        estimated = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]])
        error = compute_localization_error(gt, estimated, method="final")
        assert error == pytest.approx(2.0, abs=1e-6)

    def test_empty_trajectory_returns_inf(self):
        """Test that empty trajectories return infinity."""
        gt = np.array([[0, 0, 0]])
        empty = np.array([])
        error = compute_localization_error(gt, empty, method="ate")
        assert error == float("inf")

    def test_unknown_method_raises_error(self):
        """Test that unknown method raises ValueError."""
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        with pytest.raises(ValueError, match="Unknown error method"):
            compute_localization_error(gt, gt, method="unknown")

    def test_rpe_short_trajectory(self):
        """Test RPE with trajectory shorter than delta."""
        gt = np.array([[0, 0, 0]])
        error = compute_localization_error(gt, gt, method="rpe")
        assert error == float("inf")

    def test_ate_short_trajectory(self):
        """Test ATE with very short trajectory (less than 3 points)."""
        gt = np.array([[0, 0, 0], [1, 0, 0]])
        est = np.array([[0, 0, 0], [2, 0, 0]])
        error = compute_localization_error(gt, est, method="ate")
        # Should compute simple RMSE without alignment
        assert error > 0


class TestImperceptibility:
    """Tests for imperceptibility metrics."""

    def test_l2_identical_clouds(self):
        """Test L2 norm with identical clouds."""
        cloud = np.random.rand(100, 4)
        magnitude = compute_imperceptibility(cloud, cloud, method="l2")
        assert magnitude == pytest.approx(0.0, abs=1e-6)

    def test_l2_known_perturbation(self):
        """Test L2 norm with known perturbation."""
        original = np.zeros((10, 4))
        perturbed = np.ones((10, 4))
        magnitude = compute_imperceptibility(original, perturbed, method="l2")
        expected = np.linalg.norm(np.ones((10, 3)))
        assert magnitude == pytest.approx(expected, abs=1e-6)

    def test_linf_norm(self):
        """Test L-infinity norm."""
        original = np.zeros((10, 4))
        perturbed = original.copy()
        perturbed[0, :3] = [5, 3, 2]
        magnitude = compute_imperceptibility(original, perturbed, method="linf")
        assert magnitude == pytest.approx(5.0, abs=1e-6)

    def test_relative_norm(self):
        """Test relative L2 norm."""
        original = np.ones((10, 4))
        perturbed = original.copy()
        perturbed[:, :3] = 2.0
        magnitude = compute_imperceptibility(original, perturbed, method="relative")
        # diff = ones, orig = ones, so relative = ||diff|| / ||orig||
        expected = np.linalg.norm(np.ones((10, 3))) / np.linalg.norm(np.ones((10, 3)))
        assert magnitude == pytest.approx(expected, abs=1e-6)

    def test_relative_norm_zero_original(self):
        """Test relative norm with zero original cloud."""
        original = np.zeros((10, 4))
        perturbed = np.ones((10, 4))
        magnitude = compute_imperceptibility(original, perturbed, method="relative")
        assert magnitude == pytest.approx(0.0, abs=1e-6)

    def test_empty_clouds(self):
        """Test with empty point clouds."""
        empty = np.array([])
        cloud = np.random.rand(100, 4)
        magnitude = compute_imperceptibility(empty, cloud, method="l2")
        assert magnitude == pytest.approx(0.0, abs=1e-6)

    def test_unknown_method_raises_error(self):
        """Test that unknown method raises ValueError."""
        cloud = np.random.rand(100, 4)
        with pytest.raises(ValueError, match="Unknown imperceptibility method"):
            compute_imperceptibility(cloud, cloud, method="unknown")

    def test_size_mismatch_handled(self):
        """Test that size mismatch is handled by using minimum size."""
        original = np.ones((100, 4))
        perturbed = np.ones((50, 4))
        perturbed[:, :3] = 2.0
        magnitude = compute_imperceptibility(original, perturbed, method="l2")
        # Should use only first 50 points
        expected = np.linalg.norm(np.ones((50, 3)))
        assert magnitude == pytest.approx(expected, abs=1e-6)


class TestMultiObjectiveFitness:
    """Tests for multi-objective fitness computation."""

    def test_perfect_slam_zero_perturbation(self):
        """Test fitness with perfect SLAM and no perturbation."""
        trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cloud = np.random.rand(100, 4)

        obj1, obj2 = compute_multi_objective_fitness(
            ground_truth_trajectory=trajectory,
            estimated_trajectory=trajectory,
            original_point_cloud=cloud,
            perturbed_point_cloud=cloud,
        )
        assert obj1 == pytest.approx(0.0, abs=1e-6)
        assert obj2 == pytest.approx(0.0, abs=1e-6)


class TestNormalization:
    """Tests for fitness normalization."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        fitness = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        normalized = normalize_fitness(fitness)

        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        assert normalized[0, 0] == 0.0
        assert normalized[-1, 0] == 1.0

    def test_normalization_with_reference_point(self):
        """Test normalization with reference point."""
        fitness = np.array([[2, 20], [4, 40]])
        reference = np.array([10, 100])
        normalized = normalize_fitness(fitness, reference_point=reference)

        assert normalized[0, 0] == pytest.approx(0.2, abs=1e-6)
        assert normalized[0, 1] == pytest.approx(0.2, abs=1e-6)
        assert normalized[1, 0] == pytest.approx(0.4, abs=1e-6)

    def test_normalization_constant_values(self):
        """Test normalization with constant values (range=0)."""
        fitness = np.array([[5, 10], [5, 20], [5, 30]])
        normalized = normalize_fitness(fitness)

        # First column should all be 0 (constant - 0 range)
        assert np.all(normalized[:, 0] == 0.0)
        # Second column should be normalized normally
        assert normalized[0, 1] == 0.0
        assert normalized[-1, 1] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
