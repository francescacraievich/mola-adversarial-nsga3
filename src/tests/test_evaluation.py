"""
Tests for evaluation module.

Tests fitness computation, metrics, and MOLA interface.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_imperceptibility,
    compute_localization_error,
    compute_multi_objective_fitness,
    normalize_fitness,
)
from src.perturbations.perturbation_generator import PerturbationGenerator


class TestLocalizationError:
    """Test localization error metrics."""

    def test_ate_identical_trajectories(self):
        """Test ATE with identical trajectories."""
        trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])

        error = compute_localization_error(trajectory, trajectory, method="ate")

        assert error == pytest.approx(0.0, abs=1e-6)

    def test_ate_offset_trajectories(self):
        """Test ATE with constant offset."""
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        estimated = gt + np.array([1, 1, 0])  # Constant offset

        error = compute_localization_error(gt, estimated, method="ate")

        # Should be sqrt(1^2 + 1^2) = sqrt(2)
        assert error == pytest.approx(np.sqrt(2), abs=1e-6)

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

        assert error == pytest.approx(2.0, abs=1e-6)  # Distance between [3,0,0] and [5,0,0]

    def test_empty_trajectory_returns_inf(self):
        """Test that empty trajectories return infinity."""
        gt = np.array([[0, 0, 0]])
        empty = np.array([])

        error = compute_localization_error(gt, empty, method="ate")

        assert error == float("inf")


class TestImperceptibility:
    """Test imperceptibility metrics."""

    def test_l2_identical_clouds(self):
        """Test L2 norm with identical clouds."""
        cloud = np.random.rand(100, 4)

        magnitude = compute_imperceptibility(cloud, cloud, method="l2")

        assert magnitude == pytest.approx(0.0, abs=1e-6)

    def test_l2_known_perturbation(self):
        """Test L2 norm with known perturbation."""
        original = np.zeros((10, 4))
        perturbed = np.ones((10, 4))  # All points moved by [1, 1, 1]

        magnitude = compute_imperceptibility(original, perturbed, method="l2")

        # L2 norm of 10 points each with [1,1,1] displacement
        expected = np.linalg.norm(np.ones((10, 3)))
        assert magnitude == pytest.approx(expected, abs=1e-6)

    def test_linf_norm(self):
        """Test L-infinity norm."""
        original = np.zeros((10, 4))
        perturbed = original.copy()
        perturbed[0, :3] = [5, 3, 2]  # Max difference is 5

        magnitude = compute_imperceptibility(original, perturbed, method="linf")

        assert magnitude == pytest.approx(5.0, abs=1e-6)

    def test_relative_norm(self):
        """Test relative L2 norm."""
        original = np.ones((10, 4)) * 10  # Large cloud
        perturbed = original + 1  # Add small perturbation

        magnitude = compute_imperceptibility(original, perturbed, method="relative")

        # Should be small relative to original size
        assert magnitude < 0.1


class TestMultiObjectiveFitness:
    """Test multi-objective fitness computation."""

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

        # obj1 = -error (should be ~0, negative)
        # obj2 = imperceptibility (should be 0)
        assert obj1 == pytest.approx(0.0, abs=1e-6)
        assert obj2 == pytest.approx(0.0, abs=1e-6)

    def test_degraded_slam_large_perturbation(self):
        """Test fitness with degraded SLAM and large perturbation."""
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        estimated = gt + 5  # Large error
        original = np.zeros((100, 4))
        perturbed = original + 2  # Large perturbation

        obj1, obj2 = compute_multi_objective_fitness(
            ground_truth_trajectory=gt,
            estimated_trajectory=estimated,
            original_point_cloud=original,
            perturbed_point_cloud=perturbed,
        )

        # obj1 should be large negative (high error)
        # obj2 should be large positive (high perturbation)
        assert obj1 < -5  # Large negative error
        assert obj2 > 10  # Large perturbation


class TestNormalization:
    """Test fitness normalization."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        fitness = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

        normalized = normalize_fitness(fitness)

        # Should be in [0, 1] range
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        assert normalized[0, 0] == 0.0  # Min value
        assert normalized[-1, 0] == 1.0  # Max value

    def test_reference_point_normalization(self):
        """Test normalization with reference point."""
        fitness = np.array([[10, 20], [20, 40]])
        reference = np.array([10, 10])

        normalized = normalize_fitness(fitness, reference_point=reference)

        expected = np.array([[1.0, 2.0], [2.0, 4.0]])
        np.testing.assert_array_almost_equal(normalized, expected)


class TestOfflineEvaluator:
    """Test offline fitness evaluator."""

    def test_offline_evaluator_basic(self):
        """Test basic offline evaluation."""
        from src.evaluation.fitness_evaluator import OfflineFitnessEvaluator

        generator = PerturbationGenerator()

        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        baseline = gt.copy()
        cloud = np.random.rand(100, 4)

        evaluator = OfflineFitnessEvaluator(
            perturbation_generator=generator,
            ground_truth_trajectory=gt,
            baseline_trajectory=baseline,
            reference_point_cloud=cloud,
        )

        # Test with zero perturbation
        genome = np.zeros(8)
        obj1, obj2 = evaluator.evaluate(genome)

        # Should have low error and imperceptibility
        assert isinstance(obj1, float)
        assert isinstance(obj2, float)

    def test_offline_evaluator_with_perturbation(self):
        """Test offline evaluation with non-zero perturbation."""
        from src.evaluation.fitness_evaluator import OfflineFitnessEvaluator

        generator = PerturbationGenerator()

        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        baseline = gt.copy()
        cloud = np.random.rand(100, 4)

        evaluator = OfflineFitnessEvaluator(
            perturbation_generator=generator,
            ground_truth_trajectory=gt,
            baseline_trajectory=baseline,
            reference_point_cloud=cloud,
        )

        # Test with large perturbation
        genome = np.ones(8) * 0.5
        obj1, obj2 = evaluator.evaluate(genome)

        # Should have higher imperceptibility than zero perturbation
        assert obj2 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
