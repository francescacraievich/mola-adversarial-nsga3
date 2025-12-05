"""Tests for NSGA-III optimization (without ROS2 dependencies)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from perturbations.perturbation_generator import PerturbationGenerator  # noqa: E402

# Check if pymoo is available
try:
    import pymoo  # noqa: F401

    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False


@pytest.mark.skipif(not HAS_PYMOO, reason="pymoo not installed")
class TestNSGA3Integration:
    """Tests for NSGA-III integration with perturbation generator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = PerturbationGenerator(
            max_point_shift=0.05,
            noise_std=0.02,
            max_dropout_rate=0.15,
        )

    def test_pymoo_problem_setup(self):
        """Test pymoo problem can be created."""
        from pymoo.core.problem import Problem

        def mock_fitness(genome):
            # Mock fitness: ATE increases with perturbation, magnitude is L2 norm
            ate = np.sum(np.abs(genome)) * 0.1
            magnitude = np.linalg.norm(genome) * 0.5
            return (-ate, magnitude)

        class MockProblem(Problem):
            def __init__(self, fitness_func, genome_size):
                super().__init__(
                    n_var=genome_size,
                    n_obj=2,
                    xl=-1.0 * np.ones(genome_size),
                    xu=1.0 * np.ones(genome_size),
                )
                self.fitness_func = fitness_func

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                for genome in X:
                    obj1, obj2 = self.fitness_func(genome)
                    F.append([obj1, obj2])
                out["F"] = np.array(F)

        problem = MockProblem(mock_fitness, self.generator.get_genome_size())
        assert problem.n_var == 17  # Expanded genome with advanced attacks
        assert problem.n_obj == 2

    def test_nsga3_optimization_runs(self):
        """Test NSGA-III optimization runs without errors."""
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import Problem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize

        def mock_fitness(genome):
            ate = np.sum(np.abs(genome)) * 0.1
            magnitude = np.linalg.norm(genome) * 0.5
            return (-ate, magnitude)

        class MockProblem(Problem):
            def __init__(self, fitness_func, genome_size):
                super().__init__(
                    n_var=genome_size,
                    n_obj=2,
                    xl=-1.0 * np.ones(genome_size),
                    xu=1.0 * np.ones(genome_size),
                )
                self.fitness_func = fitness_func

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                for genome in X:
                    obj1, obj2 = self.fitness_func(genome)
                    F.append([obj1, obj2])
                out["F"] = np.array(F)

        problem = MockProblem(mock_fitness, self.generator.get_genome_size())

        from pymoo.util.ref_dirs import get_reference_directions

        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        algorithm = NSGA3(
            pop_size=10,
            ref_dirs=ref_dirs,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        result = minimize(
            problem,
            algorithm,
            ("n_gen", 5),
            seed=42,
            verbose=False,
        )

        assert result.F is not None
        assert result.X is not None
        assert len(result.F) > 0
        assert result.F.shape[1] == 2  # 2 objectives

    def test_pareto_front_properties(self):
        """Test Pareto front has expected properties."""
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import Problem
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions

        def mock_fitness(genome):
            # Conflicting objectives: can't minimize both
            ate = np.sum(genome[:6] ** 2)
            magnitude = np.sum((genome[6:] - 1) ** 2)
            return (ate, magnitude)

        class MockProblem(Problem):
            def __init__(self, genome_size):
                super().__init__(
                    n_var=genome_size,
                    n_obj=2,
                    xl=-1.0 * np.ones(genome_size),
                    xu=1.0 * np.ones(genome_size),
                )

            def _evaluate(self, X, out, *args, **kwargs):
                F = []
                for genome in X:
                    obj1, obj2 = mock_fitness(genome)
                    F.append([obj1, obj2])
                out["F"] = np.array(F)

        problem = MockProblem(12)
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        algorithm = NSGA3(
            pop_size=20,
            ref_dirs=ref_dirs,
            sampling=FloatRandomSampling(),
        )

        result = minimize(
            problem,
            algorithm,
            ("n_gen", 10),
            seed=42,
            verbose=False,
        )

        pareto_front = result.F
        pareto_set = result.X

        # Pareto front should have multiple solutions (trade-offs)
        assert len(pareto_front) >= 1

        # Each solution in Pareto set should match genome size
        assert pareto_set.shape[1] == 12

    def test_fitness_with_perturbation_generator(self):
        """Test fitness computation using actual perturbation generator."""
        np.random.seed(42)
        test_cloud = np.random.rand(100, 4) * 10
        test_cloud[:, 3] = np.random.rand(100) * 255

        def fitness_with_generator(genome):
            params = self.generator.encode_perturbation(genome)
            perturbed = self.generator.apply_perturbation(test_cloud, params, seed=42)

            # Mock ATE (in real case this comes from SLAM)
            mock_ate = params["noise_intensity"] * 10 + params["dropout_rate"] * 5

            # Real perturbation magnitude
            magnitude = self.generator.compute_perturbation_magnitude(test_cloud, perturbed, params)

            return (-mock_ate, magnitude)

        # Test a few genomes
        for _ in range(5):
            genome = self.generator.random_genome()
            obj1, obj2 = fitness_with_generator(genome)

            assert isinstance(obj1, float)
            assert isinstance(obj2, float)
            assert obj1 <= 0  # Negative ATE
            assert obj2 >= 0  # Positive magnitude


@pytest.mark.skipif(not HAS_PYMOO, reason="pymoo not installed")
class TestHistoryCallback:
    """Tests for optimization history tracking."""

    def test_callback_saves_history(self):
        """Test callback saves fitness history."""
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.callback import Callback
        from pymoo.core.problem import Problem
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions

        class SaveHistoryCallback(Callback):
            def __init__(self):
                super().__init__()
                self.all_fitness = []

            def notify(self, algorithm):
                for ind in algorithm.pop:
                    if ind.F is not None:
                        self.all_fitness.append(ind.F.copy())

        class SimpleProblem(Problem):
            def __init__(self):
                super().__init__(n_var=4, n_obj=2, xl=-1.0, xu=1.0)

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.column_stack([np.sum(X**2, axis=1), np.sum((X - 1) ** 2, axis=1)])

        callback = SaveHistoryCallback()
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        minimize(
            SimpleProblem(),
            NSGA3(pop_size=10, ref_dirs=ref_dirs, sampling=FloatRandomSampling()),
            ("n_gen", 3),
            seed=42,
            verbose=False,
            callback=callback,
        )

        # Should have saved fitness from each generation
        assert len(callback.all_fitness) > 0
        assert len(callback.all_fitness) == 30  # 10 pop * 3 gen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
