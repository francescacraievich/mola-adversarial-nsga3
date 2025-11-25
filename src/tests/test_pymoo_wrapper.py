"""
Tests for pymoo wrapper.

Tests the PerturbationOptimizer integration with pymoo.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.pymoo_wrapper import PerturbationOptimizer


# Simple test fitness functions
def simple_fitness(genome: np.ndarray) -> tuple:
    """Simple 2-objective function for testing."""
    x = genome[0]
    y = genome[1] if len(genome) > 1 else 0
    return (x**2, (x - 1) ** 2 + y**2)


def schaffer_fitness(genome: np.ndarray) -> tuple:
    """Schaffer N.1 benchmark function."""
    x = genome[0] * 10
    f1 = x**2
    f2 = (x - 2) ** 2
    return (f1, f2)


class TestPerturbationOptimizer:
    """Tests for PerturbationOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = PerturbationOptimizer(
            genome_size=8, fitness_function=simple_fitness, population_size=20, seed=42
        )

        assert optimizer.genome_size == 8
        assert optimizer.population_size == 20
        assert optimizer.n_objectives == 2

    def test_optimize(self):
        """Test optimization run."""
        optimizer = PerturbationOptimizer(
            genome_size=2, fitness_function=simple_fitness, population_size=20, seed=42
        )

        result = optimizer.optimize(n_generations=10, verbose=False)

        assert "population" in result
        assert "objectives" in result
        assert "pareto_front" in result
        assert "pareto_objectives" in result

        assert result["population"].shape[0] > 0
        assert result["population"].shape[1] == 2
        assert result["objectives"].shape[1] == 2

    def test_get_pareto_front(self):
        """Test getting Pareto front."""
        optimizer = PerturbationOptimizer(
            genome_size=2, fitness_function=simple_fitness, population_size=20, seed=42
        )

        optimizer.optimize(n_generations=10, verbose=False)
        genomes, objectives = optimizer.get_pareto_front()

        assert len(genomes) > 0
        assert len(objectives) > 0
        assert genomes.shape[0] == objectives.shape[0]

    def test_get_best_solution(self):
        """Test getting best solution."""
        optimizer = PerturbationOptimizer(
            genome_size=2, fitness_function=simple_fitness, population_size=20, seed=42
        )

        optimizer.optimize(n_generations=10, verbose=False)
        genome, objectives = optimizer.get_best_solution()

        assert len(genome) == 2
        assert len(objectives) == 2

    def test_get_best_solution_with_weights(self):
        """Test getting best solution with custom weights."""
        optimizer = PerturbationOptimizer(
            genome_size=2, fitness_function=simple_fitness, population_size=20, seed=42
        )

        optimizer.optimize(n_generations=10, verbose=False)
        genome, objectives = optimizer.get_best_solution(objective_weights=np.array([0.7, 0.3]))

        assert len(genome) == 2
        assert len(objectives) == 2

    def test_error_before_optimize(self):
        """Test error when accessing results before optimization."""
        optimizer = PerturbationOptimizer(
            genome_size=2, fitness_function=simple_fitness, population_size=20, seed=42
        )

        with pytest.raises(ValueError, match="Must run optimize"):
            optimizer.get_pareto_front()

        with pytest.raises(ValueError, match="Must run optimize"):
            optimizer.get_best_solution()


def test_integration_with_perturbations():
    """Integration test with perturbation-style problem."""
    # Simulate a perturbation optimization problem
    # Objective 1: Minimize perturbation magnitude
    # Objective 2: Maximize attack effectiveness (simulated)

    def perturbation_fitness(genome):
        # Perturbation magnitude (to minimize)
        magnitude = np.linalg.norm(genome)

        # Attack effectiveness (negative because we minimize, higher is better)
        # Simulate: effectiveness increases with certain perturbations
        effectiveness = -np.abs(np.sum(genome))

        return (magnitude, effectiveness)

    optimizer = PerturbationOptimizer(
        genome_size=8, fitness_function=perturbation_fitness, population_size=30, seed=42
    )

    optimizer.optimize(n_generations=20, verbose=False)

    # Check that we have diverse solutions
    pareto_genomes, pareto_objectives = optimizer.get_pareto_front()
    assert len(pareto_genomes) > 1

    # Check diversity in objectives
    assert np.std(pareto_objectives[:, 0]) > 0

    # Get best compromise solution
    best_genome, best_objectives = optimizer.get_best_solution()
    assert best_genome.shape == (8,)
    assert best_objectives.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
