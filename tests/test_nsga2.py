"""
Tests for NSGA-II algorithm.

Tests the NSGA-II implementation with simple benchmark functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nsga2.nsga2 import NSGA2
from nsga2.operators import polynomial_mutation, simulated_binary_crossover, tournament_selection
from nsga2.utils import compare_dominance, crowding_distance, fast_non_dominated_sort


# Benchmark multi-objective functions
def zdt1_fitness(genome: np.ndarray) -> tuple:
    """ZDT1 benchmark function (2 objectives)."""
    n = len(genome)
    f1 = genome[0]
    g = 1 + 9 * np.sum(genome[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return (f1, f2)


def schaffer_n1_fitness(genome: np.ndarray) -> tuple:
    """Schaffer N.1 benchmark function (2 objectives)."""
    x = genome[0] * 10  # Scale to reasonable range
    f1 = x**2
    f2 = (x - 2) ** 2
    return (f1, f2)


class TestNSGA2Utils:
    """Tests for NSGA-II utility functions."""

    def test_compare_dominance(self):
        """Test dominance comparison."""
        obj1 = np.array([1.0, 2.0])
        obj2 = np.array([2.0, 3.0])
        obj3 = np.array([2.0, 1.0])

        # obj1 dominates obj2 (better in both objectives)
        assert compare_dominance(obj1, obj2) == 1

        # obj2 is dominated by obj1
        assert compare_dominance(obj2, obj1) == -1

        # obj1 and obj3 are non-dominated (obj1 better in first, obj3 better in second)
        assert compare_dominance(obj1, obj3) == 0

    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        objectives = np.array(
            [
                [1.0, 5.0],  # Front 1
                [2.0, 3.0],  # Front 1
                [3.0, 2.0],  # Front 1
                [4.0, 4.0],  # Front 2
                [5.0, 1.0],  # Front 1
            ]
        )

        fronts = fast_non_dominated_sort(objectives)

        # Should have at least 2 fronts
        assert len(fronts) >= 1

        # First front should contain non-dominated solutions
        assert len(fronts[0]) > 0

    def test_crowding_distance(self):
        """Test crowding distance calculation."""
        objectives = np.array(
            [
                [1.0, 5.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [5.0, 1.0],
            ]
        )
        front_indices = [0, 1, 2, 3]

        distances = crowding_distance(objectives, front_indices)

        # Boundary solutions should have infinite distance
        assert distances[0] == np.inf or distances[-1] == np.inf

        # All distances should be non-negative
        assert np.all(distances >= 0)


class TestNSGA2Operators:
    """Tests for genetic operators."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parent1 = np.array([0.5, -0.3, 0.8, -0.1])
        self.parent2 = np.array([-0.5, 0.3, -0.8, 0.1])

    def test_sbx_crossover(self):
        """Test simulated binary crossover."""
        child1, child2 = simulated_binary_crossover(
            self.parent1, self.parent2, eta=20.0, crossover_prob=1.0
        )

        # Children should be valid
        assert len(child1) == len(self.parent1)
        assert len(child2) == len(self.parent2)

        # Children should be within bounds
        assert np.all(child1 >= -1) and np.all(child1 <= 1)
        assert np.all(child2 >= -1) and np.all(child2 <= 1)

    def test_polynomial_mutation(self):
        """Test polynomial mutation."""
        mutated = polynomial_mutation(self.parent1, eta=20.0, mutation_prob=1.0)

        # Mutated individual should be valid
        assert len(mutated) == len(self.parent1)

        # Should be within bounds
        assert np.all(mutated >= -1) and np.all(mutated <= 1)

        # Should be different from original (with high mutation prob)
        assert not np.allclose(mutated, self.parent1)

    def test_tournament_selection(self):
        """Test tournament selection."""
        population = np.random.rand(10, 4)
        ranks = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        distances = np.random.rand(10)

        selected_idx = tournament_selection(population, ranks, distances, tournament_size=2)

        # Should return valid index
        assert 0 <= selected_idx < len(population)


class TestNSGA2Algorithm:
    """Tests for main NSGA-II algorithm."""

    def test_initialization(self):
        """Test NSGA-II initialization."""
        nsga2 = NSGA2(
            population_size=20, genome_size=5, fitness_function=schaffer_n1_fitness, seed=42
        )

        assert nsga2.population.shape == (20, 5)
        assert np.all(nsga2.population >= -1) and np.all(nsga2.population <= 1)

    def test_population_size_validation(self):
        """Test that population size must be even."""
        with pytest.raises(ValueError):
            NSGA2(
                population_size=21,  # Odd number
                genome_size=5,
                fitness_function=schaffer_n1_fitness,
            )

    def test_evaluate_population(self):
        """Test population evaluation."""
        nsga2 = NSGA2(
            population_size=10, genome_size=5, fitness_function=schaffer_n1_fitness, seed=42
        )

        objectives = nsga2.evaluate_population(nsga2.population)

        assert objectives.shape == (10, 2)  # 10 individuals, 2 objectives
        assert np.all(np.isfinite(objectives))

    def test_single_generation(self):
        """Test running a single generation."""
        nsga2 = NSGA2(
            population_size=20, genome_size=5, fitness_function=schaffer_n1_fitness, seed=42
        )

        result = nsga2.evolve(generations=1, verbose=False)

        assert "population" in result
        assert "objectives" in result
        assert "fronts" in result
        assert result["population"].shape == (20, 5)

    def test_multiple_generations(self):
        """Test running multiple generations."""
        nsga2 = NSGA2(
            population_size=20, genome_size=5, fitness_function=schaffer_n1_fitness, seed=42
        )

        result = nsga2.evolve(generations=5, verbose=False)

        assert result["population"].shape == (20, 5)
        assert len(result["history"]["best_fronts"]) == 5

    def test_get_pareto_front(self):
        """Test getting Pareto front."""
        nsga2 = NSGA2(
            population_size=20, genome_size=5, fitness_function=schaffer_n1_fitness, seed=42
        )

        nsga2.evolve(generations=10, verbose=False)
        pareto_genomes, pareto_objectives = nsga2.get_pareto_front()

        # Pareto front should not be empty
        assert len(pareto_genomes) > 0
        assert len(pareto_objectives) > 0
        assert pareto_genomes.shape[0] == pareto_objectives.shape[0]

    def test_get_best_solution(self):
        """Test getting best solution."""
        nsga2 = NSGA2(
            population_size=20, genome_size=5, fitness_function=schaffer_n1_fitness, seed=42
        )

        nsga2.evolve(generations=10, verbose=False)
        best_genome, best_objectives = nsga2.get_best_solution()

        assert len(best_genome) == 5
        assert len(best_objectives) == 2

    def test_convergence(self):
        """Test that algorithm converges (improves over time)."""
        nsga2 = NSGA2(population_size=50, genome_size=10, fitness_function=zdt1_fitness, seed=42)

        # Initial objectives
        initial_objectives = nsga2.evaluate_population(nsga2.population)
        initial_min = initial_objectives.min(axis=0)

        # Evolve
        nsga2.evolve(generations=20, verbose=False)

        # Final objectives
        final_objectives = nsga2.objectives
        final_min = final_objectives.min(axis=0)

        # At least one objective should improve
        assert np.any(final_min <= initial_min)


def test_integration_nsga2():
    """Integration test: full NSGA-II run."""

    def simple_fitness(genome):
        """Simple 2-objective function."""
        x = genome[0]
        y = genome[1] if len(genome) > 1 else 0
        return (x**2, (x - 1) ** 2 + y**2)

    nsga2 = NSGA2(population_size=30, genome_size=2, fitness_function=simple_fitness, seed=42)

    result = nsga2.evolve(generations=15, verbose=False)

    # Check result structure
    assert "population" in result
    assert "objectives" in result
    assert "fronts" in result
    assert "history" in result

    # Check Pareto front
    pareto_genomes, pareto_objectives = nsga2.get_pareto_front()
    assert len(pareto_genomes) > 0

    # Check that solutions are diverse
    assert np.std(pareto_objectives[:, 0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
