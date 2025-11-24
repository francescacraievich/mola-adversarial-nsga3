"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

Main algorithm implementation for multi-objective optimization.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .operators import create_offspring_population, polynomial_mutation, simulated_binary_crossover
from .utils import assign_ranks, crowding_distance_all, fast_non_dominated_sort


class NSGA2:
    """
    NSGA-II multi-objective evolutionary algorithm.

    Optimizes multiple conflicting objectives simultaneously to find
    a Pareto-optimal set of solutions.
    """

    def __init__(
        self,
        population_size: int,
        genome_size: int,
        fitness_function: Callable,
        crossover_eta: float = 20.0,
        mutation_eta: float = 20.0,
        crossover_prob: float = 0.9,
        mutation_prob: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize NSGA-II algorithm.

        Args:
            population_size: Size of population (must be even)
            genome_size: Size of individual genome
            fitness_function: Function that takes a genome and returns (obj1, obj2, ...)
            crossover_eta: Distribution index for SBX crossover
            mutation_eta: Distribution index for polynomial mutation
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation per gene (default: 1/genome_size)
            seed: Random seed for reproducibility
        """
        if population_size % 2 != 0:
            raise ValueError("Population size must be even")

        self.population_size = population_size
        self.genome_size = genome_size
        self.fitness_function = fitness_function
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob if mutation_prob is not None else 1.0 / genome_size

        if seed is not None:
            np.random.seed(seed)

        # Initialize population
        self.population = self._initialize_population()
        self.objectives = None
        self.ranks = None
        self.distances = None

        # History tracking
        self.history = {"best_fronts": [], "hypervolume": [], "population_diversity": []}

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with random genomes.

        Returns:
            Population array of shape (population_size, genome_size)
        """
        return np.random.uniform(-1, 1, (self.population_size, self.genome_size))

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for entire population.

        Args:
            population: Population to evaluate

        Returns:
            Objectives array of shape (population_size, num_objectives)
        """
        objectives = []
        for individual in population:
            obj = self.fitness_function(individual)
            if not isinstance(obj, (list, tuple, np.ndarray)):
                obj = [obj]
            objectives.append(obj)
        return np.array(objectives)

    def evolve(self, generations: int, verbose: bool = True) -> Dict:
        """
        Run NSGA-II evolution for specified number of generations.

        Args:
            generations: Number of generations to evolve
            verbose: Print progress information

        Returns:
            Dictionary with final results and history
        """
        # Initial evaluation
        if self.objectives is None:
            self.objectives = self.evaluate_population(self.population)

        for gen in range(generations):
            # Create offspring population
            fronts = fast_non_dominated_sort(self.objectives)
            self.ranks = assign_ranks(fronts, self.population_size)
            self.distances = crowding_distance_all(self.objectives, fronts)

            # Create offspring
            offspring = create_offspring_population(
                self.population,
                self.ranks,
                self.distances,
                crossover_func=simulated_binary_crossover,
                mutation_func=polynomial_mutation,
                crossover_params={"eta": self.crossover_eta, "crossover_prob": self.crossover_prob},
                mutation_params={"eta": self.mutation_eta, "mutation_prob": self.mutation_prob},
            )

            # Evaluate offspring
            offspring_objectives = self.evaluate_population(offspring)

            # Combine parent and offspring populations
            combined_population = np.vstack([self.population, offspring])
            combined_objectives = np.vstack([self.objectives, offspring_objectives])

            # Select next generation
            self.population, self.objectives = self._environmental_selection(
                combined_population, combined_objectives
            )

            # Track history
            fronts = fast_non_dominated_sort(self.objectives)
            self.history["best_fronts"].append(len(fronts[0]))

            if verbose and (gen + 1) % max(1, generations // 10) == 0:
                print(
                    f"Generation {gen + 1}/{generations} - " f"Pareto front size: {len(fronts[0])}"
                )

        # Final evaluation
        fronts = fast_non_dominated_sort(self.objectives)
        self.ranks = assign_ranks(fronts, self.population_size)
        self.distances = crowding_distance_all(self.objectives, fronts)

        return {
            "population": self.population,
            "objectives": self.objectives,
            "fronts": fronts,
            "ranks": self.ranks,
            "distances": self.distances,
            "history": self.history,
        }

    def _environmental_selection(
        self, combined_population: np.ndarray, combined_objectives: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Environmental selection to choose next generation.

        Uses non-dominated sorting and crowding distance to select
        the best population_size individuals.

        Args:
            combined_population: Combined parent and offspring population
            combined_objectives: Combined objectives

        Returns:
            Tuple of (selected_population, selected_objectives)
        """
        fronts = fast_non_dominated_sort(combined_objectives)
        distances = crowding_distance_all(combined_objectives, fronts)

        selected_indices = []
        current_front_idx = 0

        # Add fronts until population is filled
        while len(selected_indices) < self.population_size and current_front_idx < len(fronts):
            current_front = fronts[current_front_idx]

            if len(selected_indices) + len(current_front) <= self.population_size:
                # Add entire front
                selected_indices.extend(current_front)
            else:
                # Add partial front based on crowding distance
                remaining = self.population_size - len(selected_indices)
                front_distances = distances[current_front]
                sorted_front_indices = np.argsort(front_distances)[::-1]  # Descending order
                selected_from_front = [current_front[i] for i in sorted_front_indices[:remaining]]
                selected_indices.extend(selected_from_front)

            current_front_idx += 1

        selected_indices = np.array(selected_indices[: self.population_size])

        return combined_population[selected_indices], combined_objectives[selected_indices]

    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current Pareto front (best non-dominated solutions).

        Returns:
            Tuple of (pareto_genomes, pareto_objectives)
        """
        if self.objectives is None:
            self.objectives = self.evaluate_population(self.population)

        fronts = fast_non_dominated_sort(self.objectives)
        pareto_indices = fronts[0]

        return self.population[pareto_indices], self.objectives[pareto_indices]

    def get_best_solution(
        self, objective_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best solution based on weighted objectives.

        Args:
            objective_weights: Weights for each objective (default: equal weights)

        Returns:
            Tuple of (best_genome, best_objectives)
        """
        pareto_genomes, pareto_objectives = self.get_pareto_front()

        if objective_weights is None:
            objective_weights = np.ones(pareto_objectives.shape[1]) / pareto_objectives.shape[1]

        # Normalize objectives to [0, 1]
        obj_min = pareto_objectives.min(axis=0)
        obj_max = pareto_objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1  # Avoid division by zero

        normalized_objectives = (pareto_objectives - obj_min) / obj_range

        # Compute weighted sum
        weighted_sum = (normalized_objectives * objective_weights).sum(axis=1)
        best_idx = np.argmin(weighted_sum)

        return pareto_genomes[best_idx], pareto_objectives[best_idx]

    def save_population(self, filepath: str):
        """Save current population to file."""
        np.savez(
            filepath,
            population=self.population,
            objectives=self.objectives,
            ranks=self.ranks,
            distances=self.distances,
        )

    def load_population(self, filepath: str):
        """Load population from file."""
        data = np.load(filepath)
        self.population = data["population"]
        self.objectives = data["objectives"]
        self.ranks = data["ranks"]
        self.distances = data["distances"]
