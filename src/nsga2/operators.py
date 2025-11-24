"""
Genetic operators for NSGA-II.

Implements selection, crossover, and mutation operators.
"""

from typing import List, Tuple

import numpy as np

from .utils import crowded_comparison


def tournament_selection(
    population: np.ndarray, ranks: np.ndarray, distances: np.ndarray, tournament_size: int = 2
) -> int:
    """
    Tournament selection using crowded comparison operator.

    Args:
        population: Population array
        ranks: Rank of each individual
        distances: Crowding distance of each individual
        tournament_size: Number of individuals in tournament

    Returns:
        Index of selected individual
    """
    population_size = len(population)
    tournament_indices = np.random.choice(population_size, tournament_size, replace=False)

    best_idx = tournament_indices[0]
    best_rank = ranks[best_idx]
    best_distance = distances[best_idx]

    for idx in tournament_indices[1:]:
        comparison = crowded_comparison(ranks[idx], best_rank, distances[idx], best_distance)
        if comparison > 0:
            best_idx = idx
            best_rank = ranks[idx]
            best_distance = distances[idx]

    return best_idx


def simulated_binary_crossover(
    parent1: np.ndarray, parent2: np.ndarray, eta: float = 20.0, crossover_prob: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) operator.

    Args:
        parent1: First parent genome
        parent2: Second parent genome
        eta: Distribution index (larger values produce offspring closer to parents)
        crossover_prob: Probability of crossover

    Returns:
        Tuple of two offspring genomes
    """
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    if np.random.random() > crossover_prob:
        return offspring1, offspring2

    for i in range(len(parent1)):
        if np.random.random() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-9:
                # Calculate beta
                u = np.random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

                # Calculate offspring
                offspring1[i] = 0.5 * (
                    (parent1[i] + parent2[i]) - beta * abs(parent1[i] - parent2[i])
                )
                offspring2[i] = 0.5 * (
                    (parent1[i] + parent2[i]) + beta * abs(parent1[i] - parent2[i])
                )

                # Clip to bounds [-1, 1]
                offspring1[i] = np.clip(offspring1[i], -1, 1)
                offspring2[i] = np.clip(offspring2[i], -1, 1)

    return offspring1, offspring2


def polynomial_mutation(
    individual: np.ndarray, eta: float = 20.0, mutation_prob: float = None
) -> np.ndarray:
    """
    Polynomial mutation operator.

    Args:
        individual: Individual genome to mutate
        eta: Distribution index (larger values produce smaller mutations)
        mutation_prob: Probability of mutating each gene (default: 1/genome_size)

    Returns:
        Mutated individual
    """
    if mutation_prob is None:
        mutation_prob = 1.0 / len(individual)

    mutated = individual.copy()

    for i in range(len(mutated)):
        if np.random.random() <= mutation_prob:
            u = np.random.random()
            xi = mutated[i]

            # Lower and upper bounds
            lb = -1.0
            ub = 1.0

            # Calculate delta_l and delta_r
            delta_l = (xi - lb) / (ub - lb)
            delta_r = (ub - xi) / (ub - lb)

            # Calculate mutation power
            if u <= 0.5:
                val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_l) ** (eta + 1.0)
                delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
            else:
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta_r) ** (eta + 1.0)
                delta_q = 1.0 - val ** (1.0 / (eta + 1.0))

            # Apply mutation
            mutated[i] = xi + delta_q * (ub - lb)

            # Clip to bounds
            mutated[i] = np.clip(mutated[i], lb, ub)

    return mutated


def uniform_crossover(
    parent1: np.ndarray, parent2: np.ndarray, crossover_prob: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover operator.

    Each gene is randomly selected from either parent with equal probability.

    Args:
        parent1: First parent genome
        parent2: Second parent genome
        crossover_prob: Probability of crossover occurring

    Returns:
        Tuple of two offspring genomes
    """
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    if np.random.random() > crossover_prob:
        return offspring1, offspring2

    mask = np.random.random(len(parent1)) < 0.5

    offspring1[mask] = parent2[mask]
    offspring2[mask] = parent1[mask]

    return offspring1, offspring2


def gaussian_mutation(
    individual: np.ndarray, sigma: float = 0.1, mutation_prob: float = None
) -> np.ndarray:
    """
    Gaussian mutation operator.

    Args:
        individual: Individual genome to mutate
        sigma: Standard deviation of Gaussian noise
        mutation_prob: Probability of mutating each gene (default: 1/genome_size)

    Returns:
        Mutated individual
    """
    if mutation_prob is None:
        mutation_prob = 1.0 / len(individual)

    mutated = individual.copy()

    for i in range(len(mutated)):
        if np.random.random() <= mutation_prob:
            mutated[i] += np.random.normal(0, sigma)
            mutated[i] = np.clip(mutated[i], -1, 1)

    return mutated


def create_offspring_population(
    population: np.ndarray,
    ranks: np.ndarray,
    distances: np.ndarray,
    crossover_func=simulated_binary_crossover,
    mutation_func=polynomial_mutation,
    crossover_params: dict = None,
    mutation_params: dict = None,
) -> np.ndarray:
    """
    Create offspring population using selection, crossover, and mutation.

    Args:
        population: Current population
        ranks: Rank of each individual
        distances: Crowding distance of each individual
        crossover_func: Crossover function to use
        mutation_func: Mutation function to use
        crossover_params: Parameters for crossover function
        mutation_params: Parameters for mutation function

    Returns:
        Offspring population
    """
    if crossover_params is None:
        crossover_params = {}
    if mutation_params is None:
        mutation_params = {}

    population_size = len(population)
    offspring = []

    # Create offspring in pairs
    while len(offspring) < population_size:
        # Select two parents
        parent1_idx = tournament_selection(population, ranks, distances)
        parent2_idx = tournament_selection(population, ranks, distances)

        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        # Crossover
        child1, child2 = crossover_func(parent1, parent2, **crossover_params)

        # Mutation
        child1 = mutation_func(child1, **mutation_params)
        child2 = mutation_func(child2, **mutation_params)

        offspring.append(child1)
        if len(offspring) < population_size:
            offspring.append(child2)

    return np.array(offspring[:population_size])
