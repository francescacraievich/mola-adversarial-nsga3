"""
Utility functions for NSGA-II algorithm.

Implements fast non-dominated sorting and crowding distance calculation.
"""

from typing import List

import numpy as np


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Fast non-dominated sorting algorithm.

    Args:
        objectives: Array of shape (N, M) where N is population size and M is number of objectives
                   Assumes minimization for all objectives

    Returns:
        List of fronts, where each front is a list of individual indices
    """
    population_size = len(objectives)
    domination_count = np.zeros(
        population_size, dtype=int
    )  # Number of solutions that dominate this solution
    dominated_solutions = [
        [] for _ in range(population_size)
    ]  # Solutions dominated by this solution
    fronts = [[]]

    # For each individual
    for i in range(population_size):
        for j in range(i + 1, population_size):
            # Check dominance between i and j
            dominance = compare_dominance(objectives[i], objectives[j])

            if dominance == 1:  # i dominates j
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominance == -1:  # j dominates i
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    # First front: individuals with domination_count = 0
    for i in range(population_size):
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Generate subsequent fronts
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)

        current_front += 1
        if len(next_front) > 0:
            fronts.append(next_front)

    return fronts


def compare_dominance(obj1: np.ndarray, obj2: np.ndarray) -> int:
    """
    Compare dominance between two solutions.

    Args:
        obj1: Objective values for solution 1
        obj2: Objective values for solution 2

    Returns:
        1 if obj1 dominates obj2
        -1 if obj2 dominates obj1
        0 if neither dominates (non-dominated)
    """
    better_in_any = False
    worse_in_any = False

    for i in range(len(obj1)):
        if obj1[i] < obj2[i]:
            better_in_any = True
        elif obj1[i] > obj2[i]:
            worse_in_any = True

    if better_in_any and not worse_in_any:
        return 1  # obj1 dominates obj2
    elif worse_in_any and not better_in_any:
        return -1  # obj2 dominates obj1
    else:
        return 0  # Non-dominated


def crowding_distance(objectives: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """
    Calculate crowding distance for individuals in a front.

    Args:
        objectives: Array of shape (N, M) where N is population size and M is number of objectives
        front_indices: Indices of individuals in the current front

    Returns:
        Array of crowding distances for each individual in the front
    """
    front_size = len(front_indices)
    num_objectives = objectives.shape[1]

    if front_size <= 2:
        # Boundary solutions get infinite distance
        return np.full(front_size, np.inf)

    distances = np.zeros(front_size)

    # Calculate crowding distance for each objective
    for m in range(num_objectives):
        # Sort individuals by m-th objective
        sorted_indices = np.argsort(objectives[front_indices, m])

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Range of objective m
        obj_range = (
            objectives[front_indices[sorted_indices[-1]], m]
            - objectives[front_indices[sorted_indices[0]], m]
        )

        if obj_range == 0:
            continue

        # Calculate crowding distance for intermediate points
        for i in range(1, front_size - 1):
            distances[sorted_indices[i]] += (
                objectives[front_indices[sorted_indices[i + 1]], m]
                - objectives[front_indices[sorted_indices[i - 1]], m]
            ) / obj_range

    return distances


def crowding_distance_all(objectives: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
    """
    Calculate crowding distance for all individuals across all fronts.

    Args:
        objectives: Array of shape (N, M)
        fronts: List of fronts from fast_non_dominated_sort

    Returns:
        Array of crowding distances for all individuals
    """
    population_size = len(objectives)
    distances = np.zeros(population_size)

    for front in fronts:
        if len(front) > 0:
            front_distances = crowding_distance(objectives, front)
            for idx, individual in enumerate(front):
                distances[individual] = front_distances[idx]

    return distances


def crowded_comparison(rank1: int, rank2: int, distance1: float, distance2: float) -> int:
    """
    Crowded comparison operator for tournament selection.

    An individual is preferred if:
    1. It has better (lower) rank, or
    2. Same rank but larger crowding distance

    Args:
        rank1: Rank of individual 1 (lower is better)
        rank2: Rank of individual 2
        distance1: Crowding distance of individual 1
        distance2: Crowding distance of individual 2

    Returns:
        1 if individual 1 is preferred
        -1 if individual 2 is preferred
        0 if equal
    """
    if rank1 < rank2:
        return 1
    elif rank1 > rank2:
        return -1
    else:
        # Same rank, compare crowding distance
        if distance1 > distance2:
            return 1
        elif distance1 < distance2:
            return -1
        else:
            return 0


def assign_ranks(fronts: List[List[int]], population_size: int) -> np.ndarray:
    """
    Assign rank to each individual based on fronts.

    Args:
        fronts: List of fronts from fast_non_dominated_sort
        population_size: Total population size

    Returns:
        Array of ranks for each individual
    """
    ranks = np.zeros(population_size, dtype=int)

    for rank, front in enumerate(fronts):
        for individual in front:
            ranks[individual] = rank

    return ranks
