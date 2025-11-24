"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

Multi-objective evolutionary algorithm for generating Pareto-optimal solutions.
"""

from .nsga2 import NSGA2
from .operators import polynomial_mutation, simulated_binary_crossover, tournament_selection
from .utils import crowding_distance, fast_non_dominated_sort

__all__ = [
    "NSGA2",
    "tournament_selection",
    "simulated_binary_crossover",
    "polynomial_mutation",
    "fast_non_dominated_sort",
    "crowding_distance",
]
