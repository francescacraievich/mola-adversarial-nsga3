"""
Pymoo-based optimizer for adversarial perturbations.

Integrates pymoo's NSGA-II with the PerturbationGenerator for
multi-objective optimization of point cloud perturbations.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class PerturbationProblem(Problem):
    """
    Pymoo problem definition for perturbation optimization.

    Wraps a fitness function that evaluates perturbations.
    Includes constraint handling for physical realism.
    """

    def __init__(
        self,
        n_var: int,
        fitness_function: Callable,
        n_obj: int = 2,
        max_translation: float = 0.5,
        max_rotation: float = 0.1,
    ):
        """
        Initialize problem.

        Args:
            n_var: Number of variables (genome size)
            fitness_function: Function that takes a genome and returns tuple of objectives
            n_obj: Number of objectives (default: 2)
            max_translation: Maximum translation constraint (m)
            max_rotation: Maximum rotation constraint (rad)
        """
        # n_constr = 1: single constraint for total perturbation magnitude
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=1, xl=-1.0, xu=1.0)
        self.fitness_function = fitness_function
        self.max_translation = max_translation
        self.max_rotation = max_rotation

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate population with constraint checking.

        Args:
            x: Population array of shape (pop_size, n_var)
            out: Output dictionary
        """
        objectives = []
        constraints = []

        for genome in x:
            # Evaluate fitness
            obj = self.fitness_function(genome)
            objectives.append(obj)

            # Constraint: perturbation magnitude must be within limits
            # Genome format: [tx, ty, tz, rx, ry, rz, intensity, dropout]
            translation = genome[:3] * self.max_translation
            rotation = genome[3:6] * self.max_rotation

            # Total perturbation magnitude
            trans_mag = np.linalg.norm(translation)
            rot_mag = np.linalg.norm(rotation)

            # Constraint: g(x) <= 0 (violated if positive)
            # We want: trans_mag <= max_translation AND rot_mag <= max_rotation
            # Combine into single constraint: normalized magnitude <= 1.0
            normalized_mag = (trans_mag / self.max_translation) ** 2 + (
                rot_mag / self.max_rotation
            ) ** 2
            constraint = normalized_mag - 1.0  # <= 0 if within bounds

            constraints.append(constraint)

        out["F"] = np.array(objectives)
        out["G"] = np.array(constraints).reshape(-1, 1)


class PerturbationOptimizer:
    """
    Multi-objective optimizer for adversarial perturbations using pymoo NSGA-II.

    Provides a simple interface for optimizing perturbations with multiple objectives
    (e.g., attack effectiveness vs imperceptibility).
    """

    def __init__(
        self,
        genome_size: int,
        fitness_function: Callable,
        n_objectives: int = 2,
        population_size: int = 100,
        max_translation: float = 0.5,
        max_rotation: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            genome_size: Size of genome (number of perturbation parameters)
            fitness_function: Function that evaluates a genome and returns objectives
            n_objectives: Number of objectives to optimize (default: 2)
            population_size: Population size for NSGA-II (default: 100)
            max_translation: Maximum translation constraint (m, default: 0.5)
            max_rotation: Maximum rotation constraint (rad, default: 0.1 ≈ 5.7°)
            seed: Random seed for reproducibility
        """
        self.genome_size = genome_size
        self.fitness_function = fitness_function
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.seed = seed

        # Create problem with constraints
        self.problem = PerturbationProblem(
            n_var=genome_size,
            fitness_function=fitness_function,
            n_obj=n_objectives,
            max_translation=max_translation,
            max_rotation=max_rotation,
        )

        # Create algorithm
        self.algorithm = NSGA2(pop_size=population_size)

        # Results
        self.result = None

    def optimize(self, n_generations: int = 200, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Run optimization.

        Args:
            n_generations: Number of generations to run
            verbose: Whether to print progress

        Returns:
            Dictionary with:
                - population: Final population genomes
                - objectives: Final population objectives
                - pareto_front: Pareto optimal solutions
                - pareto_objectives: Objectives of Pareto optimal solutions
        """
        termination = get_termination("n_gen", n_generations)

        self.result = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=self.seed,
            verbose=verbose,
            save_history=False,
        )

        return {
            "population": self.result.X,
            "objectives": self.result.F,
            "pareto_front": self.result.X,
            "pareto_objectives": self.result.F,
        }

    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Pareto optimal solutions.

        Returns:
            Tuple of (genomes, objectives) for Pareto front
        """
        if self.result is None:
            raise ValueError("Must run optimize() first")

        return self.result.X, self.result.F

    def get_best_solution(
        self, objective_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get best solution based on weighted objectives.

        Args:
            objective_weights: Weights for each objective (default: equal weights)

        Returns:
            Tuple of (best_genome, best_objectives)
        """
        if self.result is None:
            raise ValueError("Must run optimize() first")

        if objective_weights is None:
            objective_weights = np.ones(self.n_objectives) / self.n_objectives

        # Compute weighted sum
        weighted_scores = self.result.F @ objective_weights
        best_idx = np.argmin(weighted_scores)

        return self.result.X[best_idx], self.result.F[best_idx]
