"""
Fitness evaluation module for adversarial perturbations.

Provides integration with MOLA SLAM to evaluate the effectiveness
of perturbations on localization accuracy.
"""

from .fitness_evaluator import FitnessEvaluator
from .metrics import compute_imperceptibility, compute_localization_error
from .mola_interface import MOLAInterface

__all__ = [
    "FitnessEvaluator",
    "MOLAInterface",
    "compute_localization_error",
    "compute_imperceptibility",
]
