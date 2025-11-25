"""
Fitness evaluator for adversarial perturbations.

Coordinates the evaluation pipeline:
1. Apply perturbation to point cloud
2. Publish to MOLA via ROS 2
3. Collect SLAM trajectory
4. Compute fitness metrics
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rclpy

from ..perturbations.perturbation_generator import PerturbationGenerator
from .metrics import compute_multi_objective_fitness
from .mola_interface import MOLAInterface, load_ground_truth_trajectory


class FitnessEvaluator:
    """
    Evaluates fitness of adversarial perturbations by running MOLA SLAM.

    Integrates perturbation generation, MOLA execution, and fitness computation.
    """

    def __init__(
        self,
        perturbation_generator: PerturbationGenerator,
        ground_truth_trajectory_path: str,
        point_cloud_sequence: List[np.ndarray],
        lidar_topic: str = "/carter/lidar_with_intensity",
        odom_topic: str = "/lidar_odometry/odom",
        error_method: str = "ate",
        imperceptibility_method: str = "l2",
    ):
        """
        Initialize fitness evaluator.

        Args:
            perturbation_generator: PerturbationGenerator instance
            ground_truth_trajectory_path: Path to ground truth trajectory file
            point_cloud_sequence: List of point clouds to process
            lidar_topic: ROS 2 topic to publish point clouds
            odom_topic: ROS 2 topic to receive SLAM odometry
            error_method: Localization error metric (ate, rpe, final)
            imperceptibility_method: Imperceptibility metric (l2, linf, relative)
        """
        self.perturbation_generator = perturbation_generator
        self.point_cloud_sequence = point_cloud_sequence
        self.error_method = error_method
        self.imperceptibility_method = imperceptibility_method

        # Load ground truth trajectory
        self.ground_truth_trajectory, self.ground_truth_timestamps = load_ground_truth_trajectory(
            ground_truth_trajectory_path
        )

        # Initialize ROS 2 if not already initialized
        if not rclpy.ok():
            rclpy.init()

        # Create MOLA interface
        self.mola_interface = MOLAInterface(lidar_topic=lidar_topic, odom_topic=odom_topic)

        # Statistics
        self.evaluation_count = 0

    def evaluate(self, genome: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate fitness of a genome.

        Args:
            genome: Perturbation genome (normalized [-1, 1])

        Returns:
            Tuple of (objective1, objective2) for minimization:
            - objective1: Negative localization error (minimize = maximize attack)
            - objective2: Imperceptibility (minimize = more imperceptible)
        """
        # Decode genome to perturbation parameters
        params = self.perturbation_generator.encode_perturbation(genome)

        # Apply perturbation to point cloud sequence
        perturbed_sequence = []
        for point_cloud in self.point_cloud_sequence:
            perturbed = self.perturbation_generator.apply_perturbation(point_cloud, params)
            perturbed_sequence.append(perturbed)

        # Run MOLA with perturbed point clouds
        estimated_trajectory = self._run_mola_with_sequence(perturbed_sequence)

        # Compute fitness
        fitness = compute_multi_objective_fitness(
            ground_truth_trajectory=self.ground_truth_trajectory,
            estimated_trajectory=estimated_trajectory,
            original_point_cloud=self.point_cloud_sequence[0],  # Use first frame
            perturbed_point_cloud=perturbed_sequence[0],
            error_method=self.error_method,
            imperceptibility_method=self.imperceptibility_method,
        )

        self.evaluation_count += 1

        return fitness

    def _run_mola_with_sequence(self, point_cloud_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Run MOLA SLAM with a sequence of point clouds.

        Args:
            point_cloud_sequence: List of point clouds to publish

        Returns:
            Estimated trajectory (N, 3)
        """
        # Reset trajectory collection
        self.mola_interface.reset_trajectory()

        # Publish point cloud sequence
        rate = 10  # Hz (adjust based on your data rate)
        dt = 1.0 / rate

        for i, point_cloud in enumerate(point_cloud_sequence):
            # Publish point cloud
            timestamp = i * dt
            self.mola_interface.publish_point_cloud(point_cloud, timestamp=timestamp)

            # Spin to process callbacks
            rclpy.spin_once(self.mola_interface, timeout_sec=dt)

        # Wait for MOLA to process
        min_points = max(len(point_cloud_sequence) // 2, 5)
        success = self.mola_interface.wait_for_trajectory(min_points=min_points, timeout_sec=30.0)

        if not success:
            # If timeout, return empty trajectory (will result in inf error)
            return np.array([])

        # Get trajectory
        trajectory, _ = self.mola_interface.get_trajectory()

        return trajectory

    def evaluate_batch(self, genomes: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of genomes.

        Args:
            genomes: Array of genomes (pop_size, genome_size)

        Returns:
            Array of fitness values (pop_size, 2)
        """
        fitness_values = []

        for genome in genomes:
            fitness = self.evaluate(genome)
            fitness_values.append(fitness)

        return np.array(fitness_values)

    def get_fitness_function(self) -> Callable[[np.ndarray], Tuple[float, float]]:
        """
        Get fitness function for use with optimizer.

        Returns:
            Callable that takes genome and returns fitness tuple
        """
        return self.evaluate

    def shutdown(self):
        """Shutdown ROS 2 interface."""
        self.mola_interface.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class OfflineFitnessEvaluator:
    """
    Offline fitness evaluator that uses pre-computed SLAM results.

    Useful for testing and development without running MOLA in real-time.
    """

    def __init__(
        self,
        perturbation_generator: PerturbationGenerator,
        ground_truth_trajectory: np.ndarray,
        baseline_trajectory: np.ndarray,
        reference_point_cloud: np.ndarray,
        error_method: str = "ate",
        imperceptibility_method: str = "l2",
    ):
        """
        Initialize offline evaluator.

        Args:
            perturbation_generator: PerturbationGenerator instance
            ground_truth_trajectory: Ground truth trajectory (N, 3)
            baseline_trajectory: SLAM trajectory without perturbation (N, 3)
            reference_point_cloud: Reference point cloud for imperceptibility
            error_method: Localization error metric
            imperceptibility_method: Imperceptibility metric
        """
        self.perturbation_generator = perturbation_generator
        self.ground_truth_trajectory = ground_truth_trajectory
        self.baseline_trajectory = baseline_trajectory
        self.reference_point_cloud = reference_point_cloud
        self.error_method = error_method
        self.imperceptibility_method = imperceptibility_method

        # For simulation: assume perturbations degrade trajectory proportionally
        self.degradation_factor = 1.0

    def evaluate(self, genome: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate fitness offline (simulated).

        Args:
            genome: Perturbation genome

        Returns:
            Tuple of fitness objectives
        """
        # Decode perturbation
        params = self.perturbation_generator.encode_perturbation(genome)

        # Apply perturbation to reference point cloud
        perturbed_cloud = self.perturbation_generator.apply_perturbation(
            self.reference_point_cloud, params
        )

        # Simulate trajectory degradation based on perturbation magnitude
        # This is a simplified model - in reality you'd run MOLA
        perturbation_magnitude = np.sqrt(
            np.linalg.norm(params["translation"]) ** 2
            + np.linalg.norm(params["rotation"]) ** 2
            + params["intensity_scale"] ** 2
            + params["dropout_rate"] ** 2
        )

        # Add noise proportional to perturbation
        noise_scale = perturbation_magnitude * self.degradation_factor
        noise = np.random.normal(0, noise_scale, self.baseline_trajectory.shape)
        degraded_trajectory = self.baseline_trajectory + noise

        # Compute fitness
        fitness = compute_multi_objective_fitness(
            ground_truth_trajectory=self.ground_truth_trajectory,
            estimated_trajectory=degraded_trajectory,
            original_point_cloud=self.reference_point_cloud,
            perturbed_point_cloud=perturbed_cloud,
            error_method=self.error_method,
            imperceptibility_method=self.imperceptibility_method,
        )

        return fitness

    def get_fitness_function(self) -> Callable[[np.ndarray], Tuple[float, float]]:
        """Get fitness function for optimizer."""
        return self.evaluate
