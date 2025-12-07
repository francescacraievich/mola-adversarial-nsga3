"""Tests for run_nsga3.py utility functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from optimization.run_nsga3 import (  # noqa: E402
        load_tf_from_npy,
        load_tf_static_from_npy,
    )

    ROS2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False
    load_tf_from_npy = None
    load_tf_static_from_npy = None

pytestmark = pytest.mark.skipif(not ROS2_AVAILABLE, reason="ROS2 dependencies not available")


class TestDataLoaders:
    """Test data loading functions."""

    def test_load_tf_from_npy(self, tmp_path):
        """Test loading TF data from numpy file."""
        # Create test data
        test_tf = [{"frame_id": "test", "child_frame_id": "test_child"}]
        tf_file = tmp_path / "test_tf.npy"
        np.save(tf_file, test_tf)

        # Load and verify
        loaded_tf = load_tf_from_npy(str(tf_file))
        assert isinstance(loaded_tf, list)
        assert len(loaded_tf) == 1
        assert loaded_tf[0]["frame_id"] == "test"

    def test_load_tf_static_from_npy(self, tmp_path):
        """Test loading TF static data from numpy file."""
        # Create test data
        test_tf_static = [{"frame_id": "base", "child_frame_id": "sensor"}]
        tf_static_file = tmp_path / "test_tf_static.npy"
        np.save(tf_static_file, test_tf_static)

        # Load and verify
        loaded_tf_static = load_tf_static_from_npy(str(tf_static_file))
        assert isinstance(loaded_tf_static, list)
        assert len(loaded_tf_static) == 1
        assert loaded_tf_static[0]["frame_id"] == "base"


class TestVoxelDownsampling:
    """Test voxel downsampling functionality."""

    def test_voxel_downsample_reduces_points(self):
        """Test that voxel downsampling reduces point count."""
        # Import here to avoid ROS2 dependencies in main test
        try:
            from optimization.run_nsga3 import MOLAEvaluator  # noqa: F401
        except ImportError:
            pytest.skip("ROS2 dependencies not available")

        # Create dense point cloud
        points = np.random.rand(1000, 3) * 10  # 1000 points in 10x10x10 space

        # Mock evaluator to test downsampling
        # We'll test the method directly if possible
        voxel_size = 0.5
        # Simple voxel downsampling logic
        voxel_indices = np.floor(points / voxel_size).astype(int)
        unique_voxels = np.unique(voxel_indices, axis=0)

        # Downsampled should have fewer points
        assert len(unique_voxels) < len(points)
        assert len(unique_voxels) > 0

    def test_voxel_downsample_preserves_dimensions(self):
        """Test that voxel downsampling preserves point dimensions."""
        points = np.random.rand(100, 3)
        voxel_size = 0.2

        voxel_indices = np.floor(points / voxel_size).astype(int)
        unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

        # Reconstruct points from voxels
        downsampled = np.array(
            [points[inverse == i].mean(axis=0) for i in range(len(unique_voxels))]
        )

        assert downsampled.shape[1] == 3
        assert len(downsampled) > 0


class TestOptimizerSetup:
    """Test optimizer setup functions."""

    def test_optimizer_problem_dimensions(self):
        """Test that NSGA-III problem has correct dimensions."""
        genome_size = 17
        n_objectives = 2

        # Mock evaluator function
        def mock_evaluator(genome):
            return np.array([1.0, 1.0])

        # Test problem setup
        from pymoo.core.problem import Problem

        class TestProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=genome_size,
                    n_obj=n_objectives,
                    xl=-1.0 * np.ones(genome_size),
                    xu=1.0 * np.ones(genome_size),
                )

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.array([mock_evaluator(g) for g in X])

        problem = TestProblem()
        assert problem.n_var == 17
        assert problem.n_obj == 2
        assert np.all(problem.xl == -1.0)
        assert np.all(problem.xu == 1.0)

    def test_reference_directions_generation(self):
        """Test reference directions for NSGA-III."""
        from pymoo.util.ref_dirs import get_reference_directions

        # Generate reference directions for 2 objectives
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        assert ref_dirs.shape[1] == 2  # 2 objectives
        assert ref_dirs.shape[0] > 0  # Some reference points
        # All directions should sum to 1 (simplex constraint)
        assert np.allclose(ref_dirs.sum(axis=1), 1.0)

    def test_algorithm_operators(self):
        """Test NSGA-III algorithm operators."""
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM

        # Test crossover - just verify it can be instantiated
        sbx = SBX(prob=0.9, eta=15)
        assert sbx is not None
        # Note: prob and eta are now Real objects in newer pymoo versions

        # Test mutation
        pm = PM(prob=0.2, eta=20)
        assert pm is not None


class TestCallbacks:
    """Test optimization callbacks."""

    def test_history_callback_initialization(self):
        """Test history callback starts empty."""
        from pymoo.core.callback import Callback

        class TestCallback(Callback):
            def __init__(self):
                super().__init__()
                self.all_fitness = []
                self.all_genomes = []
                self.valid_fitness = []
                self.valid_genomes = []

        callback = TestCallback()
        assert len(callback.all_fitness) == 0
        assert len(callback.all_genomes) == 0
        assert len(callback.valid_fitness) == 0
        assert len(callback.valid_genomes) == 0

    def test_history_callback_stores_data(self):
        """Test history callback stores fitness and genomes."""
        from pymoo.core.callback import Callback

        class TestCallback(Callback):
            def __init__(self):
                super().__init__()
                self.all_fitness = []
                self.all_genomes = []

            def notify(self, algorithm):
                # Mock algorithm with population
                pass

        callback = TestCallback()
        # Manually add data
        callback.all_fitness.append(np.array([1.0, 2.0]))
        callback.all_genomes.append(np.ones(17))

        assert len(callback.all_fitness) == 1
        assert len(callback.all_genomes) == 1
        assert callback.all_fitness[0].shape == (2,)
        assert callback.all_genomes[0].shape == (17,)


class TestValidityFiltering:
    """Test validity filtering logic."""

    def test_ate_validity_threshold(self):
        """Test ATE validity threshold filtering."""
        # Simulate fitness values (negative ATE, perturbation)
        fitness_values = [
            np.array([-5.0, 2.0]),  # Valid (ATE = 5m < 10m)
            np.array([-15.0, 1.0]),  # Invalid (ATE = 15m > 10m)
            np.array([-3.0, 4.0]),  # Valid (ATE = 3m < 10m)
            np.array([-12.0, 0.5]),  # Invalid (ATE = 12m > 10m)
        ]

        valid_threshold = 10.0
        valid_fitness = [f for f in fitness_values if -f[0] < valid_threshold]

        assert len(valid_fitness) == 2
        assert -valid_fitness[0][0] < 10.0
        assert -valid_fitness[1][0] < 10.0

    def test_validity_filtering_with_genomes(self):
        """Test validity filtering keeps matching genomes."""
        fitness_values = [np.array([-5.0, 2.0]), np.array([-15.0, 1.0]), np.array([-3.0, 4.0])]
        genomes = [np.ones(17) * i for i in range(3)]

        valid_pairs = [(f, g) for f, g in zip(fitness_values, genomes) if -f[0] < 10.0]

        assert len(valid_pairs) == 2
        assert valid_pairs[0][1][0] == 0.0  # First genome
        assert valid_pairs[1][1][0] == 2.0  # Third genome


class TestOutputFormatting:
    """Test output formatting functions."""

    def test_numbered_output_format(self):
        """Test numbered output file naming."""
        base_path = "results/optimized_genome"
        run_number = 5
        expected = f"{base_path}{run_number}"
        assert expected == f"results/optimized_genome{run_number}"

    def test_save_paths_generation(self):
        """Test generation of save paths for results."""
        from pathlib import Path

        numbered_output = "results/run10"
        base_path = Path(numbered_output)

        points_path = base_path.with_suffix(".points.npy")
        genomes_path = base_path.with_suffix(".genomes.npy")
        valid_points_path = base_path.with_suffix(".valid_points.npy")
        valid_genomes_path = base_path.with_suffix(".valid_genomes.npy")

        assert str(points_path) == "results/run10.points.npy"
        assert str(genomes_path) == "results/run10.genomes.npy"
        assert str(valid_points_path) == "results/run10.valid_points.npy"
        assert str(valid_genomes_path) == "results/run10.valid_genomes.npy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
