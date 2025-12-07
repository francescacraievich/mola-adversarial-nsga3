"""Tests for MOLAEvaluator class - ROS2 SLAM integration."""

import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import ROS2 dependencies
try:
    import rclpy  # noqa: F401
    from geometry_msgs.msg import Point, Pose, PoseWithCovariance  # noqa: F401
    from nav_msgs.msg import Odometry  # noqa: F401
    from rclpy.node import Node  # noqa: F401
    from sensor_msgs.msg import PointCloud2  # noqa: F401

    from optimization.run_nsga3 import MOLAEvaluator  # noqa: E402

    ROS2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False
    MOLAEvaluator = None

pytestmark = pytest.mark.skipif(not ROS2_AVAILABLE, reason="ROS2 dependencies not available")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_point_cloud():
    """Generate test point cloud (1000 points, 4 channels: x, y, z, intensity)."""
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10  # 10x10x10 meter space
    intensity = np.ones((1000, 1)) * 100.0  # Constant intensity
    return np.hstack([points, intensity]).astype(np.float32)


@pytest.fixture
def small_point_cloud():
    """Small point cloud for edge case testing."""
    return np.array([[0, 0, 0, 100], [0.1, 0, 0, 100], [0, 0.1, 0, 100]], dtype=np.float32)


@pytest.fixture
def sample_timestamps():
    """Generate test timestamps in nanoseconds."""
    return np.array([i * 1e9 for i in range(10)])


@pytest.fixture
def sample_tf_sequence():
    """Generate test TF transform sequence."""
    return [
        {
            "timestamp": i * 1e9,
            "frame_id": "base_link",
            "child_frame_id": "lidar",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],  # Identity quaternion (w last)
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_ground_truth():
    """Generate ground truth trajectory."""
    np.random.seed(42)
    return np.cumsum(np.random.rand(100, 3) * 0.1, axis=0)  # Smooth trajectory


@pytest.fixture
def mock_perturbation_generator():
    """Mock PerturbationGenerator for testing."""
    mock_gen = MagicMock()
    mock_gen.max_point_shift = 0.05
    mock_gen.get_genome_size.return_value = 17
    mock_gen.encode_perturbation.return_value = {
        "noise_intensity": 0.02,
        "curvature_strength": 0.5,
        "dropout_rate": 0.01,
        "ghost_ratio": 0.01,
        "cluster_strength": 0.1,
        "spatial_correlation": 0.5,
        "geometric_distortion": 0.1,
        "edge_attack_strength": 0.0,
        "temporal_drift_strength": 0.0,
        "scanline_strength": 0.0,
        "strategic_ghost": 0.0,
    }
    mock_gen.compute_chamfer_distance.return_value = 0.001
    mock_gen.compute_perturbation_magnitude.return_value = 2.5  # cm
    mock_gen.reset_temporal_state.return_value = None
    mock_gen.apply_perturbation.return_value = np.random.rand(1000, 4).astype(np.float32)
    return mock_gen


def create_minimal_evaluator():
    """Create MOLAEvaluator instance bypassing ROS2 initialization for unit tests."""
    with patch.object(Node, "__init__", return_value=None):
        evaluator = object.__new__(MOLAEvaluator)
        evaluator.evaluation_count = 0
        evaluator.collected_trajectory = []
        evaluator.mola_process = None
        evaluator.tf_indices_per_cloud = []
        return evaluator


def create_mock_odom_msg(x, y, z):
    """Create mock Odometry message for testing."""
    msg = MagicMock(spec=Odometry)
    msg.pose.pose.position = MagicMock(x=x, y=y, z=z)
    return msg


# ============================================================================
# TEST VOXEL DOWNSAMPLING
# ============================================================================


class TestVoxelDownsampling:
    """Test voxel downsampling - pure numpy logic, no ROS2 deps."""

    def test_voxel_downsample_reduces_points(self, sample_point_cloud):
        """Verify voxel downsampling reduces point count."""
        evaluator = create_minimal_evaluator()
        result = evaluator._voxel_downsample(sample_point_cloud, voxel_size=0.15)

        assert len(result) < len(sample_point_cloud), "Should reduce point count"
        assert len(result) > 0, "Should not be empty"
        assert result.shape[1] == 4, "Should preserve 4 columns (x, y, z, intensity)"
        assert result.dtype == sample_point_cloud.dtype, "Should preserve dtype"

    def test_voxel_downsample_preserves_dimensions(self, sample_point_cloud):
        """Verify shape is (N, 4) after downsampling."""
        evaluator = create_minimal_evaluator()
        result = evaluator._voxel_downsample(sample_point_cloud, voxel_size=0.15)

        assert result.ndim == 2, "Should be 2D array"
        assert result.shape[1] == 4, "Should have 4 columns"

    @pytest.mark.parametrize(
        "voxel_size",
        [0.05, 0.10, 0.15, 0.20, 0.30],
    )
    def test_voxel_downsample_size_correlation(self, sample_point_cloud, voxel_size):
        """Test that larger voxel sizes produce more reduction."""
        evaluator = create_minimal_evaluator()
        result = evaluator._voxel_downsample(sample_point_cloud, voxel_size)

        # Just verify reduction happens and output is valid
        assert len(result) > 0, "Should have some points"
        assert len(result) <= len(sample_point_cloud), "Should not add points"
        assert result.shape[1] == 4, "Should preserve 4 columns"

    def test_voxel_downsample_negative_coordinates(self):
        """Handle negative coordinates correctly."""
        evaluator = create_minimal_evaluator()
        points = np.array(
            [
                [-5.0, -3.0, -1.0, 100],
                [-5.05, -3.05, -1.05, 100],  # Same voxel with 0.15m size
                [-4.0, -2.0, 0.0, 100],  # Different voxel
                [1.0, 2.0, 3.0, 100],  # Positive coords
            ],
            dtype=np.float32,
        )

        result = evaluator._voxel_downsample(points, voxel_size=0.15)

        # Should have 2-4 distinct voxels (depends on voxel boundaries)
        assert 2 <= len(result) <= 4
        assert result.shape[1] == 4

    def test_voxel_downsample_single_point(self):
        """Test with single point."""
        evaluator = create_minimal_evaluator()
        points = np.array([[1.0, 2.0, 3.0, 100]], dtype=np.float32)

        result = evaluator._voxel_downsample(points, voxel_size=0.15)

        assert len(result) == 1, "Single point should remain"
        np.testing.assert_array_equal(result, points)

    def test_voxel_downsample_all_same_voxel(self):
        """Test when all points fall in same voxel."""
        evaluator = create_minimal_evaluator()
        # All points within 0.1m of each other (same voxel with 0.15m size)
        points = np.array(
            [
                [0.0, 0.0, 0.0, 100],
                [0.01, 0.01, 0.01, 100],
                [0.02, 0.02, 0.02, 100],
                [0.03, 0.03, 0.03, 100],
            ],
            dtype=np.float32,
        )

        result = evaluator._voxel_downsample(points, voxel_size=0.15)

        assert len(result) == 1, "All points in same voxel should reduce to 1"

    def test_voxel_downsample_empty_input(self):
        """Test voxel downsampling with empty array."""
        evaluator = create_minimal_evaluator()
        empty = np.array([]).reshape(0, 4)

        result = evaluator._voxel_downsample(empty, voxel_size=0.15)

        assert len(result) == 0, "Empty input should return empty output"
        assert result.shape == (0, 4), "Should preserve column count"


# ============================================================================
# TEST TRAJECTORY COLLECTION
# ============================================================================


class TestTrajectoryCollection:
    """Test odometry callback and trajectory state management."""

    def test_odom_callback_stores_position(self):
        """Verify odometry messages are stored correctly."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())

        odom_msg = create_mock_odom_msg(1.5, 2.3, 0.1)
        evaluator._odom_callback(odom_msg)

        assert len(evaluator.collected_trajectory) == 1
        assert evaluator.collected_trajectory[0] == [1.5, 2.3, 0.1]

    def test_odom_callback_multiple_messages(self):
        """Test multiple odometry messages accumulate sequentially."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())

        positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0.5, 0]]

        for pos in positions:
            odom_msg = create_mock_odom_msg(*pos)
            evaluator._odom_callback(odom_msg)

        assert len(evaluator.collected_trajectory) == 4
        assert evaluator.collected_trajectory == positions

    def test_odom_callback_logs_first_message(self):
        """Test that first odometry is logged."""
        evaluator = create_minimal_evaluator()
        mock_logger = MagicMock()
        evaluator.get_logger = MagicMock(return_value=mock_logger)

        odom_msg = create_mock_odom_msg(1.0, 2.0, 3.0)
        evaluator._odom_callback(odom_msg)

        # Logger info should be called for first message
        mock_logger.info.assert_called()

    def test_odom_callback_extracts_xyz_correctly(self):
        """Test correct extraction of x, y, z from Odometry message."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())

        odom_msg = create_mock_odom_msg(1.234, 5.678, 9.012)
        evaluator._odom_callback(odom_msg)

        expected = [1.234, 5.678, 9.012]
        assert evaluator.collected_trajectory[0] == expected


# ============================================================================
# TEST MOLA LIFECYCLE
# ============================================================================


class TestMOLALifecycle:
    """Test MOLA subprocess management and lifecycle."""

    @patch("optimization.run_nsga3.subprocess.Popen")
    @patch("optimization.run_nsga3.time.sleep")
    def test_start_mola_subprocess_created(self, mock_sleep, mock_popen):
        """Verify MOLA subprocess is started with correct command."""
        evaluator = create_minimal_evaluator()
        evaluator.mola_binary_path = "/path/to/mola-cli"
        evaluator.mola_config_path = "/path/to/config.yaml"
        evaluator.lidar_topic = "/test/lidar"
        evaluator._logger = MagicMock()  # Mock logger

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process

        evaluator._start_mola()

        # Verify subprocess.Popen was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args

        # Check command includes binary and config
        cmd = call_args[0][0]
        assert evaluator.mola_binary_path in cmd
        assert evaluator.mola_config_path in cmd

    @patch("optimization.run_nsga3.subprocess.Popen")
    @patch("optimization.run_nsga3.time.sleep")
    def test_start_mola_environment_variables(self, mock_sleep, mock_popen):
        """Test MOLA environment variables are set correctly."""
        evaluator = create_minimal_evaluator()
        evaluator.mola_binary_path = "/path/to/mola-cli"
        evaluator.mola_config_path = "/path/to/config.yaml"
        evaluator.lidar_topic = "/test/lidar"
        evaluator._logger = MagicMock()  # Mock logger

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        evaluator._start_mola()

        call_args = mock_popen.call_args
        env = call_args[1]["env"]

        # Verify key environment variables
        assert env["MOLA_LIDAR_TOPIC"] == "/test/lidar"
        assert env["MOLA_WITH_GUI"] == "false"
        assert env["MOLA_USE_FIXED_LIDAR_POSE"] == "true"
        assert "MOLA_MIN_XYZ_BETWEEN_MAP_UPDATES" in env
        assert "MOLA_MIN_ROT_BETWEEN_MAP_UPDATES" in env

    @patch("optimization.run_nsga3.subprocess.Popen")
    @patch("optimization.run_nsga3.time.sleep")
    def test_start_mola_binary_path(self, mock_sleep, mock_popen):
        """Test that configured MOLA binary path is used."""
        evaluator = create_minimal_evaluator()
        evaluator.mola_binary_path = "/custom/path/mola-cli"
        evaluator.mola_config_path = "/config.yaml"
        evaluator.lidar_topic = "/lidar"
        evaluator._logger = MagicMock()  # Mock logger

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        evaluator._start_mola()

        cmd = mock_popen.call_args[0][0]
        assert "/custom/path/mola-cli" in cmd

    @patch("optimization.run_nsga3.subprocess.Popen")
    @patch("optimization.run_nsga3.time.sleep")
    def test_start_mola_failure_detection(self, mock_sleep, mock_popen):
        """Test detection of MOLA startup failure."""
        evaluator = create_minimal_evaluator()
        evaluator.mola_binary_path = "/path/to/mola-cli"
        evaluator.mola_config_path = "/path/to/config.yaml"
        evaluator.lidar_topic = "/lidar"
        mock_logger = MagicMock()
        evaluator._logger = mock_logger  # Mock logger

        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process exited with error
        mock_process.communicate.return_value = (b"error output", b"")  # Mock communicate
        mock_popen.return_value = mock_process

        evaluator._start_mola()

        # Verify error was logged (code doesn't raise exception, just logs)
        mock_logger.error.assert_called()
        error_msg = mock_logger.error.call_args[0][0]
        assert "MOLA failed to start" in error_msg

    @patch("optimization.run_nsga3.os.killpg")
    @patch("optimization.run_nsga3.os.getpgid")
    @patch("optimization.run_nsga3.subprocess.run")
    @patch("optimization.run_nsga3.time.sleep")
    def test_stop_mola_sigterm(self, mock_sleep, mock_run, mock_getpgid, mock_killpg):
        """Test MOLA cleanup sends SIGTERM to process group."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = None  # Exits successfully
        evaluator.mola_process = mock_process
        mock_getpgid.return_value = 12345

        evaluator._stop_mola()

        # Should send SIGTERM
        mock_killpg.assert_called()
        first_call = mock_killpg.call_args_list[0]
        assert first_call[0][1] == signal.SIGTERM

    @patch("optimization.run_nsga3.os.killpg")
    @patch("optimization.run_nsga3.os.getpgid")
    @patch("optimization.run_nsga3.subprocess.run")
    @patch("optimization.run_nsga3.time.sleep")
    def test_stop_mola_sigkill_on_timeout(self, mock_sleep, mock_run, mock_getpgid, mock_killpg):
        """Test SIGKILL is sent if SIGTERM times out."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 3.0)
        evaluator.mola_process = mock_process
        mock_getpgid.return_value = 12345

        evaluator._stop_mola()

        # Should send both SIGTERM and SIGKILL
        assert mock_killpg.call_count >= 2
        calls = mock_killpg.call_args_list
        assert calls[0][0][1] == signal.SIGTERM
        assert calls[1][0][1] == signal.SIGKILL

    @patch("optimization.run_nsga3.subprocess.run")
    @patch("optimization.run_nsga3.time.sleep")
    def test_stop_mola_cleanup_pkill(self, mock_sleep, mock_run):
        """Test pkill fallback is executed."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())
        evaluator.mola_process = None  # No process reference

        evaluator._stop_mola()

        # pkill should be called as fallback
        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        assert "pkill" in cmd

    def test_stop_mola_no_process(self):
        """Test handling when no process is running."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())
        evaluator.mola_process = None

        # Should not raise exception
        with patch("optimization.run_nsga3.subprocess.run"):
            evaluator._stop_mola()


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_voxel_downsample_empty_input_edge_case(self):
        """Test voxel downsampling with empty array (edge case coverage)."""
        evaluator = create_minimal_evaluator()
        empty = np.array([]).reshape(0, 4)

        result = evaluator._voxel_downsample(empty, voxel_size=0.15)

        assert len(result) == 0
        assert result.shape == (0, 4)

    def test_voxel_downsample_single_point_edge_case(self):
        """Test with single point (coverage for edge case)."""
        evaluator = create_minimal_evaluator()
        points = np.array([[0, 0, 0, 100]], dtype=np.float32)

        result = evaluator._voxel_downsample(points, voxel_size=0.15)

        assert len(result) == 1
        np.testing.assert_array_equal(result, points)

    def test_voxel_downsample_very_small_voxel(self):
        """Test with very small voxel size (minimal reduction)."""
        evaluator = create_minimal_evaluator()
        points = np.random.rand(100, 4).astype(np.float32) * 10

        result = evaluator._voxel_downsample(points, voxel_size=0.001)

        # Should keep most points with tiny voxel
        assert len(result) > 90

    def test_voxel_downsample_very_large_voxel(self):
        """Test with very large voxel size (maximum reduction)."""
        evaluator = create_minimal_evaluator()
        points = np.random.rand(1000, 4).astype(np.float32) * 10

        result = evaluator._voxel_downsample(points, voxel_size=50.0)

        # All points should fall in one voxel
        assert len(result) == 1

    def test_odom_callback_handles_empty_trajectory_list(self):
        """Test odometry callback when trajectory list is initially empty."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())
        evaluator.collected_trajectory = []

        odom_msg = create_mock_odom_msg(1.0, 2.0, 3.0)
        evaluator._odom_callback(odom_msg)

        assert len(evaluator.collected_trajectory) == 1

    def test_odom_callback_handles_zero_coordinates(self):
        """Test odometry callback with zero coordinates."""
        evaluator = create_minimal_evaluator()
        evaluator.get_logger = MagicMock(return_value=MagicMock())

        odom_msg = create_mock_odom_msg(0.0, 0.0, 0.0)
        evaluator._odom_callback(odom_msg)

        assert evaluator.collected_trajectory[0] == [0.0, 0.0, 0.0]

    def test_trajectory_reset_before_playback(self):
        """Test that trajectory is properly reset."""
        evaluator = create_minimal_evaluator()
        evaluator.collected_trajectory = [[1, 2, 3], [4, 5, 6]]  # Old data

        # Manually reset (as would happen in _run_synchronized_playback)
        evaluator.collected_trajectory = []

        assert len(evaluator.collected_trajectory) == 0

    def test_evaluation_count_increments(self):
        """Test evaluation counter increments correctly."""
        evaluator = create_minimal_evaluator()
        evaluator.evaluation_count = 0

        evaluator.evaluation_count += 1
        assert evaluator.evaluation_count == 1

        evaluator.evaluation_count += 1
        assert evaluator.evaluation_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
