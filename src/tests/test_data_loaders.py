"""Tests for data loaders."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loaders import (  # noqa: E402
    load_point_clouds_from_npy,
    load_timestamps_from_npy,
    load_trajectory_from_tum,
)


class TestLoadPointCloudsFromNpy:
    """Tests for load_point_clouds_from_npy."""

    def test_load_existing_file(self):
        """Test loading existing .npy file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            clouds = [np.random.rand(100, 4) for _ in range(5)]
            np.save(f.name, clouds, allow_pickle=True)

            result = load_point_clouds_from_npy(f.name)
            assert result is not None
            assert len(result) == 5

            Path(f.name).unlink()

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns None."""
        result = load_point_clouds_from_npy("/nonexistent/path.npy")
        assert result is None


class TestLoadTrajectoryFromTum:
    """Tests for load_trajectory_from_tum."""

    def test_load_existing_file(self):
        """Test loading existing TUM file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tum", delete=False) as f:
            # TUM format: timestamp tx ty tz qx qy qz qw
            f.write("1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.write("2.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.write("3.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.flush()

            result = load_trajectory_from_tum(f.name)
            assert result is not None
            assert result.shape == (3, 3)
            assert result[0, 0] == 0.0
            assert result[1, 0] == 1.0
            assert result[2, 0] == 2.0

            Path(f.name).unlink()

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns None."""
        result = load_trajectory_from_tum("/nonexistent/path.tum")
        assert result is None

    def test_skip_comments(self):
        """Test that comments are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tum", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.flush()

            result = load_trajectory_from_tum(f.name)
            assert result is not None
            assert result.shape == (1, 3)

            Path(f.name).unlink()

    def test_load_npy_trajectory(self):
        """Test loading trajectory from .npy file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            traj = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
            np.save(f.name, traj)

            result = load_trajectory_from_tum(f.name)
            assert result is not None
            assert result.shape == (3, 3)
            np.testing.assert_array_equal(result, traj)

            Path(f.name).unlink()

    def test_load_npy_nonexistent(self):
        """Test loading nonexistent .npy trajectory returns None."""
        result = load_trajectory_from_tum("/nonexistent/path.npy")
        assert result is None

    def test_empty_tum_file(self):
        """Test loading empty TUM file returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tum", delete=False) as f:
            f.write("# Only comments\n")
            f.write("# No data\n")
            f.flush()

            result = load_trajectory_from_tum(f.name)
            assert result is None

            Path(f.name).unlink()

    def test_interpolation(self):
        """Test trajectory interpolation to match point cloud timestamps."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tum", delete=False) as f:
            # Create trajectory with 3 points
            f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.write("1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.write("2.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.flush()

            # Create timestamps for 5 frames (in nanoseconds)
            pc_timestamps = np.array([0, 500000000, 1000000000, 1500000000, 2000000000])

            result = load_trajectory_from_tum(
                f.name, interpolate_to_frames=5, pc_timestamps=pc_timestamps
            )
            assert result is not None
            assert result.shape == (5, 3)
            # Check interpolated values
            assert result[0, 0] == pytest.approx(0.0, abs=0.1)
            assert result[2, 0] == pytest.approx(1.0, abs=0.1)
            assert result[4, 0] == pytest.approx(2.0, abs=0.1)

            Path(f.name).unlink()


class TestLoadTimestampsFromNpy:
    """Tests for load_timestamps_from_npy."""

    def test_load_existing_timestamps(self):
        """Test loading existing timestamps file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            timestamps = np.array([0, 100000000, 200000000, 300000000, 400000000])
            np.save(f.name, timestamps)

            result = load_timestamps_from_npy(f.name)
            assert result is not None
            assert len(result) == 5
            np.testing.assert_array_equal(result, timestamps)

            Path(f.name).unlink()

    def test_load_nonexistent_timestamps(self):
        """Test loading nonexistent timestamps file returns None."""
        result = load_timestamps_from_npy("/nonexistent/timestamps.npy")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
