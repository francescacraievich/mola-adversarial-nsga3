"""Tests for data loaders."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loaders import load_point_clouds_from_npy, load_trajectory_from_tum  # noqa: E402


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
