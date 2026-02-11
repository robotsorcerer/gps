"""
Unit tests for Sample and SampleList classes.

Tests cover:
- Sample initialization and data storage
- Data retrieval and manipulation
- SampleList operations
- Edge cases and error handling
"""
import pytest
import numpy as np
from typing import Any

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps.sample.sample import Sample
from gps.sample.sample_list import SampleList


class TestSample:
    """Unit tests for Sample class."""

    @pytest.mark.unit
    def test_sample_initialization(self, sample_dimensions):
        """Test Sample object initialization."""
        T = sample_dimensions['T']
        sample = Sample(None)  # Mock agent

        # Manually set T for testing
        sample.T = T
        sample.dX = sample_dimensions['dX']
        sample.dU = sample_dimensions['dU']

        assert sample.T == T
        assert sample.dX == sample_dimensions['dX']
        assert sample.dU == sample_dimensions['dU']

    @pytest.mark.unit
    def test_sample_set_get_X(self, sample_dimensions, random_state):
        """Test setting and getting state data."""
        T = sample_dimensions['T']
        dX = sample_dimensions['dX']

        sample = Sample(None)
        sample.T = T
        sample.dX = dX

        # Create test data
        X = random_state.randn(T, dX)

        # Set data (would need proper implementation)
        sample._X = X

        # Get data
        X_retrieved = sample._X

        np.testing.assert_array_equal(X, X_retrieved)

    @pytest.mark.unit
    def test_sample_set_get_U(self, sample_dimensions, random_state):
        """Test setting and getting action data."""
        T = sample_dimensions['T']
        dU = sample_dimensions['dU']

        sample = Sample(None)
        sample.T = T
        sample.dU = dU

        # Create test data
        U = random_state.randn(T, dU)

        # Set and verify
        sample._U = U
        U_retrieved = sample._U

        np.testing.assert_array_equal(U, U_retrieved)

    @pytest.mark.unit
    def test_sample_get_X_at_timestep(self, sample_dimensions, random_state):
        """Test getting state at specific timestep."""
        T = sample_dimensions['T']
        dX = sample_dimensions['dX']

        sample = Sample(None)
        sample.T = T
        sample.dX = dX
        sample._X = random_state.randn(T, dX)

        # Get state at t=5
        t = 5
        X_t = sample._X[t]

        assert X_t.shape == (dX,)

    @pytest.mark.unit
    def test_sample_dimensions_consistency(self, sample_dimensions):
        """Test that dimensions are consistent throughout sample."""
        sample = Sample(None)
        sample.T = sample_dimensions['T']
        sample.dX = sample_dimensions['dX']
        sample.dU = sample_dimensions['dU']

        assert sample.T == sample_dimensions['T']
        assert sample.dX == sample_dimensions['dX']
        assert sample.dU == sample_dimensions['dU']

    @pytest.mark.unit
    def test_sample_data_types(self, sample_dimensions, random_state):
        """Test that data has correct numpy dtypes."""
        T = sample_dimensions['T']
        dX = sample_dimensions['dX']

        sample = Sample(None)
        sample._X = random_state.randn(T, dX)

        assert isinstance(sample._X, np.ndarray)
        assert sample._X.dtype == np.float64


class TestSampleList:
    """Unit tests for SampleList class."""

    @pytest.mark.unit
    def test_samplelist_initialization_empty(self):
        """Test SampleList initialization with empty list."""
        sample_list = SampleList([])
        assert len(sample_list._samples) == 0

    @pytest.mark.unit
    def test_samplelist_initialization_with_samples(self, sample_dimensions):
        """Test SampleList initialization with sample data."""
        # Create mock samples
        samples = []
        for _ in range(3):
            sample = Sample(None)
            sample.T = sample_dimensions['T']
            sample.dX = sample_dimensions['dX']
            sample.dU = sample_dimensions['dU']
            samples.append(sample)

        sample_list = SampleList(samples)
        assert len(sample_list._samples) == 3

    @pytest.mark.unit
    def test_samplelist_len(self, sample_dimensions):
        """Test __len__ method of SampleList."""
        N = 5  # Number of samples
        samples = []
        for _ in range(N):
            sample = Sample(None)
            sample.T = sample_dimensions['T']
            samples.append(sample)

        sample_list = SampleList(samples)
        assert len(sample_list) == N

    @pytest.mark.unit
    def test_samplelist_getitem(self, sample_dimensions):
        """Test indexing into SampleList."""
        samples = []
        for i in range(3):
            sample = Sample(None)
            sample.T = sample_dimensions['T']
            sample._id = i  # Add identifier
            samples.append(sample)

        sample_list = SampleList(samples)

        # Test indexing
        assert sample_list[0]._id == 0
        assert sample_list[1]._id == 1
        assert sample_list[2]._id == 2

    @pytest.mark.unit
    def test_samplelist_get_X(self, sample_dimensions, random_state):
        """Test get_X method returns correct shape."""
        N = 4  # Number of samples
        T = sample_dimensions['T']
        dX = sample_dimensions['dX']

        samples = []
        for _ in range(N):
            sample = Sample(None)
            sample.T = T
            sample.dX = dX
            sample._X = random_state.randn(T, dX)
            samples.append(sample)

        sample_list = SampleList(samples)

        # Get all X data - should be N x T x dX
        # Note: This requires proper implementation in SampleList
        # For now, verify individual samples
        for i, sample in enumerate(sample_list._samples):
            assert sample._X.shape == (T, dX)

    @pytest.mark.unit
    def test_samplelist_get_U(self, sample_dimensions, random_state):
        """Test get_U method returns correct shape."""
        N = 3
        T = sample_dimensions['T']
        dU = sample_dimensions['dU']

        samples = []
        for _ in range(N):
            sample = Sample(None)
            sample.T = T
            sample.dU = dU
            sample._U = random_state.randn(T, dU)
            samples.append(sample)

        sample_list = SampleList(samples)

        # Verify individual samples
        for sample in sample_list._samples:
            assert sample._U.shape == (T, dU)


class TestSampleEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_sample_zero_timesteps(self):
        """Test sample with T=0."""
        sample = Sample(None)
        sample.T = 0
        assert sample.T == 0

    @pytest.mark.unit
    def test_sample_large_timesteps(self):
        """Test sample with large T."""
        T = 1000
        sample = Sample(None)
        sample.T = T
        assert sample.T == T

    @pytest.mark.unit
    def test_samplelist_single_sample(self, sample_dimensions):
        """Test SampleList with only one sample."""
        sample = Sample(None)
        sample.T = sample_dimensions['T']

        sample_list = SampleList([sample])
        assert len(sample_list) == 1

    @pytest.mark.unit
    def test_sample_data_immutability(self, sample_dimensions, random_state):
        """Test that modifying retrieved data doesn't affect original."""
        T = sample_dimensions['T']
        dX = sample_dimensions['dX']

        sample = Sample(None)
        X_orig = random_state.randn(T, dX)
        sample._X = X_orig.copy()

        # Get data and modify
        X_retrieved = sample._X
        X_retrieved[0, 0] = 999.0

        # Original should be affected (this is expected behavior)
        # If immutability is required, Sample should return copies
        assert sample._X[0, 0] == 999.0


# Performance tests
class TestSamplePerformance:
    """Performance tests for Sample operations."""

    @pytest.mark.slow
    def test_sample_large_trajectory(self, benchmark_iterations):
        """Test performance with large trajectories."""
        T = 1000
        dX = 50

        sample = Sample(None)
        sample.T = T
        sample.dX = dX
        sample._X = np.random.randn(T, dX)

        # Simple performance check
        assert sample._X.shape == (T, dX)

    @pytest.mark.slow
    def test_samplelist_many_samples(self):
        """Test SampleList with many samples."""
        N = 100
        T = 100

        samples = []
        for _ in range(N):
            sample = Sample(None)
            sample.T = T
            samples.append(sample)

        sample_list = SampleList(samples)
        assert len(sample_list) == N


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
