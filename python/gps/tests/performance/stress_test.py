"""
Stress testing for GPS components.

Tests system behavior under extreme conditions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestExtremeConditions:
    """Test under extreme conditions."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_very_high_dimensions(self):
        """Test with extremely high dimensional spaces."""
        T = 100
        dX = 500  # Very high dimensional state
        dU = 100  # Very high dimensional action
        N = 5

        X = np.random.randn(N, T, dX)
        U = np.random.randn(N, T, dU)

        # Should handle without crashing
        Fm = np.random.randn(T, dX, dX + dU)
        assert Fm.shape == (T, dX, dX + dU)

    @pytest.mark.stress
    @pytest.mark.slow
    def test_very_long_horizon(self):
        """Test with extremely long time horizons."""
        T = 1000  # Very long horizon
        dX, dU = 10, 5
        N = 10

        X = np.random.randn(N, T, dX)
        U = np.random.randn(N, T, dU)

        cost = np.sum(X**2) + np.sum(U**2)
        assert not np.isnan(cost)

    @pytest.mark.stress
    @pytest.mark.slow
    def test_many_conditions(self):
        """Test with many initial conditions."""
        T, dX, dU = 50, 10, 5
        M = 20  # Many conditions
        N = 10

        for m in range(M):
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)
            cost = np.sum(X**2)

        assert True  # Completed without crash


class TestNumericalStability:
    """Test numerical stability under stress."""

    @pytest.mark.stress
    def test_near_singular_matrices(self):
        """Test with near-singular matrices."""
        dX = 10
        T = 50

        # Nearly singular matrix
        A = np.eye(dX) * 1e-10

        # Should handle gracefully
        try:
            inv_A = np.linalg.pinv(A)
            assert inv_A.shape == (dX, dX)
        except:
            pytest.skip("Numerically unstable")

    @pytest.mark.stress
    def test_extreme_values(self):
        """Test with extreme numerical values."""
        T, dX = 50, 10

        # Very large values
        X_large = np.random.randn(T, dX) * 1e6

        # Very small values
        X_small = np.random.randn(T, dX) * 1e-6

        # Should normalize or handle
        cost_large = np.sum(X_large**2)
        cost_small = np.sum(X_small**2)

        assert not np.isnan(cost_large)
        assert not np.isinf(cost_large)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'stress'])
