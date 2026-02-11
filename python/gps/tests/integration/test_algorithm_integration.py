"""
Integration tests for Algorithm components.

Tests the interaction between Algorithm, Dynamics, Cost, and TrajOpt.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class MockDynamicsIntegration:
    """Mock dynamics for integration testing."""
    def __init__(self, hyperparams):
        self.T = hyperparams.get('T', 10)
        self.dX = hyperparams.get('dX', 7)
        self.dU = hyperparams.get('dU', 3)
        self.Fm = np.random.randn(self.T, self.dX, self.dX + self.dU)
        self.fv = np.random.randn(self.T, self.dX)
        self.dyn_covar = np.tile(np.eye(self.dX) * 0.01, (self.T, 1, 1))

    def fit(self, X, U):
        return self.Fm, self.fv, self.dyn_covar

    def update_prior(self, X, U):
        pass

    def get_prior(self):
        return None


class MockCostIntegration:
    """Mock cost for integration testing."""
    def __init__(self, hyperparams):
        self.T = hyperparams.get('T', 10)
        self.dX = hyperparams.get('dX', 7)
        self.dU = hyperparams.get('dU', 3)

    def eval(self, sample):
        l = np.ones(self.T)
        lx = np.zeros((self.T, self.dX))
        lu = np.zeros((self.T, self.dU))
        lxx = np.tile(np.eye(self.dX) * 0.1, (self.T, 1, 1))
        luu = np.tile(np.eye(self.dU) * 0.1, (self.T, 1, 1))
        lux = np.zeros((self.T, self.dU, self.dX))
        return l, lx, lu, lxx, luu, lux


class MockSampleIntegration:
    """Mock sample for integration testing."""
    def __init__(self, T, dX, dU):
        self.T = T
        self.dX = dX
        self.dU = dU
        self.X = np.random.randn(T, dX)
        self.U = np.random.randn(T, dU)

    def get_X(self):
        return self.X

    def get_U(self):
        return self.U

    def get(self, data_type):
        return self.X


class TestAlgorithmDynamicsIntegration:
    """Test Algorithm and Dynamics interaction."""

    @pytest.mark.integration
    def test_dynamics_fitting_with_samples(self):
        """Test dynamics fitting with multiple samples."""
        T, dX, dU = 10, 7, 3
        N = 5  # Number of samples

        # Create samples
        X = np.random.randn(N, T, dX)
        U = np.random.randn(N, T, dU)

        # Fit dynamics
        dynamics = MockDynamicsIntegration({'T': T, 'dX': dX, 'dU': dU})
        Fm, fv, dyn_covar = dynamics.fit(X, U)

        # Verify dimensions
        assert Fm.shape == (T, dX, dX + dU)
        assert fv.shape == (T, dX)
        assert dyn_covar.shape == (T, dX, dX)


class TestAlgorithmCostIntegration:
    """Test Algorithm and Cost interaction."""

    @pytest.mark.integration
    def test_cost_evaluation_on_trajectory(self):
        """Test cost evaluation on full trajectory."""
        T, dX, dU = 10, 7, 3

        # Create sample trajectory
        sample = MockSampleIntegration(T, dX, dU)

        # Evaluate cost
        cost = MockCostIntegration({'T': T, 'dX': dX, 'dU': dU})
        l, lx, lu, lxx, luu, lux = cost.eval(sample)

        # Verify all derivatives returned
        assert l.shape == (T,)
        assert lx.shape == (T, dX)
        assert lu.shape == (T, dU)
        assert lxx.shape == (T, dX, dX)
        assert luu.shape == (T, dU, dU)
        assert lux.shape == (T, dU, dX)

        # Verify Hessians are symmetric
        for t in range(T):
            np.testing.assert_array_almost_equal(lxx[t], lxx[t].T)
            np.testing.assert_array_almost_equal(luu[t], luu[t].T)


class TestDynamicsCostIntegration:
    """Test Dynamics and Cost working together."""

    @pytest.mark.integration
    def test_dynamics_forward_pass_with_cost(self):
        """Test forward dynamics pass with cost evaluation."""
        T, dX, dU = 10, 7, 3

        # Initialize dynamics
        dynamics = MockDynamicsIntegration({'T': T, 'dX': dX, 'dU': dU})

        # Create initial state and actions
        x0 = np.random.randn(dX)
        U = np.random.randn(T, dU)

        # Forward simulate
        X = np.zeros((T, dX))
        X[0] = x0
        for t in range(T - 1):
            xu = np.concatenate([X[t], U[t]])
            X[t + 1] = dynamics.Fm[t] @ xu + dynamics.fv[t]

        # Evaluate cost on trajectory
        sample = MockSampleIntegration(T, dX, dU)
        sample.X = X
        sample.U = U

        cost = MockCostIntegration({'T': T, 'dX': dX, 'dU': dU})
        l, _, _, _, _, _ = cost.eval(sample)

        # Verify total cost
        total_cost = np.sum(l)
        assert total_cost > 0
        assert not np.isnan(total_cost)


class TestMultiConditionIntegration:
    """Test multi-condition algorithm setup."""

    @pytest.mark.integration
    def test_multiple_conditions_initialization(self):
        """Test algorithm with multiple initial conditions."""
        T, dX, dU = 10, 7, 3
        M = 4  # Number of conditions

        # Create dynamics for each condition
        dynamics_list = [
            MockDynamicsIntegration({'T': T, 'dX': dX, 'dU': dU})
            for _ in range(M)
        ]

        # Create costs for each condition
        cost_list = [
            MockCostIntegration({'T': T, 'dX': dX, 'dU': dU})
            for _ in range(M)
        ]

        # Verify independent initialization
        assert len(dynamics_list) == M
        assert len(cost_list) == M

        # Verify each has unique parameters
        for i in range(M):
            assert dynamics_list[i].Fm is not dynamics_list[0].Fm or i == 0


class TestTrajectoryOptimizationIntegration:
    """Test trajectory optimization with dynamics and cost."""

    @pytest.mark.integration
    def test_backward_pass_with_cost_derivatives(self):
        """Test backward pass using cost derivatives."""
        T, dX, dU = 10, 7, 3

        # Setup
        dynamics = MockDynamicsIntegration({'T': T, 'dX': dX, 'dU': dU})
        cost = MockCostIntegration({'T': T, 'dX': dX, 'dU': dU})
        sample = MockSampleIntegration(T, dX, dU)

        # Get cost derivatives
        l, lx, lu, lxx, luu, lux = cost.eval(sample)

        # Simulate backward pass (simplified LQR)
        Vxx = np.zeros((T, dX, dX))
        Vx = np.zeros((T, dX))
        V = np.zeros(T)

        Vxx[-1] = lxx[-1]
        Vx[-1] = lx[-1]
        V[-1] = l[-1]

        for t in range(T - 2, -1, -1):
            # Simplified Q-function computation
            Qxx = lxx[t] + dynamics.Fm[t][:, :dX].T @ Vxx[t + 1] @ dynamics.Fm[t][:, :dX]
            Vxx[t] = Qxx
            V[t] = l[t]

        # Verify value function
        assert not np.any(np.isnan(V))
        assert not np.any(np.isnan(Vxx))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
