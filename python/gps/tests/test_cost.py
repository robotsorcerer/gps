"""
Unit tests for Cost classes.

Tests cover:
- Cost base class interface
- CostSum composition of multiple costs
- Multi-mode cost evaluation (standard/robust/antagonist)
- Cost derivatives and Hessians
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_sum import CostSum


class MockCost(Cost):
    """Mock cost implementation for testing."""

    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.eval_count = 0

    def eval(self, sample, **kwargs):
        """Mock evaluation returning simple cost."""
        self.eval_count += 1

        # Return dummy cost and derivatives
        T = 10  # Assume 10 timesteps
        dX = 7  # State dim
        dU = 3  # Action dim

        # Cost values (T,)
        l = np.ones(T)

        # First derivatives
        lx = np.ones((T, dX))
        lu = np.ones((T, dU))

        # Second derivatives (Hessians)
        lxx = np.tile(np.eye(dX), (T, 1, 1))
        luu = np.tile(np.eye(dU), (T, 1, 1))
        lux = np.zeros((T, dU, dX))

        if self._hyperparams.get('mode') == 'robust':
            # Robust mode returns 11-tuple
            dV = 3
            lv = np.ones((T, dV))
            lvv = np.tile(np.eye(dV), (T, 1, 1))
            luv = np.zeros((T, dU, dV))
            lvx = np.zeros((T, dV, dX))
            dist = np.zeros(dU)
            return l, lx, lu, lv, lxx, luu, lvv, luv, lux, lvx, dist
        else:
            # Standard mode returns 6-tuple
            return l, lx, lu, lxx, luu, lux


class TestCostBase:
    """Unit tests for Cost base class."""

    @pytest.mark.unit
    def test_cost_initialization(self, cost_hyperparams):
        """Test Cost base class initialization."""
        cost = MockCost(cost_hyperparams)
        assert cost._hyperparams == cost_hyperparams

    @pytest.mark.unit
    def test_cost_abstract_eval(self):
        """Test that Cost.eval() is abstract."""
        # Cannot instantiate Cost directly
        with pytest.raises(TypeError):
            Cost({})


class TestCostSum:
    """Unit tests for CostSum class."""

    @pytest.fixture
    def cost_sum_hyperparams(self):
        """Hyperparameters for CostSum."""
        # Create mock cost specifications
        cost1_params = {'mode': 'standard', 'type': MockCost}
        cost2_params = {'mode': 'standard', 'type': MockCost}
        cost3_params = {'mode': 'standard', 'type': MockCost}

        return {
            'costs': [cost1_params, cost2_params, cost3_params],
            'weights': [1.0, 0.5, 0.25],
        }

    @pytest.mark.unit
    def test_costsum_initialization(self, cost_sum_hyperparams):
        """Test CostSum initialization."""
        cost_sum = CostSum(cost_sum_hyperparams)

        assert len(cost_sum._costs) == 3
        assert len(cost_sum._weights) == 3
        assert cost_sum._weights == [1.0, 0.5, 0.25]

    @pytest.mark.unit
    def test_costsum_eval_standard_mode(self, cost_sum_hyperparams):
        """Test CostSum evaluation in standard mode."""
        cost_sum = CostSum(cost_sum_hyperparams)

        # Mock sample
        sample = None

        # Evaluate
        result = cost_sum.eval(sample)

        # Should return 6-tuple for standard mode
        assert len(result) == 6

        l, lx, lu, lxx, luu, lux = result

        # Verify shapes (assuming T=10, dX=7, dU=3)
        assert l.shape == (10,)
        assert lx.shape == (10, 7)
        assert lu.shape == (10, 3)
        assert lxx.shape == (10, 7, 7)
        assert luu.shape == (10, 3, 3)
        assert lux.shape == (10, 3, 7)

    @pytest.mark.unit
    def test_costsum_weighted_combination(self, cost_sum_hyperparams):
        """Test that costs are properly weighted."""
        cost_sum = CostSum(cost_sum_hyperparams)

        sample = None
        result = cost_sum.eval(sample)

        l = result[0]

        # Each mock cost returns ones, so weighted sum should be:
        # 1.0 * 1.0 + 0.5 * 1.0 + 0.25 * 1.0 = 1.75
        expected_l = 1.75 * np.ones(10)
        np.testing.assert_array_almost_equal(l, expected_l, decimal=10)

    @pytest.mark.unit
    def test_costsum_eval_calls_subcosts(self, cost_sum_hyperparams):
        """Test that CostSum calls all sub-costs."""
        cost_sum = CostSum(cost_sum_hyperparams)

        # Reset eval counters
        for cost in cost_sum._costs:
            cost.eval_count = 0

        # Evaluate
        cost_sum.eval(None)

        # Verify all costs were called
        for cost in cost_sum._costs:
            assert cost.eval_count == 1

    @pytest.mark.unit
    def test_costsum_mode_propagation(self):
        """Test that mode is propagated to CostSum."""
        cost_params = {'mode': 'robust', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params],
            'weights': [1.0],
        }

        cost_sum = CostSum(hyperparams)

        assert cost_sum.mode == 'robust'

    @pytest.mark.unit
    def test_costsum_robust_mode(self):
        """Test CostSum in robust mode."""
        cost_params = {'mode': 'robust', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params],
            'weights': [1.0],
        }

        cost_sum = CostSum(hyperparams)
        result = cost_sum.eval(None, sample_adv=None)

        # Should return 11-tuple for robust mode
        assert len(result) == 11

        l, lx, lu, lv, lxx, luu, lvv, luv, lux, lvx, dist = result

        # Verify shapes
        assert l.shape == (10,)
        assert lx.shape == (10, 7)
        assert lu.shape == (10, 3)
        assert lv.shape == (10, 3)
        assert dist.shape == (3,)

    @pytest.mark.unit
    def test_costsum_single_cost(self):
        """Test CostSum with single cost."""
        cost_params = {'mode': 'standard', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params],
            'weights': [2.0],
        }

        cost_sum = CostSum(hyperparams)
        result = cost_sum.eval(None)

        l = result[0]

        # Single cost with weight 2.0
        expected = 2.0 * np.ones(10)
        np.testing.assert_array_almost_equal(l, expected, decimal=10)


class TestCostDerivatives:
    """Test cost derivative computations."""

    @pytest.mark.unit
    def test_cost_derivatives_shape_consistency(self):
        """Test that all derivatives have consistent shapes."""
        cost_params = {'mode': 'standard', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params],
            'weights': [1.0],
        }

        cost_sum = CostSum(hyperparams)
        l, lx, lu, lxx, luu, lux = cost_sum.eval(None)

        T = l.shape[0]
        dX = lx.shape[1]
        dU = lu.shape[1]

        # Verify all shapes are consistent
        assert lx.shape == (T, dX)
        assert lu.shape == (T, dU)
        assert lxx.shape == (T, dX, dX)
        assert luu.shape == (T, dU, dU)
        assert lux.shape == (T, dU, dX)

    @pytest.mark.unit
    def test_hessian_symmetry(self):
        """Test that Hessian matrices are symmetric."""
        cost_params = {'mode': 'standard', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params],
            'weights': [1.0],
        }

        cost_sum = CostSum(hyperparams)
        l, lx, lu, lxx, luu, lux = cost_sum.eval(None)

        # lxx and luu should be symmetric
        for t in range(lxx.shape[0]):
            np.testing.assert_array_almost_equal(
                lxx[t], lxx[t].T, decimal=10
            )
            np.testing.assert_array_almost_equal(
                luu[t], luu[t].T, decimal=10
            )


class TestCostEdgeCases:
    """Test edge cases for cost evaluation."""

    @pytest.mark.unit
    def test_costsum_zero_weights(self):
        """Test CostSum with zero weights."""
        cost_params = {'mode': 'standard', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params, cost_params],
            'weights': [0.0, 0.0],
        }

        cost_sum = CostSum(hyperparams)
        l = cost_sum.eval(None)[0]

        # All zeros
        expected = np.zeros(10)
        np.testing.assert_array_almost_equal(l, expected, decimal=10)

    @pytest.mark.unit
    def test_costsum_negative_weights(self):
        """Test CostSum with negative weights (reward)."""
        cost_params = {'mode': 'standard', 'type': MockCost}
        hyperparams = {
            'costs': [cost_params],
            'weights': [-1.0],
        }

        cost_sum = CostSum(hyperparams)
        l = cost_sum.eval(None)[0]

        # Should be negative (reward)
        assert np.all(l < 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
