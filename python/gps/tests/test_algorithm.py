"""
Unit tests for Algorithm base class.

Tests cover:
- Algorithm initialization with hyperparameters
- IterationData management
- Cost and dynamics setup
- Condition handling
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.algorithm.config import ALG


class MockAlgorithm(Algorithm):
    """Mock implementation for testing Algorithm base class."""

    def iteration(self, sample_list):
        """Mock iteration method."""
        pass

    def iteration_cl(self, sample_lists_prot, sample_list):
        """Mock iteration_cl method."""
        pass


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, T=10, dU=3, dX=7, dO=10, dV=3):
        self.T = T
        self.dU = dU
        self.dX = dX
        self.dO = dO
        self.dV = dV
        self.x0 = np.zeros(dX)


class MockDynamics:
    """Mock dynamics for testing."""

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams


class MockTrajDist:
    """Mock trajectory distribution."""

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams


class MockTrajOpt:
    """Mock trajectory optimizer."""

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams


class MockCost:
    """Mock cost function."""

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams


class TestAlgorithmBase:
    """Unit tests for Algorithm base class."""

    @pytest.fixture
    def algorithm_hyperparams(self):
        """Basic hyperparameters for algorithm."""
        agent = MockAgent(T=10, dU=3, dX=7, dO=10, dV=3)

        return {
            'agent': agent,
            'conditions': 2,
            'T': 10,
            'fit_dynamics': True,
            'dynamics': {
                'type': MockDynamics,
            },
            'init_traj_distr': {
                'type': MockTrajDist,
            },
            'traj_opt': {
                'type': MockTrajOpt,
            },
            'cost': {
                'type': MockCost,
            },
            'kl_step': 1.0,
        }

    @pytest.mark.unit
    def test_algorithm_initialization(self, algorithm_hyperparams):
        """Test Algorithm initialization."""
        alg = MockAlgorithm(algorithm_hyperparams)

        assert alg.M == 2  # Number of conditions
        assert alg.T == 10  # Time steps
        assert alg.dU == 3  # Action dimension
        assert alg.dX == 7  # State dimension
        assert alg.dO == 10  # Observation dimension
        assert alg.dV == 3  # Adversary dimension
        assert alg.iteration_count == 0

    @pytest.mark.unit
    def test_algorithm_dimensions(self, algorithm_hyperparams):
        """Test that algorithm stores correct dimensions."""
        alg = MockAlgorithm(algorithm_hyperparams)

        assert alg._hyperparams['T'] == 10
        assert alg._hyperparams['dU'] == 3
        assert alg._hyperparams['dX'] == 7
        assert alg._hyperparams['dO'] == 10
        assert alg._hyperparams['dV'] == 3

    @pytest.mark.unit
    def test_algorithm_iteration_data(self, algorithm_hyperparams):
        """Test IterationData initialization."""
        alg = MockAlgorithm(algorithm_hyperparams)

        # Should have current and previous iteration data for each condition
        assert len(alg.cur) == alg.M
        assert len(alg.prev) == alg.M

        # Each should be IterationData instance
        for m in range(alg.M):
            assert isinstance(alg.cur[m], IterationData)
            assert isinstance(alg.prev[m], IterationData)

    @pytest.mark.unit
    def test_algorithm_traj_info(self, algorithm_hyperparams):
        """Test TrajectoryInfo initialization."""
        alg = MockAlgorithm(algorithm_hyperparams)

        for m in range(alg.M):
            assert isinstance(alg.cur[m].traj_info, TrajectoryInfo)
            assert alg.cur[m].traj_info.dynamics is not None

    @pytest.mark.unit
    def test_algorithm_cost_initialization(self, algorithm_hyperparams):
        """Test cost function initialization."""
        alg = MockAlgorithm(algorithm_hyperparams)

        # Should have one cost per condition
        assert len(alg.cost) == alg.M

        for cost in alg.cost:
            assert isinstance(cost, MockCost)

    @pytest.mark.unit
    def test_algorithm_cost_list(self, algorithm_hyperparams):
        """Test algorithm with list of costs."""
        # Create list of costs, one per condition
        algorithm_hyperparams['cost'] = [
            {'type': MockCost},
            {'type': MockCost},
        ]

        alg = MockAlgorithm(algorithm_hyperparams)

        assert len(alg.cost) == alg.M

    @pytest.mark.unit
    def test_algorithm_traj_opt(self, algorithm_hyperparams):
        """Test trajectory optimizer initialization."""
        alg = MockAlgorithm(algorithm_hyperparams)

        assert alg.traj_opt is not None
        assert isinstance(alg.traj_opt, MockTrajOpt)

    @pytest.mark.unit
    def test_algorithm_condition_indices(self, algorithm_hyperparams):
        """Test condition index handling."""
        alg = MockAlgorithm(algorithm_hyperparams)

        # Default: all conditions
        assert len(alg._cond_idx) == alg.M
        assert list(alg._cond_idx) == [0, 1]

    @pytest.mark.unit
    def test_algorithm_train_conditions(self, algorithm_hyperparams):
        """Test custom train conditions."""
        algorithm_hyperparams['train_conditions'] = [0, 2]
        alg = MockAlgorithm(algorithm_hyperparams)

        assert alg._cond_idx == [0, 2]
        assert alg.M == 2

    @pytest.mark.unit
    def test_algorithm_base_kl_step(self, algorithm_hyperparams):
        """Test KL step parameter."""
        alg = MockAlgorithm(algorithm_hyperparams)

        assert alg.base_kl_step == 1.0

    @pytest.mark.unit
    def test_algorithm_traj_distributions(self, algorithm_hyperparams):
        """Test trajectory distribution initialization."""
        alg = MockAlgorithm(algorithm_hyperparams)

        for m in range(alg.M):
            # Each condition should have three trajectory distributions
            assert alg.cur[m].traj_distr is not None
            assert alg.cur[m].traj_distr_adv is not None
            assert alg.cur[m].traj_distr_robust is not None


class TestAlgorithmAbstractMethods:
    """Test that abstract methods must be implemented."""

    @pytest.mark.unit
    def test_cannot_instantiate_algorithm_directly(self):
        """Test that Algorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Should fail because abstract methods not implemented
            Algorithm({})


class TestAlgorithmEdgeCases:
    """Test edge cases for Algorithm."""

    @pytest.mark.unit
    def test_algorithm_single_condition(self):
        """Test algorithm with single condition."""
        agent = MockAgent()
        hyperparams = {
            'agent': agent,
            'conditions': 1,
            'fit_dynamics': True,
            'dynamics': {'type': MockDynamics},
            'init_traj_distr': {'type': MockTrajDist},
            'traj_opt': {'type': MockTrajOpt},
            'cost': {'type': MockCost},
            'kl_step': 1.0,
        }

        alg = MockAlgorithm(hyperparams)

        assert alg.M == 1
        assert len(alg.cur) == 1
        assert len(alg.prev) == 1
        assert len(alg.cost) == 1

    @pytest.mark.unit
    def test_algorithm_many_conditions(self):
        """Test algorithm with many conditions."""
        agent = MockAgent()
        hyperparams = {
            'agent': agent,
            'conditions': 10,
            'fit_dynamics': True,
            'dynamics': {'type': MockDynamics},
            'init_traj_distr': {'type': MockTrajDist},
            'traj_opt': {'type': MockTrajOpt},
            'cost': {'type': MockCost},
            'kl_step': 1.0,
        }

        alg = MockAlgorithm(hyperparams)

        assert alg.M == 10
        assert len(alg.cur) == 10
        assert len(alg.cost) == 10

    @pytest.mark.unit
    def test_algorithm_without_dynamics(self):
        """Test algorithm without dynamics fitting."""
        agent = MockAgent()
        hyperparams = {
            'agent': agent,
            'conditions': 2,
            'fit_dynamics': False,
            'init_traj_distr': {'type': MockTrajDist},
            'traj_opt': {'type': MockTrajOpt},
            'cost': {'type': MockCost},
            'kl_step': 1.0,
        }

        alg = MockAlgorithm(hyperparams)

        # Should still initialize properly
        assert alg.M == 2

        # Dynamics should not be set
        for m in range(alg.M):
            assert alg.cur[m].traj_info.dynamics is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
