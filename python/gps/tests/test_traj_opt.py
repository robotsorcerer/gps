"""
Unit tests for TrajOpt base class.

Tests cover:
- TrajOpt initialization
- Abstract method interface
- Hyperparameter handling
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps.algorithm.traj_opt.traj_opt import TrajOpt


class MockTrajOpt(TrajOpt):
    """Mock implementation for testing TrajOpt base class."""

    def update(self):
        """Mock update method."""
        return None


class TestTrajOptBase:
    """Unit tests for TrajOpt base class."""

    @pytest.fixture
    def traj_opt_hyperparams(self):
        """Sample hyperparameters for trajectory optimizer."""
        return {
            'T': 10,
            'dX': 7,
            'dU': 3,
            'max_iterations': 100,
            'step_size': 0.01,
            'min_eta': 1e-8,
        }

    @pytest.mark.unit
    def test_traj_opt_initialization(self, traj_opt_hyperparams):
        """Test TrajOpt initialization."""
        traj_opt = MockTrajOpt(traj_opt_hyperparams)

        assert traj_opt._hyperparams == traj_opt_hyperparams

    @pytest.mark.unit
    def test_traj_opt_stores_hyperparams(self, traj_opt_hyperparams):
        """Test that hyperparameters are stored correctly."""
        traj_opt = MockTrajOpt(traj_opt_hyperparams)

        assert traj_opt._hyperparams['T'] == 10
        assert traj_opt._hyperparams['dX'] == 7
        assert traj_opt._hyperparams['dU'] == 3
        assert traj_opt._hyperparams['max_iterations'] == 100

    @pytest.mark.unit
    def test_traj_opt_update_callable(self, traj_opt_hyperparams):
        """Test that update method is callable."""
        traj_opt = MockTrajOpt(traj_opt_hyperparams)

        # Should be callable without error
        result = traj_opt.update()
        assert result is None


class TestTrajOptAbstractMethods:
    """Test abstract method interface."""

    @pytest.mark.unit
    def test_cannot_instantiate_traj_opt_directly(self):
        """Test that TrajOpt cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrajOpt({})

    @pytest.mark.unit
    def test_must_implement_update(self):
        """Test that update method must be implemented."""

        class IncompleteTrajOpt(TrajOpt):
            # Missing update method
            pass

        with pytest.raises(TypeError):
            IncompleteTrajOpt({})


class TestTrajOptEdgeCases:
    """Test edge cases for trajectory optimization."""

    @pytest.mark.unit
    def test_traj_opt_empty_hyperparams(self):
        """Test TrajOpt with empty hyperparameters."""
        traj_opt = MockTrajOpt({})

        assert traj_opt._hyperparams == {}

    @pytest.mark.unit
    def test_traj_opt_minimal_hyperparams(self):
        """Test TrajOpt with minimal hyperparameters."""
        hyperparams = {
            'T': 5,
            'dX': 3,
            'dU': 2,
        }

        traj_opt = MockTrajOpt(hyperparams)

        assert traj_opt._hyperparams['T'] == 5
        assert traj_opt._hyperparams['dX'] == 3
        assert traj_opt._hyperparams['dU'] == 2

    @pytest.mark.unit
    def test_traj_opt_large_dimensions(self):
        """Test TrajOpt with large dimensions."""
        hyperparams = {
            'T': 1000,
            'dX': 100,
            'dU': 50,
        }

        traj_opt = MockTrajOpt(hyperparams)

        assert traj_opt._hyperparams['T'] == 1000
        assert traj_opt._hyperparams['dX'] == 100
        assert traj_opt._hyperparams['dU'] == 50


class TestTrajOptMultipleImplementations:
    """Test multiple TrajOpt implementations."""

    @pytest.mark.unit
    def test_different_traj_opt_types(self):
        """Test that different TrajOpt types can coexist."""

        class TrajOptType1(TrajOpt):
            def update(self):
                return "type1"

        class TrajOptType2(TrajOpt):
            def update(self):
                return "type2"

        opt1 = TrajOptType1({'method': 'lqr'})
        opt2 = TrajOptType2({'method': 'ilqg'})

        assert opt1.update() == "type1"
        assert opt2.update() == "type2"

    @pytest.mark.unit
    def test_traj_opt_with_custom_update(self):
        """Test TrajOpt with custom update logic."""

        class CustomTrajOpt(TrajOpt):
            def __init__(self, hyperparams):
                super().__init__(hyperparams)
                self.update_count = 0

            def update(self):
                self.update_count += 1
                return self.update_count

        opt = CustomTrajOpt({})

        assert opt.update() == 1
        assert opt.update() == 2
        assert opt.update() == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
