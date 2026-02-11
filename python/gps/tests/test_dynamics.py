"""
Unit tests for Dynamics base class.

Tests cover:
- Dynamics initialization
- Fitted dynamics parameters (Fm, fv, dyn_covar)
- Copy method
- Abstract method interface
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps.algorithm.dynamics.dynamics import Dynamics


class MockDynamics(Dynamics):
    """Mock implementation for testing Dynamics base class."""

    def update_prior(self, X, U):
        """Mock update_prior method."""
        pass

    def get_prior(self):
        """Mock get_prior method."""
        return None

    def fit(self, sample_list):
        """Mock fit method."""
        # Set some dummy dynamics
        T = 10
        dX = 7
        dU = 3
        self.Fm = np.random.randn(dX, dX + dU)
        self.fv = np.random.randn(dX)
        self.dyn_covar = np.eye(dX)


class TestDynamicsBase:
    """Unit tests for Dynamics base class."""

    @pytest.fixture
    def dynamics_hyperparams(self):
        """Sample hyperparameters for dynamics."""
        return {
            'T': 10,
            'dX': 7,
            'dU': 3,
            'regularization': 1e-6,
        }

    @pytest.mark.unit
    def test_dynamics_initialization(self, dynamics_hyperparams):
        """Test Dynamics initialization."""
        dyn = MockDynamics(dynamics_hyperparams)

        assert dyn._hyperparams == dynamics_hyperparams
        assert dyn.Fm is not None
        assert dyn.fv is not None
        assert dyn.dyn_covar is not None

    @pytest.mark.unit
    def test_dynamics_initial_parameters_nan(self, dynamics_hyperparams):
        """Test that initial dynamics parameters are NaN."""
        dyn = MockDynamics(dynamics_hyperparams)

        # Initially should be NaN (not fitted yet)
        assert np.isnan(dyn.Fm).any()
        assert np.isnan(dyn.fv).any()
        assert np.isnan(dyn.dyn_covar).any()

    @pytest.mark.unit
    def test_dynamics_copy(self, dynamics_hyperparams, random_state):
        """Test copy method."""
        dyn = MockDynamics(dynamics_hyperparams)

        # Set some dynamics
        dX = 7
        dU = 3
        dyn.Fm = random_state.randn(dX, dX + dU)
        dyn.fv = random_state.randn(dX)
        dyn.dyn_covar = np.eye(dX)

        # Copy dynamics
        dyn_copy = dyn.copy()

        # Should have same values
        np.testing.assert_array_equal(dyn_copy.Fm, dyn.Fm)
        np.testing.assert_array_equal(dyn_copy.fv, dyn.fv)
        np.testing.assert_array_equal(dyn_copy.dyn_covar, dyn.dyn_covar)

        # But should be different objects (deep copy)
        assert dyn_copy.Fm is not dyn.Fm
        assert dyn_copy.fv is not dyn.fv
        assert dyn_copy.dyn_covar is not dyn.dyn_covar

    @pytest.mark.unit
    def test_dynamics_copy_independence(self, dynamics_hyperparams, random_state):
        """Test that copied dynamics are independent."""
        dyn = MockDynamics(dynamics_hyperparams)

        dX = 7
        dU = 3
        dyn.Fm = random_state.randn(dX, dX + dU)
        dyn.fv = random_state.randn(dX)
        dyn.dyn_covar = np.eye(dX)

        # Copy
        dyn_copy = dyn.copy()

        # Modify original
        dyn.Fm[0, 0] = 999.0
        dyn.fv[0] = 888.0
        dyn.dyn_covar[0, 0] = 777.0

        # Copy should remain unchanged
        assert dyn_copy.Fm[0, 0] != 999.0
        assert dyn_copy.fv[0] != 888.0
        assert dyn_copy.dyn_covar[0, 0] != 777.0

    @pytest.mark.unit
    def test_dynamics_fit_updates_parameters(self, dynamics_hyperparams):
        """Test that fit updates dynamics parameters."""
        dyn = MockDynamics(dynamics_hyperparams)

        # Initially NaN
        assert np.isnan(dyn.Fm).any()

        # Fit with mock data
        dyn.fit(None)

        # Should no longer be NaN
        assert not np.isnan(dyn.Fm).any()
        assert not np.isnan(dyn.fv).any()
        assert not np.isnan(dyn.dyn_covar).any()


class TestDynamicsAbstractMethods:
    """Test abstract method interface."""

    @pytest.mark.unit
    def test_cannot_instantiate_dynamics_directly(self):
        """Test that Dynamics cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Dynamics({})

    @pytest.mark.unit
    def test_abstract_methods_must_be_implemented(self):
        """Test that all abstract methods must be implemented."""

        # Missing update_prior
        class IncompleteDynamics1(Dynamics):
            def get_prior(self):
                return None

            def fit(self, sample_list):
                pass

        with pytest.raises(TypeError):
            IncompleteDynamics1({})

        # Missing get_prior
        class IncompleteDynamics2(Dynamics):
            def update_prior(self, X, U):
                pass

            def fit(self, sample_list):
                pass

        with pytest.raises(TypeError):
            IncompleteDynamics2({})

        # Missing fit
        class IncompleteDynamics3(Dynamics):
            def update_prior(self, X, U):
                pass

            def get_prior(self):
                return None

        with pytest.raises(TypeError):
            IncompleteDynamics3({})


class TestDynamicsEdgeCases:
    """Test edge cases for dynamics."""

    @pytest.mark.unit
    def test_dynamics_with_zero_covariance(self, dynamics_hyperparams, random_state):
        """Test dynamics with zero covariance (deterministic)."""
        dyn = MockDynamics(dynamics_hyperparams)

        dX = 7
        dU = 3
        dyn.Fm = random_state.randn(dX, dX + dU)
        dyn.fv = random_state.randn(dX)
        dyn.dyn_covar = np.zeros((dX, dX))

        # Copy should work even with zero covariance
        dyn_copy = dyn.copy()
        np.testing.assert_array_equal(dyn_copy.dyn_covar, np.zeros((dX, dX)))

    @pytest.mark.unit
    def test_dynamics_copy_preserves_type(self, dynamics_hyperparams):
        """Test that copy preserves dynamics type."""
        dyn = MockDynamics(dynamics_hyperparams)

        dyn_copy = dyn.copy()

        assert isinstance(dyn_copy, MockDynamics)
        assert type(dyn_copy) == type(dyn)

    @pytest.mark.unit
    def test_dynamics_large_dimensions(self):
        """Test dynamics with large state/action dimensions."""
        hyperparams = {
            'T': 100,
            'dX': 50,
            'dU': 20,
        }

        dyn = MockDynamics(hyperparams)
        assert dyn._hyperparams['dX'] == 50
        assert dyn._hyperparams['dU'] == 20


class TestDynamicsParameterShapes:
    """Test dynamics parameter shapes."""

    @pytest.mark.unit
    def test_fm_shape(self, dynamics_hyperparams, random_state):
        """Test Fm has correct shape."""
        dyn = MockDynamics(dynamics_hyperparams)

        dX = dynamics_hyperparams['dX']
        dU = dynamics_hyperparams['dU']

        # Set Fm
        dyn.Fm = random_state.randn(dX, dX + dU)

        # Verify shape: dX x (dX + dU)
        assert dyn.Fm.shape == (dX, dX + dU)

    @pytest.mark.unit
    def test_fv_shape(self, dynamics_hyperparams, random_state):
        """Test fv has correct shape."""
        dyn = MockDynamics(dynamics_hyperparams)

        dX = dynamics_hyperparams['dX']

        # Set fv
        dyn.fv = random_state.randn(dX)

        # Verify shape: (dX,)
        assert dyn.fv.shape == (dX,)

    @pytest.mark.unit
    def test_dyn_covar_shape(self, dynamics_hyperparams):
        """Test dyn_covar has correct shape."""
        dyn = MockDynamics(dynamics_hyperparams)

        dX = dynamics_hyperparams['dX']

        # Set covariance
        dyn.dyn_covar = np.eye(dX)

        # Verify shape: dX x dX
        assert dyn.dyn_covar.shape == (dX, dX)

    @pytest.mark.unit
    def test_covariance_positive_semidefinite(self, dynamics_hyperparams):
        """Test that covariance is positive semidefinite."""
        dyn = MockDynamics(dynamics_hyperparams)

        dX = dynamics_hyperparams['dX']

        # Set valid covariance
        dyn.dyn_covar = np.eye(dX) * 0.1

        # Eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvals(dyn.dyn_covar)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
