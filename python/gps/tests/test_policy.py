"""
Unit tests for Policy classes.

Tests cover:
- LinearGaussianPolicy initialization and action computation
- LinearGaussianPolicyRobust dual-controller behavior
- Policy act() method with and without noise
- Parameter validation
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gps.algorithm.policy.lin_gauss_policy import (
    LinearGaussianPolicy,
    LinearGaussianPolicyRobust
)


class TestLinearGaussianPolicy:
    """Unit tests for LinearGaussianPolicy."""

    @pytest.mark.unit
    def test_policy_initialization(self, linear_gaussian_params):
        """Test LinearGaussianPolicy initialization."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        assert policy.T == linear_gaussian_params['K'].shape[0]
        assert policy.dU == linear_gaussian_params['K'].shape[1]
        assert policy.dX == linear_gaussian_params['K'].shape[2]

    @pytest.mark.unit
    def test_policy_dimensions(self, linear_gaussian_params, sample_dimensions):
        """Test policy has correct dimensions."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        assert policy.T == sample_dimensions['T']
        assert policy.dU == sample_dimensions['dU']
        assert policy.dX == sample_dimensions['dX']

    @pytest.mark.unit
    def test_policy_act_without_noise(self, linear_gaussian_params, random_state):
        """Test policy action computation without noise."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        # State vector
        x = random_state.randn(policy.dX)
        obs = random_state.randn(policy.dX)  # Same as state for simple case
        t = 5
        noise = np.zeros(policy.dU)  # Zero noise

        # Compute action
        u = policy.act(x, obs, t, noise)

        # Verify output shape
        assert u.shape == (policy.dU,)

        # Verify computation: u = K[t] @ x + k[t]
        expected_u = policy.K[t] @ x + policy.k[t]
        np.testing.assert_array_almost_equal(u, expected_u, decimal=10)

    @pytest.mark.unit
    def test_policy_act_with_noise(self, linear_gaussian_params, random_state):
        """Test policy action computation with noise."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        x = random_state.randn(policy.dX)
        obs = random_state.randn(policy.dX)
        t = 3
        noise = random_state.randn(policy.dU)

        # Compute action with noise
        u = policy.act(x, obs, t, noise)

        # Verify shape
        assert u.shape == (policy.dU,)

        # Verify computation includes noise term
        expected_u = policy.K[t] @ x + policy.k[t] + policy.chol_pol_covar[t].T @ noise
        np.testing.assert_array_almost_equal(u, expected_u, decimal=10)

    @pytest.mark.unit
    def test_policy_fold_k(self, linear_gaussian_params, random_state):
        """Test fold_k method for folding noise into bias."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        noise = random_state.randn(policy.T, policy.dU)

        # Fold noise into k
        k_folded = policy.fold_k(noise)

        # Verify shape
        assert k_folded.shape == (policy.T, policy.dU)

        # Verify at each timestep
        for t in range(policy.T):
            expected = policy.chol_pol_covar[t].T @ noise[t] + policy.k[t]
            np.testing.assert_array_almost_equal(k_folded[t], expected, decimal=10)

    @pytest.mark.unit
    def test_policy_nans_like(self, linear_gaussian_params):
        """Test nans_like creates policy with same dimensions."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        # Create NaN policy
        nan_policy = policy.nans_like()

        # Verify dimensions match
        assert nan_policy.T == policy.T
        assert nan_policy.dU == policy.dU
        assert nan_policy.dX == policy.dX

        # Verify all values are NaN
        assert np.all(np.isnan(nan_policy.K))
        assert np.all(np.isnan(nan_policy.k))
        assert np.all(np.isnan(nan_policy.pol_covar))

    @pytest.mark.unit
    def test_policy_parameter_shapes(self, linear_gaussian_params):
        """Test that all policy parameters have correct shapes."""
        policy = LinearGaussianPolicy(**linear_gaussian_params)

        assert policy.K.shape == (policy.T, policy.dU, policy.dX)
        assert policy.k.shape == (policy.T, policy.dU)
        assert policy.pol_covar.shape == (policy.T, policy.dU, policy.dU)
        assert policy.chol_pol_covar.shape == (policy.T, policy.dU, policy.dU)
        assert policy.inv_pol_covar.shape == (policy.T, policy.dU, policy.dU)


class TestLinearGaussianPolicyRobust:
    """Unit tests for LinearGaussianPolicyRobust (dual controller)."""

    @pytest.fixture
    def robust_params(self, sample_dimensions, random_state):
        """Parameters for robust policy."""
        T = sample_dimensions['T']
        dU = sample_dimensions['dU']
        dV = sample_dimensions['dV']
        dX = sample_dimensions['dX']

        return {
            'Gu': random_state.randn(T, dU, dX),
            'gu': random_state.randn(T, dU),
            'pol_covar_u': np.tile(np.eye(dU), (T, 1, 1)),
            'chol_pol_covar_u': np.tile(np.eye(dU), (T, 1, 1)),
            'inv_pol_covar_u': np.tile(np.eye(dU), (T, 1, 1)),
            'Gv': random_state.randn(T, dV, dX),
            'gv': random_state.randn(T, dV),
            'pol_covar_v': np.tile(np.eye(dV), (T, 1, 1)),
            'chol_pol_covar_v': np.tile(np.eye(dV), (T, 1, 1)),
            'inv_pol_covar_v': np.tile(np.eye(dV), (T, 1, 1)),
        }

    @pytest.mark.unit
    def test_robust_policy_initialization(self, robust_params):
        """Test robust policy initialization."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        assert policy.T == robust_params['Gu'].shape[0]
        assert policy.dU == robust_params['Gu'].shape[1]
        assert policy.dV == robust_params['Gv'].shape[1]
        assert policy.dX == robust_params['Gu'].shape[2]

    @pytest.mark.unit
    def test_robust_policy_act_raises(self, robust_params, random_state):
        """Test that act() raises NotImplementedError."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        x = random_state.randn(policy.dX)
        obs = random_state.randn(policy.dX)
        t = 0
        noise = None

        with pytest.raises(NotImplementedError):
            policy.act(x, obs, t, noise)

    @pytest.mark.unit
    def test_robust_policy_act_u(self, robust_params, random_state):
        """Test protagonist action computation."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        x = random_state.randn(policy.dX)
        obs = random_state.randn(policy.dX)
        t = 4
        noise = np.zeros(policy.dU)

        # Compute protagonist action
        u = policy.act_u(x, obs, t, noise)

        # Verify
        assert u.shape == (policy.dU,)
        expected_u = policy.Gu[t] @ x + policy.gu[t]
        np.testing.assert_array_almost_equal(u, expected_u, decimal=10)

    @pytest.mark.unit
    def test_robust_policy_act_v(self, robust_params, random_state):
        """Test adversary action computation."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        x = random_state.randn(policy.dX)
        obs = random_state.randn(policy.dX)
        t = 6
        noise = np.zeros(policy.dV)

        # Compute adversary action
        v = policy.act_v(x, obs, t, noise)

        # Verify
        assert v.shape == (policy.dV,)
        expected_v = policy.Gv[t] @ x + policy.gv[t]
        np.testing.assert_array_almost_equal(v, expected_v, decimal=10)

    @pytest.mark.unit
    def test_robust_policy_fold_gu(self, robust_params, random_state):
        """Test fold_gu for protagonist."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        noise = random_state.randn(policy.T, policy.dU)
        gu_folded = policy.fold_gu(noise)

        assert gu_folded.shape == (policy.T, policy.dU)

        for t in range(policy.T):
            expected = policy.chol_pol_covar_u[t].T @ noise[t] + policy.gu[t]
            np.testing.assert_array_almost_equal(gu_folded[t], expected, decimal=10)

    @pytest.mark.unit
    def test_robust_policy_fold_gv(self, robust_params, random_state):
        """Test fold_gv for adversary."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        noise = random_state.randn(policy.T, policy.dV)
        gv_folded = policy.fold_gv(noise)

        assert gv_folded.shape == (policy.T, policy.dV)

        for t in range(policy.T):
            expected = policy.chol_pol_covar_v[t].T @ noise[t] + policy.gv[t]
            np.testing.assert_array_almost_equal(gv_folded[t], expected, decimal=10)

    @pytest.mark.unit
    def test_robust_policy_nans_like(self, robust_params):
        """Test nans_like for robust policy."""
        policy = LinearGaussianPolicyRobust(**robust_params)

        nan_policy = policy.nans_like()

        # Verify dimensions
        assert nan_policy.T == policy.T
        assert nan_policy.dU == policy.dU
        assert nan_policy.dV == policy.dV
        assert nan_policy.dX == policy.dX

        # Verify NaNs
        assert np.all(np.isnan(nan_policy.Gu))
        assert np.all(np.isnan(nan_policy.gu))
        assert np.all(np.isnan(nan_policy.Gv))
        assert np.all(np.isnan(nan_policy.gv))


class TestPolicyEdgeCases:
    """Test edge cases for policies."""

    @pytest.mark.unit
    def test_policy_with_identity_K(self, linear_gaussian_params):
        """Test policy with identity feedback gain."""
        params = linear_gaussian_params.copy()
        T, dU, dX = params['K'].shape

        # Set K to identity-like (square case only)
        if dU == dX:
            params['K'] = np.tile(np.eye(dU), (T, 1, 1))

            policy = LinearGaussianPolicy(**params)
            x = np.ones(dX)
            t = 0
            noise = np.zeros(dU)

            u = policy.act(x, np.ones(dX), t, noise)

            # u should be approximately x + k[0]
            expected = x + params['k'][0]
            np.testing.assert_array_almost_equal(u, expected, decimal=10)

    @pytest.mark.unit
    def test_policy_with_zero_noise_covariance(self, linear_gaussian_params):
        """Test policy with zero noise."""
        params = linear_gaussian_params.copy()
        params['chol_pol_covar'] = np.zeros_like(params['chol_pol_covar'])

        policy = LinearGaussianPolicy(**params)

        # Even with large noise input, output should be deterministic
        x = np.ones(policy.dX)
        noise = 100 * np.ones(policy.dU)
        t = 0

        u = policy.act(x, x, t, noise)

        # Should only depend on K and k
        expected = policy.K[t] @ x + policy.k[t]
        np.testing.assert_array_almost_equal(u, expected, decimal=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
