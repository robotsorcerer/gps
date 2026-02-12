"""
Tests for the observation normalisation formula used by PyTorchController.
Ports all 13 C++ GTest cases from test_obs_normalization.cpp.

Formula: obs_scaled[i] = obs[i] * scale_diag[i] + bias[i]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest


def apply_obs_norm(
    obs: np.ndarray,
    scale_diag: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Mirrors C++ apply_obs_norm — pointwise affine transform."""
    return obs * scale_diag + bias


class TestObsNormBasic:
    def test_identity_transform(self):
        obs   = np.linspace(-2.0, 5.0, 8)
        scale = np.ones(8)
        bias  = np.zeros(8)
        result = apply_obs_norm(obs, scale, bias)
        np.testing.assert_allclose(result, obs, atol=1e-12)

    def test_zero_obs_returns_bias(self):
        obs  = np.zeros(5)
        sc   = np.random.rand(5)
        bias = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        result = apply_obs_norm(obs, sc, bias)
        np.testing.assert_allclose(result, bias, atol=1e-12)

    def test_zero_scale_maps_to_bias(self):
        obs  = np.full(4, 999.0)
        sc   = np.zeros(4)
        bias = np.array([0.1, 0.2, 0.3, 0.4])
        result = apply_obs_norm(obs, sc, bias)
        np.testing.assert_allclose(result, bias, atol=1e-12)

    def test_known_values(self):
        # obs=[1,2,3], scale=[2,0.5,-1], bias=[0.1,0.2,0.3]
        # expected=[2.1, 1.2, -2.7]
        obs  = np.array([1.0, 2.0, 3.0])
        sc   = np.array([2.0, 0.5, -1.0])
        bias = np.array([0.1, 0.2, 0.3])
        expected = np.array([2.1, 1.2, -2.7])
        result = apply_obs_norm(obs, sc, bias)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_negative_bias(self):
        obs  = np.zeros(2)
        sc   = np.ones(2)
        bias = np.array([-5.0, -0.5])
        result = apply_obs_norm(obs, sc, bias)
        assert result[0] == pytest.approx(-5.0)
        assert result[1] == pytest.approx(-0.5)


class TestObsNormDimensions:
    @pytest.mark.parametrize("dO", [1, 7, 14, 64])
    def test_output_dimension_matches_input(self, dO):
        obs   = np.random.rand(dO)
        scale = np.ones(dO)
        bias  = np.zeros(dO)
        assert apply_obs_norm(obs, scale, bias).shape == (dO,)

    def test_large_vector_identity(self):
        dO  = 256
        obs = np.random.rand(dO)
        result = apply_obs_norm(obs, np.ones(dO), np.zeros(dO))
        np.testing.assert_allclose(result, obs, atol=1e-12)


class TestObsNormNumericalEdgeCases:
    def test_nan_obs_propagates(self):
        obs  = np.array([1.0, float("nan"), 3.0])
        sc   = np.ones(3)
        bias = np.zeros(3)
        result = apply_obs_norm(obs, sc, bias)
        assert not math.isnan(result[0])
        assert     math.isnan(result[1])
        assert not math.isnan(result[2])

    def test_inf_obs_propagates(self):
        obs  = np.array([float("inf"), 1.0])
        sc   = np.ones(2)
        bias = np.zeros(2)
        result = apply_obs_norm(obs, sc, bias)
        assert math.isinf(result[0])
        assert not math.isinf(result[1])

    def test_nan_scale_propagates(self):
        obs  = np.array([1.0, 2.0])
        sc   = np.array([float("nan"), 1.0])
        bias = np.zeros(2)
        result = apply_obs_norm(obs, sc, bias)
        assert     math.isnan(result[0])
        assert not math.isnan(result[1])

    def test_inf_scale_finite_obs(self):
        obs  = np.array([1.0])
        sc   = np.array([float("inf")])
        bias = np.zeros(1)
        result = apply_obs_norm(obs, sc, bias)
        assert math.isinf(result[0])


class TestObsNormLinearity:
    def test_superposition_bias_zero(self):
        # With bias=0 the transform is linear: f(k*obs) = k*f(obs)
        obs = np.array([1.0, -1.0, 0.5, 2.0])
        sc  = np.array([2.0,  3.0, 1.0, 0.5])
        b   = np.zeros(4)
        k = 3.0
        f_obs  = apply_obs_norm(obs,     sc, b)
        f_kobs = apply_obs_norm(obs * k, sc, b)
        np.testing.assert_allclose(f_kobs, f_obs * k, atol=1e-9)
