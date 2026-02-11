"""
Pytest configuration and fixtures for GPS tests.

This module provides shared fixtures and configuration for all GPS tests.
"""
import pytest
import numpy as np
from typing import Dict, Any


@pytest.fixture
def sample_dimensions():
    """Standard dimensions for test samples."""
    return {
        'T': 10,      # Time steps
        'dX': 7,      # State dimension
        'dU': 3,      # Action dimension
        'dO': 10,     # Observation dimension
        'dV': 3,      # Adversary action dimension (for robust control)
    }


@pytest.fixture
def random_state():
    """Fixed random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_trajectory(sample_dimensions, random_state):
    """Generate a sample trajectory with random data."""
    T = sample_dimensions['T']
    dX = sample_dimensions['dX']
    dU = sample_dimensions['dU']

    return {
        'X': random_state.randn(T, dX),  # States
        'U': random_state.randn(T, dU),  # Actions
        'obs': random_state.randn(T, dX + dU),  # Observations
    }


@pytest.fixture
def hyperparams_dict(sample_dimensions):
    """Sample hyperparameters dictionary for algorithm testing."""
    return {
        'T': sample_dimensions['T'],
        'dU': sample_dimensions['dU'],
        'dX': sample_dimensions['dX'],
        'dO': sample_dimensions['dO'],
        'dV': sample_dimensions['dV'],
        'conditions': 2,
        'initial_state_var': 1e-6,
    }


@pytest.fixture
def linear_gaussian_params(sample_dimensions, random_state):
    """Parameters for LinearGaussianPolicy."""
    T = sample_dimensions['T']
    dU = sample_dimensions['dU']
    dX = sample_dimensions['dX']

    return {
        'K': random_state.randn(T, dU, dX),  # Feedback gain
        'k': random_state.randn(T, dU),       # Bias term
        'pol_covar': np.tile(np.eye(dU), (T, 1, 1)),  # Covariance
        'chol_pol_covar': np.tile(np.eye(dU), (T, 1, 1)),  # Cholesky
        'inv_pol_covar': np.tile(np.eye(dU), (T, 1, 1)),   # Inverse
    }


@pytest.fixture
def cost_hyperparams():
    """Sample cost function hyperparameters."""
    return {
        'weights': [1.0, 0.5, 0.3],
        'mode': 'standard',
    }


# Test data for different scenarios
@pytest.fixture(params=['standard', 'robust', 'antagonist'])
def cost_mode(request):
    """Parametrized fixture for different cost modes."""
    return request.param


@pytest.fixture
def agent_config(sample_dimensions):
    """Mock agent configuration."""
    return {
        'T': sample_dimensions['T'],
        'dU': sample_dimensions['dU'],
        'dX': sample_dimensions['dX'],
        'dO': sample_dimensions['dO'],
        'dV': sample_dimensions['dV'],
        'x0': np.zeros(sample_dimensions['dX']),
        'sensor_dims': {
            'ACTION': sample_dimensions['dU'],
        },
        'state_include': [],
        'obs_include': [],
    }


# Performance testing fixtures
@pytest.fixture
def benchmark_iterations():
    """Number of iterations for benchmark tests."""
    return 100


# Skip markers based on environment
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: Tests that require GPU acceleration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Auto-mark slow tests
        if "benchmark" in item.nodeid or "stress" in item.nodeid:
            item.add_marker(pytest.mark.slow)
