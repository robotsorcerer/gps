"""
Top-level pytest configuration for the iDG/GPS test suite.

Adds the gps python package to sys.path and defines shared fixtures
and markers used across all test phases.
"""
import os
import sys
import logging

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: make `import gps.*` work regardless of install state
# ---------------------------------------------------------------------------
_GPS_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GPS_PYTHON not in sys.path:
    sys.path.insert(0, _GPS_PYTHON)

# ---------------------------------------------------------------------------
# Logging: surface GPS logger output during tests (visible with -s)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sensor_dims():
    """Canonical sensor dimension map used by most tests."""
    from gps.proto.gps_pb2 import (
        JOINT_ANGLES, JOINT_VELOCITIES,
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION,
    )
    return {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 6,
        END_EFFECTOR_POINT_VELOCITIES: 6,
        ACTION: 7,
    }


@pytest.fixture(scope="session")
def small_dims():
    """Tiny (dO, dU) dimensions for fast unit tests."""
    return {"dO": 14, "dU": 7}


@pytest.fixture
def rng():
    """Seeded NumPy RNG for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def random_obs(rng, small_dims):
    """Random observation array [N=5, T=10, dO]."""
    N, T = 5, 10
    dO = small_dims["dO"]
    return rng.standard_normal((N, T, dO))


@pytest.fixture
def random_tgt_mu(rng, small_dims):
    """Random target action mean [N=5, T=10, dU]."""
    N, T = 5, 10
    dU = small_dims["dU"]
    return rng.standard_normal((N, T, dU))


@pytest.fixture
def random_tgt_prc(rng, small_dims):
    """Random (positive-semi-definite-ish) precision matrices [N=5, T=10, dU, dU]."""
    N, T = 5, 10
    dU = small_dims["dU"]
    raw = rng.standard_normal((N, T, dU, dU))
    # Make PSD: A @ A^T + eps*I
    return raw @ raw.transpose(0, 1, 3, 2) + np.eye(dU) * 0.1


@pytest.fixture
def random_tgt_wt(rng, small_dims):
    """Random sample weights [N=5, T=10], non-negative."""
    N, T = 5, 10
    return np.abs(rng.standard_normal((N, T))) + 0.01
