"""
Phase 0 smoke tests — verify the test infrastructure itself is functional.

These tests MUST pass before any Phase 1 fixes are applied.
They confirm:
  1. pytest marks are registered (no PytestUnknownMarkWarning)
  2. conftest fixtures are importable and return expected shapes
  3. The gps package root is on sys.path
  4. protobuf-generated gps_pb2 is importable (proto compilation check)
"""
import sys
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mark: all infrastructure tests are 'unit'
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_gps_python_on_path():
    """The `gps` package directory must be importable."""
    import importlib.util
    spec = importlib.util.find_spec("gps")
    assert spec is not None, (
        "Cannot find 'gps' package. "
        "Ensure python/ is on PYTHONPATH or conftest.py path injection works."
    )


@pytest.mark.unit
def test_proto_importable():
    """gps_pb2 protobuf stubs must be importable (requires proto compilation)."""
    try:
        from gps.proto import gps_pb2  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"gps_pb2 not importable: {exc}\n"
            "Run: cd gps_agent_pkg/proto && "
            "protoc --python_out=../../python/gps/proto gps.proto"
        )


@pytest.mark.unit
def test_pytest_marks_registered(request):
    """All custom marks must be registered in pytest.ini (no unknown-mark warnings)."""
    # Read registered marks from the ini config (pytest 7+ API)
    ini_marks = set(request.config.getini("markers"))
    # Each entry in markers ini is a string like "unit: Fast, isolated unit tests..."
    registered_names = {line.split(":")[0].strip() for line in ini_marks}
    known = {"unit", "integration", "load", "soak", "fault", "gpu", "ros"}
    missing = known - registered_names
    assert not missing, f"Marks not registered in pytest.ini: {missing}"


@pytest.mark.unit
def test_rng_fixture_is_seeded(rng):
    """rng fixture must produce reproducible results."""
    val1 = rng.integers(0, 1_000_000)
    rng2 = np.random.default_rng(seed=42)
    val2 = rng2.integers(0, 1_000_000)
    assert val1 == val2, "rng fixture seed is not 42"


@pytest.mark.unit
def test_random_obs_shape(random_obs, small_dims):
    """random_obs fixture must have shape [5, 10, dO]."""
    assert random_obs.shape == (5, 10, small_dims["dO"])


@pytest.mark.unit
def test_random_tgt_mu_shape(random_tgt_mu, small_dims):
    """random_tgt_mu fixture must have shape [5, 10, dU]."""
    assert random_tgt_mu.shape == (5, 10, small_dims["dU"])


@pytest.mark.unit
def test_random_tgt_prc_psd(random_tgt_prc):
    """random_tgt_prc precision matrices must be positive semi-definite."""
    N, T, dU, _ = random_tgt_prc.shape
    for n in range(N):
        for t in range(T):
            eigenvalues = np.linalg.eigvalsh(random_tgt_prc[n, t])
            assert np.all(eigenvalues >= -1e-9), (
                f"Precision matrix [{n},{t}] is not PSD: "
                f"min eigenvalue = {eigenvalues.min()}"
            )


@pytest.mark.unit
def test_random_tgt_wt_nonnegative(random_tgt_wt):
    """random_tgt_wt weights must be non-negative."""
    assert np.all(random_tgt_wt >= 0), "Sample weights contain negative values"
