"""
Fault-injection and adversarial-input tests.

These tests verify that the GPS pipeline handles malformed, extreme, or
unexpected inputs gracefully — raising informative exceptions rather than
silently producing NaN / garbage or crashing the interpreter.
"""
from __future__ import annotations

import io
import pickle as _pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — conftest.py handles this for the test runner; this guard
# covers direct execution as a script.
# ---------------------------------------------------------------------------
_PYTHON_ROOT = Path(__file__).parent.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------
_dO, _dU = 14, 7
_N, _T = 3, 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fresh_opt():
    from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
    return PolicyOptPyTorch(
        {"random_seed": 0, "iterations": 2, "batch_size": 4,
         "lr": 1e-3, "weight_decay": 1e-4, "ent_reg": 0.0,
         "init_var": 0.1, "init_var_v": 0.1},
        _dO, _dU,
    )


@pytest.fixture(scope="module")
def trained_opt(fresh_opt):
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((_N, _T, _dO)).astype(np.float32)
    mu = rng.standard_normal((_N, _T, _dU)).astype(np.float32)
    prc = np.tile(np.eye(_dU, dtype=np.float32), (_N, _T, 1, 1))
    wt = np.ones((_N, _T), dtype=np.float32)
    fresh_opt.update(obs, mu, prc, wt)
    return fresh_opt


# ===========================================================================
# 1. NaN / Inf inputs to prob()
# ===========================================================================

@pytest.mark.fault
def test_prob_nan_input_does_not_crash(trained_opt):
    """prob() must not crash the interpreter on an all-NaN input array."""
    obs = np.full((_N, _T, _dO), np.nan, dtype=np.float32)
    # Either succeeds (propagating NaN) or raises a clear Python exception.
    try:
        output, _, _, _ = trained_opt.prob(obs)
        assert output.dtype == np.float32
    except (ValueError, RuntimeError, FloatingPointError):
        pass  # explicit error is acceptable


@pytest.mark.fault
def test_prob_inf_input_does_not_crash(trained_opt):
    """prob() must not crash the interpreter on an all-Inf input array."""
    obs = np.full((_N, _T, _dO), np.inf, dtype=np.float32)
    try:
        output, _, _, _ = trained_opt.prob(obs)
        assert output.dtype == np.float32
    except (ValueError, RuntimeError, FloatingPointError):
        pass


@pytest.mark.fault
def test_prob_large_values_does_not_crash(trained_opt):
    """prob() on very large finite values must not crash."""
    obs = np.full((_N, _T, _dO), 1e30, dtype=np.float32)
    try:
        output, _, _, _ = trained_opt.prob(obs)
        assert output.shape == (_N, _T, _dU)
    except (ValueError, RuntimeError):
        pass


# ===========================================================================
# 2. Wrong-shape inputs to update()
# ===========================================================================

@pytest.mark.fault
def test_update_wrong_obs_feature_dim_raises(fresh_opt):
    """update() with obs dO mismatch must raise an exception."""
    rng = np.random.default_rng(10)
    obs = rng.standard_normal((_N, _T, _dO + 1)).astype(np.float32)
    mu = rng.standard_normal((_N, _T, _dU)).astype(np.float32)
    prc = np.tile(np.eye(_dU, dtype=np.float32), (_N, _T, 1, 1))
    wt = np.ones((_N, _T), dtype=np.float32)
    with pytest.raises(Exception):
        fresh_opt.update(obs, mu, prc, wt)


@pytest.mark.fault
def test_update_wrong_mu_action_dim_raises(fresh_opt):
    """update() with mu dU mismatch must raise an exception."""
    rng = np.random.default_rng(11)
    obs = rng.standard_normal((_N, _T, _dO)).astype(np.float32)
    mu = rng.standard_normal((_N, _T, _dU + 1)).astype(np.float32)
    prc = np.tile(np.eye(_dU, dtype=np.float32), (_N, _T, 1, 1))
    wt = np.ones((_N, _T), dtype=np.float32)
    with pytest.raises(Exception):
        fresh_opt.update(obs, mu, prc, wt)


@pytest.mark.fault
def test_update_all_zero_weights_does_not_crash(fresh_opt):
    """update() with all-zero sample weights must not crash the interpreter."""
    rng = np.random.default_rng(12)
    obs = rng.standard_normal((_N, _T, _dO)).astype(np.float32)
    mu = rng.standard_normal((_N, _T, _dU)).astype(np.float32)
    prc = np.tile(np.eye(_dU, dtype=np.float32), (_N, _T, 1, 1))
    wt = np.zeros((_N, _T), dtype=np.float32)
    from gps.algorithm.policy.pytorch_policy import PyTorchPolicy
    try:
        pol = fresh_opt.update(obs, mu, prc, wt)
        assert isinstance(pol, PyTorchPolicy)
    except (ValueError, RuntimeError, ZeroDivisionError):
        pass  # degenerate weights: explicit error is acceptable


# ===========================================================================
# 3. act() with adversarial observations
# ===========================================================================

@pytest.mark.fault
def test_act_nan_observation_does_not_crash(trained_opt):
    """act() on an all-NaN observation must not crash the interpreter."""
    obs = np.full(_dO, np.nan, dtype=np.float32)
    noise = np.zeros(_dU, dtype=np.float32)
    try:
        u = trained_opt.policy.act(None, obs, 0, noise)
        assert u.shape == (_dU,)
    except (ValueError, RuntimeError):
        pass


@pytest.mark.fault
def test_act_wrong_obs_dimension_raises(trained_opt):
    """act() with wrong observation dimension must raise an exception."""
    obs = np.zeros(_dO + 5, dtype=np.float32)
    with pytest.raises(Exception):
        trained_opt.policy.act(None, obs, 0, None)


# ===========================================================================
# 4. DataLogger: corrupted and empty files
# ===========================================================================

@pytest.mark.fault
def test_unpickle_corrupted_file_returns_none(tmp_path):
    """DataLogger.unpickle must return None on a truncated / corrupted file."""
    from gps.utility.data_logger import DataLogger
    bad_file = str(tmp_path / "corrupted.pkl")
    # Write valid pickle header followed by garbage — causes UnpicklingError.
    with open(bad_file, "wb") as f:
        f.write(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00garbage_content")
    logger = DataLogger()
    result = logger.unpickle(bad_file)
    assert result is None


@pytest.mark.fault
def test_unpickle_empty_file_returns_none(tmp_path):
    """DataLogger.unpickle on a zero-byte file must return None, not raise."""
    from gps.utility.data_logger import DataLogger
    empty_file = str(tmp_path / "empty.pkl")
    open(empty_file, "wb").close()
    logger = DataLogger()
    result = logger.unpickle(empty_file)
    assert result is None


@pytest.mark.fault
def test_unpickle_missing_file_returns_none(tmp_path):
    """DataLogger.unpickle on a non-existent file must return None."""
    from gps.utility.data_logger import DataLogger
    logger = DataLogger()
    result = logger.unpickle(str(tmp_path / "ghost.pkl"))
    assert result is None


# ===========================================================================
# 5. TorchScript: garbage bytes
# ===========================================================================

@pytest.mark.fault
def test_torchscript_garbage_bytes_raises():
    """torch.jit.load on garbage bytes must raise a clear exception."""
    garbage = b"\x00\x01\x02\x03" * 64
    with pytest.raises((RuntimeError, Exception)):
        torch.jit.load(io.BytesIO(garbage))


@pytest.mark.fault
def test_export_torchscript_is_non_trivially_sized(trained_opt):
    """export_torchscript() must produce more than 100 bytes (a real model)."""
    raw = trained_opt.export_torchscript()
    assert isinstance(raw, bytes)
    assert len(raw) > 100


# ===========================================================================
# 6. SampleList edge cases
# ===========================================================================

@pytest.mark.fault
def test_samplelist_empty_num_samples_is_zero():
    """SampleList with no samples must report num_samples() == 0."""
    from gps.sample.sample_list import SampleList
    sl = SampleList([])
    assert sl.num_samples() == 0


@pytest.mark.fault
def test_samplelist_len_matches_num_samples():
    """len(SampleList) must equal num_samples()."""
    from gps.sample.sample_list import SampleList
    sl = SampleList([])
    assert len(sl) == sl.num_samples()


# ===========================================================================
# 7. Proto3 schema: default and round-trip
# ===========================================================================

@pytest.mark.fault
def test_proto3_default_T_is_zero():
    """In proto3, Sample.T defaults to 0 (no custom default like proto2 [default=100])."""
    from gps.proto.gps_pb2 import Sample
    s = Sample()
    assert s.T == 0, f"Expected proto3 default T=0, got {s.T}"


@pytest.mark.fault
def test_proto3_sample_roundtrip():
    """Sample proto message must survive serialization round-trip with integrity."""
    from gps.proto.gps_pb2 import Sample
    s = Sample()
    s.T = 10
    s.dX = 4
    s.dU = 2
    s.X.extend([1.0, 2.0, 3.0, 4.0])
    blob = s.SerializeToString()
    s2 = Sample()
    s2.ParseFromString(blob)
    assert s2.T == 10
    assert s2.dX == 4
    assert list(s2.X) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.fault
def test_proto3_pytorch_controller_enum_value():
    """PYTORCH_CONTROLLER must equal 3 in the compiled proto."""
    from gps.proto.gps_pb2 import PYTORCH_CONTROLLER
    assert PYTORCH_CONTROLLER == 3


@pytest.mark.fault
def test_proto3_total_controller_types_value():
    """TOTAL_CONTROLLER_TYPES must equal 4 (LIN_GAUSS, CAFFE, TF, PYTORCH)."""
    from gps.proto.gps_pb2 import TOTAL_CONTROLLER_TYPES
    assert TOTAL_CONTROLLER_TYPES == 4
