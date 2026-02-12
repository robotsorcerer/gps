"""
Integration tests for the GPS algorithm–agent data pipeline.

These tests exercise multiple components working together:
  - Sample creation, data packing, and SampleList access
  - CostAction and CostState evaluation on Sample objects
  - PolicyOptPyTorch.update() driven by SampleList data
  - DataLogger pickle/unpickle round-trip for algorithm state
  - PickleSampleWriter / PickleSampleReader filesystem round-trip
  - Proto3 Sample message encoding of real observation data

All tests are marked `integration` and require no ROS or GPU.
"""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PYTHON_ROOT = Path(__file__).parent.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


# ---------------------------------------------------------------------------
# Mock agent — satisfies Sample.__init__() without needing a real robot
# ---------------------------------------------------------------------------

class _MockAgent:
    """Minimal agent stub that provides the dimension attributes Sample needs."""
    def __init__(self, T: int = 10, dX: int = 14, dU: int = 7,
                 dV: int = 7, dO: int = 14, dM: int = 0) -> None:
        self.T = T
        self.dX = dX
        self.dU = dU
        self.dV = dV
        self.dO = dO
        self.dM = dM
        # Sensor layout used by agent.pack_data_obs / pack_data_x
        self._x_data_types = []
        self._obs_data_types = []
        self._sensor_dims: dict = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dims() -> dict:
    return {"T": 10, "dX": 14, "dU": 7, "dV": 7, "dO": 14}


@pytest.fixture(scope="module")
def mock_agent(dims: dict) -> _MockAgent:
    return _MockAgent(**dims)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=7)


@pytest.fixture(scope="module")
def sample_with_data(mock_agent: _MockAgent, rng: np.random.Generator, dims: dict):
    """A Sample with synthetic U, X, and obs data filled in."""
    from gps.sample.sample import Sample
    from gps.proto.gps_pb2 import ACTION, JOINT_ANGLES, JOINT_VELOCITIES

    s = Sample(mock_agent)
    T, dU, dX = dims["T"], dims["dU"], dims["dX"]

    u_data = rng.standard_normal((T, dU)).astype(np.float32)
    x_data = rng.standard_normal((T, dX)).astype(np.float32)

    s.set(ACTION, u_data)
    s.set(JOINT_ANGLES, x_data[:, :7])
    s.set(JOINT_VELOCITIES, x_data[:, 7:])
    return s


@pytest.fixture(scope="module")
def sample_list_4(mock_agent: _MockAgent, rng: np.random.Generator, dims: dict):
    """A SampleList of 4 Samples with synthetic obs data."""
    from gps.sample.sample import Sample
    from gps.sample.sample_list import SampleList
    from gps.proto.gps_pb2 import ACTION

    samples = []
    T, dU, dO = dims["T"], dims["dU"], dims["dO"]
    for i in range(4):
        s = Sample(mock_agent)
        u = rng.standard_normal((T, dU)).astype(np.float32)
        obs = rng.standard_normal((T, dO)).astype(np.float32)
        s.set(ACTION, u)
        # Directly set the obs array (bypasses sensor packing)
        s._obs = obs
        samples.append(s)
    return SampleList(samples)


@pytest.fixture(scope="module")
def trained_policy_opt(sample_list_4, dims: dict):
    """PolicyOptPyTorch trained for 5 iterations on sample_list_4 data."""
    from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
    dO, dU = dims["dO"], dims["dU"]
    N = sample_list_4.num_samples()
    T = dims["T"]

    opt = PolicyOptPyTorch(
        {"random_seed": 0, "iterations": 5, "batch_size": 8,
         "lr": 1e-3, "weight_decay": 1e-4, "ent_reg": 0.0,
         "init_var": 0.1, "init_var_v": 0.1},
        dO, dU,
    )

    obs = sample_list_4.get_obs()       # [N, T, dO]
    tgt_mu = sample_list_4.get_U()       # [N, T, dU]
    tgt_prc = np.tile(np.eye(dU, dtype=np.float32), (N, T, 1, 1))
    tgt_wt = np.ones((N, T), dtype=np.float32)
    opt.update(obs, tgt_mu, tgt_prc, tgt_wt)
    return opt


# ===========================================================================
# 1. Sample creation and data access
# ===========================================================================

@pytest.mark.integration
def test_sample_dimensions_match_agent(sample_with_data, dims: dict):
    """Sample dimensions must match the mock agent's configuration."""
    s = sample_with_data
    assert s.T == dims["T"]
    assert s.dU == dims["dU"]
    assert s.dX == dims["dX"]
    assert s.dO == dims["dO"]


@pytest.mark.integration
def test_sample_get_U_shape(sample_with_data, dims: dict):
    """sample.get_U() must return [T, dU]."""
    u = sample_with_data.get_U()
    assert u.shape == (dims["T"], dims["dU"])


@pytest.mark.integration
def test_sample_data_preserved_after_set(sample_with_data, dims: dict):
    """Data stored via sample.set() must be retrievable via sample.get()."""
    from gps.proto.gps_pb2 import ACTION
    u_retrieved = sample_with_data.get(ACTION)
    assert u_retrieved is not None
    assert u_retrieved.shape == (dims["T"], dims["dU"])
    assert np.isfinite(u_retrieved).all()


# ===========================================================================
# 2. SampleList access
# ===========================================================================

@pytest.mark.integration
def test_samplelist_get_obs_shape(sample_list_4, dims: dict):
    """SampleList.get_obs() must return [N, T, dO]."""
    obs = sample_list_4.get_obs()
    assert obs.shape == (4, dims["T"], dims["dO"])


@pytest.mark.integration
def test_samplelist_get_U_shape(sample_list_4, dims: dict):
    """SampleList.get_U() must return [N, T, dU]."""
    u = sample_list_4.get_U()
    assert u.shape == (4, dims["T"], dims["dU"])


@pytest.mark.integration
def test_samplelist_num_samples(sample_list_4):
    """SampleList.num_samples() must return 4."""
    assert sample_list_4.num_samples() == 4


@pytest.mark.integration
def test_samplelist_indexing(sample_list_4):
    """SampleList[i] must return the i-th sample."""
    from gps.sample.sample import Sample
    s = sample_list_4[0]
    assert isinstance(s, Sample)


# ===========================================================================
# 3. CostAction evaluation
# ===========================================================================

@pytest.mark.integration
def test_cost_action_protagonist_output_shapes(sample_with_data, dims: dict):
    """CostAction.eval() in protagonist mode must return correct array shapes."""
    from gps.algorithm.cost.cost_action import CostAction
    T, dU, dX = dims["T"], dims["dU"], dims["dX"]

    cost = CostAction({"wu": np.ones(dU), "gamma": 1.0, "mode": "protagonist"})
    l, lx, lu, lxx, luu, lux = cost.eval(sample_with_data)

    assert l.shape == (T,), f"l shape: {l.shape}"
    assert lu.shape == (T, dU), f"lu shape: {lu.shape}"
    assert lx.shape == (T, dX), f"lx shape: {lx.shape}"
    assert luu.shape == (T, dU, dU), f"luu shape: {luu.shape}"
    assert np.isfinite(l).all(), "CostAction produced non-finite loss"


@pytest.mark.integration
def test_cost_action_protagonist_loss_nonnegative(sample_with_data, dims: dict):
    """L2 action cost must be non-negative for any input."""
    from gps.algorithm.cost.cost_action import CostAction
    dU = dims["dU"]
    cost = CostAction({"wu": np.ones(dU), "gamma": 1.0, "mode": "protagonist"})
    l, *_ = cost.eval(sample_with_data)
    assert np.all(l >= 0.0), f"Action cost has negative values: {l.min()}"


# ===========================================================================
# 4. Full train → act cycle driven by SampleList
# ===========================================================================

@pytest.mark.integration
def test_full_train_then_act(trained_policy_opt, dims: dict, rng):
    """update() followed by act() on a fresh obs must return a finite action."""
    dO, dU = dims["dO"], dims["dU"]
    obs = rng.standard_normal(dO).astype(np.float32)
    u = trained_policy_opt.policy.act(None, obs, 0, None)
    assert u.shape == (dU,)
    assert np.isfinite(u).all()


@pytest.mark.integration
def test_prob_on_trained_policy_shapes(trained_policy_opt, sample_list_4, dims: dict):
    """prob() on the trained policy must return correctly shaped outputs."""
    dO, dU = dims["dO"], dims["dU"]
    N = sample_list_4.num_samples()
    T = dims["T"]

    obs = sample_list_4.get_obs()
    output, pol_sigma, pol_prec, pol_det_sigma = trained_policy_opt.prob(obs)
    assert output.shape == (N, T, dU)
    assert pol_sigma.shape == (N, T, dU, dU)
    assert pol_det_sigma.shape == (N, T)


# ===========================================================================
# 5. DataLogger: algorithm state persistence
# ===========================================================================

@pytest.mark.integration
def test_data_logger_saves_and_loads_policy_opt_state(trained_policy_opt, dims: dict, rng):
    """DataLogger must successfully pickle and restore a PolicyOptPyTorch."""
    from gps.utility.data_logger import DataLogger
    logger = DataLogger()

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        logger.pickle(path, trained_policy_opt)
        restored = logger.unpickle(path)
        assert restored is not None

        # The restored object must produce the same inference output.
        dO = dims["dO"]
        obs_test = rng.standard_normal((1, 1, dO)).astype(np.float32)
        out1, _, _, _ = trained_policy_opt.prob(obs_test)
        out2, _, _, _ = restored.prob(obs_test)
        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.integration
def test_data_logger_saves_and_loads_numpy_arrays():
    """DataLogger round-trip must preserve arbitrary nested numpy structures."""
    from gps.utility.data_logger import DataLogger
    logger = DataLogger()

    payload = {
        "iteration": 42,
        "costs": np.array([1.0, 2.0, 3.0]),
        "meta": {"desc": "test", "nested": np.eye(3)},
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        logger.pickle(path, payload)
        recovered = logger.unpickle(path)
        assert recovered["iteration"] == 42
        np.testing.assert_array_equal(recovered["costs"], payload["costs"])
        np.testing.assert_array_equal(recovered["meta"]["nested"], payload["meta"]["nested"])
    finally:
        Path(path).unlink(missing_ok=True)


# ===========================================================================
# 6. PickleSampleWriter / SampleList filesystem round-trip
# ===========================================================================

@pytest.mark.integration
def test_pickle_sample_writer_round_trip(sample_list_4):
    """PickleSampleWriter must write a SampleList that unpickles correctly."""
    from gps.sample.sample_list import PickleSampleWriter

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        writer = PickleSampleWriter(path)
        writer.write(sample_list_4)

        with open(path, "rb") as f:
            recovered = pickle.load(f)

        assert len(recovered) == len(sample_list_4)
    finally:
        Path(path).unlink(missing_ok=True)


# ===========================================================================
# 7. Proto3 Sample encoding of real observation data
# ===========================================================================

@pytest.mark.integration
def test_proto_sample_encodes_obs_array(sample_list_4, dims: dict):
    """Observation data packed into a proto Sample must round-trip correctly."""
    from gps.proto.gps_pb2 import Sample

    obs = sample_list_4.get_obs()[0]   # [T, dO]
    T, dO = obs.shape

    s = Sample()
    s.T = T
    s.dO = dO
    s.obs.extend(obs.flatten().tolist())

    blob = s.SerializeToString()
    s2 = Sample()
    s2.ParseFromString(blob)

    assert s2.T == T
    assert s2.dO == dO
    recovered = np.array(list(s2.obs), dtype=np.float32).reshape(T, dO)
    np.testing.assert_allclose(recovered, obs, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_proto_sample_encodes_action_array(sample_with_data, dims: dict):
    """Action data packed into a proto Sample must round-trip correctly."""
    from gps.proto.gps_pb2 import Sample

    u = sample_with_data.get_U()   # [T, dU]
    T, dU = u.shape

    s = Sample()
    s.T = T
    s.dU = dU
    s.U.extend(u.flatten().tolist())

    blob = s.SerializeToString()
    s2 = Sample()
    s2.ParseFromString(blob)

    recovered = np.array(list(s2.U), dtype=np.float32).reshape(T, dU)
    np.testing.assert_allclose(recovered, u, rtol=1e-5, atol=1e-6)


# ===========================================================================
# Rec 7: training metrics stored on PolicyOptPyTorch after update()
# ===========================================================================

@pytest.mark.integration
def test_update_stores_last_loss(trained_policy_opt):
    """Rec 7: after update(), _last_loss must be a finite float."""
    assert hasattr(trained_policy_opt, '_last_loss'), \
        "PolicyOptPyTorch missing _last_loss after update()"
    assert isinstance(trained_policy_opt._last_loss, float)
    assert np.isfinite(trained_policy_opt._last_loss)


@pytest.mark.integration
def test_update_stores_last_grad_norm(trained_policy_opt):
    """Rec 7: after update(), _last_grad_norm must be a non-negative finite float."""
    assert hasattr(trained_policy_opt, '_last_grad_norm'), \
        "PolicyOptPyTorch missing _last_grad_norm after update()"
    assert isinstance(trained_policy_opt._last_grad_norm, float)
    assert np.isfinite(trained_policy_opt._last_grad_norm)
    assert trained_policy_opt._last_grad_norm >= 0.0


# ===========================================================================
# Rec 8: proto T > 0 validation helper
# ===========================================================================

@pytest.mark.integration
def test_proto_validate_zero_T_raises():
    """Rec 8: check_sample() must raise ValueError when T == 0 (proto3 default)."""
    from gps.proto.gps_pb2 import Sample
    from gps.utility.proto_validate import check_sample
    s = Sample()   # T defaults to 0 in proto3
    with pytest.raises(ValueError, match="T == 0"):
        check_sample(s)


@pytest.mark.integration
def test_proto_validate_good_sample_passes():
    """Rec 8: check_sample() must not raise on a well-formed Sample."""
    from gps.proto.gps_pb2 import Sample
    from gps.utility.proto_validate import check_sample
    s = Sample()
    s.T = 10
    s.dU = 7
    s.U.extend([0.0] * (10 * 7))
    check_sample(s)   # must not raise


@pytest.mark.integration
def test_proto_validate_u_length_mismatch_raises():
    """Rec 8: check_sample() must raise when U length != T*dU."""
    from gps.proto.gps_pb2 import Sample
    from gps.utility.proto_validate import check_sample
    s = Sample()
    s.T = 10
    s.dU = 7
    s.U.extend([0.0] * 42)   # wrong: should be 70
    with pytest.raises(ValueError, match="T\\*dU"):
        check_sample(s)


# ===========================================================================
# Rec 9: iDG antagonist None-guard in CostAction
# ===========================================================================

@pytest.mark.integration
def test_cost_action_antagonist_none_sample_prot_raises(sample_with_data, dims: dict):
    """Rec 9: CostAction in antagonist mode must raise ValueError when sample_prot is None."""
    from gps.algorithm.cost.cost_action import CostAction
    dU = dims["dU"]
    cost = CostAction({"wu": np.ones(dU), "gamma": 1.0, "mode": "antagonist"})
    with pytest.raises(ValueError, match="sample_prot"):
        cost.eval(sample_with_data, sample_prot=None)


@pytest.mark.integration
def test_cost_action_antagonist_missing_kwarg_raises(sample_with_data, dims: dict):
    """Rec 9: CostAction in antagonist mode must raise when sample_prot kwarg is absent."""
    from gps.algorithm.cost.cost_action import CostAction
    dU = dims["dU"]
    cost = CostAction({"wu": np.ones(dU), "gamma": 1.0, "mode": "antagonist"})
    with pytest.raises((ValueError, KeyError)):
        cost.eval(sample_with_data)  # no sample_prot kwarg at all
