"""
Phase 2 tests — PyTorch policy optimizer gate.

All 11 tests must pass before Phase 3 (C++17) begins.
Tests are ordered from fast/structural checks to slower functional checks.
"""
import copy
import pickle
import inspect

import numpy as np
import pytest
import torch

from gps.algorithm.policy_opt.config import POLICY_OPT_PYTORCH
from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
from gps.algorithm.policy.pytorch_policy import PyTorchPolicy, _build_default_net

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_dO, _dU, _dV = 14, 7, 7
_N, _T = 5, 10
_HYPER = copy.deepcopy(POLICY_OPT_PYTORCH)
_HYPER['iterations'] = 5  # keep tests fast


@pytest.fixture(scope="module")
def policy_opt():
    return PolicyOptPyTorch(_HYPER, _dO, _dU, _dV)


# ---------------------------------------------------------------------------
# 1. No TensorFlow import
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_tensorflow_import():
    """policy_opt_pytorch.py must not import TensorFlow."""
    import gps.algorithm.policy_opt.policy_opt_pytorch as mod
    source = inspect.getsource(mod)
    assert 'tensorflow' not in source.lower(), (
        "TensorFlow reference found in policy_opt_pytorch.py"
    )
    assert 'import tf' not in source
    assert 'self.sess' not in source
    assert 'feed_dict' not in source


@pytest.mark.unit
def test_no_tensorflow_in_pytorch_policy():
    """pytorch_policy.py must not import TensorFlow."""
    import gps.algorithm.policy.pytorch_policy as mod
    source = inspect.getsource(mod)
    assert 'tensorflow' not in source.lower()
    assert 'self.sess' not in source
    assert 'self.obs_tensor' not in source


# ---------------------------------------------------------------------------
# 2. No deprecated torch.autograd.Variable
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_variable_deprecated():
    """Variable wrapper was deprecated in PyTorch 0.4 — must not be used."""
    import gps.algorithm.policy_opt.policy_opt_pytorch as mod
    source = inspect.getsource(mod)
    assert 'Variable(' not in source, (
        "Deprecated torch.autograd.Variable() still present"
    )


# ---------------------------------------------------------------------------
# 3. F.relu (lowercase) not F.ReLU
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_relu_lowercase():
    """Network must use F.relu / torch.relu, not F.ReLU (class, not function)."""
    import gps.algorithm.policy.pytorch_policy as mod
    source = inspect.getsource(mod)
    assert 'F.ReLU' not in source, "F.ReLU (uppercase) found — should be torch.relu"


# ---------------------------------------------------------------------------
# 4. PolicyOptPyTorch does NOT inherit from nn.Module
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_not_nn_module():
    """PolicyOptPyTorch must not inherit from nn.Module (it's a GPS optimizer, not a layer)."""
    assert not issubclass(PolicyOptPyTorch, torch.nn.Module), (
        "PolicyOptPyTorch incorrectly inherits from nn.Module"
    )


# ---------------------------------------------------------------------------
# 5. Forward pass: output shape [N, T, dU]
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_forward_output_shape(policy_opt):
    """prob() must return output of shape [N, T, dU]."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal((_N, _T, _dO))
    output, pol_sigma, pol_prec, pol_det_sigma = policy_opt.prob(obs)
    assert output.shape == (_N, _T, _dU), (
        f"Expected output shape {(_N, _T, _dU)}, got {output.shape}"
    )
    assert pol_sigma.shape == (_N, _T, _dU, _dU)
    assert pol_prec.shape == (_N, _T, _dU, _dU)
    assert pol_det_sigma.shape == (_N, _T)


# ---------------------------------------------------------------------------
# 6. Forward pass is deterministic with eval() mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_forward_deterministic(policy_opt):
    """prob() must be deterministic (net in eval mode, no dropout)."""
    rng = np.random.default_rng(7)
    obs = rng.standard_normal((_N, _T, _dO))
    out1, _, _, _ = policy_opt.prob(obs)
    out2, _, _, _ = policy_opt.prob(obs)
    np.testing.assert_array_equal(out1, out2, err_msg="prob() is non-deterministic")


# ---------------------------------------------------------------------------
# 7. update() returns a PyTorchPolicy
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_update_returns_pytorch_policy():
    """update() must return a PyTorchPolicy instance."""
    hyper = copy.deepcopy(_HYPER)
    hyper['iterations'] = 2
    pol_opt = PolicyOptPyTorch(hyper, _dO, _dU, _dV)
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((_N, _T, _dO))
    tgt_mu = rng.standard_normal((_N, _T, _dU))
    raw_prc = rng.standard_normal((_N, _T, _dU, _dU))
    tgt_prc = raw_prc @ raw_prc.transpose(0, 1, 3, 2) + np.eye(_dU) * 0.1
    tgt_wt = np.abs(rng.standard_normal((_N, _T))) + 0.01

    result = pol_opt.update(obs, tgt_mu, tgt_prc, tgt_wt)
    assert isinstance(result, PyTorchPolicy), (
        f"update() returned {type(result)}, expected PyTorchPolicy"
    )


# ---------------------------------------------------------------------------
# 8. update() actually changes the network weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_update_changes_weights():
    """Training for at least 1 iteration must change network weights."""
    hyper = copy.deepcopy(_HYPER)
    hyper['iterations'] = 3
    pol_opt = PolicyOptPyTorch(hyper, _dO, _dU, _dV)
    # Snapshot initial weights
    initial_weights = {
        name: param.clone().detach()
        for name, param in pol_opt._net.named_parameters()
    }
    rng = np.random.default_rng(99)
    obs = rng.standard_normal((_N, _T, _dO))
    tgt_mu = rng.standard_normal((_N, _T, _dU))
    raw_prc = rng.standard_normal((_N, _T, _dU, _dU))
    tgt_prc = raw_prc @ raw_prc.transpose(0, 1, 3, 2) + np.eye(_dU) * 0.1
    tgt_wt = np.ones((_N, _T))
    pol_opt.update(obs, tgt_mu, tgt_prc, tgt_wt)
    # At least one weight must have changed
    any_changed = any(
        not torch.equal(initial_weights[n], p.detach())
        for n, p in pol_opt._net.named_parameters()
    )
    assert any_changed, "No network weights changed after update() — gradient flow broken"


# ---------------------------------------------------------------------------
# 9. act() returns correct shape
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_act_output_shape(policy_opt):
    """act() must return a dU-dimensional action vector."""
    rng = np.random.default_rng(3)
    obs = rng.standard_normal(_dO)
    noise = rng.standard_normal(_dU)
    action = policy_opt.policy.act(None, obs, 0, noise)
    assert action.shape == (_dU,), (
        f"act() returned shape {action.shape}, expected ({_dU},)"
    )


@pytest.mark.unit
def test_act_no_noise(policy_opt):
    """act() with noise=None must return a finite dU-vector."""
    rng = np.random.default_rng(4)
    obs = rng.standard_normal(_dO)
    action = policy_opt.policy.act(None, obs, 0, None)
    assert action.shape == (_dU,)
    assert np.all(np.isfinite(action)), "act() returned non-finite values"


# ---------------------------------------------------------------------------
# 10. Pickle round-trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pickle_round_trip():
    """PolicyOptPyTorch must survive pickle/unpickle without errors."""
    hyper = copy.deepcopy(_HYPER)
    hyper['iterations'] = 1
    pol_opt = PolicyOptPyTorch(hyper, _dO, _dU, _dV)
    # Run one update so weights are non-trivial
    rng = np.random.default_rng(5)
    obs = rng.standard_normal((_N, _T, _dO))
    tgt_mu = rng.standard_normal((_N, _T, _dU))
    raw_prc = rng.standard_normal((_N, _T, _dU, _dU))
    tgt_prc = raw_prc @ raw_prc.transpose(0, 1, 3, 2) + np.eye(_dU) * 0.1
    tgt_wt = np.ones((_N, _T))
    pol_opt.update(obs, tgt_mu, tgt_prc, tgt_wt)

    # Pickle
    data = pickle.dumps(pol_opt)
    pol_opt2 = pickle.loads(data)

    # Check forward pass is identical
    obs_test = rng.standard_normal((_N, _T, _dO))
    out1, _, _, _ = pol_opt.prob(obs_test)
    out2, _, _, _ = pol_opt2.prob(obs_test)
    np.testing.assert_allclose(
        out1, out2, rtol=1e-5, atol=1e-6,
        err_msg="Pickle round-trip changed network output"
    )


# ---------------------------------------------------------------------------
# 11. PyTorchPolicy save / restore
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_policy_save_restore(tmp_path):
    """pickle_policy / load_policy must preserve act() output."""
    dO, dU = 8, 4
    net = _build_default_net(dO, dU)
    pol = PyTorchPolicy(
        dU=dU, dV=dU, net=net,
        var_u=0.1 * np.ones(dU),
        var_v=0.1 * np.ones(dU),
    )
    pol.scale = np.eye(dO)
    pol.bias = np.zeros(dO)
    pol.x_idx = list(range(dO))

    ckpt = str(tmp_path / "pol_ckpt")
    pol.pickle_policy(dO, dU, ckpt)

    pol2 = PyTorchPolicy.load_policy(ckpt + '/_pol')

    rng = np.random.default_rng(11)
    obs = rng.standard_normal(dO).astype(np.float32)
    a1 = pol.act(None, obs, 0, None)
    a2 = pol2.act(None, obs, 0, None)
    np.testing.assert_allclose(a1, a2, rtol=1e-5, atol=1e-6,
                               err_msg="Policy save/restore changed act() output")
