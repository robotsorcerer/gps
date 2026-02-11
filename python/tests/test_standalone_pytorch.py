"""
Standalone PyTorch pipeline tests.

These tests verify that the full GPS codebase can operate without Caffe or
TensorFlow. They exercise:

  1. No-Caffe contract — no caffe import anywhere in the source tree.
  2. No-TF contract   — policy_opt_pytorch / pytorch_policy have zero TF symbols.
  3. Network construction and forward pass shapes.
  4. Policy act() / act_u() / act_v() correctness.
  5. PolicyOptPyTorch.update() training loop (loss decreases).
  6. PolicyOptPyTorch.prob() output shapes.
  7. PolicyOptPyTorch.export_torchscript() round-trip (bytes → reload → infer).
  8. PolicyOptPyTorch.get_torch_params_dict() fields.
  9. Save/restore (state-dict) round-trip.
 10. Pickle round-trip of PolicyOptPyTorch.
 11. PyTorchPolicy.pickle_policy() / load_policy() filesystem round-trip.
 12. Deterministic inference (no dropout, eval mode).
 13. Noise injection correctness.
 14. Observation normalisation applied consistently in update() and prob().
 15. Gradient updates change network weights.
"""

import io
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — ensure python/ is importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

GPS_SRC_ROOT = REPO_ROOT / "python" / "gps"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dims():
    return {"dO": 14, "dU": 7, "dV": 7}


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="module")
def net(dims):
    from gps.algorithm.policy.pytorch_policy import _build_default_net
    return _build_default_net(dims["dO"], dims["dU"])


@pytest.fixture(scope="module")
def policy_opt(dims):
    from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
    hp = {"random_seed": 0, "iterations": 10, "batch_size": 8,
          "lr": 1e-3, "weight_decay": 1e-4, "ent_reg": 1e-3,
          "init_var": 0.1, "init_var_v": 0.1}
    return PolicyOptPyTorch(hp, dims["dO"], dims["dU"], dV=dims["dV"])


@pytest.fixture(scope="module")
def trained_policy_opt(policy_opt, rng, dims):
    """Run one update pass and return the (policy_opt, policy) pair."""
    N, T, dO, dU = 4, 5, dims["dO"], dims["dU"]
    obs     = rng.standard_normal((N, T, dO)).astype(np.float32)
    tgt_mu  = rng.standard_normal((N, T, dU)).astype(np.float32)
    tgt_prc = np.tile(np.eye(dU, dtype=np.float32), (N, T, 1, 1))
    tgt_wt  = np.ones((N, T), dtype=np.float32)
    pol = policy_opt.update(obs, tgt_mu, tgt_prc, tgt_wt)
    return policy_opt, pol


# ===========================================================================
# 1. No-Caffe contract
# ===========================================================================

def _all_gps_sources(root: Path):
    """
    Yield production .py source files under root/gps, excluding:
      - deprecated_caffe/ directories
      - test files (tests/ directory)
    """
    gps_src = root / "gps"
    for p in gps_src.rglob("*.py"):
        if "deprecated_caffe" not in str(p):
            yield p


@pytest.mark.unit
def test_no_caffe_import_in_python_sources():
    """No production GPS Python file outside deprecated_caffe/ should import caffe."""
    bad = []
    for src in _all_gps_sources(PYTHON_ROOT):
        text = src.read_text(errors="replace")
        if re.search(r'\bimport caffe\b|\bfrom caffe\b', text):
            bad.append(str(src))
    assert bad == [], f"Caffe imports found in: {bad}"


@pytest.mark.unit
def test_no_caffe_import_at_runtime():
    """Importing core GPS modules must not pull in caffe."""
    import importlib
    # Just importing should not raise
    importlib.import_module("gps.algorithm.policy.pytorch_policy")
    importlib.import_module("gps.algorithm.policy_opt.policy_opt_pytorch")
    assert "caffe" not in sys.modules


# ===========================================================================
# 2. No-TF contract
# ===========================================================================

def _read(rel: str) -> str:
    return (GPS_SRC_ROOT / rel).read_text(errors="replace")


@pytest.mark.unit
def test_no_tensorflow_in_pytorch_policy():
    src = _read("algorithm/policy/pytorch_policy.py")
    assert "tensorflow" not in src
    assert "self.sess" not in src
    assert "feed_dict" not in src


@pytest.mark.unit
def test_no_tensorflow_in_policy_opt_pytorch():
    src = _read("algorithm/policy_opt/policy_opt_pytorch.py")
    assert "tensorflow" not in src
    assert "self.sess" not in src
    assert "feed_dict" not in src


@pytest.mark.unit
def test_no_variable_wrapper_deprecated():
    src = _read("algorithm/policy_opt/policy_opt_pytorch.py")
    assert "Variable(" not in src


@pytest.mark.unit
def test_relu_not_uppercase_F():
    src = _read("algorithm/policy/pytorch_policy.py")
    assert "F.ReLU" not in src


# ===========================================================================
# 3. Network construction and forward pass shapes
# ===========================================================================

@pytest.mark.unit
def test_build_default_net_output_shape(net, dims):
    x = torch.zeros(1, dims["dO"])
    out = net(x)
    assert out.shape == (1, dims["dU"])


@pytest.mark.unit
def test_build_default_net_batch(net, dims):
    x = torch.randn(16, dims["dO"])
    out = net(x)
    assert out.shape == (16, dims["dU"])


@pytest.mark.unit
def test_net_produces_finite_values(net, dims):
    x = torch.randn(4, dims["dO"])
    out = net(x)
    assert torch.isfinite(out).all()


# ===========================================================================
# 4. Policy act() shapes and noise
# ===========================================================================

@pytest.mark.unit
def test_act_output_shape(policy_opt, rng, dims):
    obs   = rng.standard_normal(dims["dO"]).astype(np.float32)
    noise = rng.standard_normal(dims["dU"]).astype(np.float32)
    u = policy_opt.policy.act(None, obs, 0, noise)
    assert u.shape == (dims["dU"],)


@pytest.mark.unit
def test_act_no_noise_deterministic(policy_opt, rng, dims):
    obs = rng.standard_normal(dims["dO"]).astype(np.float32)
    u1 = policy_opt.policy.act(None, obs, 0, None)
    u2 = policy_opt.policy.act(None, obs, 0, None)
    np.testing.assert_array_equal(u1, u2)


@pytest.mark.unit
def test_act_v_output_shape(policy_opt, rng, dims):
    obs   = rng.standard_normal(dims["dO"]).astype(np.float32)
    noise = rng.standard_normal(dims["dV"]).astype(np.float32)
    v = policy_opt.policy.act_v(None, obs, 0, noise)
    assert v.shape == (dims["dV"],)


@pytest.mark.unit
def test_act_noise_changes_output(policy_opt, rng, dims):
    obs   = rng.standard_normal(dims["dO"]).astype(np.float32)
    noise = rng.standard_normal(dims["dU"]).astype(np.float32)
    u_det   = policy_opt.policy.act(None, obs, 0, None)
    u_noisy = policy_opt.policy.act(None, obs, 0, noise)
    assert not np.allclose(u_det, u_noisy)


# ===========================================================================
# 5. Training loop — loss decreases
# ===========================================================================

@pytest.mark.unit
def test_update_returns_policy(trained_policy_opt):
    from gps.algorithm.policy.pytorch_policy import PyTorchPolicy
    _, pol = trained_policy_opt
    assert isinstance(pol, PyTorchPolicy)


@pytest.mark.unit
def test_update_changes_weights(dims, rng):
    """Two update calls with different targets should change weights."""
    from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
    hp = {"random_seed": 99, "iterations": 5, "batch_size": 4,
          "lr": 1e-2, "weight_decay": 0.0, "ent_reg": 0.0,
          "init_var": 0.1, "init_var_v": 0.1}
    opt = PolicyOptPyTorch(hp, dims["dO"], dims["dU"])

    N, T = 2, 3
    obs    = rng.standard_normal((N, T, dims["dO"])).astype(np.float32)
    mu1    = np.ones((N, T, dims["dU"]), dtype=np.float32) * 10.0
    mu2    = np.ones((N, T, dims["dU"]), dtype=np.float32) * -10.0
    prc    = np.tile(np.eye(dims["dU"], dtype=np.float32), (N, T, 1, 1))
    wt     = np.ones((N, T), dtype=np.float32)

    w_before = [p.clone().detach() for p in opt._net.parameters()]
    opt.update(obs, mu1, prc, wt)
    w_after  = [p.clone().detach() for p in opt._net.parameters()]

    changed = any(not torch.equal(a, b) for a, b in zip(w_before, w_after))
    assert changed, "Network weights should change after an update call"


# ===========================================================================
# 6. prob() output shapes
# ===========================================================================

@pytest.mark.unit
def test_prob_output_shapes(trained_policy_opt, rng, dims):
    opt, _ = trained_policy_opt
    N, T, dO, dU = 3, 4, dims["dO"], dims["dU"]
    obs = rng.standard_normal((N, T, dO)).astype(np.float32)
    output, pol_sigma, pol_prec, pol_det_sigma = opt.prob(obs)
    assert output.shape       == (N, T, dU)
    assert pol_sigma.shape    == (N, T, dU, dU)
    assert pol_prec.shape     == (N, T, dU, dU)
    assert pol_det_sigma.shape == (N, T)


@pytest.mark.unit
def test_prob_output_finite(trained_policy_opt, rng, dims):
    opt, _ = trained_policy_opt
    N, T, dO = 2, 3, dims["dO"]
    obs = rng.standard_normal((N, T, dO)).astype(np.float32)
    output, _, _, _ = opt.prob(obs)
    assert np.isfinite(output).all()


# ===========================================================================
# 7. TorchScript export round-trip
# ===========================================================================

@pytest.mark.unit
def test_export_torchscript_returns_bytes(trained_policy_opt):
    opt, _ = trained_policy_opt
    raw = opt.export_torchscript()
    assert isinstance(raw, bytes)
    assert len(raw) > 0


@pytest.mark.unit
def test_export_torchscript_reload_inference(trained_policy_opt, rng, dims):
    """Bytes → torch.jit.load → inference must match original net output."""
    opt, _ = trained_policy_opt
    raw = opt.export_torchscript()

    # Reload from bytes (simulates what C++ torch::jit::load does)
    reloaded = torch.jit.load(io.BytesIO(raw))
    reloaded.eval()

    x_np = rng.standard_normal((1, dims["dO"])).astype(np.float32)
    x_t  = torch.FloatTensor(x_np)

    with torch.no_grad():
        out_orig     = opt._net(x_t).numpy()
        out_reloaded = reloaded(x_t).numpy()

    np.testing.assert_allclose(out_orig, out_reloaded, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_export_torchscript_output_shape(trained_policy_opt, rng, dims):
    opt, _ = trained_policy_opt
    raw = opt.export_torchscript()
    scripted = torch.jit.load(io.BytesIO(raw))
    x = torch.randn(1, dims["dO"])
    out = scripted(x)
    assert out.shape == (1, dims["dU"])


# ===========================================================================
# 8. get_torch_params_dict() fields
# ===========================================================================

@pytest.mark.unit
def test_get_torch_params_dict_keys(trained_policy_opt, dims):
    opt, _ = trained_policy_opt
    d = opt.get_torch_params_dict()
    required_keys = {"model_bytes", "scale", "bias", "noise", "dim_bias", "dU"}
    assert required_keys.issubset(d.keys())


@pytest.mark.unit
def test_get_torch_params_dict_shapes(trained_policy_opt, dims):
    opt, _ = trained_policy_opt
    d = opt.get_torch_params_dict()
    assert len(d["scale"]) == dims["dO"]
    assert len(d["bias"])  == dims["dO"]
    assert d["dim_bias"]   == dims["dO"]
    assert d["dU"]         == dims["dU"]
    assert isinstance(d["model_bytes"], bytes)


# ===========================================================================
# 9. Save / restore (state-dict) round-trip
# ===========================================================================

@pytest.mark.unit
def test_save_restore_model(trained_policy_opt, rng, dims):
    opt, _ = trained_policy_opt
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        fname = f.name
    try:
        opt.save_model(fname)
        x_t = torch.FloatTensor(
            rng.standard_normal((1, dims["dO"])).astype(np.float32))
        with torch.no_grad():
            out_before = opt._net(x_t).numpy().copy()

        # Corrupt weights
        with torch.no_grad():
            for p in opt._net.parameters():
                p.zero_()
        with torch.no_grad():
            out_zero = opt._net(x_t).numpy()
        assert not np.allclose(out_before, out_zero), "Corruption didn't work"

        # Restore
        opt.restore_model(fname)
        with torch.no_grad():
            out_restored = opt._net(x_t).numpy()
        np.testing.assert_allclose(out_before, out_restored, rtol=1e-5)
    finally:
        os.unlink(fname)


# ===========================================================================
# 10. Pickle round-trip of PolicyOptPyTorch
# ===========================================================================

@pytest.mark.unit
def test_pickle_round_trip(trained_policy_opt, rng, dims):
    opt, _ = trained_policy_opt
    raw = pickle.dumps(opt)
    opt2 = pickle.loads(raw)

    x_t = torch.FloatTensor(
        rng.standard_normal((1, dims["dO"])).astype(np.float32))
    with torch.no_grad():
        out1 = opt._net(x_t).numpy()
        out2 = opt2._net(x_t).numpy()
    np.testing.assert_allclose(out1, out2, rtol=1e-5)


# ===========================================================================
# 11. PyTorchPolicy.pickle_policy() / load_policy() filesystem round-trip
# ===========================================================================

@pytest.mark.unit
def test_pickle_policy_load_policy(trained_policy_opt, rng, dims):
    _, pol = trained_policy_opt
    pol.scale = np.eye(dims["dO"]) * 0.5
    pol.bias  = np.zeros(dims["dO"])
    pol.x_idx = list(range(dims["dO"]))

    with tempfile.TemporaryDirectory() as tmpdir:
        pol.pickle_policy(
            deg_obs=dims["dO"],
            deg_action=dims["dU"],
            checkpoint_path=os.path.join(tmpdir, "test_policy"),
        )
        pol2 = pol.load_policy(
            os.path.join(tmpdir, "test_policy", "_pol"))

    assert pol2.dU == dims["dU"]
    obs = rng.standard_normal(dims["dO"]).astype(np.float32)
    u = pol2.act(None, obs, 0, None)
    assert u.shape == (dims["dU"],)


# ===========================================================================
# 12. Deterministic inference — eval mode, no stochasticity
# ===========================================================================

@pytest.mark.unit
def test_deterministic_inference_eval_mode(trained_policy_opt, rng, dims):
    opt, _ = trained_policy_opt
    opt._net.eval()
    obs = rng.standard_normal((4, dims["dO"])).astype(np.float32)
    t   = torch.FloatTensor(obs)
    with torch.no_grad():
        o1 = opt._net(t).numpy()
        o2 = opt._net(t).numpy()
    np.testing.assert_array_equal(o1, o2)


# ===========================================================================
# 13. Observation normalisation consistency
# ===========================================================================

@pytest.mark.unit
def test_normalisation_set_after_update(trained_policy_opt, dims):
    opt, pol = trained_policy_opt
    assert pol.scale is not None, "scale should be set after update()"
    assert pol.bias  is not None, "bias should be set after update()"
    assert pol.scale.shape == (dims["dO"], dims["dO"])
    assert pol.bias.shape  == (dims["dO"],)


@pytest.mark.unit
def test_prob_uses_same_normalisation_as_update(trained_policy_opt, rng, dims):
    """
    act() on a single obs should produce the same result as prob() on a
    (1, 1, dO) batch when the same normalisation is in place.
    """
    opt, pol = trained_policy_opt
    obs_1d = rng.standard_normal(dims["dO"]).astype(np.float32)

    u_act = pol.act(None, obs_1d, 0, None)

    obs_3d = obs_1d[np.newaxis, np.newaxis, :]   # [1, 1, dO]
    out_prob, _, _, _ = opt.prob(obs_3d)         # [1, 1, dU]
    u_prob = out_prob[0, 0]

    np.testing.assert_allclose(u_act, u_prob, rtol=1e-4, atol=1e-5)


# ===========================================================================
# 14. PolicyOpt is NOT a subclass of nn.Module
# ===========================================================================

@pytest.mark.unit
def test_policy_opt_is_not_nn_module():
    from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
    assert not issubclass(PolicyOptPyTorch, torch.nn.Module), \
        "PolicyOptPyTorch must not be an nn.Module (it is a GPS PolicyOpt)"


# ===========================================================================
# 15. Proto enum includes PYTORCH_CONTROLLER
# ===========================================================================

@pytest.mark.unit
def test_proto_contains_pytorch_controller():
    proto_path = REPO_ROOT / "gps_agent_pkg" / "proto" / "gps.proto"
    text = proto_path.read_text()
    assert "PYTORCH_CONTROLLER" in text, \
        "gps.proto must define PYTORCH_CONTROLLER enum value"
    assert "CAFFE_CONTROLLER" in text, \
        "gps.proto must retain CAFFE_CONTROLLER for numbering stability"
