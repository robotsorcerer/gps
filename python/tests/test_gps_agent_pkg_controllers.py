"""
Tests for the controller hierarchy:
  LinearGaussianController, PyTorchController, TfController, PositionController.

All tests are pure-Python (no live ROS required).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import numpy as np
import pytest
from unittest.mock import MagicMock

from gps_agent_pkg.sample import ControllerSample, SAMPLE_FORMAT_VECTOR
from gps_agent_pkg.lingauss_controller import LinearGaussianController
from gps_agent_pkg.tf_controller import TfController
from gps_agent_pkg.position_controller import PositionController, NO_CONTROL, JOINT_SPACE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(T=10, dtypes=None):
    s = ControllerSample(T)
    for dt, sz in (dtypes or [(0, 4), (1, 4)]):
        s.set_meta_data(dt, sz, fmt=SAMPLE_FORMAT_VECTOR)
    return s


def _lg_options(T=3, dX=4, dU=2):
    opts = {
        "T": T,
        "dX": dX,
        "dU": dU,
        "state_datatypes": [0],
        "obs_datatypes":   [1],
    }
    for t in range(T):
        opts[f"K_{t}"] = np.eye(dU, dX)
        opts[f"k_{t}"] = np.zeros(dU)
    return opts


# ---------------------------------------------------------------------------
# LinearGaussianController
# ---------------------------------------------------------------------------

class TestLinearGaussianController:
    def test_configure(self):
        ctrl = LinearGaussianController()
        ctrl.configure_controller(_lg_options())
        assert ctrl.is_configured_

    def test_get_action_identity(self):
        ctrl = LinearGaussianController()
        opts = _lg_options(T=2, dX=3, dU=3)
        ctrl.configure_controller(opts)
        X   = np.array([1.0, 2.0, 3.0])
        obs = np.zeros(3)
        U = ctrl.get_action(0, X, obs)
        np.testing.assert_allclose(U, X)

    def test_get_action_with_bias(self):
        ctrl = LinearGaussianController()
        opts = _lg_options(T=1, dX=2, dU=2)
        opts["k_0"] = np.array([10.0, 20.0])
        ctrl.configure_controller(opts)
        X   = np.zeros(2)
        U = ctrl.get_action(0, X, np.zeros(2))
        np.testing.assert_allclose(U, [10.0, 20.0])

    def test_trial_runs_full_steps(self):
        T = 5
        ctrl = LinearGaussianController()
        ctrl.configure_controller(_lg_options(T=T, dX=2, dU=2))
        s = _make_sample(T=T, dtypes=[(0, 2), (1, 2)])

        for t in range(T):
            torques = np.zeros(2)
            ctrl.update(MagicMock(), 0.0, s, torques)

        assert ctrl.is_finished()
        assert ctrl.get_step_counter() == T

    def test_is_not_finished_initially(self):
        ctrl = LinearGaussianController()
        ctrl.configure_controller(_lg_options(T=3))
        assert not ctrl.is_finished()

    def test_step_counter_increments(self):
        ctrl = LinearGaussianController()
        ctrl.configure_controller(_lg_options(T=3, dX=2, dU=2))
        s = _make_sample(T=3, dtypes=[(0, 2), (1, 2)])
        ctrl.update(MagicMock(), 0.0, s, np.zeros(2))
        assert ctrl.get_step_counter() == 1


# ---------------------------------------------------------------------------
# PyTorchController
# ---------------------------------------------------------------------------

class TestPyTorchController:
    @pytest.fixture
    def torch_opts(self):
        """Build options with a real TorchScript model."""
        try:
            import torch, torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        dO, dU = 4, 2
        net = torch.nn.Linear(dO, dU)
        scripted = torch.jit.script(net)
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        model_bytes = buf.getvalue()

        T = 3
        opts = {
            "T": T,
            "state_datatypes": [0],
            "obs_datatypes":   [1],
            "model_bytes":    model_bytes,
            "torch_version":  torch.__version__,
            "scale":          np.ones(dO),
            "bias":           np.zeros(dO),
        }
        for t in range(T):
            opts[f"noise_{t}"] = np.zeros(dU)
        return opts, dU

    def test_configure_and_get_action(self, torch_opts):
        from gps_agent_pkg.pytorch_controller import PyTorchController
        opts, dU = torch_opts
        ctrl = PyTorchController()
        ctrl.configure_controller(opts)
        assert ctrl.is_configured_

        obs = np.ones(4)
        U = ctrl.get_action(0, np.zeros(4), obs)
        assert U.shape == (dU,)
        assert np.isfinite(U).all()

    def test_noise_is_added(self, torch_opts):
        from gps_agent_pkg.pytorch_controller import PyTorchController
        opts, dU = torch_opts
        # Set deterministic noise at t=0
        opts["noise_0"] = np.array([100.0, 200.0])
        ctrl = PyTorchController()
        ctrl.configure_controller(opts)
        obs = np.zeros(4)

        import torch, torch.nn as nn, io
        net = torch.nn.Linear(4, 2)
        # Override model to output zeros
        with torch.no_grad():
            net.weight.fill_(0.0)
            net.bias.fill_(0.0)
        scripted2 = torch.jit.script(net)
        buf = io.BytesIO()
        torch.jit.save(scripted2, buf)
        opts2 = dict(opts)
        opts2["model_bytes"] = buf.getvalue()
        ctrl2 = PyTorchController()
        ctrl2.configure_controller(opts2)
        U = ctrl2.get_action(0, np.zeros(4), obs)
        # output should be noise = [100, 200]
        np.testing.assert_allclose(U, [100.0, 200.0], atol=1e-5)

    def test_version_mismatch_raises(self, torch_opts):
        from gps_agent_pkg.pytorch_controller import PyTorchController
        opts, dU = torch_opts
        opts2 = dict(opts)
        opts2["torch_version"] = "0.1.0"  # wrong major version
        ctrl = PyTorchController()
        with pytest.raises(ValueError, match="mismatch"):
            ctrl.configure_controller(opts2)


# ---------------------------------------------------------------------------
# TfController
# ---------------------------------------------------------------------------

class TestTfController:
    def test_configure(self):
        ctrl = TfController()
        ctrl.configure_controller({"T": 5, "dU": 3,
                                    "state_datatypes": [], "obs_datatypes": []})
        assert ctrl.is_configured_

    def test_fresh_command(self):
        ctrl = TfController()
        ctrl.configure_controller({"T": 5, "dU": 3,
                                    "state_datatypes": [], "obs_datatypes": []})
        ctrl.update_action_command(1, np.array([1.0, 2.0, 3.0]))
        U = ctrl.get_action(0, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(U, [1.0, 2.0, 3.0])

    def test_stale_tolerance_up_to_2(self):
        ctrl = TfController()
        ctrl.configure_controller({"T": 5, "dU": 2,
                                    "state_datatypes": [], "obs_datatypes": []})
        ctrl.update_action_command(1, np.array([5.0, 6.0]))
        # Act on fresh command
        ctrl.get_action(0, np.zeros(2), np.zeros(2))
        # 1st stale use
        U = ctrl.get_action(1, np.zeros(2), np.zeros(2))
        np.testing.assert_allclose(U, [5.0, 6.0])
        # 2nd stale use
        U = ctrl.get_action(2, np.zeros(2), np.zeros(2))
        np.testing.assert_allclose(U, [5.0, 6.0])

    def test_stale_3rd_raises(self):
        ctrl = TfController()
        ctrl.configure_controller({"T": 5, "dU": 2,
                                    "state_datatypes": [], "obs_datatypes": []})
        ctrl.update_action_command(1, np.array([5.0, 6.0]))
        ctrl.get_action(0, np.zeros(2), np.zeros(2))  # fresh
        ctrl.get_action(1, np.zeros(2), np.zeros(2))  # stale 1
        ctrl.get_action(2, np.zeros(2), np.zeros(2))  # stale 2
        with pytest.raises(RuntimeError, match="stale"):
            ctrl.get_action(3, np.zeros(2), np.zeros(2))

    def test_new_command_resets_stale_counter(self):
        ctrl = TfController()
        ctrl.configure_controller({"T": 10, "dU": 2,
                                    "state_datatypes": [], "obs_datatypes": []})
        ctrl.update_action_command(1, np.zeros(2))
        ctrl.get_action(0, np.zeros(2), np.zeros(2))  # fresh (id 1 acted on)
        ctrl.get_action(1, np.zeros(2), np.zeros(2))  # stale 1
        ctrl.get_action(2, np.zeros(2), np.zeros(2))  # stale 2
        # Now send a new command — should work again
        ctrl.update_action_command(2, np.array([7.0, 8.0]))
        U = ctrl.get_action(3, np.zeros(2), np.zeros(2))
        np.testing.assert_allclose(U, [7.0, 8.0])

    def test_publish_obs_calls_plugin(self):
        ctrl = TfController()
        ctrl.configure_controller({"T": 3, "dU": 2,
                                    "state_datatypes": [], "obs_datatypes": []})
        plugin = MagicMock()
        obs = np.array([1.0, 2.0, 3.0])
        ctrl.publish_obs(obs, plugin)
        plugin.tf_publish_obs.assert_called_once()
        np.testing.assert_array_equal(plugin.tf_publish_obs.call_args[0][0], obs)


# ---------------------------------------------------------------------------
# PositionController
# ---------------------------------------------------------------------------

class TestPositionController:
    def _make_plugin(self, n_joints=7, q=None):
        plugin = MagicMock()
        if q is None:
            q = np.zeros(n_joints)
        plugin.get_joint_encoder_readings.return_value = q
        return plugin

    def test_no_control_zero_torques(self):
        ctrl = PositionController(MagicMock(), 0, 7)
        ctrl.configure_controller({"mode": NO_CONTROL})
        torques = np.ones(7)
        plugin = self._make_plugin()
        ctrl.update(plugin, 1.0, MagicMock(), torques)
        np.testing.assert_allclose(torques, np.zeros(7))

    def test_no_control_is_finished(self):
        ctrl = PositionController(MagicMock(), 0, 7)
        ctrl.configure_controller({"mode": NO_CONTROL})
        assert ctrl.is_finished()

    def test_joint_space_not_finished_when_far(self):
        ctrl = PositionController(MagicMock(), 0, 3)
        target = np.array([1.0, 1.0, 1.0])
        pd = np.column_stack([np.ones(3), np.zeros(3), np.zeros(3), np.ones(3)])
        ctrl.configure_controller({"mode": JOINT_SPACE, "data": target, "pd_gains": pd})
        # Plugin returns current angles = 0 → far from target
        plugin = self._make_plugin(n_joints=3, q=np.zeros(3))
        torques = np.zeros(3)
        ctrl.update(plugin, 1.0, MagicMock(), torques)
        ctrl.update(plugin, 2.0, MagicMock(), torques)
        assert not ctrl.is_finished()

    def test_joint_space_finished_when_at_target(self):
        ctrl = PositionController(MagicMock(), 0, 3)
        target = np.zeros(3)
        pd = np.column_stack([np.ones(3), np.zeros(3), np.zeros(3), np.ones(3)])
        ctrl.configure_controller({"mode": JOINT_SPACE, "data": target, "pd_gains": pd})
        # Plugin returns current angles = 0 exactly, velocity also 0
        plugin = self._make_plugin(n_joints=3, q=np.zeros(3))
        torques = np.zeros(3)
        ctrl.update(plugin, 1.0, MagicMock(), torques)
        ctrl.update(plugin, 2.0, MagicMock(), torques)
        assert ctrl.is_finished()

    def test_report_waiting_set_on_configure(self):
        ctrl = PositionController(MagicMock(), 0, 3)
        ctrl.configure_controller({"mode": NO_CONTROL})
        assert ctrl.report_waiting is True

    def test_pd_torques_negative_proportional(self):
        """Torques should oppose position error."""
        ctrl = PositionController(MagicMock(), 0, 2)
        target = np.array([0.0, 0.0])
        pd = np.array([[5.0, 0.0, 0.0, 1.0],
                       [5.0, 0.0, 0.0, 1.0]])
        ctrl.configure_controller({"mode": JOINT_SPACE, "data": target, "pd_gains": pd})
        # Current position > target → positive error → torque should be negative
        plugin = self._make_plugin(n_joints=2, q=np.array([1.0, 1.0]))
        torques = np.zeros(2)
        ctrl.update(plugin, 1.0, MagicMock(), torques)
        ctrl.update(plugin, 2.0, MagicMock(), torques)
        assert torques[0] < 0  # P term: -Kp * (1-0) < 0
        assert torques[1] < 0
