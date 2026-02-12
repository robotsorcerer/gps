"""
Tests for RobotPlugin with mocked rclpy node.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from gps_agent_pkg.position_controller import NO_CONTROL, JOINT_SPACE
from gps_agent_pkg.sample import ControllerSample


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

def _make_concrete_plugin():
    """
    Build a concrete RobotPlugin subclass backed by mock rclpy infrastructure.
    """
    from gps_agent_pkg.robot_plugin import RobotPlugin

    class ConcretePlugin(RobotPlugin):
        def __init__(self, node):
            self._joint_positions = np.zeros(7, dtype=np.float64)
            super().__init__(node)

        def get_joint_encoder_readings(self, arm, out=None):
            q = self._joint_positions.copy()
            if out is not None:
                out[:] = q
            return q

        def get_fk_solver(self, arm):
            return None, None

    node = MagicMock()
    node.get_parameter.side_effect = Exception("no param")
    plugin = ConcretePlugin(node)
    return plugin


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRobotPluginInit:
    def test_creates_active_arm_controller(self):
        plugin = _make_concrete_plugin()
        assert plugin._active_arm_controller is not None

    def test_creates_passive_arm_controller(self):
        plugin = _make_concrete_plugin()
        assert plugin._passive_arm_controller is not None

    def test_sensors_initialized(self):
        plugin = _make_concrete_plugin()
        assert plugin._sensors_initialized

    def test_sensors_list_not_empty(self):
        plugin = _make_concrete_plugin()
        assert len(plugin._sensors) > 0

    def test_current_sample_created(self):
        plugin = _make_concrete_plugin()
        assert plugin._current_sample is not None
        assert isinstance(plugin._current_sample, ControllerSample)


class TestRobotPluginPositionCallback:
    def test_active_arm_configured_on_position_command(self):
        plugin = _make_concrete_plugin()

        msg = MagicMock()
        msg.arm = 0  # TRIAL_ARM
        msg.mode = NO_CONTROL
        msg.data = [0.0] * 7
        msg.pd_gains = [1.0, 0.0, 0.0, 1.0] * 7  # 7 joints × 4 gains

        plugin._position_callback(msg)
        assert plugin._active_arm_controller._mode == NO_CONTROL

    def test_passive_arm_configured_on_position_command(self):
        plugin = _make_concrete_plugin()

        msg = MagicMock()
        msg.arm = 1  # AUXILIARY_ARM
        msg.mode = NO_CONTROL
        msg.data = [0.0] * 7
        msg.pd_gains = [1.0, 0.0, 0.0, 1.0] * 7

        plugin._position_callback(msg)
        assert plugin._passive_arm_controller._mode == NO_CONTROL


class TestRobotPluginRelaxCallback:
    def test_relax_sets_no_control(self):
        plugin = _make_concrete_plugin()

        msg = MagicMock()
        msg.arm = 0  # TRIAL_ARM
        plugin._relax_callback(msg)
        assert plugin._active_arm_controller._mode == NO_CONTROL


class TestRobotPluginDataRequestCallback:
    def test_trial_arm_request(self):
        plugin = _make_concrete_plugin()
        msg = MagicMock()
        msg.arm = 0
        plugin._data_request_callback(msg)
        assert plugin._trial_data_request_waiting

    def test_aux_arm_request(self):
        plugin = _make_concrete_plugin()
        msg = MagicMock()
        msg.arm = 1
        plugin._data_request_callback(msg)
        assert plugin._aux_data_request_waiting


class TestRobotPluginUpdateControllers:
    def test_update_controllers_no_trial(self):
        plugin = _make_concrete_plugin()
        # With no trial controller, update_controllers should not raise
        torques_before = plugin._active_arm_torques.copy()
        plugin.update_controllers(1.0, True)
        # Torques should be zero (NO_CONTROL mode)
        np.testing.assert_allclose(plugin._active_arm_torques, torques_before)

    def test_update_sensors_doesnt_crash_without_sensors_init(self):
        plugin = _make_concrete_plugin()
        plugin._sensors_initialized = False
        plugin.update_sensors(1.0, True)  # should return early without crash


class TestRobotPluginLinearGaussianTrial:
    def test_trial_callback_configures_lg_controller(self):
        plugin = _make_concrete_plugin()

        T, dX, dU = 3, 4, 2

        # Build a minimal TrialCommand mock
        lingauss = MagicMock()
        lingauss.dX = dX
        lingauss.dU = dU
        lingauss.K_t = list(np.zeros(T * dU * dX))
        lingauss.k_t = list(np.zeros(T * dU))

        controller = MagicMock()
        controller.controller_to_execute = 1  # LIN_GAUSS_CONTROLLER
        controller.lingauss = lingauss

        msg = MagicMock()
        msg.T = T
        msg.frequency = 20.0
        msg.state_datatypes = [0]
        msg.obs_datatypes   = [1]
        msg.controller      = controller
        msg.ee_points     = [0.0] * 3
        msg.ee_points_tgt = [0.0] * 3

        plugin._trial_callback(msg)

        assert plugin._trial_controller is not None
        assert plugin._controller_initialized

    def test_tf_action_callback_ignored_without_trial(self):
        plugin = _make_concrete_plugin()
        msg = MagicMock()
        msg.dU = 2
        msg.id = 1
        msg.action = [1.0, 2.0]
        # Should not raise even with no trial controller
        plugin._tf_action_callback(msg)
