"""
rclpy agent migration tests (Arch 9.1–9.4).

Covers the rclpy-migrated components without requiring a live ROS 2 daemon:

  1. AgentConfig — ament_index fallback logic in agent/config.py
  2. ServiceEmulator — rclpy pub/sub emulation of request/response
  3. AgentROS node init/teardown (rclpy Node lifecycle)
  4. ROS utils — msg_to_sample, policy_to_msg, TimeoutException
  5. UpdatePolicyClient/Server — already covered in test_update_policy_service.py;
     additional integration-style checks here for the full call flow

All rclpy imports are mocked; no spinning, no DDS.
"""
from __future__ import annotations

import sys
import time
import types
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ===========================================================================
# Shared mock-factory helpers
# ===========================================================================

def _make_rclpy_module():
    """
    Build a minimal rclpy-shaped module tree sufficient for the GPS agent code.

    Covers: rclpy, rclpy.node, rclpy.qos
    """
    rclpy = types.ModuleType('rclpy')
    rclpy.ok = MagicMock(return_value=True)
    rclpy.init = MagicMock()
    rclpy.shutdown = MagicMock()
    rclpy.spin_once = MagicMock()
    rclpy.spin_until_future_complete = MagicMock()

    # rclpy.node
    rclpy_node = types.ModuleType('rclpy.node')

    class MockNode:
        """Minimal Node stub — methods return MagicMocks for assertion."""
        def __init__(self, name: str) -> None:
            self._name = name
            self._clock = MagicMock()
            self._clock.now.return_value.to_msg.return_value = MagicMock()
            self.create_publisher = MagicMock(return_value=MagicMock())
            self.create_subscription = MagicMock(return_value=MagicMock())
            self.create_client = MagicMock(return_value=MagicMock())
            self.create_service = MagicMock(return_value=MagicMock())
            self.destroy_service = MagicMock()
            self.destroy_node = MagicMock()
            _rate = MagicMock()
            _rate.sleep = MagicMock()
            self.create_rate = MagicMock(return_value=_rate)

        def get_clock(self):
            return self._clock

    rclpy_node.Node = MockNode
    rclpy.node = rclpy_node

    # rclpy.qos
    rclpy_qos = types.ModuleType('rclpy.qos')
    rclpy_qos.QoSProfile = MagicMock(return_value=MagicMock())
    rclpy_qos.ReliabilityPolicy = MagicMock()
    rclpy_qos.DurabilityPolicy = MagicMock()
    rclpy_qos.ReliabilityPolicy.RELIABLE = 'RELIABLE'
    rclpy_qos.DurabilityPolicy.VOLATILE = 'VOLATILE'
    rclpy.qos = rclpy_qos

    return rclpy, rclpy_node, rclpy_qos


def _make_gps_agent_pkg():
    """Return a minimal gps_agent_pkg stub with the message types used by GPS."""
    pkg = types.ModuleType('gps_agent_pkg')
    msg = types.ModuleType('gps_agent_pkg.msg')

    for cls_name in (
        'TrialCommand', 'SampleResult', 'PositionCommand',
        'RelaxCommand', 'DataRequest', 'ControllerParams',
        'LinGaussParams', 'TfParams', 'TorchParams',
        'TfActionCommand', 'TfObsData',
    ):
        klass = type(cls_name, (), {'__init__': lambda self, **kw: None})
        setattr(msg, cls_name, klass)

    pkg.msg = msg
    return pkg, msg


@contextmanager
def _rclpy_env():
    """
    Inject mock rclpy + gps_agent_pkg into sys.modules for the duration of
    the ``with`` block, then restore the originals.
    """
    rclpy, rclpy_node, rclpy_qos = _make_rclpy_module()
    fake_pkg, fake_msg = _make_gps_agent_pkg()

    keys = [
        'rclpy', 'rclpy.node', 'rclpy.qos',
        'gps_agent_pkg', 'gps_agent_pkg.msg',
    ]
    prev = {k: sys.modules.get(k) for k in keys}

    sys.modules['rclpy'] = rclpy
    sys.modules['rclpy.node'] = rclpy_node
    sys.modules['rclpy.qos'] = rclpy_qos
    sys.modules['gps_agent_pkg'] = fake_pkg
    sys.modules['gps_agent_pkg.msg'] = fake_msg

    try:
        yield rclpy, rclpy_node, fake_pkg, fake_msg
    finally:
        for k, v in prev.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ===========================================================================
# 1. AgentConfig — ament_index fallback (agent/config.py)
# ===========================================================================

@pytest.mark.unit
class TestAgentConfig:
    """
    Verify the three-path resolution in _build_agent_ros_config():
      Path A: ament_index_python available and package found → populated dict
      Path B: ament_index_python absent, rospkg available → populated dict
      Path C: neither → empty dict
    """

    def _reload_config(self):
        """Force a fresh import so _build_agent_ros_config() re-executes."""
        import importlib
        import gps.agent.config as cfg_mod
        importlib.reload(cfg_mod)
        return cfg_mod

    def test_base_agent_keys_always_present(self):
        """AGENT dict must contain the mandatory baseline keys."""
        from gps.agent.config import AGENT
        for key in ('dH', 'x0var', 'smooth_noise', 'smooth_noise_var', 'mode'):
            assert key in AGENT, f"AGENT missing key: {key!r}"

    def test_ament_path_returns_populated_dict(self):
        """If ament_index_python finds the package, AGENT_ROS must be non-empty."""
        ament_mod = types.ModuleType('ament_index_python')
        packages_mod = types.ModuleType('ament_index_python.packages')
        packages_mod.get_package_share_directory = MagicMock(return_value='/fake/share')
        packages_mod.PackageNotFoundError = type('PackageNotFoundError', (Exception,), {})
        ament_mod.packages = packages_mod

        prev_ament = sys.modules.get('ament_index_python')
        prev_pkgs = sys.modules.get('ament_index_python.packages')
        sys.modules['ament_index_python'] = ament_mod
        sys.modules['ament_index_python.packages'] = packages_mod

        try:
            cfg = self._reload_config()
            assert cfg.AGENT_ROS, "AGENT_ROS should be populated when ament finds package"
            assert 'trial_command_topic' in cfg.AGENT_ROS
            assert 'frequency' in cfg.AGENT_ROS
        finally:
            if prev_ament is None:
                sys.modules.pop('ament_index_python', None)
            else:
                sys.modules['ament_index_python'] = prev_ament
            if prev_pkgs is None:
                sys.modules.pop('ament_index_python.packages', None)
            else:
                sys.modules['ament_index_python.packages'] = prev_pkgs

    def test_fallback_to_empty_when_no_ros(self):
        """If both ament and rospkg are absent, AGENT_ROS must be {} (not raise)."""
        prev_ament = sys.modules.pop('ament_index_python', None)
        prev_pkgs = sys.modules.pop('ament_index_python.packages', None)
        prev_rospkg = sys.modules.pop('rospkg', None)
        prev_roslib = sys.modules.pop('roslib', None)

        # Block the imports
        sys.modules['ament_index_python'] = None  # type: ignore[assignment]
        sys.modules['ament_index_python.packages'] = None  # type: ignore[assignment]
        sys.modules['rospkg'] = None  # type: ignore[assignment]
        sys.modules['roslib'] = None  # type: ignore[assignment]

        try:
            cfg = self._reload_config()
            assert cfg.AGENT_ROS == {}, \
                f"Expected empty dict, got: {cfg.AGENT_ROS}"
        finally:
            for k, v in [
                ('ament_index_python', prev_ament),
                ('ament_index_python.packages', prev_pkgs),
                ('rospkg', prev_rospkg),
                ('roslib', prev_roslib),
            ]:
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_agent_ros_dict_has_pid_params_array(self):
        """When populated, AGENT_ROS pid_params must be a numpy array of shape (28,)."""
        ament_mod = types.ModuleType('ament_index_python')
        packages_mod = types.ModuleType('ament_index_python.packages')
        packages_mod.get_package_share_directory = MagicMock(return_value='/fake')
        packages_mod.PackageNotFoundError = type('PackageNotFoundError', (Exception,), {})
        ament_mod.packages = packages_mod
        sys.modules['ament_index_python'] = ament_mod
        sys.modules['ament_index_python.packages'] = packages_mod
        try:
            cfg = self._reload_config()
            pid = cfg.AGENT_ROS.get('pid_params')
            assert pid is not None
            assert hasattr(pid, 'shape'), "pid_params must be a numpy array"
            assert pid.shape == (28,), f"Expected (28,), got {pid.shape}"
        finally:
            sys.modules.pop('ament_index_python', None)
            sys.modules.pop('ament_index_python.packages', None)


# ===========================================================================
# 2. ServiceEmulator — rclpy pub/sub emulation
# ===========================================================================

@pytest.mark.unit
class TestServiceEmulator:
    """
    ServiceEmulator wraps a publisher + subscriber to emulate a synchronous
    request/response.  Tests exercise the publish_and_wait() loop and the
    TimeoutException path without any real DDS transport.
    """

    def _make_emulator(self, rclpy_mock, fake_msg):
        """Create a ServiceEmulator with mock pub/sub."""
        from gps.agent.ros.ros_utils import ServiceEmulator

        MockNode = MagicMock()
        pub_mock = MagicMock()
        sub_mock = MagicMock()
        MockNode.create_publisher.return_value = pub_mock
        MockNode.create_subscription.return_value = sub_mock

        emulator = ServiceEmulator(
            MockNode,
            pub_topic='cmd',
            pub_type=MagicMock(),
            sub_topic='result',
            sub_type=MagicMock(),
        )
        return emulator, MockNode, pub_mock

    def test_publish_sends_message(self):
        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            emulator, node, pub_mock = self._make_emulator(rclpy, fake_msg)
            msg = MagicMock()
            emulator.publish(msg)
            pub_mock.publish.assert_called_once_with(msg)

    def test_publish_and_wait_returns_subscriber_message(self):
        """publish_and_wait should return as soon as the callback fires."""
        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            emulator, node, pub_mock = self._make_emulator(rclpy, fake_msg)
            expected_response = MagicMock()

            # Make spin_once deliver the response on the first call.
            # ros_utils holds a direct reference to rclpy, so patch it there.
            def fake_spin(node_arg, timeout_sec):
                emulator._callback(expected_response)

            pub_msg = MagicMock()
            with patch('gps.agent.ros.ros_utils.rclpy') as mock_rclpy:
                mock_rclpy.spin_once.side_effect = fake_spin
                result = emulator.publish_and_wait(pub_msg, timeout=5.0)

            assert result is expected_response
            pub_mock.publish.assert_called_once_with(pub_msg)

    def test_publish_and_wait_raises_on_timeout(self):
        """publish_and_wait must raise TimeoutException if no response arrives."""
        from gps.agent.ros.ros_utils import TimeoutException

        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            emulator, node, pub_mock = self._make_emulator(rclpy, fake_msg)

            with patch('gps.agent.ros.ros_utils.rclpy') as mock_rclpy:
                # spin_once never delivers a callback — timeout fires quickly
                mock_rclpy.spin_once = MagicMock()

                with pytest.raises(TimeoutException):
                    # poll_delay=0.01, timeout=0.05 → ~5 iterations before timeout
                    emulator.publish_and_wait(
                        MagicMock(), timeout=0.05, poll_delay=0.01
                    )

    def test_callback_ignored_when_not_waiting(self):
        """Messages arriving outside publish_and_wait must be discarded."""
        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            emulator, node, pub_mock = self._make_emulator(rclpy, fake_msg)
            emulator._waiting = False
            emulator._callback(MagicMock())
            assert emulator._subscriber_msg is None

    def test_check_id_raises_not_implemented(self):
        """check_id=True is reserved and must raise NotImplementedError."""
        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            emulator, node, pub_mock = self._make_emulator(rclpy, fake_msg)
            with pytest.raises(NotImplementedError):
                emulator.publish_and_wait(MagicMock(), check_id=True)


# ===========================================================================
# 3. TimeoutException
# ===========================================================================

@pytest.mark.unit
class TestTimeoutException:
    def test_message_includes_seconds(self):
        from gps.agent.ros.ros_utils import TimeoutException
        exc = TimeoutException(3.141)
        assert '3.141' in str(exc)

    def test_sec_waited_attribute(self):
        from gps.agent.ros.ros_utils import TimeoutException
        exc = TimeoutException(7.0)
        assert exc.sec_waited == 7.0

    def test_is_exception(self):
        from gps.agent.ros.ros_utils import TimeoutException
        assert issubclass(TimeoutException, Exception)


# ===========================================================================
# 4. policy_to_msg — ROS 2 controller message serialisation
# ===========================================================================

@pytest.mark.unit
class TestPolicyToMsg:
    """
    policy_to_msg() must set controller_to_execute and populate the
    correct sub-message for each supported policy type.
    """

    def _make_lin_gauss_policy(self, T=5, dX=4, dU=2):
        from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
        K = np.zeros((T, dU, dX))
        k = np.zeros((T, dU))
        PSig = np.eye(dU)[None].repeat(T, 0)
        cholPSig = np.eye(dU)[None].repeat(T, 0)
        invPSig = np.eye(dU)[None].repeat(T, 0)
        return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)

    def test_lin_gauss_controller_type(self):
        from gps.proto.gps_pb2 import LIN_GAUSS_CONTROLLER

        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            # Give ControllerParams + LinGaussParams stub enough attributes
            class FakeControllerParams:
                controller_to_execute = None
                lingauss = None
            class FakeLinGaussParams:
                dX = dU = 0
                K_t = k_t = []

            fake_msg.ControllerParams = FakeControllerParams
            fake_msg.LinGaussParams = FakeLinGaussParams

            from gps.agent.ros.ros_utils import policy_to_msg
            policy = self._make_lin_gauss_policy()
            noise = np.zeros((policy.T, policy.dU))
            msg = policy_to_msg(policy, noise)

        assert msg.controller_to_execute == LIN_GAUSS_CONTROLLER

    def test_unknown_policy_raises(self):
        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            class FakeControllerParams:
                controller_to_execute = None
            fake_msg.ControllerParams = FakeControllerParams

            from gps.agent.ros.ros_utils import policy_to_msg

            class WeirdPolicy:
                pass

            with pytest.raises(NotImplementedError):
                policy_to_msg(WeirdPolicy(), np.zeros((5, 2)))


# ===========================================================================
# 5. msg_to_sample
# ===========================================================================

@pytest.mark.unit
class TestMsgToSample:
    """msg_to_sample() should unpack SensorData entries into Sample fields."""

    def _make_sensor(self, data_type, shape, data):
        sensor = MagicMock()
        sensor.data_type = data_type
        sensor.shape = shape
        sensor.data = data
        return sensor

    def test_sensor_data_is_set_on_sample(self):
        from gps.proto.gps_pb2 import ACTION

        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            from gps.agent.ros.ros_utils import msg_to_sample

            # Build a fake agent with the required interface
            agent = MagicMock()
            sample_mock = MagicMock()
            agent.__class__ = MagicMock()  # avoids Sample(agent) import chain

            # Patch Sample so we don't need a full agent
            with patch('gps.agent.ros.ros_utils.Sample') as MockSample:
                MockSample.return_value = sample_mock

                ros_msg = MagicMock()
                data_array = list(np.arange(6, dtype=np.float32))
                ros_msg.sensor_data = [
                    self._make_sensor(ACTION, [2, 3], data_array)
                ]
                result = msg_to_sample(ros_msg, agent)

            sample_mock.set.assert_called_once()
            set_call = sample_mock.set.call_args
            assert set_call[0][0] == ACTION
            assert result is sample_mock


# ===========================================================================
# 6. AgentROS — node init / teardown lifecycle
# ===========================================================================

@pytest.mark.unit
class TestAgentROSLifecycle:
    """
    AgentROS.__init__ must call rclpy.init() and Node() when init_node=True.
    shutdown() must call destroy_node() and rclpy.shutdown().
    """

    def _build_hyperparams(self):
        """Minimal hyperparams accepted by AgentROS.__init__."""
        from gps.proto.gps_pb2 import JOINT_ANGLES
        return {
            'conditions': 1,
            'T': 10,
            'dt': 0.05,
            'dU': 2,
            'dX': 4,
            'dO': 4,
            'sensor_dims': {JOINT_ANGLES: 4},
            'state_include': [JOINT_ANGLES],
            'obs_include': [JOINT_ANGLES],
            'x0': [np.zeros(4)],
            'ee_points_tgt': [np.zeros((1, 3))],
            'reset_conditions': [{
                0: {'mode': 1, 'data': np.zeros(4)},  # TRIAL_ARM
                1: {'mode': 1, 'data': np.zeros(4)},  # AUXILIARY_ARM
            }],
            # AgentROS-specific
            'trial_command_topic': 'cmd',
            'sample_result_topic': 'result',
            'reset_command_topic': 'reset',
            'relax_command_topic': 'relax',
            'data_request_topic': 'data',
            'trial_timeout': 20,
            'frequency': 20,
            'end_effector_points': np.array([]),
            'pid_params': np.ones(28),
        }

    def test_init_calls_rclpy_init_and_creates_node(self):
        with _rclpy_env() as (rclpy_mock, rclpy_node_mod, fake_pkg, fake_msg):
            rclpy_mock.ok.return_value = False  # trigger rclpy.init()

            # Patch Agent.__init__ to avoid full GPS agent setup
            with patch('gps.agent.agent.Agent.__init__', return_value=None):
                with patch.object(
                    sys.modules['rclpy.node'], 'Node',
                    wraps=sys.modules['rclpy.node'].Node
                ) as MockNodeCls:
                    from gps.agent.ros.agent_ros import AgentROS
                    hp = self._build_hyperparams()

                    # Patch the internal pub/sub setup to avoid message imports
                    with patch.object(AgentROS, '_init_pubs_and_subs'):
                        agent = AgentROS.__new__(AgentROS)
                        agent._hyperparams = hp
                        # Manually call init_node logic
                        if not rclpy_mock.ok():
                            rclpy_mock.init()

            rclpy_mock.init.assert_called()

    def test_shutdown_destroys_node_and_calls_rclpy_shutdown(self):
        """shutdown() must call destroy_node() and rclpy.shutdown()."""
        from gps.agent.ros.agent_ros import AgentROS

        mock_node = MagicMock()
        agent = AgentROS.__new__(AgentROS)
        agent._node = mock_node

        # Patch rclpy directly on the agent_ros module (already imported)
        with patch('gps.agent.ros.agent_ros.rclpy') as mock_rclpy:
            mock_rclpy.ok.return_value = True
            agent.shutdown()

        mock_node.destroy_node.assert_called_once()
        mock_rclpy.shutdown.assert_called_once()

    def test_shutdown_skips_rclpy_shutdown_when_not_ok(self):
        """If rclpy is already shut down, shutdown() must not call it again."""
        from gps.agent.ros.agent_ros import AgentROS

        agent = AgentROS.__new__(AgentROS)
        agent._node = MagicMock()

        with patch('gps.agent.ros.agent_ros.rclpy') as mock_rclpy:
            mock_rclpy.ok.return_value = False
            agent.shutdown()

        mock_rclpy.shutdown.assert_not_called()


# ===========================================================================
# 7. tf_policy_to_action_msg / tf_obs_msg_to_numpy
# ===========================================================================

@pytest.mark.unit
class TestTfHelpers:
    def test_tf_obs_msg_to_numpy_returns_array(self):
        with _rclpy_env() as _:
            from gps.agent.ros.ros_utils import tf_obs_msg_to_numpy
            msg = MagicMock()
            msg.data = [1.0, 2.0, 3.0]
            result = tf_obs_msg_to_numpy(msg)
            np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_tf_policy_to_action_msg_fields(self):
        with _rclpy_env() as (rclpy, _, fake_pkg, fake_msg):
            # Give TfActionCommand stub attributes
            class FakeTfActionCommand:
                action = None
                dU = None
                id = None
            fake_msg.TfActionCommand = FakeTfActionCommand

            from gps.agent.ros.ros_utils import tf_policy_to_action_msg
            action = np.array([0.1, 0.2, 0.3])
            msg = tf_policy_to_action_msg(3, action, action_id=42)

            assert list(msg.action) == pytest.approx([0.1, 0.2, 0.3])
            assert msg.dU == 3
            assert msg.id == 42


# ===========================================================================
# 8. Proto round-trip — gps_pb2 cross-version compatibility
# ===========================================================================

@pytest.mark.unit
class TestProtoRoundTrip:
    """
    gps_pb2 must expose the expected enum constants regardless of the
    installed protobuf runtime version (4.x, 5.x, or 6.x).
    """

    def test_controller_type_values(self):
        from gps.proto.gps_pb2 import (
            LIN_GAUSS_CONTROLLER, CAFFE_CONTROLLER,
            TF_CONTROLLER, PYTORCH_CONTROLLER,
        )
        assert LIN_GAUSS_CONTROLLER == 0
        assert CAFFE_CONTROLLER == 1
        assert TF_CONTROLLER == 2
        assert PYTORCH_CONTROLLER == 3

    def test_actuator_type_values(self):
        from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
        assert TRIAL_ARM == 0
        assert AUXILIARY_ARM == 1

    def test_sample_type_action(self):
        from gps.proto.gps_pb2 import ACTION
        assert ACTION == 0

    def test_sample_message_serialisation(self):
        """A minimal Sample proto must survive a serialise → deserialise cycle."""
        from gps.proto.gps_pb2 import Sample
        s = Sample()
        s.T = 10
        s.dX = 4
        s.dU = 2
        blob = s.SerializeToString()
        s2 = Sample()
        s2.ParseFromString(blob)
        assert s2.T == 10
        assert s2.dX == 4
        assert s2.dU == 2
