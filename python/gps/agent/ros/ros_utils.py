"""
ROS 2 utilities for the GPS ROS agent.

Key changes from the ROS 1 version:
  - rospy.Publisher   → rclpy node.create_publisher()
  - rospy.Subscriber  → rclpy node.create_subscription()
  - rospy.sleep()     → rclpy.spin_once() with timeout
  - TimeoutException  unchanged (pure Python)
  - CaffePolicy path  retained but unconditionally guarded (deprecated)
  - PyTorchPolicy     added to policy_to_msg()
"""
from __future__ import annotations

import time
import logging
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import (
    LIN_GAUSS_CONTROLLER, CAFFE_CONTROLLER, TF_CONTROLLER, PYTORCH_CONTROLLER,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QoS profile — mirrors the one in agent_ros.py (reliable, volatile, depth=1)
# ---------------------------------------------------------------------------
_GPS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=1,
)

# ---------------------------------------------------------------------------
# Optional legacy policy imports
# ---------------------------------------------------------------------------
try:
    from gps.algorithm.policy.caffe_policy import CaffePolicy
    _NO_CAFFE = False
except ImportError:
    _NO_CAFFE = True
    LOGGER.info('Caffe not imported (expected — Caffe is deprecated)')

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:
    TfPolicy = None

try:
    from gps.algorithm.policy.pytorch_policy import PyTorchPolicy
except ImportError:
    PyTorchPolicy = None


# ---------------------------------------------------------------------------
# Message ↔ Sample conversion
# ---------------------------------------------------------------------------

def msg_to_sample(ros_msg: Any, agent: Any) -> Sample:
    """
    Convert a SampleResult ROS 2 message into a GPS Sample object.

    Args:
        ros_msg: gps_agent_pkg/SampleResult message.
        agent:   AgentROS instance (provides dimension metadata).

    Returns:
        Sample populated with all sensor data in the message.
    """
    sample = Sample(agent)
    for sensor in ros_msg.sensor_data:
        sensor_id = sensor.data_type
        shape = np.array(sensor.shape)
        data = np.array(sensor.data).reshape(shape)
        sample.set(sensor_id, data)
    return sample


def policy_to_msg(policy: Any, noise: np.ndarray) -> Any:
    """
    Convert a GPS policy object into a ROS 2 ControllerParams message.

    Supported policy types:
        LinearGaussianPolicy  → LIN_GAUSS_CONTROLLER
        CaffePolicy           → CAFFE_CONTROLLER  (deprecated)
        TfPolicy              → TF_CONTROLLER     (legacy)
        PyTorchPolicy         → PYTORCH_CONTROLLER

    Args:
        policy: A Policy object.
        noise:  Exploration noise array [T, dU].

    Returns:
        gps_agent_pkg/ControllerParams message.

    Raises:
        NotImplementedError: if the policy type is unrecognised.
    """
    from gps_agent_pkg.msg import ControllerParams, LinGaussParams, TfParams

    msg = ControllerParams()

    if isinstance(policy, LinearGaussianPolicy):
        msg.controller_to_execute = LIN_GAUSS_CONTROLLER
        msg.lingauss = LinGaussParams()
        msg.lingauss.dX = policy.dX
        msg.lingauss.dU = policy.dU
        msg.lingauss.K_t = (
            policy.K.reshape(policy.T * policy.dX * policy.dU).tolist()
        )
        msg.lingauss.k_t = (
            policy.fold_k(noise).reshape(policy.T * policy.dU).tolist()
        )

    elif not _NO_CAFFE and isinstance(policy, CaffePolicy):
        from gps_agent_pkg.msg import CaffeParams
        msg.controller_to_execute = CAFFE_CONTROLLER
        msg.caffe = CaffeParams()
        msg.caffe.net_param = policy.get_net_param()
        msg.caffe.bias = policy.bias.tolist()
        msg.caffe.dU = policy.dU
        scale_shape = policy.scale.shape
        msg.caffe.scale = (
            policy.scale.reshape(scale_shape[0] * scale_shape[1]).tolist()
        )
        msg.caffe.dim_bias = scale_shape[0]
        scaled_noise = np.zeros_like(noise)
        for i in range(noise.shape[0]):
            scaled_noise[i] = policy.chol_pol_covar.T.dot(noise[i])
        msg.caffe.noise = scaled_noise.reshape(-1).tolist()

    elif TfPolicy is not None and isinstance(policy, TfPolicy):
        msg.controller_to_execute = TF_CONTROLLER
        msg.tf = TfParams()
        msg.tf.dU = policy.dU

    elif PyTorchPolicy is not None and isinstance(policy, PyTorchPolicy):
        from gps_agent_pkg.msg import TorchParams
        # Delegate serialisation to PolicyOptPyTorch so the scale/bias/noise
        # fields are populated correctly alongside the TorchScript bytes.
        params = policy.get_torch_params_dict() if hasattr(policy, 'get_torch_params_dict') else {}
        msg.controller_to_execute = PYTORCH_CONTROLLER
        torch_msg = TorchParams()
        torch_msg.model_bytes = list(params.get('model_bytes', b''))
        torch_msg.torch_version = params.get('torch_version', '')
        torch_msg.scale = params.get('scale', [1.0] * policy.dU)
        torch_msg.bias = params.get('bias', [0.0] * policy.dU)
        torch_msg.dim_bias = params.get('dim_bias', policy.dU)
        torch_msg.dU = policy.dU
        # Flatten noise: [T, dU] → [T*dU]
        torch_msg.noise = noise.reshape(-1).tolist()
        msg.torch = torch_msg

    else:
        raise NotImplementedError(
            f'Unrecognised policy type for ROS 2 controller message: '
            f'{type(policy).__name__}'
        )

    return msg


def tf_policy_to_action_msg(deg_action: int, action: np.ndarray,
                             action_id: int) -> Any:
    """
    Convert an action array to a TfActionCommand message.

    Args:
        deg_action: Dimension of the action space (dU).
        action:     Action vector [dU].
        action_id:  Monotonically increasing counter for synchronisation.

    Returns:
        gps_agent_pkg/TfActionCommand message.
    """
    from gps_agent_pkg.msg import TfActionCommand
    msg = TfActionCommand()
    msg.action = action.tolist()
    msg.dU = deg_action
    msg.id = action_id
    return msg


def tf_obs_msg_to_numpy(obs_message: Any) -> np.ndarray:
    """
    Convert a TfObsData ROS 2 message to a flat NumPy array.

    Args:
        obs_message: gps_agent_pkg/TfObsData message.

    Returns:
        Observation array.
    """
    return np.array(obs_message.data)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TimeoutException(Exception):
    """Raised when publish_and_wait exceeds its timeout budget."""

    def __init__(self, sec_waited: float) -> None:
        super().__init__(f'Timed out after {sec_waited:.3f} seconds')
        self.sec_waited = sec_waited


# ---------------------------------------------------------------------------
# ServiceEmulator
# ---------------------------------------------------------------------------

class ServiceEmulator:
    """
    Emulates a synchronous ROS service via a publisher + subscriber pair.

    In ROS 2 the GPS C++ plugin still uses a topic-based protocol (not a
    proper ROS 2 service) for TrialCommand / SampleResult, so this wrapper
    is preserved.  The only change from the ROS 1 version is that:

      - rospy.Publisher  → node.create_publisher()
      - rospy.Subscriber → node.create_subscription()
      - rospy.sleep()    → rclpy.spin_once(node, timeout_sec=poll_delay)

    Args:
        node:      The rclpy Node that owns the pub/sub handles.
        pub_topic: Topic to publish commands on.
        pub_type:  ROS 2 message type for the publisher.
        sub_topic: Topic to subscribe for responses on.
        sub_type:  ROS 2 message type for the subscriber.
    """

    def __init__(self, node: Node,
                 pub_topic: str, pub_type: Any,
                 sub_topic: str, sub_type: Any) -> None:
        self._node = node
        self._pub = node.create_publisher(pub_type, pub_topic, _GPS_QOS)
        self._sub = node.create_subscription(
            sub_type, sub_topic, self._callback, _GPS_QOS
        )
        self._waiting: bool = False
        self._subscriber_msg: Any = None

    def _callback(self, message: Any) -> None:
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg: Any) -> None:
        """Publish a message without waiting for a response."""
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg: Any, timeout: float = 5.0,
                         poll_delay: float = 0.01,
                         check_id: bool = False) -> Any:
        """
        Publish a message and block until the subscriber receives a response.

        Args:
            pub_msg:     Message to publish.
            timeout:     Maximum seconds to wait before raising TimeoutException.
            poll_delay:  Seconds per spin_once call.
            check_id:    Reserved; raises NotImplementedError if True.

        Returns:
            The subscriber message received in response.

        Raises:
            NotImplementedError: if check_id is True (not yet supported).
            TimeoutException:    if no response arrives within timeout.
        """
        if check_id:
            raise NotImplementedError('check_id is not yet implemented in C++')

        self._waiting = True
        self.publish(pub_msg)

        time_waited = 0.0
        while self._waiting:
            rclpy.spin_once(self._node, timeout_sec=poll_delay)
            time_waited += poll_delay
            if time_waited > timeout:
                self._waiting = False
                raise TimeoutException(time_waited)

        return self._subscriber_msg
