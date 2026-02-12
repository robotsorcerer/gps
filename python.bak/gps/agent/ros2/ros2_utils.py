"""
ROS 2 (rclpy) utilities for GPS agents.

Migrated from ROS 1 (rospy) to ROS 2 Humble with rclpy.

Key changes from ROS 1:
- rospy -> rclpy
- rospy.Publisher -> rclpy.publisher
- rospy.Subscriber -> rclpy.subscription
- rospy.sleep -> time.sleep or executor spinning
- rospy.Rate -> rclpy timer or manual timing
- Message types from custom ROS 2 package
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Type, TypeVar

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.sample.sample import Sample

# Import ROS 2 message types (these would need to be generated for ROS 2)
# from gps_agent_pkg_msgs.msg import (
#     ControllerParams, LinGaussParams, TfParams, CaffeParams, TfActionCommand
# )
from gps.proto.gps_pb2 import LIN_GAUSS_CONTROLLER, CAFFE_CONTROLLER, TF_CONTROLLER

LOGGER = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from gps.algorithm.policy.caffe_policy import CaffePolicy
    NO_CAFFE = False
except ImportError:
    NO_CAFFE = True
    LOGGER.info("Caffe not imported")

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:
    TfPolicy = None

MsgT = TypeVar('MsgT')


class TimeoutException(Exception):
    """Exception thrown on timeouts."""

    def __init__(self, sec_waited: float) -> None:
        super().__init__(f"Timed out after {sec_waited:.2f} seconds")
        self.sec_waited = sec_waited


def msg_to_sample(ros_msg: Any, agent: Any) -> Sample:
    """
    Convert a SampleResult ROS 2 message into a Sample Python object.

    Args:
        ros_msg: ROS 2 SampleResult message
        agent: GPS agent instance

    Returns:
        Sample object populated with sensor data
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
    Convert a policy object to a ROS 2 ControllerParams message.

    Args:
        policy: Policy object (LinearGaussianPolicy, CaffePolicy, or TfPolicy)
        noise: Noise array of shape (T, dU)

    Returns:
        ControllerParams message

    Raises:
        NotImplementedError: If policy type is unknown or Caffe not available
    """
    # Note: ControllerParams message type needs to be defined for ROS 2
    # This is a placeholder showing the structure
    class ControllerParams:
        def __init__(self):
            self.controller_to_execute = 0
            self.lingauss = None
            self.caffe = None
            self.tf = None

    class LinGaussParams:
        def __init__(self):
            self.dX = 0
            self.dU = 0
            self.K_t = []
            self.k_t = []

    class TfParams:
        def __init__(self):
            self.dU = 0

    class CaffeParams:
        def __init__(self):
            self.net_param = b''
            self.bias = []
            self.dU = 0
            self.scale = []
            self.dim_bias = 0
            self.noise = []

    msg = ControllerParams()

    if isinstance(policy, LinearGaussianPolicy):
        msg.controller_to_execute = LIN_GAUSS_CONTROLLER
        msg.lingauss = LinGaussParams()
        msg.lingauss.dX = policy.dX
        msg.lingauss.dU = policy.dU
        msg.lingauss.K_t = policy.K.reshape(
            policy.T * policy.dX * policy.dU
        ).tolist()
        msg.lingauss.k_t = policy.fold_k(noise).reshape(
            policy.T * policy.dU
        ).tolist()

    elif not NO_CAFFE and isinstance(policy, CaffePolicy):
        msg.controller_to_execute = CAFFE_CONTROLLER
        msg.caffe = CaffeParams()
        msg.caffe.net_param = policy.get_net_param()
        msg.caffe.bias = policy.bias.tolist()
        msg.caffe.dU = policy.dU
        scale_shape = policy.scale.shape
        msg.caffe.scale = policy.scale.reshape(
            scale_shape[0] * scale_shape[1]
        ).tolist()
        msg.caffe.dim_bias = scale_shape[0]
        scaled_noise = np.zeros_like(noise)
        for i in range(noise.shape[0]):
            scaled_noise[i] = policy.chol_pol_covar.T.dot(noise[i])
        msg.caffe.noise = scaled_noise.reshape(-1).tolist()

    elif TfPolicy is not None and isinstance(policy, TfPolicy):
        msg.controller_to_execute = TF_CONTROLLER
        msg.tf = TfParams()
        msg.tf.dU = policy.dU

    else:
        raise NotImplementedError(
            f"Caffe not imported or unknown policy object: {policy}"
        )

    return msg


def tf_policy_to_action_msg(deg_action: int, action: np.ndarray,
                            action_id: int) -> Any:
    """
    Convert an action to a TfActionCommand message.

    Args:
        deg_action: Action dimensionality (dU)
        action: Action array
        action_id: Unique action identifier

    Returns:
        TfActionCommand message
    """
    class TfActionCommand:
        def __init__(self):
            self.action = []
            self.dU = 0
            self.id = 0

    msg = TfActionCommand()
    msg.action = action.tolist()
    msg.dU = deg_action
    msg.id = action_id
    return msg


def tf_obs_msg_to_numpy(obs_message: Any) -> np.ndarray:
    """
    Convert a TfObsData message to a numpy array.

    Args:
        obs_message: TfObsData ROS 2 message

    Returns:
        Observation as numpy array
    """
    return np.array(obs_message.data)


class ServiceEmulator:
    """
    Emulates a ROS 2 service (request-response) from a
    publisher-subscriber pair using rclpy.

    This is a ROS 2 port of the ROS 1 ServiceEmulator that uses
    rclpy publishers and subscriptions.

    Args:
        node: ROS 2 node instance
        pub_topic: Publisher topic name
        pub_type: Publisher message type class
        sub_topic: Subscriber topic name
        sub_type: Subscriber message type class
        qos_profile: Optional QoS profile (defaults to reliable)
    """

    def __init__(
        self,
        node: Node,
        pub_topic: str,
        pub_type: Type[MsgT],
        sub_topic: str,
        sub_type: Type[MsgT],
        qos_profile: Optional[QoSProfile] = None
    ) -> None:
        self._node = node

        # Default QoS: reliable delivery for service-like behavior
        if qos_profile is None:
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )

        self._pub = node.create_publisher(pub_type, pub_topic, qos_profile)
        self._sub = node.create_subscription(
            sub_type, sub_topic, self._callback, qos_profile
        )

        self._waiting = False
        self._subscriber_msg: Optional[Any] = None

    def _callback(self, message: Any) -> None:
        """Handle incoming subscription messages."""
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg: Any) -> None:
        """
        Publish a message without waiting for response.

        Args:
            pub_msg: Message to publish
        """
        self._pub.publish(pub_msg)

    def publish_and_wait(
        self,
        pub_msg: Any,
        timeout: float = 5.0,
        poll_delay: float = 0.01,
        check_id: bool = False
    ) -> Any:
        """
        Publish a message and wait for the response.

        Args:
            pub_msg: Message to publish
            timeout: Timeout in seconds
            poll_delay: Polling interval in seconds
            check_id: If True, only return messages with matching id

        Returns:
            Response message from subscriber

        Raises:
            NotImplementedError: If check_id is True (not yet implemented)
            TimeoutException: If response not received within timeout
        """
        if check_id:
            raise NotImplementedError(
                "ID checking not yet implemented in C++ controller"
            )

        self._waiting = True
        self._subscriber_msg = None
        self.publish(pub_msg)

        time_waited = 0.0
        while self._waiting:
            # Spin once to process callbacks
            rclpy.spin_once(self._node, timeout_sec=poll_delay)
            time_waited += poll_delay
            if time_waited > timeout:
                self._waiting = False
                raise TimeoutException(time_waited)

        return self._subscriber_msg

    def destroy(self) -> None:
        """Clean up publisher and subscription."""
        self._node.destroy_publisher(self._pub)
        self._node.destroy_subscription(self._sub)
