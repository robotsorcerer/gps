"""
ROS 2 (Humble) agent for the PR2 robot environment.

Migrated from ROS 1 (rospy) to ROS 2 (rclpy) for Humble compatibility.

Key changes from ROS 1:
- rospy.init_node -> rclpy.init() + Node class
- rospy.Publisher/Subscriber -> node.create_publisher/create_subscription
- rospy.get_rostime() -> node.get_clock().now()
- rospy.Rate -> rclpy timer or manual timing
- rospy.sleep -> time.sleep or executor spinning
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_ROS
from gps.agent.ros2.ros2_utils import (
    ServiceEmulator,
    msg_to_sample,
    policy_to_msg,
    tf_policy_to_action_msg,
    tf_obs_msg_to_numpy,
)
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

# Optional TensorFlow policy import
try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:
    TfPolicy = None

# Note: These message types need to be ported to ROS 2 message definitions
# from gps_agent_pkg_msgs.msg import (
#     TrialCommand, SampleResult, PositionCommand,
#     RelaxCommand, DataRequest, TfActionCommand, TfObsData
# )

# Placeholder message classes for type hints
class TrialCommand:
    pass

class SampleResult:
    pass

class PositionCommand:
    pass

class RelaxCommand:
    pass

class DataRequest:
    pass

class TfActionCommand:
    pass

class TfObsData:
    pass


class AgentROS2(Agent, Node):
    """
    ROS 2 agent for GPS algorithm communication.

    All communication between the algorithms and ROS 2 is done through
    this class. Inherits from both Agent (for GPS interface) and Node
    (for ROS 2 functionality).

    Args:
        hyperparams: Dictionary of hyperparameters
        init_rclpy: Whether to initialize rclpy (set False if already init)
        node_name: Name for the ROS 2 node
    """

    def __init__(
        self,
        hyperparams: Dict[str, Any],
        init_rclpy: bool = True,
        node_name: str = 'gps_agent_ros2_node'
    ) -> None:
        # Merge with default config
        config = copy.deepcopy(AGENT_ROS)
        config.update(hyperparams)

        # Initialize Agent base class
        Agent.__init__(self, config)

        # Initialize rclpy if requested
        if init_rclpy:
            rclpy.init()

        # Initialize Node base class
        Node.__init__(self, node_name)

        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands

        # Setup conditions
        conditions = self._hyperparams['conditions']
        self.x0: List[Any] = []

        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(
                self._hyperparams[field], conditions
            )
        self.x0 = self._hyperparams['x0']

        # Initial sleep to let publishers/subscribers establish
        time.sleep(1.0)

        self.use_tf = False
        self.observations_stale = True
        self._tf_subscriber_msg: Optional[Any] = None
        self.current_action_id = 0
        self.dU = 0

        self.get_logger().info('AgentROS2 initialized successfully')

    def _init_pubs_and_subs(self) -> None:
        """Initialize ROS 2 publishers and subscribers."""
        # QoS profile for reliable service-like communication
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self._trial_service = ServiceEmulator(
            self,
            self._hyperparams['trial_command_topic'], TrialCommand,
            self._hyperparams['sample_result_topic'], SampleResult,
            qos
        )
        self._reset_service = ServiceEmulator(
            self,
            self._hyperparams['reset_command_topic'], PositionCommand,
            self._hyperparams['sample_result_topic'], SampleResult,
            qos
        )
        self._relax_service = ServiceEmulator(
            self,
            self._hyperparams['relax_command_topic'], RelaxCommand,
            self._hyperparams['sample_result_topic'], SampleResult,
            qos
        )
        self._data_service = ServiceEmulator(
            self,
            self._hyperparams['data_request_topic'], DataRequest,
            self._hyperparams['sample_result_topic'], SampleResult,
            qos
        )

    def _get_next_seq_id(self) -> int:
        """Get next sequence ID for message tracking."""
        self._seq_id = (self._seq_id + 1) % (2 ** 32)
        return self._seq_id

    def get_data(self, arm: int = TRIAL_ARM) -> Any:
        """
        Request the most recent value for data/sensor readings.

        Args:
            arm: TRIAL_ARM or AUXILIARY_ARM

        Returns:
            Sample object with all available data
        """
        request = DataRequest()
        request.id = self._get_next_seq_id()
        request.arm = arm
        request.stamp = self.get_clock().now().to_msg()

        result_msg = self._data_service.publish_and_wait(request)
        sample = msg_to_sample(result_msg, self)
        return sample

    def relax_arm(self, arm: int) -> None:
        """
        Relax one of the robot arms.

        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM
        """
        relax_command = RelaxCommand()
        relax_command.id = self._get_next_seq_id()
        relax_command.stamp = self.get_clock().now().to_msg()
        relax_command.arm = arm
        self._relax_service.publish_and_wait(relax_command)

    def reset_arm(self, arm: int, mode: int, data: List[float]) -> None:
        """
        Issue a position command to an arm.

        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM
            mode: Integer code defined in gps_pb2
            data: Array of floats for position
        """
        reset_command = PositionCommand()
        reset_command.mode = mode
        reset_command.data = data
        reset_command.pd_gains = self._hyperparams['pid_params']
        reset_command.arm = arm
        reset_command.id = self._get_next_seq_id()

        timeout = self._hyperparams['trial_timeout']
        self._reset_service.publish_and_wait(reset_command, timeout=timeout)

    def reset(self, condition: int) -> None:
        """
        Reset the agent for a particular experiment condition.

        Args:
            condition: Index into hyperparams['reset_conditions']
        """
        condition_data = self._hyperparams['reset_conditions'][condition]

        self.reset_arm(
            TRIAL_ARM,
            condition_data[TRIAL_ARM]['mode'],
            condition_data[TRIAL_ARM]['data']
        )
        self.reset_arm(
            AUXILIARY_ARM,
            condition_data[AUXILIARY_ARM]['mode'],
            condition_data[AUXILIARY_ARM]['data']
        )

        # Wait for robot to stop completely
        time.sleep(2.0)

    def sample(
        self,
        policy: Any,
        condition: int,
        verbose: bool = True,
        save: bool = True,
        noisy: bool = True
    ) -> Any:
        """
        Reset and execute a policy, collecting a sample.

        Args:
            policy: A Policy object
            condition: Which condition setup to run
            verbose: Unused for this agent
            save: Whether to store the trial into samples
            noisy: Whether to use noise during sampling

        Returns:
            Sample object from trial execution
        """
        if TfPolicy is not None and isinstance(policy, TfPolicy):
            self._init_tf(policy.dU)

        self.reset(condition)

        # Generate noise
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Build trial command
        trial_command = TrialCommand()
        trial_command.id = self._get_next_seq_id()
        trial_command.controller = policy_to_msg(policy, noise)
        trial_command.T = self.T
        trial_command.frequency = self._hyperparams['frequency']

        ee_points = self._hyperparams['end_effector_points']
        trial_command.ee_points = ee_points.reshape(ee_points.size).tolist()
        trial_command.ee_points_tgt = \
            self._hyperparams['ee_points_tgt'][condition].tolist()
        trial_command.state_datatypes = self._hyperparams['state_include']
        trial_command.obs_datatypes = self._hyperparams['state_include']

        # Execute trial
        if not self.use_tf:
            sample_msg = self._trial_service.publish_and_wait(
                trial_command,
                timeout=self._hyperparams['trial_timeout']
            )
            sample = msg_to_sample(sample_msg, self)
        else:
            self._trial_service.publish(trial_command)
            sample_msg = self.run_trial_tf(
                policy,
                time_to_run=self._hyperparams['trial_timeout']
            )
            sample = msg_to_sample(sample_msg, self)

        if save:
            self._samples[condition].append(sample)

        return sample

    def run_trial_tf(self, policy: Any, time_to_run: float = 5.0) -> Any:
        """
        Run an async TensorFlow controller.

        The async controller receives observations from ROS 2 subscriptions
        and publishes actions.

        Args:
            policy: TensorFlow policy to execute
            time_to_run: Maximum trial duration in seconds

        Returns:
            SampleResult message from trial
        """
        should_stop = False
        consecutive_failures = 0
        start_time = time.time()

        while not should_stop:
            # Spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0.01)

            if not self.observations_stale:
                consecutive_failures = 0
                last_obs = tf_obs_msg_to_numpy(self._tf_subscriber_msg)
                action = self._get_new_action(policy, last_obs)
                action_msg = tf_policy_to_action_msg(
                    self.dU, action, self.current_action_id
                )
                self._tf_publish(action_msg)
                self.observations_stale = True
                self.current_action_id += 1
            else:
                consecutive_failures += 1
                elapsed = time.time() - start_time
                if elapsed > time_to_run and consecutive_failures > 5:
                    should_stop = True

        # Wait for finished trial message
        time.sleep(0.25)
        return self._trial_service._subscriber_msg

    def _get_new_action(self, policy: Any, obs: np.ndarray) -> np.ndarray:
        """Get action from policy given observation."""
        return policy.act(None, obs, None, None)

    def _tf_callback(self, message: Any) -> None:
        """Handle TensorFlow observation messages."""
        self._tf_subscriber_msg = message
        self.observations_stale = False

    def _tf_publish(self, pub_msg: Any) -> None:
        """Publish TensorFlow action message."""
        self._tf_pub.publish(pub_msg)

    def _init_tf(self, dU: int) -> None:
        """
        Initialize TensorFlow controller communication.

        Args:
            dU: Action dimensionality
        """
        self._tf_subscriber_msg = None
        self.observations_stale = True
        self.current_action_id = 1
        self.dU = dU

        if not self.use_tf:
            # QoS for TF communication
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            self._tf_pub = self.create_publisher(
                TfActionCommand,
                '/gps_controller_sent_robot_action_tf',
                qos
            )
            self._tf_sub = self.create_subscription(
                TfObsData,
                '/gps_obs_tf',
                self._tf_callback,
                qos
            )

            # Wait for publisher/subscriber to establish
            time.sleep(2.0)

        self.use_tf = True
        self.observations_stale = True

    def destroy_node(self) -> None:
        """Clean up ROS 2 resources."""
        self._trial_service.destroy()
        self._reset_service.destroy()
        self._relax_service.destroy()
        self._data_service.destroy()

        if hasattr(self, '_tf_pub'):
            self.destroy_publisher(self._tf_pub)
        if hasattr(self, '_tf_sub'):
            self.destroy_subscription(self._tf_sub)

        super().destroy_node()

    def shutdown(self) -> None:
        """Shutdown the agent and rclpy."""
        self.destroy_node()
        rclpy.shutdown()
