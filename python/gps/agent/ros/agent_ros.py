"""
Agent for the PR2 robot via ROS 2 (rclpy).

Replaces the original rospy-based implementation with rclpy equivalents:
  - rospy.init_node          → rclpy.init() + Node.__init__()
  - rospy.Publisher          → node.create_publisher()
  - rospy.Subscriber         → node.create_subscription()
  - rospy.Rate / rospy.sleep → node.create_rate() / rclpy.spin_once()
  - rospy.get_rostime()      → node.get_clock().now()
  - rospy.ServiceProxy       → node.create_client()  (via UpdatePolicyClient)

The ServiceEmulator pattern (pub + sub emulating request/response) is preserved
unchanged in structure — it is a GPS-level protocol, not ROS 1 specific.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_ROS
from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
    policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:
    TfPolicy = None


# ---------------------------------------------------------------------------
# QoS profile for GPS command topics: reliable, volatile (non-latched)
# ---------------------------------------------------------------------------
_GPS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=1,
)


class AgentROS(Agent):
    """
    All communication between the GPS algorithms and the PR2 robot is done
    through this class using ROS 2 / rclpy.
    """

    def __init__(self, hyperparams: dict, init_node: bool = True) -> None:
        """
        Initialize agent.

        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node:   Whether to initialise a new rclpy context and Node.
                         Pass False when the caller already owns the context
                         (e.g. inside a larger ROS 2 application).
        """
        config = copy.deepcopy(AGENT_ROS)
        config.update(hyperparams)
        Agent.__init__(self, config)

        if init_node:
            if not rclpy.ok():
                rclpy.init()
            self._node: Node = Node('gps_agent_ros_node')
        else:
            # Caller must supply a pre-constructed node via hyperparams['node']
            self._node = self._hyperparams['node']

        self._init_pubs_and_subs()
        self._seq_id = 0

        conditions = self._hyperparams['conditions']
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field], conditions)
        self.x0 = self._hyperparams['x0']

        # Let the publisher/subscriber infrastructure settle for one second.
        rate = self._node.create_rate(1.0)
        rate.sleep()

        self.use_tf = False
        self.observations_stale = True

    # ------------------------------------------------------------------
    # Pub / sub initialisation
    # ------------------------------------------------------------------

    def _init_pubs_and_subs(self) -> None:
        # Import message types — these are built from the gps_agent_pkg colcon
        # build; guarded so the module can be imported in unit tests without
        # a live ROS 2 workspace.
        from gps_agent_pkg.msg import (
            TrialCommand, SampleResult, PositionCommand,
            RelaxCommand, DataRequest,
        )

        self._trial_service = ServiceEmulator(
            self._node,
            self._hyperparams['trial_command_topic'], TrialCommand,
            self._hyperparams['sample_result_topic'], SampleResult,
        )
        self._reset_service = ServiceEmulator(
            self._node,
            self._hyperparams['reset_command_topic'], PositionCommand,
            self._hyperparams['sample_result_topic'], SampleResult,
        )
        self._relax_service = ServiceEmulator(
            self._node,
            self._hyperparams['relax_command_topic'], RelaxCommand,
            self._hyperparams['sample_result_topic'], SampleResult,
        )
        self._data_service = ServiceEmulator(
            self._node,
            self._hyperparams['data_request_topic'], DataRequest,
            self._hyperparams['sample_result_topic'], SampleResult,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_next_seq_id(self) -> int:
        self._seq_id = (self._seq_id + 1) % (2 ** 32)
        return self._seq_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_data(self, arm: int = TRIAL_ARM) -> Any:
        """
        Request the most recent sensor readings.

        Args:
            arm: TRIAL_ARM or AUXILIARY_ARM.

        Returns:
            Sample populated with current sensor data.
        """
        from gps_agent_pkg.msg import DataRequest
        request = DataRequest()
        request.id = self._get_next_seq_id()
        request.arm = arm
        stamp = self._node.get_clock().now().to_msg()
        request.stamp = stamp
        result_msg = self._data_service.publish_and_wait(request)
        return msg_to_sample(result_msg, self)

    def relax_arm(self, arm: int) -> None:
        """
        Relax one of the robot arms (cease torque commands).

        Args:
            arm: TRIAL_ARM or AUXILIARY_ARM.
        """
        from gps_agent_pkg.msg import RelaxCommand
        relax_command = RelaxCommand()
        relax_command.id = self._get_next_seq_id()
        relax_command.stamp = self._node.get_clock().now().to_msg()
        relax_command.arm = arm
        self._relax_service.publish_and_wait(relax_command)

    def reset_arm(self, arm: int, mode: int, data: Any) -> None:
        """
        Issue a position command to an arm.

        Args:
            arm:  TRIAL_ARM or AUXILIARY_ARM.
            mode: Position control mode (defined in gps_pb2).
            data: Array of target joint angles / positions.
        """
        from gps_agent_pkg.msg import PositionCommand
        reset_command = PositionCommand()
        reset_command.mode = mode
        reset_command.data = data
        reset_command.pd_gains = self._hyperparams['pid_params']
        reset_command.arm = arm
        reset_command.id = self._get_next_seq_id()
        timeout = self._hyperparams['trial_timeout']
        self._reset_service.publish_and_wait(reset_command, timeout=timeout)

    def reset(self, condition: int) -> None:
        """Reset the agent to the initial state for the given condition."""
        condition_data = self._hyperparams['reset_conditions'][condition]
        self.reset_arm(TRIAL_ARM,
                       condition_data[TRIAL_ARM]['mode'],
                       condition_data[TRIAL_ARM]['data'])
        self.reset_arm(AUXILIARY_ARM,
                       condition_data[AUXILIARY_ARM]['mode'],
                       condition_data[AUXILIARY_ARM]['data'])
        time.sleep(2.0)

    def sample(self, policy: Any, condition: int,
               verbose: bool = True, save: bool = True,
               noisy: bool = True) -> Any:
        """
        Reset the robot and execute one trial, returning the collected sample.

        Args:
            policy:    Policy object (LinearGaussianPolicy or PyTorchPolicy).
            condition: Which initial condition to use.
            verbose:   Unused (kept for interface compatibility).
            save:      Whether to store the trial in self._samples.
            noisy:     Whether to add exploration noise to actions.

        Returns:
            Sample object with the collected trajectory data.
        """
        if TfPolicy is not None and isinstance(policy, TfPolicy):
            self._init_tf(policy.dU)

        self.reset(condition)

        noise = (generate_noise(self.T, self.dU, self._hyperparams)
                 if noisy else np.zeros((self.T, self.dU)))

        from gps_agent_pkg.msg import TrialCommand
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

        if not self.use_tf:
            sample_msg = self._trial_service.publish_and_wait(
                trial_command,
                timeout=self._hyperparams['trial_timeout'],
            )
            sample = msg_to_sample(sample_msg, self)
        else:
            self._trial_service.publish(trial_command)
            sample_msg = self.run_trial_tf(
                policy, time_to_run=self._hyperparams['trial_timeout']
            )
            sample = msg_to_sample(sample_msg, self)

        if save:
            self._samples[condition].append(sample)
        return sample

    # ------------------------------------------------------------------
    # TF (async tensor-flow) controller path
    # ------------------------------------------------------------------

    def run_trial_tf(self, policy: Any, time_to_run: float = 5.0) -> Any:
        """
        Run an asynchronous controller that receives observations via a ROS 2
        subscription and publishes actions back.
        """
        should_stop = False
        consecutive_failures = 0
        start_time = time.time()

        while not should_stop:
            if not self.observations_stale:
                consecutive_failures = 0
                last_obs = tf_obs_msg_to_numpy(self._tf_subscriber_msg)
                action_msg = tf_policy_to_action_msg(
                    self.dU,
                    self._get_new_action(policy, last_obs),
                    self.current_action_id,
                )
                self._tf_publish(action_msg)
                self.observations_stale = True
                self.current_action_id += 1
            else:
                # Spin once to pump the rclpy callback queue.
                rclpy.spin_once(self._node, timeout_sec=0.01)
                consecutive_failures += 1
                if (time.time() - start_time > time_to_run
                        and consecutive_failures > 5):
                    should_stop = True

        # Wait for the finished-trial report.
        rclpy.spin_once(self._node, timeout_sec=0.25)
        return self._trial_service._subscriber_msg

    def _get_new_action(self, policy: Any, obs: np.ndarray) -> np.ndarray:
        return policy.act(None, obs, None, None)

    def _tf_callback(self, message: Any) -> None:
        self._tf_subscriber_msg = message
        self.observations_stale = False

    def _tf_publish(self, pub_msg: Any) -> None:
        self._pub.publish(pub_msg)

    def _init_tf(self, dU: int) -> None:
        from gps_agent_pkg.msg import TfActionCommand, TfObsData
        self._tf_subscriber_msg = None
        self.observations_stale = True
        self.current_action_id = 1
        self.dU = dU
        if not self.use_tf:
            self._pub = self._node.create_publisher(
                TfActionCommand,
                '/gps_controller_sent_robot_action_tf',
                _GPS_QOS,
            )
            self._sub = self._node.create_subscription(
                TfObsData,
                '/gps_obs_tf',
                self._tf_callback,
                _GPS_QOS,
            )
            # Give publisher/subscriber time to connect.
            rate = self._node.create_rate(0.5)
            rate.sleep()
        self.use_tf = True
        self.observations_stale = True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Destroy the ROS 2 node and shut down rclpy if we initialised it."""
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
