"""
Translates gps_agent_pkg/src/robotplugin.cpp + include/gps_agent_pkg/robotplugin.h.

RobotPlugin is an rclpy Node that:
  • subscribes to GPS controller topics (position, trial, relax, data-request, tf-action)
  • maintains sensor and controller objects
  • publishes SampleResult and TfObsData messages

Subclasses must implement:
  get_joint_encoder_readings(arm) → np.ndarray
  get_fk_solver(arm) → (fk_solver, jac_solver)
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Optional

import numpy as np

from gps_agent_pkg.sample import ControllerSample, SAMPLE_FORMAT_VECTOR
from gps_agent_pkg.sensor import Sensor, SensorType
from gps_agent_pkg.position_controller import PositionController, NO_CONTROL
from gps_agent_pkg.lingauss_controller import LinearGaussianController
from gps_agent_pkg.pytorch_controller import PyTorchController
from gps_agent_pkg.tf_controller import TfController

LOGGER = logging.getLogger(__name__)

# gps.proto ActuatorType values
TRIAL_ARM     = 0
AUXILIARY_ARM = 1

# gps.proto ControllerType values
LIN_GAUSS_CONTROLLER = 1
PYTORCH_CONTROLLER   = 3
TF_CONTROLLER        = 2

# gps.proto SampleType for ACTION
_ACTION = 19

MAX_TRIAL_LENGTH = 2000


class RobotPlugin:
    """
    Python / rclpy equivalent of C++ gps_control::RobotPlugin.

    When used with a real ROS 2 runtime subclass, pass the rclpy node as
    ``node``.  For testing purposes the node can be any duck-typed object.

    Subclasses implement ``get_joint_encoder_readings`` and ``get_fk_solver``.
    """

    def __init__(self, node) -> None:
        self._node = node

        self._trial_data_request_waiting:  bool = False
        self._aux_data_request_waiting:    bool = False
        self._sensors_initialized:         bool = False
        self._controller_initialized:      bool = False

        # Torque storage
        n_joints = 7  # default; subclasses may override after super().__init__
        self._active_arm_torques  = np.zeros(n_joints, dtype=np.float64)
        self._passive_arm_torques = np.zeros(n_joints, dtype=np.float64)

        # Controllers
        self._trial_controller:    Optional[object] = None
        self._active_arm_controller:  Optional[PositionController] = None
        self._passive_arm_controller: Optional[PositionController] = None

        # Sensor lists
        self._sensors:     list[Sensor] = []
        self._aux_sensors: list[Sensor] = []

        # Samples
        self._current_sample:     Optional[ControllerSample] = None
        self._aux_current_sample: Optional[ControllerSample] = None

        # Publishers (set up by subclass or _init_ros())
        self._report_publisher = None
        self._tf_publisher     = None

        self._init_ros()
        self._init_sensors()
        self._init_position_controllers()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_ros(self) -> None:
        """
        Create ROS 2 subscriptions and publishers.
        Mirrors C++ RobotPlugin::initialize_ros.
        """
        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            )
            from gps_agent_pkg.msg import (  # type: ignore[import]
                PositionCommand, TrialCommand, RelaxCommand,
                DataRequest, TfActionCommand, SampleResult, TfObsData,
            )
            self._node.create_subscription(
                PositionCommand, "/gps_controller_position_command",
                self._position_callback, qos,
            )
            self._node.create_subscription(
                TrialCommand, "/gps_controller_trial_command",
                self._trial_callback, qos,
            )
            self._node.create_subscription(
                RelaxCommand, "/gps_controller_relax_command",
                self._relax_callback, qos,
            )
            self._node.create_subscription(
                DataRequest, "/gps_controller_data_request",
                self._data_request_callback, qos,
            )
            self._node.create_subscription(
                TfActionCommand, "/gps_controller_sent_robot_action_tf",
                self._tf_action_callback, qos,
            )
            self._report_publisher = self._node.create_publisher(
                SampleResult, "/gps_controller_report", qos,
            )
            self._tf_publisher = self._node.create_publisher(
                TfObsData, "/gps_obs_tf", qos,
            )
        except Exception as exc:
            LOGGER.warning("RobotPlugin._init_ros: ROS not available — %s", exc)

    def _init_sensors(self) -> None:
        """
        Create encoder + ROSTopic sensors for trial and auxiliary arms.
        Mirrors C++ RobotPlugin::initialize_sensors.
        """
        self._sensors.clear()
        self._aux_sensors.clear()

        # Trial arm: encoder + ROSTopic sensors (indices 0 and 1)
        for i in range(2):
            sensor = Sensor.create_sensor(SensorType(i), self._node, self, TRIAL_ARM)
            self._sensors.append(sensor)

        n_joints = len(self._active_arm_torques)
        self._current_sample = ControllerSample(MAX_TRIAL_LENGTH)
        self._initialize_sample(self._current_sample, TRIAL_ARM)

        # Auxiliary arm: encoder sensor only (index 0)
        aux_sensor = Sensor.create_sensor(SensorType.EncoderSensorType, self._node, self, AUXILIARY_ARM)
        self._aux_sensors.append(aux_sensor)

        self._aux_current_sample = ControllerSample(1)
        self._initialize_sample(self._aux_current_sample, AUXILIARY_ARM)

        self._sensors_initialized = True

    def _initialize_sample(self, sample: ControllerSample, arm: int) -> None:
        """Register sensor metadata on a sample."""
        if arm == TRIAL_ARM:
            for sensor in self._sensors:
                sensor.set_sample_data_format(sample)
            # ACTION dtype
            sample.set_meta_data(_ACTION, len(self._active_arm_torques), fmt=SAMPLE_FORMAT_VECTOR)
        elif arm == AUXILIARY_ARM:
            for sensor in self._aux_sensors:
                sensor.set_sample_data_format(sample)

    def _configure_sensors(self, opts: dict) -> None:
        """Configure all sensors with trial-specific options."""
        self._sensors_initialized = False
        for sensor in self._sensors:
            sensor.configure_sensor(opts)
            sensor.set_sample_data_format(self._current_sample)
        # Re-register ACTION format after reconfiguration
        self._current_sample.set_meta_data(
            _ACTION, len(self._active_arm_torques), fmt=SAMPLE_FORMAT_VECTOR,
        )
        for sensor in self._aux_sensors:
            sensor.configure_sensor(opts)
            sensor.set_sample_data_format(self._aux_current_sample)
        self._sensors_initialized = True

    def _init_position_controllers(self) -> None:
        """Create passive and active arm PID controllers."""
        self._passive_arm_controller = PositionController(self._node, AUXILIARY_ARM, 7)
        self._active_arm_controller  = PositionController(self._node, TRIAL_ARM,     7)

    # ------------------------------------------------------------------
    # Update loop (called by subclass timer or test harness)
    # ------------------------------------------------------------------

    def update_sensors(self, current_time: float, is_controller_step: bool) -> None:
        """Mirrors C++ RobotPlugin::update_sensors."""
        if not self._sensors_initialized:
            return

        step_t = (
            self._trial_controller.get_step_counter()
            if self._trial_controller is not None
            else 0
        )

        for sensor in self._sensors:
            sensor.update(self, current_time, is_controller_step)
            sensor.set_sample_data(self._current_sample, step_t)

        for sensor in self._aux_sensors:
            sensor.update(self, current_time, is_controller_step)
            sensor.set_sample_data(self._aux_current_sample, 0)

        if self._trial_data_request_waiting:
            self.publish_sample_report(self._current_sample)
            self._trial_data_request_waiting = False

        if self._aux_data_request_waiting:
            self.publish_sample_report(self._aux_current_sample)
            self._aux_data_request_waiting = False

    def update_controllers(self, current_time: float, is_controller_step: bool) -> None:
        """Mirrors C++ RobotPlugin::update_controllers."""
        # Always update passive arm
        self._passive_arm_controller.update(
            self, current_time, self._current_sample, self._passive_arm_torques,
        )

        trial_init = (
            self._trial_controller is not None
            and self._trial_controller.is_configured()
            and self._controller_initialized
        )
        if not is_controller_step and trial_init:
            return

        if trial_init:
            self._trial_controller.update(
                self, current_time, self._current_sample, self._active_arm_torques,
            )
        else:
            self._active_arm_controller.update(
                self, current_time, self._current_sample, self._active_arm_torques,
            )

        # Check if trial finished
        if trial_init and self._trial_controller.is_finished():
            self.publish_sample_report(
                self._current_sample,
                T=self._trial_controller.get_trial_length(),
            )
            self._trial_controller.reset(current_time)
            self._trial_controller = None

            opts: dict = {"mode": NO_CONTROL}
            self._active_arm_controller.configure_controller(opts)

        if self._active_arm_controller.report_waiting:
            if self._active_arm_controller.is_finished():
                self.publish_sample_report(self._current_sample)
                self._active_arm_controller.report_waiting = False

        if self._passive_arm_controller.report_waiting:
            if self._passive_arm_controller.is_finished():
                self.publish_sample_report(self._current_sample)
                self._passive_arm_controller.report_waiting = False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_sample_report(
        self, sample: ControllerSample, T: int = 1
    ) -> None:
        """
        Pack ControllerSample into a SampleResult ROS message and publish.
        Mirrors C++ RobotPlugin::publish_sample_report.
        """
        if self._report_publisher is None:
            return
        try:
            from gps_agent_pkg.msg import SampleResult, SensorData  # type: ignore[import]
        except ImportError:
            return

        msg = SampleResult()
        dtypes = sample.get_available_dtypes()
        for d in dtypes:
            sd = SensorData()
            sd.data_type = d
            data = sample.get_data(T, d)
            sd.data = data.tolist()
            shape = sample.get_shape(d)
            shape_with_T = [T] + shape
            sd.shape = shape_with_T
            msg.sensor_data.append(sd)

        self._report_publisher.publish(msg)

    def tf_publish_obs(self, obs: np.ndarray) -> None:
        """Publish observation for TfController."""
        if self._tf_publisher is None:
            return
        try:
            from gps_agent_pkg.msg import TfObsData  # type: ignore[import]
            msg = TfObsData()
            msg.data = obs.tolist()
            self._tf_publisher.publish(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _position_callback(self, msg) -> None:
        """Mirrors C++ RobotPlugin::position_subscriber_callback."""
        params: dict = {}
        arm = int(msg.arm)
        params["mode"] = int(msg.mode)
        params["data"] = np.array(msg.data, dtype=np.float64)

        pd = np.array(msg.pd_gains, dtype=np.float64).reshape(-1, 4)
        params["pd_gains"] = pd

        if arm == TRIAL_ARM:
            self._active_arm_controller.configure_controller(params)
        elif arm == AUXILIARY_ARM:
            self._passive_arm_controller.configure_controller(params)

    def _trial_callback(self, msg) -> None:
        """Mirrors C++ RobotPlugin::trial_subscriber_callback."""
        self._controller_initialized = False
        T = int(msg.T)
        if T > MAX_TRIAL_LENGTH:
            LOGGER.error("Trial length %d > MAX_TRIAL_LENGTH %d", T, MAX_TRIAL_LENGTH)
            return

        self._initialize_sample(self._current_sample, TRIAL_ARM)

        frequency = float(msg.frequency)
        for sensor in self._sensors:
            sensor.set_update(1.0 / frequency)

        ctrl_params: dict = {
            "T": T,
            "state_datatypes": list(msg.state_datatypes),
            "obs_datatypes":   list(msg.obs_datatypes),
        }

        ctrl_type = int(msg.controller.controller_to_execute)

        if ctrl_type == LIN_GAUSS_CONTROLLER:
            lg = msg.controller.lingauss
            dX, dU = int(lg.dX), int(lg.dU)
            ctrl_params["dX"] = dX
            ctrl_params["dU"] = dU
            for t in range(T):
                K = np.array(lg.K_t[t * dU * dX:(t + 1) * dU * dX],
                             dtype=np.float64).reshape(dU, dX)
                k = np.array(lg.k_t[t * dU:(t + 1) * dU], dtype=np.float64)
                ctrl_params[f"K_{t}"] = K
                ctrl_params[f"k_{t}"] = k
            self._trial_controller = LinearGaussianController()
            self._trial_controller.configure_controller(ctrl_params)

        elif ctrl_type == PYTORCH_CONTROLLER:
            tp = msg.controller.torch
            dU      = int(tp.dU)
            dim_obs = int(tp.dim_bias)
            ctrl_params["model_bytes"]    = bytes(tp.model_bytes)
            ctrl_params["torch_version"] = str(tp.torch_version)
            ctrl_params["scale"] = np.array(tp.scale[:dim_obs], dtype=np.float64)
            ctrl_params["bias"]  = np.array(tp.bias[:dim_obs],  dtype=np.float64)
            for t in range(T):
                ctrl_params[f"noise_{t}"] = np.array(
                    tp.noise[t * dU:(t + 1) * dU], dtype=np.float64,
                )
            self._trial_controller = PyTorchController()
            self._trial_controller.configure_controller(ctrl_params)

        elif ctrl_type == TF_CONTROLLER:
            tf = msg.controller.tf
            ctrl_params["dU"] = int(tf.dU)
            self._trial_controller = TfController()
            self._trial_controller.configure_controller(ctrl_params)

        else:
            LOGGER.error("Unknown trial controller type: %d", ctrl_type)
            self._trial_controller = None
            return

        # Configure sensors with EE points
        sensor_params: dict = {}
        n_pts = len(msg.ee_points) // 3
        sensor_params["ee_sites"]     = np.array(msg.ee_points,     dtype=np.float64).reshape(n_pts, 3)
        sensor_params["ee_points_tgt"] = np.array(msg.ee_points_tgt, dtype=np.float64).reshape(n_pts, 3)
        self._configure_sensors(sensor_params)

        self._controller_initialized = True

    def _relax_callback(self, msg) -> None:
        """Mirrors C++ RobotPlugin::relax_subscriber_callback."""
        arm = int(msg.arm)
        params = {"mode": NO_CONTROL}
        if arm == TRIAL_ARM:
            self._active_arm_controller.configure_controller(params)
        elif arm == AUXILIARY_ARM:
            self._passive_arm_controller.configure_controller(params)

    def _data_request_callback(self, msg) -> None:
        """Mirrors C++ RobotPlugin::data_request_subscriber_callback."""
        arm = int(msg.arm)
        if arm == TRIAL_ARM:
            self._trial_data_request_waiting = True
        elif arm == AUXILIARY_ARM:
            self._aux_data_request_waiting = True

    def _tf_action_callback(self, msg) -> None:
        """Mirrors C++ RobotPlugin::tf_robot_action_command_callback."""
        trial_init = (
            self._trial_controller is not None
            and self._trial_controller.is_configured()
        )
        if trial_init:
            dU = int(msg.dU)
            action = np.array(msg.action[:dU], dtype=np.float64)
            self._trial_controller.update_action_command(int(msg.id), action)

    # ------------------------------------------------------------------
    # Abstract interface (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_joint_encoder_readings(
        self,
        arm: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return current joint positions for *arm*."""
        ...

    @abstractmethod
    def get_fk_solver(self, arm: int):
        """Return (fk_solver, jac_solver) for *arm*."""
        ...
