"""
Translates gps_agent_pkg/src/pr2plugin.cpp + include/pr2plugin.h.

In ROS 2 the PR2 real-time controller infrastructure (pr2_controller_interface,
pr2_mechanism_model) does not exist.  This class subclasses RobotPlugin and
implements the PR2-specific logic using:
  - sensor_msgs/JointState subscription for encoder readings
  - kdl_parser_py + PyKDL for FK/Jacobian solvers built from robot_description
  - std_msgs/Float64MultiArray publications for torque commands

The update loop is driven by a rclpy timer rather than a real-time thread.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from gps_agent_pkg.robot_plugin import RobotPlugin, TRIAL_ARM, AUXILIARY_ARM

LOGGER = logging.getLogger(__name__)

# Default controller step divisor (50 Hz control out of 1 kHz sensor rate)
_CONTROLLER_STEP_LENGTH = 50


class PR2RobotPlugin(RobotPlugin):
    """
    ROS 2 / Python equivalent of C++ GPSPR2Plugin.

    Construction reads robot_description + joint-name parameters from the
    ROS node, builds KDL chains, and subscribes to /joint_states.

    Parameters
    ----------
    node : rclpy.node.Node
        The rclpy node to use for subscriptions / publishers / parameters.
    """

    def __init__(self, node) -> None:
        # Joint name lists (filled before super().__init__ so that sensors can
        # call get_joint_encoder_readings during initialisation).
        self._active_arm_joint_names:  list[str] = []
        self._passive_arm_joint_names: list[str] = []
        self._active_positions:  np.ndarray = np.zeros(7, dtype=np.float64)
        self._passive_positions: np.ndarray = np.zeros(7, dtype=np.float64)

        # KDL solver pairs — set in _init_kdl_solvers()
        self._active_fk_solver  = None
        self._active_jac_solver = None
        self._passive_fk_solver  = None
        self._passive_jac_solver = None

        # Timer counter
        self._controller_counter: int = 0
        self._controller_step_length: int = _CONTROLLER_STEP_LENGTH

        # Torque publishers (populated in _init_torque_publishers)
        self._l_arm_pub = None
        self._r_arm_pub = None

        # --- Read parameters and set up KDL before super().__init__() ---
        self._read_joint_names(node)
        n_active  = len(self._active_arm_joint_names)  or 7
        n_passive = len(self._passive_arm_joint_names) or 7
        self._active_positions  = np.zeros(n_active,  dtype=np.float64)
        self._passive_positions = np.zeros(n_passive, dtype=np.float64)
        self._init_kdl_solvers(node)

        # --- Set up torque publishers before super().__init__ creates sensors ---
        self._init_torque_publishers(node)

        # --- Subscribe to /joint_states ---
        self._joint_state_sub = None
        self._init_joint_state_sub(node)

        # Now call base class (will create sensors + position controllers)
        super().__init__(node)

        # Start update timer at 1 kHz
        try:
            self._timer = node.create_timer(0.001, self._timer_callback)
        except Exception:
            pass  # not available in tests

    # ------------------------------------------------------------------
    # Parameter reading
    # ------------------------------------------------------------------

    def _read_joint_names(self, node) -> None:
        """Read active_arm_joint_name_N / passive_arm_joint_name_N parameters."""
        for prefix, dest in [
            ("active_arm_joint_name",  self._active_arm_joint_names),
            ("passive_arm_joint_name", self._passive_arm_joint_names),
        ]:
            idx = 1
            while True:
                try:
                    name = node.get_parameter(f"{prefix}_{idx}").get_parameter_value().string_value
                    if not name:
                        break
                    dest.append(name)
                    idx += 1
                except Exception:
                    break

    # ------------------------------------------------------------------
    # KDL setup
    # ------------------------------------------------------------------

    def _init_kdl_solvers(self, node) -> None:
        """
        Build KDL chains from robot_description URDF using kdl_parser_py.
        Mirrors C++ GPSPR2Plugin::init KDL chain creation.
        """
        try:
            import PyKDL as kdl
            from kdl_parser_py.urdf import treeFromString  # type: ignore[import]

            urdf_str: str = ""
            try:
                urdf_str = node.get_parameter("robot_description").get_parameter_value().string_value
            except Exception:
                pass

            if not urdf_str:
                return

            ok, tree = treeFromString(urdf_str)
            if not ok:
                LOGGER.error("PR2RobotPlugin: failed to parse robot_description URDF")
                return

            root_name: str = ""
            active_tip: str = ""
            passive_tip: str = ""
            try:
                root_name   = node.get_parameter("root_name").get_parameter_value().string_value
                active_tip  = node.get_parameter("active_tip_name").get_parameter_value().string_value
                passive_tip = node.get_parameter("passive_tip_name").get_parameter_value().string_value
            except Exception:
                pass

            if root_name and active_tip:
                active_chain = tree.getChain(root_name, active_tip)
                self._active_fk_solver  = kdl.ChainFkSolverPos_recursive(active_chain)
                self._active_jac_solver = kdl.ChainJntToJacSolver(active_chain)

            if root_name and passive_tip:
                passive_chain = tree.getChain(root_name, passive_tip)
                self._passive_fk_solver  = kdl.ChainFkSolverPos_recursive(passive_chain)
                self._passive_jac_solver = kdl.ChainJntToJacSolver(passive_chain)

        except ImportError:
            LOGGER.warning("PR2RobotPlugin: PyKDL / kdl_parser_py not available")
        except Exception as exc:
            LOGGER.error("PR2RobotPlugin._init_kdl_solvers: %s", exc)

    # ------------------------------------------------------------------
    # JointState subscription
    # ------------------------------------------------------------------

    def _init_joint_state_sub(self, node) -> None:
        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from sensor_msgs.msg import JointState
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._joint_state_sub = node.create_subscription(
                JointState, "/joint_states",
                self._joint_state_callback, qos,
            )
        except Exception as exc:
            LOGGER.warning("PR2RobotPlugin: cannot subscribe to /joint_states — %s", exc)

    def _joint_state_callback(self, msg) -> None:
        """Update joint positions from /joint_states message."""
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, name in enumerate(self._active_arm_joint_names):
            if name in name_to_pos and i < len(self._active_positions):
                self._active_positions[i] = name_to_pos[name]
        for i, name in enumerate(self._passive_arm_joint_names):
            if name in name_to_pos and i < len(self._passive_positions):
                self._passive_positions[i] = name_to_pos[name]

    # ------------------------------------------------------------------
    # Torque publishers
    # ------------------------------------------------------------------

    def _init_torque_publishers(self, node) -> None:
        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from std_msgs.msg import Float64MultiArray
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._l_arm_pub = node.create_publisher(
                Float64MultiArray, "/l_arm_controller/command", qos,
            )
            self._r_arm_pub = node.create_publisher(
                Float64MultiArray, "/r_arm_controller/command", qos,
            )
        except Exception:
            pass

    def _publish_torques(self) -> None:
        try:
            from std_msgs.msg import Float64MultiArray
            if self._l_arm_pub is not None:
                msg = Float64MultiArray()
                msg.data = self._active_arm_torques.tolist()
                self._l_arm_pub.publish(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Timer callback (replaces real-time update() in C++)
    # ------------------------------------------------------------------

    def _timer_callback(self) -> None:
        """1 kHz update loop. Mirrors C++ GPSPR2Plugin::update."""
        try:
            current_time = self._node.get_clock().now().nanoseconds * 1e-9
        except Exception:
            current_time = 0.0

        self._controller_counter += 1
        if self._controller_counter >= self._controller_step_length:
            self._controller_counter = 0
        is_controller_step = (self._controller_counter == 0)

        self.update_sensors(current_time, is_controller_step)
        self.update_controllers(current_time, is_controller_step)
        self._publish_torques()

    # ------------------------------------------------------------------
    # RobotPlugin abstract interface
    # ------------------------------------------------------------------

    def get_joint_encoder_readings(
        self,
        arm: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return current joint positions for *arm* from the JointState cache."""
        if arm == AUXILIARY_ARM:
            angles = self._passive_positions.copy()
        else:
            angles = self._active_positions.copy()

        if out is not None:
            out[:len(angles)] = angles[:len(out)]
        return angles

    def get_fk_solver(self, arm: int):
        """Return (fk_solver, jac_solver) pair for *arm*."""
        if arm == AUXILIARY_ARM:
            return self._passive_fk_solver, self._passive_jac_solver
        return self._active_fk_solver, self._active_jac_solver
