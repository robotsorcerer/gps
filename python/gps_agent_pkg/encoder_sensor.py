"""
Translates gps_agent_pkg/src/encodersensor.cpp + include/encodersensor.h.

Uses PyKDL (Python bindings for the KDL kinematics library) for FK and Jacobian,
mirroring the C++ KDL calls exactly.

SampleType constants (gps.proto values):
  JOINT_ANGLES                  = 0
  JOINT_VELOCITIES              = 1
  END_EFFECTOR_POINTS           = 2
  END_EFFECTOR_POINT_VELOCITIES = 3
  END_EFFECTOR_POSITIONS        = 13
  END_EFFECTOR_ROTATIONS        = 14
  END_EFFECTOR_JACOBIANS        = 7
  END_EFFECTOR_POINT_JACOBIANS  = 5
  END_EFFECTOR_POINT_ROT_JACOBIANS = 6
"""
from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

try:
    import PyKDL as kdl
    _KDL_AVAILABLE = True
except ImportError:
    _KDL_AVAILABLE = False

from gps_agent_pkg.sensor import Sensor
from gps_agent_pkg.encoder_filter import EncoderFilter
from gps_agent_pkg.sample import (
    ControllerSample, SAMPLE_FORMAT_VECTOR, SAMPLE_FORMAT_MATRIX,
)

# SampleType int values matching gps.proto
_JOINT_ANGLES = 0
_JOINT_VELOCITIES = 1
_END_EFFECTOR_POINTS = 2
_END_EFFECTOR_POINT_VELOCITIES = 3
_END_EFFECTOR_POINT_JACOBIANS = 5
_END_EFFECTOR_POINT_ROT_JACOBIANS = 6
_END_EFFECTOR_JACOBIANS = 7
_END_EFFECTOR_POSITIONS = 13
_END_EFFECTOR_ROTATIONS = 14


class EncoderSensor(Sensor):
    """
    Joint encoder sensor: returns joint angles, velocities, FK, and Jacobians.

    Mirrors C++ gps_control::EncoderSensor.
    """

    def __init__(self, node, plugin, actuator_type: int) -> None:
        super().__init__(node, plugin)
        self._actuator_type = actuator_type

        # Get initial joint angles from plugin.
        self._previous_angles: np.ndarray = plugin.get_joint_encoder_readings(actuator_type)
        n_joints = len(self._previous_angles)

        self._previous_velocities   = np.zeros(n_joints, dtype=np.float64)
        self._temp_joint_angles     = np.zeros(n_joints, dtype=np.float64)

        # KDL objects (set during update from plugin)
        self._fk_solver  = None
        self._jac_solver = None

        # End-effector state
        self._previous_position = np.zeros(3, dtype=np.float64)
        self._previous_rotation = np.eye(3, dtype=np.float64)
        self._previous_jacobian = np.zeros((6, n_joints), dtype=np.float64)

        # EE points (3 × n_points, default 1 point at origin)
        self._n_points = 1
        self._end_effector_points        = np.zeros((3, 1), dtype=np.float64)
        self._end_effector_points_target = np.zeros((3, 1), dtype=np.float64)
        self._previous_ee_points         = np.zeros((3, 1), dtype=np.float64)
        self._previous_ee_point_velocities = np.zeros((3, 1), dtype=np.float64)
        self._temp_ee_points             = np.zeros((3, 1), dtype=np.float64)

        # Point Jacobians (3*n_points × n_joints)
        self._point_jacobians     = np.zeros((3, n_joints), dtype=np.float64)
        self._point_jacobians_rot = np.zeros((3, n_joints), dtype=np.float64)

        # Time tracking (None = no previous sample yet)
        self._previous_angles_time: Optional[float] = None

        # Encoder filter
        self._joint_filter = EncoderFilter(node, self._previous_angles)

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def update(self, plugin, current_time: float, is_controller_step: bool) -> None:
        """
        Called every tick.  Mirrors C++ EncoderSensor::update.
        """
        # Compute elapsed time
        dt = 0.0
        if self._previous_angles_time is not None:
            dt = current_time - self._previous_angles_time

        # Get new raw joint angles from plugin and run through filter.
        plugin.get_joint_encoder_readings(self._actuator_type, out=self._temp_joint_angles)
        if self._joint_filter.is_configured:
            self._joint_filter.update(dt, self._temp_joint_angles)

        if is_controller_step:
            # Use filtered angles.
            if self._joint_filter.is_configured:
                self._temp_joint_angles = self._joint_filter.get_state()

            n_joints = len(self._previous_angles)

            if _KDL_AVAILABLE and self._fk_solver is None:
                self._fk_solver, self._jac_solver = plugin.get_fk_solver(self._actuator_type)

            if _KDL_AVAILABLE and self._fk_solver is not None:
                # Build KDL JntArray
                q = kdl.JntArray(n_joints)
                for i in range(n_joints):
                    q[i] = float(self._temp_joint_angles[i])

                # Forward kinematics
                frame = kdl.Frame()
                self._fk_solver.JntToCart(q, frame)
                for i in range(3):
                    self._previous_position[i] = frame.p[i]
                for i in range(3):
                    for j in range(3):
                        self._previous_rotation[i, j] = frame.M[i, j]

                # Jacobian
                jac = kdl.Jacobian(n_joints)
                self._jac_solver.JntToJac(q, jac)
                for i in range(6):
                    for j in range(n_joints):
                        self._previous_jacobian[i, j] = jac[i, j]

                # Compute per-point Jacobians (mirrors C++ site Jacobian computation)
                n_pts = self._n_points
                self._point_jacobians     = np.zeros((3 * n_pts, n_joints), dtype=np.float64)
                self._point_jacobians_rot = np.zeros((3 * n_pts, n_joints), dtype=np.float64)

                for pi in range(n_pts):
                    ss = pi * 3
                    ovec = self._end_effector_points[:, pi]

                    # Linear + angular parts of Jacobian
                    self._point_jacobians[ss:ss+3,     :] = self._previous_jacobian[:3, :]
                    self._point_jacobians_rot[ss:ss+3, :] = self._previous_jacobian[3:, :]

                    # Site Jacobian cross-product correction: J_site = J_lin + J_rot × ovec
                    ovec_rot = self._previous_rotation @ ovec
                    ox, oy, oz = ovec_rot
                    # x += rot_y * oz - rot_z * oy
                    self._point_jacobians[ss,   :] += (
                        self._point_jacobians_rot[ss+1, :] * oz
                        - self._point_jacobians_rot[ss+2, :] * oy
                    )
                    # y += rot_z * ox - rot_x * oz
                    self._point_jacobians[ss+1, :] += (
                        self._point_jacobians_rot[ss+2, :] * ox
                        - self._point_jacobians_rot[ss,   :] * oz
                    )
                    # z += rot_x * oy - rot_y * ox
                    self._point_jacobians[ss+2, :] += (
                        self._point_jacobians_rot[ss,   :] * oy
                        - self._point_jacobians_rot[ss+1, :] * ox
                    )

                # Compute EE points in world frame, subtract target
                self._temp_ee_points = (
                    self._previous_rotation @ self._end_effector_points
                )
                self._temp_ee_points += self._previous_position[:, np.newaxis]
                self._temp_ee_points -= self._end_effector_points_target

            # Compute velocities if we have a previous sample
            if self._previous_angles_time is not None and dt > 0:
                ratio = dt / self.sensor_step_length_
                if 0.5 <= abs(ratio) <= 2.0:
                    step = self.sensor_step_length_
                else:
                    step = dt
                self._previous_ee_point_velocities = (
                    (self._temp_ee_points - self._previous_ee_points) / step
                )
                self._previous_velocities = (
                    (self._temp_joint_angles - self._previous_angles) / step
                )

            # Update previous state
            self._previous_ee_points = self._temp_ee_points.copy()
            self._previous_angles = self._temp_joint_angles.copy()
            self._previous_angles_time = current_time

    def configure_sensor(self, options: dict) -> None:
        """
        Set end-effector point offsets and targets from the options dict.
        Mirrors C++ EncoderSensor::configure_sensor.
        """
        ee_sites = np.asarray(options["ee_sites"], dtype=np.float64)
        # C++ stores as (3, n_points) via .transpose() of (n_points, 3)
        if ee_sites.ndim == 2 and ee_sites.shape[1] == 3:
            self._end_effector_points = ee_sites.T.copy()
        else:
            self._end_effector_points = ee_sites

        self._n_points = self._end_effector_points.shape[1]
        n_joints = len(self._previous_angles)

        ee_tgt = np.asarray(options["ee_points_tgt"], dtype=np.float64)
        if ee_tgt.ndim == 2 and ee_tgt.shape[1] == 3:
            self._end_effector_points_target = ee_tgt.T.copy()
        else:
            self._end_effector_points_target = ee_tgt

        self._previous_ee_points           = np.zeros((3, self._n_points), dtype=np.float64)
        self._previous_ee_point_velocities = np.zeros((3, self._n_points), dtype=np.float64)
        self._temp_ee_points               = np.zeros((3, self._n_points), dtype=np.float64)
        self._point_jacobians     = np.zeros((3 * self._n_points, n_joints), dtype=np.float64)
        self._point_jacobians_rot = np.zeros((3 * self._n_points, n_joints), dtype=np.float64)

    def set_sample_data_format(self, sample: ControllerSample) -> None:
        n_joints = len(self._previous_angles)
        n_pts    = self._n_points

        sample.set_meta_data(_JOINT_ANGLES,                   n_joints, fmt=SAMPLE_FORMAT_VECTOR)
        sample.set_meta_data(_JOINT_VELOCITIES,               n_joints, fmt=SAMPLE_FORMAT_VECTOR)
        sample.set_meta_data(_END_EFFECTOR_POINTS,            3 * n_pts, fmt=SAMPLE_FORMAT_VECTOR)
        sample.set_meta_data(_END_EFFECTOR_POINT_VELOCITIES,  3 * n_pts, fmt=SAMPLE_FORMAT_VECTOR)
        sample.set_meta_data(_END_EFFECTOR_POINT_JACOBIANS,
                             3 * n_pts, n_joints, fmt=SAMPLE_FORMAT_MATRIX)
        sample.set_meta_data(_END_EFFECTOR_POINT_ROT_JACOBIANS,
                             3 * n_pts, n_joints, fmt=SAMPLE_FORMAT_MATRIX)
        sample.set_meta_data(_END_EFFECTOR_POSITIONS, 3, fmt=SAMPLE_FORMAT_VECTOR)
        sample.set_meta_data(_END_EFFECTOR_ROTATIONS, 3, 3, fmt=SAMPLE_FORMAT_MATRIX)
        sample.set_meta_data(_END_EFFECTOR_JACOBIANS, 6, n_joints, fmt=SAMPLE_FORMAT_MATRIX)

    def set_sample_data(self, sample: ControllerSample, t: int) -> None:
        sample.set_data(t, _JOINT_ANGLES,                   self._previous_angles)
        sample.set_data(t, _JOINT_VELOCITIES,               self._previous_velocities)
        sample.set_data(t, _END_EFFECTOR_POINTS,            self._previous_ee_points.flatten())
        sample.set_data(t, _END_EFFECTOR_POINT_VELOCITIES,  self._previous_ee_point_velocities.flatten())
        sample.set_data(t, _END_EFFECTOR_POINT_JACOBIANS,   self._point_jacobians)
        sample.set_data(t, _END_EFFECTOR_POINT_ROT_JACOBIANS, self._point_jacobians_rot)
        sample.set_data(t, _END_EFFECTOR_POSITIONS,         self._previous_position)
        sample.set_data(t, _END_EFFECTOR_ROTATIONS,         self._previous_rotation)
        sample.set_data(t, _END_EFFECTOR_JACOBIANS,         self._previous_jacobian)
