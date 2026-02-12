"""
Translates gps_agent_pkg/src/positioncontroller.cpp + include/positioncontroller.h.

PID joint-space position controller.
is_finished() threshold: position error < 0.185 rad AND velocity < 0.01 rad/s.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from gps_agent_pkg.controller import Controller
from gps_agent_pkg.sample import ControllerSample

# gps.proto PositionControlMode values
NO_CONTROL  = 0
JOINT_SPACE = 1
TASK_SPACE  = 2


class PositionController(Controller):
    """
    PD+I joint-space position controller.

    Mirrors C++ gps_control::PositionController.
    """

    EPS_POS: float = 0.185   # rad — position convergence threshold
    EPS_VEL: float = 0.01    # rad/s — velocity convergence threshold

    def __init__(self, node, arm: int, n_joints: int) -> None:
        super().__init__()
        self._arm = arm
        self._n_joints = n_joints

        self._mode: int = NO_CONTROL
        self.report_waiting: bool = False

        self._pd_gains_p = np.zeros(n_joints, dtype=np.float64)
        self._pd_gains_i = np.zeros(n_joints, dtype=np.float64)
        self._pd_gains_d = np.zeros(n_joints, dtype=np.float64)
        self._i_clamp    = np.zeros(n_joints, dtype=np.float64)

        self._pd_integral            = np.zeros(n_joints, dtype=np.float64)
        self._current_angles         = np.zeros(n_joints, dtype=np.float64)
        self._current_angle_velocities = np.zeros(n_joints, dtype=np.float64)
        self._target_angles          = np.zeros(n_joints, dtype=np.float64)

        self._last_update_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_controller(self, options: dict) -> None:
        """
        Parse mode, PD gains, and target angles.
        Mirrors C++ PositionController::configure_controller.
        """
        self.report_waiting = True
        self._mode = int(options["mode"])

        if self._mode != NO_CONTROL:
            pd_gains = np.asarray(options["pd_gains"], dtype=np.float64)
            # pd_gains has shape (n_joints, 4): [Kp, Ki, Kd, i_clamp]
            self._pd_gains_p = pd_gains[:, 0].copy()
            self._pd_gains_i = pd_gains[:, 1].copy()
            self._pd_gains_d = pd_gains[:, 2].copy()
            self._i_clamp    = pd_gains[:, 3].copy()

            if self._mode == JOINT_SPACE:
                self._target_angles = np.asarray(options["data"], dtype=np.float64).copy()
            # TASK_SPACE not yet implemented (mirrors C++ TODO)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        plugin,
        current_time: float,
        sample: ControllerSample,
        torques: np.ndarray,
    ) -> np.ndarray:
        """
        One PID step: read joint angles, compute torques.
        Mirrors C++ PositionController::update.
        """
        # Get current joint angles from plugin
        q = plugin.get_joint_encoder_readings(self._arm)

        # Estimate velocities
        dt = 0.0
        if self._last_update_time is not None:
            dt = current_time - self._last_update_time
        if dt > 0:
            self._current_angle_velocities = (q - self._current_angles) / dt

        self._current_angles = q.copy()
        self._last_update_time = current_time

        if self._mode != NO_CONTROL:
            err = self._current_angles - self._target_angles

            # Accumulate and clamp integral term
            self._pd_integral += err * dt
            safe_ki = np.where(self._pd_gains_i != 0, self._pd_gains_i, 1.0)
            np.clip(
                self._pd_integral,
                -self._i_clamp / safe_ki,
                 self._i_clamp / safe_ki,
                out=self._pd_integral,
            )

            torques[:] = -(
                self._pd_gains_p * err
                + self._pd_gains_d * self._current_angle_velocities
                + self._pd_gains_i * self._pd_integral
            )
        else:
            torques[:] = 0.0

        return torques

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_finished(self) -> bool:
        """
        Return True when position error < EPS_POS AND velocity < EPS_VEL.
        Mirrors C++ PositionController::is_finished.
        """
        if self._mode == JOINT_SPACE:
            err = np.linalg.norm(self._current_angles - self._target_angles)
            vel = np.linalg.norm(self._current_angle_velocities)
            return bool(err < self.EPS_POS and vel < self.EPS_VEL)
        return self._mode == NO_CONTROL

    def reset(self, current_time: float) -> None:
        self._pd_integral[:] = 0.0
        self._last_update_time = None
