"""
Translates gps_agent_pkg/src/encoderfilter.cpp + include/encoderfilter.h.

The filter implements:
    filtered_state_ = A @ filtered_state_ + outer(h, state)

where A is the (order x order) time matrix and h is the (order,) observation
vector.  Both are parsed from a newline-delimited ROS parameter string whose
format matches encoder_filter_params.txt:
  - Line 0: A entries in **column-major** order (order² values)
  - Line 1: h entries (order values)

get_state()    → filtered_state_[0, :]   (positions)
get_velocity() → filtered_state_[1, :]   (velocities)
"""
from __future__ import annotations

import numpy as np


class EncoderFilter:
    """
    Linear state filter for joint encoder readings.

    Parameters
    ----------
    node : rclpy Node
        Used to read the ``encoder_filter_params`` ROS parameter.
    initial_state : np.ndarray, shape (n_joints,)
        Current joint angles at construction time.
    """

    def __init__(self, node, initial_state: np.ndarray) -> None:
        self._num_joints = len(initial_state)
        self._is_configured = False

        # Try to read the filter params from the ROS node parameter.
        try:
            params: str = node.get_parameter("encoder_filter_params").get_parameter_value().string_value
        except Exception:
            params = ""

        if params:
            self._configure(params)
            # Set initial position from provided state.
            if self._is_configured:
                self._filtered_state[0, :] = initial_state

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure(self, params: str) -> None:
        """
        Parse the filter matrices from a newline-delimited parameter string.

        Line 0: time_matrix entries in column-major order (order² values space-separated)
        Line 1: observation_vector entries (order values space-separated)
        """
        lines = params.split("\n")
        if len(lines) < 2:
            return

        time_values = [v for v in lines[0].split(" ") if v]
        obs_values  = [v for v in lines[1].split(" ") if v]

        order = len(obs_values)
        if order == 0 or len(time_values) != order * order:
            return

        # time_matrix stored in column-major order in the param string
        self._time_matrix = (
            np.array([float(v) for v in time_values], dtype=np.float64)
            .reshape(order, order, order="F")  # Fortran (column-major) reshape
        )
        self._observation_vector = np.array([float(v) for v in obs_values], dtype=np.float64)

        self._filtered_state = np.zeros((order, self._num_joints), dtype=np.float64)
        self._is_configured = True

    # ------------------------------------------------------------------
    # Update / getters
    # ------------------------------------------------------------------

    def update(self, sec_elapsed: float, state: np.ndarray) -> None:
        """
        Advance the filter by one step.

        ``state`` is a 1-D array of shape (n_joints,) representing the
        latest raw joint angles.

        Mirrors C++:  filtered_state_ = A * filtered_state_ + h * state^T
        """
        if not self._is_configured:
            raise RuntimeError("EncoderFilter.update called before configure")
        self._filtered_state = (
            self._time_matrix @ self._filtered_state
            + np.outer(self._observation_vector, state)
        )

    def get_state(self) -> np.ndarray:
        """Return filtered joint positions (first row of state matrix)."""
        return self._filtered_state[0].copy()

    def get_velocity(self) -> np.ndarray:
        """Return filtered joint velocities (second row of state matrix)."""
        return self._filtered_state[1].copy()

    @property
    def is_configured(self) -> bool:
        return self._is_configured
