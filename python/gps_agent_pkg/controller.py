"""
Translates gps_agent_pkg/src/controller.cpp + include/gps_agent_pkg/controller.h.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """
    Abstract base class for all GPS controllers.

    Mirrors C++ gps_control::Controller.
    """

    def __init__(self) -> None:
        self._step_length: float = 1.0

    def configure_controller(self, options: dict) -> None:
        pass

    def set_update_delay(self, step_length: float) -> None:
        self._step_length = step_length

    def get_update_delay(self) -> float:
        return self._step_length

    def reset(self, current_time: float) -> None:
        pass

    @abstractmethod
    def update(
        self,
        plugin,
        current_time: float,
        sample,
        torques: np.ndarray,
    ) -> np.ndarray:
        """Apply one control step and return updated torques."""
        ...

    @abstractmethod
    def is_finished(self) -> bool:
        """Return True when the controller has completed its task."""
        ...
