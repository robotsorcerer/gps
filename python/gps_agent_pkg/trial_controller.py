"""
Translates gps_agent_pkg/src/trialcontroller.cpp + include/trialcontroller.h.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from gps_agent_pkg.controller import Controller
from gps_agent_pkg.sample import ControllerSample

# SampleType int for ACTION (gps.proto)
_ACTION = 19


class TrialController(Controller):
    """
    Base class for controllers that execute a fixed-length trial.

    Mirrors C++ gps_control::TrialController.
    """

    def __init__(self) -> None:
        super().__init__()
        self.step_counter_: int = 0
        self.trial_end_step_: int = 1
        self.state_datatypes_: list[int] = []
        self.obs_datatypes_: list[int] = []
        self.is_configured_: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_controller(self, options: dict) -> None:
        """
        Parse T, state_datatypes, obs_datatypes from options.
        Mirrors C++ TrialController::configure_controller.
        """
        T = int(options["T"])
        self.step_counter_   = 0
        self.trial_end_step_ = T

        self.state_datatypes_ = [int(d) for d in options.get("state_datatypes", [])]
        self.obs_datatypes_   = [int(d) for d in options.get("obs_datatypes",   [])]

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def update(
        self,
        plugin,
        current_time: float,
        sample: ControllerSample,
        torques: np.ndarray,
    ) -> np.ndarray:
        """
        Execute one trial step.
        Mirrors C++ TrialController::update.
        """
        X   = sample.get_data_vec(self.step_counter_, self.state_datatypes_)
        obs = sample.get_data_vec(self.step_counter_, self.obs_datatypes_)

        self.publish_obs(obs, plugin)
        U = self.get_action(self.step_counter_, X, obs)

        sample.set_data(self.step_counter_, _ACTION, U)
        torques[:] = U
        self.step_counter_ += 1
        return torques

    def is_finished(self) -> bool:
        return self.step_counter_ >= self.trial_end_step_

    def reset(self, current_time: float) -> None:
        self.step_counter_   = 0
        self.trial_end_step_ = 1

    def get_step_counter(self) -> int:
        return self.step_counter_

    def get_trial_length(self) -> int:
        return self.trial_end_step_

    def is_configured(self) -> bool:
        return self.is_configured_

    # ------------------------------------------------------------------
    # Abstract / overrideable
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action(self, t: int, X: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Compute and return the action vector at timestep *t*."""
        ...

    def publish_obs(self, obs: np.ndarray, plugin) -> None:
        """Override in subclasses that need to publish observations (e.g. TfController)."""
        pass

    def update_action_command(self, id: int, command) -> None:
        """Override in TfController to receive async action commands."""
        pass
