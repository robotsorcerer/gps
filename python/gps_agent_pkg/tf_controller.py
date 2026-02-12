"""
Translates gps_agent_pkg/src/tfcontroller.cpp + include/tfcontroller.h.

The TfController receives asynchronous action commands from a TF/RL policy
running in a separate process.  It tolerates up to 2 stale steps before
raising a RuntimeError.
"""
from __future__ import annotations

import numpy as np

from gps_agent_pkg.trial_controller import TrialController


class TfController(TrialController):
    """
    Async TF/external-policy trial controller.

    Mirrors C++ gps_control::TfController.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_command_id_received:   int = 0
        self.last_command_id_acted_upon: int = 0
        self.failed_attempts:            int = 0
        self._last_action: np.ndarray = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_controller(self, options: dict) -> None:
        self.last_command_id_received   = 0
        self.last_command_id_acted_upon = 0
        self.failed_attempts            = 0

        dU = int(options["dU"])
        self._last_action = np.zeros(dU, dtype=np.float64)

        # Call super *after* local init to reset step_counter_ etc.
        super().configure_controller(options)
        self.is_configured_ = True

    # ------------------------------------------------------------------
    # Async action command injection
    # ------------------------------------------------------------------

    def update_action_command(self, id: int, command) -> None:
        """
        Receive the latest action command from the TF policy.
        Mirrors C++ TfController::update_action_command.
        """
        self.last_command_id_received = id
        self._last_action = np.asarray(command, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def get_action(self, t: int, X: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Return the most recently received action if it is fresh, otherwise
        tolerate up to 2 stale steps before raising RuntimeError.

        Mirrors C++ TfController::get_action.
        """
        if self.last_command_id_acted_upon < self.last_command_id_received:
            # Fresh command available
            self.last_command_id_acted_upon = self.last_command_id_received
            self.failed_attempts = 0
            return self._last_action.copy()
        elif self.failed_attempts < 2:
            # Tolerate stale action (up to 2 times)
            self.failed_attempts += 1
            return self._last_action.copy()
        else:
            raise RuntimeError(
                "TfController: no new action command received; "
                "refusing to act on stale action."
            )

    # ------------------------------------------------------------------
    # Observation publishing
    # ------------------------------------------------------------------

    def publish_obs(self, obs: np.ndarray, plugin) -> None:
        """Forward observation to the TF policy via the plugin."""
        plugin.tf_publish_obs(obs)
