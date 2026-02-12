"""
Translates gps_agent_pkg/src/lingausscontroller.cpp + include/lingausscontroller.h.

U = K[t] @ X + k[t]
"""
from __future__ import annotations

import numpy as np

from gps_agent_pkg.trial_controller import TrialController


class LinearGaussianController(TrialController):
    """
    Time-varying linear-Gaussian controller: U_t = K_t @ X_t + k_t.

    Mirrors C++ gps_control::LinearGaussianController.
    """

    def configure_controller(self, options: dict) -> None:
        super().configure_controller(options)
        T = int(options["T"])
        self.K_: list[np.ndarray] = [
            np.asarray(options[f"K_{t}"], dtype=np.float64) for t in range(T)
        ]
        self.k_: list[np.ndarray] = [
            np.asarray(options[f"k_{t}"], dtype=np.float64) for t in range(T)
        ]
        self.is_configured_ = True

    def get_action(self, t: int, X: np.ndarray, obs: np.ndarray) -> np.ndarray:
        return (self.K_[t] @ X + self.k_[t]).astype(np.float64)
