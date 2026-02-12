"""
Translates gps_agent_pkg/src/pytorchcontroller.cpp + include/pytorchcontroller.h.

Key logic:
  1. configure_controller: load TorchScript model from raw bytes
  2. Version check: torch.__version__ major must match the torch_version field
  3. get_action(t, X, obs):
       obs_scaled = obs * scale_diag + bias    (pointwise affine normalisation)
       U = module(obs_scaled)[0] + noise_[t]
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np

from gps_agent_pkg.trial_controller import TrialController

LOGGER = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class PyTorchController(TrialController):
    """
    TorchScript-policy trial controller.

    Mirrors C++ gps_control::PyTorchController.
    """

    MAX_INFERENCE_MS: float = 5.0

    def __init__(self) -> None:
        super().__init__()
        self._module = None
        self._scale_diag: np.ndarray = np.array([], dtype=np.float64)
        self._bias:       np.ndarray = np.array([], dtype=np.float64)
        self._noise:      list[np.ndarray] = []
        self._dU: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_controller(self, options: dict) -> None:
        """
        Load TorchScript model and read normalisation / noise parameters.

        Expected keys in *options*:
          "model_bytes"   : bytes | str  — raw TorchScript bytes
          "torch_version" : str          — e.g. "2.1.0"
          "scale"         : array-like   — diagonal of scale matrix (length dO)
          "bias"          : array-like   — bias vector (length dO)
          "T"             : int          — trial length
          "noise_{t}"     : array-like   — noise for timestep t, t=0..T-1
        """
        super().configure_controller(options)

        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for PyTorchController")

        # ---- Version check -------------------------------------------
        torch_version: str = str(options["torch_version"])
        model_major = int(torch_version.split(".")[0])
        runtime_major = int(torch.__version__.split(".")[0])
        if model_major != runtime_major:
            raise ValueError(
                f"Torch major version mismatch: model={model_major} "
                f"runtime={runtime_major}"
            )

        # ---- Load model from raw bytes --------------------------------
        raw = options["model_bytes"]
        if isinstance(raw, (bytes, bytearray)):
            buf = io.BytesIO(raw)
        else:
            # Handle list[int] (how ROS messages carry byte arrays)
            buf = io.BytesIO(bytes(raw))
        buf.seek(0)
        self._module = torch.jit.load(buf, map_location="cpu")
        self._module.eval()

        # ---- Normalisation -------------------------------------------
        self._scale_diag = np.asarray(options["scale"], dtype=np.float64)
        self._bias       = np.asarray(options["bias"],  dtype=np.float64)

        # ---- Per-timestep noise --------------------------------------
        T = int(options["T"])
        self._noise = [
            np.asarray(options[f"noise_{t}"], dtype=np.float64) for t in range(T)
        ]
        self._dU = len(self._noise[0]) if T > 0 else 0

        self.is_configured_ = True

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def get_action(self, t: int, X: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Normalise obs, run the TorchScript model, add noise.

        Mirrors C++ PyTorchController::get_action.
        """
        if not self.is_configured_ or self._module is None:
            return np.zeros(self._dU, dtype=np.float64)

        # 1. Observation normalisation: obs_scaled = obs * diag(scale) + bias
        obs_scaled = obs * self._scale_diag + self._bias

        # 2. Build [1, dO] float32 input tensor
        inp = torch.tensor(obs_scaled, dtype=torch.float32).unsqueeze(0)

        # 3. Forward pass with timing
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._module(inp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > self.MAX_INFERENCE_MS:
            LOGGER.warning(
                "PyTorchController: inference %.2f ms > %.1f ms threshold",
                elapsed_ms, self.MAX_INFERENCE_MS,
            )

        # 4. Unpack [1, dU] → (dU,) float64
        U = out.squeeze(0).numpy().astype(np.float64)

        # 5. Add pre-sampled noise
        if 0 <= t < len(self._noise):
            U = U + self._noise[t]

        return U
