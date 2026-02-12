"""
Translates gps_agent_pkg/src/neuralnetwork.cpp + include/neuralnetwork.h.

NeuralNetwork is an abstract base class.  The only concrete subclass used
in this codebase is PyTorchController, so this module provides the interface
only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NeuralNetwork(ABC):
    """
    Abstract base for neural-network policy forward passes.

    Mirrors C++ gps_control::NeuralNetwork.
    """

    def __init__(self) -> None:
        self.scale_: np.ndarray = np.array([], dtype=np.float64)
        self.bias_:  np.ndarray = np.array([], dtype=np.float64)
        self.input_scaled_: np.ndarray = np.array([], dtype=np.float64)

    def set_scalebias(self, scale: np.ndarray, bias: np.ndarray) -> None:
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.bias_  = np.asarray(bias,  dtype=np.float64)

    def set_weights(self, weights) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass and return the output vector."""
        ...
