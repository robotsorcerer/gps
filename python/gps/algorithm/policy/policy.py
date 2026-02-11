""" This file defines the base class for the policy. """
from __future__ import annotations

import abc

import numpy as np


class Policy(abc.ABC):
    """ Computes actions from states/observations. """

    @abc.abstractmethod
    def act(self, x: np.ndarray, obs: np.ndarray, t: int,
            noise: np.ndarray | None) -> np.ndarray:
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def set_meta_data(self, meta: object) -> None:
        """
        Set meta data for policy (e.g., domain image, multi modal observation sizes)
        Args:
            meta: meta data.
        """
        return
