""" This file defines the base cost class. """
from __future__ import annotations

import abc
from typing import Any


class Cost(abc.ABC):
    """ Cost superclass. """

    def __init__(self, hyperparams: dict) -> None:
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def eval(self, sample: Any) -> tuple:
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample.
        """
        raise NotImplementedError("Must be implemented in subclass.")
