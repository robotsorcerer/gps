""" This file defines the base cost class. """
import abc
from typing import Dict, Any, Tuple

import numpy.typing as npt


class Cost(object):
    """ Cost superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]) -> None:
        self._hyperparams: Dict[str, Any] = hyperparams

    @abc.abstractmethod
    def eval(self, sample: Any) -> Tuple[npt.NDArray, ...]:
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample.
        """
        raise NotImplementedError("Must be implemented in subclass.")
