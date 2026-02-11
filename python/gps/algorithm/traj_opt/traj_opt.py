""" This file defines the base trajectory optimization class. """
import abc
from typing import Dict, Any


class TrajOpt(object):
    """ Trajectory optimization superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]) -> None:
        self._hyperparams: Dict[str, Any] = hyperparams

    @abc.abstractmethod
    def update(self) -> Any:
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass.")


# TODO - Interface with C++ traj opt?
