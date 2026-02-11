""" This file defines the base trajectory optimization class. """
from __future__ import annotations

import abc


class TrajOpt(abc.ABC):
    """ Trajectory optimization superclass. """

    def __init__(self, hyperparams: dict) -> None:
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def update(self) -> None:
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass.")


# TODO - Interface with C++ traj opt?
