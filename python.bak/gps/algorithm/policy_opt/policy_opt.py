""" This file defines the base policy optimization class. """
from __future__ import annotations

import abc


class PolicyOpt(abc.ABC):
    """ Policy optimization superclass. """

    def __init__(self, hyperparams: dict, dO: int, dU: int, dV: int) -> None:
        self._hyperparams = hyperparams
        self._dO = dO
        self._dU = dU
        self._dV = dV

    @abc.abstractmethod
    def update(self) -> object:
        """ Update policy. """
        raise NotImplementedError("Must be implemented in subclass.")
