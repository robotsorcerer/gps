""" This file defines the base class for dynamics estimation. """
from __future__ import annotations

import abc

import numpy as np


class Dynamics(abc.ABC):
    """ Dynamics superclass. """

    def __init__(self, hyperparams: dict) -> None:
        self._hyperparams = hyperparams

        # TODO - Currently assuming that dynamics will always be linear
        #        with X.
        # TODO - Allocate arrays using hyperparams dU, dX, T.

        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm = np.array(np.nan)
        self.fv = np.array(np.nan)
        self.dyn_covar = np.array(np.nan)  # Covariance.

    @abc.abstractmethod
    def update_prior(self, X: np.ndarray, U: np.ndarray) -> None:
        """ Update dynamics prior. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_prior(self) -> object:
        """ Returns prior object. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def fit(self, sample_list: object) -> None:
        """ Fit dynamics. """
        raise NotImplementedError("Must be implemented in subclass.")

    def copy(self) -> Dynamics:
        """ Return a copy of the dynamics estimate. """
        dyn = type(self)(self._hyperparams)
        dyn.Fm = np.copy(self.Fm)
        dyn.fv = np.copy(self.fv)
        dyn.dyn_covar = np.copy(self.dyn_covar)
        return dyn
