""" This file defines the base class for dynamics estimation. """
import abc
from typing import Dict, Any, Optional

import numpy as np
import numpy.typing as npt


class Dynamics(object):
    """ Dynamics superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]) -> None:
        self._hyperparams: Dict[str, Any] = hyperparams

        # TODO - Currently assuming that dynamics will always be linear
        #        with X.
        # TODO - Allocate arrays using hyperparams dU, dX, T.

        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm: npt.NDArray[np.float64] = np.array(np.nan)
        self.fv: npt.NDArray[np.float64] = np.array(np.nan)
        self.dyn_covar: npt.NDArray[np.float64] = np.array(np.nan)  # Covariance.

    @abc.abstractmethod
    def update_prior(self, X: npt.NDArray, U: npt.NDArray) -> None:
        """ Update dynamics prior. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_prior(self) -> Optional[Any]:
        """ Returns prior object. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def fit(self, sample_list: Any) -> None:
        """ Fit dynamics. """
        raise NotImplementedError("Must be implemented in subclass.")

    def copy(self) -> 'Dynamics':
        """ Return a copy of the dynamics estimate. """
        dyn = type(self)(self._hyperparams)
        dyn.Fm = np.copy(self.Fm)
        dyn.fv = np.copy(self.fv)
        dyn.dyn_covar = np.copy(self.dyn_covar)
        return dyn
