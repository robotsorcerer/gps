""" This file defines the torque (action) cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_ACTION
from gps.algorithm.cost.cost import Cost


class CostAction(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        if self._hyperparams['mode'] == 'protagonist':
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
            lu = self._hyperparams['wu'] * sample_u
            lx = np.zeros((T, Dx))
            luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
            lxx = np.zeros((T, Dx, Dx))
            lux = np.zeros((T, Du, Dx))
        else:
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1) - \
                self._hyperparams['gamma'] * (np.linalg.norm(sample_u, ord=2) ** 2)
            lu = self._hyperparams['wu'] * sample_u - \
                 2 * self._hyperparams['gamma'] * sample_u
            lx = np.zeros((T, Dx))
            luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1]) - \
                  2 * self._hyperparams['gamma']
            lxx = np.zeros((T, Dx, Dx))
            lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux
