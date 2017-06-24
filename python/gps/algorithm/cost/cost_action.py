""" This file defines the torque (action) cost. """
import copy, os

import numpy as np

from gps.algorithm.cost.config import COST_ACTION
from gps.algorithm.cost.cost import Cost


class CostAction(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, config)
        # self._hyperparams ='antagonist' #could also be protagonist

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        gamma, mode = 0, 'antagonist' #could also be protagonist
        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        if mode == 'protagonist':
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
            lu = self._hyperparams['wu'] * sample_u
            lx = np.zeros((T, Dx))
            luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
            lxx = np.zeros((T, Dx, Dx))
            lux = np.zeros((T, Du, Dx))

            return l, lx, lu, lxx, luu, lux
        elif mode == 'antagonist':
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1) + \
                gamma * np.linalg.norm(sample_u ** 2, ord=2) #* np.linalg.norm(sample_u ** 2, ord=2)
            lu = self._hyperparams['wu'] * sample_u - \
                 2 * gamma * sample_u
            lx = np.zeros((T, Dx))
            luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1]) - \
                  2 *  gamma
            lxx = np.zeros((T, Dx, Dx))
            lux = np.zeros((T, Du, Dx))

            return -l, -lx, -lu, -lxx, -luu, -lux
        else:
            os._exit("unknown mode. Cost Action Mode should either be protagonist or antagonist ")
