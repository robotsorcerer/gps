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
        self.config = config
        Cost.__init__(self, config)
        # self._hyperparams ='antagonist' #could also be protagonist

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        self.gamma = 1e1
        self.mode = 'antagonist'
        # gamma, mode = 0, 'antagonist' #could also be protagonist
        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        if self.mode == 'protagonist':
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
            lu = self._hyperparams['wu'] * sample_u
            lx = np.zeros((T, Dx))
            luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
            lxx = np.zeros((T, Dx, Dx))
            lux = np.zeros((T, Du, Dx))

            return l, lx, lu, lxx, luu, lux
        elif self.mode == 'antagonist':
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1) + \
                self.gamma * np.linalg.norm(sample_u ** 2, ord=2) #* np.linalg.norm(sample_u ** 2, ord=2)
            lu = self._hyperparams['wu'] * sample_u - \
                 2 * self.gamma * sample_u
            lx = np.zeros((T, Dx))
            luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1]) - \
                  2 *  self.gamma
            lxx = np.zeros((T, Dx, Dx))
            lux = np.zeros((T, Du, Dx))

            return -l, -lx, -lu, -lxx, -luu, -lux
        else:
            os._exit("unknown mode. Cost Action Mode should either be protagonist or antagonist ")
