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

        self._config = config

    def eval(self, sample, **kwargs):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """

        self.gamma = self._config['gamma']
        self.mode = self._config['mode']
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
            """
                sample_u is now adversary's local u
                sample_prot_u was the protagonist local u
                we are maximizing with respect to v as in the IROS abstract
            """
            sample_prot = kwargs['sample_prot']
            sample_prot_u = sample_prot.get_U()

            # print('sample_prot_u: ', sample_prot_u.shape, ' | sample_u: ', sample_u.shape)
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_prot_u ** 2), axis=1) - \
                self.gamma * np.sum( self._hyperparams['wu'] * (sample_u ** 2), axis=1)  # shape 100
            lv = 0.5 * np.sum(self._hyperparams['wu'] * (sample_prot_u ** 2), axis=1) - \
                 (2 * self.gamma * np.sum(self._hyperparams['wu'] * sample_u, axis=1)) # 1st term shape (100,) 2nd term shape shape 7 x 7 due to 2nd term
            lx = np.zeros((T, Dx))
            lvv_temp = 0.5 * np.sum(self._hyperparams['wu'] * (sample_prot_u ** 2), axis=1) #shape (100,)
            print('l shape: {}, lv shape: {}, lvv_temp: {}'.format(l.shape, lv.shape, lvv_temp.shape))
            lvv = np.tile(np.diag(self._hyperparams['wu'] - 2 * self.gamma * self._hyperparams['wu']), [T, 1, 1]) # shape 100, 7, 7
            lxx = np.zeros((T, Dx, Dx))
            lvx = np.zeros((T, Du, Dx))

            return -l, -lx, -lv, -lxx, -lvv, -lvx
            # l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1) - \
            #     self.gamma * np.linalg.norm(sample_u ** 2, ord=2)
            # lu = self._hyperparams['wu'] * sample_u - \
            #      2 * self.gamma * sample_u
            # lx = np.zeros((T, Dx))
            # luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1]) - \
            #       2 *  self.gamma
            # lxx = np.zeros((T, Dx, Dx))
            # lux = np.zeros((T, Du, Dx))

            return -l, -lx, -lu, -lxx, -luu, -lux

        else:
            os._exit("unknown mode. Cost Action Mode should either be protagonist or antagonist ")
