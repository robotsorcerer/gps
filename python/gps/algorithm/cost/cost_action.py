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
        self.gamma = self._hyperparams['gamma']
        self.mode = self._hyperparams['mode']

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
                lv = 0.5 * sum(wu * u^2) - 2 * gamma * wu * v
                lvv = 0.5 * sum(wu * u^2) - 2 * gamma * wu
            """
            sample_prot = kwargs['sample_prot']
            # if sample_prot is not None:
            sample_prot_u = sample_prot.get_U()

            # print('sample_prot_u: ', sample_prot_u.shape, ' | sample_u: ', sample_u.shape) #('sample_prot_u: ', (100, 7), ' | sample_u: ', (100, 7))
            l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_prot_u ** 2), axis=1) - \
                self.gamma * np.sum( self._hyperparams['wu'] * (sample_u ** 2), axis=1)  # shape 100
            lv = 0.5 * np.sum(self._hyperparams['wu'] * (sample_prot_u ** 2), axis=1) - \
                 (2 * self.gamma * np.sum(self._hyperparams['wu'] * sample_u, axis=1)) # will be of shape (100,)
            lv = np.expand_dims(lv, axis=1) #will be (100, 1)
            lv = np.tile(lv, Du) #now (100, 7)

            lx = np.zeros((T, Dx))

            # compute 2nd order control derivative
            lvv_t1= 0.5 * np.sum(self._hyperparams['wu'] * (sample_prot_u ** 2), axis=1) # shape (100,)
            lvv_t1 = np.expand_dims(lvv_t1, axis=1) # shape(100, 1)

            lvv_t1 = np.tile(lvv_t1, Du)  # shape (100, Du)
            lvv_t1 = np.expand_dims(lvv_t1, axis=2) # shape (100, Du, 1)
            lvv_t1 = np.tile(lvv_t1, Du) # shape(100, Du, Du)
            lvv_t2= np.tile(np.diag(2 * self.gamma * self._hyperparams['wu']), [T, 1, 1]) #shape (100, 7, 7)
            # print('lvv_t1.shape: {},| lvv_t2.shape: {} ', lvv_t1.shape, lvv_t2.shape)
            # if lvv_t2.shape[2] != lvv_t1.shape[2]:
            #     lvv_t2 = np.empty_like(lvv_t1)
            #     lvv_t2 = np.tile(np.diag(2 * self.gamma * np.ones(Dx)), [T, 1, 1])
            lvv = lvv_t1 - lvv_t2 # shape (100, Du, Du)

            lxx = np.zeros((T, Dx, Dx))
            lvx = np.zeros((T, Du, Dx))
            return -l, -lx, -lv, -lxx, -lvv, -lvx
        else:
            os._exit("unknown mode. Cost Action Mode should either be protagonist or antagonist ")
