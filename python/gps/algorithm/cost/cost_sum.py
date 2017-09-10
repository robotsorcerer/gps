""" This file defines a cost sum of arbitrary other costs. """
import copy, os
import numpy as np
from gps.algorithm.cost.config import COST_SUM
from gps.algorithm.cost.cost import Cost


class CostSum(Cost):
    """ A wrapper cost function that adds other cost functions. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._costs = []
        self._weights = self._hyperparams['weights']
        # print self._hyperparams['costs']

        # [torque_cost, fk_cost, final_cost] = [Cost_Action, Cost_FK, Cost_FK]
        for cost in self._hyperparams['costs']:
            self._costs.append(cost['type'](cost))
            # fix gamma and mode from hyperparams file
            self.gamma = cost['gamma'] if 'gamma' in cost else None
            self.mode = cost['mode'] if 'mode' in cost else None

    def eval(self, sample, **kwargs):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]

        # print('hyperparams file: ', self._hyperparams)
        # print('mode: ', self.mode)

        if self.mode == 'antagonist':  # 'sample_prot' in kwargs: #
            sample_prot = kwargs['sample_prot']
            # note cost[0] is torque/action cost
            l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample, sample_prot=sample_prot)

            l = l * weight
            lx = lx * weight
            lu = lu * weight
            lxx = lxx * weight
            luu = luu * weight
            lux = lux * weight
            for i in range(1, len(self._costs)):
                pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
                weight = self._weights[i]
                l = l + pl * weight
                lx = lx + plx * weight
                lu = lu + plu * weight
                lxx = lxx + plxx * weight
                luu = luu + pluu * weight
                lux = lux + plux * weight
            return l, lx, lu, lxx, luu, lux #don't negate here cause torque and fk costs are already negated

        elif self.mode=='robust': #'sample_adv' in kwargs:
            sample_adv = kwargs['sample_adv']
            l, lx, lu, lv, lxx, luu, lvv, lux, lvx = self._costs[0].eval(sample, sample_adv=sample_adv)

            l   = l   * weight

            lx  = lx  * weight
            lu  = lu  * weight
            lv  = lv  * weight

            lxx = lxx * weight
            luu = luu * weight
            lvv = lvv * weight

            lux = lux * weight
            lvx = lvx * weight

            for i in range(1, len(self._costs)): # evals fk_cost and final_cost
                pl, plx, plu,  plv, plxx, pluu, plvv, plux, plvx = self._costs[i].eval(sample)
                weight = self._weights[i]
                l = l + pl * weight
                lx = lx + plx * weight
                lu = lu + plu * weight
                lv = lv + plv * weight

                lxx = lxx + plxx * weight
                luu = luu + pluu * weight
                lvv = lvv + plvv * weight

                lux = lux + plux * weight
                lvx = lvx + plvx * weight

            return l, lx, lu, lv, lxx, luu, lvv, lux, lvx #don't negate here cause torque and fk costs are already negated

        else:
            # print('evaluating cost in protagonist')
            l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample) #we are optimizing cost action

            l = l * weight
            lx = lx * weight
            lu = lu * weight
            lxx = lxx * weight
            luu = luu * weight
            lux = lux * weight
            for i in range(1, len(self._costs)):
                pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
                weight = self._weights[i]
                l = l + pl * weight
                lx = lx + plx * weight
                lu = lu + plu * weight
                lxx = lxx + plxx * weight
                luu = luu + pluu * weight
                lux = lux + plux * weight
            return l, lx, lu, lxx, luu, lux #don't negate here cause torque and fk costs are already negated
