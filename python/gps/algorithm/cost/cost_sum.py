""" This file defines a cost sum of arbitrary other costs. """
import copy, os

from gps.algorithm.cost.config import COST_SUM
from gps.algorithm.cost.cost import Cost


class CostSum(Cost):
    """ A wrapper cost function that adds other cost functions. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        self.gamma = config['gamma']
        self.mode = config['mode']
        Cost.__init__(self, config)

        self._costs = []
        self._weights = self._hyperparams['weights']

        #[torque_cost, fk_cost, final_cost] = [Cost_Action, Cost_FK, Cost_FK]
        for cost in self._hyperparams['costs']:
            self._costs.append(cost['type'](cost))

    def eval(self, sample, **kwargs):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """
        self.mode = 'antagonist'
        if 'sample_prot' in kwargs:
            sample_prot = kwargs['sample_prot']
            l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample, sample_prot=sample_prot)
        else:
            l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample) #we are optimizing cost action

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]
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
