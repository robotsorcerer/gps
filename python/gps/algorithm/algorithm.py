""" This file defines the base algorithm class. """

import abc
import copy
import logging

import random
import numpy as np

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition


LOGGER = logging.getLogger(__name__)


class Algorithm(object):
    """ Algorithm superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG) #cost is none here
        config.update(hyperparams)  #cost becomes a dict
        self._hyperparams = config

        if 'train_conditions' in hyperparams:
            self._cond_idx = hyperparams['train_conditions']  #Non existent for mjc_mdgps
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['conditions']  #will be 4
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU # agent.py#L25. will be 7
        self.dX = self._hyperparams['dX'] = agent.dX # agent.py#L40
        self.dO = self._hyperparams['dO'] = agent.dO #not found

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)] #see algorithm_utils#L8
        self.prev = [IterationData() for _ in range(self.M)] #see algorithm_utils#L8

        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']  #is a True bool from algorithm/config.py

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo() # Line 23
            if self._hyperparams['fit_dynamics']: # True
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics) #Hyperparams: DynamicsLRPrior
            init_traj_distr = extract_condition( #L28 general_utils.py
                self._hyperparams['init_traj_distr'], self._cond_idx[m] #L84 hyperparams
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr) #will be init_lqr
            #init_lqr is defined in algorithm/policy/lin_gauss_init
        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )
        if type(hyperparams['cost']) == list: #see config.py
            self.cost = [
                hyperparams['cost'][i]['type'](hyperparams['cost'][i])  #cost is a list of CostSum
                for i in range(self.M)
            ]
        else:
            self.cost = [
                hyperparams['cost']['type'](hyperparams['cost'])
                for _ in range(self.M)
            ]
        self.base_kl_step = self._hyperparams['kl_step']

    @abc.abstractmethod
    def iteration(self, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def iteration_cl(self, sample_lists_prot, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_X()
            U = cur_data.get_U()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_dynamics_cl(self, sample_lists_prot):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """

        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            sample_prot = sample_lists_prot[m]

            X = cur_data.get_X()
            U = cur_data.get_U()

            # index last time step and sqeeze along first singleton dimesion
            U_adv = np.squeeze(sample_prot.get_U()[-1:,:,:], axis=(0,))
            # then add U
            # U += U_adv

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = \
                    self.traj_opt.update(cond, self)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)  # cur is every var in iteration data

        # Compute cost.
        cs = np.zeros((N, T))  # see algorithm_utils.py
        cc = np.zeros((N, T))   # Cost estimate constant term.
        cv = np.zeros((N, T, dX+dU))    # Cost estimate vector term.
        Cm = np.zeros((N, T, dX+dU, dX+dU)) # Cost estimate matrix term.
        for n in range(N):
            sample = self.cur[cond].sample_list[n] # cur is every var in iteration data
            # Get costs.  Self.cost will be a CostSum object see mdgps/antag/hyperparams#L128
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.

    def _eval_cost_cl(self, cond, sample_lists_prot=None):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
            sample_lists = # 5
            sample_lists_prot is the list of the protagonist policies: 8 in #
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)  # cur is every var in iteration data
        # Compute cost.
        cs = np.zeros((N, T))  # see algorithm_utils.py
        cc = np.zeros((N, T))   # Cost estimate constant term.
        cv = np.zeros((N, T, dX+dU))    # Cost estimate vector term.
        Cm = np.zeros((N, T, dX+dU, dX+dU)) # Cost estimate matrix term.
        for n in range(N):
            sample = self.cur[cond].sample_list[n] # == 1 Sample object
            sample_prot = sample_lists_prot[n]
            # Get costs.  Self.cost will be a CostSum object see mdgps/antag/hyperparams#L128
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample, sample_prot=sample_prot)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()

            #retrieve state and action for adversary
            #index last time step and sqeeze along first singleton dimesion
            # X_adv = np.squeeze(sample_prot.get_X()[-1:,:,:], axis=(0,))
            # U_adv = np.squeeze(sample_prot.get_U()[-1:,:,:], axis=(0,))

            X_adv = np.mean(sample_prot.get_X(), axis=0)
            U_adv = np.mean(sample_prot.get_U(), axis=0)

            # new state space dynamics will be the effect of the adv and protag
            # X += X_adv
            U += U_adv

            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_random_state'] = random.getstate()
        state['_np_random_state'] = np.random.get_state()
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__ = state
        random.setstate(state.pop('_random_state'))
        np.random.set_state(state.pop('_np_random_state'))
