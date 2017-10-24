""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo, PolicyInfoRobust
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList

from random import shuffle
LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPS(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        policy_prior = self._hyperparams['policy_prior'] #self._hyperparams is from algorithm.py
        for m in range(self.M):  #self.M= # conditions
            # print('operating in mode: ', self._hyperparams['cost']['mode'])
            if self._hyperparams['cost']['mode'] == 'robust':
                self.cur[m].pol_info = PolicyInfoRobust(self._hyperparams)
                # self.cur[m].pol_info_v = PolicyInfoRobust(self._hyperparams) # not needed
            else:
                self.cur[m].pol_info = PolicyInfo(self._hyperparams) # from algorithm_utils.py
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior) # in hyperparams = PolicyPriorGMM

        self.policy_opt = self._hyperparams['policy_opt']['type']( #will be PolicyOptCaffe
            self._hyperparams['policy_opt'], self.dO, self.dU, self.dV #dO and dX are from hyperparams
        )

    def iteration(self, sample_lists):
        """
            Run iteration of MDGPS-based guided policy search.

            Args:
                sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):  #self.M is the # of the condition number
            self.cur[m].sample_list = sample_lists[m] #cur is every var in iteration data
            self._eval_cost(m)  #_eval_cost is defined in algorithm.py line 129

        # Update dynamics linearizations.
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M) # defined in algorithm_utils#L13: None
            ]
            self._update_policy()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def iteration_cl(self, sample_lists_prot, sample_lists):
        # Store the samples and evaluate the costs.
        for m in range(self.M):  # self.M is the # of the condition number
            self.cur[m].sample_list = sample_lists[m] # cur is every var in iteration data
            sample_prot = sample_lists_prot[m]
            self._eval_cost_cl(m, sample_lists_prot=sample_prot)  # _eval_cost is defined in algorithm.py line 220

        # Update dynamics linearizations.
        # self._update_dynamics_cl(sample_lists_prot=sample_prot)
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [   # traj_distr defined in algorithm.py#L67
                self.cur[cond].traj_distr for cond in range(self.M) # defined in algorithm_utils#L13: None
            ]
            self._update_policy()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def iteration_idg(self, sample_lists):
        # Store the samples and evaluate the costs.
        for m in range(self.M):                       # self.M is the # of the condition number
            self.cur[m].sample_list = sample_lists[m] # cur is every var in iteration data
            self._eval_cost_idg(m)                    # _eval_cost is defined in algorithm.py line 220

        # Update dynamics linearizations.
        self._update_dynamics_idg()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            # this for the local control generated trajectory
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M) # defined in algorithm_utils#L16: None
            ]
            # this is for the adversary generated trajectory
            self.new_traj_distr_adv = [
                self.cur[cond].traj_distr_adv for cond in range(self.M) # defined in algorithm_utils#L18: None
            ]
            """
            Update each policy in turn:
                update local p_u and p_v controller linear gaussian distributions
            """
            self._update_policy_robust() # update p(v|x)

        # Fit  linearized global policy. Step 5 in alg.
        for m in range(self.M):
            """
            we've folded the update of protagonist and antagonist into robust
            """
            # update global linearization and local linearization of pi_u, pi_v
            LOGGER.debug("cond: %d:  fitting p_u and p_v "
                         "policies to modeled dynamics from GMM", m)
            self._update_policy_fit_robust(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        """
        minimize augmented reward wrt control
        maximize augmented reward wrt adversary
        update robust joint trajectory
        """
        LOGGER.debug("updating robust trajectories")
        self._update_trajectories_robust()

        # S-step
        self._update_policy_robust()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _update_policy(self):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            # Note traj is defined in base class init function as init_lqr
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info #from algorithm_utils.py#L15
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def _update_policy_robust(self):
        """
            Compute the new robust policy.
            Steps:
                1. Compute the policy for the protagonist:  \pi(u | x)
                2. Compute the policy for the protagonist:  \pi(v | x)
        """
        dU, dV, dO, T = self.dU, self.dV, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu_u = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc_u, tgt_wt_u = np.zeros((0, T, dU, dU)), np.zeros((0, T))

        # # Compute target mean, and cov for adv sample.
        tgt_mu_v, tgt_prc_v = np.zeros((0, T, dV)), np.zeros((0, T, dV, dV))
        tgt_wt_v = np.zeros((0, T))

        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            # Note traj_u is the trajectory distribution for the control only
            #      traj_v is the trajectory distribution for the adversary only
            traj_u, pol_info = self.new_traj_distr[m], self.cur[m].pol_info #from algorithm_utils.py#L15
            traj_v = self.new_traj_distr_adv[m] #from algorithm_utils.py#L15
            # pol_info_v = self.cur[m].pol_info_v
            mu_u    = np.zeros((N, T, dU))
            prc_u   = np.zeros((N, T, dU, dU))

            mu_v    = np.zeros((N, T, dV))
            prc_v   = np.zeros((N, T, dV, dV))
            wt_u    = np.zeros((N, T))
            wt_v    = np.zeros((N, T))

            # Get time-indexed actions.
            for t in range(T):
                """
                    Compute actions along this trajectory.
                    inv_pol_covar_u is in alg/policy/lin_gauss_policy.py
                """
                prc_u[:, t, :, :] = np.tile(traj_u.inv_pol_covar_u[t, :, :],
                                          [N, 1, 1])
                prc_v[:, t, :, :] = np.tile(traj_v.inv_pol_covar_v[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu_v[i, t, :]  = (traj_v.Gv[t, :, :].dot(X[i, t, :]) + traj_v.gv[t, :])
                wt_u[:, t].fill(pol_info.pol_wt[t])
                wt_v[:, t].fill(pol_info.pol_wt[t])

            tgt_mu_u = np.concatenate((tgt_mu_u, mu_u))
            tgt_prc_u = np.concatenate((tgt_prc_u, prc_u))

            tgt_mu_v = np.concatenate((tgt_mu_v, mu_v))
            tgt_prc_v = np.concatenate((tgt_prc_v, prc_v))

            tgt_wt_u = np.concatenate((tgt_wt_u, wt_u))
            tgt_wt_v = np.concatenate((tgt_wt_v, wt_v))
            obs_data = np.concatenate((obs_data, samples.get_obs()))

        """
        Update each policy in turn:
            First: update protagonist policy
            Second: update antagonist policy
            Third: update conditional of protagonist on antagonist
        """

        # will update policy_chol_pol_covar_u if prot==true
        # update pi(u|x)
        self.policy_opt.update_locals(obs_data, tgt_mu_u, tgt_prc_u, tgt_wt_u, prot=True)
        # update pi(v|x)
        self.policy_opt.update_locals(obs_data, tgt_mu_v, tgt_prc_v, tgt_wt_v, prot=False)

    def _update_policy_fit(self, m):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[m].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _update_policy_fit_u(self, m):
        """
        Deprecated. This is now collapsed into update_policy_fit_robust

        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_info.pol_mu_prot, pol_info.pol_sig_prot = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[m].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_Gu, pol_info.pol_gu, pol_info.pol_Su = \
                policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_Su[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_Su[t, :, :])

    def _update_policy_fit_v(self, m):
        """
        Deprecated. This is now collapsed into update_policy_fit_robust

        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
        """
        dX, dV, T = self.dX, self.dV, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info_v  # get robust pol info object
        X = samples.get_X()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_info.pol_mu_adv, pol_info.pol_sig_adv = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[m].sample_list_adv)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_Gv, pol_info.pol_gv, pol_info.pol_Sv = \
                policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_Sv[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_Sv[t, :, :])

    def _update_policy_fit_robust(self, m):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Here I used the normal inverse wishart prior
        Args:
            m: Condition
        """
        T = self.T
        # Choose samples to use.
        samples     = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        obs = samples.get_obs().copy()
        obs_adv = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_mu_adv, pol_sig_adv = self.policy_opt.prob_v(obs_adv)[:2]

        pol_info.pol_mu_prot, pol_info.pol_sig_prot = pol_mu, pol_sig
        pol_info.pol_mu_adv, pol_info.pol_sig_adv = pol_mu_adv, pol_sig_adv

        # Update policy prior.
        policy_prior = pol_info.policy_prior #PolicyPriorGMM
        samples = SampleList(self.cur[m].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update_robust(samples, self.policy_opt, mode)

        # Fit local prot linearization and store in pol_info prot objects.
        pol_info.pol_Gu, pol_info.pol_gu, pol_info.pol_Su = \
                        policy_prior.fit_u(X, pol_mu, pol_sig)

        # Fit local adversarial linearization and store in adv pol_info object
        pol_info.pol_Gv, pol_info.pol_gv, pol_info.pol_Sv = \
                        policy_prior.fit_v(X, pol_mu_adv, pol_sig_adv)

        # update the covariance of the protagonist, adversary ...
        # and robust distribution in turn
        for t in range(T):
            pol_info.chol_pol_Su[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_Su[t, :, :])
            pol_info.chol_pol_Sv[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_Sv[t, :, :])

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        Algorithm._advance_iteration_variables(self)
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                    self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def _stepadjust(self):
        """
        Calculate new step sizes. This version uses the same step size
        for all conditions.
        """
        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev) # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt.estimate_cost(
                    prev_nn, self.prev[m].traj_info
            ).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(
                    prev_lg, self.prev[m].traj_info
            ).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            cur_laplace[m] = self.traj_opt.estimate_cost(
                    cur_nn, self.cur[m].traj_info
            ).sum()
            cur_mc[m] = self.cur[m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()
        if self._hyperparams['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('Previous cost: Laplace: %f, MC: %f',
                     prev_laplace, prev_mc)
        LOGGER.debug('Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('Actual cost: Laplace: %f, MC: %f',
                     cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv

    def compute_costs_protagonist(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dV, dX = traj_distr.T, traj_distr.dU, traj_distr.dV, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU+dV, dX+dU+dV))
        PKLv = np.zeros((T, dX+dU+dV))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_Su = np.linalg.solve(
                pol_info.chol_pol_Su[t, :, :],
                np.linalg.solve(pol_info.chol_pol_Su[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_Gu[t, :, :], pol_info.pol_gu[t, :]
            # here I augment the coeffs pertaining to v with zeros
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_Su).dot(KB), -KB.T.dot(inv_pol_Su), np.zeros((dX, dV))]),
                np.hstack([-inv_pol_Su.dot(KB), inv_pol_Su, np.zeros_like(inv_pol_Su)]),
                np.zeros((dV, dX + dV + dU))
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_Su).dot(kB), -inv_pol_Su.dot(kB),
                np.zeros_like(inv_pol_Su.dot(kB))
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv

    def compute_costs_adversary(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dV, dX = traj_distr.T, traj_distr.dU, traj_distr.dV, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU+dV, dX+dU+dV))
        PKLv = np.zeros((T, dX+dU+dV))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_Sv = np.linalg.solve(
                pol_info.chol_pol_Sv[t, :, :],
                np.linalg.solve(pol_info.chol_pol_Sv[t, :, :].T, np.eye(dV))
            )
            KB, kB = pol_info.pol_Gv[t, :, :], pol_info.pol_gv[t, :]
            # print('KB: {}, inv_pol_Su: {}, dX: {}'.format(KB.shape, inv_pol_Su.shape, dX))
            # here I augment the coeffs pertaining to v with zeros
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_Sv).dot(KB),  np.zeros((dX, dV)), -KB.T.dot(inv_pol_Sv)]),
                np.hstack([-inv_pol_Sv.dot(KB),  np.zeros_like(inv_pol_Sv), inv_pol_Sv]),
                np.zeros((dV, dX + dV + dU))
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_Sv).dot(kB), np.zeros_like(inv_pol_Sv.dot(kB)),
                -inv_pol_Sv.dot(kB),
            ])
            # print('fCm: {}, Cm: {}, PKLm: {}', fCm.shape, Cm.shape, PKLm.shape, PKLv.shape)
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            # print('cv: {}, PKLv: {}: ', cv.shape, PKLv.shape)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv


    def compute_costs_robust(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr_robust
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dV, dX = traj_distr.T, traj_distr.dU, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU+dV, dX+dU+dV))
        PKLv = np.zeros((T, dX+dU+dV))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_Suv = np.linalg.solve(
                pol_info.chol_pol_Suv[t, :, :],
                np.linalg.solve(pol_info.chol_pol_Suv[t, :, :].T, np.eye(dU)) # follow lin_gauss_init formulation
            )
            KB, kB = pol_info.pol_Guv[t, :, :], pol_info.pol_guv[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_Suv).dot(KB), -KB.T.dot(inv_pol_Suv)]),
                np.hstack([-inv_pol_Suv.dot(KB), inv_pol_Suv])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_Suv).dot(kB), -inv_pol_Suv.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv
