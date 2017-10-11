""" This file defines a GMM prior for policy linearization. """
import copy
import logging

import numpy as np

from gps.algorithm.policy.config import POLICY_PRIOR_GMM
from gps.utility.gmm import GMM
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior


LOGGER = logging.getLogger(__name__)


class PolicyPriorGMM(object):
    """
    A policy prior encoded as a GMM over [x_t, u_t] points, where u_t is
    the output of the policy for the given state x_t. This prior is used
    when computing the linearization of the policy.

    See the method AlgorithmBADMM._update_policy_fit, in
    python/gps/algorithm.algorithm_badmm.py.

    Also see the GMM dynamics prior, in
    python/gps/algorithm/dynamics/dynamics_prior_gmm.py. This is a
    similar GMM prior that is used for the dynamics estimate.
    """
    def __init__(self, hyperparams):
        """
        Hyperparameters:
            min_samples_per_cluster: Minimum number of samples.
            max_clusters: Maximum number of clusters to fit.
            max_samples: Maximum number of trajectories to use for
                fitting the GMM at any given time.
            strength: Adjusts the strength of the prior.
        """
        config = copy.deepcopy(POLICY_PRIOR_GMM)
        config.update(hyperparams)
        self._hyperparams = config
        self.X = None
        self.obs = None
        self.gmm = GMM()
        # TODO: handle these params better (e.g. should depend on N?)
        self._min_samp = self._hyperparams['min_samples_per_cluster']
        self._max_samples = self._hyperparams['max_samples']
        self._max_clusters = self._hyperparams['max_clusters']
        self._strength = self._hyperparams['strength']

    def update(self, samples, policy_opt, mode='add'):
        """
        Update GMM using new samples or policy_opt.
        By default does not replace old samples.

        Args:
            samples: SampleList containing new samples
            policy_opt: PolicyOpt containing current policy
        """
        X, obs = samples.get_X(), samples.get_obs()

        if self.X is None or mode == 'replace':
            self.X = X
            self.obs = obs
        elif mode == 'add' and X.size > 0:
            self.X = np.concatenate([self.X, X], axis=0)
            self.obs = np.concatenate([self.obs, obs], axis=0)
            # Trim extra samples
            # TODO: how should this interact with replace_samples?
            N = self.X.shape[0]
            if N > self._max_samples:
                start = N - self._max_samples
                self.X = self.X[start:, :, :]
                self.obs = self.obs[start:, :, :]

        # Evaluate policy at samples to get mean policy action.
        U = policy_opt.prob(self.obs.copy())[0]
        # Create the dataset
        N, T = self.X.shape[:2]
        dO = self.X.shape[2] + U.shape[2]
        XU = np.reshape(np.concatenate([self.X, U], axis=2), [T * N, dO])
        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))

        LOGGER.debug('Generating %d clusters for policy prior GMM.', K)
        self.gmm.update(XU, K)

    def update_robust(self, samples, policy_opt, mode='add'):
        """
        Update GMM using new samples or policy_opt.
        By default does not replace old samples.

        Args:
            samples: SampleList containing new samples
            policy_opt: PolicyOpt containing current policy
        """
        X, obs, obs_adv = samples.get_X(), samples.get_obs(), samples.get_obs_adv()

        if self.X is None or mode == 'replace':
            self.X = X
            self.obs = obs
            self.obs_adv = obs_adv
        elif mode == 'add' and X.size > 0:
            self.X = np.concatenate([self.X, X], axis=0)
            self.obs = np.concatenate([self.obs, obs], axis=0)
            self.obs_adv = np.concatenate([self.obs_adv, obs_adv], axis=0)
            # Trim extra samples
            # TODO: how should this interact with replace_samples?
            N = self.X.shape[0]
            if N > self._max_samples:
                start = N - self._max_samples
                self.X = self.X[start:, :, :]
                self.obs = self.obs[start:, :, :]
                self.obs_adv = self.obs_adv[start:, :, :]

        # Evaluate policy at samples to get mean policy action.
        U = policy_opt.prob(self.obs.copy())[0]
        V = policy_opt.prob(self.obs_adv.copy())[0]
        # Create the dataset
        N, T = self.X.shape[:2]
        dO = self.X.shape[2] + U.shape[2] + V.shape[2]
        # XU = np.reshape(np.concatenate([self.X, U], axis=2), [T * N, dO])
        XUV = np.reshape(np.concatenate([self.X, U, V], axis=2), [T * N, dO])
        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))

        LOGGER.debug('Generating %d clusters for policy prior robust GMM.', K)
        self.gmm.update(XUV, K)

    def eval(self, Ts, Ps):
        """ Evaluate prior. """
        # Construct query data point.
        pts = np.concatenate((Ts, Ps), axis=1)
        # Perform query.
        mu0, Phi, m, n0 = self.gmm.inference(pts)
        # Factor in multiplier.
        n0 *= self._strength
        m *= self._strength
        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

    # TODO: Merge with non-GMM policy_prior?
    def fit(self, X, pol_mu, pol_sig):
        """
        Fit policy linearization.

        Args:
            X: Samples (N, T, dX)
            pol_mu: Policy means (N, T, dU)
            pol_sig: Policy covariance (N, T, dU)
        """
        N, T, dX = X.shape
        dU = pol_mu.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Collapse policy covariances. (This is only correct because
        # the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])

        # Fit policy linearization with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = \
                    gauss_fit_joint_prior(Ys,
                            mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig
        return pol_K, pol_k, pol_S

    def fit_robust(self, X, pol_mu, pol_sig, pol_mu_adv, pol_sig_adv):
        """
        Fit policy linearization.

        Args:
            X: Samples (N, T, dX)
            pol_mu: Policy means (N, T, dU)
            pol_sig: Policy covariance (N, T, dU)
            pol_mu_adv: Policy means (N, T, dV)
            pol_sig_adv: Policy covariance (N, T, dV)
        """
        N, T, dX = X.shape
        dU = pol_mu.shape[2]
        dV = pol_mu.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Collapse policy covariances. (This is only correct because
        # the policy doesn't depend on state).
        pol_sig     = np.mean(pol_sig, axis=0)
        pol_sig_adv = np.mean(pol_sig_adv, axis=0)

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])

        pol_Gv = np.zeros([T, dV, dX])
        pol_gv = np.zeros([T, dV])
        pol_Sv = np.zeros([T, dV, dV])

        # condition protagonist policy on adversarial policy
        n1, n2 = pol_mu.shape[0], pol_mu_adv.shape[0]
        # find weigted mean
        mu_weighted = (n1 * pol_mu + n2 * pol_mu_adv)/ (n1 + n2)
        # find pooled variance
        # sigma_pooled = (
        #                 (n1 * n1 * pol_sig) + (n2 * n2 * pol_sig_adv) + \
        #                 ((n1 * pol_mu) + (n2 * pol_mu_adv)) ** 2 - \
        #                 (pol_mu.dot(pol_mu_adv).T)
        #                 ) /  (n1 + n2)**2
        # Fit policy linearization with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = mu_weighted[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            print('pol_mu, pol_mu_adv, mu_weighted: ',pol_mu.shape, pol_mu_adv.shape, mu_weighted.shape)
            print('pol_sig, pol_sig_adv', pol_sig.shape, pol_sig_adv.shape)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = \
                    gauss_fit_joint_prior(Ys,
                            mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig
        return pol_K, pol_k, pol_S
