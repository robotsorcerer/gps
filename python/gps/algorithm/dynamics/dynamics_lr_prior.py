""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior


class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = \
                self._hyperparams['prior']['type'](self._hyperparams['prior']) # DynamicsPriorGMM

    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)  #this is calling update in gmm_prior

    def update_prior_robust(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        V = samples.get_V()
        self.prior.update_robust(X, U, V)  #this is calling update in gmm_prior

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    #TODO: Merge this with DynamicsLR.fit - lots of duplicated code.
    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = self._hyperparams['regularization']
            Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,
                        mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar

    def fit_robust(self, X, U, V):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU, dV = U.shape[2], V.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU+dV])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU+dV)
        ip = slice(dX+dU+dV, dX+dU+dV+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], V[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval_robust(dX, dU, dV, Ys)
            sig_reg = np.zeros((dX+dU+dV+dX, dX+dU+dV+dX))
            sig_reg[it, it] = self._hyperparams['regularization']
            Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,
                        mu0, Phi, mm, n0, dwts, dX+dU+dV, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar
