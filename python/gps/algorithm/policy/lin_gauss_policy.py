""" This file defines the linear Gaussian policy class. """
import numpy as np

from gps.algorithm.policy.policy import Policy
from gps.utility.general_utils import check_shape


class LinearGaussianPolicy(Policy):
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        Policy.__init__(self)

        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def act(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self, noise):
        """
        Fold noise into k.
        Args:
            noise: A T x Du noise vector with mean 0 and variance 1.
        Returns:
            k: A T x dU bias vector.
        """
        k = np.zeros_like(self.k)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar[i].T.dot(noise[i])
            k[i] = scaled_noise + self.k[i]
        return k

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K), np.zeros_like(self.k),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy

class LinearGaussianPolicyRobust(Policy):
    """
    Time-varying linear Gaussian policy for combined trajectory between
    adversary and protagonist.
    u^* = gu + Gu * x + noise_u, where noise_u ~ N(0, chol_pol_covar_u) # local protagonist policy
    v^* = gv + Gv * x + noise_v, where noise_v ~ N(0, chol_pol_covar_v) # local adversarial policy
    U   = g  + G  * x + noise,   where noise ~ N(0, chol_pol_covar)             # joint global policy
    """
    def __init__(self, Gu, gu, pol_covar_u, chol_pol_covar_u, inv_pol_covar_u, \
                       Gv, gv, pol_covar_v, chol_pol_covar_v, inv_pol_covar_v):
        Policy.__init__(self)

        # Assume G has the correct shape, and make sure others match.
        self.T = Gu.shape[0]
        self.dU = Gu.shape[1]
        self.dV = Gv.shape[1]
        self.dX = Gu.shape[2]

        check_shape(gu, (self.T, self.dU))
        check_shape(pol_covar_u, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar_u, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar_u, (self.T, self.dU, self.dU))

        check_shape(gv, (self.T, self.dV))
        check_shape(pol_covar_v, (self.T, self.dV, self.dV))
        check_shape(chol_pol_covar_v, (self.T, self.dV, self.dV))
        check_shape(inv_pol_covar_v, (self.T, self.dV, self.dV))

        self.Gu = Gu
        self.gu = gu
        self.pol_covar_u = pol_covar_u
        self.chol_pol_covar_u = chol_pol_covar_u
        self.inv_pol_covar_u = inv_pol_covar_u

        self.Gv = Gv
        self.gv = gv
        self.pol_covar_v = pol_covar_v
        self.chol_pol_covar_v = chol_pol_covar_v
        self.inv_pol_covar_v = inv_pol_covar_v

    def act_u(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        u = self.Gu[t].dot(x) + self.gu[t]
        u += self.chol_pol_covar_u[t].T.dot(noise)
        return u

    def act_v(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        v = self.Gv[t].dot(x) + self.gv[t]
        v += self.chol_pol_covar_v[t].T.dot(noise)
        return v

    def fold_gu(self, noise):
        """
        Fold noise into gu.
        Args:
            noise: A T x Du noise vector with mean 0 and variance 1.
        Returns:
            gu: A T x dU bias vector.
        """
        gu = np.zeros_like(self.gu)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar_u[i].T.dot(noise[i])
            gu[i] = scaled_noise + self.gu[i]
        return gu

    def fold_gv(self, noise):
        """
        Fold noise into gv.
        Args:
            noise: A T x Du noise vector with mean 0 and variance 1.
        Returns:
            gv: A T x dU bias vector.
        """
        gv = np.zeros_like(self.gv)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar_v[i].T.dot(noise[i])
            gv[i] = scaled_noise + self.gv[i]
        return gv

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = LinearGaussianPolicyRobust(
            np.zeros_like(self.Gu), np.zeros_like(self.gu),
            np.zeros_like(self.pol_covar_u), np.zeros_like(self.chol_pol_covar_u),
            np.zeros_like(self.inv_pol_covar_u),
            np.zeros_like(self.Gv), np.zeros_like(self.gv),
            np.zeros_like(self.pol_covar_v), np.zeros_like(self.chol_pol_covar_v),
            np.zeros_like(self.inv_pol_covar_v)   )
        policy.Gu.fill(np.nan)
        policy.gu.fill(np.nan)
        policy.pol_covar_u.fill(np.nan)
        policy.chol_pol_covar_u.fill(np.nan)
        policy.inv_pol_covar_u.fill(np.nan)

        policy.Gv.fill(np.nan)
        policy.gv.fill(np.nan)
        policy.pol_covar_v.fill(np.nan)
        policy.chol_pol_covar_v.fill(np.nan)
        policy.inv_pol_covar_v.fill(np.nan)

        return policy
