/**
 * @file traj_opt_lqr.cpp
 * @brief Implementation of TrajOptLQR class.
 */

#include "traj_opt_lqr.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gps {

TrajOptLQR::TrajOptLQR(const Hyperparams& hyperparams)
    : TrajOpt(hyperparams),
      del0_(get_hyperparam_or<double>(hyperparams, "del0", 1e-4)),
      min_eta_(get_hyperparam_or<double>(hyperparams, "min_eta", 1e-8)),
      max_eta_(get_hyperparam_or<double>(hyperparams, "max_eta", 1e16)),
      max_iters_(get_hyperparam_or<int>(hyperparams, "max_iters", 50)),
      kl_step_(get_hyperparam_or<double>(hyperparams, "kl_step", 1.0)) {
}

TrajOptResult TrajOptLQR::update(
    int /*condition*/, const Algorithm& /*algorithm*/) {
    // Full implementation would extract data from algorithm and run backward pass
    // This is a placeholder structure showing the interface

    throw std::runtime_error(
        "TrajOptLQR::update requires Algorithm integration - "
        "see backward_pass for core computation");
}

TrajOptResultRobust TrajOptLQR::update_robust(
    int /*condition*/, const Algorithm& /*algorithm*/) {
    // Full implementation for game-theoretic trajectory optimization
    throw std::runtime_error(
        "TrajOptLQR::update_robust requires Algorithm integration");
}

LinearGaussianPolicy TrajOptLQR::backward_pass(
    const LinearDynamics& dynamics,
    const Vector& cc,
    const Matrix& cv,
    const Tensor3d& Cm,
    const LinearGaussianPolicy& prev_traj,
    double eta) {

    const int T = static_cast<int>(dynamics.Fm.size());
    const int dX = static_cast<int>(dynamics.fv.cols());
    const int dU = static_cast<int>(dynamics.Fm[0].cols()) - dX;

    // Initialize new policy
    LinearGaussianPolicy new_traj(T, dX, dU);

    // Value function at final timestep
    auto st = [](int i) { return static_cast<std::size_t>(i); };
    Matrix Vxx = Cm[st(T-1)].topLeftCorner(dX, dX);
    Vector Vx = cv.row(T-1).head(dX).transpose();

    // Backward recursion
    for (int t = T - 1; t >= 0; --t) {
        // Extract cost terms
        Matrix Qxx = Cm[st(t)].topLeftCorner(dX, dX);
        Matrix Quu = Cm[st(t)].bottomRightCorner(dU, dU);
        Matrix Qux = Cm[st(t)].bottomLeftCorner(dU, dX);
        Vector Qx = cv.row(t).head(dX).transpose();
        Vector Qu = cv.row(t).tail(dU).transpose();

        // Extract dynamics
        Matrix Fx = dynamics.Fm[st(t)].leftCols(dX);
        Matrix Fu = dynamics.Fm[st(t)].rightCols(dU);

        // Q-function computation
        // Q_xx = C_xx + F_x^T V_xx F_x
        // Q_ux = C_ux + F_u^T V_xx F_x
        // Q_uu = C_uu + F_u^T V_xx F_u
        // Q_x = c_x + F_x^T V_x
        // Q_u = c_u + F_u^T V_x

        if (t < T - 1) {
            Qxx += Fx.transpose() * Vxx * Fx;
            Qux += Fu.transpose() * Vxx * Fx;
            Quu += Fu.transpose() * Vxx * Fu;
            Qx += Fx.transpose() * Vx;
            Qu += Fu.transpose() * Vx;
        }

        // Add KL constraint regularization
        // Q_uu += eta * inv_pol_covar
        // Q_ux += eta * inv_pol_covar * K_prev
        // Q_u += eta * inv_pol_covar * k_prev
        if (eta > 0) {
            const Matrix& inv_cov = prev_traj.inv_pol_covar()[st(t)];
            Quu += eta * inv_cov;
            Qux += eta * inv_cov * prev_traj.K()[st(t)];
            Qu += eta * inv_cov * prev_traj.k().row(t).transpose();
        }

        // Add regularization for numerical stability
        Quu += del0_ * Matrix::Identity(dU, dU);

        // Solve for feedback gains
        // K = -Q_uu^{-1} Q_ux
        // k = -Q_uu^{-1} Q_u
        Eigen::LLT<Matrix> llt(Quu);
        if (llt.info() != Eigen::Success) {
            // Quu not positive definite, add more regularization
            Matrix Quu_reg = Quu + del0_ * 10 * Matrix::Identity(dU, dU);
            llt.compute(Quu_reg);
        }

        Matrix K = -llt.solve(Qux);
        Vector k = -llt.solve(Qu);

        // Store in new policy
        new_traj.K()[st(t)] = K;
        new_traj.k().row(t) = k.transpose();

        // Policy covariance is inverse of Q_uu (approximately)
        new_traj.inv_pol_covar()[st(t)] = Quu;
        new_traj.pol_covar()[st(t)] = llt.solve(Matrix::Identity(dU, dU));
        new_traj.chol_pol_covar()[st(t)] = new_traj.pol_covar()[st(t)].llt().matrixL();

        // Update value function for next iteration
        // V_xx = Q_xx + K^T Q_uu K + K^T Q_ux + Q_ux^T K
        // V_x = Q_x + K^T Q_uu k + K^T Q_u + Q_ux^T k
        Vxx = Qxx + K.transpose() * Quu * K +
              K.transpose() * Qux + Qux.transpose() * K;
        Vx = Qx + K.transpose() * Quu * k +
             K.transpose() * Qu + Qux.transpose() * k;

        // Symmetrize for numerical stability
        Vxx = 0.5 * (Vxx + Vxx.transpose());
    }

    return new_traj;
}

double TrajOptLQR::compute_expected_cost(
    const LinearGaussianPolicy& policy,
    const LinearDynamics& dynamics,
    const Vector& x0mu,
    const Matrix& x0sigma,
    const Vector& cc,
    const Matrix& cv,
    const Tensor3d& Cm) {

    const int T = policy.T();
    const int dX = policy.dX();
    const int dU = policy.dU();

    // Forward simulate expected trajectory
    Vector xmu = x0mu;
    Matrix xsigma = x0sigma;

    double total_cost = 0.0;

    auto st = [](int i) { return static_cast<std::size_t>(i); };

    for (int t = 0; t < T; ++t) {
        // Mean action
        Vector umu = policy.K()[st(t)] * xmu + policy.k().row(t).transpose();

        // Concatenate state-action
        Vector xu(dX + dU);
        xu.head(dX) = xmu;
        xu.tail(dU) = umu;

        // Expected cost at this timestep
        // E[l] = cc + cv^T [x;u] + 0.5 * [x;u]^T Cm [x;u] + 0.5 * tr(Cm * cov)
        total_cost += cc(t);
        total_cost += cv.row(t) * xu;
        total_cost += 0.5 * xu.transpose() * Cm[st(t)] * xu;

        // Add variance term (trace)
        Matrix xu_cov = Matrix::Zero(dX + dU, dX + dU);
        xu_cov.topLeftCorner(dX, dX) = xsigma;
        xu_cov.bottomRightCorner(dU, dU) = policy.pol_covar()[st(t)];
        xu_cov.topRightCorner(dX, dU) = xsigma * policy.K()[st(t)].transpose();
        xu_cov.bottomLeftCorner(dU, dX) = policy.K()[st(t)] * xsigma;

        total_cost += 0.5 * (Cm[st(t)] * xu_cov).trace();

        // Propagate state distribution
        if (t < T - 1) {
            Matrix Fx = dynamics.Fm[st(t)].leftCols(dX);
            Matrix Fu = dynamics.Fm[st(t)].rightCols(dU);

            // x_{t+1} = Fx * x + Fu * u + fv
            xmu = dynamics.predict(xmu, umu, t);

            // Covariance propagation
            Matrix A = Fx + Fu * policy.K()[st(t)];
            xsigma = A * xsigma * A.transpose() +
                     Fu * policy.pol_covar()[st(t)] * Fu.transpose() +
                     dynamics.dyn_covar[st(t)];
        }
    }

    return total_cost;
}

double TrajOptLQR::compute_kl_divergence(
    const LinearGaussianPolicy& p,
    const LinearGaussianPolicy& q,
    const LinearDynamics& dynamics,
    const Vector& x0mu,
    const Matrix& x0sigma) {

    const int T = p.T();
    const int dX = p.dX();
    const int dU = p.dU();

    // Forward simulate to get state distribution under q
    Vector xmu = x0mu;
    Matrix xsigma = x0sigma;

    double kl = 0.0;
    auto st = [](int i) { return static_cast<std::size_t>(i); };

    for (int t = 0; t < T; ++t) {
        // KL divergence between two Gaussians with same covariance
        // at each timestep (simplified)
        Vector k_diff = p.k().row(t).transpose() - q.k().row(t).transpose();
        Matrix K_diff = p.K()[st(t)] - q.K()[st(t)];

        // Mean difference in action space
        Vector u_diff = k_diff + K_diff * xmu;

        // KL contribution: 0.5 * u_diff^T inv_cov u_diff + trace term
        kl += 0.5 * u_diff.transpose() * q.inv_pol_covar()[st(t)] * u_diff;

        // Log determinant ratio and trace terms
        double log_det_p = 0.0, log_det_q = 0.0;
        for (int i = 0; i < dU; ++i) {
            log_det_p += 2 * std::log(p.chol_pol_covar()[st(t)](i, i));
            log_det_q += 2 * std::log(q.chol_pol_covar()[st(t)](i, i));
        }
        kl += 0.5 * (log_det_q - log_det_p);
        kl += 0.5 * (q.inv_pol_covar()[st(t)] * p.pol_covar()[st(t)]).trace();
        kl -= 0.5 * dU;

        // Propagate state
        if (t < T - 1) {
            Vector umu = q.K()[st(t)] * xmu + q.k().row(t).transpose();
            xmu = dynamics.predict(xmu, umu, t);

            Matrix Fx = dynamics.Fm[st(t)].leftCols(dX);
            Matrix Fu = dynamics.Fm[st(t)].rightCols(dU);
            Matrix A = Fx + Fu * q.K()[st(t)];
            xsigma = A * xsigma * A.transpose() +
                     Fu * q.pol_covar()[st(t)] * Fu.transpose() +
                     dynamics.dyn_covar[st(t)];
        }
    }

    return kl;
}

std::unique_ptr<TrajOpt> TrajOptLQR::clone() const {
    return std::make_unique<TrajOptLQR>(hyperparams_);
}

}  // namespace gps
