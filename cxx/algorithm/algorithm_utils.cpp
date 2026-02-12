/**
 * @file algorithm_utils.cpp
 * @brief Implementation of algorithm utility functions.
 */

#include "algorithm_utils.h"

#include <cmath>
#include <stdexcept>

namespace gps {

TrajectoryInfo TrajectoryInfo::clone() const {
    TrajectoryInfo copy;

    if (dynamics) {
        copy.dynamics = dynamics->clone();
    }

    copy.cc = cc;
    copy.cv = cv;
    copy.Cm = Cm;
    copy.x0mu = x0mu;
    copy.x0sigma = x0sigma;
    copy.target_distance = target_distance;

    return copy;
}

IterationData IterationData::clone() const {
    IterationData copy;

    copy.sample_list = sample_list;
    copy.sample_list_adv = sample_list_adv;
    copy.traj_info = traj_info.clone();
    copy.traj_distr = traj_distr;
    copy.traj_distr_adv = traj_distr_adv;
    copy.traj_distr_robust = traj_distr_robust;
    copy.new_traj_distr = new_traj_distr;
    copy.new_traj_distr_adv = new_traj_distr_adv;
    copy.cs = cs;
    copy.eta = eta;
    copy.step_mult = step_mult;

    return copy;
}

double gaussian_kl(
    const Vector& mu_p, const Matrix& sigma_p,
    const Vector& mu_q, const Matrix& sigma_q) {

    const int d = static_cast<int>(mu_p.size());

    if (mu_q.size() != d || sigma_p.rows() != d || sigma_q.rows() != d) {
        throw std::invalid_argument(
            "Dimension mismatch in gaussian_kl");
    }

    // KL(p || q) = 0.5 * (log|sigma_q|/|sigma_p| + tr(sigma_q^-1 sigma_p)
    //              + (mu_q - mu_p)^T sigma_q^-1 (mu_q - mu_p) - d)

    // Compute log determinants via Cholesky
    Eigen::LLT<Matrix> llt_p(sigma_p);
    Eigen::LLT<Matrix> llt_q(sigma_q);

    if (llt_p.info() != Eigen::Success || llt_q.info() != Eigen::Success) {
        throw std::runtime_error(
            "Cholesky decomposition failed in gaussian_kl");
    }

    double log_det_p = 0.0, log_det_q = 0.0;
    for (int i = 0; i < d; ++i) {
        log_det_p += 2 * std::log(llt_p.matrixL()(i, i));
        log_det_q += 2 * std::log(llt_q.matrixL()(i, i));
    }

    // Inverse of sigma_q
    Matrix sigma_q_inv = llt_q.solve(Matrix::Identity(d, d));

    // Trace term
    double trace_term = (sigma_q_inv * sigma_p).trace();

    // Mean difference term
    Vector mu_diff = mu_q - mu_p;
    double mean_term = mu_diff.transpose() * sigma_q_inv * mu_diff;

    return 0.5 * (log_det_q - log_det_p + trace_term + mean_term - d);
}

double trajectory_kl(
    const LinearGaussianPolicy& policy_p,
    const LinearGaussianPolicy& policy_q,
    const LinearDynamics& dynamics,
    const Vector& x0mu,
    const Matrix& x0sigma) {

    const int T = policy_p.T();
    const int dX = policy_p.dX();
    const int dU = policy_p.dU();

    double total_kl = 0.0;

    // Forward simulate state distribution under q
    Vector xmu = x0mu;
    Matrix xsigma = x0sigma;

    for (int t = 0; t < T; ++t) {
        // Mean action under each policy
        Vector umu_p = policy_p.K()[t] * xmu + policy_p.k().row(t).transpose();
        Vector umu_q = policy_q.K()[t] * xmu + policy_q.k().row(t).transpose();

        // Action covariance
        Matrix sigma_u_p = policy_p.pol_covar()[t];
        Matrix sigma_u_q = policy_q.pol_covar()[t];

        // KL contribution at this timestep
        total_kl += gaussian_kl(umu_p, sigma_u_p, umu_q, sigma_u_q);

        // Propagate state under q
        if (t < T - 1) {
            Matrix Fx = dynamics.Fm[t].leftCols(dX);
            Matrix Fu = dynamics.Fm[t].rightCols(dU);

            xmu = dynamics.predict(xmu, umu_q, t);

            Matrix A = Fx + Fu * policy_q.K()[t];
            xsigma = A * xsigma * A.transpose() +
                     Fu * sigma_u_q * Fu.transpose() +
                     dynamics.dyn_covar[t];
        }
    }

    return total_kl;
}

Hyperparams extract_condition(const Hyperparams& hyperparams, int condition) {
    Hyperparams result = hyperparams;

    // Extract condition-specific values for vector-valued parameters
    // This is a simplified version - full implementation would handle
    // all array-valued parameters

    // For x0 (initial state per condition)
    if (hyperparams.contains("x0")) {
        try {
            const auto& x0_list = std::get<std::vector<double>>(
                hyperparams.at("x0"));
            // Assuming x0 is flattened for all conditions
            // Full implementation would properly index
            result["x0"] = x0_list;
        } catch (const std::bad_variant_access&) {
            // x0 might already be per-condition
        }
    }

    return result;
}

}  // namespace gps
