/**
 * @file lin_gauss_policy.cpp
 * @brief Implementation of Linear Gaussian policy classes.
 */

#include "lin_gauss_policy.h"

#include <cmath>
#include <stdexcept>

namespace gps {

// ============================================================================
// LinearGaussianPolicy Implementation
// ============================================================================

LinearGaussianPolicy::LinearGaussianPolicy(
    Tensor3d K,
    Matrix k,
    Tensor3d pol_covar,
    Tensor3d chol_pol_covar,
    Tensor3d inv_pol_covar)
    : T_(static_cast<int>(K.size())),
      dX_(K.empty() ? 0 : static_cast<int>(K[0].cols())),
      dU_(K.empty() ? 0 : static_cast<int>(K[0].rows())),
      K_(std::move(K)),
      k_(std::move(k)),
      pol_covar_(std::move(pol_covar)),
      chol_pol_covar_(std::move(chol_pol_covar)),
      inv_pol_covar_(std::move(inv_pol_covar)) {
    // Validate dimensions
    if (k_.rows() != T_ || k_.cols() != dU_) {
        throw std::invalid_argument(
            "k dimensions mismatch: expected (" + std::to_string(T_) + ", " +
            std::to_string(dU_) + "), got (" + std::to_string(k_.rows()) +
            ", " + std::to_string(k_.cols()) + ")");
    }
}

LinearGaussianPolicy::LinearGaussianPolicy(const Hyperparams& hyperparams)
    : LinearGaussianPolicy(
          get_hyperparam<int>(hyperparams, "T"),
          get_hyperparam<int>(hyperparams, "dX"),
          get_hyperparam<int>(hyperparams, "dU"),
          get_hyperparam_or<double>(hyperparams, "init_var", 1.0)) {
}

LinearGaussianPolicy::LinearGaussianPolicy(int T, int dX, int dU, double init_var)
    : T_(T), dX_(dX), dU_(dU) {
    if (T <= 0 || dX <= 0 || dU <= 0) {
        throw std::invalid_argument(
            "Policy dimensions must be positive");
    }

    // Initialize K to zeros (no feedback initially)
    K_.resize(T);
    for (int t = 0; t < T; ++t) {
        K_[t] = Matrix::Zero(dU, dX);
    }

    // Initialize k to zeros (no feedforward initially)
    k_ = Matrix::Zero(T, dU);

    // Initialize covariances to scaled identity
    pol_covar_.resize(T);
    chol_pol_covar_.resize(T);
    inv_pol_covar_.resize(T);

    double sqrt_var = std::sqrt(init_var);
    double inv_var = 1.0 / init_var;

    for (int t = 0; t < T; ++t) {
        pol_covar_[t] = Matrix::Identity(dU, dU) * init_var;
        chol_pol_covar_[t] = Matrix::Identity(dU, dU) * sqrt_var;
        inv_pol_covar_[t] = Matrix::Identity(dU, dU) * inv_var;
    }
}

Vector LinearGaussianPolicy::act(
    const Vector* x,
    const Vector* /*obs*/,
    int t,
    const Vector* noise) const {

    if (x == nullptr) {
        throw std::invalid_argument(
            "LinearGaussianPolicy requires state vector x");
    }

    if (t < 0 || t >= T_) {
        throw std::out_of_range(
            "Timestep " + std::to_string(t) + " out of range [0, " +
            std::to_string(T_) + ")");
    }

    if (x->size() != dX_) {
        throw std::invalid_argument(
            "State dimension mismatch: expected " + std::to_string(dX_) +
            ", got " + std::to_string(x->size()));
    }

    // U = K * x + k
    Vector u = K_[t] * (*x) + k_.row(t).transpose();

    // Add scaled noise if provided: u += L^T * noise (matches Python convention)
    if (noise != nullptr && noise->size() == dU_) {
        u += chol_pol_covar_[t].transpose() * (*noise);
    }

    return u;
}

std::unique_ptr<Policy> LinearGaussianPolicy::clone() const {
    return std::make_unique<LinearGaussianPolicy>(*this);
}

Matrix LinearGaussianPolicy::fold_k(const Matrix& noise) const {
    if (noise.rows() != T_ || noise.cols() != dU_) {
        throw std::invalid_argument(
            "Noise dimensions mismatch: expected (" + std::to_string(T_) +
            ", " + std::to_string(dU_) + "), got (" +
            std::to_string(noise.rows()) + ", " +
            std::to_string(noise.cols()) + ")");
    }

    Matrix result = k_;
    for (int t = 0; t < T_; ++t) {
        // k_folded = k + L^T * noise (matches Python convention)
        result.row(t) += (chol_pol_covar_[t].transpose() * noise.row(t).transpose()).transpose();
    }
    return result;
}

Vector LinearGaussianPolicy::mean_action(const Vector& x, int t) const {
    if (t < 0 || t >= T_) {
        throw std::out_of_range("Timestep out of range");
    }
    return K_[t] * x + k_.row(t).transpose();
}

double LinearGaussianPolicy::log_prob(
    const Vector& x, const Vector& u, int t) const {
    if (t < 0 || t >= T_) {
        throw std::out_of_range("Timestep out of range");
    }

    Vector mean = mean_action(x, t);
    Vector diff = u - mean;

    // Log probability of multivariate Gaussian
    // log p(u|x) = -0.5 * (u - mu)^T * Sigma^-1 * (u - mu)
    //              -0.5 * log|Sigma| - (dU/2) * log(2*pi)
    double quad_form = diff.transpose() * inv_pol_covar_[t] * diff;

    // Log determinant from Cholesky: log|Sigma| = 2 * sum(log(diag(L)))
    double log_det = 0.0;
    for (int i = 0; i < dU_; ++i) {
        log_det += 2.0 * std::log(chol_pol_covar_[t](i, i));
    }

    constexpr double LOG_2PI = 1.8378770664093453;  // log(2*pi)
    return -0.5 * quad_form - 0.5 * log_det - 0.5 * dU_ * LOG_2PI;
}

// ============================================================================
// LinearGaussianPolicyRobust Implementation
// ============================================================================

LinearGaussianPolicyRobust::LinearGaussianPolicyRobust(
    LinearGaussianPolicy protagonist,
    LinearGaussianPolicy antagonist,
    PolicyMode mode)
    : protagonist_(std::move(protagonist)),
      antagonist_(std::move(antagonist)),
      mode_(mode) {

    if (protagonist_.T() != antagonist_.T()) {
        throw std::invalid_argument(
            "Protagonist and antagonist must have same time horizon");
    }
    if (protagonist_.dX() != antagonist_.dX()) {
        throw std::invalid_argument(
            "Protagonist and antagonist must have same state dimension");
    }
}

LinearGaussianPolicyRobust::LinearGaussianPolicyRobust(
    const Hyperparams& hyperparams)
    : protagonist_(
          get_hyperparam<int>(hyperparams, "T"),
          get_hyperparam<int>(hyperparams, "dX"),
          get_hyperparam<int>(hyperparams, "dU"),
          get_hyperparam_or<double>(hyperparams, "init_var", 1.0)),
      antagonist_(
          get_hyperparam<int>(hyperparams, "T"),
          get_hyperparam<int>(hyperparams, "dX"),
          get_hyperparam<int>(hyperparams, "dV"),
          get_hyperparam_or<double>(hyperparams, "init_var", 1.0)),
      mode_(PolicyMode::PROTAGONIST) {
}

Vector LinearGaussianPolicyRobust::act(
    const Vector* x,
    const Vector* obs,
    int t,
    const Vector* noise) const {

    switch (mode_) {
        case PolicyMode::PROTAGONIST:
            return protagonist_.act(x, obs, t, noise);

        case PolicyMode::ANTAGONIST:
            return antagonist_.act(x, obs, t, noise);

        case PolicyMode::ROBUST:
            // In robust mode, return protagonist action by default
            // (caller should query both policies separately for full behavior)
            return protagonist_.act(x, obs, t, noise);
    }

    // Should not reach here
    return protagonist_.act(x, obs, t, noise);
}

int LinearGaussianPolicyRobust::dU() const noexcept {
    switch (mode_) {
        case PolicyMode::PROTAGONIST:
            return protagonist_.dU();
        case PolicyMode::ANTAGONIST:
            return antagonist_.dU();
        case PolicyMode::ROBUST:
            return protagonist_.dU();  // Return protagonist dimension
    }
    return protagonist_.dU();
}

std::unique_ptr<Policy> LinearGaussianPolicyRobust::clone() const {
    return std::make_unique<LinearGaussianPolicyRobust>(*this);
}

}  // namespace gps
