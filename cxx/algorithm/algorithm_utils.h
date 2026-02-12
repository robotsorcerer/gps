/**
 * @file algorithm_utils.h
 * @brief Algorithm utility classes and structures.
 *
 * C++20 port of python/gps/algorithm/algorithm_utils.py
 */

#ifndef GPS_CXX_ALGORITHM_ALGORITHM_UTILS_H_
#define GPS_CXX_ALGORITHM_ALGORITHM_UTILS_H_

#include <memory>
#include <optional>

#include "dynamics/dynamics.h"
#include "policy/lin_gauss_policy.h"
#include "../sample/sample_list.h"
#include "../utility/types.h"

namespace gps {

/**
 * @brief Trajectory information for optimization.
 *
 * Contains fitted dynamics, cost approximations, and state distribution.
 */
struct TrajectoryInfo {
    // Fitted dynamics model
    std::unique_ptr<Dynamics> dynamics;

    // Cost approximation (quadratic expansion)
    Vector cc;     // Constant term: (T,)
    Matrix cv;     // Linear term: (T, dX+dU) or (T, dX+dU+dV)
    Tensor3d Cm;   // Quadratic term: T x (dX+dU, dX+dU) or similar

    // Initial state distribution
    Vector x0mu;    // Mean: (dX,)
    Matrix x0sigma; // Covariance: (dX, dX)

    // For iDG: target distance metric
    double target_distance = 0.0;

    TrajectoryInfo() = default;
    ~TrajectoryInfo() = default;

    TrajectoryInfo(TrajectoryInfo&&) = default;
    TrajectoryInfo& operator=(TrajectoryInfo&&) = default;

    // Deep copy (for prev assignment)
    TrajectoryInfo clone() const;
};

/**
 * @brief Per-condition data for each algorithm iteration.
 */
struct IterationData {
    // Current sample list
    SampleList sample_list;

    // Adversary sample list (for iDG)
    SampleList sample_list_adv;

    // Trajectory information (dynamics, costs)
    TrajectoryInfo traj_info;

    // Current trajectory distributions
    LinearGaussianPolicy traj_distr;
    LinearGaussianPolicy traj_distr_adv;
    LinearGaussianPolicy traj_distr_robust;

    // Updated trajectory (after optimization)
    std::optional<LinearGaussianPolicy> new_traj_distr;
    std::optional<LinearGaussianPolicy> new_traj_distr_adv;

    // Cost values
    Matrix cs;  // True costs: (N, T)

    // Dual variable for KL constraint
    double eta = 1.0;

    // Step size multiplier
    double step_mult = 1.0;

    IterationData() = default;
    ~IterationData() = default;

    IterationData(IterationData&&) = default;
    IterationData& operator=(IterationData&&) = default;

    // Deep copy
    IterationData clone() const;
};

/**
 * @brief Compute Gaussian KL divergence.
 *
 * D_KL(p || q) for multivariate Gaussians.
 *
 * @param mu_p Mean of p
 * @param sigma_p Covariance of p
 * @param mu_q Mean of q
 * @param sigma_q Covariance of q
 * @return double KL divergence
 */
[[nodiscard]] double gaussian_kl(
    const Vector& mu_p, const Matrix& sigma_p,
    const Vector& mu_q, const Matrix& sigma_q);

/**
 * @brief Compute trajectory KL divergence.
 *
 * @param policy_p Policy p
 * @param policy_q Policy q
 * @param dynamics System dynamics
 * @param x0mu Initial state mean
 * @param x0sigma Initial state covariance
 * @return double KL divergence between trajectory distributions
 */
[[nodiscard]] double trajectory_kl(
    const LinearGaussianPolicy& policy_p,
    const LinearGaussianPolicy& policy_q,
    const LinearDynamics& dynamics,
    const Vector& x0mu,
    const Matrix& x0sigma);

/**
 * @brief Extract condition-specific configuration.
 *
 * @param hyperparams Full hyperparameters
 * @param condition Condition index to extract
 * @return Hyperparams Condition-specific configuration
 */
[[nodiscard]] Hyperparams extract_condition(
    const Hyperparams& hyperparams, int condition);

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_ALGORITHM_UTILS_H_
