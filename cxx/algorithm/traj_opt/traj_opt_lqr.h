/**
 * @file traj_opt_lqr.h
 * @brief LQR-based trajectory optimization (iLQG).
 *
 * C++20 port of python/gps/algorithm/traj_opt/traj_opt_lqr.py
 * Implements iterative LQG (iLQG) for trajectory optimization.
 */

#ifndef GPS_CXX_ALGORITHM_TRAJ_OPT_TRAJ_OPT_LQR_H_
#define GPS_CXX_ALGORITHM_TRAJ_OPT_TRAJ_OPT_LQR_H_

#include "traj_opt.h"
#include "../dynamics/dynamics.h"

namespace gps {

/**
 * @brief LQR trajectory optimization using backward pass.
 *
 * Solves the trajectory optimization problem using the
 * iterative Linear Quadratic Regulator (iLQR/iLQG) algorithm.
 */
class TrajOptLQR : public TrajOpt {
public:
    explicit TrajOptLQR(const Hyperparams& hyperparams);
    ~TrajOptLQR() override = default;

    TrajOptLQR(TrajOptLQR&&) = default;
    TrajOptLQR& operator=(TrajOptLQR&&) = default;

    /**
     * @brief Update trajectory using LQR backward pass.
     *
     * @param condition Condition index
     * @param algorithm Algorithm with dynamics and cost info
     * @return TrajOptResult New trajectory distribution
     */
    [[nodiscard]] TrajOptResult update(
        int condition, const Algorithm& algorithm) override;

    /**
     * @brief Update robust trajectory using game-theoretic LQR.
     *
     * @param condition Condition index
     * @param algorithm Algorithm with dynamics and cost info
     * @return TrajOptResultRobust Protagonist and antagonist trajectories
     */
    [[nodiscard]] TrajOptResultRobust update_robust(
        int condition, const Algorithm& algorithm) override;

    [[nodiscard]] std::unique_ptr<TrajOpt> clone() const override;

private:
    /**
     * @brief Perform LQR backward pass.
     *
     * @param dynamics Linearized dynamics
     * @param cc Cost constant term (T,)
     * @param cv Cost linear term (T, dX+dU)
     * @param Cm Cost quadratic term T x (dX+dU, dX+dU)
     * @param prev_traj Previous trajectory for KL constraint
     * @param eta Dual variable for KL constraint
     * @return LinearGaussianPolicy New policy from backward pass
     */
    [[nodiscard]] LinearGaussianPolicy backward_pass(
        const LinearDynamics& dynamics,
        const Vector& cc,
        const Matrix& cv,
        const Tensor3d& Cm,
        const LinearGaussianPolicy& prev_traj,
        double eta);

    /**
     * @brief Compute expected cost given policy and dynamics.
     *
     * @param policy Policy to evaluate
     * @param dynamics System dynamics
     * @param x0mu Initial state mean
     * @param x0sigma Initial state covariance
     * @param cc Cost constant term
     * @param cv Cost linear term
     * @param Cm Cost quadratic term
     * @return double Expected cost
     */
    [[nodiscard]] double compute_expected_cost(
        const LinearGaussianPolicy& policy,
        const LinearDynamics& dynamics,
        const Vector& x0mu,
        const Matrix& x0sigma,
        const Vector& cc,
        const Matrix& cv,
        const Tensor3d& Cm);

    /**
     * @brief Compute KL divergence between two policies.
     *
     * @param p New policy
     * @param q Previous policy
     * @param dynamics System dynamics
     * @param x0mu Initial state mean
     * @param x0sigma Initial state covariance
     * @return double KL divergence D_KL(p || q)
     */
    [[nodiscard]] double compute_kl_divergence(
        const LinearGaussianPolicy& p,
        const LinearGaussianPolicy& q,
        const LinearDynamics& dynamics,
        const Vector& x0mu,
        const Matrix& x0sigma);

    // Parameters
    double del0_;           // Initial regularization
    double min_eta_;        // Minimum dual variable
    double max_eta_;        // Maximum dual variable
    int max_iters_;         // Maximum backward pass iterations
    double kl_step_;        // Target KL step size
};

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_TRAJ_OPT_TRAJ_OPT_LQR_H_
