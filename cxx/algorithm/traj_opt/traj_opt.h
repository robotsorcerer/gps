/**
 * @file traj_opt.h
 * @brief Abstract trajectory optimization base class.
 *
 * C++20 port of python/gps/algorithm/traj_opt/traj_opt.py
 */

#ifndef GPS_CXX_ALGORITHM_TRAJ_OPT_TRAJ_OPT_H_
#define GPS_CXX_ALGORITHM_TRAJ_OPT_TRAJ_OPT_H_

#include <memory>
#include <tuple>

#include "../policy/lin_gauss_policy.h"
#include "../../utility/types.h"

namespace gps {

// Forward declaration
class Algorithm;

/**
 * @brief Result of trajectory optimization step.
 */
struct TrajOptResult {
    LinearGaussianPolicy traj_distr;  // Updated trajectory distribution
    double eta;                       // Dual variable (KL constraint)
    double expected_cost;             // Expected cost after update
    double kl_divergence;             // KL divergence from previous
};

/**
 * @brief Result of robust (iDG) trajectory optimization.
 */
struct TrajOptResultRobust {
    LinearGaussianPolicy traj_distr;      // Protagonist trajectory
    LinearGaussianPolicy traj_distr_adv;  // Antagonist trajectory
    double eta;                           // Dual variable
    double expected_cost;
};

/**
 * @brief Abstract base class for trajectory optimization.
 *
 * Computes locally optimal linear Gaussian policies given
 * cost approximations and dynamics.
 */
class TrajOpt {
public:
    explicit TrajOpt(const Hyperparams& hyperparams);
    virtual ~TrajOpt() = default;

    // Non-copyable
    TrajOpt(const TrajOpt&) = delete;
    TrajOpt& operator=(const TrajOpt&) = delete;
    TrajOpt(TrajOpt&&) = default;
    TrajOpt& operator=(TrajOpt&&) = default;

    /**
     * @brief Update trajectory distribution for a condition.
     *
     * @param condition Condition index
     * @param algorithm Algorithm containing state and cost info
     * @return TrajOptResult Updated policy and dual variables
     */
    [[nodiscard]] virtual TrajOptResult update(
        int condition, const Algorithm& algorithm) = 0;

    /**
     * @brief Update robust trajectory for iDG algorithms.
     *
     * @param condition Condition index
     * @param algorithm Algorithm containing state and cost info
     * @return TrajOptResultRobust Updated protagonist and antagonist policies
     */
    [[nodiscard]] virtual TrajOptResultRobust update_robust(
        int condition, const Algorithm& algorithm);

    /**
     * @brief Clone the trajectory optimizer.
     */
    [[nodiscard]] virtual std::unique_ptr<TrajOpt> clone() const = 0;

protected:
    Hyperparams hyperparams_;
};

/**
 * @brief Shared pointer type for trajectory optimizer.
 */
using TrajOptPtr = std::shared_ptr<TrajOpt>;

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_TRAJ_OPT_TRAJ_OPT_H_
