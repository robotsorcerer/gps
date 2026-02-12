/**
 * @file algorithm.h
 * @brief Abstract Algorithm base class.
 *
 * C++20 port of python/gps/algorithm/algorithm.py
 * Defines the interface for GPS optimization algorithms.
 */

#ifndef GPS_CXX_ALGORITHM_ALGORITHM_H_
#define GPS_CXX_ALGORITHM_ALGORITHM_H_

#include <memory>
#include <vector>

#include "algorithm_utils.h"
#include "cost/cost.h"
#include "dynamics/dynamics.h"
#include "policy/lin_gauss_policy.h"
#include "traj_opt/traj_opt.h"
#include "../sample/sample_list.h"
#include "../utility/types.h"

namespace gps {

/**
 * @brief Abstract base class for GPS algorithms.
 *
 * Provides the framework for guided policy search variants including:
 * - Standard GPS
 * - MDGPS (Mirror Descent GPS)
 * - BADMM
 * - iDG (Inverse Differential Games)
 */
class Algorithm {
public:
    /**
     * @brief Construct algorithm from hyperparameters.
     *
     * @param hyperparams Configuration containing:
     *   - T: Time horizon
     *   - dX: State dimension
     *   - dU: Action dimension
     *   - dV: Adversary dimension (for iDG)
     *   - conditions/M: Number of conditions
     *   - cost: Cost function configuration
     *   - dynamics: Dynamics model configuration
     *   - traj_opt: Trajectory optimizer configuration
     *   - init_traj_distr: Initial trajectory distribution
     */
    explicit Algorithm(const Hyperparams& hyperparams);

    virtual ~Algorithm() = default;

    // Non-copyable
    Algorithm(const Algorithm&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;
    Algorithm(Algorithm&&) = default;
    Algorithm& operator=(Algorithm&&) = default;

    /**
     * @brief Run a single iteration of the algorithm.
     *
     * @param sample_lists Samples for each condition
     */
    virtual void iteration(const std::vector<SampleList>& sample_lists) = 0;

    /**
     * @brief Run iteration for closed-loop control.
     *
     * @param sample_lists_prot Protagonist samples
     * @param sample_lists Current samples
     */
    virtual void iteration_cl(
        const std::vector<SampleList>& sample_lists_prot,
        const std::vector<SampleList>& sample_lists);

    /**
     * @brief Run iteration for iDG (game-theoretic) algorithm.
     *
     * @param sample_lists_prot Protagonist samples
     * @param sample_lists Adversary samples
     */
    virtual void iteration_idg(
        const std::vector<SampleList>& sample_lists_prot,
        const std::vector<SampleList>& sample_lists);

    /**
     * @brief Get current trajectory distribution for a condition.
     *
     * @param condition Condition index
     * @return const LinearGaussianPolicy& Current policy
     */
    [[nodiscard]] const LinearGaussianPolicy& traj_distr(int condition) const;

    /**
     * @brief Get adversary trajectory distribution for a condition.
     *
     * @param condition Condition index
     * @return const LinearGaussianPolicy& Adversary policy
     */
    [[nodiscard]] const LinearGaussianPolicy& traj_distr_adv(int condition) const;

    /**
     * @brief Get iteration count.
     */
    [[nodiscard]] int iteration_count() const noexcept {
        return iteration_count_;
    }

    // Dimension accessors
    [[nodiscard]] int T() const noexcept { return T_; }
    [[nodiscard]] int dX() const noexcept { return dX_; }
    [[nodiscard]] int dU() const noexcept { return dU_; }
    [[nodiscard]] int dV() const noexcept { return dV_; }
    [[nodiscard]] int M() const noexcept { return M_; }

    /**
     * @brief Get current iteration data for a condition.
     */
    [[nodiscard]] const IterationData& cur(int condition) const {
        return cur_.at(static_cast<std::size_t>(condition));
    }

    /**
     * @brief Get mutable current iteration data.
     */
    IterationData& cur(int condition) {
        return cur_.at(static_cast<std::size_t>(condition));
    }

    /**
     * @brief Get cost function for a condition.
     */
    [[nodiscard]] const Cost& cost(int condition) const {
        return *costs_.at(static_cast<std::size_t>(condition));
    }

protected:
    /**
     * @brief Update dynamics models from samples.
     */
    void update_dynamics();

    /**
     * @brief Update dynamics for iDG (with adversary).
     */
    void update_dynamics_idg();

    /**
     * @brief Update trajectory distributions.
     */
    void update_trajectories();

    /**
     * @brief Update trajectory distributions (robust version).
     */
    void update_trajectories_robust();

    /**
     * @brief Evaluate cost for a condition.
     *
     * @param condition Condition index
     */
    void eval_cost(int condition);

    /**
     * @brief Evaluate cost for iDG.
     *
     * @param condition Condition index
     */
    void eval_cost_idg(int condition);

    /**
     * @brief Advance iteration variables.
     */
    void advance_iteration_variables();

    /**
     * @brief Adjust step size multiplier based on improvement.
     */
    void set_new_mult(double predicted_impr, double actual_impr, int m);

    /**
     * @brief Measure trajectory entropy.
     */
    [[nodiscard]] double measure_ent(int m) const;

    // Configuration
    Hyperparams hyperparams_;

    // Dimensions
    int T_;   // Time horizon
    int dX_;  // State dimension
    int dU_;  // Action dimension
    int dV_;  // Adversary dimension
    int dO_;  // Observation dimension
    int M_;   // Number of conditions

    // Condition indices
    std::vector<int> cond_idx_;

    // Iteration data for each condition
    std::vector<IterationData> cur_;
    std::vector<IterationData> prev_;

    // New trajectory distributions (computed during iteration)
    std::vector<LinearGaussianPolicy> new_traj_distr_;
    std::vector<LinearGaussianPolicy> new_traj_distr_adv_;

    // Cost functions
    std::vector<CostPtr> costs_;

    // Trajectory optimizer
    TrajOptPtr traj_opt_;

    // Iteration counter
    int iteration_count_ = 0;

    // Step size parameters
    double base_kl_step_;
};

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_ALGORITHM_H_
