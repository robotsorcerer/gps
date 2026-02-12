/**
 * @file cost_sum.h
 * @brief Weighted sum of cost functions.
 *
 * C++20 port of python/gps/algorithm/cost/cost_sum.py
 */

#ifndef GPS_CXX_ALGORITHM_COST_COST_SUM_H_
#define GPS_CXX_ALGORITHM_COST_COST_SUM_H_

#include <memory>
#include <vector>

#include "cost.h"
#include "../../utility/types.h"

namespace gps {

/**
 * @brief Weighted sum of multiple cost functions.
 *
 * Combines arbitrary cost functions with scalar weights.
 * Supports game-theoretic modes for iDG algorithms.
 */
class CostSum : public Cost {
public:
    /**
     * @brief Construct an empty cost sum.
     */
    CostSum() = default;

    /**
     * @brief Construct with initial costs and weights.
     *
     * @param costs Vector of cost function pointers
     * @param weights Vector of scalar weights (same length as costs)
     * @param mode Cost mode for game-theoretic algorithms
     */
    CostSum(
        std::vector<CostPtr> costs,
        std::vector<double> weights,
        CostMode mode = CostMode::PROTAGONIST);

    /**
     * @brief Construct from hyperparameters.
     *
     * @param hyperparams Configuration with costs and weights
     */
    explicit CostSum(const Hyperparams& hyperparams);

    ~CostSum() override = default;

    // Non-copyable due to shared_ptr ownership
    CostSum(const CostSum&) = delete;
    CostSum& operator=(const CostSum&) = delete;
    CostSum(CostSum&&) = default;
    CostSum& operator=(CostSum&&) = default;

    /**
     * @brief Evaluate weighted sum of costs.
     *
     * @param sample Trajectory sample
     * @return CostResult Combined cost and derivatives
     */
    [[nodiscard]] CostResult eval(const Sample& sample) const override;

    [[nodiscard]] CostMode mode() const noexcept override { return mode_; }

    void set_mode(CostMode mode) override {
        mode_ = mode;
        // Propagate mode to all child costs
        for (auto& cost : costs_) {
            cost->set_mode(mode);
        }
    }

    [[nodiscard]] std::unique_ptr<Cost> clone() const override;

    /**
     * @brief Add a cost function with weight.
     *
     * @param cost Cost function to add
     * @param weight Scalar weight
     */
    void add_cost(CostPtr cost, double weight);

    /**
     * @brief Get number of cost terms.
     */
    [[nodiscard]] std::size_t num_costs() const noexcept {
        return costs_.size();
    }

    /**
     * @brief Access individual cost functions.
     */
    [[nodiscard]] const CostPtr& cost(std::size_t idx) const {
        return costs_.at(idx);
    }

    /**
     * @brief Access individual weights.
     */
    [[nodiscard]] double weight(std::size_t idx) const {
        return weights_.at(idx);
    }

private:
    std::vector<CostPtr> costs_;
    std::vector<double> weights_;
    CostMode mode_ = CostMode::PROTAGONIST;
};

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_COST_COST_SUM_H_
