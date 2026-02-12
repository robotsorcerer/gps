/**
 * @file cost.h
 * @brief Abstract Cost base class.
 *
 * C++20 port of python/gps/algorithm/cost/cost.py
 * Defines the interface for cost functions used in trajectory optimization.
 */

#ifndef GPS_CXX_ALGORITHM_COST_COST_H_
#define GPS_CXX_ALGORITHM_COST_COST_H_

#include <memory>
#include <tuple>

#include "../../sample/sample.h"
#include "../../utility/types.h"

namespace gps {

/**
 * @brief Result structure for cost evaluation.
 *
 * Contains the cost value and its derivatives with respect to
 * state and action.
 */
struct CostResult {
    Vector l;      // Cost at each timestep: (T,)
    Matrix lx;     // d cost / d state: (T, dX)
    Matrix lu;     // d cost / d action: (T, dU)
    Tensor3d lxx;  // d^2 cost / d state^2: T x (dX, dX)
    Tensor3d luu;  // d^2 cost / d action^2: T x (dU, dU)
    Tensor3d lux;  // d^2 cost / d action d state: T x (dU, dX)

    // For iDG adversary actions
    Matrix lv;     // d cost / d adversary action: (T, dV)
    Tensor3d lvv;  // d^2 cost / d adversary^2: T x (dV, dV)
    Tensor3d lvx;  // d^2 cost / d adversary d state: T x (dV, dX)
    Tensor3d luv;  // d^2 cost / d action d adversary: T x (dU, dV)
};

/**
 * @brief Abstract base class for GPS cost functions.
 *
 * Cost functions define the objective to be minimized during
 * trajectory optimization. They must provide the cost value
 * and its first/second-order derivatives.
 */
class Cost {
public:
    Cost() = default;
    virtual ~Cost() = default;

    // Moveable
    Cost(Cost&&) = default;
    Cost& operator=(Cost&&) = default;

    /**
     * @brief Evaluate cost on a sample trajectory.
     *
     * Computes the cost and its derivatives at each timestep
     * for the given sample.
     *
     * @param sample Trajectory sample to evaluate
     * @return CostResult Cost values and derivatives
     */
    [[nodiscard]] virtual CostResult eval(const Sample& sample) const = 0;

    /**
     * @brief Get cost mode for game-theoretic algorithms.
     *
     * @return CostMode Current mode (protagonist/antagonist/robust)
     */
    [[nodiscard]] virtual CostMode mode() const noexcept {
        return CostMode::PROTAGONIST;
    }

    /**
     * @brief Set cost mode for game-theoretic algorithms.
     *
     * @param mode New cost mode
     */
    virtual void set_mode(CostMode mode) {
        // Default: do nothing (single-agent algorithms ignore mode)
        (void)mode;
    }

    /**
     * @brief Clone the cost function.
     *
     * @return std::unique_ptr<Cost> Deep copy of the cost
     */
    [[nodiscard]] virtual std::unique_ptr<Cost> clone() const = 0;

protected:
    // Allow copy only for cloning in derived classes
    Cost(const Cost&) = default;
    Cost& operator=(const Cost&) = default;
};

/**
 * @brief Shared pointer type for cost functions.
 */
using CostPtr = std::shared_ptr<Cost>;
using CostConstPtr = std::shared_ptr<const Cost>;

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_COST_COST_H_
