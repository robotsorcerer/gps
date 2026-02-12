/**
 * @file policy.h
 * @brief Abstract Policy base class.
 *
 * C++20 port of python/gps/algorithm/policy/policy.py
 * Defines the interface that all GPS policies must implement.
 */

#ifndef GPS_CXX_ALGORITHM_POLICY_POLICY_H_
#define GPS_CXX_ALGORITHM_POLICY_POLICY_H_

#include <memory>
#include <optional>

#include "../../sample/sample.h"
#include "../../utility/types.h"

namespace gps {

/**
 * @brief Abstract base class for all GPS policies.
 *
 * Policies compute actions given states/observations.
 * Subclasses implement specific policy types:
 * - LinearGaussianPolicy: U = K*x + k + noise
 * - NeuralNetworkPolicy: U = NN(obs)
 */
class Policy {
public:
    Policy() = default;
    virtual ~Policy() = default;

    // Moveable
    Policy(Policy&&) = default;
    Policy& operator=(Policy&&) = default;

    /**
     * @brief Compute action given current state.
     *
     * @param x Current state vector (or nullptr for observation-based)
     * @param obs Current observation vector (for NN policies)
     * @param t Current timestep (for time-varying policies)
     * @param noise Optional noise to add to action
     * @return Vector Action to execute
     */
    [[nodiscard]] virtual Vector act(
        const Vector* x,
        const Vector* obs,
        int t,
        const Vector* noise) const = 0;

    /**
     * @brief Get action dimensionality.
     *
     * @return int Action dimension dU
     */
    [[nodiscard]] virtual int dU() const noexcept = 0;

    /**
     * @brief Get state dimensionality (if applicable).
     *
     * @return int State dimension dX, or 0 if not state-based
     */
    [[nodiscard]] virtual int dX() const noexcept { return 0; }

    /**
     * @brief Get observation dimensionality (if applicable).
     *
     * @return int Observation dimension dO, or 0 if not observation-based
     */
    [[nodiscard]] virtual int dO() const noexcept { return 0; }

    /**
     * @brief Get time horizon (for time-varying policies).
     *
     * @return int Number of timesteps, or 0 for time-invariant
     */
    [[nodiscard]] virtual int T() const noexcept { return 0; }

    /**
     * @brief Clone the policy.
     *
     * @return std::unique_ptr<Policy> Deep copy of the policy
     */
    [[nodiscard]] virtual std::unique_ptr<Policy> clone() const = 0;

protected:
    // Allow copy only for cloning in derived classes
    Policy(const Policy&) = default;
    Policy& operator=(const Policy&) = default;
};

/**
 * @brief Shared pointer type for policies.
 */
using PolicyPtr = std::shared_ptr<Policy>;
using PolicyConstPtr = std::shared_ptr<const Policy>;

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_POLICY_POLICY_H_
