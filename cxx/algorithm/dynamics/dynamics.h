/**
 * @file dynamics.h
 * @brief Abstract Dynamics base class.
 *
 * C++20 port of python/gps/algorithm/dynamics/dynamics.py
 * Defines interface for linear dynamics models: x_{t+1} = Fm * [x_t; u_t] + fv
 */

#ifndef GPS_CXX_ALGORITHM_DYNAMICS_DYNAMICS_H_
#define GPS_CXX_ALGORITHM_DYNAMICS_DYNAMICS_H_

#include <memory>

#include "../../sample/sample.h"
#include "../../utility/types.h"

namespace gps {

// Forward declaration
class SampleList;

/**
 * @brief Fitted linear dynamics model.
 *
 * Represents linearized dynamics: x_{t+1} = Fm_t * [x_t; u_t] + fv_t
 * with covariance dyn_covar.
 */
struct LinearDynamics {
    Tensor3d Fm;        // Dynamics matrices: T x (dX, dX+dU)
    Matrix fv;          // Bias terms: (T, dX)
    Tensor3d dyn_covar; // Process noise covariances: T x (dX, dX)

    /**
     * @brief Create zero-initialized dynamics.
     */
    static LinearDynamics zeros(int T, int dX, int dU);

    /**
     * @brief Predict next state.
     *
     * @param x Current state
     * @param u Action
     * @param t Timestep
     * @return Vector Predicted next state
     */
    [[nodiscard]] Vector predict(const Vector& x, const Vector& u, int t) const;
};

/**
 * @brief Abstract base class for dynamics estimation.
 *
 * Dynamics models fit linear approximations to the system dynamics
 * from trajectory samples.
 */
class Dynamics {
public:
    explicit Dynamics(const Hyperparams& hyperparams);
    virtual ~Dynamics() = default;

    // Non-copyable
    Dynamics(const Dynamics&) = delete;
    Dynamics& operator=(const Dynamics&) = delete;
    Dynamics(Dynamics&&) = default;
    Dynamics& operator=(Dynamics&&) = default;

    /**
     * @brief Update dynamics prior from data.
     *
     * @param X State trajectories (N x T x dX)
     * @param U Action trajectories (N x T x dU)
     */
    virtual void update_prior(const Tensor3d& X, const Tensor3d& U) = 0;

    /**
     * @brief Fit dynamics from sample list.
     *
     * @param samples Collection of trajectory samples
     */
    virtual void fit(const SampleList& samples) = 0;

    /**
     * @brief Get the fitted dynamics.
     */
    [[nodiscard]] const LinearDynamics& dynamics() const noexcept {
        return dynamics_;
    }

    /**
     * @brief Get mutable dynamics for modification.
     */
    LinearDynamics& dynamics() noexcept { return dynamics_; }

    /**
     * @brief Clone the dynamics estimator.
     */
    [[nodiscard]] virtual std::unique_ptr<Dynamics> clone() const = 0;

protected:
    Hyperparams hyperparams_;
    LinearDynamics dynamics_;
};

/**
 * @brief Shared pointer type for dynamics.
 */
using DynamicsPtr = std::shared_ptr<Dynamics>;

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_DYNAMICS_DYNAMICS_H_
