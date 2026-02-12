/**
 * @file lin_gauss_policy.h
 * @brief Linear Gaussian policy classes.
 *
 * C++20 port of python/gps/algorithm/policy/lin_gauss_policy.py
 * Implements time-varying linear Gaussian controllers: U = K*x + k + noise
 */

#ifndef GPS_CXX_ALGORITHM_POLICY_LIN_GAUSS_POLICY_H_
#define GPS_CXX_ALGORITHM_POLICY_LIN_GAUSS_POLICY_H_

#include <optional>

#include "policy.h"
#include "../../utility/types.h"

namespace gps {

/**
 * @brief Time-varying linear Gaussian policy.
 *
 * Computes actions as: U_t = K_t * x_t + k_t + chol(pol_covar_t) * noise
 *
 * Used as the local controller in GPS trajectory optimization.
 */
class LinearGaussianPolicy : public Policy {
public:
    /**
     * @brief Default constructor (creates empty policy).
     */
    LinearGaussianPolicy() : T_(0), dX_(0), dU_(0) {}

    /**
     * @brief Construct a LinearGaussianPolicy.
     *
     * @param K Feedback gain matrices: T x (dU, dX)
     * @param k Feedforward terms: (T, dU)
     * @param pol_covar Policy covariances: T x (dU, dU)
     * @param chol_pol_covar Cholesky of covariances: T x (dU, dU)
     * @param inv_pol_covar Inverse covariances: T x (dU, dU)
     */
    LinearGaussianPolicy(
        Tensor3d K,
        Matrix k,
        Tensor3d pol_covar,
        Tensor3d chol_pol_covar,
        Tensor3d inv_pol_covar);

    /**
     * @brief Construct from hyperparameters with initial values.
     *
     * @param hyperparams Configuration with T, dX, dU, init_var
     */
    explicit LinearGaussianPolicy(const Hyperparams& hyperparams);

    /**
     * @brief Construct with default initialization.
     *
     * Creates zero K and k, with identity covariance scaled by init_var.
     *
     * @param T Time horizon
     * @param dX State dimension
     * @param dU Action dimension
     * @param init_var Initial variance (default 1.0)
     */
    LinearGaussianPolicy(int T, int dX, int dU, double init_var = 1.0);

    ~LinearGaussianPolicy() override = default;

    // Allow copy for this concrete type
    LinearGaussianPolicy(const LinearGaussianPolicy&) = default;
    LinearGaussianPolicy& operator=(const LinearGaussianPolicy&) = default;
    LinearGaussianPolicy(LinearGaussianPolicy&&) = default;
    LinearGaussianPolicy& operator=(LinearGaussianPolicy&&) = default;

    /**
     * @brief Compute action for given state.
     *
     * @param x State vector (required, dimension dX)
     * @param obs Observation (ignored for linear Gaussian)
     * @param t Timestep
     * @param noise Noise vector to scale by chol_pol_covar (optional)
     * @return Vector Action of dimension dU
     */
    [[nodiscard]] Vector act(
        const Vector* x,
        const Vector* obs,
        int t,
        const Vector* noise) const override;

    // Dimension accessors
    [[nodiscard]] int dU() const noexcept override { return dU_; }
    [[nodiscard]] int dX() const noexcept override { return dX_; }
    [[nodiscard]] int T() const noexcept override { return T_; }

    [[nodiscard]] std::unique_ptr<Policy> clone() const override;

    /**
     * @brief Fold noise into feedforward term.
     *
     * Returns k + chol_pol_covar * noise for direct action computation.
     *
     * @param noise Noise matrix of shape (T, dU)
     * @return Matrix Folded feedforward terms (T, dU)
     */
    [[nodiscard]] Matrix fold_k(const Matrix& noise) const;

    /**
     * @brief Get mean action without noise.
     *
     * @param x State vector
     * @param t Timestep
     * @return Vector Mean action K_t * x + k_t
     */
    [[nodiscard]] Vector mean_action(const Vector& x, int t) const;

    /**
     * @brief Compute log probability of action given state.
     *
     * @param x State vector
     * @param u Action vector
     * @param t Timestep
     * @return double Log probability density
     */
    [[nodiscard]] double log_prob(const Vector& x, const Vector& u, int t) const;

    // Accessors for internal matrices
    [[nodiscard]] const Tensor3d& K() const noexcept { return K_; }
    [[nodiscard]] const Matrix& k() const noexcept { return k_; }
    [[nodiscard]] const Tensor3d& pol_covar() const noexcept { return pol_covar_; }
    [[nodiscard]] const Tensor3d& chol_pol_covar() const noexcept { return chol_pol_covar_; }
    [[nodiscard]] const Tensor3d& inv_pol_covar() const noexcept { return inv_pol_covar_; }

    // Mutable accessors for trajectory optimization updates
    Tensor3d& K() noexcept { return K_; }
    Matrix& k() noexcept { return k_; }
    Tensor3d& pol_covar() noexcept { return pol_covar_; }
    Tensor3d& chol_pol_covar() noexcept { return chol_pol_covar_; }
    Tensor3d& inv_pol_covar() noexcept { return inv_pol_covar_; }

private:
    int T_;   // Time horizon
    int dX_;  // State dimension
    int dU_;  // Action dimension

    Tensor3d K_;             // Feedback gains: T x (dU, dX)
    Matrix k_;               // Feedforward terms: (T, dU)
    Tensor3d pol_covar_;     // Policy covariances: T x (dU, dU)
    Tensor3d chol_pol_covar_; // Cholesky of covariances: T x (dU, dU)
    Tensor3d inv_pol_covar_;  // Inverse covariances: T x (dU, dU)
};

/**
 * @brief Robust Linear Gaussian policy for iDG algorithms.
 *
 * Combines protagonist and antagonist policies for game-theoretic
 * robust trajectory optimization.
 *
 * In PROTAGONIST mode: U = K_pro * x + k_pro
 * In ANTAGONIST mode: V = K_ant * x + k_ant
 * In ROBUST mode: Combined policy considering both
 */
class LinearGaussianPolicyRobust : public Policy {
public:
    /**
     * @brief Construct a robust policy from protagonist and antagonist.
     *
     * @param protagonist Protagonist (control) policy
     * @param antagonist Antagonist (adversary) policy
     * @param mode Initial policy mode (default: PROTAGONIST)
     */
    LinearGaussianPolicyRobust(
        LinearGaussianPolicy protagonist,
        LinearGaussianPolicy antagonist,
        PolicyMode mode = PolicyMode::PROTAGONIST);

    /**
     * @brief Construct from hyperparameters.
     *
     * @param hyperparams Configuration with T, dX, dU, dV, init_var
     */
    explicit LinearGaussianPolicyRobust(const Hyperparams& hyperparams);

    ~LinearGaussianPolicyRobust() override = default;

    LinearGaussianPolicyRobust(const LinearGaussianPolicyRobust&) = default;
    LinearGaussianPolicyRobust& operator=(const LinearGaussianPolicyRobust&) = default;
    LinearGaussianPolicyRobust(LinearGaussianPolicyRobust&&) = default;
    LinearGaussianPolicyRobust& operator=(LinearGaussianPolicyRobust&&) = default;

    /**
     * @brief Compute action based on current mode.
     *
     * @param x State vector
     * @param obs Observation (ignored)
     * @param t Timestep
     * @param noise Noise for stochastic action
     * @return Vector Action (U for protagonist, V for antagonist)
     */
    [[nodiscard]] Vector act(
        const Vector* x,
        const Vector* obs,
        int t,
        const Vector* noise) const override;

    [[nodiscard]] int dU() const noexcept override;
    [[nodiscard]] int dX() const noexcept override { return protagonist_.dX(); }
    [[nodiscard]] int T() const noexcept override { return protagonist_.T(); }

    /**
     * @brief Get adversary action dimension.
     */
    [[nodiscard]] int dV() const noexcept { return antagonist_.dU(); }

    [[nodiscard]] std::unique_ptr<Policy> clone() const override;

    /**
     * @brief Set policy mode.
     *
     * @param mode PROTAGONIST, ANTAGONIST, or ROBUST
     */
    void set_mode(PolicyMode mode) noexcept { mode_ = mode; }

    /**
     * @brief Get current policy mode.
     */
    [[nodiscard]] PolicyMode mode() const noexcept { return mode_; }

    // Access underlying policies
    [[nodiscard]] const LinearGaussianPolicy& protagonist() const noexcept {
        return protagonist_;
    }
    [[nodiscard]] const LinearGaussianPolicy& antagonist() const noexcept {
        return antagonist_;
    }
    LinearGaussianPolicy& protagonist() noexcept { return protagonist_; }
    LinearGaussianPolicy& antagonist() noexcept { return antagonist_; }

private:
    LinearGaussianPolicy protagonist_;  // Control policy (dU actions)
    LinearGaussianPolicy antagonist_;   // Adversary policy (dV actions)
    PolicyMode mode_;
};

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_POLICY_LIN_GAUSS_POLICY_H_
