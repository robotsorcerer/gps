/**
 * @file cost_action.h
 * @brief Action-based cost function with game-theoretic support.
 *
 * C++20 port of python/gps/algorithm/cost/cost_action.py
 * Implements quadratic action cost with protagonist/antagonist modes.
 */

#ifndef GPS_CXX_ALGORITHM_COST_COST_ACTION_H_
#define GPS_CXX_ALGORITHM_COST_COST_ACTION_H_

#include "cost.h"
#include "../../utility/types.h"

namespace gps {

/**
 * @brief Quadratic action cost with iDG game-theoretic formulation.
 *
 * Supports three modes for game-theoretic trajectory optimization:
 * - PROTAGONIST: l = 0.5 * u^T * W * u (minimize)
 * - ANTAGONIST: l = 0.5 * u_prot^T * W * u_prot - gamma * v^T * W * v (maximize via negation)
 * - ROBUST: Combined protagonist + adversary cost
 */
class CostAction : public Cost {
public:
    /**
     * @brief Construct an action cost.
     *
     * @param hyperparams Configuration with:
     *   - wu: Action weights (dU,)
     *   - gamma: Adversarial weight (for ANTAGONIST/ROBUST modes)
     *   - mode: Cost mode string ("protagonist", "antagonist", "robust")
     */
    explicit CostAction(const Hyperparams& hyperparams);

    /**
     * @brief Construct with explicit weights.
     *
     * @param wu Action cost weight vector (dU,)
     * @param gamma Adversarial weight for game-theoretic formulation
     * @param mode Cost mode
     */
    CostAction(
        Vector wu,
        double gamma = 1.0,
        CostMode mode = CostMode::PROTAGONIST);

    ~CostAction() override = default;

    CostAction(const CostAction&) = default;
    CostAction& operator=(const CostAction&) = default;
    CostAction(CostAction&&) = default;
    CostAction& operator=(CostAction&&) = default;

    /**
     * @brief Evaluate action cost on sample.
     *
     * @param sample Trajectory sample
     * @return CostResult Cost and derivatives
     */
    [[nodiscard]] CostResult eval(const Sample& sample) const override;

    /**
     * @brief Evaluate antagonist cost with protagonist sample.
     *
     * Required for iDG algorithm where antagonist cost depends on
     * the protagonist's actions.
     *
     * @param sample Antagonist sample (contains adversary actions v)
     * @param sample_prot Protagonist sample (contains control actions u)
     * @return CostResult Cost and derivatives (NEGATED for maximization)
     */
    [[nodiscard]] CostResult eval_with_protagonist(
        const Sample& sample,
        const Sample& sample_prot) const;

    [[nodiscard]] CostMode mode() const noexcept override { return mode_; }
    void set_mode(CostMode mode) override { mode_ = mode; }

    [[nodiscard]] std::unique_ptr<Cost> clone() const override;

    // Accessors
    [[nodiscard]] const Vector& wu() const noexcept { return wu_; }
    [[nodiscard]] double gamma() const noexcept { return gamma_; }

    // Setters
    void set_gamma(double gamma) noexcept { gamma_ = gamma; }

private:
    Vector wu_;        // Action cost weights (dU,)
    double gamma_;     // Adversarial weight for game-theoretic cost
    CostMode mode_;

    /**
     * @brief Evaluate protagonist mode: l = 0.5 * u^T * W * u
     */
    void eval_protagonist(const Sample& sample, CostResult& result) const;

    /**
     * @brief Evaluate antagonist mode with game-theoretic formulation.
     *
     * l = 0.5 * sum(wu * u_prot^2) - gamma * sum(wu * v^2)
     * Returns NEGATED cost for maximization objective.
     */
    void eval_antagonist_game_theoretic(
        const Sample& sample,
        const Sample& sample_prot,
        CostResult& result) const;

    /**
     * @brief Evaluate robust mode with combined cost.
     *
     * l = 0.5 * sum(wu * u^2) - gamma * sum(wu * v^2)
     * Protagonist terms are positive, antagonist terms are negative.
     */
    void eval_robust(const Sample& sample, CostResult& result) const;
};

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_COST_COST_ACTION_H_
