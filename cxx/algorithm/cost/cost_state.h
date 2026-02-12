/**
 * @file cost_state.h
 * @brief State-based cost function (L1/L2 distance to target).
 *
 * C++20 port of python/gps/algorithm/cost/cost_state.py
 */

#ifndef GPS_CXX_ALGORITHM_COST_COST_STATE_H_
#define GPS_CXX_ALGORITHM_COST_COST_STATE_H_

#include <map>
#include <string>

#include "cost.h"
#include "cost_utils.h"
#include "../../utility/types.h"

namespace gps {

/**
 * @brief Configuration for a single state data type cost term.
 */
struct StateTargetConfig {
    Vector wp;           // Per-dimension weights
    Vector target_state; // Target state for this data type
};

/**
 * @brief Computes L1/L2 distance to fixed target state(s).
 *
 * Supports multiple state data types with individual weights and targets.
 */
class CostState : public Cost {
public:
    /**
     * @brief Construct state cost from hyperparameters.
     *
     * @param hyperparams Configuration with:
     *   - l1: L1 regularization weight
     *   - l2: L2 regularization weight
     *   - alpha: Huber loss threshold
     *   - ramp_option: Cost ramping option (RAMP_CONSTANT, etc.)
     *   - wp_final_multiplier: Final weight multiplier for ramping
     *   - data_types: Map of sensor type -> {wp, target_state}
     */
    explicit CostState(const Hyperparams& hyperparams);

    ~CostState() override = default;

    CostState(const CostState&) = default;
    CostState& operator=(const CostState&) = default;
    CostState(CostState&&) = default;
    CostState& operator=(CostState&&) = default;

    /**
     * @brief Evaluate state cost on sample.
     *
     * Computes L1/L2 cost for each configured data type.
     *
     * @param sample Trajectory sample
     * @return CostResult Cost and derivatives
     */
    [[nodiscard]] CostResult eval(const Sample& sample) const override;

    [[nodiscard]] std::unique_ptr<Cost> clone() const override;

    /**
     * @brief Add a state target configuration.
     *
     * @param sensor_type Sensor type to track
     * @param config Target configuration
     */
    void add_target(SampleType sensor_type, StateTargetConfig config);

private:
    double l1_;                 // L1 weight
    double l2_;                 // L2 weight
    double alpha_;              // Huber threshold
    RampOption ramp_option_;    // Cost ramping
    double wp_final_multiplier_; // Final weight multiplier

    // Per-sensor-type configurations
    std::map<SampleType, StateTargetConfig> targets_;
};

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_COST_COST_STATE_H_
