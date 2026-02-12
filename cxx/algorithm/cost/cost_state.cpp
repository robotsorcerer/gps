/**
 * @file cost_state.cpp
 * @brief Implementation of CostState class.
 */

#include "cost_state.h"

#include <stdexcept>

namespace gps {

CostState::CostState(const Hyperparams& hyperparams)
    : l1_(get_hyperparam_or<double>(hyperparams, "l1", 0.0)),
      l2_(get_hyperparam_or<double>(hyperparams, "l2", 1.0)),
      alpha_(get_hyperparam_or<double>(hyperparams, "alpha", 1e-5)),
      wp_final_multiplier_(
          get_hyperparam_or<double>(hyperparams, "wp_final_multiplier", 1.0)) {

    // Parse ramp option
    auto ramp_str = get_hyperparam_or<std::string>(
        hyperparams, "ramp_option", "constant");

    if (ramp_str == "constant") {
        ramp_option_ = RampOption::CONSTANT;
    } else if (ramp_str == "linear") {
        ramp_option_ = RampOption::LINEAR;
    } else if (ramp_str == "quadratic") {
        ramp_option_ = RampOption::QUADRATIC;
    } else if (ramp_str == "final_only") {
        ramp_option_ = RampOption::FINAL_ONLY;
    } else {
        ramp_option_ = RampOption::CONSTANT;
    }
}

CostResult CostState::eval(const Sample& sample) const {
    const int T = sample.T();
    const int dX = sample.dX();
    const int dU = sample.dU();

    CostResult result;
    result.l = Vector::Zero(T);
    result.lx = Matrix::Zero(T, dX);
    result.lu = Matrix::Zero(T, dU);
    result.lxx.resize(T);
    result.luu.resize(T);
    result.lux.resize(T);

    for (int t = 0; t < T; ++t) {
        result.lxx[t] = Matrix::Zero(dX, dX);
        result.luu[t] = Matrix::Zero(dU, dU);
        result.lux[t] = Matrix::Zero(dU, dX);
    }

    // Get ramp multiplier
    Vector wpm = get_ramp_multiplier(ramp_option_, T, wp_final_multiplier_);

    // Evaluate cost for each configured data type
    for (const auto& [sensor_type, config] : targets_) {
        if (!sample.has(sensor_type)) {
            continue;
        }

        // Get state data for this sensor type
        const Matrix& x = sample.get(sensor_type);
        const int dim = static_cast<int>(x.cols());

        // Compute distance to target
        Matrix dist(T, dim);
        for (int t = 0; t < T; ++t) {
            dist.row(t) = x.row(t) - config.target_state.transpose();
        }

        // Apply ramped weights
        Matrix wp_ramped(T, dim);
        for (int t = 0; t < T; ++t) {
            wp_ramped.row(t) = config.wp.transpose() * wpm(t);
        }

        // Evaluate L1/L2 term
        auto [l, ls, lss] = eval_l1l2_term(wp_ramped, dist, l1_, l2_, alpha_);

        // Accumulate into result
        result.l += l;

        // Pack derivatives into state dimensions
        // This is a simplified version - full implementation would use
        // pack_data_x like the Python version
        // For now, assume the sensor type corresponds to state dimensions
        // starting at offset 0
        int offset = 0;  // Would be computed based on sensor type
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < dim && (offset + d) < dX; ++d) {
                result.lx(t, offset + d) += ls(t, d);
                for (int d2 = 0; d2 < dim && (offset + d2) < dX; ++d2) {
                    result.lxx[t](offset + d, offset + d2) += lss[t](d, d2);
                }
            }
        }
    }

    return result;
}

void CostState::add_target(SampleType sensor_type, StateTargetConfig config) {
    targets_[sensor_type] = std::move(config);
}

std::unique_ptr<Cost> CostState::clone() const {
    return std::make_unique<CostState>(*this);
}

}  // namespace gps
