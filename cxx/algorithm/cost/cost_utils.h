/**
 * @file cost_utils.h
 * @brief Cost function utilities.
 *
 * C++20 port of python/gps/algorithm/cost/cost_utils.py
 * Provides L1/L2 evaluation and cost ramping utilities.
 */

#ifndef GPS_CXX_ALGORITHM_COST_COST_UTILS_H_
#define GPS_CXX_ALGORITHM_COST_COST_UTILS_H_

#include <cmath>
#include <tuple>

#include "../../utility/types.h"

namespace gps {

/**
 * @brief Cost ramping options.
 */
enum class RampOption {
    CONSTANT,       // No ramping
    LINEAR,         // Linear ramp
    QUADRATIC,      // Quadratic ramp
    FINAL_ONLY,     // Only final timestep
};

/**
 * @brief Get ramp multiplier for each timestep.
 *
 * @param option Ramping option
 * @param T Number of timesteps
 * @param wp_final_multiplier Final weight multiplier
 * @return Vector Multipliers for each timestep (T,)
 */
[[nodiscard]] inline Vector get_ramp_multiplier(
    RampOption option, int T, double wp_final_multiplier = 1.0) {

    Vector wpm = Vector::Ones(T);

    switch (option) {
        case RampOption::CONSTANT:
            // All weights equal
            break;

        case RampOption::LINEAR:
            // Linear increase from 0 to wp_final_multiplier
            for (int t = 0; t < T; ++t) {
                wpm(t) = static_cast<double>(t) / (T - 1) * wp_final_multiplier;
            }
            break;

        case RampOption::QUADRATIC:
            // Quadratic increase
            for (int t = 0; t < T; ++t) {
                double ratio = static_cast<double>(t) / (T - 1);
                wpm(t) = ratio * ratio * wp_final_multiplier;
            }
            break;

        case RampOption::FINAL_ONLY:
            // Zero except for final timestep
            wpm.setZero();
            wpm(T - 1) = wp_final_multiplier;
            break;
    }

    return wpm;
}

/**
 * @brief Result of L1/L2 term evaluation.
 */
struct L1L2Result {
    Vector l;      // Cost values (T,)
    Matrix ls;     // First derivatives (T, dim)
    Tensor3d lss;  // Second derivatives T x (dim, dim)
};

/**
 * @brief Evaluate L1/L2 penalty term with Huber smoothing.
 *
 * Computes cost and derivatives for weighted L1/L2 distance.
 * Uses Huber loss for smooth L1 approximation when alpha > 0.
 *
 * @param wp Weight matrix (T, dim)
 * @param dist Distance vectors (T, dim)
 * @param l1 L1 regularization weight
 * @param l2 L2 regularization weight
 * @param alpha Huber threshold (0 for pure L1)
 * @return L1L2Result Cost and derivatives
 */
[[nodiscard]] inline L1L2Result eval_l1l2_term(
    const Matrix& wp, const Matrix& dist,
    double l1, double l2, double alpha) {

    const int T = static_cast<int>(dist.rows());
    const int dim = static_cast<int>(dist.cols());

    L1L2Result result;
    result.l = Vector::Zero(T);
    result.ls = Matrix::Zero(T, dim);
    result.lss.resize(T);

    for (int t = 0; t < T; ++t) {
        result.lss[t] = Matrix::Zero(dim, dim);

        for (int d = 0; d < dim; ++d) {
            double w = wp(t, d);
            double x = dist(t, d);
            double abs_x = std::abs(x);
            double sign_x = (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);

            // L2 term: 0.5 * w * x^2
            result.l(t) += l2 * 0.5 * w * x * x;
            result.ls(t, d) += l2 * w * x;
            result.lss[t](d, d) += l2 * w;

            // L1 term with Huber smoothing
            if (alpha > 0 && abs_x < alpha) {
                // Quadratic region for smooth approximation
                double huber = 0.5 * x * x / alpha;
                result.l(t) += l1 * w * huber;
                result.ls(t, d) += l1 * w * x / alpha;
                result.lss[t](d, d) += l1 * w / alpha;
            } else {
                // Linear region
                result.l(t) += l1 * w * (abs_x - 0.5 * alpha);
                result.ls(t, d) += l1 * w * sign_x;
                // Hessian is zero in linear region
            }
        }
    }

    return result;
}

/**
 * @brief Compute log barrier cost.
 *
 * For constraint satisfaction: cost increases as value approaches bound.
 *
 * @param x Value to constrain
 * @param lower Lower bound
 * @param upper Upper bound
 * @param weight Barrier weight
 * @return double Barrier cost
 */
[[nodiscard]] inline double log_barrier(
    double x, double lower, double upper, double weight) {

    constexpr double EPS = 1e-6;

    if (x <= lower + EPS || x >= upper - EPS) {
        return std::numeric_limits<double>::infinity();
    }

    return -weight * (std::log(x - lower) + std::log(upper - x));
}

}  // namespace gps

#endif  // GPS_CXX_ALGORITHM_COST_COST_UTILS_H_
