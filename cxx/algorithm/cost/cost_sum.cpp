/**
 * @file cost_sum.cpp
 * @brief Implementation of CostSum class.
 */

#include "cost_sum.h"

#include <stdexcept>

namespace gps {

CostSum::CostSum(
    std::vector<CostPtr> costs,
    std::vector<double> weights,
    CostMode mode)
    : costs_(std::move(costs)),
      weights_(std::move(weights)),
      mode_(mode) {

    if (costs_.size() != weights_.size()) {
        throw std::invalid_argument(
            "Number of costs must match number of weights");
    }
}

CostSum::CostSum(const Hyperparams& /*hyperparams*/)
    : mode_(CostMode::PROTAGONIST) {
    // Cost construction from hyperparams would require a factory pattern
    // For now, use add_cost() to add costs after construction
}

CostResult CostSum::eval(const Sample& sample) const {
    if (costs_.empty()) {
        throw std::runtime_error("CostSum has no cost functions");
    }

    // Evaluate first cost with its weight
    CostResult result = costs_[0]->eval(sample);
    double w = weights_[0];

    result.l *= w;
    result.lx *= w;
    result.lu *= w;

    for (auto& lxx : result.lxx) { lxx *= w; }
    for (auto& luu : result.luu) { luu *= w; }
    for (auto& lux : result.lux) { lux *= w; }

    // Handle adversary terms if present
    bool has_adversary = !result.lv.isZero(0);
    if (has_adversary) {
        result.lv *= w;
        for (auto& lvv : result.lvv) { lvv *= w; }
        for (auto& lvx : result.lvx) { lvx *= w; }
        for (auto& luv : result.luv) { luv *= w; }
    }

    // Add remaining costs
    for (std::size_t i = 1; i < costs_.size(); ++i) {
        CostResult part = costs_[i]->eval(sample);
        w = weights_[i];

        result.l += w * part.l;
        result.lx += w * part.lx;
        result.lu += w * part.lu;

        const int T = static_cast<int>(result.lxx.size());
        for (int t = 0; t < T; ++t) {
            result.lxx[t] += w * part.lxx[t];
            result.luu[t] += w * part.luu[t];
            result.lux[t] += w * part.lux[t];
        }

        // Handle adversary terms
        if (!part.lv.isZero(0)) {
            if (!has_adversary) {
                // Initialize adversary terms
                result.lv = w * part.lv;
                result.lvv = part.lvv;
                result.lvx = part.lvx;
                result.luv = part.luv;
                for (auto& lvv : result.lvv) { lvv *= w; }
                for (auto& lvx : result.lvx) { lvx *= w; }
                for (auto& luv : result.luv) { luv *= w; }
                has_adversary = true;
            } else {
                result.lv += w * part.lv;
                for (int t = 0; t < T; ++t) {
                    result.lvv[t] += w * part.lvv[t];
                    result.lvx[t] += w * part.lvx[t];
                    result.luv[t] += w * part.luv[t];
                }
            }
        }
    }

    return result;
}

void CostSum::add_cost(CostPtr cost, double weight) {
    costs_.push_back(std::move(cost));
    weights_.push_back(weight);
}

std::unique_ptr<Cost> CostSum::clone() const {
    // Clone all child costs
    std::vector<CostPtr> cloned_costs;
    cloned_costs.reserve(costs_.size());

    for (const auto& cost : costs_) {
        cloned_costs.push_back(
            std::shared_ptr<Cost>(cost->clone().release()));
    }

    auto cloned = std::make_unique<CostSum>(
        std::move(cloned_costs), weights_, mode_);

    return cloned;
}

}  // namespace gps
