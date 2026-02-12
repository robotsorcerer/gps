/**
 * @file cost_action.cpp
 * @brief Implementation of CostAction class.
 *
 * Implements game-theoretic cost formulation for iDG algorithm.
 */

#include "cost_action.h"

#include <stdexcept>

namespace gps {

CostAction::CostAction(const Hyperparams& hyperparams)
    : gamma_(get_hyperparam_or<double>(hyperparams, "gamma", 1.0)),
      mode_(CostMode::PROTAGONIST) {

    // Parse weights
    if (hyperparams.contains("wu")) {
        wu_ = get_hyperparam<Vector>(hyperparams, "wu");
    }

    // Parse mode
    if (hyperparams.contains("mode")) {
        const auto& mode_str = get_hyperparam<std::string>(hyperparams, "mode");
        if (mode_str == "protagonist") {
            mode_ = CostMode::PROTAGONIST;
        } else if (mode_str == "antagonist") {
            mode_ = CostMode::ANTAGONIST;
        } else if (mode_str == "robust") {
            mode_ = CostMode::ROBUST;
        } else {
            throw std::invalid_argument("Unknown cost mode: " + mode_str);
        }
    }
}

CostAction::CostAction(Vector wu, double gamma, CostMode mode)
    : wu_(std::move(wu)),
      gamma_(gamma),
      mode_(mode) {
}

CostResult CostAction::eval(const Sample& sample) const {
    const int T = sample.T();
    const int dX = sample.dX();
    const int dU = sample.dU();
    const int dV = sample.dV();

    // Initialize result with zeros
    CostResult result;
    result.l = Vector::Zero(T);
    result.lx = Matrix::Zero(T, dX);
    result.lu = Matrix::Zero(T, dU);
    result.lxx.resize(static_cast<std::size_t>(T));
    result.luu.resize(static_cast<std::size_t>(T));
    result.lux.resize(static_cast<std::size_t>(T));

    for (int t = 0; t < T; ++t) {
        result.lxx[static_cast<std::size_t>(t)] = Matrix::Zero(dX, dX);
        result.luu[static_cast<std::size_t>(t)] = Matrix::Zero(dU, dU);
        result.lux[static_cast<std::size_t>(t)] = Matrix::Zero(dU, dX);
    }

    // Initialize adversary terms if needed
    if (dV > 0 || mode_ == CostMode::ANTAGONIST || mode_ == CostMode::ROBUST) {
        int actual_dV = (dV > 0) ? dV : dU;  // For antagonist, dV may equal dU
        result.lv = Matrix::Zero(T, actual_dV);
        result.lvv.resize(static_cast<std::size_t>(T));
        result.lvx.resize(static_cast<std::size_t>(T));
        result.luv.resize(static_cast<std::size_t>(T));

        for (int t = 0; t < T; ++t) {
            result.lvv[static_cast<std::size_t>(t)] = Matrix::Zero(actual_dV, actual_dV);
            result.lvx[static_cast<std::size_t>(t)] = Matrix::Zero(actual_dV, dX);
            result.luv[static_cast<std::size_t>(t)] = Matrix::Zero(dU, actual_dV);
        }
    }

    // Evaluate based on mode
    switch (mode_) {
        case CostMode::PROTAGONIST:
            eval_protagonist(sample, result);
            break;

        case CostMode::ANTAGONIST:
            // For standalone eval without protagonist sample, use simple quadratic
            // (Full game-theoretic formulation requires eval_with_protagonist)
            eval_protagonist(sample, result);
            break;

        case CostMode::ROBUST:
            eval_robust(sample, result);
            break;
    }

    return result;
}

CostResult CostAction::eval_with_protagonist(
    const Sample& sample,
    const Sample& sample_prot) const {

    const int T = sample.T();
    const int dX = sample.dX();
    const int dU = sample.dU();

    // Initialize result
    CostResult result;
    result.l = Vector::Zero(T);
    result.lx = Matrix::Zero(T, dX);
    result.lv = Matrix::Zero(T, dU);  // Adversary action derivatives
    result.lxx.resize(static_cast<std::size_t>(T));
    result.lvv.resize(static_cast<std::size_t>(T));
    result.lvx.resize(static_cast<std::size_t>(T));

    for (int t = 0; t < T; ++t) {
        result.lxx[static_cast<std::size_t>(t)] = Matrix::Zero(dX, dX);
        result.lvv[static_cast<std::size_t>(t)] = Matrix::Zero(dU, dU);
        result.lvx[static_cast<std::size_t>(t)] = Matrix::Zero(dU, dX);
    }

    // Evaluate game-theoretic antagonist cost
    eval_antagonist_game_theoretic(sample, sample_prot, result);

    return result;
}

void CostAction::eval_protagonist(const Sample& sample, CostResult& result) const {
    const int T = sample.T();
    const int dU = sample.dU();

    // Get action data
    Matrix U = sample.get_U();

    // Use identity weights if not specified
    Vector w = wu_.size() > 0 ? wu_ : Vector::Ones(dU);

    // Weight matrix
    Matrix Wu = w.asDiagonal();

    for (int t = 0; t < T; ++t) {
        Vector u = U.row(t).transpose();

        // Cost: l = 0.5 * u^T * W * u
        result.l(t) += 0.5 * u.transpose() * Wu * u;

        // Gradient: lu = W * u
        result.lu.row(t) += (Wu * u).transpose();

        // Hessian: luu = W
        result.luu[static_cast<std::size_t>(t)] += Wu;
    }
}

void CostAction::eval_antagonist_game_theoretic(
    const Sample& sample,
    const Sample& sample_prot,
    CostResult& result) const {

    const int T = sample.T();
    const int dU = sample.dU();

    // Get action data
    Matrix V = sample.get_U();      // Adversary's actions (stored as U in antagonist sample)
    Matrix U_prot = sample_prot.get_U();  // Protagonist's actions

    // Use identity weights if not specified
    Vector w = wu_.size() > 0 ? wu_ : Vector::Ones(dU);

    for (int t = 0; t < T; ++t) {
        Vector v = V.row(t).transpose();
        Vector u_prot = U_prot.row(t).transpose();

        // Game-theoretic cost (from Python):
        // l = 0.5 * sum(wu * u_prot^2) - gamma * sum(wu * v^2)
        double prot_term = 0.5 * (w.array() * u_prot.array().square()).sum();
        double adv_term = gamma_ * (w.array() * v.array().square()).sum();
        result.l(t) = prot_term - adv_term;

        // First derivative w.r.t. v:
        // lv = 0.5 * sum(wu * u_prot^2) - 2 * gamma * wu * v
        // Note: Python tiles this scalar across dimensions, then applies -2*gamma*wu*v
        // For simplicity, we compute per-element: lv_i = -2 * gamma * w_i * v_i
        for (int i = 0; i < dU; ++i) {
            result.lv(t, i) = -2.0 * gamma_ * w(i) * v(i);
        }

        // Second derivative w.r.t. v:
        // lvv = -2 * gamma * diag(wu)
        result.lvv[static_cast<std::size_t>(t)] = -2.0 * gamma_ * w.asDiagonal();
    }

    // NEGATE everything for maximization objective (antagonist maximizes)
    result.l = -result.l;
    result.lx = -result.lx;
    result.lv = -result.lv;
    for (auto& lxx : result.lxx) { lxx = -lxx; }
    for (auto& lvv : result.lvv) { lvv = -lvv; }
    for (auto& lvx : result.lvx) { lvx = -lvx; }
}

void CostAction::eval_robust(const Sample& sample, CostResult& result) const {
    const int T = sample.T();
    const int dU = sample.dU();
    const int dV = sample.dV();

    // Get action data
    Matrix U = sample.get_U();
    Matrix V = sample.get_V();

    // Use identity weights if not specified
    Vector w = wu_.size() > 0 ? wu_ : Vector::Ones(dU);
    Matrix Wu = w.asDiagonal();

    for (int t = 0; t < T; ++t) {
        Vector u = U.row(t).transpose();
        Vector v = V.row(t).transpose();

        // Robust cost (from Python):
        // l = 0.5 * sum(wu * u^2) - gamma * sum(wu * v^2)
        double prot_term = 0.5 * (w.array() * u.array().square()).sum();
        double adv_term = gamma_ * (w.array() * v.array().square()).sum();
        result.l(t) = prot_term - adv_term;

        // Protagonist derivatives (positive, minimize u)
        result.lu.row(t) = (Wu * u).transpose();
        result.luu[static_cast<std::size_t>(t)] = Wu;

        // Antagonist derivatives (negative for maximization of v)
        // lv = -2 * gamma * wu * v (then negated)
        for (int i = 0; i < dV; ++i) {
            result.lv(t, i) = 2.0 * gamma_ * w(i) * v(i);  // Negated: sign flipped
        }

        // lvv = -(-2 * gamma * diag(wu)) = 2 * gamma * diag(wu) after negation
        result.lvv[static_cast<std::size_t>(t)] = 2.0 * gamma_ * w.head(dV).asDiagonal();
    }

    // Note: In robust mode, lu terms are positive (protagonist minimizes)
    // and lv terms are already negated (antagonist maximizes)
    // This matches the Python: "lu terms are +ve while lv terms are -ve"
}

std::unique_ptr<Cost> CostAction::clone() const {
    return std::make_unique<CostAction>(*this);
}

}  // namespace gps
