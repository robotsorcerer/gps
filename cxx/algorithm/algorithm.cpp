/**
 * @file algorithm.cpp
 * @brief Implementation of Algorithm base class.
 */

#include "algorithm.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gps {

Algorithm::Algorithm(const Hyperparams& hyperparams)
    : hyperparams_(hyperparams),
      T_(get_hyperparam<int>(hyperparams, "T")),
      dX_(get_hyperparam<int>(hyperparams, "dX")),
      dU_(get_hyperparam<int>(hyperparams, "dU")),
      dV_(get_hyperparam_or<int>(hyperparams, "dV", 0)),
      dO_(get_hyperparam_or<int>(hyperparams, "dO", 0)),
      base_kl_step_(get_hyperparam_or<double>(hyperparams, "kl_step", 1.0)) {

    // Set number of conditions
    if (hyperparams.contains("conditions")) {
        M_ = get_hyperparam<int>(hyperparams, "conditions");
    } else if (hyperparams.contains("train_conditions")) {
        const auto& train_cond = get_hyperparam<std::vector<int>>(
            hyperparams, "train_conditions");
        M_ = static_cast<int>(train_cond.size());
        cond_idx_ = train_cond;
    } else {
        M_ = 1;
    }

    // Initialize condition indices if not set
    if (cond_idx_.empty()) {
        cond_idx_.resize(static_cast<std::size_t>(M_));
        std::iota(cond_idx_.begin(), cond_idx_.end(), 0);
    }

    // Initialize iteration data for each condition
    cur_.resize(static_cast<std::size_t>(M_));
    prev_.resize(static_cast<std::size_t>(M_));

    // Initialize trajectory distributions with default values
    double init_var = get_hyperparam_or<double>(hyperparams, "init_var", 1.0);

    for (int m = 0; m < M_; ++m) {
        auto& data = cur_[static_cast<std::size_t>(m)];

        // Create initial trajectory distribution
        data.traj_distr = LinearGaussianPolicy(T_, dX_, dU_, init_var);

        if (dV_ > 0) {
            data.traj_distr_adv = LinearGaussianPolicy(T_, dX_, dV_, init_var);
            data.traj_distr_robust = LinearGaussianPolicy(T_, dX_, dU_, init_var);
        }
    }

    // Initialize new trajectory storage
    new_traj_distr_.reserve(static_cast<std::size_t>(M_));
    if (dV_ > 0) {
        new_traj_distr_adv_.reserve(static_cast<std::size_t>(M_));
    }
}

void Algorithm::iteration_cl(
    const std::vector<SampleList>& /*sample_lists_prot*/,
    const std::vector<SampleList>& /*sample_lists*/) {
    throw std::runtime_error(
        "iteration_cl not implemented in base Algorithm class");
}

void Algorithm::iteration_idg(
    const std::vector<SampleList>& /*sample_lists_prot*/,
    const std::vector<SampleList>& /*sample_lists*/) {
    throw std::runtime_error(
        "iteration_idg not implemented in base Algorithm class");
}

const LinearGaussianPolicy& Algorithm::traj_distr(int condition) const {
    return cur_.at(static_cast<std::size_t>(condition)).traj_distr;
}

const LinearGaussianPolicy& Algorithm::traj_distr_adv(int condition) const {
    return cur_.at(static_cast<std::size_t>(condition)).traj_distr_adv;
}

void Algorithm::update_dynamics() {
    for (int m = 0; m < M_; ++m) {
        auto& data = cur_[static_cast<std::size_t>(m)];

        if (data.sample_list.empty()) {
            continue;
        }

        // Get state and action data
        Tensor3d X = data.sample_list.get_X();
        Tensor3d U = data.sample_list.get_U();

        // Update prior and fit dynamics
        if (data.traj_info.dynamics) {
            data.traj_info.dynamics->update_prior(X, U);
            data.traj_info.dynamics->fit(data.sample_list);
        }

        // Fit initial state distribution
        const int N = static_cast<int>(X.size());
        const int dX = dX_;

        // Compute mean of initial states
        Vector x0mu = Vector::Zero(dX);
        for (int n = 0; n < N; ++n) {
            x0mu += X[static_cast<std::size_t>(n)].row(0).transpose();
        }
        x0mu /= N;
        data.traj_info.x0mu = x0mu;

        // Compute covariance of initial states
        Matrix x0sigma = Matrix::Zero(dX, dX);
        for (int n = 0; n < N; ++n) {
            Vector diff = X[static_cast<std::size_t>(n)].row(0).transpose() - x0mu;
            x0sigma += diff * diff.transpose();
        }
        x0sigma /= (N - 1);

        // Add minimum variance
        double min_var = get_hyperparam_or<double>(
            hyperparams_, "initial_state_var", 1e-6);
        for (int i = 0; i < dX; ++i) {
            x0sigma(i, i) = std::max(x0sigma(i, i), min_var);
        }
        data.traj_info.x0sigma = x0sigma;
    }
}

void Algorithm::update_dynamics_idg() {
    for (int m = 0; m < M_; ++m) {
        auto& data = cur_[static_cast<std::size_t>(m)];

        if (data.sample_list.empty()) {
            continue;
        }

        // Get state, action, and adversary action data
        Tensor3d X = data.sample_list.get_X();
        Tensor3d U = data.sample_list.get_U();
        Tensor3d V = data.sample_list.get_V();

        // Update prior and fit dynamics (robust version)
        if (data.traj_info.dynamics) {
            // Concatenate U and V for combined dynamics fitting
            Tensor3d UV;
            UV.reserve(U.size());
            for (std::size_t n = 0; n < U.size(); ++n) {
                Matrix uv(U[n].rows(), U[n].cols() + V[n].cols());
                uv.leftCols(U[n].cols()) = U[n];
                uv.rightCols(V[n].cols()) = V[n];
                UV.push_back(uv);
            }

            data.traj_info.dynamics->update_prior(X, UV);
            data.traj_info.dynamics->fit(data.sample_list);
        }

        // Fit initial state distribution (same as non-robust)
        const int N = static_cast<int>(X.size());
        const int dX = dX_;

        Vector x0mu = Vector::Zero(dX);
        for (int n = 0; n < N; ++n) {
            x0mu += X[static_cast<std::size_t>(n)].row(0).transpose();
        }
        x0mu /= N;
        data.traj_info.x0mu = x0mu;

        Matrix x0sigma = Matrix::Zero(dX, dX);
        for (int n = 0; n < N; ++n) {
            Vector diff = X[static_cast<std::size_t>(n)].row(0).transpose() - x0mu;
            x0sigma += diff * diff.transpose();
        }
        x0sigma /= (N - 1);

        double min_var = get_hyperparam_or<double>(
            hyperparams_, "initial_state_var", 1e-6);
        for (int i = 0; i < dX; ++i) {
            x0sigma(i, i) = std::max(x0sigma(i, i), min_var);
        }
        data.traj_info.x0sigma = x0sigma;
    }
}

void Algorithm::update_trajectories() {
    new_traj_distr_.clear();
    new_traj_distr_.reserve(static_cast<std::size_t>(M_));

    for (int m = 0; m < M_; ++m) {
        TrajOptResult result = traj_opt_->update(m, *this);
        new_traj_distr_.push_back(std::move(result.traj_distr));
        cur_[static_cast<std::size_t>(m)].eta = result.eta;
    }
}

void Algorithm::update_trajectories_robust() {
    new_traj_distr_.clear();
    new_traj_distr_adv_.clear();
    new_traj_distr_.reserve(static_cast<std::size_t>(M_));
    new_traj_distr_adv_.reserve(static_cast<std::size_t>(M_));

    for (int m = 0; m < M_; ++m) {
        TrajOptResultRobust result = traj_opt_->update_robust(m, *this);
        new_traj_distr_.push_back(std::move(result.traj_distr));
        new_traj_distr_adv_.push_back(std::move(result.traj_distr_adv));
        cur_[static_cast<std::size_t>(m)].eta = result.eta;
    }
}

void Algorithm::eval_cost(int condition) {
    auto& data = cur_[static_cast<std::size_t>(condition)];
    const int N = static_cast<int>(data.sample_list.size());
    const int dXU = dX_ + dU_;

    // Initialize cost arrays
    Matrix cs = Matrix::Zero(N, T_);
    Matrix cc = Matrix::Zero(N, T_);
    Tensor3d cv(static_cast<std::size_t>(N));
    std::vector<Tensor3d> Cm(static_cast<std::size_t>(N));

    for (int n = 0; n < N; ++n) {
        cv[static_cast<std::size_t>(n)] = Matrix::Zero(T_, dXU);
        Cm[static_cast<std::size_t>(n)].resize(static_cast<std::size_t>(T_));
        for (int t = 0; t < T_; ++t) {
            Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)] =
                Matrix::Zero(dXU, dXU);
        }
    }

    // Evaluate cost for each sample
    for (int n = 0; n < N; ++n) {
        const Sample& sample = data.sample_list[static_cast<std::size_t>(n)];
        CostResult result = costs_[static_cast<std::size_t>(condition)]->eval(sample);

        cs.row(n) = result.l.transpose();
        cc.row(n) = result.l.transpose();

        for (int t = 0; t < T_; ++t) {
            // Assemble [lx, lu] into cv
            cv[static_cast<std::size_t>(n)].row(t).head(dX_) = result.lx.row(t);
            cv[static_cast<std::size_t>(n)].row(t).tail(dU_) = result.lu.row(t);

            // Assemble [[lxx, lux^T], [lux, luu]] into Cm
            auto& Cm_nt = Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)];
            Cm_nt.topLeftCorner(dX_, dX_) = result.lxx[static_cast<std::size_t>(t)];
            Cm_nt.bottomRightCorner(dU_, dU_) = result.luu[static_cast<std::size_t>(t)];
            Cm_nt.bottomLeftCorner(dU_, dX_) = result.lux[static_cast<std::size_t>(t)];
            Cm_nt.topRightCorner(dX_, dU_) = result.lux[static_cast<std::size_t>(t)].transpose();
        }

        // Adjust for expansion around sample
        Matrix X = sample.get_X();
        Matrix U = sample.get_U();

        for (int t = 0; t < T_; ++t) {
            Vector yhat(dXU);
            yhat.head(dX_) = X.row(t).transpose();
            yhat.tail(dU_) = U.row(t).transpose();

            Vector rdiff = -yhat;
            Vector cv_update = Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)] * rdiff;

            cc(n, t) += rdiff.dot(cv[static_cast<std::size_t>(n)].row(t).transpose());
            cc(n, t) += 0.5 * rdiff.dot(cv_update);
            cv[static_cast<std::size_t>(n)].row(t) += cv_update.transpose();
        }
    }

    // Compute means
    data.traj_info.cc = cc.colwise().mean().transpose();

    data.traj_info.cv = Matrix::Zero(T_, dXU);
    for (int n = 0; n < N; ++n) {
        data.traj_info.cv += cv[static_cast<std::size_t>(n)];
    }
    data.traj_info.cv /= N;

    data.traj_info.Cm.resize(static_cast<std::size_t>(T_));
    for (int t = 0; t < T_; ++t) {
        data.traj_info.Cm[static_cast<std::size_t>(t)] = Matrix::Zero(dXU, dXU);
        for (int n = 0; n < N; ++n) {
            data.traj_info.Cm[static_cast<std::size_t>(t)] +=
                Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)];
        }
        data.traj_info.Cm[static_cast<std::size_t>(t)] /= N;
    }

    data.cs = cs;
}

void Algorithm::eval_cost_idg(int condition) {
    auto& data = cur_[static_cast<std::size_t>(condition)];
    const int N = static_cast<int>(data.sample_list.size());
    const int dXUV = dX_ + dU_ + dV_;

    // Initialize cost arrays for extended state-action space
    Matrix cs = Matrix::Zero(N, T_);
    Matrix cc = Matrix::Zero(N, T_);
    Tensor3d cv(static_cast<std::size_t>(N));
    std::vector<Tensor3d> Cm(static_cast<std::size_t>(N));

    for (int n = 0; n < N; ++n) {
        cv[static_cast<std::size_t>(n)] = Matrix::Zero(T_, dXUV);
        Cm[static_cast<std::size_t>(n)].resize(static_cast<std::size_t>(T_));
        for (int t = 0; t < T_; ++t) {
            Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)] =
                Matrix::Zero(dXUV, dXUV);
        }
    }

    // Evaluate cost for each sample (with adversary terms)
    for (int n = 0; n < N; ++n) {
        const Sample& sample = data.sample_list[static_cast<std::size_t>(n)];
        CostResult result = costs_[static_cast<std::size_t>(condition)]->eval(sample);

        cs.row(n) = result.l.transpose();
        cc.row(n) = result.l.transpose();

        for (int t = 0; t < T_; ++t) {
            // Assemble [lx, lu, lv] into cv
            cv[static_cast<std::size_t>(n)].row(t).head(dX_) = result.lx.row(t);
            cv[static_cast<std::size_t>(n)].row(t).segment(dX_, dU_) = result.lu.row(t);
            cv[static_cast<std::size_t>(n)].row(t).tail(dV_) = result.lv.row(t);

            // Assemble full Hessian into Cm
            auto& Cm_nt = Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)];

            // [lxx, lux^T, lvx^T]
            // [lux, luu,   luv  ]
            // [lvx, luv^T, lvv  ]
            Cm_nt.topLeftCorner(dX_, dX_) = result.lxx[static_cast<std::size_t>(t)];
            Cm_nt.block(dX_, dX_, dU_, dU_) = result.luu[static_cast<std::size_t>(t)];
            Cm_nt.bottomRightCorner(dV_, dV_) = result.lvv[static_cast<std::size_t>(t)];

            Cm_nt.block(dX_, 0, dU_, dX_) = result.lux[static_cast<std::size_t>(t)];
            Cm_nt.block(0, dX_, dX_, dU_) = result.lux[static_cast<std::size_t>(t)].transpose();

            Cm_nt.block(dX_ + dU_, 0, dV_, dX_) = result.lvx[static_cast<std::size_t>(t)];
            Cm_nt.block(0, dX_ + dU_, dX_, dV_) = result.lvx[static_cast<std::size_t>(t)].transpose();

            Cm_nt.block(dX_, dX_ + dU_, dU_, dV_) = result.luv[static_cast<std::size_t>(t)];
            Cm_nt.block(dX_ + dU_, dX_, dV_, dU_) = result.luv[static_cast<std::size_t>(t)].transpose();
        }

        // Adjust for expansion around sample
        Matrix X = sample.get_X();
        Matrix U = sample.get_U();
        Matrix V = sample.get_V();

        for (int t = 0; t < T_; ++t) {
            Vector yhat(dXUV);
            yhat.head(dX_) = X.row(t).transpose();
            yhat.segment(dX_, dU_) = U.row(t).transpose();
            yhat.tail(dV_) = V.row(t).transpose();

            Vector rdiff = -yhat;
            Vector cv_update = Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)] * rdiff;

            cc(n, t) += rdiff.dot(cv[static_cast<std::size_t>(n)].row(t).transpose());
            cc(n, t) += 0.5 * rdiff.dot(cv_update);
            cv[static_cast<std::size_t>(n)].row(t) += cv_update.transpose();
        }
    }

    // Compute means
    data.traj_info.cc = cc.colwise().mean().transpose();

    data.traj_info.cv = Matrix::Zero(T_, dXUV);
    for (int n = 0; n < N; ++n) {
        data.traj_info.cv += cv[static_cast<std::size_t>(n)];
    }
    data.traj_info.cv /= N;

    data.traj_info.Cm.resize(static_cast<std::size_t>(T_));
    for (int t = 0; t < T_; ++t) {
        data.traj_info.Cm[static_cast<std::size_t>(t)] = Matrix::Zero(dXUV, dXUV);
        for (int n = 0; n < N; ++n) {
            data.traj_info.Cm[static_cast<std::size_t>(t)] +=
                Cm[static_cast<std::size_t>(n)][static_cast<std::size_t>(t)];
        }
        data.traj_info.Cm[static_cast<std::size_t>(t)] /= N;
    }

    data.cs = cs;
}

void Algorithm::advance_iteration_variables() {
    iteration_count_++;

    // Move current to previous
    prev_.clear();
    prev_.reserve(static_cast<std::size_t>(M_));
    for (int m = 0; m < M_; ++m) {
        prev_.push_back(cur_[static_cast<std::size_t>(m)].clone());
        prev_.back().new_traj_distr = new_traj_distr_[static_cast<std::size_t>(m)];
    }

    // Reset current with new trajectory distributions
    cur_.clear();
    cur_.resize(static_cast<std::size_t>(M_));

    for (int m = 0; m < M_; ++m) {
        cur_[static_cast<std::size_t>(m)].traj_info = prev_[static_cast<std::size_t>(m)].traj_info.clone();
        cur_[static_cast<std::size_t>(m)].step_mult = prev_[static_cast<std::size_t>(m)].step_mult;
        cur_[static_cast<std::size_t>(m)].eta = prev_[static_cast<std::size_t>(m)].eta;
        cur_[static_cast<std::size_t>(m)].traj_distr = new_traj_distr_[static_cast<std::size_t>(m)];

        if (!new_traj_distr_adv_.empty()) {
            cur_[static_cast<std::size_t>(m)].traj_distr_adv =
                new_traj_distr_adv_[static_cast<std::size_t>(m)];
        }
    }

    new_traj_distr_.clear();
    new_traj_distr_adv_.clear();
}

void Algorithm::set_new_mult(double predicted_impr, double actual_impr, int m) {
    // Model improvement as I = predicted_dI * KL + penalty * KL^2
    // Optimize w.r.t. KL
    double new_mult = predicted_impr / (2.0 * std::max(1e-4,
        predicted_impr - actual_impr));
    new_mult = std::clamp(new_mult, 0.1, 5.0);

    double max_mult = get_hyperparam_or<double>(hyperparams_, "max_step_mult", 10.0);
    double min_mult = get_hyperparam_or<double>(hyperparams_, "min_step_mult", 0.01);

    double new_step = std::clamp(
        new_mult * cur_[static_cast<std::size_t>(m)].step_mult,
        min_mult, max_mult);

    cur_[static_cast<std::size_t>(m)].step_mult = new_step;
}

double Algorithm::measure_ent(int m) const {
    double ent = 0.0;
    const auto& traj = cur_[static_cast<std::size_t>(m)].traj_distr;

    for (int t = 0; t < T_; ++t) {
        for (int i = 0; i < dU_; ++i) {
            ent += std::log(traj.chol_pol_covar()[static_cast<std::size_t>(t)](i, i));
        }
    }

    return ent;
}

}  // namespace gps
