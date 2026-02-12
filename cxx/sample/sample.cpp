/**
 * @file sample.cpp
 * @brief Implementation of Sample class.
 */

#include "sample.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace gps {

Sample::Sample(int T, int dX, int dU, int dV)
    : T_(T), dX_(dX), dU_(dU), dV_(dV), dO_(0) {
    if (T <= 0 || dX <= 0 || dU <= 0) {
        throw std::invalid_argument(
            "Sample dimensions must be positive: T=" + std::to_string(T) +
            ", dX=" + std::to_string(dX) + ", dU=" + std::to_string(dU));
    }
    if (dV < 0) {
        throw std::invalid_argument(
            "Adversary dimension dV cannot be negative");
    }
}

Sample::Sample(const Hyperparams& hyperparams)
    : Sample(
          get_hyperparam<int>(hyperparams, "T"),
          get_hyperparam<int>(hyperparams, "dX"),
          get_hyperparam<int>(hyperparams, "dU"),
          get_hyperparam_or<int>(hyperparams, "dV", 0)) {
}

void Sample::set(SampleType sensor_type, const Vector& data, int t) {
    if (t < 0 || t >= T_) {
        throw std::out_of_range(
            "Timestep " + std::to_string(t) + " out of range [0, " +
            std::to_string(T_) + ")");
    }

    auto it = data_.find(sensor_type);
    if (it == data_.end()) {
        // Create new matrix for this sensor type
        Matrix mat = Matrix::Zero(T_, data.size());
        mat.row(t) = data.transpose();
        data_[sensor_type] = std::move(mat);
    } else {
        // Verify dimension match
        if (it->second.cols() != data.size()) {
            throw std::invalid_argument(
                "Data dimension mismatch: expected " +
                std::to_string(it->second.cols()) + ", got " +
                std::to_string(data.size()));
        }
        it->second.row(t) = data.transpose();
    }
}

void Sample::set(SampleType sensor_type, const Matrix& data) {
    if (data.rows() != T_) {
        throw std::invalid_argument(
            "Data row count must match T: expected " + std::to_string(T_) +
            ", got " + std::to_string(data.rows()));
    }
    data_[sensor_type] = data;
}

Vector Sample::get(SampleType sensor_type, int t) const {
    if (t < 0 || t >= T_) {
        throw std::out_of_range(
            "Timestep " + std::to_string(t) + " out of range [0, " +
            std::to_string(T_) + ")");
    }

    auto it = data_.find(sensor_type);
    if (it == data_.end()) {
        throw std::out_of_range("Sensor type not found in sample");
    }

    return it->second.row(t).transpose();
}

const Matrix& Sample::get(SampleType sensor_type) const {
    auto it = data_.find(sensor_type);
    if (it == data_.end()) {
        throw std::out_of_range("Sensor type not found in sample");
    }
    return it->second;
}

bool Sample::has(SampleType sensor_type) const noexcept {
    return data_.find(sensor_type) != data_.end();
}

Vector Sample::get_X(int t) const {
    if (state_types_.empty()) {
        throw std::runtime_error(
            "State types not configured. Call set_state_types() first.");
    }
    return concatenate_types(state_types_, t);
}

Matrix Sample::get_X() const {
    if (state_types_.empty()) {
        throw std::runtime_error(
            "State types not configured. Call set_state_types() first.");
    }
    return concatenate_types(state_types_);
}

Vector Sample::get_U(int t) const {
    return get(SampleType::ACTION, t);
}

Matrix Sample::get_U() const {
    return get(SampleType::ACTION);
}

Vector Sample::get_V(int t) const {
    if (dV_ == 0) {
        return Vector::Zero(0);
    }
    return get(SampleType::NOISE, t);
}

Matrix Sample::get_V() const {
    if (dV_ == 0) {
        return Matrix::Zero(T_, 0);
    }
    return get(SampleType::NOISE);
}

Vector Sample::get_obs(int t) const {
    if (obs_types_.empty()) {
        throw std::runtime_error(
            "Observation types not configured. Call set_obs_types() first.");
    }
    return concatenate_types(obs_types_, t);
}

Matrix Sample::get_obs() const {
    if (obs_types_.empty()) {
        throw std::runtime_error(
            "Observation types not configured. Call set_obs_types() first.");
    }
    return concatenate_types(obs_types_);
}

void Sample::set_state_types(std::vector<SampleType> types) {
    state_types_ = std::move(types);
}

void Sample::set_obs_types(std::vector<SampleType> types) {
    obs_types_ = std::move(types);

    // Compute observation dimensionality
    dO_ = 0;
    for (const auto& type : obs_types_) {
        auto it = data_.find(type);
        if (it != data_.end()) {
            dO_ += static_cast<int>(it->second.cols());
        }
    }
}

Vector Sample::concatenate_types(const std::vector<SampleType>& types,
                                  int t) const {
    // Calculate total dimension
    int total_dim = 0;
    for (const auto& type : types) {
        auto it = data_.find(type);
        if (it != data_.end()) {
            total_dim += static_cast<int>(it->second.cols());
        }
    }

    Vector result(total_dim);
    int offset = 0;

    for (const auto& type : types) {
        auto it = data_.find(type);
        if (it != data_.end()) {
            int dim = static_cast<int>(it->second.cols());
            result.segment(offset, dim) = it->second.row(t).transpose();
            offset += dim;
        }
    }

    return result;
}

Matrix Sample::concatenate_types(const std::vector<SampleType>& types) const {
    // Calculate total dimension
    int total_dim = 0;
    for (const auto& type : types) {
        auto it = data_.find(type);
        if (it != data_.end()) {
            total_dim += static_cast<int>(it->second.cols());
        }
    }

    Matrix result(T_, total_dim);
    int offset = 0;

    for (const auto& type : types) {
        auto it = data_.find(type);
        if (it != data_.end()) {
            int dim = static_cast<int>(it->second.cols());
            result.middleCols(offset, dim) = it->second;
            offset += dim;
        }
    }

    return result;
}

}  // namespace gps
