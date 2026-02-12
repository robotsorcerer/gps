/**
 * @file sample_list.cpp
 * @brief Implementation of SampleList class.
 */

#include "sample_list.h"

#include <stdexcept>

namespace gps {

void SampleList::add(Sample sample) {
    // Validate consistency with existing samples
    if (!samples_.empty()) {
        if (sample.T() != samples_[0].T() ||
            sample.dX() != samples_[0].dX() ||
            sample.dU() != samples_[0].dU()) {
            throw std::invalid_argument(
                "Sample dimensions must match existing samples in list");
        }
    }
    samples_.push_back(std::move(sample));
}

Tensor3d SampleList::get_X() const {
    if (samples_.empty()) {
        return {};
    }

    Tensor3d result;
    result.reserve(samples_.size());

    for (const auto& sample : samples_) {
        result.push_back(sample.get_X());
    }

    return result;
}

Tensor3d SampleList::get_U() const {
    if (samples_.empty()) {
        return {};
    }

    Tensor3d result;
    result.reserve(samples_.size());

    for (const auto& sample : samples_) {
        result.push_back(sample.get_U());
    }

    return result;
}

Tensor3d SampleList::get_V() const {
    if (samples_.empty()) {
        return {};
    }

    Tensor3d result;
    result.reserve(samples_.size());

    for (const auto& sample : samples_) {
        result.push_back(sample.get_V());
    }

    return result;
}

Tensor3d SampleList::get_obs() const {
    if (samples_.empty()) {
        return {};
    }

    Tensor3d result;
    result.reserve(samples_.size());

    for (const auto& sample : samples_) {
        result.push_back(sample.get_obs());
    }

    return result;
}

int SampleList::T() const {
    if (samples_.empty()) {
        throw std::runtime_error("Cannot get T from empty SampleList");
    }
    return samples_[0].T();
}

int SampleList::dX() const {
    if (samples_.empty()) {
        throw std::runtime_error("Cannot get dX from empty SampleList");
    }
    return samples_[0].dX();
}

int SampleList::dU() const {
    if (samples_.empty()) {
        throw std::runtime_error("Cannot get dU from empty SampleList");
    }
    return samples_[0].dU();
}

int SampleList::dV() const {
    if (samples_.empty()) {
        throw std::runtime_error("Cannot get dV from empty SampleList");
    }
    return samples_[0].dV();
}

}  // namespace gps
