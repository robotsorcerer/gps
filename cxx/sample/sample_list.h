/**
 * @file sample_list.h
 * @brief Collection of trajectory samples.
 *
 * C++20 port of python/gps/sample/sample_list.py
 */

#ifndef GPS_CXX_SAMPLE_SAMPLE_LIST_H_
#define GPS_CXX_SAMPLE_SAMPLE_LIST_H_

#include <memory>
#include <vector>

#include "sample.h"
#include "../utility/types.h"

namespace gps {

/**
 * @brief Collection of trajectory samples.
 *
 * Provides batch access to sample data across multiple trajectories.
 */
class SampleList {
public:
    SampleList() = default;
    ~SampleList() = default;

    SampleList(const SampleList&) = default;
    SampleList(SampleList&&) = default;
    SampleList& operator=(const SampleList&) = default;
    SampleList& operator=(SampleList&&) = default;

    /**
     * @brief Add a sample to the list.
     *
     * @param sample Sample to add (moved)
     */
    void add(Sample sample);

    /**
     * @brief Get number of samples.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return samples_.size();
    }

    /**
     * @brief Check if empty.
     */
    [[nodiscard]] bool empty() const noexcept {
        return samples_.empty();
    }

    /**
     * @brief Access sample by index.
     */
    [[nodiscard]] const Sample& operator[](std::size_t idx) const {
        return samples_.at(idx);
    }

    /**
     * @brief Access sample by index (mutable).
     */
    Sample& operator[](std::size_t idx) {
        return samples_.at(idx);
    }

    /**
     * @brief Get all states as 3D tensor (N x T x dX).
     *
     * @return Tensor3d States from all samples
     */
    [[nodiscard]] Tensor3d get_X() const;

    /**
     * @brief Get all actions as 3D tensor (N x T x dU).
     *
     * @return Tensor3d Actions from all samples
     */
    [[nodiscard]] Tensor3d get_U() const;

    /**
     * @brief Get all adversary actions as 3D tensor (N x T x dV).
     *
     * @return Tensor3d Adversary actions from all samples
     */
    [[nodiscard]] Tensor3d get_V() const;

    /**
     * @brief Get all observations as 3D tensor (N x T x dO).
     *
     * @return Tensor3d Observations from all samples
     */
    [[nodiscard]] Tensor3d get_obs() const;

    /**
     * @brief Get time horizon (from first sample).
     */
    [[nodiscard]] int T() const;

    /**
     * @brief Get state dimension (from first sample).
     */
    [[nodiscard]] int dX() const;

    /**
     * @brief Get action dimension (from first sample).
     */
    [[nodiscard]] int dU() const;

    /**
     * @brief Get adversary dimension (from first sample).
     */
    [[nodiscard]] int dV() const;

    // Iterators
    auto begin() { return samples_.begin(); }
    auto end() { return samples_.end(); }
    auto begin() const { return samples_.begin(); }
    auto end() const { return samples_.end(); }

    /**
     * @brief Clear all samples.
     */
    void clear() { samples_.clear(); }

private:
    std::vector<Sample> samples_;
};

}  // namespace gps

#endif  // GPS_CXX_SAMPLE_SAMPLE_LIST_H_
