/**
 * @file sample.h
 * @brief Sample class for storing trajectory rollout data.
 *
 * C++20 port of python/gps/sample/sample.py
 * Uses Eigen matrices for efficient numerical operations.
 */

#ifndef GPS_CXX_SAMPLE_SAMPLE_H_
#define GPS_CXX_SAMPLE_SAMPLE_H_

#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "../utility/types.h"

namespace gps {

/**
 * @brief Sample data from a single trajectory rollout.
 *
 * Stores time-indexed sensor data, actions, and observations
 * from a single trial. Provides methods to access state (X),
 * action (U), adversary action (V), and observation vectors.
 */
class Sample {
public:
    /**
     * @brief Construct a new Sample.
     *
     * @param T Number of timesteps
     * @param dX State dimensionality
     * @param dU Action dimensionality
     * @param dV Adversary action dimensionality (default 0)
     */
    Sample(int T, int dX, int dU, int dV = 0);

    /**
     * @brief Construct from hyperparameters.
     *
     * @param hyperparams Configuration containing T, dX, dU, dV, etc.
     */
    explicit Sample(const Hyperparams& hyperparams);

    // Default copy/move operations
    Sample(const Sample&) = default;
    Sample(Sample&&) noexcept = default;
    Sample& operator=(const Sample&) = default;
    Sample& operator=(Sample&&) noexcept = default;
    ~Sample() = default;

    /**
     * @brief Set data for a sensor type at specific timestep.
     *
     * @param sensor_type Type of sensor data
     * @param data Data vector for single timestep
     * @param t Timestep index
     */
    void set(SampleType sensor_type, const Vector& data, int t);

    /**
     * @brief Set data for a sensor type across all timesteps.
     *
     * @param sensor_type Type of sensor data
     * @param data Matrix of shape (T, dim)
     */
    void set(SampleType sensor_type, const Matrix& data);

    /**
     * @brief Get data for a sensor type at specific timestep.
     *
     * @param sensor_type Type of sensor data
     * @param t Timestep index
     * @return Vector Data at timestep t
     * @throws std::out_of_range if sensor_type not found or t invalid
     */
    [[nodiscard]] Vector get(SampleType sensor_type, int t) const;

    /**
     * @brief Get all data for a sensor type.
     *
     * @param sensor_type Type of sensor data
     * @return const Matrix& Data matrix of shape (T, dim)
     * @throws std::out_of_range if sensor_type not found
     */
    [[nodiscard]] const Matrix& get(SampleType sensor_type) const;

    /**
     * @brief Check if sample contains data for a sensor type.
     *
     * @param sensor_type Type to check
     * @return true if data exists
     */
    [[nodiscard]] bool has(SampleType sensor_type) const noexcept;

    /**
     * @brief Get state vector at timestep.
     *
     * Concatenates configured state data types.
     *
     * @param t Timestep index
     * @return Vector State vector of dimension dX
     */
    [[nodiscard]] Vector get_X(int t) const;

    /**
     * @brief Get state matrix for all timesteps.
     *
     * @return Matrix State matrix of shape (T, dX)
     */
    [[nodiscard]] Matrix get_X() const;

    /**
     * @brief Get action at timestep.
     *
     * @param t Timestep index
     * @return Vector Action vector of dimension dU
     */
    [[nodiscard]] Vector get_U(int t) const;

    /**
     * @brief Get action matrix for all timesteps.
     *
     * @return Matrix Action matrix of shape (T, dU)
     */
    [[nodiscard]] Matrix get_U() const;

    /**
     * @brief Get adversary action at timestep.
     *
     * @param t Timestep index
     * @return Vector Adversary action of dimension dV
     */
    [[nodiscard]] Vector get_V(int t) const;

    /**
     * @brief Get adversary action matrix for all timesteps.
     *
     * @return Matrix Adversary action matrix of shape (T, dV)
     */
    [[nodiscard]] Matrix get_V() const;

    /**
     * @brief Get observation vector at timestep.
     *
     * Concatenates configured observation data types.
     *
     * @param t Timestep index
     * @return Vector Observation vector
     */
    [[nodiscard]] Vector get_obs(int t) const;

    /**
     * @brief Get observation matrix for all timesteps.
     *
     * @return Matrix Observation matrix of shape (T, dO)
     */
    [[nodiscard]] Matrix get_obs() const;

    /**
     * @brief Set state data types for get_X().
     *
     * @param types Vector of SampleType to concatenate for state
     */
    void set_state_types(std::vector<SampleType> types);

    /**
     * @brief Set observation data types for get_obs().
     *
     * @param types Vector of SampleType to concatenate for observation
     */
    void set_obs_types(std::vector<SampleType> types);

    // Accessors
    [[nodiscard]] int T() const noexcept { return T_; }
    [[nodiscard]] int dX() const noexcept { return dX_; }
    [[nodiscard]] int dU() const noexcept { return dU_; }
    [[nodiscard]] int dV() const noexcept { return dV_; }
    [[nodiscard]] int dO() const noexcept { return dO_; }

private:
    int T_;   // Number of timesteps
    int dX_;  // State dimensionality
    int dU_;  // Action dimensionality
    int dV_;  // Adversary action dimensionality
    int dO_;  // Observation dimensionality (computed)

    // Sensor data storage: SampleType -> (T x dim) matrix
    std::map<SampleType, Matrix> data_;

    // Data types for state and observation composition
    std::vector<SampleType> state_types_;
    std::vector<SampleType> obs_types_;

    /**
     * @brief Concatenate data from multiple sensor types.
     *
     * @param types Sensor types to concatenate
     * @param t Timestep index
     * @return Vector Concatenated data
     */
    [[nodiscard]] Vector concatenate_types(
        const std::vector<SampleType>& types, int t) const;

    /**
     * @brief Concatenate data from multiple sensor types for all timesteps.
     *
     * @param types Sensor types to concatenate
     * @return Matrix Concatenated data of shape (T, total_dim)
     */
    [[nodiscard]] Matrix concatenate_types(
        const std::vector<SampleType>& types) const;
};

}  // namespace gps

#endif  // GPS_CXX_SAMPLE_SAMPLE_H_
