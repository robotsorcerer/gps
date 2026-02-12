/**
 * @file types.h
 * @brief Common type definitions for GPS C++20 implementation.
 *
 * Provides type aliases, constants, and enumerations used throughout
 * the GPS algorithm implementation.
 */

#ifndef GPS_CXX_UTILITY_TYPES_H_
#define GPS_CXX_UTILITY_TYPES_H_

#include <Eigen/Dense>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace gps {

// Matrix and vector type aliases using Eigen
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

// Fixed-size types for common dimensions
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Matrix4d = Eigen::Matrix4d;
using Vector4d = Eigen::Vector4d;

// Array types for time-indexed data (T x dim)
using MatrixArray = std::vector<Matrix>;
using VectorArray = std::vector<Vector>;

// Tensor type for 3D data (T x dim1 x dim2)
using Tensor3d = std::vector<Matrix>;

/**
 * @brief Sample data types enumeration.
 *
 * Maps to the protobuf SampleType enum for sensor data identification.
 */
enum class SampleType : int32_t {
    JOINT_ANGLES = 0,
    JOINT_VELOCITIES = 1,
    END_EFFECTOR_POINTS = 2,
    END_EFFECTOR_POINT_VELOCITIES = 3,
    END_EFFECTOR_POINT_JACOBIANS = 4,
    END_EFFECTOR_POINT_ROT_JACOBIANS = 5,
    END_EFFECTOR_POSITIONS = 6,
    END_EFFECTOR_ROTATIONS = 7,
    END_EFFECTOR_JACOBIANS = 8,
    END_EFFECTOR_HESSIANS = 9,
    ACTION = 10,
    TRIAL_ARM = 11,
    AUXILIARY_ARM = 12,
    JOINT_SPACE = 13,
    TASK_SPACE = 14,
    NOISE = 15,
    CONTEXT_IMAGE = 16,
    RGB_IMAGE = 17,
    DEPTH_IMAGE = 18,
    IMAGE_FEATURE = 19,
    OBJECT_POSITION = 20,
    OBJECT_ORIENTATION = 21,
    POSITION_NEAREST_OBSTACLE = 22,
    // Add more as needed from gps_pb2
};

/**
 * @brief Actuator type enumeration.
 */
enum class ActuatorType : int32_t {
    TRIAL_ARM = 0,
    AUXILIARY_ARM = 1,
};

/**
 * @brief Policy mode for game-theoretic (iDG) algorithms.
 */
enum class PolicyMode {
    PROTAGONIST,   // Maximizes expected reward
    ANTAGONIST,    // Minimizes expected reward (adversarial)
    ROBUST,        // Combined robust policy
};

/**
 * @brief Cost mode for multi-objective optimization.
 */
enum class CostMode {
    PROTAGONIST,
    ANTAGONIST,
    ROBUST,
};

/**
 * @brief Hyperparameter variant type for flexible configuration.
 */
using HyperparamValue = std::variant<
    int,
    double,
    bool,
    std::string,
    Vector,
    Matrix,
    std::vector<int>,
    std::vector<double>,
    std::vector<std::string>
>;

using Hyperparams = std::map<std::string, HyperparamValue>;

/**
 * @brief Get value from hyperparameters with type checking.
 *
 * @tparam T Expected type
 * @param params Hyperparameters map
 * @param key Parameter key
 * @return const T& Reference to the value
 * @throws std::bad_variant_access if type mismatch
 * @throws std::out_of_range if key not found
 */
template <typename T>
[[nodiscard]] const T& get_hyperparam(const Hyperparams& params,
                                       const std::string& key) {
    return std::get<T>(params.at(key));
}

/**
 * @brief Get value from hyperparameters with default.
 *
 * @tparam T Expected type
 * @param params Hyperparameters map
 * @param key Parameter key
 * @param default_value Value to return if key not found
 * @return T The value or default
 */
template <typename T>
[[nodiscard]] T get_hyperparam_or(const Hyperparams& params,
                                   const std::string& key,
                                   T default_value) {
    auto it = params.find(key);
    if (it == params.end()) {
        return default_value;
    }
    return std::get<T>(it->second);
}

}  // namespace gps

#endif  // GPS_CXX_UTILITY_TYPES_H_
