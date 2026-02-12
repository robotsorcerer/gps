/**
 * @file options.h
 * @brief Type-safe parameter passing for GPS controllers using C++20 features.
 *
 * The options object provides a map from strings to heterogeneous parameters
 * using std::variant. C++20 concepts provide compile-time type constraints.
 */
#pragma once

// C++20 standard library headers
#include <concepts>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Eigen for linear algebra types
#include <Eigen/Dense>

namespace gps_control
{

// ---------------------------------------------------------------------------
// C++20 Concepts for type constraints
// ---------------------------------------------------------------------------

/**
 * Concept for types that can be stored in OptionsVariant.
 */
template <typename T>
concept OptionValue = std::same_as<T, bool> ||
                      std::same_as<T, uint8_t> ||
                      std::same_as<T, std::vector<int>> ||
                      std::same_as<T, int> ||
                      std::same_as<T, double> ||
                      std::same_as<T, Eigen::MatrixXd> ||
                      std::same_as<T, Eigen::VectorXd> ||
                      std::same_as<T, std::string>;

/**
 * Concept for numeric types in the variant.
 */
template <typename T>
concept NumericOption = std::same_as<T, int> ||
                        std::same_as<T, double> ||
                        std::same_as<T, uint8_t>;

/**
 * Concept for Eigen types in the variant.
 */
template <typename T>
concept EigenOption = std::same_as<T, Eigen::MatrixXd> ||
                      std::same_as<T, Eigen::VectorXd>;

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

/**
 * Enum for runtime type identification (legacy compatibility).
 */
enum class OptionsDataFormat : uint8_t
{
    Bool,
    UInt8,
    IntVector,
    Int,
    Double,
    Matrix,
    Vector,
    String
};

/**
 * The parameter variant type.
 *
 * Note: argument order must match OptionsDataFormat enum for index-based access.
 */
using OptionsVariant = std::variant<
    bool,
    uint8_t,
    std::vector<int>,
    int,
    double,
    Eigen::MatrixXd,
    Eigen::VectorXd,
    std::string
>;

/**
 * The options map type - maps string keys to variant values.
 */
using OptionsMap = std::map<std::string, OptionsVariant, std::less<>>;

// ---------------------------------------------------------------------------
// C++20 Helper functions for type-safe access
// ---------------------------------------------------------------------------

/**
 * Get a value from the options map with type safety.
 *
 * @tparam T The expected type (must satisfy OptionValue concept)
 * @param opts The options map
 * @param key The key to look up
 * @return std::optional<T> containing the value if found and type matches
 */
template <OptionValue T>
[[nodiscard]] constexpr std::optional<T> get_option(
    const OptionsMap& opts,
    std::string_view key) noexcept
{
    if (auto it = opts.find(key); it != opts.end()) {
        if (auto* ptr = std::get_if<T>(&it->second)) {
            return *ptr;
        }
    }
    return std::nullopt;
}

/**
 * Get a value from the options map with a default fallback.
 *
 * @tparam T The expected type (must satisfy OptionValue concept)
 * @param opts The options map
 * @param key The key to look up
 * @param default_value Value to return if key not found or type mismatch
 * @return The value or default
 */
template <OptionValue T>
[[nodiscard]] constexpr T get_option_or(
    const OptionsMap& opts,
    std::string_view key,
    T default_value) noexcept
{
    return get_option<T>(opts, key).value_or(std::move(default_value));
}

/**
 * Check if an option exists and has the expected type.
 *
 * @tparam T The expected type
 * @param opts The options map
 * @param key The key to check
 * @return true if key exists with matching type
 */
template <OptionValue T>
[[nodiscard]] constexpr bool has_option(
    const OptionsMap& opts,
    std::string_view key) noexcept
{
    if (auto it = opts.find(key); it != opts.end()) {
        return std::holds_alternative<T>(it->second);
    }
    return false;
}

/**
 * Get the runtime type format of a variant value.
 *
 * @param v The variant value
 * @return The corresponding OptionsDataFormat enum
 */
[[nodiscard]] constexpr OptionsDataFormat get_format(const OptionsVariant& v) noexcept
{
    return static_cast<OptionsDataFormat>(v.index());
}

}  // namespace gps_control
