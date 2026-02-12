/**
 * @file util.h
 * @brief Utility functions for string manipulation and type conversion.
 *
 * C++20 modernization includes:
 * - std::string_view for non-owning string parameters
 * - std::span for generic array views
 * - Concepts for type constraints
 * - [[nodiscard]] attributes for return values
 * - constexpr where possible
 */
#pragma once

#include <charconv>
#include <concepts>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace util
{

// ---------------------------------------------------------------------------
// C++20 Concepts
// ---------------------------------------------------------------------------

/**
 * Concept for types that can be converted to string via ostringstream.
 */
template <typename T>
concept Stringifiable = requires(std::ostream& os, T value) {
    { os << value } -> std::same_as<std::ostream&>;
};

/**
 * Concept for arithmetic types that support std::to_chars.
 */
template <typename T>
concept ToCharsConvertible = std::integral<T> || std::floating_point<T>;

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

/**
 * Split a string by delimiter into a vector of substrings.
 *
 * @param s The input string to split
 * @param delim The delimiter character
 * @param elems Output vector to store the split elements
 */
void split(const std::string& s, char delim, std::vector<std::string>& elems);

/**
 * Split a string_view by delimiter (C++20 version).
 *
 * @param s The input string_view to split
 * @param delim The delimiter character
 * @return Vector of string_views pointing into the original string
 *
 * @note The returned string_views are only valid while the source string exists.
 */
[[nodiscard]] std::vector<std::string_view> split_view(
    std::string_view s,
    char delim) noexcept;

/**
 * Split a string and return the result (C++20 version).
 *
 * @param s The input string_view to split
 * @param delim The delimiter character
 * @return Vector of strings
 */
[[nodiscard]] std::vector<std::string> split(
    std::string_view s,
    char delim);

}  // namespace util

// ---------------------------------------------------------------------------
// Template implementations (must be in header)
// ---------------------------------------------------------------------------

/**
 * Convert a value to string using the most efficient method available.
 *
 * For arithmetic types in C++20, uses std::to_chars when available.
 * Falls back to ostringstream for other types.
 *
 * @tparam T The type to convert (must be Stringifiable)
 * @param value The value to convert
 * @return The string representation
 */
template <util::Stringifiable T>
[[nodiscard]] std::string to_string(T value)
{
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    }
    else if constexpr (std::is_same_v<T, std::string_view>) {
        return std::string(value);
    }
    else if constexpr (std::is_same_v<T, const char*>) {
        return std::string(value);
    }
    else if constexpr (std::integral<T> || std::floating_point<T>) {
        // Use std::to_string for numeric types (faster than ostringstream)
        return std::to_string(value);
    }
    else {
        // Fallback to ostringstream for other types
        std::ostringstream os;
        os << value;
        return os.str();
    }
}

/**
 * Convert a value to string (overload for std::string_view).
 *
 * @param sv The string_view to convert
 * @return A string copy of the string_view
 */
[[nodiscard]] inline std::string to_string(std::string_view sv)
{
    return std::string(sv);
}
