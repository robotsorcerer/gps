/**
 * @file util.cpp
 * @brief Implementation of utility functions.
 *
 * C++20 modernization:
 * - Range-based algorithms
 * - std::string_view support
 * - [[nodiscard]] compliance
 */

#include "gps_agent_pkg/util.h"

#include <algorithm>
#include <ranges>
#include <sstream>

namespace util
{

void split(const std::string& s, char delim, std::vector<std::string>& elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(std::move(item));
    }
}

std::vector<std::string_view> split_view(std::string_view s, char delim) noexcept
{
    std::vector<std::string_view> result;
    result.reserve(8);  // Pre-allocate for common cases

    std::size_t start = 0;
    std::size_t end = s.find(delim);

    while (end != std::string_view::npos) {
        result.emplace_back(s.substr(start, end - start));
        start = end + 1;
        end = s.find(delim, start);
    }

    // Add the last segment
    result.emplace_back(s.substr(start));

    return result;
}

std::vector<std::string> split(std::string_view s, char delim)
{
    std::vector<std::string> result;
    result.reserve(8);  // Pre-allocate for common cases

    std::size_t start = 0;
    std::size_t end = s.find(delim);

    while (end != std::string_view::npos) {
        result.emplace_back(s.substr(start, end - start));
        start = end + 1;
        end = s.find(delim, start);
    }

    // Add the last segment
    result.emplace_back(s.substr(start));

    return result;
}

}  // namespace util
