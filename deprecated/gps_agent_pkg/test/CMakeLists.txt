# ---------------------------------------------------------------------------
# Standalone unit-test build for GPS C++ tests.
#
# This CMakeLists.txt intentionally requires NO ROS and NO LibTorch so that:
#   1. Tests can be built and run on any machine with CMake + Eigen + GTest.
#   2. The ThreadSanitizer CI job can compile and run all tests without
#      pulling in LibTorch (which is incompatible with TSan).
#
# Usage:
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_CXX_STANDARD=17          # normal build
#   cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" \
#            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
#            -DCMAKE_CXX_COMPILER=clang++-12  # TSan build
#   make -j$(nproc) && ctest -V
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(gps_agent_pkg_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Inherit sanitizer / optimization flags from the caller if set.
# No additional flags forced here — let the CI job control them.

# ---------------------------------------------------------------------------
# Dependencies: GTest + Eigen (no ROS, no LibTorch)
# ---------------------------------------------------------------------------

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

enable_testing()

# ---------------------------------------------------------------------------
# test_options_variant — tests OptionsVariant / OptionsMap (options.h)
# Requires: Eigen3, GTest, C++17
# ---------------------------------------------------------------------------
add_executable(test_options_variant test_options_variant.cpp)
target_include_directories(test_options_variant PRIVATE
    ../include          # gps_agent_pkg/include/  (for options.h)
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(test_options_variant
    ${GTEST_LIBRARIES}
    pthread
)
add_test(NAME test_options_variant COMMAND test_options_variant)

# ---------------------------------------------------------------------------
# test_util — tests util::split() and to_string<T>() (util.h / util.cpp)
# Requires: GTest, C++ stdlib only
# ---------------------------------------------------------------------------
add_executable(test_util
    test_util.cpp
    ../src/util.cpp
)
target_include_directories(test_util PRIVATE
    ../include
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(test_util
    ${GTEST_LIBRARIES}
    pthread
)
add_test(NAME test_util COMMAND test_util)

# ---------------------------------------------------------------------------
# test_obs_normalization — tests the PyTorchController obs normalisation formula
# Requires: Eigen3, GTest — no ROS, no LibTorch, no GPS headers
# ---------------------------------------------------------------------------
add_executable(test_obs_normalization test_obs_normalization.cpp)
target_include_directories(test_obs_normalization PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(test_obs_normalization
    ${GTEST_LIBRARIES}
    pthread
)
add_test(NAME test_obs_normalization COMMAND test_obs_normalization)
