cmake_minimum_required(VERSION 3.16)
project(gps_algorithm_cxx VERSION 1.0.0 LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(GPS_BUILD_TESTS "Build unit tests" ON)
option(GPS_USE_LIBTORCH "Enable LibTorch for neural network policies" OFF)
option(GPS_BUILD_SHARED "Build shared library" ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter  # Many callbacks have unused params
        # Relaxed for Eigen compatibility
    )
endif()

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Optional: Find LibTorch
if(GPS_USE_LIBTORCH)
    find_package(Torch REQUIRED)
    add_compile_definitions(GPS_HAS_LIBTORCH)
endif()

# Collect source files
set(GPS_SOURCES
    # Utility
    # (types.h is header-only)

    # Sample
    sample/sample.cpp
    sample/sample_list.cpp

    # Algorithm - Cost
    algorithm/cost/cost_action.cpp
    algorithm/cost/cost_state.cpp
    algorithm/cost/cost_sum.cpp

    # Algorithm - Dynamics
    algorithm/dynamics/dynamics.cpp

    # Algorithm - Policy
    algorithm/policy/lin_gauss_policy.cpp

    # Algorithm - Trajectory Optimization
    algorithm/traj_opt/traj_opt.cpp
    algorithm/traj_opt/traj_opt_lqr.cpp

    # Algorithm - Core
    algorithm/algorithm_utils.cpp
    algorithm/algorithm.cpp
)

# Header files for installation
set(GPS_HEADERS
    utility/types.h

    sample/sample.h
    sample/sample_list.h

    algorithm/cost/cost.h
    algorithm/cost/cost_utils.h
    algorithm/cost/cost_action.h
    algorithm/cost/cost_state.h
    algorithm/cost/cost_sum.h

    algorithm/dynamics/dynamics.h

    algorithm/policy/policy.h
    algorithm/policy/lin_gauss_policy.h

    algorithm/traj_opt/traj_opt.h
    algorithm/traj_opt/traj_opt_lqr.h

    algorithm/algorithm_utils.h
    algorithm/algorithm.h
)

# Create library
if(GPS_BUILD_SHARED)
    add_library(gps_algorithm SHARED ${GPS_SOURCES})
else()
    add_library(gps_algorithm STATIC ${GPS_SOURCES})
endif()

target_include_directories(gps_algorithm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/gps>
)

target_link_libraries(gps_algorithm
    PUBLIC
        Eigen3::Eigen
)

if(GPS_USE_LIBTORCH)
    target_link_libraries(gps_algorithm
        PUBLIC
            "${TORCH_LIBRARIES}"
    )
endif()

# Set library properties
set_target_properties(gps_algorithm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Build tests
if(GPS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS gps_algorithm
    EXPORT gps_algorithmTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers preserving directory structure
foreach(header ${GPS_HEADERS})
    get_filename_component(dir ${header} DIRECTORY)
    install(FILES ${header}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gps/${dir}
    )
endforeach()

# Export targets
install(EXPORT gps_algorithmTargets
    FILE gps_algorithmTargets.cmake
    NAMESPACE gps::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gps_algorithm
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gps_algorithmConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(cmake/gps_algorithmConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/gps_algorithmConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gps_algorithmConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gps_algorithmConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gps_algorithm
)
