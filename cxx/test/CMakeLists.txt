# GPS Algorithm C++20 Unit Tests

# Find or fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(test_gps_algorithm
    test_sample.cpp
    test_policy.cpp
    test_cost.cpp
    test_traj_opt.cpp
    test_python_parity.cpp
    test_load_soak.cpp
    test_config.cpp
)

target_link_libraries(test_gps_algorithm
    PRIVATE
        gps_algorithm
        GTest::gtest_main
        GTest::gmock
)

# Enable testing
include(GoogleTest)
gtest_discover_tests(test_gps_algorithm)

# ThreadSanitizer build option
if(GPS_TSAN)
    target_compile_options(test_gps_algorithm PRIVATE -fsanitize=thread -g)
    target_link_options(test_gps_algorithm PRIVATE -fsanitize=thread)
endif()

# AddressSanitizer build option
if(GPS_ASAN)
    target_compile_options(test_gps_algorithm PRIVATE -fsanitize=address -g)
    target_link_options(test_gps_algorithm PRIVATE -fsanitize=address)
endif()
