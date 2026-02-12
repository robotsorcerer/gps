/**
 * @file test_options_variant.cpp
 * @brief Unit tests for OptionsVariant and OptionsMap (options.h).
 *
 * OptionsVariant is the central C++20 std::variant type used throughout the
 * GPS C++ controller to pass heterogeneous parameters. It holds one of:
 *   bool, uint8_t, std::vector<int>, int, double,
 *   Eigen::MatrixXd, Eigen::VectorXd, std::string
 *
 * These tests exercise:
 *   - std::holds_alternative<T>() for every supported type
 *   - std::get<T>() for every supported type
 *   - std::bad_variant_access thrown on type mismatch
 *   - Variant rebinding (assignment to a different type)
 *   - OptionsMap insert / retrieve / missing-key
 *   - Eigen matrix / vector round-trip through OptionsMap
 *   - C++20 concepts (OptionValue, NumericOption, EigenOption)
 *   - C++20 helper functions (get_option, get_option_or, has_option)
 *
 * No ROS or LibTorch headers are needed — options.h only depends on
 * the C++20 standard library and Eigen.
 */

#include <gtest/gtest.h>
#include "gps_agent_pkg/options.h"

#include <concepts>
#include <string>
#include <vector>

using namespace gps_control;

// ---------------------------------------------------------------------------
// C++20 Concept Tests
// ---------------------------------------------------------------------------

TEST(OptionsConcepts, OptionValueConceptSatisfied) {
    // Verify that all expected types satisfy the OptionValue concept
    static_assert(OptionValue<bool>);
    static_assert(OptionValue<uint8_t>);
    static_assert(OptionValue<std::vector<int>>);
    static_assert(OptionValue<int>);
    static_assert(OptionValue<double>);
    static_assert(OptionValue<Eigen::MatrixXd>);
    static_assert(OptionValue<Eigen::VectorXd>);
    static_assert(OptionValue<std::string>);

    // Verify that other types do NOT satisfy the concept
    static_assert(!OptionValue<float>);
    static_assert(!OptionValue<long>);
    static_assert(!OptionValue<char*>);
    static_assert(!OptionValue<std::vector<double>>);

    SUCCEED();  // If we reach here, all static_asserts passed
}

TEST(OptionsConcepts, NumericOptionConceptSatisfied) {
    static_assert(NumericOption<int>);
    static_assert(NumericOption<double>);
    static_assert(NumericOption<uint8_t>);

    static_assert(!NumericOption<bool>);
    static_assert(!NumericOption<std::string>);
    static_assert(!NumericOption<Eigen::VectorXd>);

    SUCCEED();
}

TEST(OptionsConcepts, EigenOptionConceptSatisfied) {
    static_assert(EigenOption<Eigen::MatrixXd>);
    static_assert(EigenOption<Eigen::VectorXd>);

    static_assert(!EigenOption<int>);
    static_assert(!EigenOption<double>);
    static_assert(!EigenOption<std::string>);

    SUCCEED();
}

// ---------------------------------------------------------------------------
// std::holds_alternative — one test per variant alternative
// ---------------------------------------------------------------------------

TEST(OptionsVariantHolds, Bool) {
    OptionsVariant v = true;
    EXPECT_TRUE(std::holds_alternative<bool>(v));
    EXPECT_FALSE(std::holds_alternative<int>(v));
}

TEST(OptionsVariantHolds, UInt8) {
    OptionsVariant v = static_cast<uint8_t>(42);
    EXPECT_TRUE(std::holds_alternative<uint8_t>(v));
    EXPECT_FALSE(std::holds_alternative<int>(v));
}

TEST(OptionsVariantHolds, IntVector) {
    std::vector<int> iv = {1, 2, 3};
    OptionsVariant v = iv;
    EXPECT_TRUE(std::holds_alternative<std::vector<int>>(v));
    EXPECT_FALSE(std::holds_alternative<int>(v));
}

TEST(OptionsVariantHolds, Int) {
    OptionsVariant v = 99;
    EXPECT_TRUE(std::holds_alternative<int>(v));
    EXPECT_FALSE(std::holds_alternative<double>(v));
    EXPECT_FALSE(std::holds_alternative<bool>(v));
}

TEST(OptionsVariantHolds, Double) {
    OptionsVariant v = 3.14;
    EXPECT_TRUE(std::holds_alternative<double>(v));
    EXPECT_FALSE(std::holds_alternative<int>(v));
}

TEST(OptionsVariantHolds, EigenMatrixXd) {
    Eigen::MatrixXd m = Eigen::MatrixXd::Identity(2, 2);
    OptionsVariant v = m;
    EXPECT_TRUE(std::holds_alternative<Eigen::MatrixXd>(v));
    EXPECT_FALSE(std::holds_alternative<Eigen::VectorXd>(v));
}

TEST(OptionsVariantHolds, EigenVectorXd) {
    Eigen::VectorXd vec = Eigen::VectorXd::Ones(4);
    OptionsVariant v = vec;
    EXPECT_TRUE(std::holds_alternative<Eigen::VectorXd>(v));
    EXPECT_FALSE(std::holds_alternative<Eigen::MatrixXd>(v));
}

TEST(OptionsVariantHolds, String) {
    OptionsVariant v = std::string("hello");
    EXPECT_TRUE(std::holds_alternative<std::string>(v));
    EXPECT_FALSE(std::holds_alternative<int>(v));
}

// ---------------------------------------------------------------------------
// std::get<T>() — correct-type extraction
// ---------------------------------------------------------------------------

TEST(OptionsVariantGet, Int) {
    OptionsVariant v = 7;
    EXPECT_EQ(std::get<int>(v), 7);
}

TEST(OptionsVariantGet, Double) {
    OptionsVariant v = 2.718;
    EXPECT_DOUBLE_EQ(std::get<double>(v), 2.718);
}

TEST(OptionsVariantGet, Bool) {
    OptionsVariant v = false;
    EXPECT_FALSE(std::get<bool>(v));
}

TEST(OptionsVariantGet, String) {
    OptionsVariant v = std::string("gps");
    EXPECT_EQ(std::get<std::string>(v), "gps");
}

TEST(OptionsVariantGet, UInt8) {
    OptionsVariant v = static_cast<uint8_t>(255);
    EXPECT_EQ(std::get<uint8_t>(v), 255u);
}

TEST(OptionsVariantGet, IntVector) {
    std::vector<int> iv = {10, 20, 30};
    OptionsVariant v = iv;
    auto& got = std::get<std::vector<int>>(v);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], 10);
    EXPECT_EQ(got[1], 20);
    EXPECT_EQ(got[2], 30);
}

TEST(OptionsVariantGet, EigenVector) {
    Eigen::VectorXd vec(3);
    vec << 1.0, 2.0, 3.0;
    OptionsVariant v = vec;
    auto& got = std::get<Eigen::VectorXd>(v);
    EXPECT_DOUBLE_EQ(got(0), 1.0);
    EXPECT_DOUBLE_EQ(got(1), 2.0);
    EXPECT_DOUBLE_EQ(got(2), 3.0);
}

TEST(OptionsVariantGet, EigenMatrix) {
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    OptionsVariant v = m;
    auto& got = std::get<Eigen::MatrixXd>(v);
    EXPECT_EQ(got.rows(), 2);
    EXPECT_EQ(got.cols(), 3);
    EXPECT_DOUBLE_EQ(got(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(got(1, 2), 6.0);
}

// ---------------------------------------------------------------------------
// std::bad_variant_access — wrong-type extraction must throw
// ---------------------------------------------------------------------------

TEST(OptionsVariantBadAccess, IntAsDouble) {
    OptionsVariant v = 42;
    EXPECT_THROW(std::get<double>(v), std::bad_variant_access);
}

TEST(OptionsVariantBadAccess, StringAsInt) {
    OptionsVariant v = std::string("abc");
    EXPECT_THROW(std::get<int>(v), std::bad_variant_access);
}

TEST(OptionsVariantBadAccess, VectorAsMatrix) {
    Eigen::VectorXd vec = Eigen::VectorXd::Ones(3);
    OptionsVariant v = vec;
    EXPECT_THROW(std::get<Eigen::MatrixXd>(v), std::bad_variant_access);
}

TEST(OptionsVariantBadAccess, BoolAsUInt8) {
    OptionsVariant v = true;
    EXPECT_THROW(std::get<uint8_t>(v), std::bad_variant_access);
}

// ---------------------------------------------------------------------------
// Variant rebinding — C++20 assignment changes the active alternative
// ---------------------------------------------------------------------------

TEST(OptionsVariantRebind, IntToDouble) {
    OptionsVariant v = 42;
    EXPECT_TRUE(std::holds_alternative<int>(v));
    v = 3.14;
    EXPECT_TRUE(std::holds_alternative<double>(v));
    EXPECT_FALSE(std::holds_alternative<int>(v));
    EXPECT_DOUBLE_EQ(std::get<double>(v), 3.14);
}

TEST(OptionsVariantRebind, DoubleToString) {
    OptionsVariant v = 1.0;
    v = std::string("changed");
    EXPECT_TRUE(std::holds_alternative<std::string>(v));
    EXPECT_EQ(std::get<std::string>(v), "changed");
}

TEST(OptionsVariantRebind, ScalarToEigenVector) {
    OptionsVariant v = 0;
    Eigen::VectorXd vec(2);
    vec << 9.9, 8.8;
    v = vec;
    EXPECT_TRUE(std::holds_alternative<Eigen::VectorXd>(v));
    EXPECT_DOUBLE_EQ(std::get<Eigen::VectorXd>(v)(0), 9.9);
}

// ---------------------------------------------------------------------------
// OptionsMap — the primary use-pattern throughout the controller codebase
// ---------------------------------------------------------------------------

TEST(OptionsMap, InsertAndRetrieveMultipleTypes) {
    OptionsMap opts;
    opts["lr"]    = 1e-3;
    opts["iters"] = 5000;
    opts["name"]  = std::string("policy");
    opts["flag"]  = true;

    EXPECT_DOUBLE_EQ(std::get<double>(opts.at("lr")), 1e-3);
    EXPECT_EQ(std::get<int>(opts.at("iters")), 5000);
    EXPECT_EQ(std::get<std::string>(opts.at("name")), "policy");
    EXPECT_TRUE(std::get<bool>(opts.at("flag")));
}

TEST(OptionsMap, MissingKeyThrows) {
    OptionsMap opts;
    opts["a"] = 1;
    EXPECT_THROW(opts.at("nonexistent"), std::out_of_range);
}

TEST(OptionsMap, EigenVectorRoundTrip) {
    OptionsMap opts;
    Eigen::VectorXd bias(4);
    bias << 0.1, 0.2, 0.3, 0.4;
    opts["bias"] = bias;

    auto& got = std::get<Eigen::VectorXd>(opts.at("bias"));
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(got(i), bias(i));
}

TEST(OptionsMap, EigenMatrixRoundTrip) {
    OptionsMap opts;
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(3, 3) * 2.0;
    opts["K"] = K;
    auto& got = std::get<Eigen::MatrixXd>(opts.at("K"));
    EXPECT_DOUBLE_EQ(got(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(got(0, 1), 0.0);
}

TEST(OptionsMap, OverwriteEntry) {
    OptionsMap opts;
    opts["x"] = 1;
    EXPECT_EQ(std::get<int>(opts.at("x")), 1);
    opts["x"] = 99;
    EXPECT_EQ(std::get<int>(opts.at("x")), 99);
}

TEST(OptionsMap, SimulatedControllerConfig) {
    // Mirrors the pattern in pytorchcontroller.cpp configure_controller():
    // options["T"], options["scale"], options["bias"], options["torch_version"]
    OptionsMap opts;
    opts["T"]             = 100;
    opts["torch_version"] = std::string("2.1.0");
    opts["dU"]            = 7;

    Eigen::VectorXd scale = Eigen::VectorXd::Ones(14) * 0.5;
    Eigen::VectorXd bias  = Eigen::VectorXd::Zero(14);
    opts["scale"] = scale;
    opts["bias"]  = bias;

    EXPECT_EQ(std::get<int>(opts.at("T")), 100);
    EXPECT_EQ(std::get<int>(opts.at("dU")), 7);
    EXPECT_EQ(std::get<std::string>(opts.at("torch_version")), "2.1.0");
    EXPECT_DOUBLE_EQ(std::get<Eigen::VectorXd>(opts.at("scale"))(0), 0.5);
    EXPECT_DOUBLE_EQ(std::get<Eigen::VectorXd>(opts.at("bias"))(0),  0.0);
}

// ---------------------------------------------------------------------------
// C++20 Helper Functions Tests
// ---------------------------------------------------------------------------

TEST(OptionsHelpers, GetOptionReturnsValueWhenPresent) {
    OptionsMap opts;
    opts["count"] = 42;
    opts["name"] = std::string("test");

    auto count = get_option<int>(opts, "count");
    ASSERT_TRUE(count.has_value());
    EXPECT_EQ(*count, 42);

    auto name = get_option<std::string>(opts, "name");
    ASSERT_TRUE(name.has_value());
    EXPECT_EQ(*name, "test");
}

TEST(OptionsHelpers, GetOptionReturnsNulloptWhenMissing) {
    OptionsMap opts;
    opts["count"] = 42;

    auto missing = get_option<int>(opts, "missing");
    EXPECT_FALSE(missing.has_value());
}

TEST(OptionsHelpers, GetOptionReturnsNulloptOnTypeMismatch) {
    OptionsMap opts;
    opts["count"] = 42;  // stored as int

    auto as_double = get_option<double>(opts, "count");
    EXPECT_FALSE(as_double.has_value());  // type mismatch
}

TEST(OptionsHelpers, GetOptionOrReturnsValueWhenPresent) {
    OptionsMap opts;
    opts["count"] = 42;

    int result = get_option_or<int>(opts, "count", 0);
    EXPECT_EQ(result, 42);
}

TEST(OptionsHelpers, GetOptionOrReturnsDefaultWhenMissing) {
    OptionsMap opts;

    int result = get_option_or<int>(opts, "missing", 99);
    EXPECT_EQ(result, 99);
}

TEST(OptionsHelpers, GetOptionOrReturnsDefaultOnTypeMismatch) {
    OptionsMap opts;
    opts["count"] = 42;  // stored as int

    double result = get_option_or<double>(opts, "count", 3.14);
    EXPECT_DOUBLE_EQ(result, 3.14);  // default because type mismatch
}

TEST(OptionsHelpers, HasOptionReturnsTrueForMatchingType) {
    OptionsMap opts;
    opts["count"] = 42;
    opts["name"] = std::string("test");

    EXPECT_TRUE(has_option<int>(opts, "count"));
    EXPECT_TRUE(has_option<std::string>(opts, "name"));
}

TEST(OptionsHelpers, HasOptionReturnsFalseForMissingKey) {
    OptionsMap opts;
    opts["count"] = 42;

    EXPECT_FALSE(has_option<int>(opts, "missing"));
}

TEST(OptionsHelpers, HasOptionReturnsFalseForTypeMismatch) {
    OptionsMap opts;
    opts["count"] = 42;  // stored as int

    EXPECT_FALSE(has_option<double>(opts, "count"));  // type mismatch
    EXPECT_TRUE(has_option<int>(opts, "count"));      // correct type
}

TEST(OptionsHelpers, GetFormatReturnsCorrectEnum) {
    OptionsVariant v_bool = true;
    EXPECT_EQ(get_format(v_bool), OptionsDataFormat::Bool);

    OptionsVariant v_int = 42;
    EXPECT_EQ(get_format(v_int), OptionsDataFormat::Int);

    OptionsVariant v_double = 3.14;
    EXPECT_EQ(get_format(v_double), OptionsDataFormat::Double);

    OptionsVariant v_string = std::string("test");
    EXPECT_EQ(get_format(v_string), OptionsDataFormat::String);
}

TEST(OptionsHelpers, HeterogeneousLookupWithStringView) {
    OptionsMap opts;
    opts["key"] = 42;

    // Test heterogeneous lookup with string_view
    std::string_view sv = "key";
    auto result = get_option<int>(opts, sv);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
