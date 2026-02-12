/**
 * test_options_variant.cpp
 *
 * Unit tests for OptionsVariant and OptionsMap (options.h).
 *
 * OptionsVariant is the central C++17 std::variant type used throughout the
 * GPS C++ controller to pass heterogeneous parameters.  It holds one of:
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
 *
 * No ROS or LibTorch headers are needed — options.h only depends on
 * the C++17 standard library and Eigen.
 */

#include <gtest/gtest.h>
#include "gps_agent_pkg/options.h"

#include <string>
#include <vector>

using namespace gps_control;

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
// Variant rebinding — C++17 assignment changes the active alternative
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
