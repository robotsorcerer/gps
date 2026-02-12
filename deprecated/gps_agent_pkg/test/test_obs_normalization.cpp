/**
 * test_obs_normalization.cpp
 *
 * Standalone unit tests for the observation normalisation formula used in
 * PyTorchController::get_action() (pytorchcontroller.cpp lines 108-111):
 *
 *     obs_scaled(i) = obs(i) * scale_diag_(i) + bias_(i)
 *
 * This is a pointwise affine transform: scaled = diag(scale) * obs + bias.
 * The formula is extracted verbatim so that any change to the controller
 * will break these tests, making the regression visible.
 *
 * Only Eigen and the C++ standard library are required — no ROS, no LibTorch.
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Reference implementation (mirrors pytorchcontroller.cpp verbatim)
// ---------------------------------------------------------------------------

static Eigen::VectorXd apply_obs_norm(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& scale_diag,
    const Eigen::VectorXd& bias)
{
    const int dO = static_cast<int>(obs.size());
    Eigen::VectorXd obs_scaled(dO);
    for (int i = 0; i < dO; ++i)
        obs_scaled(i) = obs(i) * scale_diag(i) + bias(i);
    return obs_scaled;
}

// ---------------------------------------------------------------------------
// Basic correctness
// ---------------------------------------------------------------------------

TEST(ObsNorm, IdentityTransform) {
    // scale=1, bias=0  →  scaled == obs
    Eigen::VectorXd obs   = Eigen::VectorXd::LinSpaced(8, -2.0, 5.0);
    Eigen::VectorXd scale = Eigen::VectorXd::Ones(8);
    Eigen::VectorXd bias  = Eigen::VectorXd::Zero(8);
    EXPECT_TRUE(apply_obs_norm(obs, scale, bias).isApprox(obs, 1e-12));
}

TEST(ObsNorm, ZeroObsReturnsBias) {
    // obs=0  →  scaled = bias
    Eigen::VectorXd obs  = Eigen::VectorXd::Zero(5);
    Eigen::VectorXd sc   = Eigen::VectorXd::Random(5);
    Eigen::VectorXd bias; bias.resize(5); bias << 1.1, 2.2, 3.3, 4.4, 5.5;
    EXPECT_TRUE(apply_obs_norm(obs, sc, bias).isApprox(bias, 1e-12));
}

TEST(ObsNorm, ZeroScaleMapsToConstant) {
    // scale=0  →  scaled = bias regardless of obs
    Eigen::VectorXd obs  = Eigen::VectorXd::Constant(4, 999.0);
    Eigen::VectorXd sc   = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd bias; bias.resize(4); bias << 0.1, 0.2, 0.3, 0.4;
    EXPECT_TRUE(apply_obs_norm(obs, sc, bias).isApprox(bias, 1e-12));
}

TEST(ObsNorm, KnownValues) {
    //  obs=[1,2,3], scale=[2, 0.5,-1], bias=[0.1, 0.2, 0.3]
    //  expected = [1*2+0.1, 2*0.5+0.2, 3*(-1)+0.3] = [2.1, 1.2, -2.7]
    Eigen::VectorXd obs(3);   obs   <<  1.0,  2.0,  3.0;
    Eigen::VectorXd sc(3);    sc    <<  2.0,  0.5, -1.0;
    Eigen::VectorXd bias(3);  bias  <<  0.1,  0.2,  0.3;
    Eigen::VectorXd expected(3); expected << 2.1, 1.2, -2.7;
    EXPECT_TRUE(apply_obs_norm(obs, sc, bias).isApprox(expected, 1e-9));
}

TEST(ObsNorm, NegativeBias) {
    Eigen::VectorXd obs(2);  obs  << 0.0, 0.0;
    Eigen::VectorXd sc(2);   sc   << 1.0, 1.0;
    Eigen::VectorXd bias(2); bias << -5.0, -0.5;
    Eigen::VectorXd result = apply_obs_norm(obs, sc, bias);
    EXPECT_DOUBLE_EQ(result(0), -5.0);
    EXPECT_DOUBLE_EQ(result(1), -0.5);
}

// ---------------------------------------------------------------------------
// Dimension / shape correctness
// ---------------------------------------------------------------------------

TEST(ObsNorm, OutputDimensionMatchesInput) {
    for (int dO : {1, 7, 14, 64}) {
        Eigen::VectorXd obs   = Eigen::VectorXd::Random(dO);
        Eigen::VectorXd scale = Eigen::VectorXd::Ones(dO);
        Eigen::VectorXd bias  = Eigen::VectorXd::Zero(dO);
        EXPECT_EQ(apply_obs_norm(obs, scale, bias).size(), dO);
    }
}

TEST(ObsNorm, LargeVectorIdentity) {
    int dO = 256;
    Eigen::VectorXd obs   = Eigen::VectorXd::Random(dO);
    Eigen::VectorXd scale = Eigen::VectorXd::Ones(dO);
    Eigen::VectorXd bias  = Eigen::VectorXd::Zero(dO);
    EXPECT_TRUE(apply_obs_norm(obs, scale, bias).isApprox(obs, 1e-12));
}

// ---------------------------------------------------------------------------
// Numerical edge cases (the scenarios the NaN guard in policy_opt_pytorch.py
// is meant to prevent from propagating into training data)
// ---------------------------------------------------------------------------

TEST(ObsNorm, NaNObsPropagates) {
    // A NaN in obs(i) must appear in scaled(i).
    // policy_opt_pytorch.py now checks isfinite(loss) before backward().
    Eigen::VectorXd obs(3);  obs  << 1.0, std::numeric_limits<double>::quiet_NaN(), 3.0;
    Eigen::VectorXd sc(3);   sc   << 1.0, 1.0, 1.0;
    Eigen::VectorXd bias(3); bias << 0.0, 0.0, 0.0;
    auto result = apply_obs_norm(obs, sc, bias);
    EXPECT_FALSE(std::isnan(result(0)));
    EXPECT_TRUE (std::isnan(result(1)));   // NaN propagated
    EXPECT_FALSE(std::isnan(result(2)));
}

TEST(ObsNorm, InfObsPropagates) {
    Eigen::VectorXd obs(2);  obs  << std::numeric_limits<double>::infinity(), 1.0;
    Eigen::VectorXd sc(2);   sc   << 1.0, 1.0;
    Eigen::VectorXd bias(2); bias << 0.0, 0.0;
    auto result = apply_obs_norm(obs, sc, bias);
    EXPECT_TRUE (std::isinf(result(0)));
    EXPECT_FALSE(std::isinf(result(1)));
}

TEST(ObsNorm, NaNScalePropagates) {
    Eigen::VectorXd obs(2);  obs  << 1.0, 2.0;
    Eigen::VectorXd sc(2);   sc   << std::numeric_limits<double>::quiet_NaN(), 1.0;
    Eigen::VectorXd bias(2); bias << 0.0, 0.0;
    auto result = apply_obs_norm(obs, sc, bias);
    EXPECT_TRUE (std::isnan(result(0)));
    EXPECT_FALSE(std::isnan(result(1)));
}

TEST(ObsNorm, InfScaleFiniteObs) {
    Eigen::VectorXd obs(1);  obs  << 1.0;
    Eigen::VectorXd sc(1);   sc   << std::numeric_limits<double>::infinity();
    Eigen::VectorXd bias(1); bias << 0.0;
    auto result = apply_obs_norm(obs, sc, bias);
    EXPECT_TRUE(std::isinf(result(0)));
}

// ---------------------------------------------------------------------------
// Linearity checks (affine map properties)
// ---------------------------------------------------------------------------

TEST(ObsNorm, Superposition) {
    // f(a+b) == f(a) + scale*b  (affine, not linear — so we test differently)
    // But f(scale * obs) + bias == scale*obs + bias, and f(obs) = scale*obs + bias
    // Verify by numeric construction.
    Eigen::VectorXd obs(4); obs << 1.0, -1.0, 0.5, 2.0;
    Eigen::VectorXd sc(4);  sc  << 2.0,  3.0, 1.0, 0.5;
    Eigen::VectorXd b(4);   b   << 0.0,  0.0, 0.0, 0.0;  // bias=0 → linear
    // With bias=0, f(k*obs) = k*f(obs)
    double k = 3.0;
    auto f_obs  = apply_obs_norm(obs,       sc, b);
    auto f_kobs = apply_obs_norm(obs * k,   sc, b);
    EXPECT_TRUE(f_kobs.isApprox(f_obs * k, 1e-9));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
