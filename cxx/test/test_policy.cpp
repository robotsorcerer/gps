/**
 * @file test_policy.cpp
 * @brief Unit tests for Policy classes.
 */

#include <gtest/gtest.h>

#include "../algorithm/policy/lin_gauss_policy.h"

namespace gps {
namespace test {

class LinearGaussianPolicyTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr double init_var = 1.0;
};

TEST_F(LinearGaussianPolicyTest, DefaultConstruction) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    EXPECT_EQ(policy.T(), T);
    EXPECT_EQ(policy.dX(), dX);
    EXPECT_EQ(policy.dU(), dU);
}

TEST_F(LinearGaussianPolicyTest, ConstructionFromHyperparams) {
    Hyperparams params;
    params["T"] = T;
    params["dX"] = dX;
    params["dU"] = dU;
    params["init_var"] = init_var;

    LinearGaussianPolicy policy(params);

    EXPECT_EQ(policy.T(), T);
    EXPECT_EQ(policy.dX(), dX);
    EXPECT_EQ(policy.dU(), dU);
}

TEST_F(LinearGaussianPolicyTest, MatrixDimensions) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    EXPECT_EQ(policy.K().size(), static_cast<std::size_t>(T));
    EXPECT_EQ(policy.K()[0].rows(), dU);
    EXPECT_EQ(policy.K()[0].cols(), dX);

    EXPECT_EQ(policy.k().rows(), T);
    EXPECT_EQ(policy.k().cols(), dU);

    EXPECT_EQ(policy.pol_covar().size(), static_cast<std::size_t>(T));
    EXPECT_EQ(policy.pol_covar()[0].rows(), dU);
    EXPECT_EQ(policy.pol_covar()[0].cols(), dU);
}

TEST_F(LinearGaussianPolicyTest, InitialValuesAreZeroAndIdentity) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    // K should be zero
    for (int t = 0; t < T; ++t) {
        EXPECT_TRUE(policy.K()[t].isZero());
    }

    // k should be zero
    EXPECT_TRUE(policy.k().isZero());

    // pol_covar should be init_var * I
    for (int t = 0; t < T; ++t) {
        EXPECT_TRUE(policy.pol_covar()[t].isApprox(
            Matrix::Identity(dU, dU) * init_var));
    }
}

TEST_F(LinearGaussianPolicyTest, ActWithoutNoise) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    // Set some feedback gains
    policy.K()[5] = Matrix::Random(dU, dX);
    policy.k().row(5) = Vector::Random(dU).transpose();

    Vector x = Vector::Random(dX);
    Vector u = policy.act(&x, nullptr, 5, nullptr);

    // u = K * x + k
    Vector expected = policy.K()[5] * x + policy.k().row(5).transpose();
    EXPECT_TRUE(u.isApprox(expected));
}

TEST_F(LinearGaussianPolicyTest, ActWithNoise) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    Vector x = Vector::Random(dX);
    Vector noise = Vector::Random(dU);

    Vector u_with_noise = policy.act(&x, nullptr, 0, &noise);
    Vector u_without_noise = policy.act(&x, nullptr, 0, nullptr);

    // Difference should be chol_pol_covar * noise
    Vector diff = u_with_noise - u_without_noise;
    Vector expected_diff = policy.chol_pol_covar()[0] * noise;
    EXPECT_TRUE(diff.isApprox(expected_diff));
}

TEST_F(LinearGaussianPolicyTest, ActThrowsOnNullState) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);
    EXPECT_THROW(policy.act(nullptr, nullptr, 0, nullptr), std::invalid_argument);
}

TEST_F(LinearGaussianPolicyTest, ActThrowsOnInvalidTimestep) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);
    Vector x = Vector::Random(dX);

    EXPECT_THROW(policy.act(&x, nullptr, -1, nullptr), std::out_of_range);
    EXPECT_THROW(policy.act(&x, nullptr, T, nullptr), std::out_of_range);
}

TEST_F(LinearGaussianPolicyTest, MeanAction) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);
    policy.K()[3] = Matrix::Random(dU, dX);
    policy.k().row(3) = Vector::Random(dU).transpose();

    Vector x = Vector::Random(dX);
    Vector mean = policy.mean_action(x, 3);

    Vector expected = policy.K()[3] * x + policy.k().row(3).transpose();
    EXPECT_TRUE(mean.isApprox(expected));
}

TEST_F(LinearGaussianPolicyTest, FoldK) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    Matrix noise = Matrix::Random(T, dU);
    Matrix folded = policy.fold_k(noise);

    EXPECT_EQ(folded.rows(), T);
    EXPECT_EQ(folded.cols(), dU);

    // k_folded[t] = k[t] + chol_pol_covar[t] * noise[t]
    for (int t = 0; t < T; ++t) {
        Vector expected = policy.k().row(t).transpose() +
                          policy.chol_pol_covar()[t] * noise.row(t).transpose();
        EXPECT_TRUE(folded.row(t).transpose().isApprox(expected));
    }
}

TEST_F(LinearGaussianPolicyTest, Clone) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);
    policy.K()[0] = Matrix::Random(dU, dX);

    auto cloned = policy.clone();
    auto* cloned_lgp = dynamic_cast<LinearGaussianPolicy*>(cloned.get());

    ASSERT_NE(cloned_lgp, nullptr);
    EXPECT_TRUE(cloned_lgp->K()[0].isApprox(policy.K()[0]));
}

// Robust policy tests
class LinearGaussianPolicyRobustTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr int dV = 2;
};

TEST_F(LinearGaussianPolicyRobustTest, Construction) {
    LinearGaussianPolicy protagonist(T, dX, dU);
    LinearGaussianPolicy antagonist(T, dX, dV);

    LinearGaussianPolicyRobust robust(
        std::move(protagonist), std::move(antagonist));

    EXPECT_EQ(robust.T(), T);
    EXPECT_EQ(robust.dX(), dX);
    EXPECT_EQ(robust.dU(), dU);
    EXPECT_EQ(robust.dV(), dV);
    EXPECT_EQ(robust.mode(), PolicyMode::PROTAGONIST);
}

TEST_F(LinearGaussianPolicyRobustTest, ConstructionFromHyperparams) {
    Hyperparams params;
    params["T"] = T;
    params["dX"] = dX;
    params["dU"] = dU;
    params["dV"] = dV;

    LinearGaussianPolicyRobust robust(params);

    EXPECT_EQ(robust.T(), T);
    EXPECT_EQ(robust.dX(), dX);
    EXPECT_EQ(robust.dU(), dU);
    EXPECT_EQ(robust.dV(), dV);
}

TEST_F(LinearGaussianPolicyRobustTest, ActInProtagonistMode) {
    Hyperparams params;
    params["T"] = T;
    params["dX"] = dX;
    params["dU"] = dU;
    params["dV"] = dV;

    LinearGaussianPolicyRobust robust(params);
    robust.set_mode(PolicyMode::PROTAGONIST);

    Vector x = Vector::Random(dX);
    Vector u = robust.act(&x, nullptr, 0, nullptr);

    EXPECT_EQ(u.size(), dU);
}

TEST_F(LinearGaussianPolicyRobustTest, ActInAntagonistMode) {
    Hyperparams params;
    params["T"] = T;
    params["dX"] = dX;
    params["dU"] = dU;
    params["dV"] = dV;

    LinearGaussianPolicyRobust robust(params);
    robust.set_mode(PolicyMode::ANTAGONIST);

    Vector x = Vector::Random(dX);
    Vector v = robust.act(&x, nullptr, 0, nullptr);

    EXPECT_EQ(v.size(), dV);  // Returns antagonist action
}

TEST_F(LinearGaussianPolicyRobustTest, SetMode) {
    Hyperparams params;
    params["T"] = T;
    params["dX"] = dX;
    params["dU"] = dU;
    params["dV"] = dV;

    LinearGaussianPolicyRobust robust(params);

    robust.set_mode(PolicyMode::ANTAGONIST);
    EXPECT_EQ(robust.mode(), PolicyMode::ANTAGONIST);

    robust.set_mode(PolicyMode::ROBUST);
    EXPECT_EQ(robust.mode(), PolicyMode::ROBUST);
}

}  // namespace test
}  // namespace gps
