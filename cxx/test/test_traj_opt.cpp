/**
 * @file test_traj_opt.cpp
 * @brief Unit tests for trajectory optimization classes.
 */

#include <gtest/gtest.h>

#include "../algorithm/traj_opt/traj_opt_lqr.h"
#include "../algorithm/dynamics/dynamics.h"

namespace gps {
namespace test {

class TrajOptLQRTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 4;
    static constexpr int dU = 2;
};

TEST_F(TrajOptLQRTest, Construction) {
    Hyperparams params;
    params["del0"] = 1e-4;
    params["min_eta"] = 1e-8;
    params["max_eta"] = 1e16;

    TrajOptLQR traj_opt(params);
    EXPECT_NE(traj_opt.clone(), nullptr);
}

TEST_F(TrajOptLQRTest, Clone) {
    Hyperparams params;
    TrajOptLQR traj_opt(params);

    auto cloned = traj_opt.clone();
    ASSERT_NE(cloned, nullptr);

    auto* cloned_lqr = dynamic_cast<TrajOptLQR*>(cloned.get());
    EXPECT_NE(cloned_lqr, nullptr);
}

// LinearDynamics tests
class LinearDynamicsTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 4;
    static constexpr int dU = 2;
};

TEST_F(LinearDynamicsTest, Zeros) {
    LinearDynamics dyn = LinearDynamics::zeros(T, dX, dU);

    EXPECT_EQ(dyn.Fm.size(), static_cast<std::size_t>(T));
    EXPECT_EQ(dyn.Fm[0].rows(), dX);
    EXPECT_EQ(dyn.Fm[0].cols(), dX + dU);
    EXPECT_TRUE(dyn.Fm[0].isZero());

    EXPECT_EQ(dyn.fv.rows(), T);
    EXPECT_EQ(dyn.fv.cols(), dX);
    EXPECT_TRUE(dyn.fv.isZero());

    EXPECT_EQ(dyn.dyn_covar.size(), static_cast<std::size_t>(T));
    EXPECT_EQ(dyn.dyn_covar[0].rows(), dX);
    EXPECT_EQ(dyn.dyn_covar[0].cols(), dX);
}

TEST_F(LinearDynamicsTest, Predict) {
    LinearDynamics dyn = LinearDynamics::zeros(T, dX, dU);

    // Set identity dynamics for x (x_{t+1} = x_t)
    for (int t = 0; t < T; ++t) {
        dyn.Fm[t].leftCols(dX) = Matrix::Identity(dX, dX);
    }

    Vector x = Vector::Random(dX);
    Vector u = Vector::Random(dU);

    Vector x_next = dyn.predict(x, u, 0);

    EXPECT_EQ(x_next.size(), dX);
    EXPECT_TRUE(x_next.isApprox(x));  // With identity dynamics
}

TEST_F(LinearDynamicsTest, PredictWithControl) {
    LinearDynamics dyn = LinearDynamics::zeros(T, dX, dU);

    // Set dynamics: x_{t+1} = x_t + B * u_t
    Matrix B = Matrix::Random(dX, dU);
    for (int t = 0; t < T; ++t) {
        dyn.Fm[t].leftCols(dX) = Matrix::Identity(dX, dX);
        dyn.Fm[t].rightCols(dU) = B;
    }

    Vector x = Vector::Random(dX);
    Vector u = Vector::Random(dU);

    Vector x_next = dyn.predict(x, u, 0);

    Vector expected = x + B * u;
    EXPECT_TRUE(x_next.isApprox(expected));
}

TEST_F(LinearDynamicsTest, PredictWithBias) {
    LinearDynamics dyn = LinearDynamics::zeros(T, dX, dU);

    // Set bias term
    Vector bias = Vector::Random(dX);
    dyn.fv.row(0) = bias.transpose();

    Vector x = Vector::Zero(dX);
    Vector u = Vector::Zero(dU);

    Vector x_next = dyn.predict(x, u, 0);

    EXPECT_TRUE(x_next.isApprox(bias));
}

}  // namespace test
}  // namespace gps
