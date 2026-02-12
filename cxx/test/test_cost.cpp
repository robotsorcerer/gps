/**
 * @file test_cost.cpp
 * @brief Unit tests for Cost classes.
 */

#include <gtest/gtest.h>

#include "../algorithm/cost/cost_action.h"
#include "../algorithm/cost/cost_sum.h"
#include "../algorithm/cost/cost_utils.h"
#include "../sample/sample.h"

namespace gps {
namespace test {

class CostActionTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr int dV = 2;

    Sample create_sample() {
        Sample sample(T, dX, dU, dV);
        sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
        sample.set(SampleType::ACTION, Matrix::Random(T, dU));
        sample.set(SampleType::NOISE, Matrix::Random(T, dV));
        sample.set_state_types({SampleType::JOINT_ANGLES});
        return sample;
    }
};

TEST_F(CostActionTest, ConstructWithWeights) {
    Vector wu = Vector::Ones(dU);
    CostAction cost(wu);

    EXPECT_EQ(cost.mode(), CostMode::PROTAGONIST);
}

TEST_F(CostActionTest, EvalProtagonist) {
    Vector wu = Vector::Ones(dU) * 2.0;
    CostAction cost(wu, 1.0, CostMode::PROTAGONIST);

    Sample sample = create_sample();
    CostResult result = cost.eval(sample);

    // Check dimensions
    EXPECT_EQ(result.l.size(), T);
    EXPECT_EQ(result.lx.rows(), T);
    EXPECT_EQ(result.lx.cols(), dX);
    EXPECT_EQ(result.lu.rows(), T);
    EXPECT_EQ(result.lu.cols(), dU);
    EXPECT_EQ(result.luu.size(), static_cast<std::size_t>(T));

    // Cost should be non-negative (quadratic)
    for (int t = 0; t < T; ++t) {
        EXPECT_GE(result.l(t), 0.0);
    }

    // Hessian luu should be positive semi-definite (equal to weight matrix)
    Matrix Wu = wu.asDiagonal();
    for (int t = 0; t < T; ++t) {
        EXPECT_TRUE(result.luu[static_cast<std::size_t>(t)].isApprox(Wu));
    }
}

TEST_F(CostActionTest, EvalZeroAction) {
    Vector wu = Vector::Ones(dU);

    CostAction cost(wu);

    // Create sample with zero actions
    Sample sample(T, dX, dU);
    sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
    sample.set(SampleType::ACTION, Matrix::Zero(T, dU));  // Zero actions
    sample.set_state_types({SampleType::JOINT_ANGLES});

    CostResult result = cost.eval(sample);

    // Cost should be zero when actions are zero
    for (int t = 0; t < T; ++t) {
        EXPECT_NEAR(result.l(t), 0.0, 1e-10);
    }
}

TEST_F(CostActionTest, Clone) {
    Vector wu = Vector::Ones(dU) * 3.0;
    CostAction cost(wu);

    auto cloned = cost.clone();
    auto* cloned_cost = dynamic_cast<CostAction*>(cloned.get());

    ASSERT_NE(cloned_cost, nullptr);
    EXPECT_TRUE(cloned_cost->wu().isApprox(cost.wu()));
}

TEST_F(CostActionTest, GammaParameter) {
    Vector wu = Vector::Ones(dU);
    double gamma = 2.5;
    CostAction cost(wu, gamma, CostMode::ROBUST);

    EXPECT_DOUBLE_EQ(cost.gamma(), gamma);

    cost.set_gamma(1.5);
    EXPECT_DOUBLE_EQ(cost.gamma(), 1.5);
}

TEST_F(CostActionTest, EvalRobustMode) {
    Vector wu = Vector::Ones(dU);
    double gamma = 0.5;
    CostAction cost(wu, gamma, CostMode::ROBUST);

    // Create sample with both protagonist and antagonist actions
    Sample sample(T, dX, dU, dV);
    sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
    sample.set(SampleType::ACTION, Matrix::Ones(T, dU));  // u = 1
    sample.set(SampleType::NOISE, Matrix::Ones(T, dV) * 0.5);  // v = 0.5
    sample.set_state_types({SampleType::JOINT_ANGLES});

    CostResult result = cost.eval(sample);

    // Check dimensions
    EXPECT_EQ(result.l.size(), T);
    EXPECT_EQ(result.lu.rows(), T);
    EXPECT_EQ(result.lu.cols(), dU);
    EXPECT_TRUE(result.lv.rows() > 0);  // Adversary derivatives should exist

    // Robust cost: l = 0.5 * sum(wu * u^2) - gamma * sum(wu * v^2)
    // For u=1, v=0.5, wu=1, gamma=0.5:
    // l = 0.5 * dU * 1 - 0.5 * dV * 0.25 = 0.5*3 - 0.5*2*0.25 = 1.5 - 0.25 = 1.25
    double expected_u_term = 0.5 * dU * 1.0;
    double expected_v_term = gamma * dV * 0.25;
    double expected_cost = expected_u_term - expected_v_term;

    for (int t = 0; t < T; ++t) {
        EXPECT_NEAR(result.l(t), expected_cost, 1e-10);
    }
}

TEST_F(CostActionTest, EvalWithProtagonist) {
    Vector wu = Vector::Ones(dU);
    double gamma = 1.0;
    CostAction cost(wu, gamma, CostMode::ANTAGONIST);

    // Create protagonist and antagonist samples
    Sample sample_prot(T, dX, dU);
    sample_prot.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
    sample_prot.set(SampleType::ACTION, Matrix::Ones(T, dU));  // u_prot = 1
    sample_prot.set_state_types({SampleType::JOINT_ANGLES});

    Sample sample_ant(T, dX, dU);
    sample_ant.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
    sample_ant.set(SampleType::ACTION, Matrix::Ones(T, dU) * 0.5);  // v = 0.5
    sample_ant.set_state_types({SampleType::JOINT_ANGLES});

    CostResult result = cost.eval_with_protagonist(sample_ant, sample_prot);

    // Check dimensions
    EXPECT_EQ(result.l.size(), T);
    EXPECT_TRUE(result.lv.rows() > 0);  // Adversary derivatives

    // Game-theoretic cost (NEGATED for antagonist maximization):
    // l_raw = 0.5 * sum(wu * u_prot^2) - gamma * sum(wu * v^2)
    // For u_prot=1, v=0.5, wu=1, gamma=1:
    // l_raw = 0.5 * dU - 1.0 * dU * 0.25 = 0.5*3 - 0.75 = 0.75
    // After negation: l = -0.75
    double expected_raw = 0.5 * dU - 1.0 * dU * 0.25;
    double expected_negated = -expected_raw;

    for (int t = 0; t < T; ++t) {
        EXPECT_NEAR(result.l(t), expected_negated, 1e-10);
    }
}

// Cost sum tests
class CostSumTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;

    Sample create_sample() {
        Sample sample(T, dX, dU);
        sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
        sample.set(SampleType::ACTION, Matrix::Random(T, dU));
        sample.set_state_types({SampleType::JOINT_ANGLES});
        return sample;
    }
};

TEST_F(CostSumTest, AddCosts) {
    CostSum sum;

    auto cost1 = std::make_shared<CostAction>(Vector::Ones(dU));
    auto cost2 = std::make_shared<CostAction>(Vector::Ones(dU) * 2);

    sum.add_cost(cost1, 1.0);
    sum.add_cost(cost2, 0.5);

    EXPECT_EQ(sum.num_costs(), 2);
    EXPECT_EQ(sum.weight(0), 1.0);
    EXPECT_EQ(sum.weight(1), 0.5);
}

TEST_F(CostSumTest, EvalSumsWeightedCosts) {
    auto cost1 = std::make_shared<CostAction>(Vector::Ones(dU));
    auto cost2 = std::make_shared<CostAction>(Vector::Ones(dU));

    CostSum sum({cost1, cost2}, {1.0, 2.0});

    Sample sample = create_sample();

    CostResult result_sum = sum.eval(sample);
    CostResult result1 = cost1->eval(sample);
    CostResult result2 = cost2->eval(sample);

    // Check that sum is weighted correctly
    for (int t = 0; t < T; ++t) {
        double expected = 1.0 * result1.l(t) + 2.0 * result2.l(t);
        EXPECT_NEAR(result_sum.l(t), expected, 1e-10);
    }
}

// Cost utilities tests
class CostUtilsTest : public ::testing::Test {};

TEST_F(CostUtilsTest, RampMultiplierConstant) {
    Vector wpm = get_ramp_multiplier(RampOption::CONSTANT, 10, 1.0);

    EXPECT_EQ(wpm.size(), 10);
    for (int t = 0; t < 10; ++t) {
        EXPECT_DOUBLE_EQ(wpm(t), 1.0);
    }
}

TEST_F(CostUtilsTest, RampMultiplierLinear) {
    Vector wpm = get_ramp_multiplier(RampOption::LINEAR, 10, 2.0);

    EXPECT_DOUBLE_EQ(wpm(0), 0.0);
    EXPECT_DOUBLE_EQ(wpm(9), 2.0);

    // Should be monotonically increasing
    for (int t = 1; t < 10; ++t) {
        EXPECT_GT(wpm(t), wpm(t - 1));
    }
}

TEST_F(CostUtilsTest, RampMultiplierFinalOnly) {
    Vector wpm = get_ramp_multiplier(RampOption::FINAL_ONLY, 10, 5.0);

    for (int t = 0; t < 9; ++t) {
        EXPECT_DOUBLE_EQ(wpm(t), 0.0);
    }
    EXPECT_DOUBLE_EQ(wpm(9), 5.0);
}

TEST_F(CostUtilsTest, EvalL1L2Term) {
    int T = 5;
    int dim = 3;

    Matrix wp = Matrix::Ones(T, dim);
    Matrix dist = Matrix::Random(T, dim);

    auto result = eval_l1l2_term(wp, dist, 0.0, 1.0, 0.0);  // Pure L2

    EXPECT_EQ(result.l.size(), T);
    EXPECT_EQ(result.ls.rows(), T);
    EXPECT_EQ(result.ls.cols(), dim);
    EXPECT_EQ(result.lss.size(), static_cast<std::size_t>(T));

    // L2 cost: 0.5 * sum(dist^2)
    for (int t = 0; t < T; ++t) {
        double expected = 0.5 * dist.row(t).squaredNorm();
        EXPECT_NEAR(result.l(t), expected, 1e-10);
    }
}

}  // namespace test
}  // namespace gps
