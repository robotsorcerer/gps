/**
 * @file test_sample.cpp
 * @brief Unit tests for Sample and SampleList classes.
 */

#include <gtest/gtest.h>

#include "../sample/sample.h"
#include "../sample/sample_list.h"

namespace gps {
namespace test {

class SampleTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr int dV = 2;
};

TEST_F(SampleTest, Construction) {
    Sample sample(T, dX, dU);
    EXPECT_EQ(sample.T(), T);
    EXPECT_EQ(sample.dX(), dX);
    EXPECT_EQ(sample.dU(), dU);
    EXPECT_EQ(sample.dV(), 0);
}

TEST_F(SampleTest, ConstructionWithAdversary) {
    Sample sample(T, dX, dU, dV);
    EXPECT_EQ(sample.T(), T);
    EXPECT_EQ(sample.dX(), dX);
    EXPECT_EQ(sample.dU(), dU);
    EXPECT_EQ(sample.dV(), dV);
}

TEST_F(SampleTest, ConstructionFromHyperparams) {
    Hyperparams params;
    params["T"] = T;
    params["dX"] = dX;
    params["dU"] = dU;

    Sample sample(params);
    EXPECT_EQ(sample.T(), T);
    EXPECT_EQ(sample.dX(), dX);
    EXPECT_EQ(sample.dU(), dU);
}

TEST_F(SampleTest, SetAndGetSingleTimestep) {
    Sample sample(T, dX, dU);

    Vector data = Vector::Random(5);
    sample.set(SampleType::JOINT_ANGLES, data, 3);

    Vector retrieved = sample.get(SampleType::JOINT_ANGLES, 3);
    EXPECT_EQ(retrieved.size(), data.size());
    EXPECT_TRUE(retrieved.isApprox(data));
}

TEST_F(SampleTest, SetAndGetAllTimesteps) {
    Sample sample(T, dX, dU);

    Matrix data = Matrix::Random(T, 7);
    sample.set(SampleType::JOINT_ANGLES, data);

    const Matrix& retrieved = sample.get(SampleType::JOINT_ANGLES);
    EXPECT_EQ(retrieved.rows(), T);
    EXPECT_EQ(retrieved.cols(), 7);
    EXPECT_TRUE(retrieved.isApprox(data));
}

TEST_F(SampleTest, HasSensorType) {
    Sample sample(T, dX, dU);

    EXPECT_FALSE(sample.has(SampleType::JOINT_ANGLES));

    sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, 7));
    EXPECT_TRUE(sample.has(SampleType::JOINT_ANGLES));
}

TEST_F(SampleTest, GetAction) {
    Sample sample(T, dX, dU);

    Matrix actions = Matrix::Random(T, dU);
    sample.set(SampleType::ACTION, actions);

    Matrix U = sample.get_U();
    EXPECT_TRUE(U.isApprox(actions));

    Vector u_3 = sample.get_U(3);
    EXPECT_TRUE(u_3.isApprox(actions.row(3).transpose()));
}

TEST_F(SampleTest, ThrowsOnInvalidTimestep) {
    Sample sample(T, dX, dU);

    EXPECT_THROW(sample.set(SampleType::JOINT_ANGLES, Vector::Random(5), -1),
                 std::out_of_range);
    EXPECT_THROW(sample.set(SampleType::JOINT_ANGLES, Vector::Random(5), T),
                 std::out_of_range);
}

TEST_F(SampleTest, ThrowsOnDimensionMismatch) {
    Sample sample(T, dX, dU);

    // First set with dim 5
    sample.set(SampleType::JOINT_ANGLES, Vector::Random(5), 0);

    // Try to set with dim 3 - should throw
    EXPECT_THROW(sample.set(SampleType::JOINT_ANGLES, Vector::Random(3), 1),
                 std::invalid_argument);
}

// SampleList tests
class SampleListTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
};

TEST_F(SampleListTest, AddAndSize) {
    SampleList list;
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0);

    list.add(Sample(T, dX, dU));
    EXPECT_FALSE(list.empty());
    EXPECT_EQ(list.size(), 1);

    list.add(Sample(T, dX, dU));
    EXPECT_EQ(list.size(), 2);
}

TEST_F(SampleListTest, AccessByIndex) {
    SampleList list;

    Sample s1(T, dX, dU);
    s1.set(SampleType::ACTION, Matrix::Random(T, dU));
    list.add(std::move(s1));

    const Sample& retrieved = list[0];
    EXPECT_TRUE(retrieved.has(SampleType::ACTION));
}

TEST_F(SampleListTest, GetX) {
    SampleList list;

    // Add samples with state data configured
    for (int n = 0; n < 3; ++n) {
        Sample s(T, dX, dU);
        s.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
        s.set_state_types({SampleType::JOINT_ANGLES});
        list.add(std::move(s));
    }

    Tensor3d X = list.get_X();
    EXPECT_EQ(X.size(), 3);
    for (const auto& x : X) {
        EXPECT_EQ(x.rows(), T);
        EXPECT_EQ(x.cols(), dX);
    }
}

TEST_F(SampleListTest, ThrowsOnDimensionMismatch) {
    SampleList list;
    list.add(Sample(T, dX, dU));

    // Try to add sample with different dimensions
    EXPECT_THROW(list.add(Sample(T + 1, dX, dU)), std::invalid_argument);
    EXPECT_THROW(list.add(Sample(T, dX + 1, dU)), std::invalid_argument);
}

TEST_F(SampleListTest, Clear) {
    SampleList list;
    list.add(Sample(T, dX, dU));
    list.add(Sample(T, dX, dU));
    EXPECT_EQ(list.size(), 2);

    list.clear();
    EXPECT_TRUE(list.empty());
}

}  // namespace test
}  // namespace gps
