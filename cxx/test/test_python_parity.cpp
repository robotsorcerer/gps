/**
 * @file test_python_parity.cpp
 * @brief Integration tests validating C++ implementation against Python reference.
 *
 * These tests verify numerical correctness of the C++20 GPS implementation
 * against the reference Python implementation in python/gps/algorithm/.
 *
 * Key areas tested:
 * 1. LinearGaussianPolicy.act() - U = K*x + k + L^T*noise formula (transpose fixed)
 * 2. CostAction.eval() - Quadratic cost with game-theoretic formulation
 * 3. CostAction.eval_with_protagonist() - Antagonist mode with negation
 * 4. Sample data packing/unpacking
 *
 * Previous discrepancies (now FIXED):
 * - LinearGaussianPolicy: Transpose on chol_pol_covar in act() and fold_k() - FIXED
 * - CostAction: Added gamma parameter and game-theoretic formulation - FIXED
 */

#include <gtest/gtest.h>
#include <cmath>

#include "../algorithm/policy/lin_gauss_policy.h"
#include "../algorithm/cost/cost_action.h"
#include "../sample/sample.h"

namespace gps {
namespace test {

// =============================================================================
// Test Constants - Matching Python test dimensions
// =============================================================================

constexpr int kT = 100;   // Time horizon (matching Python typical T=100)
constexpr int kDX = 14;   // State dimension (e.g., 7 joints * 2 for pos/vel)
constexpr int kDU = 7;    // Action dimension (7-DOF arm)
constexpr int kDV = 7;    // Adversary dimension (same as action)
constexpr double kTol = 1e-10;  // Numerical tolerance

// =============================================================================
// Utility Functions for Deterministic Test Data
// =============================================================================

/**
 * @brief Generate deterministic "random" values for reproducible tests.
 *
 * Uses a simple linear congruential generator for reproducibility.
 */
class DeterministicRNG {
public:
    explicit DeterministicRNG(uint32_t seed = 42) : state_(seed) {}

    double next() {
        state_ = state_ * 1103515245 + 12345;
        return static_cast<double>(state_ % 10000) / 10000.0 - 0.5;
    }

    Vector vector(int size) {
        Vector v(size);
        for (int i = 0; i < size; ++i) {
            v(i) = next();
        }
        return v;
    }

    Matrix matrix(int rows, int cols) {
        Matrix m(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m(i, j) = next();
            }
        }
        return m;
    }

private:
    uint32_t state_;
};

/**
 * @brief Compute Cholesky factor ensuring positive definiteness.
 */
Matrix make_spd_matrix(int dim, double scale = 1.0) {
    DeterministicRNG rng(123);
    Matrix A = rng.matrix(dim, dim);
    // A * A^T is guaranteed positive semi-definite
    // Add identity for positive definiteness
    return (A * A.transpose() + Matrix::Identity(dim, dim) * dim) * scale;
}

// =============================================================================
// LinearGaussianPolicy Tests - Numerical Validation
// =============================================================================

class LinearGaussianPolicyParityTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr double init_var = 1.0;

    void SetUp() override {
        rng_ = std::make_unique<DeterministicRNG>(42);
    }

    std::unique_ptr<DeterministicRNG> rng_;
};

/**
 * @test Verifies the action formula: U = K*x + k + L^T * noise
 *
 * Both Python and C++ now use the same formula:
 * Python: u += self.chol_pol_covar[t].T.dot(noise)
 * C++:    u += chol_pol_covar_[t].transpose() * (*noise)
 *
 * This test verifies the C++ implementation matches the Python convention.
 */
TEST_F(LinearGaussianPolicyParityTest, ActFormulaWithCholTranspose) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    // Set up non-trivial policy parameters
    int t = 5;
    policy.K()[t] = rng_->matrix(dU, dX);
    policy.k().row(t) = rng_->vector(dU).transpose();

    // Create non-identity covariance
    Matrix cov = make_spd_matrix(dU, init_var);
    Eigen::LLT<Matrix> llt(cov);
    Matrix L = llt.matrixL();

    policy.pol_covar()[t] = cov;
    policy.chol_pol_covar()[t] = L;

    Vector x = rng_->vector(dX);
    Vector noise = rng_->vector(dU);

    // Current C++ implementation (now uses L^T * noise)
    Vector u_cpp = policy.act(&x, nullptr, t, &noise);

    // Expected value per Python convention (with transpose)
    Vector u_expected = policy.K()[t] * x +
                        policy.k().row(t).transpose() +
                        L.transpose() * noise;  // Note: L^T

    // C++ should now match Python exactly
    EXPECT_TRUE(u_cpp.isApprox(u_expected, kTol))
        << "C++ implementation should match Python: U = K*x + k + L^T*noise";
}

/**
 * @test Verifies mean action (no noise): U_mean = K*x + k
 */
TEST_F(LinearGaussianPolicyParityTest, MeanActionFormula) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    int t = 3;
    policy.K()[t] = rng_->matrix(dU, dX);
    policy.k().row(t) = rng_->vector(dU).transpose();

    Vector x = rng_->vector(dX);

    Vector u_mean = policy.mean_action(x, t);
    Vector expected = policy.K()[t] * x + policy.k().row(t).transpose();

    EXPECT_TRUE(u_mean.isApprox(expected, kTol))
        << "Mean action should be K*x + k";
}

/**
 * @test Verifies fold_k formula: k_folded = k + L^T * noise
 *
 * Python: k[i] = scaled_noise + self.k[i] where scaled_noise = chol[i].T.dot(noise[i])
 * C++:    result.row(t) += (chol[t].transpose() * noise.row(t).transpose()).transpose()
 *
 * Both now use L^T (transpose) consistently.
 */
TEST_F(LinearGaussianPolicyParityTest, FoldKFormula) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    // Initialize with deterministic values
    for (int t = 0; t < T; ++t) {
        policy.k().row(t) = rng_->vector(dU).transpose();
        Matrix cov = make_spd_matrix(dU, 0.5);
        Eigen::LLT<Matrix> llt(cov);
        policy.chol_pol_covar()[t] = llt.matrixL();
    }

    Matrix noise = rng_->matrix(T, dU);
    Matrix folded = policy.fold_k(noise);

    // Verify C++ formula matches Python: k + L^T * noise
    for (int t = 0; t < T; ++t) {
        Vector expected = policy.k().row(t).transpose() +
                          policy.chol_pol_covar()[t].transpose() * noise.row(t).transpose();
        EXPECT_TRUE(folded.row(t).transpose().isApprox(expected, kTol))
            << "fold_k should compute k + L^T*noise at t=" << t;
    }
}

/**
 * @test Verifies log probability computation
 *
 * log p(u|x) = -0.5 * (u - mu)^T * Sigma^-1 * (u - mu) - 0.5 * log|Sigma| - (dU/2) * log(2*pi)
 */
TEST_F(LinearGaussianPolicyParityTest, LogProbFormula) {
    LinearGaussianPolicy policy(T, dX, dU, init_var);

    int t = 2;
    policy.K()[t] = rng_->matrix(dU, dX);
    policy.k().row(t) = rng_->vector(dU).transpose();

    Matrix cov = make_spd_matrix(dU, init_var);
    Eigen::LLT<Matrix> llt(cov);
    Matrix L = llt.matrixL();

    policy.pol_covar()[t] = cov;
    policy.chol_pol_covar()[t] = L;
    policy.inv_pol_covar()[t] = cov.inverse();

    Vector x = rng_->vector(dX);
    Vector mean = policy.K()[t] * x + policy.k().row(t).transpose();
    Vector u = mean + rng_->vector(dU) * 0.1;  // Slight deviation from mean

    double log_p = policy.log_prob(x, u, t);

    // Manual computation
    Vector diff = u - mean;
    double quad_form = diff.transpose() * cov.inverse() * diff;
    double log_det = std::log(cov.determinant());
    double expected = -0.5 * quad_form - 0.5 * log_det -
                      0.5 * dU * std::log(2.0 * M_PI);

    EXPECT_NEAR(log_p, expected, 1e-8)
        << "Log probability should match Gaussian formula";
}

// =============================================================================
// CostAction Tests - Numerical Validation
// =============================================================================

class CostActionParityTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr int dV = 2;

    void SetUp() override {
        rng_ = std::make_unique<DeterministicRNG>(123);
    }

    Sample create_sample() {
        Sample sample(T, dX, dU, dV);
        sample.set(SampleType::JOINT_ANGLES, rng_->matrix(T, dX));
        sample.set(SampleType::ACTION, rng_->matrix(T, dU));
        sample.set(SampleType::NOISE, rng_->matrix(T, dV));
        sample.set_state_types({SampleType::JOINT_ANGLES});
        return sample;
    }

    std::unique_ptr<DeterministicRNG> rng_;
};

/**
 * @test Verifies protagonist cost formula: l = 0.5 * sum(wu * u^2)
 *
 * Python: l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
 * C++:    l = 0.5 * diff^T * W * diff (where W = diag(wu), diff = u - target)
 *
 * These are equivalent when target = 0.
 */
TEST_F(CostActionParityTest, ProtagonistCostFormula) {
    Vector wu(dU);
    wu << 1.0, 2.0, 0.5;  // Non-uniform weights

    CostAction cost(wu, 1.0, CostMode::PROTAGONIST);

    Sample sample = create_sample();
    CostResult result = cost.eval(sample);

    Matrix U = sample.get_U();
    Matrix Wu = wu.asDiagonal();

    for (int t = 0; t < T; ++t) {
        Vector u = U.row(t).transpose();

        // Python formula (element-wise then sum)
        double l_python = 0.5 * (wu.array() * u.array().square()).sum();

        // C++ formula (quadratic form)
        double l_cpp = 0.5 * u.transpose() * Wu * u;

        // Both should match
        EXPECT_NEAR(l_python, l_cpp, kTol)
            << "Python and C++ formulas should be equivalent at t=" << t;

        // C++ result should match
        EXPECT_NEAR(result.l(t), l_cpp, kTol)
            << "CostAction.eval() should match formula at t=" << t;
    }
}

/**
 * @test Verifies protagonist gradient: lu = wu * u
 *
 * Python: lu = self._hyperparams['wu'] * sample_u
 * C++:    lu = W * diff (where diff = u - target)
 */
TEST_F(CostActionParityTest, ProtagonistGradientFormula) {
    Vector wu(dU);
    wu << 1.0, 2.0, 0.5;

    CostAction cost(wu, 1.0, CostMode::PROTAGONIST);

    Sample sample = create_sample();
    CostResult result = cost.eval(sample);

    Matrix U = sample.get_U();

    for (int t = 0; t < T; ++t) {
        Vector u = U.row(t).transpose();

        // Python formula: element-wise multiplication
        Vector lu_python = wu.array() * u.array();

        // C++ formula: W * u
        Matrix Wu = wu.asDiagonal();
        Vector lu_cpp = Wu * u;

        // Both should match
        EXPECT_TRUE(lu_python.isApprox(lu_cpp, kTol))
            << "Gradient formulas should match at t=" << t;

        // Result should match
        EXPECT_TRUE(result.lu.row(t).transpose().isApprox(lu_cpp, kTol))
            << "CostAction gradient should match at t=" << t;
    }
}

/**
 * @test Verifies protagonist Hessian: luu = diag(wu)
 *
 * Python: luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
 * C++:    luu = W (diagonal matrix)
 */
TEST_F(CostActionParityTest, ProtagonistHessianFormula) {
    Vector wu(dU);
    wu << 1.0, 2.0, 0.5;

    CostAction cost(wu, 1.0, CostMode::PROTAGONIST);

    Sample sample = create_sample();
    CostResult result = cost.eval(sample);

    Matrix Wu = wu.asDiagonal();

    for (int t = 0; t < T; ++t) {
        EXPECT_TRUE(result.luu[t].isApprox(Wu, kTol))
            << "Hessian should be diag(wu) at t=" << t;
    }
}

/**
 * @test Verifies cost is zero when actions are zero: l = 0.5 * u^T * W * u = 0
 */
TEST_F(CostActionParityTest, CostZeroAtOrigin) {
    Vector wu(dU);
    wu << 1.0, 2.0, 0.5;

    CostAction cost(wu, 1.0, CostMode::PROTAGONIST);

    // Create sample with zero actions
    Sample sample(T, dX, dU);
    sample.set(SampleType::JOINT_ANGLES, rng_->matrix(T, dX));
    sample.set(SampleType::ACTION, Matrix::Zero(T, dU));
    sample.set_state_types({SampleType::JOINT_ANGLES});

    CostResult result = cost.eval(sample);

    // Cost should be zero when actions are zero
    for (int t = 0; t < T; ++t) {
        EXPECT_NEAR(result.l(t), 0.0, kTol)
            << "Cost should be zero when actions are zero at t=" << t;
    }
}

/**
 * @test Verifies antagonist mode game-theoretic formulation.
 *
 * C++ now implements the Python game-theoretic formulation:
 *   l = 0.5 * sum(wu * prot_u^2) - gamma * sum(wu * ant_u^2)
 *   Returns NEGATED values (maximization objective)
 *
 * This test verifies eval_with_protagonist() behavior.
 */
TEST_F(CostActionParityTest, AntagonistModeGameTheoretic) {
    Vector wu(dU);
    wu << 1.0, 2.0, 0.5;
    double gamma = 0.5;

    CostAction cost(wu, gamma, CostMode::ANTAGONIST);

    // Create protagonist and antagonist samples
    Sample sample_prot(T, dX, dU);
    sample_prot.set(SampleType::JOINT_ANGLES, rng_->matrix(T, dX));
    sample_prot.set(SampleType::ACTION, Matrix::Ones(T, dU));  // u_prot = 1
    sample_prot.set_state_types({SampleType::JOINT_ANGLES});

    Sample sample_ant(T, dX, dU);
    sample_ant.set(SampleType::JOINT_ANGLES, rng_->matrix(T, dX));
    sample_ant.set(SampleType::ACTION, Matrix::Ones(T, dU) * 0.5);  // v = 0.5
    sample_ant.set_state_types({SampleType::JOINT_ANGLES});

    CostResult result = cost.eval_with_protagonist(sample_ant, sample_prot);

    // Game-theoretic formula (before negation):
    // l_raw = 0.5 * sum(wu * u_prot^2) - gamma * sum(wu * v^2)
    // For u_prot=1, v=0.5:
    double prot_term = 0.5 * (wu.array() * 1.0).sum();  // 0.5 * 3.5 = 1.75
    double ant_term = gamma * (wu.array() * 0.25).sum();  // 0.5 * 0.875 = 0.4375
    double l_raw = prot_term - ant_term;  // 1.75 - 0.4375 = 1.3125
    double l_expected = -l_raw;  // Negated for maximization

    for (int t = 0; t < T; ++t) {
        EXPECT_NEAR(result.l(t), l_expected, kTol)
            << "Antagonist cost should be negated game-theoretic formula at t=" << t;
    }
}

// =============================================================================
// Sample Data Packing/Unpacking Tests
// =============================================================================

class SampleParityTest : public ::testing::Test {
protected:
    static constexpr int T = 10;
    static constexpr int dX = 7;
    static constexpr int dU = 3;
    static constexpr int dV = 2;

    void SetUp() override {
        rng_ = std::make_unique<DeterministicRNG>(456);
    }

    std::unique_ptr<DeterministicRNG> rng_;
};

/**
 * @test Verifies action data storage and retrieval: U[t] = stored action at t
 */
TEST_F(SampleParityTest, ActionStorageRetrieval) {
    Sample sample(T, dX, dU);

    Matrix U = rng_->matrix(T, dU);
    sample.set(SampleType::ACTION, U);

    // Retrieve all
    Matrix U_retrieved = sample.get_U();
    EXPECT_TRUE(U_retrieved.isApprox(U, kTol))
        << "get_U() should return stored action matrix";

    // Retrieve by timestep
    for (int t = 0; t < T; ++t) {
        Vector u_t = sample.get_U(t);
        EXPECT_TRUE(u_t.isApprox(U.row(t).transpose(), kTol))
            << "get_U(t) should return stored action at t=" << t;
    }
}

/**
 * @test Verifies adversary action storage via NOISE type
 *
 * C++ maps V (adversary actions) to SampleType::NOISE.
 * This test verifies the mapping is consistent.
 */
TEST_F(SampleParityTest, AdversaryActionStorage) {
    Sample sample(T, dX, dU, dV);

    Matrix V = rng_->matrix(T, dV);
    sample.set(SampleType::NOISE, V);  // C++ uses NOISE for V

    // Retrieve via get_V()
    Matrix V_retrieved = sample.get_V();
    EXPECT_TRUE(V_retrieved.isApprox(V, kTol))
        << "get_V() should return stored adversary actions";

    // Retrieve by timestep
    for (int t = 0; t < T; ++t) {
        Vector v_t = sample.get_V(t);
        EXPECT_TRUE(v_t.isApprox(V.row(t).transpose(), kTol))
            << "get_V(t) should return stored adversary action at t=" << t;
    }
}

/**
 * @test Verifies state composition from multiple sensor types
 */
TEST_F(SampleParityTest, StateComposition) {
    Sample sample(T, dX, dU);

    int dJA = 4;  // Joint angles dimension
    int dJV = 3;  // Joint velocities dimension

    Matrix JA = rng_->matrix(T, dJA);
    Matrix JV = rng_->matrix(T, dJV);

    sample.set(SampleType::JOINT_ANGLES, JA);
    sample.set(SampleType::JOINT_VELOCITIES, JV);
    sample.set_state_types({SampleType::JOINT_ANGLES, SampleType::JOINT_VELOCITIES});

    // Composed state should concatenate [JA, JV]
    for (int t = 0; t < T; ++t) {
        Vector x = sample.get_X(t);
        EXPECT_EQ(x.size(), dJA + dJV)
            << "State dimension should be sum of components";

        EXPECT_TRUE(x.head(dJA).isApprox(JA.row(t).transpose(), kTol))
            << "State should start with joint angles at t=" << t;

        EXPECT_TRUE(x.tail(dJV).isApprox(JV.row(t).transpose(), kTol))
            << "State should end with joint velocities at t=" << t;
    }
}

// =============================================================================
// Integration Tests - Full Pipeline Validation
// =============================================================================

class IntegrationParityTest : public ::testing::Test {
protected:
    static constexpr int T = 50;
    static constexpr int dX = 14;
    static constexpr int dU = 7;

    void SetUp() override {
        rng_ = std::make_unique<DeterministicRNG>(789);
    }

    std::unique_ptr<DeterministicRNG> rng_;
};

/**
 * @test Full policy rollout: Simulates trajectory generation
 *
 * Generates a trajectory using the policy and verifies consistency.
 */
TEST_F(IntegrationParityTest, PolicyRollout) {
    // Use a separate RNG for policy initialization (deterministic)
    DeterministicRNG policy_rng(111);

    LinearGaussianPolicy policy(T, dX, dU, 0.1);

    // Initialize policy with some feedback
    for (int t = 0; t < T; ++t) {
        policy.K()[t] = policy_rng.matrix(dU, dX) * 0.1;
        policy.k().row(t) = policy_rng.vector(dU).transpose() * 0.01;
    }

    // Simulate rollout with separate RNG for trajectory
    DeterministicRNG traj_rng1(222);
    Vector x = traj_rng1.vector(dX);
    std::vector<Vector> trajectory;
    trajectory.reserve(T);

    for (int t = 0; t < T; ++t) {
        Vector noise = traj_rng1.vector(dU);
        Vector u = policy.act(&x, nullptr, t, &noise);
        trajectory.push_back(u);

        // Simple dynamics: x_next = x + small_delta (just for testing)
        x = x + traj_rng1.vector(dX) * 0.01;
    }

    // Verify trajectory dimensions
    for (int t = 0; t < T; ++t) {
        EXPECT_EQ(trajectory[t].size(), dU)
            << "Action dimension should be dU at t=" << t;
    }

    // Verify determinism: same seed should give same trajectory
    DeterministicRNG traj_rng2(222);  // Same seed as traj_rng1
    Vector x2 = traj_rng2.vector(dX);

    for (int t = 0; t < T; ++t) {
        Vector noise2 = traj_rng2.vector(dU);
        Vector u2 = policy.act(&x2, nullptr, t, &noise2);
        EXPECT_TRUE(trajectory[t].isApprox(u2, kTol))
            << "Trajectory should be deterministic with same seed at t=" << t;
        x2 = x2 + traj_rng2.vector(dX) * 0.01;
    }
}

/**
 * @test Policy + Cost integration: Compute cost on generated trajectory
 */
TEST_F(IntegrationParityTest, PolicyCostIntegration) {
    LinearGaussianPolicy policy(T, dX, dU, 0.1);

    // Initialize policy
    for (int t = 0; t < T; ++t) {
        policy.K()[t] = rng_->matrix(dU, dX) * 0.1;
    }

    // Generate trajectory
    Sample sample(T, dX, dU);
    Matrix X(T, dX);
    Matrix U(T, dU);

    Vector x = rng_->vector(dX);
    for (int t = 0; t < T; ++t) {
        X.row(t) = x.transpose();
        Vector noise = rng_->vector(dU);
        U.row(t) = policy.act(&x, nullptr, t, &noise).transpose();
        x = x + rng_->vector(dX) * 0.01;
    }

    sample.set(SampleType::JOINT_ANGLES, X);
    sample.set(SampleType::ACTION, U);
    sample.set_state_types({SampleType::JOINT_ANGLES});

    // Evaluate cost
    Vector wu = Vector::Ones(dU);
    CostAction cost(wu);
    CostResult result = cost.eval(sample);

    // Verify cost properties
    double total_cost = result.l.sum();
    EXPECT_GT(total_cost, 0.0) << "Total cost should be positive for non-zero actions";

    // Verify cost increases with larger actions
    Matrix U_large = U * 2.0;
    Sample sample_large(T, dX, dU);
    sample_large.set(SampleType::JOINT_ANGLES, X);
    sample_large.set(SampleType::ACTION, U_large);
    sample_large.set_state_types({SampleType::JOINT_ANGLES});

    CostResult result_large = cost.eval(sample_large);
    double total_cost_large = result_large.l.sum();

    EXPECT_GT(total_cost_large, total_cost)
        << "Doubling actions should increase quadratic cost";

    // For quadratic cost, doubling should give 4x cost
    EXPECT_NEAR(total_cost_large / total_cost, 4.0, 0.01)
        << "Quadratic cost should scale as action^2";
}

}  // namespace test
}  // namespace gps
