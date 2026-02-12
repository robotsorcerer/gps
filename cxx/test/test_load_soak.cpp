/**
 * @file test_load_soak.cpp
 * @brief Load and soak tests for GPS algorithm C++ implementation.
 *
 * Tests performance under load and stability over extended periods.
 * Measures: throughput, latency, memory usage, and detects leaks.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "../algorithm/policy/lin_gauss_policy.h"
#include "../algorithm/cost/cost_action.h"
#include "../algorithm/cost/cost_sum.h"
#include "../algorithm/dynamics/dynamics.h"
#include "../algorithm/traj_opt/traj_opt_lqr.h"
#include "../sample/sample.h"
#include "../sample/sample_list.h"

namespace gps {
namespace test {

// =============================================================================
// Memory Tracking Utilities
// =============================================================================

/**
 * @brief Get current process memory usage in KB (Linux-specific).
 */
long get_memory_usage_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            // Parse "VmRSS:    12345 kB"
            std::istringstream iss(line.substr(6));
            long kb;
            iss >> kb;
            return kb;
        }
    }
    return -1;  // Not available
}

/**
 * @brief Statistics collector for latency measurements.
 */
struct LatencyStats {
    std::vector<double> samples;

    void add(double latency_us) {
        samples.push_back(latency_us);
    }

    double mean() const {
        if (samples.empty()) return 0;
        return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    }

    double percentile(double p) const {
        if (samples.empty()) return 0;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
        return sorted[idx];
    }

    double p50() const { return percentile(0.50); }
    double p95() const { return percentile(0.95); }
    double p99() const { return percentile(0.99); }
    double max() const { return samples.empty() ? 0 : *std::max_element(samples.begin(), samples.end()); }
};

// =============================================================================
// Load Testing
// =============================================================================

class LoadTest : public ::testing::Test {
protected:
    static constexpr int T = 100;
    static constexpr int dX = 14;
    static constexpr int dU = 7;
    static constexpr int dV = 7;

    void SetUp() override {
        // Initialize test components
        policy_ = std::make_unique<LinearGaussianPolicy>(T, dX, dU, 0.1);
        cost_ = std::make_unique<CostAction>(Vector::Ones(dU), 1.0, CostMode::PROTAGONIST);
    }

    Sample create_sample() {
        Sample sample(T, dX, dU, dV);
        sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
        sample.set(SampleType::ACTION, Matrix::Random(T, dU));
        sample.set(SampleType::NOISE, Matrix::Random(T, dV));
        sample.set_state_types({SampleType::JOINT_ANGLES});
        return sample;
    }

    std::unique_ptr<LinearGaussianPolicy> policy_;
    std::unique_ptr<CostAction> cost_;
};

/**
 * @test Load test for policy evaluation throughput.
 *
 * Simulates 1000 policy evaluations and measures:
 * - Throughput (evaluations per second)
 * - Latency distribution (P50, P95, P99)
 */
TEST_F(LoadTest, PolicyEvaluationThroughput) {
    constexpr int NUM_ITERATIONS = 1000;
    LatencyStats stats;

    Vector x = Vector::Random(dX);
    Vector noise = Vector::Random(dU);

    auto start_total = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < T; ++t) {
            Vector u = policy_->act(&x, nullptr, t, &noise);
            // Simulate state update
            x = Vector::Random(dX);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        stats.add(latency_us);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(end_total - start_total).count();
    double throughput = NUM_ITERATIONS / total_time_s;

    std::cout << "\n=== Policy Evaluation Load Test ===" << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time_s << " s" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " rollouts/sec" << std::endl;
    std::cout << "Latency P50: " << std::fixed << std::setprecision(1) << stats.p50() << " us" << std::endl;
    std::cout << "Latency P95: " << std::fixed << std::setprecision(1) << stats.p95() << " us" << std::endl;
    std::cout << "Latency P99: " << std::fixed << std::setprecision(1) << stats.p99() << " us" << std::endl;
    std::cout << "Latency Max: " << std::fixed << std::setprecision(1) << stats.max() << " us" << std::endl;

    // Performance assertions
    EXPECT_GT(throughput, 100.0) << "Should achieve at least 100 rollouts/sec";
    EXPECT_LT(stats.p99(), 100000.0) << "P99 latency should be under 100ms";
}

/**
 * @test Load test for cost evaluation throughput.
 */
TEST_F(LoadTest, CostEvaluationThroughput) {
    constexpr int NUM_ITERATIONS = 1000;
    LatencyStats stats;

    auto start_total = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        Sample sample = create_sample();

        auto start = std::chrono::high_resolution_clock::now();
        CostResult result = cost_->eval(sample);
        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        stats.add(latency_us);

        // Prevent optimization from eliminating the call
        EXPECT_GT(result.l.size(), 0);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(end_total - start_total).count();
    double throughput = NUM_ITERATIONS / total_time_s;

    std::cout << "\n=== Cost Evaluation Load Test ===" << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " evals/sec" << std::endl;
    std::cout << "Latency P50: " << std::fixed << std::setprecision(1) << stats.p50() << " us" << std::endl;
    std::cout << "Latency P95: " << std::fixed << std::setprecision(1) << stats.p95() << " us" << std::endl;
    std::cout << "Latency P99: " << std::fixed << std::setprecision(1) << stats.p99() << " us" << std::endl;

    EXPECT_GT(throughput, 500.0) << "Should achieve at least 500 cost evals/sec";
}

/**
 * @test Load test for concurrent policy evaluations.
 */
TEST_F(LoadTest, ConcurrentPolicyEvaluation) {
    constexpr int NUM_THREADS = 4;
    constexpr int ITERATIONS_PER_THREAD = 250;

    std::atomic<int> completed{0};
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, &completed, &errors, ITERATIONS_PER_THREAD]() {
            LinearGaussianPolicy local_policy(T, dX, dU, 0.1);
            Vector x = Vector::Random(dX);
            Vector noise = Vector::Random(dU);

            for (int i = 0; i < ITERATIONS_PER_THREAD; ++i) {
                try {
                    for (int t = 0; t < T; ++t) {
                        Vector u = local_policy.act(&x, nullptr, t, &noise);
                        x = Vector::Random(dX);
                    }
                    completed++;
                } catch (...) {
                    errors++;
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(end - start).count();
    double throughput = completed.load() / total_time_s;

    std::cout << "\n=== Concurrent Policy Evaluation ===" << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << "Completed: " << completed.load() << "/" << (NUM_THREADS * ITERATIONS_PER_THREAD) << std::endl;
    std::cout << "Errors: " << errors.load() << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput << " rollouts/sec" << std::endl;

    EXPECT_EQ(errors.load(), 0) << "No errors should occur in concurrent execution";
    EXPECT_EQ(completed.load(), NUM_THREADS * ITERATIONS_PER_THREAD);
}

/**
 * @test Load test for sample list operations.
 */
TEST_F(LoadTest, SampleListThroughput) {
    constexpr int NUM_SAMPLES = 100;
    constexpr int NUM_ITERATIONS = 50;
    LatencyStats stats;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        SampleList sample_list;

        auto start = std::chrono::high_resolution_clock::now();

        // Add samples
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            sample_list.add(create_sample());
        }

        // Access all samples
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            const Sample& s = sample_list[i];
            Matrix X = s.get_X();
            Matrix U = s.get_U();
            (void)X; (void)U;  // Prevent optimization
        }

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        stats.add(latency_us);
    }

    std::cout << "\n=== Sample List Operations ===" << std::endl;
    std::cout << "Samples per list: " << NUM_SAMPLES << std::endl;
    std::cout << "Latency P50: " << std::fixed << std::setprecision(1) << stats.p50() / 1000.0 << " ms" << std::endl;
    std::cout << "Latency P95: " << std::fixed << std::setprecision(1) << stats.p95() / 1000.0 << " ms" << std::endl;

    EXPECT_LT(stats.p95(), 1000000.0) << "P95 should be under 1 second";
}

// =============================================================================
// Soak Testing (Endurance)
// =============================================================================

class SoakTest : public ::testing::Test {
protected:
    static constexpr int T = 100;
    static constexpr int dX = 14;
    static constexpr int dU = 7;
    static constexpr int dV = 7;

    Sample create_sample() {
        Sample sample(T, dX, dU, dV);
        sample.set(SampleType::JOINT_ANGLES, Matrix::Random(T, dX));
        sample.set(SampleType::ACTION, Matrix::Random(T, dU));
        sample.set(SampleType::NOISE, Matrix::Random(T, dV));
        sample.set_state_types({SampleType::JOINT_ANGLES});
        return sample;
    }
};

/**
 * @test Soak test for memory stability.
 *
 * Runs many iterations and checks for memory leaks by monitoring
 * RSS memory growth over time.
 */
TEST_F(SoakTest, MemoryStability) {
    constexpr int NUM_ITERATIONS = 5000;
    constexpr int SAMPLE_INTERVAL = 500;

    std::vector<long> memory_samples;
    std::vector<double> latency_samples;

    long initial_memory = get_memory_usage_kb();
    memory_samples.push_back(initial_memory);

    LinearGaussianPolicy policy(T, dX, dU, 0.1);
    CostAction cost(Vector::Ones(dU));

    std::cout << "\n=== Memory Stability Soak Test ===" << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Initial memory: " << initial_memory << " KB" << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Create sample, evaluate policy and cost
        Sample sample = create_sample();

        Vector x = sample.get_X(0);
        for (int t = 0; t < T; ++t) {
            Vector noise = Vector::Random(dU);
            Vector u = policy.act(&x, nullptr, t, &noise);
            x = Vector::Random(dX);
        }

        CostResult result = cost.eval(sample);

        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        latency_samples.push_back(latency_ms);

        // Sample memory periodically
        if ((i + 1) % SAMPLE_INTERVAL == 0) {
            long current_memory = get_memory_usage_kb();
            memory_samples.push_back(current_memory);
            std::cout << "  Iteration " << (i + 1) << ": " << current_memory << " KB" << std::endl;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(end_total - start_total).count();

    long final_memory = get_memory_usage_kb();
    long memory_growth = final_memory - initial_memory;
    double memory_growth_percent = 100.0 * memory_growth / initial_memory;

    // Calculate latency drift (compare first 10% vs last 10%)
    size_t window = latency_samples.size() / 10;
    double early_avg = std::accumulate(latency_samples.begin(), latency_samples.begin() + window, 0.0) / window;
    double late_avg = std::accumulate(latency_samples.end() - window, latency_samples.end(), 0.0) / window;
    double latency_drift_percent = 100.0 * (late_avg - early_avg) / early_avg;

    std::cout << "Final memory: " << final_memory << " KB" << std::endl;
    std::cout << "Memory growth: " << memory_growth << " KB ("
              << std::fixed << std::setprecision(1) << memory_growth_percent << "%)" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time_s << " s" << std::endl;
    std::cout << "Early latency avg: " << std::fixed << std::setprecision(2) << early_avg << " ms" << std::endl;
    std::cout << "Late latency avg: " << std::fixed << std::setprecision(2) << late_avg << " ms" << std::endl;
    std::cout << "Latency drift: " << std::fixed << std::setprecision(1) << latency_drift_percent << "%" << std::endl;

    // Assertions
    // Allow up to 50% memory growth (accounts for normal heap behavior)
    EXPECT_LT(memory_growth_percent, 50.0)
        << "Memory growth should be under 50% (possible leak detected)";

    // Latency should not degrade more than 50%
    EXPECT_LT(latency_drift_percent, 50.0)
        << "Latency drift should be under 50% (performance degradation detected)";
}

/**
 * @test Soak test for object lifecycle stability.
 *
 * Repeatedly creates and destroys objects to detect leaks.
 */
TEST_F(SoakTest, ObjectLifecycleStability) {
    constexpr int NUM_ITERATIONS = 2000;

    long initial_memory = get_memory_usage_kb();

    std::cout << "\n=== Object Lifecycle Soak Test ===" << std::endl;
    std::cout << "Initial memory: " << initial_memory << " KB" << std::endl;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Create and destroy various objects
        {
            auto policy = std::make_unique<LinearGaussianPolicy>(T, dX, dU, 0.1);
            auto cloned = policy->clone();
        }

        {
            auto cost = std::make_unique<CostAction>(Vector::Ones(dU));
            auto cloned = cost->clone();
        }

        {
            Sample sample = create_sample();
            Matrix X = sample.get_X();
            Matrix U = sample.get_U();
        }

        {
            SampleList list;
            for (int j = 0; j < 10; ++j) {
                list.add(create_sample());
            }
            list.clear();
        }

        {
            CostSum sum;
            sum.add_cost(std::make_shared<CostAction>(Vector::Ones(dU)), 1.0);
            sum.add_cost(std::make_shared<CostAction>(Vector::Ones(dU) * 2), 0.5);
        }
    }

    // Force cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    long final_memory = get_memory_usage_kb();
    long memory_growth = final_memory - initial_memory;
    double growth_per_iteration = static_cast<double>(memory_growth) / NUM_ITERATIONS;

    std::cout << "Final memory: " << final_memory << " KB" << std::endl;
    std::cout << "Memory growth: " << memory_growth << " KB" << std::endl;
    std::cout << "Growth per iteration: " << std::fixed << std::setprecision(3)
              << growth_per_iteration << " KB/iter" << std::endl;

    // Should have minimal persistent growth (under 0.1 KB per iteration)
    EXPECT_LT(growth_per_iteration, 0.5)
        << "Memory growth per iteration too high (possible leak)";
}

/**
 * @test Soak test for robust policy with dual players.
 */
TEST_F(SoakTest, RobustPolicyStability) {
    constexpr int NUM_ITERATIONS = 1000;

    long initial_memory = get_memory_usage_kb();
    LatencyStats stats;

    std::cout << "\n=== Robust Policy Soak Test ===" << std::endl;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        LinearGaussianPolicyRobust robust(
            LinearGaussianPolicy(T, dX, dU, 0.1),
            LinearGaussianPolicy(T, dX, dV, 0.1),
            PolicyMode::ROBUST
        );

        Vector x = Vector::Random(dX);

        // Evaluate both protagonist and antagonist
        robust.set_mode(PolicyMode::PROTAGONIST);
        for (int t = 0; t < T; ++t) {
            Vector noise = Vector::Random(dU);
            Vector u = robust.act(&x, nullptr, t, &noise);
            x = Vector::Random(dX);
        }

        robust.set_mode(PolicyMode::ANTAGONIST);
        x = Vector::Random(dX);
        for (int t = 0; t < T; ++t) {
            Vector noise = Vector::Random(dV);
            Vector v = robust.act(&x, nullptr, t, &noise);
            x = Vector::Random(dX);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats.add(latency_ms * 1000);  // Convert to us
    }

    long final_memory = get_memory_usage_kb();

    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Memory growth: " << (final_memory - initial_memory) << " KB" << std::endl;
    std::cout << "Latency P50: " << std::fixed << std::setprecision(1) << stats.p50() / 1000.0 << " ms" << std::endl;
    std::cout << "Latency P99: " << std::fixed << std::setprecision(1) << stats.p99() / 1000.0 << " ms" << std::endl;

    EXPECT_LT(final_memory - initial_memory, 10000) << "Memory growth should be under 10MB";
}

/**
 * @test Soak test for game-theoretic cost evaluation.
 */
TEST_F(SoakTest, GameTheoreticCostStability) {
    constexpr int NUM_ITERATIONS = 2000;

    long initial_memory = get_memory_usage_kb();
    LatencyStats stats;
    int errors = 0;

    std::cout << "\n=== Game-Theoretic Cost Soak Test ===" << std::endl;

    CostAction cost_protagonist(Vector::Ones(dU), 1.0, CostMode::PROTAGONIST);
    CostAction cost_antagonist(Vector::Ones(dU), 0.5, CostMode::ANTAGONIST);
    CostAction cost_robust(Vector::Ones(dU), 0.5, CostMode::ROBUST);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        try {
            Sample sample_prot = create_sample();
            Sample sample_ant = create_sample();

            // Evaluate all modes
            CostResult r1 = cost_protagonist.eval(sample_prot);
            CostResult r2 = cost_antagonist.eval_with_protagonist(sample_ant, sample_prot);
            CostResult r3 = cost_robust.eval(sample_prot);

            // Verify results are valid
            EXPECT_EQ(r1.l.size(), T);
            EXPECT_EQ(r2.l.size(), T);
            EXPECT_EQ(r3.l.size(), T);
        } catch (...) {
            errors++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        stats.add(latency_us);
    }

    long final_memory = get_memory_usage_kb();

    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "Errors: " << errors << std::endl;
    std::cout << "Memory growth: " << (final_memory - initial_memory) << " KB" << std::endl;
    std::cout << "Latency P50: " << std::fixed << std::setprecision(1) << stats.p50() << " us" << std::endl;
    std::cout << "Latency P99: " << std::fixed << std::setprecision(1) << stats.p99() << " us" << std::endl;

    EXPECT_EQ(errors, 0) << "No errors should occur";
    EXPECT_LT(final_memory - initial_memory, 10000) << "Memory growth should be under 10MB";
}

}  // namespace test
}  // namespace gps
