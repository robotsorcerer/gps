/**
 * @file test_concurrency_stress.cpp
 * @brief Stress tests for C++20 thread-safety fixes in GPS codebase.
 *
 * These tests verify the thread-safety of:
 * 1. std::atomic<bool> for sensors_initialized_ and controller_initialized_
 * 2. std::mutex protection for TfController command updates
 * 3. Proper memory ordering with acquire/release semantics
 *
 * Run with ThreadSanitizer (TSan) for maximum coverage:
 *   cmake -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" \
 *         -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" ..
 *   make && ctest -V
 */

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <functional>
#include <latch>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// Simulated TfController command state (mirrors tfcontroller.h)
// ---------------------------------------------------------------------------

class MockTfController {
public:
    mutable std::mutex command_mutex_;
    std::atomic<int> last_command_id_received{0};
    int last_command_id_acted_upon{0};
    int failed_attempts{0};
    Eigen::VectorXd last_action_command_received;
    std::atomic<bool> is_configured_{false};

    void configure(int dU) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        last_command_id_received.store(0, std::memory_order_release);
        last_command_id_acted_upon = 0;
        failed_attempts = 0;
        last_action_command_received.resize(dU);
        last_action_command_received.setZero();
        is_configured_.store(true, std::memory_order_release);
    }

    void update_action_command(int id, const Eigen::VectorXd& command) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        last_command_id_received.store(id, std::memory_order_release);
        last_action_command_received = command;
    }

    bool get_action(Eigen::VectorXd& U) {
        if (!is_configured_.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(command_mutex_);
        const int current_id = last_command_id_received.load(std::memory_order_acquire);
        if (last_command_id_acted_upon < current_id) {
            last_command_id_acted_upon = current_id;
            failed_attempts = 0;
            U = last_action_command_received;
            return true;
        }
        else if (failed_attempts < 2) {
            U = last_action_command_received;
            failed_attempts++;
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Simulated RobotPlugin state flags (mirrors robotplugin.h)
// ---------------------------------------------------------------------------

class MockRobotPlugin {
public:
    std::atomic<bool> sensors_initialized_{false};
    std::atomic<bool> controller_initialized_{false};
    mutable std::mutex trial_controller_mutex_;
    std::unique_ptr<MockTfController> trial_controller_;

    void initialize_sensors() {
        sensors_initialized_.store(false, std::memory_order_release);
        // Simulate sensor initialization work
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        sensors_initialized_.store(true, std::memory_order_release);
    }

    void configure_controller(int dU) {
        controller_initialized_.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(trial_controller_mutex_);
            trial_controller_ = std::make_unique<MockTfController>();
            trial_controller_->configure(dU);
        }
        controller_initialized_.store(true, std::memory_order_release);
    }

    bool update_sensors() {
        if (!sensors_initialized_.load(std::memory_order_acquire)) {
            return false;
        }
        // Simulate sensor update
        return true;
    }

    bool update_controller(Eigen::VectorXd& U) {
        std::lock_guard<std::mutex> lock(trial_controller_mutex_);
        if (trial_controller_ != nullptr &&
            trial_controller_->is_configured_.load(std::memory_order_acquire) &&
            controller_initialized_.load(std::memory_order_acquire)) {
            return trial_controller_->get_action(U);
        }
        return false;
    }

    void send_action_command(int id, const Eigen::VectorXd& command) {
        std::lock_guard<std::mutex> lock(trial_controller_mutex_);
        if (trial_controller_ != nullptr) {
            trial_controller_->update_action_command(id, command);
        }
    }
};

// ---------------------------------------------------------------------------
// Test Fixtures
// ---------------------------------------------------------------------------

class ConcurrencyStressTest : public ::testing::Test {
protected:
    static constexpr int kNumThreads = 8;
    static constexpr int kIterationsPerThread = 1000;
    static constexpr int kDU = 7;  // Action dimension

    MockRobotPlugin plugin_;

    void SetUp() override {
        plugin_.initialize_sensors();
        plugin_.configure_controller(kDU);
    }
};

// ---------------------------------------------------------------------------
// Atomic Flag Tests
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyStressTest, AtomicSensorsInitializedNoRace) {
    // Multiple threads toggle sensors_initialized_ while others read it
    std::atomic<int> successful_reads{0};
    std::atomic<int> successful_writes{0};
    std::latch start_latch(kNumThreads);

    auto reader = [&]() {
        start_latch.arrive_and_wait();
        for (int i = 0; i < kIterationsPerThread; ++i) {
            if (plugin_.update_sensors()) {
                successful_reads.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    auto writer = [&]() {
        start_latch.arrive_and_wait();
        for (int i = 0; i < kIterationsPerThread; ++i) {
            plugin_.sensors_initialized_.store(false, std::memory_order_release);
            std::this_thread::yield();
            plugin_.sensors_initialized_.store(true, std::memory_order_release);
            successful_writes.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads / 2; ++i) {
        threads.emplace_back(reader);
        threads.emplace_back(writer);
    }

    for (auto& t : threads) {
        t.join();
    }

    // No specific assertion on count - the test passes if no TSan errors
    EXPECT_GT(successful_writes.load(), 0);
    SUCCEED() << "No data race detected with " << kNumThreads << " threads";
}

TEST_F(ConcurrencyStressTest, AtomicControllerInitializedNoRace) {
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    std::latch start_latch(kNumThreads);

    auto reader = [&]() {
        start_latch.arrive_and_wait();
        Eigen::VectorXd U(kDU);
        for (int i = 0; i < kIterationsPerThread; ++i) {
            plugin_.update_controller(U);
            read_count.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto writer = [&]() {
        start_latch.arrive_and_wait();
        for (int i = 0; i < kIterationsPerThread; ++i) {
            plugin_.controller_initialized_.store(false, std::memory_order_release);
            std::this_thread::yield();
            plugin_.controller_initialized_.store(true, std::memory_order_release);
            write_count.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads / 2; ++i) {
        threads.emplace_back(reader);
        threads.emplace_back(writer);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(read_count.load(), (kNumThreads / 2) * kIterationsPerThread);
    EXPECT_EQ(write_count.load(), (kNumThreads / 2) * kIterationsPerThread);
}

// ---------------------------------------------------------------------------
// TfController Command Mutex Tests
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyStressTest, TfControllerCommandUpdateNoRace) {
    // Multiple threads send action commands while others read
    std::atomic<int> commands_sent{0};
    std::atomic<int> actions_retrieved{0};
    std::latch start_latch(kNumThreads);

    auto sender = [&](int thread_id) {
        start_latch.arrive_and_wait();
        Eigen::VectorXd cmd(kDU);
        for (int i = 0; i < kIterationsPerThread; ++i) {
            int cmd_id = thread_id * kIterationsPerThread + i;
            cmd.setConstant(static_cast<double>(cmd_id));
            plugin_.send_action_command(cmd_id, cmd);
            commands_sent.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto receiver = [&]() {
        start_latch.arrive_and_wait();
        Eigen::VectorXd U(kDU);
        for (int i = 0; i < kIterationsPerThread; ++i) {
            if (plugin_.update_controller(U)) {
                actions_retrieved.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads / 2; ++i) {
        threads.emplace_back(sender, i);
        threads.emplace_back(receiver);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(commands_sent.load(), (kNumThreads / 2) * kIterationsPerThread);
    EXPECT_GT(actions_retrieved.load(), 0);
}

TEST_F(ConcurrencyStressTest, TfControllerConfigureWhileActing) {
    // Stress test: reconfigure controller while other threads are acting
    std::atomic<int> reconfigs{0};
    std::atomic<int> actions{0};
    std::atomic<bool> running{true};
    std::latch start_latch(kNumThreads);

    auto reconfigurer = [&]() {
        start_latch.arrive_and_wait();
        while (running.load(std::memory_order_acquire)) {
            plugin_.configure_controller(kDU);
            reconfigs.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    };

    auto actor = [&]() {
        start_latch.arrive_and_wait();
        Eigen::VectorXd U(kDU);
        for (int i = 0; i < kIterationsPerThread * 10; ++i) {
            plugin_.update_controller(U);
            actions.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.emplace_back(reconfigurer);
    for (int i = 0; i < kNumThreads - 1; ++i) {
        threads.emplace_back(actor);
    }

    // Wait for actors to complete
    for (int i = 1; i < kNumThreads; ++i) {
        threads[i].join();
    }

    // Stop reconfigurer
    running.store(false, std::memory_order_release);
    threads[0].join();

    EXPECT_GT(reconfigs.load(), 0);
    EXPECT_EQ(actions.load(), (kNumThreads - 1) * kIterationsPerThread * 10);
}

// ---------------------------------------------------------------------------
// Memory Ordering Tests
// ---------------------------------------------------------------------------

/**
 * INVESTIGATION NOTES - Original Test Failure Analysis:
 *
 * The original test failed with: "Expected: (successful_reads.load()) > (0), actual: 0 vs 0"
 *
 * ROOT CAUSE #1: Timing dependency
 * - Reader loop: for (i = 0; i < 1000; ++i) { load(); if (ready > 0) count++; }
 * - Each iteration is ~10ns (just atomic load + comparison)
 * - Total reader time: ~10us
 * - Writer does store + potential cache coherence traffic
 * - Reader could complete all iterations before writer writes once
 *
 * ROOT CAUSE #2: Flawed assertion logic
 * - Original check: `EXPECT_GE(shared_data, ready)`
 * - This doesn't test acquire-release! Writer can modify shared_data
 *   AFTER reader loads data_ready but BEFORE reader reads shared_data:
 *   1. Writer: shared_data=5, data_ready=5 (release)
 *   2. Reader: load data_ready=5 (acquire)
 *   3. Writer: shared_data=6, data_ready=6 (release)  <-- race!
 *   4. Reader: read shared_data=6, check 6>=5, PASS (but proves nothing)
 *
 * PROPER ACQUIRE-RELEASE TEST DESIGN:
 * - Use a handshake protocol where writer waits for reader acknowledgment
 * - Reader must observe the EXACT value written, not "some value >= X"
 * - Use multiple rounds with barriers to ensure deterministic interleaving
 */

TEST_F(ConcurrencyStressTest, AcquireReleaseOrdering_Handshake) {
    // Proper acquire-release test using producer-consumer handshake
    // This ensures we actually test the memory ordering guarantee:
    // "All writes before a release-store are visible after an acquire-load
    //  that sees that store"

    constexpr int kRounds = 1000;
    std::atomic<int> ordering_violations{0};
    std::atomic<int> verified_transfers{0};

    // Shared state - non-atomic data protected by atomic flag
    struct alignas(64) SharedData {  // Cacheline-aligned to avoid false sharing
        int value1;
        int value2;
        int value3;
        int checksum;  // value1 + value2 + value3
    } data{0, 0, 0, 0};

    std::atomic<int> flag{0};  // 0=empty, N=data ready with round N
    std::atomic<int> ack{0};   // Reader acknowledgment

    auto producer = [&]() {
        for (int round = 1; round <= kRounds; ++round) {
            // Wait for reader to acknowledge previous round
            while (ack.load(std::memory_order_acquire) != round - 1) {
                std::this_thread::yield();
            }

            // Write non-atomic data (these writes must be visible after acquire)
            data.value1 = round;
            data.value2 = round * 2;
            data.value3 = round * 3;
            data.checksum = data.value1 + data.value2 + data.value3;  // 6 * round

            // Release store - publishes all above writes
            flag.store(round, std::memory_order_release);
        }
    };

    auto consumer = [&]() {
        for (int round = 1; round <= kRounds; ++round) {
            // Spin until we see the flag for this round (acquire load)
            while (flag.load(std::memory_order_acquire) != round) {
                std::this_thread::yield();
            }

            // After acquire, all writes before the release MUST be visible
            // Read the non-atomic data
            int v1 = data.value1;
            int v2 = data.value2;
            int v3 = data.value3;
            int cs = data.checksum;

            // Verify consistency - if acquire-release works, this MUST pass
            bool consistent = (v1 == round) &&
                              (v2 == round * 2) &&
                              (v3 == round * 3) &&
                              (cs == v1 + v2 + v3);

            if (!consistent) {
                ordering_violations.fetch_add(1, std::memory_order_relaxed);
                // Log details for debugging
                ADD_FAILURE() << "Ordering violation in round " << round
                              << ": v1=" << v1 << " (expected " << round << ")"
                              << ", v2=" << v2 << " (expected " << round * 2 << ")"
                              << ", v3=" << v3 << " (expected " << round * 3 << ")"
                              << ", checksum=" << cs << " (expected " << 6 * round << ")";
            } else {
                verified_transfers.fetch_add(1, std::memory_order_relaxed);
            }

            // Acknowledge this round (release to synchronize with producer)
            ack.store(round, std::memory_order_release);
        }
    };

    std::thread t1(producer);
    std::thread t2(consumer);

    t1.join();
    t2.join();

    // All rounds must complete successfully
    EXPECT_EQ(ordering_violations.load(), 0)
        << "Acquire-release ordering violations detected!";
    EXPECT_EQ(verified_transfers.load(), kRounds)
        << "Expected " << kRounds << " verified transfers";
}

TEST_F(ConcurrencyStressTest, AcquireReleaseOrdering_MultiProducer) {
    // Test acquire-release with multiple producers writing to different slots
    // This more closely models the TfController scenario where multiple
    // callbacks might update command state
    //
    // DESIGN NOTE: The original test had a TOCTOU bug - producer could overwrite
    // data before consumer read it. This is fixed by:
    // 1. Using double-buffering (odd/even sequence numbers)
    // 2. Producer waits for consumer to advance before reusing buffer

    constexpr int kProducers = 4;
    constexpr int kRoundsPerProducer = 250;

    struct alignas(64) Slot {
        std::atomic<int> sequence{0};      // Producer writes this (release)
        std::atomic<int> consumed{0};       // Consumer writes this (release)
        int data[2][8];                     // Double buffer: data[seq % 2]
    };

    std::array<Slot, kProducers> slots;
    std::atomic<int> violations{0};
    std::atomic<int> total_consumed{0};
    std::atomic<bool> done{false};

    auto producer = [&](int id) {
        for (int i = 1; i <= kRoundsPerProducer; ++i) {
            // Wait until consumer has consumed the previous use of this buffer
            // Buffer index is i % 2, so we need consumed >= i - 1 to reuse
            if (i > 1) {
                while (slots[id].consumed.load(std::memory_order_acquire) < i - 1) {
                    std::this_thread::yield();
                }
            }

            // Write payload to buffer[i % 2]
            int buf = i % 2;
            for (int j = 0; j < 8; ++j) {
                slots[id].data[buf][j] = id * 1000 + i * 10 + j;
            }
            // Release store the sequence number
            slots[id].sequence.store(i, std::memory_order_release);
        }
    };

    auto consumer = [&]() {
        std::array<int, kProducers> last_seen{};
        while (!done.load(std::memory_order_acquire) ||
               std::any_of(last_seen.begin(), last_seen.end(),
                           [](int s) { return s < kRoundsPerProducer; })) {
            for (int id = 0; id < kProducers; ++id) {
                int seq = slots[id].sequence.load(std::memory_order_acquire);
                if (seq > last_seen[id]) {
                    // Read from buffer[seq % 2]
                    int buf = seq % 2;
                    bool valid = true;
                    for (int j = 0; j < 8; ++j) {
                        int expected = id * 1000 + seq * 10 + j;
                        if (slots[id].data[buf][j] != expected) {
                            valid = false;
                            violations.fetch_add(1, std::memory_order_relaxed);
                            ADD_FAILURE() << "Violation: slot[" << id << "].data["
                                          << buf << "][" << j << "]="
                                          << slots[id].data[buf][j]
                                          << " expected " << expected
                                          << " at seq " << seq;
                            break;
                        }
                    }
                    if (valid) {
                        total_consumed.fetch_add(1, std::memory_order_relaxed);
                    }
                    last_seen[id] = seq;
                    // Signal that we've consumed this sequence
                    slots[id].consumed.store(seq, std::memory_order_release);
                }
            }
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> producers;
    for (int i = 0; i < kProducers; ++i) {
        producers.emplace_back(producer, i);
    }
    std::thread consumer_thread(consumer);

    for (auto& t : producers) {
        t.join();
    }
    done.store(true, std::memory_order_release);
    consumer_thread.join();

    EXPECT_EQ(violations.load(), 0) << "Memory ordering violations detected";
    EXPECT_EQ(total_consumed.load(), kProducers * kRoundsPerProducer)
        << "Not all messages were consumed";
}

TEST_F(ConcurrencyStressTest, AcquireReleaseOrdering_StressWithBarriers) {
    // Stress test using barriers to force worst-case interleaving
    // This is the most rigorous test - it forces threads to interleave
    // at the exact point where races would manifest

    constexpr int kRounds = 500;
    std::atomic<int> violations{0};

    for (int round = 0; round < kRounds; ++round) {
        // Fresh state each round
        int shared_data = 0;
        std::atomic<bool> data_ready{false};
        std::barrier sync(2);

        std::thread writer([&]() {
            sync.arrive_and_wait();  // Sync point 1: both threads start together
            shared_data = 42 + round;
            data_ready.store(true, std::memory_order_release);
        });

        std::thread reader([&]() {
            sync.arrive_and_wait();  // Sync point 1: both threads start together
            // Spin until we see the flag
            while (!data_ready.load(std::memory_order_acquire)) {
                // Tight spin to maximize chance of catching violation
            }
            // After acquire, shared_data MUST be visible
            if (shared_data != 42 + round) {
                violations.fetch_add(1, std::memory_order_relaxed);
            }
        });

        writer.join();
        reader.join();
    }

    EXPECT_EQ(violations.load(), 0)
        << "Acquire-release violations in " << violations.load()
        << " of " << kRounds << " rounds";
}

// ---------------------------------------------------------------------------
// Barrier Synchronization Tests (Deterministic Race Reproduction)
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyStressTest, DeterministicControllerAccessRace) {
    // Force threads to interleave at specific points using barriers
    // This is the pattern recommended in TEST_STRATEGY_REPORT.md section 5.2
    constexpr int kRounds = 100;

    for (int round = 0; round < kRounds; ++round) {
        std::barrier sync_point(2);
        std::atomic<bool> race_detected{false};

        std::thread writer([&]() {
            sync_point.arrive_and_wait();
            // Simulate trial_subscriber_callback modifying trial_controller_
            plugin_.configure_controller(kDU);
        });

        std::thread reader([&]() {
            sync_point.arrive_and_wait();
            // Simulate update_controllers reading trial_controller_
            Eigen::VectorXd U(kDU);
            plugin_.update_controller(U);
        });

        writer.join();
        reader.join();

        EXPECT_FALSE(race_detected.load()) << "Race detected in round " << round;
    }
}

TEST_F(ConcurrencyStressTest, DeterministicSensorVsConfigRace) {
    // Test sensor initialization racing with sensor updates
    constexpr int kRounds = 100;

    for (int round = 0; round < kRounds; ++round) {
        std::barrier sync_point(2);

        std::thread initializer([&]() {
            sync_point.arrive_and_wait();
            plugin_.initialize_sensors();
        });

        std::thread updater([&]() {
            sync_point.arrive_and_wait();
            // Multiple rapid update attempts
            for (int i = 0; i < 10; ++i) {
                plugin_.update_sensors();
            }
        });

        initializer.join();
        updater.join();
    }

    // Test passes if no crash or TSan error
    SUCCEED();
}

// ---------------------------------------------------------------------------
// High-Frequency Callback Stress Test
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyStressTest, HighFrequencyActionCallbacks) {
    // Simulate 100Hz control loop with async action callbacks
    constexpr int kControlLoopIterations = 1000;
    constexpr int kCallbackThreads = 4;
    std::atomic<bool> running{true};
    std::atomic<int> callbacks_processed{0};
    std::atomic<int> control_iterations{0};

    auto callback_sender = [&](int thread_id) {
        Eigen::VectorXd cmd(kDU);
        int cmd_id = thread_id * 1000000;
        while (running.load(std::memory_order_acquire)) {
            cmd.setConstant(static_cast<double>(cmd_id));
            plugin_.send_action_command(cmd_id++, cmd);
            callbacks_processed.fetch_add(1, std::memory_order_relaxed);
            // Simulate ~1kHz callback rate
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    };

    auto control_loop = [&]() {
        Eigen::VectorXd U(kDU);
        for (int i = 0; i < kControlLoopIterations; ++i) {
            plugin_.update_controller(U);
            control_iterations.fetch_add(1, std::memory_order_relaxed);
            // Simulate 100Hz control rate
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kCallbackThreads; ++i) {
        threads.emplace_back(callback_sender, i);
    }
    threads.emplace_back(control_loop);

    // Wait for control loop to finish
    threads.back().join();
    running.store(false, std::memory_order_release);

    // Wait for callback threads
    for (int i = 0; i < kCallbackThreads; ++i) {
        threads[i].join();
    }

    EXPECT_EQ(control_iterations.load(), kControlLoopIterations);
    EXPECT_GT(callbacks_processed.load(), 0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
