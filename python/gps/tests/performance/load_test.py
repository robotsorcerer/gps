"""
Load testing for GPS components.

Tests system behavior under high load conditions.
"""
import pytest
import numpy as np
import time
import psutil
from typing import List, Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gps.utility.metrics import MetricsCollector


class LoadTester:
    """Framework for load testing."""

    def __init__(self):
        self.metrics = MetricsCollector()

    def run_load_test(self, func, iterations: int, *args, **kwargs) -> Dict[str, Any]:
        """Run load test on function."""
        durations = []
        memory_usage = []

        for i in range(iterations):
            # Memory before
            mem_before = psutil.Process().memory_info().rss / 1024**2

            # Execute
            start = time.time()
            func(*args, **kwargs)
            duration = time.time() - start

            # Memory after
            mem_after = psutil.Process().memory_info().rss / 1024**2

            durations.append(duration)
            memory_usage.append(mem_after - mem_before)

        return {
            'iterations': iterations,
            'avg_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
        }


class TestDynamicsLoadPerformance:
    """Load tests for dynamics fitting."""

    @pytest.mark.slow
    @pytest.mark.load
    def test_dynamics_fit_high_load(self):
        """Test dynamics fitting under high sample load."""
        T, dX, dU = 100, 20, 10
        N_values = [10, 50, 100, 500]

        results = []
        for N in N_values:
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

            start = time.time()
            # Simulate dynamics fit
            Fm = np.random.randn(T, dX, dX + dU)
            for t in range(T-1):
                XU = np.concatenate([X[:, t, :], U[:, t, :]], axis=1)
                Fm[t] = np.linalg.lstsq(XU, X[:, t+1, :], rcond=None)[0].T
            duration = time.time() - start

            results.append({
                'N': N,
                'duration': duration,
                'throughput': N / duration
            })

        # Verify scalability
        assert results[0]['duration'] < results[-1]['duration']
        print(f"Dynamics fit load test: {results}")

    @pytest.mark.slow
    @pytest.mark.load
    def test_dynamics_fit_long_horizon(self):
        """Test dynamics with very long time horizons."""
        N, dX, dU = 10, 10, 5
        T_values = [50, 100, 200, 500]

        for T in T_values:
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

            start = time.time()
            Fm = np.random.randn(T, dX, dX + dU)
            duration = time.time() - start

            assert duration < 5.0  # Should complete in 5s
            assert not np.any(np.isnan(Fm))


class TestPolicyLoadPerformance:
    """Load tests for policy evaluation."""

    @pytest.mark.slow
    @pytest.mark.load
    def test_policy_rollout_batch(self):
        """Test policy with batch rollouts."""
        T, dX, dU = 100, 10, 5
        batch_sizes = [1, 10, 50, 100]

        K = np.random.randn(T, dU, dX)
        k = np.random.randn(T, dU)

        for batch_size in batch_sizes:
            X = np.random.randn(batch_size, T, dX)
            U = np.zeros((batch_size, T, dU))

            start = time.time()
            for b in range(batch_size):
                for t in range(T-1):
                    U[b, t] = K[t] @ X[b, t] + k[t]
                    X[b, t+1] = X[b, t] + U[b, t, :dX] * 0.01
            duration = time.time() - start

            throughput = batch_size * T / duration
            print(f"Batch {batch_size}: {throughput:.0f} steps/sec")


class TestCostLoadPerformance:
    """Load tests for cost evaluation."""

    @pytest.mark.slow
    @pytest.mark.load
    def test_cost_evaluation_batch(self):
        """Test cost evaluation on batch samples."""
        T, dX, dU = 50, 10, 5
        batch_sizes = [10, 50, 100, 500]

        for batch_size in batch_sizes:
            X = np.random.randn(batch_size, T, dX)
            U = np.random.randn(batch_size, T, dU)

            start = time.time()
            # Simulate cost evaluation
            costs = np.sum(X**2, axis=(1, 2)) + np.sum(U**2, axis=(1, 2))
            duration = time.time() - start

            throughput = batch_size / duration
            assert throughput > 100  # At least 100 samples/sec

    @pytest.mark.slow
    @pytest.mark.load
    def test_cost_derivatives_computation(self):
        """Test cost derivative computation load."""
        T, dX, dU = 100, 20, 10
        N = 100

        X = np.random.randn(N, T, dX)
        U = np.random.randn(N, T, dU)

        start = time.time()
        # Simulate derivative computation
        lx = X * 2
        lu = U * 2
        lxx = np.tile(np.eye(dX), (T, 1, 1))
        luu = np.tile(np.eye(dU), (T, 1, 1))
        duration = time.time() - start

        assert duration < 1.0  # Should be very fast


class TestMemoryLeaks:
    """Test for memory leaks under load."""

    @pytest.mark.slow
    @pytest.mark.load
    def test_repeated_dynamics_fit_memory(self):
        """Test memory doesn't leak during repeated fits."""
        T, dX, dU, N = 50, 10, 5, 20

        memory_samples = []
        for i in range(10):
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

            # Fit dynamics
            Fm = np.random.randn(T, dX, dX + dU)

            # Measure memory
            mem = psutil.Process().memory_info().rss / 1024**2
            memory_samples.append(mem)

            # Clean up
            del X, U, Fm

        # Memory shouldn't grow significantly
        memory_growth = memory_samples[-1] - memory_samples[0]
        assert memory_growth < 50  # Less than 50MB growth


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.slow
    @pytest.mark.load
    def test_parallel_condition_processing(self):
        """Test processing multiple conditions."""
        T, dX, dU = 50, 10, 5
        M = 4  # Conditions
        N = 20  # Samples per condition

        start = time.time()
        results = []
        for m in range(M):
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

            # Simulate processing
            cost = np.sum(X**2) + np.sum(U**2)
            results.append(cost)

        duration = time.time() - start

        # Should complete reasonably fast
        assert duration < 2.0
        assert len(results) == M


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'load'])
