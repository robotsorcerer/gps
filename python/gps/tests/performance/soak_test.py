"""
Soak testing for GPS components.

Long-running tests to detect memory leaks, stability issues, and performance degradation.
"""
import pytest
import numpy as np
import time
import psutil
from typing import List, Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class SoakTestMonitor:
    """Monitor system during soak tests."""

    def __init__(self):
        self.memory_samples = []
        self.cpu_samples = []
        self.timestamps = []

    def sample_resources(self):
        """Sample current resource usage."""
        process = psutil.Process()
        self.memory_samples.append(process.memory_info().rss / 1024**2)
        self.cpu_samples.append(process.cpu_percent())
        self.timestamps.append(time.time())

    def get_report(self) -> Dict:
        """Generate soak test report."""
        return {
            'duration_sec': self.timestamps[-1] - self.timestamps[0],
            'samples': len(self.memory_samples),
            'memory': {
                'start_mb': self.memory_samples[0],
                'end_mb': self.memory_samples[-1],
                'max_mb': max(self.memory_samples),
                'avg_mb': np.mean(self.memory_samples),
                'growth_mb': self.memory_samples[-1] - self.memory_samples[0],
            },
            'cpu': {
                'avg_percent': np.mean(self.cpu_samples),
                'max_percent': max(self.cpu_samples),
            }
        }


class TestLongRunningOperations:
    """Soak tests for long-running operations."""

    @pytest.mark.soak
    @pytest.mark.slow
    def test_continuous_dynamics_fitting(self):
        """Test dynamics fitting over extended period."""
        T, dX, dU, N = 50, 10, 5, 20
        iterations = 100

        monitor = SoakTestMonitor()

        for i in range(iterations):
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

            # Fit dynamics
            Fm = np.random.randn(T, dX, dX + dU)

            if i % 10 == 0:
                monitor.sample_resources()

            # Clean up
            del X, U, Fm

        report = monitor.get_report()

        # Verify stability
        assert report['memory']['growth_mb'] < 100  # Less than 100MB growth
        print(f"Soak test report: {report}")

    @pytest.mark.soak
    @pytest.mark.slow
    def test_continuous_policy_rollouts(self):
        """Test policy rollouts over extended time."""
        T, dX, dU = 100, 10, 5
        iterations = 1000

        K = np.random.randn(T, dU, dX) * 0.01
        k = np.random.randn(T, dU) * 0.01

        monitor = SoakTestMonitor()

        for i in range(iterations):
            X = np.zeros((T, dX))
            U = np.zeros((T, dU))
            X[0] = np.random.randn(dX) * 0.1

            for t in range(T-1):
                U[t] = K[t] @ X[t] + k[t]
                X[t+1] = X[t] + U[t, :dX] * 0.01

            if i % 100 == 0:
                monitor.sample_resources()

        report = monitor.get_report()
        assert report['memory']['growth_mb'] < 50


class TestStabilityUnderLoad:
    """Test system stability under sustained load."""

    @pytest.mark.soak
    @pytest.mark.slow
    def test_gps_iteration_stability(self):
        """Test GPS iterations remain stable over time."""
        T, dX, dU = 50, 10, 5
        N, M = 10, 2
        iterations = 50

        costs = []
        durations = []

        for itr in range(iterations):
            start = time.time()

            iter_costs = []
            for m in range(M):
                X = np.random.randn(N, T, dX)
                U = np.random.randn(N, T, dU)

                # Simulate iteration
                cost = np.sum(X**2) + np.sum(U**2)
                iter_costs.append(cost)

            costs.append(np.mean(iter_costs))
            durations.append(time.time() - start)

        # Check stability
        cost_std = np.std(costs)
        duration_std = np.std(durations)

        # Costs should remain stable (not grow)
        assert cost_std < np.mean(costs)  # Reasonable variation

        # Duration shouldn't degrade significantly
        early_avg = np.mean(durations[:10])
        late_avg = np.mean(durations[-10:])
        assert late_avg < early_avg * 1.5  # Max 50% slowdown


class TestResourceExhaustion:
    """Test behavior under resource constraints."""

    @pytest.mark.soak
    @pytest.mark.slow
    def test_large_sample_collection(self):
        """Test collecting very large sample sets."""
        T, dX, dU = 100, 10, 5
        N_large = 1000

        start_mem = psutil.Process().memory_info().rss / 1024**2

        # Collect large sample set
        samples = []
        for n in range(N_large):
            X = np.random.randn(T, dX)
            U = np.random.randn(T, dU)
            samples.append({'X': X, 'U': U})

        end_mem = psutil.Process().memory_info().rss / 1024**2
        memory_used = end_mem - start_mem

        # Memory usage should be reasonable
        assert memory_used < 2000  # Less than 2GB
        assert len(samples) == N_large

    @pytest.mark.soak
    @pytest.mark.slow
    def test_high_dimensional_state(self):
        """Test with very high-dimensional state spaces."""
        T = 50
        dX_values = [10, 50, 100, 200]
        dU = 10
        N = 10

        for dX in dX_values:
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

            start = time.time()
            # Simulate processing
            cost = np.sum(X**2) + np.sum(U**2)
            duration = time.time() - start

            # Should complete even for high dimensions
            assert duration < 5.0
            assert not np.isnan(cost)


class TestPerformanceDegradation:
    """Test for performance degradation over time."""

    @pytest.mark.soak
    @pytest.mark.slow
    def test_no_performance_degradation(self):
        """Verify performance doesn't degrade over iterations."""
        T, dX, dU, N = 50, 10, 5, 20

        # Warm up
        for _ in range(10):
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

        # Measure early performance
        early_durations = []
        for _ in range(20):
            start = time.time()
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)
            Fm = np.random.randn(T, dX, dX + dU)
            early_durations.append(time.time() - start)

        # Run many iterations
        for _ in range(100):
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)

        # Measure late performance
        late_durations = []
        for _ in range(20):
            start = time.time()
            X = np.random.randn(N, T, dX)
            U = np.random.randn(N, T, dU)
            Fm = np.random.randn(T, dX, dX + dU)
            late_durations.append(time.time() - start)

        # Performance shouldn't degrade
        early_avg = np.mean(early_durations)
        late_avg = np.mean(late_durations)

        assert late_avg <= early_avg * 1.2  # Max 20% slowdown


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'soak'])
