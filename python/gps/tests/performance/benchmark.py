"""
Benchmark utilities for GPS performance testing.
"""
import time
import numpy as np
import json
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, asdict
import psutil


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    throughput: float
    memory_mb: float

    def to_dict(self) -> Dict:
        return asdict(self)


class Benchmark:
    """Performance benchmarking framework."""

    def __init__(self, name: str):
        self.name = name
        self.results: List[BenchmarkResult] = []

    def run(self, func: Callable, iterations: int, *args, **kwargs) -> BenchmarkResult:
        """Run benchmark."""
        times = []
        mem_start = psutil.Process().memory_info().rss / 1024**2

        # Warm up
        for _ in range(min(10, iterations // 10)):
            func(*args, **kwargs)

        # Benchmark
        for _ in range(iterations):
            start = time.time()
            func(*args, **kwargs)
            times.append(time.time() - start)

        mem_end = psutil.Process().memory_info().rss / 1024**2

        result = BenchmarkResult(
            name=self.name,
            iterations=iterations,
            total_time=sum(times),
            avg_time=np.mean(times),
            min_time=min(times),
            max_time=max(times),
            throughput=iterations / sum(times),
            memory_mb=mem_end - mem_start
        )

        self.results.append(result)
        return result

    def save_results(self, filepath: str):
        """Save benchmark results to JSON."""
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)


def benchmark_dynamics_fit():
    """Benchmark dynamics fitting."""
    T, dX, dU, N = 50, 10, 5, 20

    def fit_dynamics():
        X = np.random.randn(N, T, dX)
        U = np.random.randn(N, T, dU)
        Fm = np.random.randn(T, dX, dX + dU)
        for t in range(T-1):
            XU = np.concatenate([X[:, t, :], U[:, t, :]], axis=1)
            Fm[t] = np.linalg.lstsq(XU, X[:, t+1, :], rcond=None)[0].T

    bench = Benchmark("Dynamics Fit")
    result = bench.run(fit_dynamics, iterations=100)
    print(f"Dynamics Fit: {result.avg_time*1000:.2f}ms avg, {result.throughput:.1f} ops/sec")
    return result


def benchmark_policy_rollout():
    """Benchmark policy rollout."""
    T, dX, dU = 100, 10, 5

    K = np.random.randn(T, dU, dX) * 0.01
    k = np.random.randn(T, dU) * 0.01

    def rollout():
        X = np.zeros((T, dX))
        U = np.zeros((T, dU))
        X[0] = np.random.randn(dX) * 0.1

        for t in range(T-1):
            U[t] = K[t] @ X[t] + k[t]
            X[t+1] = X[t] + U[t, :dX] * 0.01

    bench = Benchmark("Policy Rollout")
    result = bench.run(rollout, iterations=1000)
    print(f"Policy Rollout: {result.avg_time*1000:.2f}ms avg, {result.throughput:.1f} ops/sec")
    return result


def benchmark_cost_evaluation():
    """Benchmark cost evaluation."""
    T, dX, dU = 50, 10, 5

    X = np.random.randn(T, dX)
    U = np.random.randn(T, dU)

    def eval_cost():
        l = np.sum(X**2, axis=1) + np.sum(U**2, axis=1)
        lx = 2 * X
        lu = 2 * U

    bench = Benchmark("Cost Evaluation")
    result = bench.run(eval_cost, iterations=10000)
    print(f"Cost Eval: {result.avg_time*1000:.2f}ms avg, {result.throughput:.1f} ops/sec")
    return result


def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    print("=== GPS Performance Benchmarks ===\n")

    results = []
    results.append(benchmark_dynamics_fit())
    results.append(benchmark_policy_rollout())
    results.append(benchmark_cost_evaluation())

    print("\n=== Summary ===")
    for r in results:
        print(f"{r.name}: {r.throughput:.1f} ops/sec")

    return results


if __name__ == '__main__':
    run_all_benchmarks()
