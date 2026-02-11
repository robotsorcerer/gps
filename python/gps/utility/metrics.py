"""Performance metrics and monitoring utilities."""
import time
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json


class MetricsCollector:
    """Collects and tracks performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = defaultdict(int)

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        self.metrics[name].append(value)

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] += amount

    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        self.timers[name] = time.time()

    def stop_timer(self, name: str) -> Optional[float]:
        """Stop timing and record duration."""
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.record_metric(f"{name}_duration", duration)
            del self.timers[name]
            return duration
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {}

        # Metric statistics
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                }

        # Counter values
        summary['counters'] = dict(self.counters)

        return summary

    def save_summary(self, filepath: str) -> None:
        """Save metrics summary to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()


class GPSMetrics(MetricsCollector):
    """GPS-specific metrics tracking."""

    def record_iteration_cost(self, iteration: int, cost: float, condition: int = 0) -> None:
        """Record cost for an iteration."""
        self.record_metric(f"iteration_{iteration}_cost", cost)
        self.record_metric(f"condition_{condition}_cost", cost)
        self.record_metric("total_cost", cost)

    def record_policy_kl(self, kl_divergence: float) -> None:
        """Record KL divergence for policy update."""
        self.record_metric("kl_divergence", kl_divergence)

    def record_dynamics_fit_error(self, error: float) -> None:
        """Record dynamics fitting error."""
        self.record_metric("dynamics_fit_error", error)

    def record_sample_success_rate(self, success_rate: float) -> None:
        """Record sample collection success rate."""
        self.record_metric("sample_success_rate", success_rate)


# Global metrics collector
_global_metrics = GPSMetrics()


def get_metrics() -> GPSMetrics:
    """Get global metrics collector."""
    return _global_metrics
