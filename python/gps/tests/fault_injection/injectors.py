"""
Fault injection utilities for testing error handling.
"""
import random
import numpy as np
from typing import Any, Callable, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class FaultInjector:
    """Base fault injector."""

    def __init__(self, probability: float = 0.1):
        self.probability = probability
        self.fault_count = 0

    def should_inject(self) -> bool:
        """Determine if fault should be injected."""
        return random.random() < self.probability

    def inject(self, *args, **kwargs) -> Any:
        """Inject a fault."""
        raise NotImplementedError


class DataCorruptionInjector(FaultInjector):
    """Inject data corruption faults."""

    def corrupt_array(self, arr: np.ndarray) -> np.ndarray:
        """Corrupt numpy array data."""
        if not self.should_inject():
            return arr

        self.fault_count += 1
        corrupted = arr.copy()

        # Random corruption type
        corruption_type = random.choice(['nan', 'inf', 'zero', 'extreme'])

        if corruption_type == 'nan':
            mask = np.random.random(arr.shape) < 0.1
            corrupted[mask] = np.nan
        elif corruption_type == 'inf':
            mask = np.random.random(arr.shape) < 0.1
            corrupted[mask] = np.inf
        elif corruption_type == 'zero':
            mask = np.random.random(arr.shape) < 0.1
            corrupted[mask] = 0
        elif corruption_type == 'extreme':
            mask = np.random.random(arr.shape) < 0.1
            corrupted[mask] *= 1e10

        logger.warning(f"Injected {corruption_type} corruption")
        return corrupted

    def corrupt_dimension(self, X: np.ndarray, U: np.ndarray) -> tuple:
        """Corrupt dimensions (shape mismatch)."""
        if not self.should_inject():
            return X, U

        self.fault_count += 1

        # Random dimension corruption
        if random.random() < 0.5:
            X = X[:-1]  # Remove last sample
        else:
            U = U[:-1]

        logger.warning("Injected dimension mismatch")
        return X, U


class MemoryInjector(FaultInjector):
    """Inject memory-related faults."""

    def inject_oom(self, size_mb: int = 100):
        """Simulate out-of-memory condition."""
        if not self.should_inject():
            return

        self.fault_count += 1
        try:
            # Allocate large array
            _ = np.zeros((size_mb * 1024 * 1024 // 8,))
            logger.warning("Injected memory pressure")
        except MemoryError:
            logger.error("Actual OOM occurred")
            raise


class ComputationInjector(FaultInjector):
    """Inject computation faults."""

    def inject_singular_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Make matrix singular."""
        if not self.should_inject():
            return matrix

        self.fault_count += 1
        singular = matrix.copy()
        singular[0, :] = singular[1, :]  # Make rows identical
        logger.warning("Injected singular matrix")
        return singular

    def inject_numerical_instability(self, value: float) -> float:
        """Inject numerical instability."""
        if not self.should_inject():
            return value

        self.fault_count += 1

        # Make value very small or very large
        if random.random() < 0.5:
            value *= 1e-100  # Underflow risk
        else:
            value *= 1e100   # Overflow risk

        logger.warning("Injected numerical instability")
        return value


class RandomFailureInjector(FaultInjector):
    """Inject random failures."""

    def __init__(self, probability: float = 0.1, exception_type: type = RuntimeError):
        super().__init__(probability)
        self.exception_type = exception_type

    def maybe_fail(self, message: str = "Injected failure"):
        """Randomly raise exception."""
        if self.should_inject():
            self.fault_count += 1
            logger.warning(f"Injecting failure: {message}")
            raise self.exception_type(message)


class DelayInjector(FaultInjector):
    """Inject delays/timeouts."""

    def __init__(self, probability: float = 0.1, max_delay: float = 1.0):
        super().__init__(probability)
        self.max_delay = max_delay

    def inject_delay(self):
        """Inject random delay."""
        if not self.should_inject():
            return

        import time
        delay = random.uniform(0, self.max_delay)
        self.fault_count += 1
        logger.warning(f"Injecting {delay:.2f}s delay")
        time.sleep(delay)


@contextmanager
def fault_injection_context(injectors: list):
    """Context manager for fault injection."""
    try:
        yield injectors
    finally:
        total_faults = sum(inj.fault_count for inj in injectors)
        logger.info(f"Total faults injected: {total_faults}")


class ChaosMonkey:
    """Chaos engineering - random fault injection."""

    def __init__(self, severity: float = 0.1):
        self.data_corruptor = DataCorruptionInjector(severity)
        self.memory_injector = MemoryInjector(severity * 0.5)
        self.computation_injector = ComputationInjector(severity)
        self.random_failure = RandomFailureInjector(severity * 0.5)
        self.delay_injector = DelayInjector(severity)

    def apply_chaos(self, X: np.ndarray, U: np.ndarray) -> tuple:
        """Apply random chaos to data."""
        # Data corruption
        X = self.data_corruptor.corrupt_array(X)
        U = self.data_corruptor.corrupt_array(U)

        # Dimension corruption
        X, U = self.data_corruptor.corrupt_dimension(X, U)

        # Random failures
        self.random_failure.maybe_fail("Chaos monkey struck!")

        # Delays
        self.delay_injector.inject_delay()

        return X, U

    def total_faults(self) -> int:
        """Get total faults injected."""
        return (self.data_corruptor.fault_count +
                self.memory_injector.fault_count +
                self.computation_injector.fault_count +
                self.random_failure.fault_count +
                self.delay_injector.fault_count)
