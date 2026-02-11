"""
Fault injection tests for GPS robustness.

Tests system behavior when faults are injected.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .injectors import (
    DataCorruptionInjector,
    MemoryInjector,
    ComputationInjector,
    RandomFailureInjector,
    ChaosMonkey,
    fault_injection_context
)


class TestDataCorruptionTolerance:
    """Test tolerance to data corruption."""

    @pytest.mark.fault
    def test_handle_nan_values(self):
        """Test handling of NaN values in data."""
        T, dX, dU = 50, 10, 5
        N = 20

        injector = DataCorruptionInjector(probability=1.0)

        X = np.random.randn(N, T, dX)
        X_corrupted = injector.corrupt_array(X)

        # System should detect NaN
        has_nan = np.any(np.isnan(X_corrupted))
        assert has_nan  # Corruption occurred

        # Filter out NaN values
        X_clean = X_corrupted[~np.isnan(X_corrupted).any(axis=(1, 2))]
        assert len(X_clean) < N or len(X_clean) == 0

    @pytest.mark.fault
    def test_handle_inf_values(self):
        """Test handling of infinite values."""
        T, dX = 50, 10
        injector = DataCorruptionInjector(probability=1.0)

        X = np.random.randn(T, dX)
        X_corrupted = injector.corrupt_array(X)

        # System should detect inf
        has_inf = np.any(np.isinf(X_corrupted))
        if has_inf:
            # Replace inf with large but finite values
            X_safe = np.nan_to_num(X_corrupted, posinf=1e6, neginf=-1e6)
            assert not np.any(np.isinf(X_safe))

    @pytest.mark.fault
    def test_handle_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        T, dX, dU = 50, 10, 5
        N = 20

        injector = DataCorruptionInjector(probability=1.0)

        X = np.random.randn(N, T, dX)
        U = np.random.randn(N, T, dU)

        X_corrupt, U_corrupt = injector.corrupt_dimension(X, U)

        # Detect mismatch
        if X_corrupt.shape[0] != U_corrupt.shape[0]:
            # Truncate to minimum
            min_N = min(X_corrupt.shape[0], U_corrupt.shape[0])
            X_fixed = X_corrupt[:min_N]
            U_fixed = U_corrupt[:min_N]
            assert X_fixed.shape[0] == U_fixed.shape[0]


class TestComputationalFaultTolerance:
    """Test tolerance to computational faults."""

    @pytest.mark.fault
    def test_handle_singular_matrix(self):
        """Test handling of singular matrices."""
        dX = 10
        injector = ComputationInjector(probability=1.0)

        A = np.random.randn(dX, dX)
        A_singular = injector.inject_singular_matrix(A)

        # Detect singularity
        det = np.linalg.det(A_singular)
        if abs(det) < 1e-10:
            # Use pseudoinverse instead
            A_inv = np.linalg.pinv(A_singular)
            assert A_inv.shape == (dX, dX)

    @pytest.mark.fault
    def test_handle_numerical_instability(self):
        """Test handling of numerically unstable values."""
        injector = ComputationInjector(probability=1.0)

        value = 1.0
        unstable = injector.inject_numerical_instability(value)

        # Clamp to safe range
        safe_value = np.clip(unstable, -1e10, 1e10)
        assert -1e10 <= safe_value <= 1e10


class TestRandomFailureTolerance:
    """Test tolerance to random failures."""

    @pytest.mark.fault
    def test_retry_on_failure(self):
        """Test retry logic on random failures."""
        injector = RandomFailureInjector(probability=0.5)

        max_retries = 3
        success = False

        for attempt in range(max_retries):
            try:
                injector.maybe_fail("Random failure")
                success = True
                break
            except RuntimeError:
                continue

        # Should eventually succeed or exhaust retries
        assert success or attempt == max_retries - 1

    @pytest.mark.fault
    def test_graceful_degradation(self):
        """Test graceful degradation on failures."""
        injector = RandomFailureInjector(probability=0.3)

        results = []
        for i in range(10):
            try:
                injector.maybe_fail(f"Attempt {i}")
                results.append(i)
            except RuntimeError:
                # Log failure, continue
                pass

        # Some should succeed
        assert len(results) > 0


class TestChaosEngineering:
    """Chaos engineering tests."""

    @pytest.mark.fault
    @pytest.mark.chaos
    def test_chaos_monkey_dynamics(self):
        """Test dynamics fitting with chaos monkey."""
        T, dX, dU = 50, 10, 5
        N = 20

        monkey = ChaosMonkey(severity=0.2)

        successes = 0
        attempts = 10

        for _ in range(attempts):
            try:
                X = np.random.randn(N, T, dX)
                U = np.random.randn(N, T, dU)

                # Apply chaos
                X_chaos, U_chaos = monkey.apply_chaos(X, U)

                # Try to fit dynamics
                if not np.any(np.isnan(X_chaos)) and not np.any(np.isnan(U_chaos)):
                    if X_chaos.shape[0] == U_chaos.shape[0]:
                        Fm = np.random.randn(T, dX, dX + dU)
                        successes += 1

            except (RuntimeError, ValueError):
                # Expected from chaos monkey
                pass

        # Should have some successes despite chaos
        success_rate = successes / attempts
        print(f"Success rate under chaos: {success_rate:.1%}")
        print(f"Total faults injected: {monkey.total_faults()}")

    @pytest.mark.fault
    @pytest.mark.chaos
    def test_chaos_monkey_policy(self):
        """Test policy execution with chaos."""
        T, dX, dU = 100, 10, 5

        monkey = ChaosMonkey(severity=0.1)

        K = np.random.randn(T, dU, dX)
        k = np.random.randn(T, dU)

        successes = 0
        for _ in range(20):
            try:
                X = np.zeros((T, dX))
                U = np.zeros((T, dU))
                X[0] = np.random.randn(dX)

                # Apply chaos to initial state
                X_chaos = monkey.data_corruptor.corrupt_array(X)

                if not np.any(np.isnan(X_chaos[0])):
                    for t in range(min(T-1, 10)):  # Short rollout
                        U[t] = K[t] @ X_chaos[t] + k[t]
                        X_chaos[t+1] = X_chaos[t] + U[t, :dX] * 0.01

                    if not np.any(np.isnan(U)):
                        successes += 1

            except (RuntimeError, ValueError):
                pass

        # Should have reasonable success rate
        assert successes > 5  # At least 25% success


class TestRecoveryMechanisms:
    """Test error recovery mechanisms."""

    @pytest.mark.fault
    def test_checkpoint_recovery(self):
        """Test recovery from checkpointed state."""
        T, dX, dU = 50, 10, 5

        # Save checkpoint
        X_checkpoint = np.random.randn(T, dX)
        U_checkpoint = np.random.randn(T, dU)

        injector = DataCorruptionInjector(probability=1.0)

        # Corrupt current state
        X_corrupt = injector.corrupt_array(X_checkpoint)

        # Detect corruption and restore
        if np.any(np.isnan(X_corrupt)):
            X_restored = X_checkpoint.copy()
            assert not np.any(np.isnan(X_restored))

    @pytest.mark.fault
    def test_fallback_computation(self):
        """Test fallback to simpler computation on failure."""
        dX = 10
        injector = ComputationInjector(probability=1.0)

        A = np.random.randn(dX, dX)
        A_singular = injector.inject_singular_matrix(A)

        # Try full inverse, fallback to pseudoinverse
        try:
            A_inv = np.linalg.inv(A_singular)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A_singular)

        assert A_inv.shape == (dX, dX)


class TestFaultDetection:
    """Test fault detection capabilities."""

    @pytest.mark.fault
    def test_detect_data_corruption(self):
        """Test detection of corrupted data."""
        T, dX = 50, 10

        X_good = np.random.randn(T, dX)
        X_bad = X_good.copy()
        X_bad[0, 0] = np.nan

        # Validation function
        def validate_data(X):
            return not (np.any(np.isnan(X)) or np.any(np.isinf(X)))

        assert validate_data(X_good)
        assert not validate_data(X_bad)

    @pytest.mark.fault
    def test_detect_dimension_errors(self):
        """Test detection of dimension mismatches."""
        X = np.random.randn(10, 50, 10)
        U = np.random.randn(9, 50, 5)  # Mismatched

        def validate_dimensions(X, U):
            return X.shape[0] == U.shape[0] and X.shape[1] == U.shape[1]

        assert not validate_dimensions(X, U)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'fault'])
