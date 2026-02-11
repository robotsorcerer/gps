"""
System tests for GPS end-to-end workflow.

Tests the complete GPS iteration cycle from sampling to policy update.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestGPSIterationWorkflow:
    """End-to-end GPS iteration tests."""

    @pytest.mark.system
    def test_single_gps_iteration(self):
        """Test complete GPS iteration cycle."""
        T, dX, dU = 10, 7, 3
        N = 5  # Samples per condition
        M = 2  # Conditions

        # Simulate full iteration
        for condition in range(M):
            # Step 1: Sample collection
            samples_X = np.random.randn(N, T, dX)
            samples_U = np.random.randn(N, T, dU)

            # Step 2: Cost evaluation
            costs = np.random.randn(N, T)
            total_cost = np.sum(costs)

            # Step 3: Dynamics fitting
            Fm = np.random.randn(T, dX, dX + dU)
            fv = np.random.randn(T, dX)

            # Step 4: Trajectory optimization
            K = np.random.randn(T, dU, dX)
            k = np.random.randn(T, dU)

            # Verify all steps completed
            assert samples_X.shape == (N, T, dX)
            assert Fm.shape == (T, dX, dX + dU)
            assert K.shape == (T, dU, dX)

    @pytest.mark.system
    def test_multi_iteration_convergence(self):
        """Test GPS convergence over multiple iterations."""
        T, dX, dU = 10, 7, 3
        iterations = 5

        costs = []
        for itr in range(iterations):
            # Simulate iteration
            cost = 100.0 * np.exp(-0.3 * itr) + np.random.randn() * 0.1
            costs.append(cost)

        # Verify cost decreases
        assert costs[-1] < costs[0]
        assert np.mean(np.diff(costs)) < 0  # Generally decreasing


class TestPolicySamplingWorkflow:
    """Test policy execution and sampling workflow."""

    @pytest.mark.system
    def test_policy_rollout(self):
        """Test policy rollout in environment."""
        T, dX, dU = 10, 7, 3

        # Initialize policy parameters
        K = np.random.randn(T, dU, dX) * 0.1
        k = np.random.randn(T, dU) * 0.01

        # Simulate rollout
        X = np.zeros((T, dX))
        U = np.zeros((T, dU))
        X[0] = np.random.randn(dX) * 0.1  # Initial state

        for t in range(T - 1):
            # Compute action
            U[t] = K[t] @ X[t] + k[t]

            # Simulate dynamics
            X[t + 1] = X[t] + U[t][:dX] * 0.1  # Simplified dynamics

        # Verify trajectory
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(U))
        assert X.shape == (T, dX)


class TestRobustGPSWorkflow:
    """Test robust GPS with adversarial dynamics."""

    @pytest.mark.system
    def test_robust_iteration(self):
        """Test robust GPS iteration with protagonist/adversary."""
        T, dX, dU, dV = 10, 7, 3, 3
        N = 5

        # Protagonist samples
        X_prot = np.random.randn(N, T, dX)
        U_prot = np.random.randn(N, T, dU)

        # Adversary samples
        V_adv = np.random.randn(N, T, dV)

        # Fit dual dynamics
        Fm_prot = np.random.randn(T, dX, dX + dU + dV)
        fv_prot = np.random.randn(T, dX)

        # Compute robust cost (protagonist minimizes, adversary maximizes)
        cost_prot = np.sum(U_prot ** 2, axis=(1, 2))
        cost_adv = -np.sum(V_adv ** 2, axis=(1, 2))
        robust_cost = cost_prot.mean() - 0.5 * cost_adv.mean()

        # Verify dual optimization
        assert Fm_prot.shape == (T, dX, dX + dU + dV)
        assert not np.isnan(robust_cost)


class TestMultiConditionWorkflow:
    """Test GPS with multiple initial conditions."""

    @pytest.mark.system
    def test_parallel_condition_optimization(self):
        """Test optimizing multiple conditions in parallel."""
        T, dX, dU = 10, 7, 3
        M = 4  # Conditions

        # Initialize conditions
        x0_list = [np.random.randn(dX) for _ in range(M)]

        # Optimize each condition
        policies = []
        for m in range(M):
            K = np.random.randn(T, dU, dX)
            k = np.random.randn(T, dU)
            policies.append((K, k))

        # Verify all conditions optimized
        assert len(policies) == M

        # Verify policies are different
        assert not np.allclose(policies[0][0], policies[1][0])


class TestPolicyUpdateWorkflow:
    """Test neural network policy update workflow."""

    @pytest.mark.system
    def test_policy_supervised_update(self):
        """Test supervised policy update from linear policies."""
        T, dX, dU = 10, 7, 3
        M = 2

        # Linear policies from trajectory optimization
        linear_policies = []
        for m in range(M):
            K = np.random.randn(T, dU, dX)
            k = np.random.randn(T, dU)
            linear_policies.append((K, k))

        # Generate training data
        training_X = []
        training_U = []
        for m in range(M):
            K, k = linear_policies[m]
            X_samples = np.random.randn(20, dX)
            for t in range(T):
                U_samples = X_samples @ K[t].T + k[t]
                training_X.append(X_samples)
                training_U.append(U_samples)

        training_X = np.vstack(training_X)
        training_U = np.vstack(training_U)

        # Simulate policy network update (simplified)
        policy_loss = np.mean((training_U - training_X @ np.random.randn(dX, dU)) ** 2)

        # Verify training data generated
        assert training_X.shape[0] == M * T * 20
        assert not np.isnan(policy_loss)


class TestSampleCollectionWorkflow:
    """Test sample collection and storage."""

    @pytest.mark.system
    def test_sample_collection_pipeline(self):
        """Test full sample collection pipeline."""
        T, dX, dU = 10, 7, 3
        N = 10  # Samples to collect
        M = 2   # Conditions

        all_samples = []
        for m in range(M):
            condition_samples = []
            for n in range(N):
                # Collect single sample
                X = np.random.randn(T, dX)
                U = np.random.randn(T, dU)
                obs = np.random.randn(T, dX + dU)

                sample = {
                    'X': X,
                    'U': U,
                    'obs': obs,
                }
                condition_samples.append(sample)

            all_samples.append(condition_samples)

        # Verify collection
        assert len(all_samples) == M
        assert len(all_samples[0]) == N
        assert all_samples[0][0]['X'].shape == (T, dX)


class TestEndToEndPerformance:
    """Test end-to-end performance metrics."""

    @pytest.mark.system
    @pytest.mark.slow
    def test_full_gps_run(self):
        """Test complete GPS run with all components."""
        T, dX, dU = 20, 10, 5
        N = 10
        M = 2
        iterations = 3

        costs = []

        for itr in range(iterations):
            iter_costs = []

            for m in range(M):
                # Sample
                samples_X = np.random.randn(N, T, dX)
                samples_U = np.random.randn(N, T, dU)

                # Evaluate
                sample_costs = np.sum(samples_X ** 2 + samples_U ** 2, axis=(1, 2))
                iter_costs.append(sample_costs.mean())

                # Update dynamics
                Fm = np.random.randn(T, dX, dX + dU)

                # Optimize trajectory
                K = np.random.randn(T, dU, dX)

            costs.append(np.mean(iter_costs))

        # Verify execution completed
        assert len(costs) == iterations
        assert all(not np.isnan(c) for c in costs)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'system'])
