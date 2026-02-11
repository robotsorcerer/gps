"""
Integration tests for Policy and Agent interaction.

Tests policy execution in agent environments.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class MockPolicy:
    """Mock policy for testing."""
    def __init__(self, T, dU, dX):
        self.T = T
        self.dU = dU
        self.dX = dX
        self.K = np.random.randn(T, dU, dX) * 0.1
        self.k = np.random.randn(T, dU) * 0.01

    def act(self, x, obs, t, noise):
        u = self.K[t] @ x + self.k[t]
        if noise is not None:
            u += noise
        return u


class MockAgent:
    """Mock agent for testing."""
    def __init__(self, T, dX, dU):
        self.T = T
        self.dX = dX
        self.dU = dU
        self.x0 = np.zeros(dX)
        self._samples = []

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """Collect sample using policy."""
        X = np.zeros((self.T, self.dX))
        U = np.zeros((self.T, self.dU))
        X[0] = self.x0 + np.random.randn(self.dX) * 0.01

        for t in range(self.T - 1):
            noise = np.random.randn(self.dU) * 0.01 if noisy else None
            U[t] = policy.act(X[t], X[t], t, noise)
            X[t + 1] = X[t] + U[t][:self.dX] * 0.1

        sample = {'X': X, 'U': U}
        if save:
            self._samples.append(sample)
        return sample

    def reset(self, condition):
        self.x0 = np.random.randn(self.dX) * 0.1


class TestPolicyAgentInteraction:
    """Test policy-agent interaction."""

    @pytest.mark.integration
    def test_policy_execution_in_agent(self):
        """Test policy executing actions in agent."""
        T, dX, dU = 10, 7, 3

        policy = MockPolicy(T, dU, dX)
        agent = MockAgent(T, dX, dU)

        # Execute policy
        sample = agent.sample(policy, condition=0, noisy=False)

        # Verify execution
        assert sample['X'].shape == (T, dX)
        assert sample['U'].shape == (T, dU)
        assert not np.any(np.isnan(sample['X']))

    @pytest.mark.integration
    def test_noisy_policy_execution(self):
        """Test policy with exploration noise."""
        T, dX, dU = 10, 7, 3

        policy = MockPolicy(T, dU, dX)
        agent = MockAgent(T, dX, dU)

        # Execute with noise
        sample1 = agent.sample(policy, condition=0, noisy=True)
        agent.reset(0)
        sample2 = agent.sample(policy, condition=0, noisy=True)

        # Verify noise causes different trajectories
        assert not np.allclose(sample1['U'], sample2['U'])

    @pytest.mark.integration
    def test_deterministic_policy_execution(self):
        """Test deterministic policy execution."""
        T, dX, dU = 10, 7, 3

        policy = MockPolicy(T, dU, dX)
        agent = MockAgent(T, dX, dU)

        # Execute without noise
        agent.x0 = np.ones(dX) * 0.5  # Fixed initial state
        sample1 = agent.sample(policy, condition=0, noisy=False)

        agent.x0 = np.ones(dX) * 0.5  # Same initial state
        sample2 = agent.sample(policy, condition=0, noisy=False)

        # Should be identical
        np.testing.assert_array_almost_equal(sample1['U'], sample2['U'])

    @pytest.mark.integration
    def test_multiple_sample_collection(self):
        """Test collecting multiple samples."""
        T, dX, dU = 10, 7, 3
        N = 5

        policy = MockPolicy(T, dU, dX)
        agent = MockAgent(T, dX, dU)

        # Collect samples
        for n in range(N):
            agent.reset(0)
            agent.sample(policy, condition=0, save=True)

        # Verify collection
        assert len(agent._samples) == N

    @pytest.mark.integration
    def test_multi_condition_sampling(self):
        """Test sampling from multiple conditions."""
        T, dX, dU = 10, 7, 3
        M = 3

        policy = MockPolicy(T, dU, dX)
        agent = MockAgent(T, dX, dU)

        samples = []
        for m in range(M):
            agent.reset(m)
            sample = agent.sample(policy, condition=m)
            samples.append(sample)

        # Verify all conditions sampled
        assert len(samples) == M


class TestPolicyRolloutQuality:
    """Test quality of policy rollouts."""

    @pytest.mark.integration
    def test_policy_stability(self):
        """Test policy produces stable rollouts."""
        T, dX, dU = 20, 7, 3

        # Well-conditioned policy
        policy = MockPolicy(T, dU, dX)
        policy.K = policy.K * 0.01  # Small gains

        agent = MockAgent(T, dX, dU)
        sample = agent.sample(policy, condition=0, noisy=False)

        # Verify stability (states don't blow up)
        assert np.all(np.abs(sample['X']) < 10.0)
        assert np.all(np.abs(sample['U']) < 10.0)

    @pytest.mark.integration
    def test_policy_consistency_across_timesteps(self):
        """Test policy acts consistently at each timestep."""
        T, dX, dU = 10, 7, 3

        policy = MockPolicy(T, dU, dX)
        agent = MockAgent(T, dX, dU)

        sample = agent.sample(policy, condition=0, noisy=False)

        # Verify actions computed at each timestep
        for t in range(T - 1):
            expected_u = policy.K[t] @ sample['X'][t] + policy.k[t]
            np.testing.assert_array_almost_equal(sample['U'][t], expected_u, decimal=5)


class TestAgentResetBehavior:
    """Test agent reset and initialization."""

    @pytest.mark.integration
    def test_agent_reset_changes_initial_state(self):
        """Test reset changes initial state."""
        T, dX, dU = 10, 7, 3

        agent = MockAgent(T, dX, dU)
        policy = MockPolicy(T, dU, dX)

        # Sample from different resets
        agent.reset(0)
        x0_first = agent.x0.copy()

        agent.reset(1)
        x0_second = agent.x0.copy()

        # Should be different
        assert not np.allclose(x0_first, x0_second)

    @pytest.mark.integration
    def test_samples_cleared_per_condition(self):
        """Test sample storage per condition."""
        T, dX, dU = 10, 7, 3

        agent = MockAgent(T, dX, dU)
        policy = MockPolicy(T, dU, dX)

        # Collect samples
        agent.sample(policy, condition=0, save=True)
        agent.sample(policy, condition=0, save=True)

        assert len(agent._samples) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
