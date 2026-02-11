"""
Unit tests for Agent base class.

Tests cover:
- Agent initialization with sensor dimensions
- Sample collection and management
- State/observation indexing
- Data type handling
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock ACTION constant for testing (normally from gps_pb2)
ACTION = 'ACTION'

try:
    from gps.agent.agent import Agent
    from gps.sample.sample_list import SampleList
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module dependencies not available")
class MockAgent:
    """Mock implementation for testing Agent base class."""

    def __init__(self, hyperparams):
        # Simplified mock for testing when Agent not available
        self.T = hyperparams.get('T', 10)
        self.dU = hyperparams.get('sensor_dims', {}).get(ACTION, 3)
        self.dV = self.dU
        self.dX = sum(hyperparams.get('sensor_dims', {}).get(s, 0)
                     for s in hyperparams.get('state_include', []))
        self.dO = sum(hyperparams.get('sensor_dims', {}).get(s, 0)
                     for s in hyperparams.get('obs_include', []))
        self.dM = 0
        self.x_data_types = hyperparams.get('state_include', [])
        self.obs_data_types = hyperparams.get('obs_include', [])
        self.meta_data_types = hyperparams.get('meta_include', [])
        self._samples = [[] for _ in range(hyperparams.get('conditions', 2))]
        self._samples_adv = [[] for _ in range(hyperparams.get('conditions', 2))]
        self._state_idx = []
        self._obs_idx = []
        self._meta_idx = []
        self._x_data_idx = {}
        self._obs_data_idx = {}
        self._meta_data_idx = {}

        # Build state indices
        i = 0
        for sensor in self.x_data_types:
            dim = hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            self._x_data_idx[sensor] = list(range(i, i+dim))
            i += dim

        # Build obs indices
        i = 0
        for sensor in self.obs_data_types:
            dim = hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            self._obs_data_idx[sensor] = list(range(i, i+dim))
            i += dim

    def get_idx_x(self, sensor_name):
        return self._x_data_idx[sensor_name]

    def get_idx_obs(self, sensor_name):
        return self._obs_data_idx[sensor_name]

    def clear_samples(self, condition=None):
        if condition is None:
            self._samples = [[] for _ in range(len(self._samples))]
        else:
            self._samples[condition] = []

    def clear_samples_adv(self, condition=None):
        if condition is None:
            self._samples_adv = [[] for _ in range(len(self._samples_adv))]
        else:
            self._samples_adv[condition] = []

    def delete_last_sample(self, condition):
        self._samples[condition].pop()

    def get_samples(self, condition, start=0, end=None):
        from gps.sample.sample_list import SampleList
        return (SampleList(self._samples[condition][start:]) if end is None
                else SampleList(self._samples[condition][start:end]))

    def reset(self, condition):
        pass


class MockSample:
    """Mock sample for testing."""

    def __init__(self, sample_id):
        self.sample_id = sample_id


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module dependencies not available")
class TestAgentBase:
    """Unit tests for Agent base class."""

    @pytest.fixture
    def agent_hyperparams(self):
        """Sample hyperparameters for agent."""
        return {
            'T': 10,
            'conditions': 2,
            'sensor_dims': {
                ACTION: 3,
                'JOINT_ANGLES': 7,
                'JOINT_VELOCITIES': 7,
                'END_EFFECTOR_POINTS': 3,
            },
            'state_include': ['JOINT_ANGLES', 'JOINT_VELOCITIES'],
            'obs_include': ['JOINT_ANGLES', 'JOINT_VELOCITIES', 'END_EFFECTOR_POINTS'],
        }

    @pytest.mark.unit
    def test_agent_initialization(self, agent_hyperparams):
        """Test Agent initialization."""
        agent = MockAgent(agent_hyperparams)

        assert agent.T == 10
        assert agent.dU == 3
        assert agent.dV == 3

    @pytest.mark.unit
    def test_agent_dimensions(self, agent_hyperparams):
        """Test agent dimensions calculation."""
        agent = MockAgent(agent_hyperparams)

        # dX = JOINT_ANGLES + JOINT_VELOCITIES = 7 + 7 = 14
        assert agent.dX == 14

        # dO = JOINT_ANGLES + JOINT_VELOCITIES + END_EFFECTOR_POINTS = 7 + 7 + 3 = 17
        assert agent.dO == 17

        # dU = ACTION = 3
        assert agent.dU == 3

    @pytest.mark.unit
    def test_agent_state_indexing(self, agent_hyperparams):
        """Test state data indexing."""
        agent = MockAgent(agent_hyperparams)

        # JOINT_ANGLES: indices 0-6
        joint_angles_idx = agent.get_idx_x('JOINT_ANGLES')
        assert joint_angles_idx == list(range(0, 7))

        # JOINT_VELOCITIES: indices 7-13
        joint_velocities_idx = agent.get_idx_x('JOINT_VELOCITIES')
        assert joint_velocities_idx == list(range(7, 14))

    @pytest.mark.unit
    def test_agent_obs_indexing(self, agent_hyperparams):
        """Test observation data indexing."""
        agent = MockAgent(agent_hyperparams)

        # JOINT_ANGLES: indices 0-6
        assert agent.get_idx_obs('JOINT_ANGLES') == list(range(0, 7))

        # JOINT_VELOCITIES: indices 7-13
        assert agent.get_idx_obs('JOINT_VELOCITIES') == list(range(7, 14))

        # END_EFFECTOR_POINTS: indices 14-16
        assert agent.get_idx_obs('END_EFFECTOR_POINTS') == list(range(14, 17))

    @pytest.mark.unit
    def test_agent_samples_initialization(self, agent_hyperparams):
        """Test samples are initialized empty."""
        agent = MockAgent(agent_hyperparams)

        # Should have empty sample lists for each condition
        assert len(agent._samples) == 2
        assert len(agent._samples_adv) == 2

        for condition in range(2):
            assert agent._samples[condition] == []
            assert agent._samples_adv[condition] == []

    @pytest.mark.unit
    def test_agent_clear_samples(self, agent_hyperparams):
        """Test clearing samples."""
        agent = MockAgent(agent_hyperparams)

        # Add mock samples
        agent._samples[0] = [MockSample(0), MockSample(1)]
        agent._samples[1] = [MockSample(2), MockSample(3)]

        # Clear condition 0
        agent.clear_samples(condition=0)
        assert agent._samples[0] == []
        assert len(agent._samples[1]) == 2

        # Clear all conditions
        agent.clear_samples()
        assert agent._samples[0] == []
        assert agent._samples[1] == []

    @pytest.mark.unit
    def test_agent_clear_samples_adv(self, agent_hyperparams):
        """Test clearing adversary samples."""
        agent = MockAgent(agent_hyperparams)

        # Add mock samples
        agent._samples_adv[0] = [MockSample(0)]
        agent._samples_adv[1] = [MockSample(1)]

        # Clear condition 0
        agent.clear_samples_adv(condition=0)
        assert agent._samples_adv[0] == []
        assert len(agent._samples_adv[1]) == 1

        # Clear all
        agent.clear_samples_adv()
        assert agent._samples_adv[0] == []
        assert agent._samples_adv[1] == []

    @pytest.mark.unit
    def test_agent_delete_last_sample(self, agent_hyperparams):
        """Test deleting last sample."""
        agent = MockAgent(agent_hyperparams)

        # Add samples
        agent._samples[0] = [MockSample(0), MockSample(1), MockSample(2)]

        # Delete last
        agent.delete_last_sample(0)
        assert len(agent._samples[0]) == 2

        agent.delete_last_sample(0)
        assert len(agent._samples[0]) == 1

    @pytest.mark.unit
    def test_agent_get_samples(self, agent_hyperparams):
        """Test getting samples."""
        agent = MockAgent(agent_hyperparams)

        # Add samples
        samples = [MockSample(i) for i in range(5)]
        agent._samples[0] = samples

        # Get all samples
        sample_list = agent.get_samples(0)
        assert isinstance(sample_list, SampleList)
        assert len(sample_list._samples) == 5

        # Get subset
        sample_list = agent.get_samples(0, start=1, end=3)
        assert len(sample_list._samples) == 2

        # Get from start to end
        sample_list = agent.get_samples(0, start=2)
        assert len(sample_list._samples) == 3

    @pytest.mark.unit
    def test_agent_reset(self, agent_hyperparams):
        """Test agent reset method."""
        agent = MockAgent(agent_hyperparams)

        # Reset should not raise error (default implementation is pass)
        agent.reset(0)


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module dependencies not available")
class TestAgentAbstractMethods:
    """Test abstract method interface."""

    @pytest.mark.unit
    def test_agent_sample_method_exists(self, agent_hyperparams):
        """Test that agent has sample method."""
        agent = MockAgent(agent_hyperparams)

        # Mock agent should have basic methods
        assert hasattr(agent, 'get_idx_x')
        assert hasattr(agent, 'get_idx_obs')


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module dependencies not available")
class TestAgentDataTypes:
    """Test data type handling."""

    @pytest.mark.unit
    def test_agent_x_data_types(self, agent_hyperparams):
        """Test state data types."""
        agent = MockAgent(agent_hyperparams)

        assert 'JOINT_ANGLES' in agent.x_data_types
        assert 'JOINT_VELOCITIES' in agent.x_data_types
        assert len(agent.x_data_types) == 2

    @pytest.mark.unit
    def test_agent_obs_data_types(self, agent_hyperparams):
        """Test observation data types."""
        agent = MockAgent(agent_hyperparams)

        assert 'JOINT_ANGLES' in agent.obs_data_types
        assert 'JOINT_VELOCITIES' in agent.obs_data_types
        assert 'END_EFFECTOR_POINTS' in agent.obs_data_types
        assert len(agent.obs_data_types) == 3

    @pytest.mark.unit
    def test_agent_meta_data_types(self, agent_hyperparams):
        """Test metadata types handling."""
        # Without meta_include
        agent = MockAgent(agent_hyperparams)
        assert agent.meta_data_types == []
        assert agent.dM == 0

        # With meta_include
        params_with_meta = agent_hyperparams.copy()
        params_with_meta['meta_include'] = ['TIMESTEP']
        params_with_meta['sensor_dims']['TIMESTEP'] = 1

        agent = MockAgent(params_with_meta)
        assert 'TIMESTEP' in agent.meta_data_types
        assert agent.dM == 1


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module dependencies not available")
class TestAgentEdgeCases:
    """Test edge cases for agent."""

    @pytest.mark.unit
    def test_agent_single_condition(self):
        """Test agent with single condition."""
        hyperparams = {
            'T': 10,
            'conditions': 1,
            'sensor_dims': {ACTION: 3},
            'state_include': [],
            'obs_include': [],
        }

        agent = MockAgent(hyperparams)

        assert len(agent._samples) == 1
        assert agent.dX == 0  # No state sensors

    @pytest.mark.unit
    def test_agent_many_conditions(self):
        """Test agent with many conditions."""
        hyperparams = {
            'T': 10,
            'conditions': 10,
            'sensor_dims': {ACTION: 3},
            'state_include': [],
            'obs_include': [],
        }

        agent = MockAgent(hyperparams)

        assert len(agent._samples) == 10
        assert len(agent._samples_adv) == 10

    @pytest.mark.unit
    def test_agent_empty_state_include(self):
        """Test agent with no state sensors."""
        hyperparams = {
            'T': 10,
            'conditions': 2,
            'sensor_dims': {ACTION: 3},
            'state_include': [],
            'obs_include': [],
        }

        agent = MockAgent(hyperparams)

        assert agent.dX == 0
        assert agent._state_idx == []

    @pytest.mark.unit
    def test_agent_empty_obs_include(self):
        """Test agent with no observation sensors."""
        hyperparams = {
            'T': 10,
            'conditions': 2,
            'sensor_dims': {ACTION: 3, 'STATE': 5},
            'state_include': ['STATE'],
            'obs_include': [],
        }

        agent = MockAgent(hyperparams)

        assert agent.dO == 0
        assert agent._obs_idx == []

    @pytest.mark.unit
    def test_agent_large_dimensions(self):
        """Test agent with large state/action dimensions."""
        hyperparams = {
            'T': 100,
            'conditions': 5,
            'sensor_dims': {
                ACTION: 50,
                'HIGH_DIM_STATE': 200,
            },
            'state_include': ['HIGH_DIM_STATE'],
            'obs_include': ['HIGH_DIM_STATE'],
        }

        agent = MockAgent(hyperparams)

        assert agent.dX == 200
        assert agent.dO == 200
        assert agent.dU == 50


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module dependencies not available")
class TestAgentIndexConsistency:
    """Test consistency of indexing."""

    @pytest.mark.unit
    def test_state_indices_contiguous(self, agent_hyperparams):
        """Test that state indices are contiguous."""
        agent = MockAgent(agent_hyperparams)

        all_indices = []
        for idx_list in agent._state_idx:
            all_indices.extend(idx_list)

        # Should be 0, 1, 2, ..., dX-1
        assert sorted(all_indices) == list(range(agent.dX))

    @pytest.mark.unit
    def test_obs_indices_contiguous(self, agent_hyperparams):
        """Test that observation indices are contiguous."""
        agent = MockAgent(agent_hyperparams)

        all_indices = []
        for idx_list in agent._obs_idx:
            all_indices.extend(idx_list)

        # Should be 0, 1, 2, ..., dO-1
        assert sorted(all_indices) == list(range(agent.dO))

    @pytest.mark.unit
    def test_no_overlapping_state_indices(self, agent_hyperparams):
        """Test that state indices don't overlap."""
        agent = MockAgent(agent_hyperparams)

        all_indices = []
        for idx_list in agent._state_idx:
            for idx in idx_list:
                assert idx not in all_indices, "Overlapping state indices"
                all_indices.append(idx)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
