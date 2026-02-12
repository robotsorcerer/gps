"""
Tests for python/gps_agent_pkg/encoder_filter.py (EncoderFilter).
Verifies the linear state-filter math against hand-computed reference values.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock

from gps_agent_pkg.encoder_filter import EncoderFilter


def _make_filter(A: np.ndarray, h: np.ndarray, initial_state: np.ndarray) -> EncoderFilter:
    """
    Build an EncoderFilter without a live ROS node by calling _configure
    directly with a synthetic parameter string.

    The parameter string format is:
      Line 0: A entries in column-major order (space-separated)
      Line 1: h entries (space-separated)
    """
    order = len(h)
    # Encode A in column-major order
    A_col_major = A.flatten(order="F")
    line0 = " ".join(str(v) for v in A_col_major)
    line1 = " ".join(str(v) for v in h)
    params = line0 + "\n" + line1

    # Build a mock node that has no encoder_filter_params
    node = MagicMock()
    node.get_parameter.side_effect = Exception("no param")

    filt = EncoderFilter.__new__(EncoderFilter)
    filt._num_joints = len(initial_state)
    filt._is_configured = False
    filt._configure(params)
    filt._filtered_state[0, :] = initial_state
    return filt


class TestEncoderFilterMath:
    """
    Verify  filtered_state = A @ filtered_state + outer(h, state).

    We use a simple 2x2 system so the answer is easy to compute by hand.
    """

    def test_single_update_known_values(self):
        # 2-order filter, 1 joint
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        h = np.array([0.5, 0.2])
        q0 = np.array([1.0])
        filt = _make_filter(A, h, q0)

        state_input = np.array([2.0])
        filt.update(0.01, state_input)

        # Expected:  A @ [[1.0], [0.0]] + outer([0.5, 0.2], [2.0])
        # = [[0.9], [0.0]] + [[1.0], [0.4]] = [[1.9], [0.4]]
        expected_pos = 1.9
        expected_vel = 0.4
        assert filt.get_state()[0]    == pytest.approx(expected_pos, abs=1e-9)
        assert filt.get_velocity()[0] == pytest.approx(expected_vel, abs=1e-9)

    def test_identity_matrix_passthrough(self):
        # With A = I and h = [1, 0], state should just accumulate
        A = np.eye(2)
        h = np.array([1.0, 0.0])
        q0 = np.array([0.0, 0.0])
        filt = _make_filter(A, h, np.array([0.0, 0.0]))

        filt.update(0.01, np.array([3.0, 4.0]))
        # filtered = I @ [[0,0],[0,0]] + outer([1,0], [3,4]) = [[3,4],[0,0]]
        np.testing.assert_allclose(filt.get_state(), [3.0, 4.0])
        np.testing.assert_allclose(filt.get_velocity(), [0.0, 0.0])

    def test_two_joints(self):
        A = np.array([[0.5, 0.0], [0.0, 0.5]])
        h = np.array([1.0, 0.0])
        initial = np.array([0.0, 0.0])
        filt = _make_filter(A, h, initial)

        filt.update(0.01, np.array([2.0, 4.0]))
        # filtered = [[0,0],[0,0]] + outer([1,0],[2,4]) = [[2,4],[0,0]]
        np.testing.assert_allclose(filt.get_state(), [2.0, 4.0])

    def test_multiple_updates_accumulate(self):
        A = np.array([[0.9, 0.0], [0.0, 0.9]])
        h = np.array([0.1, 0.0])
        initial = np.zeros(1)
        filt = _make_filter(A, h, initial)

        # Step 1: filtered = 0 + outer([0.1,0],[10]) = [[1.0],[0.0]]
        filt.update(0.01, np.array([10.0]))
        assert filt.get_state()[0] == pytest.approx(1.0, abs=1e-9)

        # Step 2: filtered = A @ [[1.0],[0.0]] + outer([0.1,0],[10])
        #                   = [[0.9],[0.0]] + [[1.0],[0.0]] = [[1.9],[0.0]]
        filt.update(0.01, np.array([10.0]))
        assert filt.get_state()[0] == pytest.approx(1.9, abs=1e-9)

    def test_is_configured_true_after_configure(self):
        A = np.eye(2)
        h = np.array([1.0, 0.0])
        filt = _make_filter(A, h, np.zeros(3))
        assert filt.is_configured is True

    def test_get_state_returns_copy(self):
        A = np.eye(2)
        h = np.array([1.0, 0.0])
        filt = _make_filter(A, h, np.array([5.0, 6.0]))
        s = filt.get_state()
        s[0] = 999.0  # Mutate return value
        assert filt.get_state()[0] == pytest.approx(5.0)  # Original unchanged


class TestEncoderFilterRealParams:
    """Test using the actual encoder_filter_params.txt values."""

    def test_real_params_order_3(self):
        # From encoder_filter_params.txt (3-order filter)
        A_flat_col_major = [
            0.29752, -106.5, -6447.3059,
            0.00029752, 0.8935, -6.4473,
            1.4876e-07, 0.00094675, 0.99678,
        ]
        h_vals = [0.70248, 106.5, 6447.3059]
        A = np.array(A_flat_col_major).reshape(3, 3, order="F")
        h = np.array(h_vals)
        initial = np.zeros(7)  # 7 joints
        filt = _make_filter(A, h, initial)

        assert filt.is_configured is True
        assert filt.get_state().shape == (7,)
        assert filt.get_velocity().shape == (7,)

        # Update with random joint angles and check shapes are preserved
        filt.update(0.001, np.ones(7) * 0.1)
        assert filt.get_state().shape == (7,)
