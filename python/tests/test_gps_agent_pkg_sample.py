"""
Tests for python/gps_agent_pkg/sample.py (ControllerSample).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from gps_agent_pkg.sample import (
    ControllerSample, SAMPLE_FORMAT_VECTOR, SAMPLE_FORMAT_MATRIX,
)

# Arbitrary dtype integers (matching gps.proto SampleType values)
DTYPE_A = 0   # JOINT_ANGLES
DTYPE_B = 1   # JOINT_VELOCITIES
DTYPE_C = 2   # END_EFFECTOR_POINTS
DTYPE_M = 7   # END_EFFECTOR_JACOBIANS (matrix type)


class TestControllerSampleBasic:
    def test_init(self):
        s = ControllerSample(T=10)
        assert s.get_T() == 10

    def test_set_and_get_vector(self):
        s = ControllerSample(T=5)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        arr = np.array([1.0, 2.0, 3.0])
        s.set_data(0, DTYPE_A, arr)
        result = s.get_data_vec(0, [DTYPE_A])
        np.testing.assert_allclose(result, arr)

    def test_multiple_timesteps(self):
        s = ControllerSample(T=5)
        s.set_meta_data(DTYPE_A, 2, fmt=SAMPLE_FORMAT_VECTOR)
        for t in range(5):
            s.set_data(t, DTYPE_A, np.array([float(t), float(t + 1)]))
        for t in range(5):
            v = s.get_data_vec(t, [DTYPE_A])
            assert v[0] == pytest.approx(t)
            assert v[1] == pytest.approx(t + 1)

    def test_get_data_flattens_T_timesteps(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_A, 2, fmt=SAMPLE_FORMAT_VECTOR)
        s.set_data(0, DTYPE_A, np.array([1.0, 2.0]))
        s.set_data(1, DTYPE_A, np.array([3.0, 4.0]))
        s.set_data(2, DTYPE_A, np.array([5.0, 6.0]))
        out = s.get_data(3, DTYPE_A)
        np.testing.assert_allclose(out, [1., 2., 3., 4., 5., 6.])

    def test_get_data_vec_concat_multiple_dtypes(self):
        s = ControllerSample(T=2)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        s.set_meta_data(DTYPE_B, 3, fmt=SAMPLE_FORMAT_VECTOR)
        s.set_data(0, DTYPE_A, np.array([1., 2., 3.]))
        s.set_data(0, DTYPE_B, np.array([4., 5., 6.]))
        result = s.get_data_vec(0, [DTYPE_A, DTYPE_B])
        np.testing.assert_allclose(result, [1., 2., 3., 4., 5., 6.])

    def test_pre_allocated_zeros(self):
        s = ControllerSample(T=4)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        # Without setting data, should return zeros
        result = s.get_data_vec(0, [DTYPE_A])
        np.testing.assert_allclose(result, [0., 0., 0.])

    def test_get_available_dtypes(self):
        s = ControllerSample(T=4)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        s.set_meta_data(DTYPE_B, 3, fmt=SAMPLE_FORMAT_VECTOR)
        dtypes = s.get_available_dtypes()
        assert DTYPE_A in dtypes
        assert DTYPE_B in dtypes
        assert DTYPE_C not in dtypes

    def test_out_of_bounds_t_ignored(self):
        s = ControllerSample(T=2)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        # Setting at t >= T should be silently ignored
        s.set_data(5, DTYPE_A, np.array([9., 9., 9.]))
        result = s.get_data_vec(0, [DTYPE_A])
        np.testing.assert_allclose(result, [0., 0., 0.])


class TestControllerSampleMatrix:
    def test_set_and_get_matrix(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_M, 2, 3, fmt=SAMPLE_FORMAT_MATRIX)
        M = np.array([[1., 2., 3.], [4., 5., 6.]])
        s.set_data(0, DTYPE_M, M)
        result = s.get_data_vec(0, [DTYPE_M])
        # Row-major flattening: [1, 2, 3, 4, 5, 6]
        np.testing.assert_allclose(result, [1., 2., 3., 4., 5., 6.])

    def test_matrix_shape(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_M, 4, 7, fmt=SAMPLE_FORMAT_MATRIX)
        shape = s.get_shape(DTYPE_M)
        assert shape == [4, 7]

    def test_vector_shape(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_A, 5, fmt=SAMPLE_FORMAT_VECTOR)
        shape = s.get_shape(DTYPE_A)
        assert shape == [5]

    def test_total_size_matrix(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_M, 3, 7, fmt=SAMPLE_FORMAT_MATRIX)
        # 3*7=21 elements
        out = s.get_data(1, DTYPE_M)
        assert len(out) == 21


class TestControllerSampleSetDataVector:
    def test_set_data_vector_shape_mismatch_ignored(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        # Providing wrong-size array should be silently ignored
        s.set_data_vector(0, DTYPE_A, np.array([1., 2., 3., 4.]))
        # Data should remain zeros
        result = s.get_data_vec(0, [DTYPE_A])
        np.testing.assert_allclose(result, [0., 0., 0.])

    def test_set_data_vector_correct_shape(self):
        s = ControllerSample(T=3)
        s.set_meta_data(DTYPE_A, 3, fmt=SAMPLE_FORMAT_VECTOR)
        s.set_data_vector(0, DTYPE_A, np.array([7., 8., 9.]))
        result = s.get_data_vec(0, [DTYPE_A])
        np.testing.assert_allclose(result, [7., 8., 9.])
