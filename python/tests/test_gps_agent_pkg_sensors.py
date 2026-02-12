"""
Tests for the sensor classes:
  Sensor (factory), ROSTopicSensor, CameraSensor.
All tests are pure-Python (no live ROS required).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from gps_agent_pkg.sensor import Sensor, SensorType
from gps_agent_pkg.sample import ControllerSample, SAMPLE_FORMAT_VECTOR
from gps_agent_pkg.ros_topic_sensor import ROSTopicSensor, _IMAGE_FEAT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_node():
    node = MagicMock()
    node.get_parameter.side_effect = Exception("no param")
    return node


def _mock_plugin(n_joints=7):
    plugin = MagicMock()
    plugin.get_joint_encoder_readings.return_value = np.zeros(n_joints)
    return plugin


# ---------------------------------------------------------------------------
# Sensor factory
# ---------------------------------------------------------------------------

class TestSensorFactory:
    def test_factory_encoder(self):
        node   = _mock_node()
        plugin = _mock_plugin()
        sensor = Sensor.create_sensor(SensorType.EncoderSensorType, node, plugin, 0)
        from gps_agent_pkg.encoder_sensor import EncoderSensor
        assert isinstance(sensor, EncoderSensor)

    def test_factory_rostopic(self):
        node = _mock_node()
        sensor = Sensor.create_sensor(SensorType.ROSTopicSensorType, node, None, 0)
        assert isinstance(sensor, ROSTopicSensor)

    def test_factory_camera(self):
        node = _mock_node()
        sensor = Sensor.create_sensor(SensorType.CameraSensorType, node, None, 0)
        from gps_agent_pkg.camera_sensor import CameraSensor
        assert isinstance(sensor, CameraSensor)

    def test_factory_unknown_raises(self):
        node = _mock_node()
        with pytest.raises(ValueError):
            Sensor.create_sensor(99, node, None, 0)  # type: ignore

    def test_set_update_stores_step_length(self):
        node   = _mock_node()
        plugin = _mock_plugin()
        sensor = Sensor.create_sensor(SensorType.EncoderSensorType, node, plugin, 0)
        sensor.set_update(0.05)
        assert sensor.sensor_step_length_ == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# ROSTopicSensor
# ---------------------------------------------------------------------------

class TestROSTopicSensor:
    def test_default_data_size(self):
        node = _mock_node()
        sensor = ROSTopicSensor(node, None)
        assert sensor._data_size == 64
        assert sensor._latest_data.shape == (64,)

    def test_set_sample_data_format(self):
        node = _mock_node()
        sensor = ROSTopicSensor(node, None)
        s = ControllerSample(T=3)
        sensor.set_sample_data_format(s)
        assert _IMAGE_FEAT in s.get_available_dtypes()
        assert s.get_shape(_IMAGE_FEAT) == [64]

    def test_set_sample_data_stores_latest(self):
        node = _mock_node()
        sensor = ROSTopicSensor(node, None)
        sensor._latest_data = np.arange(64, dtype=np.float64)
        s = ControllerSample(T=3)
        sensor.set_sample_data_format(s)
        sensor.set_sample_data(s, 0)
        v = s.get_data_vec(0, [_IMAGE_FEAT])
        np.testing.assert_allclose(v, np.arange(64, dtype=np.float64))

    def test_callback_updates_data(self):
        node = _mock_node()
        sensor = ROSTopicSensor(node, None)
        # Simulate a Float64MultiArray message
        msg = MagicMock()
        msg.data = list(range(64))
        sensor._update_data_vector(msg)
        np.testing.assert_allclose(sensor._latest_data, np.arange(64, dtype=float))

    def test_callback_resizes_for_different_data(self):
        node = _mock_node()
        sensor = ROSTopicSensor(node, None)
        msg = MagicMock()
        msg.data = list(range(128))  # larger than default 64
        sensor._update_data_vector(msg)
        assert sensor._data_size == 128
        assert sensor._latest_data.shape == (128,)

    def test_update_is_noop(self):
        node = _mock_node()
        sensor = ROSTopicSensor(node, None)
        # update() should not raise
        sensor.update(None, 0.0, True)


# ---------------------------------------------------------------------------
# CameraSensor
# ---------------------------------------------------------------------------

class TestCameraSensor:
    def test_default_image_sizes(self):
        from gps_agent_pkg.camera_sensor import CameraSensor, _IMAGE_WIDTH, _IMAGE_HEIGHT
        node = _mock_node()
        sensor = CameraSensor(node, None)
        assert sensor._image_width  == _IMAGE_WIDTH
        assert sensor._image_height == _IMAGE_HEIGHT

    def test_initial_arrays_are_zero(self):
        from gps_agent_pkg.camera_sensor import CameraSensor
        node = _mock_node()
        sensor = CameraSensor(node, None)
        assert sensor._latest_rgb_image.sum()   == 0
        assert sensor._latest_depth_image.sum() == 0

    def test_set_sample_data_format(self):
        from gps_agent_pkg.camera_sensor import CameraSensor, _RGB_IMAGE, _DEPTH_IMAGE
        node = _mock_node()
        sensor = CameraSensor(node, None)
        s = ControllerSample(T=2)
        sensor.set_sample_data_format(s)
        dtypes = s.get_available_dtypes()
        assert _RGB_IMAGE   in dtypes
        assert _DEPTH_IMAGE in dtypes

    def test_rgb_callback_center_crops(self):
        from gps_agent_pkg.camera_sensor import CameraSensor
        node = _mock_node()
        sensor = CameraSensor(node, None)

        H_init = sensor._image_height_init  # 480
        W_init = sensor._image_width_init   # 640

        # Build a fake RGB image where pixel (y, x) is RGB = (y % 256, x % 256, 0)
        raw = np.zeros((H_init, W_init, 3), dtype=np.uint8)
        for y in range(H_init):
            for x in range(W_init):
                raw[y, x, 0] = y % 256
                raw[y, x, 1] = x % 256

        msg = MagicMock()
        msg.header.stamp = 0
        msg.data = raw.flatten().tobytes()

        sensor._update_rgb_image(msg)

        # Verify the cropped image has the right size
        assert len(sensor._latest_rgb_image) == sensor._image_width * sensor._image_height * 3

    def test_update_is_noop(self):
        from gps_agent_pkg.camera_sensor import CameraSensor
        node = _mock_node()
        sensor = CameraSensor(node, None)
        sensor.update(None, 0.0, True)  # must not raise


# ---------------------------------------------------------------------------
# EncoderSensor (basic — no KDL)
# ---------------------------------------------------------------------------

class TestEncoderSensorBasic:
    def test_init_reads_joint_angles(self):
        from gps_agent_pkg.encoder_sensor import EncoderSensor
        node   = _mock_node()
        plugin = _mock_plugin(n_joints=7)
        sensor = EncoderSensor(node, plugin, 0)
        plugin.get_joint_encoder_readings.assert_called()
        assert sensor._previous_angles.shape == (7,)

    def test_set_sample_data_format(self):
        from gps_agent_pkg.encoder_sensor import EncoderSensor, _JOINT_ANGLES, _JOINT_VELOCITIES
        node   = _mock_node()
        plugin = _mock_plugin(n_joints=7)
        sensor = EncoderSensor(node, plugin, 0)
        s = ControllerSample(T=3)
        sensor.set_sample_data_format(s)
        dtypes = s.get_available_dtypes()
        assert _JOINT_ANGLES    in dtypes
        assert _JOINT_VELOCITIES in dtypes

    def test_configure_sensor_updates_n_points(self):
        from gps_agent_pkg.encoder_sensor import EncoderSensor
        node   = _mock_node()
        plugin = _mock_plugin(n_joints=7)
        sensor = EncoderSensor(node, plugin, 0)
        opts = {
            "ee_sites":      np.zeros((3, 3)),   # 3 points
            "ee_points_tgt": np.zeros((3, 3)),
        }
        sensor.configure_sensor(opts)
        assert sensor._n_points == 3
