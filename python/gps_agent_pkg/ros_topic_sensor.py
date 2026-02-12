"""
Translates gps_agent_pkg/src/rostopicsensor.cpp + include/rostopicsensor.h.

Subscribes to a std_msgs/Float64MultiArray topic and stores the latest
64-element feature vector, populating gps::IMAGE_FEAT in the sample.
"""
from __future__ import annotations

import numpy as np

from gps_agent_pkg.sensor import Sensor
from gps_agent_pkg.sample import ControllerSample, SAMPLE_FORMAT_VECTOR

# SampleType int value for IMAGE_FEAT (gps.proto)
_IMAGE_FEAT = 12


class ROSTopicSensor(Sensor):
    """
    Generic ROS-topic sensor for Float64MultiArray feature vectors.

    Mirrors C++ gps_control::ROSTopicSensor.
    """

    def __init__(self, node, plugin) -> None:
        super().__init__(node, plugin)

        try:
            topic_name: str = node.get_parameter("feat_topic").get_parameter_value().string_value
        except Exception:
            topic_name = "/caffe_features_publisher"
        if not topic_name:
            topic_name = "/caffe_features_publisher"

        self._topic_name = topic_name
        self._data_size  = 64
        self._latest_data = np.zeros(self._data_size, dtype=np.float64)

        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from std_msgs.msg import Float64MultiArray
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._subscriber = node.create_subscription(
                Float64MultiArray, self._topic_name,
                self._update_data_vector, qos,
            )
        except Exception:
            pass  # rclpy not available in test environment

    # ------------------------------------------------------------------
    # ROS callback
    # ------------------------------------------------------------------

    def _update_data_vector(self, msg) -> None:
        """
        Mirrors C++ ROSTopicSensor::update_data_vector.
        Re-sizes data buffer on first message if dimensions differ.
        """
        size = len(msg.data)
        if size != self._data_size:
            self._data_size  = size
            self._latest_data = np.zeros(size, dtype=np.float64)
        for i, v in enumerate(msg.data):
            self._latest_data[i] = v

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def update(self, plugin, current_time: float, is_controller_step: bool) -> None:
        pass  # Data comes via subscription callback

    def configure_sensor(self, options: dict) -> None:
        pass

    def set_sample_data_format(self, sample: ControllerSample) -> None:
        sample.set_meta_data(_IMAGE_FEAT, self._data_size, fmt=SAMPLE_FORMAT_VECTOR)

    def set_sample_data(self, sample: ControllerSample, t: int) -> None:
        sample.set_data(t, _IMAGE_FEAT, self._latest_data)
