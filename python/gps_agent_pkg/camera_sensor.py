"""
Translates gps_agent_pkg/src/camerasensor.cpp + include/camerasensor.h.

The C++ pixel loop is replaced by numpy center-crop slicing.
rclpy subscriptions are created for RGB and depth image topics.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from gps_agent_pkg.sensor import Sensor
from gps_agent_pkg.sample import ControllerSample, SAMPLE_FORMAT_VECTOR

# Default dimensions from C++ header
_IMAGE_WIDTH       = 80
_IMAGE_HEIGHT      = 60
_IMAGE_WIDTH_INIT  = 640
_IMAGE_HEIGHT_INIT = 480

# SampleType int values matching gps.proto
_RGB_IMAGE   = 10
_DEPTH_IMAGE = 11


class CameraSensor(Sensor):
    """
    RGB + depth camera sensor.  Image callbacks perform a center-crop.

    Mirrors C++ gps_control::CameraSensor.
    """

    def __init__(self, node, plugin) -> None:
        super().__init__(node, plugin)

        # Read parameters with defaults
        def _param(name: str, default):
            try:
                return node.get_parameter(name).get_parameter_value().string_value or default
            except Exception:
                return default

        self._rgb_topic_name   = _param("rgb_topic",   "/camera/rgb/image_color")
        self._depth_topic_name = _param("depth_topic", "/camera/depth_registered/image_raw")

        self._image_width       = int(_param("image_width",       _IMAGE_WIDTH))
        self._image_height      = int(_param("image_height",      _IMAGE_HEIGHT))
        self._image_width_init  = int(_param("image_width_init",  _IMAGE_WIDTH_INIT))
        self._image_height_init = int(_param("image_height_init", _IMAGE_HEIGHT_INIT))

        self._image_size = self._image_width * self._image_height

        self._latest_rgb_image   = np.zeros(self._image_size * 3, dtype=np.uint8)
        self._latest_depth_image = np.zeros(self._image_size,     dtype=np.uint16)

        self._latest_rgb_time   = None
        self._latest_depth_time = None

        # Set up rclpy subscriptions if the node provides them
        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from sensor_msgs.msg import Image
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            )
            if self._rgb_topic_name:
                self._rgb_sub = node.create_subscription(
                    Image, self._rgb_topic_name, self._update_rgb_image, qos
                )
            if self._depth_topic_name:
                self._depth_sub = node.create_subscription(
                    Image, self._depth_topic_name, self._update_depth_image, qos
                )
        except Exception:
            pass  # rclpy not available in test environment

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _update_rgb_image(self, msg) -> None:
        """
        Center-crop the incoming RGB image.
        Mirrors C++ CameraSensor::update_rgb_image.
        """
        self._latest_rgb_time = msg.header.stamp
        h_init = self._image_height_init
        w_init = self._image_width_init
        h_crop = self._image_height
        w_crop = self._image_width

        x_start = (w_init - w_crop) // 2
        y_start = (h_init - h_crop) // 2

        data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h_init, w_init, 3)
        cropped = data[y_start:y_start + h_crop, x_start:x_start + w_crop, :]
        self._latest_rgb_image = cropped.flatten()

    def _update_depth_image(self, msg) -> None:
        """
        Center-crop the incoming depth image.
        Mirrors C++ CameraSensor::update_depth_image.
        """
        self._latest_depth_time = msg.header.stamp
        h_init = self._image_height_init
        w_init = self._image_width_init
        h_crop = self._image_height
        w_crop = self._image_width

        x_start = (w_init - w_crop) // 2
        y_start = (h_init - h_crop) // 2

        data = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h_init, w_init)
        cropped = data[y_start:y_start + h_crop, x_start:x_start + w_crop]
        self._latest_depth_image = cropped.flatten()

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def update(self, plugin, current_time: float, is_controller_step: bool) -> None:
        pass  # Data updates via callbacks

    def configure_sensor(self, options: dict) -> None:
        pass

    def set_sample_data_format(self, sample: ControllerSample) -> None:
        sample.set_meta_data(_RGB_IMAGE,   self._image_size * 3, fmt=SAMPLE_FORMAT_VECTOR)
        sample.set_meta_data(_DEPTH_IMAGE, self._image_size * 2, fmt=SAMPLE_FORMAT_VECTOR)

    def set_sample_data(self, sample: ControllerSample, t: int) -> None:
        sample.set_data(t, _RGB_IMAGE,   self._latest_rgb_image.astype(np.float64))
        sample.set_data(t, _DEPTH_IMAGE, self._latest_depth_image.astype(np.float64))
