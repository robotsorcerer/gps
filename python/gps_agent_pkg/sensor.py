"""
Translates gps_agent_pkg/src/sensor.cpp + include/gps_agent_pkg/sensor.h.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gps_agent_pkg.sample import ControllerSample


class SensorType(IntEnum):
    EncoderSensorType  = 0
    ROSTopicSensorType = 1
    CameraSensorType   = 2


class Sensor(ABC):
    """
    Abstract base class for all robot sensors.

    Mirrors C++ gps_control::Sensor.
    """

    def __init__(self, node, plugin) -> None:
        self.sensor_step_length_: float = 1.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_sensor(
        sensor_type: SensorType,
        node,
        plugin,
        actuator_type: int,
    ) -> "Sensor":
        """
        Mirrors C++ Sensor::create_sensor factory function.
        Imports are deferred to avoid circular dependencies.
        """
        from gps_agent_pkg.encoder_sensor import EncoderSensor
        from gps_agent_pkg.ros_topic_sensor import ROSTopicSensor
        from gps_agent_pkg.camera_sensor import CameraSensor

        if sensor_type == SensorType.EncoderSensorType:
            return EncoderSensor(node, plugin, actuator_type)
        elif sensor_type == SensorType.ROSTopicSensorType:
            return ROSTopicSensor(node, plugin)
        elif sensor_type == SensorType.CameraSensorType:
            return CameraSensor(node, plugin)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, plugin, current_time: float) -> None:
        pass

    def update(self, plugin, current_time: float, is_controller_step: bool) -> None:
        pass

    def set_update(self, step_length: float) -> None:
        """Set the expected time between controller steps (seconds)."""
        self.sensor_step_length_ = step_length

    def configure_sensor(self, options: dict) -> None:
        pass

    def set_sample_data_format(self, sample: "ControllerSample") -> None:
        pass

    def set_sample_data(self, sample: "ControllerSample", t: int) -> None:
        pass
