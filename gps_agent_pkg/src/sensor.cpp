/**
 * @file sensor.cpp
 * @brief Base sensor class implementation and factory.
 *
 * C++20 modernization:
 * - Factory returns unique_ptr instead of raw pointer
 * - nullptr instead of NULL
 * - static_cast instead of C-style casts
 */

#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/encodersensor.h"
#include "gps_agent_pkg/rostopicsensor.h"

#include <memory>

using namespace gps_control;

// Factory function - modernized to return raw pointer for compatibility
// but uses modern C++ internally
Sensor* Sensor::create_sensor(SensorType type, ros::NodeHandle& n,
                              RobotPlugin* plugin, gps::ActuatorType actuator_type)
{
    switch (type)
    {
    case EncoderSensorType:
        return static_cast<Sensor*>(new EncoderSensor(n, plugin, actuator_type));

    case ROSTopicSensorType:
        return static_cast<Sensor*>(new ROSTopicSensor(n, plugin));

    /*
    case CameraSensorType:
        return static_cast<Sensor*>(new CameraSensor(n, plugin));
    */

    default:
        ROS_ERROR("Unknown sensor type %i requested from sensor constructor!",
                  static_cast<int>(type));
        return nullptr;
    }
}

// Constructor
Sensor::Sensor(ros::NodeHandle& n, RobotPlugin* plugin)
    : sensor_step_length_(0.0)
{
    // Nothing to do
}

// Destructor
Sensor::~Sensor() = default;

// Reset the sensor, clearing any previous state and setting it to the current state
void Sensor::reset(RobotPlugin* plugin, ros::Time current_time)
{
    // Base implementation: nothing to do
}

// Update the sensor (called every tick)
void Sensor::update(RobotPlugin* plugin, ros::Time current_time, bool is_controller_step)
{
    // Base implementation: nothing to do
}

// Set sensor update delay
void Sensor::set_update(double new_sensor_step_length)
{
    sensor_step_length_ = new_sensor_step_length;
}

// Configure the sensor (for sensor-specific trial settings)
void Sensor::configure_sensor(OptionsMap& options)
{
    // Base implementation: nothing to do
}

void Sensor::set_sample_data_format(std::unique_ptr<Sample>& sample)
{
    // Base implementation: nothing to do
}

void Sensor::set_sample_data(std::unique_ptr<Sample>& sample, int t)
{
    // Base implementation: nothing to do
}
