cmake_minimum_required(VERSION 3.14)
project(gps_agent_pkg)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CaffeParams.msg"
  "msg/ControllerParams.msg"
  "msg/DataRequest.msg"
  "msg/DataType.msg"
  "msg/LinGaussParams.msg"
  "msg/PositionCommand.msg"
  "msg/RelaxCommand.msg"
  "msg/SampleResult.msg"
  "msg/TfActionCommand.msg"
  "msg/TfObsData.msg"
  "msg/TfParams.msg"
  "msg/TorchParams.msg"
  "msg/TrialCommand.msg"
  "srv/UpdatePolicy.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
