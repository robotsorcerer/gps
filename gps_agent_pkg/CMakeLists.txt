cmake_minimum_required(VERSION 3.14)
project(gps_agent_pkg)

# ---------------------------------------------------------------------------
# C++17 — required for std::variant, std::optional, structured bindings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ROS_DEFAULT_DIR /opt/ros/$ENV{ROS_DISTRO}/share/)

set(orocos_kdl_DIR                ${ROS_DEFAULT_DIR}/orocos_kdl)
set(pluginlib_DIR                 ${ROS_DEFAULT_DIR}/pluginlib/cmake)
set(pr2_controller_interface_DIR  ${ROS_DEFAULT_DIR}/pr2_controller_interface/cmake)
set(pr2_controller_manager_DIR    ${ROS_DEFAULT_DIR}/pr2_controller_manager/cmake/)
set(pr2_controllers_msgs_DIR      ${ROS_DEFAULT_DIR}/pr2_controllers_msgs/cmake)
set(pr2_mechanism_model_DIR       ${ROS_DEFAULT_DIR}/pr2_mechanism_model/cmake)

# ---------------------------------------------------------------------------
# LibTorch — locate via TORCH_ROOT env var or CMake find_package
# ---------------------------------------------------------------------------
if(DEFINED ENV{TORCH_ROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_ROOT}")
endif()
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

find_package(catkin REQUIRED COMPONENTS
  control_toolbox
  geometry_msgs
  orocos_kdl
  pluginlib
  pr2_controller_interface
  pr2_controller_manager
  pr2_controllers_msgs
  pr2_mechanism_model
  genmsg
  message_generation
  realtime_tools
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  tf
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build/gps)

## System dependencies are found with CMake's conventions
 find_package(Boost REQUIRED COMPONENTS system)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   CaffeParams.msg
   ControllerParams.msg
   DataRequest.msg
   DataType.msg
   LinGaussParams.msg
   PositionCommand.msg
   RelaxCommand.msg
   SampleResult.msg
   TfActionCommand.msg
   TfObsData.msg
   TfParams.msg
   TorchParams.msg
   TrialCommand.msg
)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   geometry_msgs   pr2_controllers_msgs   sensor_msgs   std_msgs
 )

## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gps_agent_pkg
#  CATKIN_DEPENDS message_runtime control_toolbox geometry_msgs #orocos_kdl pluginlib pr2_controller_interface pr2_controller_manager pr2_controllers_msgs message_runtime pr2_mechanism_model realtime_tools roscpp roslib rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(include)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  $ENV{GPS_ROOT_DIR}/build/gps
)

set(DDP_FILES src/robotplugin.cpp
              src/pr2plugin.cpp
              src/sample.cpp
              src/sensor.cpp
              src/neuralnetwork.cpp
              src/tfcontroller.cpp
              src/controller.cpp
              src/lingausscontroller.cpp
              src/camerasensor.cpp
              src/positioncontroller.cpp
              src/trialcontroller.cpp
              src/encodersensor.cpp
              src/encoderfilter.cpp
              src/rostopicsensor.cpp
              src/pytorchcontroller.cpp
              src/util.cpp)

add_library(gps_agent_lib
            ${DDP_FILES}
)

target_link_libraries(gps_agent_lib pthread ${TORCH_LIBRARIES})

# LibTorch requires position-independent code when linking into a shared lib.
set_property(TARGET gps_agent_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

add_dependencies(gps_agent_lib ${PROJECT_NAME}_gencpp)
