cmake_minimum_required(VERSION 2.8.12)
project(gps_agent_pkg)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(ROS_DEFAULT_DIR /opt/ros/indigo/share/)

set(orocos_kdl_DIR                ${ROS_DEFAULT_DIR}/orocos_kdl)
set(pluginlib_DIR           /      ${ROS_DEFAULT_DIR}/pluginlib/cmake)
set(pr2_controller_interface_DIR  ${ROS_DEFAULT_DIR}/pr2_controller_interface/cmake)
set(pr2_controller_manager_DIR    ${ROS_DEFAULT_DIR}/pr2_controller_manager/cmake/)
set(pr2_controllers_msgs_DIR      ${ROS_DEFAULT_DIR}/pr2_controllers_msgs/cmake)
set(pr2_mechanism_model_DIR      ${ROS_DEFAULT_DIR}/pr2_mechanism_model/cmake)
set(Caffe_DIR                     /home/lex/Documents/caffe/build_cudnn)

set(USE_CAFFE USE_CAFFE_GPU ON)

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
  Caffe
)

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
# Include Caffe
if (USE_CAFFE)
    # add definitions for the C++ code
    add_definitions(-DUSE_CAFFE)
    if (USE_CAFFE_GPU)
        add_definitions(-DCAFFE_GPU)
        find_package(CUDA 5.5 REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
    else (USE_CAFFE_GPU)
        add_definitions(-DCPU_ONLY)
    endif (USE_CAFFE_GPU)
    # add paths to CAFFE
    link_directories(${CAFFE_LIBRARY_PATH})  # Should be set to /path/to/caffe/build/lib
    include_directories(${CAFFE_INCLUDE_PATH})  # Should be set to /path/to/caffe/distribute/include
    # extra header paths
    include_directories(${CAFFE_EXTRA_PATH})
    # add neural network to DDP controller files
    set(DDP_FILES ${DDP_FILES} src/neuralnetworkcaffe.cpp src/caffenncontroller.cpp)
    # compile Caffe test
    #rosbuild_add_executable(caffe_test src/caffe_test.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_test caffe protobuf)

    # compile image processor node
    #rosbuild_add_executable(caffe_img_processor src/img_processor.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_img_processor caffe protobuf)
endif (USE_CAFFE)

include_directories(
  ${catkin_INCLUDE_DIRS}
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
              src/util.cpp)

add_library(gps_agent_lib
            ${DDP_FILES}
   #src/${PROJECT_NAME}/gps_agent_pkg.cpp
)

# Include Caffe in controller
if (USE_CAFFE)
    target_link_libraries(gps_agent_lib caffe protobuf)
endif (USE_CAFFE)

target_link_libraries(gps_agent_lib pthread )

add_dependencies(gps_agent_lib ${PROJECT_NAME}_gencpp)


