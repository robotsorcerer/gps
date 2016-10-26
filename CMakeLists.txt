cmake_minimum_required(VERSION 2.8.3)
project(gps_agent_pkg)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(catkin REQUIRED COMPONENTS
  control_toolbox
  geometry_msgs
  orocos_kdl
  pluginlib
  pr2_controller_interface
  pr2_controller_manager
  pr2_controllers_msgs
  pr2_mechanism_model
  realtime_tools
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  tf
)

## System dependencies are found with CMake's conventions
 find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   pr2_controllers_msgs#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gps_agent_pkg
#  CATKIN_DEPENDS control_toolbox geometry_msgs orocos_kdl pluginlib pr2_controller_interface pr2_controller_manager pr2_controllers_msgs pr2_mechanism_model realtime_tools roscpp roslib rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
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

target_link_libraries(gps_agent_lib thread 
#smart_ptr
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(gps_agent_pkg ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(gps_agent_pkg_node src/gps_agent_pkg_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(gps_agent_pkg_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(gps_agent_pkg_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS gps_agent_pkg gps_agent_pkg_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gps_agent_pkg.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
