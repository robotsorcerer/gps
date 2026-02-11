cmake_minimum_required(VERSION 3.14)
project(gps_agent_pkg)

# ---------------------------------------------------------------------------
# C++17 — required for std::variant, std::optional, structured bindings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Treat warnings as errors; suppress two categories that are endemic to
# legacy ROS code (signed/unsigned loop indices, unused callback params).
# ---------------------------------------------------------------------------
add_compile_options(
    -Wall
    -Wextra
    -Werror
    -Wno-sign-compare       # int i < container.size() patterns throughout
    -Wno-unused-parameter   # ROS callback signatures carry unused args
)

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

# ---------------------------------------------------------------------------
# ROS version detection — ROS_VERSION env var set by ROS 1 (=1) and ROS 2 (=2)
# Default to ROS 1 when unset so existing CI is unaffected.
# ---------------------------------------------------------------------------
if("$ENV{ROS_VERSION}" STREQUAL "2")
    set(GPS_ROS2_BUILD ON)
    message(STATUS "gps_agent_pkg: building for ROS 2 (ament_cmake / colcon)")
else()
    set(GPS_ROS2_BUILD OFF)
    message(STATUS "gps_agent_pkg: building for ROS 1 (catkin)")
endif()

# ===========================================================================
# ROS 2 / ament_cmake build path
# ===========================================================================
if(GPS_ROS2_BUILD)

    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(rclpy REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(rosidl_default_generators REQUIRED)

    # ── Generate ROS 2 interfaces (all messages + UpdatePolicy service) ──
    # All GPS message types use only built-in field types or cross-references
    # within this package — no PR2-specific message dependencies.
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
        DEPENDENCIES geometry_msgs sensor_msgs std_msgs
    )

    # ── Portable C++ sources (no PR2 / pluginlib dependencies) ──────────
    # PR2-specific files (pr2plugin.cpp, camerasensor.cpp, encodersensor.cpp,
    # encoderfilter.cpp, rostopicsensor.cpp) depend on PR2 controller headers
    # that don't exist in ROS 2.  The core PyTorch controller and algorithm
    # logic are fully portable.
    set(GPS_ROS2_SOURCES
        src/robotplugin.cpp
        src/sample.cpp
        src/sensor.cpp
        src/neuralnetwork.cpp
        src/tfcontroller.cpp
        src/controller.cpp
        src/lingausscontroller.cpp
        src/positioncontroller.cpp
        src/trialcontroller.cpp
        src/pytorchcontroller.cpp
        src/util.cpp
    )

    add_library(gps_agent_lib SHARED ${GPS_ROS2_SOURCES})

    ament_target_dependencies(gps_agent_lib
        rclcpp sensor_msgs geometry_msgs std_msgs tf2_ros
    )
    target_include_directories(gps_agent_lib PUBLIC
        include
        ${TORCH_INCLUDE_DIRS}
        $ENV{GPS_ROOT_DIR}/build/gps
    )
    target_link_libraries(gps_agent_lib pthread ${TORCH_LIBRARIES})
    set_property(TARGET gps_agent_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

    rosidl_target_interfaces(gps_agent_lib ${PROJECT_NAME} "rosidl_typesupport_cpp")

    install(TARGETS gps_agent_lib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY include/ DESTINATION include)

    ament_package()

# ===========================================================================
# ROS 1 / catkin build path  (unchanged from original)
# ===========================================================================
else()

    set(ROS_DEFAULT_DIR /opt/ros/$ENV{ROS_DISTRO}/share/)

    set(orocos_kdl_DIR                ${ROS_DEFAULT_DIR}/orocos_kdl)
    set(pluginlib_DIR                 ${ROS_DEFAULT_DIR}/pluginlib/cmake)
    set(pr2_controller_interface_DIR  ${ROS_DEFAULT_DIR}/pr2_controller_interface/cmake)
    set(pr2_controller_manager_DIR    ${ROS_DEFAULT_DIR}/pr2_controller_manager/cmake/)
    set(pr2_controllers_msgs_DIR      ${ROS_DEFAULT_DIR}/pr2_controllers_msgs/cmake)
    set(pr2_mechanism_model_DIR       ${ROS_DEFAULT_DIR}/pr2_mechanism_model/cmake)

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

    ## Generate services in the 'srv' folder
    add_service_files(
        FILES
        UpdatePolicy.srv
    )

    ## Generate messages and services
    generate_messages(
        DEPENDENCIES
        geometry_msgs   pr2_controllers_msgs   sensor_msgs   std_msgs
    )

    catkin_package(
        #  CATKIN_DEPENDS message_runtime control_toolbox geometry_msgs ...
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

    set(DDP_FILES
        src/robotplugin.cpp
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
        src/util.cpp
    )

    add_library(gps_agent_lib ${DDP_FILES})

    target_link_libraries(gps_agent_lib pthread ${TORCH_LIBRARIES})

    # LibTorch requires position-independent code when linking into a shared lib.
    set_property(TARGET gps_agent_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

    add_dependencies(gps_agent_lib ${PROJECT_NAME}_gencpp)

endif() # GPS_ROS2_BUILD
