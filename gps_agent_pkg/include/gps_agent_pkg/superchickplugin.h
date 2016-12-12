/*
This is the PR2-specific version of the robot plugin.
*/
#pragma once

// Headers.
#include <Eigen/Dense>
// To register the plugins
#include <pluginlib/class_list_macros.h>

// Superclass.
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps/proto/gps.pb.h"

// MoveIt!

#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

namespace gps_control
{

class GPSSuperchickPlugin: public RobotPlugin
{
private:
/*   
    robot_model_loader::RobotModelLoader robot_model_loader;
    // ("robot_description");
    robot_model::RobotModelPtr RobotModel;
    const robot_state::JointModelGroup* base_model_group;
    robot_state::RobotStatePtr RobotState(new robot_state::RobotState(RobotModel));
    const robot_state::JointModelGroup* base_model_group;
    const std::vector<std::string> *joint_names;
    std::vector<double> joint_values;
    */

    //move_it robot model loader
    robot_model_loader::RobotModelLoader *robot_model_loader;
    //state of the robot
    robot_state::RobotStatePtr RobotState;
    //Robot Model
    robot_model::RobotModelPtr RobotModel;
    //joint names
    const std::vector<std::string> joint_names;
    // We can retreive the current set of joint values stored in the state for the base bladder.
    std::vector<double> base_joint_values;
    //base_joint group
    const robot_state::JointModelGroup* base_joint_group;

    // Time of last state update.
    ros::Time last_update_time_;
    // Counter for keeping track of controller steps.
    int controller_counter_;
    // Length of controller steps in ms.
    int controller_step_length_;
public:
    // Constructor (this should do nothing).
    GPSSuperchickPlugin();
    // Destructor.
    virtual ~GPSSuperchickPlugin();
    // Functions inherited from superclass.
    // This called by the superclass to allow us to initialize all the superchick-specific stuff.
    /* IMPORTANT: note that some sensors require a KDL chain to do FK, which we need the RobotState to get... */
    virtual bool init(ros::NodeHandle& n);
    // This is called by the controller manager before starting the controller.
    virtual void starting();
    // This is called by the controller manager before stopping the controller.
    virtual void stopping();
    // This is the main update function called by the realtime thread when the controller is running.
    virtual void update();
    /* the pr2-specific update function should do the following:
       - perform whatever housekeeping is needed to note the current time.
       - update all sensors (this might be a no-op for vision, but for
         joint angle "sensors," they need to know the current robot state).
       - update the appropriate controller (position or trial) depending on
         what we're currently doing
       - if the controller wants to send something via a publisher, publish
         that at the end -- it will typically be a completion message that
         includes the recorded robot state for the controller's execution.
     */
    // Accessors.
    // Get current time.
    virtual ros::Time get_current_time() const;
    // Get current encoder readings (robot-dependent).
    // virtual void get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const;
};

}
