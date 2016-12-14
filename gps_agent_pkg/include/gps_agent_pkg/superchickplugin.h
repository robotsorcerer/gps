/*
This is the PR2-specific version of the robot plugin.
*/
#pragma once

// Headers.
// Headers.
// #include <pr2_controller_interface/controller.h>
// #include <pr2_mechanism_model/joint.h>
// #include <pr2_mechanism_model/chain.h>
// #include <pr2_mechanism_model/robot.h>
#include <Eigen/Dense>
#include <pluginlib/class_list_macros.h>

// Superclass.
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps/proto/gps.pb.h"

// MoveIt!
//moveit controller manager
#include <ros/ros.h>
#include <moveit/controller_manager/controller_manager.h>
#include <sensor_msgs/JointState.h>
#include <pluginlib/class_list_macros.h>
#include <map>
//move it trajectory and robot loader
#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

namespace gps_control
{

class GPSSuperchickPlugin: public moveit_controller_manager::MoveItControllerHandle 
{
private:
    // Time of last state update.
    ros::Time last_update_time_;
    // Counter for keeping track of controller steps.
    int controller_counter_;
    // Length of controller steps in ms.
    int controller_step_length_;
public:
    // Default Constructor (this should do nothing).
    GPSSuperchickPlugin(const std::string &name);
    //Copy Constructor
    GPSSuperchickPlugin();
    // Destructor.
    virtual ~GPSSuperchickPlugin();
    virtual bool sendTrajectory(const moveit_msgs::RobotTrajectory &t);
    virtual bool cancelExecution();
    virtual bool waitForExecution(const ros::Duration &);
    virtual moveit_controller_manager::ExecutionStatus getLastExecutionStatus();
};

class GPSSuperchickPluginManager : public RobotPlugin, public moveit_controller_manager::MoveItControllerManager
{
private:
    // ("robot_description");
    //move_it robot model loader
    robot_model_loader::RobotModelLoader *robot_model_loader;

    //Robot Model
    robot_model::RobotModelPtr RobotModel;
    //move group interface
    moveit::planning_interface::MoveGroup *group;
    //joint names
    const std::vector<std::string> joint_names;
    // We can retreive the current set of joint values stored in the state for the base bladder.
    std::vector<double> joint_values;
    const robot_state::JointModelGroup* base_model_group;
    
    // Time of last state update.
    ros::Time last_update_time_;
    // Counter for keeping track of controller steps.
    int controller_counter_;
    // Length of controller steps in ms.
    int controller_step_length_;
    //names of the three bladders
    std::string BASE_BLADDER, RIGHT_BLADDER, LEFT_BLADDER;
    // Variables.
    std::string base_group, head_name, right_name;
    //pose vector published by gps
    geometry_msgs::Pose target_pose_;

protected:
  ros::NodeHandle n_;
  std::map<std::string, moveit_controller_manager::MoveItControllerHandlePtr> controllers_;
public:
    //constructor
   GPSSuperchickPluginManager();
   //Destructor 
   virtual ~GPSSuperchickPluginManager();
   virtual moveit_controller_manager::MoveItControllerHandlePtr getControllerHandle(const std::string &name);
   virtual void getControllersList(std::vector<std::string> &names);
   virtual void getActiveControllers(std::vector<std::string> &names);
   virtual void getLoadedControllers(std::vector<std::string> &names);
   virtual void getControllerJoints(const std::string &name, std::vector<std::string> &joints);
   virtual moveit_controller_manager::MoveItControllerManager::ControllerState
   getControllerState(const std::string &name);
   virtual bool switchControllers(const std::vector<std::string> &activate, const std::vector<std::string> &deactivate);
   virtual ros::Time get_current_time() const;
   //from PR2 Superclass
   virtual bool init();
};

}
    // PR2-specific chain object necessary to construct the KDL chain.
/*     pr2_mechanism_model::Chain passive_arm_chain_, active_arm_chain_ , \
                                right_bladder_chain_, base_bladder_chain;
    // This is a pointer to the robot state, which we get when initialized and have to keep after that.
    pr2_mechanism_model::RobotState* robot_;
    // Passive arm joint states.
    std::vector<pr2_mechanism_model::JointState*> passive_arm_joint_state_;
    // Active arm joint states.
    std::vector<pr2_mechanism_model::JointState*> active_arm_joint_state_;
    // Base Bladder joint states.
    std::vector<pr2_mechanism_model::JointState*> base_bladder_joint_state_;
    // Passive arm joint names.
    std::vector<std::string> passive_arm_joint_names_;
    // Active arm joint names.
    std::vector<std::string> active_arm_joint_names_;

  
    robot_model_loader::RobotModelLoader robot_model_loader;
    // ("robot_description");
    robot_model::RobotModelPtr RobotModel;
    const robot_state::JointModelGroup* base_model_group;
    robot_state::RobotStatePtr RobotState(new robot_state::RobotState(RobotModel));
    const robot_state::JointModelGroup* base_model_group;
    const std::vector<std::string> *joint_names;
    std::vector<double> joint_values;
    

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
    // IMPORTANT: note that some sensors require a KDL chain to do FK, which we need the RobotState to get... 
    virtual bool init(ros::NodeHandle& n);
    // This is called by the controller manager before starting the controller.
    virtual void starting();
    // This is called by the controller manager before stopping the controller.
    virtual void stopping();
    // This is the main update function called by the realtime thread when the controller is running.
    virtual void update();
     // the pr2-specific update function should do the following:
     //   - perform whatever housekeeping is needed to note the current time.
     //   - update all sensors (this might be a no-op for vision, but for
     //     joint angle "sensors," they need to know the current robot state).
     //   - update the appropriate controller (position or trial) depending on
     //     what we're currently doing
     //   - if the controller wants to send something via a publisher, publish
     //     that at the end -- it will typically be a completion message that
     //     includes the recorded robot state for the controller's execution.
     
    // Accessors.
    // Get current time.
    virtual ros::Time get_current_time() const;
    // Get current encoder readings (robot-dependent).
    // virtual void get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const;
    */
// };

// }
