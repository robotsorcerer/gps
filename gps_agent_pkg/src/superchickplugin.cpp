
#include "gps_agent_pkg/superchickplugin.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/trialcontroller.h"
#include "gps_agent_pkg/util.h"

namespace gps_control
{
    //initialize constructor
  GPSSuperchickPlugin::GPSSuperchickPlugin(const std::string &name) 
  : moveit_controller_manager::MoveItControllerHandle(name)
  {
    // Some basic variable initialization.
    controller_counter_ = 0;
    controller_step_length_ = 50;
  }

  bool GPSSuperchickPlugin::sendTrajectory(const moveit_msgs::RobotTrajectory &t)
  {
    // do whatever is needed to actually execute this trajectory
    return true;
  }

  bool GPSSuperchickPlugin::cancelExecution()
  {
    // do whatever is needed to cancel execution
    return true;
  }

  bool GPSSuperchickPlugin::waitForExecution(const ros::Duration &)
  {
    // wait for the current execution to finish
    return true;
  }

  moveit_controller_manager::ExecutionStatus GPSSuperchickPlugin::getLastExecutionStatus()
  {
    return moveit_controller_manager::ExecutionStatus(moveit_controller_manager::ExecutionStatus::SUCCEEDED);
  }

  /*initialize gps plugin manager constructor*/
  GPSSuperchickPluginManager::GPSSuperchickPluginManager()
  {
    init();
  }
  //manager destructor
  GPSSuperchickPluginManager::~GPSSuperchickPluginManager(){};

  //initialize the robot
  bool GPSSuperchickPluginManager::init()
  {
      ros::AsyncSpinner spinner(1);
      spinner.start();

      // Store the robot state.
          //state of the robot
        robot_state::RobotStatePtr RobotState(new robot_state::RobotState(RobotModel));
      RobotState->setToDefaultValues();

      // Create FK solvers.
      // Get the name of the root.
      if(!n_.getParam("/GPSSuperchickPlugin/base_group", base_group)) {
          ROS_ERROR("Property base_group not found in namespace: '%s'", n_.getNamespace().c_str());
          return false;
      }

      // Get active and passive arm end-effector names.
      if(!n_.getParam("/GPSSuperchickPlugin/head_name", head_name)) {
          ROS_ERROR("Property head_name not found in namespace: '%s'", n_.getNamespace().c_str());
          return false;
      }
      if(!n_.getParam("/GPSSuperchickPlugin/right_name", right_name)) {
          ROS_ERROR("Property right_name not found in namespace: '%s'", n_.getNamespace().c_str());
          return false;
      }

      BASE_BLADDER = "base_bladder";
      RIGHT_BLADDER = "right_bladder";
      LEFT_BLADDER = "left_bladder";

      controllers_["base_bladder"] = getControllerHandle(BASE_BLADDER);
      controllers_["right_bladder"] = getControllerHandle(RIGHT_BLADDER);
      controllers_["left_bladder"] = getControllerHandle(LEFT_BLADDER);

      // Pull out joint states.      
      group = new moveit::planning_interface::MoveGroup("base_bladder");
      group->setEndEffectorLink("headnball_link");
      // We will use the :planning_scene_interface:`PlanningSceneInterface`
      // class to deal directly with the world.
      moveit::planning_interface::PlanningSceneInterface planning_scene_interface;  
      // (Optional) Create a publisher for visualizing plans in Rviz.
      ros::Publisher display_publisher = n_.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
      moveit_msgs::DisplayTrajectory display_trajectory;
      // We can print the name of the reference frame for this robot.
      ROS_INFO("Reference frame: %s", group->getPlanningFrame().c_str());

      //Im assuming trial controller topic sets target pose which is a geometry_msgs::Pose 
      group->setPoseTarget(target_pose_);

      //tell the robot that we are all set
      initialize(n_);

      return true;
  }

  moveit_controller_manager::MoveItControllerHandlePtr GPSSuperchickPluginManager::getControllerHandle(const std::string &name)
  {
    return moveit_controller_manager::MoveItControllerHandlePtr(new GPSSuperchickPlugin(name));
  }

  /*
   * Get the list of controller names.
   */
  void GPSSuperchickPluginManager::getControllersList(std::vector<std::string> &names)
  {
    names.resize(3);
    names[0] = "base_bladder";
    names[1] = "right_bladder";
    names[2] = "left_bladder";
  }

  /*
   * This plugin assumes that all controllers are already active -- and if they are not, well, it has no way to deal
   * with it anyways!
   */
  void GPSSuperchickPluginManager::getActiveControllers(std::vector<std::string> &names)
  {
    getControllersList(names);
  }

  /*
   * Controller must be loaded to be active, see comment above about active controllers...
   */
  void GPSSuperchickPluginManager::getLoadedControllers(std::vector<std::string> &names)
  {
    getControllersList(names);
  }

  /*
   * Get the list of joints that a controller can control.
   */
  void GPSSuperchickPluginManager::getControllerJoints(const std::string &name, std::vector<std::string> &joints)
  {
    joints.clear();
    if (name == "base_bladder")
    {
      // declare which joints this controller actuates
      joints.push_back("bigbladder_to_head");
    }
    else if(name == "right_bladder"){
        joints.push_back("right_bladder");
    }
    else if(name == "left_bladder"){
        joints.push_back("left_bladder");
    }
    else
    {
        ROS_WARN_STREAM("Bladder %s does not exist in the controller list" << name);
    }
  }

  /*
   * Controllers are all active and default.
   */
  moveit_controller_manager::MoveItControllerManager::ControllerState
  GPSSuperchickPluginManager::getControllerState(const std::string &name)
  {
    moveit_controller_manager::MoveItControllerManager::ControllerState state;
    state.active_ = true;
    state.default_ = true;
    return state;
  }

  /* Cannot switch our controllers */
  bool 
  GPSSuperchickPluginManager::switchControllers(const std::vector<std::string> &activate, const std::vector<std::string> &deactivate)
  {
    return false;
  }

  // Get current time.
  ros::Time GPSSuperchickPluginManager::get_current_time() const
  {
      return last_update_time_;
  }
};
// }  // end namespace moveit_controller_manager_example

PLUGINLIB_EXPORT_CLASS(gps_control::GPSSuperchickPluginManager,
                       moveit_controller_manager::MoveItControllerManager);