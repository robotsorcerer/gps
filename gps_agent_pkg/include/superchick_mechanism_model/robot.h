/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Olalekan Ogunmolu.
 *  All rights reserved.
 *
 *  Nov 24, 2016
 *********************************************************************/

#ifndef ROBOT_H
#define ROBOT_H

#include <vector>
#include <map>
#include <string>
#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <superchick_hardware_interface/hardware_interface.h>
#include <hardware_interface/hardware_interface.h>
#include "superchick_mechanism_model/joint.h"
#include "superchick_mechanism_model/transmission.h"

class TiXmlElement;

// Forward declared to avoid extra includes
namespace pluginlib {
template <class T> class ClassLoader;
}

namespace superchick_mechanism_model
{


/** \brief This class provides the controllers with an interface to the robot model.
 *
 * Most controllers that need the robot model should use the 'robot_model_', which is
 * a kinematic/dynamic model of the robot, represented by a KDL Tree structure.
 *
 * Some specialized controllers (such as the calibration controllers) can get access
 * to actuators, transmissions and special joint parameters.
 */

class Robot
{
public:
  /// Constructor
  Robot(superchick_hardware_interface::HardwareInterface *hw);

  /// Destructor
  ~Robot() { }

  /// Initialize the robot model form xml
  bool initXml(TiXmlElement *root);

  /// Initialize the robot model from the param server
  // KDL::Tree superchick_tree;
  // std::string superchick_description = "robot_description";
  // ros::param::get("robot_description", superchick_description);
  // if (!kdl_parser::treeFromString(superchick_description, superchick_tree))
  // {
  //    ROS_ERROR("Failed to construct kdl tree");
  //    return false;
  // }

  /// The kinematic/dynamic model of the robot
  urdf::Model robot_model_;

  // robot_model_.initParam(superchick_description);

  /// The list of transmissions
  std::vector<boost::shared_ptr<Transmission> > transmissions_;

  /// get the transmission index based on the transmission name. Returns -1 on failure
  int getTransmissionIndex(const std::string &name) const;

  /// get an actuator pointer based on the actuator name. Returns NULL on failure
  superchick_hardware_interface::Actuator* getActuator(const std::string &name) const;

  /// get a transmission pointer based on the transmission name. Returns NULL on failure
  boost::shared_ptr<superchick_mechanism_model::Transmission> getTransmission(const std::string &name) const;

  /// Get the time when the current controller cycle was started
  ros::Time getTime();

  /// a pointer to the superchick hardware interface. Only for advanced users
  superchick_hardware_interface::HardwareInterface* hw_;

private:
  boost::shared_ptr<pluginlib::ClassLoader<superchick_mechanism_model::Transmission> > transmission_loader_;
};



/** \brief This class provides the controllers with an interface to the robot state
 *
 * Most controllers that need the robot state should use the joint states, to get
 * access to the joint position/velocity/effort, and to command the effort a joint
 * should apply. Controllers can get access to the hard realtime clock through getTime()
 *
 * Some specialized controllers (such as the calibration controllers) can get access
 * to actuator states, and transmission states.
 */
class RobotState : public hardware_interface::HardwareInterface
{
public:
  /// constructor
  RobotState(Robot *model);

  /// The robot model containing the transmissions, urdf robot model, and hardware interface
  Robot *model_;

  /// The vector of joint states, in no particular order
  std::vector<JointState> joint_states_;

  /// Get a joint state by name
  JointState *getJointState(const std::string &name);

  /// Get a const joint state by name
  const JointState *getJointState(const std::string &name) const;

  /// Get the time when the current controller cycle was started
  ros::Time getTime() {return model_->getTime();};

 /**
  * Each transmission refers to the actuators and joints it connects by name.
  * Since name lookup is slow, for each transmission in the robot model we
  * cache pointers to the actuators and joints that it connects.
  **/
  std::vector<std::vector<superchick_hardware_interface::Actuator*> > transmissions_in_;

 /**
  * Each transmission refers to the actuators and joints it connects by name.
  * Since name lookup is slow, for each transmission in the robot model we
  * cache pointers to the actuators and joints that it connects.
  **/
  std::vector<std::vector<superchick_mechanism_model::JointState*> > transmissions_out_;

  /// Propagete the actuator positions, through the transmissions, to the joint positions
  void propagateActuatorPositionToJointPosition();
  /// Propagete the joint positions, through the transmissions, to the actuator positions
  void propagateJointPositionToActuatorPosition();

  /// Propagete the joint efforts, through the transmissions, to the actuator efforts
  void propagateJointEffortToActuatorEffort();
  /// Propagete the actuator efforts, through the transmissions, to the joint efforts
  void propagateActuatorEffortToJointEffort();

  /// Modify the commanded_effort_ of each joint state so that the joint limits are satisfied
  void enforceSafety();

  /// Checks if one (or more) of the motors are halted.
  bool isHalted();

  /// Set the commanded_effort_ of each joint state to zero
  void zeroCommands();


  std::map<std::string, JointState*> joint_states_map_;

};

}

#endif
