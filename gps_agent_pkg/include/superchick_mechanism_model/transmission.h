/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Olalekan Ogunmolu.
 *  All rights reserved.
 *
 *  Nov 24, 2016
 *********************************************************************/

#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#include <tinyxml.h>
#include "superchick_mechanism_model/joint.h"
#include "superchick_hardware_interface/hardware_interface.h"

namespace superchick_mechanism_model {

class Robot;

class Transmission
{
public:
  /// Constructor
  Transmission() {}

  /// Destructor
  virtual ~Transmission() {}

  /// Initializes the transmission from XML data
  virtual bool initXml(TiXmlElement *config, Robot *robot) = 0;

  /// Uses encoder data to fill out joint position and velocities
  virtual void propagatePosition(std::vector<superchick_hardware_interface::Actuator*>&,
                                 std::vector<superchick_mechanism_model::JointState*>&) = 0;

  /// Uses the joint position to fill out the actuator's encoder.
  virtual void propagatePositionBackwards(std::vector<superchick_mechanism_model::JointState*>&,
                                          std::vector<superchick_hardware_interface::Actuator*>&) = 0;

  /// Uses commanded joint efforts to fill out commanded motor currents
  virtual void propagateEffort(std::vector<superchick_mechanism_model::JointState*>&,
                               std::vector<superchick_hardware_interface::Actuator*>&) = 0;

  /// Uses the actuator's commanded effort to fill out the torque on the joint.
  virtual void propagateEffortBackwards(std::vector<superchick_hardware_interface::Actuator*>&,
                                        std::vector<superchick_mechanism_model::JointState*>&) = 0;

  /// the name of the transmission
  std::string name_;

  /**
   * Specifies the names of the actuators that this transmission uses.
   * In the propagate* methods, the order of actuators and joints in
   * the parameters matches the order in actuator_names_ and in
   * joint_names_.
   */
  std::vector<std::string> actuator_names_;

  /**
   * Specifies the names of the joints that this transmission uses.
   * In the propagate* methods, the order of actuators and joints in
   * the parameters matches the order in actuator_names_ and in
   * joint_names_.
   */
  std::vector<std::string> joint_names_;

  /// Initializes the transmission from XML data
  virtual bool initXml(TiXmlElement *config) { abort(); }  // In future versions, this method is mandatory in subclasses
};

} // namespace superchick_mechanism_model

#endif
