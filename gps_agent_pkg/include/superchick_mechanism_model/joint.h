/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Olalekan Ogunmolu.
 *  All rights reserved.
 *
 *  Olalekan Ogunmolu. Nov 24, 2016
 *********************************************************************/
/*
 *
 */
#ifndef JOINT_H
#define JOINT_H

#include <tinyxml.h>
#include <urdf_model/joint.h>


namespace superchick_mechanism_model {

class JointState;

class JointStatistics
{
 public:
  JointStatistics():odometer_(0.0), min_position_(0), max_position_(0),
                    max_abs_velocity_(0.0), max_abs_effort_(0.0),
                    violated_limits_(false), initialized_(false){}

  void update(JointState* s);
  void reset();

  double odometer_;
  double min_position_, max_position_;
  double max_abs_velocity_;
  double max_abs_effort_;
  bool violated_limits_;

 private:
  bool initialized_;
  double old_position_;
};



class JointState
{
public:
  /// Modify the commanded_effort_ of the joint state so that the joint limits are satisfied
  void enforceLimits();

  /// Returns the safety effort limits given the current position and velocity.
  void getLimits(double &effort_low, double &effort_high);

  /// A pointer to the corresponding urdf::Joint from the urdf::Model
  boost::shared_ptr<const urdf::Joint> joint_;

  /// The joint position in radians or meters (read-only variable)
  double position_;

  /// The joint velocity in randians/sec or meters/sec (read-only variable)
  double velocity_;

  /// The measured joint effort in Nm or N (read-only variable)
  double measured_effort_;

  // joint statistics
  JointStatistics joint_statistics_;

  /// The effort the joint should apply in Nm or N (write-to variable)
  double commanded_effort_;

  /// Bool to indicate if the joint has been calibrated or not
  bool calibrated_;

  /// The position of the optical flag that was used to calibrate this joint
  double reference_position_;

  /// Constructor
  JointState() : position_(0.0), velocity_(0.0), measured_effort_(0.0),
    commanded_effort_(0.0), calibrated_(false), reference_position_(0.0){}
};

enum
{
  JOINT_NONE,
  JOINT_ROTARY,
  JOINT_CONTINUOUS,
  JOINT_PRISMATIC,
  JOINT_FIXED,
  JOINT_PLANAR,
  JOINT_TYPES_MAX
};


};

#endif /* JOINT_H */
