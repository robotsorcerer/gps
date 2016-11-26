/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Olalekan Ogunmolu.
 *  All rights reserved.
 *
 *  Nov 24, 2016
 *********************************************************************/

#ifndef MECHANISM_MODEL_CHAIN_H
#define MECHANISM_MODEL_CHAIN_H

#include "superchick_mechanism_model/robot.h"
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jntarrayvel.hpp>
#include <kdl/jntarrayacc.hpp>

namespace superchick_mechanism_model {

class Chain
{
public:
  Chain() {}
  ~Chain() {}

  /** \brief initialize the chain object
   *
   * \param robot_state the robot state object containing the robot model and the state of each joint in the robot
   * \param root the name of the root link of the chain
   * \param tip the name of the tip link of the chain
   *
   */
  bool init(RobotState *robot_state, const std::string &root, const std::string &tip);

  /// get the joint positions of the chain as a std vector
  void getPositions(std::vector<double>&);
  /// get the joint positions of the chain as a kdl jnt array
  void getPositions(KDL::JntArray&);
  /// gets the joint positions of the chain as any type with size() and []
  template <class Vec>
  void getPositions(Vec &v)
  {
    assert((int)v.size() == (int)joints_.size());
    for (size_t i = 0; i < joints_.size(); ++i)
      v[i] = joints_[i]->position_;
  }

  /// get the joint velocities of the chain as a std vector
  void getVelocities(std::vector<double>&);
  /// get the joint velocities and positoin of the chain as a kdl jnt array vel.  Fills in the positions too.
  void getVelocities(KDL::JntArrayVel&);
  /// gets the joint velocities of the chain as any type with size() and []
  template <class Vec>
  void getVelocities(Vec &v)
  {
    assert((int)v.size() == (int)joints_.size());
    for (size_t i = 0; i < joints_.size(); ++i)
      v[i] = joints_[i]->velocity_;
  }


  /// get the measured joint efforts of the chain as a std vector
  void getEfforts(std::vector<double>&);
  /// get the measured joint efforts of the chain as a kdl jnt array
  void getEfforts(KDL::JntArray&);

  /// set the commanded joint efforts of the chain as a std vector
  void setEfforts(KDL::JntArray&);
  /// set the commanded joint efforts of the chain as a kdl jnt array
  void addEfforts(KDL::JntArray&);

  /*!
   * \brief Adds efforts from any type that implements size() and [] lookup.
   */
  template <class Vec>
  void addEfforts(const Vec& v)
  {
    assert((int)v.size() == (int)joints_.size());
    for (size_t i = 0; i < joints_.size(); ++i)
      joints_[i]->commanded_effort_ += v[i];
  }

  /// returns true if all the joints in the chain are calibrated
  bool allCalibrated();

  /// get a kdl chain object that respresents the chain from root to tip
  void toKDL(KDL::Chain &chain);

  /** \brief get a joint state of an actuated joint of the chain.
   *
   * the actuated_joint_i index starts at zero
   * fixed joints are ignored in the list of actuated joints
   */
  JointState* getJoint(unsigned int actuated_joint_i);

  /// Returns the number of actuated joints in the chain
  int size() const { return joints_.size(); }

private:
  superchick_mechanism_model::RobotState *robot_state_;
  KDL::Chain kdl_chain_;

  std::vector< JointState* > joints_;  // ONLY joints that can be actuated (not fixed joints)
};

}

#endif
