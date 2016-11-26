/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Olalekan Ogunmolu.
 *  All rights reserved.
 *
 *  Nov 24, 2016
 *********************************************************************/
#pragma once

/***************************************************/
/*! \namespace controller
 \brief The controller namespace

 \class controller::ControllerProvider
 \brief An interface to get access to controllers

 */
/***************************************************/

#include <ros/node_handle.h>

namespace superchick_controller_interface
{

class Controller;

class ControllerProvider
{
public:
  ControllerProvider(){}
  virtual ~ControllerProvider(){}

  template<class ControllerType> bool getControllerByName(const std::string& name, ControllerType*& c)
  {
    // get controller
    superchick_controller_interface::Controller* controller = getControllerByName(name);
    if (controller == NULL) return false;

    // cast controller to ControllerType
    ControllerType* controller_type = dynamic_cast< ControllerType* >(controller);
    if (controller_type == NULL)  return false;

    // copy result
    c = controller_type;
    return true;
  };

private:
  virtual superchick_controller_interface::Controller* getControllerByName(const std::string& name) = 0;

  ControllerProvider(const ControllerProvider &c);
  ControllerProvider& operator =(const ControllerProvider &c);

};
}
