#! /bin/bash

# Olalekan Ogunmolux
# April 26, 17

export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:root/catkin_ws/src/gps_agent_pkg:/root/catkin_ws/src/gps_agent_pkg/gps

export CAFFE_ROOT=/root/caffe/build
source /opt/ros/indigo/setup.bash

BUILD=$(catkin build)
CATKIN__SRC_PATH=/root/catkin_ws/src
CATKIN_PATH=/root/catkin_ws

#cd to catkin_ws and then build gps_agent_pkg
cd $CATKIN__SRC_PATH
$(catkin init)

cd $CATKIN_PATH;
catkin build

source /root/catkin_ws/devel/setup.bash

GPS_PATH=~/catkin_ws/src/gps
PIP_INSTALL=$( install -r requirements.txt)
cd $GPS_PATH;
pip $PIP_INSTALL

GPS_MAIN=python/gps/gps_main.py
MAINEXEC=$(chmod +x $GPS_MAIN)

PYEXEC=$(python python/gps/gps_main.py pr2_example)

#Run PR2 Example
$PR2EXEC
