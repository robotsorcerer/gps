#! /bin/bash

# Olalekan Ogunmolux
# April 26, 17export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python

GPS_PATH=/root/catkin_ws/src/gps/python/gps/gps_main.py
# cd $GPS_PATH;

#Run pr2_gazeo
rlc=$(roslaunch gps_agent_pkg pr2_gazebo.launch)

$rlc