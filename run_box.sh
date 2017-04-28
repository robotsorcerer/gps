#! /bin/bash

# Olalekan Ogunmolux
# April 26, 17export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python

GPS_PATH=/root/catkin_ws/src/gps/python/gps/gps_main.py
# cd $GPS_PATH;

source_indig=$(source /opt/ros/indigo/setup.bash)
$source_indig

#Run Point Mass Example
point_mass=box2d_pointmass_example

pyt=$(python $GPS_PATH ${point_mass})
$pyt

