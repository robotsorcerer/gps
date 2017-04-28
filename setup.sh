#! /bin/bash

# Olalekan Ogunmolux
# April 26, 17
#export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python

#set up uid/gid
export uid=1000 gid=1000

# mkdir -p /home/developer

echo "/root:x:${uid}:${gid}:Developer,,,:/root:/bin/bash" >> /etc/passwd && \
echo "/root:x:${uid}:" >> /etc/group && \
echo "/root ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
chmod 0440 /etc/sudoers.d/developer && \
chown ${uid}:${gid} -R /root

export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/root/catkin_ws/src/gps/gps_agent_pkg$

GPS_PATH=/root/catkin_ws/src/gps/python/gps/gps_main.py
# cd $GPS_PATH;

source_indig=$(source /opt/ros/indigo/setup.bash)
$source_indig