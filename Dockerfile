FROM ros:indigo-ros-core
LABEL maintainer "patlekano@gmail.com"

# install ros packages
RUN apt-get update && apt-get install -y \
    ros-indigo-ros-base=1.1.4-0* \
    && rm -rf /var/lib/apt/lists/*

# Install gazebo
FROM gazebo:gzserver6
# install gazebo packages
RUN apt-get update && apt-get install -q -y \
    libgazebo6-dev=6.7.0* \
    && rm -rf /var/lib/apt/lists/*

FROM ubuntu:trusty
# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys D2486D2DD83DB69272AFE98867170598AF249743
# setup sources.list
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-latest.list

#add catkin_tools key
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

#RUN wget http://packages.ros.org/ros.key -O - | sudo apt-key add -

# install gazebo packages and catkin
RUN apt-get update && apt-get install -q -y \
    gazebo6=6.7.0* \
		&& wget \
    && rm -rf /var/lib/apt/lists/*
# setup environment
EXPOSE 5000

# setup entrypoint
COPY ./gzserver_entrypoint.sh /

ENTRYPOINT ["/gzserver_entrypoint.sh"]
CMD ["gzserver"]

# Run Caffe dependencies
FROM kaixhin/caffe-deps

# Move into Caffe repo
RUN cd /root/caffe && \
# Make and move into build directory
  mkdir build && cd build && \
# CMake
  cmake .. && \
# Make
  make -j"$(nproc)" all && \
  make install

# Add to Python path
ENV PYTHONPATH=/root/caffe/python:$PYTHONPATH
ENV CAFFE_ROOT=/root/caffe
# Set ~/caffe as working directory
WORKDIR /root/caffe

# RUN apt-get update && apt-get install -y \
# 		build-essential \
#     gcc \
#     g++ \
#     make \
# 		python-catkin-tools \
#     python-pip  \
# 		python-dev \
# 		python3-pip \
#     protobuf-compiler \
#     libhdf5-dev \
# 		libprotobuf-dev \
# 		protobuf-compiler \
# 		libboost-all-dev \
# 		swig \
# 		python-pygame \
# 		git \
# 		libgflags-dev \
# 		libgoogle-glog-dev  \
# 		liblmdb-dev \

RUN	/bin/bash -c echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc \
		/bin/bash -c "source ~/.bashrc"

# RUN rosdep init

# RUN	rosdep update


RUN cd ~ && mkdir -p catkin_ws/src && \
		cd catkin_ws/src

# Get list of pakcages
COPY . ~/catkin_ws/src/gps
WORKDIR ~/catkin_ws/src/gps
RUN pip2 install -r requirements.txt

RUN /bin/bash -c echo " source /usr/local/etc/bash_completion.d/catkin_tools-completion.bash" >> ~/.bashrc \
		/bin/bash -c "source ~/.bashrc"

# RUN cd ~/catkin_ws/src && \
# 		catkin_init_workspace
#
# # Checkout all packages
# RUN cd ~/catkin_ws && \
# 		catkin_make
#
# WORKDIR ~/catkin_ws/src/gps
# RUN chmod +x compile_proto.sh && \
# 		./compile_proto.sh \

# # Box setup
# RUN cd ~ && \
# 		git clone https://github.com/pybox2d/pybox2d && \
# 		cd pybox2d && \
# 		python setup.py build && \
# 		python setup.py install
#
# ENV PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:~/catkin_ws/src/gps
#
# RUN source ~/.bashrc && \
# 	  cd ~/catkin_ws && \
# 		catkin build && \
# 		cd src/gps/gps_agent_pkg && \
# 		rosdep install --from-paths -q -r -y  .
#
# # Run Pybox example
# RUN cd ~/catkin_ws/src/gps
#
# RUN python python/gps/gps_main.py box2d_pointmass_example
