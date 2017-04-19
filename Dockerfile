FROM ros:indigo-ros-core
LABEL maintainer "patlekano@gmail.com"

# install ros packages
RUN apt-get update && apt-get install -y \
    ros-indigo-ros-base=1.1.4-0* \
    && rm -rf /var/lib/apt/lists/*

# Install gazebo
FROM gazebo:gzserver7
# install gazebo packages
RUN apt-get update && apt-get install -q -y \
    libgazebo7-dev=7.5.0* \
    && rm -rf /var/lib/apt/lists/*

FROM ubuntu:trusty

# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys D2486D2DD83DB69272AFE98867170598AF249743

# setup sources.list
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-latest.list

# install gazebo packages
RUN apt-get update && apt-get install -q -y \
    gazebo7=7.5.0* \
    && rm -rf /var/lib/apt/lists/*

# setup environment
EXPOSE 11345

# setup entrypoint
COPY ./gzserver_entrypoint.sh /

ENTRYPOINT ["/gzserver_entrypoint.sh"]
CMD ["gzserver"]
RUN apt-get update && apt-get install -y build-essential \
    libgl1-mesa-dev-lts-utopic \
    autoconf \
    gcc \
    g++ \
    make \
		python-catkin-tools \
    python-pip  \
		python-dev \
		python3-pip \
    protobuf-compiler \
    libhdf5-dev \
		libprotobuf-dev \
		protobuf-compiler \
		libboost-all-dev \
		swig \
		python-pygame \
		git

RUN pip install protobuf && \
 		rosdep init && \
		rosdep update && \
		echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc && source ~/.bashrc

RUN pip install -U catkin_tools && \
		mkdir -p catkin_ws/src && \
		cd catkin_ws/src && \
		rosinstall_generator --deps ros_tutorials > .rosinstall  # Get list of pakcages

RUN wstool update   && \             # Checkout all packages
 		cd ~/catkin_ws && \
		catkin_init && \
		mkdir -p src/gps

ADD . src/gps

RUN cd src/gps && \
		chmod +x compile_proto.sh && \
		./compile_proto.sh && \

# Box setup
RUN cd ~ && \
		git clone https://github.com/pybox2d/pybox2d && \
		cd pybox2d && \
		python setup.py build && \
		python setup.py install

ENV PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/home/$USER/catkin_ws/src/gps

RUN source ~/.bashrc && \
	  cd ~/catkin_ws && \
		cb

# Run Pybox example
RUN cd ~/catkin_ws/src/gps && \
		python python/gps/gps_main.py box2d_pointmass_example 

# # Run Caffe dependencies
# FROM kaixhin/caffe-deps
#
# # Move into Caffe repo
# RUN cd /root/caffe && \
# # Make and move into build directory
#   mkdir build && cd build && \
# # CMake
#   cmake .. && \
# # Make
#   make -j"$(nproc)" all && \
#   make install
#
# # Add to Python path
# ENV PYTHONPATH=/root/caffe/python:$PYTHONPATH
#
# # Set ~/caffe as working directory
# WORKDIR /root/caffe
