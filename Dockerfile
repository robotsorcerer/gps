FROM nvidia/cuda:8.0-devel-ubuntu14.04
LABEL maintainer "patlekano@gmail.com"

# setup environment
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends --allow-unauthenticated -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO indigo
RUN apt-get update && apt-get install --allow-unauthenticated  -y \
    ros-indigo-desktop-full=1.1.4-0* \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
# COPY ./ros_entrypoint.sh /

# ENTRYPOINT ["/ros_entrypoint.sh"]
# CMD ["bash"]

# ros-indigo-ros-base
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    ros-indigo-ros-base=1.1.4-0* \
&& rm -rf /var/lib/apt/lists/*

# moveit-indigo-ci
ENV TERM xterm

# install gazebo packages
RUN apt-get update && apt-get install -q --allow-unauthenticated -y \
		build-essential \
		gcc \
		g++ \
		wget \
		make \
		nano \
		curl \
		protobuf-compiler \
		libhdf5-dev \
		libprotobuf-dev \
		protobuf-compiler \
		libboost-all-dev \
		swig \
		python-pygame \
		python-pip \
		python-dev \
		git \
		libgflags-dev \
		libgoogle-glog-dev  \
		liblmdb-dev \
		&& rm -rf /var/lib/apt/lists/*

#one line gazebo install
RUN curl -ssL http://get.gazebosim.org | sh

# Start with Caffe dependencies

#protobuf-compiler
ENV PROTOBUF=/root/protobuf
RUN git clone https://github.com/google/protobuf.git \
		&& cd protobuf \
		&& bash autogen.sh \
		&& ./configure \
		&& make -j \
		&& make install \
		&& cd /root

#clone caffe
ENV CAFFE_ROOT=/root/caffe/
RUN git clone https://github.com/BVLC/caffe.git \
		&& cd $CAFFFE \
		&& mkdir build && cd build \
		&& cmake -DUSE_CUDNN=ON .. \
		&& make -j"$(nproc)" all \
		&& make install \
		&& make runtest \
		&& cd /root

# setup environment
EXPOSE 5000

# Setup catkin workspace
RUN /bin/bash -c echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc \
		&& /bin/bash -c echo "export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python:/root/catkin_ws/src/gps" >> ~/.bashrc \
		&& /bin/bash -c echo " source /usr/local/etc/bash_completion.d/catkin_tools-completion.bash" >> ~/.bashrc \
		&& /bin/bash -c "source /root/.bashrc"


#install catkin build
RUN pip install -U catkin_tools

ENV CATKIN_WS=/root/catkin_ws

RUN mkdir -p $CATKIN_WS/src && cd $CATKIN_WS/src
COPY . /catkin_ws/src/gps
RUN cd gps \
		&& pip2 install -r requirements.txt \
		&& ./compile_proto.sh \
		&& cd gps_agent_pkg \
		&& rosdep install --from-paths -r -y -q . \
		&& cd $CATKIN_WS/src \
		&& catkin_build

# Box setup
RUN cd /root && \
		git clone https://github.com/pybox2d/pybox2d && \
		cd pybox2d && \
		python setup.py build && \
		python setup.py install

RUN /bin/bash source /root/.bashrc
RUN python python/gps/gps_main.py box2d_pointmass_example
