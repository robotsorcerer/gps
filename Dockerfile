FROM nvidia/cuda:8.0-devel-ubuntu14.04
LABEL maintainer "patlekano@gmail.com"

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

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

# # setup entrypoint
# COPY ./ros_entrypoint.sh /

# ENTRYPOINT ["/ros_entrypoint.sh"]
# CMD ["bash"]

ENV TERM xterm

# ros-indigo-ros-base
RUN apt-get update && apt-get install -y --allow-unauthenticated \
  ros-indigo-desktop-full=1.1.4-0* \
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
	swig \
	python-pygame \
	python-pip \
	python-dev \
	git \
	libgflags-dev \
	libgoogle-glog-dev  \
	liblmdb-dev \
	autoconf  \
	automake \
	libtool \
	unzip \
	libprotobuf-dev libleveldb-dev \
	libsnappy-dev libopencv-dev \
	libhdf5-serial-dev protobuf-compiler \
	libatlas-base-dev \
	libopenblas-dev \
	&& rm -rf /var/lib/apt/lists/*
#
#
# Start with Caffe dependencies

#We need this for boost
RUN pip install --upgrade b2

ENV ROOT_DIR=/root

RUN cd $ROOT_DIR \
    && wget https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.gz \
    && tar -zvxf boost_1_61_0.tar.gz \
    && cd boost_1_61_0 \
    && ./bootstrap.sh --prefix=/usr/local --with-libraries=program_options atomic \
		link=static runtime-link=shared threading=multi \
    && ./b2 install \
    && cd $ROOT_DIR && rm boost*.gz
#
#protobuf-compiler
ENV PROTOBUF=/root/protobuf
RUN git clone https://github.com/google/protobuf.git \
		&& cd protobuf \
		&& bash autogen.sh \
		&& ./configure \
		&& make -j \
		&& make install \
		&& ldconfig \
		&& cd ../ \
		&& rm -rf protobuf

RUN cd /root \
    && wget https://ecs.utdallas.edu/~opo140030/docker_files/cudnn.tar.gz \
		&& tar -zvxf cudnn.tar.gz \
		&& cd cudnnv5.1 \
		&& cp include/cudnn.h /usr/local/cuda/include \
		&& cp include/cudnn.h /usr/local/cuda-8.0/include \
		&& cp lib64/*.* /usr/local/cuda-8.0/lib64 \
		&& cp lib64/*.* /usr/local/cuda/lib64 \
    && rm /root/cudnn* -rf

#clone caffe
ENV CAFFE_ROOT=/root/caffe/
RUN cd ~ \
    && git clone https://github.com/BVLC/caffe.git

COPY CaffeCMake.txt $CAFFE_ROOT/CMakeLists.txt

RUN cd $CAFFE_ROOT \
		&& mkdir build && cd build \
		&& cmake -DUSE_CUDNN=ON ..  \
		&& make -j"$(nproc)" all \
		&& make install

# Setup catkin workspace
RUN /bin/bash -c echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc \
		&& /bin/bash -c echo "export PYTHONPATH=${PYTHONPATH:+:${PYTHONPATH}}:/root/caffe/python:/root/catkin_ws/src/gps" >> ~/.bashrc \
		&& /bin/bash -c echo " source /usr/local/etc/bash_completion.d/catkin_tools-completion.bash" >> ~/.bashrc \
		&& /bin/bash -c "source /root/.bashrc" \
    && /bin/bash -c echo "export CAFFE_ROOT=/root/caffe/build"

RUN wget https://bootstrap.pypa.io/get-pip.py \
		&& python ./get-pip.py \
		&& apt-get install python-pip \
    && rm get-pip.py

#install catkin build
RUN pip install -U catkin_tools

ENV CATKIN_WS=/root/catkin_ws

RUN mkdir -p $CATKIN_WS/src && cd $CATKIN_WS/src \
		&& mkdir gps

COPY . $CATKIN_WS/src/gps

RUN /bin/bash -c "source /opt/ros/indigo/setup.bash" \
		&& cd $CATKIN_WS/src/gps \
    && rm CaffeCMake.txt \
		&& ./compile_proto.sh \
		&& cd gps_agent_pkg \
		&& echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list \
		&& apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 \
		&& apt-get update \
		&& rosdep install --from-paths -r -y . \
 		&& cd /root  \
		&& git clone https://github.com/pybox2d/pybox2d  \
		&& cd pybox2d  \
		&& python setup.py build  \
		&& python setup.py install \
		&& rm -rf /root/pybox2d

# split this to help debugging during build process
RUN cd $CATKIN_WS/src/gps \
		&& chmod 777 *.sh \
		&& cp *.sh $CATKIN_WS \
		&& pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

#ADD setup.sh $ROOT_DIR
# 
# RUN cd $ROOT_DIR \
#     && bash setup.sh \
#     && cd $CATKIN_WS/src \
#     && catkin init \
#     && cd $CATKIN_WS \
#     && export CAFFE_ROOT=/root/caffe/ \
#     && catkin build

RUN  echo   " ===========  Build Complete  =========   "
