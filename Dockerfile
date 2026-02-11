# ---------------------------------------------------------------------------
# GPS iDG — Development image
#
# Base  : NVIDIA CUDA 12.4 on Ubuntu 22.04
# ROS   : Noetic (backported / source build on 22.04 via unofficial PPA)
# Python: 3.10 (Ubuntu 22.04 default)
# NN    : PyTorch 2.1 (CPU + CUDA) via pip; LibTorch C++ from the same wheel
# Build : catkin (cmake 3.22+ on Ubuntu 22.04)
#
# Caffe and all Caffe dependencies have been removed.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

LABEL maintainer="gps-idg"

# ---- Locale & timezone (non-interactive) ----------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales \
        tzdata \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ---- System build tools ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        nano \
        unzip \
        pkg-config \
        libssl-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 3.10 + pip ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch 2.1 (CUDA 12.1 wheel also works on CUDA 12.4) ---------------
# Install PyTorch first so we can extract LibTorch for the C++ build later.
RUN pip install --no-cache-dir \
        torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# ---- Protocol buffers (pip wheel ships a compatible protoc binary) --------
RUN pip install --no-cache-dir \
        "protobuf>=4.23,<5.0" \
        "grpcio-tools>=1.56"

# ---- Python runtime dependencies ------------------------------------------
RUN pip install --no-cache-dir \
        "numpy>=1.24,<2.0" \
        "scipy>=1.10,<2.0" \
        "visdom>=0.2" \
        "scikit-image>=0.21" \
        "pybox2d>=2.3" \
        "catkin-tools>=0.9"

# ---- ROS Noetic (unofficial Ubuntu 22.04 PPA) -----------------------------
# The official ROS 1 Noetic targets Ubuntu 20.04. On 22.04 we use the
# community backport PPA maintained by the ROS community.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        lsb-release \
    && add-apt-repository -y ppa:v-launchpad-jochen-sprickerhof-de/ros \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-ros-base \
        ros-noetic-tf \
        ros-noetic-realtime-tools \
        ros-noetic-kdl-parser \
        ros-noetic-pr2-controller-manager \
        ros-noetic-pr2-mechanism-model \
        ros-noetic-control-toolbox \
        python3-rosdep \
        python3-catkin-tools \
        python3-wstool \
    && rm -rf /var/lib/apt/lists/*

# ---- rosdep init ----------------------------------------------------------
RUN rosdep init && rosdep update

# ---- LibTorch path for CMake find_package(Torch) --------------------------
# Point TORCH_ROOT at the pip-installed torch so catkin can find it.
ENV TORCH_ROOT=/usr/local/lib/python3.10/dist-packages/torch
ENV CMAKE_PREFIX_PATH="${TORCH_ROOT}:${CMAKE_PREFIX_PATH}"

# ---- Catkin workspace setup -----------------------------------------------
ENV ROS_DISTRO=noetic
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p ${CATKIN_WS}/src

# ---- Copy GPS source tree -------------------------------------------------
COPY . ${CATKIN_WS}/src/gps

# ---- Compile gps.proto using grpcio-tools protoc --------------------------
RUN cd ${CATKIN_WS}/src/gps \
    && python -m grpc_tools.protoc \
        -I gps_agent_pkg/proto \
        --python_out=python/gps/proto \
        gps_agent_pkg/proto/gps.proto \
    && touch python/gps/proto/__init__.py

# ---- catkin build ---------------------------------------------------------
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && cd ${CATKIN_WS}/src/gps/gps_agent_pkg \
    && rosdep install --from-paths . -r -y --ignore-src \
    && cd ${CATKIN_WS} \
    && catkin init \
    && catkin config --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DTORCH_ROOT=${TORCH_ROOT} \
    && catkin build --no-status

# ---- Environment setup in shell -------------------------------------------
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "source ${CATKIN_WS}/devel/setup.bash" >> ~/.bashrc \
    && echo "export PYTHONPATH=${CATKIN_WS}/src/gps/python:\${PYTHONPATH}" >> ~/.bashrc \
    && echo "export GPS_ROOT_DIR=${CATKIN_WS}/src/gps" >> ~/.bashrc \
    && echo "export TORCH_ROOT=${TORCH_ROOT}" >> ~/.bashrc

WORKDIR ${CATKIN_WS}/src/gps
CMD ["bash"]
