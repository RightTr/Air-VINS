FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
ARG CERES_VERSION=2.0.0
ARG TENSORRT_VERSION=8.6.1.6-1+cuda12.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_PREFIX_PATH=/opt/ros/humble:/usr/local \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64

COPY .cache/ceres.tar.gz /tmp/ceres.tar.gz
COPY .cache/ros.key /usr/share/keyrings/ros-archive-keyring.gpg

# Core build tools and native dependencies used by StirSLAM/Air-VINS.
RUN test "$(dpkg --print-architecture)" = "amd64" && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build gcc g++ git wget curl unzip \
      vim nano ca-certificates gnupg2 lsb-release locales \
      software-properties-common pkg-config \
      python3 python3-pip python3-dev python3-numpy python3-yaml \
      libopencv-dev python3-opencv \
      libeigen3-dev libboost-filesystem-dev libboost-system-dev \
      libboost-thread-dev libboost-python-dev \
      libatlas-base-dev libblas-dev liblapack-dev libsuitesparse-dev \
      libgoogle-glog-dev libgflags-dev libyaml-cpp-dev libomp-dev \
      libtbb-dev && \
    locale-gen en_US.UTF-8 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir numpy==1.23.5 PyYAML==6.0.2 && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble only. No ROS1 or catkin packages are installed.
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-desktop \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-tf2-geometry-msgs \
      ros-${ROS_DISTRO}-message-filters \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-nav-msgs \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-rosbag2 \
      ros-${ROS_DISTRO}-ament-cmake \
      ros-${ROS_DISTRO}-python-cmake-module \
      python3-colcon-common-extensions python3-rosdep && \
    rm -rf /var/lib/apt/lists/*

# TensorRT 8.6.1.6 for CUDA 12.x, pinned so a later NVIDIA repository update
# cannot silently move the development environment to TensorRT 10.
RUN curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
      libnvinfer8=${TENSORRT_VERSION} \
      libnvinfer-dev=${TENSORRT_VERSION} \
      libnvinfer-headers-dev=${TENSORRT_VERSION} \
      libnvinfer-plugin8=${TENSORRT_VERSION} \
      libnvinfer-plugin-dev=${TENSORRT_VERSION} \
      libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
      libnvonnxparsers8=${TENSORRT_VERSION} \
      libnvonnxparsers-dev=${TENSORRT_VERSION} && \
    apt-mark hold libnvinfer8 libnvinfer-dev libnvinfer-plugin8 \
      libnvinfer-headers-dev libnvinfer-plugin-dev \
      libnvinfer-headers-plugin-dev libnvonnxparsers8 \
      libnvonnxparsers-dev && \
    rm -rf /var/lib/apt/lists/*

# Ceres Solver 2.0.0 with the same sparse and logging dependencies used by StirSLAM.
RUN rm -rf /tmp/ceres && \
    mkdir -p /tmp/ceres && tar -xzf /tmp/ceres.tar.gz --strip-components=1 -C /tmp/ceres && \
    cmake -S /tmp/ceres -B /tmp/ceres-build -GNinja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
      -DSUITESPARSE=OFF -DLAPACK=ON -DGFLAGS=ON -DGLOG=ON && \
    cmake --build /tmp/ceres-build --parallel "$(nproc)" && \
    cmake --install /tmp/ceres-build && \
    rm -rf /tmp/ceres /tmp/ceres-build && ldconfig

# Fail the image build immediately if a requested version or critical library
# drifted. GPU availability itself is checked only when the container runs.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    nvcc --version | grep -q 'release 12.1' && \
    dpkg-query -W -f='${Version}' libcudnn8 | grep -q '^8.9.' && \
    opencv_version | grep -q '^4.5.' && \
    python3 -c 'import cv2; assert cv2.__version__.startswith("4.5.")' && \
    test -f /usr/local/include/ceres/ceres.h && \
    grep -Eq '^#define CERES_VERSION_MAJOR 2$' /usr/local/include/ceres/version.h && \
    grep -Eq '^#define CERES_VERSION_MINOR 0$' /usr/local/include/ceres/version.h && \
    test -f /usr/include/x86_64-linux-gnu/NvInfer.h && \
    dpkg-query -W -f='${Version}' libnvinfer8 | grep -q '^8.6.1.6-1+cuda12.0$' && \
    ros2 pkg prefix cv_bridge | grep -q "^/opt/ros/${ROS_DISTRO}$"

RUN printf '%s\n' \
      'source /opt/ros/humble/setup.bash' \
      > /etc/stir_slam_setup.bash && \
    printf '%s\n' \
      '' \
      '# StirSLAM environment' \
      'source /etc/stir_slam_setup.bash' \
      >> /etc/bash.bashrc

WORKDIR /root/stirslam_ws
CMD ["bash"]
