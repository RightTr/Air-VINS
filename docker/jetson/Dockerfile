FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
ARG L4T_REPO=r36.5
ARG BUILD_JOBS=4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    CMAKE_PREFIX_PATH=/opt/ros/humble:/usr/local \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64

COPY .cache/ceres.tar.gz /tmp/ceres.tar.gz
COPY .cache/ros.key /tmp/ros.key

# Install the same general-purpose dependencies as the x86 image.
RUN test "$(dpkg --print-architecture)" = "arm64" && \
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

# ROS 2 Humble from the official Jammy arm64 repository.
RUN gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg /tmp/ros.key && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-desktop \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-tf2-geometry-msgs \
      ros-${ROS_DISTRO}-message-filters \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-std-msgs \
      ros-${ROS_DISTRO}-nav-msgs \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
      ros-${ROS_DISTRO}-rosbag2 \
      ros-${ROS_DISTRO}-ament-cmake \
      ros-${ROS_DISTRO}-python-cmake-module \
      python3-colcon-common-extensions python3-rosdep && \
    rm -f /tmp/ros.key && rm -rf /var/lib/apt/lists/*

# Jetson development packages must be present in the image for later builds.
# The NVIDIA runtime supplies the host driver and device nodes at run time.
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/common ${L4T_REPO} main" \
      > /etc/apt/sources.list.d/nvidia-jetson.list && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/t234 ${L4T_REPO} main" \
      >> /etc/apt/sources.list.d/nvidia-jetson.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      cuda-minimal-build-12-6 libcudla-12-6 \
      libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev && \
    if [ ! -e /usr/local/cuda ]; then ln -s /usr/local/cuda-12.6 /usr/local/cuda; fi && \
    rm -rf /var/lib/apt/lists/*

# Keep the Ceres source version used by the x86 image.
RUN mkdir -p /tmp/ceres && \
    tar -xzf /tmp/ceres.tar.gz --strip-components=1 -C /tmp/ceres && \
    cmake -S /tmp/ceres -B /tmp/ceres-build -GNinja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
      -DBUILD_WITH_MARCH_NATIVE=OFF \
      -DSUITESPARSE=OFF -DLAPACK=ON -DGFLAGS=ON -DGLOG=ON && \
    cmake --build /tmp/ceres-build --parallel "${BUILD_JOBS}" && \
    cmake --install /tmp/ceres-build && \
    rm -rf /tmp/ceres /tmp/ceres-build /tmp/ceres.tar.gz && ldconfig

# Hardware access is checked in a container started with the NVIDIA runtime.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    nvcc --version | grep -q 'release 12.6' && \
    opencv_version | grep -q '^4.5.' && \
    python3 -c 'import cv2; assert cv2.__version__.startswith("4.5.")' && \
    test -f /usr/local/include/ceres/ceres.h && \
    grep -Eq '^#define CERES_VERSION_MAJOR 2$' /usr/local/include/ceres/version.h && \
    grep -Eq '^#define CERES_VERSION_MINOR 0$' /usr/local/include/ceres/version.h && \
    test -f /usr/include/aarch64-linux-gnu/NvInfer.h && \
    test -f /usr/include/aarch64-linux-gnu/NvOnnxParser.h && \
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
