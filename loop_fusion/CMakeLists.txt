cmake_minimum_required(VERSION 3.16)
project(loop_fusion)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

if(NOT ROS_EDITION)
  if("$ENV{ROS_VERSION}" STREQUAL "2")
    set(ROS_EDITION ROS2)
  elseif("$ENV{ROS_VERSION}" STREQUAL "1")
    set(ROS_EDITION ROS1)
  else()
    message(FATAL_ERROR "Set ROS_EDITION to ROS1 or ROS2")
  endif()
endif()

set(ROS_DEPENDENCIES std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs cv_bridge)
if(ROS_EDITION STREQUAL "ROS1")
  add_compile_definitions(USE_ROS1)
  find_package(catkin REQUIRED COMPONENTS roscpp roslib ${ROS_DEPENDENCIES})
  catkin_package()
elseif(ROS_EDITION STREQUAL "ROS2")
  add_compile_definitions(USE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  foreach(dependency IN LISTS ROS_DEPENDENCIES)
    find_package(${dependency} REQUIRED)
  endforeach()
else()
  message(FATAL_ERROR "Unsupported ROS_EDITION=${ROS_EDITION}; use ROS1 or ROS2")
endif()

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT}/include /usr/include /usr/include/aarch64-linux-gnu
        /usr/local/include /usr/local/TensorRT/include /usr/src/tensorrt/include)
include(${PROJECT_SOURCE_DIR}/../cmake/TensorRTApi.cmake)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h HINTS $ENV{CUDA_HOME}/include /usr/local/cuda/include /usr/include
  /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu)
find_library(NVINFER_LIBRARY nvinfer
  HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib /usr/lib/aarch64-linux-gnu
        /usr/local/lib /usr/local/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
  HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib /usr/lib/aarch64-linux-gnu
        /usr/local/lib /usr/local/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib /usr/lib/aarch64-linux-gnu
        /usr/local/lib /usr/local/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(CUDART_LIBRARY cudart HINTS $ENV{CUDA_HOME}/lib64 /usr/local/cuda/lib64 /usr/lib /usr/local/lib
  /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)

set(LOOP_INCLUDE_DIRS
  ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/.. ${PROJECT_SOURCE_DIR}/../vins_estimator/src/featureTracker
  ${PROJECT_SOURCE_DIR}/../deepFeature ${PROJECT_SOURCE_DIR}/../deepFeature/include
  ${PROJECT_SOURCE_DIR}/../deepFeature/3rdparty/tensorrtbuffer/include
  ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})

set(LOOP_FUSION_SOURCES
  src/pose_graph_node.cpp src/pose_graph.cpp src/keyframe.cpp
  src/utility/CameraPoseVisualization.cpp src/ThirdParty/DBoW/BowVector.cpp
  src/ThirdParty/DBoW/FBrief.cpp src/ThirdParty/DBoW/FeatureVector.cpp
  src/ThirdParty/DBoW/QueryResults.cpp src/ThirdParty/DBoW/ScoringObject.cpp
  src/ThirdParty/DUtils/Random.cpp src/ThirdParty/DUtils/Timestamp.cpp
  src/ThirdParty/DVision/BRIEF.cpp src/ThirdParty/VocabularyBinary.cpp
  src/vprnet/vprnet.cpp ../vins_estimator/src/featureTracker/deep_feature.cpp
  ../deepFeature/src/light_glue.cpp ../deepFeature/src/point_matcher.cc
  ../deepFeature/src/super_glue.cpp ../deepFeature/src/super_point.cpp
  ../deepFeature/src/plnet.cpp
  ../deepFeature/src/utils.cc
  ../deepFeature/3rdparty/tensorrtbuffer/src/logger.cpp
  ../deepFeature/3rdparty/tensorrtbuffer/src/sample_options.cpp
  ../deepFeature/3rdparty/tensorrtbuffer/src/sample_utils.cpp)

add_executable(loop_fusion_node ${LOOP_FUSION_SOURCES})
target_include_directories(loop_fusion_node PRIVATE ${LOOP_INCLUDE_DIRS})
set(LOOP_FUSION_TRT_LIBS ${NVINFER_LIBRARY} ${CUDART_LIBRARY})
if(NVINFER_PLUGIN_LIBRARY)
  list(APPEND LOOP_FUSION_TRT_LIBS ${NVINFER_PLUGIN_LIBRARY})
endif()
if(NVONNXPARSER_LIBRARY)
  list(APPEND LOOP_FUSION_TRT_LIBS ${NVONNXPARSER_LIBRARY})
endif()
target_link_libraries(loop_fusion_node ${OpenCV_LIBS} ${CERES_LIBRARIES}
  ${LOOP_FUSION_TRT_LIBS} yaml-cpp)

if(ROS_EDITION STREQUAL "ROS1")
  target_include_directories(loop_fusion_node PRIVATE ${catkin_INCLUDE_DIRS})
  target_link_libraries(loop_fusion_node ${catkin_LIBRARIES})
else()
  ament_target_dependencies(loop_fusion_node rclcpp ament_index_cpp tf2 tf2_ros ${ROS_DEPENDENCIES})
  install(TARGETS loop_fusion_node DESTINATION lib/${PROJECT_NAME})
  install(FILES
    ${PROJECT_SOURCE_DIR}/../support_files/brief_k10L6.bin
    ${PROJECT_SOURCE_DIR}/../support_files/brief_pattern.yml
    DESTINATION share/${PROJECT_NAME}/support_files)
  ament_package()
endif()
