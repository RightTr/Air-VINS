cmake_minimum_required(VERSION 3.16)
project(stir_slam)

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

set(ROS_DEPENDENCIES
  std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs cv_bridge image_transport)
if(ROS_EDITION STREQUAL "ROS1")
  add_compile_definitions(USE_ROS1)
  find_package(catkin REQUIRED COMPONENTS roscpp roslib tf ${ROS_DEPENDENCIES})
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
find_package(CUDA REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT}/include /usr/include /usr/include/aarch64-linux-gnu
        /usr/local/include /usr/local/TensorRT/include /usr/src/tensorrt/include)
include(${PROJECT_SOURCE_DIR}/../cmake/TensorRTApi.cmake)
find_library(NVINFER_LIBRARY nvinfer
  HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib /usr/lib/aarch64-linux-gnu
        /usr/local/lib /usr/local/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
  HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib /usr/lib/aarch64-linux-gnu
        /usr/local/lib /usr/local/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)

add_subdirectory(
  ${PROJECT_SOURCE_DIR}/../deepFeature/3rdparty/tensorrtbuffer
  ${CMAKE_CURRENT_BINARY_DIR}/deepFeature_tensorrtbuffer)

set(STIR_SLAM_INCLUDE_DIRS
  ${EIGEN3_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/.. ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/../deepFeature ${PROJECT_SOURCE_DIR}/../deepFeature/include
  ${PROJECT_SOURCE_DIR}/../deepFeature/3rdparty/tensorrtbuffer/include)

set(STIR_SLAM_SOURCES
  src/estimator/parameters.cpp src/estimator/estimator.cpp
  src/manager/feature_manager.cpp
  src/factor/pose_local_parameterization.cpp
  src/factor/projectionTwoFrameOneCamFactor.cpp
  src/factor/projectionTwoFrameTwoCamFactor.cpp
  src/factor/projectionOneFrameTwoCamFactor.cpp src/factor/marginalization_factor.cpp
  src/utility/utility.cpp
  src/utility/mappoint.cpp src/utility/visualization.cpp
  src/utility/CameraPoseVisualization.cpp
  src/initial/solve_5pts.cpp src/initial/initial_aligment.cpp
  src/initial/initial_sfm.cpp src/initial/initial_ex_rotation.cpp
  src/featureTracker/feature_tracker.cpp src/featureTracker/deep_feature.cpp
  ../deepFeature/src/super_point.cpp ../deepFeature/src/super_glue.cpp
  ../deepFeature/src/light_glue.cpp
  ../deepFeature/src/plnet.cpp
  ../deepFeature/src/point_matcher.cc ../deepFeature/src/utils.cc)

add_library(stir_slam_lib ${STIR_SLAM_SOURCES})
target_include_directories(stir_slam_lib PUBLIC ${STIR_SLAM_INCLUDE_DIRS})
target_link_libraries(stir_slam_lib ${OpenCV_LIBS} ${CERES_LIBRARIES} ${CUDA_LIBRARIES}
  ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY} tensorrtbuffer)

add_executable(stir_slam_node src/rosNodeTest.cpp)
target_include_directories(stir_slam_node PRIVATE ${STIR_SLAM_INCLUDE_DIRS})
target_link_libraries(stir_slam_node stir_slam_lib)

add_executable(nus_thermal_publisher src/nus_thermal_publisher.cpp)
target_include_directories(nus_thermal_publisher PRIVATE ${STIR_SLAM_INCLUDE_DIRS})
target_link_libraries(nus_thermal_publisher ${OpenCV_LIBS})
add_executable(tartan_thermal_publisher src/tartan_thermal_publisher.cpp)
target_include_directories(tartan_thermal_publisher PRIVATE ${STIR_SLAM_INCLUDE_DIRS})
target_link_libraries(tartan_thermal_publisher ${OpenCV_LIBS})
set_target_properties(stir_slam_node nus_thermal_publisher tartan_thermal_publisher
  PROPERTIES INSTALL_RPATH "$ORIGIN")

if(ROS_EDITION STREQUAL "ROS1")
  target_include_directories(stir_slam_lib PUBLIC ${catkin_INCLUDE_DIRS})
  target_link_libraries(stir_slam_lib ${catkin_LIBRARIES})
  target_link_libraries(nus_thermal_publisher ${catkin_LIBRARIES})
  target_link_libraries(tartan_thermal_publisher ${catkin_LIBRARIES})
else()
  set(STIR_SLAM_ROS2_DEPENDENCIES rclcpp ament_index_cpp tf2 tf2_ros ${ROS_DEPENDENCIES})
  ament_target_dependencies(stir_slam_lib ${STIR_SLAM_ROS2_DEPENDENCIES})
  ament_target_dependencies(stir_slam_node ${STIR_SLAM_ROS2_DEPENDENCIES})
  ament_target_dependencies(nus_thermal_publisher ${STIR_SLAM_ROS2_DEPENDENCIES})
  ament_target_dependencies(tartan_thermal_publisher ${STIR_SLAM_ROS2_DEPENDENCIES})
  install(TARGETS stir_slam_lib stir_slam_node nus_thermal_publisher tartan_thermal_publisher DESTINATION lib/${PROJECT_NAME})
  install(FILES src/utility/ros_utils.h DESTINATION include/${PROJECT_NAME}/utility)
  install(DIRECTORY launch_ROS2/ DESTINATION share/${PROJECT_NAME}/launch)
  install(DIRECTORY ../config DESTINATION share/${PROJECT_NAME})
  install(DIRECTORY ../output DESTINATION share/${PROJECT_NAME})
  ament_package()
endif()
