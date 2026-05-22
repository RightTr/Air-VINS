cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
    $ENV{TENSORRT_ROOT}/include
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/local/include
    /usr/local/TensorRT/include
    /usr/src/tensorrt/include
)
find_library(NVINFER_LIBRARY nvinfer
    HINTS
    $ENV{TENSORRT_ROOT}/lib
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/local/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
    HINTS
    $ENV{TENSORRT_ROOT}/lib
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/local/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/featureTracker/deepFeature/3rdparty/tensorrtbuffer)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src/featureTracker/deepFeature
  ${PROJECT_SOURCE_DIR}/src/featureTracker/deepFeature/include
  ${PROJECT_SOURCE_DIR}/src/featureTracker/deepFeature/3rdparty/tensorrtbuffer/include
)

catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/deepFeature/src/super_point.cpp
    src/featureTracker/deepFeature/src/super_glue.cpp
    src/featureTracker/deepFeature/src/light_glue.cpp
    src/featureTracker/deepFeature/src/point_matcher.cc
    src/featureTracker/deepFeature/src/utils.cc)
target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${CUDA_LIBRARIES} ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY} tensorrtbuffer)


add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib) 

add_executable(nus_thermal_publisher src/nus_thermal_publisher.cpp)
target_link_libraries(nus_thermal_publisher ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
target_link_libraries(kitti_odom_test vins_lib) 

add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
target_link_libraries(kitti_gps_test vins_lib) 
