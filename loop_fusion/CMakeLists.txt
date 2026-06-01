cmake_minimum_required(VERSION 2.8.3)
project(loop_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    )

find_package(OpenCV)


find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include /usr/local/include /usr/src/tensorrt/include
          /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    PATHS /usr/local/cuda/include /usr/include /usr/include/aarch64-linux-gnu
          /usr/include/x86_64-linux-gnu
)
find_library(NVINFER_LIBRARY nvinfer
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
)
find_library(CUDART_LIBRARY cudart
    PATHS /usr/local/cuda/lib64 /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR}
                    ${PROJECT_SOURCE_DIR}/src
                    ${TENSORRT_INCLUDE_DIR}
                    ${CUDA_INCLUDE_DIR})

catkin_package()

set(LOOP_FUSION_SOURCES
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    src/vprnet/vprnet.cpp
    )

add_executable(loop_fusion_node ${LOOP_FUSION_SOURCES})

set(LOOP_FUSION_TRT_LIBS ${NVINFER_LIBRARY} ${CUDART_LIBRARY})
if(NVINFER_PLUGIN_LIBRARY)
    list(APPEND LOOP_FUSION_TRT_LIBS ${NVINFER_PLUGIN_LIBRARY})
endif()
if(NVONNXPARSER_LIBRARY)
    list(APPEND LOOP_FUSION_TRT_LIBS ${NVONNXPARSER_LIBRARY})
endif()

target_link_libraries(loop_fusion_node ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}
                      ${LOOP_FUSION_TRT_LIBS})
