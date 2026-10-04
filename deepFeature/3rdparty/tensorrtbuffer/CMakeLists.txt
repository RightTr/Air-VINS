cmake_minimum_required(VERSION 3.6)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_library(tensorrtbuffer SHARED src/logger.cpp src/sample_options.cpp src/sample_utils.cpp)

# Install beside the ROS2 node so the executable can resolve the shared
# library from its own runtime directory after `colcon build --symlink-install`
# or a normal install build.
install(TARGETS tensorrtbuffer
    LIBRARY DESTINATION lib/${CMAKE_PROJECT_NAME}
    ARCHIVE DESTINATION lib/${CMAKE_PROJECT_NAME}
    RUNTIME DESTINATION bin)
