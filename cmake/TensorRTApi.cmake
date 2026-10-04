# Select the TensorRT API used by both ROS packages. The installed headers
# determine the ABI; an explicit selection is checked against them.
set(STIRSLAM_TRT_API "AUTO" CACHE STRING "TensorRT API major version: AUTO, 8, or 10")
set_property(CACHE STIRSLAM_TRT_API PROPERTY STRINGS AUTO 8 10)

if(NOT STIRSLAM_TRT_API STREQUAL "AUTO" AND
   NOT STIRSLAM_TRT_API STREQUAL "8" AND
   NOT STIRSLAM_TRT_API STREQUAL "10")
  message(FATAL_ERROR "STIRSLAM_TRT_API must be AUTO, 8, or 10")
endif()

if(NOT TENSORRT_INCLUDE_DIR OR NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
  message(FATAL_ERROR "TensorRT development headers (NvInferVersion.h) were not found")
endif()

file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major_line
  REGEX "^#define[ \t]+NV_TENSORRT_MAJOR[ \t]+[0-9]+")
string(REGEX MATCH "NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _trt_major_match "${_trt_major_line}")
set(_trt_major "${CMAKE_MATCH_1}")

if(NOT _trt_major STREQUAL "8" AND NOT _trt_major STREQUAL "10")
  message(FATAL_ERROR "TensorRT API ${_trt_major} is not supported; expected 8 or 10")
endif()
if(NOT STIRSLAM_TRT_API STREQUAL "AUTO" AND
   NOT STIRSLAM_TRT_API STREQUAL "${_trt_major}")
  message(FATAL_ERROR "STIRSLAM_TRT_API=${STIRSLAM_TRT_API} conflicts with installed TensorRT ${_trt_major} headers")
endif()

message(STATUS "StirSLAM TensorRT API: ${_trt_major} (selection: ${STIRSLAM_TRT_API})")
