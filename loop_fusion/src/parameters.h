/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "vins_estimator/src/utility/camera.h"
#include <eigen3/Eigen/Dense>
#include "vins_estimator/src/utility/ros_utils.h"
#ifdef USE_ROS1
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#else
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#endif

extern Camera m_camera;
extern Eigen::Vector3d tic;
extern Eigen::Matrix3d qic;
extern ros_utils::Publisher<sensor_msgs::Image> pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string BRIEF_PATTERN_FILE;
extern std::string POSE_GRAPH_SAVE_PATH;
extern int ROW;
extern int COL;
extern std::string VINS_RESULT_PATH;
extern int DEBUG_IMAGE;
