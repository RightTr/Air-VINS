/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <array>
#include "../utility/ros_utils.h"
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int KEYFRAME_MIN_INIT_STEREO_FEATURE;
extern int KEYFRAME_LOST_NUM_MATCH;
extern int KEYFRAME_MIN_NUM_MATCH;
extern int KEYFRAME_MAX_NUM_MATCH;
extern float KEYFRAME_TRACKING_POINT_RATE;
extern double KEYFRAME_TRACKING_PARALLAX_RATE;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string STIR_SLAM_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern std::string IMU_ACCEL_TOPIC, IMU_GYRO_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::array<Eigen::Vector4d, 2> CAM_INTRINSICS;
// Empty unless the input stereo images need in-node undistortion/rectification.
extern std::array<cv::Mat, 2> RECTIFY_MAP_1;
extern std::array<cv::Mat, 2> RECTIFY_MAP_2;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern int DEEP_FEATURE;
extern std::string DEEP_FEATURE_MODEL_DIR;
extern int DEEP_FEATURE_MATCHER;
extern int DEEP_FEATURE_MAX_KEYPOINTS;
extern double DEEP_FEATURE_KEYPOINT_THRESHOLD;
extern int DEEP_FEATURE_REMOVE_BORDERS;
extern double DEEP_FEATURE_STEREO_Y_THRESHOLD;
extern double DEEP_FEATURE_MIN_STEREO_DISPARITY;
extern double DEEP_FEATURE_MAX_STEREO_DISPARITY;
extern double FEATURE_MIN_DEPTH;
extern double FEATURE_MAX_DEPTH;
extern int ENABLE_NUC_HANDLE;

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
