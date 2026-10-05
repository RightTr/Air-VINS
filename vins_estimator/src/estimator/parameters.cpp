/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

double INIT_DEPTH;
double MIN_PARALLAX;
int KEYFRAME_MIN_INIT_STEREO_FEATURE;
int KEYFRAME_LOST_NUM_MATCH;
int KEYFRAME_MIN_NUM_MATCH;
int KEYFRAME_MAX_NUM_MATCH;
float KEYFRAME_TRACKING_POINT_RATE;
double KEYFRAME_TRACKING_PARALLAX_RATE;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string STIR_SLAM_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
std::string IMU_ACCEL_TOPIC, IMU_GYRO_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::array<Eigen::Vector4d, 2> CAM_INTRINSICS{
    Eigen::Vector4d::Zero(), Eigen::Vector4d::Zero()};
std::array<cv::Mat, 2> RECTIFY_MAP_1;
std::array<cv::Mat, 2> RECTIFY_MAP_2;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
int DEEP_FEATURE;
std::string DEEP_FEATURE_MODEL_DIR;
int DEEP_FEATURE_MATCHER;
int DEEP_FEATURE_MAX_KEYPOINTS;
double DEEP_FEATURE_KEYPOINT_THRESHOLD;
int DEEP_FEATURE_REMOVE_BORDERS;
double DEEP_FEATURE_STEREO_Y_THRESHOLD;
double DEEP_FEATURE_MIN_STEREO_DISPARITY;
double DEEP_FEATURE_MAX_STEREO_DISPARITY;
double FEATURE_MIN_DEPTH;
double FEATURE_MAX_DEPTH;
int ENABLE_NUC_HANDLE;

namespace
{
std::string joinPath(const std::string &folder, const std::string &name)
{
    if (folder.empty())
        return name;
    if (folder.back() == '/')
        return folder + name;
    return folder + "/" + name;
}

template <typename T>
void readOptionalParam(cv::FileStorage &fsSettings, const std::string &group,
                       const std::string &name, T &value)
{
    cv::FileNode node = fsSettings[group][name];
    if (!node.empty())
        node >> value;
}

bool readCameraIntrinsics(cv::FileStorage &fsSettings, const std::string &name,
                          Eigen::Vector4d &intrinsics)
{
    const cv::FileNode node = fsSettings["camera"][name];
    if (node.empty() || !node.isSeq() || node.size() != 4)
        return false;

    for (int i = 0; i < 4; ++i)
        intrinsics(i) = static_cast<double>(node[i]);

    return intrinsics.allFinite() && intrinsics.x() > 0.0 && intrinsics.y() > 0.0;
}

bool readCameraDistortion(cv::FileStorage &fsSettings, const std::string &name,
                          cv::Mat &distortion)
{
    const cv::FileNode node = fsSettings["camera"][name];
    if (node.empty() || !node.isSeq() || node.size() != 5)
        return false;

    distortion = cv::Mat(1, 5, CV_64F);
    for (int i = 0; i < 5; ++i)
    {
        const double value = static_cast<double>(node[i]);
        if (!std::isfinite(value))
            return false;
        distortion.at<double>(0, i) = value;
    }
    return true;
}

cv::Mat cameraMatrix(const Eigen::Vector4d &intrinsics)
{
    return (cv::Mat_<double>(3, 3) << intrinsics.x(), 0.0, intrinsics.z(),
                                      0.0, intrinsics.y(), intrinsics.w(),
                                      0.0, 0.0, 1.0);
}

void rectifyStereoParameters(cv::FileStorage &fsSettings)
{
    int rectify_stereo = 0;
    readOptionalParam(fsSettings, "camera", "rectify_stereo", rectify_stereo);
    if (!rectify_stereo)
        return;

    if (NUM_OF_CAM != 2 || ROW <= 0 || COL <= 0)
        throw std::runtime_error("rectify_stereo requires two cameras and valid image dimensions");

    cv::Mat D0, D1;
    if (!readCameraDistortion(fsSettings, "cam0_distortion", D0) ||
        !readCameraDistortion(fsSettings, "cam1_distortion", D1))
        throw std::runtime_error("rectify_stereo requires five finite distortion values per camera: [k1,k2,p1,p2,k3]");

    double alpha = 0.0;
    readOptionalParam(fsSettings, "camera", "rectify_alpha", alpha);
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0)
        throw std::runtime_error("rectify_alpha must be between 0 and 1");

    // OpenCV stereoRectify takes the transform from raw left to raw right.
    const Eigen::Matrix3d R_right_left = RIC[1].transpose() * RIC[0];
    const Eigen::Vector3d t_right_left = RIC[1].transpose() * (TIC[0] - TIC[1]);
    cv::Mat R, t;
    cv::eigen2cv(R_right_left, R);
    cv::eigen2cv(t_right_left, t);

    const cv::Size image_size(COL, ROW);
    const cv::Mat K0 = cameraMatrix(CAM_INTRINSICS[0]);
    const cv::Mat K1 = cameraMatrix(CAM_INTRINSICS[1]);
    cv::Mat R0_rect, R1_rect, P0_rect, P1_rect, Q;
    cv::stereoRectify(K0, D0, K1, D1, image_size, R, t,
                      R0_rect, R1_rect, P0_rect, P1_rect, Q,
                      cv::CALIB_ZERO_DISPARITY, alpha, image_size);
    cv::initUndistortRectifyMap(K0, D0, R0_rect, P0_rect, image_size, CV_16SC2,
                                RECTIFY_MAP_1[0], RECTIFY_MAP_2[0]);
    cv::initUndistortRectifyMap(K1, D1, R1_rect, P1_rect, image_size, CV_16SC2,
                                RECTIFY_MAP_1[1], RECTIFY_MAP_2[1]);

    for (int i = 0; i < 2; ++i)
    {
        const cv::Mat &rect_rotation = i == 0 ? R0_rect : R1_rect;
        const cv::Mat &projection = i == 0 ? P0_rect : P1_rect;
        Eigen::Matrix3d R_rect;
        cv::cv2eigen(rect_rotation, R_rect);
        // The rectified camera frame is rotated relative to the raw camera frame.
        RIC[i] = RIC[i] * R_rect.transpose();
        CAM_INTRINSICS[i] << projection.at<double>(0, 0), projection.at<double>(1, 1),
                             projection.at<double>(0, 2), projection.at<double>(1, 2);
    }
    ROS_INFO_STREAM("Stereo images rectified: " << COL << "x" << ROW
                    << ", fx=" << CAM_INTRINSICS[0].x()
                    << ", baseline=" << t_right_left.norm() << " m");
}
} // namespace


void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["topics"]["image0"] >> IMAGE0_TOPIC;
    fsSettings["topics"]["image1"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["feature_tracker"]["max_cnt"];
    MIN_DIST = fsSettings["feature_tracker"]["min_dist"];
    F_THRESHOLD = fsSettings["feature_tracker"]["F_threshold"];
    SHOW_TRACK = fsSettings["feature_tracker"]["show_track"];
    FLOW_BACK = fsSettings["feature_tracker"]["flow_back"];
    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    DEEP_FEATURE = 0;
    DEEP_FEATURE_MODEL_DIR = joinPath(configPath, "../../output");
    DEEP_FEATURE_MATCHER = 1;
    DEEP_FEATURE_MAX_KEYPOINTS = std::max(MAX_CNT, static_cast<int>(MAX_CNT * 3 / 2));
    DEEP_FEATURE_KEYPOINT_THRESHOLD = 0.003;
    DEEP_FEATURE_REMOVE_BORDERS = 4;
    DEEP_FEATURE_STEREO_Y_THRESHOLD = 2.0;
    DEEP_FEATURE_MIN_STEREO_DISPARITY = 1.0;
    DEEP_FEATURE_MAX_STEREO_DISPARITY = 96.0;
    FEATURE_MIN_DEPTH = 0.3;
    FEATURE_MAX_DEPTH = 40.0;
    ENABLE_NUC_HANDLE = 0;
    KEYFRAME_MIN_INIT_STEREO_FEATURE = 10;
    KEYFRAME_LOST_NUM_MATCH = 10;
    KEYFRAME_MIN_NUM_MATCH = 30;
    KEYFRAME_MAX_NUM_MATCH = 80;
    KEYFRAME_TRACKING_POINT_RATE = 0.7f;
    KEYFRAME_TRACKING_PARALLAX_RATE = 0.1;
    readOptionalParam(fsSettings, "deep_feature", "enabled", DEEP_FEATURE);
    readOptionalParam(fsSettings, "deep_feature", "model_dir", DEEP_FEATURE_MODEL_DIR);
    readOptionalParam(fsSettings, "deep_feature", "matcher", DEEP_FEATURE_MATCHER);
    readOptionalParam(fsSettings, "deep_feature", "max_keypoints", DEEP_FEATURE_MAX_KEYPOINTS);
    readOptionalParam(fsSettings, "deep_feature", "keypoint_threshold", DEEP_FEATURE_KEYPOINT_THRESHOLD);
    readOptionalParam(fsSettings, "deep_feature", "remove_borders", DEEP_FEATURE_REMOVE_BORDERS);
    readOptionalParam(fsSettings, "deep_feature", "stereo_y_threshold", DEEP_FEATURE_STEREO_Y_THRESHOLD);
    readOptionalParam(fsSettings, "deep_feature", "min_stereo_disparity", DEEP_FEATURE_MIN_STEREO_DISPARITY);
    readOptionalParam(fsSettings, "deep_feature", "max_stereo_disparity", DEEP_FEATURE_MAX_STEREO_DISPARITY);
    readOptionalParam(fsSettings, "feature_tracker", "min_depth", FEATURE_MIN_DEPTH);
    readOptionalParam(fsSettings, "feature_tracker", "max_depth", FEATURE_MAX_DEPTH);
    readOptionalParam(fsSettings, "nuc", "enable_handle", ENABLE_NUC_HANDLE);
    readOptionalParam(fsSettings, "keyframe", "min_init_stereo_feature", KEYFRAME_MIN_INIT_STEREO_FEATURE);
    readOptionalParam(fsSettings, "keyframe", "lost_num_match", KEYFRAME_LOST_NUM_MATCH);
    readOptionalParam(fsSettings, "keyframe", "min_num_match", KEYFRAME_MIN_NUM_MATCH);
    readOptionalParam(fsSettings, "keyframe", "max_num_match", KEYFRAME_MAX_NUM_MATCH);
    readOptionalParam(fsSettings, "keyframe", "tracking_point_rate", KEYFRAME_TRACKING_POINT_RATE);
    readOptionalParam(fsSettings, "keyframe", "tracking_parallax_rate", KEYFRAME_TRACKING_PARALLAX_RATE);

    MULTIPLE_THREAD = fsSettings["system"]["multiple_thread"];

    USE_IMU = fsSettings["system"]["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["topics"]["imu"] >> IMU_TOPIC;
        fsSettings["topics"]["imu_accel"] >> IMU_ACCEL_TOPIC;
        fsSettings["topics"]["imu_gyro"] >> IMU_GYRO_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["estimator"]["acc_n"];
        ACC_W = fsSettings["estimator"]["acc_w"];
        GYR_N = fsSettings["estimator"]["gyr_n"];
        GYR_W = fsSettings["estimator"]["gyr_w"];
        G.z() = fsSettings["estimator"]["g_norm"];
        TD = fsSettings["estimator"]["td"];
        ESTIMATE_TD = fsSettings["estimator"]["estimate_td"];
    }

    SOLVER_TIME = fsSettings["estimator"]["max_solver_time"];
    NUM_ITERATIONS = fsSettings["estimator"]["max_num_iterations"];
    MIN_PARALLAX = fsSettings["estimator"]["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["system"]["output_path"] >> OUTPUT_FOLDER;
    if (OUTPUT_FOLDER.empty())
        OUTPUT_FOLDER = joinPath(configPath, "../../output");
    STIR_SLAM_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << STIR_SLAM_RESULT_PATH << std::endl;
    std::ofstream fout(STIR_SLAM_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["camera"]["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["camera"]["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["system"]["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }
    if (!readCameraIntrinsics(fsSettings, "cam0_intrinsics", CAM_INTRINSICS[0]))
    {
        ROS_ERROR("cam0_intrinsics must contain finite [fx, fy, cx, cy] values with positive fx/fy");
        ROS_BREAK();
        return;
    }

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        if (!readCameraIntrinsics(fsSettings, "cam1_intrinsics", CAM_INTRINSICS[1]))
        {
            ROS_ERROR("cam1_intrinsics must contain finite [fx, fy, cx, cy] values with positive fx/fy");
            ROS_BREAK();
            return;
        }
        
        cv::Mat cv_T;
        fsSettings["camera"]["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["camera"]["image_height"];
    COL = fsSettings["camera"]["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    rectifyStereoParameters(fsSettings);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
