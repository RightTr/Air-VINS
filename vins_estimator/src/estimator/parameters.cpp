/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
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
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
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
int LINE_BA;
double LINE_THRESHOLD;
int LINE_MIN_OBS;
int LINE_MIN_TRACK_CNT;
int LINE_MAX_CNT;
double LINE_MIN_LENGTH;
double LINE_STEREO_Y_THRESHOLD;
double LINE_MIN_STEREO_DISPARITY;
double LINE_MAX_STEREO_DISPARITY;
int LINE_MATCH_VOTE_THRESHOLD;
double LINE_MATCH_SCORE_THRESHOLD;
double LINE_MONO_CHI2;
double LINE_STEREO_CHI2;
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
void readOptionalParam(cv::FileStorage &fsSettings, const std::string &name, T &value)
{
    cv::FileNode node = fsSettings[name];
    if (!node.empty())
        node >> value;
}
} // namespace


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

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

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];
    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    DEEP_FEATURE = 0;
    DEEP_FEATURE_MODEL_DIR = joinPath(configPath, "../../output");
    DEEP_FEATURE_MATCHER = 1;
    DEEP_FEATURE_MAX_KEYPOINTS = MAX_CNT;
    DEEP_FEATURE_KEYPOINT_THRESHOLD = 0.004;
    DEEP_FEATURE_REMOVE_BORDERS = 4;
    DEEP_FEATURE_STEREO_Y_THRESHOLD = 2.0;
    DEEP_FEATURE_MIN_STEREO_DISPARITY = 1.0;
    DEEP_FEATURE_MAX_STEREO_DISPARITY = 96.0;
    LINE_BA = 0;
    LINE_THRESHOLD = 0.7;
    LINE_MIN_OBS = 5;
    LINE_MIN_TRACK_CNT = 2;
    LINE_MAX_CNT = 100;
    LINE_MIN_LENGTH = 50.0;
    LINE_STEREO_Y_THRESHOLD = 3.0;
    LINE_MIN_STEREO_DISPARITY = 1.0;
    LINE_MAX_STEREO_DISPARITY = 128.0;
    LINE_MATCH_VOTE_THRESHOLD = 2;
    LINE_MATCH_SCORE_THRESHOLD = 0.8;
    LINE_MONO_CHI2 = 50.0;
    LINE_STEREO_CHI2 = 75.0;
    FEATURE_MIN_DEPTH = 0.3;
    FEATURE_MAX_DEPTH = 40.0;
    ENABLE_NUC_HANDLE = 0;
    readOptionalParam(fsSettings, "deep_feature", DEEP_FEATURE);
    readOptionalParam(fsSettings, "deep_feature_model_dir", DEEP_FEATURE_MODEL_DIR);
    readOptionalParam(fsSettings, "deep_feature_matcher", DEEP_FEATURE_MATCHER);
    readOptionalParam(fsSettings, "deep_feature_max_keypoints", DEEP_FEATURE_MAX_KEYPOINTS);
    readOptionalParam(fsSettings, "deep_feature_keypoint_threshold", DEEP_FEATURE_KEYPOINT_THRESHOLD);
    readOptionalParam(fsSettings, "deep_feature_remove_borders", DEEP_FEATURE_REMOVE_BORDERS);
    readOptionalParam(fsSettings, "deep_feature_stereo_y_threshold", DEEP_FEATURE_STEREO_Y_THRESHOLD);
    readOptionalParam(fsSettings, "deep_feature_min_stereo_disparity", DEEP_FEATURE_MIN_STEREO_DISPARITY);
    readOptionalParam(fsSettings, "deep_feature_max_stereo_disparity", DEEP_FEATURE_MAX_STEREO_DISPARITY);
    readOptionalParam(fsSettings, "line_ba", LINE_BA);
    readOptionalParam(fsSettings, "line_threshold", LINE_THRESHOLD);
    readOptionalParam(fsSettings, "line_min_obs", LINE_MIN_OBS);
    readOptionalParam(fsSettings, "line_min_track_cnt", LINE_MIN_TRACK_CNT);
    readOptionalParam(fsSettings, "line_max_cnt", LINE_MAX_CNT);
    readOptionalParam(fsSettings, "line_min_length", LINE_MIN_LENGTH);
    readOptionalParam(fsSettings, "line_stereo_y_threshold", LINE_STEREO_Y_THRESHOLD);
    readOptionalParam(fsSettings, "line_min_stereo_disparity", LINE_MIN_STEREO_DISPARITY);
    readOptionalParam(fsSettings, "line_max_stereo_disparity", LINE_MAX_STEREO_DISPARITY);
    readOptionalParam(fsSettings, "line_match_vote_threshold", LINE_MATCH_VOTE_THRESHOLD);
    readOptionalParam(fsSettings, "line_match_score_threshold", LINE_MATCH_SCORE_THRESHOLD);
    readOptionalParam(fsSettings, "line_mono_chi2", LINE_MONO_CHI2);
    readOptionalParam(fsSettings, "line_stereo_chi2", LINE_STEREO_CHI2);
    readOptionalParam(fsSettings, "feature_min_depth", FEATURE_MIN_DEPTH);
    readOptionalParam(fsSettings, "feature_max_depth", FEATURE_MAX_DEPTH);
    readOptionalParam(fsSettings, "enable_nuc_handle", ENABLE_NUC_HANDLE);

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
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
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
