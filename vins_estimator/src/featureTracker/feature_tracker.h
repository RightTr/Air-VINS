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

#include <cstdio>
#include <iostream>
#include <queue>
#include <memory>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../estimator/feature_manager.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

class DeepFeature;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat(), int primary_camera_id = 0, bool allow_new_features = true);
    void clearState();
    void resetTrackingState();
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    LineFeatureFrameMap getLineFrame() const;
    bool getCameraIntrinsics(int camera_id, Eigen::Vector4d &intrinsics) const;
    bool inBorder(const cv::Point2f &pt);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    cv::Mat left_prev_img, right_prev_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt, left_track_cnt, right_track_cnt;
    vector<int> left_ids;
    Eigen::Matrix<float, 259, Eigen::Dynamic> prev_deep_features, left_deep_features, right_deep_features;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    int line_n_id;
    bool hasPrediction;
    int primary_camera_id;
    std::shared_ptr<DeepFeature> deep_feature;
    std::shared_ptr<DeepFeature> line_deep_feature;

private:
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImageDeep(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat(), int primary_camera_id = 0, bool allow_new_features = true);
    void initDeepFeatureFrontend();
    void initLineFeatureFrontend();
    void processLineFeatures(const cv::Mat &left_img, const cv::Mat &right_img);
    void drawLineTrack(const cv::Mat &imLeft, const cv::Mat &imRight);

    LineFeatureFrameMap line_feature_frame;
    std::vector<Eigen::Vector4d> prev_lines, cur_lines, cur_right_lines;
    std::vector<int> line_ids, prev_line_ids, line_track_cnt;
    std::vector<int> right_line_ids, cur_right_line_match_left;
    Eigen::Matrix<float, 259, Eigen::Dynamic> prev_line_points, cur_line_points, right_line_points;
    std::vector<std::map<int, double>> prev_points_on_lines, cur_points_on_lines, right_points_on_lines;
};
