/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "ros_utils.h"
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include <map>
#include <opencv2/core.hpp>
#include "../estimator/estimator.h"

extern ros_utils::Publisher<nav_msgs::Odometry> pub_odometry;
extern ros_utils::Publisher<nav_msgs::Path> pub_path;
extern ros_utils::Publisher<visualization_msgs::Marker> pub_window_keypose;
extern ros_utils::Publisher<sensor_msgs::PointCloud> pub_keyframe_point;
extern ros_utils::Publisher<nav_msgs::Odometry> pub_keyframe_pose;
extern ros_utils::Publisher<nav_msgs::Path> pub_keyframe_path;
extern ros_utils::Publisher<visualization_msgs::Marker> pub_keyframe_marker;
extern nav_msgs::Path path;
extern nav_msgs::Path keyframe_path;
extern visualization_msgs::Marker keyframe_marker;
extern ros_utils::Publisher<sensor_msgs::Image> pub_deep_match;
extern ros_utils::Publisher<sensor_msgs::Image> pub_line_match;
extern ros_utils::Publisher<visualization_msgs::Marker> pub_map_line;
extern ros_utils::Publisher<visualization_msgs::Marker> pub_window_map_line;
extern int IMAGE_ROW, IMAGE_COL;

void registerPub(const ros_utils::NodeHandle &n);
void shutdownPub();

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

void pubTrackImage(const cv::Mat &imgTrack, const double t);
void pubDeepMatchImage(const cv::Mat &imgMatch, const double t);
void pubLineMatchImage(const cv::Mat &imgMatch, const double t);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

void pubGolbalMapLine(const Estimator &estimator, const std_msgs::Header &header);
void pubWindowMapLine(const Estimator &estimator, const std_msgs::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyframe(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

void pubCar(const Estimator & estimator, const std_msgs::Header &header);
