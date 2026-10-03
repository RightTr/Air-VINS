/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef VINS_MANAGER_FEATURE_TYPES_H_
#define VINS_MANAGER_FEATURE_TYPES_H_

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

#include "../estimator/parameters.h"
#include "utility/frameObs.h"
#include "utility/mappoint.h"

enum VisualTrackingMode
{
    TRACKING_MODE_STEREO = 0,
    TRACKING_MODE_NUC_LEFT_ONLY = 1,
    TRACKING_MODE_NUC_RIGHT_ONLY = 2
};

struct KeyframeGoodPointRecord
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int point_feature_id;
    Eigen::Vector3d point_w;
    Eigen::Vector2d point_uv;
    Eigen::Vector2d point_norm;
    Eigen::Matrix<float, 256, 1> descriptor;

    KeyframeGoodPointRecord()
        : point_feature_id(-1), point_w(Eigen::Vector3d::Zero()), point_uv(Eigen::Vector2d::Zero()), point_norm(Eigen::Vector2d::Zero())
    {
        descriptor.setZero();
    }
};

using KeyframeGoodPointRecordList = std::vector<KeyframeGoodPointRecord, Eigen::aligned_allocator<KeyframeGoodPointRecord>>;

class PointFeatureFrame
{
  public:
    PointFeatureFrame(const Eigen::Matrix<double, 7, 1> &_point, double td, int _camera_id)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
        camera_id = _camera_id;
        is_stereo = false;
    }

    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5);
        velocityRight.y() = _point(6);
        is_stereo = true;
    }

    double cur_td;
    int camera_id;
    Eigen::Vector3d point, pointRight;
    Eigen::Vector2d uv, uvRight;
    Eigen::Vector2d velocity, velocityRight;
    bool is_stereo;
};

class PointFeatureId
{
  public:
    const int point_feature_id;
    int start_frame;
    std::vector<PointFeatureFrame> point_feature_frame;
    int used_num;
    double estimated_depth;
    bool reliable_depth;
    int solve_flag;
    int depth_fail_count;

    PointFeatureId(int _point_feature_id, int _start_frame)
        : point_feature_id(_point_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), reliable_depth(false), solve_flag(0), depth_fail_count(0)
    {
    }

    const PointFeatureFrame &anchorObservation() const
    {
        return point_feature_frame.front();
    }

    PointFeatureFrame &anchorObservation()
    {
        return point_feature_frame.front();
    }

    int endFrame() const;
};

#endif  // VINS_MANAGER_FEATURE_TYPES_H_
