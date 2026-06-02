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
#include <list>
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
#include "utility/mapline.h"

enum VisualTrackingMode
{
    TRACKING_MODE_STEREO = 0,
    TRACKING_MODE_NUC_LEFT_ONLY = 1,
    TRACKING_MODE_NUC_RIGHT_ONLY = 2
};

struct LineObservationInput
{
    int camera_id;
    Eigen::Matrix<double, 8, 1> line;
    std::vector<std::pair<int, double>> supporting_points;
    int track_count;

    LineObservationInput() : camera_id(0), track_count(1)
    {
        line.setZero();
    }

    LineObservationInput(int _camera_id,
                         const Eigen::Matrix<double, 8, 1> &_line,
                         const std::vector<std::pair<int, double>> &_supporting_points = std::vector<std::pair<int, double>>(),
                         int _track_count = 1)
        : camera_id(_camera_id), line(_line), supporting_points(_supporting_points), track_count(_track_count)
    {
    }
};

using LineFeatureFrameMap = std::map<int, std::vector<LineObservationInput>>;

struct ProjectionCandidate
{
    int point_feature_id;
    int camera_id;
    int track_count;
    int observer_count;
    int window_support_count;
    int last_support_frame_id;
    Eigen::Vector3d point_cam;
    Eigen::Matrix<float, 256, 1> descriptor;

    ProjectionCandidate()
        : point_feature_id(-1), camera_id(0), track_count(0), observer_count(0),
          window_support_count(0), last_support_frame_id(-1), point_cam(Eigen::Vector3d::Zero())
    {
        descriptor.setZero();
    }
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

    int endFrame();
};

class LinePerFrame
{
  public:
    LinePerFrame(const LineObservationInput &_obs, double td)
    {
        start_point = Eigen::Vector3d(_obs.line(0), _obs.line(1), 1.0);
        end_point = Eigen::Vector3d(_obs.line(2), _obs.line(3), 1.0);
        uv_start = Eigen::Vector2d(_obs.line(4), _obs.line(5));
        uv_end = Eigen::Vector2d(_obs.line(6), _obs.line(7));
        cur_td = td;
        camera_id = _obs.camera_id;
        supporting_points = _obs.supporting_points;
        track_count = _obs.track_count;
        is_stereo = false;
        lineobs.setZero();
        lineobs_R.setZero();
        lineobs << start_point.x(), start_point.y(), end_point.x(), end_point.y();
    }

    void rightObservation(const LineObservationInput &_obs)
    {
        start_point_right = Eigen::Vector3d(_obs.line(0), _obs.line(1), 1.0);
        end_point_right = Eigen::Vector3d(_obs.line(2), _obs.line(3), 1.0);
        uv_start_right = Eigen::Vector2d(_obs.line(4), _obs.line(5));
        uv_end_right = Eigen::Vector2d(_obs.line(6), _obs.line(7));
        supporting_points_right = _obs.supporting_points;
        track_count = std::max(track_count, _obs.track_count);
        is_stereo = true;
        lineobs_R << start_point_right.x(), start_point_right.y(), end_point_right.x(), end_point_right.y();
    }

    double cur_td;
    int camera_id;
    Eigen::Vector3d start_point, end_point;
    Eigen::Vector3d start_point_right, end_point_right;
    Eigen::Vector2d uv_start, uv_end;
    Eigen::Vector2d uv_start_right, uv_end_right;
    Eigen::Vector4d lineobs;
    Eigen::Vector4d lineobs_R;
    std::vector<std::pair<int, double>> supporting_points;
    std::vector<std::pair<int, double>> supporting_points_right;
    int track_count;
    bool is_stereo;
};

class LinePerId
{
  public:
    const int line_id;
    int start_frame;
    std::vector<LinePerFrame> line_per_frame;
    int used_num;
    int solve_flag;
    Eigen::Matrix<double, 6, 1> line_3d_world;
    bool is_triangulation;
    MaplinePtr mapline;

    LinePerId(int _line_id, int _start_frame)
        : line_id(_line_id), start_frame(_start_frame), used_num(0), solve_flag(0),
          is_triangulation(false), mapline(std::make_shared<Mapline>(_line_id))
    {
        line_3d_world.setZero();
    }

    int endFrame() const
    {
        return start_frame + static_cast<int>(line_per_frame.size()) - 1;
    }
};

#endif  // VINS_MANAGER_FEATURE_TYPES_H_
