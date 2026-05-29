/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
#include "../utility/tic_toc.h"
#include "../featureTracker/deepFeature/include/utils.h"

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

typedef map<int, vector<LineObservationInput>> LineFeatureFrameMap;

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td, int _camera_id)
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
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    bool is_stereo;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    bool reliable_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), reliable_depth(false), solve_flag(0)
    {
    }

    int endFrame();
};

class LinePerFrame
{
  public:
    LinePerFrame(const LineObservationInput &_obs, double td)
    {
        start_point = Vector3d(_obs.line(0), _obs.line(1), 1.0);
        end_point = Vector3d(_obs.line(2), _obs.line(3), 1.0);
        uv_start = Vector2d(_obs.line(4), _obs.line(5));
        uv_end = Vector2d(_obs.line(6), _obs.line(7));
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
        start_point_right = Vector3d(_obs.line(0), _obs.line(1), 1.0);
        end_point_right = Vector3d(_obs.line(2), _obs.line(3), 1.0);
        uv_start_right = Vector2d(_obs.line(4), _obs.line(5));
        uv_end_right = Vector2d(_obs.line(6), _obs.line(7));
        supporting_points_right = _obs.supporting_points;
        track_count = std::max(track_count, _obs.track_count);
        is_stereo = true;
        lineobs_R << start_point_right.x(), start_point_right.y(), end_point_right.x(), end_point_right.y();
    }

    double cur_td;
    int camera_id;
    Vector3d start_point, end_point;
    Vector3d start_point_right, end_point_right;
    Vector2d uv_start, uv_end;
    Vector2d uv_start_right, uv_end_right;
    Vector4d lineobs;
    Vector4d lineobs_R;
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
    vector<LinePerFrame> line_per_frame;
    int used_num;
    int solve_flag;
    Eigen::Matrix<double, 6, 1> line_3d_world;
    bool is_triangulation;

    LinePerId(int _line_id, int _start_frame)
        : line_id(_line_id), start_frame(_start_frame), used_num(0), solve_flag(0),
          is_triangulation(false)
    {
        line_3d_world.setZero();
    }

    int endFrame() const
    {
        return start_frame + static_cast<int>(line_per_frame.size()) - 1;
    }
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[]);
    void setVisualTrackingMode(VisualTrackingMode mode);
    void setActiveCameraId(int camera_id);
    bool knownLandmarksOnly() const;
    bool allowDepthInitialization() const;
    bool hasReliableDepth(const FeaturePerId &feature_per_id) const;
    bool useFeatureForOptimization(const FeaturePerId &feature_per_id) const;
    void clearState();
    int getFeatureCount();
    int getLineFeatureCount();
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void addLineFeature(int frame_count, const LineFeatureFrameMap &lines, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulateLine(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], int active_camera_id);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                              Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                              Vector3d tic[], Matrix3d ric[]);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex);
    void removeLineOutliers(const std::map<int, std::set<int>> &outlier_observers);
    void setLineFeature(const Eigen::Matrix<double, Eigen::Dynamic, 4> &x,
                        int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    Eigen::Matrix<double, Eigen::Dynamic, 4> getLineFeatureVector();
    bool useLineForOptimization(const LinePerId &line_per_id) const;
    list<FeaturePerId> feature;
    list<LinePerId> line_feature;
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[2];
    Eigen::Vector4d line_camera_intrinsics[2];
    VisualTrackingMode visual_tracking_mode;
    int active_camera_id;
};

#endif
