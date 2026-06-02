/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef VINS_MANAGER_FEATURE_MANAGER_H_
#define VINS_MANAGER_FEATURE_MANAGER_H_

#include <memory>
#include <array>
#include "point_feature_manager.h"
#include "line_feature_manager.h"

class FeatureManager
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit FeatureManager(Eigen::Matrix3d _Rs[]);

    void clearState();
    void setRic(Eigen::Matrix3d _ric[]);
    void setLineBA(bool enabled);
    void setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[]);
    void setVisualTrackingMode(VisualTrackingMode mode);
    VisualTrackingMode getVisualTrackingMode() const;
    void setActiveCameraId(int camera_id);
    void setWindowState(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void addLocalFrame(int frameCnt, double header,
                       const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image);
    void updateLocalPoints(int frameCnt);
    bool addPointFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    KeyframeGoodPointRecordList collectGoodKeyframePoints() const;
    int countGoodLocalMappoints() const;
    int countCurrentStereoObservations() const;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r, bool only_good = false);
    void setPointFeatureDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearPointFeatureDepth();
    Eigen::VectorXd getPointFeatureDepthVector();
    void triangulatePointFeature(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    std::vector<ProjectionCandidate> collectProjectionCandidates(const Eigen::Matrix4d &nextT) const;
    void updateMappointDescriptors(const std::vector<int> &ids, const Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
    void initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[], int active_camera_id_, bool prefer_good_points = false);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D);
    void removePointFeatureBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                          Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                          Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void removeBack();
    void removeFront(int frame_count);
    void removePointFeatureOutlier(std::set<int> &outlierIndex);
    int getPointFeatureCount();
    bool usePointFeatureForOptimization(const PointFeatureId &point_feature) const;
    bool hasGoodMappoint(int point_feature_id) const;
    std::pair<int, int> windowSupportStats(int point_feature_id) const;
    bool useLineForOptimization(const LinePerId &line_feature) const;
    void addLineFeature(int frame_count, const LineFeatureFrameMap &lines, double td);
    int getLineFeatureCount() const;
    void triangulateLine(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    Eigen::Matrix<double, Eigen::Dynamic, 4> getLineFeatureVector() const;
    void setLineFeature(const Eigen::Matrix<double, Eigen::Dynamic, 4> &x,
                        int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void removeLineOutliers(const std::map<int, std::set<int>> &outlier_observers);
    void removeLineBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                  Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                  Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void removeLineBack();
    void removeLineFront(int frame_count);
    void removeLineFailures();

    PointFeatureManager &points() { return *point_manager; }
    const PointFeatureManager &points() const { return *point_manager; }
    LineFeatureManager *lines() { return line_manager.get(); }
    const LineFeatureManager *lines() const { return line_manager.get(); }

    Eigen::Matrix3d (&trackingRic())[2] { return ric; }
    const Eigen::Matrix3d (&trackingRic() const)[2] { return ric; }
    VisualTrackingMode &trackingMode() { return visual_tracking_mode; }
    const VisualTrackingMode &trackingMode() const { return visual_tracking_mode; }
    int &activeCameraId() { return active_camera_id; }
    const int &activeCameraId() const { return active_camera_id; }
    std::deque<LocalFrameObs> &localFrames() { return local_frames; }
    const std::deque<LocalFrameObs> &localFrames() const { return local_frames; }
    std::map<int, MappointPtr> &localMappoints() { return local_mappoints; }
    const std::map<int, MappointPtr> &localMappoints() const { return local_mappoints; }
    Eigen::Vector3d (&localPs())[WINDOW_SIZE + 1] { return local_Ps; }
    const Eigen::Vector3d (&localPs() const)[WINDOW_SIZE + 1] { return local_Ps; }
    Eigen::Matrix3d (&localRs())[WINDOW_SIZE + 1] { return local_Rs; }
    const Eigen::Matrix3d (&localRs() const)[WINDOW_SIZE + 1] { return local_Rs; }
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &localTic() { return local_tic; }
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &localTic() const { return local_tic; }
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> &localRic() { return local_ric; }
    const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> &localRic() const { return local_ric; }
    bool &hasLocalWindowState() { return has_local_window_state; }
    const bool &hasLocalWindowState() const { return has_local_window_state; }

  private:
    Eigen::Matrix3d ric[2];
    VisualTrackingMode visual_tracking_mode;
    int active_camera_id;
    std::deque<LocalFrameObs> local_frames;
    std::map<int, MappointPtr> local_mappoints;
    Eigen::Vector3d local_Ps[WINDOW_SIZE + 1];
    Eigen::Matrix3d local_Rs[WINDOW_SIZE + 1];
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> local_tic;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> local_ric;
    bool has_local_window_state;
    bool line_ba_enabled;
    std::unique_ptr<PointFeatureManager> point_manager;
    std::unique_ptr<LineFeatureManager> line_manager;
};

#endif  // VINS_MANAGER_FEATURE_MANAGER_H_
