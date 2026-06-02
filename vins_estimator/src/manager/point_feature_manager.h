#ifndef VINS_MANAGER_POINT_FEATURE_MANAGER_H_
#define VINS_MANAGER_POINT_FEATURE_MANAGER_H_

#include <opencv2/core.hpp>

#include "feature_types.h"

class PointFeatureManager
{
  public:
    explicit PointFeatureManager(Eigen::Matrix3d _Rs[]);

    KeyframeGoodPointRecordList collectGoodKeyframePoints() const;
    void clearState();
    void setWindowState(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void addLocalFrame(int frameCnt, double header,
                       const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                       VisualTrackingMode tracking_mode, int active_camera_id);
    void updateLocalPoints(int frameCnt);
    bool addPointFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r, bool only_good = false);
    void setPointFeatureDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearPointFeatureDepth();
    Eigen::VectorXd getPointFeatureDepthVector();
    void triangulatePointFeature(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    std::vector<ProjectionCandidate> collectProjectionCandidates(const Eigen::Matrix4d &nextT) const;
    void updateMappointDescriptors(const std::vector<int> &ids, const Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
    void initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[], int active_camera_id, bool prefer_good_points = false);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D);
    void removePointFeatureBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                          Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                          Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void removeBack();
    void removeFront(int frame_count);
    void removePointFeatureOutlier(std::set<int> &outlierIndex);

    std::list<PointFeatureId> pointFeature;
    int last_point_feature_track_num;
    double last_average_parallax;
    int new_point_feature_num;
    int long_point_feature_track_num;
    bool pending_keyframe_next_frame;

  private:
    friend class FeatureManager;

    int getPointFeatureCount();
    bool knownLandmarksOnly() const;
    bool allowDepthInitialization() const;
    bool hasReliablePointDepth(const PointFeatureId &point_feature) const;
    bool usePointFeatureForOptimization(const PointFeatureId &point_feature) const;
    bool hasGoodMappoint(int point_feature_id) const;
    double compensatedParallax2(const PointFeatureId &point_feature, int frame_count);
    void syncMappointFromPointFeature(PointFeatureId &point_feature);
    void markMappointFromPointFeature(PointFeatureId &point_feature);
    void pruneStaleLocalMappoints();
    std::pair<int, int> windowSupportStats(int point_feature_id) const;
    bool triangulateMappointWorldPoint(const PointFeatureId &point_feature, Eigen::Vector3d &point_w, double &depth) const;
    Eigen::Vector3d pointFeatureDepthToWorldPoint(const PointFeatureId &point_feature) const;
    double worldPointToPointFeatureDepth(const PointFeatureId &point_feature, const Eigen::Vector3d &point_w) const;
    bool projectWorldPointToCamera(const Eigen::Vector3d &point_w, const Eigen::Matrix4d &nextT, int camera_id, Eigen::Vector3d &point_cam) const;
    void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                          const Eigen::Vector2d &point0, const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) const;
    bool shouldUsePointFeatureForPnP(const PointFeatureId &point_feature, bool prefer_good_points) const;
    void collectPointFeaturePnPCorrespondences(int frameCnt, int active_camera_id, bool prefer_good_points,
                                               Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[],
                                               std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D) const;

    const Eigen::Matrix3d *Rs;
    Eigen::Matrix3d ric[2];
    VisualTrackingMode visual_tracking_mode;
    int active_camera_id;

    std::deque<LocalFrameObs> local_frames;
    std::map<int, MappointPtr> local_mappoints;
    Eigen::Vector3d local_Ps[WINDOW_SIZE + 1];
    Eigen::Matrix3d local_Rs[WINDOW_SIZE + 1];
    std::vector<Eigen::Vector3d> local_tic;
    std::vector<Eigen::Matrix3d> local_ric;
    bool has_local_window_state;
};

#endif  // VINS_MANAGER_POINT_FEATURE_MANAGER_H_
