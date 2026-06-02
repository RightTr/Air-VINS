#ifndef VINS_MANAGER_POINT_FEATURE_MANAGER_H_
#define VINS_MANAGER_POINT_FEATURE_MANAGER_H_

#include <opencv2/core.hpp>

#include "feature_types.h"

class FeatureManager;

class PointFeatureManager
{
  public:
    PointFeatureManager(FeatureManager *owner, Eigen::Matrix3d _Rs[]);

    void clearState();

    std::list<PointFeatureId> pointFeature;
    int last_point_feature_track_num;
    double last_average_parallax;
    int new_point_feature_num;
    int long_point_feature_track_num;
    bool pending_keyframe_next_frame;

  private:
    FeatureManager *owner;
    int getPointFeatureCount();
    bool knownLandmarksOnly() const;
    bool allowDepthInitialization() const;
    bool hasReliablePointDepth(const PointFeatureId &point_feature) const;
    double compensatedParallax2(const PointFeatureId &point_feature, int frame_count);
    void syncMappointFromPointFeature(PointFeatureId &point_feature);
    void markMappointFromPointFeature(PointFeatureId &point_feature);
    void pruneStaleLocalMappoints();
    bool triangulateMappointWorldPoint(const PointFeatureId &point_feature, Eigen::Vector3d &point_w, double &depth) const;
    Eigen::Vector3d pointFeatureDepthToWorldPoint(const PointFeatureId &point_feature) const;
    double worldPointToPointFeatureDepth(const PointFeatureId &point_feature, const Eigen::Vector3d &point_w) const;
    bool projectWorldPointToCamera(const Eigen::Vector3d &point_w, const Eigen::Matrix4d &nextT, int camera_id, Eigen::Vector3d &point_cam) const;
    void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                          const Eigen::Vector2d &point0, const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) const;
    bool shouldUsePointFeatureForPnP(const PointFeatureId &point_feature, bool prefer_good_points) const;
    void collectPointFeaturePnPCorrespondences(int frameCnt, int active_camera_id_, bool prefer_good_points,
                                               Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[],
                                               std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D) const;

    const Eigen::Matrix3d *Rs;
};

#endif  // VINS_MANAGER_POINT_FEATURE_MANAGER_H_
