#ifndef VINS_MANAGER_FEATURE_MANAGER_H_
#define VINS_MANAGER_FEATURE_MANAGER_H

#include <opencv2/core.hpp>
#include <unordered_map>

#include "feature_types.h"

class PointFeatureManager
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointFeatureManager();

    // Lifecycle
    void clearState();
    void addLocalFrame(int frameCnt, double header,
                       const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                       VisualTrackingMode tracking_mode, int active_camera_id);
    bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void updateLocalPoints(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                           Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);

    // Optimization
    int getFeatureCount();
    bool useFeatureForOptimization(const PointFeatureId &point_feature) const;
    Eigen::VectorXd getDepthVector();
    void setDepth(const Eigen::VectorXd &x, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                  Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void clearDepth();

    // Sliding window
    void removeBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                              Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                              Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void removeBack();
    void removeFront(int frame_count);
    void removeFailures();
    void removeOutlier(std::set<int> &outlierIndex);

    // PnP
    void initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[],
                            int active_camera_id, bool prefer_good_points = false);
    void updateMappointDescriptors(const std::vector<int> &ids, const Eigen::Matrix<float, 259, Eigen::Dynamic> &features);

    // Queries
    KeyframeGoodPointRecordList collectGoodKeyframePoints() const;
    int countGoodLocalMappoints() const;
    int countCurrentStereoObservations() const;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r, bool only_good = false);

    std::list<PointFeatureId> pointFeature;
    int last_point_feature_track_num;
    double last_average_parallax;
    int new_point_feature_num;
    int long_point_feature_track_num;

  private:
    std::deque<LocalFrameObs> local_frames;
    std::unordered_map<int, std::list<PointFeatureId>::iterator> pointFeatureIndex;
    std::unordered_map<int, int> windowSupportCount;
    std::unordered_map<int, int> windowSupportLastFrame;
    std::map<int, MappointPtr> local_mappoints;
    std::pair<int, int> windowSupportStats(int point_feature_id) const;
    void syncMappointFromPointFeature(PointFeatureId &point_feature,
                                      Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                      Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]);
    void pruneInactiveLocalMappoints();
    bool triangulateMappointWorldPoint(const PointFeatureId &point_feature,
                                       Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                       Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[],
                                       Eigen::Vector3d &point_w, double &depth) const;
    double worldPointToPointFeatureDepth(const PointFeatureId &point_feature,
                                         const Eigen::Vector3d &point_w,
                                         Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                         Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[]) const;
    void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                          const Eigen::Vector2d &point0, const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) const;
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D);
};

#endif  // VINS_MANAGER_POINT_FEATURE_MANAGER_H_
