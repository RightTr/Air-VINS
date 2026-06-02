#ifndef VINS_MANAGER_LINE_FEATURE_MANAGER_H_
#define VINS_MANAGER_LINE_FEATURE_MANAGER_H_
#include "feature_types.h"

class LineFeatureManager
{
  public:
    LineFeatureManager();

    void clearState();
    void setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[]);
    void addLineFeature(int frame_count, const LineFeatureFrameMap &lines, double td);
    int getLineFeatureCount() const;
    void triangulateLine(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    Eigen::Matrix<double, Eigen::Dynamic, 4> getLineFeatureVector() const;
    void setLineFeature(const Eigen::Matrix<double, Eigen::Dynamic, 4> &x,
                        int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void removeLineOutliers(const std::map<int, std::set<int>> &outlier_observers);
    void removeBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                              Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                              Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void removeBack();
    void removeFront(int frame_count);
    void removeFailures();

    std::list<LinePerId> line_feature;

  private:
    bool useLineForOptimization(const LinePerId &line_per_id) const;

    Eigen::Vector4d line_camera_intrinsics[2];
};

#endif  // VINS_MANAGER_LINE_FEATURE_MANAGER_H_
