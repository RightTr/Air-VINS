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
#include "point_feature_manager.h"
#include "line_feature_manager.h"

class FeatureManager
{
  public:
    explicit FeatureManager(Eigen::Matrix3d _Rs[]);

    void setRic(Eigen::Matrix3d _ric[]);
    void setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[]);
    void setVisualTrackingMode(VisualTrackingMode mode);
    VisualTrackingMode getVisualTrackingMode() const;
    void setActiveCameraId(int camera_id);
    int getPointFeatureCount();
    bool usePointFeatureForOptimization(const PointFeatureId &point_feature) const;
    bool useLineForOptimization(const LinePerId &line_feature) const;

    PointFeatureManager &points();
    const PointFeatureManager &points() const;
    LineFeatureManager *lines();
    const LineFeatureManager *lines() const;

  private:
    std::unique_ptr<PointFeatureManager> point_manager;
    std::unique_ptr<LineFeatureManager> line_manager;
};

#endif  // VINS_MANAGER_FEATURE_MANAGER_H_
