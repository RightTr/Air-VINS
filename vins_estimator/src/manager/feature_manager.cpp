#include "feature_manager.h"

FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[])
    : point_manager(std::make_unique<PointFeatureManager>(_Rs)),
      line_manager(LINE_BA ? std::make_unique<LineFeatureManager>() : nullptr)
{
}

void FeatureManager::setRic(Eigen::Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; ++i)
        point_manager->ric[i] = _ric[i];
}

void FeatureManager::setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[])
{
    if (line_manager)
        line_manager->setLineCameraIntrinsics(intrinsics);
}

void FeatureManager::setVisualTrackingMode(VisualTrackingMode mode)
{
    point_manager->visual_tracking_mode = mode;
}

VisualTrackingMode FeatureManager::getVisualTrackingMode() const
{
    return point_manager->visual_tracking_mode;
}

void FeatureManager::setActiveCameraId(int camera_id)
{
    if (camera_id < 0)
        camera_id = 0;
    if (camera_id >= NUM_OF_CAM)
        camera_id = NUM_OF_CAM - 1;
    point_manager->active_camera_id = camera_id;
}

bool FeatureManager::usePointFeatureForOptimization(const PointFeatureId &point_feature) const
{
    return point_manager->usePointFeatureForOptimization(point_feature);
}

int FeatureManager::getPointFeatureCount()
{
    return point_manager->getPointFeatureCount();
}

bool FeatureManager::useLineForOptimization(const LinePerId &line_feature) const
{
    return LINE_BA && line_feature.mapline && line_feature.mapline->IsValid() &&
           line_feature.solve_flag == 1 &&
           line_feature.used_num >= LINE_MIN_OBS &&
           line_feature.start_frame < WINDOW_SIZE - 2 &&
           line_feature.line_3d_world.allFinite();
}

PointFeatureManager &FeatureManager::points()
{
    return *point_manager;
}

const PointFeatureManager &FeatureManager::points() const
{
    return *point_manager;
}

LineFeatureManager *FeatureManager::lines()
{
    return line_manager.get();
}

const LineFeatureManager *FeatureManager::lines() const
{
    return line_manager.get();
}
