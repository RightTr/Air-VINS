#include "point_feature_manager.h"
#include "feature_manager.h"
#include <cmath>
#include <limits>
#include <utility>
#include <ros/console.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "camodocal/camera_models/CameraFactory.h"

using namespace std;
using namespace Eigen;

namespace
{
bool isReliableTriangulatedDepth(double depth)
{
    return depth >= FEATURE_MIN_DEPTH && depth <= FEATURE_MAX_DEPTH;
}

} // namespace

int PointFeatureId::endFrame()
{
    return start_frame + point_feature_frame.size() - 1;
}

PointFeatureManager::PointFeatureManager(FeatureManager *owner, Matrix3d _Rs[])
    : owner(owner), Rs(_Rs)
{
    last_point_feature_track_num = 0;
    last_average_parallax = 0.0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
    pending_keyframe_next_frame = false;
}

bool PointFeatureManager::knownLandmarksOnly() const
{
    return owner->trackingMode() != TRACKING_MODE_STEREO;
}

bool PointFeatureManager::allowDepthInitialization() const
{
    return !knownLandmarksOnly();
}

bool PointFeatureManager::hasReliablePointDepth(const PointFeatureId &point_feature) const
{
    return point_feature.estimated_depth > 0 && point_feature.reliable_depth;
}

void PointFeatureManager::clearState()
{
    pointFeature.clear();
    last_point_feature_track_num = 0;
    last_average_parallax = 0.0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
    pending_keyframe_next_frame = false;
}

int PointFeatureManager::getPointFeatureCount()
{
    int cnt = 0;
    for (auto &it : pointFeature)
    {
        it.used_num = it.point_feature_frame.size();
        if (owner->usePointFeatureForOptimization(it))
            ++cnt;
    }
    return cnt;
}

void PointFeatureManager::syncMappointFromPointFeature(PointFeatureId &point_feature)
{
    auto it = owner->localMappoints().find(point_feature.point_feature_id);
    if (it == owner->localMappoints().end() || !it->second)
        return;

    if (!std::isfinite(point_feature.estimated_depth) || point_feature.estimated_depth <= 0)
        return;

    if (!owner->hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return;

    it->second->SetPosition(pointFeatureDepthToWorldPoint(point_feature));
}

bool PointFeatureManager::triangulateMappointWorldPoint(const PointFeatureId &point_feature, Vector3d &point_w, double &depth) const
{
    point_w.setZero();
    depth = -1.0;

    if (!owner->hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return false;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return false;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(owner->localRic().size()))
        return false;

    if (STEREO && anchor_camera_id == 0 && point_feature.point_feature_frame[0].is_stereo)
    {
        Eigen::Matrix<double, 3, 4> leftPose;
        Eigen::Vector3d t0 = owner->localPs()[anchor_frame] + owner->localRs()[anchor_frame] * owner->localTic()[0];
        Eigen::Matrix3d R0 = owner->localRs()[anchor_frame] * owner->localRic()[0];
        leftPose.leftCols<3>() = R0.transpose();
        leftPose.rightCols<1>() = -R0.transpose() * t0;

        Eigen::Matrix<double, 3, 4> rightPose;
        Eigen::Vector3d t1 = owner->localPs()[anchor_frame] + owner->localRs()[anchor_frame] * owner->localTic()[1];
        Eigen::Matrix3d R1 = owner->localRs()[anchor_frame] * owner->localRic()[1];
        rightPose.leftCols<3>() = R1.transpose();
        rightPose.rightCols<1>() = -R1.transpose() * t1;

        const Eigen::Vector2d point0 = point_feature.point_feature_frame[0].point.head(2);
        const Eigen::Vector2d point1 = point_feature.point_feature_frame[0].pointRight.head(2);
        Eigen::Vector3d point3d;
        triangulatePoint(leftPose, rightPose, point0, point1, point3d);

        const Eigen::Vector3d localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
        depth = localPoint.z();
        if (!isReliableTriangulatedDepth(depth))
            return false;

        point_w = owner->localRs()[anchor_frame] * (owner->localRic()[0] * localPoint + owner->localTic()[0]) + owner->localPs()[anchor_frame];
        if (!point_w.allFinite())
            return false;
    }
    else
    {
        if (point_feature.point_feature_frame.size() < 2)
            return false;

        Eigen::MatrixXd svd_A(2 * point_feature.point_feature_frame.size(), 4);
        int svd_idx = 0;

        for (size_t obs_idx = 0; obs_idx < point_feature.point_feature_frame.size(); ++obs_idx)
        {
            const int frame_idx = point_feature.start_frame + static_cast<int>(obs_idx);
            if (frame_idx < 0 || frame_idx > WINDOW_SIZE)
                return false;

            const PointFeatureFrame &obs = point_feature.point_feature_frame[obs_idx];
            const int camera_id = obs.camera_id;
            if (camera_id < 0 || camera_id >= static_cast<int>(owner->localRic().size()))
                return false;

            const Eigen::Vector3d t = owner->localPs()[frame_idx] + owner->localRs()[frame_idx] * owner->localTic()[camera_id];
            const Eigen::Matrix3d R = owner->localRs()[frame_idx] * owner->localRic()[camera_id];
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;

            const Eigen::Vector3d f = obs.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - P.row(1);
        }

        if (svd_idx < 4)
            return false;

        const Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        if (std::abs(svd_V[3]) < 1e-12)
            return false;

        point_w = svd_V.head<3>() / svd_V[3];
        if (!point_w.allFinite())
            return false;

        const Eigen::Vector3d anchor_t = owner->localPs()[anchor_frame] + owner->localRs()[anchor_frame] * owner->localTic()[anchor_camera_id];
        const Eigen::Matrix3d anchor_R = owner->localRs()[anchor_frame] * owner->localRic()[anchor_camera_id];
        const Eigen::Vector3d anchor_point_cam = anchor_R.transpose() * (point_w - anchor_t);
        depth = anchor_point_cam.z();
        if (!isReliableTriangulatedDepth(depth))
            return false;

        double reproj_sum = 0.0;
        int valid_obs = 0;
        for (size_t obs_idx = 0; obs_idx < point_feature.point_feature_frame.size(); ++obs_idx)
        {
            const int frame_idx = point_feature.start_frame + static_cast<int>(obs_idx);
            const PointFeatureFrame &obs = point_feature.point_feature_frame[obs_idx];
            const Eigen::Vector3d t = owner->localPs()[frame_idx] + owner->localRs()[frame_idx] * owner->localTic()[obs.camera_id];
            const Eigen::Matrix3d R = owner->localRs()[frame_idx] * owner->localRic()[obs.camera_id];
            const Eigen::Vector3d point_cam = R.transpose() * (point_w - t);
            if (!point_cam.allFinite() || point_cam.z() <= 0.0)
                return false;
            const Eigen::Vector2d proj = point_cam.head<2>() / point_cam.z();
            reproj_sum += (proj - obs.point.head<2>()).norm();
            ++valid_obs;
        }

        if (valid_obs < 2)
            return false;
        if (reproj_sum / valid_obs > 0.01)
            return false;
    }

    return true;
}

void PointFeatureManager::markMappointFromPointFeature(PointFeatureId &point_feature)
{
    auto &mappoint = owner->localMappoints()[point_feature.point_feature_id];
    if (!mappoint)
        mappoint = std::make_shared<Mappoint>(point_feature.point_feature_id);

    mappoint->ClearObversers();
    mappoint->SetTrackCount(static_cast<int>(point_feature.point_feature_frame.size()));
    mappoint->ResetStaleCount();
    for (int i = 0; i < static_cast<int>(point_feature.point_feature_frame.size()); ++i)
        mappoint->AddObverser(point_feature.start_frame + i, i);

    if (point_feature.solve_flag == 2)
    {
        mappoint->SetBad();
        return;
    }

    if (mappoint->IsUnTriangulated())
    {
        if (mappoint->ObverserNum() > 2)
        {
            Vector3d triangulated_point = Vector3d::Zero();
            double triangulated_depth = -1.0;
            if (triangulateMappointWorldPoint(point_feature, triangulated_point, triangulated_depth))
            {
                mappoint->SetPosition(triangulated_point);
                mappoint->SetGood();
                point_feature.estimated_depth = triangulated_depth;
                point_feature.reliable_depth = true;
                point_feature.solve_flag = 1;
                point_feature.depth_fail_count = 0;
            }
            else
            {
                mappoint->SetUnTriangulated();
            }
        }
        else
        {
            mappoint->SetUnTriangulated();
        }
    }
    else if (mappoint->IsGood())
    {
        if (std::isfinite(point_feature.estimated_depth) && point_feature.estimated_depth > 0)
            syncMappointFromPointFeature(point_feature);
        else if (mappoint->HasPosition())
        {
            const double depth = worldPointToPointFeatureDepth(point_feature, mappoint->GetPosition());
            if (isReliableTriangulatedDepth(depth))
            {
                point_feature.estimated_depth = depth;
                point_feature.reliable_depth = true;
                point_feature.solve_flag = 1;
                point_feature.depth_fail_count = 0;
            }
        }
    }
    else if (!mappoint->IsBad())
    {
        mappoint->SetUnTriangulated();
    }
}

bool PointFeatureManager::projectWorldPointToCamera(const Vector3d &point_w, const Eigen::Matrix4d &nextT, int camera_id, Vector3d &point_cam) const
{
    if (!owner->hasLocalWindowState())
        return false;
    if (camera_id < 0 || camera_id >= static_cast<int>(owner->localRic().size()))
        return false;
    if (!point_w.allFinite() || !nextT.allFinite())
        return false;

    const Eigen::Matrix3d Rwb = nextT.block<3, 3>(0, 0);
    const Eigen::Vector3d Pwb = nextT.block<3, 1>(0, 3);
    const Eigen::Vector3d point_in_body = Rwb.transpose() * (point_w - Pwb);
    point_cam = owner->localRic()[camera_id].transpose() * (point_in_body - owner->localTic()[camera_id]);
    return point_cam.allFinite() && point_cam.z() > 0.0;
}

void PointFeatureManager::pruneStaleLocalMappoints()
{
    std::set<int> alive_point_feature_ids;
    for (const auto &it_per_id : pointFeature)
        alive_point_feature_ids.insert(it_per_id.point_feature_id);

    for (auto it = owner->localMappoints().begin(); it != owner->localMappoints().end();)
    {
        const bool alive = alive_point_feature_ids.find(it->first) != alive_point_feature_ids.end();
        if (!it->second)
        {
            it = owner->localMappoints().erase(it);
            continue;
        }

        if (alive)
        {
            it->second->ResetStaleCount();
            ++it;
            continue;
        }

        it->second->IncreaseStaleCount();
        const auto support_stats = owner->windowSupportStats(it->first);
        const bool strong_window_support = support_stats.first >= 2;
        const int keep_good_limit = strong_window_support ? 6 : 3;
        const int keep_non_good_limit = strong_window_support ? 4 : 2;
        const bool keep_temporarily = it->second->IsGood() && it->second->StaleCount() < keep_good_limit;
        const bool keep_non_good_temporarily = !it->second->IsGood() && it->second->StaleCount() < keep_non_good_limit;
        if (keep_temporarily || keep_non_good_temporarily)
        {
            ++it;
            continue;
        }

        it = owner->localMappoints().erase(it);
    }
}

Vector3d PointFeatureManager::pointFeatureDepthToWorldPoint(const PointFeatureId &point_feature) const
{
    if (!owner->hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return Vector3d::Zero();

    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    const int anchor_frame = point_feature.start_frame;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(owner->localRic().size()))
        return Vector3d::Zero();
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return Vector3d::Zero();
    const Vector3d pts_in_cam = owner->localRic()[anchor_camera_id] *
                                (point_feature.estimated_depth * point_feature.point_feature_frame[0].point) +
                                owner->localTic()[anchor_camera_id];
    return owner->localRs()[anchor_frame] * pts_in_cam + owner->localPs()[anchor_frame];
}

double PointFeatureManager::worldPointToPointFeatureDepth(const PointFeatureId &point_feature, const Vector3d &point_w) const
{
    if (!owner->hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return -1.0;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return -1.0;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(owner->localRic().size()))
        return -1.0;

    const Eigen::Vector3d anchor_t = owner->localPs()[anchor_frame] + owner->localRs()[anchor_frame] * owner->localTic()[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = owner->localRs()[anchor_frame] * owner->localRic()[anchor_camera_id];
    const Eigen::Vector3d point_cam = anchor_R.transpose() * (point_w - anchor_t);
    return point_cam.z();
}

void PointFeatureManager::triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                        const Eigen::Vector2d &point0, const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) const
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool PointFeatureManager::shouldUsePointFeatureForPnP(const PointFeatureId &point_feature, bool prefer_good_points) const
{
    return prefer_good_points
               ? (owner->hasGoodMappoint(point_feature.point_feature_id) && hasReliablePointDepth(point_feature))
               : (knownLandmarksOnly() ? hasReliablePointDepth(point_feature) : point_feature.estimated_depth > 0);
}

void PointFeatureManager::collectPointFeaturePnPCorrespondences(int frameCnt, int active_camera_id_, bool prefer_good_points,
                                                                Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d window_ric[],
                                                                vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D) const
{
    for (const auto &point_feature : pointFeature)
    {
        if (!shouldUsePointFeatureForPnP(point_feature, prefer_good_points))
            continue;

        const int obs_index = frameCnt - point_feature.start_frame;
        if (obs_index < 0 || static_cast<int>(point_feature.point_feature_frame.size()) < obs_index + 1)
            continue;

        const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
        const int obs_camera_id = point_feature.point_feature_frame[obs_index].camera_id;
        if (obs_camera_id != active_camera_id_)
            continue;

        const Vector3d point_in_cam = window_ric[anchor_camera_id] * (point_feature.point_feature_frame[0].point * point_feature.estimated_depth) + tic[anchor_camera_id];
        const Vector3d point_in_world = Rs[point_feature.start_frame] * point_in_cam + Ps[point_feature.start_frame];

        pts3D.emplace_back(point_in_world.x(), point_in_world.y(), point_in_world.z());
        pts2D.emplace_back(point_feature.point_feature_frame[obs_index].point.x(), point_feature.point_feature_frame[obs_index].point.y());
    }
}


// point/window lifecycle is now owned by FeatureManager


double PointFeatureManager::compensatedParallax2(const PointFeatureId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const PointFeatureFrame &frame_i = it_per_id.point_feature_frame[frame_count - 2 - it_per_id.start_frame];
    const PointFeatureFrame &frame_j = it_per_id.point_feature_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = owner->trackingRic()[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * owner->trackingRic()[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
