#include "feature_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/console.h>

using namespace std;
using namespace Eigen;

namespace
{
bool isReliableTriangulatedDepth(double depth)
{
    return depth >= FEATURE_MIN_DEPTH && depth <= FEATURE_MAX_DEPTH;
}

bool knownLandmarksOnly(const FeatureManager &fm)
{
    return fm.getVisualTrackingMode() != TRACKING_MODE_STEREO;
}

bool allowDepthInitialization(const FeatureManager &fm)
{
    return !knownLandmarksOnly(fm);
}

bool hasReliablePointDepth(const PointFeatureId &point_feature)
{
    return point_feature.estimated_depth > 0 && point_feature.reliable_depth;
}

void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                      const Eigen::Vector2d &point0, const Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

Vector3d pointFeatureDepthToWorldPoint(const FeatureManager &fm, const PointFeatureId &point_feature)
{
    if (!fm.hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return Vector3d::Zero();

    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    const int anchor_frame = point_feature.start_frame;
    if (anchor_camera_id < 0 || anchor_camera_id >= NUM_OF_CAM)
        return Vector3d::Zero();
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return Vector3d::Zero();
    const Vector3d pts_in_cam = fm.localRic()[anchor_camera_id] *
                                (point_feature.estimated_depth * point_feature.point_feature_frame[0].point) +
                                fm.localTic()[anchor_camera_id];
    return fm.localRs()[anchor_frame] * pts_in_cam + fm.localPs()[anchor_frame];
}

double worldPointToPointFeatureDepth(const FeatureManager &fm, const PointFeatureId &point_feature, const Vector3d &point_w)
{
    if (!fm.hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return -1.0;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return -1.0;
    if (anchor_camera_id < 0 || anchor_camera_id >= NUM_OF_CAM)
        return -1.0;

    const Eigen::Vector3d anchor_t = fm.localPs()[anchor_frame] + fm.localRs()[anchor_frame] * fm.localTic()[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = fm.localRs()[anchor_frame] * fm.localRic()[anchor_camera_id];
    const Eigen::Vector3d point_cam = anchor_R.transpose() * (point_w - anchor_t);
    return point_cam.z();
}

bool triangulateMappointWorldPoint(const FeatureManager &fm, const PointFeatureId &point_feature, Eigen::Vector3d &point_w, double &depth)
{
    point_w.setZero();
    depth = -1.0;

    if (!fm.hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return false;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return false;
    if (anchor_camera_id < 0 || anchor_camera_id >= NUM_OF_CAM)
        return false;

    if (STEREO && anchor_camera_id == 0 && point_feature.point_feature_frame[0].is_stereo)
    {
        Eigen::Matrix<double, 3, 4> leftPose;
        Eigen::Vector3d t0 = fm.localPs()[anchor_frame] + fm.localRs()[anchor_frame] * fm.localTic()[0];
        Eigen::Matrix3d R0 = fm.localRs()[anchor_frame] * fm.localRic()[0];
        leftPose.leftCols<3>() = R0.transpose();
        leftPose.rightCols<1>() = -R0.transpose() * t0;

        Eigen::Matrix<double, 3, 4> rightPose;
        Eigen::Vector3d t1 = fm.localPs()[anchor_frame] + fm.localRs()[anchor_frame] * fm.localTic()[1];
        Eigen::Matrix3d R1 = fm.localRs()[anchor_frame] * fm.localRic()[1];
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

        point_w = fm.localRs()[anchor_frame] * (fm.localRic()[0] * localPoint + fm.localTic()[0]) + fm.localPs()[anchor_frame];
        return point_w.allFinite();
    }

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
        if (camera_id < 0 || camera_id >= NUM_OF_CAM)
            return false;

        const Eigen::Vector3d t = fm.localPs()[frame_idx] + fm.localRs()[frame_idx] * fm.localTic()[camera_id];
        const Eigen::Matrix3d R = fm.localRs()[frame_idx] * fm.localRic()[camera_id];
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

    const Eigen::Vector3d anchor_t = fm.localPs()[anchor_frame] + fm.localRs()[anchor_frame] * fm.localTic()[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = fm.localRs()[anchor_frame] * fm.localRic()[anchor_camera_id];
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
        const Eigen::Vector3d t = fm.localPs()[frame_idx] + fm.localRs()[frame_idx] * fm.localTic()[obs.camera_id];
        const Eigen::Matrix3d R = fm.localRs()[frame_idx] * fm.localRic()[obs.camera_id];
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
    return true;
}

bool projectWorldPointToCamera(const FeatureManager &fm, const Vector3d &point_w, const Eigen::Matrix4d &nextT, int camera_id, Vector3d &point_cam)
{
    if (!fm.hasLocalWindowState())
        return false;
    if (camera_id < 0 || camera_id >= NUM_OF_CAM)
        return false;
    if (!point_w.allFinite() || !nextT.allFinite())
        return false;

    const Eigen::Matrix3d Rwb = nextT.block<3, 3>(0, 0);
    const Eigen::Vector3d Pwb = nextT.block<3, 1>(0, 3);
    const Eigen::Vector3d point_in_body = Rwb.transpose() * (point_w - Pwb);
    point_cam = fm.localRic()[camera_id].transpose() * (point_in_body - fm.localTic()[camera_id]);
    return point_cam.allFinite() && point_cam.z() > 0.0;
}

void syncMappointFromPointFeature(FeatureManager &fm, PointFeatureId &point_feature)
{
    auto it = fm.localMappoints().find(point_feature.point_feature_id);
    if (it == fm.localMappoints().end() || !it->second)
        return;
    if (!std::isfinite(point_feature.estimated_depth) || point_feature.estimated_depth <= 0)
        return;
    if (!fm.hasLocalWindowState() || point_feature.point_feature_frame.empty())
        return;
    it->second->SetPosition(pointFeatureDepthToWorldPoint(fm, point_feature));
}

void markMappointFromPointFeature(FeatureManager &fm, PointFeatureId &point_feature)
{
    auto &mappoint = fm.localMappoints()[point_feature.point_feature_id];
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
            if (triangulateMappointWorldPoint(fm, point_feature, triangulated_point, triangulated_depth))
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
            syncMappointFromPointFeature(fm, point_feature);
        else if (mappoint->HasPosition())
        {
            const double depth = worldPointToPointFeatureDepth(fm, point_feature, mappoint->GetPosition());
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

void pruneStaleLocalMappoints(FeatureManager &fm)
{
    std::set<int> alive_point_feature_ids;
    for (const auto &it_per_id : fm.points().pointFeature)
        alive_point_feature_ids.insert(it_per_id.point_feature_id);

    for (auto it = fm.localMappoints().begin(); it != fm.localMappoints().end();)
    {
        const bool alive = alive_point_feature_ids.find(it->first) != alive_point_feature_ids.end();
        if (!it->second)
        {
            it = fm.localMappoints().erase(it);
            continue;
        }

        if (alive)
        {
            it->second->ResetStaleCount();
            ++it;
            continue;
        }

        it->second->IncreaseStaleCount();
        const auto support_stats = fm.windowSupportStats(it->first);
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

        it = fm.localMappoints().erase(it);
    }
}

bool shouldUsePointFeatureForPnP(const FeatureManager &fm, const PointFeatureId &point_feature, bool prefer_good_points)
{
    return prefer_good_points
               ? (fm.hasGoodMappoint(point_feature.point_feature_id) && hasReliablePointDepth(point_feature))
               : (knownLandmarksOnly(fm) ? hasReliablePointDepth(point_feature) : point_feature.estimated_depth > 0);
}

void collectPointFeaturePnPCorrespondences(const FeatureManager &fm, int frameCnt, int active_camera_id_, bool prefer_good_points,
                                           Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d window_ric[],
                                           vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    for (const auto &point_feature : fm.points().pointFeature)
    {
        if (!shouldUsePointFeatureForPnP(fm, point_feature, prefer_good_points))
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
} // namespace

FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[])
    : visual_tracking_mode(TRACKING_MODE_STEREO),
      active_camera_id(0),
      has_local_window_state(false),
      line_ba_enabled(false),
      point_manager(std::make_unique<PointFeatureManager>(this, _Rs)),
      line_manager(std::make_unique<LineFeatureManager>(this))
{
    for (int i = 0; i < 2; ++i)
        ric[i].setIdentity();
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        local_Ps[i].setZero();
        local_Rs[i].setIdentity();
    }
}

void FeatureManager::clearState()
{
    for (int i = 0; i < 2; ++i)
        ric[i].setIdentity();
    visual_tracking_mode = TRACKING_MODE_STEREO;
    active_camera_id = 0;
    has_local_window_state = false;
    local_frames.clear();
    local_mappoints.clear();
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        local_Ps[i].setZero();
        local_Rs[i].setIdentity();
    }
    local_tic.assign(NUM_OF_CAM, Eigen::Vector3d::Zero());
    local_ric.assign(NUM_OF_CAM, Eigen::Matrix3d::Identity());

    point_manager->clearState();
    line_manager->clearState();
}

void FeatureManager::setRic(Eigen::Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; ++i)
        ric[i] = _ric[i];
}

void FeatureManager::setLineBA(bool enabled)
{
    line_ba_enabled = enabled;
    if (!enabled)
        line_manager->clearState();
}

void FeatureManager::setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[])
{
    line_manager->setLineCameraIntrinsics(intrinsics);
}

void FeatureManager::setVisualTrackingMode(VisualTrackingMode mode)
{
    visual_tracking_mode = mode;
}

VisualTrackingMode FeatureManager::getVisualTrackingMode() const
{
    return visual_tracking_mode;
}

void FeatureManager::setActiveCameraId(int camera_id)
{
    if (camera_id < 0)
        camera_id = 0;
    if (camera_id >= NUM_OF_CAM)
        camera_id = NUM_OF_CAM - 1;
    active_camera_id = camera_id;
}

void FeatureManager::setWindowState(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    if (static_cast<int>(local_tic.size()) != NUM_OF_CAM)
    {
        local_tic.assign(NUM_OF_CAM, Eigen::Vector3d::Zero());
        local_ric.assign(NUM_OF_CAM, Eigen::Matrix3d::Identity());
    }
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        local_Ps[i] = Ps[i];
        local_Rs[i] = Rs[i];
    }
    for (int i = 0; i < NUM_OF_CAM; ++i)
    {
        local_tic[i] = tic[i];
        local_ric[i] = window_ric[i];
    }
    has_local_window_state = true;
}

void FeatureManager::addLocalFrame(int frameCnt, double header,
                                   const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image)
{
    local_frames.emplace_back(frameCnt, header, static_cast<int>(visual_tracking_mode), active_camera_id);
    LocalFrameObs &local_frame = local_frames.back();
    for (const auto &id_pts : image)
    {
        if (id_pts.second.empty())
            continue;
        local_frame.addObservation(id_pts.first, id_pts.second[0].first, id_pts.second[0].second);
    }
    while (local_frames.size() > WINDOW_SIZE + 1)
        local_frames.pop_front();
}

void FeatureManager::updateLocalPoints(int frameCnt)
{
    (void)frameCnt;
    int untriangulated_count = 0;
    int good_count = 0;
    int bad_count = 0;
    int supported_good_count = 0;
    for (auto &it_per_id : point_manager->pointFeature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.point_feature_frame.size());
        markMappointFromPointFeature(*this, it_per_id);

        auto mpt_it = local_mappoints.find(it_per_id.point_feature_id);
        if (mpt_it == local_mappoints.end() || !mpt_it->second)
            continue;

        const auto support_stats = windowSupportStats(it_per_id.point_feature_id);
        mpt_it->second->SetWindowSupport(support_stats.first, support_stats.second);

        if (mpt_it->second->IsUnTriangulated())
            ++untriangulated_count;
        else if (mpt_it->second->IsGood())
        {
            ++good_count;
            if (mpt_it->second->WindowSupportCount() >= 2)
                ++supported_good_count;
        }
        else if (mpt_it->second->IsBad())
            ++bad_count;
    }
    pruneStaleLocalMappoints(*this);
}

bool FeatureManager::addPointFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input pointFeature: %d", (int)image.size());
    ROS_DEBUG("num of pointFeature: %d", getPointFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    int last_point_feature_track_num = 0;
    double last_average_parallax = 0;
    int new_point_feature_num = 0;
    int long_point_feature_track_num = 0;
    for (auto &id_pts : image)
    {
        PointFeatureFrame f_per_fra(id_pts.second[0].second, td, id_pts.second[0].first);
        if (id_pts.second.size() == 2 && id_pts.second[0].first == 0 && id_pts.second[1].first == 1)
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
        }

        int point_feature_id = id_pts.first;
        auto it = find_if(point_manager->pointFeature.begin(), point_manager->pointFeature.end(), [point_feature_id](const PointFeatureId &it)
                          { return it.point_feature_id == point_feature_id; });

        if (it == point_manager->pointFeature.end())
        {
            if (knownLandmarksOnly(*this))
                continue;
            point_manager->pointFeature.push_back(PointFeatureId(point_feature_id, frame_count));
            point_manager->pointFeature.back().point_feature_frame.push_back(f_per_fra);
            new_point_feature_num++;
        }
        else if (it->point_feature_id == point_feature_id)
        {
            it->point_feature_frame.push_back(f_per_fra);
            last_point_feature_track_num++;
            if (it->point_feature_frame.size() >= 4)
                long_point_feature_track_num++;
        }
    }

    point_manager->last_point_feature_track_num = last_point_feature_track_num;
    point_manager->last_average_parallax = 0.0;
    point_manager->new_point_feature_num = new_point_feature_num;
    point_manager->long_point_feature_track_num = long_point_feature_track_num;

    if (frame_count < 2 || last_point_feature_track_num < 20 || long_point_feature_track_num < 40 || new_point_feature_num > 0.5 * last_point_feature_track_num)
        return true;

    for (auto &it_per_id : point_manager->pointFeature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.point_feature_frame.size()) - 1 >= frame_count - 1)
        {
            const PointFeatureFrame &frame_i = it_per_id.point_feature_frame[frame_count - 2 - it_per_id.start_frame];
            const PointFeatureFrame &frame_j = it_per_id.point_feature_frame[frame_count - 1 - it_per_id.start_frame];
            double ans = 0;
            Vector3d p_j = frame_j.point;
            double u_j = p_j(0);
            double v_j = p_j(1);
            Vector3d p_i = frame_i.point;
            Vector3d p_i_comp = p_i;
            double dep_i = p_i(2);
            double u_i = p_i(0) / dep_i;
            double v_i = p_i(1) / dep_i;
            double du = u_i - u_j, dv = v_i - v_j;
            ans = std::sqrt(du * du + dv * dv);
            parallax_sum += ans;
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        point_manager->last_point_feature_track_num = last_point_feature_track_num;
        point_manager->last_average_parallax = last_average_parallax;
        point_manager->new_point_feature_num = new_point_feature_num;
        point_manager->long_point_feature_track_num = long_point_feature_track_num;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

KeyframeGoodPointRecordList FeatureManager::collectGoodKeyframePoints() const
{
    KeyframeGoodPointRecordList records;
    records.reserve(std::min<size_t>(local_mappoints.size(), 1024));
    for (const auto &it_per_id : point_manager->pointFeature)
    {
        if (!usePointFeatureForOptimization(it_per_id))
            continue;
        if (it_per_id.solve_flag != 1 || it_per_id.point_feature_frame.empty())
            continue;
        auto mpt_it = local_mappoints.find(it_per_id.point_feature_id);
        if (mpt_it == local_mappoints.end() || !mpt_it->second)
            continue;
        const auto &mappoint = mpt_it->second;
        if (!mappoint->IsGood() || !mappoint->HasPosition() || !mappoint->HasDescriptor())
            continue;
        const auto &obs = it_per_id.point_feature_frame.back();
        KeyframeGoodPointRecord record;
        record.point_feature_id = it_per_id.point_feature_id;
        record.point_w = mappoint->GetPosition();
        record.point_uv = obs.uv;
        record.point_norm = obs.point.head<2>();
        record.descriptor = mappoint->GetDescriptor().segment<256>(3);
        records.push_back(record);
    }
    return records;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r, bool only_good)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto &it : point_manager->pointFeature)
    {
        if (only_good && !hasGoodMappoint(it.point_feature_id))
            continue;
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
            if (it.point_feature_frame[idx_l].camera_id != it.point_feature_frame[idx_r].camera_id)
                continue;
            a = it.point_feature_frame[idx_l].point;
            b = it.point_feature_frame[idx_r].point;
            corres.push_back(std::make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setPointFeatureDepth(const Eigen::VectorXd &x)
{
    int point_feature_index = -1;
    for (auto &it_per_id : point_manager->pointFeature)
    {
        it_per_id.used_num = it_per_id.point_feature_frame.size();
        if (!usePointFeatureForOptimization(it_per_id))
            continue;

        const double optimized_inv_depth = x(++point_feature_index);
        const double optimized_depth = 1.0 / optimized_inv_depth;
        const bool depth_valid = std::isfinite(optimized_depth) &&
                                 optimized_depth >= FEATURE_MIN_DEPTH &&
                                 optimized_depth <= FEATURE_MAX_DEPTH;
        if (depth_valid)
        {
            it_per_id.estimated_depth = optimized_depth;
            it_per_id.reliable_depth = true;
            it_per_id.solve_flag = 1;
            it_per_id.depth_fail_count = 0;
        }
        else
        {
            it_per_id.depth_fail_count++;
            const double fallback_depth = (std::isfinite(it_per_id.estimated_depth) && it_per_id.estimated_depth > 0)
                                              ? std::min(std::max(it_per_id.estimated_depth, FEATURE_MIN_DEPTH), FEATURE_MAX_DEPTH)
                                              : INIT_DEPTH;
            it_per_id.estimated_depth = fallback_depth;
            it_per_id.reliable_depth = false;
            if (it_per_id.depth_fail_count >= 3)
            {
                it_per_id.solve_flag = 2;
                auto mpt_it = local_mappoints.find(it_per_id.point_feature_id);
                if (mpt_it != local_mappoints.end() && mpt_it->second)
                    mpt_it->second->SetBad();
            }
            else
            {
                it_per_id.solve_flag = 1;
            }
        }

        if (it_per_id.solve_flag == 1)
            syncMappointFromPointFeature(*this, it_per_id);
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = point_manager->pointFeature.begin(), it_next = point_manager->pointFeature.begin(); it != point_manager->pointFeature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            point_manager->pointFeature.erase(it);
    }
}

void FeatureManager::clearPointFeatureDepth()
{
    for (auto &it_per_id : point_manager->pointFeature)
    {
        it_per_id.estimated_depth = -1;
        it_per_id.reliable_depth = false;
        it_per_id.depth_fail_count = 0;
    }
}

Eigen::VectorXd FeatureManager::getPointFeatureDepthVector()
{
    Eigen::VectorXd dep_vec(getPointFeatureCount());
    int point_feature_index = -1;
    for (auto &it_per_id : point_manager->pointFeature)
    {
        it_per_id.used_num = it_per_id.point_feature_frame.size();
        if (!usePointFeatureForOptimization(it_per_id))
            continue;
        dep_vec(++point_feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
}

void FeatureManager::triangulatePointFeature(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    (void)frameCnt;
    (void)Ps;
    (void)Rs;
    (void)tic;
    (void)window_ric;
    if (!allowDepthInitialization(*this))
        return;

    for (auto &it_per_id : point_manager->pointFeature)
    {
        auto mpt_it = local_mappoints.find(it_per_id.point_feature_id);
        MappointPtr mappoint = (mpt_it != local_mappoints.end()) ? mpt_it->second : nullptr;

        if (mappoint && mappoint->HasPosition())
        {
            const double depth = worldPointToPointFeatureDepth(*this, it_per_id, mappoint->GetPosition());
            if (isReliableTriangulatedDepth(depth))
            {
                it_per_id.estimated_depth = depth;
                it_per_id.reliable_depth = true;
                it_per_id.solve_flag = 1;
                it_per_id.depth_fail_count = 0;
            }
            continue;
        }

        Vector3d triangulated_point = Vector3d::Zero();
        double triangulated_depth = -1.0;
        if (triangulateMappointWorldPoint(*this, it_per_id, triangulated_point, triangulated_depth))
        {
            if (!mappoint)
            {
                mappoint = std::make_shared<Mappoint>(it_per_id.point_feature_id);
                local_mappoints[it_per_id.point_feature_id] = mappoint;
            }
            mappoint->SetPosition(triangulated_point);
            mappoint->SetGood();
            it_per_id.estimated_depth = triangulated_depth;
            it_per_id.reliable_depth = true;
            it_per_id.solve_flag = 1;
            it_per_id.depth_fail_count = 0;
        }
    }
}

std::vector<ProjectionCandidate> FeatureManager::collectProjectionCandidates(const Eigen::Matrix4d &nextT) const
{
    std::vector<ProjectionCandidate> candidates;
    if (!has_local_window_state)
        return candidates;

    const int camera_id = active_camera_id;
    if (camera_id < 0 || camera_id >= static_cast<int>(local_ric.size()))
        return candidates;

    candidates.reserve(std::min<size_t>(local_mappoints.size(), 512));
    for (const auto &item : local_mappoints)
    {
        const auto &mappoint = item.second;
        if (!mappoint || !mappoint->IsGood() || !mappoint->HasPosition() || !mappoint->HasDescriptor())
            continue;

        const auto support_stats = windowSupportStats(item.first);
        ProjectionCandidate candidate;
        candidate.point_feature_id = item.first;
        candidate.camera_id = camera_id;
        candidate.track_count = mappoint->TrackCount();
        candidate.observer_count = mappoint->ObverserNum();
        candidate.window_support_count = support_stats.first;
        candidate.last_support_frame_id = support_stats.second;
        candidate.descriptor = mappoint->GetDescriptor().segment<256>(3);

        Vector3d point_cam = Vector3d::Zero();
        if (!projectWorldPointToCamera(*this, mappoint->GetPosition(), nextT, camera_id, point_cam))
            continue;

        candidate.point_cam = point_cam;
        candidates.push_back(candidate);
    }

    std::sort(candidates.begin(), candidates.end(), [](const ProjectionCandidate &a, const ProjectionCandidate &b) {
        if (a.window_support_count != b.window_support_count)
            return a.window_support_count > b.window_support_count;
        if (a.track_count != b.track_count)
            return a.track_count > b.track_count;
        if (a.observer_count != b.observer_count)
            return a.observer_count > b.observer_count;
        if (a.last_support_frame_id != b.last_support_frame_id)
            return a.last_support_frame_id > b.last_support_frame_id;
        return a.point_feature_id < b.point_feature_id;
    });

    if (candidates.size() > 512)
        candidates.resize(512);
    return candidates;
}

void FeatureManager::updateMappointDescriptors(const std::vector<int> &ids, const Eigen::Matrix<float, 259, Eigen::Dynamic> &features)
{
    if (features.rows() != 259 || features.cols() <= 0 || ids.empty())
        return;
    const int count = std::min<int>(static_cast<int>(ids.size()), static_cast<int>(features.cols()));
    for (int i = 0; i < count; ++i)
    {
        const int point_feature_id = ids[i];
        auto it = local_mappoints.find(point_feature_id);
        if (it == local_mappoints.end() || !it->second)
            continue;
        if (!features.col(i).allFinite())
            continue;
        it->second->SetDescriptor(features.col(i));
    }
}

void FeatureManager::initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[], int active_camera_id_, bool prefer_good_points)
{
    if (frameCnt <= 0)
        return;

    vector<cv::Point2f> pts2D;
    vector<cv::Point3f> pts3D;
    collectPointFeaturePnPCorrespondences(*this, frameCnt, active_camera_id_, prefer_good_points, Ps, Rs, tic, window_ric, pts2D, pts3D);
    if (prefer_good_points && pts3D.size() < 6)
        return;

    Eigen::Matrix3d RCam;
    Eigen::Vector3d PCam;
    RCam = Rs[frameCnt - 1] * window_ric[active_camera_id_];
    PCam = Rs[frameCnt - 1] * tic[active_camera_id_] + Ps[frameCnt - 1];

    if (solvePoseByPnP(RCam, PCam, pts2D, pts3D))
    {
        Rs[frameCnt] = RCam * window_ric[active_camera_id_].transpose();
        Ps[frameCnt] = -RCam * window_ric[active_camera_id_].transpose() * tic[active_camera_id_] + PCam;
    }
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                                    std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d P;

    R = R_initial.inverse();
    P = -(R * P_initial);

    if (int(pts2D.size()) < 4)
    {
        printf("pointFeature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    if (!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    R_initial = R_pnp.transpose();
    P_initial = R_initial * (-T_pnp);
    return true;
}

void FeatureManager::removePointFeatureBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                                      Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                                      Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    for (auto it = point_manager->pointFeature.begin(), it_next = point_manager->pointFeature.begin();
         it != point_manager->pointFeature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            const int old_camera_id = it->point_feature_frame[0].camera_id;
            Eigen::Vector3d uv_i = it->point_feature_frame[0].point;
            it->point_feature_frame.erase(it->point_feature_frame.begin());
            if (it->point_feature_frame.size() < 2)
            {
                point_manager->pointFeature.erase(it);
                continue;
            }
            else
            {
                const int new_camera_id = it->point_feature_frame[0].camera_id;
                Eigen::Matrix3d marg_R = back_R0 * window_ric[old_camera_id];
                Eigen::Vector3d marg_P = back_P0 + back_R0 * tic[old_camera_id];
                Eigen::Matrix3d new_R = new_R0 * window_ric[new_camera_id];
                Eigen::Vector3d new_P = new_P0 + new_R0 * tic[new_camera_id];
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                {
                    it->estimated_depth = dep_j;
                }
                else
                {
                    it->estimated_depth = INIT_DEPTH;
                    it->reliable_depth = false;
                }
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = point_manager->pointFeature.begin(), it_next = point_manager->pointFeature.begin();
         it != point_manager->pointFeature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->point_feature_frame.erase(it->point_feature_frame.begin());
            if (it->point_feature_frame.size() == 0)
                point_manager->pointFeature.erase(it);
        }
    }
    if (line_ba_enabled)
        line_manager->removeBack();
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = point_manager->pointFeature.begin(), it_next = point_manager->pointFeature.begin(); it != point_manager->pointFeature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->point_feature_frame.erase(it->point_feature_frame.begin() + j);
            if (it->point_feature_frame.size() == 0)
                point_manager->pointFeature.erase(it);
        }
    }
    if (line_ba_enabled)
        line_manager->removeFront(frame_count);
}

void FeatureManager::removePointFeatureOutlier(std::set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = point_manager->pointFeature.begin(), it_next = point_manager->pointFeature.begin();
         it != point_manager->pointFeature.end(); it = it_next)
    {
        it_next++;
        int index = it->point_feature_id;
        itSet = outlierIndex.find(index);
        if (itSet != outlierIndex.end())
        {
            auto mpt_it = local_mappoints.find(index);
            if (mpt_it != local_mappoints.end() && mpt_it->second)
                mpt_it->second->SetBad();
            point_manager->pointFeature.erase(it);
        }
    }
}

int FeatureManager::getPointFeatureCount()
{
    int cnt = 0;
    for (auto &it : point_manager->pointFeature)
    {
        it.used_num = static_cast<int>(it.point_feature_frame.size());
        if (usePointFeatureForOptimization(it))
            ++cnt;
    }
    return cnt;
}

bool FeatureManager::usePointFeatureForOptimization(const PointFeatureId &point_feature) const
{
    auto it = local_mappoints.find(point_feature.point_feature_id);
    return it != local_mappoints.end() && it->second && it->second->IsGood();
}

bool FeatureManager::hasGoodMappoint(int point_feature_id) const
{
    auto it = local_mappoints.find(point_feature_id);
    return it != local_mappoints.end() && it->second && it->second->IsGood();
}

std::pair<int, int> FeatureManager::windowSupportStats(int point_feature_id) const
{
    if (point_feature_id < 0)
        return std::make_pair(0, -1);

    int support_count = 0;
    int last_support_frame_id = -1;
    for (auto it = local_frames.rbegin(); it != local_frames.rend(); ++it)
    {
        if (!it->hasObservation(point_feature_id))
            continue;
        ++support_count;
        if (last_support_frame_id < 0)
            last_support_frame_id = it->frame_id;
    }
    return std::make_pair(support_count, last_support_frame_id);
}

bool FeatureManager::useLineForOptimization(const LinePerId &line_feature) const
{
    return line_ba_enabled && line_feature.mapline && line_feature.mapline->IsValid() &&
           line_feature.solve_flag == 1 &&
           line_feature.used_num >= LINE_MIN_OBS &&
           line_feature.start_frame < WINDOW_SIZE - 2 &&
           line_feature.line_3d_world.allFinite();
}

void FeatureManager::addLineFeature(int frame_count, const LineFeatureFrameMap &lines, double td)
{
    if (!line_ba_enabled)
        return;
    line_manager->addLineFeature(frame_count, lines, td);
}

int FeatureManager::getLineFeatureCount() const
{
    return line_ba_enabled ? line_manager->getLineFeatureCount() : 0;
}

void FeatureManager::triangulateLine(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    if (!line_ba_enabled)
        return;
    line_manager->triangulateLine(frameCnt, Ps, Rs, tic, window_ric);
}

Eigen::Matrix<double, Eigen::Dynamic, 4> FeatureManager::getLineFeatureVector() const
{
    return line_ba_enabled ? line_manager->getLineFeatureVector() : Eigen::Matrix<double, Eigen::Dynamic, 4>(0, 4);
}

void FeatureManager::setLineFeature(const Eigen::Matrix<double, Eigen::Dynamic, 4> &x,
                                    int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    if (!line_ba_enabled)
        return;
    line_manager->setLineFeature(x, frameCnt, Ps, Rs, tic, window_ric);
}

void FeatureManager::removeLineOutliers(const std::map<int, std::set<int>> &outlier_observers)
{
    if (!line_ba_enabled)
        return;
    line_manager->removeLineOutliers(outlier_observers);
}

void FeatureManager::removeLineBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                              Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                              Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    if (!line_ba_enabled)
        return;
    line_manager->removeBackShiftDepth(back_R0, back_P0, new_R0, new_P0, tic, window_ric);
}

void FeatureManager::removeLineBack()
{
    if (!line_ba_enabled)
        return;
    line_manager->removeBack();
}

void FeatureManager::removeLineFront(int frame_count)
{
    if (!line_ba_enabled)
        return;
    line_manager->removeFront(frame_count);
}

void FeatureManager::removeLineFailures()
{
    if (!line_ba_enabled)
        return;
    line_manager->removeFailures();
}
