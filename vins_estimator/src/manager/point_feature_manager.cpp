#include "point_feature_manager.h"
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

PointFeatureManager::PointFeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i].setIdentity();
    }
    visual_tracking_mode = TRACKING_MODE_STEREO;
    active_camera_id = 0;
    has_local_window_state = false;
    local_tic.resize(NUM_OF_CAM, Vector3d::Zero());
    local_ric.resize(NUM_OF_CAM, Matrix3d::Identity());
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        local_Ps[i].setZero();
        local_Rs[i].setIdentity();
    }
    last_point_feature_track_num = 0;
    last_average_parallax = 0.0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
    pending_keyframe_next_frame = false;
}

bool PointFeatureManager::knownLandmarksOnly() const
{
    return visual_tracking_mode != TRACKING_MODE_STEREO;
}

bool PointFeatureManager::allowDepthInitialization() const
{
    return !knownLandmarksOnly();
}

bool PointFeatureManager::hasReliablePointDepth(const PointFeatureId &point_feature) const
{
    return point_feature.estimated_depth > 0 && point_feature.reliable_depth;
}

bool PointFeatureManager::usePointFeatureForOptimization(const PointFeatureId &point_feature) const
{
    auto it = local_mappoints.find(point_feature.point_feature_id);
    return it != local_mappoints.end() && it->second && it->second->IsGood();
}

KeyframeGoodPointRecordList PointFeatureManager::collectGoodKeyframePoints() const
{
    KeyframeGoodPointRecordList records;
    records.reserve(std::min<size_t>(local_mappoints.size(), 1024));
    for (const auto &it_per_id : pointFeature)
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

void PointFeatureManager::clearState()
{
    pointFeature.clear();
    local_frames.clear();
    local_mappoints.clear();
    has_local_window_state = false;
    visual_tracking_mode = TRACKING_MODE_STEREO;
    active_camera_id = 0;
    last_point_feature_track_num = 0;
    last_average_parallax = 0.0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
    pending_keyframe_next_frame = false;
}

void PointFeatureManager::setWindowState(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
    if (local_tic.size() != NUM_OF_CAM)
        local_tic.resize(NUM_OF_CAM, Eigen::Vector3d::Zero());
    if (local_ric.size() != NUM_OF_CAM)
        local_ric.resize(NUM_OF_CAM, Eigen::Matrix3d::Identity());

    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        local_Ps[i] = Ps[i];
        local_Rs[i] = Rs[i];
    }
    for (int i = 0; i < NUM_OF_CAM; ++i)
    {
        local_tic[i] = tic[i];
        local_ric[i] = ric[i];
    }
    has_local_window_state = true;
}

void PointFeatureManager::addLocalFrame(int frameCnt, double header,
                                        const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                        VisualTrackingMode tracking_mode, int active_camera_id)
{
    local_frames.emplace_back(frameCnt, header, static_cast<int>(tracking_mode), active_camera_id);
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

int PointFeatureManager::getPointFeatureCount()
{
    int cnt = 0;
    for (auto &it : pointFeature)
    {
        it.used_num = it.point_feature_frame.size();
        if (usePointFeatureForOptimization(it))
            ++cnt;
    }
    return cnt;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> PointFeatureManager::getCorresponding(int frame_count_l, int frame_count_r, bool only_good)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto &it : pointFeature)
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

void PointFeatureManager::removeFailures()
{
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin(); it != pointFeature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            pointFeature.erase(it);
    }
}

void PointFeatureManager::clearPointFeatureDepth()
{
    for (auto &it_per_id : pointFeature)
    {
        it_per_id.estimated_depth = -1;
        it_per_id.reliable_depth = false;
        it_per_id.depth_fail_count = 0;
    }
}

void PointFeatureManager::updateMappointDescriptors(const std::vector<int> &ids, const Eigen::Matrix<float, 259, Eigen::Dynamic> &features)
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

bool PointFeatureManager::hasGoodMappoint(int point_feature_id) const
{
    auto it = local_mappoints.find(point_feature_id);
    return it != local_mappoints.end() && it->second && it->second->IsGood();
}

std::pair<int, int> PointFeatureManager::windowSupportStats(int point_feature_id) const
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

void PointFeatureManager::syncMappointFromPointFeature(PointFeatureId &point_feature)
{
    auto it = local_mappoints.find(point_feature.point_feature_id);
    if (it == local_mappoints.end() || !it->second)
        return;

    if (!std::isfinite(point_feature.estimated_depth) || point_feature.estimated_depth <= 0)
        return;

    if (!has_local_window_state || point_feature.point_feature_frame.empty())
        return;

    it->second->SetPosition(pointFeatureDepthToWorldPoint(point_feature));
}

bool PointFeatureManager::triangulateMappointWorldPoint(const PointFeatureId &point_feature, Vector3d &point_w, double &depth) const
{
    point_w.setZero();
    depth = -1.0;

    if (!has_local_window_state || point_feature.point_feature_frame.empty())
        return false;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return false;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(local_ric.size()))
        return false;

    if (STEREO && anchor_camera_id == 0 && point_feature.point_feature_frame[0].is_stereo)
    {
        Eigen::Matrix<double, 3, 4> leftPose;
        Eigen::Vector3d t0 = local_Ps[anchor_frame] + local_Rs[anchor_frame] * local_tic[0];
        Eigen::Matrix3d R0 = local_Rs[anchor_frame] * local_ric[0];
        leftPose.leftCols<3>() = R0.transpose();
        leftPose.rightCols<1>() = -R0.transpose() * t0;

        Eigen::Matrix<double, 3, 4> rightPose;
        Eigen::Vector3d t1 = local_Ps[anchor_frame] + local_Rs[anchor_frame] * local_tic[1];
        Eigen::Matrix3d R1 = local_Rs[anchor_frame] * local_ric[1];
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

        point_w = local_Rs[anchor_frame] * (local_ric[0] * localPoint + local_tic[0]) + local_Ps[anchor_frame];
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
            if (camera_id < 0 || camera_id >= static_cast<int>(local_ric.size()))
                return false;

            const Eigen::Vector3d t = local_Ps[frame_idx] + local_Rs[frame_idx] * local_tic[camera_id];
            const Eigen::Matrix3d R = local_Rs[frame_idx] * local_ric[camera_id];
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

        const Eigen::Vector3d anchor_t = local_Ps[anchor_frame] + local_Rs[anchor_frame] * local_tic[anchor_camera_id];
        const Eigen::Matrix3d anchor_R = local_Rs[anchor_frame] * local_ric[anchor_camera_id];
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
            const Eigen::Vector3d t = local_Ps[frame_idx] + local_Rs[frame_idx] * local_tic[obs.camera_id];
            const Eigen::Matrix3d R = local_Rs[frame_idx] * local_ric[obs.camera_id];
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
    auto &mappoint = local_mappoints[point_feature.point_feature_id];
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
    if (!has_local_window_state)
        return false;
    if (camera_id < 0 || camera_id >= static_cast<int>(local_ric.size()))
        return false;
    if (!point_w.allFinite() || !nextT.allFinite())
        return false;

    const Eigen::Matrix3d Rwb = nextT.block<3, 3>(0, 0);
    const Eigen::Vector3d Pwb = nextT.block<3, 1>(0, 3);
    const Eigen::Vector3d point_in_body = Rwb.transpose() * (point_w - Pwb);
    point_cam = local_ric[camera_id].transpose() * (point_in_body - local_tic[camera_id]);
    return point_cam.allFinite() && point_cam.z() > 0.0;
}

std::vector<ProjectionCandidate> PointFeatureManager::collectProjectionCandidates(const Eigen::Matrix4d &nextT) const
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
        if (!projectWorldPointToCamera(mappoint->GetPosition(), nextT, camera_id, point_cam))
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

void PointFeatureManager::pruneStaleLocalMappoints()
{
    std::set<int> alive_point_feature_ids;
    for (const auto &it_per_id : pointFeature)
        alive_point_feature_ids.insert(it_per_id.point_feature_id);

    for (auto it = local_mappoints.begin(); it != local_mappoints.end();)
    {
        const bool alive = alive_point_feature_ids.find(it->first) != alive_point_feature_ids.end();
        if (!it->second)
        {
            it = local_mappoints.erase(it);
            continue;
        }

        if (alive)
        {
            it->second->ResetStaleCount();
            ++it;
            continue;
        }

        it->second->IncreaseStaleCount();
        const auto support_stats = windowSupportStats(it->first);
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

        it = local_mappoints.erase(it);
    }
}

Vector3d PointFeatureManager::pointFeatureDepthToWorldPoint(const PointFeatureId &point_feature) const
{
    if (!has_local_window_state || point_feature.point_feature_frame.empty())
        return Vector3d::Zero();

    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    const int anchor_frame = point_feature.start_frame;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(local_ric.size()))
        return Vector3d::Zero();
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return Vector3d::Zero();
    const Vector3d pts_in_cam = local_ric[anchor_camera_id] *
                                (point_feature.estimated_depth * point_feature.point_feature_frame[0].point) +
                                local_tic[anchor_camera_id];
    return local_Rs[anchor_frame] * pts_in_cam + local_Ps[anchor_frame];
}

double PointFeatureManager::worldPointToPointFeatureDepth(const PointFeatureId &point_feature, const Vector3d &point_w) const
{
    if (!has_local_window_state || point_feature.point_feature_frame.empty())
        return -1.0;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.point_feature_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return -1.0;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(local_ric.size()))
        return -1.0;

    const Eigen::Vector3d anchor_t = local_Ps[anchor_frame] + local_Rs[anchor_frame] * local_tic[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = local_Rs[anchor_frame] * local_ric[anchor_camera_id];
    const Eigen::Vector3d point_cam = anchor_R.transpose() * (point_w - anchor_t);
    return point_cam.z();
}

void PointFeatureManager::updateLocalPoints(int frameCnt)
{
    (void)frameCnt;
    int untriangulated_count = 0;
    int good_count = 0;
    int bad_count = 0;
    int supported_good_count = 0;
    for (auto &it_per_id : pointFeature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.point_feature_frame.size());
        markMappointFromPointFeature(it_per_id);

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
    pruneStaleLocalMappoints();

}


bool PointFeatureManager::addPointFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input pointFeature: %d", (int)image.size());
    ROS_DEBUG("num of pointFeature: %d", getPointFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_point_feature_track_num = 0;
    last_average_parallax = 0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
    for (auto &id_pts : image)
    {
        PointFeatureFrame f_per_fra(id_pts.second[0].second, td, id_pts.second[0].first);
        if(id_pts.second.size() == 2 && id_pts.second[0].first == 0 && id_pts.second[1].first == 1)
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
        }

        int point_feature_id = id_pts.first;
        auto it = find_if(pointFeature.begin(), pointFeature.end(), [point_feature_id](const PointFeatureId &it)
                          {
            return it.point_feature_id == point_feature_id;
                          });

        if (it == pointFeature.end())
        {
            if (knownLandmarksOnly())
                continue;
            pointFeature.push_back(PointFeatureId(point_feature_id, frame_count));
            pointFeature.back().point_feature_frame.push_back(f_per_fra);
            new_point_feature_num++;
        }
        else if (it->point_feature_id == point_feature_id)
        {
            it->point_feature_frame.push_back(f_per_fra);
            last_point_feature_track_num++;
            if( it-> point_feature_frame.size() >= 4)
                long_point_feature_track_num++;
        }
    }

    if (frame_count < 2 || last_point_feature_track_num < 20 || long_point_feature_track_num < 40 || new_point_feature_num > 0.5 * last_point_feature_track_num)
        return true;

    for (auto &it_per_id : pointFeature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.point_feature_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
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
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void PointFeatureManager::setPointFeatureDepth(const VectorXd &x)
{
    int point_feature_index = -1;
    for (auto &it_per_id : pointFeature)
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

        //ROS_INFO("pointFeature id %d , start_frame %d, depth %f ", it_per_id->point_feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.solve_flag == 1)
            syncMappointFromPointFeature(it_per_id);
    }
}

VectorXd PointFeatureManager::getPointFeatureDepthVector()
{
    VectorXd dep_vec(getPointFeatureCount());
    int point_feature_index = -1;
    for (auto &it_per_id : pointFeature)
    {
        it_per_id.used_num = it_per_id.point_feature_frame.size();
        if (!usePointFeatureForOptimization(it_per_id))
            continue;
        dep_vec(++point_feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
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
               ? (hasGoodMappoint(point_feature.point_feature_id) && hasReliablePointDepth(point_feature))
               : (knownLandmarksOnly() ? hasReliablePointDepth(point_feature) : point_feature.estimated_depth > 0);
}

void PointFeatureManager::collectPointFeaturePnPCorrespondences(int frameCnt, int active_camera_id_, bool prefer_good_points,
                                                                Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[],
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

        const Vector3d point_in_cam = ric[anchor_camera_id] * (point_feature.point_feature_frame[0].point * point_feature.estimated_depth) + tic[anchor_camera_id];
        const Vector3d point_in_world = Rs[point_feature.start_frame] * point_in_cam + Ps[point_feature.start_frame];

        pts3D.emplace_back(point_in_world.x(), point_in_world.y(), point_in_world.z());
        pts2D.emplace_back(point_feature.point_feature_frame[obs_index].point.x(), point_feature.point_feature_frame[obs_index].point.y());
    }
}


bool PointFeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("pointFeature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

void PointFeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], int active_camera_id_, bool prefer_good_points)
{

    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        collectPointFeaturePnPCorrespondences(frameCnt, active_camera_id_, prefer_good_points, Ps, Rs, tic, ric, pts2D, pts3D);
        if (prefer_good_points && pts3D.size() < 6)
            return;
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
        RCam = Rs[frameCnt - 1] * ric[active_camera_id_];
        PCam = Rs[frameCnt - 1] * tic[active_camera_id_] + Ps[frameCnt - 1];

        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[active_camera_id_].transpose(); 
            Ps[frameCnt] = -RCam * ric[active_camera_id_].transpose() * tic[active_camera_id_] + PCam;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}

void PointFeatureManager::triangulatePointFeature(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if (!allowDepthInitialization())
        return;

    for (auto &it_per_id : pointFeature)
    {
        auto mpt_it = local_mappoints.find(it_per_id.point_feature_id);
        MappointPtr mappoint = (mpt_it != local_mappoints.end()) ? mpt_it->second : nullptr;

        if (mappoint && mappoint->HasPosition())
        {
            const double depth = worldPointToPointFeatureDepth(it_per_id, mappoint->GetPosition());
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
        if (triangulateMappointWorldPoint(it_per_id, triangulated_point, triangulated_depth))
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

        continue;

    }
}

void PointFeatureManager::removePointFeatureOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin();
         it != pointFeature.end(); it = it_next)
    {
        it_next++;
        int index = it->point_feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            auto mpt_it = local_mappoints.find(index);
            if (mpt_it != local_mappoints.end() && mpt_it->second)
                mpt_it->second->SetBad();
            pointFeature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

void PointFeatureManager::removePointFeatureBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                                           Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                                           Vector3d tic[], Matrix3d ric[])
{
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin();
         it != pointFeature.end(); it = it_next)
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
                pointFeature.erase(it);
                continue;
            }
            else
            {
                const int new_camera_id = it->point_feature_frame[0].camera_id;
                Eigen::Matrix3d marg_R = back_R0 * ric[old_camera_id];
                Eigen::Vector3d marg_P = back_P0 + back_R0 * tic[old_camera_id];
                Eigen::Matrix3d new_R = new_R0 * ric[new_camera_id];
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
        // remove tracking-lost pointFeature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            pointFeature.erase(it);
        }
        */
    }
}

void PointFeatureManager::removeBack()
{
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin();
         it != pointFeature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->point_feature_frame.erase(it->point_feature_frame.begin());
            if (it->point_feature_frame.size() == 0)
                pointFeature.erase(it);
        }
    }
}

void PointFeatureManager::removeFront(int frame_count)
{
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin(); it != pointFeature.end(); it = it_next)
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
                pointFeature.erase(it);
        }
    }
}


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
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
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
