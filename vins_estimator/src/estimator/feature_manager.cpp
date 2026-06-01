/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"
#include <cmath>
#include <limits>
#include <utility>
#include <opencv2/imgproc/imgproc.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "../utility/line_geometry.h"
#include "../utility/mapline.h"
#include "deepFeature/include/utils.h"

namespace
{
bool isReliableTriangulatedDepth(double depth)
{
    return depth >= FEATURE_MIN_DEPTH && depth <= FEATURE_MAX_DEPTH;
}

} // namespace

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i].setIdentity();
        line_camera_intrinsics[i] << FOCAL_LENGTH, FOCAL_LENGTH, COL * 0.5, ROW * 0.5;
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
    pending_keyframe_next_frame = false;
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[])
{
    for (int i = 0; i < NUM_OF_CAM; ++i)
        line_camera_intrinsics[i] = intrinsics[i];
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

bool FeatureManager::knownLandmarksOnly() const
{
    return visual_tracking_mode != TRACKING_MODE_STEREO;
}

bool FeatureManager::allowDepthInitialization() const
{
    return !knownLandmarksOnly();
}

bool FeatureManager::hasReliableDepth(const FeaturePerId &feature_per_id) const
{
    return feature_per_id.estimated_depth > 0 && feature_per_id.reliable_depth;
}

bool FeatureManager::useFeatureForOptimization(const FeaturePerId &feature_per_id) const
{
    auto it = local_mappoints.find(feature_per_id.feature_id);
    if (it == local_mappoints.end() || !it->second || !it->second->IsGood())
        return false;
    return true;
}

bool FeatureManager::hasGoodMappoint(int feature_id) const
{
    auto it = local_mappoints.find(feature_id);
    if (it == local_mappoints.end() || !it->second)
        return false;
    return it->second->IsGood();
}

void FeatureManager::clearState()
{
    feature.clear();
    line_feature.clear();
    local_frames.clear();
    local_mappoints.clear();
    has_local_window_state = false;
    visual_tracking_mode = TRACKING_MODE_STEREO;
    active_camera_id = 0;
    pending_keyframe_next_frame = false;
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (useFeatureForOptimization(it))
        {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::useLineForOptimization(const LinePerId &line_per_id) const
{
    return LINE_BA && line_per_id.mapline && line_per_id.mapline->IsValid() &&
           line_per_id.solve_flag == 1 &&
           line_per_id.used_num >= LINE_MIN_OBS &&
           line_per_id.start_frame < WINDOW_SIZE - 2 &&
           line_per_id.line_3d_world.allFinite();
}

int FeatureManager::getLineFeatureCount()
{
    int cnt = 0;
    for (auto &it : line_feature)
    {
        it.used_num = it.line_per_frame.size();
        if (useLineForOptimization(it))
            cnt++;
    }
    return cnt;
}

void FeatureManager::addLineFeature(int frame_count, const LineFeatureFrameMap &lines, double td)
{
    for (const auto &id_lines : lines)
    {
        if (id_lines.second.empty())
            continue;

        LinePerFrame line_per_frame(id_lines.second[0], td);
        if (id_lines.second.size() == 2 && id_lines.second[0].camera_id == 0 && id_lines.second[1].camera_id == 1)
            line_per_frame.rightObservation(id_lines.second[1]);

        const int line_id = id_lines.first;
        auto it = find_if(line_feature.begin(), line_feature.end(), [line_id](const LinePerId &existing_line) {
            return existing_line.line_id == line_id;
        });
        if (it == line_feature.end())
        {
            line_feature.push_back(LinePerId(line_id, frame_count));
            line_feature.back().line_per_frame.push_back(line_per_frame);
            if (line_feature.back().mapline)
                line_feature.back().mapline->AddObverser(frame_count, 0);
        }
        else
        {
            it->line_per_frame.push_back(line_per_frame);
            if (it->mapline)
                it->mapline->AddObverser(frame_count, static_cast<int>(it->line_per_frame.size()) - 1);
        }
    }
}

void FeatureManager::setWindowState(Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if (local_tic.size() != NUM_OF_CAM)
        local_tic.resize(NUM_OF_CAM, Vector3d::Zero());
    if (local_ric.size() != NUM_OF_CAM)
        local_ric.resize(NUM_OF_CAM, Matrix3d::Identity());

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

void FeatureManager::addLocalFrame(int frameCnt, double header,
                                   const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
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

std::pair<int, int> FeatureManager::windowSupportStats(int feature_id) const
{
    if (feature_id < 0)
        return std::make_pair(0, -1);

    int support_count = 0;
    int last_support_frame_id = -1;
    for (auto it = local_frames.rbegin(); it != local_frames.rend(); ++it)
    {
        if (!it->hasObservation(feature_id))
            continue;
        ++support_count;
        if (last_support_frame_id < 0)
            last_support_frame_id = it->frame_id;
    }
    return std::make_pair(support_count, last_support_frame_id);
}

void FeatureManager::syncMappointFromFeature(FeaturePerId &feature_per_id)
{
    auto it = local_mappoints.find(feature_per_id.feature_id);
    if (it == local_mappoints.end() || !it->second)
        return;

    if (!std::isfinite(feature_per_id.estimated_depth) || feature_per_id.estimated_depth <= 0)
        return;

    if (!has_local_window_state || feature_per_id.feature_per_frame.empty())
        return;

    it->second->SetPosition(featureDepthToWorldPoint(feature_per_id));
}

bool FeatureManager::triangulateMappointWorldPoint(const FeaturePerId &feature_per_id, Vector3d &point_w, double &depth) const
{
    point_w.setZero();
    depth = -1.0;

    if (!has_local_window_state || feature_per_id.feature_per_frame.empty())
        return false;

    const int anchor_frame = feature_per_id.start_frame;
    const int anchor_camera_id = feature_per_id.feature_per_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return false;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(local_ric.size()))
        return false;

    if (STEREO && anchor_camera_id == 0 && feature_per_id.feature_per_frame[0].is_stereo)
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

        const Eigen::Vector2d point0 = feature_per_id.feature_per_frame[0].point.head(2);
        const Eigen::Vector2d point1 = feature_per_id.feature_per_frame[0].pointRight.head(2);
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
        if (feature_per_id.feature_per_frame.size() < 2)
            return false;

        Eigen::MatrixXd svd_A(2 * feature_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        for (size_t obs_idx = 0; obs_idx < feature_per_id.feature_per_frame.size(); ++obs_idx)
        {
            const int frame_idx = feature_per_id.start_frame + static_cast<int>(obs_idx);
            if (frame_idx < 0 || frame_idx > WINDOW_SIZE)
                return false;

            const FeaturePerFrame &obs = feature_per_id.feature_per_frame[obs_idx];
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
        for (size_t obs_idx = 0; obs_idx < feature_per_id.feature_per_frame.size(); ++obs_idx)
        {
            const int frame_idx = feature_per_id.start_frame + static_cast<int>(obs_idx);
            const FeaturePerFrame &obs = feature_per_id.feature_per_frame[obs_idx];
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

void FeatureManager::updateMappointDescriptors(const std::vector<int> &ids, const Eigen::Matrix<float, 259, Eigen::Dynamic> &features)
{
    if (features.rows() != 259 || features.cols() <= 0 || ids.empty())
        return;

    const int count = std::min<int>(static_cast<int>(ids.size()), static_cast<int>(features.cols()));
    for (int i = 0; i < count; ++i)
    {
        const int feature_id = ids[i];
        auto it = local_mappoints.find(feature_id);
        if (it == local_mappoints.end() || !it->second)
            continue;
        if (!features.col(i).allFinite())
            continue;

        it->second->SetDescriptor(features.col(i));
    }
}

void FeatureManager::markMappointFromFeature(FeaturePerId &feature_per_id)
{
    auto &mappoint = local_mappoints[feature_per_id.feature_id];
    if (!mappoint)
        mappoint = std::make_shared<Mappoint>(feature_per_id.feature_id);

    mappoint->ClearObversers();
    mappoint->SetTrackCount(static_cast<int>(feature_per_id.feature_per_frame.size()));
    mappoint->ResetStaleCount();
    for (int i = 0; i < static_cast<int>(feature_per_id.feature_per_frame.size()); ++i)
        mappoint->AddObverser(feature_per_id.start_frame + i, i);

    if (feature_per_id.solve_flag == 2)
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
            if (triangulateMappointWorldPoint(feature_per_id, triangulated_point, triangulated_depth))
            {
                mappoint->SetPosition(triangulated_point);
                mappoint->SetGood();
                feature_per_id.estimated_depth = triangulated_depth;
                feature_per_id.reliable_depth = true;
                feature_per_id.solve_flag = 1;
                feature_per_id.depth_fail_count = 0;
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
        if (std::isfinite(feature_per_id.estimated_depth) && feature_per_id.estimated_depth > 0)
            syncMappointFromFeature(feature_per_id);
        else if (mappoint->HasPosition())
        {
            const double depth = worldPointToFeatureDepth(feature_per_id, mappoint->GetPosition());
            if (isReliableTriangulatedDepth(depth))
            {
                feature_per_id.estimated_depth = depth;
                feature_per_id.reliable_depth = true;
                feature_per_id.solve_flag = 1;
                feature_per_id.depth_fail_count = 0;
            }
        }
    }
    else if (!mappoint->IsBad())
    {
        mappoint->SetUnTriangulated();
    }
}

bool FeatureManager::projectWorldPointToCamera(const Vector3d &point_w, const Eigen::Matrix4d &nextT, int camera_id, Vector3d &point_cam) const
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
        candidate.feature_id = item.first;
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
        return a.feature_id < b.feature_id;
    });

    if (candidates.size() > 512)
        candidates.resize(512);
    return candidates;
}

void FeatureManager::pruneStaleLocalMappoints()
{
    std::set<int> alive_feature_ids;
    for (const auto &it_per_id : feature)
        alive_feature_ids.insert(it_per_id.feature_id);

    for (auto it = local_mappoints.begin(); it != local_mappoints.end();)
    {
        const bool alive = alive_feature_ids.find(it->first) != alive_feature_ids.end();
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

Vector3d FeatureManager::featureDepthToWorldPoint(const FeaturePerId &feature_per_id) const
{
    if (!has_local_window_state || feature_per_id.feature_per_frame.empty())
        return Vector3d::Zero();

    const int anchor_camera_id = feature_per_id.feature_per_frame[0].camera_id;
    const int anchor_frame = feature_per_id.start_frame;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(local_ric.size()))
        return Vector3d::Zero();
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return Vector3d::Zero();
    const Vector3d pts_in_cam = local_ric[anchor_camera_id] *
                                (feature_per_id.estimated_depth * feature_per_id.feature_per_frame[0].point) +
                                local_tic[anchor_camera_id];
    return local_Rs[anchor_frame] * pts_in_cam + local_Ps[anchor_frame];
}

double FeatureManager::worldPointToFeatureDepth(const FeaturePerId &feature_per_id, const Vector3d &point_w) const
{
    if (!has_local_window_state || feature_per_id.feature_per_frame.empty())
        return -1.0;

    const int anchor_frame = feature_per_id.start_frame;
    const int anchor_camera_id = feature_per_id.feature_per_frame[0].camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return -1.0;
    if (anchor_camera_id < 0 || anchor_camera_id >= static_cast<int>(local_ric.size()))
        return -1.0;

    const Eigen::Vector3d anchor_t = local_Ps[anchor_frame] + local_Rs[anchor_frame] * local_tic[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = local_Rs[anchor_frame] * local_ric[anchor_camera_id];
    const Eigen::Vector3d point_cam = anchor_R.transpose() * (point_w - anchor_t);
    return point_cam.z();
}

void FeatureManager::updateLocalPoints(int frameCnt)
{
    (void)frameCnt;
    int untriangulated_count = 0;
    int good_count = 0;
    int bad_count = 0;
    int supported_good_count = 0;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.feature_per_frame.size());
        markMappointFromFeature(it_per_id);

        auto mpt_it = local_mappoints.find(it_per_id.feature_id);
        if (mpt_it == local_mappoints.end() || !mpt_it->second)
            continue;

        const auto support_stats = windowSupportStats(it_per_id.feature_id);
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


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td, id_pts.second[0].first);
        if(id_pts.second.size() == 2 && id_pts.second[0].first == 0 && id_pts.second[1].first == 1)
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
        }

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature.end())
        {
            if (knownLandmarksOnly())
                continue;
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if( it-> feature_per_frame.size() >= 4)
                long_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
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

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r, bool only_good)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (only_good && !hasGoodMappoint(it.feature_id))
            continue;
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            if (it.feature_per_frame[idx_l].camera_id != it.feature_per_frame[idx_r].camera_id)
                continue;

            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!useFeatureForOptimization(it_per_id))
            continue;

        const double optimized_inv_depth = x(++feature_index);
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
                auto mpt_it = local_mappoints.find(it_per_id.feature_id);
                if (mpt_it != local_mappoints.end() && mpt_it->second)
                    mpt_it->second->SetBad();
            }
            else
            {
                it_per_id.solve_flag = 1;
            }
        }

        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.solve_flag == 1)
            syncMappointFromFeature(it_per_id);
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
        {
            feature.erase(it);
        }
    }

    for (auto it = line_feature.begin(), it_next = line_feature.begin();
         it != line_feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            line_feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
    {
        it_per_id.estimated_depth = -1;
        it_per_id.reliable_depth = false;
        it_per_id.depth_fail_count = 0;
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!useFeatureForOptimization(it_per_id))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}


void FeatureManager::triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
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


bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
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
        printf("feature tracking not enough, please slowly move you device! \n");
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

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], int active_camera_id_, bool prefer_good_points)
{

    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &it_per_id : feature)
        {
            const bool usable_depth = prefer_good_points
                                           ? (hasGoodMappoint(it_per_id.feature_id) && hasReliableDepth(it_per_id))
                                           : (knownLandmarksOnly() ? hasReliableDepth(it_per_id) : it_per_id.estimated_depth > 0);
            if (usable_depth)
            {
                int index = frameCnt - it_per_id.start_frame;
                if((int)it_per_id.feature_per_frame.size() >= index + 1)
                {
                    int anchor_camera_id = it_per_id.feature_per_frame[0].camera_id;
                    int obs_camera_id = it_per_id.feature_per_frame[index].camera_id;
                    if (obs_camera_id != active_camera_id_)
                        continue;

                    Vector3d ptsInCam = ric[anchor_camera_id] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[anchor_camera_id];
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d); 
                }
            }
        }
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

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if (!allowDepthInitialization())
        return;

    for (auto &it_per_id : feature)
    {
        auto mpt_it = local_mappoints.find(it_per_id.feature_id);
        MappointPtr mappoint = (mpt_it != local_mappoints.end()) ? mpt_it->second : nullptr;

        if (mappoint && mappoint->HasPosition())
        {
            const double depth = worldPointToFeatureDepth(it_per_id, mappoint->GetPosition());
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
                mappoint = std::make_shared<Mappoint>(it_per_id.feature_id);
                local_mappoints[it_per_id.feature_id] = mappoint;
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

void FeatureManager::triangulateLine(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if (!LINE_BA)
        return;

    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.line_per_frame.size());
        if (it_per_id.used_num < LINE_MIN_OBS || it_per_id.start_frame >= WINDOW_SIZE - 2)
            continue;
        if (it_per_id.solve_flag == 1 && it_per_id.is_triangulation)
            continue;

        bool solved = false;

        if (STEREO && !it_per_id.line_per_frame.empty())
        {
            const LinePerFrame &first_obs = it_per_id.line_per_frame.front();
            if (first_obs.is_stereo && first_obs.camera_id == 0)
            {
                const int imu_i = it_per_id.start_frame;
                if (imu_i >= 0 && imu_i <= frameCnt)
                {
                    const Eigen::Matrix<double, 3, 4> leftPose = worldToCameraPose(Ps[imu_i], Rs[imu_i], tic[0], ric[0]);
                    const Eigen::Matrix<double, 3, 4> rightPose = worldToCameraPose(Ps[imu_i], Rs[imu_i], tic[1], ric[1]);

                    const Eigen::Vector2d left_start_px = first_obs.uv_start;
                    const Eigen::Vector2d left_end_px = first_obs.uv_end;
                    const Eigen::Vector2d right_start_px = first_obs.uv_start_right;
                    const Eigen::Vector2d right_end_px = first_obs.uv_end_right;

                    const double dx_left = left_end_px.x() - left_start_px.x();
                    const double dy_left = left_end_px.y() - left_start_px.y();
                    const double dx_right = right_end_px.x() - right_start_px.x();
                    const double dy_right = right_end_px.y() - right_start_px.y();

                    if (std::abs(dy_left) > 3.0 && std::abs(std::atan2(dy_left, dx_left)) >= 0.175 &&
                        std::abs(dy_right) > 3.0 && std::abs(std::atan2(dy_right, dx_right)) >= 0.175 &&
                        std::abs(dy_right) > 1e-9)
                    {
                        const double k_inv_right = dx_right / dy_right;
                        const double x11_right = right_start_px.x() + k_inv_right * (left_start_px.y() - right_start_px.y());
                        const double x12_right = right_start_px.x() + k_inv_right * (left_end_px.y() - right_start_px.y());
                        const double x_diff1 = left_start_px.x() - x11_right;
                        const double x_diff2 = left_end_px.x() - x12_right;
                        if (x_diff1 >= LINE_MIN_STEREO_DISPARITY && x_diff1 <= LINE_MAX_STEREO_DISPARITY &&
                            x_diff2 >= LINE_MIN_STEREO_DISPARITY && x_diff2 <= LINE_MAX_STEREO_DISPARITY)
                        {
                            const Eigen::Vector2d right_start_interp((x11_right - line_camera_intrinsics[1](2)) / line_camera_intrinsics[1](0),
                                                                     (left_start_px.y() - line_camera_intrinsics[1](3)) / line_camera_intrinsics[1](1));
                            const Eigen::Vector2d right_end_interp((x12_right - line_camera_intrinsics[1](2)) / line_camera_intrinsics[1](0),
                                                                   (left_end_px.y() - line_camera_intrinsics[1](3)) / line_camera_intrinsics[1](1));

                            Eigen::Vector3d start_world;
                            Eigen::Vector3d end_world;
                            const Eigen::Vector2d left_start = first_obs.start_point.head<2>();
                            const Eigen::Vector2d left_end = first_obs.end_point.head<2>();
                            if (triangulateStereoEndpointFromPoses(leftPose, rightPose, left_start, right_start_interp, start_world) &&
                                triangulateStereoEndpointFromPoses(leftPose, rightPose, left_end, right_end_interp, end_world) &&
                                start_world.allFinite() && end_world.allFinite() &&
                                buildLineFromEndpoints(start_world, end_world, it_per_id.line_3d_world))
                            {
                                it_per_id.solve_flag = 1;
                                it_per_id.is_triangulation = true;
                                if (it_per_id.mapline)
                                {
                                    Vector6d endpoints;
                                    endpoints << start_world, end_world;
                                    it_per_id.mapline->SetEndpoints(endpoints, false);
                                    it_per_id.mapline->SetLine3D(it_per_id.line_3d_world);
                                    it_per_id.mapline->SetGood();
                                }
                                solved = true;
                            }
                        }
                    }
                }
            }
        }

        if (!solved)
        {
            int imu_i = it_per_id.start_frame;
            int imu_j = imu_i - 1;
            Eigen::Vector4d pii;
            Eigen::Vector3d ni;
            Eigen::Vector4d obsj;
            Eigen::Vector3d tij;
            Eigen::Matrix3d Rij;
            double min_cos_theta = 1.0;
            bool have_base_plane = false;

            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];

            for (auto &it_per_frame : it_per_id.line_per_frame)
            {
                imu_j++;
                const Eigen::Vector3d p1 = it_per_frame.start_point;
                const Eigen::Vector3d p2 = it_per_frame.end_point;
                if (imu_j == imu_i)
                {
                    pii = piFromPPP(p1, p2, Eigen::Vector3d::Zero());
                    ni = pii.head<3>();
                    if (ni.norm() > 1e-9)
                        ni.normalize();
                    have_base_plane = true;
                    continue;
                }

                const Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
                const Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
                const Eigen::Vector3d t = R0.transpose() * (t1 - t0);
                const Eigen::Matrix3d R = R0.transpose() * R1;
                Eigen::Vector3d p3 = R * p1 + t;
                Eigen::Vector3d p4 = R * p2 + t;
                const Eigen::Vector4d pij = piFromPPP(p3, p4, t);
                Eigen::Vector3d nj = pij.head<3>();
                if (ni.norm() < 1e-9 || nj.norm() < 1e-9)
                    continue;
                nj.normalize();
                const double cos_theta = ni.dot(nj);
                if (cos_theta < min_cos_theta)
                {
                    min_cos_theta = cos_theta;
                    tij = t;
                    Rij = R;
                    obsj << it_per_frame.start_point.x(), it_per_frame.start_point.y(),
                        it_per_frame.end_point.x(), it_per_frame.end_point.y();
                }
            }

            if (have_base_plane && min_cos_theta <= 0.998)
            {
                const Eigen::Vector3d p3(obsj(0), obsj(1), 1.0);
                const Eigen::Vector3d p4(obsj(2), obsj(3), 1.0);
                const Eigen::Vector4d pij = piFromPPP(Rij * p3 + tij, Rij * p4 + tij, tij);
                const Vector6d plk_camera = pipiPlk(pii, pij);
                if (plk_camera.tail<3>().norm() > 1e-9)
                {
                    it_per_id.line_3d_world = plkToWorld(plk_camera, R0, t0);
                    if (buildLineFromPlucker(it_per_id.line_3d_world, it_per_id.line_3d_world))
                    {
                        it_per_id.solve_flag = 1;
                        it_per_id.is_triangulation = true;
                        if (it_per_id.mapline)
                        {
                            it_per_id.mapline->SetLine3D(it_per_id.line_3d_world);
                            it_per_id.mapline->SetGood();
                        }
                        solved = true;
                    }
                }
            }
        }

        if (!solved)
        {
            it_per_id.solve_flag = 0;
            it_per_id.is_triangulation = false;
            if (it_per_id.mapline)
                it_per_id.mapline->SetUnTriangulated();
        }
    }
}

Eigen::Matrix<double, Eigen::Dynamic, 4> FeatureManager::getLineFeatureVector()
{
    Eigen::Matrix<double, Eigen::Dynamic, 4> line_vec(getLineFeatureCount(), 4);
    int line_index = -1;
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.line_per_frame.size());
        if (!useLineForOptimization(it_per_id))
            continue;
        line_vec.row(++line_index) = plk_to_orth(it_per_id.line_3d_world).transpose();
    }
    return line_vec;
}

void FeatureManager::setLineFeature(const Eigen::Matrix<double, Eigen::Dynamic, 4> &x,
                                    int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if (!LINE_BA)
        return;

    int line_index = -1;
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.line_per_frame.size());
        if (!useLineForOptimization(it_per_id))
            continue;
        if (++line_index >= x.rows())
            break;
        const Eigen::Vector4d line_orth = x.row(line_index).transpose();
        const Eigen::Matrix<double, 6, 1> line_plucker = orth_to_plk(line_orth);
        if (!buildLineFromPlucker(line_plucker, it_per_id.line_3d_world))
        {
            it_per_id.solve_flag = 2;
            it_per_id.is_triangulation = false;
            if (it_per_id.mapline)
                it_per_id.mapline->SetBad();
            continue;
        }
        it_per_id.solve_flag = 1;
        it_per_id.is_triangulation = true;
        if (it_per_id.mapline)
        {
            it_per_id.mapline->SetLine3D(it_per_id.line_3d_world);
            it_per_id.mapline->SetGood();
        }
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            auto mpt_it = local_mappoints.find(index);
            if (mpt_it != local_mappoints.end() && mpt_it->second)
                mpt_it->second->SetBad();
            feature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

void FeatureManager::removeLineOutliers(const std::map<int, std::set<int>> &outlier_observers)
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin(); it != line_feature.end(); it = it_next)
    {
        it_next++;
        const auto outlier_it = outlier_observers.find(it->line_id);
        if (outlier_it == outlier_observers.end())
            continue;

        const std::set<int> &bad_frames = outlier_it->second;
        for (int idx = static_cast<int>(it->line_per_frame.size()) - 1; idx >= 0; --idx)
        {
            const int frame_id = it->start_frame + idx;
            if (bad_frames.find(frame_id) == bad_frames.end())
                continue;
            if (it->mapline)
                it->mapline->RemoveObverser(frame_id);
            it->line_per_frame.erase(it->line_per_frame.begin() + idx);
        }

        if (it->line_per_frame.empty())
        {
            if (it->mapline)
                it->mapline->SetBad();
            line_feature.erase(it);
            continue;
        }

        it->used_num = static_cast<int>(it->line_per_frame.size());
        if (it->used_num < LINE_MIN_OBS)
        {
            if (it->mapline)
                it->mapline->SetBad();
            line_feature.erase(it);
            continue;
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                          Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                          Vector3d tic[], Matrix3d ric[])
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            const int old_camera_id = it->feature_per_frame[0].camera_id;
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                const int new_camera_id = it->feature_per_frame[0].camera_id;
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
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }

    for (auto it = line_feature.begin(), it_next = line_feature.begin(); it != line_feature.end(); it = it_next)
    {
        it_next++;
        if (it->start_frame != 0)
        {
            it->start_frame--;
        }
        else
        {
            if (it->mapline)
                it->mapline->RemoveObverser(it->start_frame);
            if (!it->line_per_frame.empty())
                it->line_per_frame.erase(it->line_per_frame.begin());
            if (it->line_per_frame.empty())
            {
                if (it->mapline)
                    it->mapline->SetBad();
                line_feature.erase(it);
            }
            else
                it->used_num = static_cast<int>(it->line_per_frame.size());
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }

    for (auto it = line_feature.begin(), it_next = line_feature.begin(); it != line_feature.end(); it = it_next)
    {
        it_next++;
        if (it->start_frame != 0)
        {
            it->start_frame--;
        }
        else
        {
            if (it->mapline)
                it->mapline->RemoveObverser(it->start_frame);
            if (!it->line_per_frame.empty())
                it->line_per_frame.erase(it->line_per_frame.begin());
            if (it->line_per_frame.empty())
            {
                if (it->mapline)
                    it->mapline->SetBad();
                line_feature.erase(it);
            }
            else
                it->used_num = static_cast<int>(it->line_per_frame.size());
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
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
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }

    for (auto it = line_feature.begin(), it_next = line_feature.begin(); it != line_feature.end(); it = it_next)
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
            if (j >= 0 && j < static_cast<int>(it->line_per_frame.size()))
            {
                if (it->mapline)
                    it->mapline->RemoveObverser(frame_count);
                it->line_per_frame.erase(it->line_per_frame.begin() + j);
            }
            if (it->line_per_frame.empty())
            {
                if (it->mapline)
                    it->mapline->SetBad();
                line_feature.erase(it);
            }
            else
                it->used_num = static_cast<int>(it->line_per_frame.size());
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

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
