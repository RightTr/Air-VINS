#include "feature_manager.h"
#include <cmath>
#include <limits>
#include <utility>
#include "../utility/ros_utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace Eigen;

namespace
{
bool isReliableTriangulatedDepth(double depth)
{
    return depth >= FEATURE_MIN_DEPTH && depth <= FEATURE_MAX_DEPTH;
}

bool hasReliableStereoDisparity(const PointFeatureFrame &anchor_obs)
{
    if (!anchor_obs.is_stereo || anchor_obs.camera_id != 0)
        return false;

    const double y_error = std::abs(anchor_obs.uv.y() - anchor_obs.uvRight.y());
    if (y_error > DEEP_FEATURE_STEREO_Y_THRESHOLD)
        return false;

    const double disparity = anchor_obs.uv.x() - anchor_obs.uvRight.x();
    return disparity >= DEEP_FEATURE_MIN_STEREO_DISPARITY;
}

} // namespace

int PointFeatureId::endFrame() const
{
    return start_frame + point_feature_frame.size() - 1;
}

PointFeatureManager::PointFeatureManager()
{
    last_point_feature_track_num = 0;
    last_average_parallax = 0.0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
}

void PointFeatureManager::clearState()
{
    pointFeature.clear();
    pointFeatureIndex.clear();
    local_frames.clear();
    windowSupportCount.clear();
    windowSupportLastFrame.clear();
    local_mappoints.clear();
    last_point_feature_track_num = 0;
    last_average_parallax = 0.0;
    new_point_feature_num = 0;
    long_point_feature_track_num = 0;
}

void PointFeatureManager::addLocalFrame(
    int frameCnt, double header,
    const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    VisualTrackingMode tracking_mode, int active_camera_id)
{
    if (active_camera_id < 0)
        active_camera_id = 0;
    if (active_camera_id >= NUM_OF_CAM)
        active_camera_id = NUM_OF_CAM - 1;
    local_frames.emplace_back(frameCnt, header, static_cast<int>(tracking_mode), active_camera_id);
    LocalFrameObs &local_frame = local_frames.back();
    for (const auto &id_pts : image)
    {
        if (id_pts.second.empty())
            continue;
        local_frame.addObservation(id_pts.first, id_pts.second[0].first, id_pts.second[0].second);
        ++windowSupportCount[id_pts.first];
        windowSupportLastFrame[id_pts.first] = frameCnt;
    }
    while (local_frames.size() > WINDOW_SIZE + 1)
    {
        const LocalFrameObs &oldest_frame = local_frames.front();
        for (const auto &observation : oldest_frame.observations)
        {
            const int feature_id = observation.first;
            auto support_it = windowSupportCount.find(feature_id);
            if (support_it == windowSupportCount.end())
                continue;

            if (--support_it->second <= 0)
            {
                windowSupportCount.erase(support_it);
                windowSupportLastFrame.erase(feature_id);
            }
        }
        local_frames.pop_front();
    }
}

int PointFeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : pointFeature)
    {
        it.used_num = it.point_feature_frame.size();
        if (useFeatureForOptimization(it))
            ++cnt;
    }
    return cnt;
}

void PointFeatureManager::syncMappointFromPointFeature(PointFeatureId &point_feature,
                                                       Vector3d Ps[], Matrix3d Rs[],
                                                       Vector3d tic[], Matrix3d window_ric[])
{
    auto it = local_mappoints.find(point_feature.point_feature_id);
    if (it == local_mappoints.end() || !it->second)
        return;

    if (!std::isfinite(point_feature.estimated_depth) || point_feature.estimated_depth <= 0)
        return;

    if (point_feature.point_feature_frame.empty())
        return;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.anchorObservation().camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return;
    if (anchor_camera_id < 0 || anchor_camera_id >= NUM_OF_CAM)
        return;

    const Vector3d point_in_body = window_ric[anchor_camera_id] *
                                   (point_feature.estimated_depth * point_feature.anchorObservation().point) +
                                   tic[anchor_camera_id];
    it->second->SetPosition(Rs[anchor_frame] * point_in_body + Ps[anchor_frame]);
}

bool PointFeatureManager::triangulateMappointWorldPoint(const PointFeatureId &point_feature,
                                                        Vector3d Ps[], Matrix3d Rs[],
                                                        Vector3d tic[], Matrix3d window_ric[],
                                                        Vector3d &point_w, double &depth) const
{
    point_w.setZero();
    depth = -1.0;

    if (point_feature.point_feature_frame.empty())
        return false;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.anchorObservation().camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return false;
    if (anchor_camera_id < 0 || anchor_camera_id >= NUM_OF_CAM)
        return false;

    const Eigen::Vector3d anchor_t = Ps[anchor_frame] + Rs[anchor_frame] * tic[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = Rs[anchor_frame] * window_ric[anchor_camera_id];

    const PointFeatureFrame &anchor_obs = point_feature.anchorObservation();
    if (STEREO && anchor_obs.camera_id == 0 && anchor_obs.is_stereo)
    {
        if (!hasReliableStereoDisparity(anchor_obs))
            goto multi_view_fallback;

        Eigen::Matrix<double, 3, 4> leftPose;
        const int stereo_frame = anchor_frame;

        Eigen::Vector3d t0 = Ps[stereo_frame] + Rs[stereo_frame] * tic[0];
        Eigen::Matrix3d R0 = Rs[stereo_frame] * window_ric[0];
        leftPose.leftCols<3>() = R0.transpose();
        leftPose.rightCols<1>() = -R0.transpose() * t0;

        Eigen::Matrix<double, 3, 4> rightPose;
        Eigen::Vector3d t1 = Ps[stereo_frame] + Rs[stereo_frame] * tic[1];
        Eigen::Matrix3d R1 = Rs[stereo_frame] * window_ric[1];
        rightPose.leftCols<3>() = R1.transpose();
        rightPose.rightCols<1>() = -R1.transpose() * t1;

        const Eigen::Vector2d point0 = anchor_obs.point.head(2);
        const Eigen::Vector2d point1 = anchor_obs.pointRight.head(2);
        Eigen::Vector3d point3d;
        triangulatePoint(leftPose, rightPose, point0, point1, point3d);

        const Eigen::Vector3d localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
        if (!point3d.allFinite() || !localPoint.allFinite())
            goto multi_view_fallback;

        point_w = Rs[stereo_frame] * (window_ric[0] * localPoint + tic[0]) + Ps[stereo_frame];
        if (!point_w.allFinite())
            goto multi_view_fallback;

        const Eigen::Vector3d anchor_point_cam = anchor_R.transpose() * (point_w - anchor_t);
        depth = anchor_point_cam.z();
        if (!isReliableTriangulatedDepth(depth))
            goto multi_view_fallback;

        return true;
    }

multi_view_fallback:
    if (point_feature.point_feature_frame.size() < 2)
        return false;

    size_t equation_rows = 0;
    for (size_t obs_idx = 0; obs_idx < point_feature.point_feature_frame.size(); ++obs_idx)
    {
        const PointFeatureFrame &obs = point_feature.point_feature_frame[obs_idx];
        equation_rows += 2;
        if (STEREO && obs.camera_id == 0 && obs.is_stereo)
            equation_rows += 2;
    }

    Eigen::MatrixXd svd_A(equation_rows, 4);
    int svd_idx = 0;
    auto addObservationEquation = [&](int frame_idx, int camera_id, const Eigen::Vector3d &bearing) -> bool {
        if (frame_idx < 0 || frame_idx > WINDOW_SIZE)
            return false;
        if (camera_id < 0 || camera_id >= NUM_OF_CAM)
            return false;

        const Eigen::Vector3d t = Ps[frame_idx] + Rs[frame_idx] * tic[camera_id];
        const Eigen::Matrix3d R = Rs[frame_idx] * window_ric[camera_id];
        Eigen::Matrix<double, 3, 4> P;
        P.leftCols<3>() = R.transpose();
        P.rightCols<1>() = -R.transpose() * t;

        const Eigen::Vector3d f = bearing.normalized();
        svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
        svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
        return true;
    };

    for (size_t obs_idx = 0; obs_idx < point_feature.point_feature_frame.size(); ++obs_idx)
    {
        const int frame_idx = point_feature.start_frame + static_cast<int>(obs_idx);
        const PointFeatureFrame &obs = point_feature.point_feature_frame[obs_idx];
        if (!addObservationEquation(frame_idx, obs.camera_id, obs.point))
            return false;
        if (STEREO && obs.camera_id == 0 && obs.is_stereo)
        {
            if (!addObservationEquation(frame_idx, 1, obs.pointRight))
                return false;
        }
    }

    if (svd_idx < 4)
        return false;

    const Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    if (std::abs(svd_V[3]) < 1e-12)
        return false;

    point_w = svd_V.head<3>() / svd_V[3];
    if (!point_w.allFinite())
        return false;

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
        const auto accumulateReprojection = [&](int camera_id, const Eigen::Vector3d &bearing) -> bool {
            if (camera_id < 0 || camera_id >= NUM_OF_CAM)
                return false;

            const Eigen::Vector3d t = Ps[frame_idx] + Rs[frame_idx] * tic[camera_id];
            const Eigen::Matrix3d R = Rs[frame_idx] * window_ric[camera_id];
            const Eigen::Vector3d point_cam = R.transpose() * (point_w - t);
            if (!point_cam.allFinite() || point_cam.z() <= 0.0)
                return false;
            const Eigen::Vector2d proj = point_cam.head<2>() / point_cam.z();
            reproj_sum += (proj - bearing.head<2>()).norm();
            ++valid_obs;
            return true;
        };

        if (!accumulateReprojection(obs.camera_id, obs.point))
            return false;
        if (STEREO && obs.camera_id == 0 && obs.is_stereo)
        {
            if (!accumulateReprojection(1, obs.pointRight))
                return false;
        }
    }

    if (valid_obs < 2)
        return false;
    if (reproj_sum / valid_obs > 0.01)
        return false;

    return true;
}

void PointFeatureManager::pruneInactiveLocalMappoints()
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
            it->second->ResetInactiveCount();
            ++it;
            continue;
        }

        it->second->IncreaseInactiveCount();
        const auto support_stats = windowSupportStats(it->first);
        const bool strong_window_support = support_stats.first >= 2;
        const int keep_good_limit = strong_window_support ? 6 : 3;
        const int keep_non_good_limit = strong_window_support ? 4 : 2;
        const bool keep_temporarily = it->second->IsGood() && it->second->InactiveCount() < keep_good_limit;
        const bool keep_non_good_temporarily = !it->second->IsGood() && it->second->InactiveCount() < keep_non_good_limit;
        if (keep_temporarily || keep_non_good_temporarily)
        {
            ++it;
            continue;
        }

        it = local_mappoints.erase(it);
    }
}

double PointFeatureManager::worldPointToPointFeatureDepth(const PointFeatureId &point_feature,
                                                          const Vector3d &point_w,
                                                          Vector3d Ps[], Matrix3d Rs[],
                                                          Vector3d tic[], Matrix3d window_ric[]) const
{
    if (point_feature.point_feature_frame.empty())
        return -1.0;

    const int anchor_frame = point_feature.start_frame;
    const int anchor_camera_id = point_feature.anchorObservation().camera_id;
    if (anchor_frame < 0 || anchor_frame > WINDOW_SIZE)
        return -1.0;
    if (anchor_camera_id < 0 || anchor_camera_id >= NUM_OF_CAM)
        return -1.0;

    const Eigen::Vector3d anchor_t = Ps[anchor_frame] + Rs[anchor_frame] * tic[anchor_camera_id];
    const Eigen::Matrix3d anchor_R = Rs[anchor_frame] * window_ric[anchor_camera_id];
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

// Point-feature and local-mappoint lifecycle is owned here.
void PointFeatureManager::updateLocalPoints(Vector3d Ps[], Matrix3d Rs[],
                                            Vector3d tic[], Matrix3d window_ric[])
{
    for (auto &point_feature : pointFeature)
    {
        point_feature.used_num = static_cast<int>(point_feature.point_feature_frame.size());

        auto &mappoint = local_mappoints[point_feature.point_feature_id];
        if (!mappoint)
            mappoint = std::make_shared<Mappoint>(point_feature.point_feature_id);

        mappoint->ClearObversers();
        mappoint->SetTrackCount(static_cast<int>(point_feature.point_feature_frame.size()));
        mappoint->ResetInactiveCount();
        for (int i = 0; i < static_cast<int>(point_feature.point_feature_frame.size()); ++i)
            mappoint->AddObverser(point_feature.start_frame + i, i);

        if (point_feature.solve_flag == 2)
        {
            mappoint->SetBad();
        }
        else if (mappoint->IsGood())
        {
            if (std::isfinite(point_feature.estimated_depth) && point_feature.estimated_depth > 0)
            {
                syncMappointFromPointFeature(point_feature, Ps, Rs, tic, window_ric);
            }
            else if (mappoint->HasPosition())
            {
                const double depth = worldPointToPointFeatureDepth(point_feature, mappoint->GetPosition(),
                                                                   Ps, Rs, tic, window_ric);
                if (isReliableTriangulatedDepth(depth))
                {
                    point_feature.estimated_depth = depth;
                    point_feature.reliable_depth = true;
                    point_feature.solve_flag = 1;
                    point_feature.depth_fail_count = 0;
                }
            }
        }

        const auto support_stats = windowSupportStats(point_feature.point_feature_id);
        mappoint->SetWindowSupport(support_stats.first, support_stats.second);
    }
    pruneInactiveLocalMappoints();
}

bool PointFeatureManager::addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input pointFeature: %d", (int)image.size());
    ROS_DEBUG("num of pointFeature: %d", getFeatureCount());
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
        auto index_it = pointFeatureIndex.find(point_feature_id);

        if (index_it == pointFeatureIndex.end())
        {
            pointFeature.emplace_back(point_feature_id, frame_count);
            auto it = std::prev(pointFeature.end());
            pointFeatureIndex.emplace(point_feature_id, it);
            it->point_feature_frame.push_back(f_per_fra);
            new_point_feature_num++;
        }
        else
        {
            auto it = index_it->second;
            it->point_feature_frame.push_back(f_per_fra);
            last_point_feature_track_num++;
            if (it->point_feature_frame.size() >= 4)
                long_point_feature_track_num++;
        }
    }

    this->last_point_feature_track_num = last_point_feature_track_num;
    this->last_average_parallax = 0.0;
    this->new_point_feature_num = new_point_feature_num;
    this->long_point_feature_track_num = long_point_feature_track_num;

    if (frame_count < 2 || last_point_feature_track_num < 20 || long_point_feature_track_num < 40 || new_point_feature_num > 0.5 * last_point_feature_track_num)
        return true;

    for (auto &it_per_id : pointFeature)
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
        this->last_point_feature_track_num = last_point_feature_track_num;
        this->last_average_parallax = last_average_parallax;
        this->new_point_feature_num = new_point_feature_num;
        this->long_point_feature_track_num = long_point_feature_track_num;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

KeyframeGoodPointRecordList PointFeatureManager::collectGoodKeyframePoints() const
{
    KeyframeGoodPointRecordList records;
    records.reserve(std::min<size_t>(local_mappoints.size(), 1024));
    for (const auto &it_per_id : pointFeature)
    {
        if (!useFeatureForOptimization(it_per_id))
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

int PointFeatureManager::countGoodLocalMappoints() const
{
    int count = 0;
    for (const auto &kv : local_mappoints)
    {
        if (kv.second && kv.second->IsGood())
            ++count;
    }
    return count;
}

int PointFeatureManager::countCurrentStereoObservations() const
{
    int count = 0;
    for (const auto &it_per_id : pointFeature)
    {
        if (it_per_id.point_feature_frame.empty())
            continue;
        const PointFeatureFrame &obs = it_per_id.point_feature_frame.back();
        if (obs.camera_id == 0 && obs.is_stereo)
            ++count;
    }
    return count;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> PointFeatureManager::getCorresponding(int frame_count_l, int frame_count_r, bool only_good)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto &it : pointFeature)
    {
        if (only_good)
        {
            const auto mappoint_it = local_mappoints.find(it.point_feature_id);
            if (mappoint_it == local_mappoints.end() || !mappoint_it->second ||
                !mappoint_it->second->IsGood())
                continue;
        }
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

void PointFeatureManager::setDepth(const Eigen::VectorXd &x, Vector3d Ps[], Matrix3d Rs[],
                                   Vector3d tic[], Matrix3d window_ric[])
{
    int point_feature_index = -1;
    for (auto &it_per_id : pointFeature)
    {
        it_per_id.used_num = it_per_id.point_feature_frame.size();
        if (!useFeatureForOptimization(it_per_id))
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
            syncMappointFromPointFeature(it_per_id, Ps, Rs, tic, window_ric);
    }

}

void PointFeatureManager::removeFailures()
{
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin(); it != pointFeature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
        {
            pointFeatureIndex.erase(it->point_feature_id);
            pointFeature.erase(it);
        }
    }
}

void PointFeatureManager::clearDepth()
{
    for (auto &it_per_id : pointFeature)
    {
        it_per_id.estimated_depth = -1;
        it_per_id.reliable_depth = false;
        it_per_id.depth_fail_count = 0;
    }
}

Eigen::VectorXd PointFeatureManager::getDepthVector()
{
    Eigen::VectorXd dep_vec(getFeatureCount());
    int point_feature_index = -1;
    for (auto &it_per_id : pointFeature)
    {
        it_per_id.used_num = it_per_id.point_feature_frame.size();
        if (!useFeatureForOptimization(it_per_id))
            continue;
        dep_vec(++point_feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
}

void PointFeatureManager::triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    for (auto &it_per_id : pointFeature)
    {
        auto mpt_it = local_mappoints.find(it_per_id.point_feature_id);
        MappointPtr mappoint = (mpt_it != local_mappoints.end()) ? mpt_it->second : nullptr;

        if (mappoint && mappoint->HasPosition())
        {
            const double depth = worldPointToPointFeatureDepth(it_per_id, mappoint->GetPosition(),
                                                               Ps, Rs, tic, window_ric);
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
        if (triangulateMappointWorldPoint(it_per_id, Ps, Rs, tic, window_ric,
                                          triangulated_point, triangulated_depth))
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

void PointFeatureManager::initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[], int active_camera_id, bool prefer_good_points)
{
    if (frameCnt <= 0)
        return;

    vector<cv::Point2f> pts2D;
    vector<cv::Point3f> pts3D;
    for (const auto &point_feature : pointFeature)
    {
        if (prefer_good_points)
        {
            const auto mappoint_it = local_mappoints.find(point_feature.point_feature_id);
            if (mappoint_it == local_mappoints.end() || !mappoint_it->second ||
                !mappoint_it->second->IsGood() || point_feature.estimated_depth <= 0 ||
                !point_feature.reliable_depth)
                continue;
        }
        else if (point_feature.estimated_depth <= 0)
        {
            continue;
        }

        const int obs_index = frameCnt - point_feature.start_frame;
        if (obs_index < 0 || static_cast<int>(point_feature.point_feature_frame.size()) < obs_index + 1)
            continue;

        const int anchor_camera_id = point_feature.anchorObservation().camera_id;
        const int obs_camera_id = point_feature.point_feature_frame[obs_index].camera_id;
        if (obs_camera_id != active_camera_id)
            continue;

        const Vector3d point_in_cam = window_ric[anchor_camera_id] *
                                      (point_feature.anchorObservation().point * point_feature.estimated_depth) +
                                      tic[anchor_camera_id];
        const Vector3d point_in_world = Rs[point_feature.start_frame] * point_in_cam +
                                        Ps[point_feature.start_frame];

        pts3D.emplace_back(point_in_world.x(), point_in_world.y(), point_in_world.z());
        pts2D.emplace_back(point_feature.point_feature_frame[obs_index].point.x(),
                           point_feature.point_feature_frame[obs_index].point.y());
    }
    if (prefer_good_points && pts3D.size() < 6)
        return;

    Eigen::Matrix3d RCam;
    Eigen::Vector3d PCam;
    RCam = Rs[frameCnt - 1] * window_ric[active_camera_id];
    PCam = Rs[frameCnt - 1] * tic[active_camera_id] + Ps[frameCnt - 1];

    if (solvePoseByPnP(RCam, PCam, pts2D, pts3D))
    {
        Rs[frameCnt] = RCam * window_ric[active_camera_id].transpose();
        Ps[frameCnt] = -RCam * window_ric[active_camera_id].transpose() * tic[active_camera_id] + PCam;
    }
}

bool PointFeatureManager::solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
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

void PointFeatureManager::removeBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                                      Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                                      Eigen::Vector3d tic[], Eigen::Matrix3d window_ric[])
{
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin();
         it != pointFeature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            const int old_camera_id = it->anchorObservation().camera_id;
            Eigen::Vector3d uv_i = it->anchorObservation().point;
            it->point_feature_frame.erase(it->point_feature_frame.begin());
            if (it->point_feature_frame.size() < 2)
            {
                pointFeatureIndex.erase(it->point_feature_id);
                pointFeature.erase(it);
                continue;
            }
            else
            {
                const int new_camera_id = it->anchorObservation().camera_id;
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
            {
                pointFeatureIndex.erase(it->point_feature_id);
                pointFeature.erase(it);
            }
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
            {
                pointFeatureIndex.erase(it->point_feature_id);
                pointFeature.erase(it);
            }
        }
    }
}

void PointFeatureManager::removeOutlier(std::set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = pointFeature.begin(), it_next = pointFeature.begin();
         it != pointFeature.end(); it = it_next)
    {
        it_next++;
        int index = it->point_feature_id;
        itSet = outlierIndex.find(index);
        if (itSet != outlierIndex.end())
        {
            auto mpt_it = local_mappoints.find(index);
            if (mpt_it != local_mappoints.end() && mpt_it->second)
                mpt_it->second->SetBad();
            pointFeatureIndex.erase(it->point_feature_id);
            pointFeature.erase(it);
        }
    }
}

bool PointFeatureManager::useFeatureForOptimization(const PointFeatureId &point_feature) const
{
    auto it = local_mappoints.find(point_feature.point_feature_id);
    return it != local_mappoints.end() && it->second && it->second->IsGood();
}

std::pair<int, int> PointFeatureManager::windowSupportStats(int point_feature_id) const
{
    if (point_feature_id < 0)
        return std::make_pair(0, -1);

    const auto support_it = windowSupportCount.find(point_feature_id);
    if (support_it == windowSupportCount.end())
        return std::make_pair(0, -1);

    const auto last_frame_it = windowSupportLastFrame.find(point_feature_id);
    const int last_support_frame_id = last_frame_it == windowSupportLastFrame.end()
                                          ? -1
                                          : last_frame_it->second;
    return std::make_pair(support_it->second, last_support_frame_id);
}
