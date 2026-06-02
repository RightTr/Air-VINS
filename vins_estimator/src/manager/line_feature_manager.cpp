#include "line_feature_manager.h"

#include <cmath>
#include <algorithm>

#include "utility/line_geometry.h"

using namespace std;
using namespace Eigen;

namespace
{
Eigen::Matrix<double, 3, 4> worldToCameraPoseCompat(const Eigen::Vector3d &P,
                                                    const Eigen::Matrix3d &R,
                                                    const Eigen::Vector3d &t,
                                                    const Eigen::Matrix3d &ric)
{
    return worldToCameraPose(P, R, t, ric);
}
} // namespace

LineFeatureManager::LineFeatureManager()
{
    clearState();
}

void LineFeatureManager::clearState()
{
    line_feature.clear();
    for (int i = 0; i < NUM_OF_CAM; ++i)
        line_camera_intrinsics[i] << FOCAL_LENGTH, FOCAL_LENGTH, COL * 0.5, ROW * 0.5;
}

void LineFeatureManager::setLineCameraIntrinsics(const Eigen::Vector4d intrinsics[])
{
    for (int i = 0; i < NUM_OF_CAM; ++i)
        line_camera_intrinsics[i] = intrinsics[i];
}

bool LineFeatureManager::useLineForOptimization(const LinePerId &line_per_id) const
{
    return LINE_BA && line_per_id.mapline && line_per_id.mapline->IsValid() &&
           line_per_id.solve_flag == 1 &&
           line_per_id.used_num >= LINE_MIN_OBS &&
           line_per_id.start_frame < WINDOW_SIZE - 2 &&
           line_per_id.line_3d_world.allFinite();
}

void LineFeatureManager::addLineFeature(int frame_count, const LineFeatureFrameMap &lines, double td)
{
    if (!LINE_BA)
        return;

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
                line_feature.back().mapline->AddObserver(frame_count, 0);
        }
        else
        {
            it->line_per_frame.push_back(line_per_frame);
            if (it->mapline)
                it->mapline->AddObserver(frame_count, static_cast<int>(it->line_per_frame.size()) - 1);
        }
    }
}

int LineFeatureManager::getLineFeatureCount() const
{
    int cnt = 0;
    for (const auto &it : line_feature)
    {
        const_cast<LinePerId &>(it).used_num = static_cast<int>(it.line_per_frame.size());
        if (useLineForOptimization(it))
            ++cnt;
    }
    return cnt;
}

void LineFeatureManager::triangulateLine(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
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

Eigen::Matrix<double, Eigen::Dynamic, 4> LineFeatureManager::getLineFeatureVector() const
{
    Eigen::Matrix<double, Eigen::Dynamic, 4> line_vec(getLineFeatureCount(), 4);
    int line_index = -1;
    for (const auto &it_per_id : line_feature)
    {
        if (!useLineForOptimization(it_per_id))
            continue;
        line_vec.row(++line_index) = plk_to_orth(it_per_id.line_3d_world).transpose();
    }
    return line_vec;
}

void LineFeatureManager::setLineFeature(const Eigen::Matrix<double, Eigen::Dynamic, 4> &x,
                                        int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    (void)frameCnt;
    (void)Ps;
    (void)Rs;
    (void)tic;
    (void)ric;
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

void LineFeatureManager::removeLineOutliers(const std::map<int, std::set<int>> &outlier_observers)
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
                it->mapline->RemoveObserver(frame_id);
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

void LineFeatureManager::removeBackShiftDepth(Eigen::Matrix3d back_R0, Eigen::Vector3d back_P0,
                                              Eigen::Matrix3d new_R0, Eigen::Vector3d new_P0,
                                              Vector3d tic[], Matrix3d ric[])
{
    (void)back_R0;
    (void)back_P0;
    (void)new_R0;
    (void)new_P0;
    (void)tic;
    (void)ric;

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
                it->mapline->RemoveObserver(it->start_frame);
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

void LineFeatureManager::removeBack()
{
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
                it->mapline->RemoveObserver(it->start_frame);
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

void LineFeatureManager::removeFront(int frame_count)
{
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
                    it->mapline->RemoveObserver(frame_count);
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

void LineFeatureManager::removeFailures()
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin(); it != line_feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            line_feature.erase(it);
    }
}
