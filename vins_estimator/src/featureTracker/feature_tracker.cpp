/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"

#include "../utility/visualization.h"
#include "deep_feature.h"
#include "deepFeature/include/utils.h"
#include <algorithm>
#include <set>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
cv::Mat makeDeepMatchImage(const cv::Mat &prev_img,
                           const cv::Mat &cur_img,
                           const Eigen::Matrix<float, 259, Eigen::Dynamic> &prev_features,
                           const Eigen::Matrix<float, 259, Eigen::Dynamic> &cur_features,
                           const std::vector<int> &current_prev_index,
                           const std::vector<int> &kept_indexes)
{
    if (prev_img.empty() || cur_img.empty())
        return cv::Mat();

    cv::Mat prev_bgr, cur_bgr;
    if (prev_img.channels() == 1)
        cv::cvtColor(prev_img, prev_bgr, cv::COLOR_GRAY2BGR);
    else
        prev_bgr = prev_img.clone();
    if (cur_img.channels() == 1)
        cv::cvtColor(cur_img, cur_bgr, cv::COLOR_GRAY2BGR);
    else
        cur_bgr = cur_img.clone();

    std::vector<cv::KeyPoint> prev_kpts;
    std::vector<cv::KeyPoint> cur_kpts;
    std::vector<cv::DMatch> matches;
    prev_kpts.reserve(kept_indexes.size());
    cur_kpts.reserve(kept_indexes.size());
    matches.reserve(kept_indexes.size());

    for (int idx : kept_indexes)
    {
        int prev_idx = current_prev_index[idx];
        if (prev_idx < 0 || prev_idx >= prev_features.cols())
            continue;

        prev_kpts.emplace_back(cv::Point2f(prev_features(1, prev_idx), prev_features(2, prev_idx)), 1.f);
        cur_kpts.emplace_back(cv::Point2f(cur_features(1, idx), cur_features(2, idx)), 1.f);
        matches.emplace_back(static_cast<int>(prev_kpts.size() - 1), static_cast<int>(cur_kpts.size() - 1), 0.f);
    }

    cv::Mat out;
    cv::drawMatches(prev_bgr, prev_kpts, cur_bgr, cur_kpts, matches, out,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return out;
}

double lineLength(const Eigen::Vector4d &line)
{
    return (line.head<2>() - line.tail<2>()).norm();
}

double lineAngle(const Eigen::Vector4d &line)
{
    return std::atan2(line(3) - line(1), line(2) - line(0));
}

cv::Point2f lineCenter(const Eigen::Vector4d &line)
{
    return cv::Point2f(0.5f * (line(0) + line(2)), 0.5f * (line(1) + line(3)));
}

cv::Scalar colorById(int id);

cv::Mat makeLineMatchImage(const cv::Mat &left_img,
                           const cv::Mat &right_img,
                           const std::vector<Eigen::Vector4d> &left_lines,
                           const std::vector<Eigen::Vector4d> &right_lines,
                           const std::vector<int> &left_line_ids,
                           const std::vector<int> &right_match_left,
                           const std::vector<std::map<int, double>> &left_points_on_lines,
                           const std::vector<std::map<int, double>> &right_points_on_lines,
                           const Eigen::Matrix<float, 259, Eigen::Dynamic> &left_points,
                           const Eigen::Matrix<float, 259, Eigen::Dynamic> &right_points)
{
    if (left_img.empty() || right_img.empty())
        return cv::Mat();

    cv::Mat left_bgr, right_bgr;
    if (left_img.channels() == 1)
        cv::cvtColor(left_img, left_bgr, cv::COLOR_GRAY2BGR);
    else
        left_bgr = left_img.clone();
    if (right_img.channels() == 1)
        cv::cvtColor(right_img, right_bgr, cv::COLOR_GRAY2BGR);
    else
        right_bgr = right_img.clone();

    cv::Mat canvas;
    cv::hconcat(left_bgr, right_bgr, canvas);
    const int x_offset = left_bgr.cols;

    auto drawSupportPoints = [](cv::Mat &img,
                                const std::map<int, double> &supporting_points,
                                const Eigen::Matrix<float, 259, Eigen::Dynamic> &points,
                                const cv::Scalar &color)
    {
        for (const auto &kv : supporting_points)
        {
            const int idx = kv.first;
            if (idx < 0 || idx >= points.cols())
                continue;
            cv::circle(img, cv::Point2f(points(1, idx), points(2, idx)), 3, color, -1, cv::LINE_AA);
        }
    };

    for (int left_idx = 0; left_idx < static_cast<int>(left_lines.size()); ++left_idx)
    {
        const int right_idx = (left_idx < static_cast<int>(right_match_left.size())) ? right_match_left[left_idx] : -1;
        if (right_idx < 0 || right_idx >= static_cast<int>(right_lines.size()))
            continue;

        const int line_id = (left_idx < static_cast<int>(left_line_ids.size()) && left_line_ids[left_idx] >= 0)
                                ? left_line_ids[left_idx]
                                : left_idx + 1;
        const cv::Scalar color = colorById(line_id);

        const Eigen::Vector4d &left_line = left_lines[left_idx];
        const Eigen::Vector4d &right_line = right_lines[right_idx];
        const cv::Point2f left_p1(left_line(0), left_line(1));
        const cv::Point2f left_p2(left_line(2), left_line(3));
        const cv::Point2f right_p1(right_line(0) + x_offset, right_line(1));
        const cv::Point2f right_p2(right_line(2) + x_offset, right_line(3));

        cv::line(canvas, left_p1, left_p2, color, 3, cv::LINE_AA);
        cv::line(canvas, right_p1, right_p2, color, 3, cv::LINE_AA);

        const cv::Point2f left_center = lineCenter(left_line);
        const cv::Point2f right_center(0.5f * (right_line(0) + right_line(2)) + x_offset,
                                       0.5f * (right_line(1) + right_line(3)));
        cv::line(canvas, left_center, right_center, color, 2, cv::LINE_AA);

        cv::putText(canvas, std::to_string(line_id), left_center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);

        if (left_idx < static_cast<int>(left_points_on_lines.size()))
            drawSupportPoints(canvas, left_points_on_lines[left_idx], left_points, color);
        if (right_idx < static_cast<int>(right_points_on_lines.size()))
            drawSupportPoints(canvas, right_points_on_lines[right_idx], right_points, color);
    }

    return canvas;
}

void assignPointsToLines(const std::vector<Eigen::Vector4d> &lines,
                         const Eigen::Matrix<float, 259, Eigen::Dynamic> &points,
                         std::vector<std::map<int, double>> &points_on_lines)
{
    points_on_lines.assign(lines.size(), std::map<int, double>());
    if (lines.empty() || points.cols() == 0)
        return;

    for (int l = 0; l < static_cast<int>(lines.size()); ++l)
    {
        const double lx1 = lines[l](0);
        const double ly1 = lines[l](1);
        const double lx2 = lines[l](2);
        const double ly2 = lines[l](3);
        const double A = ly2 - ly1;
        const double B = lx1 - lx2;
        const double C = lx2 * ly1 - lx1 * ly2;
        const double D = std::sqrt(A * A + B * B);
        if (D < 1e-9)
            continue;

        for (int p = 0; p < points.cols(); ++p)
        {
            cv::Point2f pt(points(1, p), points(2, p));
            if (pt.x < std::min(lx1, lx2) - 3.0 || pt.x > std::max(lx1, lx2) + 3.0 ||
                pt.y < std::min(ly1, ly2) - 3.0 || pt.y > std::max(ly1, ly2) + 3.0)
                continue;

            const double pl_distance = std::abs(A * pt.x + B * pt.y + C) / D;
            if (pl_distance > 3.0)
                continue;

            const double side1 = (lx1 - pt.x) * (lx1 - pt.x) + (ly1 - pt.y) * (ly1 - pt.y);
            const double side2 = (lx2 - pt.x) * (lx2 - pt.x) + (ly2 - pt.y) * (ly2 - pt.y);
            const double line_side = D * D;
            if (side1 <= 9.0 || side2 <= 9.0 || ((side1 < line_side + side2) && (side2 < line_side + side1)))
                points_on_lines[l][p] = pl_distance;
        }
    }
}

void matchLinesByPoints(const std::vector<std::map<int, double>> &points_on_lines0,
                        const std::vector<std::map<int, double>> &points_on_lines1,
                        int points0_count,
                        int points1_count,
                        const std::vector<cv::DMatch> &point_matches,
                        std::vector<int> &line_matches)
{
    line_matches.assign(points_on_lines0.size(), -1);
    if (points_on_lines0.empty() || points_on_lines1.empty() || points0_count <= 0 || points1_count <= 0)
        return;

    std::vector<std::vector<int>> point_lines0(points0_count), point_lines1(points1_count);
    for (int l = 0; l < static_cast<int>(points_on_lines0.size()); ++l)
    {
        for (const auto &kv : points_on_lines0[l])
            if (kv.first >= 0 && kv.first < points0_count)
                point_lines0[kv.first].push_back(l);
    }
    for (int l = 0; l < static_cast<int>(points_on_lines1.size()); ++l)
    {
        for (const auto &kv : points_on_lines1[l])
            if (kv.first >= 0 && kv.first < points1_count)
                point_lines1[kv.first].push_back(l);
    }

    Eigen::MatrixXi votes = Eigen::MatrixXi::Zero(points_on_lines0.size(), points_on_lines1.size());
    for (const auto &match : point_matches)
    {
        if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(point_lines0.size()) ||
            match.trainIdx < 0 || match.trainIdx >= static_cast<int>(point_lines1.size()))
            continue;
        for (int l0 : point_lines0[match.queryIdx])
            for (int l1 : point_lines1[match.trainIdx])
                votes(l0, l1)++;
    }

    std::vector<int> row_best_location(votes.rows(), -1);
    for (int l0 = 0; l0 < votes.rows(); ++l0)
    {
        Eigen::Index best_l1 = -1;
        int best_votes = votes.row(l0).maxCoeff(&best_l1);
        if (best_votes >= LINE_MATCH_VOTE_THRESHOLD)
            row_best_location[l0] = static_cast<int>(best_l1);
    }

    for (int l1 = 0; l1 < votes.cols(); ++l1)
    {
        Eigen::Index best_l0 = -1;
        int best_votes = votes.col(l1).maxCoeff(&best_l0);
        if (best_votes < LINE_MATCH_VOTE_THRESHOLD)
            continue;
        if (best_l0 < 0 || best_l0 >= static_cast<int>(row_best_location.size()) ||
            row_best_location[best_l0] != l1)
            continue;
        const int support0 = static_cast<int>(points_on_lines0[best_l0].size());
        const int support1 = static_cast<int>(points_on_lines1[l1].size());
        const int min_support = std::min(support0, support1);
        if (min_support <= 0)
            continue;
        const double score = static_cast<double>(best_votes * best_votes) / static_cast<double>(min_support);
        if (score < LINE_MATCH_SCORE_THRESHOLD)
            continue;
        line_matches[best_l0] = l1;
    }
}

std::vector<std::pair<int, double>> associateSupportingPoints(const std::map<int, double> &points_on_line,
                                                              const Eigen::Matrix<float, 259, Eigen::Dynamic> &point_features,
                                                              const std::vector<cv::Point2f> &tracked_points,
                                                              const std::vector<int> &tracked_ids,
                                                              const std::vector<int> &tracked_track_cnt,
                                                              int min_track_cnt,
                                                              double max_distance)
{
    std::map<int, double> supporting_points;
    if (point_features.cols() <= 0 || tracked_points.empty() || tracked_ids.empty())
        return {};

    for (const auto &kv : points_on_line)
    {
        const int point_idx = kv.first;
        if (point_idx < 0 || point_idx >= point_features.cols())
            continue;
        const cv::Point2f point(point_features(1, point_idx), point_features(2, point_idx));

        double best_distance2 = max_distance * max_distance;
        int best_track_idx = -1;
        for (int i = 0; i < static_cast<int>(tracked_points.size()) && i < static_cast<int>(tracked_ids.size()); ++i)
        {
            if (i >= static_cast<int>(tracked_track_cnt.size()) || tracked_track_cnt[i] < min_track_cnt)
                continue;
            const double dx = tracked_points[i].x - point.x;
            const double dy = tracked_points[i].y - point.y;
            const double distance2 = dx * dx + dy * dy;
            if (distance2 < best_distance2)
            {
                best_distance2 = distance2;
                best_track_idx = i;
            }
        }

        if (best_track_idx >= 0 && tracked_ids[best_track_idx] >= 0)
        {
            const int track_id = tracked_ids[best_track_idx];
            auto it = supporting_points.find(track_id);
            if (it == supporting_points.end() || kv.second < it->second)
                supporting_points[track_id] = kv.second;
        }
    }

    return std::vector<std::pair<int, double>>(supporting_points.begin(), supporting_points.end());
}

cv::Scalar colorById(int id)
{
    uint32_t x = static_cast<uint32_t>(id * 2654435761u);
    return cv::Scalar(50 + (x & 0x7F), 50 + ((x >> 8) & 0x7F), 80 + ((x >> 16) & 0x7F));
}

} // namespace

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    line_n_id = 0;
    primary_camera_id = 0;
    left_deep_features.resize(259, 0);
    right_deep_features.resize(259, 0);
    prev_line_points.resize(259, 0);
    cur_line_points.resize(259, 0);
    right_line_points.resize(259, 0);
}

void FeatureTracker::resetTrackingState()
{
    imTrack.release();
    mask.release();
    fisheye_mask.release();
    prev_img.release();
    cur_img.release();
    left_prev_img.release();
    right_prev_img.release();

    n_pts.clear();
    projection_candidates.clear();
    prev_pts.clear();
    cur_pts.clear();
    cur_right_pts.clear();
    prev_un_pts.clear();
    cur_un_pts.clear();
    cur_un_right_pts.clear();
    pts_velocity.clear();
    right_pts_velocity.clear();
    ids.clear();
    ids_right.clear();
    track_cnt.clear();
    left_ids.clear();
    left_track_cnt.clear();
    right_track_cnt.clear();
    cur_un_pts_map.clear();
    prev_un_pts_map.clear();
    cur_un_right_pts_map.clear();
    prev_un_right_pts_map.clear();
    prevLeftPtsMap.clear();
    line_feature_frame.clear();
    prev_lines.clear();
    cur_lines.clear();
    cur_right_lines.clear();
    line_ids.clear();
    prev_line_ids.clear();
    line_track_cnt.clear();
    right_line_ids.clear();
    cur_right_line_match_left.clear();
    prev_points_on_lines.clear();
    cur_points_on_lines.clear();
    right_points_on_lines.clear();
    prev_line_points.resize(259, 0);
    cur_line_points.resize(259, 0);
    right_line_points.resize(259, 0);
    left_deep_features.resize(259, 0);
    right_deep_features.resize(259, 0);
    cur_time = 0;
    prev_time = 0;
    stereo_cam = 0;
}

void FeatureTracker::clearState()
{
    resetTrackingState();
    n_id = 0;
    line_n_id = 0;
    primary_camera_id = 0;
    m_camera.clear();
}

void FeatureTracker::initLineFeatureFrontend()
{
    if (line_deep_feature)
        return;

    DeepFeatureOptions options;
    options.matcher = DEEP_FEATURE_MATCHER;
    options.model_dir = DEEP_FEATURE_MODEL_DIR;
    options.image_width = COL;
    options.image_height = ROW;
    options.max_keypoints = DEEP_FEATURE_MAX_KEYPOINTS;
    options.keypoint_threshold = static_cast<float>(DEEP_FEATURE_KEYPOINT_THRESHOLD);
    options.remove_borders = DEEP_FEATURE_REMOVE_BORDERS;
    options.line_threshold = static_cast<float>(LINE_THRESHOLD);
    options.line_length_threshold = static_cast<float>(LINE_MIN_LENGTH);

    line_deep_feature = std::make_shared<DeepFeature>();
    line_deep_feature->init(2, options);
}

void FeatureTracker::initDeepFeatureFrontend()
{
    if (deep_feature)
        return;

    DeepFeatureOptions options;
    options.matcher = DEEP_FEATURE_MATCHER;
    options.model_dir = DEEP_FEATURE_MODEL_DIR;
    options.image_width = COL;
    options.image_height = ROW;
    options.max_keypoints = DEEP_FEATURE_MAX_KEYPOINTS;
    options.keypoint_threshold = static_cast<float>(DEEP_FEATURE_KEYPOINT_THRESHOLD);
    options.remove_borders = DEEP_FEATURE_REMOVE_BORDERS;

    deep_feature = std::make_shared<DeepFeature>();
    deep_feature->init(DEEP_FEATURE, options);
}

void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1, int primary_camera_id_, bool allow_new_features)
{
    if (primary_camera_id_ < 0 || primary_camera_id_ >= static_cast<int>(m_camera.size()))
    {
        ROS_WARN_STREAM("invalid primary camera id " << primary_camera_id_ << ", fallback to 0");
        primary_camera_id_ = 0;
    }
    if (primary_camera_id_ != primary_camera_id && allow_new_features)
    {
        resetTrackingState();
        primary_camera_id = primary_camera_id_;
    }
    if (DEEP_FEATURE)
    {
        if (!deep_feature || !deep_feature->ready())
        {
            ROS_ERROR_THROTTLE(1.0, "deep feature is enabled but frontend is not ready; skip image tracking instead of using KLT");
            return map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>();
        }
        return trackImageDeep(_cur_time, _img, _img1, primary_camera_id_, allow_new_features);
    }
    if (primary_camera_id_ != primary_camera_id)
    {
        resetTrackingState();
        primary_camera_id = primary_camera_id_;
    }

    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();

    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        // reverse check
        if(FLOW_BACK)
        {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
            for(size_t i = 0; i < status.size(); i++)
            {
                if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }
        
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    if (allow_new_features)
    {
        //rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[primary_camera_id]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!_img1.empty() && stereo_cam && primary_camera_id == 0)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            if(FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if (LINE_BA)
        processLineFeatures(cur_img, rightImg);

    if(SHOW_TRACK)
    {
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
        if (LINE_BA)
            drawLineTrack(cur_img, rightImg);
    }

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = primary_camera_id;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam && primary_camera_id == 0)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImageDeep(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1, int primary_camera_id_, bool allow_new_features)
{
    if (primary_camera_id_ < 0 || primary_camera_id_ >= static_cast<int>(m_camera.size()))
        primary_camera_id_ = 0;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;

    vector<int> prev_ids;
    vector<int> prev_track_cnt;
    Eigen::Matrix<float, 259, Eigen::Dynamic> history_features(259, 0);
    cv::Mat history_img;
    bool mixed_history_sources = false;

    auto appendHistory = [&](const vector<int> &candidate_ids,
                             const vector<int> &candidate_track_cnt,
                             const Eigen::Matrix<float, 259, Eigen::Dynamic> &candidate_features,
                             const cv::Mat &candidate_img)
    {
        if (candidate_ids.empty())
            return;
        if (candidate_features.cols() != static_cast<int>(candidate_ids.size()))
            return;
        if (history_img.empty() && !candidate_img.empty())
            history_img = candidate_img;
        else if (!candidate_img.empty() && !history_img.empty() && candidate_img.data != history_img.data)
            mixed_history_sources = true;

        for (int col = 0; col < static_cast<int>(candidate_ids.size()); ++col)
        {
            const int feature_id = candidate_ids[col];
            if (std::find(prev_ids.begin(), prev_ids.end(), feature_id) != prev_ids.end())
                continue;

            const int next_col = history_features.cols();
            history_features.conservativeResize(259, next_col + 1);
            history_features.col(next_col) = candidate_features.col(col);
            prev_ids.push_back(feature_id);
            if (col < static_cast<int>(candidate_track_cnt.size()))
                prev_track_cnt.push_back(candidate_track_cnt[col]);
            else
                prev_track_cnt.push_back(1);
        }
    };

    if (allow_new_features)
    {
        if (primary_camera_id_ == 1)
            appendHistory(ids_right, right_track_cnt, right_deep_features, right_prev_img);
        else
            appendHistory(left_ids, left_track_cnt, left_deep_features, left_prev_img);
    }
    else if (primary_camera_id_ == 1)
    {
        appendHistory(ids_right, right_track_cnt, right_deep_features, right_prev_img);
        appendHistory(left_ids, left_track_cnt, left_deep_features, left_prev_img);
    }
    else
    {
        appendHistory(left_ids, left_track_cnt, left_deep_features, left_prev_img);
        appendHistory(ids_right, right_track_cnt, right_deep_features, right_prev_img);
    }

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    cur_right_pts.clear();
    ids_right.clear();
    cur_un_right_pts.clear();
    right_pts_velocity.clear();
    cur_un_right_pts_map.clear();

    const std::vector<ProjectionCandidate> projection_candidates_local = projection_candidates;
    projection_candidates.clear();

    Eigen::Matrix<float, 259, Eigen::Dynamic> current_features;
    if (!deep_feature || !deep_feature->extractPoints(cur_img, current_features))
    {
        ROS_WARN("deep frontend feature extraction failed");
        return map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>();
    }

    vector<int> current_prev_index(current_features.cols(), -1);
    vector<int> current_feature_ids(current_features.cols(), -1);
    vector<int> current_feature_track_cnt(current_features.cols(), 1);
    vector<bool> current_feature_taken(current_features.cols(), false);
    Eigen::Matrix<float, 259, Eigen::Dynamic> matched_features;
    matched_features.resize(259, 0);

    if (history_features.cols() > 0 && history_features.cols() == static_cast<int>(prev_ids.size()))
    {
        vector<cv::DMatch> matches;
        if (deep_feature && deep_feature->ready())
            deep_feature->matchPoints(history_features, current_features, matches, true);

        for (const auto &match : matches)
        {
            if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(prev_ids.size()))
                continue;
            if (match.trainIdx < 0 || match.trainIdx >= current_features.cols())
                continue;
            current_prev_index[match.trainIdx] = match.queryIdx;
            current_feature_ids[match.trainIdx] = prev_ids[match.queryIdx];
            current_feature_track_cnt[match.trainIdx] = prev_track_cnt[match.queryIdx] + 1;
            current_feature_taken[match.trainIdx] = true;
        }
    }

    if (!projection_candidates_local.empty())
    {
        const float projection_radius = 15.0f;
        const float projection_descriptor_threshold = 0.35f;
        const float projection_descriptor_ratio = 0.60f;

        for (const auto &candidate : projection_candidates_local)
        {
            if (candidate.point_feature_id < 0 || candidate.camera_id != primary_camera_id_)
                continue;

            if (!std::isfinite(candidate.point_cam.x()) || !std::isfinite(candidate.point_cam.y()) ||
                !std::isfinite(candidate.point_cam.z()) || candidate.point_cam.z() <= 0.1)
                continue;

            Eigen::Vector2d projected_uv;
            m_camera[candidate.camera_id]->spaceToPlane(candidate.point_cam, projected_uv);
            cv::Point2f projected_pt(projected_uv.x(), projected_uv.y());
            if (!inBorder(projected_pt))
                continue;

            int best_idx = -1;
            float best_desc_dist = std::numeric_limits<float>::max();
            int second_idx = -1;
            float second_desc_dist = std::numeric_limits<float>::max();

            const Eigen::Matrix<float, 256, 1> &candidate_descriptor = candidate.descriptor;

            for (int idx = 0; idx < current_features.cols(); ++idx)
            {
                if (current_feature_taken[idx])
                    continue;

                cv::Point2f pt(current_features(1, idx), current_features(2, idx));
                if (!inBorder(pt))
                    continue;
                if (cv::norm(pt - projected_pt) > projection_radius)
                    continue;

                const Eigen::Matrix<float, 256, 1> current_descriptor = current_features.block<256, 1>(3, idx);
                const float desc_dist = DescriptorDistance(candidate_descriptor, current_descriptor);
                if (desc_dist < best_desc_dist)
                {
                    second_desc_dist = best_desc_dist;
                    second_idx = best_idx;
                    best_desc_dist = desc_dist;
                    best_idx = idx;
                }
                else if (desc_dist < second_desc_dist)
                {
                    second_desc_dist = desc_dist;
                    second_idx = idx;
                }
            }

            if (best_idx >= 0 &&
                best_desc_dist <= projection_descriptor_threshold &&
                (second_idx < 0 || second_desc_dist == std::numeric_limits<float>::max() ||
                 best_desc_dist <= projection_descriptor_ratio * second_desc_dist))
            {
                current_prev_index[best_idx] = -2;
                current_feature_ids[best_idx] = candidate.point_feature_id;
                current_feature_track_cnt[best_idx] = std::max(1, candidate.track_count + 1);
                current_feature_taken[best_idx] = true;
            }
        }
    }

    vector<int> kept_indexes;
    kept_indexes.reserve(current_features.cols());
    for (int i = 0; i < current_features.cols(); ++i)
    {
        cv::Point2f pt(current_features(1, i), current_features(2, i));
        if (!inBorder(pt))
            continue;
        kept_indexes.push_back(i);
    }

    matched_features.resize(259, kept_indexes.size());
    int kept_feature_count = 0;
    for (size_t i = 0; i < kept_indexes.size(); ++i)
    {
        int idx = kept_indexes[i];
        cv::Point2f pt(current_features(1, idx), current_features(2, idx));
        if (current_feature_ids[idx] >= 0)
        {
            cur_pts.push_back(pt);
            matched_features.col(kept_feature_count++) = current_features.col(idx);
            ids.push_back(current_feature_ids[idx]);
            track_cnt.push_back(current_feature_track_cnt[idx]);
        }
        else if (allow_new_features)
        {
            cur_pts.push_back(pt);
            matched_features.col(kept_feature_count++) = current_features.col(idx);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
    }
    matched_features.conservativeResize(259, kept_feature_count);

    cur_un_pts = undistortedPts(cur_pts, m_camera[primary_camera_id_]);
    if (primary_camera_id_ == 1)
        pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_right_pts_map);
    else
        pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if (!_img1.empty() && stereo_cam && primary_camera_id_ == 0)
    {
        Eigen::Matrix<float, 259, Eigen::Dynamic> right_features;
        if (deep_feature && deep_feature->extractPoints(rightImg, right_features) && right_features.cols() > 0)
        {
            vector<cv::DMatch> stereo_matches;
            if (deep_feature && deep_feature->ready())
                deep_feature->matchPoints(matched_features, right_features, stereo_matches, false);

            ids_right.clear();
            cur_right_pts.clear();
            cur_un_right_pts.clear();
            right_pts_velocity.clear();
            cur_un_right_pts_map.clear();
            vector<int> accepted_right_feature_indexes;

            for (const auto &match : stereo_matches)
            {
                if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(cur_pts.size()))
                    continue;
                if (match.trainIdx < 0 || match.trainIdx >= right_features.cols())
                    continue;
                const cv::Point2f left_pt = cur_pts[match.queryIdx];
                const cv::Point2f right_pt(right_features(1, match.trainIdx), right_features(2, match.trainIdx));
                const double y_error = std::abs(left_pt.y - right_pt.y);
                const double disparity = left_pt.x - right_pt.x;
                if (y_error > DEEP_FEATURE_STEREO_Y_THRESHOLD || disparity < DEEP_FEATURE_MIN_STEREO_DISPARITY ||
                    disparity > DEEP_FEATURE_MAX_STEREO_DISPARITY) continue;
                ids_right.push_back(ids[match.queryIdx]);
                cur_right_pts.push_back(right_pt);
                accepted_right_feature_indexes.push_back(match.trainIdx);
            }

            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
            right_deep_features.resize(259, accepted_right_feature_indexes.size());
            right_track_cnt.clear();
            right_track_cnt.reserve(ids_right.size());
            for (size_t i = 0; i < accepted_right_feature_indexes.size(); ++i)
            {
                right_deep_features.col(i) = right_features.col(accepted_right_feature_indexes[i]);
                auto it = std::find(ids.begin(), ids.end(), ids_right[i]);
                right_track_cnt.push_back(it == ids.end() ? 1 : track_cnt[it - ids.begin()]);
            }
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    if (LINE_BA)
        processLineFeatures(cur_img, rightImg);

    if (SHOW_TRACK)
    {
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
        if (LINE_BA)
            drawLineTrack(cur_img, rightImg);
    }

    if (DEEP_FEATURE && !mixed_history_sources)
    {
        cv::Mat deep_match_img = makeDeepMatchImage(history_img, cur_img, history_features, current_features, current_prev_index, kept_indexes);
        if (!deep_match_img.empty())
            pubDeepMatchImage(deep_match_img, cur_time);
    }

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    projection_candidates.clear();

    if (primary_camera_id_ == 1)
    {
        ids_right = ids;
        cur_right_pts = cur_pts;
        cur_un_right_pts = cur_un_pts;
        right_pts_velocity = pts_velocity;
        right_deep_features = matched_features;
        right_track_cnt = track_cnt;
        right_prev_img = cur_img;
        prev_un_right_pts_map = cur_un_pts_map;
    }
    else
    {
        left_ids = ids;
        left_track_cnt = track_cnt;
        left_deep_features = matched_features;
        left_prev_img = cur_img;
    }

    if (primary_camera_id_ == 0)
        prev_un_right_pts_map = cur_un_right_pts_map;

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x = cur_un_pts[i].x;
        double y = cur_un_pts[i].y;
        double z = 1;
        double p_u = cur_pts[i].x;
        double p_v = cur_pts[i].y;
        int camera_id = primary_camera_id_;
        double velocity_x = pts_velocity[i].x;
        double velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam && primary_camera_id_ == 0)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x = cur_un_right_pts[i].x;
            double y = cur_un_right_pts[i].y;
            double z = 1;
            double p_u = cur_right_pts[i].x;
            double p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x = right_pts_velocity[i].x;
            double velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }

    return featureFrame;
}

void FeatureTracker::processLineFeatures(const cv::Mat &left_img, const cv::Mat &right_img)
{
    line_feature_frame.clear();
    if (!line_deep_feature || !line_deep_feature->ready())
        initLineFeatureFrontend();
    if (!line_deep_feature || !line_deep_feature->ready())
    {
        ROS_WARN_THROTTLE(2.0, "line_ba is enabled but PLNet line frontend is not ready");
        return;
    }

    std::vector<Eigen::Vector4d> raw_left_lines;
    Eigen::Matrix<float, 259, Eigen::Dynamic> left_points;
    if (!line_deep_feature->extractPointsLines(left_img, left_points, raw_left_lines))
        return;

    auto filterLines = [&](const std::vector<Eigen::Vector4d> &raw,
                           const Eigen::Matrix<float, 259, Eigen::Dynamic> &points,
                           std::vector<std::map<int, double>> &points_on_lines) {
        std::vector<Eigen::Vector4d> filtered;
        std::vector<std::map<int, double>> raw_points_on_lines;
        std::vector<int> kept_indices;
        assignPointsToLines(raw, points, raw_points_on_lines);
        kept_indices.reserve(raw.size());
        for (int i = 0; i < static_cast<int>(raw.size()); ++i)
        {
            const auto &line = raw[i];
            if (lineLength(line) < LINE_MIN_LENGTH)
                continue;
            cv::Point2f p1(line(0), line(1));
            cv::Point2f p2(line(2), line(3));
            if (!inBorder(p1) || !inBorder(p2))
                continue;
            filtered.push_back(line);
            kept_indices.push_back(i);
        }
        const int kept_count = std::min(static_cast<int>(filtered.size()), LINE_MAX_CNT);
        filtered.resize(kept_count);
        kept_indices.resize(kept_count);
        points_on_lines.clear();
        points_on_lines.reserve(filtered.size());
        for (int idx : kept_indices)
        {
            if (idx >= 0 && idx < static_cast<int>(raw_points_on_lines.size()))
                points_on_lines.push_back(raw_points_on_lines[idx]);
            else
                points_on_lines.emplace_back();
        }
        return filtered;
    };

    cur_lines = filterLines(raw_left_lines, left_points, cur_points_on_lines);
    cur_line_points = left_points;

    std::vector<int> current_ids(cur_lines.size(), -1);
    std::vector<int> current_track_cnt(cur_lines.size(), 1);
    std::vector<char> current_assigned(cur_lines.size(), 0);
    int temporal_match_count = 0;
    constexpr int kMinSupportingTrackCnt = 1;
    constexpr double kMaxSupportingPointDistance = 3.0;

    if (!prev_lines.empty() && prev_line_points.cols() > 0 && cur_line_points.cols() > 0)
    {
        std::vector<int> line_matches;
        std::vector<cv::DMatch> point_matches;
        line_deep_feature->matchPoints(prev_line_points, cur_line_points, point_matches, true);
        matchLinesByPoints(prev_points_on_lines, cur_points_on_lines,
                           prev_line_points.cols(), cur_line_points.cols(), point_matches, line_matches);
        for (int i = 0; i < static_cast<int>(line_matches.size()); ++i)
        {
            const int j = line_matches[i];
            if (j < 0 || j >= static_cast<int>(current_ids.size()) || current_assigned[j])
                continue;
            if (i >= static_cast<int>(prev_line_ids.size()))
                continue;
            current_ids[j] = prev_line_ids[i];
            current_track_cnt[j] = (i < static_cast<int>(line_track_cnt.size()) ? line_track_cnt[i] : 1) + 1;
            current_assigned[j] = 1;
            temporal_match_count++;
        }
    }

    for (int i = 0; i < static_cast<int>(current_ids.size()); ++i)
    {
        if (current_ids[i] < 0)
        {
            current_ids[i] = line_n_id++;
            current_track_cnt[i] = 1;
        }
    }

    cur_right_lines.clear();
    right_line_ids.clear();
    cur_right_line_match_left.clear();
    right_line_points.resize(259, 0);
    std::vector<int> stereo_match(cur_lines.size(), -1);
    int stereo_match_count = 0;

    if (!right_img.empty() && stereo_cam)
    {
        std::vector<Eigen::Vector4d> raw_right_lines;
        Eigen::Matrix<float, 259, Eigen::Dynamic> right_points;
        if (line_deep_feature->extractPointsLines(right_img, right_points, raw_right_lines))
        {
            cur_right_lines = filterLines(raw_right_lines, right_points, right_points_on_lines);
            right_line_points = right_points;
            std::vector<cv::DMatch> point_matches;
            if (cur_line_points.cols() > 0 && right_line_points.cols() > 0)
                line_deep_feature->matchPoints(cur_line_points, right_line_points, point_matches, false);
            matchLinesByPoints(cur_points_on_lines, right_points_on_lines,
                               cur_line_points.cols(), right_line_points.cols(), point_matches, stereo_match);
        }
    }
    cur_right_line_match_left.assign(cur_right_lines.size(), -1);

    auto liftLine = [&](const Eigen::Vector4d &line, int camera_id) {
        Eigen::Matrix<double, 8, 1> obs;
        Eigen::Vector3d p1, p2;
        m_camera[camera_id]->liftProjective(Eigen::Vector2d(line(0), line(1)), p1);
        m_camera[camera_id]->liftProjective(Eigen::Vector2d(line(2), line(3)), p2);
        obs << p1.x() / p1.z(), p1.y() / p1.z(), p2.x() / p2.z(), p2.y() / p2.z(),
            line(0), line(1), line(2), line(3);
        return obs;
    };

    for (int i = 0; i < static_cast<int>(cur_lines.size()); ++i)
    {
        const int id = current_ids[i];
        const std::vector<std::pair<int, double>> left_supporting_points =
            (i < static_cast<int>(cur_points_on_lines.size()))
                ? associateSupportingPoints(cur_points_on_lines[i], cur_line_points, cur_pts, ids, track_cnt,
                                            kMinSupportingTrackCnt, kMaxSupportingPointDistance)
                : std::vector<std::pair<int, double>>();

        const int right_idx = (i < static_cast<int>(stereo_match.size())) ? stereo_match[i] : -1;
        std::vector<std::pair<int, double>> right_supporting_points;
        bool stereo_confirmed = false;
        if (right_idx >= 0 && right_idx < static_cast<int>(cur_right_lines.size()))
        {
            right_supporting_points =
                (right_idx < static_cast<int>(right_points_on_lines.size()))
                    ? associateSupportingPoints(right_points_on_lines[right_idx], right_line_points,
                                                cur_right_pts, ids_right, right_track_cnt,
                                                kMinSupportingTrackCnt, kMaxSupportingPointDistance)
                    : std::vector<std::pair<int, double>>();
            stereo_confirmed = true;
        }

        line_feature_frame[id].emplace_back(0, liftLine(cur_lines[i], 0), left_supporting_points, current_track_cnt[i]);
        if (stereo_confirmed)
        {
            line_feature_frame[id].emplace_back(1, liftLine(cur_right_lines[right_idx], 1), right_supporting_points, current_track_cnt[i]);
            if (right_idx < static_cast<int>(cur_right_line_match_left.size()))
                cur_right_line_match_left[right_idx] = i;
            stereo_match_count++;
        }
    }

    line_ids.swap(current_ids);
    line_track_cnt.swap(current_track_cnt);
    prev_lines = cur_lines;
    prev_line_ids = line_ids;
    prev_line_points = cur_line_points;
    prev_points_on_lines = cur_points_on_lines;

    for (int i = 0; i < static_cast<int>(cur_right_line_match_left.size()); ++i)
    {
        const int left_idx = cur_right_line_match_left[i];
        right_line_ids.push_back((left_idx >= 0 && left_idx < static_cast<int>(line_ids.size())) ? line_ids[left_idx] : -1);
    }

    if (LINE_BA && !cur_img.empty() && !right_img.empty() && stereo_cam)
    {
        cv::Mat line_match_img = makeLineMatchImage(cur_img, right_img, cur_lines, cur_right_lines,
                                                     line_ids, cur_right_line_match_left,
                                                     cur_points_on_lines, right_points_on_lines,
                                                     cur_line_points, right_line_points);
        if (!line_match_img.empty())
            pubLineMatchImage(line_match_img, cur_time);
    }

    ROS_DEBUG("line frontend: raw_left=%lu kept_left=%lu kept_right=%lu temporal=%d stereo=%d obs=%lu",
              raw_left_lines.size(), cur_lines.size(), cur_right_lines.size(),
              temporal_match_count, stereo_match_count, line_feature_frame.size());
}

void FeatureTracker::drawLineTrack(const cv::Mat &imLeft, const cv::Mat &imRight)
{
    if (imTrack.empty())
        return;
    const int cols = imLeft.cols;
    for (size_t i = 0; i < cur_lines.size() && i < line_ids.size(); ++i)
    {
        const int track_cnt = (i < line_track_cnt.size()) ? line_track_cnt[i] : 1;
        const double len = std::min(1.0, std::max(0.0, 1.0 * track_cnt / 10.0));
        const cv::Scalar color(255 * (1.0 - len), 0, 255 * len);
        cv::line(imTrack, cv::Point2f(cur_lines[i](0), cur_lines[i](1)),
                 cv::Point2f(cur_lines[i](2), cur_lines[i](3)), color, 3, cv::LINE_AA);
        cv::putText(imTrack, std::to_string(line_ids[i]),
                    lineCenter(cur_lines[i]),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < cur_right_lines.size() && i < right_line_ids.size(); ++i)
        {
            const int left_idx = (i < cur_right_line_match_left.size()) ? cur_right_line_match_left[i] : -1;
            const bool has_left_match = left_idx >= 0 && left_idx < static_cast<int>(line_track_cnt.size());
            const int track_cnt = has_left_match ? line_track_cnt[left_idx] : 1;
            const double len = std::min(1.0, std::max(0.0, 1.0 * track_cnt / 10.0));
            const cv::Scalar color(255 * (1.0 - len), 0, 255 * len);
            cv::Point2f p1(cur_right_lines[i](0) + cols, cur_right_lines[i](1));
            cv::Point2f p2(cur_right_lines[i](2) + cols, cur_right_lines[i](3));
            cv::line(imTrack, p1, p2, color, 3, cv::LINE_AA);
            if (has_left_match)
            {
                cv::putText(imTrack, std::to_string(right_line_ids[i] >= 0 ? right_line_ids[i] : line_ids[left_idx]),
                            cv::Point2f(0.5f * (p1.x + p2.x), 0.5f * (p1.y + p2.y)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
            }
        }
    }
}

LineFeatureFrameMap FeatureTracker::getLineFrame() const
{
    return line_feature_frame;
}

bool FeatureTracker::getCameraIntrinsics(int camera_id, Eigen::Vector4d &intrinsics) const
{
    if (camera_id < 0 || camera_id >= static_cast<int>(m_camera.size()) || !m_camera[camera_id])
        return false;

    if (boost::shared_ptr<camodocal::PinholeCamera> pinhole =
            boost::dynamic_pointer_cast<camodocal::PinholeCamera>(m_camera[camera_id]))
    {
        const camodocal::PinholeCamera::Parameters &params = pinhole->getParameters();
        intrinsics << params.fx(), params.fy(), params.cx(), params.cy();
        return true;
    }

    if (boost::shared_ptr<camodocal::CataCamera> cata =
            boost::dynamic_pointer_cast<camodocal::CataCamera>(m_camera[camera_id]))
    {
        const camodocal::CataCamera::Parameters &params = cata->getParameters();
        intrinsics << params.gamma1(), params.gamma2(), params.u0(), params.v0();
        return true;
    }

    return false;
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    m_camera.clear();
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
    else
        stereo_cam = 0;

    if (DEEP_FEATURE && !deep_feature)
        initDeepFeatureFrontend();
    if (LINE_BA && !line_deep_feature)
        initLineFeatureFrontend();
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

void FeatureTracker::setProjectionCandidates(const std::vector<ProjectionCandidate> &candidates)
{
    projection_candidates = candidates;
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    auto filterHistoryById = [&removePtsIds](vector<int> &history_ids,
                                             vector<int> &history_track_cnt,
                                             Eigen::Matrix<float, 259, Eigen::Dynamic> &history_features)
    {
        const int old_size = static_cast<int>(history_ids.size());
        if (old_size == 0)
        {
            history_track_cnt.clear();
            history_features.resize(259, 0);
            return;
        }

        vector<int> keep_indexes;
        vector<int> kept_ids;
        vector<int> kept_track_cnt;
        keep_indexes.reserve(old_size);
        kept_ids.reserve(old_size);
        kept_track_cnt.reserve(old_size);

        for (int i = 0; i < old_size; i++)
        {
            if (removePtsIds.find(history_ids[i]) != removePtsIds.end())
                continue;
            keep_indexes.push_back(i);
            kept_ids.push_back(history_ids[i]);
            if (i < static_cast<int>(history_track_cnt.size()))
                kept_track_cnt.push_back(history_track_cnt[i]);
            else
                kept_track_cnt.push_back(1);
        }

        history_ids.swap(kept_ids);
        history_track_cnt.swap(kept_track_cnt);

        if (history_features.cols() == old_size)
        {
            Eigen::Matrix<float, 259, Eigen::Dynamic> filtered_features(259, keep_indexes.size());
            for (size_t i = 0; i < keep_indexes.size(); i++)
                filtered_features.col(i) = history_features.col(keep_indexes[i]);
            history_features = filtered_features;
        }
        else if (history_features.cols() != static_cast<int>(history_ids.size()))
        {
            history_features.resize(259, 0);
        }
    };

    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    // Keep deep-feature histories consistent with backend outlier rejection.
    filterHistoryById(left_ids, left_track_cnt, left_deep_features);
    filterHistoryById(ids_right, right_track_cnt, right_deep_features);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
