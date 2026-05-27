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
#include <algorithm>
#include <cmath>
#include <sys/stat.h>

namespace
{
std::string joinPath(const std::string &folder, const std::string &name)
{
    if (folder.empty())
        return name;
    if (folder.back() == '/')
        return folder + name;
    return folder + "/" + name;
}

bool fileExists(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

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
    hasPrediction = false;
    primary_camera_id = 0;
    prev_deep_features.resize(259, 0);
    left_deep_features.resize(259, 0);
    right_deep_features.resize(259, 0);
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
    predict_pts.clear();
    predict_pts_debug.clear();
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
    prev_deep_features.resize(259, 0);
    left_deep_features.resize(259, 0);
    right_deep_features.resize(259, 0);
    cur_time = 0;
    prev_time = 0;
    stereo_cam = 0;
    hasPrediction = false;
}

void FeatureTracker::clearState()
{
    resetTrackingState();
    n_id = 0;
    primary_camera_id = 0;
    m_camera.clear();
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
    options.stereo_ransac = DEEP_FEATURE_STEREO_RANSAC;
    options.ransac_threshold = static_cast<float>(DEEP_FEATURE_RANSAC_THRESHOLD);

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
        if(hasPrediction)
        {
            cur_pts = predict_pts;
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10)
               cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }
        else
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
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

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

    Eigen::Matrix<float, 259, Eigen::Dynamic> current_features;
    if (!deep_feature || !deep_feature->extractPoints(cur_img, current_features))
    {
        ROS_WARN("deep frontend feature extraction failed");
        return map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>();
    }

    vector<int> current_prev_index(current_features.cols(), -1);
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
            cv::Point2f prev_pt(history_features(1, match.queryIdx), history_features(2, match.queryIdx));
            cv::Point2f cur_pt(current_features(1, match.trainIdx), current_features(2, match.trainIdx));
            double displacement = cv::norm(cur_pt - prev_pt);
            if (displacement < DEEP_FEATURE_MIN_TEMPORAL_PARALLAX ||
                displacement > DEEP_FEATURE_MAX_TEMPORAL_PARALLAX) continue;
            current_prev_index[match.trainIdx] = match.queryIdx;
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
        if (current_prev_index[idx] >= 0)
        {
            cur_pts.push_back(pt);
            matched_features.col(kept_feature_count++) = current_features.col(idx);
            ids.push_back(prev_ids[current_prev_index[idx]]);
            track_cnt.push_back(prev_track_cnt[current_prev_index[idx]] + 1);
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
                deep_feature->matchPoints(matched_features, right_features, stereo_matches, DEEP_FEATURE_STEREO_RANSAC != 0);

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

    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

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
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

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

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    if (DEEP_FEATURE)
    {
        hasPrediction = false;
        return;
    }
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
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
    if (prev_deep_features.cols() == static_cast<int>(status.size()))
    {
        Eigen::Matrix<float, 259, Eigen::Dynamic> filtered_features(259, 0);
        vector<int> filtered_prev_index;
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i])
                filtered_prev_index.push_back(static_cast<int>(i));
        }
        filtered_features.resize(259, filtered_prev_index.size());
        for (size_t i = 0; i < filtered_prev_index.size(); i++)
            filtered_features.col(i) = prev_deep_features.col(filtered_prev_index[i]);
        prev_deep_features = filtered_features;
    }
    if (predict_pts.size() == status.size())
        reduceVector(predict_pts, status);
    if (predict_pts_debug.size() == status.size())
        reduceVector(predict_pts_debug, status);

    // Keep deep-feature histories consistent with backend outlier rejection.
    filterHistoryById(left_ids, left_track_cnt, left_deep_features);
    filterHistoryById(ids_right, right_track_cnt, right_deep_features);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
