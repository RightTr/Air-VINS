/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include "ros_utils.h"
#include <cv_bridge/cv_bridge.h>
#include <cstdint>
#include <map>
#include <tf/transform_broadcaster.h>

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_window_keypose;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;
nav_msgs::Path keyframe_path;
visualization_msgs::Marker keyframe_marker;

ros::Publisher pub_keyframe_point;
ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_path;
ros::Publisher pub_keyframe_marker;
ros::Publisher pub_extrinsic;

ros::Publisher pub_image_track;
ros::Publisher pub_deep_match;
ros::Publisher pub_line_match;
ros::Publisher pub_map_line;
ros::Publisher pub_window_map_line;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;
static std::map<int, std::pair<Eigen::Vector3d, Eigen::Vector3d>> persistent_map_lines;

// Helper: convert Plucker line (6x1) to a short segment for visualization
bool lineToSegment(const Eigen::Matrix<double, 6, 1> &line,
                          Eigen::Vector3d &seg_start,
                          Eigen::Vector3d &seg_end)
{
    const Eigen::Vector3d moment = line.head<3>();
    const Eigen::Vector3d direction = line.tail<3>();
    const double dir_norm = direction.norm();
    if (!moment.allFinite() || !direction.allFinite() || dir_norm < 1e-9)
        return false;
    const Eigen::Vector3d anchor = direction.cross(moment) / (dir_norm * dir_norm);
    const Eigen::Vector3d offset = direction.normalized() * 0.5;
    seg_start = anchor - offset;
    seg_end = anchor + offset;
    return true;
}

// Helper: append a line segment (no color) to a Marker
void appendLineSegment(visualization_msgs::Marker &marker,
                              const Eigen::Vector3d &seg_start,
                              const Eigen::Vector3d &seg_end)
{
    geometry_msgs::Point p1;
    geometry_msgs::Point p2;
    p1.x = seg_start.x();
    p1.y = seg_start.y();
    p1.z = seg_start.z();
    p2.x = seg_end.x();
    p2.y = seg_end.y();
    p2.z = seg_end.z();
    marker.points.push_back(p1);
    marker.points.push_back(p2);
}

// Helper: append a colored line segment to a Marker
void appendLineSegmentColored(visualization_msgs::Marker &marker,
                                     const Eigen::Vector3d &seg_start,
                                     const Eigen::Vector3d &seg_end,
                                     const std_msgs::ColorRGBA &color)
{
    geometry_msgs::Point p1;
    geometry_msgs::Point p2;
    p1.x = seg_start.x();
    p1.y = seg_start.y();
    p1.z = seg_start.z();
    p2.x = seg_end.x();
    p2.y = seg_end.y();
    p2.z = seg_end.z();
    marker.points.push_back(p1);
    marker.points.push_back(p2);
    marker.colors.push_back(color);
    marker.colors.push_back(color);
}

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = ros_utils::ros_advertise<nav_msgs::Odometry>(n, "imu_propagate", 1000);
    pub_path = ros_utils::ros_advertise<nav_msgs::Path>(n, "path", 1000);
    pub_odometry = ros_utils::ros_advertise<nav_msgs::Odometry>(n, "odometry", 1000);
    pub_point_cloud = ros_utils::ros_advertise<sensor_msgs::PointCloud>(n, "point_cloud", 1000);
    pub_margin_cloud = ros_utils::ros_advertise<sensor_msgs::PointCloud>(n, "margin_cloud", 1000);
    pub_window_keypose = ros_utils::ros_advertise<visualization_msgs::Marker>(n, "window_keypose", 1000);
    pub_camera_pose = ros_utils::ros_advertise<nav_msgs::Odometry>(n, "camera_pose", 1000);
    pub_camera_pose_visual = ros_utils::ros_advertise<visualization_msgs::MarkerArray>(n, "camera_pose_visual", 1000);
    pub_keyframe_point = ros_utils::ros_advertise<sensor_msgs::PointCloud>(n, "keyframe_point", 1000);
    pub_keyframe_pose = ros_utils::ros_advertise<nav_msgs::Odometry>(n, "/vins_estimator/keyframe_pose", 1000);
    pub_keyframe_path = ros_utils::ros_advertise<nav_msgs::Path>(n, "/keyframe_path", 1000);
    pub_keyframe_marker = ros_utils::ros_advertise<visualization_msgs::Marker>(n, "/global_keypose", 1000);
    pub_extrinsic = ros_utils::ros_advertise<nav_msgs::Odometry>(n, "extrinsic", 1000);
    pub_image_track = ros_utils::ros_advertise<sensor_msgs::Image>(n, "image_track", 1000);
    pub_deep_match = ros_utils::ros_advertise<sensor_msgs::Image>(n, "/deep_match", 1000);
    pub_line_match = ros_utils::ros_advertise<sensor_msgs::Image>(n, "line_match", 1000);
    pub_map_line = ros_utils::ros_advertise<visualization_msgs::Marker>(n, "global_mapline", 1000);
    pub_window_map_line = ros_utils::ros_advertise<visualization_msgs::Marker>(n, "window_mapline", 1000);

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    ros_utils::ros_publish(pub_latest_odometry, odometry);
}

void pubTrackImage(const cv::Mat &imgTrack, const double t)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    ros_utils::ros_publish(pub_image_track, *imgTrackMsg);
}

void pubDeepMatchImage(const cv::Mat &imgMatch, const double t)
{
    if (imgMatch.empty())
        return;
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgMatchMsg = cv_bridge::CvImage(header, "bgr8", imgMatch).toImageMsg();
    ros_utils::ros_publish(pub_deep_match, *imgMatchMsg);
}

void pubLineMatchImage(const cv::Mat &imgMatch, const double t)
{
    if (imgMatch.empty())
        return;
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgMatchMsg = cv_bridge::CvImage(header, "bgr8", imgMatch).toImageMsg();
    ros_utils::ros_publish(pub_line_match, *imgMatchMsg);
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        ros_utils::ros_publish(pub_odometry, odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
    ros_utils::ros_publish(pub_path, path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.toSec() * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << endl;
        foutC.close();
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "window_keypose";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    ros_utils::ros_publish(pub_window_keypose, key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        ros_utils::ros_publish(pub_camera_pose, odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.points().pointFeature)
    {
        int used_num;
        used_num = it_per_id.point_feature_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.point_feature_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    ros_utils::ros_publish(pub_point_cloud, point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.points().pointFeature)
    { 
        int used_num;
        used_num = it_per_id.point_feature_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.point_feature_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.point_feature_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    ros_utils::ros_publish(pub_margin_cloud, margin_cloud);
}

std_msgs::ColorRGBA markerColorById(int id)
{
    uint32_t x = static_cast<uint32_t>(id * 2654435761u);
    std_msgs::ColorRGBA color;
    color.r = 0.25 + static_cast<double>(x & 0xFF) / 512.0;
    color.g = 0.25 + static_cast<double>((x >> 8) & 0xFF) / 512.0;
    color.b = 0.35 + static_cast<double>((x >> 16) & 0xFF) / 512.0;
    color.a = 1.0;
    return color;
}

void pubGolbalMapLine(const Estimator &estimator, const std_msgs::Header &header)
{
    visualization_msgs::Marker map_lines;
    map_lines.header = header;
    map_lines.header.frame_id = "world";
    map_lines.ns = "map_lines";
    map_lines.id = 0;
    map_lines.type = visualization_msgs::Marker::LINE_LIST;
    map_lines.action = visualization_msgs::Marker::ADD;
    map_lines.pose.orientation.w = 1.0;
    map_lines.scale.x = 0.03;
    map_lines.color.b = 1.0;
    map_lines.color.a = 1.0;
    map_lines.lifetime = ros::Duration();

    // Only promote lines that have already stabilized inside the current window.
    auto isStableForGlobal = [&](const auto &line_it)
    {
        return estimator.f_manager.useLineForOptimization(line_it) &&
               line_it.endFrame() <= WINDOW_SIZE - 3;
    };

    if (auto *line_manager = estimator.f_manager.lines())
    {
        for (const auto &line_it : line_manager->line_feature)
        {
            if (!isStableForGlobal(line_it))
                continue;
            Eigen::Vector3d seg_start;
            Eigen::Vector3d seg_end;
            if (!::lineToSegment(line_it.line_3d_world, seg_start, seg_end))
                continue;
            persistent_map_lines[line_it.line_id] = std::make_pair(seg_start, seg_end);
        }
    }

    for (const auto &line_it : persistent_map_lines)
    {
        appendLineSegmentColored(map_lines, line_it.second.first, line_it.second.second, markerColorById(line_it.first));
    }

    if (map_lines.points.empty())
        map_lines.action = visualization_msgs::Marker::DELETE;

    ros_utils::ros_publish(pub_map_line, map_lines);
}

void pubWindowMapLine(const Estimator &estimator, const std_msgs::Header &header)
{
    visualization_msgs::Marker window_lines;
    window_lines.header = header;
    window_lines.header.frame_id = "world";
    window_lines.ns = "window_map_lines";
    window_lines.id = 0;
    window_lines.type = visualization_msgs::Marker::LINE_LIST;
    window_lines.action = visualization_msgs::Marker::ADD;
    window_lines.pose.orientation.w = 1.0;
    window_lines.scale.x = 0.03;
    window_lines.color.r = 1.0;
    window_lines.color.g = 0.0;
    window_lines.color.b = 0.0;
    window_lines.color.a = 1.0;
    window_lines.lifetime = ros::Duration();

    if (const auto *line_manager = estimator.f_manager.lines())
    {
        for (const auto &line_it : line_manager->line_feature)
        {
            if (!estimator.f_manager.useLineForOptimization(line_it))
                continue;
            const Eigen::Vector3d moment = line_it.line_3d_world.head<3>();
            const Eigen::Vector3d direction = line_it.line_3d_world.tail<3>();
            const double dir_norm = direction.norm();
            if (!moment.allFinite() || !direction.allFinite() || dir_norm < 1e-9)
                continue;
            const Eigen::Vector3d anchor = direction.cross(moment) / (dir_norm * dir_norm);
            const Eigen::Vector3d offset = direction.normalized() * 0.5;
            appendLineSegment(window_lines, anchor - offset, anchor + offset);
        }
    }

    if (window_lines.points.empty())
        window_lines.action = visualization_msgs::Marker::DELETE;

    ros_utils::ros_publish(pub_window_map_line, window_lines);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    ros_utils::ros_publish(pub_extrinsic, odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);
        std_msgs::Header header;
        header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - 2]);
        header.frame_id = "world";

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.pose.position.x = P.x();
        pose_stamped.pose.position.y = P.y();
        pose_stamped.pose.position.z = P.z();
        pose_stamped.pose.orientation.x = R.x();
        pose_stamped.pose.orientation.y = R.y();
        pose_stamped.pose.orientation.z = R.z();
        pose_stamped.pose.orientation.w = R.w();
        keyframe_path.header = header;
        keyframe_path.header.frame_id = "world";
        keyframe_path.poses.push_back(pose_stamped);
        ros_utils::ros_publish(pub_keyframe_path, keyframe_path);

        keyframe_marker.header = header;
        keyframe_marker.header.frame_id = "world";
        keyframe_marker.ns = "global_keypose";
        keyframe_marker.id = 0;
        keyframe_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        keyframe_marker.action = visualization_msgs::Marker::ADD;
        keyframe_marker.pose.orientation.w = 1.0;
        keyframe_marker.scale.x = 0.08;
        keyframe_marker.scale.y = 0.08;
        keyframe_marker.scale.z = 0.08;
        keyframe_marker.color.r = 0.1;
        keyframe_marker.color.g = 1.0;
        keyframe_marker.color.b = 0.2;
        keyframe_marker.color.a = 1.0;
        keyframe_marker.lifetime = ros::Duration();
        geometry_msgs::Point key_pose;
        key_pose.x = P.x();
        key_pose.y = P.y();
        key_pose.z = P.z();
        keyframe_marker.points.push_back(key_pose);
        ros_utils::ros_publish(pub_keyframe_marker, keyframe_marker);

        nav_msgs::Odometry keyframe_pose;
        keyframe_pose.header = header;
        keyframe_pose.header.frame_id = "world";
        keyframe_pose.pose.pose.position.x = P.x();
        keyframe_pose.pose.pose.position.y = P.y();
        keyframe_pose.pose.pose.position.z = P.z();
        keyframe_pose.pose.pose.orientation.x = R.x();
        keyframe_pose.pose.pose.orientation.y = R.y();
        keyframe_pose.pose.pose.orientation.z = R.z();
        keyframe_pose.pose.pose.orientation.w = R.w();
        ros_utils::ros_publish(pub_keyframe_pose, keyframe_pose);

        const auto good_points = estimator.f_manager.collectGoodKeyframePoints();
        ROS_INFO_STREAM_THROTTLE(5.0, "[vins_estimator] keyframe good points=" << good_points.size());
        sensor_msgs::PointCloud keyframe_point_msg;
        keyframe_point_msg.header = header;
        keyframe_point_msg.header.frame_id = "world";
        keyframe_point_msg.points.reserve(good_points.size());
        keyframe_point_msg.channels.reserve(good_points.size());
        for (const auto &good_point : good_points)
        {
            geometry_msgs::Point32 p;
            p.x = good_point.point_w.x();
            p.y = good_point.point_w.y();
            p.z = good_point.point_w.z();
            keyframe_point_msg.points.push_back(p);

            sensor_msgs::ChannelFloat32 channel;
            channel.values.reserve(261);
            channel.values.push_back(good_point.point_norm.x());
            channel.values.push_back(good_point.point_norm.y());
            channel.values.push_back(good_point.point_uv.x());
            channel.values.push_back(good_point.point_uv.y());
            channel.values.push_back(static_cast<float>(good_point.point_feature_id));
            for (int i = 0; i < good_point.descriptor.size(); ++i)
                channel.values.push_back(good_point.descriptor(i));
            keyframe_point_msg.channels.push_back(channel);
        }
        ros_utils::ros_publish(pub_keyframe_point, keyframe_point_msg);
    }
}
