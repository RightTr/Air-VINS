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

#include <stdio.h>
#include <queue>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#ifdef USE_ROS2
#include <rclcpp/executors/multi_threaded_executor.hpp>
#endif
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include "utility/ros_utils.h"

Estimator estimator;

queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

namespace
{
void enqueue_image(queue<sensor_msgs::ImageConstPtr> &image_queue,
                   const sensor_msgs::ImageConstPtr &image)
{
    if (MODE == 0 && image_queue.size() >= 2)
    {
        image_queue.pop();
    }
    image_queue.push(image);
}
} // namespace


void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    enqueue_image(img0_buf, img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    enqueue_image(img1_buf, img_msg);
    m_buf.unlock();
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

bool sameImageExact(const cv::Mat &a, const cv::Mat &b)
{
    if (a.empty() || b.empty())
        return false;
    if (a.size() != b.size() || a.type() != b.type())
        return false;
    return cv::countNonZero(a != b) == 0;
}

// extract images with same timestamp from two topics
void sync_process()
{
    static cv::Mat prev_image0, prev_image1;
    static int nuc_camera = -1;  // -1: stereo, 0: left mono, 1: right mono
    static int recovery_count = 0;
    constexpr int recovery_required = 3;

    if (STEREO)
    {
        cv::Mat image0, image1;
        sensor_msgs::ImageConstPtr image0_msg, image1_msg;
        double time = 0.0;
        {
            std::lock_guard<std::mutex> lock(m_buf);
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                const double time0 = ros_utils::time_to_sec(img0_buf.front()->header.stamp);
                const double time1 = ros_utils::time_to_sec(img1_buf.front()->header.stamp);
                // Allow up to 10 ms timestamp skew between stereo images.
                constexpr double stereo_sync_tolerance = 0.010;
                if (time0 < time1 - stereo_sync_tolerance)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if (time0 > time1 + stereo_sync_tolerance)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = time0;
                    image0_msg = img0_buf.front();
                    image1_msg = img1_buf.front();
                    img0_buf.pop();
                    img1_buf.pop();
                }
            }
        }

        if (!image0_msg || !image1_msg)
            return;

        image0 = getImageFromMsg(image0_msg);
        image1 = getImageFromMsg(image1_msg);
        cv::Mat rectified0, rectified1;
        cv::remap(image0, rectified0, RECTIFY_MAP_1[0], RECTIFY_MAP_2[0], cv::INTER_LINEAR);
        cv::remap(image1, rectified1, RECTIFY_MAP_1[1], RECTIFY_MAP_2[1], cv::INTER_LINEAR);
        image0 = rectified0;
        image1 = rectified1;

        const bool left_frozen = ENABLE_NUC_HANDLE && sameImageExact(image0, prev_image0);
        const bool right_frozen = ENABLE_NUC_HANDLE && sameImageExact(image1, prev_image1);
        const auto input_mono = [&](int camera_id)
        {
            if (camera_id == 0)
                estimator.inputImage(time, image0, cv::Mat(), 0, false, TRACKING_MODE_NUC_RIGHT_ONLY);
            else
                estimator.inputImage(time, image1, cv::Mat(), 1, false, TRACKING_MODE_NUC_LEFT_ONLY);
        };

        if (ENABLE_NUC_HANDLE && left_frozen && right_frozen)
        {
            ROS_WARN_THROTTLE(1.0, "NUC: both thermal cameras frozen, skip visual update");
            recovery_count = 0;
        }
        else if (ENABLE_NUC_HANDLE && (left_frozen || right_frozen))
        {
            nuc_camera = left_frozen ? 1 : 0;
            recovery_count = 0;
            if (left_frozen)
                ROS_WARN_THROTTLE(1.0, "NUC: left thermal camera frozen, use right camera as monocular input");
            else
                ROS_WARN_THROTTLE(1.0, "NUC: right thermal camera frozen, use left camera as monocular input");
            input_mono(nuc_camera);
        }
        else if (nuc_camera >= 0 && ++recovery_count < recovery_required)
        {
            ROS_WARN_THROTTLE(1.0, "NUC: waiting for stable stereo recovery (%d/%d)", recovery_count, recovery_required);
            input_mono(nuc_camera);
        }
        else
        {
            if (nuc_camera >= 0)
                ROS_WARN("NUC: stereo recovered, keep tracked landmarks and resume stereo updates");
            estimator.inputImage(time, image0, image1, 0, true, TRACKING_MODE_STEREO);
            nuc_camera = -1;
            recovery_count = 0;
        }

        prev_image0 = image0.clone();
        prev_image1 = image1.clone();
    }
    else
    {
        cv::Mat image;
        sensor_msgs::ImageConstPtr image_msg;
        double time = 0.0;
        {
            std::lock_guard<std::mutex> lock(m_buf);
            if (!img0_buf.empty())
            {
                image_msg = img0_buf.front();
                time = ros_utils::time_to_sec(image_msg->header.stamp);
                img0_buf.pop();
            }
        }
        if (image_msg)
        {
            image = getImageFromMsg(image_msg);
            if (!image.empty())
            {
                cv::Mat rectified;
                cv::remap(image, rectified, RECTIFY_MAP_1[0], RECTIFY_MAP_2[0], cv::INTER_LINEAR);
                estimator.inputImage(time, rectified);
            }
        }
    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = ros_utils::time_to_sec(imu_msg->header.stamp);
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = ros_utils::time_to_sec(feature_msg->header.stamp);
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeCameraType(1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeCameraType(0);
    }
    return;
}

int main(int argc, char **argv)
{
    auto node = ros_utils::init(argc, argv, "stir_slam");
    ROS_INFO("init begins");
    // Estimator is a global callback target. Initialize its state here, after
    // every translation unit's global visualization objects are constructed.
    estimator.clearState();
    string config_file;
    ros_utils::param(node, "config_file", config_file, string(""));
    if (config_file.empty())
    {
        ROS_ERROR("missing ROS parameter: config_file");
        return 1;
    }
    printf("config_file: %s\n", config_file.c_str());

    readParameters(config_file);
    if (IMAGE0_TOPIC.empty() || (STEREO && IMAGE1_TOPIC.empty()) ||
        IMU_TOPIC.empty())
    {
        ROS_ERROR("Configure image topics and topics.imu");
        return 1;
    }
    estimator.setParameter();

    // Offline stereo playback can request reliable delivery for all sensor topics.
    // Keep the original sensor-data QoS for existing live-camera launches.
    bool reliable_sensor_qos = false;
    ros_utils::param(node, "reliable_sensor_qos", reliable_sensor_qos, false);
    const auto sensor_qos = reliable_sensor_qos ? ros_utils::QosProfile::Reliable
                                                : ros_utils::QosProfile::SensorData;

    ROS_WARN("waiting for image and imu...");

    registerPub(node);

    ros_utils::Subscription<sensor_msgs::Imu> sub_imu;
#ifdef USE_ROS2
    rclcpp::CallbackGroup::SharedPtr imu_callback_group;
#endif
#ifdef USE_ROS2
        // Image processing can block the default callback group for longer
        // than one IMU period. Receive IMU on its own executor thread.
        imu_callback_group = node->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = imu_callback_group;
        sub_imu = node->create_subscription<sensor_msgs::Imu>(
            IMU_TOPIC, ros_utils::qos(2000, ros_utils::QosProfile::SensorData),
            imu_callback, imu_options);
#else
        sub_imu = ros_utils::create_subscription<sensor_msgs::Imu>(
            node, IMU_TOPIC, 2000, imu_callback, ros_utils::QosProfile::SensorData, true);
#endif
    auto sub_feature = ros_utils::create_subscription<sensor_msgs::PointCloud>(
        node, "/feature_tracker/feature", 2000, feature_callback);
    const int image_subscription_depth = MODE == 0 ? 2 : 2000;
    auto sub_img0 = ros_utils::create_subscription<sensor_msgs::Image>(
        node, IMAGE0_TOPIC, image_subscription_depth, img0_callback, sensor_qos);
    ros_utils::Subscription<sensor_msgs::Image> sub_img1;
    if(STEREO)
    {
        sub_img1 = ros_utils::create_subscription<sensor_msgs::Image>(
            node, IMAGE1_TOPIC, image_subscription_depth, img1_callback, sensor_qos);
    }
    auto sub_restart = ros_utils::create_subscription<std_msgs::Bool>(
        node, "/stir_slam/restart", 100, restart_callback);
    auto sub_cam_switch = ros_utils::create_subscription<std_msgs::Bool>(
        node, "/stir_slam/cam_switch", 100, cam_switch_callback);

    auto sync_timer = ros_utils::create_timer(node, 0.002, sync_process);
#ifdef USE_ROS2
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node);
#else
    ros_utils::spin(node);
#endif
    ros_utils::stop_timer(sync_timer);

    estimator.shutdown();
    // Release TensorRT resources before global/static destruction at process exit.
    estimator.featureTracker.deep_feature.reset();
    shutdownPub();
#ifdef USE_ROS2
    sync_timer.reset();
    sub_cam_switch.reset();
    sub_restart.reset();
    sub_img1.reset();
    sub_img0.reset();
    sub_feature.reset();
    sub_imu.reset();
    ros_utils::global_node().reset();
    node.reset();
#endif

    return 0;
}
