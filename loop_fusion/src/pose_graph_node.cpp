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

#include <vector>
#include "vins_estimator/src/utility/ros_utils.h"
#ifdef USE_ROS1
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#else
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#endif
#include <iostream>
#include <fstream>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#include "deep_feature.h"
#include "vprnet/vprnet.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::ImageConstPtr> image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<ros_utils::ConstPtr<nav_msgs::Odometry>> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

Camera m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros_utils::Publisher<sensor_msgs::Image> pub_match_img;
ros_utils::Publisher<visualization_msgs::MarkerArray> pub_camera_pose_visual;
ros_utils::Publisher<nav_msgs::Odometry> pub_odometry_rect;
std::unique_ptr<VPRNet> vprnet_extractor;
std::unique_ptr<DeepFeature> loop_deep_feature;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string STIR_SLAM_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

ros_utils::Publisher<sensor_msgs::PointCloud> pub_point_cloud, pub_margin_cloud;

void new_sequence()
{
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 5)
    {
        ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        ROS_BREAK();
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    //printf(" image time %f \n", image_msg->header.stamp.toSec());

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = ros_utils::time_to_sec(image_msg->header.stamp);
    else if (ros_utils::time_to_sec(image_msg->header.stamp) - last_image_time > 1.0 ||
             ros_utils::time_to_sec(image_msg->header.stamp) < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = ros_utils::time_to_sec(image_msg->header.stamp);
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    //ROS_INFO("point_callback!");
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
    /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
    // for visualization
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    ros_utils::publish(pub_point_cloud, point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    ros_utils::publish(pub_margin_cloud, point_cloud);
}

void pose_callback(const ros_utils::ConstPtr<nav_msgs::Odometry> &pose_msg)
{
    //ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void vio_callback(const ros_utils::ConstPtr<nav_msgs::Odometry> &pose_msg)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    ros_utils::publish(pub_odometry_rect, odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);


}

void extrinsic_callback(const ros_utils::ConstPtr<nav_msgs::Odometry> &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

void process()
{
    while (ros_utils::ok())
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        ros_utils::ConstPtr<nav_msgs::Odometry> pose_msg = nullptr;

        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (ros_utils::time_to_sec(image_buf.front()->header.stamp) >
                ros_utils::time_to_sec(pose_buf.front()->header.stamp))
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (ros_utils::time_to_sec(image_buf.front()->header.stamp) >
                     ros_utils::time_to_sec(point_buf.front()->header.stamp))
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (ros_utils::time_to_sec(image_buf.back()->header.stamp) >=
                         ros_utils::time_to_sec(pose_buf.front()->header.stamp)
                && ros_utils::time_to_sec(point_buf.back()->header.stamp) >=
                         ros_utils::time_to_sec(pose_buf.front()->header.stamp))
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (ros_utils::time_to_sec(image_buf.front()->header.stamp) <
                       ros_utils::time_to_sec(pose_msg->header.stamp))
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (ros_utils::time_to_sec(point_buf.front()->header.stamp) <
                       ros_utils::time_to_sec(pose_msg->header.stamp))
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();

        if (pose_msg != NULL)
        {
            //printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            //printf(" point time %f \n", point_msg->header.stamp.toSec());
            //printf(" image time %f \n", image_msg->header.stamp.toSec());
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                Eigen::Matrix<float, 259, Eigen::Dynamic> good_point_deep_features;
                good_point_deep_features.resize(259, point_msg->points.size());
                int good_point_count = 0;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    if (i >= point_msg->channels.size() || point_msg->channels[i].values.size() < 261)
                        continue;

                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;

                    cv::Point2f p_2d_uv, p_2d_normal;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    point_3d.push_back(p_3d);
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);

                    good_point_deep_features(0, good_point_count) = 1.0f;
                    good_point_deep_features(1, good_point_count) = p_2d_uv.x;
                    good_point_deep_features(2, good_point_count) = p_2d_uv.y;
                    for (int d = 0; d < 256; ++d)
                        good_point_deep_features(3 + d, good_point_count) = point_msg->channels[i].values[5 + d];
                    ++good_point_count;

                    //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }

                good_point_deep_features.conservativeResize(259, good_point_count);

                KeyFrame* keyframe = new KeyFrame(ros_utils::time_to_sec(pose_msg->header.stamp), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, sequence);   
                keyframe->setGoodPointDeepFeatures(good_point_deep_features);
                if (loop_deep_feature) {
                    Eigen::Matrix<float, 259, Eigen::Dynamic> deep_points;
                    if (loop_deep_feature->extractPoints(image, deep_points)) {
                        keyframe->setDeepFeatures(deep_points);
                    } else {
                        ROS_WARN_STREAM_THROTTLE(5.0, "Loop deep feature extraction failed for frame " << frame_index);
                    }
                }
                if (vprnet_extractor) {
                    Eigen::VectorXf descriptor;
                    if (vprnet_extractor->infer(image, descriptor)) {
                        std::vector<float> vprnet_vector(descriptor.data(), descriptor.data() + descriptor.size());
                        keyframe->setVPRNetDescriptor(vprnet_vector);
                    } else {
                        ROS_WARN_STREAM_THROTTLE(5.0, "VPRNet inference failed for frame " << frame_index);
                    }
                }
                m_process.lock();
                start_flag = 1;
                posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void command()
{
    while(ros_utils::ok())
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("program shutting down...\n");
            ros_utils::shutdown();
        }
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    auto node = ros_utils::init(argc, argv, "loop_fusion");
    posegraph.registerPub(node);
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0;

    string config_file;
    ros_utils::param(node, "config_file", config_file, string(""));
    if (config_file.empty())
    {
        ROS_ERROR("missing ROS parameter: config_file");
        return 0;
    }
    printf("config_file: %s\n", config_file.c_str());

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE_TOPIC;
    int LOAD_PREVIOUS_POSE_GRAPH;

    ROW = fsSettings["camera"]["image_height"];
    COL = fsSettings["camera"]["image_width"];
    std::string pkg_path = ros_utils::package_share_directory("loop_fusion");
    std::string support_files_dir = pkg_path;
#ifdef USE_ROS1
    support_files_dir += "/../support_files";
#else
    support_files_dir += "/support_files";
#endif
    string vocabulary_file = support_files_dir + "/brief_k10L6.bin";
    BRIEF_PATTERN_FILE = support_files_dir + "/brief_pattern.yml";
    if (!std::ifstream(vocabulary_file, std::ios::binary).good() ||
        !std::ifstream(BRIEF_PATTERN_FILE).good())
    {
        ROS_ERROR_STREAM("Missing loop-closure BRIEF resources in " << support_files_dir);
        return 1;
    }
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph.loadVocabulary(vocabulary_file);

    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

    const cv::FileNode cam0_node = fsSettings["camera"]["cam0_intrinsics"];
    if (cam0_node.empty() || !cam0_node.isSeq() || cam0_node.size() != 4)
    {
        ROS_ERROR("cam0_intrinsics must contain [fx, fy, cx, cy]");
        return 1;
    }
    Eigen::Vector4d cam0_intrinsics;
    for (int i = 0; i < 4; ++i)
        cam0_intrinsics(i) = static_cast<double>(cam0_node[i]);
    if (!cam0_intrinsics.allFinite() || cam0_intrinsics.x() <= 0.0 || cam0_intrinsics.y() <= 0.0)
    {
        ROS_ERROR("cam0_intrinsics must contain finite values with positive fx/fy");
        return 1;
    }
    m_camera = Camera{cam0_intrinsics.x(), cam0_intrinsics.y(),
                      cam0_intrinsics.z(), cam0_intrinsics.w()};

    fsSettings["topics"]["image0"] >> IMAGE_TOPIC;
    fsSettings["loop_closure"]["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
    fsSettings["system"]["output_path"] >> STIR_SLAM_RESULT_PATH;
    fsSettings["loop_closure"]["save_image"] >> DEBUG_IMAGE;
    std::string output_dir = STIR_SLAM_RESULT_PATH;

    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["loop_closure"]["load_previous_pose_graph"];
    STIR_SLAM_RESULT_PATH = STIR_SLAM_RESULT_PATH + "/vio_loop.csv";
    std::ofstream fout(STIR_SLAM_RESULT_PATH, std::ios::out);
    fout.close();

    int loop_descriptor_method = 1;
    if (!fsSettings["loop_closure"]["descriptor_method"].empty()) {
        fsSettings["loop_closure"]["descriptor_method"] >> loop_descriptor_method;
    }
    posegraph.setLoopDescriptorType(loop_descriptor_method);
    double vprnet_loop_threshold = 0.75;
    double vprnet_loop_margin = 0.0;
    int vprnet_loop_exclude_recent = 50;
    if (!fsSettings["loop_closure"]["loop_threshold"].empty()) {
        fsSettings["loop_closure"]["loop_threshold"] >> vprnet_loop_threshold;
    }
    if (!fsSettings["loop_closure"]["loop_margin"].empty()) {
        fsSettings["loop_closure"]["loop_margin"] >> vprnet_loop_margin;
    }
    if (!fsSettings["loop_closure"]["exclude_recent"].empty()) {
        fsSettings["loop_closure"]["exclude_recent"] >> vprnet_loop_exclude_recent;
    }
    posegraph.setVPRNetLoopParams(vprnet_loop_threshold, vprnet_loop_margin, vprnet_loop_exclude_recent);

    if (loop_descriptor_method == 2) {
        std::string vprnet_engine_path = output_dir + "/ResNet50_512_eigenplaces.engine";
        std::string vprnet_onnx_path = output_dir + "/ResNet50_512_eigenplaces.onnx";
        if (!fsSettings["loop_closure"]["vprnet_engine_path"].empty()) {
            fsSettings["loop_closure"]["vprnet_engine_path"] >> vprnet_engine_path;
        }
        if (!fsSettings["loop_closure"]["vprnet_onnx_path"].empty()) {
            fsSettings["loop_closure"]["vprnet_onnx_path"] >> vprnet_onnx_path;
        }
        VPRNet::Options options;
        options.engine_path = vprnet_engine_path;
        options.onnx_path = vprnet_onnx_path;
        options.input_width = 640;
        options.input_height = 480;
        options.verbose = false;
        vprnet_extractor.reset(new VPRNet(options));
        if (!vprnet_extractor->build()) {
            ROS_WARN_STREAM("Failed to load VPRNet TensorRT engine: " << vprnet_engine_path);
            vprnet_extractor.reset();
            posegraph.setLoopDescriptorType(1);
        }
    }

    {
        DeepFeatureOptions deep_options;
        deep_options.model_dir = output_dir;
        deep_options.matcher = 0;
        deep_options.max_keypoints = 1024;
        deep_options.keypoint_threshold = 0.004f;
        deep_options.remove_borders = 4;
        deep_options.image_width = COL;
        deep_options.image_height = ROW;
        loop_deep_feature.reset(new DeepFeature());
        if (loop_deep_feature->init(1, deep_options)) {
            posegraph.setLoopDeepFeature(loop_deep_feature.get());
            ROS_INFO_STREAM("Loop deep feature matcher initialized with model_dir=" << output_dir);
        } else {
            ROS_WARN_STREAM("Loop deep feature matcher failed to initialize, loop closure will fall back to BRIEF if needed");
            loop_deep_feature.reset();
            posegraph.setLoopDeepFeature(nullptr);
        }
    }

    int USE_IMU = fsSettings["system"]["imu"];
    posegraph.setIMUFlag(USE_IMU);
    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    auto sub_vio = ros_utils::create_subscription<nav_msgs::Odometry>(
        node, "/stir_slam/odometry", 2000, vio_callback);
    auto sub_image = ros_utils::create_subscription<sensor_msgs::Image>(
        node, IMAGE_TOPIC, 2000, image_callback, ros_utils::QosProfile::SensorData);
    auto sub_pose = ros_utils::create_subscription<nav_msgs::Odometry>(
        node, "/stir_slam/keyframe_pose", 2000, pose_callback);
    auto sub_extrinsic = ros_utils::create_subscription<nav_msgs::Odometry>(
        node, "/stir_slam/extrinsic", 2000, extrinsic_callback);
    auto sub_point = ros_utils::create_subscription<sensor_msgs::PointCloud>(
        node, "/stir_slam/keyframe_point", 2000, point_callback);
    auto sub_margin_point = ros_utils::create_subscription<sensor_msgs::PointCloud>(
        node, "/stir_slam/margin_cloud", 2000, margin_point_callback);

    pub_match_img = ros_utils::create_publisher<sensor_msgs::Image>(node, "match_image", 1000);
    pub_camera_pose_visual = ros_utils::create_publisher<visualization_msgs::MarkerArray>(
        node, "camera_pose_visual", 1000);
    pub_point_cloud = ros_utils::create_publisher<sensor_msgs::PointCloud>(
        node, "point_cloud_loop_rect", 1000);
    pub_margin_cloud = ros_utils::create_publisher<sensor_msgs::PointCloud>(
        node, "margin_cloud_loop_rect", 1000);
    pub_odometry_rect = ros_utils::create_publisher<nav_msgs::Odometry>(
        node, "odometry_rect", 1000);

    std::thread measurement_process;
    std::thread keyboard_command_process;

    measurement_process = std::thread(process);
    keyboard_command_process = std::thread(command);
    keyboard_command_process.detach();
    
    ros_utils::spin(node);
    measurement_process.join();

    return 0;
}
