#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/ColorRGBA.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace ros_utils
{
template <typename MsgT>
inline ros::Publisher ros_advertise(ros::NodeHandle &nh, const std::string &topic, uint32_t queue_size)
{
    return nh.advertise<MsgT>(topic, queue_size);
}

inline bool hasSubscribers(const ros::Publisher &pub)
{
    return pub.getNumSubscribers() > 0;
}

template <typename MsgT>
inline bool ros_publish(const ros::Publisher &pub, const MsgT &msg)
{
    if (!hasSubscribers(pub))
        return false;
    pub.publish(msg);
    return true;
}
} // namespace ros_utils
