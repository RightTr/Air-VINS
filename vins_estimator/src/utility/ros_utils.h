#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#ifdef USE_ROS1
#include <ros/package.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#elif defined(USE_ROS2)
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <builtin_interfaces/msg/duration.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/channel_float32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

// Preserve the ROS1 spelling in algorithm code. ROS-specific ownership and node
// operations remain explicit in ros_utils below.
namespace std_msgs {
using Header = msg::Header;
using Bool = msg::Bool;
using BoolConstPtr = msg::Bool::ConstSharedPtr;
using ColorRGBA = msg::ColorRGBA;
using Float32 = msg::Float32;
}  // namespace std_msgs
namespace sensor_msgs {
using Image = msg::Image;
using ImagePtr = msg::Image::SharedPtr;
using ImageConstPtr = msg::Image::ConstSharedPtr;
using Imu = msg::Imu;
using ImuPtr = msg::Imu::SharedPtr;
using ImuConstPtr = msg::Imu::ConstSharedPtr;
using PointCloud = msg::PointCloud;
using PointCloudPtr = msg::PointCloud::SharedPtr;
using PointCloudConstPtr = msg::PointCloud::ConstSharedPtr;
using ChannelFloat32 = msg::ChannelFloat32;
}  // namespace sensor_msgs
namespace geometry_msgs {
using Point = msg::Point;
using Point32 = msg::Point32;
using PoseStamped = msg::PoseStamped;
using TransformStamped = msg::TransformStamped;
}  // namespace geometry_msgs
namespace nav_msgs {
using Odometry = msg::Odometry;
using Path = msg::Path;
}  // namespace nav_msgs
namespace visualization_msgs {
using Marker = msg::Marker;
using MarkerArray = msg::MarkerArray;
}  // namespace visualization_msgs
#else
#error "Define USE_ROS1 or USE_ROS2"
#endif

namespace ros_utils {

enum class QosProfile { Reliable, SensorData };

#ifdef USE_ROS1
using NodeHandle = std::shared_ptr<ros::NodeHandle>;
using Time = ros::Time;
using Duration = ros::Duration;
template <typename MsgT> using Publisher = ros::Publisher;
template <typename MsgT> using Subscription = ros::Subscriber;
template <typename MsgT> using ConstPtr = typename MsgT::ConstPtr;
using Timer = ros::Timer;
#else
using NodeHandle = rclcpp::Node::SharedPtr;
using Time = builtin_interfaces::msg::Time;
using Duration = builtin_interfaces::msg::Duration;
template <typename MsgT> using Publisher = typename rclcpp::Publisher<MsgT>::SharedPtr;
template <typename MsgT> using Subscription = typename rclcpp::Subscription<MsgT>::SharedPtr;
template <typename MsgT> using ConstPtr = typename MsgT::ConstSharedPtr;
using Timer = rclcpp::TimerBase::SharedPtr;
#endif

inline NodeHandle &global_node()
{
    static NodeHandle node;
    return node;
}

inline NodeHandle init(int argc, char **argv, const std::string &name)
{
#ifdef USE_ROS1
    ros::init(argc, argv, name);
    global_node() = std::make_shared<ros::NodeHandle>("~");
#else
    rclcpp::init(argc, argv);
    global_node() = rclcpp::Node::make_shared(name);
#endif
    return global_node();
}

inline bool ok()
{
#ifdef USE_ROS1
    return ros::ok();
#else
    return rclcpp::ok();
#endif
}

inline void shutdown()
{
#ifdef USE_ROS1
    ros::shutdown();
#else
    rclcpp::shutdown();
#endif
}

inline void spin(const NodeHandle &node)
{
#ifdef USE_ROS1
    (void)node;
    ros::spin();
#else
    rclcpp::spin(node);
#endif
}

inline void spin_once(const NodeHandle &node = global_node())
{
#ifdef USE_ROS1
    (void)node;
    ros::spinOnce();
#else
    rclcpp::spin_some(node);
#endif
}

template <typename CallbackT>
inline Timer create_timer(const NodeHandle &node, double period_seconds, CallbackT &&callback)
{
#ifdef USE_ROS1
    return node->createTimer(ros::Duration(period_seconds), std::forward<CallbackT>(callback));
#else
    const auto period = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(period_seconds));
    return node->create_wall_timer(period, std::forward<CallbackT>(callback));
#endif
}

inline void publish_transform(const geometry_msgs::TransformStamped &message)
{
#ifdef USE_ROS1
    static tf::TransformBroadcaster broadcaster;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(message.transform.translation.x,
                                    message.transform.translation.y,
                                    message.transform.translation.z));
    transform.setRotation(tf::Quaternion(message.transform.rotation.x,
                                         message.transform.rotation.y,
                                         message.transform.rotation.z,
                                         message.transform.rotation.w));
    broadcaster.sendTransform(tf::StampedTransform(transform, message.header.stamp,
                                                   message.header.frame_id,
                                                   message.child_frame_id));
#else
    static auto broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(global_node());
    broadcaster->sendTransform(message);
#endif
}

inline double time_to_sec(const Time &stamp)
{
#ifdef USE_ROS1
    return stamp.toSec();
#else
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
#endif
}

inline Time time_from_sec(double seconds)
{
#ifdef USE_ROS1
    return ros::Time().fromSec(seconds);
#else
    const auto nanoseconds = static_cast<int64_t>(std::llround(seconds * 1e9));
    builtin_interfaces::msg::Time result;
    auto sec = nanoseconds / 1000000000LL;
    auto nanosec = nanoseconds % 1000000000LL;
    if (nanosec < 0) {
        nanosec += 1000000000LL;
        --sec;
    }
    result.sec = static_cast<int32_t>(sec);
    result.nanosec = static_cast<uint32_t>(nanosec);
    return result;
#endif
}

inline Duration zero_duration()
{
#ifdef USE_ROS1
    return ros::Duration();
#else
    return builtin_interfaces::msg::Duration{};
#endif
}

inline void sleep_for(double seconds)
{
#ifdef USE_ROS1
    ros::Duration(seconds).sleep();
#else
    if (seconds > 0.0)
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
#endif
}

inline std::string private_topic(const std::string &topic)
{
    if (topic.empty() || topic.front() == '/' || topic.front() == '~')
        return topic;
#ifdef USE_ROS1
    return topic;
#else
    return "~/" + topic;
#endif
}

#ifdef USE_ROS2
inline rclcpp::QoS qos(std::size_t queue_size, QosProfile profile)
{
    rclcpp::QoS result{rclcpp::KeepLast(queue_size)};
    return profile == QosProfile::SensorData ? result.best_effort() : result.reliable();
}
#endif

template <typename MsgT>
inline Publisher<MsgT> create_publisher(const NodeHandle &node, const std::string &topic,
                                        std::size_t queue_size, bool use_private_topic = true,
                                        QosProfile profile = QosProfile::Reliable)
{
#ifdef USE_ROS1
    (void)profile;
    return node->advertise<MsgT>(use_private_topic ? private_topic(topic) : topic, queue_size);
#else
    const std::string resolved = use_private_topic ? private_topic(topic) : topic;
    return node->create_publisher<MsgT>(resolved, qos(queue_size, profile));
#endif
}

template <typename MsgT>
inline Publisher<MsgT> ros_advertise(const NodeHandle &node, const std::string &topic,
                                     std::size_t queue_size, bool use_private_topic = true,
                                     QosProfile profile = QosProfile::Reliable)
{
    return create_publisher<MsgT>(node, topic, queue_size, use_private_topic, profile);
}

template <typename MsgT, typename CallbackT>
inline Subscription<MsgT> create_subscription(const NodeHandle &node, const std::string &topic,
                                              std::size_t queue_size, CallbackT &&callback,
                                              QosProfile profile = QosProfile::Reliable,
                                              bool tcp_no_delay = false)
{
#ifdef USE_ROS1
    (void)profile;
    ros::TransportHints hints;
    if (tcp_no_delay)
        hints.tcpNoDelay();
    return node->subscribe<MsgT>(topic, queue_size, std::forward<CallbackT>(callback), hints);
#else
    (void)tcp_no_delay;
    return node->create_subscription<MsgT>(topic, qos(queue_size, profile),
                                            std::forward<CallbackT>(callback));
#endif
}

template <typename MsgT>
inline std::size_t subscription_count(const Publisher<MsgT> &publisher)
{
#ifdef USE_ROS1
    return publisher.getNumSubscribers();
#else
    return publisher ? publisher->get_subscription_count() : 0;
#endif
}

template <typename MsgT>
inline bool publish(const Publisher<MsgT> &publisher, const MsgT &message)
{
    if (subscription_count<MsgT>(publisher) == 0)
        return false;
#ifdef USE_ROS1
    publisher.publish(message);
#else
    publisher->publish(message);
#endif
    return true;
}

template <typename MsgT>
inline bool ros_publish(const Publisher<MsgT> &publisher, const MsgT &message)
{
    return publish(publisher, message);
}

template <typename T>
inline void param(const NodeHandle &node, const std::string &name, T &value, const T &default_value)
{
#ifdef USE_ROS1
    node->param<T>(name, value, default_value);
#else
    if (!node->has_parameter(name))
        node->declare_parameter<T>(name, default_value);
    node->get_parameter(name, value);
#endif
}

inline std::string package_share_directory(const std::string &package_name)
{
#ifdef USE_ROS1
    return ros::package::getPath(package_name);
#else
    return ament_index_cpp::get_package_share_directory(package_name);
#endif
}

#ifdef USE_ROS2
inline rclcpp::Logger logger()
{
    return global_node() ? global_node()->get_logger() : rclcpp::get_logger("air_vins");
}

inline rclcpp::Clock &clock()
{
    static rclcpp::Clock fallback_clock;
    return global_node() ? *global_node()->get_clock() : fallback_clock;
}
#endif

}  // namespace ros_utils

#ifdef USE_ROS2
#define ROS_DEBUG(...) RCLCPP_DEBUG(::ros_utils::logger(), __VA_ARGS__)
#define ROS_INFO(...) RCLCPP_INFO(::ros_utils::logger(), __VA_ARGS__)
#define ROS_WARN(...) RCLCPP_WARN(::ros_utils::logger(), __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(::ros_utils::logger(), __VA_ARGS__)
#define ROS_ERROR_THROTTLE(period, ...) \
    RCLCPP_ERROR_THROTTLE(::ros_utils::logger(), ::ros_utils::clock(), \
                          static_cast<int64_t>((period) * 1000.0), __VA_ARGS__)
#define ROS_WARN_THROTTLE(period, ...) \
    RCLCPP_WARN_THROTTLE(::ros_utils::logger(), ::ros_utils::clock(), \
                         static_cast<int64_t>((period) * 1000.0), __VA_ARGS__)
#define ROS_ASSERT(condition) assert(condition)
#define ROS_BREAK() raise(SIGTRAP)
#define ROS_STREAM_LOG(level, expression) \
    do { std::ostringstream ros_stream; ros_stream << expression; \
         RCLCPP_##level(::ros_utils::logger(), "%s", ros_stream.str().c_str()); } while (false)
#define ROS_DEBUG_STREAM(expression) ROS_STREAM_LOG(DEBUG, expression)
#define ROS_INFO_STREAM(expression) ROS_STREAM_LOG(INFO, expression)
#define ROS_WARN_STREAM(expression) ROS_STREAM_LOG(WARN, expression)
#define ROS_ERROR_STREAM(expression) ROS_STREAM_LOG(ERROR, expression)
#define ROS_STREAM_LOG_THROTTLE(level, period, expression) \
    do { std::ostringstream ros_stream; ros_stream << expression; \
         RCLCPP_##level##_THROTTLE(::ros_utils::logger(), ::ros_utils::clock(), \
             static_cast<int64_t>((period) * 1000.0), "%s", ros_stream.str().c_str()); } while (false)
#define ROS_INFO_STREAM_THROTTLE(period, expression) ROS_STREAM_LOG_THROTTLE(INFO, period, expression)
#define ROS_WARN_STREAM_THROTTLE(period, expression) ROS_STREAM_LOG_THROTTLE(WARN, period, expression)
#endif
