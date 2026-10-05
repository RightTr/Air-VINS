#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include "utility/ros_utils.h"

namespace
{
struct FramePair
{
    std::string left_path;
    std::string right_path;
    double stamp = 0.0;
};

struct ImuSample
{
    double stamp = 0.0;
    double gx = 0.0, gy = 0.0, gz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
};

std::string joinPath(const std::string &base, const std::string &name)
{
    return base.empty() || base.back() == '/' ? base + name : base + "/" + name;
}

std::string resolvePath(const std::string &root, const std::string &path)
{
    return !path.empty() && path.front() == '/' ? path : joinPath(root, path);
}

std::string frameName(size_t index)
{
    std::ostringstream name;
    name << std::setw(8) << std::setfill('0') << index << ".png";
    return name.str();
}

bool fileExists(const std::string &path)
{
    return std::ifstream(path).good();
}

std::vector<double> loadTimestamps(const std::string &path)
{
    std::ifstream input(path);
    if (!input)
    {
        ROS_ERROR_STREAM("Cannot open timestamps: " << path);
        return {};
    }

    std::vector<double> stamps;
    double stamp = 0.0;
    while (input >> stamp)
    {
        if (!std::isfinite(stamp) || stamp < 0.0 || (!stamps.empty() && stamp <= stamps.back()))
        {
            ROS_ERROR_STREAM("Invalid or non-increasing timestamp at index " << stamps.size() << " in " << path);
            return {};
        }
        stamps.push_back(stamp);
    }
    if (!input.eof() || stamps.empty())
    {
        ROS_ERROR_STREAM("Empty or malformed timestamps: " << path);
        return {};
    }
    return stamps;
}

std::vector<FramePair> loadPairs(const std::string &left_dir, const std::string &right_dir,
                                 double stamp_tolerance)
{
    const auto left_stamps = loadTimestamps(joinPath(left_dir, "timestamps.txt"));
    const auto right_stamps = loadTimestamps(joinPath(right_dir, "timestamps.txt"));
    if (left_stamps.empty() || right_stamps.empty() || left_stamps.size() != right_stamps.size())
    {
        ROS_ERROR_STREAM("Left/right timestamp count differs: " << left_stamps.size()
                         << " / " << right_stamps.size());
        return {};
    }

    std::vector<FramePair> pairs;
    pairs.reserve(left_stamps.size());
    for (size_t i = 0; i < left_stamps.size(); ++i)
    {
        const std::string name = frameName(i);
        const std::string left = joinPath(left_dir, name);
        const std::string right = joinPath(right_dir, name);
        if (std::abs(left_stamps[i] - right_stamps[i]) > stamp_tolerance ||
            !fileExists(left) || !fileExists(right))
        {
            ROS_WARN_STREAM("Skip incomplete or unsynchronized stereo frame " << i
                            << " (dt=" << std::abs(left_stamps[i] - right_stamps[i]) << " s)");
            continue;
        }
        // Both images represent one stereo observation. Use the left exposure time for both
        // messages so the estimator's stereo queue never discards a valid pair.
        pairs.push_back({left, right, left_stamps[i]});
    }
    return pairs;
}

std::vector<ImuSample> loadImu(const std::string &imu_dir)
{
    const auto stamps = loadTimestamps(joinPath(imu_dir, "timestamps.txt"));
    std::ifstream input(joinPath(imu_dir, "data.txt"));
    if (stamps.empty() || !input)
    {
        ROS_ERROR_STREAM("Cannot load IMU data from " << imu_dir);
        return {};
    }

    std::vector<ImuSample> samples;
    samples.reserve(stamps.size());
    std::string line;
    for (size_t i = 0; i < stamps.size(); ++i)
    {
        if (!std::getline(input, line))
        {
            ROS_ERROR_STREAM("IMU data ends before timestamp " << i);
            return {};
        }
        ImuSample sample;
        sample.stamp = stamps[i];
        std::istringstream values(line);
        if (!(values >> sample.gx >> sample.gy >> sample.gz
                     >> sample.ax >> sample.ay >> sample.az) ||
            !std::isfinite(sample.gx) || !std::isfinite(sample.gy) || !std::isfinite(sample.gz) ||
            !std::isfinite(sample.ax) || !std::isfinite(sample.ay) || !std::isfinite(sample.az))
        {
            ROS_ERROR_STREAM("Invalid IMU row " << i << " in " << imu_dir);
            return {};
        }
        samples.push_back(sample);
    }
    if (std::getline(input, line))
    {
        ROS_ERROR_STREAM("IMU data has more rows than timestamps in " << imu_dir);
        return {};
    }
    return samples;
}

sensor_msgs::ImagePtr readImage(const std::string &path, double stamp, const std::string &frame_id)
{
    const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty() || image.type() != CV_8UC1)
    {
        ROS_ERROR_STREAM("Expected an 8-bit single-channel image: " << path);
        return {};
    }
    std_msgs::Header header;
    header.stamp = ros_utils::time_from_sec(stamp);
    header.frame_id = frame_id;
    return cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, image).toImageMsg();
}

sensor_msgs::Imu makeImuMessage(const ImuSample &sample, const std::string &frame_id)
{
    sensor_msgs::Imu message;
    message.header.stamp = ros_utils::time_from_sec(sample.stamp);
    message.header.frame_id = frame_id;
    message.angular_velocity.x = sample.gx;
    message.angular_velocity.y = sample.gy;
    message.angular_velocity.z = sample.gz;
    message.linear_acceleration.x = sample.ax;
    message.linear_acceleration.y = sample.ay;
    message.linear_acceleration.z = sample.az;
    return message;
}

template <typename ImagePublisher, typename ImuPublisher>
void waitForSubscribers(const ImagePublisher &left, const ImagePublisher &right,
                        const ImuPublisher &imu, bool publish_imu, double timeout_seconds)
{
    if (timeout_seconds <= 0.0)
        return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
    while (ros_utils::ok() && std::chrono::steady_clock::now() < deadline)
    {
        if (ros_utils::subscription_count<sensor_msgs::Image>(left) > 0 &&
            ros_utils::subscription_count<sensor_msgs::Image>(right) > 0 &&
            (!publish_imu || ros_utils::subscription_count<sensor_msgs::Imu>(imu) > 0))
            return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ROS_WARN("Subscriber wait timed out; playback will continue");
}
} // namespace

int main(int argc, char **argv)
{
    auto node = ros_utils::init(argc, argv, "tartan_thermal_publisher");
    std::string data_root, left_dir, right_dir, imu_dir;
    std::string image0_topic, image1_topic, imu_topic;
    std::string left_frame_id, right_frame_id, imu_frame_id;
    double playback_rate = 1.0, stereo_stamp_tolerance = 0.003, subscriber_wait_timeout = 10.0;
    int start_index = 0, max_frames = 0;
    bool publish_imu = true, loop = false;

    ros_utils::param(node, "data_root", data_root,
                     std::string("/mnt/Data/TartanThermal/TartanRGBT_dataset/day1/undistorted_images_all_cameras_20250822_115703"));
    ros_utils::param(node, "left_dir", left_dir, std::string("thermal_left_rect_8"));
    ros_utils::param(node, "right_dir", right_dir, std::string("thermal_right_rect_8"));
    ros_utils::param(node, "imu_dir", imu_dir, std::string("zed_imu_raw"));
    ros_utils::param(node, "image0_topic", image0_topic, std::string("/cam0/image_raw"));
    ros_utils::param(node, "image1_topic", image1_topic, std::string("/cam1/image_raw"));
    ros_utils::param(node, "imu_topic", imu_topic, std::string("/imu0"));
    ros_utils::param(node, "left_frame_id", left_frame_id, std::string("thermal_left/optical_frame"));
    ros_utils::param(node, "right_frame_id", right_frame_id, std::string("thermal_right/optical_frame"));
    ros_utils::param(node, "imu_frame_id", imu_frame_id, std::string("zed_imu_link"));
    ros_utils::param(node, "playback_rate", playback_rate, 1.0);
    ros_utils::param(node, "stereo_stamp_tolerance", stereo_stamp_tolerance, 0.003);
    ros_utils::param(node, "subscriber_wait_timeout", subscriber_wait_timeout, 10.0);
    ros_utils::param(node, "start_index", start_index, 0);
    ros_utils::param(node, "max_frames", max_frames, 0);
    ros_utils::param(node, "publish_imu", publish_imu, true);
    ros_utils::param(node, "loop", loop, false);

    if (playback_rate <= 0.0 || stereo_stamp_tolerance < 0.0 || subscriber_wait_timeout < 0.0 ||
        start_index < 0 || max_frames < 0)
    {
        ROS_ERROR("Invalid playback_rate, stereo_stamp_tolerance, subscriber_wait_timeout, start_index or max_frames");
        return 1;
    }

    const auto pairs = loadPairs(resolvePath(data_root, left_dir), resolvePath(data_root, right_dir),
                                 stereo_stamp_tolerance);
    const auto imu_samples = publish_imu ? loadImu(resolvePath(data_root, imu_dir)) : std::vector<ImuSample>{};
    if (pairs.empty() || (publish_imu && imu_samples.empty()) ||
        static_cast<size_t>(start_index) >= pairs.size())
    {
        ROS_ERROR_STREAM("No usable data or start_index out of range: pairs=" << pairs.size()
                         << " imu=" << imu_samples.size() << " start_index=" << start_index);
        return 1;
    }
    const size_t end_index = max_frames == 0 ? pairs.size() :
        std::min(pairs.size(), static_cast<size_t>(start_index) + static_cast<size_t>(max_frames));

    auto pub_left = ros_utils::ros_advertise<sensor_msgs::Image>(
        node, image0_topic, 10, false, ros_utils::QosProfile::SensorData);
    auto pub_right = ros_utils::ros_advertise<sensor_msgs::Image>(
        node, image1_topic, 10, false, ros_utils::QosProfile::SensorData);
    ros_utils::Publisher<sensor_msgs::Imu> pub_imu;
    if (publish_imu)
        pub_imu = ros_utils::ros_advertise<sensor_msgs::Imu>(
            node, imu_topic, 2000, false, ros_utils::QosProfile::SensorData);

    ROS_INFO_STREAM("TartanThermal: " << (end_index - static_cast<size_t>(start_index))
                    << " stereo pairs, " << imu_samples.size() << " IMU samples, rate=" << playback_rate);
    waitForSubscribers(pub_left, pub_right, pub_imu, publish_imu, subscriber_wait_timeout);

    do
    {
        auto imu_it = std::lower_bound(imu_samples.begin(), imu_samples.end(), pairs[start_index].stamp,
                                       [](const ImuSample &sample, double stamp) { return sample.stamp < stamp; });
        size_t imu_index = static_cast<size_t>(imu_it - imu_samples.begin());
        if (imu_index > 0)
            --imu_index;
        const auto wall_start = std::chrono::steady_clock::now();
        const double first_stamp = pairs[start_index].stamp;

        for (size_t i = static_cast<size_t>(start_index); ros_utils::ok() && i < end_index; ++i)
        {
            const auto elapsed = std::chrono::duration<double>((pairs[i].stamp - first_stamp) / playback_rate);
            std::this_thread::sleep_until(wall_start + elapsed);
            if (!ros_utils::ok())
                break;

            // Include the first IMU sample after this image, as VINS needs it to
            // interpolate IMU integration to the image timestamp.
            if (publish_imu)
            {
                while (imu_index < imu_samples.size() && imu_samples[imu_index].stamp < pairs[i].stamp)
                    ros_utils::ros_publish(pub_imu, makeImuMessage(imu_samples[imu_index++], imu_frame_id));
                if (imu_index < imu_samples.size())
                    ros_utils::ros_publish(pub_imu, makeImuMessage(imu_samples[imu_index++], imu_frame_id));
            }

            auto left = readImage(pairs[i].left_path, pairs[i].stamp, left_frame_id);
            auto right = readImage(pairs[i].right_path, pairs[i].stamp, right_frame_id);
            if (!left || !right)
                return 1;
            ros_utils::ros_publish(pub_left, *left);
            ros_utils::ros_publish(pub_right, *right);
        }
    } while (ros_utils::ok() && loop);

    return 0;
}
