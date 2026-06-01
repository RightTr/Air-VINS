#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include "utility/ros_utils.h"

namespace
{
struct Frame
{
    std::string path;
    double stamp = 0.0;
};

struct ImuSample
{
    double stamp = 0.0;
    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
};

struct Rectifier
{
    bool enabled = true;
    cv::Mat map_left_x;
    cv::Mat map_left_y;
    cv::Mat map_right_x;
    cv::Mat map_right_y;
};

std::string joinPath(const std::string &base, const std::string &name)
{
    if (base.empty())
        return name;
    if (base.back() == '/')
        return base + name;
    return base + "/" + name;
}

bool hasPngExtension(const std::string &name)
{
    const std::string suffix = ".png";
    if (name.size() < suffix.size())
        return false;
    for (size_t i = 0; i < suffix.size(); ++i)
    {
        const char a = static_cast<char>(std::tolower(name[name.size() - suffix.size() + i]));
        if (a != suffix[i])
            return false;
    }
    return true;
}

bool parseStamp(const std::string &name, double &stamp)
{
    const size_t end = name.find_last_of('.');
    if (end == std::string::npos)
        return false;
    const std::string stem = name.substr(0, end);
    char *parse_end = nullptr;
    errno = 0;
    const double value = std::strtod(stem.c_str(), &parse_end);
    if (errno != 0 || parse_end == stem.c_str() || *parse_end != '\0')
        return false;
    stamp = value;
    return true;
}

bool parseImuSample(const std::string &line, ImuSample &sample)
{
    std::istringstream ss(line);
    return static_cast<bool>(ss >> sample.stamp
                                >> sample.gx >> sample.gy >> sample.gz
                                >> sample.ax >> sample.ay >> sample.az);
}

std::vector<Frame> loadFrames(const std::string &directory)
{
    std::vector<Frame> frames;
    DIR *dir = opendir(directory.c_str());
    if (dir == nullptr)
    {
        ROS_ERROR_STREAM("Failed to open directory: " << directory);
        return frames;
    }

    while (dirent *entry = readdir(dir))
    {
        const std::string name(entry->d_name);
        if (!hasPngExtension(name))
            continue;

        double stamp = 0.0;
        if (!parseStamp(name, stamp))
        {
            ROS_WARN_STREAM("Skip image with non-timestamp filename: " << joinPath(directory, name));
            continue;
        }
        frames.push_back(Frame{joinPath(directory, name), stamp});
    }
    closedir(dir);

    std::sort(frames.begin(), frames.end(), [](const Frame &a, const Frame &b) {
        return a.stamp < b.stamp;
    });
    return frames;
}

std::vector<ImuSample> loadImuSamples(const std::string &path)
{
    std::vector<ImuSample> samples;
    std::ifstream input(path);
    if (!input.is_open())
    {
        ROS_ERROR_STREAM("Failed to open IMU file: " << path);
        return samples;
    }

    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        ImuSample sample;
        if (!parseImuSample(line, sample))
            continue;
        samples.push_back(sample);
    }

    std::sort(samples.begin(), samples.end(), [](const ImuSample &a, const ImuSample &b) {
        return a.stamp < b.stamp;
    });
    return samples;
}

size_t firstImuIndexForImageTime(const std::vector<ImuSample> &samples, double image_time)
{
    if (samples.empty())
        return 0;
    auto it = std::lower_bound(samples.begin(), samples.end(), image_time,
                               [](const ImuSample &sample, double stamp) {
                                   return sample.stamp < stamp;
                               });
    if (it == samples.begin())
        return 0;
    return static_cast<size_t>((it - samples.begin()) - 1);
}

Rectifier makeNusThermalRectifier(bool enabled, double alpha)
{
    Rectifier rectifier;
    rectifier.enabled = enabled;
    if (!enabled)
        return rectifier;

    const cv::Size image_size(640, 512);
    const cv::Mat K0 = (cv::Mat_<double>(3, 3) << 344.08059125, 0.0, 320.20695996,
                       0.0, 343.71871889, 271.73937497,
                       0.0, 0.0, 1.0);
    const cv::Mat D0 = (cv::Mat_<double>(1, 5) << -0.21823207, 0.04657087, 0.0, 0.0, 0.0);
    const cv::Mat K1 = (cv::Mat_<double>(3, 3) << 342.69763215, 0.0, 323.56293972,
                       0.0, 342.45604858, 267.49213891,
                       0.0, 0.0, 1.0);
    const cv::Mat D1 = (cv::Mat_<double>(1, 5) << -0.22042839, 0.04895111, 0.0, 0.0, 0.0);
    const cv::Mat R = (cv::Mat_<double>(3, 3) << 0.9998928479139975, 0.006664201062753905, 0.013033844967273217,
                      -0.0065988424903717065, 0.999965470127153, -0.005051121773369857,
                      -0.01306705660135398, 0.004964572245152228, 0.9999022977542356);
    const cv::Mat T = (cv::Mat_<double>(3, 1) << -0.1216520307054718,
                       0.00037876701143810795,
                      -0.0015966275775746094);

    cv::Mat R0_rect, R1_rect, P0_rect, P1_rect, Q;
    cv::Rect roi0, roi1;
    cv::stereoRectify(K0, D0, K1, D1, image_size, R, T,
                      R0_rect, R1_rect, P0_rect, P1_rect, Q,
                      cv::CALIB_ZERO_DISPARITY, alpha, image_size, &roi0, &roi1);

    cv::initUndistortRectifyMap(K0, D0, R0_rect, P0_rect, image_size, CV_16SC2,
                                rectifier.map_left_x, rectifier.map_left_y);
    cv::initUndistortRectifyMap(K1, D1, R1_rect, P1_rect, image_size, CV_16SC2,
                                rectifier.map_right_x, rectifier.map_right_y);

    ROS_INFO_STREAM("NUS thermal rectification enabled alpha=" << alpha
                    << " fx=" << P0_rect.at<double>(0, 0)
                    << " cx=" << P0_rect.at<double>(0, 2)
                    << " cy=" << P0_rect.at<double>(1, 2)
                    << " baseline=" << -P1_rect.at<double>(0, 3) / P1_rect.at<double>(0, 0));
    return rectifier;
}

sensor_msgs::ImagePtr readMonoImage(const std::string &path,
                                    const ros::Time &stamp,
                                    const std::string &frame_id,
                                    const Rectifier &rectifier,
                                    bool is_left)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        ROS_ERROR_STREAM("Failed to read image: " << path);
        return sensor_msgs::ImagePtr();
    }
    if (rectifier.enabled)
    {
        cv::Mat rectified;
        const cv::Mat &map_x = is_left ? rectifier.map_left_x : rectifier.map_right_x;
        const cv::Mat &map_y = is_left ? rectifier.map_left_y : rectifier.map_right_y;
        cv::remap(image, rectified, map_x, map_y, cv::INTER_LINEAR);
        image = rectified;
    }

    std_msgs::Header header;
    header.stamp = stamp;
    header.frame_id = frame_id;
    return cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, image).toImageMsg();
}

sensor_msgs::Imu makeImuMsg(const ImuSample &sample, const std::string &frame_id)
{
    sensor_msgs::Imu msg;
    msg.header.stamp = ros::Time().fromSec(sample.stamp);
    msg.header.frame_id = frame_id;
    msg.angular_velocity.x = sample.gx;
    msg.angular_velocity.y = sample.gy;
    msg.angular_velocity.z = sample.gz;
    msg.linear_acceleration.x = sample.ax;
    msg.linear_acceleration.y = sample.ay;
    msg.linear_acceleration.z = sample.az;
    return msg;
}
} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "nus_thermal_publisher");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string data_root;
    std::string left_dir;
    std::string right_dir;
    std::string image0_topic;
    std::string image1_topic;
    std::string imu_file;
    std::string imu_topic;
    std::string left_frame_id;
    std::string right_frame_id;
    std::string imu_frame_id;
    double playback_rate = 1.0;
    int start_index = 0;
    int max_frames = 0;
    bool loop = false;
    bool rectify_images = true;
    bool publish_imu = false;
    double rectify_alpha = 0.0;

    pnh.param<std::string>("data_root", data_root, "/mnt/Data/homebrew_thermal/Nus/run_20250912_180927");
    pnh.param<std::string>("left_dir", left_dir, "left_thermal/left_motion");
    pnh.param<std::string>("right_dir", right_dir, "right_thermal/right_motion");
    pnh.param<std::string>("imu_file", imu_file, "realsense/imu/imu_synced.txt");
    pnh.param<std::string>("image0_topic", image0_topic, "/cam0/image_raw");
    pnh.param<std::string>("image1_topic", image1_topic, "/cam1/image_raw");
    pnh.param<std::string>("imu_topic", imu_topic, "/imu0");
    pnh.param<std::string>("left_frame_id", left_frame_id, "nus_thermal_left");
    pnh.param<std::string>("right_frame_id", right_frame_id, "nus_thermal_right");
    pnh.param<std::string>("imu_frame_id", imu_frame_id, "realsense_imu");
    pnh.param("playback_rate", playback_rate, 1.0);
    pnh.param("start_index", start_index, 0);
    pnh.param("max_frames", max_frames, 0);
    pnh.param("loop", loop, false);
    pnh.param("rectify_images", rectify_images, true);
    pnh.param("publish_imu", publish_imu, false);
    pnh.param("rectify_alpha", rectify_alpha, 0.0);

    if (playback_rate <= 0.0)
    {
        ROS_ERROR_STREAM("playback_rate must be positive, got " << playback_rate);
        return 1;
    }
    if (start_index < 0)
    {
        ROS_ERROR_STREAM("start_index must be non-negative, got " << start_index);
        return 1;
    }

    const std::string left_path = left_dir.empty() || left_dir[0] == '/' ? left_dir : joinPath(data_root, left_dir);
    const std::string right_path = right_dir.empty() || right_dir[0] == '/' ? right_dir : joinPath(data_root, right_dir);
    const std::string imu_path = imu_file.empty() || imu_file[0] == '/' ? imu_file : joinPath(data_root, imu_file);
    const std::vector<Frame> left_frames = loadFrames(left_path);
    const std::vector<Frame> right_frames = loadFrames(right_path);
    const std::vector<ImuSample> imu_samples = publish_imu ? loadImuSamples(imu_path) : std::vector<ImuSample>();
    if (left_frames.empty() || right_frames.empty())
    {
        ROS_ERROR_STREAM("No frames found. left=" << left_frames.size() << " right=" << right_frames.size());
        return 1;
    }
    if (publish_imu && imu_samples.empty())
    {
        ROS_ERROR_STREAM("publish_imu is true but no IMU samples were loaded from " << imu_path);
        return 1;
    }

    const size_t pair_count = std::min(left_frames.size(), right_frames.size());
    if (left_frames.size() != right_frames.size())
    {
        ROS_WARN_STREAM("Left/right frame count differs; publishing " << pair_count << " pairs"
                        << " left=" << left_frames.size() << " right=" << right_frames.size());
    }
    if (static_cast<size_t>(start_index) >= pair_count)
    {
        ROS_ERROR_STREAM("start_index " << start_index << " is outside pair count " << pair_count);
        return 1;
    }

    size_t end_index = pair_count;
    if (max_frames > 0)
        end_index = std::min(pair_count, static_cast<size_t>(start_index + max_frames));

    ros::Publisher pub_left = ros_utils::ros_advertise<sensor_msgs::Image>(nh, image0_topic, 10);
    ros::Publisher pub_right = ros_utils::ros_advertise<sensor_msgs::Image>(nh, image1_topic, 10);
    ros::Publisher pub_imu;
    if (publish_imu)
        pub_imu = ros_utils::ros_advertise<sensor_msgs::Imu>(nh, imu_topic, 2000);
    const Rectifier rectifier = makeNusThermalRectifier(rectify_images, rectify_alpha);

    ROS_INFO_STREAM("Publishing NUS thermal stereo sequence"
                    << " left=" << left_path
                    << " right=" << right_path
                    << " imu=" << (publish_imu ? imu_path : std::string("disabled"))
                    << " pairs=" << (end_index - static_cast<size_t>(start_index))
                    << " playback_rate=" << playback_rate);

    do
    {
        size_t imu_index = publish_imu ? firstImuIndexForImageTime(imu_samples, left_frames[start_index].stamp) : 0;
        for (size_t i = static_cast<size_t>(start_index); ros::ok() && i < end_index; ++i)
        {
            const ros::Time stamp = ros::Time().fromSec(left_frames[i].stamp);
            if (publish_imu)
            {
                while (imu_index < imu_samples.size() && imu_samples[imu_index].stamp < left_frames[i].stamp)
                {
                    ros_utils::ros_publish(pub_imu, makeImuMsg(imu_samples[imu_index], imu_frame_id));
                    ++imu_index;
                }
                if (imu_index < imu_samples.size())
                {
                    ros_utils::ros_publish(pub_imu, makeImuMsg(imu_samples[imu_index], imu_frame_id));
                    ++imu_index;
                }
            }
            sensor_msgs::ImagePtr left_msg = readMonoImage(left_frames[i].path, stamp, left_frame_id, rectifier, true);
            sensor_msgs::ImagePtr right_msg = readMonoImage(right_frames[i].path, stamp, right_frame_id, rectifier, false);
            if (!left_msg || !right_msg)
                return 1;

            ros_utils::ros_publish(pub_left, *left_msg);
            ros_utils::ros_publish(pub_right, *right_msg);
            ros::spinOnce();

            if (i + 1 < end_index)
            {
                double dt = left_frames[i + 1].stamp - left_frames[i].stamp;
                if (dt < 0.0)
                    dt = 0.0;
                ros::Duration(dt / playback_rate).sleep();
            }
        }
    } while (ros::ok() && loop);

    return 0;
}
