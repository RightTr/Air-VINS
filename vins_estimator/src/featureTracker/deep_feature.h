#pragma once

#include <string>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class PointMatcher;
class SuperPoint;
class PLNet;

struct DeepFeatureOptions
{
    std::string model_dir;
    int matcher = 1;              // 0: LightGlue, 1: SuperGlue
    int max_keypoints = 150;
    float keypoint_threshold = 0.004f;
    int remove_borders = 4;
    int stereo_ransac = 0;
    int image_width = 640;
    int image_height = 512;
    float line_threshold = 0.5f;
    float line_length_threshold = 50.0f;
};

class DeepFeature
{
public:
    DeepFeature() = default;
    ~DeepFeature() = default;

    // mode: 1 = SuperPoint, 2 = PLNET
    bool init(int mode, const DeepFeatureOptions &options);

    bool ready() const { return ready_; }

    bool extractPoints(const cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &points);

    bool extractLines(const cv::Mat &image, std::vector<Eigen::Vector4d> &lines);

    bool matchPoints(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features0,
                     const Eigen::Matrix<float, 259, Eigen::Dynamic> &features1,
                     std::vector<cv::DMatch> &matches,
                     bool outlier_rejection = false);

private:
    int mode_ = 0;
    DeepFeatureOptions options_;
    bool ready_ = false;

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<PLNet> plnet_;
    std::shared_ptr<PointMatcher> point_matcher_;
};
