#include "deep_feature.h"

#include <ros/ros.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "deepFeature/include/super_point.h"
#include "deepFeature/include/super_glue.h"
#include "deepFeature/include/light_glue.h"
#include "deepFeature/include/point_matcher.h"
#include "deepFeature/include/plnet.h"
#include "deepFeature/include/read_configs.h"
#include "deepFeature/include/utils.h"

#include <sys/stat.h>

static std::string joinPath(const std::string &folder, const std::string &name)
{
    if (folder.empty())
        return name;
    if (folder.back() == '/')
        return folder + name;
    return folder + "/" + name;
}

static bool fileExists(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

static PointMatcherConfig buildPointMatcherConfig(const DeepFeatureOptions &options)
{
    PointMatcherConfig config;
    config.matcher = options.matcher;
    config.image_width = options.image_width;
    config.image_height = options.image_height;

    config.dla_core = -1;

    if (options.matcher == 0) // LightGlue
    {
        config.onnx_file = joinPath(options.model_dir, "superpoint_lightglue.onnx");
        config.engine_file = joinPath(options.model_dir, "superpoint_lightglue.engine");
    }
    else // SuperGlue
    {
        const std::string indoor_onnx = joinPath(options.model_dir, "superglue_indoor_sim_int32.onnx");
        const std::string indoor_engine = joinPath(options.model_dir, "superglue_indoor_sim_int32.engine");
        const std::string outdoor_onnx = joinPath(options.model_dir, "superglue_outdoor_sim_int32.onnx");
        const std::string outdoor_engine = joinPath(options.model_dir, "superglue_outdoor_sim_int32.engine");
        if (fileExists(indoor_onnx) || fileExists(indoor_engine))
        {
            config.onnx_file = indoor_onnx;
            config.engine_file = indoor_engine;
        }
        else
        {
            config.onnx_file = outdoor_onnx;
            config.engine_file = outdoor_engine;
        }
    }

    ROS_INFO("DeepFeature matcher config: matcher=%d, onnx=%s, engine=%s",
             config.matcher, config.onnx_file.c_str(), config.engine_file.c_str());
    return config;
}

bool DeepFeature::init(int mode, const DeepFeatureOptions &options)
{
    mode_ = mode;
    options_ = options;

    switch (mode)
    {
    case 1:
    {
        SuperPointConfig sp_config;
        sp_config.max_keypoints = options.max_keypoints;
        sp_config.keypoint_threshold = options.keypoint_threshold;
        sp_config.remove_borders = options.remove_borders;
        sp_config.dla_core = -1;
        sp_config.input_tensor_names = {"input"};
        sp_config.output_tensor_names = {"scores", "descriptors"};
        sp_config.onnx_file = joinPath(options.model_dir, "superpoint_v1_sim_int32.onnx");
        sp_config.engine_file = joinPath(options.model_dir, "superpoint_v1_sim_int32.engine");

        superpoint_ = std::make_shared<SuperPoint>(sp_config);
        if (!superpoint_->build())
        {
            ROS_ERROR("DeepFeature: Failed to build SuperPoint engine");
            ready_ = false;
            return false;
        }

        PointMatcherConfig pm_config = buildPointMatcherConfig(options);
        point_matcher_ = std::make_shared<PointMatcher>(pm_config);
        if (!point_matcher_->ready())
        {
            ROS_ERROR("DeepFeature: Failed to initialize PointMatcher");
            ready_ = false;
            return false;
        }

        ready_ = true;
        return true;
    }

    case 2:
    {
        PLNetConfig plnet_config;
        plnet_config.use_superpoint = 0;
        plnet_config.max_keypoints = options.max_keypoints;
        plnet_config.keypoint_threshold = options.keypoint_threshold;
        plnet_config.remove_borders = options.remove_borders;
        plnet_config.line_threshold = options.line_threshold;
        plnet_config.line_length_threshold = options.line_length_threshold;
        plnet_config.SetModelPath(options.model_dir);

        plnet_ = std::make_shared<PLNet>(plnet_config);
        if (!plnet_->build())
        {
            ROS_ERROR("DeepFeature: Failed to build PLNET engine");
            ready_ = false;
            return false;
        }

        PointMatcherConfig pm_config = buildPointMatcherConfig(options);
        point_matcher_ = std::make_shared<PointMatcher>(pm_config);
        if (!point_matcher_->ready())
        {
            ROS_ERROR("DeepFeature: Failed to initialize PointMatcher for PLNET");
            ready_ = false;
            return false;
        }

        ready_ = true;
        return true;
    }

    default:
        ready_ = false;
        ROS_ERROR("DeepFeature: invalid mode %d. Supported: 1=SuperPoint, 2=PLNET", mode);
        return false;
    }
}

bool DeepFeature::extractPoints(const cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &points)
{
    switch (mode_)
    {
    case 1:
    {
        if (!ready_ || !superpoint_)
        {
            ROS_WARN_THROTTLE(5.0, "ExtractPoints: SuperPoint not ready");
            points.resize(259, 0);
            return false;
        }
        return superpoint_->infer(image, points);
    }

    case 2:
    {
        if (!ready_ || !plnet_)
        {
            ROS_WARN_THROTTLE(5.0, "ExtractPoints: PLNET not ready");
            points.resize(259, 0);
            return false;
        }
        std::vector<Eigen::Vector4d> lines;
        Eigen::Matrix<float, 259, Eigen::Dynamic> junctions;
        return plnet_->infer(image, points, lines, junctions);
    }

    default:
        ROS_WARN_THROTTLE(5.0, "ExtractPoints: unsupported mode %d. Returning empty features.", mode_);
        points.resize(259, 0);
        return false;
    }
}

bool DeepFeature::extractLines(const cv::Mat &image, std::vector<Eigen::Vector4d> &lines)
{
    switch (mode_)
    {
    case 2:
    {
        if (!ready_ || !plnet_)
        {
            ROS_WARN_THROTTLE(5.0, "ExtractLines: PLNET not ready");
            lines.clear();
            return false;
        }
        Eigen::Matrix<float, 259, Eigen::Dynamic> points;
        Eigen::Matrix<float, 259, Eigen::Dynamic> junctions;
        return plnet_->infer(image, points, lines, junctions);
    }

    case 1:
    default:
        ROS_WARN_THROTTLE(5.0, "ExtractLines: not supported in mode %d. Returning empty lines.", mode_);
        lines.clear();
        return false;
    }
}

bool DeepFeature::extractPointsLines(const cv::Mat &image,
                                    Eigen::Matrix<float, 259, Eigen::Dynamic> &points,
                                    std::vector<Eigen::Vector4d> &lines)
{
    if (mode_ != 2 || !ready_ || !plnet_)
    {
        ROS_WARN_THROTTLE(5.0, "ExtractPointLines: PLNET not ready");
        points.resize(259, 0);
        lines.clear();
        return false;
    }
    Eigen::Matrix<float, 259, Eigen::Dynamic> junctions;
    return plnet_->infer(image, points, lines, junctions);
}

bool DeepFeature::matchPoints(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features0,
                              const Eigen::Matrix<float, 259, Eigen::Dynamic> &features1,
                              std::vector<cv::DMatch> &matches,
                              bool outlier_rejection)
{
    if (!ready_ || !point_matcher_)
    {
        ROS_WARN_THROTTLE(5.0, "MatchPoints: not ready or matcher not initialized");
        matches.clear();
        return false;
    }

    int num_matches = point_matcher_->MatchingPoints(features0, features1, matches, outlier_rejection);
    return num_matches >= 0;
}
