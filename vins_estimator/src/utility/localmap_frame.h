#pragma once

#include <map>
#include <vector>

#include <eigen3/Eigen/Dense>

struct LocalFrameObservation
{
    int camera_id;
    Eigen::Matrix<double, 7, 1> feature;

    LocalFrameObservation()
        : camera_id(0)
    {
        feature.setZero();
    }

    LocalFrameObservation(int _camera_id, const Eigen::Matrix<double, 7, 1> & _feature)
        : camera_id(_camera_id), feature(_feature)
    {
    }
};

class LocalFrame
{
  public:
    int frame_id;
    double timestamp;
    int tracking_mode;
    int active_camera_id;
    int track_count;
    std::map<int, LocalFrameObservation> observations;

    LocalFrame()
        : frame_id(-1), timestamp(0.0), tracking_mode(0), active_camera_id(0), track_count(0)
    {
    }

    LocalFrame(int _frame_id, double _timestamp, int _tracking_mode, int _active_camera_id)
        : frame_id(_frame_id), timestamp(_timestamp), tracking_mode(_tracking_mode),
          active_camera_id(_active_camera_id), track_count(0)
    {
    }

    void addObservation(int feature_id, int camera_id, const Eigen::Matrix<double, 7, 1> &feature)
    {
        observations[feature_id] = LocalFrameObservation(camera_id, feature);
        track_count = static_cast<int>(observations.size());
    }

    bool hasObservation(int feature_id) const
    {
        return observations.find(feature_id) != observations.end();
    }

    std::vector<int> featureIds() const
    {
        std::vector<int> ids;
        ids.reserve(observations.size());
        for (const auto &kv : observations)
            ids.push_back(kv.first);
        return ids;
    }
};
