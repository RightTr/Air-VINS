#include "mappoint.h"

#include <algorithm>

Mappoint::Mappoint(int mappoint_id)
    : mappoint_id(mappoint_id), state_(State::UnTriangulated), position_(Eigen::Vector3d::Zero()),
      has_position_(false), track_count_(0), has_descriptor_(false), stale_count_(0),
      window_support_count_(0), last_window_support_frame_id_(-1)
{
    descriptor_.setZero();
}

Mappoint::Mappoint(int mappoint_id, const Eigen::Vector3d &position)
    : mappoint_id(mappoint_id), state_(State::UnTriangulated), position_(position),
      has_position_(true), track_count_(0), has_descriptor_(false), stale_count_(0),
      window_support_count_(0), last_window_support_frame_id_(-1)
{
    descriptor_.setZero();
}

Mappoint::State Mappoint::GetState() const
{
    return state_;
}

bool Mappoint::IsUnTriangulated() const
{
    return state_ == State::UnTriangulated;
}

bool Mappoint::IsGood() const
{
    return state_ == State::Good;
}

bool Mappoint::IsBad() const
{
    return state_ == State::Bad;
}

bool Mappoint::IsValid() const
{
    return IsGood();
}

bool Mappoint::HasPosition() const
{
    return has_position_;
}

const Eigen::Vector3d &Mappoint::GetPosition() const
{
    return position_;
}

void Mappoint::SetPosition(const Eigen::Vector3d &position)
{
    position_ = position;
    has_position_ = position.allFinite();
}

void Mappoint::SetUnTriangulated()
{
    if (!IsBad())
        state_ = State::UnTriangulated;
}

void Mappoint::SetGood()
{
    if (!IsBad())
        state_ = State::Good;
}

void Mappoint::SetBad()
{
    state_ = State::Bad;
}

void Mappoint::AddObverser(int frame_id, int slot_id)
{
    observers_[frame_id] = slot_id;
}

void Mappoint::RemoveObverser(int frame_id)
{
    observers_.erase(frame_id);
}

void Mappoint::ClearObversers()
{
    observers_.clear();
}

int Mappoint::ObverserNum() const
{
    return static_cast<int>(observers_.size());
}

const std::map<int, int> &Mappoint::GetAllObversers() const
{
    return observers_;
}

void Mappoint::SetTrackCount(int track_count)
{
    track_count_ = track_count;
}

int Mappoint::TrackCount() const
{
    return track_count_;
}

void Mappoint::ResetStaleCount()
{
    stale_count_ = 0;
}

void Mappoint::IncreaseStaleCount()
{
    ++stale_count_;
}

int Mappoint::StaleCount() const
{
    return stale_count_;
}

void Mappoint::SetWindowSupport(int support_count, int last_support_frame_id)
{
    window_support_count_ = std::max(0, support_count);
    last_window_support_frame_id_ = last_support_frame_id;
}

int Mappoint::WindowSupportCount() const
{
    return window_support_count_;
}

int Mappoint::LastWindowSupportFrameId() const
{
    return last_window_support_frame_id_;
}

void Mappoint::SetDescriptor(const Eigen::Matrix<float, 259, 1> &descriptor)
{
    descriptor_ = descriptor;
    has_descriptor_ = true;
}

bool Mappoint::HasDescriptor() const
{
    return has_descriptor_;
}

const Eigen::Matrix<float, 259, 1> &Mappoint::GetDescriptor() const
{
    return descriptor_;
}
