#pragma once

#include <map>
#include <memory>

#include <eigen3/Eigen/Dense>

class Mappoint
{
  public:
    enum class State
    {
        UnTriangulated = 0,
        Good = 1,
        Bad = 2
    };

    const int mappoint_id;

    explicit Mappoint(int mappoint_id);
    Mappoint(int mappoint_id, const Eigen::Vector3d &position);

    State GetState() const;
    bool IsUnTriangulated() const;
    bool IsGood() const;
    bool IsBad() const;
    bool IsValid() const;

    bool HasPosition() const;
    const Eigen::Vector3d &GetPosition() const;
    void SetPosition(const Eigen::Vector3d &position);

    void SetUnTriangulated();
    void SetGood();
    void SetBad();

    void AddObverser(int frame_id, int slot_id = -1);
    void RemoveObverser(int frame_id);
    void ClearObversers();
    int ObverserNum() const;

    void SetTrackCount(int track_count);
    int TrackCount() const;

    void SetDescriptor(const Eigen::Matrix<float, 259, 1> &descriptor);
    bool HasDescriptor() const;
    const Eigen::Matrix<float, 259, 1> &GetDescriptor() const;

  private:
    State state_;
    Eigen::Vector3d position_;
    bool has_position_;
    std::map<int, int> observers_;
    int track_count_;
    bool has_descriptor_;
    Eigen::Matrix<float, 259, 1> descriptor_;
};

using MappointPtr = std::shared_ptr<Mappoint>;
