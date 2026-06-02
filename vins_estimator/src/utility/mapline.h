#ifndef MAPLINE_H_
#define MAPLINE_H_

#include <limits>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "line_geometry.h"

class Mapline
{
  public:
    enum Type
    {
        UnTriangulated = 0,
        Good = 1,
        Bad = 2,
    };

    Mapline();
    Mapline(int line_id);
    void SetId(int id);
    int GetId();
    void SetType(const Type &type);
    Type GetType();
    void SetUnTriangulated();
    bool IsUnTriangulated();
    void SetBad();
    bool IsBad();
    void SetGood();
    bool IsValid();

    void SetEndpoints(Vector6d &p, bool compute_line3d = true);
    Vector6d &GetEndpoints();
    void SetEndpointsValidStatus(bool status);
    bool EndpointsValid();
    void SetEndpointsUpdateStatus(bool status);
    bool ToUpdateEndpoints();
    void SetLine3D(const Vector6d &line_3d);
    const Vector6d &GetLine3D() const;

    void AddObserver(const int &frame_id, const int &line_index);
    void RemoveObserver(const int &frame_id);
    int ObserverNum();
    const std::map<int, int> &GetAllObservers();
    int GetLineIdx(int frame_id);

    void SetObserverEndpointStatus(int frame_id, int status = 1);
    int GetObserverEndpointStatus(int frame_id);
    const std::map<int, int> &GetAllObserverEndpointStatus();

  public:
    int local_map_optimization_frame_id;

  private:
    int _id;
    Type _type;
    bool _to_update_endpoints;
    bool _endpoints_valid;
    Vector6d _endpoints;
    Vector6d _line_3d_world;
    std::map<int, int> _obversers;
    std::map<int, int> _included_endpoints;
};
typedef std::shared_ptr<Mapline> MaplinePtr;

#endif  // MAPLINE_H_
