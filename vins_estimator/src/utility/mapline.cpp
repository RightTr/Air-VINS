#include "mapline.h"
#include "line_geometry.h"

Mapline::Mapline() : local_map_optimization_frame_id(-1), _type(Type::UnTriangulated),
                     _to_update_endpoints(false), _endpoints_valid(false)
{
    _id = -1;
    _endpoints.setZero();
    _line_3d_world.setZero();
}

Mapline::Mapline(int line_id) : local_map_optimization_frame_id(-1), _id(line_id),
                                _type(Type::UnTriangulated), _to_update_endpoints(false), _endpoints_valid(false)
{
    _endpoints.setZero();
    _line_3d_world.setZero();
}

void Mapline::SetId(int id)
{
    _id = id;
}

int Mapline::GetId()
{
    return _id;
}

void Mapline::SetType(const Type &type)
{
    _type = type;
}

Mapline::Type Mapline::GetType()
{
    return _type;
}

void Mapline::SetUnTriangulated()
{
    _type = Type::UnTriangulated;
}

bool Mapline::IsUnTriangulated()
{
    return (_type == Type::UnTriangulated);
}

void Mapline::SetBad()
{
    _type = Type::Bad;
    _obversers.clear();
}

bool Mapline::IsBad()
{
    return (_type == Type::Bad);
}

void Mapline::SetGood()
{
    _type = Type::Good;
}

bool Mapline::IsValid()
{
    return (_type == Type::Good);
}

void Mapline::SetEndpoints(Vector6d &p, bool compute_line3d)
{
    if (compute_line3d)
    {
        if (!buildLineFromEndpoints(p.head<3>(), p.tail<3>(), _line_3d_world))
            return;
        if (_type == Type::UnTriangulated)
            _type = Type::Good;
    }
    _endpoints = p;
    _endpoints_valid = true;
}

Vector6d &Mapline::GetEndpoints()
{
    return _endpoints;
}

void Mapline::SetEndpointsValidStatus(bool status)
{
    _endpoints_valid = status;
}

bool Mapline::EndpointsValid()
{
    return _endpoints_valid;
}

void Mapline::SetEndpointsUpdateStatus(bool status)
{
    _to_update_endpoints = status;
}

bool Mapline::ToUpdateEndpoints()
{
    return _to_update_endpoints;
}

void Mapline::SetLine3D(const Vector6d &line_3d)
{
    if (!buildLineFromPlucker(line_3d, _line_3d_world))
        return;
    _to_update_endpoints = true;
    if (_type == Type::UnTriangulated)
        _type = Type::Good;
}

const Vector6d &Mapline::GetLine3D() const
{
    return _line_3d_world;
}

void Mapline::AddObverser(const int &frame_id, const int &line_index)
{
    _obversers[frame_id] = line_index;
}

void Mapline::RemoveObverser(const int &frame_id)
{
    auto it = _obversers.find(frame_id);
    if (it != _obversers.end())
        _obversers.erase(it);

    it = _included_endpoints.find(frame_id);
    if (it != _included_endpoints.end())
        _included_endpoints.erase(it);
}

int Mapline::ObverserNum()
{
    int obverser_num = 0;
    for (const auto &kv : _obversers)
    {
        if (kv.second >= 0)
            ++obverser_num;
    }
    return obverser_num;
}

const std::map<int, int> &Mapline::GetAllObversers()
{
    return _obversers;
}

int Mapline::GetLineIdx(int frame_id)
{
    if (_obversers.count(frame_id) > 0)
        return _obversers[frame_id];
    return -1;
}

void Mapline::SetObverserEndpointStatus(int frame_id, int status)
{
    _included_endpoints[frame_id] = status;
}

int Mapline::GetObverserEndpointStatus(int frame_id)
{
    auto it = _included_endpoints.find(frame_id);
    if (it == _included_endpoints.end())
        return -1;
    return it->second;
}

const std::map<int, int> &Mapline::GetAllObverserEndpointStatus()
{
    return _included_endpoints;
}
