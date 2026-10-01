#pragma once

#include <eigen3/Eigen/Core>

struct Camera
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    Eigen::Vector3d liftProjective(const Eigen::Vector2d &pixel) const
    {
        return Eigen::Vector3d((pixel.x() - cx) / fx,
                               (pixel.y() - cy) / fy,
                               1.0);
    }

    void liftProjective(const Eigen::Vector2d &pixel, Eigen::Vector3d &bearing) const
    {
        bearing = liftProjective(pixel);
    }

    Eigen::Vector2d project(const Eigen::Vector3d &point) const
    {
        return Eigen::Vector2d(fx * point.x() / point.z() + cx,
                               fy * point.y() / point.z() + cy);
    }
};
