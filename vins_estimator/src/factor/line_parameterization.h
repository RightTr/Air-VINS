#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

class LineOrthParameterization : public ceres::LocalParameterization
{
  public:
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
    bool ComputeJacobian(const double *x, double *jacobian) const override;
    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 4; }
};
