#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

class lineProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 4>
{
  public:
    explicit lineProjectionFactor(const Eigen::Vector4d &_obs_i);
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
    Eigen::Vector4d obs_i;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
