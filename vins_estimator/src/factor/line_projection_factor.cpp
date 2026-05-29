#include "line_projection_factor.h"
#include "../utility/line_geometry.h"
#include "line_parameterization.h"
#include "../utility/utility.h"
#include <cmath>

Eigen::Matrix2d lineProjectionFactor::sqrt_info;
double lineProjectionFactor::sum_t;

lineProjectionFactor::lineProjectionFactor(const Eigen::Vector4d &_pts_i) : obs_i(_pts_i)
{
}

bool lineProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector4d line_orth(parameters[2][0], parameters[2][1], parameters[2][2], parameters[2][3]);
    Vector6d line_w = orth_to_plk(line_orth);

    Eigen::Matrix3d Rwb(Qi);
    Eigen::Vector3d twb(Pi);
    Vector6d line_b = plk_from_pose(line_w, Rwb, twb);

    Eigen::Matrix3d Rbc(qic);
    Eigen::Vector3d tbc(tic);
    Vector6d line_c = plk_from_pose(line_b, Rbc, tbc);

    Eigen::Vector3d nc = line_c.head(3);
    double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
    double l_sqrtnorm = std::sqrt(l_norm + 1e-12);
    double l_trinorm = l_norm * l_sqrtnorm + 1e-12;

    double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
    double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual(0) = e1 / l_sqrtnorm;
    residual(1) = e2 / l_sqrtnorm;
    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix<double, 2, 3> jaco_e_l(2, 3);
        jaco_e_l << (obs_i(0) / l_sqrtnorm - nc(0) * e1 / l_trinorm),
            (obs_i(1) / l_sqrtnorm - nc(1) * e1 / l_trinorm), 1.0 / l_sqrtnorm,
            (obs_i(2) / l_sqrtnorm - nc(0) * e2 / l_trinorm),
            (obs_i(3) / l_sqrtnorm - nc(1) * e2 / l_trinorm), 1.0 / l_sqrtnorm;

        jaco_e_l = sqrt_info * jaco_e_l;

        Eigen::Matrix<double, 3, 6> jaco_l_Lc;
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 2, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Matrix6d invTbc;
            invTbc << Rbc.transpose(), -Rbc.transpose() * skew_symmetric(tbc),
                Eigen::Matrix3d::Zero(), Rbc.transpose();

            Vector3d nw = line_w.head(3);
            Vector3d dw = line_w.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_pose;
            jaco_Lc_pose.setZero();
            jaco_Lc_pose.block(0, 0, 3, 3) = Rwb.transpose() * skew_symmetric(dw);
            jaco_Lc_pose.block(0, 3, 3, 3) = skew_symmetric(Rwb.transpose() * (nw + skew_symmetric(dw) * twb));
            jaco_Lc_pose.block(3, 3, 3, 3) = skew_symmetric(Rwb.transpose() * dw);

            jaco_Lc_pose = invTbc * jaco_Lc_pose;

            jacobian_pose_i.leftCols<6>() = jaco_e_Lc * jaco_Lc_pose;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);

            Vector3d nb = line_b.head(3);
            Vector3d db = line_b.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_ex;
            jaco_Lc_ex.setZero();
            jaco_Lc_ex.block(0, 0, 3, 3) = Rbc.transpose() * skew_symmetric(db);
            jaco_Lc_ex.block(0, 3, 3, 3) = skew_symmetric(Rbc.transpose() * (nb + skew_symmetric(db) * tbc));
            jaco_Lc_ex.block(3, 3, 3, 3) = skew_symmetric(Rbc.transpose() * db);

            jacobian_ex_pose.leftCols<6>() = jaco_e_Lc * jaco_Lc_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }

        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[2]);

            Eigen::Matrix3d Rwc = Rwb * Rbc;
            Eigen::Vector3d twc = Rwb * tbc + twb;
            Matrix6d invTwc;
            invTwc << Rwc.transpose(), -Rwc.transpose() * skew_symmetric(twc),
                Eigen::Matrix3d::Zero(), Rwc.transpose();

            Vector3d nw = line_w.head(3);
            Vector3d vw = line_w.tail(3);
            Vector3d u1 = nw / nw.norm();
            Vector3d u2 = vw / vw.norm();
            Vector3d u3 = u1.cross(u2);
            Vector2d w(nw.norm(), vw.norm());
            w = w / w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lw_orth;
            jaco_Lw_orth.setZero();
            jaco_Lw_orth.block(3, 0, 3, 1) = w[1] * u3;
            jaco_Lw_orth.block(0, 1, 3, 1) = -w[0] * u3;
            jaco_Lw_orth.block(0, 2, 3, 1) = w(0) * u2;
            jaco_Lw_orth.block(3, 2, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(0, 3, 3, 1) = -w(1) * u1;
            jaco_Lw_orth.block(3, 3, 3, 1) = w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc * invTwc * jaco_Lw_orth;
        }
    }

    return true;
}
