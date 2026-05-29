#include "line_geometry.h"
#include <cmath>

Vector4d line_to_orth(Vector6d line)
{
    Vector4d orth;
    Vector3d p = line.head(3);
    Vector3d v = line.tail(3);
    Vector3d n = p.cross(v);

    Vector3d u1 = n / n.norm();
    Vector3d u2 = v / v.norm();
    Vector3d u3 = u1.cross(u2);

    orth[0] = atan2(u2(2), u3(2));
    orth[1] = asin(-u1(2));
    orth[2] = atan2(u1(1), u1(0));
    Vector2d w(n.norm(), v.norm());
    w = w / w.norm();
    orth[3] = asin(w(1));
    return orth;
}

Vector6d orth_to_line(Vector4d orth)
{
    Vector6d line;
    Vector3d theta = orth.head(3);
    double phi = orth[3];
    double s1 = sin(theta[0]), c1 = cos(theta[0]);
    double s2 = sin(theta[1]), c2 = cos(theta[1]);
    double s3 = sin(theta[2]), c3 = cos(theta[2]);
    Matrix3d R;
    R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
         c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
         -s2,     s1 * c2,                c1 * c2;
    double w1 = cos(phi);
    double w2 = sin(phi);
    double d = w1 / w2;
    line.head(3) = -R.col(2) * d;
    line.tail(3) = R.col(1);
    return line;
}

Vector4d plk_to_orth(Vector6d plk)
{
    Vector4d orth;
    Vector3d n = plk.head(3);
    Vector3d v = plk.tail(3);
    Vector3d u1 = n / n.norm();
    Vector3d u2 = v / v.norm();
    Vector3d u3 = u1.cross(u2);
    orth[0] = atan2(u2(2), u3(2));
    orth[1] = asin(-u1(2));
    orth[2] = atan2(u1(1), u1(0));
    Vector2d w(n.norm(), v.norm());
    w = w / w.norm();
    orth[3] = asin(w(1));
    return orth;
}

Vector6d orth_to_plk(Vector4d orth)
{
    Vector6d plk;
    Vector3d theta = orth.head(3);
    double phi = orth[3];
    double s1 = sin(theta[0]), c1 = cos(theta[0]);
    double s2 = sin(theta[1]), c2 = cos(theta[1]);
    double s3 = sin(theta[2]), c3 = cos(theta[2]);
    Matrix3d R;
    R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3,
         c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3,
         -s2,     s1 * c2,                c1 * c2;
    double w1 = cos(phi);
    double w2 = sin(phi);
    Vector3d u1 = R.col(0);
    Vector3d u2 = R.col(1);
    plk.head(3) = w1 * u1;
    plk.tail(3) = w2 * u2;
    return plk;
}

Matrix3d skewSymmetric(Vector3d v)
{
    return skew_symmetric(v);
}

Vector4d piFromPPP(Vector3d x1, Vector3d x2, Vector3d x3)
{
    return pi_from_ppp(x1, x2, x3);
}

Vector6d pipiPlk(Vector4d pi1, Vector4d pi2)
{
    return pipi_plk(pi1, pi2);
}

Vector4d pi_from_ppp(Vector3d x1, Vector3d x2, Vector3d x3)
{
    Vector4d pi;
    pi << (x1 - x3).cross(x2 - x3), -x3.dot(x1.cross(x2));
    return pi;
}

Vector6d pipi_plk(Vector4d pi1, Vector4d pi2)
{
    Vector6d plk;
    Matrix4d dp = pi1 * pi2.transpose() - pi2 * pi1.transpose();
    plk << dp(0, 3), dp(1, 3), dp(2, 3), -dp(1, 2), dp(0, 2), -dp(0, 1);
    return plk;
}

Vector3d plucker_origin(Vector3d n, Vector3d v)
{
    return v.cross(n) / v.dot(v);
}

Matrix3d skew_symmetric(Vector3d v)
{
    Matrix3d S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}

Vector3d point_to_pose(Eigen::Matrix3d Rcw, Eigen::Vector3d tcw, Vector3d pt_w)
{
    return Rcw * pt_w + tcw;
}

Vector3d poit_from_pose(Eigen::Matrix3d Rcw, Eigen::Vector3d tcw, Vector3d pt_c)
{
    Eigen::Matrix3d Rwc = Rcw.transpose();
    Vector3d twc = -Rwc * tcw;
    return point_to_pose(Rwc, twc, pt_c);
}

Vector6d line_to_pose(Vector6d line_w, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw)
{
    Vector6d line_c;
    Vector3d cp_w = line_w.head(3);
    Vector3d dv_w = line_w.tail(3);
    Vector3d cp_c = point_to_pose(Rcw, tcw, cp_w);
    Vector3d dv_c = Rcw * dv_w;
    line_c.head(3) = cp_c;
    line_c.tail(3) = dv_c;
    return line_c;
}

Vector6d line_from_pose(Vector6d line_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw)
{
    Eigen::Matrix3d Rwc = Rcw.transpose();
    Vector3d twc = -Rwc * tcw;
    return line_to_pose(line_c, Rwc, twc);
}

Vector6d plk_to_pose(Vector6d plk_w, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw)
{
    Vector3d nw = plk_w.head(3);
    Vector3d vw = plk_w.tail(3);
    Vector3d nc = Rcw * nw + skew_symmetric(tcw) * Rcw * vw;
    Vector3d vc = Rcw * vw;
    Vector6d plk_c;
    plk_c.head(3) = nc;
    plk_c.tail(3) = vc;
    return plk_c;
}

Vector6d plk_from_pose(Vector6d plk_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw)
{
    Eigen::Matrix3d Rwc = Rcw.transpose();
    Vector3d twc = -Rwc * tcw;
    return plk_to_pose(plk_c, Rwc, twc);
}

Vector6d plkFromPose(Vector6d plk_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw)
{
    return plk_to_pose(plk_c, Rcw, tcw);
}

Vector6d plkToWorld(Vector6d plk_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw)
{
    return plk_from_pose(plk_c, Rcw, tcw);
}

Eigen::Matrix<double, 3, 4> worldToCameraPose(const Eigen::Vector3d &P,
                                              const Eigen::Matrix3d &R,
                                              const Eigen::Vector3d &tic,
                                              const Eigen::Matrix3d &ric)
{
    Eigen::Matrix<double, 3, 4> pose;
    Eigen::Vector3d t = P + R * tic;
    Eigen::Matrix3d q = R * ric;
    pose.leftCols<3>() = q.transpose();
    pose.rightCols<1>() = -q.transpose() * t;
    return pose;
}

bool normalizePluckerLine(Vector6d &line)
{
    Eigen::Vector3d moment = line.head<3>();
    Eigen::Vector3d direction = line.tail<3>();
    const double direction_norm = direction.norm();
    if (!std::isfinite(direction_norm) || direction_norm < 1e-9)
        return false;
    direction /= direction_norm;
    moment /= direction_norm;
    moment -= direction * direction.dot(moment);
    if (!moment.allFinite() || !direction.allFinite())
        return false;
    line.head<3>() = moment;
    line.tail<3>() = direction;
    return true;
}

bool buildLineFromEndpoints(const Eigen::Vector3d &start_point,
                            const Eigen::Vector3d &end_point,
                            Vector6d &line)
{
    const Eigen::Vector3d d = end_point - start_point;
    if (d.norm() < 1e-9)
        return false;
    line.head<3>() = start_point.cross(d);
    line.tail<3>() = d;
    return normalizePluckerLine(line);
}

bool buildLineFromPlucker(const Vector6d &line_plucker, Vector6d &line)
{
    if (!line_plucker.allFinite())
        return false;
    line = line_plucker;
    return normalizePluckerLine(line);
}

bool triangulateStereoEndpointFromPoses(const Eigen::Matrix<double, 3, 4> &pose_left,
                                        const Eigen::Matrix<double, 3, 4> &pose_right,
                                        const Eigen::Vector2d &point_left,
                                        const Eigen::Vector2d &point_right,
                                        Eigen::Vector3d &point_world)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point_left[0] * pose_left.row(2) - pose_left.row(0);
    design_matrix.row(1) = point_left[1] * pose_left.row(2) - pose_left.row(1);
    design_matrix.row(2) = point_right[0] * pose_right.row(2) - pose_right.row(0);
    design_matrix.row(3) = point_right[1] * pose_right.row(2) - pose_right.row(1);
    const Eigen::Vector4d triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    if (std::abs(triangulated_point(3)) < 1e-12)
        return false;
    point_world = Eigen::Vector3d(triangulated_point(0) / triangulated_point(3),
                                  triangulated_point(1) / triangulated_point(3),
                                  triangulated_point(2) / triangulated_point(3));
    return point_world.allFinite();
}
