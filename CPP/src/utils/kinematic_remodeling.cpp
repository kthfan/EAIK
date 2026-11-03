#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <utility>

#include "kinematic_remodeling.h"


namespace EAIK
{
    bool do_axes_intersect(const Eigen::Vector3d& h1, const Eigen::Vector3d& h2, const Eigen::Vector3d& p12, const double ZERO_THRESHOLD, const double AXIS_INTERSECT_THRESHOLD)
    {
        return fabs((h1.cross(h2)).transpose()*p12) < AXIS_INTERSECT_THRESHOLD && (h1.cross(h2)).norm() > ZERO_THRESHOLD;
    }

    bool is_point_on_Axis(const Eigen::Vector3d& h, const Eigen::Vector3d& p0h, const Eigen::Vector3d& p, const double AXIS_INTERSECT_THRESHOLD)
    {
        const Eigen::Vector3d p_h = p - p0h;
        const Eigen::Vector3d proj_p = p_h.dot(h)*h;

        return (proj_p-p_h).norm() < AXIS_INTERSECT_THRESHOLD;
    }

    Eigen::Vector3d calc_intersection(const Eigen::Vector3d& hj, const Eigen::Vector3d& hk, const Eigen::Vector3d& p0j, const Eigen::Vector3d& pkj, const double ZERO_THRESHOLD)
    {
        const Eigen::Vector3d hj_cross_hk = hj.cross(hk);

        if(hj_cross_hk.squaredNorm() < ZERO_THRESHOLD)
        {
            throw std::runtime_error("Intersection point can't be calculated for two parallel axes!");
        }

        Eigen::Matrix3d A;
        A << pkj, hk, hj_cross_hk;
        double lambda_1 = A.determinant()/hj_cross_hk.squaredNorm();
        
        return p0j + hj*lambda_1;
        /*
        To get an error-measure calculate the second lambda and calc the eucledian difference between the two intersection points
        A.col(1) = hj;
        double lambda_2 = A.determinant()/hj_cross_hk.squaredNorm();
        */
    }

    Eigen::MatrixXd remodel_kinematics(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P, const double ZERO_THRESHOLD, const double AXIS_INTERSECT_THRESHOLD)
    {
        Eigen::MatrixXd P_new = P;
        Eigen::Vector3d p0_i_plus1(0,0,0);

        if(P.cols() == 7)
        {
            // Remodel Robot "Base"
            for(unsigned i = 0; i < H.cols()-4; i++)
            {
                p0_i_plus1 += P_new.col(i);
                if (do_axes_intersect(H.col(i), H.col(i+1), P.col(i+1), ZERO_THRESHOLD, AXIS_INTERSECT_THRESHOLD))
                {
                    Eigen::Vector3d intersection = calc_intersection(H.col(i), H.col(i+1), p0_i_plus1, P.col(i+1), ZERO_THRESHOLD);
                    P_new.col(i+1) = Eigen::Vector3d(0,0,0);
                
                    unsigned j = i+2;
                    Eigen::Vector3d p0j = p0_i_plus1+P.col(j-1)+P.col(j);

                    while(j < H.cols()-3 && is_point_on_Axis(H.col(j), p0j, intersection, AXIS_INTERSECT_THRESHOLD))
                    {
                        P_new.col(j) = Eigen::Vector3d(0,0,0);
                        j++;
                        p0j += P.col(j);
                    }
                    P_new.col(i) = intersection - (p0_i_plus1-P_new.col(i));
                    P_new.col(j) = p0j - intersection;
                    i = j-1;
                    p0_i_plus1 = p0j - P.col(j);
                }
            }

            p0_i_plus1+=P.col(H.cols()-4);

            // Remodel Robot "Wrist"
            for(unsigned i = H.cols()-3; i < H.cols()-1; i++)
            {
                p0_i_plus1 += P_new.col(i);
                if (do_axes_intersect(H.col(i), H.col(i+1), P.col(i+1), ZERO_THRESHOLD, AXIS_INTERSECT_THRESHOLD))
                {
                    Eigen::Vector3d intersection = calc_intersection(H.col(i), H.col(i+1), p0_i_plus1, P_new.col(i+1), AXIS_INTERSECT_THRESHOLD);
                
                    unsigned j = i+2;
                    Eigen::Vector3d p0_j_plus1 = p0_i_plus1+P_new.col(i+1)+P_new.col(j);

                    P_new.col(i+1) = Eigen::Vector3d(0,0,0);

                    while(j < H.cols() && is_point_on_Axis(H.col(j), p0_j_plus1, intersection, AXIS_INTERSECT_THRESHOLD))
                    {
                        P_new.col(j) = Eigen::Vector3d(0,0,0);
                        j++;
                        p0_j_plus1 += P.col(j);
                    }
                    P_new.col(i) = intersection - (p0_i_plus1-P_new.col(i));
                    P_new.col(j) = p0_j_plus1 - intersection;
                    i = j-2;
                    p0_i_plus1 = p0_j_plus1 - P.col(j)-P.col(j-1);
                }
            }
        }
        else
        {
            for(unsigned i = 0; i < H.cols()-1; i++)
            {
                p0_i_plus1 += P_new.col(i);
                if (do_axes_intersect(H.col(i), H.col(i+1), P.col(i+1), ZERO_THRESHOLD, AXIS_INTERSECT_THRESHOLD))
                {
                    Eigen::Vector3d intersection = calc_intersection(H.col(i), H.col(i+1), p0_i_plus1, P_new.col(i+1), AXIS_INTERSECT_THRESHOLD);
                
                    unsigned j = i+2;
                    Eigen::Vector3d p0_j_plus1 = p0_i_plus1+P_new.col(i+1)+P_new.col(j);

                    P_new.col(i+1) = Eigen::Vector3d(0,0,0);

                    while(j < H.cols() && is_point_on_Axis(H.col(j), p0_j_plus1, intersection, AXIS_INTERSECT_THRESHOLD))
                    {
                        P_new.col(j) = Eigen::Vector3d(0,0,0);
                        j++;
                        p0_j_plus1 += P.col(j);
                    }
                    P_new.col(i) = intersection - (p0_i_plus1-P_new.col(i));
                    P_new.col(j) = p0_j_plus1 - intersection;
                    i = j-2;
                    p0_i_plus1 = p0_j_plus1 - P.col(j)-P.col(j-1);
                }
            }
        }

        return P_new;
    }

    namespace
    {
        // Point on line (oj, hj) closest to line (ok, hk); lines must not be parallel
        Eigen::Vector3d closest_point_on_first(const Eigen::Vector3d& oj, const Eigen::Vector3d& hj, const Eigen::Vector3d& ok, const Eigen::Vector3d& hk)
        {
            const Eigen::Vector3d w = oj - ok;
            const double a = hj.dot(hj), b = hj.dot(hk), c = hk.dot(hk), d = hj.dot(w), e = hk.dot(w);
            const double s = (b*e - c*d) / (a*c - b*b);
            return oj + s*hj;
        }

        // Axis k passes through the sum of P.col(0..k)
        Eigen::Vector3d axis_point(const Eigen::MatrixXd& P, unsigned k)
        {
            Eigen::Vector3d o = Eigen::Vector3d::Zero();
            for(unsigned m = 0; m <= k; ++m) o += P.col(m);
            return o;
        }

        // Below this |h_i x h_j| two axes count as parallel for the intersection steps
        constexpr double PARALLEL_FOR_INTERSECTION = 1e-6;
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> idealize_kinematics(const Eigen::MatrixXd &H_in, const Eigen::MatrixXd &P_in, const double parallel_tol, const double intersect_tol)
    {
        Eigen::MatrixXd H = H_in;
        Eigen::MatrixXd P = P_in;
        const unsigned n = H.cols();
        for(unsigned k = 0; k < n; ++k) H.col(k).normalize();

        // 1) Groups of consecutive (anti-)parallel axes share one exact direction
        if(parallel_tol > 0)
        {
            const double sin_tol = std::sin(std::min(parallel_tol, 1.5707963267948966));
            unsigned i = 0;
            while(i < n)
            {
                unsigned j = i;
                while(j + 1 < n && Eigen::Vector3d(H.col(i)).cross(Eigen::Vector3d(H.col(j+1))).norm() < sin_tol) ++j;
                if(j > i)
                {
                    Eigen::Vector3d m = Eigen::Vector3d::Zero();
                    for(unsigned k = i; k <= j; ++k) m += (H.col(k).dot(H.col(i)) >= 0 ? 1.0 : -1.0) * H.col(k);
                    m.normalize();
                    for(unsigned k = i; k <= j; ++k)
                    {
                        const double s = H.col(k).dot(H.col(i)) >= 0 ? 1.0 : -1.0;
                        H.col(k) = s*m;
                    }
                }
                i = j + 1;
            }
        }

        if(intersect_tol > 0 && n >= 2)
        {
            // 2) Consecutive non-parallel axes with a small common-normal distance intersect exactly
            //    (shift the offset between them along the common normal; downstream axes move along)
            std::vector<bool> intersects(n, false);  // intersects[i]: axis i meets axis i+1
            for(unsigned i = 0; i + 1 < n; ++i)
            {
                const Eigen::Vector3d c = Eigen::Vector3d(H.col(i)).cross(Eigen::Vector3d(H.col(i+1)));
                if(c.norm() < PARALLEL_FOR_INTERSECTION) continue;
                const Eigen::Vector3d nc = c.normalized();
                const double d = nc.dot(P.col(i+1));
                if(std::fabs(d) < intersect_tol)
                {
                    P.col(i+1) -= d*nc;
                    intersects.at(i) = true;
                }
            }

            // 3) Axes i, i+1, i+2 pairwise intersecting at two nearby points on axis i+1:
            //    move axis i+2 along axis i+1 so that all three meet in one point
            for(unsigned i = 0; i + 2 < n; ++i)
            {
                if(!intersects.at(i) || !intersects.at(i+1)) continue;
                const Eigen::Vector3d A = closest_point_on_first(axis_point(P, i+1), H.col(i+1), axis_point(P, i), H.col(i));
                const Eigen::Vector3d B = closest_point_on_first(axis_point(P, i+1), H.col(i+1), axis_point(P, i+2), H.col(i+2));
                const Eigen::Vector3d delta = B - A;  // parallel to H.col(i+1)
                if(delta.norm() < intersect_tol)
                {
                    P.col(i+2) -= delta;
                }
            }
        }
        return {H, P};
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::Matrix<double, 3, 3>> partial_joint_parametrization(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P, std::vector<std::pair<int, double>> fixed_axes, const Eigen::Matrix<double, 3, 3>& R6T)
    {
        // Sort fixed axes parametrization to start with last axis first
        std::sort(fixed_axes.begin(), fixed_axes.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {return a.first > b.first;});
        Eigen::MatrixXd H_new = H;
        Eigen::MatrixXd P_new = P;
        Eigen::Matrix<double, 3, 3> r6t_new = R6T;

        for(const auto&[axis_index, q] : fixed_axes)
        {
            const Eigen::Matrix3d r_fixed = Eigen::AngleAxisd(q, H_new.col(axis_index).normalized()).toRotationMatrix();
            P_new.col(axis_index) += r_fixed*P_new.col(axis_index+1);
            
            P_new.block(0, axis_index+1, 3, P_new.cols()-axis_index-2) = r_fixed*P_new.block(0, axis_index+2, 3, P_new.cols()-axis_index-2).eval();

            // Only adjust r6t if last axis is locked
            if(axis_index+1 < H_new.cols())
            {
                H_new.col(axis_index) = r_fixed*H_new.col(axis_index+1);
                H_new.block(0, axis_index+1, 3, H_new.cols()-axis_index-2) = r_fixed*H_new.block(0, axis_index+2, 3, H_new.cols()-axis_index-2).eval();
            }

            H_new = H_new.block(0, 0, 3, H_new.cols()-1).eval();
            P_new = P_new.block(0, 0, 3, P_new.cols()-1).eval();
            r6t_new = r_fixed*r6t_new;
        }

        return {H_new, P_new,r6t_new};
    }
} // namespace EAIK
