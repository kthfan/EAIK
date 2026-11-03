#ifndef EAIK_IMPL
#define EAIK_IMPL

#include <Eigen/Dense>
#include <memory>
#include <tuple>

#include "IKS.h"

namespace EAIK
{
    class Robot
    {
    public:
        // parallel_tol [rad], intersect_tol [m]: idealization tolerances (see idealize_kinematics); <= 0 disables.
        //   IK is solved on the idealized model; fwdkin and the reported errors use the original model.
        // ls_pos_tol [m], ls_rot_tol [rad]: a 6R solution is flagged is_LS if fwdkin(Q) misses the target
        //   pose by more than these.
        Robot(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P, const Eigen::Matrix<double, 3, 3> &R6T=Eigen::Matrix<double, 3, 3>::Identity(), const std::vector<std::pair<int, double>>& fixed_axes={}, bool is_double_precision=true,
              double parallel_tol=0.0, double intersect_tol=0.0, double ls_pos_tol=1e-8, double ls_rot_tol=1e-8);
        Robot(const Eigen::VectorXd& dh_alpha, const Eigen::VectorXd& dh_a, const Eigen::VectorXd& dh_d, const Eigen::Matrix<double, 3, 3> &R6T=Eigen::Matrix<double, 3, 3>::Identity(), const std::vector<std::pair<int, double>>& fixed_axes={}, bool is_double_precision=true,
              double parallel_tol=0.0, double intersect_tol=0.0, double ls_pos_tol=1e-8, double ls_rot_tol=1e-8);

        IKS::IK_Solution calculate_IK(const IKS::Homogeneous_T &ee_position_orientation) const;
        std::vector<IKS::IK_Solution> calculate_IK_batched(std::vector<IKS::Homogeneous_T> EE_pose_batch, const unsigned worker_threads) const;

        IKS::IK_Eigen_Solution calculate_Eigen_IK(const IKS::Homogeneous_T &ee_position_orientation) const;
        std::vector<IKS::IK_Eigen_Solution> calculate_Eigen_IK_batched(std::vector<IKS::Homogeneous_T> EE_pose_batch, const unsigned worker_threads) const;

        IKS::Homogeneous_T fwdkin(const std::vector<double> &Q) const;
        IKS::Homogeneous_T fwdkin_Eigen(const Eigen::VectorXd &Q) const;

        bool is_spherical() const;
        bool has_known_decomposition() const;

        Eigen::MatrixXd get_remodeled_H() const;
        Eigen::MatrixXd get_remodeled_P() const;

        Eigen::MatrixXd get_original_H() const;
        Eigen::MatrixXd get_original_P() const;

        // Model the analytical solver works on: after fixed-axis reduction and idealization, before remodeling
        Eigen::MatrixXd get_idealized_H() const;
        Eigen::MatrixXd get_idealized_P() const;
        Eigen::Matrix<double, 3, 3> get_R6T() const;

        std::string get_kinematic_family() const;
        
    private:
        // Init function to allow nice constructor overloading
        void init(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P, const std::vector<std::pair<int, double>>& fixed_axes={}, bool is_double_precision=true);
        std::unique_ptr<IKS::General_Robot> bot_kinematics;
        std::unique_ptr<IKS::General_Robot> original_kinematics; // Forward Kinematics calculated by original kinematics

        bool spherical_wrist{false};

        void evaluate_solution(IKS::IK_Solution& solution, const IKS::Homogeneous_T& target) const;
        double parallel_tol{0.0};
        double intersect_tol{0.0};
        double ls_pos_tol{1e-8};
        double ls_rot_tol{1e-8};
        unsigned solver_dof{0};
        Eigen::MatrixXd H_idealized;
        Eigen::MatrixXd P_idealized;

        // single precision default thresholds
        double ZERO_THRESHOLD = 1e-7; 
        double AXIS_INTERSECT_THRESHOLD = 1e-6;
        
        Eigen::Matrix<double, 3, 3> R6T;
        Eigen::Matrix<double, 3, 3> R6T_partial;
        std::vector<std::pair<int, double>> fixed_axes; // Locked axes sorted by rising indices (!)

        const std::string decomposition_unknown_exception_info{"No Subproblem-Decomposition for the given manipulator is currently known! Analytical IK computation is not possible."};

    };
} // namespace EAIK
#endif