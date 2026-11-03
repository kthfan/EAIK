#include <iostream>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <cmath>
#include <tuple>

#include "kinematic_remodeling.h"
#include "kinematic_utils.h"
#include "EAIK.h"

namespace EAIK
{
    Robot::Robot(const Eigen::VectorXd& dh_alpha, const Eigen::VectorXd& dh_a, const Eigen::VectorXd& dh_d, const Eigen::Matrix<double, 3, 3> &R6T, const std::vector<std::pair<int, double>>& fixed_axes, bool is_double_precision,
                 double parallel_tol, double intersect_tol, double ls_pos_tol, double ls_rot_tol)
        : parallel_tol(parallel_tol), intersect_tol(intersect_tol), ls_pos_tol(ls_pos_tol), ls_rot_tol(ls_rot_tol)
    {
        const auto&[H, P, R6T_dh] = IKS::dh_to_H_P(dh_alpha, dh_a, dh_d);
        this->R6T = R6T*R6T_dh;
        init(H, P, fixed_axes, is_double_precision);
    }

    Robot::Robot(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P, const Eigen::Matrix<double, 3, 3> &R6T, const std::vector<std::pair<int, double>>& fixed_axes, bool is_double_precision,
                 double parallel_tol, double intersect_tol, double ls_pos_tol, double ls_rot_tol)
        : parallel_tol(parallel_tol), intersect_tol(intersect_tol), ls_pos_tol(ls_pos_tol), ls_rot_tol(ls_rot_tol), R6T(R6T)
    {
        init(H, P, fixed_axes, is_double_precision);
    }

    void Robot::init(const Eigen::MatrixXd &H, const Eigen::MatrixXd &P, const std::vector<std::pair<int, double>>& fixed_axes, bool is_double_precision)
    {
        if(is_double_precision)
        {
            ZERO_THRESHOLD = 1e-13;
            AXIS_INTERSECT_THRESHOLD = 1e-9;
        }

        if(P.cols() != H.cols() + 1)
        {
            throw std::runtime_error("Wrong input dimensions for H and P. Note that #P = #H+1.");
        }

        Eigen::MatrixXd P_remodeled;
        Eigen::MatrixXd H_remodeled;

        // Insert fixed axes (e.g., to cope with >6DOF manipulators) into the kinematic chain
        if(fixed_axes.size()>0)
        {
            this->fixed_axes = fixed_axes;
            std::sort(this->fixed_axes.begin(), this->fixed_axes.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {return a.first < b.first;});

            const auto&[H_part, P_part, R6T_part] = partial_joint_parametrization(H, P, fixed_axes, R6T);
            std::tie(H_idealized, P_idealized) = idealize_kinematics(H_part, P_part, parallel_tol, intersect_tol);
            this->R6T_partial = R6T_part;
        }
        else
        {
            std::tie(H_idealized, P_idealized) = idealize_kinematics(H, P, parallel_tol, intersect_tol);
            this->R6T_partial = R6T;
        }
        P_remodeled = remodel_kinematics(H_idealized, P_idealized, ZERO_THRESHOLD, AXIS_INTERSECT_THRESHOLD);
        H_remodeled = H_idealized;
        solver_dof = H_remodeled.cols();
        switch (H_remodeled.cols())
        {
        case 6:
            bot_kinematics = std::make_unique<IKS::General_6R>(H_remodeled, P_remodeled);
            original_kinematics = std::make_unique<IKS::General_Robot>(H, P);
            break;
        case 5:
            bot_kinematics = std::make_unique<IKS::General_5R>(H_remodeled, P_remodeled);
            original_kinematics = std::make_unique<IKS::General_Robot>(H, P);
            break;
        case 4:
            bot_kinematics = std::make_unique<IKS::General_4R>(H_remodeled, P_remodeled);
            original_kinematics = std::make_unique<IKS::General_Robot>(H, P);  
            break;  
        case 3:
            bot_kinematics = std::make_unique<IKS::General_3R>(H_remodeled, P_remodeled);
            original_kinematics = std::make_unique<IKS::General_Robot>(H, P);  
            break;
        case 2:
            bot_kinematics = std::make_unique<IKS::General_2R>(H_remodeled, P_remodeled);
            original_kinematics = std::make_unique<IKS::General_Robot>(H, P);  
            break;
        case 1:
            bot_kinematics = std::make_unique<IKS::General_1R>(H_remodeled, P_remodeled);
            original_kinematics = std::make_unique<IKS::General_Robot>(H, P);  
            break;
        default:
            throw std::runtime_error("Currently, only 1-6R robots are solvable with EAIK. Consider locking redundant joints.");
            break;
        }
    }


    Eigen::MatrixXd Robot::get_remodeled_H() const
    {
        return bot_kinematics->get_H();
    }

    Eigen::MatrixXd Robot::get_remodeled_P() const
    {
        return bot_kinematics->get_P();
    }

    Eigen::MatrixXd Robot::get_original_H() const
    {
        return original_kinematics->get_H();
    }

    Eigen::MatrixXd Robot::get_original_P() const
    {
        return original_kinematics->get_P();
    }

    Eigen::MatrixXd Robot::get_idealized_H() const
    {
        return H_idealized;
    }

    Eigen::MatrixXd Robot::get_idealized_P() const
    {
        return P_idealized;
    }

    Eigen::Matrix<double, 3, 3> Robot::get_R6T() const
    {
        return R6T;
    }

    void Robot::evaluate_solution(IKS::IK_Solution& solution, const IKS::Homogeneous_T& target) const
    {
        solution.is_LS_vec.resize(solution.Q.size(), true);
        solution.pos_error_vec.resize(solution.Q.size());
        solution.rot_error_vec.resize(solution.Q.size());
        for(unsigned i = 0; i < solution.Q.size(); ++i)
        {
            const IKS::Homogeneous_T F = fwdkin(solution.Q.at(i));
            const double dp = (F.block<3,1>(0,3) - target.block<3,1>(0,3)).norm();
            const Eigen::Matrix3d R = target.block<3,3>(0,0).transpose() * F.block<3,3>(0,0);
            const Eigen::Vector3d w(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
            const double dr = std::atan2(0.5*w.norm(), 0.5*(R.trace()-1.0));  // accurate also for tiny angles
            solution.pos_error_vec.at(i) = dp;
            solution.rot_error_vec.at(i) = dr;

            // Full-pose (6R) solutions: is_LS reflects whether the returned Q actually reaches the target.
            // For <6R robots the full pose is generally not reachable, keep the subproblem flags.
            if(solver_dof == 6)
            {
                solution.is_LS_vec.at(i) = dp > ls_pos_tol || dr > ls_rot_tol;
            }
        }
    }

    std::string Robot::get_kinematic_family() const
    {
        return bot_kinematics->get_kinematic_family();
    }

    IKS::IK_Solution Robot::calculate_IK(const IKS::Homogeneous_T &ee_position_orientation) const
    {
        if (!bot_kinematics->has_known_decomposition())
        {
            throw std::runtime_error(decomposition_unknown_exception_info);
        }

        IKS::Homogeneous_T ee_06_rot = ee_position_orientation;
        ee_06_rot.block<3,3>(0,0) *= R6T_partial.transpose();
        IKS::IK_Solution solution = bot_kinematics->calculate_IK(ee_06_rot);

        // TODO: Find alternative to this costly insert operation
        for(const auto&[joint_index, value] : fixed_axes)
        {
            for(auto& Qs : solution.Q)
            {
                Qs.insert(Qs.begin()+joint_index, value);
            }
        }

        evaluate_solution(solution, ee_position_orientation);
        return solution;
    }    
    
    std::vector<IKS::IK_Solution> Robot::calculate_IK_batched(std::vector<IKS::Homogeneous_T> EE_pose_batch, const unsigned worker_threads) const
    {
        if (!bot_kinematics->has_known_decomposition())
        {
            throw std::runtime_error(decomposition_unknown_exception_info);
        }

        std::vector<IKS::IK_Solution> solutions(EE_pose_batch.size());
        std::vector<std::thread> thread_pool;
        thread_pool.reserve(worker_threads);

        std::mutex pose_mutex;

        auto kernel = 
        [this, &pose_mutex, &solutions, &EE_pose_batch]{
            pose_mutex.lock();
            int solution_index = EE_pose_batch.size()-1;

            while(solution_index >= 0)
            {
                const IKS::Homogeneous_T& pose = EE_pose_batch.back();
                EE_pose_batch.pop_back();  
                pose_mutex.unlock(); 

                solutions.at(solution_index) = calculate_IK(pose);

                pose_mutex.lock();
                solution_index = EE_pose_batch.size()-1;
            }
            pose_mutex.unlock(); 
        };

        for(unsigned i = 0; i < worker_threads; i++)
        {
            thread_pool.push_back(std::thread(kernel));
        }

        for(unsigned i = 0; i < worker_threads; i++)
        {
            thread_pool.at(i).join();
        }

        return solutions;
    }

    // Eigen matrices as solutions for the pybindings
    IKS::IK_Eigen_Solution Robot::calculate_Eigen_IK(const IKS::Homogeneous_T &ee_position_orientation) const
    {
        IKS::IK_Solution vector_solution = calculate_IK(ee_position_orientation);
        IKS::IK_Eigen_Solution eigen_solution;

        if(vector_solution.Q.size() > 0)
        {
            eigen_solution.is_LS_vec.resize(vector_solution.is_LS_vec.size());
            eigen_solution.pos_error_vec = Eigen::VectorXd::Map(vector_solution.pos_error_vec.data(), vector_solution.pos_error_vec.size());
            eigen_solution.rot_error_vec = Eigen::VectorXd::Map(vector_solution.rot_error_vec.data(), vector_solution.rot_error_vec.size());

            // TODO: Find a way to avoid slow loop in future releases
            for(unsigned i = 0; i < vector_solution.is_LS_vec.size(); ++i)
            {
                eigen_solution.is_LS_vec(i) = vector_solution.is_LS_vec.at(i);
            }

            eigen_solution.Q = Eigen::MatrixXd(vector_solution.Q.size(), vector_solution.Q.at(0).size());
            for (unsigned i = 0; i < vector_solution.Q.size(); i++)
            {
                eigen_solution.Q.row(i) = Eigen::VectorXd::Map(vector_solution.Q.at(i).data(),vector_solution.Q.at(0).size());
            }
        }
        return eigen_solution;                
    }

    std::vector<IKS::IK_Eigen_Solution> Robot::calculate_Eigen_IK_batched(std::vector<IKS::Homogeneous_T> EE_pose_batch, const unsigned worker_threads) const
    {
        std::vector<IKS::IK_Eigen_Solution> solutions(EE_pose_batch.size());
        std::vector<std::thread> thread_pool;
        thread_pool.reserve(worker_threads);

        std::mutex pose_mutex;

        auto kernel = 
        [this, &pose_mutex, &solutions, &EE_pose_batch]{
            pose_mutex.lock();
            int solution_index = EE_pose_batch.size()-1;

            while(solution_index >= 0)
            {
                const IKS::Homogeneous_T& pose = EE_pose_batch.back();
                EE_pose_batch.pop_back();  
                pose_mutex.unlock(); 

                solutions.at(solution_index) = calculate_Eigen_IK(pose);

                pose_mutex.lock();
                solution_index = EE_pose_batch.size()-1;
            }
            pose_mutex.unlock(); 
        };

        for(unsigned i = 0; i < worker_threads; i++)
        {
            thread_pool.push_back(std::thread(kernel));
        }

        for(unsigned i = 0; i < worker_threads; i++)
        {
            thread_pool.at(i).join();
        }

        return solutions;
    }    

    IKS::Homogeneous_T Robot::fwdkin(const std::vector<double> &Q) const
    {
        IKS::Homogeneous_T sol_r06 = original_kinematics->fwdkin(Q);
        sol_r06.block<3,3>(0,0) *= R6T;
        return sol_r06;
    }

    // Function for use with pybindings and the corresponding numpy arrays
    IKS::Homogeneous_T Robot::fwdkin_Eigen(const Eigen::VectorXd &Q) const
    {
        std::vector<double> Q_stdVector(Q.data(), Q.data() + Q.size());
        IKS::Homogeneous_T sol_r06 = original_kinematics->fwdkin(Q_stdVector);
        sol_r06.block<3,3>(0,0) *= R6T;
        return sol_r06;
    }
    
    bool Robot::is_spherical() const
    {
        return this->bot_kinematics->is_spherical();
    }

    bool Robot::has_known_decomposition() const
    {
        return this->bot_kinematics->has_known_decomposition();
    }
} // namespace EAIK
