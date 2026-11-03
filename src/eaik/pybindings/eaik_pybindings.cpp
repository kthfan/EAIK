#include "IKS.h"
#include "EAIK.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // Convert numpy <-> Eigen
#include <pybind11/stl.h>   // Convert python list <-> std::list

namespace py = pybind11;


PYBIND11_MODULE(EAIK, m)
{
    m.doc() = R"pbdoc(
    Pybind11 of EAIK
    )pbdoc";

    py::class_<IKS::IK_Eigen_Solution>(m, "IKSolution")
        .def(py::init<>())
        .def_readwrite("Q", &IKS::IK_Eigen_Solution::Q)
        .def_readwrite("is_LS", &IKS::IK_Eigen_Solution::is_LS_vec)
        .def_readwrite("pos_error", &IKS::IK_Eigen_Solution::pos_error_vec, "Position error of fwdkin(Q) w.r.t. the target pose on the original model [m]")
        .def_readwrite("rot_error", &IKS::IK_Eigen_Solution::rot_error_vec, "Rotation error angle of fwdkin(Q) w.r.t. the target pose on the original model [rad]")
        .def("num_solutions", &IKS::IK_Eigen_Solution::num_solutions, R"pbdoc(
            Returns number of total IK solutions (including Least-Squares)
            :return:   int
        )pbdoc")
        .def("__bool__", [](const IKS::IK_Eigen_Solution &self) {
        return self.num_solutions() > 0;
        });

    py::class_<EAIK::Robot>(m, "Robot")
        .def(py::init<const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::Matrix<double, 3, 3> &, const std::vector<std::pair<int, double>>&, bool, double, double, double, double>(), R"pbdoc(
            The EAIK Robot class.

            :param H:  Unit vectors defining the joint axes
            :param P:  Linear Joint offsets
            :param R6T:  Endeffector orientation w.r.t. joint 6
            :param fixed_axes:  List of tuples defining fixed joints (zero-indexed) (i, q_i+1)    
            :param use_double_precision:  Use double precision (standard)
            :param parallel_tol:  Idealization: consecutive axes within this angle [rad] are made exactly parallel (0 = off)
            :param intersect_tol:  Idealization: consecutive axes within this distance [m] are made to intersect exactly (0 = off)
            :param ls_pos_tol:  6R solutions missing the target position by more than this [m] are flagged is_LS
            :param ls_rot_tol:  6R solutions missing the target rotation by more than this [rad] are flagged is_LS
        )pbdoc",
        py::arg("H"), py::arg("P"), py::arg("R6T"), py::arg("fixed_axes"), py::arg("use_double_precision"),
        py::arg("parallel_tol") = 0.0, py::arg("intersect_tol") = 0.0, py::arg("ls_pos_tol") = 1e-8, py::arg("ls_rot_tol") = 1e-8)
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::Matrix<double, 3, 3> &, const std::vector<std::pair<int, double>>&, bool, double, double, double, double>(), R"pbdoc(
            The EAIK Robot class.

            :param dh_alpha:  DH-Parameters: alpha
            :param dh_a:  DH-Parameters: a
            :param dh_d:  DH-Parameters: d
            :param R6T:  Endeffector orientation w.r.t. joint 6
            :param fixed_axes:  List of tuples defining fixed joints (zero-indexed) (i, q_i+1)    
            :param use_double_precision:  Use double precision (standard)
            :param parallel_tol:  Idealization: consecutive axes within this angle [rad] are made exactly parallel (0 = off)
            :param intersect_tol:  Idealization: consecutive axes within this distance [m] are made to intersect exactly (0 = off)
            :param ls_pos_tol:  6R solutions missing the target position by more than this [m] are flagged is_LS
            :param ls_rot_tol:  6R solutions missing the target rotation by more than this [rad] are flagged is_LS
        )pbdoc",
        py::arg("dh_alpha"), py::arg("dh_a"), py::arg("dh_d"), py::arg("R6T"), py::arg("fixed_axes"), py::arg("use_double_precision"),
        py::arg("parallel_tol") = 0.0, py::arg("intersect_tol") = 0.0, py::arg("ls_pos_tol") = 1e-8, py::arg("ls_rot_tol") = 1e-8)
        .def("calculate_IK", &EAIK::Robot::calculate_Eigen_IK, R"pbdoc(
            Run inverse kinematics.

            :param pose:  4x4 Homogeneous transformation matrix
            :return:      IK-Solution class
        )pbdoc",
             py::arg("pose"))
        .def("calculate_IK_batched", &EAIK::Robot::calculate_Eigen_IK_batched, R"pbdoc(
            Run inverse kinematics for a batch of EE poses.

            :param pose:  Batch of 4x4 Homogeneous transformation matrix
            :param num_worker_threads: Number of total worker threads to assign
            :return:      Batch of IK-Solution classes
        )pbdoc",
             py::arg("pose_batch"), py::arg("num_worker_threads"))
        .def("fwdkin", &EAIK::Robot::fwdkin_Eigen, R"pbdoc(
            Run forward kinematics.

            :param Q:  Array of six joint angles
            :return:   4x4 Homogeneous transformation matrix
        )pbdoc",
             py::arg("Q"))
        .def("is_spherical", &EAIK::Robot::is_spherical, R"pbdoc(
            Returns if robot has spherical wrist.
            :return:   bool
        )pbdoc")
        .def("has_known_decomposition", &EAIK::Robot::has_known_decomposition, R"pbdoc(
            Returns if robot has a known SP decomposition
            :return:   bool
        )pbdoc")
        .def("get_remodeled_H", &EAIK::Robot::get_remodeled_H)
        .def("get_remodeled_P", &EAIK::Robot::get_remodeled_P)
        .def("get_original_H", &EAIK::Robot::get_original_H)
        .def("get_original_P", &EAIK::Robot::get_original_P)
        .def("get_idealized_H", &EAIK::Robot::get_idealized_H)
        .def("get_idealized_P", &EAIK::Robot::get_idealized_P)
        .def("get_R6T", &EAIK::Robot::get_R6T)
        .def("get_kinematic_family", &EAIK::Robot::get_kinematic_family);
}
