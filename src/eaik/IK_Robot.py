from abc import ABC

import numpy as np

import eaik.pybindings.EAIK as EAIK


class IKRobot(ABC):
    """An interface for the python-side robot implementation of this toolbox"""

    _robot: EAIK.Robot

    @staticmethod
    def urdf_to_sp_conv(axis_trafo, axis, parent_p):
        """
        Convert urchin axis to axis-translation convention for subproblems

        :param axis_trafo: 4x4 homogeneous transformation of a joint w.r.t. a world frame
        :param axis: Joint axis within axis_trafo (e.g., z-axis)
        :param parent_p: Linear global offset of last joint
        :return: (Axis vector, translation)
        """
        R = axis_trafo[:-1, :-1]  # Rotation in global basis frame
        T = axis_trafo[:-1, -1] - parent_p  # Translation in local joint-frame
        axis_n = R.dot(axis)
        return axis_n, T

    def hasSphericalWrist(self) -> bool:
        """Is true iff the robot has a spherical wrist."""
        return self._robot.is_spherical()

    def hasKnownDecomposition(self) -> bool:
        """Is true iff there is a geometric decomposition that allows solving IK for this robot."""
        return self._robot.has_known_decomposition()

    def getKinematicFamily(self) -> str:
        """Returns information on the kinematic family of the robot."""
        return self._robot.get_kinematic_family()

    def getRemodeled_H(self) -> np.ndarray:
        """Returns the joint axes of this robot after kinematic remodeling"""
        return self._robot.get_remodeled_H()

    def getRemodeled_P(self) -> np.ndarray:
        """Returns the joint axes' reference points for this robot after kinematic remodeling"""
        return self._robot.get_remodeled_P()

    def getOriginal_H(self) -> np.ndarray:
        """Returns the original joint axes of this robot before kinematic remodeling"""
        return self._robot.get_original_H()

    def getOriginal_P(self) -> np.ndarray:
        """Returns the original joint axes' reference points for this robot before kinematic remodeling"""
        return self._robot.get_original_P()

    def getIdealized_H(self) -> np.ndarray:
        """Returns the joint axes the analytical solver uses (after fixed-axis reduction and idealization)"""
        return self._robot.get_idealized_H()

    def getIdealized_P(self) -> np.ndarray:
        """Returns the joint offsets the analytical solver uses (after fixed-axis reduction and idealization)"""
        return self._robot.get_idealized_P()

    def getR6T(self) -> np.ndarray:
        """Returns the end-effector orientation w.r.t. the last joint frame (at zero configuration, base frame)"""
        return self._robot.get_R6T()

    def fwdKin(self, q: np.array) -> np.ndarray:
        """Calculate the forward kinematics for a given joint configuration"""
        return self._robot.fwdkin(q)

    def IK(self, pose: np.ndarray):
        """
        Calculate the inverse kinematics for a desired pose.

        The returned solution holds Q, is_LS, and per solution pos_error [m] / rot_error [rad]
        of fwdKin(Q) w.r.t. the pose on the original (non-idealized) model. For 6R robots,
        is_LS is True iff these exceed the robot's ls_pos_tol / ls_rot_tol.
        """
        return self._robot.calculate_IK(pose)

    def IK_batched(self, pose_batch, num_worker_threads=4):
        """
        Returns multiple IK for a batch of desired poses.

        :param pose_batch: A numpy array with shape (batch_size, 4, 4) or a list of 2D numpy arrays with shape (4, 4)
        :param num_worker_threads: Number of worker threads to use for parallel IK calculation
        """
        # Check if the input is a pure numpy array
        if isinstance(pose_batch, np.ndarray) and pose_batch.ndim == 3 and pose_batch.shape[1:] == (4, 4):
            pose_list = [pose_batch[i] for i in range(pose_batch.shape[0])]
        elif isinstance(pose_batch, list) and all(isinstance(p, np.ndarray) and p.shape == (4, 4) for p in pose_batch):
            pose_list = pose_batch
        else:
            raise ValueError("Input must be a numpy array with shape (batch_size, 4, 4) or a list of 2D numpy arrays "
                             "with shape (4, 4)")
        return self._robot.calculate_IK_batched(pose_list, num_worker_threads)
