import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot


class HomogeneousRobot(IKRobot):
    """A robot parametrized by homogeneous joint transformations"""

    def __init__(self,
                 joint_trafos: np.ndarray,
                 fixed_axes: list[tuple[int, float]] = None,
                 joint_axis: np.array = np.array([0, 0, 1]),
                 parallel_tol: float = 0.0,
                 intersect_tol: float = 0.0,
                 ls_pos_tol: float = 1e-8,
                 ls_rot_tol: float = 1e-8):
        """
        EAIK Robot parametrized by homogeneous joint transformations

        :param joint_trafos: (N+1)x4x4 numpy array of homogeneous transformations of each of N joints w.r.t. the world
            frame, i.e. (T01, T02, ..., T0EE)
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param joint_axis: Unit vector of joint axis orientation within each frame in joint_trafos (e.g., z-axis)
        :param parallel_tol: Idealization [rad]: consecutive joint axes closer than this to parallel are
            treated as exactly parallel by the analytical solver (0 = off)
        :param intersect_tol: Idealization [m]: consecutive joint axes closer than this to intersecting are
            treated as exactly intersecting by the analytical solver (0 = off)
        :param ls_pos_tol: 6R solutions missing the target position by more than this [m] are flagged is_LS
        :param ls_rot_tol: 6R solutions missing the target rotation by more than this [rad] are flagged is_LS
        """
        super().__init__()
        if fixed_axes is None:
            fixed_axes = []

        H = np.array([], dtype=np.float64).reshape(0, 3)  # axes
        P = np.array([], dtype=np.float64).reshape(0, 3)  # offsets

        # Derive Kinematic from zero-position
        parent_p = np.zeros(3, dtype=np.float64)
        for frame in joint_trafos[:-1]:
            h, p = IKRobot.urdf_to_sp_conv(frame, joint_axis, parent_p)
            H = np.vstack([H, h])
            P = np.vstack([P, p])
            parent_p += p
            
        # Account for end effector pose
        p_EE = joint_trafos[-1][:-1, -1] - joint_trafos[-2][:-1, -1] # Translation in local joint-frame
        P = np.vstack([P, p_EE])
        rNt = joint_trafos[-1][:-1, :-1]  # Rotation in global basis frame

        self._robot = EAIK.Robot(H.T, P.T, rNt, fixed_axes, True, parallel_tol, intersect_tol, ls_pos_tol, ls_rot_tol)
