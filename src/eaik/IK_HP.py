import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot

class HPRobot(IKRobot):
    """A robot parameterized by by H and P vectors"""

    def __init__(self,
                 H: np.ndarray,
                 P: np.ndarray,
                 fixed_axes: list[tuple[int, float]] = None,
                 R6T: np.ndarray = None,
                 parallel_tol: float = 0.0,
                 intersect_tol: float = 0.0,
                 ls_pos_tol: float = 1e-8,
                 ls_rot_tol: float = 1e-8):
        """
        EAIK Robot parametrized by H and P vectors

        :param H: (N, 3) numpy array resembling the unit direction vectors of the joints
        :param P: (N+1, 3) numpy array resembling the offsets between the joint axes 
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param R6T: (3, 3) end-effector orientation at zero configuration (base frame); identity if None
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

        if R6T is None:
            R6T = np.eye(3)
        self._robot = EAIK.Robot(H.T, P.T, np.asarray(R6T, dtype=float), fixed_axes, True, parallel_tol, intersect_tol, ls_pos_tol, ls_rot_tol)