import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot

class DhRobot(IKRobot):
    """A robot parameterized by standard Denavit-Hartenberg parameters"""

    def __init__(self,
                 dh_alpha: np.ndarray,
                 dh_a: np.ndarray,
                 dh_d: np.ndarray,
                 fixed_axes: list[tuple[int, float]] = None,
                 parallel_tol: float = 0.0,
                 intersect_tol: float = 0.0,
                 ls_pos_tol: float = 1e-8,
                 ls_rot_tol: float = 1e-8):
        """
        EAIK Robot parametrized by homogeneous joint transformations

        :param dh_alpha
        :param dh_a
        :param dh_d
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
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

        self._robot = EAIK.Robot(dh_alpha, dh_a, dh_d, np.eye(3), fixed_axes, True, parallel_tol, intersect_tol, ls_pos_tol, ls_rot_tol)
