from urchin import URDF
import numpy as np

import eaik.pybindings.EAIK as EAIK
from eaik.IK_Robot import IKRobot


class UrdfRobot(IKRobot):
    """A robot for which the kinematic chain is parsed from a URDF file."""

    def __init__(self,
                 file_path: str,
                 fixed_axes: list[tuple[int, float]] = None,
                 tip_link: str = None,
                 parallel_tol: float = 0.0,
                 intersect_tol: float = 0.0,
                 ls_pos_tol: float = 1e-8,
                 ls_rot_tol: float = 1e-8):
        """
        EAIK Robot parametrized by URDF file

        :param file_path: Path to URDF file
        :param fixed_axes: List of tuples defining fixed joints (zero-indexed) (i, q_i+1)
        :param tip_link: Link whose pose is the IK target / fwdKin output. It must be attached to the child
            link of the last actuated joint through fixed joints only. If None, the end effector is the
            origin of the last actuated joint with the base-frame orientation at zero configuration
            (behaviour of eaik <= 1.2.2).
        :param parallel_tol: Idealization [rad]: consecutive joint axes closer than this to parallel are
            treated as exactly parallel by the analytical solver (0 = off)
        :param intersect_tol: Idealization [m]: consecutive joint axes closer than this to intersecting are
            treated as exactly intersecting by the analytical solver (0 = off)
        :param ls_pos_tol: 6R solutions missing the target position by more than this [m] are flagged is_LS
        :param ls_rot_tol: 6R solutions missing the target rotation by more than this [rad] are flagged is_LS
        """
        if fixed_axes is None:
            fixed_axes = []
        super().__init__()
        robot = URDF.load(file_path, lazy_load_meshes=True)
        joints = robot._sort_joints(robot.actuated_joints)

        fk_zero_pose = robot.link_fk()  # Calculate FK

        parent_p = np.zeros(3)
        H = np.array([], dtype=np.int64).reshape(0, 3)  # axes
        P = np.array([], dtype=np.int64).reshape(0, 3)  # offsets
        for i in range(len(joints)):
            joint_child_link = robot.link_map[joints[i].child]
            h, p = self.urdf_to_sp_conv(fk_zero_pose[joint_child_link], joints[i].axis, parent_p)
            H = np.vstack([H, h])
            P = np.vstack([P, p])
            parent_p += p

        if tip_link is None:
            # End effector displacement is (0,0,0), orientation = base frame
            P = np.vstack([P, np.zeros(3)])
            R6T = np.eye(3)
        else:
            self._check_rigidly_attached(robot, tip_link, joints[-1].child)
            T_tip = fk_zero_pose[robot.link_map[tip_link]]
            P = np.vstack([P, T_tip[:3, 3] - parent_p])   # last joint origin -> tip, base frame, q = 0
            R6T = np.array(T_tip[:3, :3])                   # tip orientation at q = 0, base frame

        self.tip_link = tip_link
        self._robot = EAIK.Robot(H.T, P.T, R6T, fixed_axes, True,
                                 parallel_tol, intersect_tol, ls_pos_tol, ls_rot_tol)

    @staticmethod
    def _check_rigidly_attached(robot, tip_link: str, last_child: str):
        if tip_link not in robot.link_map:
            raise ValueError(f"tip_link '{tip_link}' is not a link of the URDF")
        parent_joint = {j.child: j for j in robot.joints}
        link = tip_link
        while link != last_child:
            joint = parent_joint.get(link)
            if joint is None:
                raise ValueError(f"tip_link '{tip_link}' is not downstream of the last actuated joint")
            if joint.joint_type != "fixed":
                raise ValueError(f"tip_link '{tip_link}' is not rigidly attached to '{last_child}' "
                                 f"(the path crosses the non-fixed joint '{joint.name}')")
            link = joint.parent
