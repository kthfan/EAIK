# EAIK 1.2.2+robust.1: changes against v1.2.2

All new parameters default to the 1.2.2 behaviour of the analytical solver: with default
arguments, `Q` is bit-identical to v1.2.2 (checked on 5,004 poses of a calibrated TM5S).
The meaning of `is_LS` for 6R robots is the one intended change of behaviour.

## 1. `UrdfRobot(..., tip_link=None)`

v1.2.2 ignores everything after the last actuated joint: the end effector is the origin of
the last joint with the base orientation at zero configuration (`R6T = I`, `p6T = 0`).
With `tip_link="flange"` (any link rigidly attached to the last joint's child link through
fixed joints) `R6T` and `p6T` are taken from the URDF, so `IK` takes and `fwdKin` returns the
pose of that link. A link that is not rigidly attached raises `ValueError`.

## 2. `is_LS` from the final pose error; new `pos_error`, `rot_error`

v1.2.2 ORs the least-squares flags of the intermediate subproblems, each tested against an
absolute 1e-8. On calibrated geometry these flags are set for exact solutions too, and SP6
never sets its flag. Now every solution is evaluated with `fwdkin` of the original model:

- `sol.pos_error[i]`: position error [m]
- `sol.rot_error[i]`: rotation angle error [rad], computed with atan2 so it is accurate
  near 0
- 6R robots: `is_LS[i] = pos_error > ls_pos_tol or rot_error > ls_rot_tol`, with defaults
  of 1e-8 m and 1e-8 rad. For robots with fewer than 6 joints the subproblem flags are kept,
  because a full pose is not reachable in general.

## 3. Idealization (`parallel_tol` [rad], `intersect_tol` [m], 0 = off)

EAIK picks its decomposition from exact geometric relations tested against absolute
thresholds. Calibrated robots satisfy them only approximately, and that has two effects:

- The robot falls into a different (general) branch whose subproblem is ill-conditioned
  near the relation. Example: the TM5S axes 5/6 meet within 19 nm, so EAIK uses SP6 instead
  of SP4 and returns solutions that miss by up to 0.1 m.
- No decomposition is found at all. Calibrated UR-like and KUKA-like models give
  "6R-Unknown Kinematic Class".

Before classification, `idealize_kinematics` (kinematic_remodeling.cpp) does three things:

1. Consecutive axes within `parallel_tol` get one exact common direction.
2. Consecutive non-parallel axes whose common normal is shorter than `intersect_tol` are made
   to intersect. The offset between them is shifted along the common normal, and the
   downstream chain moves with it.
3. Axes i, i+1, i+2 that pairwise intersect at points less than `intersect_tol` apart are made
   to meet in one point. Axis i+2 is shifted along axis i+1.

The analytical solver uses the idealized model (`getIdealized_H/P`). `fwdKin`, `pos_error`
and `is_LS` use the original model. The solutions are therefore exact for the idealized
model and off by about the idealization error on the original one. Removing that error is
left to a numerical layer, such as Newton refinement on the original model.

With fixed axes, the idealization is applied to the reduced chain.

## API summary

- `EAIK.Robot(H, P, R6T, fixed_axes, use_double_precision, parallel_tol=0, intersect_tol=0,
  ls_pos_tol=1e-8, ls_rot_tol=1e-8)`, same for the DH constructor
- `UrdfRobot`, `HPRobot` (new `R6T` argument), `DhRobot`, `HomogeneousRobot`: the same
  keyword arguments
- `IKRobot.getIdealized_H()`, `getIdealized_P()`, `getR6T()`
- `IKSolution.pos_error`, `IKSolution.rot_error`
