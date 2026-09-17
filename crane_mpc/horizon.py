"""
Carry the horizon as data.

Read off the wire, resampled onto the OCP's grid, written back as
`trajectory_msgs/JointTrajectory`. Python port of `src/horizon_source.cpp`,
knot-for-knot: arrays, not a vector of structs, since resample is one
Hermite evaluation over all N knots, not N of them. ROS-free apart from the
two marshalling functions, so the maths is tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from builtin_interfaces.msg import Duration
from crane_model import symbolic as cs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ACTUATED_DOF = cs.K_ACTUATED_DOF
PASSIVE_DOF = cs.K_PASSIVE_DOF


class Rejection(Enum):
    """Why a resample produced nothing. `str()` is the clause the node logs."""

    EMPTY_REFERENCE = "the reference carries no points"
    DEGENERATE_GRID = (
        "the horizon grid is shorter than two knots or its step is not positive"
    )
    NON_MONOTONIC_TIME = "the reference's times are not strictly increasing"
    NON_FINITE = "the reference carries a value that is not finite"
    OFFSET_BEFORE_REFERENCE = (
        "the horizon would start before the reference's own first point"
    )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Grid:
    """`Ts` seconds between `horizon_length` knots."""

    Ts: float
    horizon_length: int

    def duration(self) -> float:
        return (self.horizon_length - 1) * self.Ts if self.horizon_length >= 2 else 0.0


@dataclass
class Knots:
    """
    N knots. In a reference `t` is time_from_start; in a horizon, index*Ts.

    Sway pair is the solved one, shifted by `Cycle.adopt_solution`; a
    reference carries no sway, so it stays zero there.
    """

    t: np.ndarray  # (N,)
    q_a_ref: np.ndarray  # (N, 6)
    dq_a_ref: np.ndarray  # (N, 6)
    ddq_a_ref: np.ndarray  # (N, 6)
    q_u_ref: np.ndarray  # (N, 2)
    dq_u_ref: np.ndarray  # (N, 2)

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @staticmethod
    def zeros(count: int) -> Knots:
        return Knots(
            np.zeros(count),
            np.zeros((count, ACTUATED_DOF)),
            np.zeros((count, ACTUATED_DOF)),
            np.zeros((count, ACTUATED_DOF)),
            np.zeros((count, PASSIVE_DOF)),
            np.zeros((count, PASSIVE_DOF)),
        )

    def copy(self) -> Knots:
        return Knots(
            self.t.copy(),
            self.q_a_ref.copy(),
            self.dq_a_ref.copy(),
            self.ddq_a_ref.copy(),
            self.q_u_ref.copy(),
            self.dq_u_ref.copy(),
        )


def duration_seconds(duration) -> float:
    return duration.sec + duration.nanosec / 1e9


def seconds_duration(seconds: float) -> Duration:
    whole = np.floor(seconds)
    nanoseconds = round((seconds - whole) * 1e9)
    if nanoseconds >= 1_000_000_000:
        nanoseconds -= 1_000_000_000
        whole += 1
    return Duration(sec=int(whole), nanosec=int(nanoseconds))


def resample(reference: Knots, offset: float, grid: Grid):
    """
    Evaluate the reference at `offset + index*Ts`.

    Cubic Hermite in position and velocity; past the plan's last point every
    knot holds the goal at rest. Returns `(rejection, horizon)`, exactly one
    of which is None.
    """
    if grid.horizon_length < 2 or not np.isfinite(grid.Ts) or grid.Ts <= 0.0:
        return Rejection.DEGENERATE_GRID, None
    if len(reference) == 0:
        return Rejection.EMPTY_REFERENCE, None
    if not np.isfinite(offset):
        return Rejection.NON_FINITE, None
    if not (
        np.all(np.isfinite(reference.t))
        and np.all(np.isfinite(reference.q_a_ref))
        and np.all(np.isfinite(reference.dq_a_ref))
    ):
        return Rejection.NON_FINITE, None
    if len(reference) > 1 and not np.all(np.diff(reference.t) > 0.0):
        return Rejection.NON_MONOTONIC_TIME, None
    if offset < reference.t[0]:
        return Rejection.OFFSET_BEFORE_REFERENCE, None

    horizon = Knots.zeros(grid.horizon_length)
    horizon.t[:] = np.arange(grid.horizon_length) * grid.Ts
    t = offset + horizon.t

    # Past the end: goal at rest. `hold` is never empty once the plan is
    # spent, and `t` increasing keeps the two halves contiguous.
    hold = t >= reference.t[-1]
    horizon.q_a_ref[hold] = reference.q_a_ref[-1]

    live = ~hold
    if np.any(live):
        # C++ walks `segment` forward while ref[segment+1].t <= t, stops at
        # len-2; searchsorted on increasing t gives the same index.
        left = np.clip(
            np.searchsorted(reference.t, t[live], side="right") - 1,
            0,
            len(reference) - 2,
        )
        _hermite(reference, left, t[live], horizon, live)
    return None, horizon


def _hermite(reference: Knots, left: np.ndarray, t: np.ndarray, out: Knots, rows):
    right = left + 1
    dt = (reference.t[right] - reference.t[left])[:, None]
    s = ((t - reference.t[left]) / dt[:, 0])[:, None]

    q0 = reference.q_a_ref[left]
    v0 = reference.dq_a_ref[left]
    q1 = reference.q_a_ref[right]
    v1 = reference.dq_a_ref[right]

    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    d00 = 6.0 * s2 - 6.0 * s
    d10 = 3.0 * s2 - 4.0 * s + 1.0
    d01 = -6.0 * s2 + 6.0 * s
    d11 = 3.0 * s2 - 2.0 * s
    e00 = 12.0 * s - 6.0
    e10 = 6.0 * s - 4.0
    e01 = -12.0 * s + 6.0
    e11 = 6.0 * s - 2.0

    out.q_a_ref[rows] = h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1
    out.dq_a_ref[rows] = (d00 * q0 + d01 * q1) / dt + d10 * v0 + d11 * v1
    out.ddq_a_ref[rows] = (e00 * q0 + e01 * q1) / (dt * dt) + (e10 * v0 + e11 * v1) / dt


def reference_from_message(message: JointTrajectory, joints):
    """
    Lift the six named joints out of the wire order.

    Returns `(reference, why)` or `(None, why)`: missing a joint, position or
    velocity refuses the whole reference -- a partial plan is not a plan.
    """
    if not message.points:
        return None, "the reference carries no points"
    names = list(message.joint_names)
    column = []
    for joint in joints:
        if joint not in names:
            return None, f"the reference does not name {joint}"
        column.append(names.index(joint))
    width = len(names)

    reference = Knots.zeros(len(message.points))
    for index, point in enumerate(message.points):
        if len(point.positions) != width:
            return None, (
                f"the reference carries no position for every joint at point {index}"
            )
        if len(point.velocities) != width:
            return None, (
                f"the reference carries no velocity for every joint at point {index}"
            )
        reference.t[index] = duration_seconds(point.time_from_start)
        positions = np.asarray(point.positions)
        velocities = np.asarray(point.velocities)
        reference.q_a_ref[index] = positions[column]
        reference.dq_a_ref[index] = velocities[column]
    return reference, ""


def horizon_to_message(horizon: Knots, joints, first_knot_valid_at):
    """
    Write the horizon as a `JointTrajectory`.

    `joints` is the canonical eight, in contract order; JTC's `dof_` list is
    eight wide, and `allow_partial_joints_goal: false` rejects six names
    whole. The two passive columns are the solved sway -- a fill, not a
    computation.

    No accelerations: `ddq_a_ref` is the OCP's stage residual, not a command.
    """
    position = np.zeros((len(horizon), cs.K_GENERALIZED_DOF))
    velocity = np.zeros_like(position)
    position[:, cs.K_ACTUATED_ROWS] = horizon.q_a_ref
    velocity[:, cs.K_ACTUATED_ROWS] = horizon.dq_a_ref
    position[:, cs.K_PASSIVE_ROWS] = horizon.q_u_ref
    velocity[:, cs.K_PASSIVE_ROWS] = horizon.dq_u_ref

    message = JointTrajectory()
    message.header.stamp = first_knot_valid_at
    message.joint_names = list(joints)
    message.points = [
        JointTrajectoryPoint(
            positions=position[index].tolist(),
            velocities=velocity[index].tolist(),
            time_from_start=seconds_duration(float(horizon.t[index])),
        )
        for index in range(len(horizon))
    ]
    return message
