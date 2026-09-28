"""
Carry the horizon as data.

Read off the wire, resampled onto the OCP's grid, written back as
`trajectory_msgs/JointTrajectory`. Python port of `src/horizon_source.cpp`,
knot-for-knot: arrays, not a vector of structs, since resample is one
Hermite evaluation over all N knots, not N of them. ROS-free apart from the
marshalling functions, so the maths is tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from builtin_interfaces.msg import Duration
from crane_model import symbolic as cs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import problem

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
    reference carries no sway, so it stays zero there. `u` is likewise the
    solved command and stays zero on a reference, which carries none.
    """

    t: np.ndarray  # (N,)
    q_a_ref: np.ndarray  # (N, 6)
    dq_a_ref: np.ndarray  # (N, 6)
    ddq_a_ref: np.ndarray  # (N, 6)
    q_u_ref: np.ndarray  # (N, 2)
    dq_u_ref: np.ndarray  # (N, 2)
    u: np.ndarray  # (N, 6) -- joint velocity command at Psi's input

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
            np.zeros((count, ACTUATED_DOF)),
        )

    def copy(self) -> Knots:
        return Knots(
            self.t.copy(),
            self.q_a_ref.copy(),
            self.dq_a_ref.copy(),
            self.ddq_a_ref.copy(),
            self.q_u_ref.copy(),
            self.dq_u_ref.copy(),
            self.u.copy(),
        )


def duration_seconds(duration) -> float:
    return duration.sec + duration.nanosec / 1e9


#: How long before a knot the previous command is still held; well under a JTC tick.
HOLD_EDGE_S = 0.001


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


def path_from_message(message, joints):
    """
    Lift the planned columns out of a `crane_msgs/JointPath`.

    Returns `(samples, duration, why)` -- `samples` is `None` and `why` says why
    on a message this node cannot use. The rows are uniform in the planner's own
    `sigma`, which is exactly what `solver.path_control` fits, so nothing here
    interpolates: refusing a malformed message is the whole job.
    """
    names = list(message.joint_names)
    columns = []
    for joint in joints:
        if joint not in names:
            return None, 0.0, f"the path does not name {joint}"
        columns.append(names.index(joint))
    width = len(names)
    if width == 0:
        return None, 0.0, "the path names no joints"

    flat = np.asarray(message.q_path, dtype=float)
    if flat.size == 0 or flat.size % width:
        return (
            None,
            0.0,
            (
                f"the path carries {flat.size} values, which is not a whole number of "
                f"{width}-joint rows"
            ),
        )
    rows = flat.size // width
    if rows < problem.PATH_POINTS:
        return (
            None,
            0.0,
            (
                f"the path carries {rows} samples, fewer than the "
                f"{problem.PATH_POINTS} control points it is fitted to, so the fit "
                "would be underdetermined rather than an approximation of the path"
            ),
        )
    if not np.all(np.isfinite(flat)):
        return None, 0.0, "the path carries a value that is not finite"

    duration = duration_seconds(message.duration)
    if not np.isfinite(duration) or duration <= 0.0:
        return (
            None,
            0.0,
            (
                f"the path is spent in {duration:.3g} s; a plan that takes no time has "
                "no pace to price progress against"
            ),
        )
    return flat.reshape(rows, width)[:, columns], duration, ""


def horizon_to_message(
    horizon: Knots,
    joints,
    stamp,
    lead: float = 0.0,
    period: float = 0.0,
    in_flight: Knots | None = None,
    in_flight_u=None,
    linear: bool = False,
):
    """
    Write the horizon as a `JointTrajectory`.

    `joints` is the canonical eight, in contract order; JTC's `dof_` list is
    eight wide, and `allow_partial_joints_goal: false` rejects six names
    whole. The two passive columns are the solved sway -- a fill, not a
    computation.

    No accelerations: `ddq_a_ref` is the OCP's stage residual, not a command.

    `lead` is the plant's own dead time. When it is non-zero, `stamp` is the
    measurement instant, `in_flight_u` the command published last cycle (the
    JTC plays it until knot 0, one `Ts` on; `in_flight` is its horizon, for
    the positions) and the message is `_command_message`'s, which the
    JTC (`period` its tick) plays back exactly as the OCP assumes. Issue 161
    placed knot 0 at `lead` with the in-flight knot prepended instead: that
    counted the dead time twice on the command, and with the JTC's linear
    effort ramp and one-tick feedforward lookahead it put `u` 30-60 ms late
    against a loop with ~10 ms of margin -- the Gazebo divergence.
    Without a lead the plain form below goes out (shadow and offline paths).
    `linear` (C4): `u` is the command state at each knot, ramped between them,
    and `in_flight_u` the in-flight segment's start command.

    `effort` carries a feed-forward *velocity*, not a torque, read under the
    JTC's `effort_field_is_feedforward`. `u - dq_a_ref` is the same identity
    `crane_planning` writes: the plugin adds `ff_velocity_scale*dq_ref` whether
    or not anyone wants it, so the difference is what makes the open-loop branch
    the OCP's own `u`. Without it the machine is commanded the reference velocity
    and C3's force state never charges -- `tau_dot = k*(u_f - dq)` is zero at
    `u == dq`. The tool row falls out at zero: it is pinned, so both terms are.
    """
    position = np.zeros((len(horizon), cs.K_GENERALIZED_DOF))
    velocity = np.zeros_like(position)
    position[:, cs.K_ACTUATED_ROWS] = horizon.q_a_ref
    velocity[:, cs.K_ACTUATED_ROWS] = horizon.dq_a_ref
    position[:, cs.K_PASSIVE_ROWS] = horizon.q_u_ref
    velocity[:, cs.K_PASSIVE_ROWS] = horizon.dq_u_ref
    effort = np.zeros_like(position)
    effort[:, cs.K_ACTUATED_ROWS] = horizon.u - horizon.dq_a_ref

    message = JointTrajectory()
    message.header.stamp = stamp
    message.joint_names = list(joints)
    message.points = [
        JointTrajectoryPoint(
            positions=position[index].tolist(),
            velocities=velocity[index].tolist(),
            effort=effort[index].tolist(),
            time_from_start=seconds_duration(float(horizon.t[index])),
        )
        for index in range(len(horizon))
    ]
    if lead > 0.0:
        return _command_message(
            horizon, joints, stamp, period, in_flight, in_flight_u, linear
        )
    return message


def _jtc_cubic(t0, t1, p0, v0, p1, v1, t):
    """Interpolate as the JTC does between two points (cubic Hermite)."""
    T, x = t1 - t0, t - t0
    a2 = (3.0 * (p1 - p0) - (2.0 * v0 + v1) * T) / T**2
    a3 = (2.0 * (p0 - p1) + (v0 + v1) * T) / T**3
    return p0 + v0 * x + a2 * x**2 + a3 * x**3, v0 + 2.0 * a2 * x + 3.0 * a3 * x**2


def _command_message(
    horizon: Knots,
    joints,
    stamp,
    period: float,
    in_flight=None,
    in_flight_u=None,
    linear=False,
) -> JointTrajectory:
    """
    Encode the horizon so the JTC plays exactly what the OCP assumes.

    Stamped at the measurement instant, knot i at `t_i`: the plant's own dead
    time delays the command, so `u_i` must leave the JTC over [t_i, t_i+Ts)
    while its position reference already stands where the machine will be.
    Three JTC details each break the loop (margin ~10 ms) if ignored:
    - `effort` is interpolated linearly: a point `HOLD_EDGE_S` before each
      step keeps the previous command, so it is a zero-order hold;
    - feedforward is sampled one `period` ahead: the step sits at t_i+period;
    - pos/vel are a cubic Hermite: actuated velocities are the slope of the
      positions sent (the curve's, not the solver's), or the cubic swings
      between knots and `dq_ref + effort` stops summing to `u`.
    `linear` (C4): the command is piecewise linear through `u` on the knots, so
    no hold edges; every point carries `c(s - period) - v(s)`.
    """
    t = np.asarray(horizon.t, dtype=float)
    q_a, q_u, dq_u = horizon.q_a_ref, horizon.q_u_ref, horizon.dq_u_ref
    u = np.asarray(horizon.u, dtype=float)
    if in_flight_u is not None:
        # [0, Ts) is last cycle's command; knot 0 stands one `Ts` on.
        first = horizon if in_flight is None else in_flight
        t = np.concatenate([[0.0], t + (t[1] - t[0])])
        q_a = np.vstack([first.q_a_ref[:1], q_a])
        q_u = np.vstack([first.q_u_ref[:1], q_u])
        dq_u = np.vstack([first.dq_u_ref[:1], dq_u])
        now = np.zeros((1, u.shape[1]))
        now[0, : cs.K_PLANNED_DOF] = np.asarray(in_flight_u)[: cs.K_PLANNED_DOF]
        u = np.vstack([now, u])
    position = np.zeros((len(t), cs.K_GENERALIZED_DOF))
    velocity = np.zeros_like(position)
    actuated, passive = list(cs.K_ACTUATED_ROWS), list(cs.K_PASSIVE_ROWS)
    position[:, actuated] = q_a
    position[:, passive] = q_u
    velocity[:, actuated] = np.gradient(q_a, t, axis=0, edge_order=2)
    velocity[:, passive] = dq_u
    # u_i takes over at t_i + period; just before, the previous one still holds.
    steps = t[1:] + period
    # Linear: c(s - period) kinks at every t_j + period, t_0's included.
    edges = [t[:1] + period] if linear else [steps - HOLD_EDGE_S]
    # Rounded to the wire's nanoseconds, or float near-duplicates go out as equal
    # `time_from_start` and the JTC rejects the whole message.
    grid = np.unique(np.round(np.concatenate([t, *edges, steps]), 9))
    grid = grid[grid <= t[-1]]

    points = []
    for at in grid:
        i = min(np.searchsorted(t, at, side="right") - 1, len(t) - 2)
        p, v = _jtc_cubic(
            t[i],
            t[i + 1],
            position[i],
            velocity[i],
            position[i + 1],
            velocity[i + 1],
            at,
        )
        if linear:
            command = np.array([np.interp(at - period, t, c) for c in u.T])
        else:
            command = u[np.searchsorted(steps, at, side="right")]
        effort = np.zeros(cs.K_GENERALIZED_DOF)
        effort[actuated] = command - v[actuated]
        points.append(
            JointTrajectoryPoint(
                positions=p.tolist(),
                velocities=v.tolist(),
                effort=effort.tolist(),
                time_from_start=seconds_duration(float(at)),
            )
        )
    message = JointTrajectory()
    message.header.stamp = stamp
    message.joint_names = list(joints)
    message.points = points
    return message
