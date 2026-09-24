"""The reference on the wire, resampled onto the OCP's grid."""

import numpy as np
import pytest
from builtin_interfaces.msg import Duration
from crane_mpc import problem
from crane_mpc.horizon import (
    Grid,
    Knots,
    Rejection,
    horizon_to_message,
    path_from_message,
    reference_from_message,
    resample,
)
from crane_msgs.msg import JointPath
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = ["sw", "ha", "ka", "sa", "ro", "gr"]
#: The canonical eight, with the two passive coordinates at rows 4 and 5.
CANONICAL = ["sw", "ha", "ka", "sa", "tip", "tilt", "ro", "gr"]


def ramp(times, slope=0.3):
    reference = Knots.zeros(len(times))
    reference.t[:] = times
    reference.q_a_ref[:, 0] = slope * np.asarray(times)
    reference.dq_a_ref[:, 0] = slope
    return reference


def test_the_resample_reproduces_the_reference_at_its_own_knots():
    reference = ramp([0.0, 0.4, 0.8, 1.2])
    rejection, horizon = resample(reference, 0.0, Grid(0.4, 4))
    assert rejection is None
    assert np.allclose(horizon.q_a_ref, reference.q_a_ref)
    # Last knot lands on the plan's own end: held goal at rest, not interpolation.
    assert np.allclose(horizon.dq_a_ref[:-1], reference.dq_a_ref[:-1])
    assert np.allclose(horizon.dq_a_ref[-1], 0.0)
    # The horizon's own time is knot-local, not the reference's.
    assert np.allclose(horizon.t, [0.0, 0.4, 0.8, 1.2])


def test_past_the_plan_the_goal_is_held_at_rest():
    reference = ramp([0.0, 0.4])
    rejection, horizon = resample(reference, 0.2, Grid(0.4, 4))
    assert rejection is None
    assert np.allclose(horizon.q_a_ref[1:], reference.q_a_ref[-1])
    assert np.allclose(horizon.dq_a_ref[1:], 0.0)
    assert np.allclose(horizon.ddq_a_ref[1:], 0.0)


@pytest.mark.parametrize(
    "reference, offset, grid, expected",
    [
        (ramp([0.0, 0.4]), 0.0, Grid(0.4, 1), Rejection.DEGENERATE_GRID),
        (Knots.zeros(0), 0.0, Grid(0.04, 5), Rejection.EMPTY_REFERENCE),
        (ramp([0.0, 0.4, 0.4]), 0.0, Grid(0.04, 5), Rejection.NON_MONOTONIC_TIME),
        (ramp([0.0, np.inf]), 0.0, Grid(0.04, 5), Rejection.NON_FINITE),
        (ramp([1.0, 2.0]), 0.5, Grid(0.04, 5), Rejection.OFFSET_BEFORE_REFERENCE),
    ],
)
def test_a_reference_the_horizon_cannot_carry_is_refused(
    reference, offset, grid, expected
):
    rejection, horizon = resample(reference, offset, grid)
    assert rejection is expected
    assert horizon is None


def test_the_reference_is_read_by_joint_name_and_never_by_index():
    """The two stacks publish different name sets in different orders."""
    message = JointTrajectory()
    message.joint_names = ["gr", "sa", "sw", "ka", "extra", "ha", "ro"]
    point = JointTrajectoryPoint()
    point.positions = [6.0, 4.0, 1.0, 3.0, 99.0, 2.0, 5.0]
    point.velocities = [0.6, 0.4, 0.1, 0.3, 9.9, 0.2, 0.5]
    point.time_from_start = Duration(sec=0, nanosec=500000000)
    message.points = [point]

    reference, why = reference_from_message(message, JOINTS)
    assert why == ""
    assert np.allclose(reference.q_a_ref[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(reference.dq_a_ref[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert reference.t[0] == pytest.approx(0.5)


def test_a_reference_missing_a_joint_is_refused_whole():
    message = JointTrajectory()
    message.joint_names = JOINTS[:-1]
    point = JointTrajectoryPoint()
    point.positions = [0.0] * 5
    point.velocities = [0.0] * 5
    message.points = [point]
    reference, why = reference_from_message(message, JOINTS)
    assert reference is None
    assert "gr" in why


def test_the_wire_form_carries_positions_velocities_and_no_accelerations():
    horizon = ramp([0.0, 0.04, 0.08])
    horizon.ddq_a_ref[:] = 7.0
    from builtin_interfaces.msg import Time

    message = horizon_to_message(horizon, CANONICAL, Time(sec=3, nanosec=0))
    assert message.joint_names == CANONICAL
    assert message.header.stamp.sec == 3
    assert len(message.points) == 3
    assert message.points[2].time_from_start == Duration(sec=0, nanosec=80000000)
    assert message.points[1].positions[0] == pytest.approx(0.3 * 0.04)
    assert list(message.points[0].accelerations) == []


def test_the_wire_form_names_the_canonical_eight_and_carries_the_sway():
    """
    JTC's `joints` is eight wide; `allow_partial_joints_goal: false` rejects a
    six-name trajectory outright, so passive columns go out in canonical rows.
    """
    from builtin_interfaces.msg import Time

    horizon = ramp([0.0, 0.04])
    horizon.q_u_ref[:] = [0.1, 1.5]
    horizon.dq_u_ref[:] = [-0.2, 0.3]

    message = horizon_to_message(horizon, CANONICAL, Time(sec=0, nanosec=0))
    point = message.points[1]
    assert len(point.positions) == len(CANONICAL)
    assert [point.positions[row] for row in (4, 5)] == pytest.approx([0.1, 1.5])
    assert [point.velocities[row] for row in (4, 5)] == pytest.approx([-0.2, 0.3])
    # The actuated six keep their canonical rows around the sway pair.
    assert point.positions[0] == pytest.approx(0.3 * 0.04)
    assert point.positions[6] == pytest.approx(0.0)


def test_the_effort_field_is_the_feedforward_velocity_not_a_torque():
    """
    `u - dq_a_ref`, the identity `crane_planning` writes.

    The JTC adds `ff_velocity_scale*dq_ref` whether or not anyone wants it, so
    the difference is what makes the open-loop branch the OCP's own `u`. Sending
    `dq_a_ref` alone leaves C3's force state uncharged -- `tau_dot = k*(u_f - dq)`
    is zero at `u == dq` -- which is the drift this field exists to remove.
    """
    from builtin_interfaces.msg import Time

    horizon = ramp([0.0, 0.04])
    horizon.u[:, 0] = 0.5  # slew: commanded faster than the 0.3 rad/s it moves
    horizon.dq_a_ref[:, 4] = 0.2  # rotator, moving
    horizon.u[:, 4] = 0.7  # and commanded ahead of itself

    point = horizon_to_message(horizon, CANONICAL, Time(sec=0, nanosec=0)).points[1]
    assert len(point.effort) == len(CANONICAL)
    # Canonical rows 0 and 6 are the slew and the rotator: the actuated six sit
    # around the sway pair, and effort has to land on the same rows as velocity.
    assert point.effort[0] == pytest.approx(0.5 - 0.3)
    assert point.effort[6] == pytest.approx(0.7 - 0.2)
    # Passive rows carry no command, and the tool is pinned so both terms are 0.
    assert [point.effort[row] for row in (4, 5, 7)] == pytest.approx([0.0, 0.0, 0.0])


# -- the planner's curve on the wire --------------------------------------------

PLANNED = ["sw", "ha", "ka", "sa", "ro"]
#: Enough rows to determine the fit; fewer is refused, not approximated.
ROWS = problem.PATH_POINTS


def joint_path(rows=ROWS, names=None, duration=2.0):
    """A `JointPath` whose value at row `i`, column `j` is `i + j / 10`."""
    names = PLANNED if names is None else names
    message = JointPath()
    message.joint_names = list(names)
    message.q_path = [
        float(row) + column / 10.0
        for row in range(rows)
        for column in range(len(names))
    ]
    whole = int(duration)
    message.duration = Duration(sec=whole, nanosec=int((duration - whole) * 1e9))
    return message


def test_the_path_is_lifted_by_name_and_not_by_column():
    # Wire order reversed against this node's: read by column the rows come back
    # mirrored, and a mirrored path is a different move at the same cost.
    message = joint_path(names=list(reversed(PLANNED)))
    samples, duration, why = path_from_message(message, PLANNED)
    assert why == ""
    assert duration == pytest.approx(2.0)
    expected = np.array(
        [
            [row + (len(PLANNED) - 1 - column) / 10.0 for column in range(len(PLANNED))]
            for row in range(ROWS)
        ]
    )
    assert samples == pytest.approx(expected)


@pytest.mark.parametrize(
    "spoil, clause",
    [
        (lambda m: m.q_path.pop(), "whole number"),
        (lambda m: setattr(m, "duration", Duration()), "no pace"),
        # A coarse path is what a second producer would send, and it is the one
        # that fails silently: `pinv` answers an underdetermined fit with a curve
        # that dives toward zero rather than refusing.
        (
            lambda m: setattr(m, "q_path", m.q_path[: 10 * len(PLANNED)]),
            "underdetermined",
        ),
        (lambda m: setattr(m, "q_path", m.q_path[: len(PLANNED)]), "underdetermined"),
        (
            lambda m: setattr(m, "joint_names", ["sw", "ha", "ka", "sa", "zz"]),
            "not name ro",
        ),
    ],
)
def test_a_path_this_node_cannot_use_is_refused_whole(spoil, clause):
    message = joint_path()
    spoil(message)
    samples, _, why = path_from_message(message, PLANNED)
    assert samples is None
    assert clause in why


def test_nothing_on_the_wire_is_still_in_the_future_when_it_arrives():
    """
    Issue 161. `Ts == sensor_to_valve_delay`, so a horizon stamped at the
    instant knot 0 falls due is replaced by the next one exactly as it becomes
    current: the JTC spends every cycle in `Trajectory::sample`'s
    before-the-first-point branch, interpolating from the *measured* state with
    `effort` forced to zero. The command then carries the machine's own
    velocity back at cycle rate and only a ramped fraction of `u`.

    The dead time belongs in `time_from_start`, with the command already in
    flight standing at zero.
    """
    from builtin_interfaces.msg import Time

    lead = 0.06
    in_flight = ramp([0.0, 0.06], slope=0.2)
    in_flight.u[:, 0] = 0.5
    horizon = ramp([0.0, 0.06], slope=0.3)
    horizon.u[:, 0] = 0.9

    message = horizon_to_message(
        horizon, CANONICAL, Time(sec=10, nanosec=0), lead=lead, in_flight=in_flight
    )

    # Every point is valid at or before the stamp's own instant.
    assert message.points[0].time_from_start == Duration(sec=0, nanosec=0)
    assert len(message.points) == len(horizon) + 1
    # Knot 0 still takes effect one dead time after the stamp, as it did before.
    assert message.points[1].time_from_start == Duration(sec=0, nanosec=60000000)
    # Point zero is the command the machine is already running, not a zero-effort
    # blend of its own measurement: `u - dq_a_ref` off the previous horizon.
    assert message.points[0].effort[0] == pytest.approx(0.5 - 0.2)
    assert message.points[1].effort[0] == pytest.approx(0.9 - 0.3)


def test_without_a_lead_the_wire_form_is_unchanged():
    """No dead time, no prepended knot -- the shadow and offline paths."""
    from builtin_interfaces.msg import Time

    horizon = ramp([0.0, 0.06])
    message = horizon_to_message(horizon, CANONICAL, Time(sec=1, nanosec=0))
    assert len(message.points) == len(horizon)
    assert message.points[0].time_from_start == Duration(sec=0, nanosec=0)
