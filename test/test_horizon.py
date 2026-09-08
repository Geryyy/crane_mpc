"""The reference on the wire, resampled onto the OCP's grid."""

import numpy as np
import pytest
from builtin_interfaces.msg import Duration
from crane_mpc.horizon import (
    Grid,
    Knots,
    Rejection,
    horizon_to_message,
    reference_from_message,
    resample,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = ["sw", "ha", "ka", "sa", "ro", "gr"]


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
    # The last knot lands on the plan's own end, which is the held goal at rest
    # rather than an interpolation.
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

    message = horizon_to_message(horizon, JOINTS, Time(sec=3, nanosec=0))
    assert message.joint_names == JOINTS
    assert message.header.stamp.sec == 3
    assert len(message.points) == 3
    assert message.points[2].time_from_start == Duration(sec=0, nanosec=80000000)
    assert message.points[1].positions[0] == pytest.approx(0.3 * 0.04)
    assert list(message.points[0].accelerations) == []
