"""
The settled verdict: two rates and a staleness flag, as one function.

Must not happen: a degraded estimate reading as settled -- so every
non-measurement path is asserted here, not just the two that are.
"""

import math

import pytest
import rclpy
from builtin_interfaces.msg import Time
from crane_model import canonical_joints
from crane_model import symbolic as cs
from crane_mpc import reports
from crane_mpc.node import MpcNode
from crane_msgs.msg import SwaySettled
from sensor_msgs.msg import JointState

ACTUATED = [canonical_joints()[row] for row in cs.K_ACTUATED_ROWS]
PASSIVE = [canonical_joints()[row] for row in cs.K_PASSIVE_ROWS]
MAX_STATE_AGE = 0.2
SETTLED = [0.04, 0.04]
STAMP = Time(sec=7)


def verdict(dq_u, age=0.01, max_state_age=MAX_STATE_AGE):
    return reports.sway_settled(dq_u, age, max_state_age, SETTLED, STAMP)


def test_rates_inside_the_threshold_are_settled():
    settled = verdict([0.01, -0.039])
    assert settled.settled == SwaySettled.SETTLED_YES
    assert list(settled.velocity) == [0.01, -0.039]


def test_one_row_over_the_threshold_is_enough_to_be_unsettled():
    settled = verdict([0.001, 0.05])
    assert settled.settled == SwaySettled.SETTLED_NO
    assert list(settled.velocity) == [0.001, 0.05]


@pytest.mark.parametrize(
    "rates,age",
    [
        ([0.0, 0.0], None),  # no rate has arrived; the zeros are not a reading
        ([0.0, 0.0], 0.5),  # stale by max_state_age
        ([float("nan"), 0.0], 0.01),  # non-finite
    ],
)
def test_a_degraded_estimate_is_unknown_and_never_settled(rates, age):
    settled = verdict(rates, age=age)
    assert settled.settled == SwaySettled.SETTLED_UNKNOWN
    assert all(math.isnan(rate) for rate in settled.velocity)
    assert settled.message


def test_a_changed_verdict_is_exempt_from_the_decimation():
    """What decimation exists not to drop: the cycle the load settles on."""
    unsettled = verdict([0.5, 0.0])
    settled = verdict([0.0, 0.0])
    assert reports.settled_is_due(2, 2, unsettled, settled)
    assert not reports.settled_is_due(2, 2, settled, settled)


@pytest.fixture
def node():
    rclpy.init()
    node = MpcNode()
    node.published = []
    node._settled_publisher.publish = node.published.append
    yield node
    node.destroy_node()
    rclpy.shutdown()


def joint_state(node, velocities=True):
    message = JointState()
    message.header.stamp = node.get_clock().now().to_msg()
    message.name = list(PASSIVE)
    message.position = [0.0, 0.5 * math.pi]
    if velocities:
        message.velocity = [0.0, 0.0]
    return message


def test_a_cycle_that_publishes_no_horizon_still_publishes_a_verdict(node):
    """The gates swallow the cycles the verdict is least allowed to skip."""
    node.update()
    assert "no robot description" in node._last_silence
    assert node.published[0].settled == SwaySettled.SETTLED_UNKNOWN
    node.update()
    assert len(node.published) == 1, "the second cycle is off cadence and unchanged"


def test_a_pose_without_a_rate_is_not_a_load_that_has_stopped_swinging(node):
    """
    `/joint_states` carries no validity flag and its velocity array is optional.

    Taking the pose's stamp for the rate's would publish this node's starting
    zeros as a measurement of a still machine, on the stream a grip gates on.
    """
    node._joints, node._passive_joints = list(ACTUATED), list(PASSIVE)
    node.on_joint_state(joint_state(node, velocities=False))
    node.update()
    assert node._passive_stamp is not None, "the pose was taken"
    assert node.published[0].settled == SwaySettled.SETTLED_UNKNOWN

    node.on_joint_state(joint_state(node, velocities=True))
    node.update()
    assert node.published[-1].settled == SwaySettled.SETTLED_YES
