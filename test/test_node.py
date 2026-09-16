"""The node's cycle: what it publishes, and what it refuses to publish."""

import math

import pytest
import rclpy
from crane_model import canonical_joints
from crane_model import symbolic as cs
from crane_mpc import problem
from crane_mpc.node import MpcNode
from crane_msgs.msg import SolverHealth
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

NAMES = canonical_joints()
ACTUATED = [NAMES[row] for row in cs.K_ACTUATED_ROWS]
PASSIVE = [NAMES[row] for row in cs.K_PASSIVE_ROWS]
POSE = [0.0, 0.5, 1.0, 0.2, 0.0, 0.3]
# The tool hanging at `POSE`, not a pair of zeros. The tilt joint's URDF range is
# [0.785, 2.356] and it hangs at pi/2, so the zeros these used to be pinned the
# tool 0.785 rad outside its own stop, some 90 degrees off hanging, with the sway
# box centred there. Nothing refuses that -- `lbx = ubx = x0` at stage 0 takes it
# -- it just made the feedback QP hard: 48 of its 50 iterations against 13 from
# here, so every assertion in this file sat two iterations from the cap and
# `weights.q_u` looked like what decided it (issue 137). Tip is
# `pi/2 - q_boom - q_arm`; tilt does not move.
PASSIVE_POSE = [0.5 * math.pi - POSE[1] - POSE[2], 0.5 * math.pi]


@pytest.fixture(scope="module")
def context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(context):
    node = MpcNode()
    # These tests are about what the node publishes, not about how fast this
    # container is: a budget overrun is a real verdict and it is tested for
    # separately rather than left to decide every other assertion.
    node.set_parameters([Parameter("solve_budget", Parameter.Type.DOUBLE, 10.0)])
    published = {}
    for name in (
        "_horizon_publisher",
        "_shadow_horizon_publisher",
        "_health_publisher",
    ):
        publisher = getattr(node, name)
        publisher.publish = lambda message, name=name: published.setdefault(
            name, []
        ).append(message)
    node.published = published
    yield node
    node.destroy_node()


def joint_state(node, position=POSE, velocity=None, permuted=False):
    message = JointState()
    message.header.stamp = node.get_clock().now().to_msg()
    names = list(ACTUATED) + list(PASSIVE)
    values = list(position) + list(PASSIVE_POSE)
    rates = list(velocity if velocity is not None else [0.0] * 6) + [0.0, 0.0]
    if permuted:
        order = [7, 2, 0, 5, 6, 1, 4, 3]
        names = [names[row] for row in order]
        values = [values[row] for row in order]
        rates = [rates[row] for row in order]
    message.name = names
    message.position = values
    message.velocity = rates
    return message


def reference(node, target=POSE):
    message = JointTrajectory()
    message.header.stamp = node.get_clock().now().to_msg()
    message.joint_names = list(ACTUATED)
    for index in range(2):
        point = JointTrajectoryPoint()
        point.positions = list(target)
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 4 * index
        message.points.append(point)
    return message


def configured(node):
    node.on_robot_description(String(data=problem.default_description().read_text()))
    node.on_reference(reference(node))
    node.on_joint_state(joint_state(node))
    return node


def test_without_a_description_nothing_is_published(node):
    node.update()
    assert node.published == {}
    assert "no robot description" in node._last_silence


def test_without_a_measured_state_nothing_is_published(node):
    node.on_robot_description(String(data=problem.default_description().read_text()))
    node.on_reference(reference(node))
    node.update()
    assert node.published == {}
    assert "no measured state" in node._last_silence


def test_in_shadow_the_horizon_goes_to_the_private_topic(node):
    configured(node)
    node.update()
    assert "_horizon_publisher" not in node.published
    horizons = node.published["_shadow_horizon_publisher"]
    assert len(horizons) == 1
    horizon = horizons[0]
    # The canonical eight, not the actuated six: the JTC rejects a six-name
    # trajectory whole under `allow_partial_joints_goal: false`.
    assert list(horizon.joint_names) == list(NAMES)
    assert len(horizon.points) == node.grid.horizon_length
    # The plan starts where the machine is, not where the last cycle left off.
    assert horizon.points[0].positions[1] == pytest.approx(POSE[1], abs=0.05)
    # The two passive rows carry the solved sway, not a pair of zeros.
    assert [horizon.points[0].positions[row] for row in cs.K_PASSIVE_ROWS] == (
        pytest.approx(PASSIVE_POSE, abs=0.05)
    )
    health = node.published["_health_publisher"][0]
    assert health.outcome in (
        SolverHealth.SOLVE_CONVERGED,
        SolverHealth.SOLVE_BUDGET_EXCEEDED,
    )
    assert health.joint_names == ACTUATED


def test_the_two_status_streams_are_stamped_with_the_cycle_they_share(node):
    """One instant per cycle: a panel reading either cannot disagree with the other."""
    settled = []
    node._settled_publisher.publish = settled.append
    configured(node)
    node.update()
    health = node.published["_health_publisher"][0]
    assert len(settled) == 1
    assert settled[0].header.stamp == health.header.stamp


def test_the_measurement_is_keyed_by_name_and_not_by_index(node):
    configured(node)
    node.on_joint_state(joint_state(node, permuted=True))
    node.update()
    horizon = node.published["_shadow_horizon_publisher"][0]
    assert horizon.points[0].positions[1] == pytest.approx(POSE[1], abs=0.05)
    # Canonical row 7 is the tool -- the pinned axis -- and not actuated row 5.
    assert horizon.points[0].positions[7] == pytest.approx(POSE[5])


def test_a_stale_measurement_stops_the_publisher(node):
    configured(node)
    stale = joint_state(node)
    stale.header.stamp.sec -= 5
    node.on_joint_state(stale)
    node.update()
    assert node.published == {}
    assert "max_state_age" in node._last_silence


def test_the_active_mode_publishes_on_the_contract_topic(node):
    node.set_parameters([Parameter("mode", Parameter.Type.STRING, "active")])
    configured(node)
    node.update()
    assert node.mode == "active"
    assert "_shadow_horizon_publisher" not in node.published
    assert len(node.published["_horizon_publisher"]) == 1


def test_a_non_finite_measurement_is_not_written_through(node):
    """
    It is a stale sensor, not a solver fault: dropped here the node keeps the
    last finite pose and the stamp stops advancing, so `max_state_age` is what
    speaks. Written through, the NaN reaches `x0` and comes back as a solve
    failure counted against `max_consecutive_failures`.
    """
    configured(node)
    fresh = node._actuated_stamp
    broken = joint_state(node)
    broken.position[2] = float("nan")
    node.on_joint_state(broken)

    assert node._q_a[2] == POSE[2]
    assert node._actuated_stamp == fresh
    # The passive rows of that same message are finite and are taken: the two
    # groups are read and stamped apart, so one bad axis does not stop the sway
    # measurement.
    assert node._passive_stamp == Time.from_msg(broken.header.stamp)
