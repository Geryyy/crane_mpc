"""The node's cycle: what it publishes, and what it refuses to publish."""

import math

import pytest
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from control_msgs.msg import JointTrajectoryControllerState
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
# Tool hangs at `POSE`, not a pair of zeros: tilt's URDF range is [0.785,
# 2.356] and it hangs at pi/2, so zeros pinned it 0.785 rad outside its own
# stop. Not rejected by the solver, but made the feedback QP hard (48/50
# iterations vs 13 here) and looked like a weights.q_u issue (issue 137). Tip
# is `pi/2 - q_boom - q_arm`; tilt does not move.
PASSIVE_POSE = [0.5 * math.pi - POSE[1] - POSE[2], 0.5 * math.pi]


@pytest.fixture(scope="module")
def context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(context):
    node = MpcNode()
    # Budget overrun is tested separately, not left to decide every other assertion.
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
    # Canonical eight, not actuated six: JTC rejects a six-name trajectory
    # under `allow_partial_joints_goal: false`.
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
    Stale sensor, not a solver fault: dropped here, `max_state_age` speaks via
    a frozen stamp. Written through, NaN reaches `x0` and counts against
    `max_consecutive_failures`.
    """
    configured(node)
    fresh = node._actuated_stamp
    broken = joint_state(node)
    broken.position[2] = float("nan")
    node.on_joint_state(broken)

    assert node._q_a[2] == POSE[2]
    assert node._actuated_stamp == fresh
    # Passive rows are finite and taken: groups read/stamped apart, so one
    # bad axis doesn't stop sway.
    assert node._passive_stamp == Time.from_msg(broken.header.stamp)


def controller_state(node, names, velocities):
    message = JointTrajectoryControllerState()
    message.header.stamp = node.get_clock().now().to_msg()
    message.joint_names = list(names)
    message.output.velocities = list(velocities)
    return message


def test_a_shared_controller_state_topic_keeps_only_the_actuated_one(node):
    """The grasping controller shares the topic; its state is not the follower.

    Taking the last message when interleaved would leave `follower_command`
    naming none of the six every other cycle.
    """
    configured(node)
    driving = controller_state(node, ACTUATED, [0.1] * 6)
    node.on_controller_state(driving)
    node.on_controller_state(controller_state(node, PASSIVE, [0.0, 0.0]))
    assert node._controller_state is driving

    follower = node.follower_command(node.get_clock().now().nanoseconds)
    assert follower.source == "output.velocities"
    assert follower.velocity[0] == pytest.approx(0.1)


def gated(node, action="/trajectory_controller_a2b/follow_joint_trajectory"):
    """
    Put a configured node behind the start gate.

    Set rather than passed: `start_signal_action` is read_only, read once in
    the constructor, so a fixture cannot move it afterwards.
    """
    configured(node)
    node._start_signal_action = action
    node._start_signal_topic = f"{action}/_action/status"
    node._start_signal_open = False
    # Stand-in for the controller's status publisher; gate closes when it
    # goes away, so tests need one to exist.
    node._status_publisher = node.create_publisher(
        GoalStatusArray, node._start_signal_topic, 1
    )
    return node


def goal_status(*statuses):
    message = GoalStatusArray()
    for status in statuses:
        entry = GoalStatus()
        entry.status = status
        message.status_list.append(entry)
    return message


def test_a_gated_node_publishes_nothing_until_a_goal_is_accepted(node):
    one = gated(node)
    one.update()
    assert one.published == {}
    assert "no goal is live" in one._last_silence

    one.on_start_signal(goal_status(GoalStatus.STATUS_EXECUTING))
    one.update()
    assert len(one.published["_shadow_horizon_publisher"]) == 1


def test_a_finished_goal_closes_the_gate_again(node):
    one = gated(node)
    one.on_start_signal(goal_status(GoalStatus.STATUS_ACCEPTED))
    one.update()
    assert "_shadow_horizon_publisher" in one.published

    one.on_start_signal(goal_status(GoalStatus.STATUS_SUCCEEDED))
    published = len(one.published["_shadow_horizon_publisher"])
    one.update()
    assert len(one.published["_shadow_horizon_publisher"]) == published


def test_a_held_plan_is_not_spent_while_the_gate_is_closed(node):
    """
    Reference waits for a human, so it may not age while it waits.

    `max_reference_age` counts from a plan that has begun spending; 30 gated
    cycles = 1.2 s against a 0.5 s bound and a 4 s plan, so a gate running
    after `Cycle.gates` would anchor here and refuse the held plan.
    """
    one = gated(node)
    for _ in range(30):
        one.update()
    assert one.published == {}
    assert not one._cycle.reference_anchored

    one.on_start_signal(goal_status(GoalStatus.STATUS_EXECUTING))
    one.update()
    assert len(one.published["_shadow_horizon_publisher"]) == 1


def test_a_vanished_controller_closes_the_gate_it_had_opened(node):
    """
    A status topic reports goals, not liveness.

    A dying controller's last status stays cached, so an open gate closes via
    the graph going quiet, not a message -- else a JTC dying mid-goal leaves
    this node driving a controller that no longer exists.
    """
    one = gated(node)
    one.on_start_signal(goal_status(GoalStatus.STATUS_EXECUTING))
    assert one._start_signal_open

    # Graph's own publisher count. Set here, not by destroying the publisher
    # above: rmw updates it asynchronously, so waiting on discovery flakes.
    one.count_publishers = lambda topic: 0
    one.update()
    assert not one._start_signal_open
    assert one.published == {}
