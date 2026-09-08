"""The node's cycle: what it publishes, and what it refuses to publish."""

import pytest
import rclpy
from crane_model import canonical_joints
from crane_model import symbolic as cs
from crane_mpc import problem
from crane_mpc.node import MpcNode
from crane_msgs.msg import SolverHealth
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

NAMES = canonical_joints()
ACTUATED = [NAMES[row] for row in cs.K_ACTUATED_ROWS]
PASSIVE = [NAMES[row] for row in cs.K_PASSIVE_ROWS]
POSE = [0.0, 0.5, 1.0, 0.2, 0.0, 0.3]


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
    values = list(position) + [0.0, 0.0]
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
    assert list(horizon.joint_names) == ACTUATED
    assert len(horizon.points) == node.grid.horizon_length
    # The plan starts where the machine is, not where the last cycle left off.
    assert horizon.points[0].positions[1] == pytest.approx(POSE[1], abs=0.05)
    health = node.published["_health_publisher"][0]
    assert health.outcome in (
        SolverHealth.SOLVE_CONVERGED,
        SolverHealth.SOLVE_BUDGET_EXCEEDED,
    )
    assert health.joint_names == ACTUATED


def test_the_measurement_is_keyed_by_name_and_not_by_index(node):
    configured(node)
    node.on_joint_state(joint_state(node, permuted=True))
    node.update()
    horizon = node.published["_shadow_horizon_publisher"][0]
    assert horizon.points[0].positions[1] == pytest.approx(POSE[1], abs=0.05)
    assert horizon.points[0].positions[5] == pytest.approx(POSE[5])


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
