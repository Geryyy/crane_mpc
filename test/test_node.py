"""The node's cycle: what it publishes, and what it refuses to publish."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Duration
from conftest import export_for, shipped_config, shipped_init_args
from control_msgs.msg import JointTrajectoryControllerState
from crane_model import canonical_joints, hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import config, problem
from crane_mpc.node import MpcNode, StartGate, jtc_rate_mismatch
from crane_msgs.msg import JointPath, SolverHealth
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
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
    rclpy.init(args=shipped_init_args())
    yield
    rclpy.shutdown()


class Ticks:
    """
    The node's clock, moved by hand: one `Ts` per `update`.

    Every age the node computes is a difference against `get_clock().now()`, so
    under real time a loaded host ages a measurement past `max_state_age`
    between the message and the cycle that reads it -- and which test fails then
    depends on the load, not on the code. The tick runs *after* `update`, so a
    message stamped just before a cycle is that cycle's, age zero.
    """

    def __init__(self, node):
        self.now_ns = 1_000_000_000_000
        step = int(node.Ts * 1e9)
        cycle = node.update
        node.get_clock = lambda: self

        def update():
            cycle()
            self.now_ns += step

        node.update = update

    def now(self) -> Time:
        return Time(nanoseconds=self.now_ns)


@pytest.fixture
def node(context, export_base):
    node = MpcNode()
    Ticks(node)
    # Every assertion below is on the configuration the repo ships: the
    # declaration carries no defaults, so `context` hands the node
    # `config/crane_mpc.yaml` and there is no second controller to run. The
    # export is built here and not in `configured` because several tests hand a
    # description over themselves.
    export_for(
        export_base,
        config.parameter_dict(node._values),
        hydraulic_limits(),
        problem.default_description().read_text(),
    )
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


def test_the_node_runs_on_the_shipped_values(node):
    """
    Not on a declaration default, because there are none (issue 183).

    Without this every assertion below could be measuring a controller that
    never ships -- which is what they did while the two yamls disagreed on
    `q_a`, `dq_u`, `du`, `tool`, `progress`, `terminal_scale` and `command_state`.
    """
    shipped = yaml.safe_load(shipped_config().read_text())["crane_mpc"][
        "ros__parameters"
    ]
    assert node.Ts == shipped["Ts"]
    assert node.grid.horizon_length == shipped["horizon_length"]
    assert node._values.command_state == shipped["command_state"]
    assert list(node._values.weights.q_a) == shipped["weights"]["q_a"]


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
    # Knot 0 at the stamp, every knot after it, plus the hold edges between.
    assert len(horizon.points) > node.grid.horizon_length
    assert horizon.points[0].time_from_start == Duration(sec=0, nanosec=0)
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
    node._gate = StartGate(action)
    node._gate.open = False
    # The graph's count for the controller's status publisher. Stubbed and not
    # published for real: rmw counts publishers asynchronously, so a real one
    # is a discovery race that drops the gate shut under host load.
    node.count_publishers = lambda topic: 1
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
    cycles = 1.8 s against a 0.5 s bound and a 4 s plan, so a gate running
    after `Cycle.gates` would anchor here and refuse the held plan. The joint
    state is fed each cycle because the machine keeps publishing while gated.
    """
    one = gated(node)
    for _ in range(30):
        one.on_joint_state(joint_state(one))
        one.update()
    assert one.published == {}
    assert not one._cycle.reference_anchored

    one.on_start_signal(goal_status(GoalStatus.STATUS_EXECUTING))
    one.on_joint_state(joint_state(one))
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
    assert one._gate.open

    # The graph going quiet, as `gated`'s stub reports it.
    one.count_publishers = lambda topic: 0
    one.update()
    assert not one._gate.open
    assert one.published == {}


def joint_path(stamp, target=POSE):
    """The planner's curve for `reference`'s move: a joint-space line to `target`."""
    message = JointPath()
    message.header.stamp = stamp
    message.joint_names = list(ACTUATED)[: cs.K_PLANNED_DOF]
    rows = np.linspace(POSE[: cs.K_PLANNED_DOF], target[: cs.K_PLANNED_DOF], 40)
    message.q_path = [float(value) for value in rows.reshape(-1)]
    message.duration = Duration(sec=4)
    return message


def test_the_planners_curve_replaces_the_window_fit(node):
    node.on_robot_description(String(data=problem.default_description().read_text()))
    plan = reference(node)
    node.on_reference(plan)
    node.on_joint_state(joint_state(node))
    # Nothing published geometry yet, so the OCP fits this cycle's own window.
    assert node._cycle.resolved_path() is None

    node.on_path(joint_path(plan.header.stamp))
    path = node._cycle.resolved_path()
    assert path is not None
    # Fitted once on arrival, not once per cycle inside the OCP, and it is the fit
    # that travels into the parameter vector.
    assert path.control.shape == (problem.PATH_POINTS, cs.K_PLANNED_DOF)
    # 1/duration: the pace the plan was written at, which is what the progress row
    # is priced against.
    assert path.nominal_rate == pytest.approx(0.25)
    node.update()
    assert node.published["_shadow_horizon_publisher"]


class Logs:
    """What the node said, by level. Any level, so nothing else has to be stubbed."""

    def __init__(self):
        self.lines = []

    def __getattr__(self, level):
        def say(text, **_):
            self.lines.append((level, text))

        return say


def test_which_branch_drives_the_position_reference_is_said_out_loud(node):
    """
    A curve that does not pair by stamp was dropped without a word (issue 162).

    From outside the node that read exactly like a followed curve, which is the
    mode a run is meant to be under test in.
    """
    logs = Logs()
    node.get_logger = lambda: logs
    node.on_robot_description(String(data=problem.default_description().read_text()))
    plan = reference(node)
    node.on_reference(plan)
    node.on_joint_state(joint_state(node))

    paired = Time.from_msg(plan.header.stamp).nanoseconds
    node.on_path(joint_path(Time(nanoseconds=paired + 1).to_msg()))
    node.update()
    assert any(
        f"{paired + 1} ns" in text and f"{paired} ns" in text for _, text in logs.lines
    )

    logs.lines.clear()
    node.on_path(joint_path(plan.header.stamp))
    node.update()
    assert any("the planner's curve" in text for _, text in logs.lines)
    # Once per change and not once per cycle: it is said in the 25 Hz callback.
    logs.lines.clear()
    node.update()
    assert not any("solving against" in text for _, text in logs.lines)


def test_a_cycle_late_end_to_end_is_not_published_as_converged(node):
    """acados' solve_time misses the sample's age; 100 ms old is past a 60 ms budget."""
    configured(node)
    node.set_parameters([Parameter("solve_budget", Parameter.Type.DOUBLE, 0.06)])
    old = joint_state(node)
    now = node.get_clock().now().nanoseconds
    old.header.stamp = Time(nanoseconds=now - 100_000_000).to_msg()
    node.on_joint_state(old)
    node.update()
    assert "_shadow_horizon_publisher" not in node.published
    health = node.published["_health_publisher"][0]
    assert health.outcome == SolverHealth.SOLVE_BUDGET_EXCEEDED
    assert "late cycle" in health.message


def rate(value):
    response = GetParameters.Response(values=[ParameterValue(integer_value=value)])
    return SimpleNamespace(result=lambda: response)


def test_a_jtc_off_the_assumed_tick_is_refused(node):
    assert jtc_rate_mismatch(0.01, 100, 0) == ""  # 0: the manager's rate
    assert jtc_rate_mismatch(0.01, 50, 100) == ""
    assert jtc_rate_mismatch(0.01, 100, 20)
    configured(node)
    node._rate_query.on_rate(0, rate(100))
    node._rate_query.on_rate(1, rate(20))
    node.update()
    assert node.published == {}
    assert "20 Hz" in node._last_silence
