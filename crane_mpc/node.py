"""
The `crane_mpc` node.

An adapter and nothing else: `cycle.py` decides, `reports.py` marshals. State
stays in the OCP's own `x` end to end, never rebuilt at a crossing.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from control_msgs.msg import JointTrajectoryControllerState
from crane_model import CraneModel, Payload, Tool, canonical_joints, hydraulic_limits
from crane_model.velocity_loop import load_velocity_loop
from crane_msgs.msg import JointPath, SolverHealth, SwaySettled
from crane_msgs.srv import SetPayload
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from . import config, problem, reports
from . import horizon as hz
from .cycle import (
    ACTUATED_DOF,
    ACTUATED_INDICES,
    CONTROLLER_STATE_TOPIC,
    HORIZON_TOPIC,
    JOINT_PATH_TOPIC,
    JOINT_STATE_TOPIC,
    NANOSECONDS,
    PASSIVE_DOF,
    PASSIVE_INDICES,
    PLANNED_DOF,
    REFERENCE_TOPIC,
    ROBOT_DESCRIPTION_TOPIC,
    SET_PAYLOAD_SERVICE,
    SHADOW_COMPARISON_TOPIC,
    SHADOW_HORIZON_TOPIC,
    SOLVER_HEALTH_TOPIC,
    SWAY_SETTLED_TOPIC,
    TCP_HORIZON_TOPIC,
    Cycle,
    FollowerCommand,
    Measurement,
    Silence,
)
from .parameters import crane_mpc as parameter_library
from .solver import Ocp, path_control

#: How often a repeated complaint is logged, ms.
WARN_PERIOD = 5.0
#: Seconds (one query each) to wait for the controllers' `update_rate`.
RATE_QUERIES = 10
CONTROLLER_MANAGER = "/controller_manager"


def jtc_rate_mismatch(assumed_period: float, manager_rate: int, jtc_rate: int) -> str:
    """Refusal text if the JTC does not tick at `assumed_period`, else `""`."""
    rate = jtc_rate or manager_rate  # a controller's 0 means the manager's
    if rate > 0 and abs(1.0 / rate - assumed_period) <= 1e-6:
        return ""
    return (
        f"the JTC ticks at {rate} Hz (update_rate {jtc_rate}, controller_manager "
        f"{manager_rate}) and the horizon is built for a {assumed_period} s tick "
        "from velocity_loop.yaml rate_hz; the feed-forward would be sampled off-tick"
    )


class StartGate:
    """
    Does the action this node is gated on carry a goal. No action: always open.

    A status topic reports goals, not liveness, so a vanished publisher closes
    the gate rather than leaving this node driving a controller that is gone.
    """

    LIVE = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)

    def __init__(self, action: str) -> None:
        self.action = action
        self.topic = f"{action}/_action/status" if action else ""
        self.open = not action

    def adopt(self, message: GoalStatusArray) -> bool:
        """Read one status array; `True` if the gate moved."""
        was_open = self.open
        self.open = any(entry.status in self.LIVE for entry in message.status_list)
        return self.open != was_open

    def closed_with_its_publisher(self, count_publishers) -> bool:
        """Close an open gate whose status publisher has gone; `True` if it moved."""
        if not self.open or not self.topic or count_publishers(self.topic):
            return False
        self.open = False
        return True


class RateQuery:
    """
    Ask the controller manager and the JTC for `update_rate`, once.

    One query a second for `RATE_QUERIES` s: the controllers come up after this
    node does. Unanswered, the `velocity_loop.yaml` tick stands unchecked; a
    mismatch goes to `refuse`, so the node owns the refusal.
    """

    def __init__(self, node: Node, jtc_name: str, period: float, refuse) -> None:
        self._node = node
        self._jtc_name = jtc_name
        self._assumed_period = period
        self._refuse = refuse
        names = (CONTROLLER_MANAGER, f"/{jtc_name}")
        self._clients = [AsyncParameterClient(node, name) for name in names]
        self._rates: list = [None, None]
        self._queries = 0
        self._timer = node.create_timer(1.0, self.query)

    def query(self) -> None:
        self._queries += 1
        if self._queries > RATE_QUERIES:
            self._timer.cancel()
            self._node.get_logger().warn(
                f"no update_rate from {CONTROLLER_MANAGER} or /{self._jtc_name} in "
                f"{RATE_QUERIES} s; the JTC is assumed to tick every "
                f"{self._assumed_period} s (velocity_loop.yaml), unchecked."
            )
            return
        for slot, client in enumerate(self._clients):
            if self._rates[slot] is None and client.services_are_ready():
                client.get_parameters(
                    ["update_rate"],
                    callback=lambda future, slot=slot: self.on_rate(slot, future),
                )

    def on_rate(self, slot: int, future) -> None:
        """Adopt one answer; once both are in, refuse a tick off the horizon's."""
        response = future.result()
        if response is None or not response.values:
            return
        self._rates[slot] = int(response.values[0].integer_value)
        if None in self._rates or self._timer.is_canceled():
            return
        self._timer.cancel()
        why = jtc_rate_mismatch(self._assumed_period, *self._rates)
        if why:
            self._refuse(why)


def _qos(durability=DurabilityPolicy.VOLATILE) -> QoSProfile:
    return QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=durability
    )


def _latched() -> QoSProfile:
    return _qos(DurabilityPolicy.TRANSIENT_LOCAL)


def _measured(message: JointState, index: dict, names: list):
    """
    `(q, dq)` for `names`, or `None` if the message doesn't carry them finitely.

    `dq` is `None` where the message carries no velocity (last one stands). A
    non-finite row is dropped, not written through: it surfaces as staleness via
    `max_state_age` instead of a solve failure.
    """
    rows = [index.get(name) for name in names]
    if any(row is None for row in rows) or len(message.position) <= max(rows):
        return None
    position = np.array([message.position[row] for row in rows])
    velocity = (
        np.array([message.velocity[row] for row in rows])
        if len(message.velocity) > max(rows)
        else None
    )
    if not np.all(np.isfinite(position)):
        return None
    if velocity is not None and not np.all(np.isfinite(velocity)):
        return None
    return position, velocity


class MpcNode(Node):
    def __init__(self) -> None:
        super().__init__("crane_mpc")
        self._listener = parameter_library.ParamListener(self)
        self._values = self._listener.get_params()

        self.Ts = float(self._values.Ts)
        self.grid = hz.Grid(self.Ts, int(self._values.horizon_length))
        self.delay = float(self._values.sensor_to_valve_delay)
        self._cycle = Cycle(
            self.Ts,
            self.grid,
            str(self._values.mode),
            self.delay,
            list(self._values.dq_a_feedback),
            list(self._values.dq_a_divergence_max),
        )

        self._ocp: Ocp | None = None
        self._model: CraneModel | None = None
        self._description: str | None = None
        self._configuration_failure = ""
        #: The canonical eight, in contract order: what the horizon is named by.
        self._canonical_joints = list(canonical_joints())
        #: The two joint groups, empty until the description configures this node.
        self._joints: list[str] = []
        self._passive_joints: list[str] = []

        # By name, never index -- the two stacks publish different `/joint_states` sets.
        self._q_a = np.zeros(ACTUATED_DOF)
        self._dq_a = np.zeros(ACTUATED_DOF)
        self._q_u = np.zeros(PASSIVE_DOF)
        self._dq_u = np.zeros(PASSIVE_DOF)
        self._actuated_stamp: Time | None = None
        self._passive_stamp: Time | None = None
        #: When a passive *velocity* (not pose) was last written; settled verdict ages against this.
        self._passive_velocity_stamp: Time | None = None

        #: The JTC's tick; it samples feedforward one tick ahead (`hz._command_message`).
        self._jtc_period = 1.0 / load_velocity_loop()[1]
        #: `(stamp_ns, planned-axis output)` from the JTC, oldest first.
        self._jtc_outputs: deque = deque(maxlen=64)

        self._reference_message: JointTrajectory | None = None
        self._path_message: JointPath | None = None
        self._payload = Payload()
        self._controller_state: JointTrajectoryControllerState | None = None
        self._last_health: SolverHealth | None = None
        self._last_settled: SwaySettled | None = None
        #: Last `Cycle.path_source` said out loud; said again only when it changes.
        self._last_path_source: str | None = None
        #: One instant per cycle, shared by every report that cycle writes.
        self._stamp = self.get_clock().now().to_msg()

        #: The action whose accepted goal releases this node; no action, no gate.
        self._gate = StartGate(str(self._values.start_signal_action))
        self._rate_query: RateQuery | None = None

        self._create_publishers()
        self._create_subscriptions()
        self.create_timer(self.Ts, self.update)

        self.get_logger().info(
            f"crane_mpc in {self.mode} mode: {self.grid.horizon_length} knots at "
            f"{self.Ts} s, {self.delay} s of transport delay, a "
            f"{self._values.solve_budget} s budget."
        )
        self.warn_open_loop_axes()

    @property
    def mode(self) -> str:
        return self._cycle.mode

    def warn_open_loop_axes(self) -> None:
        """Say once, at startup, which axes do not close the velocity loop."""
        open_loop = [
            self._canonical_joints[ACTUATED_INDICES[axis]]
            for axis in range(PLANNED_DOF)
            if not self._cycle.dq_a_feedback[axis]
        ]
        if not open_loop:
            return
        self.get_logger().warn(
            f"{', '.join(open_loop)} take dq in x_0 from the model, not from "
            f"{JOINT_STATE_TOPIC} (dq_a_feedback); the state can walk away from the "
            f"machine. dq_a_divergence_max watches it, on {SHADOW_COMPARISON_TOPIC}, "
            "and nothing corrects it."
        )

    def say_path_source(self) -> None:
        """
        Say what this cycle is solved against, whenever that changes.

        Whether the curve is followed was only knowable by reading source: a curve
        that does not pair by stamp is dropped without a word, and reads from
        outside exactly like one that drove. Info and not warn: every re-plan has a
        window where the new reference is in hand and its curve is not, so a warn
        there would fire on every healthy move. What is solved against, not what
        went out -- the shift rung publishes the previous horizon.
        """
        source = self._cycle.path_source()
        if source == self._last_path_source:
            return
        self._last_path_source = source
        self.get_logger().info(f"crane_mpc is solving against {source}.")

    def warn_divergence(self) -> None:
        """Report a carried velocity that has walked away from the measured one."""
        carry = self._cycle.velocity_carry
        if not carry.any_diverged:
            return
        for axis, joint in enumerate(self._joints):
            if not carry.diverged[axis]:
                continue
            # Own call site: rclpy throttles by caller, not shared with `self.warn`.
            self.get_logger().warn(
                f"the model-carried velocity of {joint} has walked "
                f"{carry.divergence[axis]:g} from the measured one, past "
                f"dq_a_divergence_max {self._cycle.dq_a_divergence_max[axis]:g}; this "
                "axis runs open-loop (dq_a_feedback), so it is reported, not corrected.",
                throttle_duration_sec=WARN_PERIOD,
            )

    @property
    def _last_silence(self) -> str:
        return self._cycle.last_silence

    def _create_publishers(self) -> None:
        self._horizon_publisher = self.create_publisher(
            JointTrajectory, HORIZON_TOPIC, _qos()
        )
        self._shadow_horizon_publisher = self.create_publisher(
            JointTrajectory, SHADOW_HORIZON_TOPIC, _qos()
        )
        self._tcp_publisher = self.create_publisher(Path, TCP_HORIZON_TOPIC, _qos())
        self._health_publisher = self.create_publisher(
            SolverHealth, SOLVER_HEALTH_TOPIC, _qos()
        )
        self._settled_publisher = self.create_publisher(
            SwaySettled, SWAY_SETTLED_TOPIC, _qos()
        )
        self._comparison_publisher = self.create_publisher(
            DiagnosticArray, SHADOW_COMPARISON_TOPIC, _qos()
        )

    def _create_subscriptions(self) -> None:
        self.create_subscription(
            JointTrajectoryControllerState,
            CONTROLLER_STATE_TOPIC,
            self.on_controller_state,
            _qos(),
        )
        self.create_subscription(
            JointTrajectory, REFERENCE_TOPIC, self.on_reference, _latched()
        )
        self.create_subscription(JointPath, JOINT_PATH_TOPIC, self.on_path, _latched())
        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self.on_robot_description, _latched()
        )
        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self.on_joint_state, _qos()
        )
        self.create_service(SetPayload, SET_PAYLOAD_SERVICE, self.on_set_payload)
        if self._gate.topic:
            # Status topic, not the action -- latched+reliable so a late-starting
            # node still sees an accepted goal.
            self.create_subscription(
                GoalStatusArray, self._gate.topic, self.on_start_signal, _latched()
            )

    # -- what arrives ------------------------------------------------------------

    def on_robot_description(self, message: String) -> None:
        if self._description is not None:
            return
        self._description = message.data
        self.configure()

    def on_joint_state(self, message: JointState) -> None:
        if not self._joints:
            return
        index = {name: row for row, name in enumerate(message.name)}
        stamp = Time.from_msg(message.header.stamp)
        measured = _measured(message, index, self._joints)
        if measured is not None:
            self._q_a, velocity = measured
            if velocity is not None:
                self._dq_a = velocity
            self._actuated_stamp = stamp
        measured = _measured(message, index, self._passive_joints)
        if measured is not None:
            self._q_u, velocity = measured
            if velocity is not None:
                self._dq_u = velocity
                self._passive_velocity_stamp = stamp
            self._passive_stamp = stamp

    def on_reference(self, message: JointTrajectory) -> None:
        self._reference_message = message
        self.adopt_reference()

    def on_path(self, message: JointPath) -> None:
        self._path_message = message
        self.adopt_path()

    def on_start_signal(self, message: GoalStatusArray) -> None:
        """Is a goal live on the action this node is gated on."""
        if not self._gate.adopt(message):
            return
        drives = (
            "accepted a goal, so this node drives"
            if self._gate.open
            else "has no goal left, so this node stops driving"
        )
        self.get_logger().info(f"{self._gate.action} {drives}.")

    def on_controller_state(self, message: JointTrajectoryControllerState) -> None:
        # Timber bringup profiles put both trajectory controllers on one topic;
        # without this, `follower_command` reads a message naming no actuated
        # joint every other cycle. Names, not indices.
        if not set(self._joints) & set(message.joint_names):
            return
        self._controller_state = message
        names = list(message.joint_names)
        output = message.output.velocities
        rows = [names.index(j) for j in self._joints[:PLANNED_DOF] if j in names]
        if len(rows) == PLANNED_DOF and max(rows) < len(output):
            velocity = np.array([output[row] for row in rows])
            if np.all(np.isfinite(velocity)):
                stamp = Time.from_msg(message.header.stamp).nanoseconds
                self._jtc_outputs.append((stamp, velocity))

    # -- configuration -----------------------------------------------------------

    def configure(self) -> None:
        """Build the model and the problem once the description has arrived."""
        if self._description is None or self._ocp is not None:
            return
        try:
            self._model = CraneModel(self._description, Tool(problem.TOOL))
            names = self._canonical_joints
            self._joints = [names[row] for row in ACTUATED_INDICES]
            self._passive_joints = [names[row] for row in PASSIVE_INDICES]
            # `Ocp.__init__` runs `config.check_settings`, which is where
            # `delay == Ts` is refused -- inside this `try`, so it is a reported
            # configuration failure and not a dead process in `MpcNode.__init__`.
            self._ocp = Ocp(
                problem.default_description().read_text(),
                config.parameter_dict(self._values),
                hydraulic_limits(),
            )
        except Exception as error:  # the description decides whether this node runs
            self._configuration_failure = str(error)
            self.get_logger().error(f"The MPC could not be configured: {error}")
            return
        self._cycle.ocp = self._ocp
        self.get_logger().info(
            f"crane_mpc configured on {self._model.tool.value}: joints "
            f"{', '.join(self._joints)}, sway {', '.join(self._passive_joints)}."
        )
        # `CRANE_MPC_OCP_OPTIONS` is inherited from the environment and changes
        # which solver is compiled, so a variant left over from a sweep must
        # not reach a machine unlogged.
        patched = {
            name: value
            for name, value in problem.solver_tuning().items()
            if value != problem.SOLVER_TUNING[name]
        }
        if patched:
            self.get_logger().warning(
                f"the acados backend is patched by {problem.TUNING_ENV}: {patched} -- "
                "a swept solver, not the one this package ships."
            )
        self.adopt_reference()
        self.adopt_path()
        self._rate_query = RateQuery(
            self, str(self._values.jtc_name), self._jtc_period, self.refuse
        )

    def refuse(self, why: str) -> None:
        """Stop driving on a configuration this node cannot honour."""
        self._configuration_failure = why
        self.get_logger().error(f"the MPC refuses its configuration: {why}.")

    def ready(self) -> bool:
        return (
            self._ocp is not None
            and self._model is not None
            and not self._configuration_failure
        )

    def adopt_reference(self) -> None:
        if self._reference_message is None or not self.ready():
            return
        reference, why = hz.reference_from_message(
            self._reference_message, self._joints
        )
        if reference is None:
            self.get_logger().warn(
                f"the reference on {REFERENCE_TOPIC} was refused, the previous one "
                f"stands: {why}."
            )
            self._reference_message = None
            return
        self._cycle.adopt_reference(
            reference, Time.from_msg(self._reference_message.header.stamp).nanoseconds
        )
        self._reference_message = None

    def adopt_path(self) -> None:
        """
        Fit the planner's curve once, here, rather than once per cycle in the OCP.

        Refused, the node keeps whatever curve it had and `Cycle.resolved_path`
        pairs by stamp, so a refusal cannot silently attach the last plan's
        geometry to this one -- it falls back to the window fit instead.
        """
        if self._path_message is None or not self.ready():
            return
        samples, duration, why = hz.path_from_message(
            self._path_message, self._joints[:PLANNED_DOF]
        )
        if samples is None:
            self.get_logger().warn(
                f"the path on {JOINT_PATH_TOPIC} was refused, so the horizon's own "
                f"knots are the path this node follows: {why}."
            )
            self._path_message = None
            return
        self._cycle.adopt_path(
            path_control(samples),
            duration,
            Time.from_msg(self._path_message.header.stamp).nanoseconds,
        )
        self._path_message = None

    def adopt_mode(self, requested: str) -> None:
        previous = self._cycle.adopt_mode(requested)
        self.get_logger().info(
            f"crane_mpc moved from {previous} to {requested}; a mode change is this "
            "node's reactivation, so the next solve cold starts."
        )

    def refresh_parameters(self) -> None:
        if not self._listener.is_old(self._values):
            return
        self._values = self._listener.get_params()
        if self._ocp is not None:
            self._ocp.solve_budget_s = float(self._values.solve_budget)
        if str(self._values.mode) != self.mode:
            self.adopt_mode(str(self._values.mode))

    # -- the cycle ---------------------------------------------------------------

    def update(self) -> None:
        self.refresh_parameters()
        cycle = self._cycle
        cycle.begin()
        clock = self.get_clock().now()
        now = clock.nanoseconds
        self._stamp = clock.to_msg()
        # Before every gate: measures the machine, not the command path.
        self.report_sway_settled(now)

        if not self.ready():
            why = (
                "no robot description has arrived"
                if self._description is None
                else (self._configuration_failure or "configuration is incomplete")
            )
            self.fall_silent(self.unconfigured(why))
            return

        # Before `cycle.gates` deliberately: it anchors the reference, so a held
        # plan still starts at its first knot whenever the goal comes.
        if not self.start_signal_still_open():
            self.fall_silent(self.ungated())
            return

        # The sample's own instant, not the tick: it lags 0-10 ms plus transport.
        stamp = self._actuated_stamp
        measured = now if stamp is None else min(stamp.nanoseconds, now)
        silence = cycle.gates(
            now,
            float(self._values.max_clock_skew),
            float(self._values.max_reference_age),
            measured,
        )
        if silence is None:
            silence = cycle.read_state(
                self.measurement(now), float(self._values.max_state_age)
            )
        if silence is None:
            # u^+ is crane_model's, resolved when the problem was posed -- not a
            # parameter, so the only copy is the one the solver was configured on
            cycle.adopt_follower(
                self.follower_command(now), self._ocp.parameters["limits"]["u_max"]
            )
            cycle.jtc_output = (
                self.jtc_output(measured) if self.mode == "active" else None
            )
            silence = cycle.propagate()
        if silence is not None:
            self.fall_silent(silence)
            return
        self.say_path_source()
        self.warn_divergence()

        solution, refusal = cycle.solve(
            lambda: (self.get_clock().now().nanoseconds - measured) / 1e9
        )
        if refusal:
            self.report(None, refusal)
            self.warn(f"The OCP refused its arguments: {refusal}.")
            return

        verdict = cycle.ladder(
            solution,
            int(self._values.max_consecutive_failures),
            float(self._ocp.solve_budget_s),
        )
        if not verdict.published:
            self.report(solution, verdict.text)
            self.complain(verdict)
            return

        self.complain(verdict)
        self.publish_horizon()
        was_stalled = cycle.progress_stalled
        cycle.advance(
            solution,
            now,
            float(self._values.min_progress_rate),
            float(self._values.max_stall_time),
        )
        # After `advance`, which moves the progress and the cadence anchor the
        # next cycle resamples on; and after the horizon went out, so the
        # linearisation runs in the part of the cycle nobody is waiting on.
        cycle.prepare_next(solution)
        self.report(solution, self.say_stalled(verdict.text, was_stalled))

    def say_stalled(self, text: str, was_stalled: bool) -> str:
        """Add the stall to the verdict, and say it once on the transition."""
        if not self._cycle.progress_stalled:
            return text
        text += (
            "; and the plan is stalled, not merely slow: virtual time advanced by "
            f"less than {self._values.min_progress_rate} of nominal for "
            f"{self._values.max_stall_time} s of wall clock"
        )
        if not was_stalled:
            self.get_logger().error(
                f"{text}. The horizon still goes out, but {SOLVER_HEALTH_TOPIC} now "
                "carries FAULT_SOLVER; this node reports the stall, it does not "
                "recover from it."
            )
        return text

    def complain(self, verdict) -> None:
        if verdict.severity == "error":
            self.get_logger().error(verdict.text)
        elif verdict.severity:
            # Own call site: sharing `self.warn`'s bucket used to swallow the
            # ladder's rungs below the escalation.
            self.get_logger().warn(verdict.text, throttle_duration_sec=WARN_PERIOD)

    def unconfigured(self, why: str) -> Silence:
        return Silence(
            why,
            f"the MPC is not configured ({why}), so nothing goes out on "
            f"{HORIZON_TOPIC}; it needs one latched {ROBOT_DESCRIPTION_TOPIC}.",
        )

    def start_signal_still_open(self) -> bool:
        """Read the gate; its publisher having gone closes it, fail-closed."""
        gate = self._gate
        if gate.closed_with_its_publisher(self.count_publishers):
            self.get_logger().warn(
                f"{gate.action} has no status publisher left, so its goal cannot "
                "still be live and this node stops driving."
            )
        return gate.open

    def ungated(self) -> Silence:
        return Silence(
            f"no goal is live on {self._gate.action}",
            f"{self._gate.action} carries no goal, so nothing goes out on "
            f"{HORIZON_TOPIC}; the reference is held unanchored (start_signal_action) "
            "until the controller is given one.",
        )

    def fall_silent(self, silence) -> None:
        self._cycle.stay_silent(silence.why)
        self.warn(silence.warning)

    def measurement(self, now_ns: int) -> Measurement:
        """Collect the two joint groups and the age of each, as the cycle reads them."""

        def age(stamp):
            return None if stamp is None else (now_ns - stamp.nanoseconds) / 1e9

        return Measurement(
            self._q_a,
            self._dq_a,
            self._q_u,
            self._dq_u,
            age(self._actuated_stamp),
            age(self._passive_stamp),
        )

    def jtc_output(self, measured_ns: int):
        """
        Mean JTC output over `[measured - Ts, measured)`, or `None`.

        What the plant runs over the dead time, PI included. `None` below half
        the expected ticks.
        """
        start = measured_ns - int(self.Ts * NANOSECONDS)
        window = [v for t, v in self._jtc_outputs if start <= t < measured_ns]
        if len(window) < 0.5 * self.Ts / self._jtc_period:
            return None
        return np.mean(window, axis=0)

    def follower_command(self, now_ns: int) -> FollowerCommand:
        """Read what the velocity controller is doing; only shadow mode compares."""
        return reports.follower_command(
            self._controller_state if self.mode == "shadow" else None,
            self._joints,
            now_ns,
            float(self._values.max_clock_skew),
            float(self._values.max_state_age),
        )

    def warn(self, message: str) -> None:
        self.get_logger().warn(message, throttle_duration_sec=WARN_PERIOD)

    # -- what goes out -----------------------------------------------------------

    def publish_horizon(self) -> None:
        cycle = self._cycle
        # Stamped at the measurement instant; knot 0 is due `command_delay` later.
        # See `hz.horizon_to_message`.
        lead_ns = int(cycle.command_delay * NANOSECONDS)
        message = hz.horizon_to_message(
            cycle.horizon,
            self._canonical_joints,
            Time(nanoseconds=cycle.next_first_knot_ns - lead_ns).to_msg(),
            lead=cycle.delay,
            period=self._jtc_period,
            in_flight=cycle.last_horizon,
            in_flight_u=cycle.in_flight_command(),
            linear=cycle.command_state,
        )
        self.publish_tcp_horizon()
        if self.mode == "active":
            self._horizon_publisher.publish(message)
        else:
            self._shadow_horizon_publisher.publish(message)
            cycle.take_shadow_command()

    def publish_tcp_horizon(self) -> None:
        path, why = reports.tcp_path(self._cycle, self._model)
        if why is not None:
            self.warn(f"The TCP horizon could not be drawn: {why}.")
        elif path is not None:
            self._tcp_publisher.publish(path)

    def report_sway_settled(self, now_ns: int) -> None:
        """Say whether the load has stopped swinging, on this cycle's stamp."""
        cycle = self._cycle
        measured = self._passive_velocity_stamp
        verdict = reports.sway_settled(
            self._dq_u,
            None if measured is None else (now_ns - measured.nanoseconds) / 1e9,
            float(self._values.max_state_age),
            list(self._values.sway.dq_u_settled),
            self._stamp,
        )
        # Reuses `solver_health_decimation`, not a second knob -- two numbers
        # for one decision could disagree.
        if not reports.settled_is_due(
            cycle.cycles,
            int(self._values.solver_health_decimation),
            self._last_settled,
            verdict,
        ):
            return
        self._settled_publisher.publish(verdict)
        self._last_settled = verdict

    def report(self, solution, why: str) -> None:
        """Write both reports, in the order every exit from `update` writes them."""
        self.report_solver_health(solution, why)
        if self.mode == "shadow":
            self._comparison_publisher.publish(
                reports.shadow_comparison(
                    self._cycle,
                    why,
                    solution,
                    self._joints,
                    float(self._values.solve_budget),
                    self._stamp,
                )
            )

    def report_solver_health(self, solution, why: str) -> None:
        cycle = self._cycle
        health = reports.solver_health(
            cycle,
            solution,
            why,
            self._joints,
            float(self._values.solve_budget),
            self._stamp,
        )
        if not reports.health_is_due(
            cycle.solves,
            int(self._values.solver_health_decimation),
            self._last_health,
            health,
        ):
            return
        if solution is not None and self._ocp is not None:
            try:
                cycle.cost_terms = self._ocp.cost_terms(solution, cycle.q_eq)
                reports.fill_cost_terms(health, cycle.cost_terms)
            except Exception as error:
                self.warn(f"The cost could not be split by term: {error}.")
        self._health_publisher.publish(health)
        self._last_health = health

    # -- the payload -------------------------------------------------------------

    def on_set_payload(self, request, response):
        response.active_mass = self._payload.mass_kg
        if not self.ready():
            response.success = False
            response.message = (
                f"the MPC is not configured -- nothing on {ROBOT_DESCRIPTION_TOPIC} "
                "yet -- so there is no problem to set a payload on"
            )
            self.get_logger().warn(
                f"{SET_PAYLOAD_SERVICE} was called before configuration and refused."
            )
            return response

        payload, why = reports.payload_from_message(request.payload)
        if payload is None:
            response.success = False
            response.message = (
                f"{why}. The payload this node is solving with is unchanged"
            )
            self.get_logger().warn(f"{SET_PAYLOAD_SERVICE} refused a payload: {why}.")
            return response

        changed = self._ocp.set_payload(payload.mass_kg, payload.center_of_mass_k8_m)
        self._payload = payload
        response.success = True
        response.active_mass = payload.mass_kg
        if not changed:
            response.message = (
                "the payload is unchanged, so nothing was dropped and the next solve "
                "is still warm-started"
            )
            return response

        # A payload step makes the warm plan and the force state seeded for it
        # stale, which is the same reactivation every other break in output is.
        self._cycle.forget_plan()
        com = payload.center_of_mass_k8_m
        response.message = (
            f"the payload is now {payload.mass_kg} kg at ({com[0]}, {com[1]}, "
            f"{com[2]}) m in K8, on every stage of the next horizon, which cold starts"
        )
        self.get_logger().info(response.message)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
