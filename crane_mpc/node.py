"""
The `crane_mpc` node: `src/mpc_node.cpp` in Python.

Nothing on the wire changes -- the topics, their types, their QoS, the service
and the 25 Hz cadence are the contract `wiki/implementation/ros2_interfaces.md`
§4 fixes, and this is the same node behind them.

What does change is that the state never leaves the OCP's own coordinates. The
C++ carried a 16-wide `crane_model::State` between the node and the solver and
rebuilt the actuator rows on each crossing; here `x` is the problem's 25 rows
from `/joint_states` to the published horizon.

An adapter and nothing else: parameters, subscriptions, publishers, the service,
the timer and the conversions at the boundary. `cycle.py` decides, `reports.py`
marshals, `config.py` shapes the parameters.
"""

from __future__ import annotations

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from crane_model import CraneModel, Payload, Tool, canonical_joints
from crane_msgs.msg import PayloadEstimate, SolverHealth
from crane_msgs.srv import SetPayload
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Path
from rclpy.node import Node
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
    JOINT_STATE_TOPIC,
    PASSIVE_DOF,
    PASSIVE_INDICES,
    PAYLOAD_ESTIMATE_TOPIC,
    PLANNED_DOF,
    REFERENCE_TOPIC,
    ROBOT_DESCRIPTION_TOPIC,
    SET_PAYLOAD_SERVICE,
    SHADOW_COMPARISON_TOPIC,
    SHADOW_HORIZON_TOPIC,
    SOLVER_HEALTH_TOPIC,
    TCP_HORIZON_TOPIC,
    Cycle,
    FollowerCommand,
    Measurement,
    Silence,
)
from .parameters import crane_mpc as parameter_library
from .solver import Ocp

#: How often a repeated complaint is logged, ms.
WARN_PERIOD = 5.0


def _qos(durability=DurabilityPolicy.VOLATILE) -> QoSProfile:
    return QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=durability
    )


def _latched() -> QoSProfile:
    return _qos(DurabilityPolicy.TRANSIENT_LOCAL)


def _measured(message: JointState, index: dict, names: list):
    """
    `(q, dq)` for `names`, or `None` if the message does not carry them finitely.

    `dq` is `None` where the message carries no velocity for them, which leaves
    the last one standing.

    **A non-finite row is dropped rather than written through**
    (`mpc_node.cpp:360-364`). A NaN that reaches `x0` comes back as a solve
    failure: it counts against `max_consecutive_failures` and is reported as
    FAULT_SOLVER, when what happened is that one sensor went stale. Dropped
    here, the group's stamp does not advance and it surfaces as staleness
    through `max_state_age`, which is what it is.
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
        self._joints: list[str] = []
        self._passive_joints: list[str] = []
        #: The canonical eight, in contract order: what the horizon is named by.
        self._canonical_joints: list[str] = []

        # The measurement, by name and never by index: the two stacks publish
        # different `/joint_states` name sets in different orders.
        self._q_a = np.zeros(ACTUATED_DOF)
        self._dq_a = np.zeros(ACTUATED_DOF)
        self._q_u = np.zeros(PASSIVE_DOF)
        self._dq_u = np.zeros(PASSIVE_DOF)
        self._actuated_stamp: Time | None = None
        self._passive_stamp: Time | None = None

        self._reference_message: JointTrajectory | None = None
        self._payload = Payload()
        self._payload_estimate: PayloadEstimate | None = None
        self._controller_state: JointTrajectoryControllerState | None = None
        self._last_health: SolverHealth | None = None

        self._create_publishers()
        self._create_subscriptions()
        self.create_timer(self.Ts, self.update)

        if abs(1.0 / self.Ts - 1.0 / 0.04) > 1e-9:
            self.get_logger().warn(
                f"Ts is {self.Ts} s, so this node publishes at {1.0 / self.Ts:.1f} Hz "
                "and ros2_interfaces §4 fixes /crane/mpc/horizon at 25 Hz."
            )
        self.get_logger().info(
            f"crane_mpc in {self.mode} mode: {self.grid.horizon_length} knots at "
            f"{self.Ts} s, {self.delay} s of transport delay, a "
            f"{self._values.solve_budget} s budget."
        )
        self.warn_open_loop_axes()

    @property
    def mode(self) -> str:
        return self._cycle.mode

    def axis_names(self) -> list[str]:
        """Name the six actuated joints, before the description has arrived too."""
        names = canonical_joints()
        return [names[row] for row in ACTUATED_INDICES]

    def warn_open_loop_axes(self) -> None:
        """Say once, at startup, which axes do not close the velocity loop."""
        cycle = self._cycle
        names = self.axis_names()
        open_loop = [
            names[axis] for axis in range(PLANNED_DOF) if not cycle.dq_a_feedback[axis]
        ]
        if not open_loop:
            return
        self.get_logger().warn(
            f"{', '.join(open_loop)} take their velocity in x_0 from the **model** "
            f"and not from {JOINT_STATE_TOPIC} (dq_a_feedback). That is Marc's "
            "deployed setting -- his node closes the velocity loop on the slewing "
            "axis alone -- and it buys quiet on noisy hydraulic velocity signals at "
            "the cost of an actuator state that can walk away from the machine. "
            f"dq_a_divergence_max is what watches for that; it is reported on "
            f"{SHADOW_COMPARISON_TOPIC} and warned about, and never corrected."
        )

    def warn_divergence(self) -> None:
        """Report a carried velocity that has walked away from the measured one."""
        carry = self._cycle.velocity_carry
        if not carry.any_diverged:
            return
        for axis, joint in enumerate(self._joints):
            if not carry.diverged[axis]:
                continue
            # Its own call site, not `self.warn`: rclpy keys the throttle by
            # caller, so sharing that one would put the divergence in the same
            # five-second bucket as every silence this node reports.
            self.get_logger().warn(
                f"The model-carried velocity of {joint} has walked "
                f"{carry.divergence[axis]:g} away from the measured one, past the "
                f"{self._cycle.dq_a_divergence_max[axis]:g} dq_a_divergence_max on "
                "that axis. This axis runs open-loop in velocity by configuration "
                "(dq_a_feedback), so nothing corrects it and the state the OCP is "
                "solved from is the model's, not the machine's. Reported and not "
                "corrected: substituting the measurement on a threshold would be a "
                "second controller nobody configured.",
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
        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self.on_robot_description, _latched()
        )
        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self.on_joint_state, _qos()
        )
        self.create_subscription(
            PayloadEstimate,
            PAYLOAD_ESTIMATE_TOPIC,
            self.on_payload_estimate,
            _latched(),
        )
        self.create_service(SetPayload, SET_PAYLOAD_SERVICE, self.on_set_payload)

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
            self._passive_stamp = stamp

    def on_reference(self, message: JointTrajectory) -> None:
        self._reference_message = message
        self.adopt_reference()

    def on_payload_estimate(self, message: PayloadEstimate) -> None:
        # Stored and reported at startup; the payload the OCP solves with moves
        # only through the service, exactly as the C++ node read it.
        self._payload_estimate = message

    def on_controller_state(self, message: JointTrajectoryControllerState) -> None:
        self._controller_state = message

    # -- configuration -----------------------------------------------------------

    def configure(self) -> None:
        """Build the model and the problem once the description has arrived."""
        if self._description is None or self._ocp is not None:
            return
        try:
            self._model = CraneModel(self._description, Tool(problem.TOOL))
            names = canonical_joints()
            self._joints = [names[row] for row in ACTUATED_INDICES]
            self._passive_joints = [names[row] for row in PASSIVE_INDICES]
            self._canonical_joints = list(names)
            self._ocp = Ocp(
                problem.default_description().read_text(),
                config.parameter_dict(self._values),
                config.hydraulics_dict(self._values),
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
        self.adopt_reference()

    def ready(self) -> bool:
        return self._ocp is not None and self._model is not None

    def adopt_reference(self) -> None:
        if self._reference_message is None or not self.ready():
            return
        reference, why = hz.reference_from_message(
            self._reference_message, self._joints
        )
        if reference is None:
            self.get_logger().warn(
                f"The reference on {REFERENCE_TOPIC} was refused and the previous "
                f"one stands: {why}."
            )
            self._reference_message = None
            return
        self._cycle.adopt_reference(
            reference, Time.from_msg(self._reference_message.header.stamp).nanoseconds
        )
        self._reference_message = None

    def adopt_mode(self, requested: str) -> None:
        previous = self._cycle.adopt_mode(requested)
        self.get_logger().info(
            f"crane_mpc moved from {previous} to {requested}; the next solve cold "
            "starts, because a mode change is this node's reactivation."
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

        if not self.ready():
            why = (
                "no robot description has arrived"
                if self._description is None
                else (self._configuration_failure or "configuration is incomplete")
            )
            self.fall_silent(self.unconfigured(why))
            return

        now = self.get_clock().now().nanoseconds
        silence = cycle.gates(
            now,
            float(self._values.max_clock_skew),
            float(self._values.max_reference_age),
        )
        if silence is None:
            silence = cycle.read_state(
                self.measurement(now), float(self._values.max_state_age)
            )
        if silence is None:
            cycle.adopt_follower(self.follower_command(now), self._values.limits.u_max)
            silence = cycle.propagate()
        if silence is not None:
            self.fall_silent(silence)
            return
        self.warn_divergence()

        solution, refusal = cycle.solve()
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
        cycle.advance(solution)
        self.report(solution, verdict.text)

    def complain(self, verdict) -> None:
        if verdict.severity == "error":
            self.get_logger().error(verdict.text)
        elif verdict.severity:
            self.warn(verdict.text)

    def unconfigured(self, why: str) -> Silence:
        return Silence(
            why,
            f"The MPC is not configured ({why}), so nothing is published on "
            f"{HORIZON_TOPIC}. Configuration requires one latched message on "
            f"{ROBOT_DESCRIPTION_TOPIC} and nothing else.",
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

    def follower_command(self, now_ns: int) -> FollowerCommand:
        """Read what the velocity controller is actually doing, per axis."""
        follower = FollowerCommand()
        state = self._controller_state
        if self.mode != "shadow" or state is None:
            return follower
        follower.age = (now_ns - Time.from_msg(state.header.stamp).nanoseconds) / 1e9
        if -follower.age > float(self._values.max_clock_skew):
            follower.source = "future"
            return follower
        if follower.age > float(self._values.max_state_age):
            follower.source = "stale"
            return follower

        names = list(state.joint_names)
        sources = set()
        complete = True
        for axis, joint in enumerate(self._joints):
            if joint not in names:
                complete = False
                continue
            row = names.index(joint)
            taken = None
            for field in ("output", "reference"):
                velocities = getattr(state, field).velocities
                if row < len(velocities) and np.isfinite(velocities[row]):
                    follower.velocity[axis] = velocities[row]
                    taken = f"{field}.velocities"
                    break
            if taken is None:
                complete = False
            else:
                follower.have_velocity[axis] = True
                sources.add(taken)
            errors = state.error.velocities
            if row < len(errors) and np.isfinite(errors[row]):
                follower.velocity_error[axis] = errors[row]
                follower.have_velocity_error[axis] = True
        follower.complete = complete
        follower.source = (
            "mixed" if len(sources) > 1 else (sources.pop() if sources else "none")
        )
        return follower

    def warn(self, message: str) -> None:
        self.get_logger().warn(message, throttle_duration_sec=WARN_PERIOD)

    # -- what goes out -----------------------------------------------------------

    def publish_horizon(self) -> None:
        cycle = self._cycle
        message = hz.horizon_to_message(
            cycle.horizon,
            self._canonical_joints,
            Time(nanoseconds=cycle.next_first_knot_ns).to_msg(),
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
                    self.get_clock().now().to_msg(),
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
            self.get_clock().now().to_msg(),
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
                cycle.cost_terms = self._ocp.cost_terms(
                    solution, cycle.horizon, cycle.q_eq
                )
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
                f"the MPC is not configured -- no robot description on "
                f"{ROBOT_DESCRIPTION_TOPIC} yet -- so there is no optimal control "
                "problem to set a payload on"
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

        self._cycle.payload_changed()
        com = payload.center_of_mass_k8_m
        response.message = (
            f"the payload is now {payload.mass_kg} kg at ({com[0]}, {com[1]}, "
            f"{com[2]}) m in K8, on every stage of the next horizon. The next solve "
            "cold starts"
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
