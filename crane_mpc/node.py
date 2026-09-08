"""
The `crane_mpc` node: `src/mpc_node.cpp` in Python.

Nothing on the wire changes -- the topics, their types, their QoS, the service
and the 25 Hz cadence are the contract `wiki/implementation/ros2_interfaces.md`
§4 fixes, and this is the same node behind them.

What does change is that the state never leaves the OCP's own coordinates. The
C++ carried a 16-wide `crane_model::State` between the node and the solver and
rebuilt the actuator rows on each crossing; here `x` is the problem's 25 rows
from `/joint_states` to the published horizon.
"""

from __future__ import annotations

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from crane_model import CraneModel, Frame, Payload, Tool, canonical_joints
from crane_model import symbolic as cs
from crane_msgs.msg import PayloadEstimate, SolverHealth, SupervisorStatus
from crane_msgs.srv import SetPayload
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from . import horizon as hz
from . import problem
from .parameters import crane_mpc as parameter_library
from .solver import Ocp, Outcome

REFERENCE_TOPIC = "/crane/reference"
JOINT_STATE_TOPIC = "/joint_states"
PAYLOAD_ESTIMATE_TOPIC = "/crane/payload_estimate"
CONTROLLER_STATE_TOPIC = "/crane/controller_state"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
HORIZON_TOPIC = "/crane/mpc/horizon"
SOLVER_HEALTH_TOPIC = "/crane/mpc/solver_health"
TCP_HORIZON_TOPIC = "/crane/mpc/tcp_horizon"
TCP_HORIZON_FRAME = "K0_mounting_base"
SET_PAYLOAD_SERVICE = "/crane/mpc/set_payload"
SHADOW_HORIZON_TOPIC = "~/shadow_horizon"
SHADOW_COMPARISON_TOPIC = "~/shadow_comparison"

#: How often a repeated complaint is logged, ms.
WARN_PERIOD = 5.0

#: Where the canonical eight carry the six actuated and the two passive.
ACTUATED_INDICES = cs.K_ACTUATED_ROWS
PASSIVE_INDICES = cs.K_PASSIVE_ROWS
ACTUATED_DOF = cs.K_ACTUATED_DOF
PLANNED_DOF = cs.K_PLANNED_DOF
PASSIVE_DOF = cs.K_PASSIVE_DOF
TOOL_AXIS = cs.K_TOOL_AXIS


def _qos(durability=DurabilityPolicy.VOLATILE) -> QoSProfile:
    return QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=durability
    )


def _latched() -> QoSProfile:
    return _qos(DurabilityPolicy.TRANSIENT_LOCAL)


def _text(value: float) -> str:
    """Format a double that round-trips: a comparison is read off these."""
    return f"{value:.17g}"


class MpcNode(Node):
    def __init__(self) -> None:
        super().__init__("crane_mpc")
        self._listener = parameter_library.ParamListener(self)
        self._values = self._listener.get_params()

        self.Ts = float(self._values.Ts)
        self.grid = hz.Grid(self.Ts, int(self._values.horizon_length))
        self.mode = str(self._values.mode)
        self.delay = float(self._values.sensor_to_valve_delay)

        self._ocp: Ocp | None = None
        self._model: CraneModel | None = None
        self._description: str | None = None
        self._configuration_failure = ""
        self._joints: list[str] = []
        self._passive_joints: list[str] = []

        # The measurement, by name and never by index: the two stacks publish
        # different `/joint_states` name sets in different orders.
        self._q_a = np.zeros(ACTUATED_DOF)
        self._dq_a = np.zeros(ACTUATED_DOF)
        self._q_u = np.zeros(PASSIVE_DOF)
        self._dq_u = np.zeros(PASSIVE_DOF)
        self._actuated_stamp: Time | None = None
        self._passive_stamp: Time | None = None

        self._reference: hz.Knots | None = None
        self._reference_message: JointTrajectory | None = None
        self._reference_stamp = Time()
        self._reference_progress = 0.0
        self._reference_anchored = False
        self._next_first_knot = Time()
        self._cadence_anchored = False

        self._horizon: hz.Knots | None = None
        self._last_horizon: hz.Knots | None = None
        self._tcp_states: np.ndarray | None = None
        self._last_tcp_states: np.ndarray | None = None

        # The OCP's own rows that no sensor carries: C3's lagged command, its
        # force state and the progress pair. Seeded at the first solve.
        self._carried = np.zeros(cs.NX)
        self._carried[cs.X_PROGRESS_RATE] = problem.K_PROGRESS_RATE_REFERENCE
        self._seed_force = True
        self._last_input = np.zeros(cs.NU_PROGRESS)
        self._guess = None
        self._last_solution = None

        self._payload = Payload()
        self._payload_estimate: PayloadEstimate | None = None
        self._controller_state: JointTrajectoryControllerState | None = None

        self._solves = 0
        self._consecutive_failures = 0
        self._escalated = False
        self._applied_previous = False
        self._last_silence = ""
        self._last_health: SolverHealth | None = None
        self._cost_terms = None
        self._q_eq = np.zeros(PASSIVE_DOF)
        self._shadow_command = np.zeros(ACTUATED_DOF)
        self._shadow_command_valid = False

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
        rows = [index.get(name) for name in self._joints]
        if all(row is not None for row in rows) and len(message.position) > max(rows):
            self._q_a = np.array([message.position[row] for row in rows])
            if len(message.velocity) > max(rows):
                self._dq_a = np.array([message.velocity[row] for row in rows])
            self._actuated_stamp = stamp
        rows = [index.get(name) for name in self._passive_joints]
        if all(row is not None for row in rows) and len(message.position) > max(rows):
            self._q_u = np.array([message.position[row] for row in rows])
            if len(message.velocity) > max(rows):
                self._dq_u = np.array([message.velocity[row] for row in rows])
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
            self._ocp = Ocp(
                problem.default_description().read_text(),
                self._parameter_dict(),
                self._hydraulics_dict(),
            )
        except Exception as error:  # the description decides whether this node runs
            self._configuration_failure = str(error)
            self.get_logger().error(f"The MPC could not be configured: {error}")
            return
        self.get_logger().info(
            f"crane_mpc configured on {self._model.tool.value}: joints "
            f"{', '.join(self._joints)}, sway {', '.join(self._passive_joints)}."
        )
        self.adopt_reference()

    def _parameter_dict(self) -> dict:
        """
        Shape the parameters as the problem reads them.

        One shape, whether they came off the parameter server or out of a yaml
        on a driver's disk.
        """
        values = self._values
        return {
            "Ts": float(values.Ts),
            "horizon_length": int(values.horizon_length),
            "levenberg_marquardt": float(values.levenberg_marquardt),
            "solve_budget": float(values.solve_budget),
            "sensor_to_valve_delay": float(values.sensor_to_valve_delay),
            "weights": {
                name: getattr(values.weights, name)
                for name in (
                    "q_a",
                    "dq_a",
                    "q_u",
                    "dq_u",
                    "tau_a",
                    "u",
                    "lag",
                    "progress_rate",
                    "progress_accel",
                    "terminal_scale",
                )
            },
            "limits": {
                name: getattr(values.limits, name)
                for name in (
                    "q_a_lower",
                    "q_a_upper",
                    "q_a_margin",
                    "dq_a_max",
                    "q_u_max",
                    "dq_u_max",
                    "u_max",
                    "progress_rate_max",
                    "progress_accel_max",
                )
            },
            "slack": {
                name: getattr(values.slack, name)
                for name in ("q_u", "dq_u", "cylinder_force", "pump_flow")
            },
        }

    def _hydraulics_dict(self) -> dict:
        hydraulics = self._values.hydraulics
        return {
            "pump_flow_max": float(hydraulics.pump_flow_max),
            "pump_flow_planning_factor": float(hydraulics.pump_flow_planning_factor),
            "system_pressure_pa": float(hydraulics.system_pressure_pa),
        }

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
        self._reference = reference
        self._reference_stamp = Time.from_msg(self._reference_message.header.stamp)
        self._reference_message = None
        self._reference_anchored = False

    # -- the cycle ---------------------------------------------------------------

    def update(self) -> None:
        if self._listener.is_old(self._values):
            self._values = self._listener.get_params()
            if self._ocp is not None:
                self._ocp.solve_budget_s = float(self._values.solve_budget)
            if str(self._values.mode) != self.mode:
                self.adopt_mode(str(self._values.mode))
        self._shadow_command_valid = False
        self._cost_terms = None

        if not self.ready():
            why = (
                "no robot description has arrived"
                if self._description is None
                else (self._configuration_failure or "configuration is incomplete")
            )
            self.stay_silent(why)
            self.warn(
                f"The MPC is not configured ({why}), so nothing is published on "
                f"{HORIZON_TOPIC}. Configuration requires one latched message on "
                f"{ROBOT_DESCRIPTION_TOPIC} and nothing else."
            )
            return

        if self._reference is None or len(self._reference) == 0:
            self.stay_silent("no reference has arrived")
            self.warn(
                f"No reference has arrived on {REFERENCE_TOPIC}, so nothing is "
                f"published on {HORIZON_TOPIC}. A horizon of zeros would be a plan "
                "to stop where the crane is not, and a stale one is worse."
            )
            return

        now = self.get_clock().now()
        lead = (self._reference_stamp - now).nanoseconds / 1e9
        if lead > float(self._values.max_clock_skew):
            self.stay_silent("the reference is stamped in this node's future")
            self.warn(
                f"The reference on {REFERENCE_TOPIC} is stamped {lead:.3f} s ahead "
                f"of this node against a {self._values.max_clock_skew} s bound, so "
                f"nothing is published on {HORIZON_TOPIC}."
            )
            return

        spent = self._reference_progress - float(self._reference.t[-1])
        if self._reference_anchored and spent > float(self._values.max_reference_age):
            self.stay_silent(
                "the reference's plan has been spent past its end by more than "
                "max_reference_age"
            )
            self.warn(
                f"The plan on {REFERENCE_TOPIC} ended {spent:.3f} s of virtual time "
                f"ago against a {self._values.max_reference_age} s bound, so nothing "
                f"is published on {HORIZON_TOPIC}."
            )
            return

        self.anchor_cadence(now)

        if not self._reference_anchored:
            self._reference_progress = (
                self._next_first_knot - self._reference_stamp
            ).nanoseconds / 1e9
            self._reference_anchored = True
        self._reference_progress = max(
            self._reference_progress, float(self._reference.t[0])
        )

        rejection, horizon = hz.resample(
            self._reference, self._reference_progress, self.grid
        )
        if rejection is not None:
            self.stay_silent(str(rejection))
            self.warn(
                f"The reference on {REFERENCE_TOPIC} could not be resampled onto the "
                f"horizon, so nothing is published on {HORIZON_TOPIC}: {rejection}."
            )
            return
        self._horizon = horizon

        measured, why = self.measured_state(now)
        if measured is None:
            self.stay_silent(why)
            self.warn(
                f"Nothing was published on {HORIZON_TOPIC} because there is no state "
                f"to solve from: {why}."
            )
            return

        if self.mode == "shadow":
            self._follower = self.follower_command(now)
            self._last_input = self.follower_input(self._follower)
        else:
            self._follower = FollowerCommand()

        self._ocp.pin_tool(float(self._q_a[TOOL_AXIS]))
        try:
            x0 = self._ocp.propagate(measured, self._last_input)
        except Exception as error:
            self.stay_silent(
                "the measured state could not be propagated to the instant the plan "
                "takes effect"
            )
            self.warn(
                f"The measured state could not be carried {self.delay} s forward, so "
                f"nothing is published on {HORIZON_TOPIC}: {error}."
            )
            return

        # The sway box is centred on where the tool is actually swinging, not on
        # the hanging pose. `weights.q_u` being zero and this centre justify each
        # other in a circle -- both are recorded, neither is this port's business.
        q_eq = x0[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF].copy()
        self._q_eq = q_eq

        try:
            solution = self._ocp.solve(x0, self._horizon, q_eq, self._guess)
        except Exception as error:
            self._applied_previous = False
            self._consecutive_failures += 1
            self._solves += 1
            self.report_solver_health(None, str(error))
            if self.mode == "shadow":
                self.publish_shadow_comparison(str(error), None)
            self.stay_silent_after_failure(
                "the optimal control problem did not return a usable horizon"
            )
            self.warn(f"The OCP refused its arguments: {error}.")
            return

        self._solves += 1
        self._last_solution = solution
        self._guess = (
            None if solution.outcome is Outcome.FAILED else self._ocp.shifted(solution)
        )
        converged = solution.outcome is Outcome.CONVERGED
        if converged:
            self._consecutive_failures = 0
            self._escalated = False
        else:
            self._consecutive_failures += 1
        destination = HORIZON_TOPIC if self.mode == "active" else SHADOW_HORIZON_TOPIC

        if converged:
            self.adopt_solution(solution)
            self._applied_previous = False
            verdict = (
                "the solve converged inside the budget and its horizon was published "
                f"on {destination}"
            )
        elif self._consecutive_failures >= int(self._values.max_consecutive_failures):
            self._escalated = True
            self._applied_previous = False
            verdict = (
                f"mpc §6's repeated-failure escalation: {self._consecutive_failures} "
                "consecutive solves did not converge against a ceiling of "
                f"{self._values.max_consecutive_failures} (acados last answered "
                f"{solution.status_word}, {solution.outcome}), so this node has "
                "stopped publishing and handed control back rather than shifting a "
                "plan it no longer believes"
            )
            self.report_solver_health(solution, verdict)
            if self.mode == "shadow":
                self.publish_shadow_comparison(verdict, solution)
            self.stay_silent_after_failure(
                "mpc §6's repeated-failure escalation has stopped the publisher"
            )
            self.get_logger().error(verdict)
            return
        elif self.shift_previous_horizon():
            self._applied_previous = True
            verdict = (
                f"acados answered {solution.status_word} ({solution.outcome}) after "
                f"{round(1000 * solution.solve_time_s)} ms against a "
                f"{round(1000 * self._ocp.solve_budget_s)} ms budget, so mpc §6's "
                "previous solution shifted by one step was published on "
                f"{destination} instead"
            )
            self.warn(verdict)
        else:
            self._applied_previous = False
            verdict = (
                f"acados answered {solution.status_word} ({solution.outcome}) and "
                "there was no previous solution to shift, so nothing was published"
            )
            self.report_solver_health(solution, verdict)
            if self.mode == "shadow":
                self.publish_shadow_comparison(verdict, solution)
            self.stay_silent_after_failure(
                "the solve did not converge and there is no previous solution to shift"
            )
            self.warn(verdict)
            return

        message = hz.horizon_to_message(
            self._horizon, self._joints, self._next_first_knot.to_msg()
        )
        self.publish_tcp_horizon()
        if self.mode == "active":
            self._horizon_publisher.publish(message)
        else:
            self._shadow_horizon_publisher.publish(message)
            self._shadow_command = self._horizon.dq_a_ref[1].copy()
            self._shadow_command_valid = True

        self._last_horizon = self._horizon.copy()
        self._last_tcp_states = (
            None if self._tcp_states is None else self._tcp_states.copy()
        )
        if converged:
            self._last_input = solution.u0.copy()
        else:
            self._last_input = np.zeros(cs.NU_PROGRESS)
            self._last_input[:PLANNED_DOF] = self._horizon.dq_a_ref[1, :PLANNED_DOF]
        if solution.outcome is not Outcome.FAILED:
            # C3's own rows, carried: the command in flight and the force the
            # actuators are at. No sensor reports either.
            self._carried = solution.states[1].copy()

        self._next_first_knot = self._next_first_knot + Duration(
            nanoseconds=int(self.Ts * 1e9)
        )
        advance = solution.progress_advance
        self._reference_progress += (
            advance if np.isfinite(advance) and advance >= 0.0 else self.Ts
        )
        self._last_silence = ""
        self.report_solver_health(solution, verdict)
        if self.mode == "shadow":
            self.publish_shadow_comparison(verdict, solution)

    # -- the pieces of one cycle -------------------------------------------------

    def anchor_cadence(self, now: Time) -> None:
        """
        Decide when the first knot takes effect.

        `now` plus the transport delay, re-anchored whenever the cadence has
        drifted off the horizon.
        """
        anchor = now + Duration(nanoseconds=int(self.delay * 1e9))
        ceiling = now + Duration(
            nanoseconds=int((self.delay + self.grid.duration()) * 1e9)
        )
        if (
            not self._cadence_anchored
            or self._next_first_knot < now
            or self._next_first_knot > ceiling
        ):
            self._next_first_knot = anchor
            self._cadence_anchored = True

    def measured_state(self, now: Time):
        """
        `x` from `/joint_states` alone, or `(None, why)`.

        The actuator rows are the ones carried from the last accepted solve:
        nothing measures the command in flight or the force the cylinders are at.
        """
        if self._actuated_stamp is None:
            return None, f"no measured state has arrived on {JOINT_STATE_TOPIC}"
        if self._passive_stamp is None:
            return None, f"no passive state has arrived on {JOINT_STATE_TOPIC}"
        age = float(self._values.max_state_age)
        actuated = (now - self._actuated_stamp).nanoseconds / 1e9
        passive = (now - self._passive_stamp).nanoseconds / 1e9
        if actuated > age or passive > age:
            return None, "the measured state is older than max_state_age"

        x = self._carried.copy()
        x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF] = self._q_a[
            :PLANNED_DOF
        ]
        x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF] = self._dq_a[
            :PLANNED_DOF
        ]
        x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF] = self._q_u
        x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE_DOF] = self._dq_u
        x[cs.X_PROGRESS] = 0.0
        if self._seed_force:
            # C3 block 3 starts holding the machine's own weight: zero would be
            # the hydraulics switched off, and there is no force measurement.
            self._ocp.pin_tool(float(self._q_a[TOOL_AXIS]))
            x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED_DOF] = (
                self._ocp.static_hold_force(x)
            )
            self._carried = x.copy()
            self._seed_force = False
        return x, ""

    def adopt_solution(self, solution) -> None:
        """Write the solved horizon as the six actuated joints the wire carries."""
        states = solution.states
        self._horizon.q_a_ref[:, :PLANNED_DOF] = states[
            :, cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF
        ]
        self._horizon.dq_a_ref[:, :PLANNED_DOF] = states[
            :, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF
        ]
        # The tool is pinned, not planned: it holds where it was measured.
        self._horizon.q_a_ref[:, TOOL_AXIS] = self._q_a[TOOL_AXIS]
        self._horizon.dq_a_ref[:, TOOL_AXIS] = 0.0
        self._tcp_states = states.copy()

    def shift_previous_horizon(self) -> bool:
        """Shift the last horizon one knot on, duplicating its last: mpc §6."""
        if (
            self._last_horizon is None
            or self._last_tcp_states is None
            or len(self._last_horizon) != len(self._horizon)
            or len(self._last_tcp_states) != len(self._horizon)
        ):
            return False
        for field in ("q_a_ref", "dq_a_ref"):
            previous = getattr(self._last_horizon, field)
            current = getattr(self._horizon, field)
            current[:-1] = previous[1:]
            current[-1] = previous[-1]
        self._tcp_states = np.vstack(
            [self._last_tcp_states[1:], self._last_tcp_states[-1:]]
        )
        return True

    def stay_silent(self, why: str) -> None:
        self._consecutive_failures = 0
        self._escalated = False
        self.stay_silent_after_failure(why)

    def stay_silent_after_failure(self, why: str) -> None:
        """Nothing goes out, so nothing may be shifted next cycle either."""
        self._last_silence = why
        self._last_horizon = None
        self._tcp_states = None
        self._last_tcp_states = None
        self._guess = None
        if self._reference_anchored:
            # One nominal interval spent while nobody was driving.
            self._reference_progress += self.Ts

    def adopt_mode(self, requested: str) -> None:
        previous, self.mode = self.mode, requested
        self._last_horizon = None
        self._tcp_states = None
        self._last_tcp_states = None
        self._guess = None
        self._consecutive_failures = 0
        self._escalated = False
        self._cadence_anchored = False
        self._last_input = np.zeros(cs.NU_PROGRESS)
        self.get_logger().info(
            f"crane_mpc moved from {previous} to {requested}; the next solve cold "
            "starts, because a mode change is this node's reactivation."
        )

    def warn(self, message: str) -> None:
        self.get_logger().warn(message, throttle_duration_sec=WARN_PERIOD)

    # -- the follower, in shadow -------------------------------------------------

    def follower_command(self, now: Time) -> FollowerCommand:
        """Read what the velocity controller is actually doing, per axis."""
        follower = FollowerCommand()
        state = self._controller_state
        if state is None:
            return follower
        follower.age = (now - Time.from_msg(state.header.stamp)).nanoseconds / 1e9
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

    def follower_input(self, follower: FollowerCommand) -> np.ndarray:
        """
        Take the follower's command as `u`.

        Under C3 `u` **is** a joint velocity, so it is clipped and never
        differenced into an acceleration.
        """
        command = np.zeros(cs.NU_PROGRESS)
        if not follower.complete:
            return command
        bound = np.asarray(self._values.limits.u_max[:PLANNED_DOF], dtype=float)
        command[:PLANNED_DOF] = np.clip(follower.velocity[:PLANNED_DOF], -bound, bound)
        return command

    # -- what goes out -----------------------------------------------------------

    def publish_tcp_horizon(self) -> None:
        if (
            self._model is None
            or self._horizon is None
            or self._tcp_states is None
            or len(self._tcp_states) != len(self._horizon)
        ):
            return
        path = Path()
        path.header.frame_id = TCP_HORIZON_FRAME
        path.header.stamp = self._next_first_knot.to_msg()
        for index, state in enumerate(self._tcp_states):
            q = np.zeros(cs.K_GENERALIZED_DOF)
            q[list(ACTUATED_INDICES[:PLANNED_DOF])] = state[
                cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF
            ]
            q[list(PASSIVE_INDICES)] = state[
                cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
            ]
            q[ACTUATED_INDICES[TOOL_AXIS]] = self._q_a[TOOL_AXIS]
            try:
                pose = self._model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
            except Exception as error:
                self.warn(f"The TCP horizon could not be drawn: {error}.")
                return
            stamped = PoseStamped()
            stamped.header.frame_id = TCP_HORIZON_FRAME
            stamped.header.stamp = (
                self._next_first_knot
                + Duration(nanoseconds=int(self._horizon.t[index] * 1e9))
            ).to_msg()
            stamped.pose.position.x = float(pose.position_m[0])
            stamped.pose.position.y = float(pose.position_m[1])
            stamped.pose.position.z = float(pose.position_m[2])
            stamped.pose.orientation.x = float(pose.orientation_xyzw[0])
            stamped.pose.orientation.y = float(pose.orientation_xyzw[1])
            stamped.pose.orientation.z = float(pose.orientation_xyzw[2])
            stamped.pose.orientation.w = float(pose.orientation_xyzw[3])
            path.poses.append(stamped)
        self._tcp_publisher.publish(path)

    def report_solver_health(self, solution, why: str) -> None:
        health = SolverHealth()
        health.header.stamp = self.get_clock().now().to_msg()
        health.joint_names = list(self._joints)
        health.solve_budget = float(self._values.solve_budget)
        health.applied_previous_solution = self._applied_previous
        health.message = f"{self.mode}: {why}"
        if solution is None:
            health.outcome = SolverHealth.SOLVE_UNKNOWN
            health.fault = SupervisorStatus.FAULT_SOLVER
        else:
            health.outcome = {
                Outcome.CONVERGED: SolverHealth.SOLVE_CONVERGED,
                Outcome.BUDGET_EXCEEDED: SolverHealth.SOLVE_BUDGET_EXCEEDED,
                Outcome.FAILED: SolverHealth.SOLVE_FAILED,
            }[solution.outcome]
            health.fault = (
                SupervisorStatus.FAULT_NONE
                if solution.outcome is Outcome.CONVERGED
                else SupervisorStatus.FAULT_SOLVER
            )
            health.status = solution.status
            health.status_word = solution.status_word
            health.iterations = solution.iterations
            health.qp_status = solution.qp_status
            health.qp_iterations = solution.qp_iterations
            health.solve_time = solution.solve_time_s
            health.used_slack = solution.used_slack
            health.slack_penalty = solution.slack_penalty
            health.constraint_violation = [
                solution.violation.q_u,
                solution.violation.dq_u,
                solution.violation.cylinder_force,
                solution.violation.pump_flow,
            ]

        # Decimated, because the consumer is the 20 Hz supervisor -- but never a
        # change of verdict, which is what decimation exists not to drop.
        decimation = max(1, int(self._values.solver_health_decimation))
        on_cadence = (self._solves - 1) % decimation == 0
        changed = self._last_health is None or (
            self._last_health.fault != health.fault
            or self._last_health.outcome != health.outcome
        )
        if not on_cadence and not changed:
            return

        if solution is not None and self._ocp is not None:
            try:
                terms = self._cost_terms = self._ocp.cost_terms(
                    solution, self._horizon, self._q_eq
                )
                health.cost_term = [
                    terms.q_a,
                    terms.dq_a,
                    terms.q_u,
                    terms.dq_u,
                    terms.tau_a,
                    terms.u,
                    terms.terminal,
                    terms.slack,
                ]
            except Exception as error:
                self.warn(f"The cost could not be split by term: {error}.")
        self._health_publisher.publish(health)
        self._last_health = health

    def publish_shadow_comparison(self, verdict: str, solution) -> None:
        """Judge the shadow command against what drove, per user story 68."""
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "crane_mpc: the shadow solution against what drove"
        status.hardware_id = problem.TOOL
        values = status.values
        follower = getattr(self, "_follower", FollowerCommand())

        def put(key: str, value: str) -> None:
            values.append(KeyValue(key=key, value=value))

        put("mode", self.mode)
        put("follower.topic", CONTROLLER_STATE_TOPIC)
        put("follower.velocity_source", follower.source)
        put("follower.age", _text(follower.age))
        put("follower.error_field", "error.velocities")
        converged = solution is not None and solution.outcome is Outcome.CONVERGED
        put("solve.converged", "true" if converged else "false")
        put(
            "solve.outcome",
            str(solution.outcome) if solution else "the OCP refused its arguments",
        )
        put(
            "solve.applied_previous_solution",
            "true" if self._applied_previous else "false",
        )
        put("solve.escalated", "true" if self._escalated else "false")
        if solution is not None:
            put("solve.time", _text(solution.solve_time_s))
            put("solve.budget", _text(float(self._values.solve_budget)))
            put("constraint.used_slack", "true" if solution.used_slack else "false")
            put("constraint.slack_penalty", _text(solution.slack_penalty))
            put("constraint.sway", _text(solution.violation.q_u))
            put("constraint.sway_rate", _text(solution.violation.dq_u))
            put("constraint.cylinder_force", _text(solution.violation.cylinder_force))
            put("constraint.pump_flow", _text(solution.violation.pump_flow))
            # The two cost rows `crane_msgs/SolverHealth` has no field for. 119
            # made them the interesting ones, so they are reported here rather
            # than computed and left invisible as the C++ left them.
            if self._cost_terms is not None:
                put("cost.lag", _text(self._cost_terms.lag))
                put("cost.progress", _text(self._cost_terms.progress))

        compared = 0
        largest = 0.0
        for axis, joint in enumerate(self._joints):
            if self._shadow_command_valid:
                put(f"{joint}.shadow_velocity", _text(self._shadow_command[axis]))
            if follower.have_velocity[axis]:
                put(f"{joint}.follower_velocity", _text(follower.velocity[axis]))
            if self._shadow_command_valid and follower.have_velocity[axis]:
                difference = self._shadow_command[axis] - follower.velocity[axis]
                put(f"{joint}.difference", _text(difference))
                largest = max(largest, abs(difference))
                compared += 1
            if follower.have_velocity_error[axis]:
                put(
                    f"{joint}.follower_velocity_error",
                    _text(follower.velocity_error[axis]),
                )
        put("difference.axes_compared", str(compared))
        if compared > 0:
            put("difference.largest_absolute", _text(largest))

        if compared == ACTUATED_DOF:
            status.level = DiagnosticStatus.OK
            status.message = (
                "the shadow command and the follower's are both present on all six "
                f"axes; {verdict}"
            )
        else:
            status.level = DiagnosticStatus.WARN
            status.message = (
                "this cycle produced a shadow command and the follower's velocity "
                "is missing or stale on at least one axis, so those axes carry no "
                "difference; "
                if self._shadow_command_valid
                else "this cycle produced no shadow command at all, so there is "
                "nothing to compare; "
            ) + verdict
        message.status = [status]
        self._comparison_publisher.publish(message)

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

        payload, why = _payload_from_message(request.payload)
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

        # A payload step is a model change: the plan that was warm was a plan for
        # the old one, and the force state was seeded against the old weight.
        self._guess = None
        self._last_horizon = None
        self._tcp_states = None
        self._last_tcp_states = None
        self._seed_force = True
        com = payload.center_of_mass_k8_m
        response.message = (
            f"the payload is now {payload.mass_kg} kg at ({com[0]}, {com[1]}, "
            f"{com[2]}) m in K8, on every stage of the next horizon. The next solve "
            "cold starts"
        )
        self.get_logger().info(response.message)
        return response


class FollowerCommand:
    """What `/crane/controller_state` said the machine was doing, per axis."""

    def __init__(self) -> None:
        self.velocity = np.zeros(ACTUATED_DOF)
        self.have_velocity = [False] * ACTUATED_DOF
        self.velocity_error = np.zeros(ACTUATED_DOF)
        self.have_velocity_error = [False] * ACTUATED_DOF
        self.source = "none"
        self.age = 0.0
        self.complete = False


def _payload_from_message(message):
    """
    Read `crane_msgs/Payload` as the model's own.

    A point mass: the message carries no inertia.
    """
    payload = Payload()
    payload.valid = True
    if message.shape == message.SHAPE_NONE:
        return payload, ""
    if not np.isfinite(message.mass) or message.mass <= 0.0:
        return None, (
            f"a payload shape was declared with a mass of {message.mass} kg; an "
            "unknown payload is not a zero-mass payload, so declare SHAPE_NONE for "
            "an empty gripper instead"
        )
    com = np.array([message.com.x, message.com.y, message.com.z])
    if not np.all(np.isfinite(com)):
        return None, (
            "the payload's centre of mass carries a number that is not finite; it is "
            "the moment arm the whole gravity load hangs on"
        )
    payload.mass_kg = float(message.mass)
    payload.center_of_mass_k8_m = com
    return payload, ""


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
