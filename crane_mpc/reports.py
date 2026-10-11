"""
What the cycle says about itself, as messages.

Pure marshalling: solver health, shadow comparison, TCP horizon, payload, pump flow. Every
function takes a `Cycle` and returns a message; nothing publishes here.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
from crane_model import Frame, Payload
from crane_model import symbolic as cs
from crane_msgs.msg import SolverHealth, SupervisorStatus, SwaySettled
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.time import Time

from . import bspline, problem
from .cycle import (
    ACTUATED_DOF,
    ACTUATED_INDICES,
    CONTROLLER_STATE_TOPIC,
    JOINT_STATE_TOPIC,
    NANOSECONDS,
    PASSIVE_DOF,
    PASSIVE_INDICES,
    PLANNED_DOF,
    TCP_HORIZON_FRAME,
    TOOL_AXIS,
    Cycle,
    FollowerCommand,
)
from .solver import CostTerms, Outcome


def _text(value: float) -> str:
    """Format a double that round-trips: a comparison is read off these."""
    return f"{value:.17g}"


def solver_health(cycle: Cycle, solution, why: str, joints, budget: float, stamp):
    """`crane_msgs/SolverHealth` for one cycle, verdict and all."""
    health = SolverHealth()
    health.header.stamp = stamp
    health.joint_names = list(joints)
    health.solve_budget = budget
    health.applied_previous_solution = cycle.applied_previous
    health.message = f"{cycle.mode}: {why}"
    if solution is None:
        health.outcome = SolverHealth.SOLVE_UNKNOWN
        health.fault = SupervisorStatus.FAULT_SOLVER
        return health

    # `LATE` and `BUDGET_EXCEEDED` share the wire value: both are "too slow",
    # the message has no fourth constant and crane_msgs is not this package's to
    # extend. What separates them from a *failure* is this field against
    # `SOLVE_FAILED`, and what separates them from each other is `status` (0 on
    # a late cycle, since it converged) and `message`, which carries both clocks
    # on every rung -- `solve_time` is acados' own and is inside the budget on
    # every late cycle, so counting overruns off it reads 0 while a fifth of
    # cycles publish a shifted horizon (issue 226 §4, issue 230).
    health.outcome = {
        Outcome.CONVERGED: SolverHealth.SOLVE_CONVERGED,
        Outcome.BUDGET_EXCEEDED: SolverHealth.SOLVE_BUDGET_EXCEEDED,
        Outcome.LATE: SolverHealth.SOLVE_BUDGET_EXCEEDED,
        Outcome.FAILED: SolverHealth.SOLVE_FAILED,
    }[solution.outcome]
    # A stalled plan is a *converged* solve (answering "wait"); without this it reads
    # healthy forever. Its own code: the solve is converged and on time, so
    # FAULT_SOLVER would name the wrong component (issue 273). A non-converged solve
    # outranks the stall -- the solver really did fail, and that is the actionable
    # half. No node here acts on either yet. `health_is_due` exempts a changed verdict
    # from decimation, so a new stall goes out immediately.
    if solution.outcome is not Outcome.CONVERGED:
        health.fault = SupervisorStatus.FAULT_SOLVER
    elif cycle.progress_stalled:
        health.fault = SupervisorStatus.FAULT_PLAN_STALLED
    else:
        health.fault = SupervisorStatus.FAULT_NONE
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
    return health


def report_is_due(count: int, decimation: int, changed: bool) -> bool:
    """Decimated (consumer at 20 Hz, cycle at 25), but never a change of verdict."""
    return changed or (count - 1) % max(1, decimation) == 0


def health_is_due(solves: int, decimation: int, last, health) -> bool:
    return report_is_due(
        solves,
        decimation,
        last is None or last.fault != health.fault or last.outcome != health.outcome,
    )


def settled_is_due(cycles: int, decimation: int, last, verdict) -> bool:
    """Count in cycles and not in solves: this stream reports the machine."""
    return report_is_due(
        cycles, decimation, last is None or last.settled != verdict.settled
    )


def sway_settled(dq_u, age: float | None, max_state_age: float, dq_u_settled, stamp):
    """
    `crane_msgs/SwaySettled` for one cycle: has the load stopped swinging.

    Reported, never acted on (damping/stopping/refusing is the task layer's). Every
    non-measurement path is SETTLED_UNKNOWN with NaN rates: a degraded estimate
    read as settled is a grip descending onto a swinging block. `age` is the
    **rate**'s age, not the pose's -- a still crane's zeros must not read as fresh.
    """
    verdict = SwaySettled()
    verdict.header.stamp = stamp
    verdict.velocity = [float("nan")] * PASSIVE_DOF
    verdict.settled = SwaySettled.SETTLED_UNKNOWN

    dq_u = np.asarray(dq_u, dtype=float).reshape(-1)
    if age is None:
        verdict.message = (
            f"no passive sway rate has arrived on {JOINT_STATE_TOPIC}, so whether "
            "the load is swinging is unknown rather than settled"
        )
        return verdict
    if age > max_state_age:
        verdict.message = (
            f"the passive sway rate is {age:.3f} s old against a max_state_age of "
            f"{max_state_age} s, so it is not a reading of the machine now"
        )
        return verdict
    if dq_u.size != PASSIVE_DOF or not np.all(np.isfinite(dq_u)):
        verdict.message = (
            f"the sway rate on {JOINT_STATE_TOPIC} is not two finite numbers, so "
            "there is nothing to compare against dq_u_settled"
        )
        return verdict

    verdict.velocity = [float(rate) for rate in dq_u]
    bound = np.asarray(dq_u_settled, dtype=float)
    settled = bool(np.all(np.abs(dq_u) <= bound))
    verdict.settled = SwaySettled.SETTLED_YES if settled else SwaySettled.SETTLED_NO
    verdict.message = (
        f"the measured sway rate {np.abs(dq_u).round(4).tolist()} rad/s is "
        f"{'inside' if settled else 'outside'} dq_u_settled "
        f"{bound.tolist()} on the passive pair, measured {age:.3f} s ago"
    )
    return verdict


def follower_command(state, joints, now_ns, max_clock_skew, max_state_age):
    """
    Read what the velocity controller is doing, per axis, off one controller state.

    `state is None` -- no message yet, or a mode that does not compare -- is the
    default `FollowerCommand`: nothing taken, nothing compared. Velocity comes
    from `output` and falls back to `reference`; `source` names which, or why not.
    """
    follower = FollowerCommand()
    if state is None:
        return follower
    follower.age = (now_ns - Time.from_msg(state.header.stamp).nanoseconds) / 1e9
    if -follower.age > max_clock_skew:
        follower.source = "future"
        return follower
    if follower.age > max_state_age:
        follower.source = "stale"
        return follower

    names = list(state.joint_names)
    sources = set()
    complete = True
    for axis, joint in enumerate(joints):
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


def fill_cost_terms(health, terms) -> None:
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


def cost_terms(terms, stamp) -> DiagnosticArray:
    """
    Report the cost rows `SolverHealth` cannot carry, beside the eight it can.

    `cost_term` is `float64[8]`, so lag, tool, progress and C4's rate cost went
    unpublished in active mode -- `~/shadow_comparison` has the first three, shadow
    only. Same stamp and same decimation as the health message they belong with, and
    `None` (no solve to split) reports zeros, as `cost_term` does. PlotJuggler names
    these `~/cost_terms/cost/<key>`, so `hardware_id` stays empty.
    """
    terms = terms if terms is not None else CostTerms()
    message = DiagnosticArray()
    message.header.stamp = stamp
    status = DiagnosticStatus(name="cost", level=DiagnosticStatus.OK)
    status.message = "the cost rows SolverHealth/cost_term has no slot for"
    status.values = [
        KeyValue(key="lag", value=_text(terms.lag)),
        KeyValue(key="tool", value=_text(terms.tool)),
        KeyValue(key="progress", value=_text(terms.progress)),
        KeyValue(key="du", value=_text(terms.du)),
    ]
    message.status = [status]
    return message


def shadow_comparison(cycle: Cycle, verdict: str, solution, joints, budget, stamp):
    """Judge the shadow command against what drove, per user story 68."""
    message = DiagnosticArray()
    message.header.stamp = stamp
    status = DiagnosticStatus()
    status.name = "crane_mpc: the shadow solution against what drove"
    status.hardware_id = problem.TOOL
    values = status.values
    follower = cycle.follower

    def put(key: str, value: str) -> None:
        values.append(KeyValue(key=key, value=value))

    put("mode", cycle.mode)
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
        "true" if cycle.applied_previous else "false",
    )
    put("solve.escalated", "true" if cycle.escalated else "false")
    if solution is not None:
        put("solve.time", _text(solution.solve_time_s))
        put("solve.budget", _text(budget))
        put("constraint.used_slack", "true" if solution.used_slack else "false")
        put("constraint.slack_penalty", _text(solution.slack_penalty))
        put("constraint.sway", _text(solution.violation.q_u))
        put("constraint.sway_rate", _text(solution.violation.dq_u))
        put("constraint.cylinder_force", _text(solution.violation.cylinder_force))
        put("constraint.pump_flow", _text(solution.violation.pump_flow))
        # The cost rows `SolverHealth` has no field for (issue 119 made them interesting).
        if cycle.cost_terms is not None:
            put("cost.lag", _text(cycle.cost_terms.lag))
            put("cost.tool", _text(cycle.cost_terms.tool))
            put("cost.progress", _text(cycle.cost_terms.progress))

    compared = 0
    largest = 0.0
    for axis, joint in enumerate(joints):
        if cycle.shadow_command_valid:
            put(f"{joint}.shadow_velocity", _text(cycle.shadow_command[axis]))
        if follower.have_velocity[axis]:
            put(f"{joint}.follower_velocity", _text(follower.velocity[axis]))
        if cycle.shadow_command_valid and follower.have_velocity[axis]:
            difference = cycle.shadow_command[axis] - follower.velocity[axis]
            put(f"{joint}.difference", _text(difference))
            largest = max(largest, abs(difference))
            compared += 1
        if follower.have_velocity_error[axis]:
            put(
                f"{joint}.follower_velocity_error",
                _text(follower.velocity_error[axis]),
            )
        # Machine-readable divergence, open-loop-velocity axes: OCP state vs machine state.
        if cycle.velocity_carry.carried[axis]:
            put(
                f"{joint}.dq_a_divergence", _text(cycle.velocity_carry.divergence[axis])
            )
            put(
                f"{joint}.dq_a_diverged",
                "true" if cycle.velocity_carry.diverged[axis] else "false",
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
            if cycle.shadow_command_valid
            else "this cycle produced no shadow command at all, so there is "
            "nothing to compare; "
        ) + verdict
    message.status = [status]
    return message


def _tcp_q(state, planned, tool):
    q = np.zeros(cs.GENERALIZED_DOF)
    q[list(ACTUATED_INDICES[:PLANNED_DOF])] = planned
    q[list(PASSIVE_INDICES)] = state[
        cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
    ]
    q[ACTUATED_INDICES[TOOL_AXIS]] = tool
    return q


# Half-width of the path window searched around the solve's progress, and its samples.
TCP_ERROR_WINDOW = 0.15
TCP_ERROR_SAMPLES = 31


def tcp_path_error(cycle: Cycle, model):
    """
    Distance of the measured TCP to the planned TCP path, metres; `(None, why)` on refusal.

    Geometric, not timed: the nearest point of the polyline `FK(c(s))` over a window
    around the solve's progress, so neither progress lag nor dead-time propagation
    enters. Both sides use the measured passive angles: `/crane/reference` carries no
    planned sway, so this is the actuated chain's deviation.
    """
    if model is None or cycle.path is None or cycle.measured is None:
        return None, None
    x = cycle.measured
    origin = cycle.path.origin
    s = np.linspace(
        max(origin - TCP_ERROR_WINDOW, 0.0),
        min(origin + TCP_ERROR_WINDOW, 1.0),
        TCP_ERROR_SAMPLES,
    )
    knots = bspline.value(s, cycle.path.control, problem.PATH_KNOTS)
    try:
        p = model.forward_kinematics(
            _tcp_q(x, x[cs.X_PLANNED_POSITION :][:PLANNED_DOF], cycle.tool_position),
            Frame.MOUNTING_BASE,
            Frame.TCP,
        ).position_m
        line = np.array(
            [
                model.forward_kinematics(
                    _tcp_q(x, knot, cycle.tool_position), Frame.MOUNTING_BASE, Frame.TCP
                ).position_m
                for knot in knots
            ]
        )
    except Exception as error:
        return None, str(error)
    a, d = line[:-1], np.diff(line, axis=0)
    t = np.clip(
        np.einsum("ij,ij->i", p - a, d) / np.maximum((d * d).sum(1), 1e-12), 0, 1
    )
    return float(np.min(np.linalg.norm(a + t[:, None] * d - p, axis=1))), None


def tcp_path(cycle: Cycle, model):
    """
    Draw the horizon through forward kinematics as `nav_msgs/Path`.

    Returns `(path, why)`: `(None, None)` is nothing to draw, `why` a refusal.
    """
    if (
        model is None
        or cycle.horizon is None
        or cycle.tcp_states is None
        or len(cycle.tcp_states) != len(cycle.horizon)
    ):
        return None, None
    first_knot = Time(nanoseconds=cycle.next_first_knot_ns)
    path = Path()
    path.header.frame_id = TCP_HORIZON_FRAME
    path.header.stamp = first_knot.to_msg()
    for index, state in enumerate(cycle.tcp_states):
        planned = state[cs.X_PLANNED_POSITION :][:PLANNED_DOF]
        q = _tcp_q(state, planned, cycle.tool_position)
        try:
            pose = model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        except Exception as error:
            return None, str(error)
        stamped = PoseStamped()
        stamped.header.frame_id = TCP_HORIZON_FRAME
        stamped.header.stamp = (
            first_knot + Duration(nanoseconds=int(cycle.horizon.t[index] * NANOSECONDS))
        ).to_msg()
        stamped.pose.position.x = float(pose.position_m[0])
        stamped.pose.position.y = float(pose.position_m[1])
        stamped.pose.position.z = float(pose.position_m[2])
        stamped.pose.orientation.x = float(pose.orientation_xyzw[0])
        stamped.pose.orientation.y = float(pose.orientation_xyzw[1])
        stamped.pose.orientation.z = float(pose.orientation_xyzw[2])
        stamped.pose.orientation.w = float(pose.orientation_xyzw[3])
        path.poses.append(stamped)
    return path, None


def payload_from_message(message):
    """Read `crane_msgs/Payload` as the model's own: a point mass, no inertia."""
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


class PumpFlow:
    """
    `Q_i = A^±_i |J_c,i(q) dq_i|` on all six actuated axes, from the measurement.

    Constraint 7's summands without its smoothing, and with the tool the OCP
    leaves out (it is held still there, not here -- nomenclature §8).
    """

    def __init__(self, hydraulics: dict, constants=None):
        constants = constants if constants is not None else cs.load_constants()
        q2, q3 = ca.SX.sym("q2"), ca.SX.sym("q3")
        self._ratio = ca.Function(
            "J_c", [q2, q3], [ca.vertcat(*cs.jacobian_diagonal(constants, q2, q3))]
        )
        areas = cs.axis_areas(constants)
        self._extend = np.array([area.a_eff_pos for area in areas])
        self._retract = np.array([area.a_eff_neg for area in areas])
        self.q_pump_max = float(hydraulics["pump_flow_max"])
        #: Where constraint 7 binds, as a share: the planning factor, not 1.
        self.planned_share = float(hydraulics["pump_flow_planning_factor"])

    def __call__(self, q_a, dq_a) -> np.ndarray:
        ratio = np.asarray(self._ratio(q_a[1], q_a[2]), dtype=float).reshape(-1)
        velocity = ratio * np.asarray(dq_a, dtype=float)
        return np.where(velocity >= 0.0, self._extend, self._retract) * np.abs(velocity)


def pump_flow(flow: np.ndarray, pump: PumpFlow, joints, stamp) -> DiagnosticArray:
    """`Q_pump`, its share of `Q_pump_max`, the share the MPC binds at, each axis' `Q`."""
    message = DiagnosticArray()
    message.header.stamp = stamp
    status = DiagnosticStatus(name="pump", level=DiagnosticStatus.OK)
    total = float(np.sum(flow))
    status.values = [
        KeyValue(key="Q_pump", value=_text(total)),
        KeyValue(key="Q_pump_share", value=_text(total / pump.q_pump_max)),
        KeyValue(key="Q_pump_share_planned", value=_text(pump.planned_share)),
    ] + [KeyValue(key=f"Q.{joint}", value=_text(q)) for joint, q in zip(joints, flow)]
    message.status = [status]
    return message
