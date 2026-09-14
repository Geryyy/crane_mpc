"""
What the cycle says about itself, as messages.

Pure marshalling: the solver-health report, the shadow comparison of user story
68, the TCP horizon drawn through the forward kinematics, and the payload read
off a service request. Every function here takes a `Cycle` and returns a
message; nothing publishes, so the node keeps its publishers and this keeps the
field-by-field detail out of it.
"""

from __future__ import annotations

import numpy as np
from crane_model import Frame, Payload
from crane_model import symbolic as cs
from crane_msgs.msg import SolverHealth, SupervisorStatus
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.time import Time

from . import problem
from .cycle import (
    ACTUATED_DOF,
    ACTUATED_INDICES,
    CONTROLLER_STATE_TOPIC,
    NANOSECONDS,
    PASSIVE_DOF,
    PASSIVE_INDICES,
    PLANNED_DOF,
    TCP_HORIZON_FRAME,
    TOOL_AXIS,
    Cycle,
)
from .solver import Outcome


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
    return health


def health_is_due(solves: int, decimation: int, last, health) -> bool:
    """
    Decimated, because the consumer is the 20 Hz supervisor.

    Never a change of verdict, though, which is what decimation exists not to
    drop.
    """
    on_cadence = (solves - 1) % max(1, decimation) == 0
    changed = last is None or (
        last.fault != health.fault or last.outcome != health.outcome
    )
    return on_cadence or changed


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
        # The two cost rows `crane_msgs/SolverHealth` has no field for. 119 made
        # them the interesting ones, so they are reported here rather than
        # computed and left invisible as the C++ left them.
        if cycle.cost_terms is not None:
            put("cost.lag", _text(cycle.cost_terms.lag))
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


def tcp_path(cycle: Cycle, model):
    """
    Draw the horizon through the forward kinematics, as `nav_msgs/Path`.

    Returns `(path, why)`. `(None, None)` is nothing to draw; a `why` that is
    not None is a kinematics refusal the node logs -- including the empty
    string, which is what an exception with no message says.
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
        q = np.zeros(cs.K_GENERALIZED_DOF)
        q[list(ACTUATED_INDICES[:PLANNED_DOF])] = state[
            cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF
        ]
        q[list(PASSIVE_INDICES)] = state[
            cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
        ]
        q[ACTUATED_INDICES[TOOL_AXIS]] = cycle.tool_position
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
