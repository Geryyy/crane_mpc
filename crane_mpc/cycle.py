"""
One MPC cycle, without ROS.

What `node.update()` does between reading the graph and writing to it: the
silence gates, the cadence anchor, the state one solve carries to the next and
`mpc` §6's fallback ladder. `node.py` reads the graph, hands the numbers in and
publishes what comes back.

Times are integer nanoseconds rather than `rclpy.time.Time`. The conversion is
exact both ways and it was the only thing rclpy carried into the cycle, so the
ladder, the anchor and the shift are constructible and tested without a node --
which is the whole reason this module exists.

The wire names live here rather than in `node.py` because the refusals quote
them: a gate that says "nothing is published on /crane/mpc/horizon" needs the
name, and `reports.py` needs the same strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from crane_model import symbolic as cs

from . import horizon as hz
from . import problem
from .solver import Outcome

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

#: Where the canonical eight carry the six actuated and the two passive.
ACTUATED_INDICES = cs.K_ACTUATED_ROWS
PASSIVE_INDICES = cs.K_PASSIVE_ROWS
ACTUATED_DOF = cs.K_ACTUATED_DOF
PLANNED_DOF = cs.K_PLANNED_DOF
PASSIVE_DOF = cs.K_PASSIVE_DOF
TOOL_AXIS = cs.K_TOOL_AXIS

NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class Silence:
    """Why nothing goes out. `why` is recorded, `warning` is what is logged."""

    why: str
    warning: str


@dataclass(frozen=True)
class Verdict:
    """What the ladder decided, and the sentence that says so."""

    text: str
    published: bool
    #: `""`, `"warn"` or `"error"` -- how loudly `text` is logged.
    severity: str = ""


@dataclass
class FollowerCommand:
    """What `/crane/controller_state` said the machine was doing, per axis."""

    velocity: np.ndarray = field(default_factory=lambda: np.zeros(ACTUATED_DOF))
    have_velocity: list = field(default_factory=lambda: [False] * ACTUATED_DOF)
    velocity_error: np.ndarray = field(default_factory=lambda: np.zeros(ACTUATED_DOF))
    have_velocity_error: list = field(default_factory=lambda: [False] * ACTUATED_DOF)
    source: str = "none"
    age: float = 0.0
    complete: bool = False


@dataclass
class Measurement:
    """
    What `/joint_states` last carried, with the age of each group of rows.

    An age of `None` is "nothing has arrived", which is a different refusal from
    "what arrived is stale".
    """

    q_a: np.ndarray
    dq_a: np.ndarray
    q_u: np.ndarray
    dq_u: np.ndarray
    actuated_age: float | None
    passive_age: float | None

    def refusal(self, max_state_age: float) -> str:
        if self.actuated_age is None:
            return f"no measured state has arrived on {JOINT_STATE_TOPIC}"
        if self.passive_age is None:
            return f"no passive state has arrived on {JOINT_STATE_TOPIC}"
        if self.actuated_age > max_state_age or self.passive_age > max_state_age:
            return "the measured state is older than max_state_age"
        return ""


def follower_input(follower: FollowerCommand, u_max) -> np.ndarray:
    """
    Take the follower's command as `u`.

    Under C3 `u` **is** a joint velocity, so it is clipped and never
    differenced into an acceleration.
    """
    command = np.zeros(cs.NU_PROGRESS)
    if not follower.complete:
        return command
    bound = np.asarray(u_max[:PLANNED_DOF], dtype=float)
    command[:PLANNED_DOF] = np.clip(follower.velocity[:PLANNED_DOF], -bound, bound)
    return command


class Cycle:
    """
    The MPC's own state between two solves, and the decisions it makes on it.

    Everything the node published last cycle is reachable from here; nothing
    here reaches back into the node.
    """

    def __init__(self, Ts: float, grid: hz.Grid, mode: str, delay: float) -> None:
        self.Ts = Ts
        self.grid = grid
        self.mode = mode
        self.delay = delay
        #: Set once the description has arrived and the problem is posed.
        self.ocp = None

        self.reference: hz.Knots | None = None
        self.reference_stamp_ns = 0
        self.reference_progress = 0.0
        self.reference_anchored = False
        self.next_first_knot_ns = 0
        self.cadence_anchored = False

        self.horizon: hz.Knots | None = None
        self.last_horizon: hz.Knots | None = None
        self.tcp_states: np.ndarray | None = None
        self.last_tcp_states: np.ndarray | None = None

        # The OCP's own rows that no sensor carries: C3's lagged command, its
        # force state and the progress pair. Seeded at the first solve.
        self.carried = np.zeros(cs.NX)
        self.carried[cs.X_PROGRESS_RATE] = problem.K_PROGRESS_RATE_REFERENCE
        self.seed_force = True
        self.last_input = np.zeros(cs.NU_PROGRESS)
        self.guess = None
        self.last_solution = None
        self.measured: np.ndarray | None = None
        self.x0: np.ndarray | None = None
        self.q_eq = np.zeros(PASSIVE_DOF)
        self.tool_position = 0.0

        self.solves = 0
        self.consecutive_failures = 0
        self.escalated = False
        self.applied_previous = False
        self.last_silence = ""
        self.cost_terms = None
        self.follower = FollowerCommand()
        self.shadow_command = np.zeros(ACTUATED_DOF)
        self.shadow_command_valid = False

    # -- what the node hands in --------------------------------------------------

    def begin(self) -> None:
        """Nothing has been compared or costed yet this cycle."""
        self.shadow_command_valid = False
        self.cost_terms = None

    def adopt_reference(self, reference: hz.Knots, stamp_ns: int) -> None:
        self.reference = reference
        self.reference_stamp_ns = stamp_ns
        self.reference_anchored = False

    def adopt_mode(self, requested: str) -> str:
        """Move to `requested` and cold start. Returns the mode left behind."""
        previous, self.mode = self.mode, requested
        self.forget_plan()
        self.consecutive_failures = 0
        self.escalated = False
        self.cadence_anchored = False
        self.last_input = np.zeros(cs.NU_PROGRESS)
        return previous

    def adopt_follower(self, follower: FollowerCommand, u_max) -> None:
        """In shadow the follower's command is what `u` was; in active it is not."""
        self.follower = follower
        if self.mode == "shadow":
            self.last_input = follower_input(follower, u_max)

    def payload_changed(self) -> None:
        """
        Drop what a payload step made stale.

        A payload step is a model change: the plan that was warm was a plan for
        the old payload, and the force state was seeded against its weight.
        """
        self.forget_plan()
        self.seed_force = True

    # -- the gates ---------------------------------------------------------------

    def gates(self, now_ns: int, max_clock_skew: float, max_reference_age: float):
        """
        Run the reference's three refusals, then the anchor and the resample.

        Returns the `Silence` that stops this cycle, or `None` and a horizon on
        `self.horizon`.
        """
        if self.reference is None or len(self.reference) == 0:
            return Silence(
                "no reference has arrived",
                f"No reference has arrived on {REFERENCE_TOPIC}, so nothing is "
                f"published on {HORIZON_TOPIC}. A horizon of zeros would be a plan "
                "to stop where the crane is not, and a stale one is worse.",
            )

        lead = (self.reference_stamp_ns - now_ns) / 1e9
        if lead > max_clock_skew:
            return Silence(
                "the reference is stamped in this node's future",
                f"The reference on {REFERENCE_TOPIC} is stamped {lead:.3f} s ahead "
                f"of this node against a {max_clock_skew} s bound, so nothing is "
                f"published on {HORIZON_TOPIC}.",
            )

        spent = self.reference_progress - float(self.reference.t[-1])
        if self.reference_anchored and spent > max_reference_age:
            return Silence(
                "the reference's plan has been spent past its end by more than "
                "max_reference_age",
                f"The plan on {REFERENCE_TOPIC} ended {spent:.3f} s of virtual time "
                f"ago against a {max_reference_age} s bound, so nothing is published "
                f"on {HORIZON_TOPIC}.",
            )

        self.anchor_cadence(now_ns)
        if not self.reference_anchored:
            self.reference_progress = (
                self.next_first_knot_ns - self.reference_stamp_ns
            ) / 1e9
            self.reference_anchored = True
        self.reference_progress = max(
            self.reference_progress, float(self.reference.t[0])
        )

        rejection, horizon = hz.resample(
            self.reference, self.reference_progress, self.grid
        )
        if rejection is not None:
            return Silence(
                str(rejection),
                f"The reference on {REFERENCE_TOPIC} could not be resampled onto the "
                f"horizon, so nothing is published on {HORIZON_TOPIC}: {rejection}.",
            )
        self.horizon = horizon
        return None

    def anchor_cadence(self, now_ns: int) -> None:
        """
        Decide when the first knot takes effect.

        `now` plus the transport delay, re-anchored whenever the cadence has
        drifted off the horizon.
        """
        anchor = now_ns + int(self.delay * NANOSECONDS)
        ceiling = now_ns + int((self.delay + self.grid.duration()) * NANOSECONDS)
        if (
            not self.cadence_anchored
            or self.next_first_knot_ns < now_ns
            or self.next_first_knot_ns > ceiling
        ):
            self.next_first_knot_ns = anchor
            self.cadence_anchored = True

    def read_state(self, measurement: Measurement, max_state_age: float):
        """
        `x` from `/joint_states` alone, or the `Silence` that stops the cycle.

        The actuator rows are the ones carried from the last accepted solve:
        nothing measures the command in flight or the force the cylinders are at.
        """
        why = measurement.refusal(max_state_age)
        if why:
            return Silence(
                why,
                f"Nothing was published on {HORIZON_TOPIC} because there is no state "
                f"to solve from: {why}.",
            )

        # One snapshot of the tool row for the whole cycle. The node read
        # `self._q_a` live at four points; under the single-threaded executor no
        # callback can land between them, so this is the same number.
        self.tool_position = float(measurement.q_a[TOOL_AXIS])
        x = self.carried.copy()
        x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF] = (
            measurement.q_a[:PLANNED_DOF]
        )
        x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF] = (
            measurement.dq_a[:PLANNED_DOF]
        )
        x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF] = measurement.q_u
        x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE_DOF] = (
            measurement.dq_u
        )
        x[cs.X_PROGRESS] = 0.0
        if self.seed_force:
            # C3 block 3 starts holding the machine's own weight: zero would be
            # the hydraulics switched off, and there is no force measurement.
            self.ocp.pin_tool(self.tool_position)
            x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED_DOF] = (
                self.ocp.static_hold_force(x)
            )
            self.carried = x.copy()
            self.seed_force = False
        self.measured = x
        return None

    def propagate(self):
        """Carry the measurement over the dead time, or refuse the cycle."""
        self.ocp.pin_tool(self.tool_position)
        try:
            x0 = self.ocp.propagate(self.measured, self.last_input)
        except Exception as error:
            return Silence(
                "the measured state could not be propagated to the instant the plan "
                "takes effect",
                f"The measured state could not be carried {self.delay} s forward, so "
                f"nothing is published on {HORIZON_TOPIC}: {error}.",
            )
        # The sway box is centred on where the tool is actually swinging, not on
        # the hanging pose. `weights.q_u` being zero and this centre justify each
        # other in a circle -- both are recorded, neither is this port's business.
        self.q_eq = x0[
            cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
        ].copy()
        self.x0 = x0
        return None

    # -- the solve and its ladder ------------------------------------------------

    def solve(self):
        """
        Solve. Returns `(solution, refusal)`; exactly one of the two is falsy.

        A refusal has already stopped the publisher, so the caller only reports
        it.
        """
        try:
            solution = self.ocp.solve(self.x0, self.horizon, self.q_eq, self.guess)
        except Exception as error:
            self.applied_previous = False
            self.consecutive_failures += 1
            self.solves += 1
            self.stay_silent_after_failure(
                "the optimal control problem did not return a usable horizon"
            )
            return None, str(error)

        self.solves += 1
        self.last_solution = solution
        self.guess = (
            None if solution.outcome is Outcome.FAILED else self.ocp.shifted(solution)
        )
        if solution.outcome is Outcome.CONVERGED:
            self.consecutive_failures = 0
            self.escalated = False
        else:
            self.consecutive_failures += 1
        return solution, ""

    def ladder(self, solution, max_consecutive_failures: int, solve_budget_s: float):
        """`mpc` §6: publish this solve, shift the last one, or hand control back."""
        destination = HORIZON_TOPIC if self.mode == "active" else SHADOW_HORIZON_TOPIC

        if solution.outcome is Outcome.CONVERGED:
            self.adopt_solution(solution)
            self.applied_previous = False
            return Verdict(
                "the solve converged inside the budget and its horizon was published "
                f"on {destination}",
                published=True,
            )

        if self.consecutive_failures >= max_consecutive_failures:
            self.escalated = True
            self.applied_previous = False
            text = (
                f"mpc §6's repeated-failure escalation: {self.consecutive_failures} "
                "consecutive solves did not converge against a ceiling of "
                f"{max_consecutive_failures} (acados last answered "
                f"{solution.status_word}, {solution.outcome}), so this node has "
                "stopped publishing and handed control back rather than shifting a "
                "plan it no longer believes"
            )
            self.stay_silent_after_failure(
                "mpc §6's repeated-failure escalation has stopped the publisher"
            )
            return Verdict(text, published=False, severity="error")

        if self.shift_previous_horizon():
            self.applied_previous = True
            return Verdict(
                f"acados answered {solution.status_word} ({solution.outcome}) after "
                f"{round(1000 * solution.solve_time_s)} ms against a "
                f"{round(1000 * solve_budget_s)} ms budget, so mpc §6's previous "
                "solution shifted by one step was published on "
                f"{destination} instead",
                published=True,
                severity="warn",
            )

        self.applied_previous = False
        text = (
            f"acados answered {solution.status_word} ({solution.outcome}) and there "
            "was no previous solution to shift, so nothing was published"
        )
        self.stay_silent_after_failure(
            "the solve did not converge and there is no previous solution to shift"
        )
        return Verdict(text, published=False, severity="warn")

    def adopt_solution(self, solution) -> None:
        """Write the solved horizon as the six actuated joints the wire carries."""
        states = solution.states
        self.horizon.q_a_ref[:, :PLANNED_DOF] = states[
            :, cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF
        ]
        self.horizon.dq_a_ref[:, :PLANNED_DOF] = states[
            :, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF
        ]
        # The tool is pinned, not planned: it holds where it was measured.
        self.horizon.q_a_ref[:, TOOL_AXIS] = self.tool_position
        self.horizon.dq_a_ref[:, TOOL_AXIS] = 0.0
        self.tcp_states = states.copy()

    def shift_previous_horizon(self) -> bool:
        """Shift the last horizon one knot on, duplicating its last: mpc §6."""
        if (
            self.last_horizon is None
            or self.last_tcp_states is None
            or len(self.last_horizon) != len(self.horizon)
            or len(self.last_tcp_states) != len(self.horizon)
        ):
            return False
        for field_name in ("q_a_ref", "dq_a_ref"):
            previous = getattr(self.last_horizon, field_name)
            current = getattr(self.horizon, field_name)
            current[:-1] = previous[1:]
            current[-1] = previous[-1]
        self.tcp_states = np.vstack(
            [self.last_tcp_states[1:], self.last_tcp_states[-1:]]
        )
        return True

    # -- what the cycle leaves behind --------------------------------------------

    def take_shadow_command(self) -> None:
        """Keep the second knot: what would have driven, had this been active."""
        self.shadow_command = self.horizon.dq_a_ref[1].copy()
        self.shadow_command_valid = True

    def advance(self, solution) -> None:
        """Carry what the next cycle starts from, now that a horizon went out."""
        self.last_horizon = self.horizon.copy()
        self.last_tcp_states = (
            None if self.tcp_states is None else self.tcp_states.copy()
        )
        if solution.outcome is Outcome.CONVERGED:
            self.last_input = solution.u0.copy()
        else:
            self.last_input = np.zeros(cs.NU_PROGRESS)
            self.last_input[:PLANNED_DOF] = self.horizon.dq_a_ref[1, :PLANNED_DOF]
        if solution.outcome is not Outcome.FAILED:
            # C3's own rows, carried: the command in flight and the force the
            # actuators are at. No sensor reports either.
            self.carried = solution.states[1].copy()

        self.next_first_knot_ns += int(self.Ts * NANOSECONDS)
        advance = solution.progress_advance
        self.reference_progress += (
            advance if np.isfinite(advance) and advance >= 0.0 else self.Ts
        )
        self.last_silence = ""

    def stay_silent(self, why: str) -> None:
        """Fall silent on a gate, which is not a solver failure and does not count."""
        self.consecutive_failures = 0
        self.escalated = False
        self.stay_silent_after_failure(why)

    def stay_silent_after_failure(self, why: str) -> None:
        """Nothing goes out, so nothing may be shifted next cycle either."""
        self.last_silence = why
        self.forget_plan()
        if self.reference_anchored:
            # One nominal interval spent while nobody was driving.
            self.reference_progress += self.Ts

    def forget_plan(self) -> None:
        """Drop the warm start and the horizon a shift would be taken from."""
        self.last_horizon = None
        self.tcp_states = None
        self.last_tcp_states = None
        self.guess = None
