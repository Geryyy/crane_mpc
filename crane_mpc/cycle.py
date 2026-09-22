"""
One MPC cycle, without ROS.

`node.update()` calls into this: silence gates, cadence anchor, state carried
solve to solve, the repeated-failure ladder. Nanosecond ints, not
`rclpy.time.Time`, keep it testable without a node. Wire names live here, not
in `node.py`, because refusals quote them and `reports.py` reuses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from crane_model import symbolic as cs

from . import horizon as hz
from .solver import Outcome

REFERENCE_TOPIC = "/crane/reference"
JOINT_STATE_TOPIC = "/joint_states"
PAYLOAD_ESTIMATE_TOPIC = "/crane/payload_estimate"
CONTROLLER_STATE_TOPIC = "/crane/controller_state"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
HORIZON_TOPIC = "/crane/mpc/horizon"
SOLVER_HEALTH_TOPIC = "/crane/mpc/solver_health"
SWAY_SETTLED_TOPIC = "/crane/sway_settled"
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
class VelocityCarry:
    """What `carry_actuated_velocity` did to `x_0`, per actuated axis."""

    #: True where this axis' velocity was taken from the model, not measured.
    carried: list = field(default_factory=lambda: [False] * ACTUATED_DOF)
    #: Model-carried minus measured. Zero on an axis that was not carried.
    divergence: np.ndarray = field(default_factory=lambda: np.zeros(ACTUATED_DOF))
    #: True where `|divergence|` crossed that axis' `dq_a_divergence_max`.
    diverged: list = field(default_factory=lambda: [False] * ACTUATED_DOF)
    any_diverged: bool = False


def carry_actuated_velocity(
    x: np.ndarray, carried, from_measurement, divergence_max
) -> VelocityCarry:
    """
    Put the model's own velocity into `x` on axes that opt out of feedback.

    `from_measurement[axis]` false uses `carried[axis]` instead; non-finite
    falls back to measurement. All-true (default) makes this a no-op.
    """
    report = VelocityCarry()
    for axis in range(PLANNED_DOF):
        if from_measurement[axis] or not np.isfinite(carried[axis]):
            continue
        row = cs.X_PLANNED_VELOCITY + axis
        difference = float(carried[axis]) - float(x[row])
        x[row] = carried[axis]
        report.carried[axis] = True
        report.divergence[axis] = difference
        report.diverged[axis] = abs(difference) > float(divergence_max[axis])
    report.any_diverged = any(report.diverged)
    return report


@dataclass
class Measurement:
    """What `/joint_states` last carried; `None` age means nothing arrived, not stale."""

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
    """`u` is the follower's command: a clipped joint velocity (C3), never differenced."""
    command = np.zeros(cs.NU_PROGRESS)
    if not follower.complete:
        return command
    bound = np.asarray(u_max[:PLANNED_DOF], dtype=float)
    command[:PLANNED_DOF] = np.clip(follower.velocity[:PLANNED_DOF], -bound, bound)
    return command


def watch_progress(held, advance, nominal, wall_dt, min_rate, max_stall_time):
    """
    Fold one published cycle into the wall-clock stall watch. `(held, stalled)`.

    Checked against the wall clock, not plan time: the optimizer can choose
    zero progress and never trip max_reference_age otherwise. A cycle under
    `min_rate` adds wall-clock duration, capped at the timeout; any other
    cycle clears it -- flags sustained near-zero progress, not a slowdown.
    """
    step = wall_dt if wall_dt > 0.0 else 0.0
    progressing = nominal > 0.0 and advance >= min_rate * nominal
    held = 0.0 if progressing else min(held + step, max_stall_time)
    # The cap bounds the count; `>=` keeps it reported for as long as it lasts.
    return held, not progressing and held >= max_stall_time


class Cycle:
    """The MPC's own state between two solves; nothing here reaches back into the node."""

    def __init__(
        self,
        Ts: float,
        grid: hz.Grid,
        mode: str,
        delay: float,
        dq_a_feedback=None,
        dq_a_divergence_max=None,
    ) -> None:
        self.Ts = Ts
        self.grid = grid
        self.mode = mode
        self.delay = delay
        #: Per axis, whether x_0's velocity is measured or model-carried.
        self.dq_a_feedback = (
            [True] * ACTUATED_DOF if dq_a_feedback is None else list(dq_a_feedback)
        )
        self.dq_a_divergence_max = (
            np.full(ACTUATED_DOF, np.inf)
            if dq_a_divergence_max is None
            else np.asarray(dq_a_divergence_max, dtype=float)
        )
        #: Set once the description has arrived and the problem is posed.
        self.ocp = None

        self.reference: hz.Knots | None = None
        self.reference_stamp_ns = 0
        self.reference_progress = 0.0
        self.reference_anchored = False
        # `watch_progress` state; only published cycles feed it.
        self.progress_held = 0.0
        self.progress_stalled = False
        self.progress_mark_ns = 0
        self.progress_marked = False
        self.next_first_knot_ns = 0
        self.cadence_anchored = False

        self.horizon: hz.Knots | None = None
        self.last_horizon: hz.Knots | None = None
        self.tcp_states: np.ndarray | None = None
        self.last_tcp_states: np.ndarray | None = None

        # OCP rows no sensor carries: lagged command, force state, progress pair.
        self.carried = np.zeros(cs.NX)
        # Path parameter now, so rest is where a fresh cycle starts from.
        self.carried[cs.X_PROGRESS_RATE] = 0.0
        self.seed_force = True
        # Previous cycle's propagated velocity, read back as the carry -- not a
        # second integrator. `None` until propagated once.
        self.dq_a_carried: np.ndarray | None = None
        self.velocity_carry = VelocityCarry()
        self.last_input = np.zeros(cs.NU_PROGRESS)
        # What this node put in flight, newest first. Dead time is 1.5 intervals
        # on the shipped 40 ms / 60 ms, so two commands span the propagation
        # window. Kept as deep as `Ocp.replay` needs.
        self.applied_inputs: list = []
        self.guess = None
        #: A preparation standing for this cycle, and the resampled reference it
        #: was linearised about. The horizon is what gets re-checked: it is the
        #: only thing the preparation bakes that a new plan, a re-anchored
        #: cadence or a slipped clock can move underneath it.
        self.prepared = False
        self.prepared_horizon: hz.Knots | None = None
        self.last_solution = None
        self.measured: np.ndarray | None = None
        self.x0: np.ndarray | None = None
        self.q_eq = np.zeros(PASSIVE_DOF)
        self.tool_position = 0.0

        self.solves = 0
        # Every cycle, silent ones included -- `solves` freezes on a silent
        # cycle, wrong clock for a stream reporting the machine, not the solve.
        self.cycles = 0
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
        self.cycles += 1
        self.shadow_command_valid = False
        self.cost_terms = None

    def adopt_reference(self, reference: hz.Knots, stamp_ns: int) -> None:
        self.reference = reference
        self.reference_stamp_ns = stamp_ns
        self.reference_anchored = False
        # Stall watch is per plan: a new reference is the re-plan a stall asks for.
        self.forget_stall()
        # A preparation is a linearisation of the plan that was; this is another
        # plan. `forget_plan` is not called here -- the warm start survives a
        # re-plan -- so the preparation has to be dropped on its own.
        self.forget_preparation()

    def adopt_mode(self, requested: str) -> str:
        """Move to `requested` and cold start. Returns the mode left behind."""
        previous, self.mode = self.mode, requested
        self.forget_plan()
        # In shadow this node's plan drives nothing, so progress may sit near
        # zero; carried to active that would falsely report a stall.
        self.forget_stall()
        self.consecutive_failures = 0
        self.escalated = False
        self.cadence_anchored = False
        self.last_input = np.zeros(cs.NU_PROGRESS)
        # Rest of what was in flight: replaying it would carry `x_0` under another commander's plan.
        self.applied_inputs.clear()
        return previous

    def adopt_follower(self, follower: FollowerCommand, u_max) -> None:
        """In shadow the follower's command is what `u` was; in active it is not."""
        self.follower = follower
        if self.mode == "shadow":
            self.last_input = follower_input(follower, u_max)

    def payload_changed(self) -> None:
        """Drop what a payload step made stale: the warm plan, the force state seeded for it."""
        self.forget_plan()
        self.seed_force = True

    # -- the gates ---------------------------------------------------------------

    def gates(self, now_ns: int, max_clock_skew: float, max_reference_age: float):
        """
        Run the reference's three refusals, then the anchor and the resample.

        Returns the `Silence` that stops this cycle, or `None` with `self.horizon` set.
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
        """Decide when the first knot takes effect: `now` plus transport delay."""
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

        Actuator force/command rows come from the last accepted solve -- unmeasured.
        """
        why = measurement.refusal(max_state_age)
        if why:
            return Silence(
                why,
                f"Nothing was published on {HORIZON_TOPIC} because there is no state "
                f"to solve from: {why}.",
            )

        # One snapshot of the tool row: single-threaded executor, no callback
        # lands between reads.
        self.tool_position = float(measurement.q_a[TOOL_AXIS])
        x = self.carried.copy()
        x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED_DOF] = (
            measurement.q_a[:PLANNED_DOF]
        )
        x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF] = (
            measurement.dq_a[:PLANNED_DOF]
        )
        # `dq_a_feedback[axis]` false: velocity is the model's, not the
        # measurement just written. All-true ships, so this is inert by default.
        self.velocity_carry = (
            VelocityCarry()
            if self.dq_a_carried is None
            else carry_actuated_velocity(
                x, self.dq_a_carried, self.dq_a_feedback, self.dq_a_divergence_max
            )
        )
        x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF] = measurement.q_u
        x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE_DOF] = (
            measurement.dq_u
        )
        x[cs.X_PROGRESS] = 0.0
        if self.seed_force:
            # C3 seeds holding the machine's weight, not zero (hydraulics off) --
            # no force is measured.
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
        # Recorded here, not beside `last_input` writes: shadow mode writes it
        # twice per cycle but only one command applies here.
        self.applied_inputs.insert(0, self.last_input.copy())
        depth = self.ocp.replay[0].age + 1 if self.ocp.replay else 1
        del self.applied_inputs[depth:]
        try:
            x0 = self.ocp.propagate_applied(self.measured, self.applied_inputs)
        except Exception as error:
            return Silence(
                "the measured state could not be propagated to the instant the plan "
                "takes effect",
                f"The measured state could not be carried {self.delay} s forward, so "
                f"nothing is published on {HORIZON_TOPIC}: {error}.",
            )
        # Sway box and offset residual centred on where the tool actually
        # swings, not the hanging pose -- issue 137 measured centre worth 3-4x
        # peak sway, weight worth nothing (issue 049).
        self.q_eq = x0[
            cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
        ].copy()
        # Next cycle's carry on an opted-out axis: this propagation's velocity,
        # one cycle old -- inside what dq_a_divergence_max watches.
        self.dq_a_carried = x0[
            cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF
        ].copy()
        self.x0 = x0
        return None

    # -- the solve and its ladder ------------------------------------------------

    def solve(self):
        """
        Solve. Returns `(solution, refusal)`; exactly one is falsy.

        A refusal has already stopped the publisher; the caller only reports it.

        Takes the feedback phase alone when last cycle prepared this one on the
        same plan; otherwise the whole solve, which is what every cycle did
        before the split existed.
        """
        on_preparation = self.prepared and self.preparation_still_applies()
        self.forget_preparation()
        try:
            if on_preparation:
                solution = self.ocp.feedback(self.x0)
            else:
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

    def preparation_still_applies(self) -> bool:
        """
        Whether the plan the preparation linearised is the plan this cycle has.

        The reference is compared, not the state: the preparation is *meant* to
        stand on a predicted state, and the feedback phase carries the measured
        one in through the initial-state bound. What it may not do is answer on
        another plan -- a re-plan, a re-anchored cadence or a slipped clock all
        move the resampled knots, and any of them makes the linearisation wrong
        rather than merely stale.
        """
        if self.prepared_horizon is None or self.horizon is None:
            return False
        return all(
            np.array_equal(
                getattr(self.prepared_horizon, block), getattr(self.horizon, block)
            )
            for block in ("t", "q_a_ref", "dq_a_ref", "ddq_a_ref")
        )

    def prepare_next(self, solution) -> None:
        """
        Linearise the next cycle now, about the state this one predicts for it.

        `states[1]` is that prediction already: the optimizer's own one step
        ahead, on the plan that is about to go out, so no second integration is
        needed. Runs after `advance`, which is what moves `reference_progress`
        and the cadence anchor to where the next cycle will resample them.

        Best effort throughout -- anything unexpected leaves no preparation, and
        the next cycle solves whole.
        """
        if self.ocp is None or not getattr(self.ocp, "split_rti", False):
            return
        if solution.outcome is Outcome.FAILED or self.guess is None:
            return
        if self.reference is None or not self.reference_anchored:
            return
        rejection, horizon = hz.resample(
            self.reference, self.reference_progress, self.grid
        )
        if rejection is not None:
            return
        predicted = solution.states[1]
        q_eq = predicted[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF]
        try:
            self.ocp.prepare(predicted, horizon, q_eq, self.guess)
        except Exception:
            self.forget_preparation()
            return
        self.prepared = True
        self.prepared_horizon = horizon

    def ladder(self, solution, max_consecutive_failures: int, solve_budget_s: float):
        """Publish this solve, shift the last one, or hand control back."""
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
            # Handing back happens once; later cycles just wait for a believable
            # solve. SolverHealth keeps FAULT_SOLVER either way -- only log
            # severity changes, so it doesn't bury the explaining cycle.
            handing_back = not self.escalated
            self.escalated = True
            self.applied_previous = False
            spent = (
                f"{round(1000 * solution.solve_time_s)} ms against a "
                f"{round(1000 * solve_budget_s)} ms budget"
            )
            if handing_back:
                text = (
                    f"mpc §6's repeated-failure escalation: "
                    f"{self.consecutive_failures} consecutive solves did not "
                    f"converge against a ceiling of {max_consecutive_failures} "
                    f"(acados last answered {solution.status_word}, "
                    f"{solution.outcome}, {spent}), so this node has stopped "
                    "publishing and handed control back rather than shifting a "
                    "plan it no longer believes"
                )
            else:
                text = (
                    f"mpc §6's escalation still holds: {self.consecutive_failures} "
                    f"consecutive failures, the last one {spent}. Nothing goes out "
                    f"on {destination} until a solve converges or the mode changes"
                )
            self.stay_silent_after_failure(
                "mpc §6's repeated-failure escalation has stopped the publisher"
            )
            return Verdict(
                text,
                published=False,
                severity="error" if handing_back else "warn",
            )

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
        """Write the solved horizon as the canonical eight the wire carries."""
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
        # Sway that JTC tracks but doesn't command -- OCP's own prediction.
        self.horizon.q_u_ref[:] = states[
            :, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
        ]
        self.horizon.dq_u_ref[:] = states[
            :, cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE_DOF
        ]
        # One input short of a knot: N intervals between N+1 knots. The last
        # interval's command is held, as a zero-order hold already holds it.
        commands = solution.inputs[:, : cs.NU]
        if commands.size:
            self.horizon.u[:-1, :PLANNED_DOF] = commands
            self.horizon.u[-1, :PLANNED_DOF] = commands[-1]
        self.horizon.u[:, TOOL_AXIS] = 0.0
        self.tcp_states = states.copy()

    def shift_previous_horizon(self) -> bool:
        """Shift the last horizon one knot on, duplicating its last knot."""
        if (
            self.last_horizon is None
            or self.last_tcp_states is None
            or len(self.last_horizon) != len(self.horizon)
            or len(self.last_tcp_states) != len(self.horizon)
        ):
            return False
        for field_name in ("q_a_ref", "dq_a_ref", "q_u_ref", "dq_u_ref", "u"):
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

    def advance(
        self,
        solution,
        now_ns: int,
        min_progress_rate: float,
        max_stall_time: float,
    ) -> None:
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
            # C3's own rows carried: command in flight, actuator force -- unmeasured.
            self.carried = solution.states[1].copy()

        self.next_first_knot_ns += int(self.Ts * NANOSECONDS)
        # `s` is a path parameter over the horizon's own window, not seconds:
        # one unit of it is the whole window, so `duration()` converts. It was
        # virtual time until the progress state became the path parameter
        # (86ec918) and this consumer was not moved with it -- the plan then
        # advanced at `s` per cycle, a factor `duration()` short, and only ran
        # at all because the old cost pinned `s` to its ceiling every cycle.
        #
        # Only a converged solve gets to say how much plan it bought. On the
        # other rungs `ladder` published the previous horizon shifted by one
        # knot, so what the machine consumes is one interval of *that* plan --
        # the same nominal interval `stay_silent_after_failure` charges. Reading
        # `s` off a refused iterate instead let the window run up to 0.21 s of
        # plan per cycle ahead of the machine, and `max_consecutive_failures`
        # allows five in a row before the publisher stops.
        if solution.outcome is Outcome.CONVERGED:
            advance = solution.progress_advance * self.grid.duration()
            spent = advance if np.isfinite(advance) and advance >= 0.0 else self.Ts
        else:
            spent = self.Ts
        self.reference_progress += spent

        # Liveness runs on the wall clock deliberately -- the one quantity the
        # optimizer doesn't choose. Reported, not recovered from: horizon keeps
        # going out, machine just isn't moving through the plan. `Failed`
        # clears the count on purpose -- that's `max_consecutive_failures`'s job.
        wall_dt = (
            (now_ns - self.progress_mark_ns) / 1e9 if self.progress_marked else 0.0
        )
        self.progress_mark_ns = now_ns
        self.progress_marked = True
        self.progress_held, self.progress_stalled = watch_progress(
            self.progress_held,
            spent,
            self.Ts,
            wall_dt,
            min_progress_rate,
            max_stall_time,
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
        # Stall count survives a silence; only the wall-clock mark doesn't.
        # Charging wall time here would call a dead producer a stopped machine;
        # clearing the count would hide a dropped `/joint_states` sample.
        self.progress_marked = False

    def forget_stall(self) -> None:
        """Drop the stall watch: what it was gathered about is gone."""
        self.progress_held = 0.0
        self.progress_stalled = False
        self.progress_marked = False

    def forget_preparation(self) -> None:
        """Drop a standing preparation: next cycle solves whole, as it used to."""
        self.prepared = False
        self.prepared_horizon = None

    def forget_plan(self) -> None:
        """Drop the warm start and the horizon a shift would be taken from."""
        # Every break in output -- gate, escalation, payload step, mode change,
        # failure -- reaches here, and a preparation never outlives one of them.
        self.forget_preparation()
        # Carry goes with them: a break in output (gate, escalation, payload
        # step, mode change) is reactivation, so next cycle re-seeds from the
        # measurement, not a state that kept integrating unmanned.
        self.dq_a_carried = None
        self.velocity_carry = VelocityCarry()
        self.last_horizon = None
        self.tcp_states = None
        self.last_tcp_states = None
        self.guess = None
