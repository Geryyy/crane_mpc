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
    Put the model's own velocity into `x` on the axes that opt out of feedback.

    `from_measurement[axis]` false overwrites that planned velocity row with
    `carried[axis]`, the previous cycle's propagated value, instead of the
    measurement. Positions and both passive rows are never touched, and a carry
    that is not finite falls back to the measurement rather than poisoning `x`.

    All-true is the shipped default and then this writes nothing at all, which
    is what makes the feature bit-identical until it is configured.
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


def watch_progress(held, advance, nominal, wall_dt, min_rate, max_stall_time):
    """
    Fold one published cycle into the wall-clock stall watch. `(held, stalled)`.

    `advance` is the virtual time this cycle spent, `nominal` what it would have
    spent at `v_s = 1`, so their ratio is the realized progress rate. The gate in
    `gates` asks whether the plan is **finished** and asks it in plan time, which
    is the right clock for that question. This asks whether it is
    **progressing**, and that one is only meaningful against the wall clock: the
    progress rate is a decision variable with zero inside its feasible set, so an
    optimizer that stops spending the plan also stops ageing it and the
    max_reference_age gate slows down with the machine and never fires.

    A cycle under `min_rate` adds its own wall-clock duration, capped at the
    timeout so a long stall cannot run the count away; any other cycle clears it
    outright. The verdict is therefore on *sustained* near-zero progress and not
    on a slowdown -- slowing down is the feature working.

    `nominal` is one grid step, so the rate is per cycle: it is the progress rate
    the optimizer chose and nothing else. A node whose own loop runs slower than
    `Ts` also spends plan slower than real time, and that is a timing fault with
    `solve_budget` for an oracle.

    Every comparison reads a NaN as *not progressing* and as *no wall time*.
    Neither is reachable from `advance`, which sanitises both, but a guard that
    reads a corrupt number as a healthy cycle is the one that costs something.
    """
    step = wall_dt if wall_dt > 0.0 else 0.0
    progressing = nominal > 0.0 and advance >= min_rate * nominal
    held = 0.0 if progressing else min(held + step, max_stall_time)
    # The cap bounds the count; `>=` keeps it reported for as long as it lasts.
    return held, not progressing and held >= max_stall_time


class Cycle:
    """
    The MPC's own state between two solves, and the decisions it makes on it.

    Everything the node published last cycle is reachable from here; nothing
    here reaches back into the node.
    """

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
        # `watch_progress`'s state. Only cycles that published feed it, because
        # only those spent a decision of this node's.
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

        # The OCP's own rows that no sensor carries: C3's lagged command, its
        # force state and the progress pair. Seeded at the first solve.
        self.carried = np.zeros(cs.NX)
        self.carried[cs.X_PROGRESS_RATE] = problem.K_PROGRESS_RATE_REFERENCE
        self.seed_force = True
        # The previous cycle's propagated planned velocity, read back as the
        # carry. **Not a second integrator**: a read of `Ocp.propagate`'s own
        # output, the same one x_0 is built from. `None` until one has been
        # propagated, and the first cycle then takes the measurement.
        self.dq_a_carried: np.ndarray | None = None
        self.velocity_carry = VelocityCarry()
        self.last_input = np.zeros(cs.NU_PROGRESS)
        # What this node put in flight, newest first. `last_input` is one command
        # and the dead time is 1.5 intervals on the shipped 40 ms / 60 ms, so the
        # machine really did execute two different numbers over the window the
        # propagation carries `x_0` across. One entry per cycle, kept only as deep
        # as `Ocp.replay` can ask for.
        self.applied_inputs: list = []
        self.guess = None
        self.last_solution = None
        self.measured: np.ndarray | None = None
        self.x0: np.ndarray | None = None
        self.q_eq = np.zeros(PASSIVE_DOF)
        self.tool_position = 0.0

        self.solves = 0
        # Every cycle, silent ones included. `solves` is the wrong clock for a
        # stream that reports the machine rather than the solve: it stops
        # advancing on a silent cycle and would freeze that stream's cadence.
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
        # The stall watch is per plan: a new reference is the re-plan a stall
        # asks for, and counting the old plan's stall against it reports the
        # answer.
        self.forget_stall()

    def adopt_mode(self, requested: str) -> str:
        """Move to `requested` and cold start. Returns the mode left behind."""
        previous, self.mode = self.mode, requested
        self.forget_plan()
        # In shadow this node's plan drives nothing, so its own progress may sit
        # near zero for as long as the follower is doing something else. Carried
        # into active that reports a stall on the first cycle that takes the
        # machine, on evidence gathered while it did not.
        self.forget_stall()
        self.consecutive_failures = 0
        self.escalated = False
        self.cadence_anchored = False
        self.last_input = np.zeros(cs.NU_PROGRESS)
        # And the rest of what was in flight: every entry older than this instant
        # was issued by whatever was driving before, and replaying those would
        # carry `x_0` forward under another commander's plan.
        self.applied_inputs.clear()
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
        # On an axis whose `dq_a_feedback` entry is false the velocity is the
        # model's own from the previous cycle rather than the measurement just
        # written. All-true ships, so this writes nothing until it is configured.
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
        # One entry per cycle, recorded here rather than beside each `last_input`
        # write: in shadow mode there are two of those per cycle and still only
        # one command, and by this line `last_input` is whichever of them applies.
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
        # The sway box, and §2's offset residual with it, are centred on where the
        # tool is actually swinging and not on the hanging pose. `weights.q_u`
        # being zero and this centre still justify each other in a circle, but the
        # circle is no longer only recorded: issue 137 measured it, and the centre
        # is worth 3-4x the peak sway the weight is worth nothing. Issue 049's.
        self.q_eq = x0[
            cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
        ].copy()
        # What the next cycle carries on an axis that opted out of feedback: what
        # this propagation says the velocity is when the plan takes effect. One
        # cycle old and half a delay ahead of the measurement instant, which is
        # inside what dq_a_divergence_max watches.
        self.dq_a_carried = x0[
            cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF
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
            # Handing control back happens once; the cycles after it are this
            # node waiting for a solve it can believe, and they are not further
            # hand-backs. The count keeps rising and `SolverHealth` keeps
            # carrying `FAULT_SOLVER` either way -- only how loudly this is said
            # changes, because an error per cycle for as long as the condition
            # holds buries the cycle that explains it.
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
        # The sway the JTC tracks but does not command. Nothing is planned for
        # it -- this is the OCP's own prediction of where the tool will be.
        self.horizon.q_u_ref[:] = states[
            :, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE_DOF
        ]
        self.horizon.dq_u_ref[:] = states[
            :, cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE_DOF
        ]
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
        for field_name in ("q_a_ref", "dq_a_ref", "q_u_ref", "dq_u_ref"):
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
            # C3's own rows, carried: the command in flight and the force the
            # actuators are at. No sensor reports either.
            self.carried = solution.states[1].copy()

        self.next_first_knot_ns += int(self.Ts * NANOSECONDS)
        advance = solution.progress_advance
        spent = advance if np.isfinite(advance) and advance >= 0.0 else self.Ts
        self.reference_progress += spent

        # The liveness check, on the wall clock on purpose: what the plan cost in
        # wall-clock seconds is the one quantity the optimizer does not choose,
        # and how much plan that bought is exactly what it does. Reported and not
        # recovered from -- the horizon keeps going out, the machine is where the
        # plan says it is, it is merely not moving through it. A `Failed` cycle
        # spends a nominal interval above and so clears the count, deliberately:
        # "the optimizer did not answer" is `max_consecutive_failures`, and this
        # one is about the answer "wait".
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
        # **The stall count survives a silence; only the wall-clock mark does
        # not.** A silent cycle is not evidence either way -- this node is not
        # controlling, so no plan is being spent -- and charging its wall time to
        # the stall would report a producer that died as a machine that stopped.
        # But clearing the count would mean a stalled machine that drops one
        # `/joint_states` sample a second never reports at all, which is this
        # hole one level up.
        self.progress_marked = False

    def forget_stall(self) -> None:
        """Drop the stall watch: what it was gathered about is gone."""
        self.progress_held = 0.0
        self.progress_stalled = False
        self.progress_marked = False

    def forget_plan(self) -> None:
        """Drop the warm start and the horizon a shift would be taken from."""
        # The carry goes with them. It was propagated under whatever drove
        # before, and a break in this node's own output stream -- a gate, the
        # escalation, a payload step, a mode change -- is this node's
        # reactivation, so the next cycle re-seeds the row from the measurement
        # rather than from a state that kept integrating while nobody drove.
        self.dq_a_carried = None
        self.velocity_carry = VelocityCarry()
        self.last_horizon = None
        self.tcp_states = None
        self.last_tcp_states = None
        self.guess = None
