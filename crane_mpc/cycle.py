"""
One MPC cycle, without ROS.

`Cycle.step` owns the call order; `node.update()` and `scripts/trials/wire_chain`
both run it rather than each copying it. `Timing` owns every instant that order
turns on. Nanosecond ints, not `rclpy.time.Time`, keep it testable without a
node. Wire names live here, not in `node.py`, because refusals quote them and
`reports.py` reuses them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
from crane_model import symbolic as cs
from crane_model.velocity_loop import load_velocity_loop

from . import bspline, problem
from . import horizon as hz
from .solver import PROGRESS_ROWS, Outcome, PathCycle

REFERENCE_TOPIC = "/crane/reference"
JOINT_PATH_TOPIC = "/crane/joint_path"
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
PUMP_FLOW_TOPIC = "~/pump_flow"

#: Where the canonical eight carry the six actuated and the two passive.
ACTUATED_INDICES = cs.K_ACTUATED_ROWS
PASSIVE_INDICES = cs.K_PASSIVE_ROWS
ACTUATED_DOF = cs.K_ACTUATED_DOF
PLANNED_DOF = cs.K_PLANNED_DOF
PASSIVE_DOF = cs.K_PASSIVE_DOF
TOOL_AXIS = cs.K_TOOL_AXIS

NANOSECONDS = 1_000_000_000

#: JTC outputs kept at least, whatever the tick: several cycles' worth at 100 Hz.
OUTPUT_HISTORY = 64


class Timing:
    """
    Every instant one cycle runs on, in one place.

    The dead time knot 0 is due after, the cadence anchor a horizon is stamped
    from, the JTC's own output over that dead time, the tick the JTC is assumed
    to run at, and whether a solve came back late. `Cycle` holds one of these
    and the node reads it rather than keeping a second copy -- split over the
    two files, the copies drifted.

    `say` is called once per episode of each silent fallback -- re-armed when
    that fallback stops happening, so a cold start does not spend the line for
    the run. Both of them read from outside exactly like the healthy case,
    which is why they are said at all.
    """

    def __init__(self, Ts: float, delay: float, jtc_period=None, say=None) -> None:
        self.Ts = float(Ts)
        self.delay = float(delay)
        #: A solve started at `now` drives from the next cycle on, so its result may
        #: arrive any time inside `Ts`: until then the JTC plays the command
        #: published last cycle, and the predictor replays exactly that. Without it
        #: the first `solve latency` of every interval ran a command nobody
        #: predicted -- 10 ms of it diverges the loop.
        self.command_delay = self.delay + self.Ts
        self.command_delay_ns = int(self.command_delay * NANOSECONDS)
        #: The JTC's tick; it samples feedforward one tick ahead (`hz._command_message`).
        self.jtc_period = (
            1.0 / load_velocity_loop()[1] if jtc_period is None else float(jtc_period)
        )
        #: When knot 0 of the horizon this cycle builds takes effect.
        self.next_first_knot_ns = 0
        # `(stamp_ns, planned-axis output)` from the JTC, oldest first. Two cycles
        # of ticks at least, or a fast tick (a trial's `--control-rate`) would
        # push the far end of the dead-time window out before it is averaged.
        ticks = max(1, round(self.Ts / self.jtc_period))
        self._outputs: deque = deque(maxlen=max(OUTPUT_HISTORY, 2 * ticks))
        self._say, self._said = say, set()

    # -- the cadence -------------------------------------------------------------

    def measured_ns(self, now_ns: int, stamp_ns: int | None) -> int:
        """
        Give the instant to anchor on: the sample's own, not the tick's.

        The sample lags the tick by 0-10 ms plus transport. A stamp in this
        node's future is another clock, not a prediction, so it is clamped back
        to `now` -- and said, because a clamped stamp reads like a timely one.
        """
        if stamp_ns is None:
            return now_ns
        if stamp_ns > now_ns:
            self._once(
                "future stamp",
                f"a state stamp is {(stamp_ns - now_ns) / 1e9:.3f} s ahead of this "
                f"node's clock on {JOINT_STATE_TOPIC} and is clamped to it, so the "
                "cadence anchors on the tick instead of the sample.",
            )
            return now_ns
        self._said.discard("future stamp")
        return stamp_ns

    def anchor(self, measured_ns: int) -> None:
        """
        Knot 0 takes effect `command_delay` after this cycle's measurement.

        Re-anchored every cycle, not once and advanced by `Ts`: the timer ticks on
        its own grid, so a once-anchored stamp sat a fixed 0-60 ms off the real
        measurement instant, random per run, against ~10 ms of timing margin.
        """
        self.next_first_knot_ns = measured_ns + self.command_delay_ns

    def stamp_ns(self) -> int:
        """Give what a horizon is stamped: the measurement knot 0 counts from."""
        return self.next_first_knot_ns - self.command_delay_ns

    # -- what the JTC did --------------------------------------------------------

    def record_output(self, stamp_ns: int, velocity) -> None:
        self._outputs.append((int(stamp_ns), np.asarray(velocity, dtype=float)))

    def jtc_output(self, measured_ns: int):
        """
        Mean JTC output over `[measured - Ts, measured)`, or `None`.

        What the plant ran over the dead time, PI included -- the command that
        was sent carries none of it. `None` below half the expected ticks: two
        samples of ten are not the interval's mean.
        """
        start = measured_ns - int(self.Ts * NANOSECONDS)
        window = [v for t, v in self._outputs if start <= t < measured_ns]
        if len(window) < 0.5 * self.Ts / self.jtc_period:
            self._once(
                "no jtc output",
                f"fewer than half a cycle's ticks arrived on {CONTROLLER_STATE_TOPIC}, "
                "so the dead time replays the command that was sent instead of the "
                "JTC's own output -- which carries the PI term and the command does not.",
            )
            return None
        self._said.discard("no jtc output")
        return np.mean(window, axis=0)

    # -- what the controllers promise --------------------------------------------

    def rate_mismatch(self, manager_rate: int, jtc_rate: int) -> str:
        """Refusal text if the JTC does not tick at `jtc_period`, else `""`."""
        rate = jtc_rate or manager_rate  # a controller's 0 means the manager's
        if rate > 0 and abs(1.0 / rate - self.jtc_period) <= 1e-6:
            return ""
        return (
            f"the JTC ticks at {rate} Hz (update_rate {jtc_rate}, controller_manager "
            f"{manager_rate}) and the horizon is built for a {self.jtc_period} s tick "
            "from velocity_loop.yaml rate_hz; the feed-forward would be sampled off-tick"
        )

    def late(self, latency_s, budget_s: float) -> bool:
        """End-to-end from the measurement, not acados' own `solve_time`."""
        return latency_s is not None and float(latency_s) > float(budget_s)

    def _once(self, key: str, message: str) -> None:
        """Say this fallback once per episode; the healthy branch re-arms `key`."""
        if self._say is None or key in self._said:
            return
        self._said.add(key)
        self._say(message)


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


@dataclass(frozen=True)
class Bounds:
    """One read of the gate parameters, for the cycle that is about to run."""

    max_clock_skew: float
    max_reference_age: float
    max_state_age: float
    max_consecutive_failures: int
    min_progress_rate: float
    max_stall_time: float


@dataclass
class Step:
    """What one `Cycle.step` did: the first thing that stopped it, or the verdict."""

    silence: Silence | None = None
    solution: object = None
    refusal: str = ""
    verdict: Verdict | None = None
    #: Whether the plan was already stalled before `advance` folded this cycle in.
    was_stalled: bool = False


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
        timing: Timing,
        grid: hz.Grid,
        mode: str,
        dq_a_feedback=None,
        dq_a_divergence_max=None,
    ) -> None:
        self.timing = timing
        self.grid = grid
        self.mode = mode
        #: `x` rolled to the next measurement instant; the next cycle's C3 rows.
        self.x_next = None
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
        # The planner's own geometry, when it publishes it: control points fitted
        # once per plan, the seconds it means to spend the whole path in, and how
        # much of the path earlier cycles already spent. Without it `Ocp` fits
        # this cycle's own window instead and `path_origin` goes unread.
        self.path_control: np.ndarray | None = None
        self.path_duration = 0.0
        self.path_stamp_ns = 0
        self.path_origin = 0.0
        #: The path this cycle is written against, or `None` for the window fit.
        self.path: PathCycle | None = None
        # `watch_progress` state; only published cycles feed it.
        self.progress_held = 0.0
        self.progress_stalled = False
        self.progress_mark_ns = 0
        self.progress_marked = False

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
        #: C4: `c` where `last_input`'s ramp starts (`last_input` is then `dc`).
        self.last_start = np.zeros(PLANNED_DOF)
        # What this node put in flight, newest first; C4: `(start, input)` pairs.
        # Two deep: the dead time is one `Ts`, so `command_delay` is this cycle's
        # command over one interval and the previous one over the interval before.
        self.applied_inputs: list = []
        #: Mean JTC output over the dead time before the measurement, planned
        #: axes; replaces `applied_inputs[1]` in that roll -- it carries the PI.
        self.jtc_output: np.ndarray | None = None
        #: Measurement-to-solved seconds of the last solve, `None` if unmeasured.
        self.latency_s: float | None = None
        self.guess = None
        #: A preparation standing for this cycle, and the resampled reference it
        #: was linearised about. The horizon is what gets re-checked: it is the
        #: only thing the preparation bakes that a new plan, a re-anchored
        #: cadence or a slipped clock can move underneath it.
        self.prepared = False
        self.prepared_horizon: hz.Knots | None = None
        self.prepared_in_flight: np.ndarray | None = None
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

    # `Timing`'s, read through here: `reports` and the node's publisher want one
    # of these per cycle and there is only one place it can be right.
    Ts = property(lambda self: self.timing.Ts)
    command_delay = property(lambda self: self.timing.command_delay)
    next_first_knot_ns = property(lambda self: self.timing.next_first_knot_ns)

    @property
    def command_state(self) -> bool:
        """C4: the command is a state and `u` its rate."""
        return bool(getattr(self.ocp, "command_state", False))

    def in_flight_command(self):
        """Give what the JTC plays until knot 0: C3 its hold, C4 its ramp's start."""
        entry = self.applied_inputs[0]
        return entry[0] if self.command_state else entry

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
        # A new plan starts at its own beginning. The reference anchors the plan,
        # so this belongs here and not in `adopt_path`: a curve is geometry and
        # says nothing about how much of it has been spent.
        self.path_origin = 0.0
        # Stall watch is per plan: a new reference is the re-plan a stall asks for.
        self.forget_stall()
        # A preparation is a linearisation of the plan that was; this is another
        # plan. `forget_plan` is not called here -- the warm start survives a
        # re-plan -- so the preparation has to be dropped on its own.
        self.forget_preparation()

    def adopt_path(self, control: np.ndarray, duration: float, stamp_ns: int) -> None:
        """
        Take the planner's curve, already fitted, for the plan stamped `stamp_ns`.

        Stored rather than used: `resolved_path` only reaches for it once the
        reference of the same stamp is the one this cycle is running, so a curve
        that arrives before its reference, or after the next one, drives nothing.
        """
        self.path_control = np.asarray(control, dtype=float)
        self.path_duration = float(duration)
        self.path_stamp_ns = int(stamp_ns)
        if self.reference_anchored and self.resolved_path() is not None:
            # It arrived after its own reference, so that reference was anchored as
            # a time-indexed one and has been spent as one. Both readings go back to
            # the plan's start together: nothing has been spent *on the curve*, and
            # leaving them apart is the truncation the anchor above describes.
            self.reference_progress = 0.0
            self.path_origin = 0.0
        # A standing preparation was linearised about whichever path last cycle
        # resolved; this may be the cycle that stops being the window fit.
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
        self.last_input = np.zeros(cs.NU_PROGRESS)
        # Rest of what was in flight: replaying it would carry `x_0` under another commander's plan.
        self.applied_inputs.clear()
        return previous

    def adopt_follower(self, follower: FollowerCommand, u_max) -> None:
        """In shadow the follower's command is what `u` was; in active it is not."""
        self.follower = follower
        if self.mode == "shadow":
            self.last_input = follower_input(follower, u_max)
            if self.command_state:
                self.last_start = self.last_input[:PLANNED_DOF].copy()
                self.last_input = np.zeros(cs.NU_PROGRESS)

    # -- one cycle, in order -----------------------------------------------------

    def step(
        self,
        now_ns: int,
        measured_ns: int,
        measurement: Measurement,
        bounds: Bounds,
        follower: FollowerCommand | None = None,
        latency=None,
        publish=None,
    ) -> Step:
        """
        Run the whole cycle, in the one order it may be run in.

        Gates, state, follower, the dead-time roll, the solve, the ladder,
        `publish()`, then `advance` and the next cycle's preparation. Every
        caller runs this rather than copying the order: the copy in
        `wire_chain` had already drifted -- its own `jtc_output` rule and no
        `adopt_follower` at all.

        `begin()` stays the caller's: it marks cycles this never reaches, the
        ones a node refuses before it is configured. A gate silences the cycle
        here, so the caller only reports the `Silence` it gets back.
        """
        silence = self.gates(
            now_ns, bounds.max_clock_skew, bounds.max_reference_age, measured_ns
        )
        if silence is None:
            silence = self.read_state(measurement, bounds.max_state_age)
        if silence is None:
            # u^+ is crane_model's, resolved when the problem was posed -- not a
            # parameter, so the only copy is the one the solver was configured on.
            self.adopt_follower(
                follower if follower is not None else FollowerCommand(),
                self.ocp.parameters["limits"]["u_max"],
            )
            # In shadow this node drives nothing, so the JTC's output is another
            # commander's and may not stand in for what ours put in flight.
            self.jtc_output = (
                self.timing.jtc_output(measured_ns) if self.mode == "active" else None
            )
            silence = self.propagate()
        if silence is not None:
            self.stay_silent(silence.why)
            return Step(silence=silence)

        solution, refusal = self.solve(latency)
        if solution is None:
            # On the refusal `solve` has already fallen silent. Tested on the
            # solution and not on `refusal`: `str(AssertionError())` is `""`,
            # and a falsy refusal used to carry `None` into `ladder` and take
            # the node's timer callback down with it.
            return Step(refusal=refusal or "the OCP raised without a message")

        verdict = self.ladder(
            solution, bounds.max_consecutive_failures, self.ocp.solve_budget_s
        )
        done = Step(
            solution=solution, verdict=verdict, was_stalled=self.progress_stalled
        )
        if not verdict.published:
            return done
        if publish is not None:
            publish()
        self.advance(solution, now_ns, bounds.min_progress_rate, bounds.max_stall_time)
        # After `advance`, which moves the progress and the cadence anchor the next
        # cycle resamples on; and after the horizon went out, so the linearisation
        # runs in the part of the cycle nobody is waiting on.
        self.prepare_next(solution)
        return done

    # -- the gates ---------------------------------------------------------------

    def gates(
        self,
        now_ns: int,
        max_clock_skew: float,
        max_reference_age: float,
        measured_ns: int | None = None,
    ):
        """
        Run the reference's three refusals, then the anchor and the resample.

        The cadence anchors on `measured_ns`, the state's stamp, else on `now_ns`.

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

        self.timing.anchor(now_ns if measured_ns is None else measured_ns)
        if not self.reference_anchored:
            # Skipping the planning latency is what a time-indexed reference wants:
            # the plan is that old, so its first seconds are past. A path-following
            # cost wants the opposite -- its first stage is `c(origin)` and `origin`
            # starts at the curve's start, so anchoring this reading L seconds in
            # puts the two readings of one plan at different places on it, and it is
            # this one the `max_reference_age` gate above reads. Anchored at L, that
            # gate fires (L - max_reference_age)/duration of the path early.
            self.reference_progress = (
                0.0
                if self.resolved_path() is not None
                else (self.next_first_knot_ns - self.reference_stamp_ns) / 1e9
            )
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
        self.path = self.resolved_path()
        return None

    def resolved_path(self) -> PathCycle | None:
        """
        Give the planner's curve for the plan now running, or `None` to fit the window.

        Paired by stamp: the reference and the curve are one plan in two forms
        and are published together, so a mismatch means one of them is the other
        plan's. Falling back rather than refusing keeps a node whose planner does
        not publish geometry working exactly as it did.
        """
        if self.path_control is None or self.path_duration <= 0.0:
            return None
        if self.path_stamp_ns != self.reference_stamp_ns:
            return None
        return PathCycle(
            control=self.path_control,
            origin=min(max(self.path_origin, 0.0), 1.0),
            nominal_rate=1.0 / self.path_duration,
        )

    def path_source(self) -> str:
        """
        Name the branch `q_a_ref` comes from, and say why it is not the other one.

        Said out loud because falling back is silent: a stamp mismatch reads
        exactly like a followed curve from outside the node, and that is the
        mode a run is meant to be under test in.
        """
        if self.resolved_path() is not None:
            return f"the planner's curve on {JOINT_PATH_TOPIC}"
        if self.path_control is None or self.path_duration <= 0.0:
            return (
                "the horizon's own knots, because no usable curve has arrived on "
                f"{JOINT_PATH_TOPIC}"
            )
        # Only the pairing is left: `resolved_path` refused above and a curve is stored.
        return (
            f"the horizon's own knots, because the curve on {JOINT_PATH_TOPIC} is "
            f"stamped {self.path_stamp_ns} ns against the reference's "
            f"{self.reference_stamp_ns} ns, so it is another plan's geometry"
        )

    def path_span(self) -> float:
        """Seconds one whole unit of path parameter stands for, on this cycle's path."""
        if self.path is not None:
            return 1.0 / self.path.nominal_rate
        # The window fit spans one horizon, which is what `nominal_progress_rate`
        # inverts -- so this is the same number, read off the grid.
        return self.grid.duration()

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
        nx = getattr(self.ocp, "nx", cs.NX)
        if self.carried.size != nx:
            self.carried = np.zeros(nx)
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
        # C4 replays `(start, dc)`: x0's `c` is then last solve's `states[1]` exactly.
        c4 = self.command_state
        if self.mode == "active" and not self.applied_inputs:
            # Nothing of ours in flight (first cycle, after a silence or a mode
            # change): the machine's own velocity is the best reading of what the
            # JTC runs -- zero at rest, and neither a stale `u0` nor a hard stop
            # when it moves.
            velocity = self.measured[
                cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF
            ]
            self.last_input = np.zeros(cs.NU_PROGRESS)
            if c4:
                # Held, inside Psi's domain (`c`'s hard box).
                bound = np.asarray(
                    self.ocp.parameters["limits"]["u_max"][:PLANNED_DOF], dtype=float
                )
                self.last_start = np.clip(velocity, -bound, bound)
            else:
                self.last_input[:PLANNED_DOF] = velocity
        self.applied_inputs.insert(
            0,
            (self.last_start.copy(), self.last_input.copy())
            if c4
            else self.last_input.copy(),
        )
        del self.applied_inputs[2:]
        dead_time = self.applied_inputs[1:] or self.applied_inputs
        if self.jtc_output is not None:
            held = np.zeros(cs.NU_PROGRESS)
            if c4:
                dead_time = [(np.asarray(self.jtc_output, dtype=float), held)]
            else:
                held[:PLANNED_DOF] = self.jtc_output
                dead_time = [held]
        try:
            # Over the dead time the plant runs what left the JTC one cycle ago;
            # then one `Ts` of the command already published (`command_delay`).
            self.x_next = self.ocp.propagate_applied(self.measured, dead_time)
            x0 = self.ocp.propagate_applied(self.x_next, self.applied_inputs[:1])
        except Exception as error:
            return Silence(
                "the measured state could not be propagated to the instant the plan "
                "takes effect",
                f"The measured state could not be carried {self.command_delay} s "
                f"forward, so "
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

    def solve(self, latency=None):
        """
        Solve. Returns `(solution, refusal)`; exactly one is falsy.

        `latency()`, seconds since the measurement, is read once the OCP answers;
        past `solve_budget_s` a converged solve is late end-to-end and is handled
        as over budget -- acados' `solve_time` misses the Python around it.

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
                solution = self.ocp.solve(
                    self.x0,
                    self.horizon,
                    self.q_eq,
                    self.guess,
                    path=self.path,
                    # C4 prices `dc` on every stage; no stage-0 move cost.
                    in_flight=None if self.command_state else self.applied_inputs[0],
                )
        except Exception as error:
            self.applied_previous = False
            self.consecutive_failures += 1
            self.solves += 1
            self.stay_silent_after_failure(
                "the optimal control problem did not return a usable horizon"
            )
            return None, str(error)

        self.solves += 1
        self.latency_s = None if latency is None else float(latency())
        if solution.outcome is Outcome.CONVERGED and self.timing.late(
            self.latency_s, self.ocp.solve_budget_s
        ):
            solution = replace(solution, outcome=Outcome.BUDGET_EXCEEDED)
        self.guess = (
            None if solution.outcome is Outcome.FAILED else self.ocp.carried(solution)
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
        # Stage 0's move cost stands on the preparer's in-flight command; in
        # shadow the follower's replaces it.
        if self.prepared_in_flight is not None and not np.array_equal(
            self.prepared_in_flight, self.applied_inputs[0]
        ):
            return False
        return all(
            np.array_equal(
                getattr(self.prepared_horizon, block), getattr(self.horizon, block)
            )
            for block in ("t", "q_a_ref", "dq_a_ref")
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
            # `advance` has already moved `path_origin` on, so this resolves the
            # path as the *next* cycle will -- otherwise the preparation would
            # stand on a path one interval behind the feedback phase's.
            # `advance` set `last_input` to what `propagate` will put in flight
            # next cycle.
            self.ocp.prepare(
                predicted,
                horizon,
                q_eq,
                self.guess,
                path=self.resolved_path(),
                in_flight=None if self.command_state else self.last_input,
            )
        except Exception:
            self.forget_preparation()
            return
        self.prepared = True
        self.prepared_horizon = horizon
        if self.ocp.suppresses_moves:
            self.prepared_in_flight = self.last_input.copy()

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

        late = self.timing.late(self.latency_s, solve_budget_s)
        spent = (
            f"{round(1000 * self.latency_s)} ms end-to-end from the measurement, a "
            "late cycle,"
            if late
            else f"{round(1000 * solution.solve_time_s)} ms"
        ) + f" against a {round(1000 * solve_budget_s)} ms budget"
        if self.consecutive_failures >= max_consecutive_failures:
            # Handing back happens once; later cycles just wait for a believable
            # solve. SolverHealth keeps FAULT_SOLVER either way -- only log
            # severity changes, so it doesn't bury the explaining cycle.
            handing_back = not self.escalated
            self.escalated = True
            self.applied_previous = False
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
                f"{spent}, so mpc §6's previous solution shifted by one step was published on "
                f"{destination} instead",
                published=True,
                severity="warn",
            )

        self.applied_previous = False
        text = (
            f"acados answered {solution.status_word} ({solution.outcome}) after "
            f"{spent} and there was no previous solution to shift, so nothing was "
            "published"
        )
        self.stay_silent_after_failure(
            "the solve did not converge and there is no previous solution to shift"
        )
        return Verdict(text, published=False, severity="warn")

    def adopt_solution(self, solution) -> None:
        """
        Write the solved horizon as the canonical eight the wire carries.

        The planned positions are a **plan** quantity, never the solved states:
        those pin stage 0 to the measurement, which leaves the JTC's `p e_pos` --
        the integral action on a velocity interface -- no error to integrate, and
        the machine parks on the solver's offset (11 mrad, measured, not
        decaying). The plan is `gates`' resampled reference on the window fit, and
        `c(origin + s)` per knot on the planner's curve, which is the quantity the
        cost was written against. `dq_a_ref` stays the solver's so that
        `effort = u - dq_a_ref` keeps the two open-loop branches summing to `u`.
        """
        states = solution.states
        self.horizon.dq_a_ref[:, :PLANNED_DOF] = states[
            :, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED_DOF
        ]
        if self.path is not None:
            sigma = self.path.origin + states[:, cs.X_PROGRESS]
            self.horizon.q_a_ref[:, :PLANNED_DOF] = bspline.value(
                sigma, self.path.control, problem.PATH_KNOTS
            )
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
        # C4: the command is the state `c`, one per knot, ramped between them.
        commands = solution.inputs[:, : cs.NU]
        if self.command_state:
            self.horizon.u[:, :PLANNED_DOF] = states[
                :, cs.X_COMMAND : cs.X_COMMAND + PLANNED_DOF
            ]
        elif commands.size:
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
        c4 = self.command_state
        if solution.outcome is Outcome.CONVERGED:
            self.last_input = solution.u0.copy()
            if c4:
                self.last_start = solution.states[
                    0, cs.X_COMMAND : cs.X_COMMAND + PLANNED_DOF
                ].copy()
        elif c4:
            # The shifted knots' ramp 0 -> 1 is what went out.
            self.last_start = self.horizon.u[0, :PLANNED_DOF].copy()
            self.last_input = np.zeros(cs.NU_PROGRESS)
            self.last_input[:PLANNED_DOF] = (
                self.horizon.u[1, :PLANNED_DOF] - self.last_start
            ) / self.Ts
        else:
            # `ladder` shifted the previous horizon into `self.horizon` and
            # published it, so `u[0]` is what the machine got -- the same row
            # `adopt_solution` fills from `solution.u0` on the converged rung.
            # Not `dq_a_ref`: `u != dq_ref`, and `propagate_applied` replays
            # this to build C3's force, so the reference understates the
            # build-up once per missed budget and compounds over a run of them.
            self.last_input = np.zeros(cs.NU_PROGRESS)
            self.last_input[:PLANNED_DOF] = self.horizon.u[0, :PLANNED_DOF]
        if solution.outcome is not Outcome.FAILED:
            # C3's lag and force rows are physical, so they enter next cycle's
            # roll at its *measurement* instant -- `x_next`, not a stage of this
            # solution. The progress pair is not physical and no sensor overwrites
            # it, so it wants the stage standing where the next `x0` will:
            # `states[1]`, which `Ocp.propagate` holds out of the roll.
            self.carried = (
                self.x_next if self.x_next is not None else solution.states[0]
            ).copy()
            self.carried[PROGRESS_ROWS] = solution.states[1][PROGRESS_ROWS]

        # `s` is a path parameter, not seconds, so `path_span` converts -- one
        # window for the window fit, the whole plan for the planner's curve. Both
        # readings move off the one number: `path_origin` is where on the curve
        # the next cycle starts, `reference_progress` the same place as the plan's
        # own time, which places the fallback's window.
        #
        # Only a converged solve says how much plan it bought. The other rungs
        # published the previous horizon shifted one knot, so the machine consumes
        # one nominal interval of *that* plan -- what
        # `stay_silent_after_failure` charges. Off a refused iterate the window
        # ran up to 0.21 s of plan per cycle ahead of the machine.
        span = self.path_span()
        nominal = self.Ts / span if span > 0.0 else 0.0
        bought = (
            float(solution.progress_advance)
            if solution.outcome is Outcome.CONVERGED
            else nominal
        )
        if not (np.isfinite(bought) and bought >= 0.0):
            bought = nominal
        if self.path is not None:
            # Past the end of the curve `remaining()` is zero, so a converged solve
            # buys nothing and neither reading would ever move again: no expiry, and
            # `watch_progress` calling a finished move a stall. What is spent there
            # is time at the goal, which is what `max_reference_age` bounds.
            if self.path_origin >= 1.0:
                bought = nominal
            self.path_origin = min(self.path_origin + bought, 1.0)
        spent = bought * span
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
        # What the JTC runs through a silence is not ours; `propagate` reseeds.
        self.applied_inputs.clear()
        if self.reference_anchored:
            # One nominal interval spent while nobody was driving, charged to both
            # readings of where the plan is, as `advance` charges them -- and to the
            # curve only while the curve is what drove, since nothing was spent on a
            # path this cycle was not written against.
            span = self.path_span()
            if self.path is not None and span > 0.0:
                self.path_origin = min(self.path_origin + self.Ts / span, 1.0)
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
        self.prepared_in_flight = None

    def forget_plan(self) -> None:
        """Drop the warm start and the horizon a shift would be taken from."""
        # Every break in output -- gate, escalation, payload step, mode change,
        # failure -- reaches here, and none of this may outlive one: a break is
        # reactivation, so the next cycle re-seeds from the measurement rather
        # than a state that kept integrating unmanned. `seed_force` is what makes
        # that true of `carried`: `read_state` rebuilds the force row from
        # `static_hold_force` at the measured pose, which after §6's escalation
        # released the arm claim is the only pose still known to hold.
        self.forget_preparation()
        self.seed_force = True
        self.dq_a_carried = None
        self.velocity_carry = VelocityCarry()
        self.last_horizon = None
        self.tcp_states = None
        self.last_tcp_states = None
        self.guess = None
