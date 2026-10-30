"""
The escalation ladder and its shift, without a node.

Two paths load-bearing but unreachable from a node test: the escalation
that hands control back, and the shift that keeps driving on a solve that
did not converge.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from crane_model import CraneModel, Payload, Tool
from crane_model import symbolic as cs
from crane_mpc import horizon as hz
from crane_mpc import problem, reports
from crane_mpc.cycle import (
    HORIZON_TOPIC,
    JOINT_PATH_TOPIC,
    PASSIVE_DOF,
    PLANNED_DOF,
    Bounds,
    Cycle,
    Timing,
    Verdict,
    watch_progress,
)
from crane_mpc.equilibrium import RO, Equilibrium
from crane_mpc.solver import CostTerms, Outcome, PathCycle, Solution, Violation
from crane_mpc.solver import path_control as ocp_path_control
from crane_msgs.msg import SolverHealth, SupervisorStatus
from rclpy.time import Time

KNOTS = 5
Ts = 0.04
BOUNDS = Bounds(1.0, 0.5, 0.2, 3, 0.1, 2.0)


def line(places=40):
    """A joint-space straight line, 0 -> 1 on every planned axis, as control points."""
    samples = np.linspace(np.zeros(cs.PLANNED_DOF), np.ones(cs.PLANNED_DOF), places)
    return ocp_path_control(samples)


def cycle(mode="active"):
    # `delay == Ts` is the one timing invariant `config.check_settings` enforces.
    made = Cycle(Timing(Ts, Ts), hz.Grid(Ts, KNOTS), mode)
    made.horizon = hz.Knots.zeros(KNOTS)
    # Every cycle past `gates` is written against a paired curve (188). This one
    # spans exactly one grid window, so `path_span()` is `grid.duration()` and the
    # advance arithmetic below still reads in `Ts`.
    made.path = PathCycle(
        control=line(), origin=0.0, nominal_rate=1.0 / hz.Grid(Ts, KNOTS).duration()
    )
    return made


def solution(outcome, progress_advance=Ts, nx=cs.NX, status=2):
    """A solve whose states count up, so a shift of them is visible."""
    states = np.arange(KNOTS * nx, dtype=float).reshape(KNOTS, nx)
    return Solution(
        states=states,
        inputs=np.zeros((KNOTS - 1, cs.NU_PROGRESS)),
        outcome=outcome,
        status=status,
        status_word="ACADOS_SUCCESS" if status == 0 else "ACADOS_MAXITER",
        qp_status=0,
        iterations=1,
        qp_iterations=1,
        solve_time_s=0.01,
        used_slack=False,
        slack_penalty=0.0,
        violation=Violation(),
        progress_advance=progress_advance,
        warm_started=False,
    )


def test_a_converged_solve_publishes_and_does_not_escalate():
    one = cycle()
    one.consecutive_failures = 0
    verdict = one.ladder(solution(Outcome.CONVERGED), 3, 0.02)
    assert verdict.published
    assert not one.escalated
    assert not one.applied_previous
    assert verdict.severity == ""


def test_the_first_failures_shift_the_previous_horizon():
    one = cycle()
    one.last_horizon = hz.Knots.zeros(KNOTS)
    one.last_horizon.q_a_ref[:] = np.arange(KNOTS)[:, None]
    one.last_tcp_states = np.arange(KNOTS * cs.NX, dtype=float).reshape(KNOTS, cs.NX)
    one.consecutive_failures = 1

    verdict = one.ladder(solution(Outcome.BUDGET_EXCEEDED), 3, 0.02)

    assert verdict.published
    assert one.applied_previous
    assert not one.escalated
    assert one.horizon.q_a_ref[:, 0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 4.0])
    assert one.tcp_states[0] == pytest.approx(one.last_tcp_states[1])
    assert one.tcp_states[-1] == pytest.approx(one.last_tcp_states[-1])


def test_without_a_previous_horizon_nothing_is_published():
    one = cycle()
    one.consecutive_failures = 1
    verdict = one.ladder(solution(Outcome.FAILED), 3, 0.02)
    assert not verdict.published
    assert not one.applied_previous
    assert not one.escalated
    assert "no previous solution to shift" in one.last_silence


def test_the_ceiling_escalates_and_stops_the_publisher():
    one = cycle()
    # Shiftable previous horizon deliberately present: ceiling is checked
    # before the shift, so escalation must win over it.
    one.last_horizon = hz.Knots.zeros(KNOTS)
    one.last_tcp_states = np.zeros((KNOTS, cs.NX))
    one.consecutive_failures = 3

    verdict = one.ladder(solution(Outcome.BUDGET_EXCEEDED), 3, 0.02)

    assert not verdict.published
    assert verdict.severity == "error"
    assert one.escalated
    assert not one.applied_previous
    assert "escalation" in one.last_silence
    assert "10 ms of solver time, latency unmeasured, against a 20 ms budget" in (
        verdict.text
    )
    assert one.last_horizon is None
    assert one.guess is None


def test_the_escalation_is_an_error_once_and_a_standing_warning_after():
    """
    Handing control back happens once; later cycles keep reporting
    `FAULT_SOLVER` as a warning, not an error -- a wedged node once wrote a
    hundred identical error lines in twenty seconds.
    """
    one = cycle()
    one.consecutive_failures = 3

    first = one.ladder(solution(Outcome.BUDGET_EXCEEDED), 3, 0.02)
    assert first.severity == "error"
    assert "handed control back" in first.text

    for _ in range(3):
        one.consecutive_failures += 1
        standing = one.ladder(solution(Outcome.BUDGET_EXCEEDED), 3, 0.02)
        assert standing.severity == "warn"
        assert not standing.published
        assert "still holds" in standing.text

    # Clearing the latch (gate, solve, mode change) re-arms the error.
    one.stay_silent("a gate")
    one.consecutive_failures = 3
    assert one.ladder(solution(Outcome.FAILED), 3, 0.02).severity == "error"


def test_a_gate_clears_the_failure_count_and_the_ladder_does_not():
    """`stay_silent` clears the count; the ladder's own silences do not."""
    one = cycle()
    one.consecutive_failures = 3
    one.ladder(solution(Outcome.FAILED), 3, 0.02)
    assert one.consecutive_failures == 3

    one.stay_silent("a gate refused")
    assert one.consecutive_failures == 0
    assert not one.escalated


def test_a_horizon_of_the_wrong_length_is_not_shifted():
    one = cycle()
    one.last_horizon = hz.Knots.zeros(KNOTS + 1)
    one.last_tcp_states = np.zeros((KNOTS + 1, cs.NX))
    assert not one.shift_previous_horizon()


def solving(answer, budget=0.02):
    """A cycle whose OCP answers `answer`; nothing else about it is exercised."""
    one = cycle()
    one.ocp = SimpleNamespace(
        solve_budget_s=budget,
        solve=lambda *_, **__: answer,
        carried=lambda _: None,
    )
    one.applied_inputs = [np.zeros(cs.NU_PROGRESS)]
    return one


def health_of(one, answer, joints=("a", "b", "c", "d", "e", "f")):
    return reports.solver_health(
        one, answer, "a verdict", list(joints), one.ocp.solve_budget_s, Time().to_msg()
    )


def test_a_late_cycle_is_reported_apart_from_a_failed_one():
    """
    Issue 226 read a run of late cycles as refusals, and the stream let it.

    Both rungs shift the previous horizon and both raise `FAULT_SOLVER`, so the
    only separation is `outcome` against `status` -- and acados' `solve_time`,
    the one duration the message carries, is *inside* the budget on every late
    cycle, which is why counting overruns off it read 0 while a fifth of cycles
    published a shifted horizon.
    """
    late = solving(solution(Outcome.CONVERGED, status=0))
    answer, refusal = late.solve(latency=lambda: 0.1)
    assert not refusal
    assert answer.outcome is Outcome.LATE
    assert answer.status == 0  # it converged; nothing acados says is wrong

    failed = solving(solution(Outcome.FAILED))
    assert failed.solve(latency=lambda: 0.001)[0].outcome is Outcome.FAILED

    assert health_of(late, answer).outcome == SolverHealth.SOLVE_BUDGET_EXCEEDED
    assert (
        health_of(failed, solution(Outcome.FAILED)).outcome == SolverHealth.SOLVE_FAILED
    )


def test_every_verdict_carries_both_clocks():
    """
    The solver's own time and the end-to-end latency, converged cycles included.

    A reader handed only the overruns cannot see the margin shrinking, and the
    one duration `SolverHealth` has a field for is the one that misses the
    Python around the solve.
    """
    one = solving(solution(Outcome.CONVERGED))
    answer, _ = one.solve(latency=lambda: 0.1)
    assert answer.outcome is Outcome.LATE
    one.last_horizon = hz.Knots.zeros(KNOTS)
    one.last_tcp_states = np.zeros((KNOTS, cs.NX))
    text = one.ladder(answer, 3, one.ocp.solve_budget_s).text
    assert "10 ms of solver time" in text
    assert "100 ms end-to-end from the measurement (a late cycle)" in text
    assert "20 ms budget" in text

    on_time = solving(solution(Outcome.CONVERGED))
    answer, _ = on_time.solve(latency=lambda: 0.012)
    assert answer.outcome is Outcome.CONVERGED
    converged = on_time.ladder(answer, 3, on_time.ocp.solve_budget_s).text
    assert "10 ms of solver time, 12 ms end-to-end" in converged
    assert "late cycle" not in converged


def test_an_unmeasured_latency_says_so_rather_than_reading_as_zero():
    one = solving(solution(Outcome.CONVERGED))
    assert one.solve(latency=None)[0].outcome is Outcome.CONVERGED
    assert (
        "latency unmeasured"
        in one.ladder(solution(Outcome.FAILED), 3, one.ocp.solve_budget_s).text
    )


def test_the_shadow_mode_names_its_own_topic_in_the_verdict():
    one = cycle("shadow")
    verdict = one.ladder(solution(Outcome.CONVERGED), 3, 0.02)
    assert HORIZON_TOPIC not in verdict.text
    assert "shadow_horizon" in verdict.text


# -- where x_0's actuated velocity comes from, per axis -------------------------

DOF = cs.PLANNED_DOF


def test_the_cost_rows_solver_health_cannot_carry_go_out_under_plottable_names():
    """
    PlotJuggler names a DiagnosticArray series `<topic>/<status.name>/<key>` only
    while `hardware_id` is empty -- a non-empty one (as `shadow_comparison` sets)
    renames every series and silently breaks the cost panel's layout.
    """
    terms = CostTerms(tool=2.0, progress=3.0, du=4.0, q_a=5.0)
    status = reports.cost_terms(terms, Time().to_msg()).status[0]

    assert (status.name, status.hardware_id) == ("cost", "")
    assert {entry.key: float(entry.value) for entry in status.values} == {
        "tool": 2.0,
        "progress": 3.0,
        "du": 4.0,
    }


# -- the wall-clock stall watch -------------------------------------------------

MIN_RATE = 0.1
MAX_STALL = 2.0
NOMINAL_NS = int(Ts * 1e9)


def watch(held, rate, cycles):
    """`cycles` cycles of a machine whose progress rate is `rate` times nominal."""
    stalled = False
    for _ in range(cycles):
        held, stalled = watch_progress(held, rate * Ts, Ts, Ts, MIN_RATE, MAX_STALL)
    return held, stalled


def stall(one):
    """
    Publish converged cycles that spend no plan at all, from a cold start.

    51 not 50: the first cycle after a clear has no mark yet, so charges no
    wall time (cold-start case; in steady state the bound falls on the 50th).
    """
    for step in range(51):
        one.advance(
            solution(Outcome.CONVERGED, 0.0), step * NOMINAL_NS, MIN_RATE, MAX_STALL
        )
    return one


def test_a_held_plan_reports_after_the_stall_time_and_not_before():
    """With the mark already set, the bound is counted from the last good cycle."""
    held, stalled = watch(0.0, 0.0, 49)
    assert not stalled, "49 cycles is 1.96 s, inside the 2.0 s bound"
    held, stalled = watch(held, 0.0, 1)
    assert stalled, "the 50th is 2.00 s, the bound itself"
    assert held == pytest.approx(MAX_STALL)
    held, stalled = watch(held, 0.0, 500)
    assert stalled
    assert held == pytest.approx(MAX_STALL)


def test_a_rate_just_under_the_threshold_is_a_stall_and_just_over_is_not():
    """AC1 is *below* min_rate, not zero; every other case here spends nothing."""
    assert watch(0.0, 0.999 * MIN_RATE, 50)[1]
    assert not watch(0.0, 1.001 * MIN_RATE, 10000)[1]


def test_a_slowdown_is_not_a_stall_and_the_count_is_wall_clock():
    """The watch must not fire on the feature working."""
    # Issue 119's cannot-follow run: legitimate over-ask on the slewing axis,
    # held longer than the stall time, at 0.760 minimum.
    assert not watch(0.0, 0.760, 10000)[1]
    # 0.319 is what a 200x-too-low time price dawdles at -- a tuning defect
    # this watch should not name as a stall.
    assert not watch(0.0, 0.319, 10000)[1]

    # Wall clock, not cycles: half rate reports after the same 2s wall time.
    held, stalled = 0.0, False
    for _ in range(26):
        held, stalled = watch_progress(held, 0.0, Ts, 2.0 * Ts, MIN_RATE, MAX_STALL)
    assert stalled

    # A non-duration nominal isn't progress; non-finite buys neither progress
    # nor wall time.
    assert watch_progress(0.0, Ts, 0.0, Ts, MIN_RATE, MAX_STALL) == (Ts, False)
    assert watch_progress(MAX_STALL, np.nan, Ts, np.nan, MIN_RATE, MAX_STALL) == (
        MAX_STALL,
        True,
    )


def test_one_progressing_cycle_inside_the_window_clears_the_count():
    held, stalled = watch(0.0, 0.0, 60)
    assert stalled
    held, stalled = watch(held, 1.0, 1)
    assert not stalled
    assert held == 0.0
    # A machine crawling just above the threshold never reports: 0.1 of
    # nominal still finishes the plan.
    assert not watch(held, MIN_RATE, 10000)[1]


def test_a_stall_and_a_solver_failure_carry_different_codes():
    """
    Issue 273: a stalled plan is converged and on time, so FAULT_SOLVER there
    points the reader at a component that did nothing wrong.
    """
    joints = [f"axis{axis}" for axis in range(cs.ACTUATED_DOF)]

    def health_for(one, answer):
        return reports.solver_health(
            one, answer, "a verdict", joints, 0.02, Time().to_msg()
        )

    stalled = stall(cycle())
    assert stalled.progress_stalled
    health = health_for(stalled, solution(Outcome.CONVERGED, 0.0))
    assert health.outcome == SolverHealth.SOLVE_CONVERGED
    assert health.fault == SupervisorStatus.FAULT_PLAN_STALLED

    # A solve that failed keeps FAULT_SOLVER, and outranks a stall if both hold.
    failed = health_for(cycle(), solution(Outcome.FAILED, 0.0))
    assert failed.outcome == SolverHealth.SOLVE_FAILED
    assert failed.fault == SupervisorStatus.FAULT_SOLVER
    assert (
        health_for(stall(cycle()), solution(Outcome.FAILED, 0.0)).fault
        == SupervisorStatus.FAULT_SOLVER
    )

    # And a converged, progressing solve is still healthy.
    assert (
        health_for(cycle(), solution(Outcome.CONVERGED, 0.0)).fault
        == SupervisorStatus.FAULT_NONE
    )


def test_a_new_reference_and_a_mode_change_each_clear_the_stall():
    one = stall(cycle())
    one.adopt_reference(hz.Knots.zeros(KNOTS), 0)
    assert not one.progress_stalled
    assert one.progress_held == 0.0

    one = stall(cycle())
    one.adopt_mode("shadow")
    assert not one.progress_stalled
    assert one.progress_held == 0.0


def test_a_silence_keeps_the_count_and_drops_the_wall_clock_mark():
    """Silent seconds are charged to nobody; a stalled machine still reports."""
    one = stall(cycle())
    one.stay_silent("a gate refused")
    assert one.progress_held == pytest.approx(MAX_STALL)
    assert not one.progress_marked
    # The field `reports.solver_health` reads stands through the silence too.
    assert one.progress_stalled


@pytest.mark.parametrize(
    "break_it",
    [
        lambda one: one.forget_plan(),
        lambda one: one.stay_silent("a gate"),
        lambda one: one.stay_silent_after_failure("a failure"),
        lambda one: one.adopt_mode("shadow"),
    ],
)
def test_every_break_in_the_output_stream_is_a_reactivation(break_it):
    one = cycle()
    # A cycle that has already seeded, so re-arming below means something.
    one.seed_force = False
    break_it(one)
    # The force row comes back from the measurement, not from before the break:
    # after the escalation the arm claim is released, so the pose it held may
    # not be the pose it comes back to.
    assert one.seed_force


def test_the_progress_the_solver_bought_is_read_as_a_path_parameter():
    """
    `s` spans the path, so a cycle's advance is `s` x `path_span()` seconds.

    Read as seconds it is a factor `path_span()` short -- here 4.8x on `cycle`'s
    window-wide curve -- which is a plan that crawls, not one that stalls, so
    nothing else in this file sees it.
    """
    window = hz.Grid(Ts, KNOTS).duration()
    one = cycle()
    one.reference_anchored = True

    # The whole path in one cycle: the plan advances by the whole window.
    one.advance(solution(Outcome.CONVERGED, 1.0), NOMINAL_NS, MIN_RATE, MAX_STALL)
    assert one.reference_progress == pytest.approx(window)

    # The plan's own pace: one interval of `s` buys one `Ts` of plan.
    two = cycle()
    two.reference_anchored = True
    two.advance(
        solution(Outcome.CONVERGED, Ts / window), NOMINAL_NS, MIN_RATE, MAX_STALL
    )
    assert two.reference_progress == pytest.approx(Ts)


def test_a_refused_solve_spends_one_interval_and_not_what_its_iterate_claims():
    """
    The fallback rung publishes the previous horizon shifted one knot, so what
    the machine consumes is one interval of *that* plan -- not whatever `s` a
    solver that did not converge happens to hold.

    Reading the iterate let both readings run `progress_rate_headroom *
    path_span()` ahead per cycle, five cycles deep before the publisher stops,
    leaving the next converged cycle origined where the machine is not.
    """
    window = hz.Grid(Ts, KNOTS).duration()
    for outcome in (Outcome.FAILED, Outcome.BUDGET_EXCEEDED):
        one = cycle()
        one.reference_anchored = True
        # An iterate claiming the whole path: believed, it would spend `window`.
        one.advance(solution(outcome, 1.0), NOMINAL_NS, MIN_RATE, MAX_STALL)
        assert one.reference_progress == pytest.approx(Ts)
        assert one.reference_progress < window


# -- the planner's curve, the only thing a cycle may be written against ----------


def curve_cycle(duration=2.0, stamp_ns=1000):
    """A cycle holding a reference and the curve of the same plan."""
    one = cycle()
    one.reference = hz.Knots.zeros(KNOTS)
    one.reference.t[:] = np.linspace(0.0, Ts * (KNOTS - 1), KNOTS)
    one.reference_stamp_ns = stamp_ns
    one.reference_anchored = True
    one.adopt_path(line(), duration, stamp_ns)
    one.path = one.resolved_path()
    return one


def test_a_curve_drives_only_the_plan_it_was_published_with():
    one = curve_cycle(stamp_ns=1000)
    assert one.resolved_path() is not None
    # The planner stamps the two alike, so a mismatch means one of them belongs to
    # another plan -- and attaching last plan's geometry to this reference would
    # aim the cost at a path the machine is not on.
    one.reference_stamp_ns = 2000
    assert one.resolved_path() is None


def test_a_plan_without_its_curve_refuses_the_cycle_and_names_both_stamps():
    """
    The mismatch is the case every re-plan passes through, and the one that used
    to fall back to a window fit no oracle covered (188). It now reads like the
    missing-reference refusal: nothing published, and the reason said.
    """
    one = curve_cycle(stamp_ns=1000)
    one.reference_stamp_ns = 2000
    silence = one.gates(0, 1.0, 0.5)
    assert silence is not None
    # Both numbers, so the pairing can be checked without reading source.
    assert "1000 ns" in silence.why and "2000 ns" in silence.why
    assert HORIZON_TOPIC in silence.warning and JOINT_PATH_TOPIC in silence.warning

    # No curve at all is the other half, and says so rather than naming a stamp.
    one.path_control = None
    assert "no usable curve has arrived" in one.gates(0, 1.0, 0.5).why


def test_a_reference_without_a_curve_does_not_anchor_and_so_does_not_age():
    """
    The refusal sits before the anchor, so the re-plan window costs no plan.

    Anchored first, the gap between the reference and its curve would be spent
    against `max_reference_age` -- and a late curve would start part-way along
    itself, which is what `adopt_path` used to have to undo.
    """
    one = curve_cycle(duration=2.0, stamp_ns=0)
    one.reference_anchored = False
    one.path_control = None
    assert one.gates(1_000_000_000, 1.0, 0.5) is not None
    assert not one.reference_anchored
    assert one.reference_progress == 0.0 and one.path_origin == 0.0


def test_the_advance_moves_both_readings_of_where_the_plan_is():
    one = curve_cycle(duration=2.0)
    one.advance(solution(Outcome.CONVERGED, progress_advance=0.25), 0, 0.0, 1.0)
    # A quarter of the path bought is a quarter of the path spent, which on a 2 s
    # plan is 0.5 s of it: one quantity, two units.
    assert one.path_origin == pytest.approx(0.25)
    assert one.reference_progress == pytest.approx(0.5)


def test_the_last_sliver_of_path_is_taken_whole_so_the_plan_can_end():
    """Buying a share of what is left never reaches 1: live, every goal stalled there."""
    one = curve_cycle(duration=2.0)
    nominal = one.Ts / one.path_span()
    one.path_origin = 1.0 - 0.5 * nominal
    one.advance(
        solution(Outcome.CONVERGED, progress_advance=0.05 * nominal), 0, 0.0, 1.0
    )
    assert one.path_origin == 1.0
    one.advance(solution(Outcome.CONVERGED, progress_advance=0.0), 0, 0.1, 1.0)
    assert not one.progress_stalled


def test_the_curve_is_what_the_follower_is_given_to_chase():
    """
    `s = origin + progress`, and the origin is not zero on any cycle past the first.

    Pinned part-way along the curve on purpose: with the window fit gone (188)
    every driving cycle runs at `origin > 0`, so a dropped or sign-flipped origin
    would command the right curve at the wrong place -- smooth motion to
    somewhere nobody planned, which the JTC's integral action would chase.
    """
    one = curve_cycle()
    one.path_origin = 0.4
    one.path = one.resolved_path()
    one.tool_position = 0.0
    answer = solution(Outcome.CONVERGED)
    progress = np.linspace(0.0, 0.5, KNOTS)
    answer.states[:, cs.X_PROGRESS] = progress
    one.adopt_solution(answer)
    # A joint-space line, so the point at `s` is `s` on every planned axis:
    # the position reference is the path at the progress the solve chose, not the
    # solved state (which counts up into the thousands here) and not the resample.
    for knot, place in enumerate(progress):
        assert one.horizon.q_a_ref[knot, : cs.PLANNED_DOF] == pytest.approx(
            np.full(cs.PLANNED_DOF, 0.4 + place), abs=1e-9
        )


def test_a_curve_anchors_at_the_plans_start_and_not_the_planning_latency():
    one = curve_cycle(duration=2.0, stamp_ns=0)
    one.reference_anchored = False
    # The plan was solved a second before this cycle. A time-indexed reference would
    # skip that second, because its first seconds are past; a path-following one may
    # not, since `path_origin` starts at the curve's start and it is *this* reading
    # the `max_reference_age` gate tests. Anchored a second in, that gate fires half
    # a second of plan before the path is spent.
    assert one.gates(1_000_000_000, 1.0, 0.5) is None
    assert one.reference_progress == 0.0


def test_a_spent_path_charges_time_so_the_plan_can_still_expire():
    one = curve_cycle(duration=2.0)
    one.path_origin = 1.0
    # `remaining()` is zero at the end of the curve, so a converged solve buys no
    # progress. Without charging the interval anyway, neither the expiry gate nor
    # `watch_progress` would ever move again: the node would hold the goal forever
    # and report a stall for every move it finished.
    one.advance(solution(Outcome.CONVERGED, progress_advance=0.0), 0, 0.5, 1.0)
    assert one.reference_progress == pytest.approx(Ts)
    assert not one.progress_stalled


def test_a_shifted_horizon_records_the_command_it_published():
    """
    `last_input` is what `propagate_applied` replays, so it must be what went out.

    On the shift rung the machine gets the previous horizon's `u`, not the
    reference velocity: `u != dq_ref` (CLAUDE.md), and force builds on
    `tau_dot = k*(u_f - dq)`, so recording `dq_a_ref` understates the build-up
    once per missed budget and compounds over consecutive misses.
    """
    one = cycle()
    one.last_horizon = hz.Knots.zeros(KNOTS)
    # Distinct per knot and distinct between the two fields, so which one was
    # taken -- and at which index -- is visible in the failure.
    one.last_horizon.u[:, :DOF] = 10.0 + np.arange(KNOTS)[:, None]
    one.last_horizon.dq_a_ref[:, :DOF] = 100.0 + np.arange(KNOTS)[:, None]
    one.last_tcp_states = np.arange(KNOTS * cs.NX, dtype=float).reshape(KNOTS, cs.NX)
    one.consecutive_failures = 1

    verdict = one.ladder(solution(Outcome.BUDGET_EXCEEDED), 3, 0.02)
    assert verdict.published and one.applied_previous
    published = one.horizon.u[0, :DOF].copy()

    one.advance(solution(Outcome.BUDGET_EXCEEDED), 10**9, 0.1, 2.0)

    assert one.last_input[:DOF] == pytest.approx(published)


def test_the_machine_carries_force_from_the_next_measurement_and_progress_from_x0_on():
    """
    `propagate` rolls the measurement one `Ts` under the command the plant runs
    now (`x_next`, the next measurement instant) and then under the one already
    published. C3's physical rows enter the next cycle from `x_next`; the
    progress pair from `states[1]`, where the next `x_0` stands.
    """
    made = cycle()
    made.x_next = np.full(cs.NX, 7.0)
    solved = solution(Outcome.CONVERGED, 1.0)
    made.advance(solved, NOMINAL_NS, MIN_RATE, MAX_STALL)
    assert made.carried[cs.X_ACTUATED_FORCE] == pytest.approx(7.0)
    assert made.carried[cs.X_PROGRESS_RATE] == pytest.approx(
        solved.states[1][cs.X_PROGRESS_RATE]
    )


class ReplayOcp:
    """Records the commands `propagate` replays; the roll itself is the identity."""

    def __init__(self):
        self.replayed = []

    def pin_tool(self, _):
        pass

    def propagate_applied(self, x, applied):
        self.replayed.append([np.asarray(u).copy() for u in applied])
        return np.asarray(x).copy()


def test_after_a_silence_the_command_in_flight_is_the_machines_own_velocity():
    """
    A silence mid-motion (cancelled goal, stale `/joint_states`, escalation)
    must not resume by replaying the `u0` from before it -- up to full speed on
    an axis that has since stopped -- nor by assuming a hard stop.
    """
    one = cycle()
    one.ocp = ReplayOcp()
    one.last_input = np.full(cs.NU_PROGRESS, 0.7)
    one.applied_inputs = [one.last_input.copy()]
    one.stay_silent("the gate closed")

    one.measured = np.zeros(cs.NX)
    one.measured[cs.X_PLANNED_VELOCITY] = 0.2
    assert one.propagate() is None
    replayed = one.ocp.replayed[-1][0]
    assert replayed[0] == pytest.approx(0.2)
    assert np.allclose(replayed[1 : cs.PLANNED_DOF], 0.0)


def test_the_horizon_is_anchored_on_the_measurement_and_not_the_tick():
    """A sample 10 ms old puts knot 0, and with it the stamp, 10 ms earlier."""
    one = curve_cycle(stamp_ns=0)
    now = 1_000_000_000
    measured = now - 10_000_000
    assert one.gates(now, 1.0, 0.5, measured) is None
    assert one.next_first_knot_ns == measured + int(one.command_delay * 1e9)


def test_the_dead_time_replays_what_the_jtc_output_and_not_what_was_sent():
    """The JTC's output carries the PI the OCP does not model; the command does not."""
    one = cycle()
    one.ocp = ReplayOcp()
    one.applied_inputs = [np.full(cs.NU_PROGRESS, 0.7)]
    one.last_input = np.full(cs.NU_PROGRESS, 0.3)
    one.measured = np.zeros(cs.NX)
    one.jtc_output = np.full(cs.PLANNED_DOF, 0.5)
    assert one.propagate() is None
    dead_time, published = one.ocp.replayed
    assert np.allclose(dead_time[0][: cs.PLANNED_DOF], 0.5)
    assert np.allclose(dead_time[0][cs.PLANNED_DOF :], 0.0)
    assert np.allclose(published[0], 0.3)


ORDER = (
    "gates",
    "read_state",
    "adopt_follower",
    "propagate",
    "solve",
    "ladder",
    "publish",
    "advance",
)


def test_step_runs_the_one_order_both_callers_used_to_copy():
    """
    The node and `wire_chain` each carried this order and had drifted apart.

    `publish` before `advance` is the load-bearing pair: `advance` moves the
    progress and the cadence anchor the *next* cycle resamples on, so a horizon
    written after it is stamped for a cycle that has already been charged.
    `adopt_follower` is the call `wire_chain`'s copy had dropped.
    """
    one = cycle()
    one.ocp = SimpleNamespace(solve_budget_s=1.0, parameters={"limits": {"u_max": []}})
    said = []
    answer = solution(Outcome.CONVERGED)
    returns = {
        "gates": None,
        "read_state": None,
        "adopt_follower": None,
        "propagate": None,
        "solve": (answer, ""),
        "ladder": Verdict("published", published=True),
        "advance": None,
    }
    for name, value in returns.items():
        setattr(one, name, lambda *_, n=name, v=value, **__: (said.append(n), v)[1])

    step = one.step(0, 0, None, BOUNDS, publish=lambda: said.append("publish"))

    assert said == list(ORDER)
    assert step.verdict.published and step.solution is answer


def test_a_solver_exception_without_a_message_is_still_a_refusal():
    """`str(AssertionError())` is `""`, which used to read as "no refusal"."""
    one = cycle()
    one.ocp = SimpleNamespace(solve_budget_s=1.0, parameters={"limits": {"u_max": []}})
    for name in ("gates", "read_state", "adopt_follower", "propagate"):
        setattr(one, name, lambda *_, **__: None)
    one.solve = lambda *_, **__: (None, "")

    step = one.step(0, 0, None, BOUNDS)

    assert step.refusal and step.verdict is None


def test_the_two_silent_fallbacks_are_said_once_per_episode():
    """
    Both read from outside like the healthy case, which is why they are said.

    Once and not per cycle: at 25 Hz a repeat is a log nobody reads, and the
    conditions that raise them last for runs, not cycles. Once per *episode*
    and not once per process: the first episode of the missing JTC output is
    the cold start, and latching on it would spend the line before the machine
    has moved.
    """
    said = []
    ticking = Timing(Ts, Ts, jtc_period=0.01, say=said.append)
    now = 1_000_000_000

    assert ticking.measured_ns(now, now + 5_000_000) == now
    assert ticking.measured_ns(now, now + 9_000_000) == now
    assert ticking.measured_ns(now, now - 5_000_000) == now - 5_000_000
    assert ticking.measured_ns(now, None) == now
    assert len(said) == 1 and "ahead of this" in said[0]

    # One of the four ticks a 0.04 s cycle expects: not the interval's mean.
    ticking.record_output(now, np.zeros(cs.PLANNED_DOF))
    assert ticking.jtc_output(now + int(Ts * 1e9)) is None
    assert ticking.jtc_output(now + int(Ts * 1e9)) is None
    assert len(said) == 2 and "half a cycle's ticks" in said[1]
    # The count rides along: the publish rate was measured flat while this
    # fired, so it is the number that says a reader dropped them.
    assert "(1 of 4)" in said[1]

    for tick in range(1, 4):
        ticking.record_output(now + tick * 10_000_000, np.full(cs.PLANNED_DOF, 2.0))
    mean = ticking.jtc_output(now + int(Ts * 1e9))
    assert mean == pytest.approx(np.full(cs.PLANNED_DOF, 1.5))
    assert len(said) == 2

    # Re-armed by the healthy cycle above: the next episode is said again. A
    # window this far on holds no ticks at all, which is the follower not
    # driving and not a delivery loss, so it is said as its own thing -- live,
    # a quarter of all windows are empty between moves and reporting them as
    # starvation buried the one window that really was thinned.
    assert ticking.jtc_output(now + int(5 * Ts * 1e9)) is None
    assert len(said) == 3 and "nothing arrived" in said[2]
    assert "half a cycle's ticks" not in said[2]
    assert ticking.measured_ns(now, now + 5_000_000) == now
    assert len(said) == 4


# -- C4: the command a state, `u` its rate ----------------------------------------

U_MAX = [1.0] * cs.ACTUATED_DOF


class C4ReplayOcp(ReplayOcp):
    """`ReplayOcp` under C4: entries are `(start, dc)` pairs, recorded as given."""

    command_state = True
    nx = cs.NX_COMMAND
    parameters = {"limits": {"u_max": U_MAX}}

    def propagate_applied(self, x, applied):
        self.replayed.append(list(applied))
        return np.asarray(x).copy()


def c4_cycle():
    one = cycle()
    one.ocp = C4ReplayOcp()
    one.measured = np.zeros(cs.NX_COMMAND)
    return one


def test_c4_seeds_a_held_start_inside_psis_domain_from_the_measured_velocity():
    one = c4_cycle()
    one.measured[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + 2] = [0.2, 3.0]
    assert one.propagate() is None
    start, rate = one.ocp.replayed[-1][0]
    assert start[:2] == pytest.approx([0.2, 1.0])
    assert np.allclose(rate, 0.0)
    assert one.in_flight_command() is start


def test_c4_dead_time_holds_the_jtc_output_then_ramps_the_command_in_flight():
    one = c4_cycle()
    one.applied_inputs = [(np.full(DOF, 0.7), np.zeros(cs.NU_PROGRESS))]
    one.last_start, one.last_input = np.full(DOF, 0.3), np.full(cs.NU_PROGRESS, 2.0)
    one.jtc_output = np.full(DOF, 0.5)
    assert one.propagate() is None
    (dead_time,), (published,) = one.ocp.replayed
    assert np.allclose(dead_time[0], 0.5) and np.allclose(dead_time[1], 0.0)
    assert np.allclose(published[0], 0.3) and np.allclose(published[1], 2.0)


def test_c4_publishes_the_command_state_and_carries_the_ramp_that_went_out():
    one = c4_cycle()
    solved = solution(Outcome.CONVERGED, nx=cs.NX_COMMAND)
    solved.inputs[:] = 0.25
    one.adopt_solution(solved)
    command = solved.states[:, cs.X_COMMAND : cs.X_COMMAND + DOF]
    assert np.array_equal(one.horizon.u[:, :DOF], command)
    one.advance(solved, NOMINAL_NS, MIN_RATE, MAX_STALL)
    assert np.array_equal(one.last_start, command[0])
    assert np.allclose(one.last_input, 0.25)

    # Shifted: knot 0 -> 1 of the previous horizon is the ramp that went out.
    one.last_tcp_states = solved.states
    one.consecutive_failures = 1
    failed = solution(Outcome.BUDGET_EXCEEDED, nx=cs.NX_COMMAND)
    assert one.ladder(failed, 3, 0.02).published
    one.advance(failed, NOMINAL_NS, MIN_RATE, MAX_STALL)
    assert one.last_start == pytest.approx(command[1])
    assert one.last_input[:DOF] == pytest.approx((command[2] - command[1]) / Ts)


def test_the_sway_centre_is_carried_along_the_plan_and_not_frozen():
    """
    The centre is what sway costs, so it has to be the plan's hanging pose.

    Frozen at the measurement it charges the hanging pose's *own* travel as
    sway -- that pose moves 1:1 with `-(q_boom + q_arm)` -- which is why
    `weights.q_u` could not be paid before issue 229. The fallback with no model
    to ask is still the frozen vector, and nothing downstream may tell the two
    apart by anything but the shape.
    """
    one = cycle()
    one.ocp = ReplayOcp()
    one.measured = np.zeros(cs.NX)
    # Swinging, not hanging: the frozen centre is this, the carried one is not.
    one.measured[cs.X_PASSIVE_POSITION] = 0.3
    assert one.propagate() is None
    assert one.q_eq.shape == (PASSIVE_DOF,)
    assert one.q_eq[0] == pytest.approx(0.3)

    model = CraneModel(problem.default_description().read_text(), Tool(problem.TOOL))
    one.equilibrium = Equilibrium(model)
    # A plan that luffs the arm through 0.4 rad, part spent, on a law that is not the
    # mean ramp: the centre has to sit where the progress row asks each stage to be.
    luff = np.outer(np.linspace(0.0, 1.0, 40), [0.0, 0.0, 0.4, 0.0, 0.0])
    one.path = PathCycle(
        control=ocp_path_control(luff),
        origin=0.25,
        nominal_rate=one.path.nominal_rate,
        s_profile=np.linspace(0.0, 1.0, 50) ** 2,
    )
    assert one.propagate() is None
    assert one.q_eq.shape == (KNOTS, PASSIVE_DOF)
    progress = one.path.progress_at(np.arange(KNOTS) * Ts)
    hanging = [
        model.passive_equilibrium([0.0, 0.0, 0.4 * s, 0.0, 0.0, 0.0]) for s in progress
    ]
    assert np.allclose(one.q_eq, hanging, atol=1e-6)
    assert one.q_eq[-1, 0] - one.q_eq[0, 0] == pytest.approx(
        -0.4 * (progress[-1] - progress[0]), abs=1e-6
    )


def test_the_centre_follows_an_off_axis_payload_around_the_rotator():
    """
    An off-axis CoM is the only thing besides the luff that moves the centre.

    It is also what a Jacobian fitted at the plan's start gets wrong: the curve
    is a harmonic in the rotator, so carrying its slope over a turn put the tilt
    centre 0.23 rad out -- past the 0.2 rad sway box, on a weight that is only
    payable while the centre is right.
    """
    model = CraneModel(problem.default_description().read_text(), Tool(problem.TOOL))
    payload = Payload(880.0, np.array([0.5, 0.0, 0.0]), valid=True)
    q_a = np.zeros(cs.ACTUATED_DOF)
    stages = np.zeros((KNOTS, PLANNED_DOF))
    stages[:, RO] = np.linspace(0.0, 2.0, KNOTS)
    table = Equilibrium(model, payload).table(q_a, stages)
    truth = np.array(
        [model.passive_equilibrium(np.append(knot, 0.0), payload) for knot in stages]
    )
    assert np.allclose(table, truth, atol=0.01)
    # Worth having: the centre really does travel here, so `atol` is not passing
    # on a curve the table failed to follow at all.
    assert np.abs(truth - truth[0]).max() > 0.1
