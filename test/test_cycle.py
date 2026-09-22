"""
The escalation ladder and its shift, without a node.

Two paths load-bearing but unreachable from a node test: the escalation
that hands control back, and the shift that keeps driving on a solve that
did not converge.
"""

import numpy as np
import pytest
from crane_model import symbolic as cs
from crane_mpc import horizon as hz
from crane_mpc import reports
from crane_mpc.cycle import (
    HORIZON_TOPIC,
    Cycle,
    carry_actuated_velocity,
    watch_progress,
)
from crane_mpc.solver import Outcome, Solution, Violation
from crane_msgs.msg import SolverHealth, SupervisorStatus
from rclpy.time import Time

KNOTS = 5
Ts = 0.04


def cycle(mode="active"):
    made = Cycle(Ts, hz.Grid(Ts, KNOTS), mode, 0.0)
    made.horizon = hz.Knots.zeros(KNOTS)
    return made


def solution(outcome, progress_advance=Ts):
    """A solve whose states count up, so a shift of them is visible."""
    states = np.arange(KNOTS * cs.NX, dtype=float).reshape(KNOTS, cs.NX)
    return Solution(
        states=states,
        inputs=np.zeros((KNOTS - 1, cs.NU_PROGRESS)),
        outcome=outcome,
        status=2,
        status_word="ACADOS_MAXITER",
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
    assert "10 ms against a 20 ms budget" in verdict.text
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


def test_the_shadow_mode_names_its_own_topic_in_the_verdict():
    one = cycle("shadow")
    verdict = one.ladder(solution(Outcome.CONVERGED), 3, 0.02)
    assert HORIZON_TOPIC not in verdict.text
    assert "shadow_horizon" in verdict.text


# -- where x_0's actuated velocity comes from, per axis -------------------------

DOF = cs.K_PLANNED_DOF
VELOCITY = cs.X_PLANNED_VELOCITY


def measured_state():
    """A state whose rows all count up, so a write to the wrong one is caught."""
    return np.arange(cs.NX, dtype=float)


def test_every_axis_on_measurement_leaves_the_state_untouched():
    x = measured_state()
    report = carry_actuated_velocity(x, [99.0] * DOF, [True] * DOF, [1e-9] * DOF)
    assert x == pytest.approx(measured_state())
    assert not report.any_diverged
    assert not any(report.carried)


def test_an_opted_out_axis_takes_the_model_velocity_and_nothing_else_moves():
    x = measured_state()
    carried = [0.0] * DOF
    carried[1] = 0.25
    from_measurement = [True] * DOF
    from_measurement[1] = False

    report = carry_actuated_velocity(x, carried, from_measurement, [1.0] * DOF)

    expected = measured_state()
    expected[VELOCITY + 1] = 0.25
    assert x == pytest.approx(expected)
    assert report.carried[1]
    assert report.divergence[1] == pytest.approx(0.25 - measured_state()[VELOCITY + 1])
    # The tool row is six wide for symmetry and is never carried.
    assert not report.carried[cs.K_TOOL_AXIS]


def test_divergence_past_the_bound_is_reported_and_at_it_is_not():
    from_measurement = [True] * DOF
    from_measurement[0] = False
    bounds = [1.0] * DOF
    measured = measured_state()[VELOCITY]

    carried = [0.0] * DOF
    carried[0] = measured + 0.1
    bounds[0] = 0.05
    report = carry_actuated_velocity(
        measured_state(), carried, from_measurement, bounds
    )
    assert report.diverged[0] and report.any_diverged
    assert report.divergence[0] == pytest.approx(0.1)

    bounds[0] = 0.1
    report = carry_actuated_velocity(
        measured_state(), carried, from_measurement, bounds
    )
    assert not report.diverged[0]
    assert not report.any_diverged


def test_a_carried_axis_reports_its_divergence_on_the_comparison():
    one = cycle("shadow")
    x = measured_state()
    carried = [0.0] * DOF
    carried[1] = x[VELOCITY + 1] + 0.5
    from_measurement = [True] * DOF
    from_measurement[1] = False
    one.velocity_carry = carry_actuated_velocity(
        x, carried, from_measurement, [0.1] * DOF
    )

    joints = [f"axis{axis}" for axis in range(cs.K_ACTUATED_DOF)]
    message = reports.shadow_comparison(
        one, "a verdict", None, joints, 0.02, Time().to_msg()
    )
    keys = {entry.key: entry.value for entry in message.status[0].values}

    assert float(keys["axis1.dq_a_divergence"]) == pytest.approx(0.5)
    assert keys["axis1.dq_a_diverged"] == "true"
    assert not [key for key in keys if key.startswith("axis0.dq_a")]


def test_a_break_in_the_output_stream_drops_the_carry():
    """A silence is this node's reactivation: the next cycle re-seeds from measurement."""
    one = cycle()
    one.dq_a_carried = np.zeros(DOF)
    one.stay_silent("a gate refused")
    assert one.dq_a_carried is None


def test_a_carry_that_is_not_finite_falls_back_to_the_measurement():
    x = measured_state()
    carried = [np.nan] * DOF
    report = carry_actuated_velocity(x, carried, [False] * DOF, [1.0] * DOF)
    assert x == pytest.approx(measured_state())
    assert not any(report.carried)


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


def test_a_stalled_plan_reports_fault_solver_on_a_converged_solve():
    one = stall(cycle())
    assert one.progress_stalled
    joints = [f"axis{axis}" for axis in range(cs.K_ACTUATED_DOF)]
    health = reports.solver_health(
        one,
        solution(Outcome.CONVERGED, 0.0),
        "a verdict",
        joints,
        0.02,
        Time().to_msg(),
    )
    assert health.outcome == SolverHealth.SOLVE_CONVERGED
    assert health.fault == SupervisorStatus.FAULT_SOLVER


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


# -- the preparation, and everything that must invalidate it -------------------
#
# A preparation is last cycle's linearisation of this cycle's problem. Standing
# on a stale one is silent: the solve returns 0 and the plan is optimal for a
# problem the machine is not in. These check that every way the problem can move
# puts the cycle back on the whole solve.


def prepared_cycle():
    one = cycle()
    one.prepared = True
    one.prepared_horizon = one.horizon.copy()
    return one


def test_a_preparation_stands_only_for_the_plan_it_was_taken_on():
    one = prepared_cycle()
    assert one.preparation_still_applies()

    # A resample the cadence moved: same reference, different knots.
    one.horizon = one.horizon.copy()
    one.horizon.q_a_ref[2, 0] += 1e-6
    assert not one.preparation_still_applies()


def test_a_new_reference_drops_the_preparation():
    """`adopt_reference` does not call `forget_plan` -- the warm start survives
    a re-plan -- so the preparation has to be dropped on its own."""
    one = prepared_cycle()
    one.adopt_reference(hz.Knots.zeros(KNOTS), 0)
    assert not one.prepared


@pytest.mark.parametrize(
    "break_it",
    [
        lambda one: one.forget_plan(),
        lambda one: one.stay_silent("a gate"),
        lambda one: one.stay_silent_after_failure("a failure"),
        lambda one: one.payload_changed(),
        lambda one: one.adopt_mode("shadow"),
    ],
)
def test_every_break_in_the_output_stream_drops_the_preparation(break_it):
    one = prepared_cycle()
    break_it(one)
    assert not one.prepared
    assert one.prepared_horizon is None


class StubOcp:
    """Records whether `prepare_next` got as far as linearising anything."""

    split_rti = True

    def __init__(self):
        self.prepared_with = []

    def prepare(self, x0, horizon, q_eq, guess):
        self.prepared_with.append((x0, horizon, q_eq, guess))
        return 0.001


def preparable_cycle():
    """A cycle with everything `prepare_next` needs, so a refusal means something."""
    one = cycle()
    one.ocp = StubOcp()
    one.guess = object()
    one.reference = hz.Knots.zeros(KNOTS)
    one.reference.t[:] = np.linspace(0.0, Ts * (KNOTS - 1), KNOTS)
    one.reference_anchored = True
    one.reference_progress = 0.0
    return one


def test_a_prepared_cycle_linearises_about_the_state_it_predicts():
    one = preparable_cycle()
    answer = solution(Outcome.CONVERGED)
    one.prepare_next(answer)

    assert one.prepared
    x0, _, _, _ = one.ocp.prepared_with[0]
    # states[1], the optimizer's own one step ahead -- not the state this cycle
    # measured, and not a second integration.
    assert np.array_equal(x0, answer.states[1])


def test_a_failed_solve_prepares_nothing_for_the_next_cycle():
    one = preparable_cycle()
    one.prepare_next(solution(Outcome.FAILED))
    assert not one.prepared
    assert one.ocp.prepared_with == []


def test_without_a_warm_start_there_is_nothing_to_prepare_from():
    one = preparable_cycle()
    one.guess = None
    one.prepare_next(solution(Outcome.CONVERGED))
    assert not one.prepared
    assert one.ocp.prepared_with == []


def test_the_progress_the_solver_bought_is_read_as_a_path_parameter():
    """
    `s` spans the horizon's window, so a cycle's advance is `s` x that window.

    Read as seconds it is a factor `duration()` short -- here 4.8x -- which is a
    plan that crawls, not one that stalls, so nothing else in this file sees it.
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
