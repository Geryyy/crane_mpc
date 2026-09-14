"""
`mpc` §6's ladder and its shift, without a node.

The two paths `architecture-audit.md` §4 names as both load-bearing and
unreachable from a node test: the escalation that hands control back, and the
shift that keeps driving on a solve that did not converge.
"""

import numpy as np
import pytest
from crane_model import symbolic as cs
from crane_mpc import horizon as hz
from crane_mpc.cycle import HORIZON_TOPIC, Cycle
from crane_mpc.solver import Outcome, Solution, Violation

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
    # One knot on, the last duplicated: mpc §6.
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
    # A shiftable previous horizon is deliberately present: the ceiling is
    # checked before the shift, so escalation must win over it.
    one.last_horizon = hz.Knots.zeros(KNOTS)
    one.last_tcp_states = np.zeros((KNOTS, cs.NX))
    one.consecutive_failures = 3

    verdict = one.ladder(solution(Outcome.BUDGET_EXCEEDED), 3, 0.02)

    assert not verdict.published
    assert verdict.severity == "error"
    assert one.escalated
    assert not one.applied_previous
    assert "escalation" in one.last_silence
    # Nothing may be shifted next cycle either.
    assert one.last_horizon is None
    assert one.guess is None


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
