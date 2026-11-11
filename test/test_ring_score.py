"""
The ring score's contract: one metric at either rate, scored on every axis.

Issue 312 moved the score from the 0.12 s cycle log to the control tick and has
to stay comparable to 307's grid, which was taken at the cycle. The high-pass
was a 9-*sample* mean -- at 100 Hz that passes 11 Hz and leaves the plan's own
motion inside the window, so the same signal would have scored differently at
the two rates.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wire_chain"))

from growth import rings  # noqa: E402

LINE_HZ = 1.1  # the machine's ring, records/2026-11-09_16-16-56-stacking_mpc
GROWTH = 1.2  # per ring cycle
PLAN_PEAK = 0.05  # rad/s, stands in for the floor wire_chain passes


def ringing(step, line=LINE_HZ, growth=GROWTH, seconds=20.0, start=0.004):
    """The plan's own ramp plus a sinusoid growing `growth` per ring cycle."""
    t = step * (1 + np.arange(int(round(seconds / step))))
    envelope = start * growth ** (line * t)
    return t, PLAN_PEAK * np.tanh(t / 4.0) + envelope * np.sin(2 * np.pi * line * t)


def quiet(step, seconds=20.0):
    t = step * (1 + np.arange(int(round(seconds / step))))
    return t, PLAN_PEAK * np.tanh(t / 4.0)


def one_axis(step, line=8.0, growth=1.02):
    """One fast ring, scored alone: `(ptp, growth per cycle, Hz)`."""
    t, dq = ringing(step, line=line, growth=growth)
    return rings(t, dq[:, None], [PLAN_PEAK], ["ka"])[0]["ka"]


def score_one(step):
    t, dq = ringing(step)
    scores, worst = rings(t, dq[:, None], [PLAN_PEAK], ["ka"])
    assert worst == "ka"
    return scores["ka"]


def test_score_is_the_same_at_the_tick_and_at_the_cycle():
    """The 0.12 s cycle and a 100 Hz tick read one ring the same way."""
    cycle_ptp, cycle_growth, cycle_hz = score_one(0.12)
    tick_ptp, tick_growth, tick_hz = score_one(0.01)
    assert abs(tick_hz - cycle_hz) < 0.05  # one spectral bin of a 20 s run
    assert abs(tick_growth / cycle_growth - 1.0) < 0.05
    assert abs(max(tick_ptp) / max(cycle_ptp) - 1.0) < 0.05


def test_worst_axis_is_the_fastest_growing_one():
    t, fast = ringing(0.01, growth=1.2)
    _, slow = ringing(0.01, line=0.9, growth=1.15)
    _, still = quiet(0.01)
    scores, worst = rings(
        t,
        np.column_stack([still, slow, fast]),
        [PLAN_PEAK] * 3,
        ["sw", "ha", "ka"],
    )
    assert worst == "ka"
    assert not np.isfinite(scores["sw"][1])  # a ramp alone is no ring
    assert scores["ha"][1] < scores["ka"][1]


def test_no_axis_rings_names_nobody():
    t, still = quiet(0.01)
    scores, worst = rings(
        t, np.column_stack([still] * 2), [PLAN_PEAK] * 2, ["tip", "tilt"]
    )
    assert worst is None
    assert all(not np.isfinite(s[1]) for s in scores.values())


def test_a_decaying_ring_is_not_a_worst_axis():
    """
    The calm baseline cell scores a 0.90 /cycle rotator (`records/analysis_312`).

    Naming it would make `None` unreachable, so a gate on "an axis was named"
    would fire on the one cell that is meant to pass.
    """
    t, decaying = ringing(0.01, growth=1.0 / 1.2, start=0.4)
    scores, worst = rings(t, decaying[:, None], [PLAN_PEAK], ["ro"])
    assert scores["ro"][1] < 1.0  # scored, and decaying
    assert worst is None


def test_a_mode_over_the_cycle_nyquist_is_only_seen_at_the_tick():
    """4.17 Hz is Nyquist at Ts 0.12 s -- the whole reason for scoring at 10 ms."""
    tick = one_axis(0.01)
    cycle = one_axis(0.12)
    assert abs(tick[2] - 8.0) < 0.1  # the line, Hz
    # aliased to 0.33 Hz, where the high-pass attenuates it under the floor:
    # the 0.12 s log reports no line, and could not report this one if it did
    assert np.isnan(cycle[2]) or cycle[2] < 4.17
