"""
`looptiming.py` has to read back the timing a capture was built with -- and refuse the rest.

The live half of issue 189 is human-run (it needs a Gazebo), so the measurement
is checked against a synthetic capture whose latency and tick rate are known.
The refusals are tested too: a wrong number in the issue's table does not leave
the harness unvalidated, it declares it validated, which is worse than an empty
cell.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trials"))

import looptiming  # noqa: E402

JOINT = "q1_slewing_joint"
SRC = "/trajectory_controllers/controller_state"
TS = 0.06
TICK = 0.01  # NOMINAL_JTC_HZ
LATENCY = 0.018


def build(latency=LATENCY, tick=TICK, rtf=1.0, stamps=True, seconds=12.0):
    """A capture on a `tick` controller grid, horizons every `TS`, delivered `latency` late."""

    def row(stamp, **kw):
        at = {"sim": stamp, "rx": stamp} if stamps else {}
        return dict(t=1.7e9 + stamp / rtf, **at, **kw)

    js, ctrl = [], []
    for k in range(int(seconds / tick)):
        t = k * tick
        ctrl.append(row(t, src=SRC, names=[JOINT], out=[np.sin(t)]))
        js.append(row(t, names=[JOINT], pos=[0.0], vel=[np.sin(t)]))
    hz = [
        row(k * TS, names=[JOINT], pos=[0.0], vel=[0.0], eff=[0.0])
        | {"rx": k * TS + latency if stamps else None}
        for k in range(int(seconds / TS))
    ]
    if not stamps:
        hz = [{k: v for k, v in r.items() if v is not None} for r in hz]
    return dict(js=js, ctrl=ctrl, hz=hz, health=[], ref=[], path=[], refmsg={})


def test_recovers_latency_and_the_tick_rate():
    cap = build()
    assert looptiming.stamped(cap)
    assert looptiming.cycle_period(cap) == pytest.approx(TS, rel=1e-6)
    assert looptiming.latency(cap)["p50"] == pytest.approx(LATENCY, abs=1e-9)
    rate = looptiming.controller_rate(cap)
    assert rate["hz"] == pytest.approx(1 / TICK, rel=1e-3)
    assert rate["per_cycle"] == pytest.approx(TS / TICK, rel=1e-3)
    assert rate["starved"] == 0.0


def test_uniform_decimation_is_counted_as_starvation():
    """
    The fault the issue reports is load-driven, so it is not phase-locked to the cycle.

    Normalised against the *measured* rate this reads as a healthy slow loop;
    the quorum has to be the nominal tick count. Two of every three dropped
    leaves 2 ticks against a quorum of 3.
    """
    cap = build()
    cap["ctrl"] = cap["ctrl"][::3]
    rate = looptiming.controller_rate(cap)
    assert rate["hz"] == pytest.approx(1 / (3 * TICK), rel=1e-2)
    assert rate["expected"] == pytest.approx(TS / TICK)
    assert rate["starved"] > 0.9


def test_the_busiest_source_is_taken_alone():
    """controller_state is shared with the grasping controller; merged it reads double."""
    cap = build()
    other = [
        dict(r, src="/trajectory_controller_a2b/controller_state") for r in cap["ctrl"]
    ]
    cap["ctrl"] = cap["ctrl"] + other[: len(other) // 4]
    rate = looptiming.controller_rate(cap)
    assert rate["sources"] == 2 and rate["src"] == SRC
    assert rate["hz"] == pytest.approx(1 / TICK, rel=1e-3)


def test_cadence_phase_is_refused_rather_than_reported():
    """It is 0 by construction: the horizon stamp *is* a /joint_states stamp."""
    text = looptiming.report(build())
    assert "cadence phase  NOT MEASURABLE" in text
    assert "construction" in text and "tick instant" in text


def test_dead_time_is_refused_and_points_at_the_real_lever():
    """`sensor_to_valve_delay` is pinned to Ts by `config.check_settings`."""
    text = looptiming.report(build())
    assert "dead time      NOT MEASURABLE" in text
    assert "steptest.py" in text and "--lag-shift" in text


def test_a_capture_without_stamps_is_refused_not_rescaled():
    """Wall `t` at RTF != 1 is out by 1/rtf; a rate off it would be silently halved."""
    cap = build(rtf=0.5, stamps=False)
    assert not looptiming.stamped(cap)
    text = looptiming.report(cap)
    assert "no usable sim/rx stamps" in text
    # The wall clock really is out by 1/rtf here, which is why it is refused.
    wall = np.diff(sorted(r["t"] for r in cap["ctrl"]))
    assert float(np.median(wall)) == pytest.approx(TICK / 0.5, rel=1e-3)


def test_a_pid_capture_with_no_horizons_reports_instead_of_crashing():
    """`trial.sh CTRL=pid` is the stable reference arm: no MPC, so no horizons."""
    cap = build()
    cap["hz"] = []
    assert looptiming.latency(cap) is None
    assert looptiming.cycle_period(cap) is None
    text = looptiming.report(cap)
    assert "no horizons" in text
    # The rate is still measurable without a cycle grid; the quorum is not.
    rate = looptiming.controller_rate(cap)
    assert rate["hz"] == pytest.approx(1 / TICK, rel=1e-3)
    assert "expected" not in rate


def test_a_clock_that_lags_reads_negative_and_is_not_passed_on_as_a_delay():
    cap = build(latency=-0.004)
    lag = looptiming.latency(cap)
    assert lag["p50"] == pytest.approx(-0.004, abs=1e-9) and lag["negative"] == lag["n"]
    # wire_chain would pop a past arrival immediately; clamp rather than hand it over.
    assert "--latency 0.000" in looptiming.report(cap)
