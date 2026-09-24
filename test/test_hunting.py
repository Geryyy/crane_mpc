"""
The hunting detector: it has to fire on the failure it was built for and only on it.

The oscillating case is the one issue 155 measured on the live graph -- slewing's
command flipping sign every cycle at +-0.8 to 1.1 rad/s against a plan moving
that axis 0.015 rad in total. The quiet case is the `p = 0` rungs, which park on
the solver's 11 mrad steady-state offset and do not decay: expected, not
instability, and a detector that flags it cannot tell the two apart.
"""

from __future__ import annotations

import numpy as np
from crane_mpc.hunting import hunting_report

CYCLES = 120
AXES = 5


def _run(command, dq):
    """One synthetic run, slewing carrying the signal and the rest at rest."""
    u = np.zeros((CYCLES, AXES))
    rate = np.zeros((CYCLES + 1, AXES))
    u[:, 0] = command
    rate[:, 0] = dq
    return u, rate


#: The run's last second, as `sim_chain.hunting_score` computes it at Ts = 60 ms.
SETTLED = CYCLES - 16


def test_it_fires_on_a_command_alternating_at_the_cycle_rate():
    """The live-graph failure: +-0.9 rad/s, sign flipping every cycle."""
    alternating = 0.9 * (-1.0) ** np.arange(CYCLES)
    u, rate = _run(alternating, 0.25 * (-1.0) ** np.arange(CYCLES + 1))
    report = hunting_report(u, rate, settled=SETTLED, names=list("abcde"))
    assert report["hunting"]
    assert report["hunting_axes"] == ["a"]
    # Every consecutive pair reverses, so the fraction is 1 up to the last pair.
    assert report["reversals_worst"] > 0.99


def test_it_stays_quiet_on_a_run_that_merely_parked_off_target():
    """
    A one-sided command, and an axis still settling through the last second.

    That last part is the point. A long move leaves real rate at the end --
    0.127 rad/s, measured on a run reversing on 3 % of cycles -- so a threshold
    in rad/s would call this a hunt. The reversal count does not.
    """
    move = np.concatenate([np.linspace(0.0, 0.4, 60), np.linspace(0.4, 0.05, 60)])
    u, rate = _run(move, np.concatenate([move, [0.05]]) * 0.5)
    report = hunting_report(u, rate, settled=SETTLED)
    assert not report["hunting"]
    assert report["hunting_axes"] == []
    assert report["reversals_worst"] == 0.0
    # Reported, never flagged on its own: the axis is demonstrably still moving.
    assert report["dq_sustained_worst"] > 0.0


def test_dither_around_zero_is_not_a_reversal():
    """Below the dead band the command is the solver's numerics, not a flip."""
    u, rate = _run(1.0e-3 * (-1.0) ** np.arange(CYCLES), np.zeros(CYCLES + 1))
    assert hunting_report(u, rate, settled=SETTLED)["reversals_worst"] == 0.0
