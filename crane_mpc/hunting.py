"""
Is the chain off, or is it hunting? Two numbers, because those are two faults.

`sim_chain.tracking_report` scores where the tool ended up. That cannot tell
"arrived 11 mrad off" -- which is what a `p = 0` rung does, and is expected --
from "arrived oscillating", which is a chain about to break a crane. One scalar
covering both would rank the harmless one as the worse. This is the second
number, and it is reported beside the first, never folded into it.

The observable is the one issue 155 measured on the live graph: the node's own
command reversing sign *every cycle*, at the axis's `u_clamp`, while the
position reference underneath it stayed a smooth ramp. Sign alternation at the
cycle rate is a one-step delay instability. So the detector counts reversals at
the cycle rate rather than looking for energy near `f_n`: `f_n` moves with pose
(`M_ii` by x132 on slewing) while the cycle rate does not.

**The reversal count is the flag; the rate left at the end is reported beside
it and does not flag on its own.** Measured on `sim_chain --random`: a quiet run
at zero delay still carried 0.127 rad/s in its last second while reversing on
3 % of cycles, and a hunting run at 60 ms of delay carried 3.55 rad/s while
reversing on 92 %. The rate separates the two by a factor of 28 but only through
a threshold in rad/s, which depends on how big the move was and how much settle
it was given -- so it would call a long move that is still settling a hunt, and
that is exactly the `p = 0` rung's parked offset misread as instability.
"""

from __future__ import annotations

import numpy as np

#: A command reversing on at least this fraction of consecutive cycles is
#: hunting. A move reverses an axis a handful of times over hundreds of cycles;
#: a delay instability reverses it on every one. Halfway between, so that
#: neither case sits near the threshold.
REVERSAL_FRACTION = 0.5

#: Commands nearer zero than this are the solver's own dither, not a reversal.
#: 1 % of slewing's `u_clamp` (`velocity_loop.yaml`, -0.96/+0.94 rad/s), which
#: is the axis the failure was measured on. One number across all five axes, so
#: on the telescope it is 1.8 % of a clamp in m/s rather than rad/s -- same
#: order, and a per-axis band would be a tuning decision this issue does not make.
COMMAND_DEAD_BAND = 0.01


def reversal_fraction(command, dead_band: float = COMMAND_DEAD_BAND) -> np.ndarray:
    """
    Per axis: the share of consecutive cycles on which the command changes sign.

    Over *all* cycle pairs, not just the ones outside the dead band: normalising
    by the live pairs alone would score an axis that moved twice and reversed
    once as a full-blown oscillation.
    """
    u = np.atleast_2d(np.asarray(command, dtype=float))
    pairs = u.shape[0] - 1
    if pairs < 1:
        return np.zeros(u.shape[1])
    first, second = u[:-1], u[1:]
    live = (np.abs(first) > dead_band) & (np.abs(second) > dead_band)
    return np.count_nonzero(live & (np.sign(first) != np.sign(second)), axis=0) / pairs


def hunting_report(
    command,
    dq,
    settled: int,
    names=None,
    dead_band: float = COMMAND_DEAD_BAND,
) -> dict:
    """
    Score hunting on one run: command reversals, and the rate left at the end.

    `command` is the per-cycle `u` the node would publish and `dq` the plant's
    joint rates on the same grid. `settled` is the first row of the window the
    run is judged to have *finished* in -- the caller's last second, as
    `tracking_report` judges `sway_end` -- not the arrival: a move arrives and
    then settles, and peak rate over the whole settle is settling, not hunting.
    """
    reversals = reversal_fraction(command, dead_band)
    rate = np.atleast_2d(np.asarray(dq, dtype=float))
    sustained = np.abs(rate[min(settled, rate.shape[0] - 1) :]).max(axis=0)
    flagged = reversals >= REVERSAL_FRACTION
    labels = (
        [f"axis{axis}" for axis in range(len(reversals))] if names is None else names
    )
    return {
        "reversals": [float(value) for value in reversals],
        "dq_sustained": [float(value) for value in sustained],
        "reversals_worst": float(reversals.max()),
        "dq_sustained_worst": float(sustained.max()),
        "hunting": bool(flagged.any()),
        "hunting_axes": [
            str(labels[axis]) for axis in range(len(flagged)) if flagged[axis]
        ],
    }
