"""
Is the chain off, or is it hunting? A trial score, kept apart from goal error.

A final-error score cannot tell "arrived 11 mrad off" -- what a `p = 0` rung
does, and expected -- from "arrived oscillating", a chain about to break a crane.
The observable of the second is the node's own command reversing sign *every*
cycle at the axis's `u_clamp` while the position reference under it stays a
smooth ramp: a one-step delay instability. So this counts reversals at the cycle
rate rather than looking for energy near `f_n`, which moves with pose (`M_ii` by
x132 on slewing) where the cycle rate does not.

A rad/s threshold on the rate left at the end would do it too -- a quiet run
carried 0.127 rad/s against a hunting one's 3.55 -- but only for a known move
length and settle, so it reads a long move still settling as a hunt.
"""

from __future__ import annotations

import numpy as np

#: A command reversing on at least this fraction of consecutive cycles is
#: hunting. A move reverses an axis a handful of times over hundreds of cycles;
#: a delay instability reverses it on every one. Halfway between, so neither case
#: sits near the threshold.
REVERSAL_FRACTION = 0.5

#: Commands nearer zero than this are the solver's own dither, not a reversal.
#: 1 % of slewing's `u_clamp` (`velocity_loop.yaml`), the axis the failure was
#: measured on; one number across all five axes, so on the telescope it is 1.8 %
#: of a clamp in m/s -- same order, and a per-axis band would be a tuning choice.
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
