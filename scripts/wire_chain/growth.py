#!/usr/bin/env python3
"""
Score a trial on the divergence growth rate, not on goal error.

dev(t) = max |q_u - passive_equilibrium(q_a)| from /joint_states. Fit
log(dev) against t over the band where the mode is still growing and not yet
saturated (2e-3 < dev < 0.2). The slope is 1/s: <= 0 is stable.

Goal error is noise here -- identical config scatters 20x -- because the end
state depends on *when* an exponential saturates. The slope does not.
"""

import json
import sys

import numpy as np
from crane_model.conventions import (
    ACTUATED_INDICES,
    PASSIVE_INDICES,
    Tool,
    canonical_joints,
)
from crane_model.model import CraneModel
from crane_mpc import problem

LO, HI = 2e-3, 0.2
#: Move-start ring scoring, the window the machine was read on
#: (`records/analysis_304/critic2/onset.py`, issue 308's correction).
RING_WINDOW_S = 0.8
RING_SMOOTH = 9  # cycles of mean the ring is read against
RING_FLOOR = 2e-3  # rad/s ptp; the sensor-noise floor, under the plan's own peak
CANON = canonical_joints()
ACT = [CANON[i] for i in ACTUATED_INDICES]
PAS = [CANON[i] for i in PASSIVE_INDICES]


def deviation(js, model):
    """
    |q_u - q_eq(q_a)| per sample, against sim time where the capture has it.

    Wall clock is the wrong clock: sim RTF varies with machine load, so the
    same physical divergence scores differently run to run. Older captures
    carry only `t` and fall back to it.
    """
    t, dev = [], []
    for row in js:
        idx = {n: i for i, n in enumerate(row["names"])}
        if any(j not in idx for j in ACT + PAS):
            continue
        q_a = np.array([row["pos"][idx[j]] for j in ACT])
        q_u = np.array([row["pos"][idx[j]] for j in PAS])
        t.append(row.get("sim", row["t"]))
        dev.append(np.max(np.abs(q_u - model.passive_equilibrium(q_a))))
    return np.asarray(t) - (t[0] if t else 0.0), np.asarray(dev)


def growth(t, dev):
    # First rise only. Past the first crossing of HI the mode has saturated and
    # what comes back through the band is the load rattling in its stops, which
    # fits a slope of its own and hides the one that matters.
    over = np.nonzero(dev >= HI)[0]
    band = np.zeros(len(dev), bool)
    band[: over[0] if len(over) else len(dev)] = True
    band &= dev > LO
    if band.sum() < 10:
        return float("nan"), int(band.sum())
    return float(np.polyfit(t[band], np.log(dev[band]), 1)[0]), int(band.sum())


def ring(t, dq, floor=RING_FLOOR):
    """
    One axis' ring from move start: `(ptp per window, growth per cycle, Hz)`.

    The machine never arrived, so a window scored after arrival measures
    nothing it has (issue 308's correction) -- this one starts where the move
    does, which is what makes the bench comparable to
    `records/2026-11-09_16-16-56-stacking_mpc` (arm 0.017 .. 0.243 rad/s
    at 1.10 Hz).

    `dq` is high-passed by subtracting a `RING_SMOOTH`-cycle mean, so the plan's
    own motion drops out. Mind the shape: at `Ts` 0.12 s the mean nulls
    0.93 Hz, so the high-pass *passes* 0.93 Hz at unity, overshoots to ~1.2x
    around 1.3 Hz and attenuates 0.35 Hz to ~0.2 -- two cells at different
    lines are read through different gains. The window is `round(0.8/Ts)`
    cycles, i.e. 0.84 s at the shipped `Ts` and not 0.8.

    `floor` is the amplitude below which there is no ring to score -- pass the
    plan's own peak `|dq|` on this axis, because a window the size of the
    planned motion is the planned motion. Under it the growth and the line come
    back `nan` rather than fitting the plan's ramp to arrival and dividing it by
    whatever bin the spectrum happens to peak in. Growth is the first rise only:
    past the largest window the ring has saturated and fits a slope of its own.
    """
    step = float(np.median(np.diff(t)))
    high = dq - np.convolve(dq, np.ones(RING_SMOOTH) / RING_SMOOTH, "same")
    width = max(2, int(round(RING_WINDOW_S / step)))
    starts = np.arange(0, len(high) - width + 1, width)
    ptp = np.array([np.ptp(high[i : i + width]) for i in starts])
    band = ptp > max(float(floor), RING_FLOOR)
    peak = int(np.argmax(ptp))
    rise = band.copy()
    if band[: peak + 1].sum() >= 3:
        rise[peak + 1 :] = False  # growing: past the peak it has saturated
    if rise.sum() < 3:
        return ptp.tolist(), float("nan"), float("nan")
    live = np.zeros(len(high), bool)
    for index in np.nonzero(band)[0]:
        live[starts[index] : starts[index] + width] = True
    spectrum = np.abs(np.fft.rfft(high * live * np.hanning(len(high))))
    frequency = np.fft.rfftfreq(len(high), step)
    keep = frequency > 0.2  # below that is the move's own settling, not a ring
    line = float(frequency[keep][np.argmax(spectrum[keep])])
    slope = np.polyfit(t[starts][rise] + 0.5 * RING_WINDOW_S, np.log(ptp[rise]), 1)[0]
    return ptp.tolist(), float(np.exp(slope / line)), line


def main():
    model = CraneModel(problem.default_description().read_text(), Tool.PZS100)
    print(f"{'arm':>12} {'growth /s':>10} {'doubling':>9} {'n':>5} {'dev max':>8}")
    for path, label in zip(sys.argv[1::2], sys.argv[2::2]):
        t, dev = deviation(json.load(open(path))["js"], model)
        rate, n = growth(t, dev)
        doubling = np.log(2) / rate if rate > 0 else float("inf")
        print(f"{label:>12} {rate:10.3f} {doubling:9.1f} {n:5d} {dev.max():8.4f}")


if __name__ == "__main__":
    main()
