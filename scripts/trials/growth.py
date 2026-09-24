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


def main():
    model = CraneModel(problem.default_description().read_text(), Tool.PZS100)
    print(f"{'arm':>12} {'growth /s':>10} {'doubling':>9} {'n':>5} {'dev max':>8}")
    for path, label in zip(sys.argv[1::2], sys.argv[2::2]):
        t, dev = deviation(json.load(open(path))["js"], model)
        rate, n = growth(t, dev)
        doubling = np.log(2) / rate if rate > 0 else float("inf")
        print(f"{label:>12} {rate:10.3f} {doubling:9.1f} {n:5d} {dev.max():8.4f}")


main()
