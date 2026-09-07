#!/usr/bin/env python3
"""
Give the MPC a reference it cannot follow, and show it spends time instead.

This is issue 119's acceptance test and it is the whole point of the progress
state. The same move is run twice on the **same generated solver**, differing
only in the runtime price of spending time:

* **time-scaling** -- the shipped `weights.progress_rate`. The optimizer may let
  the progress rate fall below one, which holds the reference back and stops the
  tracking residual from growing;
* **time-indexed** -- the same problem with that weight raised until the rate is
  pinned at one to several decimals. That *is* the controller this design
  replaces: `wiki/robot_model.md` §4.4 before issue 119, and `crane_mpc` before
  it, indexed the horizon by wall clock off the reference stamp.

The reference is the ordinary A-to-B move compressed into a duration the machine
cannot achieve, so the demanded joint velocities exceed `dq_a_max` and no input
sequence tracks it. What has to come out is the progress rate dropping below one
in the first run and **not** in the second, with the tracking error and its cost
lower in the first -- the machine arriving late rather than never.

    ./scripts/cannot_follow.py                     # the shipped weights
    ./scripts/cannot_follow.py --move-duration 1.0 # harder still

Everything is reused from `mpc_a2b.py`: the same plant, the same solver cache,
the same reference and the same closed loop. Nothing is simulated twice here.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

warnings.filterwarnings(
    "ignore", message="to-Python converter for pinocchio.*", category=RuntimeWarning
)

import mpc_a2b  # noqa: E402

cs = mpc_a2b.cs

#: What "pinned at one" costs. Large enough that the rate holds to ~1e-4 against
#: this problem's other terms, small enough that the Hessian is not degenerate.
PINNED_PROGRESS_SCALE = 1.0e6

#: The move. It is deliberately **long, unreachable, and otherwise ordinary.**
#:
#: A pure slew sweep from -1.8 rad to +1.8 rad in six seconds. The quintic's peak
#: rate is 1.875 x 3.6 / 6 = 1.125 rad/s against the slewing axis' control-safe
#: 0.802, so the reference asks for **1.40x** what constraint 2 allows, and it
#: asks for it for most of the six seconds. Every other axis holds the pose
#: `mpc_a2b`'s default A-to-B starts in, which is the pose issues 116 and 117
#: measured on.
#:
#: Two earlier shapes of this test were wrong and are worth not repeating.
#: **Short and violent is the wrong test**: `mpc_a2b`'s default move compressed
#: into 1.5 s does ask for 1.45x the arm's limit, but the reference is *finished*
#: 1.5 s in, so from then on holding the plan back changes nothing -- there is no
#: plan left to hold. **Large and multi-axis is a different test**: a six-second
#: move to `(2.4, 1.4, 2.4, 1.8, 2.0)` folds the boom and arm toward the region
#: `docs/features/mpc-full-authority/brief.md` says the MPC does not drive and
#: issue 126 is about, and what it measures is that configuration and not the
#: reference. This one moves one axis, inside its range, in the ordinary pose.
UNREACHABLE_START = (-1.80, 0.30, 0.80, 0.60, 0.00)
UNREACHABLE_GOAL = (1.80, 0.30, 0.80, 0.60, 0.00)
UNREACHABLE_DURATION = 6.0


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--move-duration",
        type=float,
        default=UNREACHABLE_DURATION,
        help="seconds the reference allots to the move. Shortening it makes the "
        "reference harder, and shortening it far enough makes it a different "
        "test -- see UNREACHABLE_GOAL",
    )
    parser.add_argument("--settle-duration", type=float, default=6.0)
    parser.add_argument(
        "--progress-scale",
        type=float,
        default=1.0,
        help="multiplier on the shipped weights.progress_rate for the first run, "
        "so the price of time can be swept without editing the deployment yaml",
    )
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--horizon-knots", type=int, default=50)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="how far below one the progress rate must fall for the "
        "time-scaling run to count as having spent time",
    )
    return parser.parse_args()


def run(arguments: argparse.Namespace, progress_scale: float) -> mpc_a2b.RunData:
    """One closed-loop run of `mpc_a2b`'s own simulation at a given time price."""
    argv = [
        "mpc_a2b.py",
        "--move-duration",
        str(arguments.move_duration),
        "--settle-duration",
        str(arguments.settle_duration),
        "--dt",
        str(arguments.dt),
        "--horizon-knots",
        str(arguments.horizon_knots),
        "--progress-scale",
        str(progress_scale),
        "--a",
        *[str(value) for value in UNREACHABLE_START],
        "--b",
        *[str(value) for value in UNREACHABLE_GOAL],
        "--no-csv",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        parsed = mpc_a2b.parse_arguments()
    finally:
        sys.argv = saved
    parameters, hydraulics = mpc_a2b.load_settings(parsed)
    a, b = mpc_a2b.validate_movement(parsed, parameters)
    solver, model, scale = mpc_a2b.create_solver(parsed, parameters, hydraulics)
    return mpc_a2b.simulate(
        parsed, parameters, hydraulics, solver, model, scale, a, b
    ), parameters


def demanded_rate(arguments: argparse.Namespace) -> np.ndarray:
    """Peak joint velocity the reference asks for, per planned axis."""
    a = np.asarray(UNREACHABLE_START, dtype=float)
    b = np.asarray(UNREACHABLE_GOAL, dtype=float)
    # The quintic's peak rate is 1.875 * displacement / duration.
    return 1.875 * np.abs(b - a) / arguments.move_duration


def tracking(data: mpc_a2b.RunData, parameters: dict) -> tuple[np.ndarray, float]:
    """Return the tracking error per sample and the cost the controller paid."""
    error = (
        data.state[:, cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
        - data.q_ref
    )
    weight = np.asarray(parameters["weights"]["q_a"][: cs.K_PLANNED_DOF], dtype=float)
    cost = float(np.sum(0.5 * weight * error * error))
    return error, cost


def report(name: str, data: mpc_a2b.RunData, parameters: dict) -> dict:
    error, cost = tracking(data, parameters)
    rate = data.state[:, cs.X_PROGRESS_RATE]
    norm = np.linalg.norm(error, axis=1)
    row = {
        "name": name,
        "rate_min": float(rate.min()),
        "rate_mean": float(rate.mean()),
        "peak_error": float(norm.max()),
        "final_error": float(norm[-1]),
        "tracking_cost": cost,
        "virtual_time": float(data.virtual_time[-1]),
        "wall_time": float(data.time[-1]),
        "not_converged": int(np.count_nonzero(data.status)),
        "fallback": int(np.count_nonzero(data.fallback)),
        "samples": int(data.status.size),
    }
    return row


def main() -> int:
    arguments = parse()
    demanded = demanded_rate(arguments)
    print("The reference this run asks for, against the control-safe limits:")
    time_scaling, parameters = run(arguments, arguments.progress_scale)
    limit = np.asarray(
        parameters["limits"]["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float
    )
    for axis, name in enumerate(mpc_a2b.AXIS_NAMES):
        print(
            f"  {name:<10} peak |dq_ref| {demanded[axis]:6.3f} against "
            f"dq_a_max {limit[axis]:6.3f}  "
            f"({demanded[axis] / limit[axis]:5.2f}x)"
        )
    if np.max(demanded / limit) <= 1.0:
        print(
            "\nerror: this reference is followable, so it tests nothing. "
            "Shorten --move-duration.",
            file=sys.stderr,
        )
        return 2

    pinned, _ = run(arguments, PINNED_PROGRESS_SCALE)

    rows = [
        report("time-scaling (shipped)", time_scaling, parameters),
        report("time-indexed (rate pinned at one)", pinned, parameters),
    ]

    print(
        f"\n{'run':<34}{'v_s min':>9}{'v_s mean':>10}{'peak |e|':>10}"
        f"{'final |e|':>11}{'tracking cost':>15}{'!conv':>7}{'fallback':>10}"
    )
    for row in rows:
        print(
            f"{row['name']:<34}{row['rate_min']:9.4f}{row['rate_mean']:10.4f}"
            f"{row['peak_error']:10.4f}{row['final_error']:11.4f}"
            f"{row['tracking_cost']:15.4f}{row['not_converged']:7d}"
            f"{row['fallback']:10d}"
        )
    for row in rows:
        print(
            f"  {row['name']}: spent {row['virtual_time']:.2f} s of plan in "
            f"{row['wall_time']:.2f} s of wall clock"
        )

    scaling, indexed = rows
    verdicts = [
        (
            "the progress rate fell below one",
            scaling["rate_min"] < 1.0 - arguments.tolerance,
            f"min {scaling['rate_min']:.4f}",
        ),
        (
            "the pinned run really is time-indexed",
            abs(indexed["rate_min"] - 1.0) < 1.0e-3,
            f"min {indexed['rate_min']:.6f}",
        ),
        (
            "spending time cost less tracking than not spending it",
            scaling["tracking_cost"] < indexed["tracking_cost"],
            f"{scaling['tracking_cost']:.4f} against {indexed['tracking_cost']:.4f}",
        ),
        (
            "the peak tracking error is smaller with the progress state",
            scaling["peak_error"] < indexed["peak_error"],
            f"{scaling['peak_error']:.4f} against {indexed['peak_error']:.4f}",
        ),
        (
            "the time-scaling run neither fell back nor failed to converge",
            scaling["not_converged"] == 0 and scaling["fallback"] == 0,
            f"{scaling['not_converged']} non-converged, {scaling['fallback']} "
            "fallback of "
            f"{scaling['samples']} -- and a fully-fallback run produces no NaN "
            "and completes the move, so 'no NaN' is not the check",
        ),
    ]
    # The pinned run's own convergence is **reported, not required**. It is the
    # controller this design replaces, run against a reference that defeats it,
    # and it failing to converge is a finding rather than a defect in this
    # change: the QP it is handed has no good answer, so it hits HPIPM's
    # iteration limit and the node falls back on the previous plan.
    print(
        f"\n  the time-indexed baseline: {indexed['not_converged']} non-converged "
        f"and {indexed['fallback']} fallback of {indexed['samples']} cycles"
    )
    print()
    failed = 0
    for claim, held, evidence in verdicts:
        print(f"  [{'ok  ' if held else 'FAIL'}] {claim} -- {evidence}")
        failed += 0 if held else 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
