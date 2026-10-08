#!/usr/bin/env python3
"""
Score C3's force-state estimate against Gazebo's own joint effort.

    ./scripts/trials/force_truth.py <bag>/bag_0.mcap [--json out.json] [--set k=v]

`tau_a` is the one state row no sensor in this stack carries, which is why issue
227 exists: `x_0` carried it by integrating `tau_dot = k (u_f - dq)` over the
**measured** `dq`, and `k` runs to 3.5e6. Gazebo *does* carry it -- the hydraulic
plugin writes each actuated joint's applied force to the `effort` state
interface, so a live `/joint_states` bag is a ground truth for the force rows
and for nothing else in this workspace.

What this replays is the estimator alone, over recorded inputs:

    q_a, dq_a, q_u, dq_u   /joint_states
    the command            /trajectory_controllers/controller_state, `output.velocities`
                           averaged over the preceding `Ts`, which is what
                           `Timing.jtc_output` hands the node

and nothing else -- no solve, no horizon. Both estimators see the same inputs on
the same cycle grid, so the two columns differ by the estimator and by nothing
else. That is deliberately *not* a closed loop: a closed one would move the
machine and there would be no shared truth left to score against.

**Conventions, checked before the numbers are believed.** Gazebo's effort is
joint-side, same sign and same units as `tau_a` (N m, N on the telescope), and
on the telescope it is reported *per mimic half*, so the truth is the sum of
`q4_big_telescope` and `q5_small_telescope`. At rest on the i226 bags that sum
reads 1969 N against `h_eff`'s 2005 and the arm reads 20025 against 20051 --
0.1-2 % apart, which is the plant's own mass against the description's and is
the floor this scores above. The slewing and rotator rows hold ~0 N m at rest in
both, so a *relative* error on those two is meaningless and only the absolute
one is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
from crane_model.conventions import ACTUATED_INDICES, canonical_joints  # noqa: E402
from crane_mpc import force_filter as ff  # noqa: E402

cs = harness.cs
PLANNED = cs.K_PLANNED_DOF
PASSIVE = cs.K_PASSIVE_DOF

JOINT_STATE_TOPIC = "/joint_states"
#: The JTC the A2B capture activates; `--jtc` for any other profile.
CONTROLLER_STATE_TOPIC = "/trajectory_controllers/controller_state"

#: The passive pair, by name: `/joint_states` carries different name sets in
#: different orders in the two stacks, so nothing here reads by index.
PASSIVE_JOINTS = ("theta6_tip_joint", "theta7_tilt_joint")

#: One cylinder, two mimicked bodies, and Gazebo reports an effort for each.
TELESCOPE_HALVES = ("q4_big_telescope", "q5_small_telescope")

#: Rows whose hold force is ~0 N m, so a ratio against it says nothing.
NEAR_ZERO_ROWS = (0, 4)

#: Position noise per unit of velocity noise, measured on
#: `control_recordings/2026-08-19_12-06-08-record`: the actuated rows read
#: 1.6e-5 to 2.0e-5 rad of white position noise against 1.2e-3 to 2.3e-3 rad/s
#: of velocity noise, so the ratio is ~1e-2 s and the same on every axis.
MACHINE_Q_RATIO = 1.0e-2


def actuated_joints() -> list:
    joints = list(canonical_joints())
    return [joints[index] for index in ACTUATED_INDICES]


def read_bag(path: Path, jtc_topic: str) -> dict:
    """Pull the measurement, the truth and the command off one mcap."""
    from mcap_ros2.reader import read_ros2_messages

    joints = actuated_joints()
    state = {name: [] for name in ("t", "q_a", "dq_a", "q_u", "dq_u", "tau_truth")}
    command = {"t": [], "u": []}
    for message in read_ros2_messages(str(path), topics=[JOINT_STATE_TOPIC, jtc_topic]):
        body = message.ros_msg
        stamp = body.header.stamp.sec + 1.0e-9 * body.header.stamp.nanosec
        if message.channel.topic == JOINT_STATE_TOPIC:
            row = {name: index for index, name in enumerate(body.name)}
            if not all(name in row for name in joints + list(PASSIVE_JOINTS)):
                continue
            truth = np.array(
                [body.effort[row[name]] for name in joints[:PLANNED]], dtype=float
            )
            if all(half in row for half in TELESCOPE_HALVES):
                truth[3] = sum(body.effort[row[half]] for half in TELESCOPE_HALVES)
            state["t"].append(stamp)
            state["q_a"].append([body.position[row[name]] for name in joints])
            state["dq_a"].append([body.velocity[row[name]] for name in joints])
            state["q_u"].append([body.position[row[n]] for n in PASSIVE_JOINTS])
            state["dq_u"].append([body.velocity[row[n]] for n in PASSIVE_JOINTS])
            state["tau_truth"].append(truth)
            continue
        row = {name: index for index, name in enumerate(body.joint_names)}
        if not all(name in row for name in joints[:PLANNED]):
            continue
        # `output.velocities` is the controller's own output, PI included, which
        # is what the node's dead-time roll replays; the fork leaves it empty on
        # no profile here, so there is no `reference` fallback to write.
        command["t"].append(stamp)
        command["u"].append([body.output.velocities[row[n]] for n in joints[:PLANNED]])
    read = {key: np.asarray(value, dtype=float) for key, value in state.items()}
    read["command_t"] = np.asarray(command["t"], dtype=float)
    read["command_u"] = np.asarray(command["u"], dtype=float)
    return read


def mean_command(bag: dict, start: float, end: float) -> np.ndarray:
    """Give what the JTC averaged over `[start, end)`, zero where it said nothing."""
    inside = (bag["command_t"] >= start) & (bag["command_t"] < end)
    if not inside.any():
        return np.zeros(PLANNED)
    window = bag["command_u"][inside]
    # A non-finite row is the tool's or a passive one leaking in; drop it rather
    # than letting one NaN sample zero the whole interval's mean.
    return np.nanmean(np.where(np.isfinite(window), window, np.nan), axis=0)


def cycle_grid(bag: dict, Ts: float, skip: float) -> np.ndarray:
    """
    Give the joint-state samples a cycle at `Ts` would have read, as indices.

    The node runs off a timer and reads whatever sample is newest, so the grid
    is the bag's own samples nearest each multiple of `Ts` -- not a resample:
    interpolating a velocity is exactly the smoothing this is measuring.
    """
    stamps = bag["t"]
    first = stamps[0] + float(skip)
    instants = np.arange(first, stamps[-1], Ts)
    return np.unique(np.searchsorted(stamps, instants).clip(0, stamps.size - 1))


def state_at(
    bag: dict,
    index: int,
    nx: int,
    carried: np.ndarray,
    measured_q: np.ndarray,
    measured_dq: np.ndarray,
) -> np.ndarray:
    """Write the measured rows into a copy of `carried`, as `read_state` does."""
    x = carried.copy()
    x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED] = measured_q[:PLANNED]
    x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED] = measured_dq[:PLANNED]
    x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE] = bag["q_u"][index]
    x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE] = bag["dq_u"][index]
    x[cs.X_PROGRESS] = 0.0
    x[cs.X_PROGRESS_RATE] = 0.0
    return x


def sensor_noise(bag: dict, grid: np.ndarray, noise, seed: int) -> dict:
    """
    One white sequence to add to the measured rows, shared by both estimators.

    Gazebo's `/joint_states` is all but noiseless (the one-step velocity
    prediction residual over the i226 bags is 1e-4 rad/s), so the quantity issue
    227 is about does not exist in the capture and has to be put there. Drawn
    once and handed to every variant, so the two columns differ by the estimator
    and not by the draw.
    """
    if noise is None:
        return {"q_a": None, "dq_a": None}
    rng = np.random.default_rng(seed)
    spread = np.asarray(noise, dtype=float)
    return {
        # Position noise rides along at the machine's own measured ratio to the
        # velocity's, so a sweep of one sweeps a sensor and not half of one.
        "q_a": rng.normal(0.0, 1.0, (grid.size, PLANNED)) * spread * MACHINE_Q_RATIO,
        "dq_a": rng.normal(0.0, 1.0, (grid.size, PLANNED)) * spread,
    }


def replay(ocp, bag: dict, grid: np.ndarray, filtered, noise=None) -> dict:
    """
    One estimator over the whole bag. `filtered` is a `ForceFilter` or `None`.

    `None` is the pre-227 carry exactly: measured `q_a`/`dq_a`, the force row
    whatever last cycle's roll left, which is what `Cycle.read_state` did.

    `noise` is `sensor_noise`'s draw, added to the two measured rows.
    """
    rows = np.array([cs.X_PLANNED_VELOCITY, cs.X_ACTUATED_FORCE])
    index = (rows[:, None] + np.arange(PLANNED)[None, :]).T
    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + PLANNED)
    Ts = ocp.Ts
    carried = np.zeros(ocp.nx)
    estimate, velocity_residual, seeded = [], [], False
    for cycle, sample in enumerate(grid):
        ocp.pin_tool(float(bag["q_a"][sample][cs.K_TOOL_AXIS]))
        measured_q = bag["q_a"][sample].copy()
        measured_dq = bag["dq_a"][sample].copy()
        if noise is not None and noise["dq_a"] is not None:
            measured_q[:PLANNED] += noise["q_a"][cycle]
            measured_dq[:PLANNED] += noise["dq_a"][cycle]
        x = state_at(bag, sample, ocp.nx, carried, measured_q, measured_dq)
        if not seeded:
            x[force] = ocp.static_hold_force(x)
            carried = x.copy()
            seeded = True
            if filtered is not None:
                filtered.forget()
        elif filtered is not None:
            if not filtered.armed:
                filtered.derive(ocp.axis_transition(carried, np.zeros(cs.NU_PROGRESS)))
            x[index] = filtered.correct(carried[index], measured_dq)
        # Scored before the roll: the estimate standing at *this* measurement
        # instant is what stage 0 of this cycle's solve would have been pinned at.
        estimate.append(x[force].copy())
        velocity_residual.append(
            measured_dq[:PLANNED]
            - carried[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED]
        )
        instant = bag["t"][sample]
        held = mean_command(bag, instant - Ts, instant)
        if ocp.command_state:
            # C4: `(c at the entry's first instant, dc)`, as `Cycle.propagate` rolls it.
            entry = (held, np.zeros(cs.NU_PROGRESS))
        else:
            entry = np.zeros(cs.NU_PROGRESS)
            entry[:PLANNED] = held
        try:
            carried = ocp.propagate_applied(x, [entry])
        except Exception as refusal:
            print(f"the roll refused at t = {instant:.3f}: {refusal}", file=sys.stderr)
            carried = x.copy()
    return {
        "tau_a": np.asarray(estimate),
        "dq_a_residual": np.asarray(velocity_residual),
    }


def score(estimate: np.ndarray, truth: np.ndarray) -> dict:
    error = estimate - truth
    return {
        "rms": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "peak": np.max(np.abs(error), axis=0).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="an mcap carrying /joint_states")
    parser.add_argument("--jtc", default=CONTROLLER_STATE_TOPIC)
    parser.add_argument(
        "--skip",
        type=float,
        default=0.0,
        help="seconds of the bag to leave before the first cycle",
    )
    parser.add_argument(
        "--dq-noise",
        type=float,
        nargs="*",
        default=None,
        metavar="SIGMA",
        help="white velocity noise added to the measurement, rad/s: one number "
        f"for every axis or {PLANNED}. No flag leaves the capture as it is, which "
        "measures the mismatch floor and not the noise.",
    )
    parser.add_argument("--noise-seed", type=int, default=227)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    options = parser.parse_args(argv)

    parameters, hydraulics = harness.settings(options.set)
    ocp = harness.create_solver(parameters, hydraulics)
    bag = read_bag(options.bag, options.jtc)
    if bag["t"].size < 2:
        print(f"{options.bag} carries no usable {JOINT_STATE_TOPIC}", file=sys.stderr)
        return 2
    grid = cycle_grid(bag, ocp.Ts, options.skip)
    truth = bag["tau_truth"][grid]

    covariances = ff.Noise.from_parameters(parameters["force_filter"])
    spread = None
    if options.dq_noise is not None:
        spread = np.resize(np.asarray(options.dq_noise, dtype=float), PLANNED)
    draw = sensor_noise(bag, grid, spread, options.noise_seed)

    report = {
        "bag": str(options.bag),
        "cycles": int(grid.size),
        "seconds": float(bag["t"][grid[-1]] - bag["t"][grid[0]]),
        "dq_noise": None if spread is None else spread.tolist(),
        "truth_rms": np.sqrt(np.mean(truth**2, axis=0)).tolist(),
        "truth_span": (np.max(truth, axis=0) - np.min(truth, axis=0)).tolist(),
        "near_zero_rows": list(NEAR_ZERO_ROWS),
    }
    for name, filtered in (
        ("open_loop", None),
        ("filtered", ff.ForceFilter(covariances)),
    ):
        # The same estimator twice: once on the capture, once on the capture
        # plus the draw. The difference between the two **is** the force error
        # the measurement noise produced, which is the one quantity this issue
        # is about and the only one free of the plant mismatch below.
        clean = replay(ocp, bag, grid, filtered)
        if filtered is not None:
            filtered.forget()
        noisy = clean if spread is None else replay(ocp, bag, grid, filtered, draw)
        report[name] = score(noisy["tau_a"], truth)
        report[name]["mismatch_rms"] = score(clean["tau_a"], truth)["rms"]
        report[name]["noise_rms"] = np.sqrt(
            np.mean((noisy["tau_a"] - clean["tau_a"]) ** 2, axis=0)
        ).tolist()
        report[name]["noise_peak"] = np.max(
            np.abs(noisy["tau_a"] - clean["tau_a"]), axis=0
        ).tolist()
        report[name]["dq_a_residual_rms"] = np.sqrt(
            np.mean(noisy["dq_a_residual"] ** 2, axis=0)
        ).tolist()

    for column in ("rms", "noise_rms", "mismatch_rms"):
        report[f"{column}_improvement"] = (
            np.asarray(report["open_loop"][column])
            / np.maximum(np.asarray(report["filtered"][column]), 1e-12)
        ).tolist()

    joints = actuated_joints()[:PLANNED]
    width = max(len(name) for name in joints)
    print(f"{report['cycles']} cycles, {report['seconds']:.1f} s of {options.bag}")
    print(f"dq noise added: {report['dq_noise']}")
    header = (
        f"{'axis'.ljust(width)}  {'truth rms':>11} {'mismatch':>11} "
        f"{'noise open':>11} {'noise filt':>11} {'x':>7} {'total open':>11} "
        f"{'total filt':>11}"
    )
    print(header)
    for axis, name in enumerate(joints):
        print(
            f"{name.ljust(width)}  {report['truth_rms'][axis]:11.4g} "
            f"{report['open_loop']['mismatch_rms'][axis]:11.4g} "
            f"{report['open_loop']['noise_rms'][axis]:11.4g} "
            f"{report['filtered']['noise_rms'][axis]:11.4g} "
            f"{report['noise_rms_improvement'][axis]:7.2f} "
            f"{report['open_loop']['rms'][axis]:11.4g} "
            f"{report['filtered']['rms'][axis]:11.4g}"
        )
    if options.json is not None:
        options.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
