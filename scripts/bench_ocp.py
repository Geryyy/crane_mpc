#!/usr/bin/env python3
"""
Run the MPC over a corpus of sampled moves and log what the solver did on each.

`crane_planning/scripts/bench_ocp.py`'s counterpart, on the other OCP. The unit
there is a plan; here it is a **cycle**, because that is what a receding-horizon
controller has to finish inside `solve_budget` a few thousand times in a row.
So the cost columns pool every cycle of every move and report median/p90/max,
and the p90 is the one to read: a median inside budget with a p90 outside it is
a controller that drops cadence under exactly the poses that are hard.

This sweeps nothing. Change the backend, re-run, compare the logs;
`scripts/sweep_ocp.py` is what drives it once per variant.

    ./scripts/bench_ocp.py --moves 25
    ./scripts/bench_ocp.py --moves 10 --seed 3 --out build/before.json

Plant is the model itself (`mpc_a2b`'s ERK4 rollout), deliberately: a bench that
also changed the plant would price the integrator's disagreement as solver cost.
`scripts/tune_mpc.py` is where a plant that is not the model belongs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))
if (PACKAGE / "crane_mpc").is_dir():
    sys.path.insert(0, str(PACKAGE))
    sys.path.insert(0, str(PACKAGE.parent / "crane_model"))

import mpc_a2b  # noqa: E402
from crane_mpc import problem  # noqa: E402
from crane_mpc import solver as ocp_runtime  # noqa: E402

cs = mpc_a2b.cs

#: Sampled onto a bound, a move is about the bound and not about the solver.
INSET = 0.10

#: The rotator is +-4pi, so a uniform draw over it is mostly winding, not crane.
ROTATOR = math.pi

#: Peak rate a move asks for, as a fraction of `dq_a_max` on its binding axis.
#: At 1.0 the reference sits exactly on constraint 2 and every cycle of the
#: corpus is measured with that bound active, which is `cannot_follow.py`'s
#: regime and not a plan `crane_planning` would ever emit.
SPEED_FRACTION = 0.6

#: A minimum-jerk quintic peaks at 15/8 of its average rate, so a duration read
#: off the average would ask for 1.875x what it looks like it asks for. Stated
#: once here and in `cannot_follow.py:138`, on the same reference.
QUINTIC_PEAK = 1.875


def arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--moves", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed-fraction",
        type=float,
        default=SPEED_FRACTION,
        help="move duration is sized so the reference's peak rate reaches this "
        "fraction of the binding axis' control-safe speed (default: %(default)s)",
    )
    parser.add_argument("--min-duration", type=float, default=2.0)
    # two sway periods (~2 s each, `config/crane_mpc.yaml`): one would score a
    # variant that leaves the tool swinging over a single swing
    parser.add_argument("--settle-duration", type=float, default=4.0)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument(
        "--enforce-budget",
        action="store_true",
        help="treat a cycle over `solve_budget` as a failed one, as the node "
        "does; off by default so the timing columns measure the whole corpus",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=PACKAGE / "build" / "bench_ocp.json"
    )
    return parser.parse_args(argv)


def corpus(parameters: dict, count: int, seed: int) -> list[tuple]:
    """`count` (A, B) pairs drawn uniformly from the control-safe box."""
    limits = parameters["limits"]
    planned = slice(0, cs.K_PLANNED_DOF)
    lower = np.asarray(limits["q_a_lower"][planned], dtype=float)
    upper = np.asarray(limits["q_a_upper"][planned], dtype=float)
    margin = np.asarray(limits["q_a_margin"][planned], dtype=float)
    lower, upper = lower + margin, upper - margin
    lower = np.maximum(lower, -ROTATOR)
    upper = np.minimum(upper, ROTATOR)
    span = upper - lower
    lower, upper = lower + INSET * span, upper - INSET * span

    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(lower, upper), rng.uniform(lower, upper)) for _ in range(count)
    ]


def duration_of(parameters: dict, a: np.ndarray, b: np.ndarray, options) -> float:
    """
    How long to give this move: its binding axis at a fraction of its bound.

    On the reference's **peak** rate, not its average. A fixed duration would
    make every long move a speed-limited one and every short move a wait, so
    the corpus would measure the sampler instead of the solver; sizing on the
    average instead would put every sample 1.875x over the bound, which
    measures the fallback path.
    """
    dq_max = np.asarray(
        parameters["limits"]["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float
    )
    needed = QUINTIC_PEAK * np.max(np.abs(b - a) / (options.speed_fraction * dq_max))
    return float(max(options.min_duration, needed))


def statistics(values: np.ndarray) -> dict:
    """Median/p90/max over the finite entries, NaN when there are none."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"median": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def measure(data, parameters: dict) -> dict:
    """One closed-loop run, as the numbers a solver change moves."""
    budget = float(parameters["solve_budget"])
    period = float(parameters["Ts"])
    sway = (
        data.state[:, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
        - data.q_eq
    )
    error = data.state[-1, : cs.K_PLANNED_DOF] - data.q_ref[-1]
    rate = data.state[:, cs.X_PROGRESS_RATE]
    return {
        # --- per cycle, kept as arrays: the corpus pools them ------------------
        "solve_time_s": data.solve_time.tolist(),
        "qp_iterations": data.timing["qp_iter"].tolist(),
        "linearisation_s": data.timing["time_lin"].tolist(),
        "qp_time_s": data.timing["time_qp"].tolist(),
        # --- per cycle, counted -----------------------------------------------
        "cycles": int(data.solve_time.size),
        "nonzero_status": int(np.count_nonzero(data.status)),
        "fallback_cycles": int(np.count_nonzero(data.fallback)),
        "over_budget_cycles": int(np.count_nonzero(data.solve_time > budget)),
        # `solve_budget` is 0.08 against a 0.06 cycle, so a variant can be
        # inside it and still miss every control deadline
        "over_period_cycles": int(np.count_nonzero(data.solve_time > period)),
        # --- per run: what the cost bought ------------------------------------
        "terminal_error": float(np.linalg.norm(error)),
        "peak_sway_rad": float(np.max(np.abs(sway))),
        "peak_cylinder_force": float(np.max(data.hydraulic_use[:, : cs.K_PLANNED_DOF])),
        "peak_pump_use": float(np.max(data.hydraulic_use[:, cs.K_PLANNED_DOF])),
        "min_progress_rate": float(rate.min()),
        "plan_time_spent": float(data.virtual_time[-1] / data.time[-1]),
    }


#: Solver cost, pooled over every cycle in the corpus. What a sweep is for.
COST = ("solve_time_s", "qp_iterations", "linearisation_s", "qp_time_s")

#: What must not get worse when the cost above gets better. Per run.
QUALITY = (
    "terminal_error",
    "peak_sway_rad",
    "peak_cylinder_force",
    "peak_pump_use",
    "min_progress_rate",
)

#: Robustness, summed over the corpus. A solver that is fast and refuses is not
#: a faster solver.
COUNTED = (
    "cycles",
    "nonzero_status",
    "fallback_cycles",
    "over_budget_cycles",
    "over_period_cycles",
)


def summary(runs: list[dict], failures: list[dict]) -> dict:
    solved = [run for run in runs if "failed" not in run]
    out = {
        "moves_sampled": len(runs) + len(failures),
        "moves_run": len(solved),
        "moves_failed": len(failures),
    }
    if not solved:
        return out
    for name in COUNTED:
        out[name] = int(sum(run[name] for run in solved))
    for name in COST:
        pooled = np.concatenate([np.asarray(run[name], dtype=float) for run in solved])
        out[name] = statistics(pooled)
    for name in QUALITY:
        out[name] = statistics(np.array([run[name] for run in solved]))
    return out


def baked(parameters: dict, hydraulics: dict, description: str) -> dict:
    """
    Name this run's solver, so it cannot be attributed to another one.

    The signature is the same digest the cache directory is named after, so two
    logs carrying the same one were produced by the same compiled `.so`.
    """
    return {
        "export_key": ocp_runtime.export_key(parameters, hydraulics, description),
        "Ts": float(parameters["Ts"]),
        "horizon_length": int(parameters["horizon_length"]),
        "levenberg_marquardt": float(parameters["levenberg_marquardt"]),
        "tuning": problem.solver_tuning(),
    }


def main(argv=None) -> int:
    options = arguments(argv)
    settings = mpc_a2b.parse_arguments([])
    settings.rebuild = options.rebuild
    settings.enforce_budget = options.enforce_budget
    settings.settle_duration = options.settle_duration
    settings.payload_mass = options.payload_mass
    settings.no_plot = settings.no_csv = True

    parameters, hydraulics = mpc_a2b.load_settings(settings)
    description = (
        mpc_a2b.export_ocp.DEFAULT_DESCRIPTIONS / mpc_a2b.export_ocp.DESCRIPTION
    ).read_text()
    solver, model, scale = mpc_a2b.create_solver(settings, parameters, hydraulics)

    print(f"{'move':>5} {'cycles':>7} {'qp':>7} {'solve ms':>10} {'p90 ms':>9} ")
    runs, failures = [], []
    for index, (a, b) in enumerate(corpus(parameters, options.moves, options.seed)):
        settings.a, settings.b = a, b
        settings.move_duration = duration_of(parameters, a, b, options)
        started = time.perf_counter()
        try:
            checked_a, checked_b = mpc_a2b.validate_movement(settings, parameters)
            # simulate narrates per second of plan; the table below is the report
            with contextlib.redirect_stdout(io.StringIO()):
                data = mpc_a2b.simulate(
                    settings,
                    parameters,
                    hydraulics,
                    solver,
                    model,
                    scale,
                    checked_a,
                    checked_b,
                )
        except (ValueError, RuntimeError) as error:
            row = {"move": index, "failed": str(error)}
            failures.append(row)
            print(f"{index:>5} {'failed':>7}  {str(error)[:60]}", flush=True)
            continue

        row = {
            "move": index,
            "a": np.asarray(a).tolist(),
            "b": np.asarray(b).tolist(),
            "move_duration_s": settings.move_duration,
            "wall_s": time.perf_counter() - started,
        } | measure(data, parameters)
        runs.append(row)
        solve = statistics(np.asarray(row["solve_time_s"]))
        print(
            f"{index:>5} {row['cycles']:>7} "
            f"{np.nanmedian(row['qp_iterations']):>7.0f} "
            f"{1e3 * solve['median']:>10.2f} {1e3 * solve['p90']:>9.2f}",
            flush=True,
        )

    report = summary(runs, failures)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(
        json.dumps(
            {
                "seed": options.seed,
                "baked": baked(parameters, hydraulics, description),
                # this corpus measured 26 ms median in the run that compiled the
                # solver and 8.6 ms minutes later on the same box -- read a
                # regression off iterations, and off wall clock only at equal load
                "load_average": os.getloadavg()[0],
                "summary": report,
                "runs": runs + failures,
            },
            indent=1,
            default=float,
        )
    )

    print(f"\nmoves run {report['moves_run']} of {report['moves_sampled']}")
    for name in COUNTED:
        if name in report:
            print(f"  {name:<20} {report[name]}")
    print("\ncost, pooled over every cycle")
    for name in COST:
        if name in report:
            block = report[name]
            scale_ms = 1e3 if name.endswith("_s") else 1.0
            print(
                f"  {name:<20} median {scale_ms * block['median']:>9.3f}  "
                f"p90 {scale_ms * block['p90']:>9.3f}  "
                f"max {scale_ms * block['max']:>9.3f}"
            )
    print("\nwhat it bought")
    for name in QUALITY:
        if name in report:
            block = report[name]
            print(
                f"  {name:<20} median {block['median']:>9.4f}  "
                f"p90 {block['p90']:>9.4f}  max {block['max']:>9.4f}"
            )
    print("-- these must not get worse when the cost above gets better")
    print(f"wrote {options.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
