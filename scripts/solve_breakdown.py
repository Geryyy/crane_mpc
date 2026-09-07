#!/usr/bin/env python3
"""
Where the solve time actually goes: linearisation, integration, or the QP.

Issue 115 measured that median solve time is strongly sub-linear in the horizon --
cutting 50 knots to 30 buys 1.27x, not 1.67x -- which says a large fixed cost
dominates and that the horizon is not the lever the plan assumed. This asks acados
which part that is, by reading the timing breakdown it already keeps beside
`time_tot`.

It reuses `mpc_a2b.py`'s own solver construction, so the problem being timed is the
deployed one. It does not simulate: it re-solves from one pinned state, which is
what isolates per-solve cost from trajectory-dependent behaviour.

**INCOMPLETE, and the missing piece is named so nobody rediscovers it.** Pinning
`x_0` is not enough to pose a solvable problem: the per-stage `yref` and the box
bounds are left at their defaults, so the tracking cost asks a mid-move machine to
be at the origin and the sway box is centred on zero rather than on `q_eq`. HPIPM
gets a long way in and then answers status 3 at QP iteration 38. Two seeding traps
are already fixed here and are worth keeping whatever comes next: a synthesised
zero state is outside the sway box, and a zero force state starts the problem with
the hydraulics switched off.

Finishing this means either copying `mpc_a2b.simulate`'s per-stage setup, which is
most of that function, or -- better -- adding `time_lin`, `time_sim`, `time_qp` and
`time_qp_xcond` as columns to the harness CSV and reading them off a real run.
The second is a few lines and gives the breakdown on a feasible problem. Do that.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import casadi as ca
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

import export_ocp  # noqa: E402
import mpc_a2b  # noqa: E402

cs = export_ocp.cs

# acados keeps these beside `time_tot`. `time_lin` is building the QP (model and
# constraint Jacobians), `time_sim` the integrator inside it, `time_qp` the QP
# itself and `time_qp_xcond` the partial condensing in front of it. Adding states
# inflates linearisation roughly quadratically and the QP roughly cubically, so
# which one dominates decides how expensive this feature's states are.
# Note the field names differ by acados version; these are what this build offers.
FIELDS = ("time_tot", "time_lin", "time_sim", "time_qp", "time_qp_xcond", "time_reg")


def harness_defaults() -> argparse.Namespace:
    """Build the harness's own argument defaults without going through the command line."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        return mpc_a2b.parse_arguments()
    finally:
        sys.argv = saved


def seed_state(path: Path, model, payload: np.ndarray) -> np.ndarray:
    """
    Build a pinned state from a harness CSV, in the OCP's own layout.

    The CSV predates C3 and carries only the rigid rows, so the actuator rows are
    filled the way the harness fills them: the lagged command at the measured
    velocity, which is the PT1's steady state, and the force state at `h_eff`, the
    force that holds the machine still here. **Zero is not a neutral seed** -- it
    starts the problem with the hydraulics switched off, and HPIPM refuses it with
    status 3 rather than solving something meaningless.
    """
    import csv

    rows = list(csv.DictReader(path.open()))
    row = rows[len(rows) // 2]
    axes = ["slew", "boom", "arm", "telescope", "rotator"]
    passive = ["sway_1", "sway_2"]
    state = np.zeros(cs.NX)
    for index, name in enumerate(axes):
        state[cs.X_PLANNED_POSITION + index] = float(row[f"q_{name}"])
        state[cs.X_PLANNED_VELOCITY + index] = float(row[f"dq_{name}"])
    for index, name in enumerate(passive):
        state[cs.X_PASSIVE_POSITION + index] = float(row[f"q_{name}"])
        state[cs.X_PASSIVE_VELOCITY + index] = float(row[f"dq_{name}"])
    for slot, axis in enumerate(cs.K_LAG_AXES):
        state[cs.X_COMMAND_LAG + slot] = state[cs.X_PLANNED_VELOCITY + axis]
    # The plan is being spent at its nominal rate at the pinned state; a zero
    # rate would be a machine that has stopped following, which is a different
    # problem to time.
    state[cs.X_PROGRESS_RATE] = export_ocp.K_PROGRESS_RATE_REFERENCE
    static = ca.Function(
        "breakdown_static", [model.x, model.p], [model.actuated_force_static]
    )
    state[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF] = np.asarray(
        static(state, payload)
    ).reshape(-1)
    return state


def qp_iterations(solver) -> int:
    """`qp_iter` comes back as a scalar or per-iteration array depending on version."""
    value = solver.get_stats("qp_iter")
    try:
        return int(np.sum(value))
    except TypeError:
        return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knots", type=int, nargs="+", default=[50, 40, 30])
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument(
        "--seed",
        type=Path,
        default=PACKAGE / "build" / "sweep_grid" / "k50_dt40" / "run.csv",
        help="a run CSV to lift the pinned state out of",
    )
    cli = parser.parse_args()

    header = f"{'knots':>6} {'dt':>6} " + " ".join(f"{name:>13}" for name in FIELDS)
    print(header + f" {'qp_iter':>8}")

    for knots in cli.knots:
        namespace = harness_defaults()
        namespace.horizon_knots = knots
        if cli.dt is not None:
            namespace.dt = cli.dt

        parameters, hydraulics = mpc_a2b.load_settings(namespace)
        solver, model, _scale = mpc_a2b.create_solver(namespace, parameters, hydraulics)

        # One pinned state, the same every repeat, taken from a real run rather than
        # synthesised: a zero state is outside the sway box and HPIPM refuses it with
        # status 3, which measures nothing.
        intervals = export_ocp.shooting_intervals(parameters)
        payload = mpc_a2b.parameter_vector(namespace)
        state = seed_state(cli.seed, model, payload)
        for stage in range(intervals + 1):
            # `p` also carries the stage's local reference model now. This tool
            # times one pinned state, so the reference is the state's own pose
            # held still: a zero first and second derivative is a plan that is
            # not moving, which is what a single-state timing probe means.
            solver.set(
                stage,
                "p",
                mpc_a2b.stage_parameters(
                    payload,
                    stage * float(namespace.dt or parameters["Ts"]),
                    state[
                        cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF
                    ],
                    np.zeros(cs.K_PLANNED_DOF),
                    np.zeros(cs.K_PLANNED_DOF),
                ),
            )
        solver.set(0, "lbx", state)
        solver.set(0, "ubx", state)

        samples = {name: [] for name in FIELDS}
        iterations = []
        for _ in range(cli.repeats):
            solver.solve()
            for name in FIELDS:
                samples[name].append(float(solver.get_stats(name)))
            iterations.append(qp_iterations(solver))

        medians = {name: 1e3 * statistics.median(samples[name]) for name in FIELDS}
        print(
            f"{knots:>6} {float(parameters['Ts']):>6.3f} "
            + " ".join(f"{medians[name]:>13.3f}" for name in FIELDS)
            + f" {statistics.median(iterations):>8.0f}"
        )

    print()
    print("Medians in ms. time_lin + time_qp (+ xcond) should account for most of")
    print("time_tot; whatever is left is acados overhead that no change to the")
    print("problem size will move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
