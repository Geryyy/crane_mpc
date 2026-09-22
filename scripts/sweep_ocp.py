#!/usr/bin/env python3
"""
Drive `bench_ocp.py` once per acados variant and table what each one cost.

`crane_planning/scripts/sweep_ocp.py`'s counterpart. One variant per
subprocess, because the setting is compiled into the solver and a process holds
exactly one of them; the variant reaches the child as `CRANE_MPC_OCP_OPTIONS`,
which `crane_mpc.problem.solver_tuning` layers over `SOLVER_TUNING` and
`solver_signature` hashes -- so each variant compiles its own `.so` instead of
opening its predecessor's and reading as a null result.

    ./scripts/sweep_ocp.py --moves 25
    ./scripts/sweep_ocp.py --only hpipm_speed cond_N_10 --moves 40
    ./scripts/sweep_ocp.py --variants build/round2.json

`--variants` is a JSON object `{name: patch}`; a patch is either a bare options
dict or `{"options": {...}, "env": {...}}` when it also moves something that is
not compiled in. Baseline is always row 0, and every other row is read against
it -- not against the previous sweep, whose box was under a different load.

**Read a regression off `qp_iter` first and off milliseconds second**: the same
solver on the same corpus measured 26 ms median in the run that compiled it and
8.6 ms minutes later on the same box. `wall_s` is not comparable at all between
rows, since a cold variant pays for a compile -- which is why every row records
the load average its child saw and the signature of the `.so` it opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
BENCH = PACKAGE / "scripts" / "bench_ocp.py"
OUT_ROOT = PACKAGE / "build" / "sweep_ocp"

#: One knob each, so a row names its own cause. Combinations are round two:
#: write the ones the singles rewarded into a `--variants` file.
DEFAULT_VARIANTS = {
    # --- the QP ---------------------------------------------------------------
    "hpipm_speed": {"hpipm_mode": "SPEED"},
    "hpipm_speed_abs": {"hpipm_mode": "SPEED_ABS"},
    "hpipm_robust": {"hpipm_mode": "ROBUST"},
    "qp_full_condensing": {"qp_solver": "FULL_CONDENSING_HPIPM"},
    # Issue 129 measured `qp_solver_cond_N = 1` far worse (61-81 ms against
    # 10.3 ms at `N`); these ask where in between the crossover sits.
    "cond_N_10": {"qp_solver_cond_N": 10},
    "cond_N_20": {"qp_solver_cond_N": 20},
    "qp_iter_25": {"qp_solver_iter_max": 25},
    "qp_iter_100": {"qp_solver_iter_max": 100},
    # Shipped is 2. `0` is the regression row -- it is what this solver did
    # before the QP memory survived a warm cycle, and it is the row to re-run
    # if anyone reinstates an unconditional `reset_qp_solver_mem=1`.
    "qp_warm_start_0": {"qp_solver_warm_start": 0},
    "qp_warm_start_1": {"qp_solver_warm_start": 1},
    "qp_ric_alg_0": {"qp_solver_ric_alg": 0},
    # No `qp_scaling` row: acados 0.5.5 refuses both qpscaling options under
    # `SQP_RTI` outright (`acados_ocp.py:1301-1303`, NotImplementedError), so
    # the row can only ever fail while this solver is an RTI one. Reinstate it
    # in the same patch that moves `nlp_solver_type`, not before.
    # --- the integrator, where C3's stiffness is paid for ----------------------
    "irk_4_stages": {"sim_method_num_stages": 4},
    "irk_2_steps": {"sim_method_num_steps": 2},
    "irk_newton_2": {"sim_method_newton_iter": 2},
    "irk_newton_5": {"sim_method_newton_iter": 5},
    "gnsf": {"integrator_type": "GNSF"},
    # Shipped is Radau IIA with Jacobian reuse. These two are the regression
    # rows -- what the integrator cost before each half of that, measured at
    # 4.96 ms median / 16.1 p90 against the shipped 3.85 / 10.1.
    "irk_legendre": {"collocation_type": "GAUSS_LEGENDRE"},
    "irk_no_jac_reuse": {"sim_method_jac_reuse": 0},
    # --- one Newton step, or more ---------------------------------------------
    # RTI is the shipped trade; these price what convergence would cost per
    # cycle, and `Ts` is what says whether it is affordable.
    # **Their quality columns are not about the solver.** Measured on
    # `sqp_2_iter`: 1134 of 1134 cycles came back non-zero and fell back, so
    # the error and sway columns describe the shifted previous plan. acados
    # answers `ACADOS_MAXITER` when an SQP run spends `nlp_solver_max_iter`
    # without meeting `nlp_solver_tol_*` (1e-6), and `mpc_a2b.simulate` accepts
    # only status 0. Relax the tolerances in the same patch, or read these rows
    # for cost alone.
    "sqp_2_iter": {"nlp_solver_type": "SQP", "nlp_solver_max_iter": 2},
    "sqp_4_iter": {"nlp_solver_type": "SQP", "nlp_solver_max_iter": 4},
    "sqp_merit": {
        "nlp_solver_type": "SQP",
        "nlp_solver_max_iter": 4,
        "globalization": "MERIT_BACKTRACKING",
    },
    # --- regularisation -------------------------------------------------------
    "regularize_project": {"regularize_method": "PROJECT"},
    "regularize_mirror": {"regularize_method": "MIRROR"},
}

#: Table order: cost first, then what must not get worse for it.
COLUMNS = (
    ("moves", ("moves_run", None)),
    ("fallback", ("fallback_cycles", None)),
    ("status!=0", ("nonzero_status", None)),
    ("over Ts", ("over_period_cycles", None)),
    ("qp_med", ("qp_iterations", "median")),
    ("qp_p90", ("qp_iterations", "p90")),
    ("ms_med", ("solve_time_s", "median")),
    ("ms_p90", ("solve_time_s", "p90")),
    ("ms_max", ("solve_time_s", "max")),
    ("err_med", ("terminal_error", "median")),
    ("sway_p90", ("peak_sway_rad", "p90")),
    ("force_p90", ("peak_cylinder_force", "p90")),
    ("pump_p90", ("peak_pump_use", "p90")),
)

#: Seconds columns printed in milliseconds; the table is about a 60 ms cycle.
MILLISECONDS = ("solve_time_s",)


def arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--variants", type=Path, default=None)
    parser.add_argument("--moves", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-fraction", type=float, default=None)
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=0.0,
        help="forwarded to every variant; an empty gripper is not the load "
        "case the machine works in (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="run these variants only; the baseline is always included",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="seconds per variant, compile included (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=PACKAGE / "build" / "sweep_ocp.json"
    )
    return parser.parse_args(argv)


def variants_of(options) -> dict:
    table = DEFAULT_VARIANTS
    if options.variants is not None:
        table = json.loads(options.variants.read_text())
    if options.only is not None:
        unknown = set(options.only) - set(table)
        if unknown:
            raise SystemExit(f"no such variant: {sorted(unknown)}")
        table = {name: table[name] for name in options.only}
    return {"baseline": {}, **table}


def run(name: str, patch: dict, options) -> dict:
    """One variant, in its own process, its own solver and its own log."""
    acados = patch.get("options", patch)
    environment = {str(k): str(v) for k, v in patch.get("env", {}).items()}
    log = OUT_ROOT / f"{name}.json"
    log.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(BENCH),
        "--moves",
        str(options.moves),
        "--seed",
        str(options.seed),
        "--out",
        str(log),
    ]
    if options.speed_fraction is not None:
        command += ["--speed-fraction", str(options.speed_fraction)]
    if options.payload_mass:
        command += ["--payload-mass", str(options.payload_mass)]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        env=os.environ | {"CRANE_MPC_OCP_OPTIONS": json.dumps(acados)} | environment,
        capture_output=True,
        text=True,
        timeout=options.timeout,
    )
    wall = time.perf_counter() - started
    if completed.returncode != 0 or not log.is_file():
        return {"name": name, "patch": patch, "failed": completed.stderr[-1500:]}

    document = json.loads(log.read_text())
    return {
        "name": name,
        "patch": patch,
        # not comparable between rows: a cold variant pays for a `make`
        "wall_s": wall,
        "compiled": "compiling the crane_mpc" in completed.stdout,
        "load_average": document["load_average"],
        "signature": document["baked"]["signature"],
        "summary": document["summary"],
    }


def cell(row: dict, key: tuple) -> float:
    name, statistic = key
    block = row.get("summary", {}).get(name)
    if block is None:
        return math.nan
    value = block if statistic is None else block.get(statistic, math.nan)
    return 1e3 * value if name in MILLISECONDS else float(value)


def table(rows: list[dict]) -> None:
    header = f"{'variant':<22}" + "".join(f"{label:>11}" for label, _ in COLUMNS)
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        if "failed" in row:
            print(f"{row['name']:<22}{'FAILED':>11}")
            continue
        line = "".join(f"{cell(row, key):>11.3f}" for _, key in COLUMNS)
        print(f"{row['name']:<22}{line}")


def main(argv=None) -> int:
    options = arguments(argv)
    rows = []
    for name, patch in variants_of(options).items():
        print(f"== {name} {json.dumps(patch.get('options', patch))}", flush=True)
        try:
            row = run(name, patch, options)
        except subprocess.TimeoutExpired:
            row = {"name": name, "patch": patch, "failed": "timed out"}
        rows.append(row)
        if "failed" in row:
            print("   FAILED", row["failed"].strip().splitlines()[-1:], flush=True)
        else:
            print(
                f"   qp {cell(row, ('qp_iterations', 'median')):.0f}"
                f" / {cell(row, ('solve_time_s', 'median')):.2f} ms"
                f" / fallback {cell(row, ('fallback_cycles', None)):.0f}"
                f" / {row['wall_s']:.0f} s"
                f"{' (compiled)' if row['compiled'] else ''}",
                flush=True,
            )
        # rewritten every variant, so a killed sweep keeps what it finished
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(
            json.dumps(
                {"moves": options.moves, "seed": options.seed, "rows": rows},
                indent=1,
                default=float,
            )
        )

    table(rows)
    print(f"wrote {options.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
