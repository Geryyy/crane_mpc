#!/usr/bin/env python3
"""
Sweep the MPC over a grid of horizon knots and sample times and report solve cost.

Issue 115's oracle. Drives `mpc_a2b.py` once per cell as a subprocess -- same OCP,
same integrator, same RTI backend as the deployed solver -- and reads the CSV it
writes beside its plot. Nothing here reimplements the problem.

A cell is only a candidate if its horizon covers a sway period; a grid that does
not cover one lets the optimizer excite an oscillation that develops after the
horizon ends, whatever it times at.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
HARNESS = PACKAGE / "scripts" / "mpc_a2b.py"
OUT_ROOT = PACKAGE / "build" / "sweep_grid"

# crane_mpc/README.md design value, slowest sway period ~2s, never computed. Screen, not proof.
SWAY_PERIOD_S = 2.0

# config/crane_mpc.yaml value; hardcoded since callers most often override budget per run.
DEFAULT_BUDGET_S = 0.03


def median_column(rows: list[dict], column: str) -> float:
    """Median of one CSV column over the finite cycles, NaN if there are none."""
    values = []
    for row in rows:
        try:
            value = float(row.get(column, ""))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return statistics.median(values) if values else math.nan


def run_cell(knots: int, dt: float, extra: list[str]) -> dict | None:
    """One harness run. Returns solve-time statistics, or None if it failed."""
    out_dir = OUT_ROOT / f"k{knots}_dt{int(round(dt * 1000))}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot = out_dir / "run.png"

    command = [
        sys.executable,
        str(HARNESS),
        # figures unused, dominate a cell's wall clock
        "--no-plot",
        "--horizon-knots",
        str(knots),
        "--dt",
        str(dt),
        "--output",
        str(plot),
        *extra,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    # shared box; same solver measured 12.6ms idle vs 73ms at load 7.9 -- load matters
    load = os.getloadavg()[0]
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        return {
            "failed": tail[-1] if tail else "no output",
            "returncode": completed.returncode,
        }

    rows = list(csv.DictReader((plot.with_suffix(".csv")).open()))
    if not rows:
        return {"failed": "the harness wrote no rows"}

    # non-finite solve time = refused/fallback cycle; count separately, a NaN would poison max
    raw = [float(row["solve_time_s"]) for row in rows]
    times = sorted(value for value in raw if math.isfinite(value))
    non_finite = len(raw) - len(times)
    if not times:
        return {
            "failed": f"every one of {len(raw)} cycles reported a non-finite solve time"
        }
    # non-converged count matters more than an average that would hide it
    not_converged = sum(
        1 for row in rows if row.get("status", "") not in ("0", "ACADOS_SUCCESS")
    )
    fallbacks = sum(
        1 for row in rows if row.get("fallback", "").lower() in ("1", "true")
    )
    # acados' own timing split; read off the header so files can't disagree on columns
    breakdown = {
        f"{column}_ms": 1e3 * median_column(rows, column)
        for column in rows[0]
        if column.startswith("time_")
    }
    return {
        "load_1min": load,
        **breakdown,
        "qp_iter_median": median_column(rows, "qp_iter"),
        "cycles": len(times),
        "median_ms": 1e3 * statistics.median(times),
        "p90_ms": 1e3 * times[min(len(times) - 1, int(0.9 * len(times)))],
        "max_ms": 1e3 * times[-1],
        "not_converged": not_converged,
        "fallbacks": fallbacks,
        "non_finite": non_finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knots", type=int, nargs="+", default=[50, 40, 30])
    parser.add_argument("--dt", type=float, nargs="+", default=[0.04, 0.05, 0.06])
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S)
    parser.add_argument(
        "--json",
        type=Path,
        default=OUT_ROOT / "sweep.json",
        help="machine-readable result",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="pass through to the harness on the FIRST cell only; later cells reuse the cache",
    )
    arguments, passthrough = parser.parse_known_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    first = True
    for knots in arguments.knots:
        for dt in arguments.dt:
            extra = list(passthrough)
            if arguments.rebuild and first:
                extra.append("--rebuild")
            first = False
            horizon_s = (knots - 1) * dt
            print(f"k={knots:3d} dt={dt:.3f} horizon={horizon_s:.2f}s ...", flush=True)
            cell = run_cell(knots, dt, extra)
            cell.update(knots=knots, dt=dt, horizon_s=horizon_s)
            cell["covers_sway"] = horizon_s >= SWAY_PERIOD_S
            results.append(cell)

    arguments.json.write_text(json.dumps(results, indent=2))

    print()
    print(
        "| knots | dt | horizon s | covers sway | median ms | p90 ms | max ms | "
        "lin ms | sim ms | qp ms | qp it | load | !conv | fallback | non-finite |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in results:
        if "failed" in cell:
            print(
                f"| {cell['knots']} | {cell['dt']} | {cell['horizon_s']:.2f} | "
                f"{'yes' if cell['covers_sway'] else 'no'} | FAILED: {cell['failed']} |"
                + " |"
                * 10
            )
            continue
        print(
            f"| {cell['knots']} | {cell['dt']} | {cell['horizon_s']:.2f} | "
            f"{'yes' if cell['covers_sway'] else 'no'} | {cell['median_ms']:.2f} | "
            f"{cell['p90_ms']:.2f} | {cell['max_ms']:.2f} | "
            f"{cell.get('time_lin_ms', float('nan')):.2f} | "
            f"{cell.get('time_sim_ms', float('nan')):.2f} | "
            f"{cell.get('time_qp_ms', float('nan')):.2f} | "
            f"{cell['qp_iter_median']:.0f} | {cell['load_1min']:.2f} | "
            f"{cell['not_converged']} | {cell['fallbacks']} | {cell['non_finite']} |"
        )

    budget_ms = 1e3 * arguments.budget
    candidates = [
        cell
        for cell in results
        if "failed" not in cell and cell["covers_sway"] and cell["p90_ms"] <= budget_ms
    ]
    print()
    if candidates:
        best = min(candidates, key=lambda cell: cell["median_ms"])
        print(
            f"Candidates (cover a sway period, p90 inside {budget_ms:.0f} ms): "
            + ", ".join(f"k={cell['knots']}/dt={cell['dt']}" for cell in candidates)
        )
        print(
            f"Cheapest: knots={best['knots']} dt={best['dt']} median={best['median_ms']:.2f} ms"
        )
    else:
        print(f"No cell covers a sway period within {budget_ms:.0f} ms at p90.")
    print(f"Wrote: {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
