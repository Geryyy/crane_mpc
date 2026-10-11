#!/usr/bin/env python3
"""
Issue 276: the gate columns of one cost, per offline arm.

What the effort row is since 276 is `tau_a - h_g(q_k)`, gravity at the stage's
own pose, symbolic in `cost_y_expr` with `yref`'s force rows zero. The four
`--reference` arms 265/267 drove this file with are **gone**: they wrote a
constant into rows that now subtract gravity themselves, so an arm would price
`tau - h_g(q_k) - h_bar` and mean nothing (271's note, made good).

    ./scripts/trials/effort_split.py --goal out --payload-mass 800 \
        --plant-payload 800 --report build/276_ship_out_800.json

Calls `wire_chain.main` -- the node's own `Cycle` through the wire -- under
monkeypatches and no edit to the package. Every `wire_chain` flag passes
through, so `--payload-mass 800 --plant-payload 800` is the loaded arm and
`--set weights.<key>=<value>` is the ablation lever (a row at weight 0 is a row
removed, Gauss-Newton: no cost, no gradient, no curvature).

A wrapper on `Ocp.cost_terms` records, per scored cycle and stage, `tau_a`,
`h_eff(x0)`, `h_eff(x_k)` and `h(q_ref,k)`; one on `Cycle.step` records
`tcp_path_error` and the stall watch's verdict. The `drift_*`/`*_of_square`
keys decompose what the **retired** frozen row would have charged on these
trajectories (265's question); `effort_*` and `term_mean` are what this cost
actually paid.

`--split` dumps the per-stage arrays as npz.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import harness
import numpy as np
import wire_chain
from crane_model import CraneModel, Tool
from crane_mpc import bspline, problem, reports
from crane_mpc import cycle as cy
from crane_mpc.solver import Ocp, stage_equilibrium
from harness import cs

Q_A = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + cs.PLANNED_DOF)
Q_U = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF)
TAU = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + cs.PLANNED_DOF)
FORCE = slice(problem.Y_ACTUATED_FORCE, problem.Y_ACTUATED_FORCE + cs.PLANNED_DOF)


def hold_force(ocp: Ocp, q_a, q_u) -> np.ndarray:
    """`h_eff` at a pose alone: velocities and the force state left at rest."""
    x = np.zeros(ocp.nx)
    x[Q_A], x[Q_U] = q_a, q_u
    return ocp.static_hold_force(x)


def path_poses(ocp: Ocp, path) -> np.ndarray:
    """`q_a` on the planned curve at each stage's nominal progress."""
    s = path.progress_at(np.arange(ocp.intervals + 1) * ocp.Ts)
    return bspline.value(s, path.control, problem.PATH_KNOTS)


def per_stage_reference(ocp: Ocp, path, q_eq) -> np.ndarray:
    """`h(q_ref,k)` per running stage; the terminal stage prices no effort."""
    poses = path_poses(ocp, path)
    return np.array(
        [
            hold_force(ocp, poses[stage], stage_equilibrium(q_eq, stage))
            for stage in range(ocp.intervals)
        ]
    )


def frozen_reference(ocp: Ocp, path, q_eq) -> np.ndarray:
    """`h_eff(x0)` for every stage: the row 276 retired, as a yardstick."""
    frozen = ocp.static_hold_force(ocp._written["x0"])
    return np.repeat(frozen[None], ocp.intervals, axis=0)


def install_recorder(log: list) -> None:
    """Record the per-stage split beside every cost split `wire_chain` takes."""
    original = Ocp.cost_terms

    def cost_terms(self, solution, q_eq):
        terms = original(self, solution, q_eq)
        log.append(split(self, solution, q_eq, terms))
        return terms

    Ocp.cost_terms = cost_terms


def install_tcp_error(log: list, stalled: list, model) -> None:
    """
    Record, per cycle, `reports.tcp_path_error` and the stall watch's own verdict.

    `progress_stalled` is the flag 269's loaded lift fires -- the plan standing
    still for `max_stall_time` with no constraint binding -- read off the node's
    own `Cycle`, so an offline arm answers the same question the live one does.
    """
    original = cy.Cycle.step

    def step(self, *args, **kwargs):
        outcome = original(self, *args, **kwargs)
        error, _ = reports.tcp_path_error(self, model)
        if error is not None:
            log.append(float(error))
        stalled.append(bool(self.progress_stalled))
        return outcome

    cy.Cycle.step = step


def split(ocp: Ocp, solution, q_eq, terms) -> dict:
    """One cycle's effort term, per stage, against four force references."""
    stages = ocp.intervals
    states = solution.states
    tau = states[:stages, TAU]
    frozen = np.repeat(ocp.static_hold_force(states[0])[None], stages, axis=0)
    # The stage's *whole* state, not its pose with the velocities zeroed:
    # `h_eff` carries velocity terms too, and zeroing them puts a non-zero
    # drift on stage 0, where by construction there is none.
    solved = np.array([ocp.static_hold_force(states[stage]) for stage in range(stages)])
    written = ocp._written["yref"][:stages, FORCE]
    weight = np.diag(ocp._weight)[FORCE]
    return {
        # N*m (N on the telescope), one row per running stage.
        "tau": tau,
        "frozen": frozen,
        "solved": solved,
        "path": per_stage_reference(ocp, ocp._last_path, q_eq),
        "written": written,
        "weight": weight,
        # As `cost_terms` reports it: since 267 it reads the effort reference
        # back off `_written["yref"]` (zero since 276), so `tau_a` is the cost
        # the cycle paid.
        "terms": dataclasses.asdict(terms),
    }


def summarise(log: list, tcp: list, stalled: list, scores: dict) -> dict:
    """Reduce the log to the two numbers the issue asks for, and the A/B columns."""
    if not log:
        return {}
    weight = log[0]["weight"]
    tau = np.concatenate([row["tau"] for row in log])
    frozen = np.concatenate([row["frozen"] for row in log])
    solved = np.concatenate([row["solved"] for row in log])
    path = np.concatenate([row["path"] for row in log])
    # `tau - h(x0) = drift + inertial`, so the retired frozen row's square is
    # `drift^2 + 2 drift inertial + inertial^2` and the three shares are 265's
    # question: how much of it a reference tracking the pose would not charge.
    #
    # Every `drift_*`, `*_share_of_square` and `effort_vs_*` key below is that
    # **retired** row decomposed on this run's trajectories -- a counterfactual
    # since 276. `effort_*` and `term_mean` are what this cost paid.
    drift = solved - frozen
    inertial = tau - solved
    total = float(np.sum(0.5 * weight * (drift + inertial) ** 2))
    effort = np.array([row["terms"]["tau_a"] for row in log])
    whole = np.array([sum(row["terms"].values()) for row in log])
    biggest = np.array([max(row["terms"].values()) for row in log])
    stages = log[0]["tau"].shape[0]
    by_stage = (0.5 * weight * drift**2).reshape(len(log), stages, -1).sum(axis=2)
    return {
        "cycles": len(log),
        # Issue 265 acceptance 1: how much of the shipped effort term is gravity
        # drift rather than inertia.
        "drift_share_of_square": float(np.sum(0.5 * weight * drift**2) / total),
        "cross_share_of_square": float(np.sum(weight * drift * inertial) / total),
        "inertial_share_of_square": float(np.sum(0.5 * weight * inertial**2) / total),
        # Per axis, N*m (N on the telescope): slew, boom, arm, telescope, rotator.
        "drift_rms": np.sqrt(np.mean(drift**2, axis=0)).tolist(),
        "inertial_rms": np.sqrt(np.mean(inertial**2, axis=0)).tolist(),
        "drift_abs_p95": np.percentile(np.abs(drift), 95, axis=0).tolist(),
        # Where along the horizon the drift sits: mean priced drift at stage
        # 0 / N/4 / N/2 / 3N/4 / N-1.
        "drift_cost_by_stage": by_stage.mean(axis=0)[
            [0, stages // 4, stages // 2, 3 * stages // 4, stages - 1]
        ].tolist(),
        # What each candidate reference would charge, same solved trajectories.
        "effort_vs_frozen": total,
        "effort_vs_path_point": float(np.sum(0.5 * weight * (tau - path) ** 2)),
        "effort_vs_solved_pose": float(np.sum(0.5 * weight * inertial**2)),
        # The A/B columns: what the term took of the cycle's whole cost,
        # measured against the reference that cycle was written with.
        "effort_mean": float(np.mean(effort)),
        "effort_median": float(np.median(effort)),
        "effort_share_median": float(np.median(effort / whole)),
        "effort_largest_fraction": float(np.mean(effort >= biggest)),
        "term_mean": {
            name: float(np.mean([row["terms"][name] for row in log]))
            for name in log[0]["terms"]
        },
        "tcp_path_error_m": (np.percentile(tcp, [50, 95, 100]).tolist() if tcp else []),
        # Share of stepped cycles the stall watch called stalled (269).
        "stalled_share": float(np.mean(stalled)) if stalled else 0.0,
        # Straight off `wire_chain`, so the A/B reads one json per arm.
        "sway_peak": scores.get("sway_peak"),
        "sway_rate_after_2s": scores.get("sway_rate_after_2s"),
        "final_error": scores.get("final_error"),
        "solve_time_s": scores.get("solve_time_s"),
        "qp_iterations": scores.get("qp_iterations"),
        "qp_iter_capped": scores.get("qp_iter_capped"),
        "nonzero_status": scores.get("nonzero_status"),
        "handback": scores.get("handback"),
        "silence": scores.get("silence"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1], add_help=False
    )
    parser.add_argument(
        "--cost-decimation",
        type=int,
        default=2,
        help="split every n-th solved cycle; wire_chain's own default is 10. "
        "Not 1: its gate is `len(riders) %% n == 1`, which never fires at 1",
    )
    parser.add_argument("--split", type=Path, default=None, help="npz, per stage")
    parser.add_argument("--report", type=Path, default=None, help="json, the summary")
    mine, rest = parser.parse_known_args(argv)

    wire_chain.COST_DECIMATION = mine.cost_decimation
    log: list = []
    tcp: list = []
    stalled: list = []
    install_recorder(log)
    install_tcp_error(
        tcp, stalled, CraneModel(harness.description_xml(), Tool(problem.TOOL))
    )
    scores = Path("build") / "effort_split_scores.json"
    scores.parent.mkdir(parents=True, exist_ok=True)
    status = wire_chain.main([*rest, "--json", str(scores)])
    read = json.loads(scores.read_text()) if scores.exists() else {}
    summary = summarise(log, tcp, stalled, read if isinstance(read, dict) else read[-1])
    summary["wire_chain_status"] = status
    print(json.dumps(summary, indent=1))
    if mine.report:
        mine.report.parent.mkdir(parents=True, exist_ok=True)
        mine.report.write_text(json.dumps(summary, indent=1))
    if mine.split and log:
        np.savez_compressed(
            mine.split,
            **{
                name: np.stack([row[name] for row in log])
                for name in ("tau", "frozen", "solved", "path", "written")
            },
            weight=log[0]["weight"],
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
