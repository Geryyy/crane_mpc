#!/usr/bin/env python3
"""
The gate columns and the per-row cost share of one cost, per offline arm.

Replaces `effort_split.py`, whose whole subject -- the effort row and the four
force references 265/267/271/275/276 tried on it -- issue 277 deleted. What is
left is the part every cost change needs: the gate's quality columns straight
off `wire_chain`, and the split of the cost by row, so "no single row dominates
by construction" is a measurement and not a hope.

    ./scripts/trials/cost_shares.py --goal out --payload-mass 800 \
        --plant-payload 800 --report build/277_ship_out_800.json

Calls `wire_chain.main` -- the node's own `Cycle` through the wire -- under
monkeypatches and no edit to the package. Every `wire_chain` flag passes
through, so `--payload-mass 800 --plant-payload 800` is the loaded arm and
`--set weights.<key>=<value>` is the lever (a row at weight 0 is a row removed,
Gauss-Newton: no cost, no gradient, no curvature).

`term_mean` is each `CostTerms` row averaged over scored cycles; `term_share` is
the same row's share of that cycle's whole cost, median over cycles -- the
median, because the first cycle of a move starts the horizon metres off path and
carries the terminal row's whole mean (276's averaging artefact).
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
from crane_mpc import cycle as cy
from crane_mpc import problem, reports
from crane_mpc.solver import Ocp


def install_recorder(log: list) -> None:
    """Record the cost split beside every one `wire_chain` takes."""
    original = Ocp.cost_terms

    def cost_terms(self, solution, q_eq):
        terms = original(self, solution, q_eq)
        log.append(dataclasses.asdict(terms))
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


def summarise(log: list, tcp: list, stalled: list, scores: dict) -> dict:
    """Reduce the log to the cost split, and carry `wire_chain`'s own columns."""
    if not log:
        return {}
    names = list(log[0])
    whole = np.array([sum(row.values()) for row in log])
    share = {
        name: float(np.median([row[name] for row in log] / whole)) for name in names
    }
    return {
        "cycles": len(log),
        "term_mean": {
            name: float(np.mean([row[name] for row in log])) for name in names
        },
        "term_share": share,
        # The row that dominates, and by how much: the ticket's own question.
        "largest_share": max(share, key=share.get),
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
    scores = Path("build") / "cost_shares_scores.json"
    scores.parent.mkdir(parents=True, exist_ok=True)
    status = wire_chain.main([*rest, "--json", str(scores)])
    read = json.loads(scores.read_text()) if scores.exists() else {}
    summary = summarise(log, tcp, stalled, read if isinstance(read, dict) else read[-1])
    summary["wire_chain_status"] = status
    print(json.dumps(summary, indent=1))
    if mine.report:
        mine.report.parent.mkdir(parents=True, exist_ok=True)
        mine.report.write_text(json.dumps(summary, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
