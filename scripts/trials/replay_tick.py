#!/usr/bin/env python3
"""
Replay a tick the node dumped, through `Ocp.solve`.

Two questions a live log cannot answer: which of the inputs carried the
non-finite number, and whether the status is a property of the data or of the
node that was running. So this says the first off the file and answers the
second by solving the recorded problem again on a freshly opened solver.

    ./scripts/trials/replay_tick.py ~/.ros/log/crane_mpc_ticks/tick_7461_000517.npz

`harness` is imported inside `main` on purpose: it pulls in `crane_planning` and
an exporter, and `test/test_failing_tick_dump.py` imports `replay` without
either.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from crane_mpc import dump
from crane_mpc.solver import Guess, PathCycle, status_word

PACKAGE = Path(__file__).resolve().parents[2]


def report(record) -> str:
    """Say what the file carries, before anything is solved again."""
    found = dump.nonfinite(record)
    qp_status = int(record["qp_status"])
    if found:
        verdict = f"non-finite inputs: {', '.join(found)}"
    else:
        verdict = (
            "every recorded input is finite, so nothing the cycle was handed "
            f"carried a NaN in: at {status_word(qp_status)} the QP was hard, not "
            "poisoned"
        )
    return "\n".join(
        [
            f"tick {int(record['cycle'])}, stamped {int(record['stamp_ns'])} ns: "
            f"{status_word(int(record['status']))}, QP "
            f"{status_word(qp_status)} (status {qp_status}), "
            f"{int(record['iterations'])} SQP and "
            f"{int(record['qp_iterations'])} QP iterations in "
            f"{1e3 * float(record['solve_time_s']):.1f} ms, "
            f"{'warm' if bool(record['warm']) else 'cold'} started",
            f"residuals [stat, eq, ineq, comp]: "
            f"{np.asarray(record['residuals'], dtype=float)}",
            f"payload {float(record['payload_mass_kg']):g} kg at "
            f"{np.asarray(record['payload_com_m'])} m, tool "
            f"{float(record['tool_position']):g}, path origin "
            f"{float(record['path_origin']):g} at "
            f"{float(record['path_nominal_rate']):g} /s",
            verdict,
        ]
    )


def replay(ocp, record):
    """
    Solve the recorded problem again. Returns the `Solution`.

    The recorded iterate goes back as the warm start, so what is written is the
    seed the dumped cycle actually solved from, clipping included. `horizon` is
    `None`: the resampled reference reaches the QP only through the path, which
    every record carries -- a cycle without one never solved (issue 188).

    One `solve`, because that is what the node takes. A dump written before
    issue 226 carries a `split` flag and a second initial state (`bound`) from
    the preparation/feedback split; both are ignored, so such a tick replays as
    the node would run it today and not as it ran then.

    `cold=False`: a fresh solver stands at zero payload, so a dump off a loaded
    move would read as a payload step, force the cold start and throw away that
    seed -- and the seed is the suspect.
    """
    ocp.pin_tool(float(record["tool_position"]))
    ocp.set_payload(
        float(record["payload_mass_kg"]), record["payload_com_m"], cold=False
    )
    in_flight = np.asarray(record["in_flight"], dtype=float)
    arguments = (
        record["x0"],
        None,
        record["q_eq"],
        Guess(record["guess_states"], record["guess_inputs"]),
    )
    keywords = dict(
        path=PathCycle(
            control=record["path_control"],
            origin=float(record["path_origin"]),
            nominal_rate=float(record["path_nominal_rate"]),
        ),
        in_flight=None if in_flight.size == 0 else in_flight,
    )
    return ocp.solve(*arguments, **keywords)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="a .npz written by crane_mpc")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="yaml lever"
    )
    parser.add_argument("--rebuild", action="store_true", help="re-export the solver")
    options = parser.parse_args(argv)

    sys.path.insert(0, str(PACKAGE / "scripts" / "trials"))
    import harness

    record = dump.load(options.file)
    print(report(record))

    parameters, hydraulics = harness.settings(options.set)
    ocp = harness.create_solver(parameters, hydraulics, options.rebuild)
    try:
        solution = replay(ocp, record)
    except Exception as refusal:
        # Which is a finding: the refused argument is the one carrying the NaN,
        # and the node solved anyway because `check_payload`/`_checked_x0` only
        # guard what the node writes, not what it was given.
        print(f"the replay refused the dump's own inputs: {refusal}")
        return 2

    recorded = int(record["qp_status"])
    print(
        f"replayed: {solution.status_word}, {solution.iterations} SQP and "
        f"{solution.qp_iterations} QP iterations"
    )
    if solution.qp_status == recorded and solution.status == int(record["status"]):
        print("the status reproduces, so it is the recorded data and not the node")
        return 0
    print(
        f"the status does not reproduce (recorded QP {recorded}, "
        f"{status_word(int(record['status']))}): what failed is not in the file, so "
        "it is the solver's own memory, the settings, or the export"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
