"""
The tick a solve failed on, written where an offline harness can replay it.

Nothing recorded the inputs of a failing cycle before this: `SolverHealth`
carries the diagnosis and not the problem, so issue 213 guessed at three offline
reproductions of `mpc` §6's escalation and every one of them solved clean.

Written on any non-zero QP status, which is the QP not being solved -- a NaN in
it, or a step that collapsed, and the two want opposite investigations.
**`qp_stat` is an acados status, not HPIPM's**: `ocp_qp_hpipm.c` translates
before storing it, so 1 is HPIPM's `NAN_SOL`, 2 its `MAX_ITER`, 3 its
`MIN_STEP` and 9 its `INCONS_EQ`. `solver.status_word` names it; issue 213 read
the raw HPIPM enum and had 1 and 3 the wrong way round.

One `.npz` per failing tick, bounded per run. `nonfinite` names the recorded
input that is not finite -- none of them, on a `MIN_STEP`, which is the point:
it says the fault is conditioning and not data. `scripts/trials/replay_tick.py`
solves the file again.

ROS-free, like `cycle.py`: where the files go is the node's to decide.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from crane_model import symbolic as cs

from . import problem
from .solver import status_word

#: One file per tick: the pid separates two runs sharing a directory, the cycle
#: count orders the ticks of one run.
FILE = "tick_{pid}_{cycle:06d}.npz"

#: What a human is told to run on the file.
REPLAY_COMMAND = "crane_mpc/scripts/trials/replay_tick.py"

#: Empty stands for "the solve was not given one" -- under C4 no stage carries a
#: move cost, so there is no command in flight. `np.savez` carries no `None`.
ABSENT = np.zeros(0)

#: Recorded inputs `nonfinite` scans by name. `p` is scanned by block instead:
#: the model's rows, the path origin and the control points share one vector and
#: only the block says which of them carried the NaN in.
INPUTS = (
    "x0",
    "bound",
    # Carries its own NaN source: the force rows are `h_eff` at `x0` and
    # nothing checks them, so a finite state can still write a non-finite one.
    "yref",
    "q_eq",
    "q_a_ref",
    "dq_a_ref",
    "path_control",
    "path_origin",
    "payload_mass_kg",
    "payload_com_m",
    "tool_position",
    "guess_states",
    "guess_inputs",
    "in_flight",
)


def tick(cycle, solution, reference) -> dict:
    """
    Everything one solve was handed, and what it answered.

    `Ocp.last_problem` and not the cycle's own fields for what reached the QP: a
    cycle that took the feedback phase had its problem written by `prepare`, one
    cycle and one predicted state earlier, and the cycle's `x0`/`guess`/path of
    *now* would be a problem nobody solved. `reference` is passed rather than
    read off the cycle because the ladder has had it by then --
    `Cycle.failing_reference` takes the copy.
    """
    ocp = cycle.ocp
    written = ocp.last_problem()
    path = written["path"]
    guess = written["guess"]
    in_flight = written["in_flight"]
    return {
        "cycle": cycle.cycles,
        "stamp_ns": cycle.timing.stamp_ns(),
        "x0": written["x0"],
        "bound": written["bound"],
        "split": written["split"],
        "yref": written["yref"],
        "t": reference.t,
        "q_a_ref": reference.q_a_ref,
        "dq_a_ref": reference.dq_a_ref,
        "q_eq": written["q_eq"],
        "tool_position": cycle.tool_position,
        "path_control": path.control,
        "path_origin": path.origin,
        "path_nominal_rate": path.nominal_rate,
        "payload_mass_kg": ocp.payload_mass_kg,
        "payload_com_m": ocp.payload_com_m,
        "p": written["p"],
        "guess_states": guess.states,
        "guess_inputs": guess.inputs,
        "in_flight": ABSENT if in_flight is None else in_flight,
        "warm": written["warm"],
        "status": solution.status,
        "qp_status": solution.qp_status,
        "iterations": solution.iterations,
        "qp_iterations": solution.qp_iterations,
        "solve_time_s": solution.solve_time_s,
        "residuals": written["residuals"],
    }


def load(path) -> dict:
    """Read a dump back as the dict `tick` wrote; scalars come back 0-d."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _finite(value) -> bool:
    array = np.asarray(value, dtype=float)
    return bool(array.size == 0 or np.all(np.isfinite(array)))


def nonfinite(record) -> list:
    """Name every recorded input carrying a non-finite number, `p` by block."""
    found = [name for name in INPUTS if not _finite(record[name])]
    p = np.atleast_2d(np.asarray(record["p"], dtype=float))
    blocks = (
        ("p (the model's own rows)", p[:, : cs.NP]),
        ("p (path origin)", p[:, problem.P_PATH_ORIGIN]),
        ("p (path control)", p[:, problem.P_PATH_CONTROL :]),
    )
    return found + [name for name, block in blocks if not _finite(block)]


class Sink:
    """
    Where failing ticks go, and how many of them one run may write.

    Bounded because the failure repeats: `mpc` §6 escalates on five consecutive
    non-converged solves and the cycles after it keep solving, so an unbounded
    dump would fill a disk with one fault. A refused write spends the budget
    rather than retrying every cycle.

    `say` is called on the first file and on a refusal, so the path is in the
    log of the run that produced it.
    """

    def __init__(self, directory, limit: int, say=None) -> None:
        self.directory = Path(directory)
        self.limit = max(0, int(limit))
        self.written = 0
        self._say = say

    def wants(self, qp_status: int) -> bool:
        """Whether a tick with this status would be taken: reading one is not free."""
        return bool(qp_status) and self.written < self.limit

    def write(self, record) -> Path | None:
        """Write one tick, or `None` if there is no budget left for it."""
        if record is None or self.written >= self.limit:
            return None
        path = self.directory / FILE.format(pid=os.getpid(), cycle=int(record["cycle"]))
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            np.savez(path, **record)
        except OSError as error:
            self.refuse(f"no failing tick could be written to {path}: {error}")
            return None
        self.written += 1
        if self.written == 1:
            qp_status = int(record["qp_status"])
            self._tell(
                f"the QP came back {status_word(qp_status)} (QP status {qp_status}) "
                f"and the tick it failed on is written to {self.directory}, at most "
                f"{self.limit} of them this run. Replay one with "
                f"{REPLAY_COMMAND} <file>"
            )
        return path

    def refuse(self, message: str) -> None:
        """
        Give up on this run's dumps, out loud.

        Spends the whole budget rather than retrying: a refusal is the same
        every cycle, and the failure it fires on repeats, so retrying would
        print the escalation's length and cost the next cycle its deadline each
        time.
        """
        self.written = self.limit
        self._tell(message)

    def _tell(self, message: str) -> None:
        if self._say is not None:
            self._say(message)
