"""
A refused tick has to leave a file, and that file has to solve the same.

The escalation this exists for is only seen in Gazebo -- issue 213 reproduced
none of three guesses offline -- so what is tested is the path, not the fault: a
non-finite path control point reaches `p`, acados answers a non-zero QP status,
the dump names the input that carried it and `replay_tick` solves it back to the
same status. A dump that could not be replayed would be a second guess.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from conftest import export_for
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import dump, problem
from crane_mpc.config import machine_limits
from crane_mpc.cycle import Cycle, Timing
from crane_mpc.horizon import Grid, Knots, resample
from crane_mpc.solver import Ocp, Outcome, PathCycle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trials"))

import replay_tick  # noqa: E402

PLANNED = cs.K_PLANNED_DOF
POSE = [0.0, 0.5, 1.0, 0.2, 0.0, 0.3]
#: Tool hangs at `POSE`; zeros pin tilt outside its own stop (see `test_node`).
PASSIVE_POSE = [0.5 * math.pi - POSE[1] - POSE[2], 0.5 * math.pi]


@pytest.fixture(scope="module")
def ocp(export_base):
    """The shipped problem, which is the one a dump off a deployment carries."""
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/crane_mpc.yaml") as stream:
        parameters = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
    parameters["limits"].update(machine_limits())
    hydraulics = hydraulic_limits()
    description = problem.default_description().read_text()
    export_for(export_base, parameters, hydraulics, description)
    return Ocp(description, parameters, hydraulics)


def holding_cycle(ocp):
    """A cycle standing still at `POSE`, with everything one solve reads set."""
    cycle = Cycle(Timing(ocp.Ts, ocp.Ts), Grid(ocp.Ts, ocp.intervals + 1), "active")
    cycle.ocp = ocp
    x = np.zeros(ocp.nx)
    x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED] = POSE[:PLANNED]
    x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = PASSIVE_POSE
    cycle.tool_position = POSE[-1]
    ocp.pin_tool(cycle.tool_position)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    cycle.x0 = x
    cycle.q_eq = np.asarray(PASSIVE_POSE)

    reference = Knots.zeros(2)
    reference.t[:] = [0.0, ocp.Ts * ocp.intervals]
    reference.q_a_ref[:, :PLANNED] = POSE[:PLANNED]
    _, cycle.horizon = resample(reference, 0.0, Grid(ocp.Ts, ocp.intervals + 1))
    return cycle


def spoil(ocp, horizon):
    """The planner's curve with one non-finite control point: `p`'s own NaN."""
    window = ocp.window_path(horizon)
    control = window.control.copy()
    control[len(control) // 2, 0] = np.nan
    return PathCycle(control, origin=0.0, nominal_rate=window.nominal_rate)


def test_a_non_finite_parameter_is_dumped_and_replays_to_the_same_status(ocp, tmp_path):
    said = []
    cycle = holding_cycle(ocp)
    cycle.failing_ticks = dump.Sink(tmp_path / "ticks", 2, say=said.append)

    # The same cycle, solved clean first: a dump off a tick nothing was wrong
    # with would be noise, and it is what says the NaN below is what failed it.
    healthy, refusal = cycle.solve()
    assert not refusal and healthy.qp_status == 0
    assert cycle.failing_reference(healthy) is None

    cycle.path = spoil(ocp, cycle.horizon)
    solution, refusal = cycle.solve()
    assert not refusal, refusal
    assert solution.qp_status != 0, "a NaN in `p` has to reach the QP"

    reference = cycle.failing_reference(solution)
    resampled = reference.q_a_ref.copy()
    # What `ladder`'s shift rung does between the solve and the write: the
    # reference this tick ran on is gone unless it was copied before the ladder.
    cycle.horizon.q_a_ref[:] = 0.0
    cycle.write_failing_tick(solution, reference)

    written = sorted((tmp_path / "ticks").glob("*.npz"))
    assert len(written) == 1, "the failing tick left no file"
    # Said once, and it carries the directory: a file nobody is told about is
    # not a diagnosis.
    assert len(said) == 1 and str(tmp_path) in said[0]

    record = dump.load(written[0])
    assert record["q_a_ref"] == pytest.approx(resampled)
    assert dump.nonfinite(record) == ["path_control", "p (path control)"]
    assert np.isnan(np.asarray(record["residuals"])).any()

    replayed = replay_tick.replay(ocp, record)
    assert replayed.status == solution.status
    assert replayed.qp_status == solution.qp_status
    # Named and not just numbered: `qp_stat` is an acados status, and issue 213
    # read the number against HPIPM's own enum and chased the wrong fault.
    assert "ACADOS_NAN_DETECTED" in replay_tick.report(record)


def test_a_tick_that_cannot_be_read_back_is_said_once_and_not_retried(ocp, tmp_path):
    """
    `Ocp.last_problem` crosses into acados, so it can refuse -- out loud.

    Swallowing it would make the dump write nothing for a whole run with
    nothing in the log saying so, and the fault repeats every cycle.
    """
    said = []
    cycle = holding_cycle(ocp)
    cycle.failing_ticks = dump.Sink(tmp_path / "ticks", 5, say=said.append)
    solution, refusal = cycle.solve()
    assert not refusal
    cycle.ocp = SimpleNamespace(
        last_problem=lambda: (_ for _ in ()).throw(RuntimeError("no")),
    )

    cycle.write_failing_tick(solution, cycle.horizon)
    assert len(said) == 1 and "could not be read back" in said[0]
    # Spent, not retried: the refusal is the same every cycle and reading it
    # back costs the next cycle its deadline.
    assert not cycle.failing_ticks.wants(True)
    assert not (tmp_path / "ticks").exists()


def test_the_dumps_are_bounded_and_off_by_default_at_zero(tmp_path):
    """The fault repeats, so an unbounded dump is a full disk per escalation."""
    sink = dump.Sink(tmp_path / "ticks", 1)
    record = {"cycle": 3, "status": 4, "qp_status": 3, "x0": np.zeros(2)}
    assert sink.wants(True) and not sink.wants(False)
    assert sink.write(record) is not None
    assert not sink.wants(True)
    assert sink.write(dict(record, cycle=4)) is None
    assert len(list((tmp_path / "ticks").glob("*.npz"))) == 1
    assert not dump.Sink(tmp_path / "off", 0).wants(True)


def test_a_refusal_the_qp_status_does_not_see_is_still_dumped(ocp, tmp_path):
    """
    The two refusal shapes the old `qp_status` gate dumped nothing for.

    `Ocp.solve` refuses on a non-zero NLP status, a non-zero QP status *or* a
    non-finite state, and the gate read only the second: every live refusal in
    227's twelve captures was NLP status 4 with `qp_status` 0, and 233's own r3
    was both statuses 0 with a state that was not finite. Both are faked onto a
    solve that really ran, so what is tested is `Cycle`'s gate and the write
    behind it, not an arithmetic on two ints.
    """
    for status, qp_status in ((4, 0), (0, 0)):
        said = []
        cycle = holding_cycle(ocp)
        cycle.failing_ticks = dump.Sink(tmp_path / f"ticks{status}", 2, say=said.append)
        solved, refusal = cycle.solve()
        assert not refusal and solved.qp_status == 0 and solved.status == 0
        assert cycle.failing_reference(solved) is None, "a clean solve leaves no dump"

        failed = replace(
            solved, outcome=Outcome.FAILED, status=status, qp_status=qp_status
        )
        reference = cycle.failing_reference(failed)
        assert reference is not None, f"NLP {status}, QP {qp_status} left no dump"
        cycle.write_failing_tick(failed, reference)
        assert len(list((tmp_path / f"ticks{status}").glob("*.npz"))) == 1
        assert f"NLP status {status}" in said[0] and "QP status 0" in said[0]
        # Neither status set is the third fault, and the line has to say so
        # rather than reporting the refusal as a success.
        assert ("not finite" in said[0]) == (status == 0)
