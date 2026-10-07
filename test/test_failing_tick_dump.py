"""
A tick the QP failed on has to leave a file, and that file has to solve the same.

The escalation this exists for is only seen in Gazebo -- issue 213 reproduced
none of three guesses offline -- so what is tested is the path, not the fault: a
non-finite path control point reaches `p`, acados answers a non-zero QP status,
the dump names the input that carried it and `replay_tick` solves it back to the
same status. A dump that could not be replayed would be a second guess.
"""

import math
import sys
from pathlib import Path

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
from crane_mpc.solver import Ocp, PathCycle

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


def test_a_tick_off_the_feedback_phase_replays_on_both_of_its_initial_states(
    ocp, tmp_path
):
    """
    The onset tick of every escalation is a split one, and it has two `x0`s.

    The cycle before it converged, so it prepared -- so the tick that explains
    the fault is linearised about a *predicted* state and pinned to a
    *measured* one. Stage 0 is pinned with no slack, so a replay that collapses
    the pair solves a different QP and reports "does not reproduce" on a tick
    whose data was fine. What is asserted is that the replay rebuilds the
    problem the dump recorded, read back through `Ocp.last_problem`.
    """
    cycle = holding_cycle(ocp)
    cycle.failing_ticks = dump.Sink(tmp_path / "ticks", 2)
    clean, refusal = cycle.solve()
    assert not refusal and clean.qp_status == 0

    # What `prepare_next` does, without the reference plumbing a node has: the
    # next cycle linearised about this one's own one-step prediction.
    predicted = clean.states[1]
    q_eq = predicted[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    ocp.prepare(
        predicted,
        cycle.horizon,
        q_eq,
        ocp.carried(clean),
        path=spoil(ocp, cycle.horizon),
    )
    cycle.prepared = True
    cycle.prepared_horizon = cycle.horizon
    cycle.prepared_in_flight = None
    # The measurement the preparation did not have, which is the whole of what
    # the feedback phase is allowed to move.
    cycle.x0 = predicted.copy()
    cycle.x0[cs.X_PLANNED_POSITION] += 0.02

    solution, refusal = cycle.solve()
    assert not refusal, refusal
    assert solution.qp_status != 0
    reference = cycle.failing_reference(solution)
    cycle.write_failing_tick(solution, reference)

    record = dump.load(sorted((tmp_path / "ticks").glob("*.npz"))[0])
    assert bool(record["split"]), "a feedback-phase tick has to say so"
    assert record["bound"] != pytest.approx(record["x0"]), "the two states collapsed"

    replayed = replay_tick.replay(ocp, record)
    rebuilt = ocp.last_problem()
    assert bool(rebuilt["split"])
    assert rebuilt["x0"] == pytest.approx(record["x0"])
    assert rebuilt["bound"] == pytest.approx(record["bound"])
    assert np.array_equal(rebuilt["p"], record["p"], equal_nan=True)
    assert replayed.qp_status == solution.qp_status
    assert "prepare/feedback" in replay_tick.report(record)


def test_a_tick_that_cannot_be_read_back_is_said_once_and_not_retried(ocp, tmp_path):
    """
    `Ocp.last_problem` refuses on top of a standing preparation -- out loud.

    Unreachable today (a non-zero QP status is `FAILED` and `prepare_next`
    early-returns on that), but widening the dump's trigger to a status that
    does prepare lands here, and swallowing it would make the dump write
    nothing for a whole run with nothing in the log saying so.
    """
    said = []
    cycle = holding_cycle(ocp)
    cycle.failing_ticks = dump.Sink(tmp_path / "ticks", 5, say=said.append)
    solution, refusal = cycle.solve()
    assert not refusal
    ocp.prepare(solution.states[1], cycle.horizon, cycle.q_eq, ocp.carried(solution))

    cycle.write_failing_tick(solution, cycle.horizon)
    assert len(said) == 1 and "could not be read back" in said[0]
    # Spent, not retried: the refusal is the same every cycle and reading it
    # back costs the next cycle its deadline.
    assert not cycle.failing_ticks.wants(3)
    assert not (tmp_path / "ticks").exists()


def test_the_dumps_are_bounded_and_off_by_default_at_zero(tmp_path):
    """The fault repeats, so an unbounded dump is a full disk per escalation."""
    sink = dump.Sink(tmp_path / "ticks", 1)
    record = {"cycle": 3, "qp_status": 3, "x0": np.zeros(2)}
    assert sink.wants(3) and not sink.wants(0)
    assert sink.write(record) is not None
    assert not sink.wants(3)
    assert sink.write(dict(record, cycle=4)) is None
    assert len(list((tmp_path / "ticks").glob("*.npz"))) == 1
    assert not dump.Sink(tmp_path / "off", 0).wants(3)
