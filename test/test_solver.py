"""The solver wrapper: what it predicts, and what it plans."""

import json

import crane_ocp_export as ox
import numpy as np
import pytest
import yaml
from acados_template import AcadosOcpSolver, AcadosSimSolver
from ament_index_python.packages import get_package_share_directory
from conftest import export_for, window_path
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import bspline, problem
from crane_mpc.config import machine_limits
from crane_mpc.horizon import Grid, Knots, resample
from crane_mpc.solver import (
    Ocp,
    Outcome,
    PathCycle,
    export_key,
    move_suppression,
    predictor_label,
    slack_prices,
    stage_reference,
    weight_matrices,
)

PLANNED = cs.PLANNED_DOF


def read_parameters(name):
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/{name}") as stream:
        return yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]


@pytest.fixture(scope="module")
def parameters():
    values = read_parameters("crane_mpc.yaml")
    values.update({"hydraulics": hydraulic_limits()})
    # neither the box nor u^+ is in the yaml; crane_model owns both
    values["limits"].update(machine_limits())
    # These tests pin C3's semantics (ZOH replay, stage-0 move cost); C4 has its
    # own fixture below, whatever the shipped yaml selects.
    values["command_state"] = False
    values["weights"]["du"] = [10.0] * 6
    return values


@pytest.fixture(scope="module")
def ocp(parameters, export_base):
    description = problem.default_description().read_text()
    export_for(export_base, parameters, parameters["hydraulics"], description)
    return Ocp(description, parameters, parameters["hydraulics"])


def hold_state(ocp):
    x = np.zeros(cs.NX)
    x[cs.X_PLANNED_POSITION + 1] = 0.5
    # Inside `v_s`'s box, which is the declared headroom against this grid's
    # pace. A state above it is infeasible at stage 1 rather than merely tight:
    # `progress_accel_max` cannot brake into the box in one interval, and the
    # row carries no slack. It only read 1.0 here while the box was an absolute
    # 1.5/s that happened to be 3.5x nominal.
    x[cs.X_PROGRESS_RATE] = float(
        ocp.parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(ocp.parameters)
    ocp.pin_tool(0.3)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    return x


@pytest.fixture
def state(ocp):
    return hold_state(ocp)


def horizon_holding(ocp, position):
    """A reference that asks the machine to stay where it is."""
    reference = Knots.zeros(2)
    reference.t[:] = [0.0, ocp.Ts * ocp.intervals]
    reference.q_a_ref[:, :PLANNED] = position
    _, horizon = resample(reference, 0.0, Grid(ocp.Ts, ocp.intervals + 1))
    return horizon


def solved(ocp, x0, horizon, q_eq, guess=None, **keywords):
    """
    `Ocp.solve` with the horizon's own knots as the path, unless one is passed.

    Every solve needs the curve its cost is written against -- issue 188 deleted
    the fit-the-window fallback that used to supply it. These tests are about the
    boxes, the warm start and the cost rows, so any consistent path will do and
    the window is the one they had.
    """
    keywords.setdefault("path", window_path(ocp, horizon))
    return ocp.solve(x0, horizon, q_eq, guess, **keywords)


#: A plan that rests at both ends: s(t) = (1 - cos(pi t/T))/2, whose pace peaks at
#: pi/2 = 1.571x its own mean -- the shape the A2B profile has (1.759x, issue 250).
LAW_PEAK_OVER_MEAN = np.pi / 2.0


def resting_path(duration=10.0, origin=0.0, samples=201):
    """A `PathCycle` carrying that law, and nothing else the pacing reads."""
    grid = np.linspace(0.0, duration, samples)
    law = 0.5 * (1.0 - np.cos(np.pi * grid / duration))
    control = np.zeros((problem.PATH_POINTS, PLANNED))
    return PathCycle(control, origin, 1.0 / duration, law), grid, law


def test_the_progress_reference_is_paced_by_the_plans_own_time_law():
    """
    Priced against the law anchored at `origin`, and `v_s` capped off its peak.

    Both against the mean they were before issue 253: the reference ran the fast
    middle of a move at 1/1.76 of the plan's pace and the ceiling could not have
    held it anyway. Anchored at `origin` rather than at wall time, so a cycle
    that fell behind is not handed a catch-up demand -- the pace is the plan's,
    the schedule stays the optimizer's.
    """
    path, grid, law = resting_path(origin=0.5)
    assert path.peak_rate() == pytest.approx(
        LAW_PEAK_OVER_MEAN * path.nominal_rate, rel=1e-3
    )

    # `origin` 0.5 is the law's half-way point, which is where it is fastest.
    elapsed = np.array([0.0, 1.0, 2.0])
    assert path.progress_at(elapsed) == pytest.approx(
        np.interp(0.5 * grid[-1] + elapsed, grid, law)
    )
    assert path.progress_at(0.4) > 0.5 + 0.4 * path.nominal_rate
    # Still a path parameter past the end: the cap is what keeps `tangent(1)`
    # from asking for motion away from the goal.
    assert path.progress_at(2.0 * grid[-1]) == pytest.approx(1.0)


def test_a_path_with_no_time_law_is_priced_at_the_mean_pace_exactly():
    """The pre-253 ramp, to the bit: an empty field changes nothing (issue 253)."""
    control = np.zeros((problem.PATH_POINTS, PLANNED))
    for law in (None, [], np.zeros(0)):
        path = PathCycle(control, 0.25, 0.1, law)
        assert path.s_profile is None
        assert path.peak_rate() == path.nominal_rate
        elapsed = np.arange(5) * 0.6
        assert np.array_equal(
            path.progress_at(elapsed),
            np.minimum(1.0, 0.25 + elapsed * 0.1),
        )


def test_the_rate_row_is_priced_against_the_plans_pace_and_not_against_standing():
    """
    Issue 276: a `v_s` reference of 0 is a cost that prefers a standing plan.

    That is what 269's loaded lift was -- nothing bound while `s` stood, the
    rate row charged for moving, and the `s` row could not win it back because
    its residual is pinned to 0 at stage 0. Both ends matter: the reference is
    the plan's own pace where the path runs, and 0 past its end, where a
    surviving demand would price motion away from the goal.
    """
    step = 0.5
    path, _, _ = resting_path(duration=10.0)
    pace = path.pace_at(np.array([0.0, 5.0]), step)
    # The law rests at the start and is fastest at the half-way point.
    assert pace[0] < path.nominal_rate
    assert pace[1] == pytest.approx(LAW_PEAK_OVER_MEAN * path.nominal_rate, rel=0.02)
    assert path.pace_at(2.0 * 10.0, step) == pytest.approx(0.0)

    # The mean ramp has no shape, so its pace is the mean until the path ends.
    plain = PathCycle(np.zeros((problem.PATH_POINTS, PLANNED)), 0.0, 0.1, None)
    assert plain.pace_at(np.arange(3) * step, step) == pytest.approx(0.1)
    assert plain.pace_at(10.0, step) == pytest.approx(0.0)

    # And it is what lands in the residual's reference, for stage and terminal.
    for terminal in (False, True):
        reference = stage_reference(np.zeros(cs.PASSIVE_DOF), terminal, 0.3, 0.07)
        assert reference[problem.Y_PROGRESS_RATE] == pytest.approx(0.07)


def test_a_time_law_that_does_not_advance_is_refused():
    control = np.zeros((problem.PATH_POINTS, PLANNED))
    for law in ([0.0, 0.5, 0.4, 1.0], [0.3, 0.3], [0.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match="time law"):
            PathCycle(control, 0.0, 0.1, law)


def test_the_predictor_carries_the_command_that_is_in_flight(ocp, state):
    """
    Issue 125: input reaches `ddq` only via the command-lag/force rows; a
    predictor dropping them returns the same state for every command.
    """
    still = ocp.propagate(state, np.zeros(cs.NU_PROGRESS))
    command = np.zeros(cs.NU_PROGRESS)
    command[0] = 0.4
    driven = ocp.propagate(state, command)

    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + PLANNED)
    assert not np.allclose(driven[force], still[force])
    assert not np.allclose(driven[cs.X_PLANNED_VELOCITY], still[cs.X_PLANNED_VELOCITY])
    # A prediction, not a reset: pose barely moves in 60 ms.
    assert np.allclose(
        driven[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED],
        state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED],
        atol=0.02,
    )


def test_the_prediction_replays_the_one_command_that_covers_the_window(ocp, state):
    """
    `delay == Ts`, so one command covers the dead time: `propagate_applied` is
    `propagate` on the newest entry and an older one never reaches the roll.
    Empty stays refused -- it is not the same thing as a zero command.
    """
    command = np.zeros(cs.NU_PROGRESS)
    command[0] = 0.4
    held = ocp.propagate(state, command)
    assert np.allclose(ocp.propagate_applied(state, [command]), held)
    # An older, opposite command is worth whole units on the force rows; it is
    # out of the window and must move nothing.
    assert np.allclose(ocp.propagate_applied(state, [command, -command]), held)
    with pytest.raises(ValueError, match="empty history"):
        ocp.propagate_applied(state, [])


def test_a_solve_keeps_its_plan_inside_the_boxes_it_was_given(ocp, state):
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]

    solution = solved(ocp, state, horizon, q_eq)
    assert solution.outcome is not Outcome.FAILED
    assert solution.status == 0
    u_max = np.asarray(ocp.parameters["limits"]["u_max"][:PLANNED])
    assert np.all(np.abs(solution.inputs[:, :PLANNED]) <= u_max + 1e-9)
    # Constraint 1: stage zero is the given state, every row incl. actuator states.
    assert np.allclose(solution.states[0], state, atol=1e-6)
    # Constraint 2, on every stage rather than only the first.
    dq_max = np.asarray(ocp.parameters["limits"]["dq_a_max"][:PLANNED])
    velocity = solution.states[
        :, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED
    ]
    assert np.all(np.abs(velocity) <= dq_max + 1e-6)


def test_the_solution_reads_flat_the_way_it_reads_per_stage(ocp, state):
    """
    `get_flat` is stage-major, which is what reading the solution in one call assumes.

    Pinned because nothing downstream would tell a relaid-out `sl` from a real
    violation: the slack only feeds the health report (issue 187).

    `x` comes back off the solver scaled (issue 226) and `Solution` carries it
    physical, so that one row is compared through the scaling -- which also
    pins the direction of the conversion.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    solution = solved(ocp, state, horizon_holding(ocp, position), q_eq)

    stages = range(ocp.intervals + 1)
    scale = problem.state_scale(ocp.parameters, ocp.scale)
    assert np.array_equal(
        solution.states, [ocp.solver.get(k, "x") * scale for k in stages]
    )
    assert np.array_equal(
        solution.inputs, [ocp.solver.get(k, "u") for k in range(ocp.intervals)]
    )
    for field in ("sl", "su"):
        assert np.array_equal(
            ocp.solver.get_flat(field),
            np.concatenate([ocp.solver.get(k, field) for k in stages]),
        )


def test_the_cost_split_names_where_the_plan_spent(ocp, state):
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    # Ask for a pose the machine is not in: the tracking row has to carry it.
    horizon.q_a_ref[:, 0] += 0.4
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]

    solution = solved(ocp, state, horizon, q_eq)
    terms = ocp.cost_terms(solution, q_eq)
    assert terms.q_a > 0.0
    # Off pose means off path at the tool too, which is the row the split used to drop.
    assert terms.tool > 0.0
    assert all(np.isfinite(value) and value >= 0.0 for value in terms.__dict__.values())


def test_the_cost_split_leaves_no_residual_row_out(ocp, state):
    """
    acados' own number is the oracle: an omitted row cannot hide from it.

    Two conventions have to be undone first. acados scales every stage by its
    interval (`cost_scaling` defaults to [Ts] * N + [1]) while the split sums
    stage and terminal alike, and it counts the slack penalty into the same
    number -- unscaled in `slack_penalty`, so only the terminal stage's share has
    to be told apart.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    # Off pose, so every tracking row carries something -- the tool's included.
    horizon.q_a_ref[:, 0] += 0.4
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]

    solution = solved(ocp, state, horizon, q_eq)
    terms = ocp.cost_terms(solution, q_eq)
    assert terms.tool > 0.0

    taken = np.maximum(ocp.solver.get_flat("sl"), 0.0) + np.maximum(
        ocp.solver.get_flat("su"), 0.0
    )
    price = slack_prices(ocp.parameters)["terminal"]
    terminal_slack = float(price @ taken[-price.size :])
    stage = sum(
        value
        for name, value in vars(terms).items()
        if name not in ("terminal", "slack")
    )
    total = (
        ocp.Ts * (stage + terms.slack - terminal_slack)
        + terms.terminal
        + terminal_slack
    )
    assert total == pytest.approx(ocp.solver.get_cost(), rel=1e-9)


def test_the_force_state_is_priced_by_no_row_at_all(ocp, state):
    """
    Issue 277: the effort row is gone, so `tau_a` enters the cost nowhere.

    Pushing the force state far off any hold moves the residual by exactly
    nothing. That is what leaves `levenberg_marquardt` the only curvature on
    those four QP directions, and why `problem.state_scale` still scales them.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    solved(ocp, state, horizon_holding(ocp, position), q_eq)
    parameter = ocp._last_parameter
    no_input = np.zeros(cs.NU_PROGRESS)

    def residual(x):
        value = ocp._residual(ocp._to_qp(x), no_input, parameter)
        return np.asarray(value).reshape(-1)

    pushed = state.copy()
    pushed[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] += 5.0e4
    assert residual(pushed) == pytest.approx(residual(state), abs=1e-9)


def test_the_velocity_row_prices_deviation_from_the_plans_pace_not_motion(ocp, state):
    """
    Issue 277: what replaces the effort row's hidden `D dq_a` damper.

    The row is `dq_a - tangent(s) v_s`, so a machine running the plan at the
    plan's own pace pays nothing on it, while standing still *while the plan
    asks for a pace* costs the whole of it -- which the absolute velocity price
    the effort row smuggled in had backwards (269's stall).
    """
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    # A path that goes somewhere: a holding one has no tangent to be paced by.
    reference = Knots.zeros(2)
    reference.t[:] = [0.0, ocp.Ts * ocp.intervals]
    reference.q_a_ref[0, :PLANNED] = position
    reference.q_a_ref[1, :PLANNED] = position + 0.2
    _, horizon = resample(reference, 0.0, Grid(ocp.Ts, ocp.intervals + 1))
    solved(ocp, state, horizon, q_eq)
    parameter = ocp._last_parameter
    no_input = np.zeros(cs.NU_PROGRESS)
    rows = slice(problem.Y_PLANNED_VELOCITY, problem.Y_PLANNED_VELOCITY + PLANNED)
    velocity = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + PLANNED)

    def row(x):
        value = ocp._residual(ocp._to_qp(x), no_input, parameter)
        return np.asarray(value).reshape(-1)[rows]

    # Mid path: `resample` rests the window at both ends, so the tangent at
    # `s = 0` is zero and the stage would have nothing to be paced against.
    resting = state.copy()
    resting[cs.X_PROGRESS] = 0.5
    resting[cs.X_PROGRESS_RATE] = 0.0
    resting[velocity] = 0.0
    assert row(resting) == pytest.approx(np.zeros(PLANNED), abs=1e-9)

    # The tangent off `bspline.value`, not off the row being tested: derived
    # from the row, the zero below is true of any `dq_a - f(x) v_s` and would
    # not catch a sign, a transpose or the wrong knot vector.
    path = ocp._last_path
    step = 1e-6
    s = path.origin + 0.5
    control, knots = path.control, problem.PATH_KNOTS
    tangent = (
        bspline.value(np.array([s + step]), control, knots)
        - bspline.value(np.array([s - step]), control, knots)
    ).reshape(-1) / (2.0 * step)
    assert np.max(np.abs(tangent)) > 0.1

    # Standing still while the plan asks for a pace is what the row charges.
    standing = resting.copy()
    standing[cs.X_PROGRESS_RATE] = 0.1
    assert row(standing) == pytest.approx(-0.1 * tangent, abs=1e-6)

    on_plan = standing.copy()
    on_plan[velocity] = 0.1 * tangent
    assert row(on_plan) == pytest.approx(np.zeros(PLANNED), abs=1e-6)


def test_each_weight_block_lands_on_the_rows_it_is_named_for(parameters):
    """
    The diagonal against the residual's own row order, block by block.

    `test_the_cost_split_leaves_no_residual_row_out` cannot see this: it applies
    the same `W` on both sides, so two blocks swapped cancel and it passes.
    Since 277 that matters -- `q_a` 40 sits directly above `dq_a` 27431 on the
    boom, both five wide, a factor of 686 apart.
    """
    diagonal = np.diag(weight_matrices(parameters)[0])
    weights = parameters["weights"]
    blocks = {
        problem.Y_PLANNED_POSITION: weights["q_a"][:PLANNED],
        problem.Y_PLANNED_VELOCITY: weights["dq_a"][:PLANNED],
        problem.Y_PASSIVE_POSITION: weights["q_u"],
        problem.Y_PASSIVE_VELOCITY: weights["dq_u"],
        problem.Y_TOOL: [weights["tcp"]] * 3,
        problem.Y_PROGRESS: [weights["progress"], weights["progress_rate"]],
        problem.Y_INPUT: weights["u"][:PLANNED],
    }
    for row, expected in blocks.items():
        assert diagonal[row : row + len(expected)] == pytest.approx(expected), row
    assert diagonal[problem.Y_INPUT + PLANNED] == weights["progress_accel"]
    assert diagonal.size == problem.NY


def test_the_warm_start_is_the_last_solution_unshifted(ocp, state):
    """`Ocp.carried`: knot 0 is kept, not dropped -- a shift spends the one step."""
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]

    solution = solved(ocp, state, horizon, q_eq)
    guess = ocp.carried(solution)
    assert guess.warm(ocp.intervals)
    assert np.array_equal(guess.states, solution.states)
    assert np.array_equal(guess.inputs, solution.inputs)
    assert solved(ocp, solution.states[1], horizon, q_eq, guess).warm_started


def test_a_payload_step_drops_the_warm_start_whatever_the_caller_passes(ocp, state):
    """
    `h_eff` jumps discontinuously at a grasp; a warm plan is then for another
    model. Held in the solver, not left to the caller's `guess=None`.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]

    guess = ocp.carried(solved(ocp, state, horizon, q_eq))
    assert solved(ocp, state, horizon, q_eq, guess).warm_started

    ocp.set_payload(120.0, [0.1, 0.0, -0.4])
    assert not solved(ocp, state, horizon, q_eq, guess).warm_started
    # And only the step: the cycle after it is warm again.
    assert solved(ocp, state, horizon, q_eq, guess).warm_started
    ocp.set_payload(0.0, np.zeros(3))


def test_opening_a_solver_compiles_nothing(ocp, parameters, export_base, monkeypatch):
    """
    The contract the export step bought: a startup opens, it never builds.

    Issue 134 was the weaker version of this -- the integrators passed neither
    `generate` nor `build`, acados defaults both True, and every startup
    regenerated over an already-warm cache. Now nothing may reach a compiler at
    all: a node that can build can build the wrong thing, and this is what says
    it cannot. `ocp` is a parameter so its export exists before this runs.
    """

    def refuse(*arguments, **keywords):
        raise AssertionError("a warm cache was regenerated or rebuilt")

    description = problem.default_description().read_text()
    # Already exported by the `ocp` fixture, so this only re-points the runtime
    # at it -- another fixture may have moved the environment since.
    export_for(export_base, parameters, parameters["hydraulics"], description)

    for backend in (AcadosOcpSolver, AcadosSimSolver):
        monkeypatch.setattr(backend, "generate", staticmethod(refuse))
        monkeypatch.setattr(backend, "build", staticmethod(refuse))

    Ocp(description, parameters, parameters["hydraulics"])


def test_a_model_change_moves_the_export_key(parameters):
    """
    A description edit must not be openable by a solver exported before it.

    The integrators used to carry a second, weaker key of their own, which is
    how a stale one survived a model change the solver correctly rebuilt for.
    They ride in the same export now, so there is one key and that cannot
    happen; what is left to hold is that the key sees the description at all.
    """
    hydraulics = parameters["hydraulics"]
    description = problem.default_description().read_text()
    changed = description + "<!-- a link moved -->"
    assert export_key(parameters, hydraulics, description) != export_key(
        parameters, hydraulics, changed
    )
    # Sub-step count is generated code too; the interval alone fixes it only
    # while `T_s` is held, so the label carries both.
    assert predictor_label(0.06, 0.04) != predictor_label(0.06, 0.02)


def test_an_unexported_problem_is_refused_and_says_what_differs(parameters, tmp_path):
    """
    The whole of what replaced compiling at startup.

    Asking for a problem nothing exported must not start, and the refusal has
    to name the field -- the usual cause is an export made for another machine
    or another setting, and "differs in horizon_length" is actionable where
    "not found" is not.
    """
    hydraulics = parameters["hydraulics"]
    description = problem.default_description().read_text()
    exported = export_key(parameters, hydraulics, description)
    root = ox.solver_root(tmp_path, exported)
    root.mkdir(parents=True)
    (root / ox.MANIFEST).write_text(json.dumps({"key": exported, "sims": []}))

    moved = dict(parameters, horizon_length=parameters["horizon_length"] + 1)
    with pytest.raises(ox.StaleExport, match="horizon_length"):
        ox.manifest(tmp_path, export_key(moved, hydraulics, description), "re-export")


def test_a_non_finite_input_never_reaches_a_solve(ocp, state):
    """
    Both are the same defect: acados returns a NaN state with status 0, so a
    `FAULT_SOLVER` would be the only sign of an unchecked reference.
    """
    broken = state.copy()
    broken[cs.X_PLANNED_VELOCITY] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ocp.propagate(broken, np.zeros(cs.NU_PROGRESS))

    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    horizon.dq_a_ref[3, 1] = np.nan
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    with pytest.raises(ValueError, match="the reference"):
        solved(ocp, state, horizon, q_eq)


def test_a_swept_acados_setting_moves_the_export_key(parameters, monkeypatch):
    """
    The one thing a variant sweep rests on.

    Every setting in `SOLVER_TUNING` is compiled into the `.so`, so a variant
    the key does not see opens its predecessor's solver and measures the
    baseline again -- a null result that reads as "no difference".
    """
    hydraulics = parameters["hydraulics"]
    description = problem.default_description().read_text()
    before = export_key(parameters, hydraulics, description)

    monkeypatch.setenv(problem.TUNING_ENV, '{"hpipm_mode": "SPEED"}')
    assert problem.solver_tuning()["hpipm_mode"] == "SPEED"
    assert export_key(parameters, hydraulics, description) != before


def test_a_misspelt_knob_is_refused_rather_than_ignored(monkeypatch):
    """Silently dropped, it would re-measure the baseline under another name."""
    monkeypatch.setenv(problem.TUNING_ENV, '{"hpipm_modes": "SPEED"}')
    with pytest.raises(ValueError, match="hpipm_modes"):
        problem.solver_tuning()


def test_hpipms_memory_survives_a_warm_cycle_and_not_a_cold_one(ocp, state):
    """
    The QP keeps its factorisation exactly when the plan it came from still
    applies. Worth 5 QP iterations against 15 (scripts/bench_ocp.py), so an
    unconditional `reset_qp_solver_mem=1` is a 3x regression that nothing else
    here would notice -- tracking and sway are identical either way.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]

    dropped = []
    original = ocp.solver.reset
    ocp.solver.reset = lambda **kw: (
        dropped.append(kw["reset_qp_solver_mem"]),
        original(**kw),
    )[1]
    try:
        # no guess is a cold cycle, and so is the payload step after it
        guess = ocp.carried(solved(ocp, state, horizon, q_eq))
        assert solved(ocp, state, horizon, q_eq, guess).warm_started
        ocp.set_payload(120.0, [0.1, 0.0, -0.4])
        assert not solved(ocp, state, horizon, q_eq, guess).warm_started
        ocp.set_payload(0.0, np.zeros(3))
    finally:
        ocp.solver.reset = original
    assert dropped == [1, 0, 1]


def resting_state(ocp, parameters):
    """
    A state the machine can actually sit in: the passive pair at *its own*
    equilibrium for the pose, not at zero.

    `hold_state` seeds the passive rows at zero, which for this tool is 1.57 rad
    from where the load hangs, so a solve on it is the optimizer fighting a
    pendulum held sideways. That is a legitimate fixture for the box tests and a
    trap for anything reading `u`.
    """
    from crane_model.conventions import Tool
    from crane_model.model import CraneModel

    description = problem.default_description().read_text()
    q_a = np.array([0.785, 0.5236, 0.5236, 0.25, 0.0, 0.21])
    q_eq = np.asarray(CraneModel(description, Tool.PZS100).passive_equilibrium(q_a))
    x = np.zeros(cs.NX)
    x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED] = q_a[:PLANNED]
    x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF] = q_eq
    x[cs.X_PROGRESS_RATE] = float(
        parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(parameters)
    ocp.pin_tool(0.21)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    return x, q_eq


def test_a_machine_at_rest_asked_to_stay_there_is_commanded_nothing(ocp, parameters):
    """
    The null test. `|u| <= u_max` is not enough on its own: a bang-bang solution
    sitting exactly on the bound satisfies it, which is what the live chain was
    doing while every box assertion passed.

    Both halves matter. Zero command has to hold the state, and the solve on a
    reference that asks for no motion has to return zero -- otherwise the plan
    the machine executes is full-scale velocity on a stationary crane.
    """
    x, q_eq = resting_state(ocp, parameters)
    still = ocp.propagate(x, np.zeros(cs.NU_PROGRESS))
    rows = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + PLANNED)
    assert np.allclose(still[rows], x[rows], atol=1e-6)

    horizon = horizon_holding(ocp, x[rows])
    u_max = np.asarray(parameters["limits"]["u_max"][:PLANNED])
    solution = solved(ocp, x, horizon, q_eq)
    assert solution.outcome is not Outcome.FAILED
    assert np.max(np.abs(solution.inputs[0, :PLANNED]) / u_max) < 0.05


def test_move_suppression_is_the_sum_of_the_two_quadratics(parameters):
    """Issue 121: w_u*u^2 + w_du*(u - u_f)^2 == W*(u - yref)^2 + const, per row."""
    weight = weight_matrices(parameters)[0]
    reference = np.zeros(problem.NY)
    du = np.array([3.0, 0.0, 10.0, 1.0, 30.0])
    in_flight = np.array([0.4, -0.2, 0.1, 0.0, -0.3, 7.0])  # last: progress accel
    folded, shifted = move_suppression(weight, reference, du, in_flight)

    rows = np.arange(problem.Y_INPUT, problem.Y_INPUT + PLANNED)
    w_u = np.diag(weight)[rows]

    def split(u):
        return w_u * u**2 + du * (u - in_flight[:PLANNED]) ** 2

    def joint(u):
        return np.diag(folded)[rows] * (u - shifted[rows]) ** 2

    offset = split(np.zeros(PLANNED)) - joint(np.zeros(PLANNED))
    for u in (np.full(PLANNED, 0.5), np.linspace(-1.0, 1.0, PLANNED)):
        assert np.allclose(split(u) - joint(u), offset)
    # Every other row, the progress-accel input included, is untouched.
    others = np.setdiff1d(np.arange(problem.NY), rows)
    assert np.array_equal(np.diag(folded)[others], np.diag(weight)[others])
    assert np.array_equal(shifted[others], reference[others])


@pytest.mark.parametrize("du", [0.0, 10.0])
def test_stage_zero_prices_the_command_in_flight_only_when_asked(
    parameters, export_base, du
):
    """w_du = 0 writes stage 0 exactly as before; w_du > 0 moves only the u rows."""
    values = dict(parameters)
    values["weights"] = dict(parameters["weights"], du=[du] * cs.ACTUATED_DOF)
    description = problem.default_description().read_text()
    export_for(export_base, values, values["hydraulics"], description)
    ocp = Ocp(description, values, values["hydraulics"])
    state = hold_state(ocp)
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    horizon = horizon_holding(ocp, position)
    in_flight = np.full(cs.NU_PROGRESS, 0.3)

    solved(ocp, state, horizon, q_eq)
    plain = ocp.solver.cost_get(0, "W"), ocp.solver.cost_get(0, "yref")
    solved(ocp, state, horizon, q_eq, in_flight=in_flight)
    weight, reference = ocp.solver.cost_get(0, "W"), ocp.solver.cost_get(0, "yref")

    assert np.array_equal(plain[0], weight_matrices(values)[0])
    rows = slice(problem.Y_INPUT, problem.Y_INPUT + PLANNED)
    if du == 0.0:
        assert np.array_equal(weight, plain[0])
        assert np.array_equal(reference, plain[1])
    else:
        w_u = np.diag(plain[0])[rows]
        assert np.allclose(np.diag(weight)[rows], w_u + du)
        assert np.allclose(reference[rows], du * 0.3 / (w_u + du))
        assert np.array_equal(reference[: problem.Y_INPUT], plain[1][: problem.Y_INPUT])


@pytest.fixture(scope="module")
def c4_ocp(parameters, export_base):
    """C4: the command a state, `u[:5]` its rate."""
    values = dict(parameters, command_state=True)
    description = problem.default_description().read_text()
    export_for(export_base, values, values["hydraulics"], description)
    return Ocp(description, values, values["hydraulics"])


def c4_state(ocp, command):
    """`hold_state` with `c` appended."""
    x = np.zeros(cs.NX_COMMAND)
    x[cs.X_PLANNED_POSITION + 1] = 0.5
    x[cs.X_PROGRESS_RATE] = float(
        ocp.parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(ocp.parameters)
    x[cs.X_COMMAND :] = command
    ocp.pin_tool(0.3)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    return x


def test_c4_opens_on_its_own_dimensions(c4_ocp):
    """Opening ran `_check_dimensions` against the export; C3 offsets hold."""
    assert c4_ocp.nx == cs.NX_COMMAND == cs.X_COMMAND + PLANNED
    assert problem.residual_rows(c4_ocp.parameters) == problem.NY + PLANNED
    assert not c4_ocp.suppresses_moves


def test_c4_replays_a_command_ramp_from_its_start(c4_ocp):
    """An applied `(start, du)` leaves `c = start + Ts*du`, whatever `x` carried."""
    start = np.array([0.1, -0.05, 0.02, 0.0, 0.3])
    rate = np.zeros(cs.NU_PROGRESS)
    rate[:PLANNED] = [0.5, -0.2, 0.1, 0.3, -1.0]
    x = c4_state(c4_ocp, np.zeros(PLANNED))
    moved = c4_ocp.propagate_applied(x, [(start, rate)])
    assert np.allclose(moved[cs.X_COMMAND :], start + c4_ocp.Ts * rate[:PLANNED])
    # ZOH form: `x`'s own `c` is the start. Not to the bit: acados warm-starts
    # IRK Newton from the last solve.
    held = c4_ocp.propagate_applied(c4_state(c4_ocp, start), [rate])
    assert np.allclose(held, moved, rtol=1e-5, atol=1e-5)


def test_c4_solves_at_rest_inside_its_boxes(c4_ocp):
    x = c4_state(c4_ocp, np.zeros(PLANNED))
    position = x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    q_eq = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    solution = solved(c4_ocp, x, horizon_holding(c4_ocp, position), q_eq)
    assert solution.status == 0
    limits = c4_ocp.parameters["limits"]
    assert np.all(
        np.abs(solution.inputs[:, :PLANNED]) <= np.asarray(limits["du_max"]) + 1e-9
    )
    u_max = np.asarray(limits["u_max"][:PLANNED])
    assert np.all(np.abs(solution.states[:, cs.X_COMMAND :]) <= u_max + 1e-6)
    assert c4_ocp.cost_terms(solution, q_eq).du >= 0.0


def test_c4_converges_at_the_end_of_the_path_with_progress_still_running(c4_ocp):
    """
    Live, every goal ended in QP status 3: `v_s` carried onto a spent path.

    Progress was zero to 1e-9 here until `du_max` became a control bound (issue
    214): with a tenth of the joint command rate the optimizer buys a little path
    instead, and it creeps to ~8e-3 over the horizon. That moves nothing --
    `bspline.casadi_value`/`casadi_slope` clamp `s` to 1, so the reference is
    the goal either way, the `s` residual prices the creep at its own weight,
    and `_origined_x0` re-clamps progress next cycle so it cannot accumulate. The
    status is what this test was diagnosed by and it is still zero; the bound
    keeps a *runaway* `v_s` -- the actual fault -- failing.
    """
    x = c4_state(c4_ocp, np.zeros(PLANNED))
    position = x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    q_eq = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.PASSIVE_DOF]
    horizon = horizon_holding(c4_ocp, position)
    end = window_path(c4_ocp, horizon, origin=1.0)
    solution = solved(c4_ocp, x, horizon, q_eq, path=end)
    assert solution.status == 0
    progress = solution.states[:, cs.X_PROGRESS]
    # `s >= 0` is a hard box, held to the QP's own absolute tolerance and no
    # better: one RTI step at the sway offset weight issue 229 pays lands a few
    # 1e-4 outside it. Nothing reads these rows but `progress_advance`, which
    # `_read_solution` clips into `[0, progress_ceiling]`, so the slop cannot
    # move the origin backwards.
    assert progress.min() > -1.0e-3, progress.min()
    assert progress.max() < 0.02, progress.max()
