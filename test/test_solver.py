"""The solver wrapper: what it predicts, and what it plans."""

import json

import crane_ocp_export as ox
import numpy as np
import pytest
import yaml
from acados_template import AcadosOcpSolver, AcadosSimSolver
from ament_index_python.packages import get_package_share_directory
from conftest import export_for
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import problem
from crane_mpc.config import machine_limits
from crane_mpc.horizon import Grid, Knots, resample
from crane_mpc.solver import (
    Ocp,
    Outcome,
    export_key,
    move_suppression,
    predictor_label,
    weight_matrices,
)

PLANNED = cs.K_PLANNED_DOF


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
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    solution = ocp.solve(state, horizon, q_eq)
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


def test_the_cost_split_names_where_the_plan_spent(ocp, state):
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    # Ask for a pose the machine is not in: the tracking row has to carry it.
    horizon.q_a_ref[:, 0] += 0.4
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    solution = ocp.solve(state, horizon, q_eq)
    terms = ocp.cost_terms(solution, q_eq)
    assert terms.q_a > 0.0
    assert all(np.isfinite(value) and value >= 0.0 for value in terms.__dict__.values())


def test_the_warm_start_is_the_last_horizon_shifted(ocp, state):
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    solution = ocp.solve(state, horizon, q_eq)
    guess = ocp.shifted(solution)
    assert guess.warm(ocp.intervals)
    rows = [row for row in range(cs.NX) if row != cs.X_PROGRESS]
    assert np.allclose(guess.states[0][rows], solution.states[1][rows])
    # `s` restarts every cycle: shifted plan re-origins by spend, not the old origin.
    assert guess.states[0, cs.X_PROGRESS] <= solution.states[1, cs.X_PROGRESS] + 1e-12
    assert ocp.solve(solution.states[1], horizon, q_eq, guess).warm_started


def test_a_payload_step_drops_the_warm_start_whatever_the_caller_passes(ocp, state):
    """
    `h_eff` jumps discontinuously at a grasp; a warm plan is then for another
    model. Held in the solver, not left to caller's `guess=None`
    (`ocp_solver.cpp:1396-1400`).
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    guess = ocp.shifted(ocp.solve(state, horizon, q_eq))
    assert ocp.solve(state, horizon, q_eq, guess).warm_started

    ocp.set_payload(120.0, [0.1, 0.0, -0.4])
    assert not ocp.solve(state, horizon, q_eq, guess).warm_started
    # And only the step: the cycle after it is warm again.
    assert ocp.solve(state, horizon, q_eq, guess).warm_started
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
    Both are the same defect: acados returns a NaN state with status 0, so
    `FAULT_SOLVER` means an unchecked reference. `ddq_a_ref` is the sharp one:
    resample's second derivative carries unbounded `1/dt^2` of knot spacing.
    """
    broken = state.copy()
    broken[cs.X_PLANNED_VELOCITY] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ocp.propagate(broken, np.zeros(cs.NU_PROGRESS))

    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    horizon.ddq_a_ref[3, 1] = np.nan
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    with pytest.raises(ValueError, match="curvature"):
        ocp.solve(state, horizon, q_eq)


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
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    dropped = []
    original = ocp.solver.reset
    ocp.solver.reset = lambda **kw: (
        dropped.append(kw["reset_qp_solver_mem"]),
        original(**kw),
    )[1]
    try:
        # no guess is a cold cycle, and so is the payload step after it
        guess = ocp.shifted(ocp.solve(state, horizon, q_eq))
        assert ocp.solve(state, horizon, q_eq, guess).warm_started
        ocp.set_payload(120.0, [0.1, 0.0, -0.4])
        assert not ocp.solve(state, horizon, q_eq, guess).warm_started
        ocp.set_payload(0.0, np.zeros(3))
    finally:
        ocp.solver.reset = original
    assert dropped == [1, 0, 1]


def test_the_split_is_the_same_computation_as_the_whole_solve(ocp, state):
    """
    Preparation plus feedback is one RTI step cut in two, not a second method.

    The tolerance is measured here rather than picked: `reset` does not restore
    a deterministic interior point, so two "cold" combined solves land a QP
    iteration apart and disagree by ~2e-2 on the inputs depending on what the
    solver did before them. The split has to sit no further from a combined
    solve than two combined solves sit from each other -- measured, it is about
    three orders closer.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    first = ocp.solve(state, horizon, q_eq)
    reference = ocp.solve(state, horizon, q_eq)
    spread = np.max(np.abs(first.inputs - reference.inputs))

    ocp.prepare(state, horizon, q_eq)
    split = ocp.feedback(state)

    assert split.status == reference.status
    assert np.max(np.abs(split.inputs - reference.inputs)) <= max(spread, 1e-3)
    # The budget is about latency, so the preparation is reported beside the
    # solve time and never inside it.
    assert split.preparation_time_s > 0.0
    assert reference.preparation_time_s == 0.0


def test_the_feedback_phase_reads_the_state_it_is_given(ocp, state):
    """
    The measurement still reaches the QP after the linearisation is standing.

    This is the initial value embedding, and it is the one thing the split must
    not lose: if the feedback phase answered on the state the preparation
    assumed, the controller would be open loop for a cycle and nothing else in
    this package would notice.
    """
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    moved = state.copy()
    moved[cs.X_PLANNED_VELOCITY] += 0.05

    ocp.prepare(state, horizon, q_eq)
    on_predicted = ocp.feedback(state)
    ocp.prepare(state, horizon, q_eq)
    on_measured = ocp.feedback(moved)

    # Same preparation both times, so any difference is the measurement's.
    assert np.allclose(on_measured.states[0], moved, atol=1e-6)
    assert not np.allclose(on_measured.inputs, on_predicted.inputs, atol=1e-6)


def test_the_feedback_phase_refuses_to_run_on_nothing(ocp, state):
    """A cycle whose preparation was invalidated has to take `solve`, not guess."""
    with pytest.raises(RuntimeError, match="without a preparation"):
        ocp.feedback(state)


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
    x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = q_eq
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
    solution = ocp.solve(x, horizon, q_eq)
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
    values["weights"] = dict(parameters["weights"], du=[du] * cs.K_ACTUATED_DOF)
    description = problem.default_description().read_text()
    export_for(export_base, values, values["hydraulics"], description)
    ocp = Ocp(description, values, values["hydraulics"])
    state = hold_state(ocp)
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    horizon = horizon_holding(ocp, position)
    in_flight = np.full(cs.NU_PROGRESS, 0.3)

    ocp.solve(state, horizon, q_eq)
    plain = ocp.solver.cost_get(0, "W"), ocp.solver.cost_get(0, "yref")
    ocp.solve(state, horizon, q_eq, in_flight=in_flight)
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
    q_eq = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    solution = c4_ocp.solve(x, horizon_holding(c4_ocp, position), q_eq)
    assert solution.status == 0
    limits = c4_ocp.parameters["limits"]
    assert np.all(
        np.abs(solution.inputs[:, :PLANNED]) <= np.asarray(limits["du_max"]) + 1e-9
    )
    u_max = np.asarray(limits["u_max"][:PLANNED])
    assert np.all(np.abs(solution.states[:, cs.X_COMMAND :]) <= u_max + 1e-6)
    assert c4_ocp.cost_terms(solution, q_eq).du >= 0.0
