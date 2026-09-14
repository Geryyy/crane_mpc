"""The solver wrapper: what it predicts, and what it plans."""

import numpy as np
import pytest
import yaml
from acados_template import AcadosOcpSolver, AcadosSimSolver
from ament_index_python.packages import get_package_share_directory
from crane_model import symbolic as cs
from crane_mpc import problem
from crane_mpc.horizon import Grid, Knots, resample
from crane_mpc.solver import (
    Ocp,
    Outcome,
    predictor_cache,
    solver_cache,
    solver_signature,
)

PLANNED = cs.K_PLANNED_DOF


def read_parameters(name):
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/{name}") as stream:
        return yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]


@pytest.fixture(scope="module")
def parameters():
    values = read_parameters("crane_mpc.yaml")
    values.update(read_parameters("hydraulic_limits.yaml"))
    return values


@pytest.fixture(scope="module")
def ocp(parameters):
    return Ocp(
        problem.default_description().read_text(),
        parameters,
        parameters["hydraulics"],
    )


@pytest.fixture
def state(ocp):
    x = np.zeros(cs.NX)
    x[cs.X_PLANNED_POSITION + 1] = 0.5
    x[cs.X_PROGRESS_RATE] = 1.0
    ocp.pin_tool(0.3)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    return x


def horizon_holding(ocp, position):
    """A reference that asks the machine to stay where it is."""
    reference = Knots.zeros(2)
    reference.t[:] = [0.0, ocp.Ts * ocp.intervals]
    reference.q_a_ref[:, :PLANNED] = position
    _, horizon = resample(reference, 0.0, Grid(ocp.Ts, ocp.intervals + 1))
    return horizon


def test_the_predictor_carries_the_command_that_is_in_flight(ocp, state):
    """
    Issue 125. Under C3 the input reaches `ddq` only through the command-lag and
    force rows, so a predictor that drops them returns the same state for every
    command and the delay compensation is a coast at a frozen force.
    """
    still = ocp.propagate(state, np.zeros(cs.NU_PROGRESS))
    command = np.zeros(cs.NU_PROGRESS)
    command[0] = 0.4
    driven = ocp.propagate(state, command)

    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + PLANNED)
    assert not np.allclose(driven[force], still[force])
    assert not np.allclose(driven[cs.X_PLANNED_VELOCITY], still[cs.X_PLANNED_VELOCITY])
    # And it is a prediction, not a reset: the pose barely moves in 60 ms.
    assert np.allclose(
        driven[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED],
        state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED],
        atol=0.02,
    )


def test_a_solve_keeps_its_plan_inside_the_boxes_it_was_given(ocp, state):
    position = state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED]
    horizon = horizon_holding(ocp, position)
    q_eq = state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]

    solution = ocp.solve(state, horizon, q_eq)
    assert solution.outcome is not Outcome.FAILED
    assert solution.status == 0
    u_max = np.asarray(ocp.parameters["limits"]["u_max"][:PLANNED])
    assert np.all(np.abs(solution.inputs[:, :PLANNED]) <= u_max + 1e-9)
    # Constraint 1's initial condition: stage zero is the state it was given,
    # every row of it -- the actuator states are pinned there too.
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
    terms = ocp.cost_terms(solution, horizon, q_eq)
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
    # `s` restarts every cycle, so the shifted plan is re-origined by what was
    # spent rather than carried forward with the old origin.
    assert guess.states[0, cs.X_PROGRESS] <= solution.states[1, cs.X_PROGRESS] + 1e-12
    assert ocp.solve(solution.states[1], horizon, q_eq, guess).warm_started


def test_a_payload_step_drops_the_warm_start_whatever_the_caller_passes(ocp, state):
    """
    `h_eff` moves discontinuously at a grasp, so the plan that was warm was a
    plan for another model. Held in the solver rather than trusted to the caller
    passing `guess=None` (`ocp_solver.cpp:1396-1400`).
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


def test_a_second_construction_over_a_warm_cache_compiles_nothing(
    ocp, parameters, monkeypatch
):
    """
    Issue 134. The two integrators beside the solver passed neither `generate`
    nor `build`, and acados defaults both to True, so every startup ran CasADi
    code generation and `make` over a cache that was already warm.
    """

    def refuse(*arguments, **keywords):
        raise AssertionError("a warm cache was regenerated or rebuilt")

    for backend in (AcadosOcpSolver, AcadosSimSolver):
        monkeypatch.setattr(backend, "generate", staticmethod(refuse))
        monkeypatch.setattr(backend, "build", staticmethod(refuse))

    Ocp(
        problem.default_description().read_text(),
        parameters,
        parameters["hydraulics"],
    )


def test_the_predictor_is_rebuilt_for_the_model_changes_the_solver_is(parameters):
    """
    The predictor integrates the same model, so a second and weaker key is how a
    stale one survives a model change the solver correctly rebuilds for.
    """
    hydraulics = parameters["hydraulics"]
    description = problem.default_description().read_text()
    changed = description + "<!-- a link moved -->"
    assert solver_cache(parameters, hydraulics, description) != solver_cache(
        parameters, hydraulics, changed
    )

    signature = solver_signature(parameters, hydraulics, description)
    warm = predictor_cache(signature, 0.06, 0.04)
    assert warm != predictor_cache(
        solver_signature(parameters, hydraulics, changed), 0.06, 0.04
    )
    # The sub-step count is generated code too, and the interval alone fixes it
    # only while `T_s` is held.
    assert warm != predictor_cache(signature, 0.06, 0.02)


def test_a_non_finite_input_never_reaches_a_solve(ocp, state):
    """
    Both are the same defect: acados answers a NaN state with status 0, so what
    comes back is a `FAULT_SOLVER` for a measurement or a reference nobody
    checked. `ddq_a_ref` is the sharp one -- the resample's second derivative
    carries `1/dt^2` in the incoming knot spacing, which nothing bounds below.
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
