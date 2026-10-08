"""
The force observer: its gain, its linearisation, and what it keeps out of `x_0`.

The one claim worth a test is the issue's own -- velocity noise reaching the
force row through C3's `k` -- plus the two things that would be silently wrong
rather than loud: the gain's own arithmetic, and the scaled-to-physical
conversion of the linearisation it is derived from (`problem.state_scale` puts
the force rows at 1e4-1e5, so a conversion dropped in either direction gives a
plausible-looking gain that is wrong by that factor).
"""

import numpy as np
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from conftest import export_for
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import horizon as hz
from crane_mpc import problem
from crane_mpc.config import machine_limits
from crane_mpc.cycle import Cycle, Measurement, Timing
from crane_mpc.force_filter import ForceFilter, Noise, steady_state_gain
from crane_mpc.solver import Ocp

PLANNED = cs.K_PLANNED_DOF
PASSIVE = cs.K_PASSIVE_DOF


@pytest.fixture(scope="module")
def parameters():
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/crane_mpc.yaml") as stream:
        values = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
    values["hydraulics"] = hydraulic_limits()
    values["limits"].update(machine_limits())
    return values


@pytest.fixture(scope="module")
def ocp(parameters, export_base):
    description = problem.default_description().read_text()
    export_for(export_base, parameters, parameters["hydraulics"], description)
    return Ocp(description, parameters, parameters["hydraulics"])


def resting_state(ocp):
    """A pose the machine holds, with the force row at what holds it."""
    x = np.zeros(ocp.nx)
    x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED] = [
        0.785,
        0.3,
        1.4,
        0.2,
        0.0,
    ]
    ocp.pin_tool(0.0)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    return x


def test_the_steady_state_gain_is_the_analytic_one_on_a_scalar_random_walk():
    """
    One state, one measurement: `P = qP + qr` has a closed form, and `K = P/(P+r)`.

    The whole filter rests on this recursion, and a sweep that stopped one
    iteration short, or a posterior/prior mix-up, still returns a plausible
    number between 0 and 1.
    """
    process, sensor = 4.0e-4, 1.0e-2
    covariance = 0.5 * (process + np.sqrt(process**2 + 4.0 * process * sensor))
    gain = steady_state_gain(np.ones((1, 1)), np.ones((1, 1)), [[process]], [[sensor]])
    assert gain[0, 0] == pytest.approx(covariance / (covariance + sensor), rel=1e-9)


def test_a_perfect_sensor_takes_the_whole_innovation_and_a_useless_one_takes_none():
    """The two ends of the trade, so a sign or an inversion cannot hide."""
    transition = np.array([[1.0, 0.06], [0.0, 1.0]])
    observation = np.eye(2)[:1]
    process = np.diag([1.0e-6, 1.0e-6])
    assert steady_state_gain(transition, observation, process, [[1.0e-18]])[
        0, 0
    ] == pytest.approx(1.0, abs=1e-6)
    assert steady_state_gain(transition, observation, process, [[1.0e12]])[
        0, 0
    ] == pytest.approx(0.0, abs=1e-6)


def test_the_linearisation_agrees_with_the_step_it_is_taken_from(ocp):
    """
    `axis_transition` against a finite difference of `propagate`, in N m and rad.

    This is the test that catches a missing `_x_scale` conversion: the force
    column would come back 1e4-1e5 times too small and the gain would still
    look like a gain. Central differences, each row perturbed on its own, with
    a step sized to the row's own units.
    """
    x = resting_state(ocp)
    u = np.zeros(cs.NU_PROGRESS)
    blocks = ocp.axis_transition(x, u)
    rows = [cs.X_PLANNED_VELOCITY, cs.X_ACTUATED_FORCE]
    # The force column's step is a fraction of that axis' own force allowance:
    # a step of newton-metres that is nothing on the boom is the rotator's whole
    # working range, and the rotator is where a flat step reads as curvature.
    allowance = problem.state_scale(ocp.parameters, ocp.scale)[
        cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED
    ]
    for axis in range(PLANNED):
        steps = (1.0e-5, 1.0e-4 * allowance[axis])
        for column, (row, step) in enumerate(zip(rows, steps)):
            up, down = x.copy(), x.copy()
            up[row + axis] += step
            down[row + axis] -= step
            difference = (ocp.propagate(up, u) - ocp.propagate(down, u)) / (2.0 * step)
            expected = [difference[entry + axis] for entry in rows]
            assert blocks[axis][:, column] == pytest.approx(
                expected, rel=2.0e-2, abs=1.0e-7
            ), f"axis {axis}, column {column}"


def test_velocity_noise_reaches_the_force_row_open_loop_and_the_filter_holds_it_back(
    ocp, parameters
):
    """
    Issue 227's own claim, as a number: how much force error the noise makes.

    Each estimator is run twice over the same trajectory, once on a clean
    measurement and once with a white draw added, and scored on the *difference*
    -- which is the force error the noise produced and nothing else. Not against
    a standing truth: a pose held by `static_hold_force` is no equilibrium of
    the whole model (the passive pair swings, and only the planned rows are
    rewritten each cycle), so the force wanders thousands of N m on a noiseless
    run and that transient swamps the quantity under test.

    Scored on the two luffing rows, the ones whose force is worth thousands of
    N m. 5x is the issue's own acceptance number; the shipped covariances read
    11-115x over a captured move, and the tuning that cancels the filter --
    `tau_a` counting the velocity coupling the transition already carries --
    read 0.9-2.1x there.
    """
    u = np.zeros(cs.NU_PROGRESS)
    x = resting_state(ocp)
    rows = np.array([cs.X_PLANNED_VELOCITY, cs.X_ACTUATED_FORCE])
    index = (rows[:, None] + np.arange(PLANNED)[None, :]).T
    position = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + PLANNED)
    velocity = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + PLANNED)
    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + PLANNED)

    rng = np.random.default_rng(227)
    # Position and velocity noise at the ratio `control_recordings` carries them.
    spread = np.array([1.0e-5, 1.0e-3])
    draw = rng.normal(0.0, 1.0, (60, PLANNED, 2)) * spread

    def run(filtered, noisy):
        carried, trace = x.copy(), []
        for cycle in range(draw.shape[0]):
            measured_q = x[position] + (draw[cycle, :, 0] if noisy else 0.0)
            measured_dq = x[velocity] + (draw[cycle, :, 1] if noisy else 0.0)
            state = carried.copy()
            state[position] = measured_q
            state[velocity] = measured_dq
            if filtered is not None:
                if not filtered.armed:
                    filtered.derive(ocp.axis_transition(carried, u))
                state[index] = filtered.correct(carried[index], measured_dq)
            trace.append(state[force].copy())
            carried = ocp.propagate(state, u)
        return np.asarray(trace)

    wander = {}
    for name in ("open_loop", "filtered"):
        made = (
            None
            if name == "open_loop"
            else ForceFilter(Noise.from_parameters(parameters["force_filter"]))
        )
        clean = run(made, noisy=False)
        if made is not None:
            made.forget()
        wander[name] = np.sqrt(np.mean((run(made, noisy=True) - clean) ** 2, axis=0))

    luffing = [1, 2]
    assert np.all(wander["open_loop"][luffing] > 50.0), wander["open_loop"]
    assert np.all(wander["filtered"][luffing] < 0.2 * wander["open_loop"][luffing]), (
        wander
    )


def test_nothing_is_corrected_before_a_gain_is_derived(parameters):
    """
    A gain belongs to the pose it was solved at, so a reset must leave none.

    Correcting against a dropped gain is the one way this could silently keep
    using a payload's or a pose's old numbers.
    """
    filtered = ForceFilter(Noise.from_parameters(parameters["force_filter"]))
    assert not filtered.armed
    with pytest.raises(RuntimeError):
        filtered.correct(np.zeros((PLANNED, 2)), np.zeros(6))

    filtered.derive(np.tile(np.eye(2), (PLANNED, 1, 1)))
    assert filtered.armed
    filtered.forget()
    assert not filtered.armed


def measurement(x) -> Measurement:
    return Measurement(
        q_a=np.concatenate(
            [x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED], [0.0]]
        ),
        dq_a=np.concatenate(
            [x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + PLANNED], [0.0]]
        ),
        q_u=x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + PASSIVE],
        dq_u=x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + PASSIVE],
        actuated_age=0.0,
        passive_age=0.0,
    )


def built_cycle(ocp) -> Cycle:
    made = Cycle(Timing(ocp.Ts, ocp.Ts), hz.Grid(ocp.Ts, ocp.intervals + 1), "active")
    made.ocp = ocp
    return made


def test_switching_the_filter_off_carries_the_force_the_way_it_did_before(
    ocp, parameters
):
    """
    `enabled: false` is the rollback, so it has to be the pre-227 arithmetic.

    Two cycles, because the first one seeds the force from `h_eff` either way
    and the second is the first that could differ: off, `x_0`'s rows are the
    measurement and the carried force untouched; on, the velocity and the force
    move and **the position row still does not**. That last one is not cosmetic
    -- `position_box` pins the hard box on `x_0`'s own `q_a`, and 226 measured
    the QP tolerating 3e-4 rad of movement in that bound.
    """
    x = resting_state(ocp)
    moved = measurement(x)
    moved.dq_a[:PLANNED] = 2.0e-2

    states = {}
    for enabled in (False, True):
        ocp.parameters["force_filter"] = dict(
            parameters["force_filter"], enabled=enabled
        )
        one = built_cycle(ocp)
        assert one.read_state(measurement(x), 1.0) is None
        one.carried = ocp.propagate(one.measured, np.zeros(cs.NU_PROGRESS))
        assert one.read_state(moved, 1.0) is None
        states[enabled] = one.measured.copy()
    ocp.parameters["force_filter"] = parameters["force_filter"]

    planned = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + PLANNED)
    position = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + PLANNED)
    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + PLANNED)
    assert states[False][planned] == pytest.approx(moved.dq_a[:PLANNED])
    assert not np.allclose(states[True][planned], moved.dq_a[:PLANNED])
    assert not np.allclose(states[True][force], states[False][force])
    for enabled in (False, True):
        assert states[enabled][position] == pytest.approx(moved.q_a[:PLANNED])
