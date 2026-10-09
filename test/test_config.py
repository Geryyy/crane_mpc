"""What the node refuses to start on, by class of refusal."""

import copy

import numpy as np
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from crane_model import hydraulic_limits
from crane_mpc.config import (
    MpcConfigError,
    check_payload,
    check_settings,
    machine_limits,
)

#: The sway period `du_max` is sized against, s. A design value of order two
#: seconds; `wiki/mpc.md` §4 says the slowest case has never been computed.
SWAY_PERIOD_S = 2.34
#: How far inside one sway period the command may traverse its clamp. 4 from 1
#: (issue 246): the hunt 1 was picked to stop does not reproduce at any setting
#: in today's harness, and 4x buys peak sway and hold quiet for nothing.
DU_RELAXATION = 4.0


def read_parameters(name):
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/{name}") as stream:
        return yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]


@pytest.fixture
def shipped():
    parameters = read_parameters("crane_mpc.yaml")
    parameters.update({"hydraulics": hydraulic_limits()})
    # the shipped yaml carries neither; the node resolves both off crane_model
    parameters["limits"].update(machine_limits())
    return copy.deepcopy(parameters)


def refusal(parameters) -> str:
    with pytest.raises(MpcConfigError) as raised:
        check_settings(parameters, parameters["hydraulics"])
    return str(raised.value)


def test_the_shipped_configuration_is_accepted(shipped):
    """A guard that rejects what the repo ships is a bug in the guard."""
    check_settings(shipped, shipped["hydraulics"])
    # Guard on weights is non-negativity. The sway offset weight is payable only
    # because `q_eq` is a per-stage table (issue 229); zero is what it shipped at
    # while the centre was frozen, and the guard covers either.
    assert list(shipped["weights"]["q_u"]) == [6400.0, 6400.0]


def test_the_command_may_not_slew_faster_than_the_sway_it_shapes(shipped):
    """
    `du_max` is a control bound, not the valve's slew limit.

    Nothing in `check_settings` ties the two -- the guard is positivity -- so
    this is the only thing holding the rule the five numbers were picked by. At
    the valve's own limit (`[5.2, 1.7, 1.8, 3.5, 14.0]`, 0.058 per 10 ms
    normalised) `u` crosses its whole clamp inside one 60 ms knot and the
    single-iteration SQP reversed the command every other cycle, which shook the
    empty pendulum for the length of the hold (issue 214).

    **The time it is divided by is the sway period, not the horizon.** The two
    were the same 2.34 s when the five numbers were fitted and the horizon has
    since moved to 3.54 s (issue 223), so this used to read `horizon_length`.
    What the rule is about is the mode the command must not excite; retrying it
    to the longer horizon was measured and is a small loss everywhere -- max
    solve 25.6 -> 40.8 ms, non-zero status 1.7 -> 2.2 %, late sway rate and
    final error marginally worse, peak sway marginally better.

    `DU_RELAXATION` is the quarter-period the command is now allowed instead of
    the whole one (issue 246). What still holds is the *form*: the five numbers
    are one rule about the sway mode, not five independent trims, and nothing
    in `check_settings` ties them to anything.
    """
    command = np.asarray(shipped["limits"]["u_max"][: len(shipped["limits"]["du_max"])])
    assert np.allclose(
        shipped["limits"]["du_max"],
        command * DU_RELAXATION / SWAY_PERIOD_S,
        rtol=5.0e-3,
    )
    # Still clear of the valve's own ceiling, which is what the rule is against.
    assert np.all(np.asarray(shipped["limits"]["du_max"]) < [5.2, 1.7, 1.8, 3.5, 14.0])


def test_a_scalar_that_is_not_a_number_is_refused(shipped):
    shipped["Ts"] = 0.0
    assert "control cycle" in refusal(shipped)

    shipped["Ts"] = 0.04
    shipped["solve_budget"] = float("nan")
    assert "solve_budget = nan" in refusal(shipped)

    shipped["solve_budget"] = 0.03
    shipped["levenberg_marquardt"] = -1.0e-6
    assert "subtracts from it" in refusal(shipped)


def test_a_delay_that_is_not_the_step_is_refused_here_and_only_here(shipped):
    """
    The one timing invariant. It used to live in `node.configure` alone, so
    `Ocp` and `Cycle` took any delay and `--set sensor_to_valve_delay=0.04` ran
    silently wrong everywhere but the node.
    """
    shipped["sensor_to_valve_delay"] = 0.04
    assert "must equal Ts" in refusal(shipped)

    # Off the JTC's 100 Hz tick: a grid the horizon's knots cannot be sampled on.
    shipped["Ts"] = shipped["sensor_to_valve_delay"] = 0.065
    assert "JTC ticks" in refusal(shipped)

    shipped["Ts"] = shipped["sensor_to_valve_delay"] = 0.04
    check_settings(shipped, shipped["hydraulics"])


def test_a_negative_weight_is_refused_and_names_the_entry(shipped):
    shipped["weights"]["dq_a"][2] = -0.4
    message = refusal(shipped)
    assert "dq_a[2] = -0.4" in message
    assert "J' W J" in message


def test_the_only_price_on_getting_anywhere_may_not_be_zero(shipped):
    """
    Checked strictly, unlike other weights: the progress state is the path
    parameter, so at zero nothing makes the machine travel the path it is on.

    Its rate may be zero -- that row is damping, and the velocity and flow rows
    still bound how fast the path may be spent.
    """
    shipped["weights"]["progress"] = 0.0
    message = refusal(shipped)
    assert "progress = 0" in message
    assert "never arrives" in message

    shipped["weights"]["progress"] = 8.0
    shipped["weights"]["progress_rate"] = 0.0
    check_settings(shipped, shipped["hydraulics"])


def test_a_control_safe_range_that_is_not_a_range_is_refused(shipped):
    limits = shipped["limits"]
    limits["q_a_upper"][1] = limits["q_a_lower"][1]
    assert "empty control-safe position range" in refusal(shipped)

    limits["q_a_upper"][1] = 1.563
    limits["q_a_margin"][1] = 0.5 * (limits["q_a_upper"][1] - limits["q_a_lower"][1])
    message = refusal(shipped)
    assert "q_a_margin[1]" in message
    assert "a tightening, not a replacement" in message


def test_a_bound_that_is_not_a_bound_is_refused(shipped):
    shipped["limits"]["dq_a_max"][0] = 0.0
    message = refusal(shipped)
    assert "dq_a_max[0] = 0" in message
    assert "silent stub" in message


def test_a_progress_ceiling_under_the_plans_own_pace_is_refused(shipped):
    """
    Headroom is a multiple of nominal, so the floor is one whatever the grid is.

    The rate it stands for is not: `Ocp.progress_rate_max` reads 0.427/s at the
    shipped grid and 12.5/s at the four-knot one, off this same 1.5.
    """
    shipped["limits"]["progress_rate_headroom"] = 0.9
    assert "below one" in refusal(shipped)

    shipped["limits"]["progress_rate_headroom"] = 1.0
    check_settings(shipped, shipped["hydraulics"])


def test_a_vector_the_problem_is_not_posed_on_is_refused(shipped):
    shipped["limits"]["u_max"] = shipped["limits"]["u_max"][:4]
    assert "limits.u_max carries 4 entries" in refusal(shipped)


def test_a_soft_constraint_priced_at_zero_is_refused(shipped):
    shipped["slack"]["pump_flow"] = 0.0
    assert "absent one" in refusal(shipped)


def test_a_hydraulic_number_that_is_not_physical_is_refused(shipped):
    shipped["hydraulics"]["pump_flow_planning_factor"] = 1.2
    assert "outside (0, 1]" in refusal(shipped)

    shipped["hydraulics"]["pump_flow_planning_factor"] = 0.95
    shipped["hydraulics"]["system_pressure_pa"] = 0.0
    assert "relief pressure" in refusal(shipped)


def test_a_payload_the_dynamics_cannot_carry_is_refused():
    # Declared empty payload is an empty gripper, and is accepted.
    check_payload(0.0, np.zeros(3))
    check_payload(120.0, [0.1, 0.0, -0.4])

    with pytest.raises(MpcConfigError, match="not a light load"):
        check_payload(-1.0, np.zeros(3))
    with pytest.raises(MpcConfigError, match="moment arm"):
        check_payload(120.0, [0.1, float("nan"), 0.0])
    with pytest.raises(MpcConfigError, match="three finite"):
        check_payload(120.0, [0.1, 0.0])
