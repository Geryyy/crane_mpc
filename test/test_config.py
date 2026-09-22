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
    # Guard on weights is non-negativity, so shipped zero sway offset weight
    # passes (issue 137 moved the declared default to zero; guard stays >= 0).
    assert list(shipped["weights"]["q_u"]) == [0.0, 0.0]


def test_a_scalar_that_is_not_a_number_is_refused(shipped):
    shipped["Ts"] = 0.0
    assert "control cycle" in refusal(shipped)

    shipped["Ts"] = 0.04
    shipped["solve_budget"] = float("nan")
    assert "solve_budget = nan" in refusal(shipped)

    shipped["solve_budget"] = 0.03
    shipped["levenberg_marquardt"] = -1.0e-6
    assert "subtracts from it" in refusal(shipped)


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
