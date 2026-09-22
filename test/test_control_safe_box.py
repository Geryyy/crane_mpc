"""
Configured boom and arm boxes stay on the working branch of constraint 6
(issues 126 and 128, ported here by 141).

A box admitting a sign change in the transmission ratio bounds nothing. Both
ratios cross inside the declared range (arm dead point, boom four-bar
closure), so the box alone keeps constraint 6 well formed -- the exported
CasADi graph carries no guard. Test on configuration, not code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from crane_model import symbolic as cs
from crane_mpc.config import machine_limits

PACKAGE = Path(__file__).resolve().parent.parent
BOOM_ROW = 1
ARM_ROW = 2

# Where the arm's moment arm reverses (issue 126), reproduced by
# `test_the_dead_point_has_not_moved`: pivot and both cylinder attachments collinear.
ARM_DEAD_POINT = 1.8466647

# Where the boom's four-bar stops closing: d = r_13 - r_23's discriminant
# turns negative and `boom_ratio` is not finite below it (issue 128).
BOOM_CLOSURE = -1.1841609


def _ratio(name: str):
    """ds_i/dq_i as a callable, built from the shipped hydraulic constants."""
    constants = cs.load_constants()
    q = cs.ca.MX.sym("q")
    return cs.ca.Function(name, [q], [getattr(cs, name)(constants, q)])


def _shipped() -> dict:
    """The box the node poses constraint 1 on. crane_model's, via one reader."""
    return machine_limits()


def _box(row: int) -> tuple:
    limits = _shipped()
    return limits["q_a_lower"][row], limits["q_a_upper"][row]


@pytest.fixture(scope="module")
def ratio():
    return _ratio("arm_ratio")


@pytest.fixture(scope="module")
def boom_ratio():
    return _ratio("boom_ratio")


@pytest.fixture(scope="module")
def arm_box() -> tuple:
    return _box(ARM_ROW)


@pytest.fixture(scope="module")
def boom_box() -> tuple:
    return _box(BOOM_ROW)


def test_the_box_has_no_second_home():
    """
    The box used to live in the declaration and in the shipped yaml both, so
    fixing a number in one put the singularities back in reach with the suite
    green (issue 137's divergence, on a different row). Now it lives in
    crane_model and neither file may carry a row of it.
    """
    declared = yaml.safe_load((PACKAGE / "crane_mpc_parameters.yaml").read_text())
    shipped = yaml.safe_load((PACKAGE / "config" / "crane_mpc.yaml").read_text())
    for block in (
        declared["crane_mpc"]["limits"],
        shipped["crane_mpc"]["ros__parameters"]["limits"],
    ):
        assert set(block) & set(machine_limits()) == set()


def test_the_ratio_is_finite_and_single_signed_over_the_configured_arm_box(
    ratio, arm_box
):
    lower, upper = arm_box
    values = np.array([float(ratio(q)) for q in np.linspace(lower, upper, 2001)])
    assert np.all(np.isfinite(values))
    assert values.min() > 0.0, (
        f"the arm box [{lower}, {upper}] reaches the transmission dead point at "
        f"q3 = {ARM_DEAD_POINT}; constraint 6's extend and retract bounds are "
        "exchanged above it"
    )


def test_the_configured_boom_box_is_inside_the_four_bars_closing_range(
    boom_ratio, boom_box
):
    """
    The boom fails harder than the arm: below the closure the pose doesn't exist.

    Non-finite ratio makes constraint 6 non-finite, so finiteness is what
    matters; the sign assert covers the 0.033 rad above closure where the
    ratio is negative and large.
    """
    lower, upper = boom_box
    values = np.array([float(boom_ratio(q)) for q in np.linspace(lower, upper, 2001)])
    assert np.all(np.isfinite(values)), (
        f"the boom box [{lower}, {upper}] reaches past q2 = {BOOM_CLOSURE}, where "
        "the four-bar stops closing and the machine has no such pose"
    )
    assert values.min() > 0.0


def test_the_four_bar_closure_has_not_moved(boom_ratio):
    """The counterpart of `test_the_dead_point_has_not_moved`, on finiteness."""
    low, high = -1.5, 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if np.isfinite(float(boom_ratio(middle))):
            high = middle
        else:
            low = middle
    assert 0.5 * (low + high) == pytest.approx(BOOM_CLOSURE, abs=1.0e-6)


def test_the_dead_point_has_not_moved(ratio):
    """
    The box is pinned to a number, so the number it's clear of must be pinned too.

    A hydraulics constant moving the crossing toward the configured upper is
    otherwise silent until the sign test above starts failing.
    """
    low, high = 1.5, 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if float(ratio(middle)) > 0.0:
            low = middle
        else:
            high = middle
    assert 0.5 * (low + high) == pytest.approx(ARM_DEAD_POINT, abs=1.0e-6)
