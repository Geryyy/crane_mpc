"""
The configured boom and arm boxes stay on the working branch of constraint 6
(issues 126 and 128, ported here by 141).

Constraint 6 reaches force space by dividing the joint torque by the diagonal
transmission ratio, so a box that admits a sign change admits a row that bounds
nothing: `wiki/mpc.md` §3 writes the extend and retract limits as acados' two
separate sides, and a negative ratio exchanges them. Both ratios leave the
working branch inside the *declared* range -- the arm at a dead point, the boom
where its four-bar stops closing and the pose ceases to exist at all -- so what
keeps the row well formed is the box and nothing else; the exported CasADi graph
carries no guard. Hence a test on the configuration rather than on the code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from crane_model import symbolic as cs

PACKAGE = Path(__file__).resolve().parent.parent
BOOM_ROW = 1
ARM_ROW = 2

# Where the arm's moment arm reverses, measured in issue 126 and reproduced by
# `test_the_dead_point_has_not_moved`. Pivot and both cylinder attachments are
# collinear there.
ARM_DEAD_POINT = 1.8466647

# Where the boom's four-bar stops closing: d = r_13 - r_23, the discriminant of
# `wiki/hydraulics.md` §2.2 turns negative and `boom_ratio` is not finite below
# it. Issue 128; `wiki/hydraulics.md` §2.2 rounds it to -1.184.
BOOM_CLOSURE = -1.1841609


def _ratio(name: str):
    """ds_i/dq_i as a callable, built from the shipped hydraulic constants."""
    constants = cs.load_constants()
    q = cs.ca.MX.sym("q")
    return cs.ca.Function(name, [q], [getattr(cs, name)(constants, q)])


def _shipped() -> dict:
    with open(PACKAGE / "config" / "crane_mpc.yaml") as stream:
        return yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]["limits"]


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


def test_the_declared_default_box_is_the_shipped_one():
    """
    The box has two homes and the tests above read one of them.

    `crane_mpc_parameters.yaml`'s default is what runs when the config file is not
    passed, so a number fixed in one place only puts the singularities back in
    reach with the suite green -- issue 137's divergence on a different row.
    """
    with open(PACKAGE / "crane_mpc_parameters.yaml") as stream:
        declared = yaml.safe_load(stream)["crane_mpc"]["limits"]
    shipped = _shipped()
    for row in ("q_a_lower", "q_a_upper"):
        assert declared[row]["default_value"] == shipped[row], row


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
    The boom fails harder than the arm: below the closure the pose does not exist.

    A non-finite ratio makes constraint 6 non-finite, so the finiteness assert is
    the one that matters; the sign assert covers the 0.033 rad above the closure
    where the ratio is negative and large.
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
    The box is pinned to a number, so the number it is clear of must be pinned too.

    A hydraulics constant that moves the crossing down toward the configured
    upper is otherwise silent until the sign test above starts failing.
    """
    low, high = 1.5, 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if float(ratio(middle)) > 0.0:
            low = middle
        else:
            high = middle
    assert 0.5 * (low + high) == pytest.approx(ARM_DEAD_POINT, abs=1.0e-6)
