"""
The path object a path-following cost is written against.

Checked against robocrane's `sspp.BSplines` when it was vendored -- knots,
control points, basis and curve agree to 0.0 -- but that library is not
reachable from here, so what is pinned is the contract the cost needs: the
planner's own paths come back exactly, the goal is on the curve, and the
expression the solver differentiates is the curve the fit produced.
"""

import casadi as ca
import numpy as np
import pytest
from crane_mpc import bspline

#: The planner's usual answer is a straight line in joint space ("joint line"),
#: which any order reproduces; the curved candidates are what needs the points.
LINE = np.linspace([0.0, 1.0, -2.0], [1.5, -0.5, 3.0], 60)


@pytest.mark.parametrize("count", (5, 12, 30))
def test_a_joint_line_survives_the_fit(count):
    control = bspline.fit(LINE, count)
    curve = bspline.value(
        np.linspace(0.0, 1.0, 97), control, bspline.knot_vector(count)
    )
    expected = np.linspace(LINE[0], LINE[-1], 97)
    assert np.abs(curve - expected).max() < 1.0e-12


def test_the_ends_are_interpolated_not_approximated():
    """A path's last point is a goal. Least squares near it is not good enough."""
    curved = np.column_stack(
        [np.linspace(0, 1, 40) ** 2, np.cos(np.linspace(0, 3, 40))]
    )
    control = bspline.fit(curved, 8)
    ends = bspline.value([0.0, 1.0], control, bspline.knot_vector(8))
    assert np.allclose(ends[0], curved[0], atol=1.0e-12)
    assert np.allclose(ends[1], curved[-1], atol=1.0e-12)


def test_the_expression_is_the_curve_and_its_slope():
    """What acados differentiates has to be what the fit produced."""
    curved = np.column_stack(
        [np.linspace(0, 1, 40) ** 3, np.sin(np.linspace(0, 4, 40))]
    )
    control = bspline.fit(curved, 10)
    knots = bspline.knot_vector(10)
    theta = ca.SX.sym("theta")
    value = ca.Function(
        "v", [theta], [bspline.casadi_value(theta, ca.SX(control), knots)]
    )
    slope = ca.Function(
        "s", [theta], [bspline.casadi_slope(theta, ca.SX(control), knots)]
    )

    probe = np.linspace(0.02, 0.98, 53)
    symbolic = np.array([np.asarray(value(place)).ravel() for place in probe])
    assert np.abs(symbolic - bspline.value(probe, control, knots)).max() < 1.0e-12

    step = 1.0e-6
    difference = np.array(
        [
            (np.asarray(value(p + step)).ravel() - np.asarray(value(p - step)).ravel())
            / (2.0 * step)
            for p in probe
        ]
    )
    analytic = np.array([np.asarray(slope(place)).ravel() for place in probe])
    assert np.abs(analytic - difference).max() < 1.0e-6


def test_de_boor_agrees_with_the_basis_sum():
    """`value` evaluates by local support; the basis sum is what it has to match."""
    rng = np.random.default_rng(0)
    control = bspline.fit(np.cumsum(rng.normal(size=(120, 5)), axis=0) / 120, 30)
    knots = bspline.knot_vector(30)
    # The ends and outside the domain included: `value` clips, and `_basis` closes
    # its last span, so the two have to agree about `theta = 1` as well.
    theta = np.concatenate([[0.0, 1.0, -0.2, 1.4], rng.random(40)])
    summed = np.array(
        [
            sum(
                control[index] * bspline._basis(place, bspline.ORDER, index, knots)
                for index in range(control.shape[0])
            )
            for place in np.clip(theta, 0.0, 1.0)
        ]
    )
    assert bspline.value(theta, control, knots) == pytest.approx(summed, abs=1e-14)


def test_a_fit_that_cannot_be_determined_is_refused_not_approximated():
    # Two samples leave every interior control point at the minimum-norm answer,
    # which in control-point space is joint space: the curve would leave the start,
    # dive toward all-joints-zero and come back to the goal.
    with pytest.raises(ValueError, match="cannot determine"):
        bspline.fit(np.linspace(np.zeros(5), np.ones(5), 2), 30)
