"""
Clamped B-spline path, the object a path-following cost is written against.

The formulation and this basis are robocrane's (`sspp/sspp/BSplines.py`, the
`pfc`/`tspfc` controllers), validated on a KUKA. Vendored rather than imported:
`sspp` sits outside this workspace and a deployed package may not reach it.

The path is geometry, `c(theta)` on `theta` in [0, 1], with no time in it --
`crane_planning` already plans that way and carries its own `sigma`. What a
solver needs from it is the value and the slope at a `theta` it is free to
choose, which is what `casadi_value`/`casadi_slope` give as expression graphs.
"""

from __future__ import annotations

from functools import lru_cache

import casadi as ca
import numpy as np

#: Cubic. Two continuous derivatives is what the cost differentiates through.
ORDER = 3


def knot_vector(control_points: int, order: int = ORDER) -> np.ndarray:
    """Clamped uniform knots on [0, 1]: the ends interpolate their control point."""
    interior = np.linspace(0.0, 1.0, control_points + order + 1 - 2 * order)
    return np.concatenate([np.zeros(order), interior, np.ones(order)])


def _closes(index: int, knots: np.ndarray) -> bool:
    """
    Whether span `index` is the last non-empty one, so it owns `theta = 1`.

    Spans are half-open or the basis would sum past one; the last one has to be
    closed or the curve has no value at its own end. `sspp` says this outside the
    basis instead (`bspline` returns the last control point for theta >= 1),
    which leaves its symbolic basis reading 1 on every repeated end knot.
    """
    return knots[index] < knots[index + 1] == knots[-1]


def _basis(theta: float, order: int, index: int, knots: np.ndarray) -> float:
    """Cox-de Boor, numerically. `casadi_basis` is the same recursion symbolically."""
    if order == 0:
        inside = knots[index] <= theta < knots[index + 1]
        return 1.0 if inside or (_closes(index, knots) and theta >= knots[-1]) else 0.0
    left, right = 0.0, 0.0
    if knots[index + order] != knots[index]:
        left = (theta - knots[index]) / (knots[index + order] - knots[index])
        left *= _basis(theta, order - 1, index, knots)
    if knots[index + order + 1] != knots[index + 1]:
        right = (knots[index + order + 1] - theta) / (
            knots[index + order + 1] - knots[index + 1]
        )
        right *= _basis(theta, order - 1, index + 1, knots)
    return left + right


def casadi_basis(theta, order: int, index: int, knots: np.ndarray):
    """`_basis` as an expression in `theta`; the knots stay numeric."""
    if order == 0:
        inside = ca.logic_and(knots[index] <= theta, theta < knots[index + 1])
        if _closes(index, knots):
            inside = ca.logic_or(inside, theta >= knots[-1])
        return ca.if_else(inside, 1.0, 0.0)
    left, right = 0.0, 0.0
    if knots[index + order] != knots[index]:
        left = (theta - knots[index]) / (knots[index + order] - knots[index])
        left = left * casadi_basis(theta, order - 1, index, knots)
    if knots[index + order + 1] != knots[index + 1]:
        right = (knots[index + order + 1] - theta) / (
            knots[index + order + 1] - knots[index + 1]
        )
        right = right * casadi_basis(theta, order - 1, index + 1, knots)
    return left + right


def value(theta, control, knots: np.ndarray, order: int = ORDER) -> np.ndarray:
    """`c(theta)` per row of `theta`, numerically; `control` is (points, axes)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    control = np.asarray(control, dtype=float)
    out = np.zeros((theta.size, control.shape[1]))
    for row, place in enumerate(np.clip(theta, 0.0, 1.0)):
        for index in range(control.shape[0]):
            out[row] += control[index] * _basis(place, order, index, knots)
    return out


def casadi_value(theta, control, knots: np.ndarray, order: int = ORDER):
    """`c(theta)` as an expression. `control` is a (points, axes) SX or array."""
    place = ca.fmax(0.0, ca.fmin(1.0, theta))
    total = 0
    for index in range(control.shape[0]):
        total = total + control[index, :] * casadi_basis(place, order, index, knots)
    return total


def casadi_slope(theta, control, knots: np.ndarray, order: int = ORDER):
    """
    `dc/dtheta` as an expression -- the path's tangent, which a velocity row needs.

    The derivative of a clamped B-spline is one of order `k-1` on the interior
    knots, written here as the usual difference of neighbouring control points.
    """
    place = ca.fmax(0.0, ca.fmin(1.0, theta))
    inner = knots[1:-1]
    total = 0
    for index in range(control.shape[0] - 1):
        span = knots[index + order + 1] - knots[index + 1]
        if span == 0.0:
            continue
        step = order * (control[index + 1, :] - control[index, :]) / span
        total = total + step * casadi_basis(place, order - 1, index, inner)
    return total


@lru_cache(maxsize=None)
def _design(samples: int, control_points: int, order: int):
    """
    Give the basis at each sample, and the pseudo-inverse of its interior columns.

    Cached: the basis is a recursion in Python and the shape never changes, but
    the path behind it is refitted every cycle.
    """
    knots = knot_vector(control_points, order)
    places = np.linspace(0.0, 1.0, samples)
    design = np.array(
        [
            [_basis(place, order, index, knots) for index in range(control_points)]
            for place in places
        ]
    )
    return design, np.linalg.pinv(design[:, 1:-1])


def fit(samples, control_points: int, order: int = ORDER) -> np.ndarray:
    """
    Least-squares control points for a path sampled uniformly in its own parameter.

    `samples` is (n, axes), read at `theta = i/(n-1)`. The first and last rows are
    pinned rather than fitted: a path whose end is a goal may not be approximated
    at the goal. Fewer control points than samples is the point -- the whole path
    travels as `control_points x axes` numbers.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if control_points < order + 1:
        raise ValueError(
            f"a clamped order-{order} spline needs at least {order + 1} control "
            f"points, got {control_points}"
        )
    design, interior = _design(len(samples), control_points, order)
    # Clamped, so the end control points *are* the end of the curve. Taking them
    # as known and fitting the interior against what is left keeps them exact;
    # writing them as two more least-squares rows only asks for them politely,
    # and a goal approximated to a millimetre is a goal missed.
    control = np.zeros((control_points, samples.shape[1]))
    control[0], control[-1] = samples[0], samples[-1]
    held = np.outer(design[:, 0], control[0]) + np.outer(design[:, -1], control[-1])
    control[1:-1] = interior @ (samples - held)
    return control
