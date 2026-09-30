"""Measured pump draw: the area step per direction, and the tool the OCP leaves out."""

import numpy as np
import pytest
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import reports

POSE = np.array([0.0, 0.5, 1.0, 0.2, 0.0, 0.1])


def test_the_draw_takes_the_chamber_the_direction_fills_and_counts_the_tool():
    flow = reports.PumpFlow(hydraulic_limits())
    constants = cs.load_constants()
    areas = cs.axis_areas(constants)

    # Differential arm cylinder: extending (`J_c dq` > 0) fills the piston side,
    # as the OCP's tanh selects it.
    ratio = float(cs.jacobian_diagonal(constants, POSE[1], POSE[2])[2])
    extending = [0.0, 0.0, 0.05 * np.sign(ratio), 0.0, 0.0, 0.0]
    retracting = [-v for v in extending]
    speed = abs(ratio) * 0.05
    assert flow(POSE, extending)[2] == pytest.approx(areas[2].a_eff_pos * speed)
    assert flow(POSE, retracting)[2] == pytest.approx(areas[2].a_eff_neg * speed)

    # The rail draws although no plan moves it: legacy "5-DoF" balance left it out.
    rail = flow(POSE, [0.0, 0.0, 0.0, 0.0, 0.0, -0.1])
    assert rail[5] == pytest.approx(areas[5].a_eff_neg * 0.1)
    assert np.count_nonzero(rail) == 1
