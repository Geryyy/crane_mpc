"""
`--plant hyd`'s contract: two regimes, one switch, and the one measured load law.

The two things worth pinning are the ones that were wrong first time. The shut
branch must stand on *gravity* and not on `MujocoPlant.h_eff` -- `h_eff`
compensates the joint damper along with gravity, and a spring handed that force
cancels its own decay and rings forever. And the boom's lowering branch has to
reproduce 314 step 5's single measured number, x1.30 loaded against empty at
matched valve current, on this bench's own holding forces.

A stub plant, not MuJoCo: the law reads `mj_fullM`, the bias rows and the
projection, and nothing else.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wire_chain"))

import hyd_plant  # noqa: E402
from crane_model.conventions import GENERALIZED_DOF, PLANNED_INDICES  # noqa: E402

TIMESTEP = 5.0e-4
BOOM, ARM = hyd_plant.HYD_AXES
#: the loaded 11-09 slew pose, `records/analysis_316/probe.py`: M_ii diag, h_eff
#: diagonal, so the sway rows solve to zero and the planned holds *are* the bias
MASS = np.array([8.46e4, 9.51e4, 5.50e4, 5.74e3, 1.1e3, 1.1e3, 87.6, 1.0])
GRAVITY = np.array([-518.0, 1.264e5, 7.03e4, -1.07e4, 0.0, 0.0, 0.1, 0.0])
DAMPING = np.array([7520.0, 165600.0, 47000.0, 133920.0, 0.0, 0.0, 484.0, 0.0])


class FakePlant:
    """What `HydraulicActuator` reads off a plant: the mass matrix and two bias rows."""

    def __init__(self, dq) -> None:
        self.model = SimpleNamespace(nv=GENERALIZED_DOF)
        self.data = SimpleNamespace(
            qfrc_bias=GRAVITY.copy(), qfrc_passive=-DAMPING * dq
        )
        self.projection = np.eye(GENERALIZED_DOF)
        self._mj = SimpleNamespace(mj_fullM=self._full)

    @staticmethod
    def _full(model, data, out) -> None:
        out[:, :] = np.diag(MASS)


def build(dq_planned, k_scale=hyd_plant.K_SHUT_SCALE):
    """An actuator on a plant standing still except for `dq_planned`."""
    dq = np.zeros(GENERALIZED_DOF)
    dq[list(PLANNED_INDICES)] = dq_planned
    plant = FakePlant(dq)
    actuator = hyd_plant.HydraulicActuator(
        plant, timestep=TIMESTEP, k_shut_scale=k_scale
    )
    actuator.reset(tau=GRAVITY[list(PLANNED_INDICES)])
    return actuator


def drive(actuator, u_boom, dq_planned, seconds):
    """Hold one command on the boom; the stub's state does not move."""
    u = np.zeros(len(PLANNED_INDICES))
    u[BOOM] = u_boom
    for _ in range(int(round(seconds / TIMESTEP))):
        tau = actuator.step(u, dq_planned)
    return tau


def test_the_lowering_branch_carries_314s_measured_load_factor():
    """x1.30 loaded against empty, and 1.0 at the condition 313 measured the gain on."""
    actuator = build(np.zeros(len(PLANNED_INDICES)))
    assert actuator.lower_factor(hyd_plant.TAU_HOLD_REF) == pytest.approx(1.0)
    empty = actuator.lower_factor(6.11e4)  # the same pose, payload off
    assert 1.0 / empty == pytest.approx(1.30, abs=0.02)
    # lifting is load-blind (meter-in with a working compensator, 314 step 5)
    actuator.u_f = np.array([0.1, 0.1])
    assert actuator.flow_gain(np.array([6.11e4, 0.0]))[0] == hyd_plant.GAIN_LIFT[0]


def test_the_shut_spool_is_a_spring_that_gravity_holds_up_and_d_damps():
    """Not `h_eff`: a spring handed the damper's compensation never decays."""
    dq = np.zeros(len(PLANNED_INDICES))
    dq[BOOM] = 0.05
    actuator = build(dq)
    seconds = 0.2
    tau = drive(actuator, 0.0, dq, seconds)
    k_shut = hyd_plant.K_SHUT_SCALE * np.asarray(actuator.c3.fit.k, float)[BOOM]
    # the spool was shut from the first step: gravity, less the spring's integral
    expected = GRAVITY[BOOM] - k_shut * dq[BOOM] * seconds
    assert tau[BOOM] == pytest.approx(expected, rel=1e-3)
    assert actuator.k_shut[0] / actuator.c3.fit.k[BOOM] == pytest.approx(
        hyd_plant.K_SHUT_SCALE
    )


def test_the_open_spool_is_a_velocity_source_behind_the_measured_lag():
    """Dead time first, then the PT1, then flow at the per-direction gain."""
    dq = np.zeros(len(PLANNED_INDICES))
    actuator = build(dq)
    command = -0.2  # lowering, at the reference load: gain 0.95 x factor 1.0
    held = drive(actuator, command, dq, hyd_plant.DEAD_TIME_S - TIMESTEP)
    assert held[BOOM] == pytest.approx(GRAVITY[BOOM], rel=1e-6)  # still dead
    settled = drive(actuator, command, dq, 1.0)
    gain = hyd_plant.GAIN_LOWER[0] * actuator.lower_factor(GRAVITY[BOOM])
    servo = MASS[BOOM] * hyd_plant.SERVO_RAD_S * gain * command
    assert settled[BOOM] == pytest.approx(GRAVITY[BOOM] + servo, rel=1e-3)


def test_the_other_three_axes_are_still_c3():
    """Slew, telescope and rotator keep the actuator the staircases fitted."""
    dq = np.zeros(len(PLANNED_INDICES))
    actuator = build(dq)
    reference = hyd_plant.C3Actuator(timestep=TIMESTEP)
    reference.reset(tau=GRAVITY[list(PLANNED_INDICES)])
    u = np.zeros(len(PLANNED_INDICES))
    u[0], u[3], u[4] = 0.1, 0.05, 0.2
    for _ in range(400):
        mine, theirs = actuator.step(u, dq), reference.step(u, dq)
    assert mine[[0, 3, 4]] == pytest.approx(theirs[[0, 3, 4]])
