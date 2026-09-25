#!/usr/bin/env python3
"""
The same velocity step on every plant we have, so they can be held side by side.

MuJoCo through `C3Actuator`, the OCP's own integrator, and -- given a capture
from `steptest.py` -- Gazebo. Normalised to the commanded step, so only the
shape is compared.

Two traps this exists to keep straight. `M_ii` is pose-dependent (slewing runs
836 to 23400 over the working range), so every plant must be stepped from the
*same* pose or the numbers mean nothing. And the OCP carries the transport dead
time in its predictor rather than its plant model, so `_stepper` alone answers
one dead time early and has to be shifted before it lines up with a plant.
"""

import json
import sys

import numpy as np
from crane_model import symbolic as cs
from crane_model.actuator import C3Actuator
from crane_model.conventions import canonical_joints
from crane_model.mujoco_plant import PLANNED_INDICES, MujocoPlant
from crane_mpc import problem
from crane_mpc.solver import Ocp

TIMES = (0.06, 0.12, 0.20, 0.30, 0.50, 0.80)
CANON = list(canonical_joints())
PL = cs.K_PLANNED_DOF
POS = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + PL)
VEL = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + PL)
PAS = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF)


def gazebo(path, joint):
    """Read one axis' step out of a `steptest.py` capture."""
    run = [r for r in json.load(open(path)) if r["joint"] == joint][0]
    trace = run["trace"]
    t = np.array([s["sim"] for s in trace])
    t = t - t[0]
    index = trace[0]["names"].index(joint)
    dq = np.array([s["vel"][index] for s in trace])
    sampled = np.array([float(np.interp(f, t, dq)) for f in TIMES]) / run["amplitude"]
    return sampled, run["hold"], run["amplitude"]


def mujoco(hold, amplitude, joint):
    """Step MuJoCo through C3, which is where the dead time lives."""
    plant = MujocoPlant(problem.default_description().read_text())
    plant.set_state(np.array([hold[j][0] for j in CANON]), np.zeros(len(CANON)))
    plant.forward()
    actuator = C3Actuator(timestep=plant.model.opt.timestep)
    actuator.reset(tau=plant.holding_force)
    row = CANON.index(joint)
    u = np.zeros(len(PLANNED_INDICES))
    u[list(PLANNED_INDICES).index(row)] = amplitude
    t, dq, step = [0.0], [0.0], 0.01
    for k in range(int(0.9 / step)):
        plant.drive(actuator, u, step)
        t.append((k + 1) * step)
        dq.append(float(plant.dq[row]))
    return np.array([float(np.interp(f, t, np.asarray(dq) / amplitude)) for f in TIMES])


def ocp_model(hold, amplitude, parameters):
    """Step the OCP's own integrator, then shift it by the dead time it omits."""
    ocp = Ocp(
        problem.default_description().read_text(), parameters, parameters["hydraulics"]
    )
    q = np.array([hold[j][0] for j in CANON])
    x = np.zeros(cs.NX)
    x[POS] = q[[0, 1, 2, 3, 6]]
    x[PAS] = q[[4, 5]]
    x[cs.X_PROGRESS_RATE] = float(
        parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(parameters)
    ocp.pin_tool(float(q[7]))
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PL] = ocp.static_hold_force(x)
    t, dq = [0.0], [0.0]
    for k in range(int(0.95 / ocp.Ts)):
        u = np.zeros(cs.NU_PROGRESS)
        u[0] = amplitude
        x = ocp._integrate(ocp._stepper, x, u)
        t.append((k + 1) * ocp.Ts)
        dq.append(float(x[VEL][0]) / amplitude)
    return np.array([float(np.interp(max(f - ocp.delay_s, 0.0), t, dq)) for f in TIMES])


def main():
    import copy

    import yaml
    from ament_index_python.packages import get_package_share_directory
    from crane_model import hydraulic_limits
    from crane_mpc.config import machine_limits

    joint = sys.argv[2] if len(sys.argv) > 2 else CANON[0]
    gz, hold, amplitude = gazebo(sys.argv[1], joint)

    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/crane_mpc.yaml") as stream:
        parameters = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
    parameters.update({"hydraulics": hydraulic_limits()})
    parameters["limits"].update(machine_limits())

    print(
        f"\n{joint}, step {amplitude:+.2f}, normalised; pose "
        f"{ {k: round(v[0], 3) for k, v in hold.items()} }\n"
    )
    print(f"{'':>30} " + " ".join(f"{int(f * 1000):>7}ms" for f in TIMES))
    for label, row in (
        ("Gazebo", gz),
        ("MuJoCo", mujoco(hold, amplitude, joint)),
        (
            "OCP model, dead time added",
            ocp_model(hold, amplitude, copy.deepcopy(parameters)),
        ),
    ):
        print(f"{label:>30} " + " ".join(f"{v:>9.3f}" for v in row))


main()
