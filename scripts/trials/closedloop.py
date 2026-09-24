#!/usr/bin/env python3
"""
The node's feedback loop, offline, against the OCP's own model as the plant.

Zero model/plant mismatch, zero ROS, zero Gazebo, deterministic. Replicates
Cycle exactly: carried rows from the last solution's `carry_stage`, positions
and velocities overwritten from the "measurement", the roll over the dead time
under the command actually in flight, `q_eq` from x0.

If this diverges, the fault is in the loop, not in the sim.
"""

import copy
import sys

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_model.conventions import Tool
from crane_model.model import CraneModel
from crane_mpc import problem
from crane_mpc.config import machine_limits
from crane_mpc.cycle import carry_stage
from crane_mpc.horizon import Grid, Knots, resample
from crane_mpc.solver import Ocp, Outcome

PL = cs.K_PLANNED_DOF
PA = cs.K_PASSIVE_DOF
POS = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + PL)
VEL = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + PL)
PAS = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + PA)
PAV = slice(cs.X_PASSIVE_VELOCITY, cs.X_PASSIVE_VELOCITY + PA)
PROGRESS_ROWS = [cs.X_PROGRESS, cs.X_PROGRESS_RATE]

Q_A = np.array([0.785, 0.5236, 0.5236, 0.25, 0.0, 0.21])


def shipped(overrides=()):
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/crane_mpc.yaml") as stream:
        p = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
    p.update({"hydraulics": hydraulic_limits()})
    p["limits"].update(machine_limits())
    p = copy.deepcopy(p)
    for override in overrides:
        path, value = override.split("=")
        node, keys = p, path.split(".")
        for key in keys[:-1]:
            node = node[key]
        current = node[keys[-1]]
        node[keys[-1]] = (
            [float(value)] * len(current) if isinstance(current, list) else float(value)
        )
        print(f"override {path} = {node[keys[-1]]}")
    return p


def rest_state(ocp, model, q_a):
    """Build a state the machine really sits in: pendulum at its own equilibrium."""
    x = np.zeros(cs.NX)
    x[POS] = q_a[:PL]
    x[PAS] = np.asarray(model.passive_equilibrium(q_a))
    x[cs.X_PROGRESS_RATE] = float(
        ocp.parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(ocp.parameters)
    ocp.pin_tool(float(q_a[cs.K_TOOL_AXIS]))
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PL] = ocp.static_hold_force(x)
    return x


def hold_horizon(ocp, position):
    reference = Knots.zeros(2)
    reference.t[:] = [0.0, ocp.Ts * ocp.intervals]
    reference.q_a_ref[:, :PL] = position
    _, horizon = resample(reference, 0.0, Grid(ocp.Ts, ocp.intervals + 1))
    return horizon


def run(ocp, model, cycles, kick, plant_step, plant_delay=True):
    """Close the loop from rest with one small kick on the pendulum."""
    stage = carry_stage(ocp.delay_s, ocp.Ts)
    x_rest = rest_state(ocp, model, Q_A)
    horizon = hold_horizon(ocp, x_rest[POS])

    plant = x_rest.copy()
    plant[PAS] = plant[PAS] + kick
    carried = x_rest.copy()
    last_input = np.zeros(cs.NU_PROGRESS)
    guess = None

    rows = []
    for n in range(cycles):
        # --- Cycle.read_state: carried rows, measured positions and velocities
        x = carried.copy()
        x[POS] = plant[POS]
        x[VEL] = plant[VEL]
        x[PAS] = plant[PAS]
        x[PAV] = plant[PAV]
        x[cs.X_PROGRESS] = 0.0
        # --- Cycle.propagate: roll over the dead time under the command in flight
        x0 = ocp.propagate_applied(x, [last_input])
        q_eq = x0[PAS].copy()
        # --- Cycle.solve
        solution = ocp.solve(x0, horizon, q_eq, guess)
        u0 = solution.u0[:PL].copy()
        # --- Cycle.advance
        if solution.outcome is not Outcome.FAILED:
            carried = solution.states[stage].copy()
            carried[PROGRESS_ROWS] = solution.states[1][PROGRESS_ROWS]
        guess = None if solution.outcome is Outcome.FAILED else ocp.carried(solution)

        q_a_now = np.append(plant[POS], Q_A[cs.K_TOOL_AXIS])
        dev = float(np.max(np.abs(plant[PAS] - model.passive_equilibrium(q_a_now))))
        rows.append((n, dev, u0.copy(), solution.outcome.value))
        # --- the plant runs one Ts under the command that is in effect now,
        #     which is the one the previous cycle computed (dead time = Ts).
        plant = plant_step(plant, last_input if plant_delay else solution.u0)
        last_input = solution.u0.copy()
    return rows


def report(rows, u_max, label):
    dev = np.array([r[1] for r in rows])
    u = np.array([r[2] for r in rows])
    n = np.arange(len(rows))
    band = (dev > 2e-3) & (dev < 0.2)
    per_cycle = (
        float(np.polyfit(n[band], np.log(dev[band]), 1)[0])
        if band.sum() >= 5
        else float("nan")
    )
    sign = np.sign(u[:, 0])
    nz = sign[np.abs(u[:, 0]) > 1e-4]
    rev = float(np.mean(nz[1:] != nz[:-1])) if len(nz) > 1 else 0.0
    print(f"\n=== {label} ===")
    print(f"{'n':>3} {'dev':>10} {'|u|/u_max':>10} {'u_slew':>9}  outcome")
    for i in list(range(min(6, len(rows)))) + list(
        range(6, len(rows), max(1, len(rows) // 12))
    ):
        k, d, ui, out = rows[i]
        print(
            f"{k:>3} {d:10.6f} {np.max(np.abs(ui) / u_max):10.3f} {ui[0]:9.4f}  {out}"
        )
    print(
        f"growth {per_cycle:+.4f} /cycle -> {per_cycle / 0.06:+.3f} /s   "
        f"slew reversals {rev * 100:.1f}%   dev {dev[0]:.2e} -> {dev[-1]:.4f}"
    )
    return per_cycle


def main():
    p = shipped(sys.argv[1:])
    description = problem.default_description().read_text()
    ocp = Ocp(description, p, p["hydraulics"])
    model = CraneModel(description, Tool.PZS100)
    u_max = np.asarray(p["limits"]["u_max"][:PL])

    def nominal_plant(x, u):
        y = ocp._integrate(ocp._stepper, x, u)
        y[PROGRESS_ROWS] = x[PROGRESS_ROWS]
        return y

    for delayed, label in (
        (True, "plant HAS the modelled 60 ms dead time"),
        (False, "plant has NO dead time, model assumes 60 ms"),
    ):
        rows = run(
            ocp,
            model,
            cycles=60,
            kick=np.array([1e-3, 0.0]),
            plant_step=nominal_plant,
            plant_delay=delayed,
        )
        report(rows, u_max, label)


main()
