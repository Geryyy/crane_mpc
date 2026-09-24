#!/usr/bin/env python3
"""
The null test: machine at rest, reference says stay put. Optimal u is ~0.

Offline, no ROS graph. If u comes back at +/- u_max here, the defect is in the
OCP itself and everything above it -- node, cycle, planner, JTC -- is innocent.
Repeated cycles so a sign flip between them is visible.
"""

import copy
import sys

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from crane_model import hydraulic_limits
from crane_model import symbolic as cs
from crane_mpc import problem
from crane_mpc.config import machine_limits
from crane_mpc.horizon import Grid, Knots, resample
from crane_mpc.solver import Ocp

PLANNED = cs.K_PLANNED_DOF


def shipped():
    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/crane_mpc.yaml") as stream:
        p = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
    p.update({"hydraulics": hydraulic_limits()})
    p["limits"].update(machine_limits())
    return copy.deepcopy(p)


def hold_state(ocp):
    """At rest: no velocity, no progress rate, force holding its own weight."""
    x = np.zeros(cs.NX)
    x[cs.X_PLANNED_POSITION + 1] = 0.5
    # As test_solver.hold_state: v_s inside its box, or stage 1 is infeasible.
    x[cs.X_PROGRESS_RATE] = float(
        ocp.parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(ocp.parameters)
    ocp.pin_tool(0.3)
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PLANNED] = ocp.static_hold_force(x)
    return x


def horizon_holding(ocp, position):
    reference = Knots.zeros(2)
    reference.t[:] = [0.0, ocp.Ts * ocp.intervals]
    reference.q_a_ref[:, :PLANNED] = position
    _, horizon = resample(reference, 0.0, Grid(ocp.Ts, ocp.intervals + 1))
    return horizon


def main():
    parameters = shipped()
    for override in sys.argv[1:]:
        path, value = override.split("=")
        node = parameters
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        current = node[keys[-1]]
        node[keys[-1]] = (
            [float(value)] * len(current) if isinstance(current, list) else float(value)
        )
        print(f"override {path} = {node[keys[-1]]}")

    ocp = Ocp(
        problem.default_description().read_text(),
        parameters,
        parameters["hydraulics"],
    )
    x = hold_state(ocp)
    position = x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + PLANNED].copy()
    horizon = horizon_holding(ocp, position)
    q_eq = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF].copy()
    u_max = np.asarray(parameters["limits"]["u_max"][:PLANNED])

    print(f"u_max = {np.round(u_max, 4)}")
    print(
        f"{'cycle':>5} {'outcome':>10} "
        + " ".join(f"{n:>9}" for n in ("sw", "ha", "ka", "sa", "ro"))
        + "   |u|/u_max"
    )
    guess = None
    for cycle in range(8):
        solution = ocp.solve(x, horizon, q_eq, guess)
        u = solution.inputs[0, :PLANNED]
        frac = np.max(np.abs(u) / u_max)
        print(
            f"{cycle:>5} {solution.outcome.value:>10} "
            + " ".join(f"{v:9.4f}" for v in u)
            + f"   {frac:6.3f}"
        )
        guess = ocp.shifted(solution)
        # Machine does not move: same state every cycle, so any sign flip is
        # the optimizer's own, not the plant's.
    print(
        "\nVERDICT: bang-bang at the bound"
        if frac > 0.9
        else "\nVERDICT: u is small -- the OCP holds correctly"
    )


main()
