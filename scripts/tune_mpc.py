#!/usr/bin/env python3
"""
Run the MPC on a real plan, against a plant that is not its own model.

`mpc_a2b.py` closes the loop on a quintic it invents and on the dynamics its own
solver was generated from. This replaces both halves and keeps the controller:
the reference is a `crane_planning` plan, the same one the node would be handed
on `/crane/reference`, and the plant is MuJoCo.

    ./scripts/tune_mpc.py --goal out
    ./scripts/tune_mpc.py --goal here --sway-scale 4 --viewer

Any flag this parser does not know goes to `mpc_a2b`'s own, so every cost
multiplier, `--horizon-knots`, `--dt` and `--enforce-budget` mean the same.

C3 stays in Python: MuJoCo carries the rigid fourteen, the command lag, progress
and force rows are rolled forward by the exported model, and it is that force
state that MuJoCo is driven with.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

import mpc_a2b  # noqa: E402

cs = mpc_a2b.cs


def _sibling(package: str, module: str):
    """Import a script out of a sibling package's `scripts/`; there is no path."""
    path = PACKAGE.parent / package / "scripts" / f"{module}.py"
    if not path.is_file():
        raise SystemExit(f"no {module}.py at {path}")
    spec = importlib.util.spec_from_file_location(module, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module] = loaded
    spec.loader.exec_module(loaded)
    return loaded


tune_planner = _sibling("crane_planning", "tune_planner")
plan_example = sys.modules["plan_example"]

from crane_model.conventions import PASSIVE_INDICES  # noqa: E402
from crane_model.mujoco_plant import NX_RIGID, MujocoPlant, leave  # noqa: E402
from crane_planning import Planner, PlanningError  # noqa: E402


def arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--goal",
        choices=("out", "here"),
        default="out",
        help="'out' = (4, 0, 2) m; 'here' = start x, y lifted to z = 2 m",
    )
    parser.add_argument(
        "--goal-pose",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "YAW"),
        help="TCP pose in K0_mounting_base instead of a preset goal",
    )
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--ocp-duration-max", type=float, default=None)
    parser.add_argument("--ocp-integrator", choices=("ERK", "IRK"), default=None)
    parser.add_argument("--margin-safety", type=float, default=None)
    parser.add_argument("--margin-interp", type=float, default=None)
    parser.add_argument("--resettle-start", action="store_true")
    parser.add_argument("--timestep", type=float, default=5.0e-4)
    parser.add_argument(
        "--cosim-step",
        type=float,
        default=2.0e-3,
        help="seconds between exchanges of force and state with MuJoCo; sized "
        "for C3's stiffness, not for the control rate",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer playback factor; 0 is as fast as it computes. Pacing only",
    )
    return parser.parse_known_args(argv)


def plan_reference(plan):
    """
    Wrap a `crane_planning` plan as the callable `mpc_a2b.simulate` asks for.

    The MPC evaluates at its progress state, between the plan's 40 ms samples,
    so they are interpolated -- linearly, because the node itself resamples with
    a cubic Hermite and a third curve here would be a third answer. Past the end
    `np.interp` clamps onto the last sample, which is pinned to rest.
    """
    rows = list(cs.K_PLANNED_ROWS)

    def sample(time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.array([np.interp(time, plan.time, field[:, row]) for row in rows])
            for field in (plan.q, plan.dq, plan.ddq)
        )

    return sample


def mujoco_plant(description: str, model, q_tool: float, options):
    """
    Build the plant: MuJoCo for the rigid fourteen, the model for the rest.

    C3's command lag, progress and force rows are states of the actuator, not of
    the body, so the full state is still rolled forward by `rk4_step` and only
    rows 0..13 are replaced -- consistent because the force rows integrate off
    the positions MuJoCo just produced.

    The two interleave every `--cosim-step` rather than once per control sample.
    C3 block 3 is a stiffness and is stiff at `Ts`; holding the force across a
    whole 60 ms while MuJoCo moves under it diverges, measured, by 7.5 rad on
    the rotator.
    """
    dynamics, _, _, _ = mpc_a2b.make_numeric_functions(model)
    plant = MujocoPlant(description, timestep=options.timestep)
    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF)
    seeded = False

    def advance(state, control, parameter, dt):
        nonlocal seeded
        if not seeded:
            plant.set_rigid_state(state[:NX_RIGID], q_tool)
            if options.viewer:
                plant.open_viewer(options.realtime)
            seeded = True
        exchanges = max(1, int(round(dt / options.cosim_step)))
        step = dt / exchanges
        carried = np.asarray(state, dtype=float).copy()
        for _ in range(exchanges):
            carried = mpc_a2b.rk4_step(
                dynamics, carried, control, parameter, step, substeps=1
            )
            plant.step(carried[force], step)
            carried[:NX_RIGID] = plant.state
        return carried

    advance.mujoco = plant
    return advance


def main(argv: list[str] | None = None) -> int:
    options, forwarded = arguments(sys.argv[1:] if argv is None else argv)
    mpc = mpc_a2b.parse_arguments(forwarded)

    description = plan_example.description()
    planner = Planner(description, plan_example.configure(options))
    start = tune_planner.start_of(planner, options)
    position, yaw = tune_planner.goal_of(planner, start, options)
    print(
        f"goal: [{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}] m, "
        f"yaw {np.degrees(yaw):.1f} deg"
    )
    try:
        plan = planner.plan(
            start,
            position,
            yaw,
            scene=[],
            avoid_collisions=True,
            speed_scale=options.speed_scale,
        )
    except PlanningError as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1
    print(plan.message)

    # The plan decides the movement, so what `mpc_a2b` took from the command
    # line is overwritten: A and B are the plan's endpoints, still what the
    # control-safe box is checked against, and the duration is the plan's.
    rows = list(cs.K_PLANNED_ROWS)
    mpc.a, mpc.b = plan.q[0, rows], plan.q[-1, rows]
    mpc.move_duration = plan.duration
    mpc.tool_position = float(start.q[-1])
    if mpc.output == mpc_a2b.PACKAGE / "build" / "mpc_a2b.png":
        mpc.output = mpc_a2b.PACKAGE / "build" / "tune_mpc.png"

    try:
        parameters, hydraulics = mpc_a2b.load_settings(mpc)
        a, b = mpc_a2b.validate_movement(mpc, parameters)
        solver, model, scale = mpc_a2b.create_solver(mpc, parameters, hydraulics)
        plant = mujoco_plant(description, model, mpc.tool_position, options)
        data = mpc_a2b.simulate(
            mpc,
            parameters,
            hydraulics,
            solver,
            model,
            scale,
            a,
            b,
            reference=plan_reference(plan),
            plant=plant,
            passive_guess=plan.q[0, list(PASSIVE_INDICES)],
        )
    except (KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not mpc.no_plot:
        mpc_a2b.plot(mpc.output.resolve(), data, parameters, hydraulics, mpc.show)
    if not mpc.no_csv:
        csv_path = mpc.output.resolve().with_suffix(".csv")
        mpc_a2b.write_csv(csv_path, data)
        print(f"Wrote data: {csv_path}")
    mpc_a2b.print_summary(data, parameters)
    print(f"reference: {plan.duration:.2f} s plan, {plan.time.size} samples")
    if options.viewer:
        print("close the viewer window to finish")
    plant.mujoco.hold_viewer()
    return 0


if __name__ == "__main__":
    sys.exit(leave(main()))
