#!/usr/bin/env python3
"""
Run the MPC on a real plan, against a plant that is not its own model.

`mpc_a2b.py` closes the loop on the quintic it makes up and the dynamics the
solver was generated from. That measures the controller and nothing else, which
is what it is for. This replaces both halves and keeps the controller:

* the reference is a `crane_planning` plan -- the same `Planner` the node runs,
  solved between the same preset poses `tune_planner.py` uses, so what the MPC
  tracks here is what it would be handed on `/crane/reference`;
* the plant is MuJoCo (`crane_model.mujoco_plant`), an independent
  articulated-body solver on the same URDF, so tracking error and residual sway
  are answers about the model and not tautologies.

    ./scripts/tune_mpc.py --goal out
    ./scripts/tune_mpc.py --goal here --sway-scale 4 --model-matched

Any flag this parser does not know is passed straight to `mpc_a2b`'s own, so
every cost multiplier, `--horizon-knots`, `--dt` and `--enforce-budget` work
unchanged and mean the same thing.

C3 stays in Python. MuJoCo integrates the rigid fourteen; the command lag, the
progress pair and the force rows are rolled forward by the exported model
exactly as before, and it is that force state -- `X_ACTUATED_FORCE`, N and N*m
per planned axis -- that MuJoCo is driven with. So the actuator is the fitted
one and the body is MuJoCo's. Neither is the hydraulics
(`wiki/hydraulics.md` §5.1).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

import mpc_a2b  # noqa: E402

cs = mpc_a2b.cs


def _sibling(package: str, module: str):
    """
    Import a script out of a sibling package's `scripts/` directory.

    The tuning tools are scripts and not installed modules, so there is no
    import path between them. Reaching across by file is the same trick
    `plan_example.py` uses to reach `crane_model`, and the alternative is a
    second copy of the start pose, the goals and the planner call -- which is
    exactly the thing that drifts.
    """
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
from crane_model.mujoco_plant import (  # noqa: E402
    NX_RIGID,
    MujocoPlant,
    viewer_was_opened,
)
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
    parser.add_argument(
        "--timestep",
        type=float,
        default=5.0e-4,
        help="MuJoCo step; the C3 force is held across one control sample",
    )
    parser.add_argument(
        "--cosim-step",
        type=float,
        default=2.0e-3,
        help=(
            "seconds between exchanges of force and state between the C3 block "
            "and MuJoCo. C3 is stiff at the control sample, so this is sized "
            "for it and not for the control rate"
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="watch it in MuJoCo's passive viewer while it runs; there is "
        "nothing to watch under --model-matched",
    )
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help=(
            "viewer playback factor: 1.0 is a simulated second per second, 0 is "
            "as fast as it computes. Pacing only, the run is the same either way"
        ),
    )
    parser.add_argument(
        "--model-matched",
        action="store_true",
        help=(
            "keep mpc_a2b's own ERK4 plant instead of MuJoCo. The reference is "
            "still the plan, so this isolates the modelling error from the "
            "tracking error: run both and the difference is MuJoCo's"
        ),
    )
    return parser.parse_known_args(argv)


def plan_reference(plan, tool_position: float):
    """
    Wrap a `crane_planning` plan as the callable `mpc_a2b.simulate` asks for.

    The MPC evaluates its reference at the progress state, at times between the
    plan's own 40 ms samples, so the samples are interpolated. Linear, not
    Hermite: the plan is C4 and sampled ten times per sway period, so the
    interpolation error is far below the tracking error being measured, and the
    node itself resamples with a cubic Hermite off positions and velocities --
    a third curve here would be a third answer to compare against.

    Past the end the plan holds: `np.interp` clamps, and the plan's last sample
    is pinned to zero rate and zero acceleration, so the settle phase asks the
    controller to stand still at the goal.
    """
    del tool_position  # the tool rides in `p`, not in the reference
    rows = list(mpc_a2b.cs.K_PLANNED_ROWS)

    def sample(time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.array([np.interp(time, plan.time, field[:, row]) for row in rows])
            for field in (plan.q, plan.dq, plan.ddq)
        )

    return sample


def mujoco_plant(description: str, model, start_q, options):
    """
    Build the plant: MuJoCo for the rigid fourteen, the model for the rest.

    C3's command lag, progress pair and force rows are states of the *actuator*,
    not of the body, and MuJoCo has no opinion about them -- so the full state
    is still rolled forward by `rk4_step` and only rows 0..13 are replaced. The
    two stay consistent because the force rows are integrated off the positions
    and velocities MuJoCo just produced.

    **The two have to be interleaved, not alternated once per control sample.**
    C3 block 3 is a stiffness -- the force state chases the commanded velocity
    against the joint's own -- and it is stiff at `T_s`: `mpc_a2b` says
    `|lambda| T_s = 8.5` at an ordinary pose, which is why its own plant
    substeps ten times. Holding the force across a whole 60 ms while MuJoCo
    moves underneath it is that same explicit step, and it diverges the same
    way: measured, a run with one exchange per cycle leaves the rotator 7.5 rad
    out and the slew 2.8. So the exchange happens every `--cosim-step`, short
    enough that neither side runs open loop on the other.
    """
    dynamics, _, _, _ = mpc_a2b.make_numeric_functions(model)
    plant = MujocoPlant(description, timestep=options.timestep)
    seeded = False
    force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF)

    def advance(state, control, parameter, dt):
        nonlocal seeded
        if not seeded:
            plant.set_rigid_state(state[:NX_RIGID], float(start_q[-1]))
            if options.viewer:
                plant.open_viewer(realtime=options.realtime)
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

    # The caller needs the plant itself to hold the window open once the run is
    # over, and a closure is otherwise the only reference to it.
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

    # The plan decides the movement, so what `mpc_a2b` would have taken from the
    # command line is overwritten: its A and B become the plan's own endpoints
    # (they are still what the control-safe box is checked against) and its
    # nominal duration becomes the plan's.
    rows = list(cs.K_PLANNED_ROWS)
    mpc.a = plan.q[0, rows]
    mpc.b = plan.q[-1, rows]
    mpc.move_duration = plan.duration
    mpc.tool_position = float(start.q[-1])
    if mpc.output == mpc_a2b.PACKAGE / "build" / "mpc_a2b.png":
        mpc.output = mpc_a2b.PACKAGE / "build" / "tune_mpc.png"

    plant = None
    try:
        parameters, hydraulics = mpc_a2b.load_settings(mpc)
        a, b = mpc_a2b.validate_movement(mpc, parameters)
        solver, model, scale = mpc_a2b.create_solver(mpc, parameters, hydraulics)
        if not options.model_matched:
            plant = mujoco_plant(description, model, start.q, options)
        data = mpc_a2b.simulate(
            mpc,
            parameters,
            hydraulics,
            solver,
            model,
            scale,
            a,
            b,
            reference=plan_reference(plan, mpc.tool_position),
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
    print(
        f"plant: {'the exported model' if options.model_matched else 'MuJoCo'}; "
        f"reference: {plan.duration:.2f} s plan, {plan.time.size} samples"
    )
    if plant is not None:
        # The summary comes first; the window stays until it is closed.
        if options.viewer:
            print("close the viewer window to finish")
        plant.mujoco.hold_viewer()
    return 0


if __name__ == "__main__":
    status = main()
    # A viewer run would otherwise exit 139: MuJoCo's viewer segfaults on
    # interpreter teardown here, after every file is written. `os._exit` leaves
    # without tearing down.
    if viewer_was_opened():
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)
