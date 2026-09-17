#!/usr/bin/env python3
"""
Run the MPC on a real plan, against a plant that is not its own model.

Replaces `mpc_a2b.py`'s invented quintic and generated-model plant: reference is
a `crane_planning` plan (same as `/crane/reference`), plant is MuJoCo.

    ./scripts/tune_mpc.py --goal out
    ./scripts/tune_mpc.py --goal here --sway-scale 4 --viewer

Unknown flags pass through to `mpc_a2b` (cost multipliers, --horizon-knots, --dt,
--enforce-budget, ...).

C3 stays in Python: MuJoCo carries the rigid fourteen; command lag, progress and
force rows are rolled forward by the exported model, which drives MuJoCo via
that force state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

import mpc_a2b  # noqa: E402

cs = mpc_a2b.cs


# scripts, not installed modules -- no import path between packages; bootstrap manually
sys.path.insert(0, str(PACKAGE.parent / "crane_planning" / "scripts"))

import tune_planner  # noqa: E402
from crane_model.conventions import PASSIVE_INDICES  # noqa: E402
from crane_model.mujoco_plant import NX_RIGID, MujocoPlant, leave  # noqa: E402


def arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    tune_planner.plan_arguments(parser)
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

    Interpolates linearly between the plan's 40ms samples (the node itself uses
    cubic Hermite; a third curve here would be a third answer). Past the end
    `np.interp` clamps onto the last sample, pinned to rest.
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

    Full state still rolled forward by `rk4_step`, only rows 0..13 replaced --
    force rows integrate off the positions MuJoCo just produced.

    Interleaved every `--cosim-step`, not once per control sample: holding the
    force across a full 60ms while MuJoCo moves under it diverges, measured, by
    7.5 rad on the rotator (C3 block 3 is stiff at `Ts`).
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

    return advance, plant


def main(argv: list[str] | None = None) -> int:
    options, forwarded = arguments(sys.argv[1:] if argv is None else argv)
    mpc = mpc_a2b.parse_arguments(forwarded)
    description, _, start, plan = tune_planner.plan_for(options)

    # plan overrides mpc_a2b's CLI: A/B are the plan's endpoints (still checked
    # against the control-safe box), duration is the plan's
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
        plant, mujoco = mujoco_plant(description, model, mpc.tool_position, options)
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
    mujoco.hold_viewer()
    return 0


if __name__ == "__main__":
    sys.exit(leave(main()))
