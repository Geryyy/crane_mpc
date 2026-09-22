#!/usr/bin/env python3
"""
Simulate the whole control chain offline, from the plan down to a moving crane.

Every block here is the one the machine runs:

    plan (crane_planning)     where the tool should go
      -> MPC (crane_mpc)      solves at 16.7 Hz, outputs joint velocities u
      -> velocity loop        100 Hz PI + feedforward, as the JTC plugin runs it
      -> C3 actuator          dead time, lag, force build-up, clamp
      -> MuJoCo               the crane moves

    ./scripts/sim_chain.py --goal out --viewer
    ./scripts/sim_chain.py --random --viewer    space starts the next move

`--random` chains moves, each starting where the last one ended. Unknown flags
are passed on to `mpc_a2b`.

Four ways this is not the real machine -- read them before trusting a number:

* the total command is never clamped. The `PI clamp` column only counts the PI
  part hitting its limit, so `max|u|` can be bigger than that limit.
* the tool axis is not driven here; on the machine a separate PID holds it.
* the MPC reads the true actuator forces out of the simulation. The real node
  has no force sensor and estimates them instead.
* the MPC result is used the instant it exists. On the machine solve time and
  network delay sit in between.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

import mpc_a2b  # noqa: E402
import tune_mpc  # noqa: E402

cs = mpc_a2b.cs

sys.path.insert(0, str(PACKAGE.parent / "crane_planning" / "scripts"))

import tune_planner  # noqa: E402
from crane_model import viewing  # noqa: E402
from crane_model.actuator import C3Actuator  # noqa: E402
from crane_model.conventions import (  # noqa: E402
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    Frame,
    canonical_joints,
)
from crane_model.errors import CraneModelError  # noqa: E402
from crane_model.mujoco_plant import (  # noqa: E402
    NX_RIGID,
    PLANNED_INDICES,
    TOOL_INDEX,
    MujocoPlant,
    leave,
)
from crane_model.symbolic import K_ACTUATOR_FIT  # noqa: E402
from crane_model.velocity_loop import VelocityLoop, load_velocity_loop  # noqa: E402

AXIS_NAMES = mpc_a2b.AXIS_NAMES
PLANNED = list(PLANNED_INDICES)
POSITION = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + cs.NU)
VELOCITY = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + cs.NU)
PASSIVE = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF)
LAG = slice(cs.X_COMMAND_LAG, cs.X_COMMAND_LAG + cs.K_COMMAND_LAG_DOF)
FORCE = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF)

#: overlays: what was planned, what the solver holds right now.
PLAN_RGBA = (0.15, 0.85, 0.45, 0.6)
HORIZON_RGBA = (1.0, 0.45, 0.05, 0.9)


def arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    tune_planner.plan_arguments(parser)
    parser.add_argument(
        "--timestep",
        type=float,
        default=5.0e-4,
        help="MuJoCo step, and C3's. Must divide the dead time",
    )
    parser.add_argument(
        "--control-rate",
        type=float,
        default=None,
        help="inner loop, Hz; default is velocity_loop.yaml's own rate_hz",
    )
    parser.add_argument(
        "--dead-time",
        type=float,
        default=None,
        help=f"C3 block 1, s; default is the fit's {K_ACTUATOR_FIT.dead_time_s}. "
        "0 deletes it, which is what an ideal predictor would leave behind",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="solve from the measured state instead of predicting through the "
        "dead time",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="drop C3 block 4; the force state is then unbounded",
    )
    parser.add_argument(
        "--random",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="chain N random goals instead of --goal, each move starting where "
        "the last ended; no N is as many as the window stays open. Space in the "
        "viewer releases the next one. Plot and CSV are off",
    )
    parser.add_argument("--seed", type=int, default=None, help="--random's stream")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer playback factor; 0 is as fast as it computes. Pacing only",
    )
    return parser.parse_known_args(argv)


class Chain:
    """The three blocks under the MPC, plus what they did, per controller tick."""

    def __init__(self, plant, model, q_tool: float, options, markers=None) -> None:
        self.options = options
        self.q_tool = float(q_tool)
        self.dynamics, _, _, _ = mpc_a2b.make_numeric_functions(model)
        #: one plant per process: the viewer follows its data, a second one would
        #: open a second window.
        self.plant = plant
        self.markers = markers

        gains_by_joint, rate_hz = load_velocity_loop()
        gains = [gains_by_joint[canonical_joints()[index]] for index in PLANNED]
        self.step_s = 1.0 / (options.control_rate or rate_hz)
        self.wrap = plant.continuous_axes
        self.loop = VelocityLoop(gains, self.step_s, continuous=self.wrap)
        self.ff_scale = np.array([axis.ff_velocity_scale for axis in gains])
        self.low = np.array([axis.u_clamp_min for axis in gains])
        self.high = np.array([axis.u_clamp_max for axis in gains])

        self.dead_time_s = (
            K_ACTUATOR_FIT.dead_time_s
            if options.dead_time is None
            else float(options.dead_time)
        )
        self.predict = not options.no_predict and self.dead_time_s > 0.0
        self.tau_max = None if options.no_clamp else plant.effort_limits
        self.actuator = C3Actuator(
            timestep=plant.model.opt.timestep,
            dead_time_s=options.dead_time,
            tau_max=self.tau_max,
        )
        self.log: dict[str, list] = {key: [] for key in ("u", "e_pos", "pi", "tau")}
        #: true plant state per cycle; `advance` hands the solver a prediction, so
        #: `simulate`'s own log is not the machine.
        self.measured: list[np.ndarray] = []
        self._seeded = False

    def _seed(self, state: np.ndarray) -> None:
        self.plant.set_rigid_state(state[:NX_RIGID], self.q_tool)
        # the cylinders hold the crane up; from tau = 0 the boom drops before the
        # loop has an error to answer.
        self.actuator.reset(tau=self.plant.holding_force)
        self._seeded = True

    def advance(self, state, control, parameter, dt):
        """
        One MPC cycle: hold `control` while the inner loop runs underneath it.

        The reference the loop tracks is the horizon's first interval, integrated
        with the exported model from the measured state.
        """
        if not self._seeded:
            self._seed(state)

        # a ragged remainder would have the loop integrate over a step it was not
        # built with. Refuse it, as C3 refuses a fractional dead time.
        exact = dt / self.step_s
        if abs(exact - round(exact)) > 1.0e-9:
            raise ValueError(
                f"a {1.0 / self.step_s:.4g} Hz inner loop does not divide "
                f"T_s = {dt} s; pick a rate that does"
            )

        u_mpc = np.asarray(control[: cs.NU], dtype=float)
        reference = np.asarray(state, dtype=float).copy()
        for _ in range(int(round(exact))):
            # the JTC samples the trajectory twice per tick: the PI error against
            # the control instant, both feedforward branches one period later.
            q_ref, dq_ref = reference[POSITION].copy(), reference[VELOCITY].copy()
            reference = mpc_a2b.rk4_step(
                self.dynamics, reference, control, parameter, self.step_s
            )
            # effort = u - dq_a_ref per knot, as crane_planning writes it. The
            # loop adds ff_scale*dq_ref itself, so the two open-loop branches sum
            # to the OCP's own `u` -- the command C3's force state needs.
            dq_next = reference[VELOCITY]
            forward = self.ff_scale * (dq_next - dq_ref) + (u_mpc - dq_next)
            command = self.loop.step(
                q_ref,
                dq_ref,
                self.plant.q[PLANNED],
                self.plant.dq[PLANNED],
                feedforward=forward,
            )
            self._record(command, q_ref, dq_ref, forward)
            self.plant.drive(self.actuator, command, self.step_s)
            # sampled once per tick while C3 runs at 0.5 ms, so the clamp duty is
            # a lower bound.
            self.log["tau"].append(self.actuator.tau.copy())

        # the progress pair is virtual -- no plant holds it, and `reference`
        # already rolled it the full T_s under the same constant input.
        following = np.asarray(state, dtype=float).copy()
        following[cs.X_PROGRESS] = reference[cs.X_PROGRESS]
        following[cs.X_PROGRESS_RATE] = reference[cs.X_PROGRESS_RATE]
        following[:NX_RIGID] = self.plant.state
        following[LAG] = self.actuator.u_f[list(cs.K_LAG_AXES)]
        following[FORCE] = self.actuator.tau
        self.measured.append(following.copy())

        if self.predict:
            # the node solves from a state rolled through the transport delay
            # under the command already in flight. On the shipped grid
            # Ts == sensor_to_valve_delay, so that is exactly one command. The
            # exported model carries no delay, so this does not double-count.
            following = mpc_a2b.rk4_step(
                self.dynamics, following, control, parameter, self.dead_time_s
            )
        return following

    def show_horizon(self, states) -> None:
        """Draw the horizon in force this cycle, as the tool would trace it."""
        if self.markers is None:
            return
        rows = viewing.canonical_rows(
            states[:, POSITION], states[:, PASSIVE], self.q_tool
        )
        self.markers.path("horizon", rows, HORIZON_RGBA, width=0.05, samples=12)

    def _record(self, command, q_ref, dq_ref, forward) -> None:
        open_loop = self.ff_scale * dq_ref + (0.0 if forward is None else forward)
        error = q_ref - self.plant.q[PLANNED]
        # the rotator is continuous and the loop takes its error the short way
        # round; a plain subtraction would report ~2pi where it saw ~0.
        error = np.where(self.wrap, (error + np.pi) % (2.0 * np.pi) - np.pi, error)
        self.log["u"].append(command.copy())
        self.log["e_pos"].append(error)
        self.log["pi"].append(command - open_loop)

    def report(self) -> None:
        """Per axis: how hard the inner loop worked, and what it ran out of."""
        pi = np.array(self.log["pi"])
        error = np.array(self.log["e_pos"])
        tau = np.array(self.log["tau"])
        command = np.array(self.log["u"])
        saturated = np.isclose(pi, self.low) | np.isclose(pi, self.high)
        print(
            f"\ninner loop: {len(pi)} ticks at {1.0 / self.step_s:.0f} Hz, "
            f"dead time {self.dead_time_s * 1e3:.0f} ms "
            f"({'predicted' if self.predict else 'uncompensated'}), "
            f"block 4 {'off' if self.tau_max is None else 'on'}"
        )
        # e_pos is against the MPC's horizon, which restarts at the measured state
        # every cycle: what the inner loop is left to fix, not the plan error.
        print(
            f"{'axis':<10}{'max|e_pos|':>11}{'max|u|':>9}{'PI clamp':>10}{'tau clamp':>11}"
        )
        for axis, name in enumerate(AXIS_NAMES):
            clamped = (
                f"{np.mean(np.isclose(np.abs(tau[:, axis]), self.tau_max[axis])):10.1%}"
                if self.tau_max is not None
                else f"{'off':>10}"
            )
            print(
                f"{name:<10} {np.abs(error[:, axis]).max():10.4f} "
                f"{np.abs(command[:, axis]).max():8.4f} "
                f"{np.mean(saturated[:, axis]):9.1%} {clamped}"
            )


#: draws before a goal, or `plan_move`, gives up. At ~1/3 accepted, a 0.8% tail.
GOAL_DRAWS = 12


def random_goal(planner, start, rng) -> tuple[np.ndarray, float]:
    """
    Draw a TCP pose the machine can hold: sample the joints, take FK off them.

    Sampling joint space keeps the pose reachable by construction; a far draw can
    still leave the planner's IK in a local minimum, and `plan_move` redraws.
    """
    lower = np.where(planner.limits.bounded, planner.limits.lower, -np.pi)
    upper = np.where(planner.limits.bounded, planner.limits.upper, np.pi)
    q = np.zeros(GENERALIZED_DOF)
    q[TOOL_INDEX] = start.q_tool
    for _ in range(GOAL_DRAWS):
        q[PLANNED] = rng.uniform(lower, upper)
        q[list(PASSIVE_INDICES)] = planner.model.passive_equilibrium(
            q[list(ACTUATED_INDICES)]
        )
        # folded draws are thrown back: a plan ending there leaves the next move
        # with a start the planner will not measure.
        clearance = planner.model.collision_query(q, [])
        if (
            not clearance.collision
            and clearance.minimum_distance_m > planner.config.margin_safety
        ):
            break
    pose = planner.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    sample = tune_planner.Start(q=q, dq_a=np.zeros(len(PLANNED)))
    return np.asarray(pose.position_m), tune_planner.start_yaw(planner, sample)


def plan_move(planner, start, options, rng):
    """
    Plan one move, or say so: a refused random goal is redrawn, a named one is not.

    None means give up -- a return rather than `SystemExit`, since the viewer is
    open by now and only `leave` may end a process that has one.
    """
    for _ in range(GOAL_DRAWS):
        if rng is None:
            position, yaw = tune_planner.goal_of(planner, start, options)
        else:
            position, yaw = random_goal(planner, start, rng)
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
        except tune_planner.PlanningError as refusal:
            print(f"refused: {refusal}")
            if rng is None:
                return None
            continue
        print(plan.message)
        return plan
    print(f"no goal this planner would take in {GOAL_DRAWS} draws", file=sys.stderr)
    return None


def next_start(plant, planner) -> object:
    """
    Where the next move starts: the pose this one reached, as both boxes take it.

    A move often ends a hair outside the planner's box, which `validate_movement`
    would refuse; the rotator is continuous, so it is wrapped rather than clipped.
    """
    q = plant.q
    inside = np.clip(q[PLANNED], planner.limits.lower, planner.limits.upper)
    wrapped = (q[PLANNED] + np.pi) % (2.0 * np.pi) - np.pi
    q[PLANNED] = np.where(plant.continuous_axes, wrapped, inside)
    return tune_planner.Start(q=q, dq_a=np.zeros(len(PLANNED)))


def summarise(mpc, data, parameters, hydraulics, plan, chain, figures: bool) -> None:
    """Print what the move did, and write the figure and CSV where they are wanted."""
    if figures and not mpc.no_plot:
        mpc_a2b.plot(mpc.output.resolve(), data, parameters, hydraulics, mpc.show)
    if figures and not mpc.no_csv:
        csv_path = mpc.output.resolve().with_suffix(".csv")
        mpc_a2b.write_csv(csv_path, data)
        print(f"Wrote data: {csv_path}")
    mpc_a2b.print_summary(data, parameters)
    print(f"reference: {plan.duration:.2f} s plan, {plan.time.size} samples")
    chain.report()


def main(argv: list[str] | None = None) -> int:
    options, forwarded = arguments(sys.argv[1:] if argv is None else argv)
    mpc = mpc_a2b.parse_arguments(forwarded)
    if mpc.output == mpc_a2b.PACKAGE / "build" / "mpc_a2b.png":
        mpc.output = mpc_a2b.PACKAGE / "build" / "sim_chain.png"
    rng = None if options.random is None else np.random.default_rng(options.seed)
    description = tune_planner.plan_example.description()

    try:
        planner = tune_planner.Planner(
            description, tune_planner.plan_example.configure(options)
        )
        parameters, hydraulics = mpc_a2b.load_settings(mpc)
        solver, model, scale = mpc_a2b.create_solver(mpc, parameters, hydraulics)
    except (KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if options.random == 0 and not options.viewer:
        print("error: --random with no count needs --viewer to end it", file=sys.stderr)
        return 2

    # the solver carries no A and B -- set per cycle -- so one compile serves
    # every move, and one plant keeps one window.
    plant = MujocoPlant(description, timestep=options.timestep)
    start = anchor = tune_planner.start_of(planner, options.resettle_start)
    offset = start.q[list(PASSIVE_INDICES)] - planner.model.passive_equilibrium(
        start.q[list(ACTUATED_INDICES)]
    )
    print(f"start passive pair sits {np.degrees(offset)} deg off rest")
    gate = viewing.SpaceGate()
    markers = None
    if options.viewer:
        # placed before the window opens: planning takes seconds and nothing syncs
        # meanwhile, so an unplaced plant is what would be on screen.
        plant.set_state(start.q)
        plant.open_viewer(options.realtime, key_callback=gate)
        markers = viewing.Markers(plant)

    moves = 0
    while True:
        plan = plan_move(planner, start, options, rng)
        if plan is None:
            if rng is None or start is anchor:
                break
            # half a radian of tracking error folds the machine into itself and the
            # planner measures no start there; fall back rather than end the run.
            print("no move from here; back to the pose this run opened at")
            start = anchor
            continue
        if markers is not None:
            markers.path("plan", plan.q, PLAN_RGBA, width=0.02)

        rows = list(cs.K_PLANNED_ROWS)
        mpc.a, mpc.b = plan.q[0, rows], plan.q[-1, rows]
        mpc.move_duration = plan.duration
        mpc.tool_position = start.q_tool
        try:
            a, b = mpc_a2b.validate_movement(mpc, parameters)
            chain = Chain(plant, model, mpc.tool_position, options, markers)
            data = mpc_a2b.simulate(
                mpc,
                parameters,
                hydraulics,
                solver,
                model,
                scale,
                a,
                b,
                reference=tune_mpc.plan_reference(plan),
                plant=chain.advance,
                passive_guess=plan.q[0, list(PASSIVE_INDICES)],
                horizon=None if markers is None else chain.show_horizon,
            )
        except (CraneModelError, KeyError, ValueError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        if chain.predict:
            # `simulate` logged the predicted states `advance` handed the solver;
            # put the machine's own back before summarising.
            data = replace(data, state=np.vstack([data.state[:1], chain.measured]))
        summarise(mpc, data, parameters, hydraulics, plan, chain, figures=rng is None)

        moves += 1
        if rng is None or (options.random and moves >= options.random):
            break
        if markers is not None:
            markers.drop("horizon")
        if plant.viewer is not None:
            print("\nspace in the viewer for the next goal")
        if not gate.wait(plant):
            break
        # only the pose carries over: `simulate` builds its own initial state, so
        # the next move opens at rest with the load hanging at equilibrium.
        start = next_start(plant, planner)

    if plant.viewer is not None:
        print("close the viewer window to finish")
    plant.hold_viewer()
    return 0


if __name__ == "__main__":
    sys.exit(leave(main()))
