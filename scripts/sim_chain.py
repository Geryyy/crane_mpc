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

With the viewer open the chain keeps running once the plan is spent -- MPC,
inner loop and plant, all of it -- so the sway after arrival is something you
can watch. Space ends that and starts the next move; those cycles are printed
but not scored, or the score would depend on how long the key was held.

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
    GROUND_Z,
    NX_RIGID,
    PLANNED_INDICES,
    TOOL_INDEX,
    MujocoPlant,
    leave,
)
from crane_model.symbolic import K_ACTUATOR_FIT  # noqa: E402
from crane_model.velocity_loop import VelocityLoop, load_velocity_loop  # noqa: E402
from crane_mpc import solver as ocp_runtime  # noqa: E402
from crane_mpc.problem import PATH_POINTS  # noqa: E402
from crane_planning.ocp import evaluate  # noqa: E402

AXIS_NAMES = mpc_a2b.AXIS_NAMES
PLANNED = list(PLANNED_INDICES)
POSITION = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + cs.NU)
VELOCITY = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + cs.NU)
PASSIVE = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF)
PASSIVE_RATE = slice(cs.X_PASSIVE_VELOCITY, cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF)
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

    def advance(self, state, control, parameter, dt, knots=None):
        """
        One MPC cycle: hold `control` while the inner loop runs underneath it.

        `knots` is the position reference the node publishes for this interval --
        the plan, not a roll of the measurement, since `Cycle.adopt_solution`
        stopped overwriting those rows. The loop's `p e_pos` is its integral
        action, so this is the error it gets to integrate. Velocity and the
        feedforward branches stay the solver's, integrated with the exported
        model as before, which is what keeps `effort = u - dq_a_ref` summing to
        the OCP's own `u`.
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
        ticks = int(round(exact))
        for tick in range(ticks):
            # the JTC samples the trajectory twice per tick: the PI error against
            # the control instant, both feedforward branches one period later.
            q_ref, dq_ref = reference[POSITION].copy(), reference[VELOCITY].copy()
            if knots is not None:
                # between the two published knots, as the JTC interpolates.
                q_ref = knots[0] + (knots[1] - knots[0]) * (tick / ticks)
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
            #
            # The progress pair sits it out: `reference` above already rolled it
            # the full T_s and nothing about it travels to a valve.
            # `Ocp.propagate` holds the same rows back for the node.
            progress = following[ocp_runtime.PROGRESS_ROWS].copy()
            following = mpc_a2b.rk4_step(
                self.dynamics, following, control, parameter, self.dead_time_s
            )
            following[ocp_runtime.PROGRESS_ROWS] = progress
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


#: draws before a goal, or `plan_move`, gives up. 62% of joint draws clear both
#: collision and ground (2000 draws), so the tail is ~1e-5.
GOAL_DRAWS = 12


def random_goal(planner, start, rng) -> tuple[np.ndarray, float]:
    """
    Draw a TCP pose the machine can hold: sample the joints, take FK off them.

    Sampling joint space keeps the pose reachable by construction; a far draw can
    still leave the planner's IK in a local minimum, and `plan_move` redraws.

    Joint limits alone let the tool swing below the ground the machine stands on
    -- nothing in `collision_query` knows about the floor. Such a goal is one no
    operator would give, so it is redrawn, with the same clearance the geometry
    check uses.
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
        pose = planner.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        if (
            not clearance.collision
            and clearance.minimum_distance_m > planner.config.margin_safety
            and pose.position_m[2] > GROUND_Z + planner.config.margin_safety
        ):
            break
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


def planned_path(plan):
    """
    Take the planner's curve as the cost wants it: geometry, and how it is spent.

    `crane_planning` solves for a path and a timing law separately and only
    resamples the two together on the way out. This takes them apart again --
    `coefficients` is the curve in its own `sigma`, and `sigma(t)` is the law.
    Fitting the resample instead would bake the timing into the geometry, and a
    straight path still bends in time.
    """
    places = np.linspace(0.0, 1.0, 4 * PATH_POINTS)
    return mpc_a2b.Path(
        control=ocp_runtime.path_control(evaluate(plan.timing.coefficients, places))
    )


def coast_gate(plant, gate, message: str):
    """
    Build a `simulate` settle gate: cycle on while the window is open and unpressed.

    Headless there is nobody to press anything, so there is no coast and the run
    ends where it was asked to. A closed window leaves `pressed` false, which is
    how the caller tells "next goal" from "that is enough".
    """
    said = False

    def keep_going() -> bool:
        nonlocal said
        viewer = plant.viewer
        if viewer is None or not viewer.is_running() or gate.pressed:
            return False
        if not said:
            said = True
            print(message)
        return True

    return keep_going


#: How finely the path is sampled when measuring distance to it. The metric is a
#: minimum over `theta`, so this sets its resolution, not the path's.
PATH_SAMPLES = 400


def tool_positions(planner, rows: np.ndarray) -> np.ndarray:
    """TCP position per canonical configuration, in the mounting base."""
    return np.array(
        [
            planner.model.forward_kinematics(
                row, Frame.MOUNTING_BASE, Frame.TCP
            ).position_m
            for row in rows
        ]
    )


def path_tool_positions(planner, path, q_tool: float) -> np.ndarray:
    """
    TCP position along the whole path, the passive pair at rest.

    The curve every off-path number is measured against, the score's and the
    progress line's alike -- computed once per plan because it is `PATH_SAMPLES`
    forward-kinematics calls and neither caller wants its own answer.
    """
    places = np.linspace(0.0, 1.0, PATH_SAMPLES)
    on_path = ocp_runtime.bspline.value(
        places, path.control, ocp_runtime.problem.PATH_KNOTS
    )
    # `passive_equilibrium` wants all six actuated rows, tool included, so the
    # path's five go through a canonical frame first.
    framed = viewing.canonical_rows(
        on_path, np.zeros((len(on_path), len(PASSIVE_INDICES))), q_tool
    )
    resting = np.array(
        [
            planner.model.passive_equilibrium(row[list(ACTUATED_INDICES)])
            for row in framed
        ]
    )
    return tool_positions(planner, viewing.canonical_rows(on_path, resting, q_tool))


def off_path_now(planner, plant, curve):
    """
    Build a `simulate` progress readout: how far off the path the tool is, in m.

    Reads the plant rather than the state handed round the loop -- with the
    predictor on, that state is the one the solver gets, a dead time ahead of
    the machine, and this is the number you watch the machine by.
    """

    def gap() -> float:
        here = tool_positions(planner, plant.q[None, :])[0]
        return float(np.linalg.norm(curve - here, axis=1).min())

    return gap


def tracking_report(planner, data, chain, curve) -> dict:
    """
    Score this move the way the brief does: millimetres at the tool.

    Two different questions, and the formulation answers them separately:

    * **off the path** -- the shortest distance from the tool to the curve as a
      *set*, minimised over `theta`. A machine exactly on the path but behind it
      scores zero here, which is the whole point of following a path rather than
      a resample of one.
    * **off the goal** -- where the tool stopped against where the move was for.
      Spending time is free on the first metric and not on this one. The goal is
      the path's own end embedded at rest, not `plan.q[-1]`: that row carries the
      solve's terminal sway, which the planner lets run to `terminal_q_sway_max`
      (0.02 rad, ~18 mm at the tool), so scoring against it charged a machine
      that arrived and let the load hang.

    The path is joint-space, so its tool positions are taken with the passive
    pair at rest; the machine's are taken with the sway it actually had, because
    that is where the tool actually was.

    Scored on `data.scored` cycles, never on the coast the viewer added: the
    coast is however long a key was held and a number that moves with that is
    not a number. The coast is reported on its own line underneath.
    """
    measured = tool_positions(
        planner,
        viewing.canonical_rows(
            data.state[:, POSITION], data.state[:, PASSIVE], chain.q_tool
        ),
    )
    end = data.scored + 1
    # (samples, path samples) -- small enough to take whole, and a nearest-point
    # search that starts from the wrong end is worse than no metric at all.
    gap = np.linalg.norm(measured[:, None, :] - curve[None, :, :], axis=2).min(axis=1)
    # Sway, the two numbers that say different things: how far the load was
    # thrown, and how much of that was still swinging when the move ended. The
    # second is the damping one -- a move can stay inside the box all the way
    # and still hand the next one a pendulum.
    offset = data.state[:end, PASSIVE] - data.q_eq[:end]
    rate = data.state[:end, PASSIVE_RATE]
    last_second = max(1, int(round(1.0 / float(data.time[1] - data.time[0]))))
    score = {
        "path_worst": 1000 * float(gap[:end].max()),
        "path_rms": 1000 * float(np.sqrt((gap[:end] ** 2).mean())),
        "goal_final": 1000 * float(np.linalg.norm(measured[data.scored] - curve[-1])),
        "sway_peak": float(np.abs(offset).max()),
        "sway_end": float(np.abs(rate[-last_second:]).max()),
        "cycles": int(data.scored),
        "refused": int(np.count_nonzero(data.status[: data.scored])),
    }
    print(
        f"  TCP off the path:     worst {score['path_worst']:7.1f} mm   "
        f"rms {score['path_rms']:7.1f} mm"
    )
    print(f"  TCP off the goal:     final {score['goal_final']:7.1f} mm")
    print(
        f"  sway:                 peak {score['sway_peak']:7.3f} rad   "
        f"still moving {score['sway_end']:6.3f} rad/s"
    )
    coasted = data.fallback.size - data.scored
    if coasted > 0:
        print(
            f"  after {coasted * (data.time[1] - data.time[0]):5.1f} s of coast:"
            f"   goal {1000 * float(np.linalg.norm(measured[-1] - curve[-1])):7.1f} mm   "
            f"still moving {float(np.abs(data.state[-last_second:, PASSIVE_RATE]).max()):6.3f} rad/s"
        )
    return score


def summarise(
    mpc, data, parameters, hydraulics, plan, chain, planner, curve, figures: bool
) -> dict:
    """Print what the move did, and write the figure and CSV where they are wanted."""
    if figures and not mpc.no_plot:
        mpc_a2b.plot(mpc.output.resolve(), data, parameters, hydraulics, mpc.show)
    if figures and not mpc.no_csv:
        csv_path = mpc.output.resolve().with_suffix(".csv")
        mpc_a2b.write_csv(csv_path, data)
        print(f"Wrote data: {csv_path}")
    mpc_a2b.print_summary(data, parameters)
    score = tracking_report(planner, data, chain, curve)
    print(f"reference: {plan.duration:.2f} s plan, {plan.time.size} samples")
    chain.report()
    return score


def benchmark(scores: list[dict]) -> None:
    """
    One block per run, so two runs of the same seed are read side by side.

    The worst move is the number that matters and the median is the one that
    flatters, so both are here. `refused` rides along because a move that fails
    its way to a short trajectory can otherwise look like a well-tracked one.
    """
    if len(scores) < 2:
        return
    worst = lambda key: max(score[key] for score in scores)  # noqa: E731
    median = lambda key: float(np.median([score[key] for score in scores]))  # noqa: E731
    cycles = sum(score["cycles"] for score in scores)
    refused = sum(score["refused"] for score in scores)
    print(f"\nBenchmark over {len(scores)} moves")
    print(
        f"  TCP off the path:     worst {worst('path_worst'):7.1f} mm   "
        f"median rms {median('path_rms'):7.1f} mm"
    )
    print(
        f"  TCP off the goal:     worst {worst('goal_final'):7.1f} mm   "
        f"median     {median('goal_final'):7.1f} mm"
    )
    print(
        f"  sway:                 worst {worst('sway_peak'):7.3f} rad   "
        f"median     {median('sway_peak'):7.3f} rad"
    )
    print(
        f"  sway still moving:    worst {worst('sway_end'):7.3f} rad/s "
        f"median     {median('sway_end'):7.3f} rad/s"
    )
    print(
        f"  cycles:               {cycles} of which {refused} refused "
        f"({100.0 * refused / max(1, cycles):.1f}%)"
    )


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
    scores: list[dict] = []
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
        path = planned_path(plan)
        mpc.a, mpc.b = plan.q[0, rows], plan.q[-1, rows]
        mpc.move_duration = plan.duration
        mpc.tool_position = start.q_tool
        curve = path_tool_positions(planner, path, mpc.tool_position)
        # The coast consumes the press, so it starts from unpressed: a press
        # left over from the last move would end this one before it began.
        gate.pressed = False
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
                plant=chain.advance,
                passive_guess=plan.q[0, list(PASSIVE_INDICES)],
                path=path,
                horizon=None if markers is None else chain.show_horizon,
                off_path=off_path_now(planner, plant, curve),
                settle_gate=coast_gate(
                    plant, gate, "\nsettling -- space in the viewer for the next goal"
                ),
            )
        except (CraneModelError, KeyError, ValueError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        if chain.predict:
            # `simulate` logged the predicted states `advance` handed the solver;
            # put the machine's own back before summarising.
            data = replace(data, state=np.vstack([data.state[:1], chain.measured]))
        scores.append(
            summarise(
                mpc,
                data,
                parameters,
                hydraulics,
                plan,
                chain,
                planner,
                curve,
                figures=rng is None,
            )
        )

        moves += 1
        if rng is None or (options.random and moves >= options.random):
            break
        if markers is not None:
            markers.drop("horizon")
        # The coast did the waiting. With a window open, only a press ends it
        # with more to come -- unpressed means it closed. Headless there was no
        # coast and nobody to press, so the chain goes on.
        if plant.viewer is not None and not gate.pressed:
            break
        # only the pose carries over: `simulate` builds its own initial state, so
        # the next move opens at rest with the load hanging at equilibrium.
        start = next_start(plant, planner)

    benchmark(scores)
    if plant.viewer is not None:
        print("close the viewer window to finish")
    plant.hold_viewer()
    return 0


if __name__ == "__main__":
    sys.exit(leave(main()))
