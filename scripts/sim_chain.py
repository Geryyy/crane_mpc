#!/usr/bin/env python3
"""
Run the whole deployed chain offline: plan -> MPC -> velocity loop -> C3 -> MuJoCo.

`tune_mpc.py` hands the MPC's `u` straight to the exported C3 rows, so the inner
loop the machine actually runs is not in it. This closes that gap -- every block
between the plan and the joint is the shipped one:

    crane_planning plan          the node's own planner, via tune_planner.plan_for
      -> crane_mpc OCP           16.7 Hz (Ts = 0.06, crane_mpc.yaml), measured
                                 state each cycle
      -> VelocityLoop            100 Hz, the PidTrajectoryPlugin transcription
      -> C3Actuator              blocks 1-4: dead time, PT1 lag, force state, clamp
      -> MujocoPlant             qfrc_applied on the five planned axes

    ./scripts/sim_chain.py --goal out --viewer
    ./scripts/sim_chain.py --goal out --dead-time 0
    ./scripts/sim_chain.py --random --viewer    space for the next goal

`--random` chains moves: each goal is a pose sampled inside the planner's own
box, each move starts where the last one ended. With the viewer the overlays
say what is being asked of the machine -- the plan's TCP curve in green, the
horizon the solver is holding this cycle in orange, redrawn at T_s.

Unknown flags pass through to `mpc_a2b` (cost multipliers, --horizon-knots, ...).

The JTC samples the trajectory twice per tick and hands the plugin both points:
the PI error is taken against the reference at the control instant, while both
feedforward branches read the reference one period later
(`joint_trajectory_controller.cpp:269-279`, `:391-394`). This reproduces that
split, and it is worth getting right in one direction: on `--goal out`, putting
*both* at the later instant ends 0.947 rad off against this split's 0.482,
because the PI then sees a standing `dq_ref*h` error that is not there. Putting
both at the control instant costs only the feedforward's half-tick lead and
lands within 4 % (0.462). The bias scales with `p`, which spans 70x here.

One block here is still ahead of what ships, and it is what the 0.20 rad final
error on `--goal out` rests on. The other one now ships:

* **the C3 feedforward.** `horizon_to_message` fills the trajectory's `effort`
  field with `u - dq_a_ref` per knot -- the identity `crane_planning` already
  used, the difference cancelling the plugin's own `ff_velocity_scale*dq_ref` so
  the open-loop branch is the OCP's own `u`. Without it the machine is commanded
  `dq_ref`, which is not what C3 needs: a force state charges on
  `k*(u_f - dq)`, so a command equal to the rate stops it charging. Measured off
  a plan the correction is only 5-8 % of `dq_ref` in RMS and still worth 0.48 rad
  against 0.20 on this move, because nothing closes a loop on it: `e_pos` stays
  near 1e-4 rad, so the PI never develops the authority to notice.
* **the dead-time predictor.** C3 block 1 is a real 60 ms, which on the shipped
  grid is exactly one cycle (`Ts == sensor_to_valve_delay`), so the state handed
  to the solver is rolled one `dead_time_s` forward under the command already in
  flight -- the single-command replay `solver.py`'s `replay_schedule` reduces to
  there. `mpc_a2b` does not do this on its own. Measured: 0.20 rad with it, 0.82
  and 32 of 178 QP failures on `--no-predict`, 0.18 at `--dead-time 0`. So the
  predictor buys back nearly all of it. `simulate` logs the predicted states it
  was handed, so `main` puts the plant's own back before anything is summarised.

Not the deployment, in four more places worth knowing before quoting a number:

* **nothing clamps the total command.** `u_clamp_*` bounds the PI branch alone
  (`velocity_loop.py`), so the `PI clamp` column is not a bound on `max|u|`
  beside it. Psi is not modelled and does not need to be: it exists on the
  machine, and C3 takes `u` directly here.
* **the tool axis is not driven.** q9 is held only by the description; the
  deployed JTC runs it through the PID plugin's `pid_joints`.
* **the MPC's stage-0 force rows are the plant's**, read back from C3. The node
  has no force measurement and seeds them instead (`solver.py`).
* **there is no MPC-to-JTC transport latency.** The horizon is consumed the
  instant it is produced; on the machine a solve time and DDS sit in between.
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

#: The two overlays, as the tool traces them: what was planned, what the solver
#: is holding right now.
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
        help="solve from the measured state. The node predicts through the dead "
        "time; mpc_a2b on its own does not",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="drop C3 block 4. The force state is then unbounded -- see actuator.py",
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
        "viewer releases the next one. Plot and CSV are off -- one run, one figure",
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
        #: One plant per process, not per move: the viewer follows its data, so
        #: a second one would open a second window.
        self.plant = plant
        self.markers = markers

        gains_by_joint, rate_hz = load_velocity_loop()
        gains = [gains_by_joint[canonical_joints()[index]] for index in PLANNED_INDICES]
        self.step_s = 1.0 / (options.control_rate or rate_hz)
        self.wrap = self.plant.continuous_axes
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
        self.tau_max = None if options.no_clamp else self.plant.effort_limits
        self.actuator = C3Actuator(
            timestep=self.plant.model.opt.timestep,
            dead_time_s=options.dead_time,
            tau_max=self.tau_max,
        )
        self.log: dict[str, list] = {key: [] for key in ("u", "e_pos", "pi", "tau")}
        #: the true plant state per cycle. `advance` hands the solver a
        #: prediction, so `simulate`'s own log is not the machine.
        self.measured: list[np.ndarray] = []
        self._seeded = False

    def _seed(self, state: np.ndarray) -> None:
        self.plant.set_rigid_state(state[:NX_RIGID], self.q_tool)
        # The cylinders hold the crane up before anyone commands anything; from
        # tau = 0 the boom drops before the loop has an error to answer.
        self.actuator.reset(tau=self.plant.holding_force)
        self._seeded = True

    def advance(self, state, control, parameter, dt):
        """
        One MPC cycle: hold `control` while the inner loop runs underneath it.

        `control` is the OCP's first input, zero-order held across the cycle as
        `/crane/mpc/horizon` is. The reference the loop tracks is that horizon's
        first interval, integrated with the exported model from the measured
        state -- the curve the JTC would interpolate, without a third answer for
        what lies between two knots.
        """
        if not self._seeded:
            self._seed(state)
        planned = list(PLANNED_INDICES)
        position = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + cs.NU)
        velocity = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + cs.NU)

        # A ragged remainder would have the loop integrate over a step it was not
        # built with. Inert at i == 0, wrong the moment anyone raises it to study
        # windup -- which is what this script is for. Refuse it, as C3 refuses a
        # fractional dead time for the same reason.
        exact = dt / self.step_s
        if abs(exact - round(exact)) > 1.0e-9:
            raise ValueError(
                f"a {1.0 / self.step_s:.4g} Hz inner loop does not divide "
                f"T_s = {dt} s; pick a rate that does"
            )

        u_mpc = np.asarray(control[: cs.NU], dtype=float)
        reference = np.asarray(state, dtype=float).copy()
        for _ in range(int(round(exact))):
            # The JTC samples the trajectory *twice* per tick and hands the
            # plugin both: `state_desired_` at the control instant, which the PI
            # error is taken against, and `command_next_` one period later, which
            # is what the feedforward branches read
            # (joint_trajectory_controller.cpp:269-279 and :391-394). Passing the
            # difference as `feedforward` moves the open-loop branch forward on
            # its own, since VelocityLoop adds `ff_scale*dq_ref` itself.
            q_ref, dq_ref = reference[position].copy(), reference[velocity].copy()
            reference = mpc_a2b.rk4_step(
                self.dynamics, reference, control, parameter, self.step_s
            )
            # What the node would publish if `horizon_to_message` filled the
            # `effort` field the way `crane_planning` does: effort = u - dq_a_ref
            # per knot, read at the same t+h as the velocity branch. With
            # ff_velocity_scale at 1.0 the two cancel to the OCP's own `u`, which
            # is the command C3 needs; `dq_ref` alone leaves its force state
            # uncharged. This is the one block here that the MPC path does not
            # ship yet.
            dq_next = reference[velocity]
            forward = self.ff_scale * (dq_next - dq_ref) + (u_mpc - dq_next)
            command = self.loop.step(
                q_ref,
                dq_ref,
                self.plant.q[planned],
                self.plant.dq[planned],
                feedforward=forward,
            )
            self._record(command, q_ref, dq_ref, forward)
            self.plant.drive(self.actuator, command, self.step_s)
            # Sampled once per tick while C3 runs at 0.5 ms, so the clamp duty is
            # a lower bound: an excursion shorter than a tick is invisible.
            self.log["tau"].append(self.actuator.tau.copy())

        # The progress pair is virtual -- no plant holds it. `reference` already
        # rolled it the full T_s under the same constant input, and it is a
        # decoupled double integrator, so a second rollout would only re-derive it.
        following = np.asarray(state, dtype=float).copy()
        following[cs.X_PROGRESS] = reference[cs.X_PROGRESS]
        following[cs.X_PROGRESS_RATE] = reference[cs.X_PROGRESS_RATE]
        following[:NX_RIGID] = self.plant.state
        lag = slice(cs.X_COMMAND_LAG, cs.X_COMMAND_LAG + cs.K_COMMAND_LAG_DOF)
        following[lag] = self.actuator.u_f[list(cs.K_LAG_AXES)]
        force = slice(cs.X_ACTUATED_FORCE, cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF)
        following[force] = self.actuator.tau
        self.measured.append(following.copy())

        if self.predict:
            # The node solves from a state predicted through the transport delay:
            # the command this cycle produces does not reach the valve for
            # `dead_time_s`, and the machine keeps moving meanwhile under the one
            # already in flight. `Ts == sensor_to_valve_delay` on the shipped
            # grid, so that is exactly one command -- replay it, which is what
            # `solver.py`'s `replay_schedule` reduces to there. The exported model
            # carries no delay of its own, so applying `control` to it directly is
            # "what happens while this command executes".
            following = mpc_a2b.rk4_step(
                self.dynamics, following, control, parameter, self.dead_time_s
            )
        return following

    def show_horizon(self, states) -> None:
        """Draw the horizon in force this cycle, as the tool would trace it."""
        if self.markers is None:
            return
        position = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + cs.NU)
        passive = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF)
        rows = viewing.canonical_rows(
            states[:, position], states[:, passive], self.q_tool
        )
        self.markers.path("horizon", rows, HORIZON_RGBA, width=0.05)

    def _record(self, command, q_ref, dq_ref, forward) -> None:
        open_loop = self.ff_scale * dq_ref + (0.0 if forward is None else forward)
        error = q_ref - self.plant.q[list(PLANNED_INDICES)]
        # The rotator is a continuous joint and the loop takes its error the short
        # way round; a plain subtraction here would report ~2pi where it saw ~0.
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
        # e_pos is against the MPC's own horizon, which restarts at the measured
        # state every cycle -- it bounds what the inner loop is left to fix, not
        # what the machine is off the plan. That is the summary's |q-qref|.
        head = ("axis", "max|e_pos|", "max|u|", "PI clamp", "tau clamp")
        print(f"{head[0]:<10}{head[1]:>11}{head[2]:>9}{head[3]:>10}{head[4]:>11}")
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


#: Draws before a goal, or `plan_move`, gives up. At ~1/3 accepted per plan,
#: twelve is a 0.8% tail.
GOAL_DRAWS = 12


def random_goal(planner, start, rng) -> tuple[np.ndarray, float]:
    """
    Draw a TCP pose the machine can hold: sample the joints, take FK off them.

    Sampling joint space and taking FK, rather than a box in Cartesian space:
    the pose is one the box admits by construction, so nothing here can ask for
    a reach the machine does not have. It can still be refused -- the planner's
    IK is seeded near the start and a far draw leaves it in a local minimum, at
    which point `plan_move` draws again. Roughly a third of draws survive.

    Folded draws are thrown back here instead: the box admits poses the machine
    is inside itself at, and a plan that ends at one leaves the next move with a
    start the planner will not measure.
    """
    lower = np.where(planner.limits.bounded, planner.limits.lower, -np.pi)
    upper = np.where(planner.limits.bounded, planner.limits.upper, np.pi)
    q = np.zeros(GENERALIZED_DOF)
    q[TOOL_INDEX] = start.q_tool
    for _ in range(GOAL_DRAWS):
        q[list(PLANNED_INDICES)] = rng.uniform(lower, upper)
        q[list(PASSIVE_INDICES)] = planner.model.passive_equilibrium(
            q[list(ACTUATED_INDICES)]
        )
        clearance = planner.model.collision_query(q, [])
        if (
            not clearance.collision
            and clearance.minimum_distance_m > planner.config.margin_safety
        ):
            break
    pose = planner.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
    sample = tune_planner.Start(q=q, dq_a=np.zeros(len(PLANNED_INDICES)))
    return np.asarray(pose.position_m), tune_planner.start_yaw(planner, sample)


def plan_move(planner, start, options, rng):
    """
    Plan one move, or say so: a refused random goal is redrawn, a named one is not.

    None means give up, and it has to be a return rather than a `SystemExit`:
    the viewer is open by now and only `leave` may end a process that has one.
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

    A move ends where C3 and the inner loop left the machine, which is a hair
    outside the planner's box often enough -- the telescope creeps past its stop
    every move. Unclipped, the plan's own A lands outside the MPC's control-safe
    box and `validate_movement` ends the whole chain over a millimetre. The
    rotator is continuous and turn-counted to 4pi, so it is wrapped rather than
    clipped: a 2pi is the same pose, and clipping one would be a real move.
    """
    rows = list(PLANNED_INDICES)
    q = plant.q
    inside = np.clip(q[rows], planner.limits.lower, planner.limits.upper)
    wrapped = (q[rows] + np.pi) % (2.0 * np.pi) - np.pi
    q[rows] = np.where(plant.continuous_axes, wrapped, inside)
    return tune_planner.Start(q=q, dq_a=np.zeros(len(rows)))


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

    # The solver carries no A and B -- those are set per cycle -- so one compile
    # serves every move, and one plant keeps one window.
    plant = MujocoPlant(description, timestep=options.timestep)
    start = anchor = tune_planner.start_of(planner, options.resettle_start)
    offset = start.q[list(PASSIVE_INDICES)] - planner.model.passive_equilibrium(
        start.q[list(ACTUATED_INDICES)]
    )
    print(f"start passive pair sits {np.degrees(offset)} deg off rest")
    gate = viewing.SpaceGate()
    markers = None
    if options.viewer:
        # Placed before the window opens: planning takes seconds and nothing
        # syncs meanwhile, so an unplaced plant is what would be on screen.
        plant.set_state(start.q)
        plant.open_viewer(options.realtime, key_callback=gate)
        markers = viewing.Markers(plant)

    moves = 0
    while True:
        plan = plan_move(planner, start, options, rng)
        if plan is None:
            if rng is None or start is anchor:
                break
            # Half a radian of tracking error folds the machine into itself, and
            # the planner measures no start there. Go back to the pose the run
            # opened at rather than end the session on one bad move.
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
            # `advance` handed the solver a state predicted through the dead time,
            # so that is what `simulate` logged. Put the machine's own states back
            # before anything is summarised, plotted or written out.
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
        # The pose this move reached carries over; nothing else does. `simulate`
        # builds its own initial state, so the next one opens at rest with the
        # load hanging at equilibrium, whatever was still swinging here.
        start = next_start(plant, planner)

    if plant.viewer is not None:
        print("close the viewer window to finish")
    plant.hold_viewer()
    return 0


if __name__ == "__main__":
    sys.exit(leave(main()))
