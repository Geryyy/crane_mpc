#!/usr/bin/env python3
"""
The node's own `Cycle`, closed through the wire: horizon message -> JTC -> MuJoCo.

Runs `Cycle.step` -- the node's own call order, not a copy of it -- writes each
horizon with `hz.horizon_to_message` and plays it as `trajectory.cpp` does into
the velocity loop, C3 actuator and MuJoCo plant of `harness.py`: horizon encoding, JTC
interpolation, its one-tick feedforward lookahead, delivery latency and cadence
phase included -- the five faults behind the 2026-09-25 Gazebo divergence
(issue 171). Nothing is published.

    ./scripts/trials/wire_chain.py --goal out --latency 0.02
    ./scripts/trials/wire_chain.py --goal out --wire 161         # must diverge
    ./scripts/trials/wire_chain.py --goal out --cadence-phase 0.03  # must diverge
    ./scripts/trials/wire_chain.py --goal out --viewer --realtime 0
    ./scripts/trials/wire_chain.py --random --viewer   # space: next move

`--viewer` keeps the chain running past the plan, unscored, until space or the
window closes. `--random N` chains N moves, each from where the last ended.

Plant against OCP, i.e. the mismatch: `--plant-payload`/`--plant-payload-com`
against `--payload-mass`/`--payload-com`, `--k-scale`/`--d-scale`,
`--tau-v-scale`/`--tau-v-offset`, `--dead-time`, `--psi-gain-*`. The robustness
map they were cut for is `robustness.md`.

Plan, plant mismatch and solver flags are `harness.add_arguments`'. Exit 1 on
divergence, hunting (`hunting.py`: u0 reversing on half the cycles) or a JTC
command outside `u_clamp_*`, Psi's fitted domain (`past_clamp`).

Levers: `--set weights.du=30 --set dq_a_feedback=false` overrides crane_mpc.yaml
(yaml value; a scalar on a list key fills every axis). A key in
`solver.export_key` (Ts, horizon_length, ...) re-exports the solver: minutes,
and unlocked, so run that setting once before a parallel sweep. `Ts` moves
`sensor_to_valve_delay` with it -- `check_settings` refuses any other pair, and
`harness.create_solver` refuses before the export rather than after it. Payload: `--payload-mass` is what the OCP believes, `--plant-payload`
what MuJoCo carries (default: the same), both at `--payload-com` in K8.

Not the machine: measurement is exact at the cycle instant, the solve takes no
time (latency is delivery only; budget off), the tool axis is not driven, and
the JTC's tolerance checks, goal handling and angle wraparound are skipped.
"""

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import harness
import numpy as np
from builtin_interfaces.msg import Time
from crane_model.mismatch import perturb_fit
from crane_model.model import Payload
from crane_model.mujoco_plant import MujocoPlant, leave
from crane_model.symbolic import PAYLOAD_MOUNT_LINK
from crane_model.viewing import SpaceGate
from crane_mpc import cycle as cy
from crane_mpc import horizon as hz
from crane_mpc import problem as ocp_problem
from crane_mpc.equilibrium import Equilibrium
from crane_mpc.solver import SLACK_NOTICEABLE
from growth import growth
from harness import ACTUATED, PASSIVE, PLANNED, cs, tune_planner
from hunting import REVERSAL_FRACTION, reversal_fraction
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ROTATOR = 4
#: ros2_control yaml `topic_timeout`: past it the JTC holds position.
TOPIC_TIMEOUT = 0.18
#: Sim clock origin; trajectory.cpp reads a zero stamp as "start on receipt".
EPOCH = 1.0
#: |sway| past this is a diverged run, not a bad one.
DIVERGED_SWAY = 1.2
#: The sway period `limits.du_max` is derived from; an envelope is read over one.
SWAY_PERIOD_S = 2.34
#: Where "2 s after arrival" is read: +-0.5 s, ~a fifth of a sway period either
#: side, so the number is the mode's local envelope and not whichever phase one
#: cycle happened to fall on.
AFTER_ARRIVAL_S = (1.5, 2.5)
#: A box row counts as riding its bound within this fraction of it. Not equality:
#: the QP stops at its own tolerance and `du_max` is reached by a ramp.
ACTIVE_MARGIN = 0.99
#: Every cycle would cost N+1 residual evaluations of the solve's own latency.
COST_DECIMATION = 10
#: What `--relax-h` widens a row to. `h` is a fraction of its own allowance, so
#: ten is already ~11x the worst draw anything here reaches (`wiki/mpc.md` puts a
#: resting boom+arm at 0.91 of `F_max`, and the planner's peak pump draw is 0.76).
#: Not 1e6, which is out of reach the same way and costs HPIPM its conditioning:
#: at 1e6 the same runs read QP iterations p90 8 -> 12-14 and max 12 -> 60 with
#: every quality column unmoved, i.e. an artifact of the number and not of the
#: constraint (issue 248).
RELAXED = 10.0


def envelope_tau(t, rate):
    """
    Decay time constant of `|rate|`'s one-period rolling max, per column.

    `inf` where the envelope does not decay, `nan` where it is already at zero.
    Fitted on the envelope and not on the signal: the rate oscillates, so a fit
    on it reads whichever phase the window ends on, and "damped within one sway
    period" is a claim about the envelope.
    """
    width = max(2, int(round(SWAY_PERIOD_S / (t[1] - t[0]))))
    taus = []
    for column in np.abs(rate).T:
        envelope = np.array(
            [column[i : i + width].max() for i in range(len(column) - width + 1)]
        )
        keep = envelope > 1e-6
        if keep.sum() < 4:
            taus.append(float("nan"))
            continue
        slope = np.polyfit(t[: envelope.size][keep], np.log(envelope[keep]), 1)[0]
        taus.append(-1.0 / slope if slope < 0.0 else float("inf"))
    return taus


def scale_h(ocp, rows, factor):
    """
    Multiply constraint 6 (cylinder force) and/or 7 (pump flow)'s allowance.

    `factor` past one removes the row -- `h` is a fraction of its own allowance,
    so nothing a move does reaches ten. Under one it *tightens*, which is how the
    headroom gets measured: lower it until slack appears and the row's distance
    from binding is the factor it binds at.

    Not a `--set`: `problem.constraint_scale` divides each row by its own
    allowance, so `lh`/`uh` come out in `[-1, 1]` and `[0, 1]` whatever the
    hydraulics yaml says, and no number in either yaml can move them. Written
    straight onto the solver, which holds because `configure_fixed_data` writes
    the pair once at construction and no cycle rewrites it.

    `nh` and the two rows' slack columns stay in the QP either way, so this
    measures what the constraint *does*, not what dropping it would save.
    """
    force = slice(0, cs.PLANNED_DOF)
    for stage in range(ocp.intervals):
        lower = np.asarray(ocp.solver.constraints_get(stage, "lh"), float)
        upper = np.asarray(ocp.solver.constraints_get(stage, "uh"), float)
        if rows in ("force", "both"):
            lower[force] *= factor
            upper[force] *= factor
        if rows in ("pump", "both"):
            upper[cs.NU] *= factor
        ocp.solver.constraints_set(stage, "lh", lower)
        ocp.solver.constraints_set(stage, "uh", upper)


def riding(solution, limits):
    """
    Which of the OCP's bounds this cycle's horizon rides, and what it slacked.

    The hard boxes (`du_max`, constraint 5's `u^+`, the sway-rate bound) are read
    as proximity over the horizon; the softened rows are read as slack taken,
    which is a *violation* and so a stricter test than being active -- a soft
    bound can bind exactly with no slack and not be counted here. Against
    `SLACK_NOTICEABLE`, not zero: HPIPM leaves every slack a positive 1e-11 or so,
    which read as "every softened row violated on every cycle".
    """
    du_max = np.asarray(limits["du_max"][: cs.PLANNED_DOF], float)
    u_max = np.asarray(limits["u_max"][: cs.PLANNED_DOF], float)
    rate = np.abs(solution.inputs[:, : cs.PLANNED_DOF]).max(0) / du_max
    passive = slice(cs.X_PASSIVE_VELOCITY, cs.X_PASSIVE_VELOCITY + cs.PASSIVE_DOF)
    sway_rate = np.abs(solution.states[:, passive]).max(0) / np.asarray(
        limits["dq_sway_max"], float
    )
    command = (
        np.abs(solution.states[:, cs.X_COMMAND : cs.X_COMMAND + cs.PLANNED_DOF]).max(0)
        / u_max
        if solution.states.shape[1] > cs.X_COMMAND
        else np.zeros(cs.PLANNED_DOF)
    )
    violation = solution.violation
    return {
        "du_max": bool(rate.max() >= ACTIVE_MARGIN),
        "u_clamp": bool(command.max() >= ACTIVE_MARGIN),
        "dq_sway": bool(sway_rate.max() >= ACTIVE_MARGIN),
        "q_sway_slacked": bool(violation.q_u > SLACK_NOTICEABLE),
        "cylinder_force_slacked": bool(violation.cylinder_force > SLACK_NOTICEABLE),
        "pump_flow_slacked": bool(violation.pump_flow > SLACK_NOTICEABLE),
        "slack": bool(solution.used_slack),
    }


class Playback:
    """One received `JointTrajectory`, sampled as `Trajectory::sample` does (planned axes)."""

    def __init__(self, message, received, q, dq):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        points = message.points
        self.t = stamp + np.array(
            [hz.duration_seconds(p.time_from_start) for p in points]
        )
        self.pos, self.vel, self.eff = (
            np.array([np.asarray(getattr(p, field))[PLANNED] for p in points])
            for field in ("positions", "velocities", "effort")
        )
        # `set_point_before_trajectory_msg`: measured state at receipt, effort 0.
        self.before = (received, q, dq, np.zeros_like(q))

    @staticmethod
    def hold(now, q):
        """`set_hold_position`: where the machine stands, at rest."""
        message = JointTrajectory()
        zero = np.zeros(cs.GENERALIZED_DOF)
        full = zero.copy()
        full[PLANNED] = q
        message.points = [
            JointTrajectoryPoint(
                positions=full.tolist(), velocities=zero.tolist(), effort=zero.tolist()
            )
        ]
        message.header.stamp = stamp_of(now)
        return Playback(message, now, q, np.zeros_like(q))

    def sample(self, t):
        """Return `(q_ref, dq_ref, effort)` at `t`."""
        if t < self.t[0]:
            t0, p0, v0, e0 = self.before
            t1, p1, v1, e1 = self.t[0], self.pos[0], self.vel[0], self.eff[0]
        elif t >= self.t[-1]:
            return self.pos[-1], self.vel[-1], self.eff[-1]
        else:
            i = np.searchsorted(self.t, t, side="right") - 1
            t0, p0, v0, e0 = self.t[i], self.pos[i], self.vel[i], self.eff[i]
            t1, p1, v1, e1 = (
                self.t[i + 1],
                self.pos[i + 1],
                self.vel[i + 1],
                self.eff[i + 1],
            )
        q, dq = hz._jtc_cubic(t0, t1, p0, v0, p1, v1, t)
        return q, dq, e0 + (e1 - e0) * (t - t0) / (t1 - t0)


def stamp_of(seconds):
    ns = int(round(seconds * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def carry_payload(plant, mass, com):
    """Put a point mass on MuJoCo's payload mount, where the OCP's sits."""
    if not mass:
        return
    mj, model = plant._mj, plant.model
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, PAYLOAD_MOUNT_LINK)
    if model.body_mass[body] != 0.0:
        raise SystemExit(f"{PAYLOAD_MOUNT_LINK} has mass already; merge not done")
    model.body_mass[body] = mass
    model.body_ipos[body] = com
    model.body_iquat[body] = (1.0, 0.0, 0.0, 0.0)
    model.body_inertia[body] = 1e-4 * mass  # near-point; MuJoCo wants > 0
    mj.mj_setConst(model, plant.data)
    plant.forward()


def scale_tau_v(fit, scale):
    """
    Plant `tau_v` under a per-axis multiplier; the OCP keeps the shipped fit.

    Not `perturb_fit`'s `lag_shift_s`, which moves `tau_v` and the dead time at
    constant sum: the shipped arm sits at `tau_v = 0`, so every shift is refused
    (`mismatch.py`). A multiplier keeps 0 at 0, so `lag_axes` -- hence NX and the
    export key -- does not move, and what it changes is the dead time + lag
    *sum*, which is the half the staircase campaign never separated.
    """
    scale = np.asarray(scale, float)
    if not np.all(np.isfinite(scale) & (scale > 0.0)):
        raise SystemExit(f"--tau-v-scale wants positive finite multipliers: {scale}")
    return dataclasses.replace(fit, tau_v=tuple(np.asarray(fit.tau_v, float) * scale))


def offset_tau_v(fit, offset):
    """
    Plant `tau_v` under a per-axis *additive* offset, floored at zero.

    `scale_tau_v` cannot move an axis fitted at zero and the arm is
    (`tau_v = [0.1, 0.025, 0.0, 0.05, 0.125]`), so every multiplicative lag
    family leaves the axis carrying the payload moment arm untouched --
    `robustness.md` §1 says so and §4 asks for this knob. An offset moves it.
    Still not `perturb_fit`'s `lag_shift_s`, which holds dead time + `tau_v`
    constant and is refused at that same zero: here the *sum* moves, which is
    the half the staircase campaign never separated anyway.

    Floored, not refused, below zero: `C3Actuator` reads `tau_v < 0` as
    alpha = 1, i.e. no lag at all, so a negative would silently clip. Zero is
    what "no lag" means and the axes already there cannot get faster.
    `LAG_AXES` moves with the floor, but only on the plant's fit -- the export
    key digests `c3_full_model.json`, not this.
    """
    offset = np.asarray(offset, float)
    if not np.all(np.isfinite(offset)):
        raise SystemExit(f"--tau-v-offset wants finite seconds: {offset}")
    lagged = np.maximum(np.asarray(fit.tau_v, float) + offset, 0.0)
    return dataclasses.replace(fit, tau_v=tuple(lagged))


def wire_161(horizon, joints, stamp, lead, in_flight):
    """Issue 161's encoding: knot i at lead + t_i, the in-flight knot at 0, u - dq per knot."""
    message = hz.horizon_to_message(horizon, joints, stamp)
    for point in message.points:
        point.time_from_start = hz.seconds_duration(
            lead + hz.duration_seconds(point.time_from_start)
        )
    standing = horizon if in_flight is None else in_flight
    message.points.insert(0, hz.horizon_to_message(standing, joints, stamp).points[0])
    return message


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--latency", type=float, default=0.0, help="s, measurement to JTC receipt"
    )
    parser.add_argument(
        "--seconds", type=float, default=None, help="default plan + 5 s"
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="per-cycle log, last move's"
    )
    # Regression checks: each reverts one fix of issue 171 and must diverge.
    parser.add_argument("--wire", choices=("today", "161"), default="today")
    parser.add_argument(
        "--cadence-phase",
        type=float,
        default=0.0,
        help="s; stamp every horizon this far off the measurement, as the "
        "once-anchored grid did",
    )
    parser.add_argument(
        "--plant-payload", type=float, default=None, help="kg; default --payload-mass"
    )
    parser.add_argument(
        "--plant-payload-com",
        type=float,
        nargs=3,
        default=None,
        metavar="M",
        help="m in K8, what MuJoCo hangs; default --payload-com",
    )
    parser.add_argument(
        "--tau-v-scale",
        type=float,
        nargs=harness.AXES,
        default=None,
        metavar="S",
        help="plant C3 tau_v multiplier per planned axis",
    )
    parser.add_argument(
        "--tau-v-offset",
        type=float,
        nargs=harness.AXES,
        default=None,
        metavar="S",
        help="plant C3 tau_v += s per planned axis, floored at 0; reaches the "
        "arm, which is fitted at 0 and no multiplier can move",
    )
    parser.add_argument(
        "--relax-h",
        choices=("force", "pump", "both"),
        default=None,
        help="scale constraint 6 / 7's allowance; the one row group no --set can "
        "move, since h is scaled by its own allowance",
    )
    parser.add_argument(
        "--h-factor",
        type=float,
        default=RELAXED,
        help=f"--relax-h's multiplier; default {RELAXED:g} removes the row, "
        "under 1 tightens it to read the headroom",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="scores; a list per move if chained"
    )
    parser.add_argument("--viewer", action="store_true", help="MuJoCo passive viewer")
    parser.add_argument(
        "--start",
        type=float,
        nargs=10,
        metavar="X",
        help="first move from a live state: canonical eight positions, sway included, "
        "then tip/tilt rates (e.g. off a cap.json); default the preset, hanging still",
    )
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer playback factor; 0 is as fast as it computes. Pacing only",
    )
    harness.add_arguments(parser)
    options = parser.parse_args(argv)
    if options.random == 0 and not options.viewer:
        print("error: --random with no count needs --viewer to end it", file=sys.stderr)
        return 2

    description = tune_planner.plan_example.description()
    planner = tune_planner.Planner(
        description, tune_planner.plan_example.configure(options)
    )
    parameters, hydraulics = harness.settings(options.set)
    plant_fit, damping_scale = perturb_fit(
        k=options.k_scale, d=options.d_scale, lag_shift_s=options.lag_shift
    )
    if options.tau_v_scale is not None:
        plant_fit = scale_tau_v(plant_fit, options.tau_v_scale)
    if options.tau_v_offset is not None:
        plant_fit = offset_tau_v(plant_fit, options.tau_v_offset)
    ocp = harness.create_solver(
        parameters, hydraulics, options.rebuild, options.verbose_build
    )
    # Offline wall time is not the machine's; `--latency` stands for it.
    ocp.solve_budget_s = 10.0
    if options.relax_h:
        scale_h(ocp, options.relax_h, options.h_factor)
    ocp.set_payload(options.payload_mass, options.payload_com)
    plant = MujocoPlant(description, timestep=options.timestep)
    carried = (
        options.payload_mass if options.plant_payload is None else options.plant_payload
    )
    # The plant's COM, not the OCP's: `equilibrium` scores sway against where the
    # load actually hangs, so a COM mismatch must not move the reference too.
    carried_com = np.asarray(
        options.payload_com
        if options.plant_payload_com is None
        else options.plant_payload_com,
        float,
    )
    carry_payload(plant, carried, carried_com)
    payload = Payload(carried, carried_com, valid=True)

    def equilibrium(q_a):
        return planner.model.passive_equilibrium(q_a, payload)

    # The OCP's payload this time, not the plant's: this is the sway centre the
    # node's own `Cycle` solves against, so a COM mismatch must move it.
    hanging = Equilibrium(
        planner.model, Payload(ocp.payload_mass_kg, ocp.payload_com_m, valid=True)
    )

    plant.scale_planned_damping(damping_scale)
    start = anchor = tune_planner.start_of(planner, options.resettle_start)
    if options.start is not None:
        start = anchor = tune_planner.Start(
            q=np.asarray(options.start[:8]),
            dq_a=np.zeros(len(PLANNED)),
            dq_u=np.asarray(options.start[8:]),
        )
    rng = None if options.random is None else np.random.default_rng(options.seed)
    gate = SpaceGate()
    if options.viewer:
        plant.set_state(start.q)  # planning takes seconds; show the start meanwhile
        plant.open_viewer(options.realtime, key_callback=gate)
    said = False

    def coast():
        """Cycle on past the plan while the window is open and space unpressed."""
        nonlocal said
        if plant.viewer is None or not plant.viewer.is_running() or gate.pressed:
            return False
        if not said:
            said = True
            print("\nsettling -- space in the viewer for the next goal")
        return True

    moves, failed, scores = 0, False, []
    while True:
        plan = harness.plan_move(planner, start, options, rng)
        if plan is None:
            if rng is None or start is anchor:
                return 2
            print("no move from here; back to the pose this run opened at")
            start = anchor
            continue
        gate.pressed, said = False, False
        chain = harness.Chain(plant, start.q_tool, options, fit=plant_fit)
        # Only the first move starts swinging; later ones start where the last ended.
        swing = (
            None if options.start is None or moves else (start.q[PASSIVE], start.dq_u)
        )
        reason, score = fly(
            options, parameters, ocp, chain, equilibrium, plan, coast, swing, hanging
        )
        failed, moves = failed or bool(reason), moves + 1
        scores.append(score)
        if rng is None or (options.random and moves >= options.random):
            break
        if plant.viewer is not None and not gate.pressed:
            break  # window closed
        start = harness.next_start(plant, planner)
    if options.json:
        options.json.write_text(
            json.dumps(scores[0] if moves == 1 else scores, indent=1)
        )
    if plant.viewer is not None:
        print("close the viewer window to finish")
    plant.hold_viewer()
    return 1 if failed else 0


def fly(
    options, parameters, ocp, chain, equilibrium, plan, coast, swing=None, hanging=None
):
    """One move through the wire; `(failure reasons, scores)`. Coast cycles unscored."""
    plant = chain.plant
    Ts, N = float(parameters["Ts"]), int(parameters["horizon_length"])
    delay = float(parameters["sensor_to_valve_delay"])
    timing = cy.Timing(Ts, delay, jtc_period=chain.step_s)
    cycle = cy.Cycle(timing, hz.Grid(Ts, N), "active", parameters["dq_a_feedback"])
    cycle.ocp = ocp
    cycle.equilibrium = hanging
    bounds = cy.Bounds(
        float(parameters["max_clock_skew"]),
        float(parameters["max_reference_age"]),
        float(parameters["max_state_age"]),
        int(parameters["max_consecutive_failures"]),
        float(parameters["min_progress_rate"]),
        float(parameters["max_stall_time"]),
    )
    epoch_ns = int(EPOCH * 1e9)
    reference = hz.Knots.zeros(len(plan.time))
    reference.t[:], reference.q_a_ref[:], reference.dq_a_ref[:] = (
        plan.time,
        plan.q_a,
        plan.dq_a,
    )
    cycle.adopt_reference(reference, epoch_ns)
    cycle.adopt_path(harness.path_control(plan), plan.duration, epoch_ns)
    if options.cadence_phase:
        anchor = timing.anchor
        phase_ns = int(round(options.cadence_phase * 1e9))
        timing.anchor = lambda ns: anchor(ns + phase_ns)

    q0 = plan.q[0].copy()
    q0[PASSIVE] = equilibrium(q0[ACTUATED]) if swing is None else swing[0]
    x_init = np.zeros(cs.NX)
    x_init[: cs.PLANNED_DOF] = q0[PLANNED]
    x_init[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + 2] = q0[PASSIVE]
    if swing is not None:
        x_init[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + 2] = swing[1]
    chain.seed(x_init)

    joints = [f"j{i}" for i in range(cs.GENERALIZED_DOF)]
    h, ff = chain.step_s, chain.ff_scale
    ticks = int(round(Ts / h))
    events = {"silence": 0, "handback": 0}
    log = []
    # The JTC's total command `forward + correction` per tick -- `u`, i.e. Psi's
    # input, which is the signal `u_clamp_*` is the fitted domain of. Not a bound
    # the loop enforces: it clamps the PI branch alone, so the total is free to
    # leave the domain, which is the whole point of scoring it. Per tick, because
    # every tick is a separate command; `log` is per cycle, so `dq` beside it is
    # sampled 1-in-`ticks` and the two clamp ratios are not on one grid.
    u_tick = []
    solve_times = []
    # Per cycle, beside the milliseconds: solve time is ~linear in this and the
    # ms are the machine's, so a regression reads off iterations first (issue 223).
    qp_iterations = []
    # Which bound each solved cycle's horizon rides, and the cost split that says
    # whether the goal rows outweigh the sway rows (issue 244 (d)).
    riders, costs = [], []
    iteration_cap = int(ocp_problem.solver_tuning()["qp_solver_iter_max"])
    # Wall seconds from the cycle instant to the horizon on the wire -- the
    # margin the timer callback has left, solve included. Offline wall time is
    # not the machine's, but what is Python here is Python there (issue 187).
    tick_to_publish = []
    seconds = options.seconds or float(plan.duration) + 5.0
    diverged = False
    scored = int(round(seconds / Ts))

    def cycles():
        """Run the node's cycle then the JTC's ticks, once per `next`; `(solution, u0)`."""
        clock = EPOCH
        live = Playback.hold(clock, plant.q[PLANNED])
        last_receipt = clock
        flight = []  # (arrival, message)
        h_ns = int(round(h * 1e9))
        u0 = np.full(cs.PLANNED_DOF, np.nan)

        def send():
            """`Cycle.step`'s publisher: encode the horizon and put it on the wire."""
            nonlocal u0
            stamp = stamp_of(timing.stamp_ns() / 1e9)
            if options.wire == "161":
                message = wire_161(
                    cycle.horizon, joints, stamp, delay, cycle.last_horizon
                )
            else:
                message = hz.horizon_to_message(
                    cycle.horizon,
                    joints,
                    stamp,
                    lead=delay,
                    period=h,
                    in_flight=cycle.last_horizon,
                    in_flight_u=cycle.in_flight_command(),
                    linear=cycle.command_state,
                )
            flight.append((clock + options.latency, message))
            u0 = cycle.horizon.u[0, : cs.PLANNED_DOF].copy()
            tick_to_publish.append(time.perf_counter() - started)

        started = 0.0
        for k in itertools.count():
            now_ns = epoch_ns + k * int(round(Ts * 1e9))
            cycle.begin()
            q, dq = plant.q, plant.dq
            u0 = np.full(cs.PLANNED_DOF, np.nan)
            started = time.perf_counter()
            # Measurement is exact at the cycle instant here, so it is its own stamp.
            step = cycle.step(
                now_ns,
                now_ns,
                cy.Measurement(
                    q[ACTUATED], dq[ACTUATED], q[PASSIVE], dq[PASSIVE], 0.0, 0.0
                ),
                bounds,
                publish=send,
            )
            if step.solution is not None:
                solve_times.append(step.solution.solve_time_s)
                qp_iterations.append(step.solution.qp_iterations)
                riders.append(riding(step.solution, parameters["limits"]))
                # Right after the solve or not at all: `cost_terms` scores against
                # the parameter vector and path that solve was written with.
                if len(riders) % COST_DECIMATION == 1:
                    costs.append(ocp.cost_terms(step.solution, cycle.q_eq))
            if step.silence is not None:
                events["silence"] += 1
            elif step.verdict is None or not step.verdict.published:
                events["handback"] += 1

            for tick in range(ticks):
                # JTC update: take the newest arrived message, then sample at t and t + h.
                while flight and flight[0][0] <= clock + 1e-9:
                    live = Playback(
                        flight.pop(0)[1], clock, plant.q[PLANNED], plant.dq[PLANNED]
                    )
                    last_receipt = clock
                if clock - last_receipt > TOPIC_TIMEOUT:
                    live, last_receipt = Playback.hold(clock, plant.q[PLANNED]), np.inf
                q_ref, dq_ref, _ = live.sample(clock)
                _, dq_next, effort_next = live.sample(clock + h)
                command = chain.loop.step(
                    q_ref,
                    dq_ref,
                    plant.q[PLANNED],
                    plant.dq[PLANNED],
                    feedforward=ff * (dq_next - dq_ref) + effort_next,
                )
                # Stamped off `now_ns`, not the float `clock`: these are the ticks
                # `Timing.jtc_output` averages over `[measured - Ts, measured)` next
                # cycle, and an accumulated float would drop the one on the edge.
                timing.record_output(now_ns + tick * h_ns, command)
                u_tick.append(command)
                plant.drive(chain.actuator, chain.psi.apply(command), h)
                clock += h
            yield step.solution, u0

    run = cycles()
    for k, (solution, u0) in zip(range(scored), run):
        q, dq = plant.q, plant.dq
        sway = q[PASSIVE] - equilibrium(q[ACTUATED])
        status = -1 if solution is None else solution.status
        log.append(
            [
                (k + 1) * Ts,
                cycle.path_origin,
                *q[PLANNED],
                *sway,
                *dq[PLANNED],
                *dq[PASSIVE],
                status,
                *u0,
            ]
        )
        if not np.all(np.isfinite(q)) or np.abs(sway).max() > DIVERGED_SWAY:
            print(f"DIVERGED at t = {(k + 1) * Ts:.2f} s")
            diverged = True
            break

    log = np.array(log)
    nq = cs.PLANNED_DOF
    sway = log[:, 2 + nq : 4 + nq]
    late = sway[-int(round(3.0 / Ts)) :]
    # The damping half of issue 223 is scored on the *rate*, not the angle: the
    # published angle carries an uncalibrated offset and `dq_u_settled` is the
    # limit the tree gates on. Same window as `late`.
    rate_late = log[-int(round(3.0 / Ts)) :, 4 + 2 * nq : 6 + 2 * nq]
    # Issue 244's window: the plan's own end is arrival, and everything after it
    # is the hold the damping is scored in. `sway_rate_late` above is the tail of
    # whatever `--seconds` was, which is a different question once the hold is long.
    rate_u = log[:, 4 + 2 * nq : 6 + 2 * nq]
    arrival = float(plan.duration)
    after = log[:, 0] >= arrival
    window = (log[:, 0] >= arrival + AFTER_ARRIVAL_S[0]) & (
        log[:, 0] <= arrival + AFTER_ARRIVAL_S[1]
    )
    rate_2s = np.abs(rate_u[window]).max(axis=0) if window.any() else np.full(2, np.nan)
    tau_u = (
        envelope_tau(log[after, 0], rate_u[after])
        if after.sum() > int(round(SWAY_PERIOD_S / Ts)) + 4
        else [float("nan")] * 2
    )
    # 214's hunting metric: what the actuated axes are still doing in the hold.
    dq_a_hold = float(np.sqrt(np.mean(log[after, 2 + nq + 2 : 2 + 2 * nq + 2] ** 2)))
    u0 = log[:, -nq:]
    u0 = u0[np.all(np.isfinite(u0), axis=1)]
    # Ticks of the coast loop land in `u_tick` too, but only after this scoring.
    u = np.array(u_tick)
    print(
        f"wire {options.wire}, latency {1e3 * options.latency:.0f} ms, "
        f"cadence phase {1e3 * options.cadence_phase:.0f} ms, {len(log)} cycles\n"
        f"  final error      {np.abs(log[-1, 2 : 2 + nq] - plan.q[-1, PLANNED]).max():.4f}\n"
        f"  sway peak        {np.round(np.abs(sway).max(axis=0), 3)} rad\n"
        f"  sway last 3 s    {np.round(np.abs(late).max(axis=0), 3)} rad\n"
        f"  rate at +2 s     {np.round(rate_2s, 4)} rad/s (arrival {arrival:.2f} s)\n"
        f"  hold rate tau    {np.round(tau_u, 1)} s\n"
        f"  hold dq_a RMS    {dq_a_hold:.4f} rad/s\n"
        f"  bounds ridden    "
        + " ".join(
            f"{name} {np.mean([r[name] for r in riders]):.2f}"
            for name in (riders[0] if riders else {})
        )
        + "\n"
        f"  max|u_rot|       {np.abs(u[:, ROTATOR]).max():.3f} rad/s "
        f"(u_clamp {chain.low[ROTATOR]:+.2f}/{chain.high[ROTATOR]:+.2f})\n"
        f"  max|dq_rot|      {np.abs(log[:, 4 + nq + ROTATOR]).max():.3f} rad/s\n"
        f"  nonzero status   {int(np.sum(log[:, -nq - 1] > 0))}\n"
        f"  u0 reversals     {np.round(reversal_fraction(u0), 3)}\n"
        f"  silence/handback {events['silence']}/{events['handback']}"
    )
    if options.csv:
        names = ["t", "origin"] + [f"q{i}" for i in range(nq)] + ["sway0", "sway1"]
        names += [f"dq{i}" for i in range(nq)] + ["dq_u0", "dq_u1", "status"]
        names += [f"u0_{i}" for i in range(nq)]
        np.savetxt(options.csv, log, delimiter=",", header=",".join(names), comments="")
    # A command past `u_clamp_*` is out of Psi's fitted domain, so the compensator
    # extrapolates a spline past its data: an exit, and one the reversal gate does
    # not catch (a hunting run at 1/3 reversals, PI x4, issue 173, passed that
    # alone). Scored on `u`, not on the achieved `dq`: `u != dq`, the clamp is on
    # u, and `velocity_loop` applies it to the PI branch only. The `dq` test it
    # replaces (issue 190 scored it, and 149 of 166 failures were this gate) is an
    # overspeed check with no clamp of its own -- kept as a column, not an exit.
    dq = log[:, 2 + nq + 2 : 2 + 2 * nq + 2]
    past_clamp = bool(np.any(u > chain.high + 1e-3) or np.any(u < chain.low - 1e-3))
    overspeed = bool(np.any(dq > chain.high + 1e-3) or np.any(dq < chain.low - 1e-3))
    hunting = reversal_fraction(u0).max() >= REVERSAL_FRACTION
    reason = [
        n
        for n, bad in (
            ("diverged", diverged),
            ("hunting", hunting),
            ("past_clamp", past_clamp),
        )
        if bad
    ]
    scores = None
    if options.json:
        rate, _ = growth(log[:, 0], np.abs(sway).max(axis=1))
        scores = {
            "exit": reason[0] if reason else "ok",
            "cycles": len(log),
            "final_error": float(
                np.abs(log[-1, 2 : 2 + nq] - plan.q[-1, PLANNED]).max()
            ),
            "sway_peak": np.abs(sway).max(axis=0).tolist(),
            "sway_late": np.abs(late).max(axis=0).tolist(),
            "sway_rate_late": np.abs(rate_late).max(axis=0).tolist(),
            "arrival_s": arrival,
            "sway_rate_after_2s": rate_2s.tolist(),
            "sway_rate_tau_s": tau_u,
            "dq_a_rms_hold": dq_a_hold,
            # The gate: |u| over u_clamp on its side, per axis. > 1 is outside
            # Psi's fitted domain, i.e. `past_clamp`.
            "u_clamp_ratio": np.maximum(
                u.max(0) / chain.high, u.min(0) / chain.low
            ).tolist(),
            "u_peak": np.abs(u).max(0).tolist(),
            # The achieved joint velocity over the *command* clamp, per axis: an
            # overspeed ratio. `u_clamp_*` bounds u, not dq, so > 1 here says the
            # joint ran faster than u is allowed to ask, not that Psi was
            # extrapolated. `overspeed` is that test as a bool.
            "dq_clamp_ratio": np.maximum(
                dq.max(0) / chain.high, dq.min(0) / chain.low
            ).tolist(),
            "overspeed": overspeed,
            "reversals": reversal_fraction(u0).tolist(),
            "nonzero_status": int(np.sum(log[:, -nq - 1] > 0)),
            **events,
            "growth_per_s": rate,
            "u0_first": u0[:10, 0].tolist(),
            "solve_time_s": (
                np.percentile(solve_times, [50, 90, 100]).tolist()
                if solve_times
                else []
            ),
            "qp_iterations": (
                np.percentile(qp_iterations, [50, 90, 100]).tolist()
                if qp_iterations
                else []
            ),
            # Share of cycles that stopped at `qp_solver_iter_max` rather than
            # converging. Not the same as `nonzero_status`: a capped QP answers
            # `ACADOS_MAXITER`, which the NLP accepts, so it publishes its own
            # truncated step and reads as healthy -- the only way to see the
            # cap being hit is to count it (issue 226).
            "qp_iter_capped": (
                float(np.mean(np.asarray(qp_iterations) >= iteration_cap))
                if qp_iterations
                else 0.0
            ),
            # p50/p90/p100, cycle instant to publish
            "tick_to_publish_s": (
                np.percentile(tick_to_publish, [50, 90, 100]).tolist()
                if tick_to_publish
                else []
            ),
            # Issue 244 (d): the share of solved cycles each bound was ridden on,
            # and the cost split averaged over `COST_DECIMATION`-th cycles.
            "active": {
                name: float(np.mean([r[name] for r in riders]))
                for name in (riders[0] if riders else {})
            },
            "cost": {
                name: float(np.mean([getattr(c, name) for c in costs]))
                for name in (dataclasses.asdict(costs[0]) if costs else {})
            },
        }
    # The viewer's coast: same chain, past the plan, unscored.
    while not diverged and coast() and np.all(np.isfinite(plant.q)):
        next(run)
    return reason, scores


if __name__ == "__main__":
    sys.exit(leave(main()))
