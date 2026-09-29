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

Plan, plant mismatch and solver flags are `harness.add_arguments`'. Exit 1 on
divergence or hunting (`hunting.py`: u0 reversing on half the cycles).

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
        zero = np.zeros(cs.K_GENERALIZED_DOF)
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
        "--json", type=Path, default=None, help="scores; a list per move if chained"
    )
    parser.add_argument("--viewer", action="store_true", help="MuJoCo passive viewer")
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
    ocp = harness.create_solver(
        parameters, hydraulics, options.rebuild, options.verbose_build
    )
    # Offline wall time is not the machine's; `--latency` stands for it.
    ocp.solve_budget_s = 10.0
    ocp.set_payload(options.payload_mass, options.payload_com)
    plant = MujocoPlant(description, timestep=options.timestep)
    carried = (
        options.payload_mass if options.plant_payload is None else options.plant_payload
    )
    carry_payload(plant, carried, options.payload_com)
    payload = Payload(carried, np.asarray(options.payload_com, float), valid=True)

    def equilibrium(q_a):
        return planner.model.passive_equilibrium(q_a, payload)

    plant.scale_planned_damping(damping_scale)
    start = anchor = tune_planner.start_of(planner, options.resettle_start)
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
        reason, score = fly(options, parameters, ocp, chain, equilibrium, plan, coast)
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


def fly(options, parameters, ocp, chain, equilibrium, plan, coast):
    """One move through the wire; `(failure reasons, scores)`. Coast cycles unscored."""
    plant = chain.plant
    Ts, N = float(parameters["Ts"]), int(parameters["horizon_length"])
    delay = float(parameters["sensor_to_valve_delay"])
    timing = cy.Timing(Ts, delay, jtc_period=chain.step_s)
    cycle = cy.Cycle(timing, hz.Grid(Ts, N), "active", parameters["dq_a_feedback"])
    cycle.ocp = ocp
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
    q0[PASSIVE] = equilibrium(q0[ACTUATED])
    x_init = np.zeros(cs.NX)
    x_init[: cs.K_PLANNED_DOF] = q0[PLANNED]
    x_init[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + 2] = q0[PASSIVE]
    chain.seed(x_init)

    joints = [f"j{i}" for i in range(cs.K_GENERALIZED_DOF)]
    h, ff = chain.step_s, chain.ff_scale
    ticks = int(round(Ts / h))
    events = {"silence": 0, "handback": 0}
    log = []
    solve_times = []
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
        u0 = np.full(cs.K_PLANNED_DOF, np.nan)

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
            u0 = cycle.horizon.u[0, : cs.K_PLANNED_DOF].copy()
            tick_to_publish.append(time.perf_counter() - started)

        started = 0.0
        for k in itertools.count():
            now_ns = epoch_ns + k * int(round(Ts * 1e9))
            cycle.begin()
            q, dq = plant.q, plant.dq
            u0 = np.full(cs.K_PLANNED_DOF, np.nan)
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
    nq = cs.K_PLANNED_DOF
    sway = log[:, 2 + nq : 4 + nq]
    late = sway[-int(round(3.0 / Ts)) :]
    u0 = log[:, -nq:]
    u0 = u0[np.all(np.isfinite(u0), axis=1)]
    print(
        f"wire {options.wire}, latency {1e3 * options.latency:.0f} ms, "
        f"cadence phase {1e3 * options.cadence_phase:.0f} ms, {len(log)} cycles\n"
        f"  final error      {np.abs(log[-1, 2 : 2 + nq] - plan.q[-1, PLANNED]).max():.4f}\n"
        f"  sway peak        {np.round(np.abs(sway).max(axis=0), 3)} rad\n"
        f"  sway last 3 s    {np.round(np.abs(late).max(axis=0), 3)} rad\n"
        f"  max|dq_rot|      {np.abs(log[:, 4 + nq + ROTATOR]).max():.3f} rad/s "
        f"(u_clamp {chain.low[ROTATOR]:+.2f}/{chain.high[ROTATOR]:+.2f})\n"
        f"  nonzero status   {int(np.sum(log[:, -nq - 1] > 0))}\n"
        f"  u0 reversals     {np.round(reversal_fraction(u0), 3)}\n"
        f"  silence/handback {events['silence']}/{events['handback']}"
    )
    if options.csv:
        names = ["t", "origin"] + [f"q{i}" for i in range(nq)] + ["sway0", "sway1"]
        names += [f"dq{i}" for i in range(nq)] + ["dq_u0", "dq_u1", "status"]
        names += [f"u0_{i}" for i in range(nq)]
        np.savetxt(options.csv, log, delimiter=",", header=",".join(names), comments="")
    # A joint past its command clamp is out of Psi's domain: a hunting run at 1/3
    # reversals (PI x4, issue 173) passed the reversal gate alone.
    dq = log[:, 2 + nq + 2 : 2 + 2 * nq + 2]
    past_clamp = bool(np.any(dq > chain.high + 1e-3) or np.any(dq < chain.low - 1e-3))
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
            # |dq| over the clamp on its side, per axis: > 1 is outside Psi's domain
            "dq_clamp_ratio": np.maximum(
                dq.max(0) / chain.high, dq.min(0) / chain.low
            ).tolist(),
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
            # p50/p90/p100, cycle instant to publish
            "tick_to_publish_s": (
                np.percentile(tick_to_publish, [50, 90, 100]).tolist()
                if tick_to_publish
                else []
            ),
        }
    # The viewer's coast: same chain, past the plan, unscored.
    while not diverged and coast() and np.all(np.isfinite(plant.q)):
        next(run)
    return reason, scores


if __name__ == "__main__":
    sys.exit(leave(main()))
