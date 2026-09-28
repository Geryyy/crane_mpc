#!/usr/bin/env python3
"""
The node's own `Cycle`, closed through the wire: horizon message -> JTC -> MuJoCo.

`sim_chain.py` and `closedloop.py` hand `u` to the plant as a zero-order hold,
so neither sees the horizon encoding, the JTC's interpolation, its one-tick
feedforward lookahead, delivery latency or the cadence phase -- the five
faults behind the 2026-09-25 Gazebo divergence (issue 171). This runs
`Cycle` in the node's order, writes each horizon with `hz.horizon_to_message`
and plays it as `trajectory.cpp` does into sim_chain's velocity loop, C3
actuator and MuJoCo plant. Nothing is published.

    ./scripts/trials/wire_chain.py --goal out --latency 0.02
    ./scripts/trials/wire_chain.py --goal out --wire 161         # must diverge
    ./scripts/trials/wire_chain.py --goal out --cadence-phase 0.03  # must diverge

Unknown flags go to sim_chain / mpc_a2b (plant mismatch included). Exit 1 on
divergence or hunting (`crane_mpc.hunting`: u0 reversing on half the cycles).

Levers: `--set weights.du=30 --set dq_a_feedback=false` overrides crane_mpc.yaml
(yaml value; a scalar on a list key fills every axis). A key in
`solver.export_key` (Ts, horizon_length, sensor_to_valve_delay, ...) re-exports
the solver: minutes, and unlocked, so run that setting once before a parallel
sweep. Payload: `--payload-mass` is what the OCP believes, `--plant-payload`
what MuJoCo carries (default: the same), both at `--payload-com` in K8.

Not the machine: measurement is exact at the cycle instant, the solve takes no
time (latency is delivery only; budget off), the tool axis is not driven, and
the JTC's tolerance checks, goal handling and angle wraparound are skipped.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim_chain  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from crane_model.model import Payload  # noqa: E402
from crane_model.symbolic import PAYLOAD_MOUNT_LINK  # noqa: E402
from crane_mpc import cycle as cy  # noqa: E402
from crane_mpc import horizon as hz  # noqa: E402
from crane_mpc.hunting import REVERSAL_FRACTION, reversal_fraction  # noqa: E402
from growth import growth  # noqa: E402
from sim_chain import (  # noqa: E402
    ACTUATED_INDICES,
    PASSIVE_INDICES,
    PLANNED,
    cs,
    mpc_a2b,
    tune_planner,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

PASSIVE = list(PASSIVE_INDICES)
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


def override(parameters, assignment):
    """Apply one `--set key.path=value` to the loaded parameters."""
    key, _, text = assignment.partition("=")
    *path, leaf = key.split(".")
    node = parameters
    for part in path:
        node = node[part]
    if leaf not in node:
        raise SystemExit(f"--set {key}: no such parameter")
    value = yaml.safe_load(text)
    if isinstance(node[leaf], list) and not isinstance(value, list):
        value = [value] * len(node[leaf])
    node[leaf] = value


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
    parser.add_argument("--csv", type=Path, default=None, help="per-cycle log")
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
        "--pi-scale", type=float, default=1.0, help="multiply the JTC's p, i, d"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="yaml lever"
    )
    parser.add_argument(
        "--plant-payload", type=float, default=None, help="kg; default --payload-mass"
    )
    parser.add_argument("--json", type=Path, default=None, help="scores")
    mine, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    options, forwarded = sim_chain.arguments(rest)
    mpc = mpc_a2b.parse_arguments(forwarded)

    description = tune_planner.plan_example.description()
    planner = tune_planner.Planner(
        description, tune_planner.plan_example.configure(options)
    )
    parameters, hydraulics = mpc_a2b.load_settings(mpc)
    xml = (
        mpc_a2b.export_ocp.DEFAULT_DESCRIPTIONS / mpc_a2b.export_ocp.DESCRIPTION
    ).read_text()
    shipped = sim_chain.ocp_runtime.export_key(parameters, hydraulics, xml)
    for assignment in mine.set:
        override(parameters, assignment)
    moved = sim_chain.ocp_runtime.export_key(parameters, hydraulics, xml)
    moved = [key for key in shipped if moved[key] != shipped[key]]
    if moved:
        print(f"warning: --set moves compiled {moved}; re-exports", file=sys.stderr)
    plant_fit, damping_scale = sim_chain.perturb_fit(
        k=options.k_scale, d=options.d_scale, lag_shift_s=options.lag_shift
    )
    ocp, model, _ = mpc_a2b.create_solver(mpc, parameters, hydraulics)
    # Offline wall time is not the machine's; `--latency` stands for it.
    ocp.solve_budget_s = 10.0
    ocp.set_payload(mpc.payload_mass, mpc.payload_com)
    plant = sim_chain.MujocoPlant(description, timestep=options.timestep)
    carried = mpc.payload_mass if mine.plant_payload is None else mine.plant_payload
    carry_payload(plant, carried, mpc.payload_com)
    payload = Payload(carried, np.asarray(mpc.payload_com, float), valid=True)

    def equilibrium(q_a):
        return planner.model.passive_equilibrium(q_a, payload)

    plant.scale_planned_damping(damping_scale)
    start = tune_planner.start_of(planner, options.resettle_start)
    rng = None if options.random is None else np.random.default_rng(options.seed)
    plan = sim_chain.plan_move(planner, start, options, rng)
    if plan is None:
        return 2
    path = sim_chain.planned_path(plan)
    chain = sim_chain.Chain(plant, model, start.q_tool, options, fit=plant_fit)
    chain.loop = type(chain.loop)(
        [
            dataclasses.replace(
                a, p=a.p * mine.pi_scale, i=a.i * mine.pi_scale, d=a.d * mine.pi_scale
            )
            for a in chain.loop.gains
        ],
        chain.step_s,
        continuous=chain.wrap,
    )

    Ts, N = float(parameters["Ts"]), int(parameters["horizon_length"])
    delay = float(parameters["sensor_to_valve_delay"])
    cycle = cy.Cycle(Ts, hz.Grid(Ts, N), "active", delay, parameters["dq_a_feedback"])
    cycle.ocp = ocp
    epoch_ns = int(EPOCH * 1e9)
    reference = hz.Knots.zeros(len(plan.time))
    reference.t[:], reference.q_a_ref[:], reference.dq_a_ref[:] = (
        plan.time,
        plan.q_a,
        plan.dq_a,
    )
    cycle.adopt_reference(reference, epoch_ns)
    cycle.adopt_path(path.control, plan.duration, epoch_ns)
    if mine.cadence_phase:
        anchor = cycle.anchor_cadence
        phase_ns = int(round(mine.cadence_phase * 1e9))
        cycle.anchor_cadence = lambda ns: anchor(ns + phase_ns)

    q0 = plan.q[0].copy()
    q0[PASSIVE] = equilibrium(q0[list(ACTUATED_INDICES)])
    x_init = np.zeros(cs.NX)
    x_init[: cs.K_PLANNED_DOF] = q0[PLANNED]
    x_init[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + 2] = q0[PASSIVE]
    chain._seed(x_init)

    joints = [f"j{i}" for i in range(cs.K_GENERALIZED_DOF)]
    h, ff = chain.step_s, chain.ff_scale
    ticks = int(round(Ts / h))
    clock = EPOCH
    live = Playback.hold(clock, plant.q[PLANNED])
    last_receipt = clock
    flight = []  # (arrival, message)
    outputs = []  # JTC output over the last cycle, for `cycle.jtc_output`
    events = {"silence": 0, "handback": 0}
    log = []
    seconds = mine.seconds or float(plan.duration) + 5.0
    diverged = False
    for k in range(int(round(seconds / Ts))):
        now_ns = epoch_ns + k * int(round(Ts * 1e9))
        cycle.begin()
        q, dq = plant.q, plant.dq
        silence = cycle.gates(
            now_ns,
            float(parameters["max_clock_skew"]),
            float(parameters["max_reference_age"]),
        )
        if silence is None:
            silence = cycle.read_state(
                cy.Measurement(
                    q[list(ACTUATED_INDICES)],
                    dq[list(ACTUATED_INDICES)],
                    q[PASSIVE],
                    dq[PASSIVE],
                    0.0,
                    0.0,
                ),
                float(parameters["max_state_age"]),
            )
        if silence is None:
            cycle.jtc_output = (
                np.mean(outputs, axis=0) if len(outputs) == ticks else None
            )
            silence = cycle.propagate()
        solution, u0 = None, np.full(cs.K_PLANNED_DOF, np.nan)
        if silence is not None:
            cycle.stay_silent(silence.why)
            events["silence"] += 1
        else:
            solution, refusal = cycle.solve()
            verdict = (
                None
                if refusal
                else cycle.ladder(
                    solution,
                    int(parameters["max_consecutive_failures"]),
                    ocp.solve_budget_s,
                )
            )
            if verdict is None or not verdict.published:
                events["handback"] += 1
            else:
                stamp = stamp_of(
                    (cycle.next_first_knot_ns - cycle.command_delay * 1e9) / 1e9
                )
                if mine.wire == "161":
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
                        in_flight_u=cycle.applied_inputs[0],
                    )
                flight.append((clock + mine.latency, message))
                u0 = cycle.horizon.u[0, : cs.K_PLANNED_DOF].copy()
                cycle.advance(
                    solution,
                    now_ns,
                    float(parameters["min_progress_rate"]),
                    float(parameters["max_stall_time"]),
                )
                cycle.prepare_next(solution)

        outputs = []
        for _ in range(ticks):
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
            outputs.append(command)
            plant.drive(chain.actuator, chain.psi.apply(command), h)
            clock += h

        q, dq = plant.q, plant.dq
        sway = q[PASSIVE] - equilibrium(q[list(ACTUATED_INDICES)])
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
        f"wire {mine.wire}, latency {1e3 * mine.latency:.0f} ms, "
        f"cadence phase {1e3 * mine.cadence_phase:.0f} ms, {len(log)} cycles\n"
        f"  final error      {np.abs(log[-1, 2 : 2 + nq] - plan.q[-1, PLANNED]).max():.4f}\n"
        f"  sway peak        {np.round(np.abs(sway).max(axis=0), 3)} rad\n"
        f"  sway last 3 s    {np.round(np.abs(late).max(axis=0), 3)} rad\n"
        f"  max|dq_rot|      {np.abs(log[:, 4 + nq + ROTATOR]).max():.3f} rad/s "
        f"(u_clamp {chain.low[ROTATOR]:+.2f}/{chain.high[ROTATOR]:+.2f})\n"
        f"  nonzero status   {int(np.sum(log[:, -nq - 1] > 0))}\n"
        f"  u0 reversals     {np.round(reversal_fraction(u0), 3)}\n"
        f"  silence/handback {events['silence']}/{events['handback']}"
    )
    if mine.csv:
        names = ["t", "origin"] + [f"q{i}" for i in range(nq)] + ["sway0", "sway1"]
        names += [f"dq{i}" for i in range(nq)] + ["dq_u0", "dq_u1", "status"]
        names += [f"u0_{i}" for i in range(nq)]
        np.savetxt(mine.csv, log, delimiter=",", header=",".join(names), comments="")
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
    if mine.json:
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
        }
        mine.json.write_text(json.dumps(scores, indent=1))
    return 1 if reason else 0


if __name__ == "__main__":
    sys.exit(sim_chain.leave(main()))
