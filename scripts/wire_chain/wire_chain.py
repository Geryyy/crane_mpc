#!/usr/bin/env python3
"""
The node's own `Cycle`, closed through the wire: horizon message -> JTC -> MuJoCo.

    ./scripts/wire_chain/wire_chain.py --goal out --latency 0.02
    ./scripts/wire_chain/wire_chain.py --goal out --wire 161            # must diverge
    ./scripts/wire_chain/wire_chain.py --goal out --cadence-phase 0.03  # must diverge
    ./scripts/wire_chain/wire_chain.py --goal out --viewer --realtime 0
    ./scripts/wire_chain/wire_chain.py --random --viewer                # space: next move
    ./scripts/wire_chain/wire_chain.py --goal out --set weights.du=30   # override crane_mpc.yaml

Plan, plant mismatch and solver flags: `harness.add_arguments`. A `solver.export_key`
setting (Ts, horizon_length, ...) re-exports the solver: minutes, unlocked.
Exit 1 on divergence, hunting (`hunting.py`) or a JTC command outside `u_clamp_*`.

Not the machine: measured at the cycle instant, solve takes no time (latency is delivery
only), tool axis not driven, JTC tolerance checks, goal handling and wraparound skipped.
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
from crane_model.viewing import Markers, SpaceGate, canonical_rows
from crane_mpc import cycle as cy
from crane_mpc import horizon as hz
from crane_mpc import problem as ocp_problem
from crane_mpc.equilibrium import Equilibrium
from crane_mpc.solver import SLACK_NOTICEABLE
from growth import growth
from harness import ACTUATED, PASSIVE, PLANNED, cs, tune_planner
from hunting import REVERSAL_FRACTION, reversal_fraction
from plotjuggler import PORT, PlotJuggler
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ROTATOR = 4
TOPIC_TIMEOUT = 0.24  # a2b JTC yaml topic_timeout, pzs100 sim
EPOCH = 1.0  # sim clock origin; zero stamp reads as "start on receipt"
DIVERGED_SWAY = 1.2  # rad
SWAY_PERIOD_S = 2.34  # `limits.du_max` is derived from it
AFTER_ARRIVAL_S = (1.5, 2.5)  # "2 s after arrival" +-0.5 s, ~1/5 sway period
ACTIVE_MARGIN = 0.99  # not equality: QP tolerance, du_max reached by a ramp
COST_DECIMATION = 10  # each cost split costs N+1 residual evaluations
RELAXED = 10.0  # --relax-h widening; 1e6 hurts HPIPM conditioning (issue 248)
PLAN_RGBA = (0.15, 0.85, 0.45, 0.6)
HORIZON_RGBA = (1.0, 0.45, 0.05, 0.9)
#: `gz_system_actuators` measurement params, in `Sensor`'s order
NOISE_KEYS = (
    "position_noise_std",
    "position_noise_tau",
    "velocity_noise_std",
    "velocity_noise_tau",
    "position_bias",
    "velocity_bias",
    "sample_rate",
)


def sensor_noise() -> np.ndarray:
    """(8, 7) noise params per canonical joint, off the Gazebo description; 0 if absent."""
    import xml.etree.ElementTree as ET

    import xacro
    from ament_index_python.packages import get_package_share_directory
    from crane_model.conventions import canonical_joints

    source = Path(get_package_share_directory("epsilon_crane_description"))
    document = xacro.process_file(
        str(source / "urdf" / "timber_loader_AIT.urdf.xacro"),
        mappings={
            "gazebo": "true",
            "rigid": "true",
            "sim_hydraulics": "true",
            "tool": "pzs100_description",  # = problem.DESCRIPTION
        },
    )
    joints = {
        j.get("name"): {p.get("name"): p.text for p in j.iter("param")}
        for system in ET.fromstring(document.toxml()).iter("ros2_control")
        for j in system.iter("joint")
    }
    return np.array(
        [
            [float(joints.get(name, {}).get(key) or 0.0) for key in NOISE_KEYS]
            for name in canonical_joints()
        ]
    )


def mrad(values) -> str:
    return " ".join(f"{1e3 * v:+5.0f}" for v in values)


class Sensor:
    """`gz_system_actuators`' measurement model: Gauss-Markov noise, bias, sample-hold."""

    def __init__(self, plant, noise, scale: float, dt: float, seed: int = 0) -> None:
        self.plant, self.dt = plant, dt
        q_std, q_tau, v_std, v_tau, self.q_bias, self.v_bias, rate = noise.T * [
            [scale],
            [1],
            [scale],
            [1],
            [scale],
            [scale],
            [1],
        ]
        self.q_a, self.q_k = self._gm(q_std, q_tau)
        self.v_a, self.v_k = self._gm(v_std, v_tau)
        rate = np.where(rate > 0, rate, 1.0 / dt)  # 0 Hz = every tick
        self.every = np.round(1.0 / (rate * dt)).astype(int)
        self.rng = np.random.default_rng(seed)
        self.q_noise = self.v_noise = np.zeros(len(noise))
        self.ticks = 0
        self.q, self.dq = plant.q, plant.dq

    def _gm(self, std, tau):
        a = np.where(tau > 0, np.exp(-self.dt / np.where(tau > 0, tau, 1)), 0.0)
        return a, std * np.sqrt(1 - a**2)

    def tick(self) -> None:
        """Advance the noise, resample where due."""
        n = len(self.q_bias)
        self.q_noise = self.q_a * self.q_noise + self.q_k * self.rng.standard_normal(n)
        self.v_noise = self.v_a * self.v_noise + self.v_k * self.rng.standard_normal(n)
        due = self.ticks % self.every == 0
        self.ticks += 1
        q = self.plant.q + self.q_bias + self.q_noise
        dq = self.plant.dq + self.v_bias + self.v_noise
        self.q, self.dq = np.where(due, q, self.q), np.where(due, dq, self.dq)


def envelope_tau(t, rate):
    """Per-column decay tau of |rate|'s rolling-max envelope; inf no decay, nan at zero."""
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
    """Scale constraint 6 (cylinder force) and/or 7 (pump flow) allowance; <1 tightens."""
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
    """Bounds this horizon rides: hard boxes by proximity, soft rows by slack taken."""
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
        # before-trajectory point: measured state at receipt, effort 0
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
    """Plant tau_v times a per-axis multiplier, 0 stays 0; OCP keeps the shipped fit."""
    scale = np.asarray(scale, float)
    if not np.all(np.isfinite(scale) & (scale > 0.0)):
        raise SystemExit(f"--tau-v-scale wants positive finite multipliers: {scale}")
    return dataclasses.replace(fit, tau_v=tuple(np.asarray(fit.tau_v, float) * scale))


def offset_tau_v(fit, offset):
    """Plant tau_v plus a per-axis offset, floored at 0 (negative reads as no lag)."""
    offset = np.asarray(offset, float)
    if not np.all(np.isfinite(offset)):
        raise SystemExit(f"--tau-v-offset wants finite seconds: {offset}")
    lagged = np.maximum(np.asarray(fit.tau_v, float) + offset, 0.0)
    return dataclasses.replace(fit, tau_v=tuple(lagged))


def wire_161(horizon, joints, stamp, lead, in_flight):
    """Issue 161's encoding: knot i at lead + t_i, in-flight knot at 0, u - dq per knot."""
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
        "--topic-timeout",
        type=float,
        default=TOPIC_TIMEOUT,
        help="s; JTC holds after this long without a horizon",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        help="fraction of horizons lost on the wire; lab 2026-10-12 got ~1.3 of 8.3 Hz, i.e. 0.84",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="per-cycle log, last move's"
    )
    # regression checks: each reverts a fix and must diverge
    parser.add_argument("--wire", choices=("today", "161"), default="today")
    parser.add_argument(
        "--cadence-phase",
        type=float,
        default=0.0,
        help="s; stamp every horizon this far off the measurement, as the "
        "once-anchored grid did",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="divide the duration handed to adopt_path, i.e. compress the "
        "progress reference in time. 1 is the plan's own pace (issue 250)",
    )
    parser.add_argument(
        "--mean-pace",
        action="store_true",
        help="hand adopt_path no time law, so the progress row is priced "
        "against the plan's mean pace -- every cycle before issue 253",
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
        "--pj",
        type=int,
        nargs="?",
        const=PORT,
        default=None,
        metavar="PORT",
        help=f"stream each cycle to PlotJuggler's UDP server (JSON); default {PORT}",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="x the Gazebo description's joint noise and bias; 0 = exact",
    )
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
    ocp.solve_budget_s = (
        10.0  # offline wall time is not the machine's; --latency stands in
    )
    if options.relax_h:
        scale_h(ocp, options.relax_h, options.h_factor)
    ocp.set_payload(options.payload_mass, options.payload_com)
    plant = MujocoPlant(description, timestep=options.timestep)
    carried = (
        options.payload_mass if options.plant_payload is None else options.plant_payload
    )
    # plant's COM, not the OCP's: sway is scored where the load hangs
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

    # OCP's payload: the sway centre the node's `Cycle` solves against
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
    markers = Markers(plant) if options.viewer else None
    noise = sensor_noise() if options.noise_scale else np.zeros((cs.GENERALIZED_DOF, 7))
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
        # only the first move starts swinging
        swing = (
            None if options.start is None or moves else (start.q[PASSIVE], start.dq_u)
        )
        if markers is not None:
            markers.path("plan", plan.q, PLAN_RGBA, width=0.02)
        reason, score = fly(
            options,
            parameters,
            ocp,
            chain,
            equilibrium,
            plan,
            coast,
            swing,
            hanging,
            Sensor(plant, noise, options.noise_scale, chain.step_s),
            markers,
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
    options,
    parameters,
    ocp,
    chain,
    equilibrium,
    plan,
    coast,
    swing=None,
    hanging=None,
    sensor=None,
    markers=None,
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
    # --pace compresses the progress reference in time only (curve and knots untouched)
    cycle.adopt_path(
        harness.path_control(plan),
        plan.duration / options.pace,
        epoch_ns,
        None if options.mean_pace else plan.timing.s,
    )
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
    sensor.tick()
    status_line = sys.stderr.isatty()
    pj = None if options.pj is None else PlotJuggler(options.pj)

    joints = [f"j{i}" for i in range(cs.GENERALIZED_DOF)]
    h, ff = chain.step_s, chain.ff_scale
    ticks = int(round(Ts / h))
    events = {"silence": 0, "handback": 0, "dropped": 0}
    loss = np.random.default_rng(1)
    log = []
    # per-tick JTC command (u, Psi's input); scored against u_clamp_*, not enforced
    u_tick = []
    solve_times = []
    # per cycle; solve ms are ~linear in it
    qp_iterations = []
    # bound ridden per cycle; cost split (goal vs sway rows)
    riders, costs = [], []
    iteration_cap = int(ocp_problem.solver_tuning()["qp_solver_iter_max"])
    # wall s from cycle instant to horizon on the wire, solve included
    tick_to_publish = []
    # MPC ask per horizon: knot 0 (JTC plays next) and whole-horizon max
    asked_now, asked_any = [], []
    seconds = options.seconds or float(plan.duration) + 5.0
    diverged = False
    scored = int(round(seconds / Ts))

    def cycles():
        """Run the node's cycle then the JTC's ticks, once per `next`; `(solution, u0)`."""
        clock = EPOCH
        live = Playback.hold(clock, sensor.q[PLANNED])
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
                message = hz.truncated(message, float(parameters["horizon_publish_s"]))
            if loss.random() < options.drop:
                events["dropped"] += 1
            else:
                flight.append((clock + options.latency, message))
            u0 = cycle.horizon.u[0, : cs.PLANNED_DOF].copy()
            asked = np.abs(cycle.horizon.dq_a_ref)
            asked_now.append(asked[0])
            asked_any.append(asked.max(0))
            tick_to_publish.append(time.perf_counter() - started)

        started = 0.0
        for k in itertools.count():
            now_ns = epoch_ns + k * int(round(Ts * 1e9))
            cycle.begin()
            sensor.tick()  # the cycle instant is the JTC's tick 0
            q, dq = sensor.q, sensor.dq
            u0 = np.full(cs.PLANNED_DOF, np.nan)
            started = time.perf_counter()
            # measured at the cycle instant: stamp = now
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
                # right after the solve: cost_terms needs that solve's parameters and path
                if len(riders) % COST_DECIMATION == 1:
                    costs.append(ocp.cost_terms(step.solution, cycle.q_eq))
            if markers is not None and cycle.horizon is not None:
                rows = canonical_rows(
                    cycle.horizon.q_a_ref[:, : cs.PLANNED_DOF],
                    cycle.horizon.q_u_ref,
                    chain.q_tool,
                )
                markers.path("horizon", rows, HORIZON_RGBA, width=0.05, samples=12)
            t = (now_ns - epoch_ns) / 1e9
            sway = plant.q[PASSIVE] - equilibrium(plant.q[ACTUATED])
            if pj is not None:
                dq_ref = (
                    cycle.horizon.dq_a_ref[0, : cs.PLANNED_DOF]
                    if cycle.horizon is not None
                    else np.full(cs.PLANNED_DOF, np.nan)
                )
                pj.cycle(
                    t,
                    plant,
                    sensor,
                    sway,
                    step.solution,
                    u0,
                    dq_ref,
                    (PLANNED, PASSIVE),
                )
            if status_line:
                solved = step.solution
                print(
                    f"\r{'move' if t < plan.duration else 'hold'} t {t:6.2f}/"
                    f"{plan.duration:.1f} s  sway {mrad(sway)} mrad"
                    f"  rate {mrad(plant.dq[PASSIVE])} mrad/s"
                    + (
                        "  no solve"
                        if solved is None
                        else f"  solve {1e3 * solved.solve_time_s:4.0f} ms"
                        f"  qp {solved.qp_iterations:3d}  status {solved.status}"
                    )
                    + "   ",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            if step.silence is not None:
                events["silence"] += 1
            elif step.verdict is None or not step.verdict.published:
                events["handback"] += 1

            for tick in range(ticks):
                # JTC update: newest arrived message, sample at t and t + h
                if tick:
                    sensor.tick()
                while flight and flight[0][0] <= clock + 1e-9:
                    live = Playback(
                        flight.pop(0)[1], clock, sensor.q[PLANNED], sensor.dq[PLANNED]
                    )
                    last_receipt = clock
                if clock - last_receipt > options.topic_timeout:
                    live, last_receipt = Playback.hold(clock, sensor.q[PLANNED]), np.inf
                q_ref, dq_ref, _ = live.sample(clock)
                _, dq_next, effort_next = live.sample(clock + h)
                command = chain.loop.step(
                    q_ref,
                    dq_ref,
                    sensor.q[PLANNED],
                    sensor.dq[PLANNED],
                    feedforward=ff * (dq_next - dq_ref) + effort_next,
                )
                # off now_ns, not float clock: Timing.jtc_output averages the edge tick
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
            print(f"\nDIVERGED at t = {(k + 1) * Ts:.2f} s")
            diverged = True
            break

    if status_line:
        print(file=sys.stderr)
    log = np.array(log)
    nq = cs.PLANNED_DOF
    sway = log[:, 2 + nq : 4 + nq]
    late = sway[-int(round(3.0 / Ts)) :]
    # damping scored on rate, not angle: published angle has an uncalibrated offset
    rate_late = log[-int(round(3.0 / Ts)) :, 4 + 2 * nq : 6 + 2 * nq]
    # arrival = plan end; damping is scored in the hold after it
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
    # hunting metric: what the actuated axes still do in the hold
    dq_a_hold = float(np.sqrt(np.mean(log[after, 2 + nq + 2 : 2 + 2 * nq + 2] ** 2)))
    u0 = log[:, -nq:]
    u0 = u0[np.all(np.isfinite(u0), axis=1)]
    # coast ticks land in u_tick after this
    u = np.array(u_tick)
    # peak speed per axis over the whole log; lagging moves run past plan end
    plan_peak = np.abs(plan.dq_a[:, :nq]).max(0)  # planned five lead the six
    # peak/mean of s(t) speed; ceiling on achieved/plan under --mean-pace
    plan_pace = float(np.abs(plan.timing.speed).max() * plan.duration)
    dq_peak = np.abs(log[:, 2 + nq + 2 : 2 + 2 * nq + 2]).max(0)

    def peak(rows):
        return np.asarray(rows)[:, :nq].max(0) if rows else np.zeros(nq)

    now_peak, any_peak = peak(asked_now), peak(asked_any)
    moving = plan_peak > 1e-6

    def over_plan(values):
        return np.divide(values, plan_peak, out=np.full(nq, np.nan), where=moving)

    ratio, now_ratio, any_ratio = (
        over_plan(dq_peak),
        over_plan(now_peak),
        over_plan(any_peak),
    )
    print(
        f"wire {options.wire}, latency {1e3 * options.latency:.0f} ms, "
        f"dropped {events['dropped']} horizons, "
        f"cadence phase {1e3 * options.cadence_phase:.0f} ms, {len(log)} cycles\n"
        f"  final error      {np.abs(log[-1, 2 : 2 + nq] - plan.q[-1, PLANNED]).max():.4f}\n"
        f"  sway peak        {np.round(np.abs(sway).max(axis=0), 3)} rad\n"
        f"  sway last 3 s    {np.round(np.abs(late).max(axis=0), 3)} rad\n"
        f"  rate at +2 s     {np.round(rate_2s, 4)} rad/s (arrival {arrival:.2f} s)\n"
        f"  hold rate tau    {np.round(tau_u, 1)} s\n"
        f"  hold dq_a RMS    {dq_a_hold:.4f} rad/s\n"
        f"  plan peak |dq_a| {np.round(plan_peak, 3)} rad/s\n"
        f"  plan v_s pk/mean {plan_pace:.3f} ({plan.duration:.2f} s)\n"
        f"  asked now / plan {np.round(now_ratio, 3)}\n"
        f"  asked any / plan {np.round(any_ratio, 3)}\n"
        f"  achieved / plan  {np.round(ratio, 3)}\n"
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
    # u past u_clamp_* leaves Psi's fitted domain; scored on u, not dq
    dq = log[:, 2 + nq + 2 : 2 + 2 * nq + 2]
    past_clamp = bool(np.any(u > chain.high + 1e-3) or np.any(u < chain.low - 1e-3))
    # dq check is the overspeed column only, not an exit
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
            # peak per planned axis; ratios nan on a held axis
            "plan_dq_peak": plan_peak.tolist(),
            "plan_pace_peak_over_mean": plan_pace,
            "asked_now_dq_peak": now_peak.tolist(),
            "asked_any_dq_peak": any_peak.tolist(),
            "dq_peak": dq_peak.tolist(),
            "asked_now_plan_ratio": now_ratio.tolist(),
            "asked_any_plan_ratio": any_ratio.tolist(),
            "dq_plan_ratio": ratio.tolist(),
            # |u| over u_clamp per axis; > 1 is past_clamp
            "u_clamp_ratio": np.maximum(
                u.max(0) / chain.high, u.min(0) / chain.low
            ).tolist(),
            "u_peak": np.abs(u).max(0).tolist(),
            # dq over the command clamp; > 1 is overspeed, not Psi extrapolation
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
            # share of cycles stopped at qp_solver_iter_max; status stays healthy
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
            # share of solved cycles per bound ridden; mean cost split
            "active": {
                name: float(np.mean([r[name] for r in riders]))
                for name in (riders[0] if riders else {})
            },
            "cost": {
                name: float(np.mean([getattr(c, name) for c in costs]))
                for name in (dataclasses.asdict(costs[0]) if costs else {})
            },
        }
    # viewer coast: past the plan, unscored
    while not diverged and coast() and np.all(np.isfinite(plant.q)):
        next(run)
    if status_line and plant.viewer is not None:
        print(file=sys.stderr)
    return reason, scores


if __name__ == "__main__":
    sys.exit(leave(main()))
