"""
What `wire_chain.py` closes the loop around: settings, solver, plan, plant.

The shipped yaml plus `--set` overrides, the node's own `Ocp` (exported on
demand), a `crane_planning` plan and MuJoCo under C3 and the JTC's velocity
loop, with the plant mismatch of `crane_model.mismatch`. The cycle itself is
`crane_mpc.cycle.Cycle`; nothing here runs one.
"""

import dataclasses
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "scripts"))
sys.path.insert(0, str(PACKAGE.parent / "crane_planning" / "scripts"))

import export_ocp  # noqa: E402
import tune_planner  # noqa: E402
from crane_model.actuator import C3Actuator  # noqa: E402
from crane_model.conventions import (  # noqa: E402
    ACTUATED_INDICES,
    GENERALIZED_DOF,
    PASSIVE_INDICES,
    Frame,
    canonical_joints,
)
from crane_model.mismatch import PI_RUNGS, PsiGain, pi_rung  # noqa: E402
from crane_model.mujoco_plant import (  # noqa: E402
    GROUND_Z,
    NX_RIGID,
    PLANNED_INDICES,
    TOOL_INDEX,
)
from crane_model.symbolic import K_ACTUATOR_FIT  # noqa: E402
from crane_model.velocity_loop import VelocityLoop, load_velocity_loop  # noqa: E402
from crane_mpc import solver as ocp_runtime  # noqa: E402
from crane_mpc.problem import PATH_POINTS  # noqa: E402
from crane_planning.ocp import evaluate  # noqa: E402

ox, cs = export_ocp.ox, export_ocp.cs
PLANNED = list(PLANNED_INDICES)
ACTUATED, PASSIVE = list(ACTUATED_INDICES), list(PASSIVE_INDICES)
AXES = 5
#: draws before a random goal gives up; 62 % of joint draws clear collision and ground.
GOAL_DRAWS = 12


def add_arguments(parser) -> None:
    """Plan, plant, mismatch and solver flags; all identity by default."""
    tune_planner.plan_arguments(parser)
    parser.add_argument(
        "--timestep", type=float, default=5.0e-4, help="MuJoCo and C3 step"
    )
    parser.add_argument(
        "--control-rate", type=float, default=None, help="JTC Hz; default the yaml's"
    )
    parser.add_argument(
        "--dead-time", type=float, default=None, help="C3 plant dead time, s"
    )
    parser.add_argument("--no-clamp", action="store_true", help="drop C3's force clamp")
    parser.add_argument(
        "--random",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="chain N random goals, each from where the last ended; no N: until "
        "the viewer closes",
    )
    parser.add_argument("--seed", type=int, default=None, help="--random's stream")
    for side in ("positive", "negative"):
        parser.add_argument(
            f"--psi-gain-{side}", type=float, nargs=AXES, default=None, metavar="G"
        )
    parser.add_argument(
        "--k-scale",
        type=float,
        nargs=AXES,
        default=None,
        help="plant C3 k; with --d-scale",
    )
    parser.add_argument(
        "--d-scale", type=float, nargs=AXES, default=None, help="plant joint damping"
    )
    parser.add_argument(
        "--lag-shift", type=float, default=0.0, help="plant tau_v += s, dead time -= s"
    )
    parser.add_argument("--pi-rung", choices=PI_RUNGS, default="full")
    parser.add_argument(
        "--pi-scale", type=float, default=1.0, help="multiply the JTC's p, i, d"
    )
    parser.add_argument("--payload-mass", type=float, default=0.0, help="kg, the OCP's")
    parser.add_argument(
        "--payload-com", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="m, in K8"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="yaml lever"
    )
    parser.add_argument("--rebuild", action="store_true", help="re-export the solver")
    parser.add_argument("--verbose-build", action="store_true")


def description_xml() -> str:
    return (export_ocp.DEFAULT_DESCRIPTIONS / export_ocp.DESCRIPTION).read_text()


def override(parameters: dict, assignment: str) -> None:
    """Apply one `key.path=value`; a scalar on a list key fills every entry."""
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


def settings(overrides=()) -> tuple[dict, dict]:
    """Read the shipped crane_mpc.yaml as the node does, plus `overrides`."""
    parameters = ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    parameters["limits"].update(export_ocp.ocp_config.machine_limits())
    hydraulics = export_ocp.hydraulic_limits()
    xml = description_xml()
    shipped = ocp_runtime.export_key(parameters, hydraulics, xml)
    for assignment in overrides:
        override(parameters, assignment)
    moved = ocp_runtime.export_key(parameters, hydraulics, xml)
    moved = [key for key in shipped if moved[key] != shipped[key]]
    if moved:
        print(f"warning: --set moves compiled {moved}; re-exports", file=sys.stderr)
    return parameters, hydraulics


def create_solver(parameters, hydraulics, rebuild=False, verbose=False):
    """Open the node's `Ocp` for these settings, exporting first if none matches."""
    # Before the export, not after it: a `--set` the node would refuse otherwise
    # costs a minute of code generation to find out. `Ocp` checks again anyway.
    export_ocp.ocp_config.check_settings(parameters, hydraulics)
    xml = description_xml()
    key = ocp_runtime.export_key(parameters, hydraulics, xml)
    base = ocp_runtime.export_base()
    if rebuild:
        shutil.rmtree(ox.solver_root(base, key), ignore_errors=True)
    try:
        ox.manifest(base, key, "the harness exports on demand")
    except ox.StaleExport:
        acados_ocp, *_ = export_ocp.build_ocp(xml, parameters, hydraulics)
        sims = ocp_runtime.predictor_sims(acados_ocp, parameters)
        ox.export(acados_ocp, base, key, sims=sims, verbose=verbose)
    return ocp_runtime.Ocp(xml, parameters, hydraulics, verbose=verbose)


class Chain:
    """What sits under the MPC: the JTC's velocity loop, Psi and C3, on `plant`."""

    def __init__(self, plant, q_tool: float, options, fit=None) -> None:
        self.plant, self.q_tool = plant, float(q_tool)
        fit = K_ACTUATOR_FIT if fit is None else fit
        gains_by_joint, rate_hz = load_velocity_loop()
        gains = [gains_by_joint[canonical_joints()[i]] for i in PLANNED]
        self.step_s = 1.0 / (options.control_rate or rate_hz)
        s = options.pi_scale
        loop_gains = [
            dataclasses.replace(a, p=a.p * s, i=a.i * s, d=a.d * s)
            for a in pi_rung(gains, options.pi_rung)
        ]
        self.loop = VelocityLoop(
            loop_gains, self.step_s, continuous=plant.continuous_axes
        )
        self.ff_scale = np.array([a.ff_velocity_scale for a in gains])
        self.low = np.array([a.u_clamp_min for a in gains])
        self.high = np.array([a.u_clamp_max for a in gains])
        self.psi = PsiGain(options.psi_gain_positive, options.psi_gain_negative)
        self.actuator = C3Actuator(
            timestep=plant.model.opt.timestep,
            fit=fit,
            dead_time_s=fit.dead_time_s
            if options.dead_time is None
            else options.dead_time,
            tau_max=None if options.no_clamp else plant.effort_limits,
        )

    def seed(self, state: np.ndarray) -> None:
        """Place the plant; C3 starts holding it up, not from tau = 0."""
        self.plant.set_rigid_state(state[:NX_RIGID], self.q_tool)
        self.actuator.reset(tau=self.plant.holding_force)


def random_goal(planner, start, rng):
    """Draw a TCP pose off the joints: reachable, collision-free, above ground."""
    lower = np.where(planner.limits.bounded, planner.limits.lower, -np.pi)
    upper = np.where(planner.limits.bounded, planner.limits.upper, np.pi)
    q = np.zeros(GENERALIZED_DOF)
    q[TOOL_INDEX] = start.q_tool
    for _ in range(GOAL_DRAWS):
        q[PLANNED] = rng.uniform(lower, upper)
        q[PASSIVE] = planner.model.passive_equilibrium(q[ACTUATED])
        clearance = planner.model.collision_query(q, [])
        pose = planner.model.forward_kinematics(q, Frame.MOUNTING_BASE, Frame.TCP)
        margin = planner.config.margin_safety
        if (
            not clearance.collision
            and clearance.minimum_distance_m > margin
            and pose.position_m[2] > GROUND_Z + margin
        ):
            break
    sample = tune_planner.Start(q=q, dq_a=np.zeros(len(PLANNED)))
    return np.asarray(pose.position_m), tune_planner.start_yaw(planner, sample)


def plan_move(planner, start, options, rng):
    """Plan one move, or None; a refused random goal is redrawn, a named one is not."""
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


def next_start(plant, planner):
    """Start where this move ended, inside the planner's box; the rotator wraps."""
    q = plant.q
    inside = np.clip(q[PLANNED], planner.limits.lower, planner.limits.upper)
    wrapped = (q[PLANNED] + np.pi) % (2.0 * np.pi) - np.pi
    q[PLANNED] = np.where(plant.continuous_axes, wrapped, inside)
    return tune_planner.Start(q=q, dq_a=np.zeros(len(PLANNED)))


def path_control(plan) -> np.ndarray:
    """Take the planner's curve in its own `sigma`, not its timed resample."""
    places = np.linspace(0.0, 1.0, 4 * PATH_POINTS)
    return ocp_runtime.path_control(evaluate(plan.timing.coefficients, places))
