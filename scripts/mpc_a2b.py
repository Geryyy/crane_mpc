#!/usr/bin/env python3
"""
Run the crane MPC offline on a joint-space A-to-B movement and plot it.

Developer tuning tool, not a second controller: imports ``export_ocp.py`` so
dynamics, cost residuals, constraints, integrator and RTI backend match the
deployed solver. No ROS, DDS or controller manager required.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
import yaml
from acados_template import AcadosOcpSolver
from scipy.optimize import root

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

# pinocchio's bool-converter warning is harmless but clutters CLI help/summaries
warnings.filterwarnings(
    "ignore", message="to-Python converter for pinocchio.*", category=RuntimeWarning
)

import export_ocp  # noqa: E402

# same solver-config pieces the node writes, defined once; source tree is the
# fallback for a from-scratch build, same as export_ocp resolves cs
try:
    from crane_mpc import config as ocp_config  # noqa: E402
    from crane_mpc import solver as ocp_runtime  # noqa: E402
except ImportError:
    sys.path.insert(0, str(PACKAGE))
    from crane_mpc import config as ocp_config  # noqa: E402
    from crane_mpc import solver as ocp_runtime  # noqa: E402

configure_fixed_data = ocp_runtime.configure_fixed_data
constraint_data = ocp_runtime.constraint_data
position_box = ocp_runtime.position_box
solver_signature = ocp_runtime.solver_signature
stage_parameters = ocp_runtime.stage_parameters
stage_reference = ocp_runtime.stage_reference
state_bounds = ocp_runtime.state_bounds
weight_matrices = ocp_runtime.weight_matrices

cs = export_ocp.cs

AXIS_NAMES = ("slew", "boom", "arm", "telescope", "rotator")
# acados' split of time_tot + QP iterations, per cycle (issue 129: time_tot alone
# can't say model vs QP cost). Seconds except qp_iter; time_sim accumulates over
# the solver's life so it's differenced below like the others.
TIMING_FIELDS = ("time_lin", "time_sim", "time_qp", "time_qp_xcond", "qp_iter")
CUMULATIVE_FIELDS = ("time_sim",)
PASSIVE_NAMES = ("sway 1", "sway 2")
DEFAULT_A = np.array([0.0, 0.30, 0.80, 0.60, 0.0])
DEFAULT_B = np.array([0.60, 0.60, 1.20, 1.00, 0.50])


@dataclass
class Path:
    """
    The curve the cost is written against, and how the plan means to spend it.

    `control` is the path in its own parameter, `progress` takes a virtual time
    to where the plan is on it. Separated because that is the point: geometry
    does not bend in time, and a timing law does not leave the path.
    """

    control: np.ndarray
    progress: object


def line_path(a: np.ndarray, b: np.ndarray, duration: float) -> Path:
    """Build this script's own A-to-B: a joint-space line, quintic in time."""
    samples = np.linspace(a, b, 4 * ocp_runtime.problem.PATH_POINTS)
    return Path(
        control=ocp_runtime.path_control(samples),
        progress=lambda time: minimum_jerk(time, duration)[0],
    )


@dataclass
class RunData:
    """Closed-loop samples, including the terminal sample at ``time[-1]``."""

    time: np.ndarray
    #: virtual time (s of plan): reference origin per wall-clock sample; advances
    #: by T_s only at progress rate one -- the gap is what the MPC bought
    virtual_time: np.ndarray
    state: np.ndarray
    q_ref: np.ndarray
    dq_ref: np.ndarray
    q_eq: np.ndarray
    control: np.ndarray
    desired_velocity: np.ndarray
    hydraulic_use: np.ndarray
    pump_flow: np.ndarray
    solve_time: np.ndarray
    status: np.ndarray
    fallback: np.ndarray
    #: One array per `TIMING_FIELDS` key, same length as `solve_time`.
    timing: dict[str, np.ndarray]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone simulation and tuning arguments."""
    parser = argparse.ArgumentParser(
        description="Run and plot the real crane MPC on an offline A-to-B simulation."
    )
    parser.add_argument(
        "--a",
        nargs=cs.K_PLANNED_DOF,
        type=float,
        default=DEFAULT_A,
        metavar=("SW", "BOOM", "ARM", "TELESCOPE", "ROTATOR"),
        help="start joint pose; radians except telescope in metres",
    )
    parser.add_argument(
        "--b",
        nargs=cs.K_PLANNED_DOF,
        type=float,
        default=DEFAULT_B,
        metavar=("SW", "BOOM", "ARM", "TELESCOPE", "ROTATOR"),
        help="goal joint pose; radians except telescope in metres",
    )
    parser.add_argument("--tool-position", type=float, default=0.30)
    parser.add_argument("--move-duration", type=float, default=4.0)
    parser.add_argument("--settle-duration", type=float, default=3.0)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument(
        "--payload-com",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="payload centre of mass in K8, metres",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="sample time; defaults to config/crane_mpc.yaml Ts",
    )
    parser.add_argument(
        "--horizon-knots",
        type=int,
        default=None,
        help="number of prediction knots; defaults to the deployment value",
    )

    tuning = parser.add_argument_group("cost tuning (multipliers of deployment YAML)")
    tuning.add_argument("--q-scale", type=float, default=1.0)
    tuning.add_argument("--dq-scale", type=float, default=1.0)
    tuning.add_argument("--sway-scale", type=float, default=1.0)
    tuning.add_argument("--sway-rate-scale", type=float, default=1.0)
    tuning.add_argument("--effort-scale", type=float, default=1.0)
    tuning.add_argument("--input-scale", type=float, default=1.0)
    tuning.add_argument("--lag-scale", type=float, default=1.0)
    tuning.add_argument(
        "--progress-scale",
        type=float,
        default=1.0,
        help="multiplier on weights.progress_rate -- the price of spending time. "
        "A very large value pins the progress rate at one, which is the "
        "time-indexed controller this design replaces",
    )
    tuning.add_argument(
        "--terminal-scale",
        type=float,
        default=None,
        help="absolute terminal multiplier; defaults to deployment YAML",
    )
    tuning.add_argument(
        "--levenberg-marquardt",
        type=float,
        default=None,
        help="absolute regularisation; defaults to deployment YAML",
    )
    parser.add_argument(
        "--qp-cond-n",
        type=int,
        default=None,
        help="HPIPM partial-condensing block count; defaults to acados' own default",
    )

    parser.add_argument(
        "--enforce-budget",
        action="store_true",
        help="apply the node's previous-plan fallback when solve_budget is exceeded",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE / "build" / "mpc_a2b.png",
        help="plot path (a CSV is written beside it)",
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the figure. Every automated driver passes this: they read the "
        "CSV and the summary, and matplotlib is most of a run's wall clock",
    )
    parser.add_argument(
        "--show", action="store_true", help="also open the matplotlib window"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="recompile even when an identical solver is cached",
    )
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args(argv)


def positive(value: float, name: str, allow_zero: bool = False) -> None:
    """Validate a finite positive or non-negative scalar."""
    good = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not good:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}, got {value}")


def load_settings(arguments: argparse.Namespace) -> tuple[dict, dict]:
    """Load deployment YAML and apply command-line tuning overrides."""
    # same two reads export_ocp.main makes, through the same helpers -- harness
    # can't drift from the exporter about what a deployment is
    parameters = export_ocp.ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    hydraulics = export_ocp.hydraulic_limits()
    # deep-copy through YAML: nested mapping also used to form the cache signature below
    parameters = yaml.safe_load(yaml.safe_dump(parameters))
    # the box and u^+ are crane_model's; the yaml carries neither, same as the node
    parameters["limits"].update(ocp_config.machine_limits())
    if arguments.dt is not None:
        parameters["Ts"] = arguments.dt
    if arguments.horizon_knots is not None:
        parameters["horizon_length"] = arguments.horizon_knots
    if arguments.terminal_scale is not None:
        parameters["weights"]["terminal_scale"] = arguments.terminal_scale
    if arguments.levenberg_marquardt is not None:
        parameters["levenberg_marquardt"] = arguments.levenberg_marquardt
    if arguments.qp_cond_n is not None:
        parameters["qp_solver_cond_N"] = arguments.qp_cond_n

    scales = {
        "q_a": arguments.q_scale,
        "dq_a": arguments.dq_scale,
        "q_u": arguments.sway_scale,
        "dq_u": arguments.sway_rate_scale,
        "tau_a": arguments.effort_scale,
        "u": arguments.input_scale,
    }
    for name, scale in scales.items():
        positive(scale, f"--{name.replace('_', '-')}-scale", allow_zero=True)
        parameters["weights"][name] = [
            scale * float(value) for value in parameters["weights"][name]
        ]
    positive(arguments.lag_scale, "--lag-scale", allow_zero=True)
    positive(arguments.progress_scale, "--progress-scale")
    parameters["weights"]["lag"] = arguments.lag_scale * float(
        parameters["weights"]["lag"]
    )
    parameters["weights"]["progress_rate"] = arguments.progress_scale * float(
        parameters["weights"]["progress_rate"]
    )

    positive(float(parameters["Ts"]), "--dt")
    positive(float(parameters["weights"]["terminal_scale"]), "--terminal-scale", True)
    positive(float(parameters["levenberg_marquardt"]), "--levenberg-marquardt", True)
    if int(parameters["horizon_length"]) < 3:
        raise ValueError("--horizon-knots must be at least 3")
    return parameters, hydraulics


def validate_movement(
    arguments: argparse.Namespace, parameters: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Validate A and B against the configured control-safe box."""
    a = np.asarray(arguments.a, dtype=float)
    b = np.asarray(arguments.b, dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("A and B must contain only finite values")
    lower = np.asarray(parameters["limits"]["q_a_lower"][: cs.K_PLANNED_DOF])
    upper = np.asarray(parameters["limits"]["q_a_upper"][: cs.K_PLANNED_DOF])
    for label, pose in (("A", a), ("B", b)):
        bad = np.flatnonzero((pose < lower) | (pose > upper))
        if bad.size:
            names = ", ".join(AXIS_NAMES[index] for index in bad)
            raise ValueError(
                f"{label} is outside the control-safe position box on {names}"
            )
    positive(arguments.move_duration, "--move-duration")
    positive(arguments.settle_duration, "--settle-duration", allow_zero=True)
    positive(arguments.payload_mass, "--payload-mass", allow_zero=True)
    if not math.isfinite(arguments.tool_position):
        raise ValueError("--tool-position must be finite")
    if not np.all(np.isfinite(arguments.payload_com)):
        raise ValueError("--payload-com must contain finite values")
    return a, b


def parameter_vector(arguments: argparse.Namespace) -> np.ndarray:
    """Build the tool and point-payload acados parameter vector."""
    parameter = np.zeros(cs.NP)
    parameter[cs.P_TOOL_POSITION] = arguments.tool_position
    parameter[cs.P_PAYLOAD_MASS] = arguments.payload_mass
    parameter[cs.P_PAYLOAD_COM : cs.P_PAYLOAD_COM + 3] = arguments.payload_com
    # payload inertia stays zero: crane_msgs/Payload is a point mass, matching the node
    return parameter


def create_solver(
    arguments: argparse.Namespace, parameters: dict, hydraulics: dict
) -> tuple[AcadosOcpSolver, object, np.ndarray]:
    """
    Open this problem's compiled solver, compiling once if needed.

    Node opens the same cache: two callers, one compile.
    """
    description = (export_ocp.DEFAULT_DESCRIPTIONS / export_ocp.DESCRIPTION).read_text()
    cache = ocp_runtime.solver_cache(parameters, hydraulics, description)
    if arguments.rebuild and cache.is_dir():
        # generated, gitignored cache dir; no source/user output can land here
        shutil.rmtree(cache)
    ocp, scale, model = export_ocp.build_ocp(description, parameters, hydraulics)
    solver, _ = ocp_runtime.load_or_build(
        ocp, parameters, hydraulics, description, verbose=arguments.verbose_build
    )
    return solver, model, scale


def solver_stat(solver: AcadosOcpSolver, field: str) -> float:
    """
    One acados statistic per solve, as a scalar.

    `qp_iter` comes back per SQP iteration (one entry under RTI); summing keeps
    the column meaningful if that changes.
    """
    return float(np.sum(np.asarray(solver.get_stats(field), dtype=float)))


def minimum_jerk(time: float, duration: float) -> tuple[float, float, float]:
    """Return quintic progress and its first two rates, zero at both endpoints."""
    if time <= 0.0:
        return 0.0, 0.0, 0.0
    if time >= duration:
        return 1.0, 0.0, 0.0
    s = time / duration
    position = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    rate = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    accel = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (duration * duration)
    return position, rate, accel


def reference_at(
    time: float, a: np.ndarray, b: np.ndarray, duration: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the reference and its first two derivatives at a **virtual** time.

    MPC needs a local second-order model per stage, not a sample -- quintic is
    analytic so both derivatives are exact; node takes them off its own cubic
    Hermite resample instead.
    """
    progress, rate, accel = minimum_jerk(time, duration)
    displacement = b - a
    return (
        a + progress * displacement,
        rate * displacement,
        accel * displacement,
    )


def make_numeric_functions(
    model: object,
) -> tuple[ca.Function, ca.Function, ca.Function, ca.Function]:
    """Create numeric dynamics, equilibrium, output-map and static-force functions."""
    dynamics = ca.Function("a2b_dynamics", [model.x, model.u, model.p], [model.xdot])
    bias_u = ca.Function("a2b_passive_bias", [model.x, model.p], [model.bias_u])
    outputs = ca.Function("a2b_outputs", [model.x, model.u, model.p], [model.z])
    static = ca.Function(
        "a2b_static_force", [model.x, model.p], [model.actuated_force_static]
    )
    return dynamics, bias_u, outputs, static


#: ERK4 substeps per control sample. Fastest eigenvalue of the linearised plant
#: is `|lambda| T_s = 8.5` at an ordinary pose (telescope `k = 3.5e6 N/m` vs its
#: effective mass), ERK4 stable only to ~2.8 -- one explicit step diverges in 3
#: samples, NaN to the solver. Solver itself is IRK and doesn't need this.
PLANT_SUBSTEPS = 10


def rk4_step(
    dynamics: ca.Function,
    state: np.ndarray,
    control: np.ndarray,
    parameter: np.ndarray,
    dt: float,
    substeps: int = PLANT_SUBSTEPS,
) -> np.ndarray:
    """
    Integrate one model-matched plant sample with substepped ERK4.

    `substeps` is an argument so a co-simulation with an already short `dt`
    doesn't pay ten inner steps per its own.
    """

    def evaluate(value: np.ndarray) -> np.ndarray:
        return np.asarray(dynamics(value, control, parameter)).reshape(-1)

    step = dt / substeps
    for _ in range(substeps):
        k1 = evaluate(state)
        k2 = evaluate(state + 0.5 * step * k1)
        k3 = evaluate(state + 0.5 * step * k2)
        k4 = evaluate(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def equilibrium_table(
    bias_u: ca.Function,
    parameter: np.ndarray,
    reference,
    dt: float,
    count: int,
    guess=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute continuous-branch passive equilibria over **virtual** time.

    Indexed by seconds of nominal plan, so a slow-spending run reads the same
    table more slowly rather than needing a second one; grid is the horizon's
    own, `equilibrium_at` interpolates between knots.

    `guess` seeds the first solve, later ones continue from their predecessor,
    picking the branch the table sits on. Zero suits a pose near the origin; a
    tilt near pi/2 (`initialization_outside.yaml`) converges onto a different
    solution, simulating a machine holding its load sideways.
    """
    times = dt * np.arange(count)
    q_eq = np.zeros((count, cs.K_PASSIVE_DOF))
    guess = (
        np.zeros(cs.K_PASSIVE_DOF)
        if guess is None
        else np.asarray(guess, dtype=float).reshape(cs.K_PASSIVE_DOF).copy()
    )

    for index, time in enumerate(times):
        q_ref_index, _, _ = reference(time)

        def residual(passive: np.ndarray, q_ref_index=q_ref_index) -> np.ndarray:
            state = np.zeros(cs.NX)
            state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF] = (
                q_ref_index
            )
            state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = (
                passive
            )
            return np.asarray(bias_u(state, parameter)).reshape(-1)

        solution = root(residual, guess)
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise RuntimeError(
                f"passive equilibrium failed at t={time:.3f} s: {solution.message}"
            )
        # Continuation from the previous knot keeps the same periodic branch.
        guess = solution.x
        q_eq[index] = guess
    return times, q_eq


def equilibrium_at(times: np.ndarray, table: np.ndarray, tau: float) -> np.ndarray:
    """Read the equilibrium table at a virtual time, linearly between knots."""
    clamped = float(np.clip(tau, times[0], times[-1]))
    return np.array(
        [np.interp(clamped, times, table[:, row]) for row in range(table.shape[1])]
    )


def hydraulic_utilisation(
    outputs: ca.Function,
    state: np.ndarray,
    control: np.ndarray,
    parameter: np.ndarray,
    extend: np.ndarray,
    retract: np.ndarray,
    hydraulics: dict,
) -> np.ndarray:
    """Return force and shared-pump use as fractions of their limits."""
    values = np.asarray(outputs(state, control, parameter)).reshape(-1)
    force = values[
        cs.K_CYLINDER_FORCE_OFFSET : cs.K_CYLINDER_FORCE_OFFSET + cs.K_PLANNED_DOF
    ]
    force_use = np.where(force >= 0.0, force / extend, -force / retract)
    flow = np.sum(
        values[cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_PLANNED_DOF]
    )
    flow_limit = float(hydraulics["pump_flow_planning_factor"]) * float(
        hydraulics["pump_flow_max"]
    )
    return np.concatenate([force_use, [flow / flow_limit]])


def simulate(
    arguments: argparse.Namespace,
    parameters: dict,
    hydraulics: dict,
    solver: AcadosOcpSolver,
    model: object,
    scale: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reference=None,
    plant=None,
    passive_guess=None,
    horizon=None,
    path=None,
) -> RunData:
    """
    Run the receding-horizon controller against a plant.

    `reference`/`plant` default to this script's own quintic and model-matched
    ERK4 rollout; passing either replaces that half, leaves the controller alone.

    `plant` is `(state, control, parameter, dt) -> state` over the full `cs.NX`
    -- must carry C3's lag/progress/force rows since the controller reads them
    back. `passive_guess` picks the equilibrium branch, i.e. where the run starts.

    `horizon`, if given, is handed the (N+1, NX) states in force each cycle --
    the accepted solution, or the shifted previous one where a solve was refused.
    """
    dt = float(parameters["Ts"])
    intervals = export_ocp.shooting_intervals(parameters)
    steps = int(math.ceil((arguments.move_duration + arguments.settle_duration) / dt))
    base_parameter = parameter_vector(arguments)
    dynamics, bias_u, outputs, static_force = make_numeric_functions(model)
    if reference is None:
        # (q, dq, ddq) at a virtual time is the whole contract for a driver to swap curves
        def reference(time):
            return reference_at(time, a, b, arguments.move_duration)

    if plant is None:

        def plant(state, control, parameter, step_s):
            return rk4_step(dynamics, state, control, parameter, step_s)

    # The cost reads a path, not samples of one. Absent a planned curve this is
    # the straight line the quintic above walks.
    if path is None:
        path = line_path(a, b, arguments.move_duration)
    # `s`'s ceiling over the horizon, so the fitted window covers everywhere the
    # optimizer may put it -- past that, `casadi_value` clamps and the reference
    # would flatten where the progress ran fastest.
    span = intervals * dt * float(parameters["limits"]["progress_rate_max"])

    # table is over virtual time, read at whatever virtual time the horizon reached;
    # sized for the worst case, a horizon that never slows down
    table_count = steps + intervals + 1
    eq_times, q_eq_table = equilibrium_table(
        bias_u, base_parameter, reference, dt, table_count, passive_guess
    )
    u_max, extend, retract = configure_fixed_data(
        solver, model, scale, parameters, hydraulics
    )

    state = np.zeros(cs.NX)
    state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF] = a
    state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = (
        equilibrium_at(eq_times, q_eq_table, 0.0)
    )
    # The plan is spent at nominal rate until a solve says otherwise.
    state[cs.X_PROGRESS_RATE] = export_ocp.K_PROGRESS_RATE_REFERENCE
    # C3 block 3 starts holding its own weight; no force measurement in the stack,
    # so h_eff at the initial pose seeds it (zero would start with hydraulics off)
    state[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF] = np.asarray(
        static_force(state, base_parameter)
    ).reshape(-1)

    states = np.zeros((steps + 1, cs.NX))
    controls = np.zeros((steps, cs.NU_PROGRESS))
    hydraulics_used = np.zeros((steps, cs.NU + 1))
    solve_times = np.zeros(steps)
    timings = {field: np.full(steps, math.nan) for field in TIMING_FIELDS}
    cumulative = {field: 0.0 for field in CUMULATIVE_FIELDS}
    statuses = np.zeros(steps, dtype=int)
    fallback = np.zeros(steps, dtype=bool)
    q_ref_log = np.zeros((steps + 1, cs.K_PLANNED_DOF))
    dq_ref_log = np.zeros_like(q_ref_log)
    q_eq_log = np.zeros((steps + 1, cs.K_PASSIVE_DOF))
    virtual_time = np.zeros(steps + 1)
    states[0] = state

    previous_x: list[np.ndarray] | None = None
    previous_u: list[np.ndarray] | None = None
    budget = float(parameters["solve_budget"])
    rate_max = float(parameters["limits"]["progress_rate_max"])

    # reference origin, in virtual time -- what the progress state buys: horizon
    # samples from here, not wall clock; each cycle advances by what the last
    # solve's first interval was worth. Wall-clock indexing let the reference
    # run away from a machine that fell behind.
    origin = 0.0

    for step in range(steps):
        virtual_time[step] = origin
        q_ref_log[step], dq_ref_log[step], _ = reference(origin)
        q_eq_log[step] = equilibrium_at(eq_times, q_eq_table, origin)
        # one evaluation per cycle at measured state, held across horizon --
        # ocp_solver.cpp's approach, same trade mpc_node.cpp makes for q_eq
        tau_hold = np.asarray(static_force(state, base_parameter)).reshape(-1)
        # progress restart: s pinned at zero every cycle, origin above carries
        # what the last one bought (timber_crane_mpc.cpp:173-181)
        state[cs.X_PROGRESS] = 0.0
        # iterate always dropped, HPIPM's memory only on a cold cycle -- same
        # split as `crane_mpc/solver.py`, which this harness has to mirror
        warm_cycle = previous_x is not None and previous_u is not None
        solver.reset(reset_qp_solver_mem=0 if warm_cycle else 1)
        # The path and the window are the whole horizon's, so this is built once
        # a cycle; which stage a stage is, `s` carries.
        cycle_parameter = stage_parameters(
            base_parameter,
            span,
            ocp_runtime.timing_control(path.progress, origin, span),
            path.control,
        )
        for stage in range(intervals + 1):
            # stage's nominal virtual time: where s would be if nothing slipped.
            # anchoring on the previous solution's progress instead was tried and
            # measured worse (issue 119 notes) -- so this is k*T_s
            nominal = stage * dt
            tau_virtual = origin + nominal
            equilibrium = equilibrium_at(eq_times, q_eq_table, tau_virtual)
            solver.set(stage, "p", cycle_parameter)
            solver.cost_set(
                stage,
                "yref",
                stage_reference(equilibrium, tau_hold, stage == intervals),
            )
            if stage == 0:
                lower = upper = state
            else:
                lower, upper = state_bounds(
                    parameters, equilibrium, state[: cs.K_PLANNED_DOF]
                )
                # s's ceiling: what nominal seconds at the fastest allowed rate can reach
                upper[cs.X_PROGRESS] = nominal * rate_max
            solver.constraints_set(stage, "lbx", lower)
            solver.constraints_set(stage, "ubx", upper)

        if previous_x is None or previous_u is None:
            guess_x = [state.copy()]
            for _ in range(intervals):
                guess_x.append(
                    rk4_step(
                        dynamics,
                        guess_x[-1],
                        np.zeros(cs.NU_PROGRESS),
                        base_parameter,
                        dt,
                    )
                )
            guess_u = [np.zeros(cs.NU_PROGRESS) for _ in range(intervals)]
        else:
            guess_x = previous_x[1:] + [previous_x[-1].copy()]
            guess_u = previous_u[1:] + [previous_u[-1].copy()]

        for stage in range(intervals + 1):
            value = guess_x[stage].copy()
            if stage == 0:
                value = state.copy()
            else:
                equilibrium = equilibrium_at(eq_times, q_eq_table, origin + stage * dt)
                lower, upper = state_bounds(
                    parameters, equilibrium, state[: cs.K_PLANNED_DOF]
                )
                upper[cs.X_PROGRESS] = stage * dt * rate_max
                # only boxed prefix has bounds; force states held by constraint 6,
                # left as the rollout produced them
                boxed = lower.size
                value[:boxed] = np.clip(value[:boxed], lower, upper)
            solver.set(stage, "x", value)
            if stage < intervals:
                solver.set(stage, "u", np.clip(guess_u[stage], -u_max, u_max))

        status = int(solver.solve())
        solve_time = float(solver.get_stats("time_tot"))
        statuses[step] = status
        solve_times[step] = solve_time
        for field in TIMING_FIELDS:
            value = solver_stat(solver, field)
            if field in CUMULATIVE_FIELDS:
                value, cumulative[field] = value - cumulative[field], value
            timings[field][step] = value
        candidate_x = [
            np.asarray(solver.get(stage, "x")).copy() for stage in range(intervals + 1)
        ]
        candidate_u = [
            np.asarray(solver.get(stage, "u")).copy() for stage in range(intervals)
        ]
        finite = all(np.all(np.isfinite(value)) for value in candidate_x + candidate_u)
        accepted = (
            status == 0
            and finite
            and (not arguments.enforce_budget or solve_time <= budget)
        )

        if accepted:
            control = candidate_u[0]
            # QP's own iterate at node one (satisfies linearised, not integrated,
            # dynamics), clamped to what the rate box reaches in one interval --
            # same as ocp_solver.cpp; an unclamped jump would skip plan the machine
            # never tracked, while staying finite
            advance = float(
                np.clip(
                    candidate_x[1][cs.X_PROGRESS],
                    0.0,
                    dt * float(parameters["limits"]["progress_rate_max"]),
                )
            )
            previous_x, previous_u = candidate_x, candidate_u
        elif previous_u is not None:
            # Same shifted-previous-plan fallback used by mpc_node.
            control = (
                previous_u[1].copy() if len(previous_u) > 1 else previous_u[0].copy()
            )
            advance = dt
            previous_x = previous_x[1:] + [previous_x[-1].copy()]
            previous_u = previous_u[1:] + [previous_u[-1].copy()]
            fallback[step] = True
        else:
            control = np.zeros(cs.NU_PROGRESS)
            advance = dt
            fallback[step] = True

        controls[step] = control
        hydraulics_used[step] = hydraulic_utilisation(
            outputs, state, control, base_parameter, extend, retract, hydraulics
        )
        if horizon is not None and previous_x is not None:
            horizon(np.array(previous_x))
        state = plant(state, control, base_parameter, dt)
        states[step + 1] = state
        origin += advance if math.isfinite(advance) and advance >= 0.0 else dt

        if (
            step == 0
            or (step + 1) % max(1, int(round(1.0 / dt))) == 0
            or step + 1 == steps
        ):
            reached, _, _ = reference(origin)
            error = np.linalg.norm(state[: cs.K_PLANNED_DOF] - reached)
            print(
                f"t={(step + 1) * dt:6.2f} s  s={origin:6.2f} s  "
                f"v_s={state[cs.X_PROGRESS_RATE]:5.3f}  status={status:2d}  "
                f"solve={1e3 * solve_time:7.2f} ms  |q-qref|={error:.4f}"
            )

    virtual_time[steps] = origin
    q_ref_log[steps], dq_ref_log[steps], _ = reference(origin)
    q_eq_log[steps] = equilibrium_at(eq_times, q_eq_table, origin)

    # u is a joint velocity at Psi's input under C3, not acceleration -- desired
    # velocity is the command directly
    desired_velocity = np.vstack(
        [
            states[0, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF],
            controls[:, : cs.NU],
        ]
    )
    flow_limit = float(hydraulics["pump_flow_planning_factor"]) * float(
        hydraulics["pump_flow_max"]
    )

    return RunData(
        time=dt * np.arange(steps + 1),
        virtual_time=virtual_time,
        state=states,
        q_ref=q_ref_log,
        dq_ref=dq_ref_log,
        q_eq=q_eq_log,
        control=controls,
        desired_velocity=desired_velocity,
        hydraulic_use=hydraulics_used,
        pump_flow=hydraulics_used[:, -1] * flow_limit,
        solve_time=solve_times,
        status=statuses,
        fallback=fallback,
        timing=timings,
    )


def write_csv(path: Path, data: RunData) -> None:
    """Write every plotted sample to a flat CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headings = ["time_s"]
    headings += [f"q_{name}" for name in AXIS_NAMES]
    headings += [f"q_ref_{name}" for name in AXIS_NAMES]
    headings += [f"dq_{name}" for name in AXIS_NAMES]
    headings += [f"dq_ref_{name}" for name in AXIS_NAMES]
    headings += [f"dq_desired_{name}" for name in AXIS_NAMES]
    headings += [f"q_{name.replace(' ', '_')}" for name in PASSIVE_NAMES]
    headings += [f"q_eq_{name.replace(' ', '_')}" for name in PASSIVE_NAMES]
    headings += [f"dq_{name.replace(' ', '_')}" for name in PASSIVE_NAMES]
    headings += [f"u_{name}" for name in AXIS_NAMES]
    headings += [f"force_use_{name}" for name in AXIS_NAMES]
    headings += ["pump_use", "pump_flow_m3_s", "solve_time_s", "status", "fallback"]
    headings += list(TIMING_FIELDS)
    headings += ["virtual_time_s", "progress_rate", "progress_accel"]

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        for index, time in enumerate(data.time):
            state = data.state[index]
            row = [time]
            row += state[
                cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF
            ].tolist()
            row += data.q_ref[index].tolist()
            row += state[
                cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF
            ].tolist()
            row += data.dq_ref[index].tolist()
            row += data.desired_velocity[index].tolist()
            row += state[
                cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF
            ].tolist()
            row += data.q_eq[index].tolist()
            row += state[
                cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF
            ].tolist()
            if index < data.control.shape[0]:
                row += data.control[index, : cs.NU].tolist()
                row += data.hydraulic_use[index].tolist()
                row += [data.pump_flow[index]]
                row += [
                    data.solve_time[index],
                    int(data.status[index]),
                    int(data.fallback[index]),
                ]
                row += [data.timing[field][index] for field in TIMING_FIELDS]
                row += [
                    data.virtual_time[index],
                    state[cs.X_PROGRESS_RATE],
                    data.control[index, cs.U_PROGRESS_ACCEL],
                ]
            else:
                row += [math.nan] * (cs.NU + cs.NU + 2)
                row += [math.nan, 0, 0]
                row += [math.nan] * len(TIMING_FIELDS)
                row += [data.virtual_time[index], state[cs.X_PROGRESS_RATE], math.nan]
            writer.writerow(row)


def plot(
    path: Path,
    data: RunData,
    parameters: dict,
    hydraulics: dict,
    show: bool,
) -> None:
    """Render all closed-loop trajectories, commands, limits, and diagnostics."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    time = data.time
    control_time = time[:-1]
    limits = parameters["limits"]
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)

    q = data.state[:, cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
    dq = data.state[:, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF]
    q_u = data.state[
        :, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF
    ]
    dq_u = data.state[
        :, cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF
    ]

    q_axis = axes[0, 0]
    dq_axis = axes[0, 1]
    input_axis = axes[1, 0]
    desired_velocity_axis = axes[1, 1]
    flow_axis = axes[2, 0]
    solve_axis = axes[2, 1]

    q_lower = np.asarray(limits["q_a_lower"][: cs.K_PLANNED_DOF], dtype=float)
    q_upper = np.asarray(limits["q_a_upper"][: cs.K_PLANNED_DOF], dtype=float)
    dq_limit = np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float)
    u_limit = np.asarray(limits["u_max"][: cs.K_PLANNED_DOF], dtype=float)

    for index, name in enumerate(AXIS_NAMES):
        color = colors[index]
        q_axis.plot(time, q[:, index], color=color, lw=1.6, label=f"{name} actuated")
        q_axis.plot(
            time,
            data.q_ref[:, index],
            ":",
            color=color,
            lw=1.0,
            alpha=0.8,
            label=f"{name} reference",
        )
        q_axis.axhline(
            q_lower[index],
            color=color,
            ls="--",
            lw=0.8,
            alpha=0.35,
            label="position limit" if index == 0 else "_nolegend_",
        )
        q_axis.axhline(q_upper[index], color=color, ls="--", lw=0.8, alpha=0.35)

        dq_axis.plot(time, dq[:, index], color=color, lw=1.6, label=f"{name} actuated")
        dq_axis.plot(
            time,
            data.dq_ref[:, index],
            ":",
            color=color,
            lw=1.0,
            alpha=0.8,
            label=f"{name} reference",
        )
        dq_axis.axhline(
            dq_limit[index],
            color=color,
            ls="--",
            lw=0.8,
            alpha=0.35,
            label="velocity limit" if index == 0 else "_nolegend_",
        )
        dq_axis.axhline(-dq_limit[index], color=color, ls="--", lw=0.8, alpha=0.35)

        input_axis.step(
            control_time,
            data.control[:, index],
            where="post",
            color=color,
            lw=1.4,
            label=name,
        )
        input_axis.axhline(
            u_limit[index],
            color=color,
            ls="--",
            lw=0.8,
            alpha=0.35,
            label="input limit" if index == 0 else "_nolegend_",
        )
        input_axis.axhline(-u_limit[index], color=color, ls="--", lw=0.8, alpha=0.35)

        desired_velocity_axis.plot(
            time,
            data.desired_velocity[:, index],
            color=color,
            lw=1.6,
            label=name,
        )
        desired_velocity_axis.plot(
            time,
            data.dq_ref[:, index],
            ":",
            color=color,
            lw=1.0,
            alpha=0.8,
            label=f"{name} reference",
        )
        desired_velocity_axis.axhline(
            dq_limit[index],
            color=color,
            ls="--",
            lw=0.8,
            alpha=0.35,
            label="velocity limit" if index == 0 else "_nolegend_",
        )
        desired_velocity_axis.axhline(
            -dq_limit[index], color=color, ls="--", lw=0.8, alpha=0.35
        )

    q_axis.set_title("Full joint position trajectory")
    q_axis.set_ylabel("rad; telescope in m")
    dq_axis.set_title("Full joint velocity trajectory")
    dq_axis.set_ylabel("rad/s; telescope in m/s")

    q_u_limit = np.asarray(limits["q_u_max"], dtype=float)
    for index, name in enumerate(PASSIVE_NAMES):
        color = colors[cs.K_PLANNED_DOF + index]
        q_axis.plot(
            time,
            q_u[:, index],
            "--",
            color=color,
            lw=1.6,
            label=f"{name} passive",
        )
        q_axis.plot(
            time,
            data.q_eq[:, index],
            ":",
            color=color,
            lw=1.0,
            alpha=0.8,
            label=f"{name} equilibrium",
        )
        q_axis.plot(
            time,
            data.q_eq[:, index] - q_u_limit[index],
            "--",
            color=color,
            lw=0.8,
            alpha=0.35,
            label="passive position limit" if index == 0 else "_nolegend_",
        )
        q_axis.plot(
            time,
            data.q_eq[:, index] + q_u_limit[index],
            "--",
            color=color,
            lw=0.8,
            alpha=0.35,
        )

        dq_axis.plot(
            time,
            dq_u[:, index],
            "--",
            color=color,
            lw=1.6,
            label=f"{name} passive",
        )
        dq_axis.axhline(
            limits["dq_u_max"][index],
            color=color,
            ls="--",
            lw=0.8,
            alpha=0.35,
            label="passive velocity limit" if index == 0 else "_nolegend_",
        )
        dq_axis.axhline(
            -limits["dq_u_max"][index], color=color, ls="--", lw=0.8, alpha=0.35
        )

    q_axis.legend(ncol=3, fontsize=7)
    dq_axis.legend(ncol=3, fontsize=7)

    input_axis.set_title("Actuated acceleration input $u$")
    input_axis.set_ylabel("rad/s²; telescope in m/s²")
    input_axis.legend(ncol=3, fontsize=8)
    desired_velocity_axis.set_title("Desired velocity: integral of applied $u$")
    desired_velocity_axis.set_ylabel("rad/s; telescope in m/s")
    desired_velocity_axis.legend(ncol=3, fontsize=7)

    flow_to_l_min = 60.0e3
    planning_flow_limit = float(hydraulics["pump_flow_planning_factor"])
    pump_flow_limit = planning_flow_limit * float(hydraulics["pump_flow_max"])
    flow_axis.step(
        control_time,
        flow_to_l_min * data.pump_flow,
        where="post",
        color="black",
        lw=1.6,
        label="pump flow",
    )
    flow_axis.axhline(
        flow_to_l_min * pump_flow_limit,
        color="black",
        ls="--",
        lw=1.0,
        label="planning flow limit",
    )
    flow_axis.axhline(
        flow_to_l_min * float(hydraulics["pump_flow_max"]),
        color="black",
        ls="--",
        lw=0.8,
        alpha=0.45,
        label="pump maximum",
    )
    flow_axis.set_title("Pump flow")
    flow_axis.set_ylabel("L/min")
    flow_axis.legend(fontsize=8)

    solve_axis.plot(
        control_time, 1e3 * data.solve_time, color=colors[0], label="acados time"
    )
    solve_axis.axhline(
        1e3 * float(parameters["solve_budget"]),
        color="black",
        ls="--",
        label="node budget",
    )
    failed = data.status != 0
    if np.any(failed):
        solve_axis.scatter(
            control_time[failed],
            1e3 * data.solve_time[failed],
            color="red",
            label="failed",
        )
    if np.any(data.fallback):
        solve_axis.scatter(
            control_time[data.fallback],
            1e3 * data.solve_time[data.fallback],
            facecolors="none",
            edgecolors="orange",
            label="fallback",
        )
    solve_axis.set_title("RTI solver time")
    solve_axis.set_ylabel("ms")
    solve_axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("time [s]")
    figure.suptitle("crane_mpc offline A-to-B closed-loop simulation")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150)
    print(f"Wrote plot: {path}")
    if show:
        plt.show()
    plt.close(figure)


def print_summary(data: RunData, parameters: dict) -> None:
    """Print compact terminal, constraint, and timing metrics."""
    q = data.state[-1, : cs.K_PLANNED_DOF]
    final_error = q - data.q_ref[-1]
    sway = (
        data.state[:, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
        - data.q_eq
    )
    over_budget = data.solve_time > float(parameters["solve_budget"])
    print("\nSummary")
    print(f"  final per-axis error: {np.array2string(final_error, precision=5)}")
    print(f"  final error norm:     {np.linalg.norm(final_error):.6f}")
    print(f"  peak sway offset:     {np.max(np.abs(sway), axis=0)} rad")
    print(f"  peak hydraulic use:   {np.max(data.hydraulic_use, axis=0)}")
    print(
        f"  solve time median/max:{1e3 * np.median(data.solve_time):.2f}/{1e3 * np.max(data.solve_time):.2f} ms"
    )
    split = "  ".join(
        f"{field[5:]}={1e3 * np.nanmedian(data.timing[field]):.2f}"
        for field in TIMING_FIELDS
        if field.startswith("time_")
    )
    print(f"  acados median ms:     {split}")
    print(f"  qp iterations median: {np.nanmedian(data.timing['qp_iter']):.0f}")
    # box is shared; same solver measured 12.6ms idle vs 73ms at load 7.9 -- load matters
    print(
        f"  load average 1/5 min: {os.getloadavg()[0]:.2f} / {os.getloadavg()[1]:.2f}"
    )
    print(
        f"  nonzero statuses:     {np.count_nonzero(data.status)} / {data.status.size}"
    )
    print(
        f"  over budget:          {np.count_nonzero(over_budget)} / {over_budget.size}"
    )
    print(
        f"  fallback cycles:      {np.count_nonzero(data.fallback)} / {data.fallback.size}"
    )
    rate = data.state[:, cs.X_PROGRESS_RATE]
    print(
        f"  progress rate min/mean/max: {rate.min():.4f} / {rate.mean():.4f} / {rate.max():.4f}"
    )
    print(
        f"  virtual time spent:   {data.virtual_time[-1]:.3f} s of plan in "
        f"{data.time[-1]:.3f} s of wall clock"
    )


def main() -> int:
    """Run the command-line tuning workflow."""
    arguments = parse_arguments()
    try:
        parameters, hydraulics = load_settings(arguments)
        a, b = validate_movement(arguments, parameters)
        solver, model, scale = create_solver(arguments, parameters, hydraulics)
        data = simulate(arguments, parameters, hydraulics, solver, model, scale, a, b)
        if not arguments.no_plot:
            plot(
                arguments.output.resolve(), data, parameters, hydraulics, arguments.show
            )
        if not arguments.no_csv:
            csv_path = arguments.output.resolve().with_suffix(".csv")
            write_csv(csv_path, data)
            print(f"Wrote data: {csv_path}")
        print_summary(data, parameters)
        return 0
    except (KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
