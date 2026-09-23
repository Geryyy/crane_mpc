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
from dataclasses import dataclass, replace
from pathlib import Path

import casadi as ca
import numpy as np
import yaml
from acados_template import AcadosOcpSolver
from scipy.optimize import root

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "scripts"))

# `crane_ocp_export` is installed (crane_ocp declares it), but these scripts are
# meant to run from a source checkout without the overlay sourced, the same way
# they fall back to the source tree for `crane_mpc` and `crane_model`. The
# runtime imports it too -- `crane_mpc.solver` does -- so this has to go on the
# path *before* that module, not just before the exporter.
try:
    import crane_ocp_export  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PACKAGE.parent / "crane_ocp"))

import crane_ocp_export as ox  # noqa: E402

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

    `control` is the path in its own parameter. The timing law that used to ride
    alongside it went when `s` became the path parameter: where on the path to be
    is the optimizer's choice now, so nothing here reads a `sigma(t)`.
    """

    control: np.ndarray


def line_path(a: np.ndarray, b: np.ndarray, duration: float) -> Path:
    """Build this script's own A-to-B: a joint-space line, quintic in time."""
    samples = np.linspace(a, b, 4 * ocp_runtime.problem.PATH_POINTS)
    return Path(control=ocp_runtime.path_control(samples))


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
    #: Cycles the run was asked for. A `settle_gate` may have kept the loop going
    #: past it; everything after is coast, and scoring it would make the number
    #: depend on how long somebody held a key.
    scored: int = 0


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
    parameters["weights"]["progress"] = arguments.progress_scale * float(
        parameters["weights"]["progress"]
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
) -> tuple[ocp_runtime.Ocp, object, np.ndarray]:
    """
    Open this problem's compiled solver, exporting it first if need be.

    Returns the node's own `Ocp`, not a bare acados handle: everything a cycle
    writes lives there, so this harness and the node cannot drift apart without
    a test noticing.

    **This exports; the node never does.** A sweep walks configurations that no
    deliberate export step could have anticipated -- `sweep_ocp.py` alone visits
    one per acados variant -- so requiring a hand-run export per variant would
    make a sweep unusable. It is the same `ox.export` the export script calls,
    into a directory named for the problem, so variants do not clobber each
    other and a repeat run opens what the last one built.
    """
    description = (export_ocp.DEFAULT_DESCRIPTIONS / export_ocp.DESCRIPTION).read_text()
    key = ocp_runtime.export_key(parameters, hydraulics, description)
    base = ocp_runtime.export_base()
    if arguments.rebuild:
        # generated export dir, named for this problem; no source or user output
        shutil.rmtree(ox.solver_root(base, key), ignore_errors=True)
    acados_ocp, scale, model, _chamber = export_ocp.build_ocp(
        description, parameters, hydraulics
    )
    try:
        ox.manifest(base, key, "the harness exports on demand")
    except ox.StaleExport:
        ox.export(
            acados_ocp,
            base,
            key,
            sims=ocp_runtime.predictor_sims(acados_ocp, parameters),
            verbose=arguments.verbose_build,
        )
    ocp = ocp_runtime.Ocp(
        description, parameters, hydraulics, verbose=arguments.verbose_build
    )
    return ocp, model, scale


def solver_stat(solver: AcadosOcpSolver, field: str) -> float:
    """
    One acados statistic per solve, as a scalar.

    `qp_iter` comes back per SQP iteration (one entry under RTI); summing keeps
    the column meaningful if that changes.
    """
    return float(np.sum(np.asarray(solver.get_stats(field), dtype=float)))


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
    Precompute continuous-branch passive equilibria along the **path**.

    Indexed by the path parameter, so a run that spends the path slowly reads
    the same table more slowly rather than needing a second one; the grid is
    uniform in theta and `equilibrium_at` interpolates between knots.

    `guess` seeds the first solve, later ones continue from their predecessor,
    picking the branch the table sits on. Zero suits a pose near the origin; a
    tilt near pi/2 (`initialization_outside.yaml`) converges onto a different
    solution, simulating a machine holding its load sideways.
    """
    times = np.linspace(0.0, 1.0, count)
    q_eq = np.zeros((count, cs.K_PASSIVE_DOF))
    guess = (
        np.zeros(cs.K_PASSIVE_DOF)
        if guess is None
        else np.asarray(guess, dtype=float).reshape(cs.K_PASSIVE_DOF).copy()
    )

    for index, time in enumerate(times):
        q_ref_index = reference(time)

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


def _grown(array: np.ndarray, extra: int) -> np.ndarray:
    """Append `extra` zero rows to a log, keeping every axis past the first."""
    return np.concatenate([array, np.zeros((extra, *array.shape[1:]), array.dtype)])


def simulate(
    arguments: argparse.Namespace,
    parameters: dict,
    hydraulics: dict,
    ocp: ocp_runtime.Ocp,
    model: object,
    scale: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    plant=None,
    passive_guess=None,
    horizon=None,
    path=None,
    off_path=None,
    settle_gate=None,
) -> RunData:
    """
    Run the receding-horizon controller against a plant.

    `plant` defaults to a model-matched ERK4 rollout; passing one replaces that
    half and leaves the controller alone. It is
    `(state, control, parameter, dt, knots) -> state` over the full `cs.NX` --
    must carry C3's lag/progress/force rows since the controller reads them back.
    `knots` is the pair of planned positions the node would publish for this
    interval, which a plant running an inner loop tracks and the default ignores.
    `passive_guess` picks the equilibrium branch, i.e. where the run starts.

    `horizon`, if given, is handed the (N+1, NX) states in force each cycle --
    the accepted solution, or the shifted previous one where a solve was refused.

    `off_path`, if given, is asked for the tool's distance off the path in
    metres whenever the progress line prints. It is the caller's metric, not
    this function's: everything here is joint space, and the number worth
    watching is millimetres at the tool.

    `settle_gate`, if given, is asked at the end of the run whether to keep
    cycling; while it says yes the whole chain keeps running with the path
    already spent, which is how the sway after arrival gets to be watched. Those
    cycles are logged but sit past `RunData.scored`.
    """
    dt = float(parameters["Ts"])
    intervals = export_ocp.shooting_intervals(parameters)
    steps = int(math.ceil((arguments.move_duration + arguments.settle_duration) / dt))
    base_parameter = parameter_vector(arguments)
    dynamics, bias_u, outputs, static_force = make_numeric_functions(model)
    if plant is None:

        def plant(state, control, parameter, step_s, knots=None):
            return rk4_step(dynamics, state, control, parameter, step_s)

    # The cost reads a path, not samples of one. Absent a planned curve this is
    # the straight line the quintic above walks.
    if path is None:
        path = line_path(a, b, arguments.move_duration)
    path_knots = ocp_runtime.problem.PATH_KNOTS

    def on_path(theta):
        """Give the planned pose at a place on the path, which is what the cost reads."""
        return ocp_runtime.bspline.value(
            np.clip(theta, 0.0, 1.0), path.control, path_knots
        )[0]

    # Along the path, not along a clock: `origin` is where on it this cycle
    # starts, and the table is read at the same parameter the cost is.
    table_count = steps + intervals + 1
    eq_times, q_eq_table = equilibrium_table(
        bias_u, base_parameter, on_path, dt, table_count, passive_guess
    )
    # Written by `Ocp` at construction, against the same weights and bounds the
    # node writes; read back here only for the hydraulic-utilisation column.
    extend = ocp.cylinder_force_max
    retract = ocp.cylinder_force_retract

    state = np.zeros(cs.NX)
    state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF] = a
    state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = (
        equilibrium_at(eq_times, q_eq_table, 0.0)
    )
    # The path parameter starts at rest; what spends it is the row that wants it
    # at the end, not a nominal rate.
    state[cs.X_PROGRESS_RATE] = 0.0
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

    #: last accepted solve, the warm start for the next cycle
    previous: ocp_runtime.Solution | None = None
    # The plan's own pace, as path parameter per second: what the progress row
    # asks for, until the end of the path is nearer than that. This path is the
    # whole move, not a horizon window, so the nominal is the move's own and
    # `problem.nominal_progress_rate` (which is the node's window) is not it.
    nominal_rate = 1.0 / float(arguments.move_duration)
    rate_max = float(parameters["limits"]["progress_rate_headroom"]) * nominal_rate

    # Where on the path this cycle starts, in [0, 1]. `s` is pinned to zero each
    # cycle and this carries what the last solve bought -- the same bookkeeping
    # as when it was virtual time, in the parameter the cost now reads.
    origin = 0.0

    #: what the run was asked for; `settle_gate` may push `steps` past it.
    scored = steps
    settle_chunk = max(1, int(round(1.0 / dt)))

    step = 0
    while step < steps:
        virtual_time[step] = origin
        q_ref_log[step] = on_path(origin)
        dq_ref_log[step] = state[
            cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF
        ]
        q_eq_log[step] = equilibrium_at(eq_times, q_eq_table, origin)
        # Everything the cycle writes is `Ocp`'s, so this harness and the node
        # solve the same problem by construction rather than by inspection.
        # What stays here is what the node gets from elsewhere: the path (the
        # planner's, once stage 2 lands) and a sway equilibrium solved along it
        # instead of frozen at the measurement.
        path_cycle = ocp_runtime.PathCycle(
            control=path.control,
            origin=origin,
            nominal_rate=nominal_rate,
        )
        q_eq_stage = np.array(
            [
                equilibrium_at(eq_times, q_eq_table, origin + stage * dt * rate_max)
                for stage in range(intervals + 1)
            ]
        )
        # `Ocp.shifted` is the warm start: last horizon one knot left, with the
        # progress row re-origined by what that solve bought. Rebuilding the
        # shift here is what let this harness and the node drift.
        guess = None if previous is None else ocp.carried(previous)
        solution = ocp.solve(state, None, q_eq_stage, guess, path=path_cycle)

        status = solution.status
        solve_time = solution.solve_time_s
        statuses[step] = status
        solve_times[step] = solve_time
        for field in TIMING_FIELDS:
            value = solver_stat(ocp.solver, field)
            if field in CUMULATIVE_FIELDS:
                value, cumulative[field] = value - cumulative[field], value
            timings[field][step] = value
        # `Ocp` calls a slow solve BUDGET_EXCEEDED whether or not this run cares;
        # --enforce-budget is what decides whether that costs the cycle.
        accepted = solution.outcome is ocp_runtime.Outcome.CONVERGED or (
            solution.outcome is ocp_runtime.Outcome.BUDGET_EXCEEDED
            and not arguments.enforce_budget
        )

        if accepted:
            control = solution.inputs[0].copy()
            advance = solution.progress_advance
            previous = solution
        elif previous is not None:
            # Same shifted-previous-plan fallback used by mpc_node. The path
            # parameter stays where it is: a refused solve bought no path.
            inputs = previous.inputs
            control = (inputs[1] if len(inputs) > 1 else inputs[0]).copy()
            advance = 0.0
            carried = ocp.shifted(previous)
            previous = replace(previous, states=carried.states, inputs=carried.inputs)
            fallback[step] = True
        else:
            control = np.zeros(cs.NU_PROGRESS)
            advance = 0.0
            fallback[step] = True

        controls[step] = control
        hydraulics_used[step] = hydraulic_utilisation(
            outputs, state, control, base_parameter, extend, retract, hydraulics
        )
        if horizon is not None and previous is not None:
            horizon(previous.states)
        # What the node publishes as this interval's position reference: the
        # plan, one interval on -- the plan and not a roll of the measurement,
        # which is what gives the loop underneath a `p e_pos` to integrate.
        # `Cycle.adopt_solution` writes the same curve, but at the progress the
        # solve chose rather than at the nominal pace used here, so the two
        # differ by exactly as much as the MPC has slowed the plan down.
        knots = np.array(
            [on_path(origin), on_path(min(1.0, origin + dt * nominal_rate))]
        )
        state = plant(state, control, base_parameter, dt, knots)
        states[step + 1] = state
        origin = min(
            1.0,
            origin + (advance if math.isfinite(advance) and advance >= 0.0 else 0.0),
        )

        if (
            step == 0
            or (step + 1) % max(1, int(round(1.0 / dt))) == 0
            or step + 1 == steps
        ):
            reached = on_path(origin)
            error = np.linalg.norm(state[: cs.K_PLANNED_DOF] - reached)
            # `|q-qref|` is joint space against the path point at this `theta`;
            # `off path` is millimetres at the tool against the path as a set,
            # which is what the move is finally scored on.
            gap = "" if off_path is None else f"  off path={1e3 * off_path():6.1f} mm"
            print(
                f"t={(step + 1) * dt:6.2f} s  theta={origin:5.3f}  "
                f"v_s={state[cs.X_PROGRESS_RATE]:5.3f}  status={status:2d}  "
                f"solve={1e3 * solve_time:7.2f} ms  |q-qref|={error:.4f}{gap}"
            )

        step += 1
        if step == steps and settle_gate is not None and settle_gate():
            # Asked for is over and somebody is still watching: keep the whole
            # chain turning with the path spent, so the sway after arrival is
            # something you can look at. Grown a chunk at a time -- a coast has
            # no length to preallocate for.
            steps += settle_chunk
            states = _grown(states, settle_chunk)
            q_ref_log = _grown(q_ref_log, settle_chunk)
            dq_ref_log = _grown(dq_ref_log, settle_chunk)
            q_eq_log = _grown(q_eq_log, settle_chunk)
            virtual_time = _grown(virtual_time, settle_chunk)
            controls = _grown(controls, settle_chunk)
            hydraulics_used = _grown(hydraulics_used, settle_chunk)
            solve_times = _grown(solve_times, settle_chunk)
            statuses = _grown(statuses, settle_chunk)
            fallback = _grown(fallback, settle_chunk)
            timings = {
                key: _grown(value, settle_chunk) for key, value in timings.items()
            }

    virtual_time[steps] = origin
    q_ref_log[steps] = on_path(origin)
    dq_ref_log[steps] = state[
        cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF
    ]
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
        scored=scored,
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
    # The coast a `settle_gate` added is however many cycles somebody watched
    # for; reading it here would put that into every number below.
    end = data.scored or data.status.size
    q = data.state[end, : cs.K_PLANNED_DOF]
    final_error = q - data.q_ref[end]
    sway = (
        data.state[
            : end + 1, cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF
        ]
        - data.q_eq[: end + 1]
    )
    over_budget = data.solve_time[:end] > float(parameters["solve_budget"])
    print("\nSummary")
    print(f"  final per-axis error: {np.array2string(final_error, precision=5)}")
    print(f"  final error norm:     {np.linalg.norm(final_error):.6f}")
    print(f"  peak sway offset:     {np.max(np.abs(sway), axis=0)} rad")
    print(f"  peak hydraulic use:   {np.max(data.hydraulic_use[:end], axis=0)}")
    print(
        f"  solve time median/max:{1e3 * np.median(data.solve_time[:end]):.2f}/"
        f"{1e3 * np.max(data.solve_time[:end]):.2f} ms"
    )
    split = "  ".join(
        f"{field[5:]}={1e3 * np.nanmedian(data.timing[field][:end]):.2f}"
        for field in TIMING_FIELDS
        if field.startswith("time_")
    )
    print(f"  acados median ms:     {split}")
    print(f"  qp iterations median: {np.nanmedian(data.timing['qp_iter'][:end]):.0f}")
    # box is shared; same solver measured 12.6ms idle vs 73ms at load 7.9 -- load matters
    print(
        f"  load average 1/5 min: {os.getloadavg()[0]:.2f} / {os.getloadavg()[1]:.2f}"
    )
    print(f"  nonzero statuses:     {np.count_nonzero(data.status[:end])} / {end}")
    print(
        f"  over budget:          {np.count_nonzero(over_budget)} / {over_budget.size}"
    )
    print(f"  fallback cycles:      {np.count_nonzero(data.fallback[:end])} / {end}")
    rate = data.state[: end + 1, cs.X_PROGRESS_RATE]
    print(
        f"  progress rate min/mean/max: {rate.min():.4f} / {rate.mean():.4f} / {rate.max():.4f}"
    )
    print(
        f"  virtual time spent:   {data.virtual_time[end]:.3f} s of plan in "
        f"{data.time[end]:.3f} s of wall clock"
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
