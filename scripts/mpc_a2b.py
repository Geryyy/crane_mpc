#!/usr/bin/env python3
"""
Run the crane MPC offline on a joint-space A-to-B movement and plot it.

This is a developer tuning tool, not a second controller.  It imports
``export_ocp.py`` so the dynamics, cost residuals, constraints, integrator and
RTI backend are the same ones used to generate the deployed solver.  ROS, DDS
and a running controller manager are not required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

# Pinocchio registers an old and a new bool converter while the shared symbolic
# model imports.  That harmless binding warning otherwise obscures CLI help and
# every tuning summary.
warnings.filterwarnings(
    "ignore", message="to-Python converter for pinocchio.*", category=RuntimeWarning
)

import export_ocp  # noqa: E402

cs = export_ocp.cs

AXIS_NAMES = ("slew", "boom", "arm", "telescope", "rotator")
PASSIVE_NAMES = ("sway 1", "sway 2")
DEFAULT_A = np.array([0.0, 0.30, 0.80, 0.60, 0.0])
DEFAULT_B = np.array([0.60, 0.60, 1.20, 1.00, 0.50])


@dataclass
class RunData:
    """Closed-loop samples, including the terminal sample at ``time[-1]``."""

    time: np.ndarray
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


def parse_arguments() -> argparse.Namespace:
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
        "--show", action="store_true", help="also open the matplotlib window"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="recompile even when an identical solver is cached",
    )
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


def positive(value: float, name: str, allow_zero: bool = False) -> None:
    """Validate a finite positive or non-negative scalar."""
    good = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not good:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}, got {value}")


def load_settings(arguments: argparse.Namespace) -> tuple[dict, dict]:
    """Load deployment YAML and apply command-line tuning overrides."""
    # Same two reads `export_ocp.generate` makes, through the same helper, so the
    # harness cannot drift from the exporter about what a deployment is.
    parameters = export_ocp.ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    hydraulics = export_ocp.ox.read_ros_parameters(
        PACKAGE / "config" / "hydraulic_limits.yaml", "crane_mpc"
    )["hydraulics"]
    # Deep-copy through YAML because the source mapping is nested and is also
    # used to form the cache signature below.
    parameters = yaml.safe_load(yaml.safe_dump(parameters))
    if arguments.dt is not None:
        parameters["Ts"] = arguments.dt
    if arguments.horizon_knots is not None:
        parameters["horizon_length"] = arguments.horizon_knots
    if arguments.terminal_scale is not None:
        parameters["weights"]["terminal_scale"] = arguments.terminal_scale
    if arguments.levenberg_marquardt is not None:
        parameters["levenberg_marquardt"] = arguments.levenberg_marquardt

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
    # Payload inertia remains zero: crane_msgs/Payload is a point mass in this
    # stack, matching the runtime node's interpretation.
    return parameter


def solver_signature(parameters: dict, hydraulics: dict, description: Path) -> str:
    """Hash inputs that materially change generated solver code."""
    digest = hashlib.sha256()
    # W, all bounds and slack prices are set through the runtime API.  Keeping
    # them out of this signature is important: changing a cost multiplier is the
    # main tuning loop and must reuse the compiled solver.  Grid dimensions and
    # regularisation are part of the generated plan used here.
    generated_settings = {
        "Ts": parameters["Ts"],
        "horizon_length": parameters["horizon_length"],
        "levenberg_marquardt": parameters["levenberg_marquardt"],
    }
    digest.update(json.dumps(generated_settings, sort_keys=True, default=str).encode())
    digest.update(json.dumps(hydraulics, sort_keys=True, default=str).encode())
    digest.update(description.read_bytes())
    digest.update((PACKAGE / "scripts" / "export_ocp.py").read_bytes())
    # C3 is folded in as constants, so the model and the fit are generated code
    # here exactly as the exporter is.
    digest.update(Path(cs.__file__).read_bytes())
    digest.update(Path(cs.default_actuator_path()).read_bytes())
    return digest.hexdigest()[:16]


def create_solver(
    arguments: argparse.Namespace, parameters: dict, hydraulics: dict
) -> tuple[AcadosOcpSolver, object, np.ndarray]:
    """Load a cached solver or generate and compile an exact OCP solver."""
    description = export_ocp.DEFAULT_DESCRIPTIONS / export_ocp.DESCRIPTION
    signature = solver_signature(parameters, hydraulics, description)
    cache = PACKAGE / "build" / "mpc_a2b" / f"{export_ocp.TOOL}_{signature}"
    json_path = cache / "ocp.json"
    solver_name = f"{export_ocp.SOLVER_PREFIX}_{export_ocp.TOOL}"
    shared_library = cache / "code" / f"libacados_ocp_solver_{solver_name}.so"

    if arguments.rebuild and cache.is_dir():
        # This is a generated, ignored directory resolved below this package's
        # own build/mpc_a2b root; no source or user output can be selected here.
        shutil.rmtree(cache)

    if json_path.is_file() and shared_library.is_file():
        print(f"Reusing cached solver: {cache}")
        solver = AcadosOcpSolver(
            None,
            json_file=str(json_path),
            generate=False,
            build=False,
            verbose=arguments.verbose_build,
        )
        # The symbolic model is still needed for plant integration and plots.
        model = cs.CraneSymbolicModel(
            description.read_text(), export_ocp.TOOL, actuator=cs.load_actuator_fit()
        )
        scale = export_ocp.constraint_scale(model, hydraulics)
        return solver, model, scale

    cache.mkdir(parents=True, exist_ok=True)
    print(f"Generating and compiling solver in: {cache}")
    ocp, scale, model = export_ocp.build_ocp(
        description.read_text(), parameters, hydraulics
    )
    ocp.code_export_directory = str(cache / "code")
    # The devcontainer keeps the acados source (and t_renderer) under /opt but
    # installs the headers and libraries under /usr/local.  acados_template
    # otherwise derives unusable /opt/acados/{include,lib} paths from
    # ACADOS_SOURCE_DIR.  Native installs whose source tree contains its own
    # installed layout keep the paths build_ocp supplied.
    installed_prefix = Path("/usr/local")
    if (installed_prefix / "include" / "acados").is_dir() and (
        installed_prefix / "lib" / "libacados.so"
    ).is_file():
        ocp.acados_include_path = str(installed_prefix / "include")
        ocp.acados_lib_path = str(installed_prefix / "lib")

    AcadosOcpSolver.generate(ocp, json_file=str(json_path))
    # acados emits exact nonlinear-cost Hessians even under GAUSS_NEWTON.  They
    # are not registered by the solver (the checked-in deployment tree prunes
    # them for the same reason) and are by far the slowest files to compile.
    makefile = cache / "code" / "Makefile"
    makefile.write_text(
        "\n".join(
            line for line in makefile.read_text().splitlines() if "_hess.c" not in line
        )
        + "\n"
    )
    AcadosOcpSolver.build(
        str(cache / "code"), with_cython=False, verbose=arguments.verbose_build
    )
    if not shared_library.is_file():
        raise RuntimeError(
            "acados failed to build the shared solver; rerun with --verbose-build for details"
        )
    solver = AcadosOcpSolver(
        None,
        json_file=str(json_path),
        generate=False,
        build=False,
        verbose=arguments.verbose_build,
    )
    return solver, model, scale


def minimum_jerk(time: float, duration: float) -> tuple[float, float]:
    """Return quintic progress and its rate, with zero endpoint velocity."""
    if time <= 0.0:
        return 0.0, 0.0
    if time >= duration:
        return 1.0, 0.0
    s = time / duration
    position = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    rate = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    return position, rate


def reference_at(
    time: float, a: np.ndarray, b: np.ndarray, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the minimum-jerk A-to-B position and velocity reference."""
    progress, rate = minimum_jerk(time, duration)
    displacement = b - a
    return a + progress * displacement, rate * displacement


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


#: ERK4 substeps per control sample in the plant. C3 is stiff at `T_s`: the
#: fastest eigenvalue of the linearised plant is `|lambda| T_s = 8.5` at an
#: ordinary pose -- the telescope's `k = 3.5e6 N/m` against its effective mass --
#: and ERK4 is stable only to about 2.8, so a single explicit step diverges in
#: three samples and hands the solver a NaN guess. The solver itself is IRK and
#: does not need this; the plant here is explicit and does.
PLANT_SUBSTEPS = 10


def rk4_step(
    dynamics: ca.Function,
    state: np.ndarray,
    control: np.ndarray,
    parameter: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate one model-matched plant sample with substepped ERK4."""

    def evaluate(value: np.ndarray) -> np.ndarray:
        return np.asarray(dynamics(value, control, parameter)).reshape(-1)

    step = dt / PLANT_SUBSTEPS
    for _ in range(PLANT_SUBSTEPS):
        k1 = evaluate(state)
        k2 = evaluate(state + 0.5 * step * k1)
        k3 = evaluate(state + 0.5 * step * k2)
        k4 = evaluate(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def equilibrium_table(
    bias_u: ca.Function,
    parameter: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    duration: float,
    dt: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute references and continuous-branch passive equilibria."""
    times = dt * np.arange(count)
    q_ref = np.zeros((count, cs.K_PLANNED_DOF))
    dq_ref = np.zeros_like(q_ref)
    q_eq = np.zeros((count, cs.K_PASSIVE_DOF))
    guess = np.zeros(cs.K_PASSIVE_DOF)

    for index, time in enumerate(times):
        q_ref[index], dq_ref[index] = reference_at(time, a, b, duration)

        def residual(passive: np.ndarray) -> np.ndarray:
            state = np.zeros(cs.NX)
            state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF] = (
                q_ref[index]
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
    return q_ref, dq_ref, q_eq


def weight_matrices(parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Construct stage and terminal nonlinear-least-squares weights."""
    weights = parameters["weights"]
    planned = cs.K_PLANNED_DOF
    passive = cs.K_PASSIVE_DOF
    diagonal = np.concatenate(
        [
            np.asarray(weights["q_a"][:planned], dtype=float),
            np.asarray(weights["dq_a"][:planned], dtype=float),
            np.asarray(weights["q_u"][:passive], dtype=float),
            np.asarray(weights["dq_u"][:passive], dtype=float),
            np.asarray(weights["tau_a"][:planned], dtype=float),
            np.asarray(weights["u"][:planned], dtype=float),
        ]
    )
    terminal = float(weights["terminal_scale"]) * diagonal[: 2 * planned + 2 * passive]
    return np.diag(diagonal), np.diag(terminal)


def constraint_data(
    model: object, scale: np.ndarray, parameters: dict, hydraulics: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive scaled nonlinear bounds and physical force limits."""
    pressure = float(hydraulics["system_pressure_pa"])
    full = np.full(cs.K_ACTUATED_DOF, pressure)
    zero = np.zeros(cs.K_ACTUATED_DOF)
    extend = np.abs(np.asarray(ca.evalf(model.chamber_force(full, zero))).reshape(-1))
    retract = np.abs(np.asarray(ca.evalf(model.chamber_force(zero, full))).reshape(-1))
    lower_h = np.concatenate([-retract[: cs.K_PLANNED_DOF] / scale[: cs.NU], [0.0]])
    upper_h = np.concatenate(
        [
            extend[: cs.K_PLANNED_DOF] / scale[: cs.NU],
            [
                float(hydraulics["pump_flow_planning_factor"])
                * float(hydraulics["pump_flow_max"])
                / scale[cs.NU]
            ],
        ]
    )
    return lower_h, upper_h, extend[: cs.K_PLANNED_DOF], retract[: cs.K_PLANNED_DOF]


def configure_fixed_data(
    solver: AcadosOcpSolver,
    model: object,
    scale: np.ndarray,
    parameters: dict,
    hydraulics: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write runtime-tunable data that does not move during a run."""
    intervals = export_ocp.shooting_intervals(parameters)
    W, W_e = weight_matrices(parameters)
    lower_h, upper_h, extend, retract = constraint_data(
        model, scale, parameters, hydraulics
    )
    u_max = np.asarray(parameters["limits"]["u_max"][: cs.K_PLANNED_DOF], dtype=float)
    slack = parameters["slack"]
    soft_state = np.concatenate(
        [np.asarray(slack["q_u"], dtype=float), np.asarray(slack["dq_u"], dtype=float)]
    )
    soft_h = np.concatenate(
        [
            np.asarray(slack["cylinder_force"][: cs.K_PLANNED_DOF], dtype=float),
            [float(slack["pump_flow"])],
        ]
    )

    for stage in range(intervals + 1):
        if stage < intervals:
            solver.cost_set(stage, "W", W)
            solver.constraints_set(stage, "lbu", -u_max)
            solver.constraints_set(stage, "ubu", u_max)
            solver.constraints_set(stage, "lh", lower_h)
            solver.constraints_set(stage, "uh", upper_h)
        else:
            solver.cost_set(stage, "W", W_e)

        price = (
            soft_h
            if stage == 0
            else (soft_state if stage == intervals else np.r_[soft_state, soft_h])
        )
        zeros = np.zeros(price.size)
        solver.cost_set(stage, "Zl", zeros)
        solver.cost_set(stage, "Zu", zeros)
        solver.cost_set(stage, "zl", price)
        solver.cost_set(stage, "zu", price)
    return u_max, extend, retract


def state_bounds(
    parameters: dict, equilibrium: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build one stage's state box around its passive equilibrium."""
    limits = parameters["limits"]
    u_max = np.asarray(limits["u_max"][: cs.K_PLANNED_DOF], dtype=float)
    # The boxed rows are the rigid state and the lagged command; the force states
    # are bounded by constraint 6 and carry no box (`export_ocp.py`). `u_f` is a
    # filtered `u`, so it gets `u`'s own box.
    lag = u_max[list(cs.K_LAG_AXES)]
    lower = np.concatenate(
        [
            np.asarray(limits["q_a_lower"][: cs.K_PLANNED_DOF], dtype=float),
            equilibrium - np.asarray(limits["q_u_max"], dtype=float),
            -np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float),
            -np.asarray(limits["dq_u_max"], dtype=float),
            -lag,
        ]
    )
    upper = np.concatenate(
        [
            np.asarray(limits["q_a_upper"][: cs.K_PLANNED_DOF], dtype=float),
            equilibrium + np.asarray(limits["q_u_max"], dtype=float),
            np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float),
            np.asarray(limits["dq_u_max"], dtype=float),
            lag,
        ]
    )
    return lower, upper


def stage_reference(
    q_ref: np.ndarray, dq_ref: np.ndarray, q_eq: np.ndarray, terminal: bool
) -> np.ndarray:
    """Build an acados stage or terminal residual reference."""
    base = np.concatenate([q_ref, dq_ref, q_eq, np.zeros(cs.K_PASSIVE_DOF)])
    if terminal:
        return base
    return np.concatenate([base, np.zeros(2 * cs.K_PLANNED_DOF)])


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
) -> RunData:
    """Run the receding-horizon controller against the model-matched plant."""
    dt = float(parameters["Ts"])
    intervals = export_ocp.shooting_intervals(parameters)
    steps = int(math.ceil((arguments.move_duration + arguments.settle_duration) / dt))
    parameter = parameter_vector(arguments)
    dynamics, bias_u, outputs, static_force = make_numeric_functions(model)
    q_ref_all, dq_ref_all, q_eq_all = equilibrium_table(
        bias_u, parameter, a, b, arguments.move_duration, dt, steps + intervals + 1
    )
    u_max, extend, retract = configure_fixed_data(
        solver, model, scale, parameters, hydraulics
    )

    state = np.zeros(cs.NX)
    state[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF] = a
    state[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF] = q_eq_all[
        0
    ]
    # C3 block 3 starts where the machine starts: holding its own weight. There
    # is no force measurement in the stack, so h_eff at the initial pose is the
    # seed, and zero would start the run with the hydraulics switched off.
    state[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + cs.K_PLANNED_DOF] = np.asarray(
        static_force(state, parameter)
    ).reshape(-1)

    states = np.zeros((steps + 1, cs.NX))
    controls = np.zeros((steps, cs.NU))
    hydraulics_used = np.zeros((steps, cs.NU + 1))
    solve_times = np.zeros(steps)
    statuses = np.zeros(steps, dtype=int)
    fallback = np.zeros(steps, dtype=bool)
    states[0] = state

    previous_x: list[np.ndarray] | None = None
    previous_u: list[np.ndarray] | None = None
    budget = float(parameters["solve_budget"])

    for step in range(steps):
        solver.reset(reset_qp_solver_mem=1)
        for stage in range(intervals + 1):
            index = step + stage
            solver.set(stage, "p", parameter)
            solver.cost_set(
                stage,
                "yref",
                stage_reference(
                    q_ref_all[index],
                    dq_ref_all[index],
                    q_eq_all[index],
                    stage == intervals,
                ),
            )
            if stage == 0:
                lower = upper = state
            else:
                lower, upper = state_bounds(parameters, q_eq_all[index])
            solver.constraints_set(stage, "lbx", lower)
            solver.constraints_set(stage, "ubx", upper)

        if previous_x is None or previous_u is None:
            guess_x = [state.copy()]
            for _ in range(intervals):
                guess_x.append(
                    rk4_step(dynamics, guess_x[-1], np.zeros(cs.NU), parameter, dt)
                )
            guess_u = [np.zeros(cs.NU) for _ in range(intervals)]
        else:
            guess_x = previous_x[1:] + [previous_x[-1].copy()]
            guess_u = previous_u[1:] + [previous_u[-1].copy()]

        for stage in range(intervals + 1):
            value = guess_x[stage].copy()
            if stage == 0:
                value = state.copy()
            else:
                lower, upper = state_bounds(parameters, q_eq_all[step + stage])
                # Only the boxed prefix has bounds; the force states are held by
                # constraint 6 and are left as the rollout produced them.
                boxed = lower.size
                value[:boxed] = np.clip(value[:boxed], lower, upper)
            solver.set(stage, "x", value)
            if stage < intervals:
                solver.set(stage, "u", np.clip(guess_u[stage], -u_max, u_max))

        status = int(solver.solve())
        solve_time = float(solver.get_stats("time_tot"))
        statuses[step] = status
        solve_times[step] = solve_time
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
            previous_x, previous_u = candidate_x, candidate_u
        elif previous_u is not None:
            # Same shifted-previous-plan fallback used by mpc_node.
            control = (
                previous_u[1].copy() if len(previous_u) > 1 else previous_u[0].copy()
            )
            previous_x = previous_x[1:] + [previous_x[-1].copy()]
            previous_u = previous_u[1:] + [previous_u[-1].copy()]
            fallback[step] = True
        else:
            control = np.zeros(cs.NU)
            fallback[step] = True

        controls[step] = control
        hydraulics_used[step] = hydraulic_utilisation(
            outputs, state, control, parameter, extend, retract, hydraulics
        )
        state = rk4_step(dynamics, state, control, parameter, dt)
        states[step + 1] = state

        if (
            step == 0
            or (step + 1) % max(1, int(round(1.0 / dt))) == 0
            or step + 1 == steps
        ):
            error = np.linalg.norm(state[: cs.K_PLANNED_DOF] - q_ref_all[step + 1])
            print(
                f"t={(step + 1) * dt:6.2f} s  status={status:2d}  "
                f"solve={1e3 * solve_time:7.2f} ms  |q-qref|={error:.4f}"
            )

    # `u` is a joint velocity at Psi's input under C3, not an acceleration, so
    # the desired velocity is the command and nothing is integrated to get it.
    desired_velocity = np.vstack(
        [
            states[0, cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF],
            controls,
        ]
    )
    flow_limit = float(hydraulics["pump_flow_planning_factor"]) * float(
        hydraulics["pump_flow_max"]
    )

    return RunData(
        time=dt * np.arange(steps + 1),
        state=states,
        q_ref=q_ref_all[: steps + 1],
        dq_ref=dq_ref_all[: steps + 1],
        q_eq=q_eq_all[: steps + 1],
        control=controls,
        desired_velocity=desired_velocity,
        hydraulic_use=hydraulics_used,
        pump_flow=hydraulics_used[:, -1] * flow_limit,
        solve_time=solve_times,
        status=statuses,
        fallback=fallback,
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
                row += data.control[index].tolist()
                row += data.hydraulic_use[index].tolist()
                row += [data.pump_flow[index]]
                row += [
                    data.solve_time[index],
                    int(data.status[index]),
                    int(data.fallback[index]),
                ]
            else:
                row += [math.nan] * (cs.NU + cs.NU + 2)
                row += [math.nan, 0, 0]
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
    print(
        f"  nonzero statuses:     {np.count_nonzero(data.status)} / {data.status.size}"
    )
    print(
        f"  over budget:          {np.count_nonzero(over_budget)} / {over_budget.size}"
    )
    print(
        f"  fallback cycles:      {np.count_nonzero(data.fallback)} / {data.fallback.size}"
    )


def main() -> int:
    """Run the command-line tuning workflow."""
    arguments = parse_arguments()
    try:
        parameters, hydraulics = load_settings(arguments)
        a, b = validate_movement(arguments, parameters)
        solver, model, scale = create_solver(arguments, parameters, hydraulics)
        data = simulate(arguments, parameters, hydraulics, solver, model, scale, a, b)
        plot(arguments.output.resolve(), data, parameters, hydraulics, arguments.show)
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
