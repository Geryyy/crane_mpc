"""
The generated acados solver, opened and driven: `src/ocp_solver.cpp` in Python.

acados' own artifact is unchanged -- the same problem `problem.build_ocp`
assembles, generated to C and compiled to a shared library, opened through
`acados_template.AcadosOcpSolver`. What is here is the half the C++ called the
binding: what is written onto the solver before a solve, what is read back
after, and the two things that are not the solver -- the dead-time predictor and
the static hold force.

**The state is the OCP's own 25 rows end to end.** The C++ carried a 16-wide
`crane_model::State` at this boundary and truncated the command-lag and force
rows on every crossing, which is what made its dead-time predictor a coast at a
frozen force (issue 125). There is no fixed-width state here to truncate to.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from crane_model import symbolic as cs

from . import config, problem

#: Slack below this is round-off, not a violation (`ocp_solver.hpp` kSlackNoticeable).
SLACK_NOTICEABLE = 1e-6

#: acados' own return codes, spelled as `crane_ocp::status_word` spells them.
STATUS_WORDS = {
    0: "ACADOS_SUCCESS",
    1: "ACADOS_NAN_DETECTED",
    2: "ACADOS_MAXITER",
    3: "ACADOS_MINSTEP",
    4: "ACADOS_QP_FAILURE",
    5: "ACADOS_READY",
    6: "ACADOS_TIMEOUT",
}


def status_word(status: int) -> str:
    return STATUS_WORDS.get(int(status), f"acados status {int(status)}")


class Outcome(Enum):
    CONVERGED = "converged"
    BUDGET_EXCEEDED = "the solve exceeded its budget"
    FAILED = "the solve failed"

    def __str__(self) -> str:
        return self.value


@dataclass
class Violation:
    """The worst slack taken on each softened constraint, over the horizon."""

    q_u: float = 0.0
    dq_u: float = 0.0
    cylinder_force: float = 0.0
    pump_flow: float = 0.0

    def worst(self) -> float:
        return max(self.q_u, self.dq_u, self.cylinder_force, self.pump_flow)


@dataclass
class Guess:
    """A warm start: `N+1` states and the `N` inputs between them."""

    states: np.ndarray
    inputs: np.ndarray

    def warm(self, intervals: int) -> bool:
        return (
            self.states.shape == (intervals + 1, cs.NX)
            and self.inputs.shape == (intervals, cs.NU_PROGRESS)
            and np.all(np.isfinite(self.states))
            and np.all(np.isfinite(self.inputs))
        )


@dataclass
class Solution:
    states: np.ndarray  # (N+1, NX)
    inputs: np.ndarray  # (N, NU_PROGRESS)
    outcome: Outcome
    status: int
    status_word: str
    qp_status: int
    iterations: int
    qp_iterations: int
    solve_time_s: float
    used_slack: bool
    slack_penalty: float
    violation: Violation
    progress_advance: float
    warm_started: bool

    @property
    def u0(self) -> np.ndarray:
        return self.inputs[0]


@dataclass
class CostTerms:
    """`wiki/mpc.md` §2 term by term, summed over the horizon."""

    q_a: float = 0.0
    dq_a: float = 0.0
    q_u: float = 0.0
    dq_u: float = 0.0
    lag: float = 0.0
    progress: float = 0.0
    tau_a: float = 0.0
    u: float = 0.0
    terminal: float = 0.0
    slack: float = 0.0


# ------------------------------------------------------- what the solver is told


def weight_matrices(parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stage and terminal nonlinear-least-squares weights, `wiki/mpc.md` §2."""
    weights = parameters["weights"]
    planned = cs.K_PLANNED_DOF
    passive = cs.K_PASSIVE_DOF
    diagonal = np.concatenate(
        [
            np.asarray(weights["q_a"][:planned], dtype=float),
            np.asarray(weights["dq_a"][:planned], dtype=float),
            np.asarray(weights["q_u"][:passive], dtype=float),
            np.asarray(weights["dq_u"][:passive], dtype=float),
            [float(weights["lag"]), float(weights["progress_rate"])],
            np.asarray(weights["tau_a"][:planned], dtype=float),
            np.asarray(weights["u"][:planned], dtype=float),
            [float(weights["progress_accel"])],
        ]
    )
    terminal = float(weights["terminal_scale"]) * diagonal[: problem.NY_TERMINAL]
    return np.diag(diagonal), np.diag(terminal)


def constraint_data(
    model, scale: np.ndarray, parameters: dict, hydraulics: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scaled bounds of `h` (constraints 6 and 7) and the physical force limits."""
    extend, retract = problem.chamber_forces(
        model, float(hydraulics["system_pressure_pa"]), cs.K_ACTUATED_DOF
    )
    extend = np.abs(extend)
    retract = np.abs(retract)
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


def position_box(
    parameters: dict, measured: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constraint 1's box with `q_a_margin`, never excluding `measured`.

    A margin that excluded the pose the machine is in would make stage 1 chase a
    position no input reaches in one interval.
    """
    limits = parameters["limits"]
    margin = np.asarray(limits["q_a_margin"][: cs.K_PLANNED_DOF], dtype=float)
    lower = np.asarray(limits["q_a_lower"][: cs.K_PLANNED_DOF], dtype=float) + margin
    upper = np.asarray(limits["q_a_upper"][: cs.K_PLANNED_DOF], dtype=float) - margin
    return np.minimum(lower, measured), np.maximum(upper, measured)


def state_bounds(
    parameters: dict, equilibrium: np.ndarray, measured: np.ndarray, elapsed: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one stage's box over the boxed prefix of `x`.

    The rigid rows, the lagged command and the progress pair. The force rows
    carry no box -- constraint 6 holds them, because their transmission is not
    constant.
    """
    limits = parameters["limits"]
    u_max = np.asarray(limits["u_max"][: cs.K_PLANNED_DOF], dtype=float)
    q_lower, q_upper = position_box(parameters, measured)
    lag = u_max[list(cs.K_LAG_AXES)]
    rate_max = float(limits["progress_rate_max"])
    lower = np.concatenate(
        [
            q_lower,
            equilibrium - np.asarray(limits["q_u_max"], dtype=float),
            -np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float),
            -np.asarray(limits["dq_u_max"], dtype=float),
            -lag,
            [0.0, 0.0],
        ]
    )
    upper = np.concatenate(
        [
            q_upper,
            equilibrium + np.asarray(limits["q_u_max"], dtype=float),
            np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float),
            np.asarray(limits["dq_u_max"], dtype=float),
            lag,
            [elapsed * rate_max, rate_max],
        ]
    )
    return lower, upper


def stage_reference(
    q_eq: np.ndarray, tau_ref: np.ndarray, terminal: bool
) -> np.ndarray:
    """
    Build an acados stage or terminal residual reference.

    **The tracking rows are zero.** The reference is evaluated at the progress
    state, so it rides in `p` and the residual carries the error itself.
    `tau_ref` is `h_eff`: a zero reference there prices the machine holding its
    own weight and buys droop (issue 117).
    """
    reference = np.zeros(problem.NY_TERMINAL if terminal else problem.NY)
    reference[
        problem.Y_PASSIVE_POSITION : problem.Y_PASSIVE_POSITION + cs.K_PASSIVE_DOF
    ] = q_eq
    reference[problem.Y_PROGRESS_RATE] = problem.K_PROGRESS_RATE_REFERENCE
    if not terminal:
        reference[
            problem.Y_ACTUATED_FORCE : problem.Y_ACTUATED_FORCE + cs.K_PLANNED_DOF
        ] = tau_ref
    return reference


def stage_parameters(
    base: np.ndarray,
    nominal: float,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    ddq_ref: np.ndarray,
) -> np.ndarray:
    """One stage's local reference model, bound into the acados parameter vector."""
    parameter = np.zeros(problem.NP)
    parameter[: cs.NP] = base
    parameter[problem.P_PROGRESS_NOMINAL] = nominal
    parameter[
        problem.P_REFERENCE_POSITION : problem.P_REFERENCE_POSITION + cs.K_PLANNED_DOF
    ] = q_ref
    parameter[
        problem.P_REFERENCE_FIRST : problem.P_REFERENCE_FIRST + cs.K_PLANNED_DOF
    ] = dq_ref
    parameter[
        problem.P_REFERENCE_SECOND : problem.P_REFERENCE_SECOND + cs.K_PLANNED_DOF
    ] = ddq_ref
    return parameter


def slack_prices(parameters: dict) -> dict:
    """Price each softened row, per kind of stage."""
    slack = parameters["slack"]
    state = np.concatenate(
        [
            np.asarray(slack["q_u"], dtype=float),
            np.asarray(slack["dq_u"], dtype=float),
        ]
    )
    nonlinear = np.concatenate(
        [
            np.asarray(slack["cylinder_force"][: cs.K_PLANNED_DOF], dtype=float),
            [float(slack["pump_flow"])],
        ]
    )
    return {
        "initial": nonlinear,
        "path": np.concatenate([state, nonlinear]),
        "terminal": state,
    }


def configure_fixed_data(
    solver: AcadosOcpSolver,
    model,
    scale: np.ndarray,
    parameters: dict,
    hydraulics: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Write what is runtime-settable and does not move during a run.

    Every weight, every bound of constraints 5, 6 and 7 and every slack price.
    What moves per cycle -- `yref`, the state box and `p` -- is written in
    `Ocp.solve`.
    """
    intervals = problem.shooting_intervals(parameters)
    weight, terminal_weight = weight_matrices(parameters)
    lower_h, upper_h, extend, retract = constraint_data(
        model, scale, parameters, hydraulics
    )
    # The input box is the five joint commands and then the progress
    # acceleration, which is not an actuated axis and has its own bound.
    u_max = np.concatenate(
        [
            np.asarray(parameters["limits"]["u_max"][: cs.K_PLANNED_DOF], dtype=float),
            [float(parameters["limits"]["progress_accel_max"])],
        ]
    )
    price = slack_prices(parameters)
    for stage in range(intervals + 1):
        if stage < intervals:
            solver.cost_set(stage, "W", weight)
            solver.constraints_set(stage, "lbu", -u_max)
            solver.constraints_set(stage, "ubu", u_max)
            solver.constraints_set(stage, "lh", lower_h)
            solver.constraints_set(stage, "uh", upper_h)
        else:
            solver.cost_set(stage, "W", terminal_weight)
        kind = (
            "initial" if stage == 0 else ("terminal" if stage == intervals else "path")
        )
        zeros = np.zeros(price[kind].size)
        solver.cost_set(stage, "Zl", zeros)
        solver.cost_set(stage, "Zu", zeros)
        solver.cost_set(stage, "zl", price[kind])
        solver.cost_set(stage, "zu", price[kind])
    return u_max, extend, retract


# ------------------------------------------------------------- the compiled solver


def _cache_root() -> Path:
    """
    Locate where solvers are compiled.

    Outside the source tree, as `crane_planning`'s is: a built tree and a
    shippable one are not the same tree.
    """
    return (
        Path(os.environ.get("CRANE_MPC_OCP_CACHE", tempfile.gettempdir()))
        / "crane_mpc_ocp"
    )


def solver_signature(parameters: dict, hydraulics: dict, description_xml: str) -> str:
    """
    Hash what materially changes generated code.

    `W`, every bound and every slack price are written through the runtime API,
    so they are deliberately **not** in here: retuning a weight must reuse the
    compiled solver, which is the whole tuning loop.
    """
    digest = hashlib.sha256()
    generated = {
        "Ts": parameters["Ts"],
        "horizon_length": parameters["horizon_length"],
        "levenberg_marquardt": parameters["levenberg_marquardt"],
        "qp_solver_cond_N": parameters.get("qp_solver_cond_N"),
    }
    digest.update(json.dumps(generated, sort_keys=True, default=str).encode())
    digest.update(json.dumps(hydraulics, sort_keys=True, default=str).encode())
    digest.update(description_xml.encode())
    digest.update(Path(problem.__file__).read_bytes())
    digest.update(Path(cs.__file__).read_bytes())
    digest.update(Path(cs.default_actuator_path()).read_bytes())
    return digest.hexdigest()[:16]


def _installed_acados(target) -> None:
    """
    Point acados_template at the installed headers and libraries.

    The devcontainer keeps the acados source under `/opt` and installs under
    `/usr/local`. Take `/usr/local` only when that install is complete:
    acados_template needs `lib/link_libs.json` to know what to link against, and
    the `/usr/local` copy does not always carry it. Without it, fall through to
    acados_template's own `ACADOS_SOURCE_DIR` default, which does.
    """
    prefix = Path("/usr/local")
    if (
        (prefix / "include" / "acados").is_dir()
        and (prefix / "lib" / "libacados.so").is_file()
        and (prefix / "lib" / "link_libs.json").is_file()
    ):
        target.acados_include_path = str(prefix / "include")
        target.acados_lib_path = str(prefix / "lib")


def solver_cache(parameters: dict, hydraulics: dict, description_xml: str) -> Path:
    """Where this exact problem's compiled solver lives."""
    signature = solver_signature(parameters, hydraulics, description_xml)
    return _cache_root() / f"{problem.TOOL}_{signature}"


def load_or_build(
    ocp, parameters: dict, hydraulics: dict, description_xml: str, verbose: bool = False
) -> tuple[AcadosOcpSolver, dict]:
    """
    Open the compiled solver for this problem, compiling it once if need be.

    Returns it with the dimensions acados wrote beside it: opened from a json
    the solver keeps them private, and the startup check needs them.
    """
    cache = solver_cache(parameters, hydraulics, description_xml)
    json_path = cache / "ocp.json"
    name = f"{problem.SOLVER_PREFIX}_{problem.TOOL}"
    library = cache / "code" / f"libacados_ocp_solver_{name}.so"

    def opened():
        # acados refuses ocp=None since 0.5; the formulation is read back from the
        # json the generate step wrote, and nothing is regenerated or rebuilt.
        solver = AcadosOcpSolver(
            AcadosOcp.from_json(str(json_path)),
            json_file=str(json_path),
            generate=False,
            build=False,
            verbose=verbose,
        )
        return solver, json.loads(json_path.read_text())["dims"]

    if json_path.is_file() and library.is_file():
        return opened()

    cache.mkdir(parents=True, exist_ok=True)
    if parameters.get("qp_solver_cond_N") is not None:
        ocp.solver_options.qp_solver_cond_N = int(parameters["qp_solver_cond_N"])
    ocp.code_export_directory = str(cache / "code")
    _installed_acados(ocp)
    AcadosOcpSolver.generate(ocp, json_file=str(json_path))
    # acados emits exact nonlinear-cost Hessians even under GAUSS_NEWTON. Nothing
    # registers them and they are by far the slowest files to compile; the
    # checked-in tree prunes them for the same reason.
    makefile = cache / "code" / "Makefile"
    makefile.write_text(
        "\n".join(
            line for line in makefile.read_text().splitlines() if "_hess.c" not in line
        )
        + "\n"
    )
    AcadosOcpSolver.build(str(cache / "code"), with_cython=False, verbose=verbose)
    if not library.is_file():
        raise RuntimeError(f"acados built no shared solver in {cache / 'code'}")
    return opened()


def _build_predictor(ocp, seconds: float, sample_time: float, verbose: bool):
    """
    Build an acados integrator over the same model, for the transport dead time.

    IRK for the reason the horizon is IRK: C3 is stiff at `T_s` (`|lambda| T_s`
    is 8.5 against an explicit stability limit near 2.8), and the delay is longer
    than one interval. Sub-stepped so no step exceeds `T_s`.
    """
    sim = AcadosSim()
    sim.model = ocp.model
    sim.parameter_values = ocp.parameter_values
    sim.solver_options.T = seconds
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_stages = 2
    sim.solver_options.num_steps = max(1, int(np.ceil(seconds / sample_time)))
    tree = _cache_root() / f"predictor_{problem.TOOL}_{seconds:g}"
    sim.code_export_directory = str(tree / "code")
    _installed_acados(sim)
    tree.mkdir(parents=True, exist_ok=True)
    return AcadosSimSolver(sim, json_file=str(tree / "sim.json"), verbose=verbose)


class Ocp:
    """The optimal control problem of `wiki/mpc.md` §1, solved once per cycle."""

    def __init__(
        self,
        description_xml: str,
        parameters: dict,
        hydraulics: dict,
        *,
        verbose: bool = False,
    ) -> None:
        # Before anything is built: generating and compiling a solver for a
        # configuration that is then refused costs a minute for nothing, and a
        # configuration that is *not* refused here is one no later stage checks.
        config.check_settings(parameters, hydraulics)
        self.parameters = parameters
        self.hydraulics = hydraulics
        self.Ts = float(parameters["Ts"])
        self.intervals = problem.shooting_intervals(parameters)
        self.solve_budget_s = float(parameters["solve_budget"])

        self._ocp, self.scale, self.model = problem.build_ocp(
            description_xml, parameters, hydraulics
        )
        self.solver, self._dims = load_or_build(
            self._ocp, parameters, hydraulics, description_xml, verbose=verbose
        )
        self._check_dimensions()

        # The residual, evaluated outside the solver for `wiki/mpc.md` §5.3
        # requirement 4: acados reports one cost and the split is what says
        # whether the expense is tracking, sway or effort.
        model = self._ocp.model
        self._residual = ca.Function(
            "y", [model.x, model.u, model.p], [model.cost_y_expr]
        )
        self._terminal_residual = ca.Function(
            "y_e", [model.x, model.p], [model.cost_y_expr_e]
        )
        self._static_force = ca.Function(
            "h_eff",
            [self.model.x, self.model.p],
            [self.model.actuated_force_static],
        )

        self._weight, self._terminal_weight = weight_matrices(parameters)
        self._u_max, self.cylinder_force_max, self.cylinder_force_retract = (
            self._configure_fixed()
        )
        self._slack_price = slack_prices(parameters)

        self._stepper = _build_predictor(self._ocp, self.Ts, self.Ts, verbose)
        delay = float(parameters["sensor_to_valve_delay"])
        self._predictor = (
            _build_predictor(self._ocp, delay, self.Ts, verbose)
            if delay > 0.0
            else None
        )
        self.delay_s = delay

        self._base_parameter = np.zeros(cs.NP)
        self.payload_mass_kg = 0.0
        self.payload_com_m = np.zeros(3)
        self._payload_changed = False

    # -- what the problem is, checked against what was compiled -----------------

    def _check_dimensions(self) -> None:
        """
        Check the eleven dimensions `ocp_solver.cpp:48-71` asserts at compile time.

        The compiled solver and this module derive the same numbers from the same
        module, but they derive them at different times: a solver compiled before
        the fit changed the lag-axis count is a solver for another problem.
        """
        expected = {
            "nx": cs.NX,
            "nbx": cs.NBX,
            "nbx_e": cs.NBX,
            "nbx_0": cs.NX,
            "nu": cs.NU_PROGRESS,
            "np": problem.NP,
            "ny": problem.NY,
            "ny_e": problem.NY_TERMINAL,
            "nh": cs.NU + 1,
            "nh_e": 0,
            "nsbx": 2 * cs.K_PASSIVE_DOF,
            "nsh": cs.NU + 1,
            "nbu": cs.NU_PROGRESS,
            "N": self.intervals,
        }
        for name, want in expected.items():
            got = self._dims.get(name)
            if got is None:
                continue
            if int(got) != int(want):
                raise RuntimeError(
                    f"the compiled solver has {name} = {int(got)} and this problem "
                    f"needs {int(want)}; the solver was generated from another "
                    "model, another fit or another grid"
                )

    def _configure_fixed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.solver.options_set(
            "levenberg_marquardt", float(self.parameters["levenberg_marquardt"])
        )
        return configure_fixed_data(
            self.solver, self.model, self.scale, self.parameters, self.hydraulics
        )

    # -- the model, asked the two questions the solver does not answer ----------

    def set_payload(self, mass_kg: float, com_m) -> bool:
        """Bind a payload into `p`. Returns whether it changed the model."""
        config.check_payload(mass_kg, com_m)
        com = np.asarray(com_m, dtype=float)
        changed = mass_kg != self.payload_mass_kg or not np.array_equal(
            com, self.payload_com_m
        )
        # Held here rather than trusted to the caller passing `guess=None`
        # (`ocp_solver.cpp:1396-1400`): `h_eff` moves discontinuously at a
        # grasp, so a plan that was warm was a plan for another model.
        self._payload_changed = self._payload_changed or changed
        self.payload_mass_kg = float(mass_kg)
        self.payload_com_m = com
        self._base_parameter[cs.P_PAYLOAD_MASS] = float(mass_kg)
        self._base_parameter[cs.P_PAYLOAD_COM : cs.P_PAYLOAD_COM + 3] = com
        # `crane_msgs/Payload` carries no inertia: a point mass, as the C++ node
        # also read it.
        return changed

    def pin_tool(self, position: float) -> None:
        """Bind the tool coordinate, which is not planned, into `p`."""
        self._base_parameter[cs.P_TOOL_POSITION] = float(position)

    def _model_parameter(self) -> np.ndarray:
        """
        Build the full stage parameter with an empty reference.

        What the integrator needs is the model's own eleven and nothing this
        problem appended.
        """
        parameter = np.zeros(problem.NP)
        parameter[: cs.NP] = self._base_parameter
        return parameter

    def static_hold_force(self, x: np.ndarray) -> np.ndarray:
        """
        `h_eff` at `x`: the force that holds the machine where it is.

        Seeds C3's force state, and is the effort term's own reference -- priced
        against zero the optimizer buys droop (issue 117).
        """
        value = self._static_force(x, self._base_parameter[: cs.NP])
        return np.asarray(value).reshape(-1)[: cs.K_PLANNED_DOF]

    def propagate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        `x` carried forward through the transport dead time under `u`.

        The whole state, actuator rows included: the plan is computed for the
        instant it takes effect, and under C3 the input reaches `ddq` only
        through the command-lag and force rows.
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
            raise ValueError(
                "the measured state and the applied input must be finite to be "
                "carried through the transport dead time"
            )
        if self._predictor is None:
            return x.copy()
        self._predictor.set("p", self._model_parameter())
        self._predictor.set("x", x)
        self._predictor.set("u", u)
        status = self._predictor.solve()
        if status != 0:
            raise RuntimeError(
                f"the dead-time predictor answered {status_word(status)}"
            )
        predicted = self._predictor.get("x")
        # acados reports a successful step whose state is NaN, and this is the
        # state `x0` is built from (`ocp_solver.cpp:1156-1162`).
        if not np.all(np.isfinite(predicted)):
            raise RuntimeError(
                "propagating the measured state forward under the applied command "
                "left it non-finite"
            )
        return predicted

    # -- one cycle ---------------------------------------------------------------

    def cold_start(self, x0: np.ndarray) -> Guess:
        """`x0` rolled forward under zero input: what a first cycle starts from."""
        states = np.zeros((self.intervals + 1, cs.NX))
        inputs = np.zeros((self.intervals, cs.NU_PROGRESS))
        states[0] = x0
        parameter = self._model_parameter()
        for stage in range(self.intervals):
            self._stepper.set("p", parameter)
            self._stepper.set("x", states[stage])
            self._stepper.set("u", inputs[stage])
            if self._stepper.solve() != 0:
                states[stage + 1 :] = x0
                break
            states[stage + 1] = self._stepper.get("x")
        return Guess(states, inputs)

    def shifted(self, solution: Solution) -> Guess:
        """
        Shift the accepted horizon one knot left, as the next warm start.

        `s` restarts at zero every cycle while the reference origin the caller
        tracks advances, so the progress row is re-origined by what was spent.
        """
        states = np.vstack([solution.states[1:], solution.states[-1:]])
        inputs = np.vstack([solution.inputs[1:], solution.inputs[-1:]])
        states[:, cs.X_PROGRESS] = np.maximum(
            states[:, cs.X_PROGRESS] - max(solution.progress_advance, 0.0), 0.0
        )
        return Guess(states, inputs)

    def solve(
        self,
        x0: np.ndarray,
        horizon,
        q_eq: np.ndarray,
        guess: Guess | None = None,
    ) -> Solution:
        """One RTI step. `horizon` is the resampled reference, `q_eq` the sway centre."""
        intervals = self.intervals
        x0 = np.asarray(x0, dtype=float).copy()
        if x0.shape != (cs.NX,) or not np.all(np.isfinite(x0)):
            raise ValueError("x0 is not a finite state of this problem")
        if len(horizon) != intervals + 1:
            raise ValueError(
                f"the horizon carries {len(horizon)} knots and the problem is posed "
                f"on {intervals + 1}"
            )
        # The curvature is the sharp one of the four: the resample's Hermite
        # second derivative carries `1/dt^2` in the *incoming* knot spacing,
        # which nothing bounds from below, and it reaches every stage of the
        # horizon through `p`, multiplied by `ds^2`, into a Gauss-Newton Hessian
        # (`ocp_solver.cpp:1250-1266`).
        if not all(
            np.all(np.isfinite(block))
            for block in (
                horizon.q_a_ref,
                horizon.dq_a_ref,
                horizon.ddq_a_ref,
                np.asarray(q_eq, dtype=float),
            )
        ):
            raise ValueError(
                "the reference, its curvature or the sway equilibrium carries a "
                "value that is not finite"
            )
        # `s` is virtual time *within one cycle*: it restarts here, and the
        # caller's own reference origin is what carries between cycles.
        x0[cs.X_PROGRESS] = 0.0

        # acados' own reset zeroes the iterate and the QP memory, so nothing of
        # the last cycle survives except what is written below.
        self.solver.reset(reset_qp_solver_mem=1)

        force_reference = self.static_hold_force(x0)
        measured = x0[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
        # Every stage's box is the same box but for one entry: `s`'s ceiling is
        # what `elapsed` seconds at the fastest rate allowed can have reached.
        # Built once and that one entry moved, because fifty of these cost more
        # than the solve can spare.
        rate_max = float(self.parameters["limits"]["progress_rate_max"])
        lower, upper = state_bounds(self.parameters, q_eq, measured, 0.0)
        running = stage_reference(q_eq, force_reference, terminal=False)
        terminal = stage_reference(q_eq, force_reference, terminal=True)
        for stage in range(intervals + 1):
            self.solver.set(
                stage,
                "p",
                stage_parameters(
                    self._base_parameter,
                    stage * self.Ts,
                    horizon.q_a_ref[stage, : cs.K_PLANNED_DOF],
                    horizon.dq_a_ref[stage, : cs.K_PLANNED_DOF],
                    horizon.ddq_a_ref[stage, : cs.K_PLANNED_DOF],
                ),
            )
            self.solver.cost_set(
                stage, "yref", terminal if stage == intervals else running
            )
            if stage == 0:
                # Constraint 1's initial condition: every row, actuator states
                # included, pinned at the state the plan is computed for.
                self.solver.constraints_set(0, "lbx", x0)
                self.solver.constraints_set(0, "ubx", x0)
                continue
            upper[cs.X_PROGRESS] = stage * self.Ts * rate_max
            self.solver.constraints_set(stage, "lbx", lower)
            self.solver.constraints_set(stage, "ubx", upper)

        # A payload step forces the cold start, whatever the caller passed.
        warm = not self._payload_changed and guess is not None and guess.warm(intervals)
        self._payload_changed = False
        seed = guess if warm else self.cold_start(x0)
        for stage in range(intervals + 1):
            if stage == 0:
                state = x0
            else:
                state = seed.states[stage].copy()
                upper[cs.X_PROGRESS] = stage * self.Ts * rate_max
                state[: cs.NBX] = np.clip(state[: cs.NBX], lower, upper)
            self.solver.set(stage, "x", state)
            if stage < intervals:
                self.solver.set(
                    stage,
                    "u",
                    np.clip(seed.inputs[stage], -self._u_max, self._u_max),
                )

        status = int(self.solver.solve())
        solve_time = float(self.solver.get_stats("time_tot"))
        qp_status = _last(self.solver.get_stats("qp_stat"))
        qp_iterations = _last(self.solver.get_stats("qp_iter"))
        iterations = int(self.solver.get_stats("sqp_iter"))

        states = np.array(
            [self.solver.get(stage, "x") for stage in range(intervals + 1)]
        )
        inputs = np.array([self.solver.get(stage, "u") for stage in range(intervals)])

        violation, penalty, used_slack = self._slack_taken()

        finite = np.all(np.isfinite(states)) and np.all(np.isfinite(inputs))
        word = status_word(status)
        if qp_status != 0:
            word = f"{word} (QP status {qp_status})"
        if not finite or status != 0 or qp_status != 0:
            outcome = Outcome.FAILED
        elif solve_time > self.solve_budget_s:
            outcome = Outcome.BUDGET_EXCEEDED
        else:
            outcome = Outcome.CONVERGED

        advance = float(states[1][cs.X_PROGRESS]) if len(states) > 1 else 0.0
        advance = float(np.clip(advance, 0.0, self.Ts * rate_max))

        return Solution(
            states=states,
            inputs=inputs,
            outcome=outcome,
            status=status,
            status_word=word,
            qp_status=qp_status,
            iterations=iterations,
            qp_iterations=qp_iterations,
            solve_time_s=solve_time,
            used_slack=used_slack,
            slack_penalty=penalty,
            violation=violation,
            progress_advance=advance,
            warm_started=warm,
        )

    def _slack_taken(self) -> tuple[Violation, float, bool]:
        """
        Read what the softened constraints spent, off `sl`/`su` per stage.

        Row order is the problem's own: the sway pair and the sway-rate pair
        first (`idxsbx`), then the five cylinder-force rows and the pump row.
        Sway is reported as a fraction of its own allowance; the hydraulic rows
        are already conditioned by `constraint_scale`.
        """
        limits = self.parameters["limits"]
        q_u_max = np.asarray(limits["q_u_max"], dtype=float)
        dq_u_max = np.asarray(limits["dq_u_max"], dtype=float)
        violation = Violation()
        penalty = 0.0
        used = False
        for stage in range(self.intervals + 1):
            kind = (
                "initial"
                if stage == 0
                else ("terminal" if stage == self.intervals else "path")
            )
            price = self._slack_price[kind]
            if price.size == 0:
                continue
            lower = np.maximum(np.asarray(self.solver.get(stage, "sl")), 0.0)
            upper = np.maximum(np.asarray(self.solver.get(stage, "su")), 0.0)
            taken = np.maximum(lower, upper)
            penalty += float(price @ lower + price @ upper)
            entry = 0
            if stage > 0:
                sway = taken[entry : entry + cs.K_PASSIVE_DOF]
                entry += cs.K_PASSIVE_DOF
                rate = taken[entry : entry + cs.K_PASSIVE_DOF]
                entry += cs.K_PASSIVE_DOF
                violation.q_u = max(violation.q_u, float(np.max(sway / q_u_max)))
                violation.dq_u = max(violation.dq_u, float(np.max(rate / dq_u_max)))
            if stage < self.intervals:
                force = taken[entry : entry + cs.K_PLANNED_DOF]
                entry += cs.K_PLANNED_DOF
                violation.cylinder_force = max(
                    violation.cylinder_force, float(np.max(force))
                )
                violation.pump_flow = max(violation.pump_flow, float(taken[entry]))
            if float(np.max(taken)) > SLACK_NOTICEABLE:
                used = True
        return violation, penalty, used

    def cost_terms(self, solution: Solution, horizon, q_eq: np.ndarray) -> CostTerms:
        """
        `wiki/mpc.md` §2 term by term: what the plan spent, and on what.

        Re-evaluated from the residual rather than read off the QP -- acados
        reports one number and requirement 4 asks for the split.
        """
        terms = CostTerms(slack=solution.slack_penalty)
        force_reference = self.static_hold_force(solution.states[0])
        diagonal = np.diag(self._weight)
        terminal_diagonal = np.diag(self._terminal_weight)
        planned = cs.K_PLANNED_DOF
        passive = cs.K_PASSIVE_DOF
        for stage in range(self.intervals + 1):
            parameter = stage_parameters(
                self._base_parameter,
                stage * self.Ts,
                horizon.q_a_ref[stage, :planned],
                horizon.dq_a_ref[stage, :planned],
                horizon.ddq_a_ref[stage, :planned],
            )
            terminal = stage == self.intervals
            reference = stage_reference(q_eq, force_reference, terminal=terminal)
            if terminal:
                value = np.asarray(
                    self._terminal_residual(solution.states[stage], parameter)
                ).reshape(-1)
                terms.terminal += float(
                    0.5 * terminal_diagonal @ (value - reference) ** 2
                )
                continue
            value = np.asarray(
                self._residual(
                    solution.states[stage], solution.inputs[stage], parameter
                )
            ).reshape(-1)
            row = 0.5 * diagonal * (value - reference) ** 2
            terms.q_a += float(np.sum(row[problem.Y_PLANNED_POSITION :][:planned]))
            terms.dq_a += float(np.sum(row[problem.Y_PLANNED_VELOCITY :][:planned]))
            terms.q_u += float(np.sum(row[problem.Y_PASSIVE_POSITION :][:passive]))
            terms.dq_u += float(np.sum(row[problem.Y_PASSIVE_VELOCITY :][:passive]))
            terms.lag += float(row[problem.Y_LAG])
            terms.progress += float(row[problem.Y_PROGRESS_RATE])
            terms.tau_a += float(np.sum(row[problem.Y_ACTUATED_FORCE :][:planned]))
            terms.u += float(np.sum(row[problem.Y_INPUT :][: cs.NU_PROGRESS]))
        return terms


def _last(value) -> int:
    """Take the last entry: acados reports these per RTI call."""
    array = np.atleast_1d(np.asarray(value)).reshape(-1)
    return int(array[-1]) if array.size else 0
