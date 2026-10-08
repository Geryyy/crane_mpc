"""
The generated acados solver, opened and driven.

The binding around acados' unchanged artifact: what is written before a solve,
read back after, plus the dead-time predictor and static hold force. State is the
OCP's own rows end to end -- a truncation that drops the actuator rows coasts the
predictor at a frozen force (issue 125).

Every state crossing this binding is **physical** (N m, rad, rad/s). The
compiled problem is posed in a scaled one (`problem.state_scale`), and
`Ocp._to_qp`/`_to_physical` are the only conversion: nothing above this module
or beside it -- bounds, tick dumps, `replay_tick.py`, the predictor's callers --
sees a scaled row.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import casadi as ca
import crane_ocp_export as ox
import numpy as np
from acados_template import AcadosOcpSolver, AcadosSim
from crane_model import symbolic as cs

from . import bspline, config, problem

#: Slack below this is round-off, not a violation.
SLACK_NOTICEABLE = 1e-6

#: acados' own return codes, by name.
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

    def warm(self, intervals: int, nx: int = cs.NX) -> bool:
        return (
            self.states.shape == (intervals + 1, nx)
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


#: The optimizer's own bookkeeping rows: no sensor reads them, no valve sees
#: them, and no transport delay applies to them.
PROGRESS_ROWS = [cs.X_PROGRESS, cs.X_PROGRESS_RATE]


@dataclass
class PathCycle:
    """
    The path a cycle is written against, and where on it the cycle starts.

    `s` is pinned to zero at stage zero, so `origin` carries what earlier cycles
    already spent and `s` carries only the rest -- `theta = origin + s`.

    `nominal_rate` is the plan's own pace in path parameter per second, which
    depends on what the path spans and so cannot be derived here: the node fits
    one horizon window per cycle (`Ocp.window_path`), an offline harness fits a
    whole move once. Everything else about the two is the same, which is the
    point of naming it.
    """

    control: np.ndarray
    origin: float
    nominal_rate: float

    def remaining(self) -> float:
        """How much path parameter is left, which is `s`'s own ceiling."""
        return max(0.0, problem.K_PROGRESS_RATE_REFERENCE - float(self.origin))


@dataclass
class CostTerms:
    """The NLS cost, term by term, summed over the horizon."""

    q_a: float = 0.0
    dq_a: float = 0.0
    q_u: float = 0.0
    dq_u: float = 0.0
    lag: float = 0.0
    #: The three TCP-position rows -- the cm-scale metric, in metres, not joints.
    tool: float = 0.0
    progress: float = 0.0
    tau_a: float = 0.0
    #: The command (C3: the input rows; C4: the `c` rows) plus progress accel.
    u: float = 0.0
    #: C4 only: the command's rate rows.
    du: float = 0.0
    terminal: float = 0.0
    slack: float = 0.0


# ------------------------------------------------------- what the solver is told


def weight_matrices(parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage and terminal nonlinear-least-squares weights.

    C4: the input rows are `dc`, priced by `du` every stage; `u` prices `c`.
    """
    weights = parameters["weights"]
    planned = cs.K_PLANNED_DOF
    passive = cs.K_PASSIVE_DOF
    command = np.asarray(weights["u"][:planned], dtype=float)
    rate = np.asarray(weights["du"][:planned], dtype=float)
    c4 = problem.command_state(parameters)
    diagonal = np.concatenate(
        [
            np.asarray(weights["q_a"][:planned], dtype=float),
            np.asarray(weights["dq_a"][:planned], dtype=float),
            np.asarray(weights["q_u"][:passive], dtype=float),
            np.asarray(weights["dq_u"][:passive], dtype=float),
            [
                float(weights["lag"]),
            ],
            np.full(3, float(weights["tool"])),
            [
                float(weights["progress"]),
                float(weights["progress_rate"]),
            ],
            np.asarray(weights["tau_a"][:planned], dtype=float),
            rate if c4 else command,
            [float(weights["progress_accel"])],
            command if c4 else [],
        ]
    )
    terminal = float(weights["terminal_scale"]) * diagonal[: problem.NY_TERMINAL]
    return np.diag(diagonal), np.diag(terminal)


def move_suppression(
    weight: np.ndarray, reference: np.ndarray, du: np.ndarray, in_flight
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fold `w_du*(u0 - u_f)^2` into stage 0's `w_u*u0^2`, per planned input row.

    Same minimiser and curvature: `W = w_u + w_du`, `yref = w_du*u_f/W`; only a
    constant is dropped. Progress-accel row untouched.
    """
    weight, reference = weight.copy(), reference.copy()
    rows = np.arange(problem.Y_INPUT, problem.Y_INPUT + du.size)
    total = weight[rows, rows] + du
    weight[rows, rows] = total
    reference[rows] = np.divide(
        du * np.asarray(in_flight, dtype=float)[: du.size],
        total,
        out=np.zeros_like(total),
        where=total > 0.0,
    )
    return weight, reference


def constraint_data(
    chamber: tuple, scale: np.ndarray, hydraulics: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaled bounds of `h` (constraints 6 and 7) and the physical force limits.

    `chamber` is `build_ocp`'s own `(extend, retract)`, so these bounds are
    conditioned by the numbers the divisors were built from. Half of what comes
    back is `+-1` by construction -- `scale` is the larger of each pair, and the
    pump divisor *is* the planning limit -- so this only names the other half.
    """
    extend, retract = chamber
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
    """Position box with `q_a_margin`, never excluding `measured` (else stage 1 chases nothing)."""
    limits = parameters["limits"]
    margin = np.asarray(limits["q_a_margin"][: cs.K_PLANNED_DOF], dtype=float)
    lower = np.asarray(limits["q_a_lower"][: cs.K_PLANNED_DOF], dtype=float) + margin
    upper = np.asarray(limits["q_a_upper"][: cs.K_PLANNED_DOF], dtype=float) - margin
    return np.minimum(lower, measured), np.maximum(upper, measured)


def state_bounds(
    parameters: dict,
    equilibrium: np.ndarray,
    measured: np.ndarray,
    rate_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One stage's box: rigid rows, lagged command, progress pair.

    Force rows are uncapped (constraint 6 holds them); `s`'s ceiling is zero, moved per stage.

    `rate_max` is passed rather than read off `parameters`: the yaml declares
    catch-up headroom, and turning that into a rate needs to know what the
    path spans, which this function cannot see.
    """
    limits = parameters["limits"]
    u_max = np.asarray(limits["u_max"][: cs.K_PLANNED_DOF], dtype=float)
    q_lower, q_upper = position_box(parameters, measured)
    lag = u_max[list(cs.K_LAG_AXES)]
    # C4's `c` rows, hard in Psi's domain.
    command = u_max if problem.command_state(parameters) else []
    lower = np.concatenate(
        [
            q_lower,
            equilibrium - np.asarray(limits["q_u_max"], dtype=float),
            -np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float),
            -np.asarray(limits["dq_u_max"], dtype=float),
            -lag,
            [0.0, 0.0],
            -np.asarray(command),
        ]
    )
    upper = np.concatenate(
        [
            q_upper,
            equilibrium + np.asarray(limits["q_u_max"], dtype=float),
            np.asarray(limits["dq_a_max"][: cs.K_PLANNED_DOF], dtype=float),
            np.asarray(limits["dq_u_max"], dtype=float),
            lag,
            [0.0, rate_max],
            command,
        ]
    )
    return lower, upper


def stage_equilibrium(q_eq: np.ndarray, stage: int) -> np.ndarray:
    """
    Return this stage's sway equilibrium: one held vector, or one row per stage.

    The node reads one equilibrium at the measured state and holds it; a harness
    that knows the whole path can solve one per stage. Taking either here is what
    lets the two share a cycle.
    """
    table = np.asarray(q_eq, dtype=float)
    return table if table.ndim == 1 else table[stage]


def stage_reference(
    q_eq: np.ndarray,
    tau_ref: np.ndarray,
    terminal: bool,
    progress: float,
    rows: int = problem.NY,
) -> np.ndarray:
    """
    Build a stage/terminal residual reference; tracking rows are zero (rides in `p`).

    `tau_ref` is `h_eff`; zeroed, it prices holding own weight and buys droop (issue 117).

    `progress` is where on the path this stage is asked to be -- no default, so
    that a caller has to say. robocrane asks for the end of the path at every
    stage, which works where a horizon covers most of one; here it covers a
    third, so asking for the end is asking for flat out everywhere -- measured,
    460 mm off path against 448, and 31 of 178 solves refused. Asking for as far
    as the plan's own pace reaches by this stage is the same thing wherever the
    end is in sight, and a pace everywhere else. `Ocp.nominal_progress` is
    that pace. `rows` is the stage residual's width (C4 appends `c`, referenced
    to zero like C3's `u`).
    """
    reference = np.zeros(problem.NY_TERMINAL if terminal else rows)
    reference[
        problem.Y_PASSIVE_POSITION : problem.Y_PASSIVE_POSITION + cs.K_PASSIVE_DOF
    ] = q_eq
    reference[problem.Y_PROGRESS] = min(
        float(problem.K_PROGRESS_RATE_REFERENCE), float(progress)
    )
    if not terminal:
        reference[
            problem.Y_ACTUATED_FORCE : problem.Y_ACTUATED_FORCE + cs.K_PLANNED_DOF
        ] = tau_ref
    return reference


def stage_parameters(
    base: np.ndarray, origin: float, control: np.ndarray
) -> np.ndarray:
    """
    Pack the path and where this cycle starts on it into the parameter vector.

    One vector for the whole horizon, not one per stage: which stage this is, is
    carried by `s`, and the path does not know about stages.
    """
    parameter = np.zeros(problem.NP)
    parameter[: cs.NP] = base
    parameter[problem.P_PATH_ORIGIN] = float(origin)
    # Point-major, which is what `ca.reshape` unpacks column by column.
    parameter[problem.P_PATH_CONTROL :] = np.asarray(control, dtype=float).reshape(-1)
    return parameter


def path_control(samples) -> np.ndarray:
    """Fit the planned curve itself, in its own parameter. Once per plan."""
    return bspline.fit(np.asarray(samples, dtype=float), problem.PATH_POINTS)


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
    chamber: tuple,
    scale: np.ndarray,
    parameters: dict,
    hydraulics: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Write what is runtime-settable and fixed for a run.

    Every weight, bound of constraints 5/6/7, slack price; `yref`/box/`p` move per cycle.
    """
    intervals = problem.shooting_intervals(parameters)
    weight, terminal_weight = weight_matrices(parameters)
    lower_h, upper_h, extend, retract = constraint_data(chamber, scale, hydraulics)
    # Input box: five joint commands (C4: their rates) plus progress acceleration
    # (not an axis, own bound).
    joint = "du_max" if problem.command_state(parameters) else "u_max"
    u_max = np.concatenate(
        [
            np.asarray(parameters["limits"][joint][: cs.K_PLANNED_DOF], dtype=float),
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


#: Env var naming where the export is written and read.
EXPORT_ENV = "CRANE_MPC_OCP_EXPORT"

#: What a caller is told to run when the export is missing or is another problem.
#: The script path, not `ros2 run`: `setup.py` declares one console script and it
#: is the node. An exporter needs the source checkout anyway -- it reads
#: `config/`, the description and the C3 fit out of it.
EXPORT_COMMAND = "crane_mpc/scripts/export_ocp.py"


def export_base() -> Path:
    """Where `scripts/export_ocp.py` writes and this opens, one dir per problem."""
    return ox.export_base(EXPORT_ENV, f"crane_mpc_{problem.TOOL}")


def _file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def export_key(parameters: dict, hydraulics: dict, description_xml: str) -> dict:
    """
    Everything baked into generated code, field by field.

    `W`/bounds/slack are excluded on purpose -- they go in through the runtime
    API every cycle, so retuning must reuse the exported solver, and it does.
    The acados settings are **not** excluded: they are compiled in, so a variant
    that left them out would open its predecessor's `.so` and read as a null
    result.

    A dict rather than one hash so `ox.divergence` can say *which* of these
    moved. Re-exporting costs minutes; "the horizon moved" is actionable and
    "the hash differs" is not.
    """
    return {
        # `Ts` alone names the integrators: the dead time is one `Ts`
        # (`config.check_settings`), so the delay cannot move the export.
        "Ts": parameters["Ts"],
        "command_state": problem.command_state(parameters),
        "horizon_length": parameters["horizon_length"],
        "levenberg_marquardt": parameters["levenberg_marquardt"],
        "qp_solver_cond_N": parameters.get("qp_solver_cond_N"),
        "tuning": problem.solver_tuning(),
        "hydraulics": hydraulics,
        "description": hashlib.sha256(description_xml.encode()).hexdigest()[:16],
        "problem.py": _file_digest(problem.__file__),
        "symbolic.py": _file_digest(cs.__file__),
        "actuator_fit": _file_digest(cs.default_actuator_path()),
        # The whole file, not the three numbers `hydraulics` carries: the
        # chamber areas, `r_gear` and `V_m` live in it too, and since issue 226
        # they set `problem.state_scale` -- the **units of the force state
        # across this binding**. An area edited without a re-export would leave
        # the compiled code reading `S_old x_qp` while `Ocp._x_scale` writes
        # `tau/S_new`, and nothing else in the package reads `tau_a` to notice.
        "hydraulics.yaml": _file_digest(cs.default_hydraulics_path()),
    }


def _sub_steps(seconds: float, sample_time: float) -> int:
    """How many integrator steps the interval takes with none longer than `T_s`."""
    return max(1, int(np.ceil(seconds / sample_time)))


def predictor_label(seconds: float, sample_time: float) -> str:
    """One integrator's name in the export: its interval and its sub-step count."""
    return f"{seconds:g}_{_sub_steps(seconds, sample_time)}"


def predictor_sim(ocp, seconds: float, sample_time: float) -> AcadosSim:
    """
    One acados integrator over the same model as the horizon.

    IRK like the horizon: C3 is stiff at `T_s` (`|lambda| T_s` = 8.5 vs an
    explicit stability limit near 2.8); sub-stepped so no step exceeds `T_s`.

    Built here and read by both sides -- `scripts/export_ocp.py` compiles exactly
    the set `predictor_labels` names, and `Ocp` opens exactly that set, so the
    two cannot disagree about which integrators exist.

    `newton_iter` is acados' default 3 until the passive rows carry Coulomb
    friction (issue 222): the bias turns over inside the tanh's smoothing width,
    three iterations stop short, and two replays of one ZOH command then part by
    2 N m of actuated force -- `test_c4_replays_a_command_ramp_from_its_start`.
    It is a runtime option, applied when the solver is opened, so it is not in
    `export_key` and moving it costs no re-export.
    """
    sim = AcadosSim()
    sim.model = ocp.model
    sim.parameter_values = ocp.parameter_values
    sim.solver_options.T = seconds
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_stages = 2
    sim.solver_options.num_steps = _sub_steps(seconds, sample_time)
    sim.solver_options.newton_iter = 20
    return sim


def predictor_sims(ocp, parameters: dict) -> list:
    """Every `(label, AcadosSim)` an export must carry: one `Ts` step, and only one."""
    sample_time = float(parameters["Ts"])
    return [
        (
            predictor_label(sample_time, sample_time),
            predictor_sim(ocp, sample_time, sample_time),
        )
    ]


class Ocp:
    """The optimal control problem, solved once per cycle."""

    def __init__(
        self,
        description_xml: str,
        parameters: dict,
        hydraulics: dict,
        *,
        verbose: bool = False,
    ) -> None:
        # Refuse before building: a rejected config still costs a minute to compile.
        config.check_settings(parameters, hydraulics)
        self.parameters = parameters
        self.hydraulics = hydraulics
        self.Ts = float(parameters["Ts"])
        self.intervals = problem.shooting_intervals(parameters)
        self.solve_budget_s = float(parameters["solve_budget"])

        self._ocp, self.scale, self.model, self._chamber = problem.build_ocp(
            description_xml, parameters, hydraulics
        )
        #: C4: `c` rides at the end of `x` and the joint inputs are its rate.
        self.command_state = problem.command_state(parameters)
        self.nx = int(self.model.x.shape[0])
        #: `x = S x_s`: the compiled problem is posed in a scaled state
        #: (`problem.state_scale`). Everything in this class and everything
        #: outside it -- bounds, dumps, the predictor's callers -- is physical;
        #: `_to_qp`/`_to_physical` convert at the four places a state is written
        #: to or read off acados.
        self._x_scale = problem.state_scale(parameters, self.scale)
        self._boxed = problem.boxed_state_rows(parameters)
        self._ny = problem.residual_rows(parameters)
        self._command_max = np.asarray(
            parameters["limits"]["u_max"][: cs.K_PLANNED_DOF], dtype=float
        )
        # Opened, never built: `scripts/export_ocp.py` compiled this and wrote
        # the key beside it, and `ox.open_solver` refuses anything that is not
        # this problem rather than starting on another machine's dynamics.
        base = export_base()
        key = export_key(parameters, hydraulics, description_xml)
        self.solver, self._dims = ox.open_solver(
            self._ocp, base, key, EXPORT_COMMAND, verbose=verbose
        )
        self._check_dimensions()

        # Residual outside the solver: acados reports one cost, this splits
        # tracking/sway/effort. Posed on the *scaled* state, like the generated
        # code -- `cost_terms` divides before it calls either.
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
        self._du = np.asarray(
            parameters["weights"]["du"][: cs.K_PLANNED_DOF], dtype=float
        )
        #: Whether stage 0 carries a move cost, i.e. depends on the command in flight.
        #: Never under C4, where `du` prices the rate on every stage instead.
        self.suppresses_moves = bool(self._du.any()) and not self.command_state
        #: The in-flight command stage 0 was last written against, for `cost_terms`.
        self._in_flight: np.ndarray | None = None
        self._u_max, self.cylinder_force_max, self.cylinder_force_retract = (
            self._configure_fixed()
        )
        # Priced per stage, held flat: `_slack_taken` reads `sl`/`su` for the
        # whole horizon in one call and never slices stage by stage.
        price = slack_prices(parameters)
        self._slack_price = np.concatenate(
            [price["initial"]]
            + [price["path"]] * (self.intervals - 1)
            + [price["terminal"]]
        )
        #: Widths of the two soft blocks: sway pair + sway-rate pair, and the
        #: cylinder-force rows + the pump row. Stage 0 carries the second alone,
        #: the terminal stage the first alone, every stage between both.
        self._soft_state = 2 * cs.K_PASSIVE_DOF
        self._soft_nonlinear = cs.NU + 1

        def predictor(seconds: float):
            return ox.open_sim(
                predictor_sim(self._ocp, seconds, self.Ts),
                base,
                key,
                predictor_label(seconds, self.Ts),
                verbose=verbose,
            )

        #: Cold start, one horizon stage, and the dead time -- all one `Ts`.
        self._stepper = predictor(self.Ts)

        self._base_parameter = np.zeros(cs.NP)
        self.payload_mass_kg = 0.0
        self.payload_com_m = np.zeros(3)
        self._payload_changed = False
        #: The parameter vector the last written problem carried, path included,
        #: and the `PathCycle` it was built from -- what `_read_solution` and
        #: `cost_terms` have to read the solution against.
        self._last_parameter: np.ndarray | None = None
        self._last_path: PathCycle | None = None
        #: What the last written problem was built from, for `last_problem`.
        self._written: dict = {}

    # -- what the problem is, checked against what was compiled -----------------

    def _check_dimensions(self) -> None:
        """Check every dimension: a stale solver is one for another problem."""
        expected = {
            "nx": self.nx,
            "nbx": self._boxed.size,
            "nbx_e": self._boxed.size,
            "nbx_0": self.nx,
            "nu": cs.NU_PROGRESS,
            "np": problem.NP,
            "ny": problem.residual_rows(self.parameters),
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
            self.solver, self._chamber, self.scale, self.parameters, self.hydraulics
        )

    # -- physical state <-> the QP's own coordinates -----------------------------

    def _to_qp(self, x: np.ndarray) -> np.ndarray:
        """Convert a physical state, or a stack of them, to what the solver holds."""
        return np.asarray(x, dtype=float) / self._x_scale

    def _to_physical(self, x: np.ndarray) -> np.ndarray:
        """Convert what the solver holds back to N m and rad."""
        return np.asarray(x, dtype=float) * self._x_scale

    # -- the model, asked the two questions the solver does not answer ----------

    def set_payload(self, mass_kg: float, com_m, *, cold: bool = True) -> bool:
        """
        Bind a payload into `p`. Returns whether it changed the model.

        `cold=False` takes it without forcing the cold start, which only a
        replay wants: opening a fresh solver starts at zero payload, so every
        dump off a loaded move would read as a payload step and throw away the
        recorded seed -- and the seed is the suspect.
        """
        config.check_payload(mass_kg, com_m)
        com = np.asarray(com_m, dtype=float)
        changed = mass_kg != self.payload_mass_kg or not np.array_equal(
            com, self.payload_com_m
        )
        # Tracked here, not via caller's `guess=None`: `h_eff` jumps at a grasp.
        self._payload_changed = self._payload_changed or (changed and cold)
        self.payload_mass_kg = float(mass_kg)
        self.payload_com_m = com
        self._base_parameter[cs.P_PAYLOAD_MASS] = float(mass_kg)
        self._base_parameter[cs.P_PAYLOAD_COM : cs.P_PAYLOAD_COM + 3] = com
        # `crane_msgs/Payload` carries no inertia: a point mass.
        return changed

    def pin_tool(self, position: float) -> None:
        """Bind the tool coordinate, which is not planned, into `p`."""
        self._base_parameter[cs.P_TOOL_POSITION] = float(position)

    def _model_parameter(self) -> np.ndarray:
        """Full stage parameter, empty reference: the model's own eleven, nothing appended."""
        parameter = np.zeros(problem.NP)
        parameter[: cs.NP] = self._base_parameter
        return parameter

    def static_hold_force(self, x: np.ndarray) -> np.ndarray:
        """
        `h_eff` at `x`: the force that holds the machine where it is.

        Seeds C3's force state and effort term's reference; zeroed there, buys droop (issue 117).
        """
        value = self._static_force(x, self._base_parameter[: cs.NP])
        return np.asarray(value).reshape(-1)[: cs.K_PLANNED_DOF]

    def _integrate(self, predictor, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        predictor.set("p", self._model_parameter())
        predictor.set("x", self._to_qp(x))
        predictor.set("u", u)
        status = predictor.solve()
        if status != 0:
            raise RuntimeError(
                f"the dead-time predictor answered {status_word(status)}"
            )
        predicted = self._to_physical(predictor.get("x"))
        # acados can report success with a NaN state.
        if not np.all(np.isfinite(predicted)):
            raise RuntimeError(
                "propagating the measured state forward under the applied command "
                "left it non-finite"
            )
        return predicted

    def axis_transition(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        One `Ts` of `(dq_a, tau_a)` per planned axis, as a `(5, 2, 2)` matrix.

        What `force_filter` derives its gain from. acados' own forward
        sensitivity of exactly the step `propagate` takes, so the filter is
        linearised about the integrator the horizon runs rather than about a
        second model written by hand -- and the force row's only path into
        something a sensor reads, `d dq / d tau`, comes out of it rather than
        out of an inertia this package would have to restate.

        Diagonal sub-blocks: the filter is one two-row observer per axis, so the
        cross terms are deliberately dropped, and `q_a` is not a row of it at
        all (the filter takes the encoder as exact -- `force_filter` says why).
        Converted out of the scaled coordinates like every other state crossing
        this binding; the pair is a sub-block, so that is one outer product per
        axis.
        """
        x, u = _finite(x, u)
        self._stepper.set("p", self._model_parameter())
        self._stepper.set("x", self._to_qp(x))
        self._stepper.set("u", u)
        status = self._stepper.solve()
        if status != 0:
            raise RuntimeError(
                f"linearising the force filter's step answered {status_word(status)}"
            )
        # `(nx, nx + nu)`; the input columns belong to no state row.
        sensitivity = np.asarray(self._stepper.get("S_forw"), dtype=float)[:, : self.nx]
        rows = np.array([cs.X_PLANNED_VELOCITY, cs.X_ACTUATED_FORCE])
        blocks = np.zeros((cs.K_PLANNED_DOF, rows.size, rows.size))
        for axis in range(cs.K_PLANNED_DOF):
            index = rows + axis
            factor = self._x_scale[index]
            blocks[axis] = (
                sensitivity[np.ix_(index, index)] * factor[:, None] / factor[None, :]
            )
        if not np.all(np.isfinite(blocks)):
            raise RuntimeError(
                "the force filter's own linearisation came back non-finite, so no "
                "gain can be derived from it"
            )
        return blocks

    def propagate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        `x` carried forward through the transport dead time -- one `Ts` -- under `u`.

        Whole state, actuator rows included: under C3, `u` reaches `ddq` only via
        lag/force rows, so dropping them would return one state for every command
        (issue 125).

        Except the progress pair. Nothing about `s`/`v_s` travels to a valve --
        they are the optimizer's own bookkeeping, and `Cycle` carries them from
        the last solution's stage 1, which already stands at the instant the next
        plan takes effect. Rolling them over the delay as well advanced them
        twice per cycle, which walked `v_s` out of its own box and made
        `_origined_x0`'s clamp load-bearing rather than a backstop.
        """
        x, u = _finite(x, u)
        moved = self._integrate(self._stepper, x, u)
        moved[PROGRESS_ROWS] = x[PROGRESS_ROWS]
        return moved

    def propagate_applied(self, x: np.ndarray, applied) -> np.ndarray:
        """
        Carry `x` under the command in flight, newest first.

        The dead time is one `Ts`, so one command covers it and only `applied[0]`
        is read -- the caller keeps the rest for its next roll. Empty is refused.

        An entry is an input (ZOH), or C4's `(start, input)`: `start` is `c` at the
        entry's first instant, written into `x` before the roll; `None` keeps `x`'s.
        """
        if len(applied) == 0:
            raise ValueError(
                "replaying the commands in flight needs at least the one applied "
                "now; an empty history is not the same thing as a zero command "
                "and is not guessed at here"
            )
        entry = applied[0]
        start, command = entry if isinstance(entry, tuple) else (None, entry)
        state, command = _finite(x, command)
        if start is not None:
            state = state.copy()
            state[cs.X_COMMAND : cs.X_COMMAND + cs.K_PLANNED_DOF] = _finite(start)[0]
        return self.propagate(state, command)

    # -- one cycle ---------------------------------------------------------------

    def cold_start(self, x0: np.ndarray) -> Guess:
        """`x0` rolled forward under zero input: what a first cycle starts from."""
        states = np.zeros((self.intervals + 1, self.nx))
        inputs = np.zeros((self.intervals, cs.NU_PROGRESS))
        states[0] = x0
        parameter = self._model_parameter()
        for stage in range(self.intervals):
            self._stepper.set("p", parameter)
            self._stepper.set("x", self._to_qp(states[stage]))
            self._stepper.set("u", inputs[stage])
            if self._stepper.solve() != 0:
                states[stage + 1 :] = x0
                break
            states[stage + 1] = self._to_physical(self._stepper.get("x"))
        return Guess(states, inputs)

    def carried(self, solution: Solution) -> Guess:
        """
        Return the next cycle's warm start: the last solution, **not** shifted.

        The textbook RTI shift is the wrong move at one Newton step per cycle: it
        perturbs the iterate and the single iteration spends itself repairing that
        instead of improving the plan, so the correction is deferred one knot
        further every cycle while only knot 0 is executed. Measured, the shifted
        loop walks away from the goal (0.139 -> 0.209 rad over 29 s) where this
        decays to 1.4e-4 rad.
        """
        return Guess(states=solution.states.copy(), inputs=solution.inputs.copy())

    def _origined_x0(self, x0: np.ndarray, path: PathCycle) -> np.ndarray:
        """
        `x0` as this problem's own state, with the progress pair put in its box.

        `s` is virtual time within one cycle, so it is pinned to zero and the
        path's origin carries what came before.

        `v_s` is projected into `[0, rate_max]`. Neither row is measured -- they
        are the optimizer's own bookkeeping, carried from the last solution --
        so a value outside the box is stale arithmetic, not an observation, and
        clamping it is not overruling the machine. Left alone it is an
        **infeasible** problem rather than a tight one: stage 0 is pinned with no
        slack, the running stages box `v_s`, and `progress_accel_max` cannot
        brake into the box within one interval. That is the loop that took
        44.4% of cycles on the five-move benchmark -- a refused solve applies
        the previous command, which carries `v_s` further out, which refuses the
        next one.
        """
        x0 = np.asarray(x0, dtype=float).copy()
        if x0.shape != (self.nx,) or not np.all(np.isfinite(x0)):
            raise ValueError("x0 is not a finite state of this problem")
        if self.command_state:
            # `c`'s box is hard from stage 1 and `du` finite: start inside it.
            command = slice(cs.X_COMMAND, cs.X_COMMAND + cs.K_PLANNED_DOF)
            x0[command] = np.clip(x0[command], -self._command_max, self._command_max)
        x0[cs.X_PROGRESS] = 0.0
        # Nor faster than stops inside what is left: `s` never falls and is capped
        # at `remaining()` on every stage, so a rate carried onto the last sliver
        # is QP status 3 to the goal. Stage 1 bounds it by `r/Ts`, braking at
        # `progress_accel_max` by `sqrt(2 a r)`.
        left = path.remaining()
        accel = float(self.parameters["limits"]["progress_accel_max"])
        x0[cs.X_PROGRESS_RATE] = min(
            max(0.0, float(x0[cs.X_PROGRESS_RATE])),
            self.progress_rate_max(path),
            left / self.Ts,
            float(np.sqrt(2.0 * accel * left)),
        )
        return x0

    def _checked_x0(
        self, x0: np.ndarray, horizon, q_eq: np.ndarray, path: PathCycle
    ) -> np.ndarray:
        """
        Validate the cycle's arguments and return `x0` with `s` re-origined.

        `horizon` may be absent: a caller that brought its own `PathCycle` has no
        resampled knots to check, and the path it did bring is checked by
        `_resolved_path`.
        """
        x0 = self._origined_x0(x0, path)
        if horizon is None:
            if not np.all(np.isfinite(np.asarray(q_eq, dtype=float))):
                raise ValueError(
                    "the sway equilibrium carries a value that is not finite"
                )
            return x0
        intervals = self.intervals
        if len(horizon) != intervals + 1:
            raise ValueError(
                f"the horizon carries {len(horizon)} knots and the problem is posed "
                f"on {intervals + 1}"
            )
        # acados answers a NaN reference with status 0, so nothing downstream
        # would tell this from a solver fault.
        if not all(
            np.all(np.isfinite(block))
            for block in (
                horizon.q_a_ref,
                horizon.dq_a_ref,
                np.asarray(q_eq, dtype=float),
            )
        ):
            raise ValueError(
                "the reference or the sway equilibrium carries a value that is "
                "not finite"
            )
        return x0

    def _path_parameters(self, path: PathCycle) -> np.ndarray:
        """Pack the model parameters, this path and where the cycle starts on it."""
        return stage_parameters(self._base_parameter, path.origin, path.control)

    def window_path(self, horizon) -> PathCycle:
        """
        Fit this horizon's own knots as the path, starting at 0: the interim source.

        `crane_planning` holds the curve as geometry and stage 2 of
        `docs/features/path-following-mpc` hands it over whole -- at which point
        a caller builds the `PathCycle` itself and this stops being the default.
        `bspline.fit` reads knot `i` at `i/intervals`, so a window's own pace is
        one window per `intervals * Ts`.
        """
        return PathCycle(
            control=path_control(horizon.q_a_ref[:, : cs.K_PLANNED_DOF]),
            origin=0.0,
            nominal_rate=problem.nominal_progress_rate(self.parameters),
        )

    def nominal_progress(self, stage: int, path: PathCycle) -> float:
        """
        Where on the path the plan's own pace puts this stage, as `theta`.

        Asking every stage for the end of the path instead is asking for flat
        out everywhere: `s` runs to its ceiling in the first 0.7 s of a 2.34 s
        horizon and the tracking rows spend the rest resisting it (measured,
        460 mm off path against 448 and 31 of 178 solves refused).
        """
        reached = float(path.origin) + stage * self.Ts * float(path.nominal_rate)
        return min(problem.K_PROGRESS_RATE_REFERENCE, reached)

    def progress_rate_max(self, path: PathCycle) -> float:
        """`v_s`'s ceiling: declared headroom against this path's own pace."""
        headroom = float(self.parameters["limits"]["progress_rate_headroom"])
        return headroom * float(path.nominal_rate)

    def progress_ceiling(self, stage: int, path: PathCycle) -> float:
        """
        `s`'s box top at this stage: as far as the rate reaches, or the path's end.

        The end cap is what keeps `s` a path parameter. `bspline` clamps the
        value past one but its *slope* stays the end tangent, so an uncapped `s`
        leaves the velocity row asking for `tangent(1) * v_s` at every stage
        beyond the path -- motion away from the goal, priced at the tracking
        weight. Both robocrane variants box `theta` in [0, 1] for this reason.
        """
        reach = stage * self.Ts * self.progress_rate_max(path)
        return min(reach, path.remaining())

    def _write_problem(
        self,
        x0: np.ndarray,
        path: PathCycle,
        q_eq: np.ndarray,
        guess: Guess | None,
        in_flight=None,
    ) -> bool:
        """
        Write everything the solve reads except the initial-state bound's own value.

        Returns whether the cycle is warm. `in_flight`: the command the JTC
        plays over [t, t+Ts), what stage 0's move cost prices against; `None`
        leaves it out.

        Boxes, references and the seed are built for the whole horizon at once and
        `p`/`x`/`u` go down flat: at N = 80 the per-stage Python cost 7 ms of the
        timer callback, which is not on the command's path but is on the next
        cycle's (issue 187).
        """
        intervals = self.intervals

        # A payload step forces the cold start, whatever the caller passed.
        # Decided here rather than below because the reset reads it.
        warm = (
            not self._payload_changed
            and guess is not None
            and guess.warm(intervals, self.nx)
        )
        self._payload_changed = False

        # The **iterate** is always dropped: `seed` below writes every stage, so
        # what starts the solve is this call's own guess and never last cycle's
        # leftovers. HPIPM's own memory is dropped only on a cold cycle -- it is
        # an interior-point solver, and re-entering near the previous
        # factorisation is worth 5 QP iterations against 15, 5.5 ms against
        # 8.8 ms and six fewer cycles over `T_s` (scripts/bench_ocp.py, 25 moves
        # / 6252 cycles; tracking, sway, force and pump identical to 4 decimals).
        # A cold cycle must still drop it: that is what makes a payload step, a
        # failure or a first cycle start from the central path rather than from
        # a factorisation of a problem this one is not.
        self.solver.reset(reset_qp_solver_mem=0 if warm else 1)

        force_reference = self.static_hold_force(x0)
        measured = x0[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
        rate_max = self.progress_rate_max(path)
        equilibrium = np.asarray(q_eq, dtype=float)
        elapsed = np.arange(intervals + 1) * self.Ts
        # One box per stage, all the same but for two entries: `s`'s ceiling, and
        # the sway rows when the caller knows an equilibrium per stage.
        lower, upper = state_bounds(
            self.parameters, stage_equilibrium(q_eq, 0), measured, rate_max
        )
        lower = np.repeat(lower[None], intervals + 1, axis=0)
        upper = np.repeat(upper[None], intervals + 1, axis=0)
        if equilibrium.ndim > 1:
            sway = slice(
                cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF
            )
            q_u_max = np.asarray(self.parameters["limits"]["q_u_max"], dtype=float)
            lower[:, sway] = equilibrium - q_u_max
            upper[:, sway] = equilibrium + q_u_max
        # `progress_ceiling` per stage: as far as the rate reaches, or the path's end.
        upper[:, cs.X_PROGRESS] = np.minimum(elapsed * rate_max, path.remaining())

        # `stage_reference` per stage: the tracking rows ride in `p` and stay zero,
        # the terminal row set is the leading `NY_TERMINAL` of a stage's.
        reference = np.zeros((intervals + 1, self._ny))
        passive = slice(
            problem.Y_PASSIVE_POSITION,
            problem.Y_PASSIVE_POSITION + cs.K_PASSIVE_DOF,
        )
        reference[:, passive] = equilibrium
        # `nominal_progress` per stage: where the plan's own pace reaches by then.
        reference[:, problem.Y_PROGRESS] = np.minimum(
            problem.K_PROGRESS_RATE_REFERENCE,
            float(path.origin) + elapsed * float(path.nominal_rate),
        )
        reference[
            :intervals,
            problem.Y_ACTUATED_FORCE : problem.Y_ACTUATED_FORCE + cs.K_PLANNED_DOF,
        ] = force_reference

        # Kept because `cost_terms` has to score against the path this cycle was
        # written to, not a refit of it.
        parameter = self._path_parameters(path)
        self._last_parameter = parameter
        self._last_path = path
        self._in_flight = (
            None
            if in_flight is None or not self.suppresses_moves
            else np.array(in_flight, dtype=float)
        )
        if self.suppresses_moves:
            weight = self._weight
            if self._in_flight is not None:
                weight, reference[0] = move_suppression(
                    weight, reference[0], self._du, self._in_flight
                )
            self.solver.cost_set(0, "W", weight)

        # These boxes go down unconverted, and that is an invariant and not an
        # omission: `problem.state_scale` raises unless every boxed row stands
        # at scale 1, so the box *is* in the QP's own coordinates already.
        self.solver.set_flat("p", np.tile(parameter, intervals + 1))
        for stage in range(intervals):
            self.solver.cost_set(stage, "yref", reference[stage])
            self.solver.constraints_set(stage + 1, "lbx", lower[stage + 1])
            self.solver.constraints_set(stage + 1, "ubx", upper[stage + 1])
        self.solver.cost_set(
            intervals, "yref", reference[intervals, : problem.NY_TERMINAL]
        )
        # Initial condition: every row, actuator states included, pinned at x0.
        self.solver.constraints_set(0, "lbx", self._to_qp(x0))
        self.solver.constraints_set(0, "ubx", self._to_qp(x0))

        seed = guess if warm else self.cold_start(x0)
        states = seed.states.copy()
        states[0] = x0
        states[1:, self._boxed] = np.clip(states[1:, self._boxed], lower[1:], upper[1:])
        inputs = np.clip(seed.inputs, -self._u_max, self._u_max)
        self.solver.set_flat("x", self._to_qp(states).reshape(-1))
        self.solver.set_flat("u", inputs.reshape(-1))
        # The iterate as written, not the caller's guess: a cold cycle rolled its
        # own and a warm one was clipped, so this is the only seed a replay can
        # hand back and get the same QP.
        self._written = {
            # One state: the point this problem is linearised about *and* the
            # initial-state bound. The two were separate fields while the
            # preparation/feedback split existed, which is what let them differ
            # (issue 226 removed it).
            "x0": x0,
            "q_eq": np.asarray(q_eq, dtype=float),
            "path": path,
            "guess": Guess(states, inputs),
            # Not derivable from `x0`: the force rows are `h_eff` at `x0`, a
            # casadi evaluation through the cylinder kinematics, so a finite
            # state can still write a non-finite reference.
            "yref": reference,
            "in_flight": self._in_flight,
            "warm": warm,
        }
        return warm

    def _read_solution(self, status: int, solve_time: float, warm: bool) -> Solution:
        """Read back what the solver answered."""
        intervals = self.intervals
        qp_status = _last(self.solver.get_stats("qp_stat"))
        qp_iterations = _last(self.solver.get_stats("qp_iter"))
        iterations = int(self.solver.get_stats("sqp_iter"))

        # One call per field rather than one per stage: 161 crossings into acados
        # sat between the solve and the publish (issue 187).
        states = self._to_physical(
            self.solver.get_flat("x").reshape(intervals + 1, self.nx)
        )
        inputs = self.solver.get_flat("u").reshape(intervals, cs.NU_PROGRESS)

        violation, penalty, used_slack = self._slack_taken()

        finite = np.all(np.isfinite(states)) and np.all(np.isfinite(inputs))
        word = status_word(status)
        if qp_status != 0:
            # Named, not just numbered: `qp_stat` is an acados status and not
            # HPIPM's (`ocp_qp_hpipm.c` translates it), so 3 is `MIN_STEP` and 1
            # is `NAN_SOL`. Issue 213 read the raw HPIPM enum off the number and
            # chased a NaN that was a collapsed step.
            word = f"{word} ({status_word(qp_status)}, QP status {qp_status})"
        if not finite or status != 0 or qp_status != 0:
            outcome = Outcome.FAILED
        elif solve_time > self.solve_budget_s:
            outcome = Outcome.BUDGET_EXCEEDED
        else:
            outcome = Outcome.CONVERGED

        advance = float(states[1][cs.X_PROGRESS]) if len(states) > 1 else 0.0
        advance = float(
            np.clip(advance, 0.0, self.progress_ceiling(1, self._last_path))
        )

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

    # -- one cycle, solved whole -------------------------------------------------

    def _resolved_path(self, horizon, path: PathCycle | None) -> PathCycle:
        """
        Return the path this cycle is written against: the caller's, or the window's.

        A caller that hands one over owns where it starts and how fast the plan
        means to spend it; one that does not gets the horizon's own knots fitted
        per cycle, which is what the node has until `crane_planning` publishes
        geometry.
        """
        if path is None:
            if horizon is None:
                raise ValueError("a cycle needs either a horizon or a path")
            return self.window_path(horizon)
        if float(path.nominal_rate) <= 0.0:
            raise ValueError(
                f"the path's nominal rate is {path.nominal_rate:g}; a plan that never "
                "advances cannot be the pace the progress row is priced against"
            )
        return path

    def solve(
        self,
        x0: np.ndarray,
        horizon,
        q_eq: np.ndarray,
        guess: Guess | None = None,
        path: PathCycle | None = None,
        in_flight=None,
    ) -> Solution:
        """
        One whole RTI step, linearisation and QP together, every cycle.

        **The preparation/feedback split is gone** (issue 226). It cost nothing
        offline and bought 4.7 ms of measurement-to-command latency, but live it
        refused 24-28 % of cycles, every one of them a feedback phase. Replaying
        the 45 dumped ticks said why, and it is not conditioning: with the bound
        left at the preparation's own linearisation point the feedback phase
        converges on exactly the ticks a whole solve converges on (19/45, tick
        for tick); with the measured bound it converges on none. Moving the
        bound alone is what fails, and `q_a` alone reproduces it -- the machine
        sits outside the control-safe position box (see `position_box`), so that
        box is pinned to the *predicted* position and a measurement a
        thousandth of a radian further out leaves stage 1 unreachable.
        Falling back to a whole solve in the same cycle recovers 20 of the 21
        but costs 16 ms p50 and 59 ms max against a 60 ms budget, before live
        contention; solving whole costs 6.8 ms p50 (issue 223) and recovers 21.
        """
        cycle = self._resolved_path(horizon, path)
        x0 = self._checked_x0(x0, horizon, q_eq, cycle)
        warm = self._write_problem(x0, cycle, q_eq, guess, in_flight)
        status = int(self.solver.solve())
        return self._read_solution(
            status, float(self.solver.get_stats("time_tot")), warm
        )

    def last_problem(self) -> dict:
        """
        Give what the last written problem was built from, for `dump.tick`.

        `p` is read back off the solver rather than rebuilt: it is where a
        non-finite payload row, tool row or path control point actually lands,
        and reading it is the only way to see one that was written. The
        residuals are recomputed because `SQP_RTI` keeps none -- all four come
        back zero otherwise -- and `res_stat` alone separates a NaN that arrived
        in the data from one the linearisation produced.
        """
        return dict(
            self._written,
            p=self.solver.get_flat("p").reshape(self.intervals + 1, problem.NP),
            residuals=np.asarray(
                self.solver.get_residuals(recompute=True), dtype=float
            ),
        )

    def _slack_taken(self) -> tuple[Violation, float, bool]:
        """
        Read what the softened constraints spent, off `sl`/`su` over the horizon.

        Row order (`idxsbx`): sway pair, sway-rate pair, cylinder-force rows, pump row.

        Two `get_flat` calls, not two per stage: this sits between the solve and
        the publish and only feeds the health report, so at N = 80 the per-stage
        loop cost 3 ms of the command's own latency (issue 187).
        """
        limits = self.parameters["limits"]
        q_u_max = np.asarray(limits["q_u_max"], dtype=float)
        dq_u_max = np.asarray(limits["dq_u_max"], dtype=float)
        lower = np.maximum(self.solver.get_flat("sl"), 0.0)
        upper = np.maximum(self.solver.get_flat("su"), 0.0)
        taken = np.maximum(lower, upper)
        penalty = float(self._slack_price @ (lower + upper))

        state, nonlinear = self._soft_state, self._soft_nonlinear
        # The stages between the ends carry both blocks, so they are one rectangle;
        # stage 0's nonlinear rows and the terminal stage's state rows sit outside it.
        path = taken[nonlinear : taken.size - state].reshape(-1, state + nonlinear)
        soft_state = np.concatenate(
            [path[:, :state], taken[None, taken.size - state :]]
        )
        soft_nonlinear = np.concatenate([taken[None, :nonlinear], path[:, state:]])

        passive, planned = cs.K_PASSIVE_DOF, cs.K_PLANNED_DOF
        violation = Violation(
            q_u=float(np.max(soft_state[:, :passive] / q_u_max)),
            dq_u=float(np.max(soft_state[:, passive:] / dq_u_max)),
            cylinder_force=float(np.max(soft_nonlinear[:, :planned])),
            pump_flow=float(np.max(soft_nonlinear[:, planned])),
        )
        return violation, penalty, bool(np.max(taken) > SLACK_NOTICEABLE)

    def cost_terms(self, solution: Solution, q_eq: np.ndarray) -> CostTerms:
        """Cost, term by term: re-evaluated from the residual, since acados reports one number."""
        if self._last_parameter is None:
            raise RuntimeError(
                "no problem has been written, so there is no cost to split"
            )
        terms = CostTerms(slack=solution.slack_penalty)
        force_reference = self.static_hold_force(solution.states[0])
        diagonal = np.diag(self._weight)
        terminal_diagonal = np.diag(self._terminal_weight)
        planned = cs.K_PLANNED_DOF
        passive = cs.K_PASSIVE_DOF
        parameter = self._last_parameter
        for stage in range(self.intervals + 1):
            terminal = stage == self.intervals
            reference = stage_reference(
                stage_equilibrium(q_eq, stage),
                force_reference,
                terminal,
                self.nominal_progress(stage, self._last_path),
                self._ny,
            )
            if terminal:
                value = np.asarray(
                    self._terminal_residual(
                        self._to_qp(solution.states[stage]), parameter
                    )
                ).reshape(-1)
                terms.terminal += float(
                    0.5 * terminal_diagonal @ (value - reference) ** 2
                )
                continue
            value = np.asarray(
                self._residual(
                    self._to_qp(solution.states[stage]),
                    solution.inputs[stage],
                    parameter,
                )
            ).reshape(-1)
            weight = diagonal
            if stage == 0 and self._in_flight is not None:
                weight, reference = move_suppression(
                    np.diag(diagonal), reference, self._du, self._in_flight
                )
                weight = np.diag(weight)
            row = 0.5 * weight * (value - reference) ** 2
            terms.q_a += float(np.sum(row[problem.Y_PLANNED_POSITION :][:planned]))
            terms.dq_a += float(np.sum(row[problem.Y_PLANNED_VELOCITY :][:planned]))
            terms.q_u += float(np.sum(row[problem.Y_PASSIVE_POSITION :][:passive]))
            terms.dq_u += float(np.sum(row[problem.Y_PASSIVE_VELOCITY :][:passive]))
            terms.lag += float(row[problem.Y_LAG])
            terms.tool += float(np.sum(row[problem.Y_TOOL :][:3]))
            # Both rows of the pair: the path-following one carries the weight
            # (8 against 0.1), so reporting the rate alone hid the term.
            terms.progress += float(
                row[problem.Y_PROGRESS] + row[problem.Y_PROGRESS_RATE]
            )
            terms.tau_a += float(np.sum(row[problem.Y_ACTUATED_FORCE :][:planned]))
            inputs = float(np.sum(row[problem.Y_INPUT :][: cs.NU_PROGRESS]))
            if self.command_state:
                rate = float(np.sum(row[problem.Y_INPUT :][:planned]))
                terms.du += rate
                terms.u += inputs - rate + float(np.sum(row[problem.Y_COMMAND :]))
            else:
                terms.u += inputs
        return terms


def _finite(*values) -> list[np.ndarray]:
    """Return the arguments as float arrays, refused unless every entry is finite."""
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(
            "the measured state and the applied input must be finite to be "
            "carried through the transport dead time"
        )
    return arrays


def _last(value) -> int:
    """Take the last entry: acados reports these per RTI call."""
    array = np.atleast_1d(np.asarray(value)).reshape(-1)
    return int(array[-1]) if array.size else 0
