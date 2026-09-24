"""
The generated acados solver, opened and driven: `src/ocp_solver.cpp`'s Python counterpart.

The binding around acados' unchanged artifact: what is written before a solve, read
back after, plus the dead-time predictor and static hold force. State is the OCP's
own 25 rows end to end, unlike the C++'s 16-wide truncated `crane_model::State`,
which coasted its dead-time predictor at a frozen force (issue 125).
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
    #: Seconds the preparation phase cost, or zero when the cycle was solved
    #: whole. Not counted in `solve_time_s`, which is what the budget is about:
    #: under the split only the feedback phase sits between the measurement and
    #: the command, and the preparation ran before the measurement existed.
    preparation_time_s: float = 0.0

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
    progress: float = 0.0
    tau_a: float = 0.0
    u: float = 0.0
    terminal: float = 0.0
    slack: float = 0.0


# ------------------------------------------------------- what the solver is told


def weight_matrices(parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stage and terminal nonlinear-least-squares weights."""
    weights = parameters["weights"]
    planned = cs.K_PLANNED_DOF
    passive = cs.K_PASSIVE_DOF
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
            np.asarray(weights["u"][:planned], dtype=float),
            [float(weights["progress_accel"])],
        ]
    )
    terminal = float(weights["terminal_scale"]) * diagonal[: problem.NY_TERMINAL]
    return np.diag(diagonal), np.diag(terminal)


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
            [0.0, rate_max],
        ]
    )
    return lower, upper


def stage_equilibrium(q_eq: np.ndarray, stage: int) -> np.ndarray:
    """
    Return this stage's sway equilibrium: one held vector, or one row per stage.

    The node reads one equilibrium at the measured state and holds it
    (`ocp_solver.cpp`'s trade); a harness that knows the whole path can solve one
    per stage. Taking either here is what lets the two share a cycle.
    """
    table = np.asarray(q_eq, dtype=float)
    return table if table.ndim == 1 else table[stage]


def stage_reference(
    q_eq: np.ndarray, tau_ref: np.ndarray, terminal: bool, progress: float
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
    that pace.
    """
    reference = np.zeros(problem.NY_TERMINAL if terminal else problem.NY)
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
    # Input box: five joint commands plus progress acceleration (not an axis, own bound).
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
        "Ts": parameters["Ts"],
        # The delay decides which integrators `predictor_label` names into the
        # export, so it is compiled in exactly as the acados settings are. It was
        # missing here, and a changed delay therefore opened its predecessor's
        # `.so` and read as a null result.
        "sensor_to_valve_delay": parameters["sensor_to_valve_delay"],
        "horizon_length": parameters["horizon_length"],
        "levenberg_marquardt": parameters["levenberg_marquardt"],
        "qp_solver_cond_N": parameters.get("qp_solver_cond_N"),
        "tuning": problem.solver_tuning(),
        "hydraulics": hydraulics,
        "description": hashlib.sha256(description_xml.encode()).hexdigest()[:16],
        "problem.py": _file_digest(problem.__file__),
        "symbolic.py": _file_digest(cs.__file__),
        "actuator_fit": _file_digest(cs.default_actuator_path()),
    }


def _sub_steps(seconds: float, sample_time: float) -> int:
    """How many integrator steps the interval takes with none longer than `T_s`."""
    return max(1, int(np.ceil(seconds / sample_time)))


@dataclass(frozen=True)
class ReplaySegment:
    """One command in flight: which history entry it is, and how much it covers."""

    #: Index into the newest-first applied-input history. 0 is what is applied now.
    age: int
    #: Seconds of the delay window that entry was the applied command for.
    seconds: float


#: yaml floats: `0.08/0.04` can land just under 2; a one-ulp remainder is round-off.
GRID_TOLERANCE = 1e-9

#: A delay of a thousand intervals is a unit slip (60 read as seconds for 0.06).
MAX_REPLAY_INTERVALS = 1024.0


def replay_schedule(delay_s: float, step_s: float) -> list[ReplaySegment]:
    """
    Cut the dead time into one segment per command the machine was actually given.

    Oldest first; the partial interval is oldest, belonging to entry
    `floor(delay_s / step_s)`, not the newest. Empty for a non-positive, non-finite
    or unit-slip-deep delay.
    """
    if not (np.isfinite(delay_s) and np.isfinite(step_s)):
        return []
    if delay_s <= 0.0 or step_s <= 0.0 or delay_s / step_s > MAX_REPLAY_INTERVALS:
        return []
    full = int(np.floor(delay_s / step_s + GRID_TOLERANCE))
    remainder = delay_s - full * step_s
    segments = []
    if remainder > GRID_TOLERANCE * step_s:
        segments.append(ReplaySegment(full, remainder))
    segments.extend(ReplaySegment(age, step_s) for age in range(full - 1, -1, -1))
    return segments


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
    """
    sim = AcadosSim()
    sim.model = ocp.model
    sim.parameter_values = ocp.parameter_values
    sim.solver_options.T = seconds
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_stages = 2
    sim.solver_options.num_steps = _sub_steps(seconds, sample_time)
    return sim


def predictor_sims(ocp, parameters: dict) -> list:
    """Every `(label, AcadosSim)` an export must carry for this configuration."""
    sample_time = float(parameters["Ts"])
    return [
        (
            predictor_label(seconds, sample_time),
            predictor_sim(ocp, seconds, sample_time),
        )
        for seconds in predictor_intervals(parameters)
    ]


def predictor_intervals(parameters: dict) -> list:
    """Every interval an `Ocp` will want an integrator for, longest first."""
    sample_time = float(parameters["Ts"])
    delay = float(parameters["sensor_to_valve_delay"])
    wanted = {sample_time}
    if delay > 0.0:
        wanted.add(delay)
        wanted.update(
            segment.seconds for segment in replay_schedule(delay, sample_time)
        )
    return sorted(wanted, reverse=True)


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
        # Opened, never built: `scripts/export_ocp.py` compiled this and wrote
        # the key beside it, and `ox.open_solver` refuses anything that is not
        # this problem rather than starting on another machine's dynamics.
        base = export_base()
        key = export_key(parameters, hydraulics, description_xml)
        self.solver, self._dims = ox.open_solver(
            self._ocp, base, key, EXPORT_COMMAND, verbose=verbose
        )
        self._check_dimensions()

        # Residual outside the solver: acados reports one cost, this splits tracking/sway/effort.
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

        def predictor(seconds: float):
            return ox.open_sim(
                predictor_sim(self._ocp, seconds, self.Ts),
                base,
                key,
                predictor_label(seconds, self.Ts),
                verbose=verbose,
            )

        self._stepper = predictor(self.Ts)
        delay = float(parameters["sensor_to_valve_delay"])
        self._predictor = predictor(delay) if delay > 0.0 else None
        self.delay_s = delay
        #: How the delay window splits between the commands in flight (issue 125).
        self.replay = replay_schedule(delay, self.Ts)
        # One integrator per segment, built here not in the cycle (too slow for a callback).
        self._segment_predictor = {
            segment.seconds: (
                self._stepper
                if segment.seconds == self.Ts
                else predictor(segment.seconds)
            )
            for segment in self.replay
        }

        self._base_parameter = np.zeros(cs.NP)
        self.payload_mass_kg = 0.0
        self.payload_com_m = np.zeros(3)
        self._payload_changed = False
        #: The parameter vector the last written problem carried, path included,
        #: and the `PathCycle` it was built from -- what `_read_solution` and
        #: `cost_terms` have to read the solution against.
        self._last_parameter: np.ndarray | None = None
        self._last_path: PathCycle | None = None

        #: Whether the preparation/feedback split is reachable at all. acados
        #: refuses `rti_phase` outside `SQP_RTI`, so a swept `nlp_solver_type`
        #: leaves every cycle on the combined `solve`.
        self.split_rti = problem.solver_tuning()["nlp_solver_type"] == "SQP_RTI"
        #: Whether a `prepare` is standing that `feedback` may consume.
        self._prepared = False
        self._prepared_warm = False
        self._preparation_time = 0.0

    # -- what the problem is, checked against what was compiled -----------------

    def _check_dimensions(self) -> None:
        """Check dims `ocp_solver.cpp:48-71` asserts: a stale solver is one for another problem."""
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
            self.solver, self._chamber, self.scale, self.parameters, self.hydraulics
        )

    # -- the model, asked the two questions the solver does not answer ----------

    def set_payload(self, mass_kg: float, com_m) -> bool:
        """Bind a payload into `p`. Returns whether it changed the model."""
        config.check_payload(mass_kg, com_m)
        com = np.asarray(com_m, dtype=float)
        changed = mass_kg != self.payload_mass_kg or not np.array_equal(
            com, self.payload_com_m
        )
        # Tracked here, not via caller's `guess=None`: `h_eff` jumps at a grasp.
        self._payload_changed = self._payload_changed or changed
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
        predictor.set("x", x)
        predictor.set("u", u)
        status = predictor.solve()
        if status != 0:
            raise RuntimeError(
                f"the dead-time predictor answered {status_word(status)}"
            )
        predicted = predictor.get("x")
        # acados can report success with a NaN state (`ocp_solver.cpp:1156-1162`).
        if not np.all(np.isfinite(predicted)):
            raise RuntimeError(
                "propagating the measured state forward under the applied command "
                "left it non-finite"
            )
        return predicted

    def propagate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        `x` carried forward through the transport dead time under `u`.

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
        if self._predictor is None:
            return x.copy()
        moved = self._integrate(self._predictor, x, u)
        moved[PROGRESS_ROWS] = x[PROGRESS_ROWS]
        return moved

    def propagate_applied(self, x: np.ndarray, applied) -> np.ndarray:
        """
        Carry `x` under the commands actually applied over the delay, newest first.

        One hold per command (`self.replay`); a short history clamps oldest, empty is refused.
        """
        if len(applied) == 0:
            raise ValueError(
                "replaying the commands in flight needs at least the one applied "
                "now; an empty history is not the same thing as a zero command "
                "and is not guessed at here"
            )
        state, *history = _finite(x, *applied)
        if not self.replay:
            # Nothing to cut up: zero delay, or the single-input path complains.
            return self.propagate(state, history[0])
        # Held back across the whole replay, same reason as `propagate`.
        progress = state[PROGRESS_ROWS].copy()
        for segment in self.replay:
            state = self._integrate(
                self._segment_predictor[segment.seconds],
                state,
                history[min(segment.age, len(history) - 1)],
            )
        state[PROGRESS_ROWS] = progress
        return state

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

    def carried(self, solution: Solution) -> Guess:
        """
        Return the next cycle's warm start: the last solution, **not** shifted.

        Shifting is the textbook RTI move and it is the wrong one here. One
        Newton step per cycle is a small budget and the shift spends it:
        dropping knot 0 and duplicating knot N perturbs the iterate, and the
        single iteration goes on repairing that perturbation instead of
        improving the plan. The result is a plan that defers its own correction
        one knot further every cycle, and since only knot 0 is ever executed,
        the correction never happens.

        Measured on the model-matched plant (`mpc_a2b.py`, 25 s settle): shifted
        walks *away* from the goal, 0.139 -> 0.209 rad over 29 s, monotonically,
        with every solve converged and no fallback; unshifted decays to
        1.4e-4 rad. The QP's own numbers say the same -- shifted, it moves `u0`
        by 9e-5 per cycle against a 0.039 gap to the converged answer.

        Over `sim_chain --random 10` on two seeds the endpoint median halves
        (18.3 -> 9.5 mm, 42.8 -> 16.3 mm) and -- unlike buying more iterations --
        so does the worst (40.2 -> 21.1 mm, 74.3 -> 24.9 mm), with path error a
        shade better and six refused cycles gone. Sway median moves 0.142 ->
        0.155 rad on one seed, which is the one column that pays.

        `shifted` stays for the fallback below, where a shift is not a guess but
        the answer: those knots get published against instants that have moved.
        """
        return Guess(states=solution.states.copy(), inputs=solution.inputs.copy())

    def shifted(self, solution: Solution) -> Guess:
        """Shift the horizon one knot left as the next warm start; progress row re-origined."""
        states = np.vstack([solution.states[1:], solution.states[-1:]])
        inputs = np.vstack([solution.inputs[1:], solution.inputs[-1:]])
        states[:, cs.X_PROGRESS] = np.maximum(
            states[:, cs.X_PROGRESS] - max(solution.progress_advance, 0.0), 0.0
        )
        return Guess(states, inputs)

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
        if x0.shape != (cs.NX,) or not np.all(np.isfinite(x0)):
            raise ValueError("x0 is not a finite state of this problem")
        x0[cs.X_PROGRESS] = 0.0
        x0[cs.X_PROGRESS_RATE] = min(
            max(0.0, float(x0[cs.X_PROGRESS_RATE])), self.progress_rate_max(path)
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
        # Curvature is the sharp one: Hermite's 2nd derivative carries `1/dt^2` (unbounded below).
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
    ) -> bool:
        """
        Write everything the solve reads except the initial-state bound's own value.

        Returns whether the cycle is warm. Split out of `solve` so `prepare` can
        run it against a predicted state one cycle early.
        """
        intervals = self.intervals

        # A payload step forces the cold start, whatever the caller passed.
        # Decided here rather than below because the reset reads it.
        warm = not self._payload_changed and guess is not None and guess.warm(intervals)
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
        # Same box but for two entries: `s`'s ceiling, and the sway rows when the
        # caller knows an equilibrium per stage. Built once, moved per stage.
        lower, upper = state_bounds(
            self.parameters, stage_equilibrium(q_eq, 0), measured, rate_max
        )
        sway = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF)
        travelling = np.asarray(q_eq, dtype=float).ndim > 1
        q_u_max = np.asarray(self.parameters["limits"]["q_u_max"], dtype=float)

        def box(stage: int) -> tuple[np.ndarray, np.ndarray]:
            """Return the box at this stage; only the travelling entries move."""
            if travelling:
                equilibrium = stage_equilibrium(q_eq, stage)
                lower[sway] = equilibrium - q_u_max
                upper[sway] = equilibrium + q_u_max
            upper[cs.X_PROGRESS] = self.progress_ceiling(stage, path)
            return lower, upper

        # Kept because `cost_terms` has to score against the path this cycle was
        # written to, not a refit of it.
        parameter = self._path_parameters(path)
        self._last_parameter = parameter
        self._last_path = path
        for stage in range(intervals + 1):
            self.solver.set(stage, "p", parameter)
            self.solver.cost_set(
                stage,
                "yref",
                stage_reference(
                    stage_equilibrium(q_eq, stage),
                    force_reference,
                    stage == intervals,
                    self.nominal_progress(stage, path),
                ),
            )
            if stage == 0:
                # Initial condition: every row, actuator states included, pinned at x0.
                self.solver.constraints_set(0, "lbx", x0)
                self.solver.constraints_set(0, "ubx", x0)
                continue
            self.solver.constraints_set(stage, "lbx", box(stage)[0])
            self.solver.constraints_set(stage, "ubx", upper)

        seed = guess if warm else self.cold_start(x0)
        for stage in range(intervals + 1):
            if stage == 0:
                state = x0
            else:
                state = seed.states[stage].copy()
                box(stage)
                state[: cs.NBX] = np.clip(state[: cs.NBX], lower, upper)
            self.solver.set(stage, "x", state)
            if stage < intervals:
                self.solver.set(
                    stage,
                    "u",
                    np.clip(seed.inputs[stage], -self._u_max, self._u_max),
                )
        return warm

    def _read_solution(
        self, status: int, solve_time: float, warm: bool, preparation_time: float
    ) -> Solution:
        """Read back what the solver answered, whichever phases produced it."""
        intervals = self.intervals
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
            preparation_time_s=preparation_time,
        )

    # -- the three ways to spend a cycle -----------------------------------------

    def _set_phase(self, phase: int) -> None:
        """
        Select PREPARATION_AND_FEEDBACK (0), PREPARATION (1) or FEEDBACK (2).

        acados refuses the option outside `SQP_RTI`, so a swept `nlp_solver_type`
        keeps the combined path and never reaches here with a split phase.
        """
        self.solver.options_set("rti_phase", int(phase))

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
    ) -> Solution:
        """
        One whole RTI step, linearisation and QP together.

        The path taken when nothing was prepared for this cycle: the first one,
        one after a failure, or one whose problem moved under the preparation.
        """
        cycle = self._resolved_path(horizon, path)
        x0 = self._checked_x0(x0, horizon, q_eq, cycle)
        # A preparation this call overwrites is gone, and must not be consumable
        # by a later `feedback`.
        self._prepared = False
        warm = self._write_problem(x0, cycle, q_eq, guess)
        if self.split_rti:
            self._set_phase(0)
        status = int(self.solver.solve())
        return self._read_solution(
            status, float(self.solver.get_stats("time_tot")), warm, 0.0
        )

    def prepare(
        self,
        x0: np.ndarray,
        horizon,
        q_eq: np.ndarray,
        guess: Guess | None = None,
        path: PathCycle | None = None,
    ) -> float:
        """
        Everything but the QP, against a **predicted** state, one cycle early.

        `x0` is where the machine is expected to be when the next cycle's plan
        takes effect, not where it is now: the whole point is that this runs
        before that cycle's measurement exists. Returns the seconds it cost,
        which are not on the measurement-to-command path.
        """
        cycle = self._resolved_path(horizon, path)
        x0 = self._checked_x0(x0, horizon, q_eq, cycle)
        self._prepared_warm = self._write_problem(x0, cycle, q_eq, guess)
        self._set_phase(1)
        self.solver.solve()
        self._prepared = True
        self._preparation_time = float(self.solver.get_stats("time_tot"))
        return self._preparation_time

    def feedback(self, x0: np.ndarray) -> Solution:
        """
        Solve the QP alone, against the measured state, on the last `prepare`.

        Only the initial-state bound moves between the two phases -- that is the
        initial value embedding, and it is the whole of what the measurement is
        allowed to change once the linearisation is standing.
        """
        if not self._prepared:
            raise RuntimeError(
                "the feedback phase was asked for without a preparation to stand "
                "on; a cycle whose preparation was invalidated takes `solve`"
            )
        self._prepared = False
        # The preparation's own path: only the initial-state bound may move here.
        x0 = self._origined_x0(x0, self._last_path)
        # The bound only. Writing `x` here as well would move the point the
        # preparation linearised about, and the QP step -- already computed to
        # travel from that point to this bound -- would land on top of it: a
        # measurement 0.05 off the prediction came back as 0.1.
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        self._set_phase(2)
        status = int(self.solver.solve())
        feedback_time = float(self.solver.get_stats("time_tot"))
        return self._read_solution(
            status, feedback_time, self._prepared_warm, self._preparation_time
        )

    def _slack_taken(self) -> tuple[Violation, float, bool]:
        """
        Read what the softened constraints spent, off `sl`/`su` per stage.

        Row order (`idxsbx`): sway pair, sway-rate pair, cylinder-force rows, pump row.
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
            )
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
            # Both rows of the pair: the path-following one carries the weight
            # (8 against 0.1), so reporting the rate alone hid the term.
            terms.progress += float(
                row[problem.Y_PROGRESS] + row[problem.Y_PROGRESS_RATE]
            )
            terms.tau_a += float(np.sum(row[problem.Y_ACTUATED_FORCE :][:planned]))
            terms.u += float(np.sum(row[problem.Y_INPUT :][: cs.NU_PROGRESS]))
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
