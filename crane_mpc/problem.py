"""
The optimal control problem, as an `AcadosOcp`.

Built once here; the node and `scripts/export_ocp.py` both import this
definition rather than restating it. Algebra comes from `crane_model.symbolic`
untouched; this module adds the shooting problem, residual, constraints, grid.

Baked at code generation: constraint structure, `h`'s row order/divisors,
dynamics. Not baked: payload (`p`, issue 072) and every weight/bound/price,
set by `solver.py` at configure time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from crane_model import symbolic as cs

from . import bspline


def chamber_forces(model, relief_pa: float, actuated_dof: int) -> tuple:
    """
    Return `(extend, retract)` at relief pressure: `F_i = A_A p_A - A_B p_B`.

    The live one. `crane_ocp_export` carries a byte-identical copy, written when
    an exporter could not reach this package; `export_ocp.py` resolves
    `crane_mpc.problem` off the source tree now, nothing calls that copy, and it
    can go once `concrete_block_stack` has a branch to take it on.

    Both exporters condition their force rows by the **larger** of the pair, so
    a row the planner conditions and a row the MPC conditions are one number.
    """
    pressure = np.full(actuated_dof, float(relief_pa))
    zero = np.zeros(actuated_dof)
    extend = np.array(ca.evalf(model.chamber_force(pressure, zero))).ravel()
    retract = np.array(ca.evalf(model.chamber_force(zero, pressure))).ravel()
    return extend, retract


# --- the acados backend, and the one place it is written down -----------------
#
# Every setting that is compiled into the solver and is a choice rather than a
# consequence. `scripts/sweep_ocp.py` layers a variant over this through
# `CRANE_MPC_OCP_OPTIONS`, and `solver.solver_signature` hashes the result --
# without that a variant opens its predecessor's `.so` and reads as a null
# result, which is how a sweep measures the baseline 33 times.
#
#     CRANE_MPC_OCP_OPTIONS='{"nlp_solver_type": "SQP", "nlp_solver_max_iter": 3}'
#
SOLVER_TUNING = {
    # One Newton step per cycle: deterministic solve time, worth more than a
    # converged step since the plant moves between cycles anyway.
    "nlp_solver_type": "SQP_RTI",
    # Read only when `nlp_solver_type` is not RTI, which is what makes a
    # converging variant comparable against one Newton step.
    "nlp_solver_max_iter": 1,
    "qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "hpipm_mode": "BALANCE",
    # None is acados' default, `N`: issue 129 measured smaller blocks worse
    # (10.3 ms QP at `N`, 61-81 ms at 1).
    "qp_solver_cond_N": None,
    "qp_solver_iter_max": 50,
    # Primal **and** dual. Reachable only because `solver.py` keeps HPIPM's
    # memory across a warm cycle; with the memory dropped every cycle this flag
    # measured exactly the baseline, which is a null result that reads as "no
    # effect". 2 against 0: QP iterations 15 -> 5 median, 18 -> 11 at p90.
    "qp_solver_warm_start": 2,
    "qp_solver_ric_alg": 1,
    # C3's actuator lag makes the model stiff: linearised fastest eigenvalue
    # `|lambda| T_s = 8.5` at an ordinary pose (telescope `k = 3.5e6 N/m` vs
    # effective mass), where ERK4 is stable only to ~2.8 and diverges in three
    # intervals (HPIPM status 3). Two implicit stages carry that comfortably.
    "integrator_type": "IRK",
    "sim_method_num_stages": 2,
    "sim_method_num_steps": 1,
    "sim_method_newton_iter": 3,
    # Not for speed -- for what happens at a pose stiffer than the one above.
    # Legendre is A-stable with `R(z) -> 1` as `z -> -inf`, so the stiff mode is
    # bounded but undamped and rings; Radau IIA is L-stable (`R -> 0`) at one
    # order less (3 against 4). At `z = -8.5` that is `|R|` 0.098 against 0.25,
    # and the gap widens with the telescope out and a payload on.
    #
    # The order it gives up costs nothing measurable at `T_s`: 25 moves / 7k
    # cycles, terminal error 0.434 -> 0.432, sway/pump identical, force 0.557 ->
    # 0.556. It is not slower either (below).
    "collocation_type": "GAUSS_RADAU_IIA",
    # Reuse the IRK Jacobian across a step's Newton iterations instead of
    # re-forming and re-factorising it three times. Same corpus: solve 4.96 ->
    # 3.85 ms median and 16.1 -> 10.1 ms at p90 against Legendre without reuse,
    # with `qp_iter` 5/11 unmoved -- this is integrator cost, not QP cost -- and
    # every quality column within a digit. Two runs agree on the median; read
    # the p90 with the load average the sweep recorded, not on its own.
    "sim_method_jac_reuse": 1,
    # Gauss-Newton, so `J' W J` is positive semi-definite by construction and
    # the exact nonlinear-cost Hessians acados would emit go unused.
    "hessian_approx": "GAUSS_NEWTON",
    # No line search: one full Newton step is RTI. No adaptive regularisation
    # either -- `levenberg_marquardt` is a fixed constant added every cycle.
    "globalization": "FIXED_STEP",
    "regularize_method": "NO_REGULARIZE",
    "qpscaling_scale_constraints": "NO_CONSTRAINT_SCALING",
    "qpscaling_scale_objective": "NO_OBJECTIVE_SCALING",
    # Unread under RTI, which never checks convergence. Here so that an
    # `nlp_solver_type` variant can be swept honestly: at acados' 1e-6 a
    # four-iteration SQP answers `ACADOS_MAXITER` and every cycle of it is
    # scored as a failed one.
    "nlp_solver_tol_stat": 1.0e-6,
    "nlp_solver_tol_eq": 1.0e-6,
    "nlp_solver_tol_ineq": 1.0e-6,
    "nlp_solver_tol_comp": 1.0e-6,
}

#: Env var a sweep patches the table through. JSON object; an unknown key is an
#: error, since a misspelt knob would otherwise re-measure the baseline.
TUNING_ENV = "CRANE_MPC_OCP_OPTIONS"


def solver_tuning() -> dict:
    """`SOLVER_TUNING` with the environment's patch layered over it."""
    patch = json.loads(os.environ.get(TUNING_ENV, "{}"))
    unknown = set(patch) - set(SOLVER_TUNING)
    if unknown:
        raise ValueError(
            f"{TUNING_ENV} carries {sorted(unknown)}, which is not a setting this "
            f"problem writes; known: {sorted(SOLVER_TUNING)}"
        )
    return SOLVER_TUNING | patch


TOOL = "pzs100"
DESCRIPTION = "pzs100.urdf"


def default_description() -> Path:
    """
    Locate the description the solver's dynamics are baked from.

    Not `/robot_description`: this is the plant the solver compiled for, one
    file so one compiled solver serves every deployment. `crane_model`
    installs it beside its config.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = (
            Path(get_package_share_directory("crane_model"))
            / "description"
            / DESCRIPTION
        )
        if installed.is_file():
            return installed
    except Exception:
        pass
    return (
        Path(__file__).resolve().parents[2]
        / "crane_model"
        / "test"
        / "description"
        / DESCRIPTION
    )


# Prefix every generated symbol carries (acados derives it from the model
# name, e.g. `crane_mpc_pzs100_acados_create`); two solvers in one lib can't
# collide.
SOLVER_PREFIX = "crane_mpc"


# --- the OCP's own parameters, appended to the model's eleven ------------------
#
# `crane_symbolic`'s `p` is the tool coordinate and payload; what's appended
# here is local to this problem: the path, and where on it this cycle starts.
#
# A spline *can* bake into a generated solver -- its control points are
# parameters and its basis is an expression, which is what robocrane's path
# following does and what replaced the second-order expansion that used to sit
# here. The expansion was only the path near one point; this is the path.
#
# `c(theta)`, the planner's curve in its own parameter, fitted once per plan.
# The progress state *is* `theta`, so no timing law lives in here: where on the
# path to be is the optimizer's choice, and how fast to get there is what the
# limits and the cost decide. `ORIGIN` is where this cycle starts, since `s` is
# pinned to zero at stage zero and carries only the rest.
#
# The planner's usual answer is a straight line in joint space, exact at any
# count; its curved candidate needs 30 points for 1.3 mm at the tool.
PATH_POINTS = 30

P_PATH_ORIGIN = cs.NP
P_PATH_CONTROL = P_PATH_ORIGIN + 1
NP = P_PATH_CONTROL + PATH_POINTS * cs.K_PLANNED_DOF

PATH_KNOTS = bspline.knot_vector(PATH_POINTS)

# --- the residual blocks -------------------------------------------------------
#
# Terminal residual is the leading prefix of the stage one, sharing offsets.
# Lag/progress rows sit inside that prefix; effort/smoothness (need an input)
# come after.
Y_PLANNED_POSITION = 0
Y_PLANNED_VELOCITY = cs.K_PLANNED_DOF
Y_PASSIVE_POSITION = 2 * cs.K_PLANNED_DOF
Y_PASSIVE_VELOCITY = Y_PASSIVE_POSITION + cs.K_PASSIVE_DOF
Y_LAG = Y_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF
#: Where the tool is against where the path says it should be, in metres. The
#: target is a centimetre at the tool and every other tracking row is in joint
#: space, where the same error is worth 2174 to 11905 mm depending on the pose
#: -- a factor of 5.5 that no per-axis weight can absorb, which is why sweeping
#: them is inert. Three rows, position only; orientation is a second metric.
Y_TOOL = Y_LAG + 1
Y_PROGRESS = Y_TOOL + 3
Y_PROGRESS_RATE = Y_PROGRESS + 1
NY_TERMINAL = Y_PROGRESS_RATE + 1
Y_ACTUATED_FORCE = NY_TERMINAL
Y_INPUT = Y_ACTUATED_FORCE + cs.K_PLANNED_DOF
NY = Y_INPUT + cs.NU_PROGRESS

# Axis the lag row is written on: Marc's slewing joint alone
# (`timber_crane_cost_js_pfc_pt2.cpp:62-74`), not generalised across axes --
# the five planned coords mix four radians and one metre, so a unit tangent
# has no fixed conversion. Slewing is slowest, drives the pendulum laterally,
# and is the axis the progress variable buys time for.
K_LAG_AXIS = 0

# `theta = 1`: the end of the path, which is what the progress row is priced
# against now that the progress state is the path parameter. It was the rate's
# reference -- one second of plan per wall-clock second -- until the cost
# stopped saying when to be somewhere and started saying where to end up.
K_PROGRESS_RATE_REFERENCE = 1.0


# ------------------------------------------------------------------ the numbers


def shooting_intervals(parameters: dict) -> int:
    """
    Return `N`, the shooting-interval count, from the yaml's knot count.

    `horizon_length` counts knots (fifty = two seconds); `mpc_node.cpp:
    214-219` subtracts the terminal knot, which ends no interval. Same
    arithmetic here or the shipped solver is one interval too long.
    """
    knots = int(parameters["horizon_length"])
    if knots < 2:
        raise ValueError(
            f"horizon_length is {knots}; a horizon needs at least two knots"
        )
    return knots - 1


def nominal_progress_rate(parameters: dict) -> float:
    """
    Return the plan's own pace, in path parameter per second, over one window.

    `s` spans the window the path was fitted to, so running the plan at the
    speed it was written at is one window per `intervals * Ts` seconds. A caller
    whose path spans something else -- an offline harness fitting a whole move
    -- has its own nominal and does not read this one.
    """
    return 1.0 / (shooting_intervals(parameters) * float(parameters["Ts"]))


def constraint_scale(model: cs.CraneSymbolicModel, hydraulics: dict) -> tuple:
    """
    Return `(scale, extend, retract)`: each `h` row's divisor and what set it.

    The chamber pair rides along because `solver.constraint_data` needs the same
    two numbers to bound the rows these divisors condition, and evaluating
    `chamber_force` a second time there only risks the two disagreeing.

    Force row: divided by the larger of its two chamber forces at relief
    pressure, so the wider box end lands at one (dividing by the smaller
    would put the arm row at 2.5).

    Pump row: divided by `0.95 Q_P^max`, the planning factor on pump flow
    max -- not a second kappa, which is reserved for the MPC's own
    correction margin.
    """
    extend, retract = chamber_forces(
        model, hydraulics["system_pressure_pa"], cs.K_ACTUATED_DOF
    )

    scale = np.zeros(cs.NU + 1)
    for axis in cs.K_PLANNED_AXES:
        scale[axis] = max(abs(extend[axis]), abs(retract[axis]))
    scale[cs.NU] = float(hydraulics["pump_flow_planning_factor"]) * float(
        hydraulics["pump_flow_max"]
    )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(
            "every row of h needs a finite positive divisor; a zero one is a "
            "constraint 6 or 7 with no right-hand side"
        )
    return scale, extend, retract


# ------------------------------------------------------------------- assembly


def build_ocp(description_xml: str, parameters: dict, hydraulics: dict) -> tuple:
    """
    Assemble the OCP for the description, and return it with its scales.

    Every expression is a slice of `crane_symbolic`'s graph or an arithmetic
    combination of two; no equation of motion, transmission ratio or
    smoothing width is written here.
    """
    # C3 on every planned axis: PT1 command lag, force state, forward
    # dynamics, with fitted `k`/`tau_v` from file. `u` is joint velocity
    # (rad/s) at Psi's input, not acceleration.
    model = cs.CraneSymbolicModel(
        description_xml, TOOL, actuator=cs.load_actuator_fit()
    )
    scale, extend, retract = constraint_scale(model, hydraulics)

    # --- `p`: the pinned tool coordinate and the payload body -----------------
    #
    # Eleven parameters, bound whole, nothing substituted out (issue 072).
    # `crane_symbolic` carries the payload as a symbol (mass, COM, six
    # `Theta_L` entries) so it can be a runtime acados parameter.
    #
    # Tool coordinate: not planned (gripper driven by low-level controller),
    # but its inertia is in `M(q)`, so it's evaluated at the incoming state's
    # gripper value each cycle.
    # Payload: set via `crane_msgs/SetPayload` at grasp/release;
    # `ocp_solver.cpp` treats a change as discrete and drops the warm start.
    #
    # `ocp_solver.cpp` writes both every stage; export ships zero default
    # (empty gripper).
    xdot = model.xdot
    tau_a = model.tau_a
    output = model.z

    x = model.x
    u = model.u

    q_a = x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
    q_u = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    dq_a = x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF]
    dq_u = x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF]

    # Progress pair: `s` virtual time, `s_nom` what it'd be unslipped, `ds`
    # how far the plan moved off clock.
    progress = x[cs.X_PROGRESS]
    progress_rate = x[cs.X_PROGRESS_RATE]
    reference_parameters = ca.SX.sym("p_ref", NP - cs.NP)
    parameter_vector = ca.vertcat(model.p, reference_parameters)

    def block(offset):
        return parameter_vector[offset : offset + cs.K_PLANNED_DOF]

    # Where on the path this stage is. `s` is pinned to zero at stage zero, so
    # the parameter carries where the cycle starts and `s` carries the rest.
    control = ca.reshape(
        parameter_vector[P_PATH_CONTROL : P_PATH_CONTROL + PATH_POINTS * cs.NU],
        cs.K_PLANNED_DOF,
        PATH_POINTS,
    ).T
    theta = parameter_vector[P_PATH_ORIGIN] + progress
    reference_position = bspline.casadi_value(theta, control, PATH_KNOTS).T
    # The path's tangent, which is the velocity a machine travelling the path at
    # `v_theta` would have. robocrane prices `dq_a` toward rest instead and lets
    # the pull on `theta` buy the motion; on a KUKA under acceleration control
    # that works, here it fights C3's force state and the dead time -- measured,
    # 460 mm off path and 29 of 178 solves refused. So the row stays a tangent
    # and the formulation's freedom lives in `theta`, which is where it matters.
    tangent = bspline.casadi_slope(theta, control, PATH_KNOTS).T

    acados_model = AcadosModel()
    acados_model.name = f"{SOLVER_PREFIX}_{TOOL}"
    acados_model.x = x
    acados_model.u = u
    acados_model.p = parameter_vector
    acados_model.f_expl_expr = xdot
    # IRK reads `f_impl_expr` (errors if empty), ERK reads `f_expl_expr`; both
    # set so the integrator stays a solver option, not a re-model.
    acados_model.xdot = ca.SX.sym("xdot", cs.NX)
    acados_model.f_impl_expr = acados_model.xdot - xdot

    # --- the cost, as a nonlinear least-squares residual -----------------------
    #
    # Row order: tracking, sway, effort, smoothness. `yref` carries `q_a_ref`,
    # `dq_a_ref`, `q_eq`; every other row is zero.
    #
    # Effort rows are `tau_a`, not `u`: pricing acceleration carries the
    # effective inertia, so the optimizer is gentler extended/loaded -- why
    # inverse dynamics is in the loop.
    #
    # Four actuated blocks are planned rows only: no tool row (follows the
    # low-level controller, optimizer can't move it).
    #
    # Tracking rows carry their own reference; `yref` is zero there. `q_a,ref`
    # is a function of `s`, so it can't live in `yref` (acados subtracts it
    # as a constant) -- this is what makes `s` a decision variable.
    tracking = q_a - reference_position
    # Both tool positions are taken at *this* stage's passive pair, so the row
    # is the planned axes' own contribution and does not double-count the sway
    # the two rows below already price. The reference is the path point, not the
    # goal: this is the task-space reading of the tracking row above it, not a
    # second terminal condition.
    canonical = ca.SX.zeros(cs.K_GENERALIZED_DOF)
    canonical[list(cs.K_PLANNED_ROWS)] = q_a
    canonical[list(cs.K_PASSIVE_ROWS)] = q_u
    canonical[cs.K_ACTUATED_ROWS[cs.K_TOOL_AXIS]] = model.p[cs.P_TOOL_POSITION]
    at_machine = model.tool_position(canonical)
    canonical[list(cs.K_PLANNED_ROWS)] = reference_position
    at_path = model.tool_position(canonical)
    tool = at_machine - at_path
    #
    # Lag row (`timber_crane_cost_js_pfc_pt2.cpp:62-74`): tracking error
    # projected on reference direction of travel, slewing axis only,
    # unnormalised -- weight absorbs scale.
    lag = tracking[K_LAG_AXIS] * tangent[K_LAG_AXIS]
    #
    # Progress row: quadratic regulator toward one (time-scaling); `yref`
    # carries the one, row itself is `v_s`.
    residual = ca.vertcat(
        tracking,
        dq_a - tangent * progress_rate,
        q_u,
        dq_u,
        lag,
        tool,
        theta,
        progress_rate,
        tau_a[: cs.K_PLANNED_DOF],
        u,
    )
    terminal_residual = residual[:NY_TERMINAL]
    acados_model.cost_y_expr_0 = residual
    acados_model.cost_y_expr = residual
    # Terminal condition is a raised-weight cost, never a terminal set: a
    # hard set is a reliable infeasibility source under single-iteration SQP.
    acados_model.cost_y_expr_e = terminal_residual

    # --- force and flow constraints, out of the same output map ---------------
    #
    # `F_cyl,i = tau_a,i / J_c,ii(q_i)` and the per-axis `Q` sum, both read
    # out of `crane_symbolic`'s output map. Smoothing (`A±(v) sqrt(v² + eps²)`,
    # tanh width `eps_v`) is already inside `Q`; not applied again.
    #
    # Planned axes only: the tool is held still, so no plan here moves that
    # cylinder or draws supply through it.
    cylinder_force = output[
        cs.K_CYLINDER_FORCE_OFFSET : cs.K_CYLINDER_FORCE_OFFSET + cs.K_ACTUATED_DOF
    ]
    axis_flow = output[
        cs.K_AXIS_FLOW_OFFSET : cs.K_AXIS_FLOW_OFFSET + cs.K_ACTUATED_DOF
    ]
    rows = [cylinder_force[axis] / scale[axis] for axis in cs.K_PLANNED_AXES]
    rows.append(ca.sum1(axis_flow[: cs.K_PLANNED_DOF]) / scale[cs.NU])
    constraint = ca.vertcat(*rows)
    acados_model.con_h_expr_0 = constraint
    acados_model.con_h_expr = constraint
    # Terminal stage has no input, so `tau_a` is undefined and neither
    # nonlinear row exists there.

    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.parameter_values = np.zeros(NP)

    # `horizon_length` counts knots, not intervals: `mpc_node.cpp` passes
    # `horizon_length - 1` (last knot ends no interval); fifty knots is
    # forty-nine.
    horizon = shooting_intervals(parameters)
    step = float(parameters["Ts"])
    ocp.solver_options.N_horizon = horizon
    ocp.solver_options.tf = horizon * step

    nx = cs.NX
    nu = cs.NU_PROGRESS
    nh = cs.NU + 1

    # --- the cost data ---------------------------------------------------------
    #
    # Placeholders, overwritten by `ocp_solver.cpp` from deployment
    # parameters before the first solve; written here only for shape.
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ny = residual.shape[0]
    ny_e = terminal_residual.shape[0]
    if ny != NY or ny_e != NY_TERMINAL:
        raise ValueError(
            f"the residual is {ny} rows against the offsets' {NY} and the terminal "
            f"{ny_e} against {NY_TERMINAL}; the offsets are what the C++ reads"
        )
    ocp.cost.W_0 = np.eye(ny)
    ocp.cost.W = np.eye(ny)
    ocp.cost.W_e = np.eye(ny_e)
    ocp.cost.yref_0 = np.zeros(ny)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # --- constraints 1 to 5, as the boxes acados takes natively ---------------
    #
    # `x0` pins stage zero; `idxbx` boxes rows up to the force state at later
    # stages; `idxbu` the six inputs (five joints, progress acceleration).
    # Values arrive at runtime: limits are ROS parameters, sway box travels
    # with each stage's equilibrium.
    ocp.constraints.x0 = np.zeros(nx)
    # Force states not boxed: `|tau_a,i| <= J_c,ii(q) F_i^max` has `J_c,ii`
    # changing sign across the arm's range, so a constant box is wrong. The
    # nonlinear row stays, reduced to `x_j / J_c,ii(q)`.
    #
    # Boxed rows: rigid-body state plus lagged command, a contiguous prefix
    # of `x`, so `idxbx`/`idxsbx` share indices.
    nbx = cs.NBX
    ocp.constraints.idxbx = np.arange(nbx)
    ocp.constraints.lbx = -np.ones(nbx)
    ocp.constraints.ubx = np.ones(nbx)
    ocp.constraints.idxbx_e = np.arange(nbx)
    ocp.constraints.lbx_e = -np.ones(nbx)
    ocp.constraints.ubx_e = np.ones(nbx)
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.lbu = -np.ones(nu)
    ocp.constraints.ubu = np.ones(nu)

    # Every `h` row is a fraction of its allowance: force rows in `[-1, 1]`,
    # flow row one-sided (`Q` is a pump draw, magnitudes summed, never
    # negative).
    ocp.constraints.lh_0 = np.concatenate([-np.ones(cs.NU), [0.0]])
    ocp.constraints.uh_0 = np.ones(nh)
    ocp.constraints.lh = np.concatenate([-np.ones(cs.NU), [0.0]])
    ocp.constraints.uh = np.ones(nh)

    # --- softening, as dimensions ----------------------------------------------
    #
    # Three stage kinds:
    #   * stage 0 softens nonlinear rows only -- its box is `x_0`, a slack
    #     there is a plan for a state the machine isn't in. Flow constraint 7
    #     depends on `dq_a` alone, fully determined at stage 0, so unsoftened
    #     it's infeasible whenever the machine is already over the pump limit;
    #   * running stages soften 3, 4, 6, 7, leave 1, 2, 5 hard (command
    #     bounds, always achievable);
    #   * terminal has no input, so 6/7 absent; 3/4 stay boxed and soft.
    soft_state = np.array(
        [
            cs.X_PASSIVE_POSITION,
            cs.X_PASSIVE_POSITION + 1,
            cs.X_PASSIVE_VELOCITY,
            cs.X_PASSIVE_VELOCITY + 1,
        ]
    )
    ocp.constraints.idxsbx = soft_state
    ocp.constraints.idxsbx_e = soft_state
    ocp.constraints.idxsh_0 = np.arange(nh)
    ocp.constraints.idxsh = np.arange(nh)

    n_soft_path = soft_state.size + nh
    # `Zl`/`Zu` (quadratic weight) stay zero: L1 is exact, quadratic leaks
    # violation near the boundary. `zl`/`zu` are the linear prices,
    # runtime-set.
    ocp.cost.Zl_0 = np.zeros(nh)
    ocp.cost.Zu_0 = np.zeros(nh)
    ocp.cost.zl_0 = np.ones(nh)
    ocp.cost.zu_0 = np.ones(nh)
    ocp.cost.Zl = np.zeros(n_soft_path)
    ocp.cost.Zu = np.zeros(n_soft_path)
    ocp.cost.zl = np.ones(n_soft_path)
    ocp.cost.zu = np.ones(n_soft_path)
    ocp.cost.Zl_e = np.zeros(soft_state.size)
    ocp.cost.Zu_e = np.zeros(soft_state.size)
    ocp.cost.zl_e = np.ones(soft_state.size)
    ocp.cost.zu_e = np.ones(soft_state.size)

    # --- grid and backend -------------------------------------------------------
    #
    # Applied last, so a swept setting beats anything set above it; `None`
    # leaves acados' own default (`qp_solver_cond_N` has no other spelling
    # for "the horizon").
    ocp.solver_options.levenberg_marquardt = float(parameters["levenberg_marquardt"])
    for name, value in solver_tuning().items():
        if value is not None:
            setattr(ocp.solver_options, name, value)

    return ocp, scale, model, (extend, retract)
