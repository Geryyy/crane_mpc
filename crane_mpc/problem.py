"""
The optimal control problem of `wiki/mpc.md` §1, as an `AcadosOcp`.

Installed rather than left in `scripts/`, because the node builds this problem at
startup and `scripts/` is not on any installed path. `scripts/export_ocp.py` is
the same problem written out as generated C; it imports from here, so the solver
the node loads and the tree that is checked in are one definition.

The algebra is **imported and never restated**: `crane_model.symbolic` carries
the dynamics, the transmission and the output map. What is added here is the
four things that make those an OCP -- §1's shooting problem, §2's residual, §3's
seven constraints and §4's grid.

Baked at code generation: the structure of every constraint, the row order of
`h`, each `h` row's conditioning divisor, and the dynamics. **Not baked**: the
payload, which rides in `p` (issue 072), and every weight, bound and price,
which `solver.py` writes onto the loaded solver at configure time.
"""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from crane_model import symbolic as cs


def chamber_forces(model, relief_pa: float, actuated_dof: int) -> tuple:
    """
    `(extend, retract)`: what each cylinder carries at the relief pressure.

    `wiki/hydraulics.md` §4's `F_i = A_A p_A - A_B p_B`, asked of the shared
    model rather than restated. `crane_ocp_export.chamber_forces` is the same
    six lines for the exporters, which cannot import this package.
    """
    pressure = np.full(actuated_dof, float(relief_pa))
    zero = np.zeros(actuated_dof)
    extend = np.array(ca.evalf(model.chamber_force(pressure, zero))).ravel()
    retract = np.array(ca.evalf(model.chamber_force(zero, pressure))).ravel()
    return extend, retract


TOOL = "pzs100"
DESCRIPTION = "pzs100.urdf"


def default_description() -> Path:
    """
    Locate the description the solver's dynamics are baked from.

    **Not `/robot_description`.** What the node parses off the graph decides the
    kinematics it draws a TCP horizon with; what is baked here decides the plant
    the solver was compiled for, and that is one file so that one compiled
    solver serves every deployment. `crane_model` installs it beside its config.
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


# The prefix every generated symbol carries. acados derives it from the model
# name, so `crane_mpc_pzs100_acados_create` and its neighbours are what the C++
# calls; two solvers in one library therefore cannot collide.
SOLVER_PREFIX = "crane_mpc"


# --- the OCP's own parameters, appended to the model's eleven ------------------
#
# `crane_symbolic`'s `p` is the pinned tool coordinate and the payload body, and
# it stays exactly that -- `crane_planning` builds the same module and a wider
# model parameter vector would reach it. What is appended here belongs to *this*
# problem: the local model of the reference each stage is tracking.
#
# The progress state `s` is virtual time, so the reference the cost compares
# against is `q_a,ref(s)` and not `q_a,ref(k T_s)`. A spline cannot be baked into
# a generated solver, so each stage carries a **second-order expansion of its own
# reference about its nominal virtual time** -- the value, the first derivative
# and the second, exactly the three quantities Marc's cost reads off the spline
# at `s` (`timber_crane_cost_js_pfc_pt2.cpp:35-37`). Inside one stage's expansion
# the residual is then an ordinary expression of `x` and `p`, and both derivatives
# reach the Gauss-Newton gradient and Hessian through it.
#
# At `s = s_nom` and `v_s = 1` every residual below collapses to today's
# time-indexed one, which is what "nominal behaviour is exactly time-indexed
# tracking" means and is the property the offsets exist to keep checkable.
P_PROGRESS_NOMINAL = cs.NP
P_REFERENCE_POSITION = P_PROGRESS_NOMINAL + 1
P_REFERENCE_FIRST = P_REFERENCE_POSITION + cs.K_PLANNED_DOF
P_REFERENCE_SECOND = P_REFERENCE_FIRST + cs.K_PLANNED_DOF
NP = P_REFERENCE_SECOND + cs.K_PLANNED_DOF

# --- the residual blocks, `wiki/mpc.md` 2 -------------------------------------
#
# The terminal residual is the leading prefix of the stage one, so one set of
# offsets addresses both. The lag and progress rows are inside that prefix and
# the effort and smoothness rows -- which need an input -- are after it.
Y_PLANNED_POSITION = 0
Y_PLANNED_VELOCITY = cs.K_PLANNED_DOF
Y_PASSIVE_POSITION = 2 * cs.K_PLANNED_DOF
Y_PASSIVE_VELOCITY = Y_PASSIVE_POSITION + cs.K_PASSIVE_DOF
Y_LAG = Y_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF
Y_PROGRESS_RATE = Y_LAG + 1
NY_TERMINAL = Y_PROGRESS_RATE + 1
Y_ACTUATED_FORCE = NY_TERMINAL
Y_INPUT = Y_ACTUATED_FORCE + cs.K_PLANNED_DOF
NY = Y_INPUT + cs.NU_PROGRESS

# The axis the lag row is written on. **Marc's is the slewing joint alone**
# (`timber_crane_cost_js_pfc_pt2.cpp:62-74`) where a textbook lag error sums the
# tangential projection over every axis, and it is reproduced rather than
# generalised. The reason is that there is no tangent to project onto: the five
# planned coordinates carry four radians and one metre, so the unit tangent a
# single scalar projection needs is a norm over mixed units and nothing in the
# wiki says how many radians of slew a metre of telescope is worth. The
# per-axis alternative is not the generalisation either -- `e_i * q_ref,i'(s)`
# is the tracking error times the local reference speed, i.e. a speed-dependent
# reweighting of a term that already exists per axis, so it would add a knob and
# not a mechanism. Slewing is meanwhile the axis whose lag the progress variable
# is there to buy time for: it is the slowest, it is the one that drives the
# pendulum laterally, and it is the one Marc kept.
K_LAG_AXIS = 0

# `v_s = 1` is one second of plan per second of wall clock. It is a **constant**
# and not a parameter: any other value is a reference that is not the plan, and
# `wiki/trajectory_planning.md` 5.3's "a reference the MPC would reject is a
# planner bug" is written about the plan as issued. Marc declares `sDot_ref` and
# every deployed config sets it to 1.0.
K_PROGRESS_RATE_REFERENCE = 1.0


# ------------------------------------------------------------------ the numbers


def shooting_intervals(parameters: dict) -> int:
    """
    Return `N`, the shooting-interval count, from the yaml's knot count.

    `horizon_length` counts knots -- fifty of them is `wiki/mpc.md` §4's two
    seconds of plan -- and `mpc_node.cpp:214-219` turns that into the `N` the OCP
    is posed on by subtracting the terminal knot, which ends no interval. The
    same arithmetic has to happen here or the shipped solver is one interval
    longer than every caller asks for.
    """
    knots = int(parameters["horizon_length"])
    if knots < 2:
        raise ValueError(
            f"horizon_length is {knots}; a horizon needs at least two knots"
        )
    return knots - 1


def constraint_scale(model: cs.CraneSymbolicModel, hydraulics: dict) -> np.ndarray:
    """
    Return the divisor of each row of `h`, in that row's own physical unit.

    A force row is divided by the **larger** of its two chamber forces at the
    relief pressure, so the wider end of the two-sided box lands exactly on one
    and the narrower end at that axis' area ratio. Dividing by the smaller
    instead would put the arm row at 2.5, which is the spread the normalisation
    exists to remove.

    The pump row is divided by `0.95 Q_P^max`, which is `wiki/implementation/
    parameters.md` §4's planning factor on a number that page states once. It is
    not a second kappa: `wiki/trajectory_planning.md` §5.5 reserves kappa so the
    MPC has authority left to correct with, and this is that consumer.
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
    return scale


# ------------------------------------------------------------------- assembly


def build_ocp(description_xml: str, parameters: dict, hydraulics: dict) -> tuple:
    """
    Assemble `wiki/mpc.md` §1--§4 for the description, and return it with its scales.

    Every expression below is a slice of `crane_symbolic`'s graph or an
    arithmetic combination of two of them. No equation of motion, no
    transmission ratio and no smoothing width is written here.
    """
    # C3 in full on every planned axis (`hydraulic_actuator_model.md` §1): the
    # PT1 command lag, the force state and forward dynamics, with the fitted
    # `k` and `tau_v` read out of the file rather than hand-copied. `u` is a
    # joint velocity in rad/s at Psi's input from here on, not an acceleration.
    model = cs.CraneSymbolicModel(
        description_xml, TOOL, actuator=cs.load_actuator_fit()
    )
    scale = constraint_scale(model, hydraulics)

    # --- `p`: the pinned tool coordinate and the payload body -----------------
    #
    # The module's eleven parameters, bound whole and **nothing substituted
    # out**. `crane_symbolic` carries the payload as a symbol -- mass, centre of
    # mass and the six independent entries of `Theta_L` -- precisely so that an
    # OCP can make it a runtime acados parameter without rebuilding the model,
    # and this is that binding (issue 072).
    #
    # The two halves are runtime for different reasons:
    #
    #   * the **tool coordinate** is not planned. The gripper is opened and
    #     closed by the low-level velocity/position controller, so this problem
    #     does not plan it -- but the tool link's inertia is still in `M(q)`, so
    #     the model has to be evaluated somewhere on that axis, and it is
    #     evaluated where the incoming state says the gripper is. It changes
    #     every cycle;
    #   * the **payload** is not this node's to measure. A block is on the
    #     gripper because the behaviour tree grasped it, so the tree is what says
    #     so, through `crane_msgs/SetPayload`. It changes at a grasp and at a
    #     release and at no other time -- which is why `ocp_solver.cpp` treats a
    #     change as a discrete model change and drops the warm start built for
    #     the old one.
    #
    # `ocp_solver.cpp` writes both onto every stage before each solve, in the
    # order `write_header` records. Nothing here decides what the payload *is*;
    # the export ships zero as the artifact's default parameter value, which is
    # an empty gripper.
    xdot = model.xdot
    tau_a = model.tau_a
    output = model.z

    x = model.x
    u = model.u

    q_a = x[cs.X_PLANNED_POSITION : cs.X_PLANNED_POSITION + cs.K_PLANNED_DOF]
    q_u = x[cs.X_PASSIVE_POSITION : cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF]
    dq_a = x[cs.X_PLANNED_VELOCITY : cs.X_PLANNED_VELOCITY + cs.K_PLANNED_DOF]
    dq_u = x[cs.X_PASSIVE_VELOCITY : cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF]

    # The progress pair and the local reference model. `s` is virtual time and
    # `s_nom` is the virtual time this stage would be at if nothing had slipped,
    # so `ds` is how far the optimizer has moved the plan away from the clock.
    progress = x[cs.X_PROGRESS]
    progress_rate = x[cs.X_PROGRESS_RATE]
    reference_parameters = ca.SX.sym("p_ref", NP - cs.NP)
    parameter_vector = ca.vertcat(model.p, reference_parameters)

    def block(offset):
        return parameter_vector[offset : offset + cs.K_PLANNED_DOF]

    ds = progress - parameter_vector[P_PROGRESS_NOMINAL]
    reference_position = (
        block(P_REFERENCE_POSITION)
        + block(P_REFERENCE_FIRST) * ds
        + 0.5 * block(P_REFERENCE_SECOND) * ds * ds
    )
    # d/ds of the line above, which is what the chain rule needs: the reference
    # velocity the machine should hold is `q_ref'(s) * v_s`, so a horizon that
    # spends the plan more slowly asks for a proportionally slower axis. Tracking
    # `q_ref'(s_nom)` regardless would have the velocity term fighting the
    # progress term, which is the defect that makes a progress state decoration.
    reference_velocity = block(P_REFERENCE_FIRST) + block(P_REFERENCE_SECOND) * ds

    acados_model = AcadosModel()
    acados_model.name = f"{SOLVER_PREFIX}_{TOOL}"
    acados_model.x = x
    acados_model.u = u
    acados_model.p = parameter_vector
    acados_model.f_expl_expr = xdot
    # IRK reads `f_impl_expr` and errors on an empty one, ERK reads `f_expl_expr`;
    # both are set so the integrator stays a solver option rather than a re-model,
    # which is what `crane_planning`'s exporter does for the same reason.
    acados_model.xdot = ca.SX.sym("xdot", cs.NX)
    acados_model.f_impl_expr = acados_model.xdot - xdot

    # --- §2's cost, as a nonlinear least-squares residual ---------------------
    #
    # Row order is the page's: tracking, sway, effort, smoothness. `yref` carries
    # `q_a_ref`, `dq_a_ref` and `q_eq`; every other reference row is zero, which
    # is the difference between "track this" and "penalise this".
    #
    # **The effort rows are `tau_a` and not `u`.** That is the first of §2's
    # three deliberate properties: the price of an acceleration then carries the
    # effective inertia, so the optimizer is automatically gentler with the arm
    # extended or loaded, and a plain input penalty cannot express that. It is
    # the entire reason inverse dynamics is in the loop.
    #
    # The four actuated blocks are the **planned** rows: there is no tool row to
    # track, because the tool follows the low-level controller and not this
    # reference, and none to price, because the optimizer cannot move it.
    #
    # **The tracking rows carry their own reference and `yref` is zero on them.**
    # `q_a,ref` is a function of the decision variable `s`, so it cannot live in
    # `yref`, which acados subtracts as a constant. Moving it into the residual
    # is what makes `s` a decision variable rather than decoration, and it leaves
    # exactly one home for the reference instead of a copy in `yref` and a copy
    # in `p`.
    tracking = q_a - reference_position
    #
    # The lag row, `timber_crane_cost_js_pfc_pt2.cpp:62-74`: the tracking error
    # projected on the reference's own direction of travel, on the slewing axis.
    # Unnormalised, as the original is -- the weight absorbs the scale, and a
    # unit tangent would need the mixed-unit norm `K_LAG_AXIS` exists to avoid.
    lag = tracking[K_LAG_AXIS] * reference_velocity[K_LAG_AXIS]
    #
    # The progress row is a **quadratic regulator toward one**, not a linear
    # progress reward: this is time-scaling. `yref` carries the one, so the row
    # itself is just `v_s`.
    residual = ca.vertcat(
        tracking,
        dq_a - reference_velocity * progress_rate,
        q_u,
        dq_u,
        lag,
        progress_rate,
        tau_a[: cs.K_PLANNED_DOF],
        u,
    )
    terminal_residual = residual[:NY_TERMINAL]
    acados_model.cost_y_expr_0 = residual
    acados_model.cost_y_expr = residual
    # §2's third deliberate property: the terminal condition is a raised-weight
    # **cost** and never a terminal set. A hard terminal set is a dependable
    # source of infeasibility under a single-iteration scheme, and an infeasible
    # solve is worse than a slightly unsettled tool.
    acados_model.cost_y_expr_e = terminal_residual

    # --- §3's constraints 6 and 7, out of the same output map -----------------
    #
    # `F_cyl,i = tau_a,i / J_c,ii(q_i)` is constraint 6's left-hand side and the
    # per-axis `Q` sums to constraint 7's, both read out of `crane_symbolic`'s
    # output map rather than rebuilt. §3.1's smoothing -- `A±(v) sqrt(v² + eps²)`
    # with a `tanh` of width `eps_v` -- is already inside `Q`, and **nothing here
    # applies it a second time**. `crane_planning`'s timing OCP takes the same
    # two slices of the same map, which is what makes `wiki/trajectory_planning.md`
    # §5.3's "a reference the MPC would reject is a planner bug" true by
    # construction rather than by intention.
    #
    # Both run over the **planned** axes alone. The tool's rows are not dropped
    # because they are small; they are dropped because the tool is held still, so
    # no plan this solver writes can move that cylinder or draw supply through it.
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
    # The terminal stage has no input, so `tau_a` is undefined there and neither
    # nonlinear row exists. `crane_planning` drops its own at the same node for
    # the same reason.

    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.parameter_values = np.zeros(NP)

    # `horizon_length` counts **knots**, not shooting intervals: `mpc_node.cpp`
    # reads the same key and passes `horizon_length - 1` to the OCP, because the
    # published horizon carries a point per knot and the last knot ends no
    # interval. acados' `N_horizon` is the interval count, so the yaml's fifty
    # knots are forty-nine of them.
    horizon = shooting_intervals(parameters)
    step = float(parameters["Ts"])
    ocp.solver_options.N_horizon = horizon
    ocp.solver_options.tf = horizon * step

    nx = cs.NX
    nu = cs.NU_PROGRESS
    nh = cs.NU + 1

    # --- the cost data ---------------------------------------------------------
    #
    # Placeholders. Every one of these is set again by `ocp_solver.cpp` from the
    # deployment's own parameters before the first solve, which is grill D5's
    # "weights and constraint bounds remain runtime-settable on a generated
    # solver". They are written here only because acados wants a shape.
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

    # --- §3's constraints 1 to 5, as the boxes acados takes natively ----------
    #
    # `x0` is 1's initial condition and pins every row at stage zero; `idxbx`
    # boxes the rows up to the force state at every later stage, and `idxbu` the
    # six inputs -- five joint commands and the progress acceleration. The *values* arrive at runtime, because the control-safe limits of
    # `wiki/implementation/parameters.md` §2 are ROS parameters and the sway box
    # of constraint 3 travels with each stage's own equilibrium.
    ocp.constraints.x0 = np.zeros(nx)
    # The **force states are not boxed.** Constraint 6 is
    # `|tau_a,i| <= J_c,ii(q) F_i^max`, and `J_c,ii` is not a constant on the boom
    # and changes sign across the arm's control-safe range, so no constant box on
    # `tau_a` is that constraint -- it is either slack or wrong. The nonlinear row
    # stays and is what bounds the force state; what C3 buys there is that the row
    # is now `x_j / J_c,ii(q)` instead of a whole inverse dynamics.
    #
    # The boxed rows are therefore the rigid-body state plus the lagged command,
    # which is a **contiguous prefix** of `x`, so acados' `idxbx` positions and the
    # state rows still coincide and `idxsbx` below indexes the same numbers.
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

    # Every `h` row is a fraction of its own allowance, so these bounds carry no
    # machine number: the force rows land inside `[-1, 1]` and the flow row is
    # one-sided, because `Q` is a pump *draw* and not a signed force --
    # `wiki/hydraulics.md` §3 sums magnitudes and a negative total draw is not a
    # state the machine has.
    ocp.constraints.lh_0 = np.concatenate([-np.ones(cs.NU), [0.0]])
    ocp.constraints.uh_0 = np.ones(nh)
    ocp.constraints.lh = np.concatenate([-np.ones(cs.NU), [0.0]])
    ocp.constraints.uh = np.ones(nh)

    # --- §3.2's softening, as dimensions --------------------------------------
    #
    # Three kinds of stage and each has its own reason for the slacks it does and
    # does not carry:
    #
    #   * **stage 0 softens the nonlinear rows and nothing else.** Its box is
    #     `x_0` itself, and a slack on that is a plan for a state the machine is
    #     not in. The nonlinear rows there must be soft all the same, and
    #     constraint 7 is the sharpest case: `Q` depends on `dq_a` alone, so at
    #     stage 0 it is fully determined by the pinned measurement and no input
    #     can relieve it. A machine already drawing over the pump limit would
    #     otherwise have no solvable problem at all -- which is precisely §3.2's
    #     "a measurement transient can render the problem infeasible for a state
    #     the machine is already in";
    #   * **the running stages soften 3, 4, 6 and 7** and leave 1, 2 and 5 hard:
    #     those three bound the command, which is always achievable;
    #   * **the terminal stage has no input**, so constraints 6 and 7 are absent
    #     there. Constraints 3 and 4 are still boxes and are still soft.
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
    # `Zl`/`Zu` are the **quadratic** slack weights and stay zero: §3.2 asks for
    # an `L1` penalty, and a quadratic price is cheap near the boundary, so it
    # leaks a little violation everywhere. A linear one with a large enough
    # coefficient is exact -- zero violation wherever the constraint can be met
    # at all. `zl`/`zu` are the linear prices and are runtime-settable.
    # Stage 0 carries slacks only because of the nonlinear rows: its box is
    # `x_0` itself, and a slack on that is a plan for a state the machine is not
    # in.
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

    # --- §4's grid and §5.2's backend ----------------------------------------
    #
    # **RTI: exactly one iteration per cycle**, which is what makes the solve time
    # deterministic. For a hard-real-time loop that is worth more than the extra
    # accuracy of a converged SQP step, because the plant moves between cycles
    # anyway.
    #
    # **IRK, and it is C3 that made it necessary.** The comment that stood here
    # said ERK4 was sound only because the ideal inner loop left the model
    # non-stiff, and to revisit it if actuator lag came back. It came back: the
    # fastest eigenvalue of the linearised C3 plant is `|lambda| T_s = 8.5` at an
    # ordinary pose -- the telescope's `k = 3.5e6 N/m` against its effective mass
    # -- and ERK4 is stable only to about 2.8, so an explicit step diverges inside
    # three intervals and HPIPM answers status 3. Two Gauss stages are order four
    # and A-stable, so the step size stops being a stability question at all.
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # No `qp_solver_cond_N`: acados' default is `N`, and issue 129 measured every
    # smaller block size to be worse -- 10.3 ms of QP at `N`, 61-81 ms at 1. The
    # harness can still override it per run; nothing deployed should.
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 1
    # No line search: one full Newton step is what RTI *is*, and a search would
    # make the cost of a cycle depend on the problem. Nothing is regularised
    # adaptively either, so `wiki/mpc.md` §5.3 requirement 2 has nothing that
    # could latch -- the Levenberg-Marquardt term below is a constant acados adds
    # identically on every cycle.
    ocp.solver_options.globalization = "FIXED_STEP"
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    ocp.solver_options.levenberg_marquardt = float(parameters["levenberg_marquardt"])

    return ocp, scale, model
