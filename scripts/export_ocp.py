#!/usr/bin/env python3
"""
Write the optimal control problem of `wiki/mpc.md` §1 as a generated acados solver.

This is `docs/features/cbs-ocp-python/grill.md` D1 and D8 for `crane_mpc`: the
problem is *defined* here, in Python, over `crane_model/scripts/crane_symbolic.py`,
and *shipped* as generated C under `generated/`. Nothing in this package assembles
an OCP against the raw acados C API any more, and nothing hands acados a live
`casadi::Function`.

    ./scripts/export_ocp.py                # rewrite `generated/`
    ./scripts/export_ocp.py --check        # regenerate into a scratch tree and diff

The algebra is **imported and never restated**. `crane_symbolic` carries the
dynamics, the transmission and the output map; what is added here is the four
things that make those an OCP and that the model has no opinion about:
`wiki/mpc.md` §1's shooting problem, §2's nonlinear-least-squares residual, §3's
seven constraints with §3.1's smoothing already inside the output map, and §4's
grid. If this file ever needs an expression the module does not expose, the
expression belongs in the module.

## What is baked and what stays runtime-settable

Baked, because acados fixes them at code generation:

* the *structure* of every constraint: which rows are boxed, which are soft, and
  the row order of `h`;
* the conditioning divisor of each `h` row -- see `constraint_scale` below;
* the dynamics, and with them the description the solver was built from.

The **payload is not baked** (issue 072). It rides in `p` beside the pinned tool
coordinate -- mass, centre of mass and the six independent entries of `Theta_L`,
which is `crane_symbolic`'s own parameter vector bound whole -- and
`ocp_solver.cpp` writes it onto every stage before each solve. A block picked up
mid-run therefore changes the model without a reconfigure and without a
regeneration, which is what a behaviour tree calling `crane_msgs/SetPayload` at
grasp and release needs.

Runtime-settable on the generated solver, and set by `ocp_solver.cpp` at
configure time from the deployment's own parameters: `W`, `yref`, `lbx`/`ubx`,
`lbu`/`ubu`, `lh`/`uh`, the `L1` slack prices `zl`/`zu`, the Levenberg-Marquardt
term, **and `N` and `T_s`**. The last pair is the one grill D5 expected to become
regenerate-only and did not: acados generates
`<name>_acados_create_with_discretization(capsule, N, steps)` beside the fixed-`N`
entry point, so the grid is an argument and the offline tests keep the short
horizons they need. `N` and `T_s` are read from `config/crane_mpc.yaml` here only
to write the artifact's *default* into `crane_mpc_ocp_generated.h`, off the same
key the node reads.

## `constraint_scale`, and why it is a constant rather than a limit

Every row of `h` is divided by a fixed number before it reaches acados, because
written in physical units the rows span eight decades -- newtons near `1e5`
beside cubic metres per second near `1e-3` -- and HPIPM fails on the constraint
Jacobian that produces.

The divisor is **conditioning and not a bound**. It is baked, and the bound is
not: `|F_cyl,i| <= F_i^max` goes in as `lh_i = -F_i^max / scale_i`, so a
deployment that moves its relief pressure moves `lh`/`uh` and leaves the
generated expression alone. `ocp_solver.cpp` reads the divisors back out of
`generated/crane_mpc_ocp_generated.h` rather than deriving its own, because two
derivations of one number is exactly the drift the divisor being a constant
removes.

## There is no staleness guard, by decision

`config/hydraulics.yaml`, `config/hydraulic_limits.yaml` and the description
all require this script to be re-run and the workspace rebuilt. **Nothing fails
when one of them and the checked-in tree disagree.** grill §4 records that the
alternative -- hashing the inputs into the generated code and comparing in a test,
which is what `crane_model`'s own fixture does -- was considered and rejected, and
that this is therefore a deliberate divergence from both existing generator
scripts in this repository. `generated/README.md` says the same thing where a
reader of the tree will find it.

## What is boilerplate lives in `crane_ocp`

The scratch-JSON code generation, the pruning, the whitespace normalisation and
the `--check` byte-compare are `crane_ocp/scripts/crane_ocp_export.py` -- the same
module `crane_planning/scripts/export_timing_ocp.py` uses, because that half of
the two exporters was the same code twice. What is left here is the problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp

PACKAGE = Path(__file__).resolve().parent.parent

# Neither `crane_ocp`'s nor `crane_model`'s `scripts/` is installed (issue 070's
# notes), so both shared modules are imported by path out of the source tree.
sys.path.insert(0, str(PACKAGE.parent / "crane_ocp" / "scripts"))

import crane_ocp_export as ox  # noqa: E402

cs = ox.import_crane_symbolic(PACKAGE)

# The one machine this package ships a solver for, and the description its
# dynamics are baked from. The PZS100 is the machine that has to run. The
# Epsilon 7040 is still a real machine in `crane_model` -- description,
# hydraulics and collision model -- and what was retired here is only its
# *solver*, which nothing planned on.
TOOL = "pzs100"
DESCRIPTION = "pzs100.urdf"

# Where the description lives by default, so `scripts/mpc_a2b.py` can reach the
# same file without re-deriving the path.
DEFAULT_DESCRIPTIONS = ox.default_descriptions(PACKAGE)

# The prefix every generated symbol carries. acados derives it from the model
# name, so `crane_mpc_pzs100_acados_create` and its neighbours are what the C++
# calls; two solvers in one library therefore cannot collide.
SOLVER_PREFIX = "crane_mpc"

# The header this script writes beside the solver, carrying the numbers the
# C++ would otherwise have to derive a second time.
GENERATED_HEADER = "crane_mpc_ocp_generated.h"


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
    extend, retract = ox.chamber_forces(
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
    model = cs.CraneSymbolicModel(description_xml, TOOL)
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

    acados_model = AcadosModel()
    acados_model.name = f"{SOLVER_PREFIX}_{TOOL}"
    acados_model.x = x
    acados_model.u = u
    acados_model.p = model.p
    acados_model.f_expl_expr = xdot

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
    residual = ca.vertcat(q_a, dq_a, q_u, dq_u, tau_a[: cs.K_PLANNED_DOF], u)
    terminal_residual = ca.vertcat(q_a, dq_a, q_u, dq_u)
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
    ocp.parameter_values = np.zeros(cs.NP)

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
    nu = cs.NU
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
    ocp.cost.W_0 = np.eye(ny)
    ocp.cost.W = np.eye(ny)
    ocp.cost.W_e = np.eye(ny_e)
    ocp.cost.yref_0 = np.zeros(ny)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # --- §3's constraints 1 to 5, as the boxes acados takes natively ----------
    #
    # `x0` is §1's initial condition and pins all fourteen rows at stage zero;
    # `idxbx` boxes the same fourteen at every later stage, and `idxbu` the five
    # inputs. The *values* arrive at runtime, because the control-safe limits of
    # `wiki/implementation/parameters.md` §2 are ROS parameters and the sway box
    # of constraint 3 travels with each stage's own equilibrium.
    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.idxbx = np.arange(nx)
    ocp.constraints.lbx = -np.ones(nx)
    ocp.constraints.ubx = np.ones(nx)
    ocp.constraints.idxbx_e = np.arange(nx)
    ocp.constraints.lbx_e = -np.ones(nx)
    ocp.constraints.ubx_e = np.ones(nx)
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.lbu = -np.ones(nu)
    ocp.constraints.ubu = np.ones(nu)

    # Every `h` row is a fraction of its own allowance, so these bounds carry no
    # machine number: the force rows land inside `[-1, 1]` and the flow row is
    # one-sided, because `Q` is a pump *draw* and not a signed force --
    # `wiki/hydraulics.md` §3 sums magnitudes and a negative total draw is not a
    # state the machine has.
    ocp.constraints.lh_0 = np.concatenate([-np.ones(nu), [0.0]])
    ocp.constraints.uh_0 = np.ones(nh)
    ocp.constraints.lh = np.concatenate([-np.ones(nu), [0.0]])
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
    # anyway. ERK with four stages **is** ERK4, one step per shooting interval, so
    # the integrator step is `T_s`: with the ideal inner loop the model is not
    # stiff, so an explicit integrator is enough and cheaper than IRK, and grill
    # D1's whole argument is that changing this line is now one line.
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
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


# ------------------------------------------------------------------ generation


def write_header(
    output: Path, parameters: dict, scale: np.ndarray, constants: cs.Constants
) -> Path:
    """
    Write the numbers `ocp_solver.cpp` would otherwise derive a second time.

    The dimensions are acados' own, in `acados_solver_<name>.h`, and are not
    repeated here. What is here is the grid, the row offsets this script chose,
    and the conditioning divisors -- the three things the C++ cannot read off the
    solver and must not invent.
    """
    path = output / GENERATED_HEADER
    guard = "CRANE_MPC_OCP_GENERATED_H_"
    scales = ", ".join(f"{value!r}" for value in scale.tolist())
    lines = [
        "// Generated by `scripts/export_ocp.py`. Do not edit; re-run the script.",
        "//",
        "// The OCP of `wiki/mpc.md` §1 as it was written, in the numbers the C++",
        "// side must agree with rather than re-derive. Dimensions are acados' own,",
        "// in `acados_solver_crane_mpc_<tool>.h`, and are deliberately not repeated.",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "// §4's grid. Regenerate-only (grill D5): acados fixes both at code",
        "// generation, so a parameter file that disagrees is refused at configure",
        "// rather than silently solved on the shipped horizon. The horizon is the",
        "// **shooting-interval** count `OcpSettings::horizon_length` carries, one",
        "// less than the yaml's knot count.",
        f"#define CRANE_MPC_OCP_HORIZON {shooting_intervals(parameters)}",
        f"#define CRANE_MPC_OCP_SAMPLE_TIME_S {float(parameters['Ts'])!r}",
        "",
        "// `p`, which is `crane_symbolic`'s own parameter vector bound whole: the",
        "// pinned tool coordinate, then the payload body -- mass, centre of mass in",
        "// K8, and the six independent entries of Theta_L. acados' own",
        "// `<name>_acados_update_params` carries it; what it cannot say is the",
        "// **order**, so this is where `ocp_solver.cpp` reads the packing rather than",
        "// inventing a second one. Both halves are set on every stage before every",
        "// solve, so neither is a property of the artifact (issue 072).",
        f"#define CRANE_MPC_OCP_PARAMETER_DOF {cs.NP}",
        f"#define CRANE_MPC_OCP_PARAMETER_TOOL_POSITION {cs.P_TOOL_POSITION}",
        f"#define CRANE_MPC_OCP_PARAMETER_PAYLOAD_MASS {cs.P_PAYLOAD_MASS}",
        f"#define CRANE_MPC_OCP_PARAMETER_PAYLOAD_COM {cs.P_PAYLOAD_COM}",
        f"#define CRANE_MPC_OCP_PARAMETER_PAYLOAD_INERTIA {cs.P_PAYLOAD_INERTIA}",
        "",
        "// The (row, column) of each of those six entries, in the order they are",
        "// packed. Theta_L is symmetric and about the payload's own centre of mass",
        "// with the axes of K8, which is the URDF `<inertial>` convention.",
        "#define CRANE_MPC_OCP_PARAMETER_INERTIA_ENTRIES {"
        + ", ".join(f"{{{row}, {column}}}" for row, column in cs.INERTIA_ENTRIES)
        + "}",
        "",
        "// The blocks of the stage residual `y = [q_a, dq_a, q_u, dq_u, tau_a, u]`.",
        "// The order is §2's -- tracking, sway, effort, smoothness -- and not the",
        "// state's, and everything that has to agree with `W` reads it from here.",
        "#define CRANE_MPC_OCP_RESIDUAL_PLANNED_POSITION 0",
        f"#define CRANE_MPC_OCP_RESIDUAL_PLANNED_VELOCITY {cs.K_PLANNED_DOF}",
        f"#define CRANE_MPC_OCP_RESIDUAL_PASSIVE_POSITION {2 * cs.K_PLANNED_DOF}",
        "#define CRANE_MPC_OCP_RESIDUAL_PASSIVE_VELOCITY "
        f"{2 * cs.K_PLANNED_DOF + cs.K_PASSIVE_DOF}",
        "#define CRANE_MPC_OCP_RESIDUAL_ACTUATED_FORCE "
        f"{2 * cs.K_PLANNED_DOF + 2 * cs.K_PASSIVE_DOF}",
        "#define CRANE_MPC_OCP_RESIDUAL_INPUT "
        f"{3 * cs.K_PLANNED_DOF + 2 * cs.K_PASSIVE_DOF}",
        "",
        "// The rows of `h`: constraint 6 once per planned axis, then constraint 7.",
        "#define CRANE_MPC_OCP_CONSTRAINT_CYLINDER_FORCE 0",
        f"#define CRANE_MPC_OCP_CONSTRAINT_PUMP_FLOW {cs.NU}",
        "",
        "// What each row of `h` was divided by, in that row's own physical unit:",
        "// newtons on the five force rows and m^3/s on the pump row. Conditioning",
        "// and **not** a bound -- see `export_ocp.py`. `lh`/`uh` carry the limits.",
        f"#define CRANE_MPC_OCP_CONSTRAINT_SCALE {{{scales}}}",
        "",
        "// §3.1's two smoothing widths, as `config/hydraulics.yaml` states them and",
        "// as they went into `Q`. Here so that a test can assert the identity",
        "// `A±(v) sqrt(v² + eps²)` from outside without restating either number --",
        "// `crane_model`'s own `hydraulics::Constants` is private to that package.",
        f"#define CRANE_MPC_OCP_SMOOTHING_EPS_ABS {constants.eps_abs!r}",
        f"#define CRANE_MPC_OCP_SMOOTHING_EPS_V {constants.eps_v!r}",
        "",
        "// The soft rows of §3.2 among the state box, in acados' own slack order.",
        f"#define CRANE_MPC_OCP_SOFT_PASSIVE_POSITION {cs.X_PASSIVE_POSITION}",
        f"#define CRANE_MPC_OCP_SOFT_PASSIVE_VELOCITY {cs.X_PASSIVE_VELOCITY}",
        "",
        f"#endif  // {guard}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


README = """\
# `crane_mpc/generated/` -- the shipped solver

Machine output. **Do not edit any file here**: change
`scripts/export_ocp.py`, `crane_model/scripts/crane_symbolic.py` or one of the
config files below, re-run

    ./scripts/export_ocp.py

and commit what changes. `./scripts/export_ocp.py --check` regenerates into a
scratch tree and diffs, which is what says the tree still matches its inputs.

## What is in here

    crane_mpc_ocp_generated.h        the grid, the residual offsets and the
                                     conditioning divisors -- written by the
                                     export, read by `src/ocp_solver.cpp`
    crane_mpc_pzs100/                the PZS100 solver, constrained

**One solver.** The description is *baked in*, so a generated solver is one
machine's, and the PZS100 is the machine that has to run. The Epsilon 7040 is
still a real machine in `crane_model` -- description, hydraulics and collision
model -- and what was retired here is only its solver, which nothing planned on.

`docs/features/cbs-ocp-python/grill.md` D6's second artifact is retired too, and
the question it existed to settle is settled: dropping `wiki/mpc.md` §3's
nonlinear cylinder-force and pump-flow rows moves this OCP's solve from 19 QP
iterations and 7.15 ms to 20 and 7.02 ms. Those rows cost essentially nothing,
so there is no case for shipping a second artifact without them.

Inside the solver directory, `crane_mpc_pzs100_output.{c,h}` is not acados' --
it is the output map of `wiki/nomenclature.md` §10 code-generated beside the
solver, all six axes of `tau_a`, `F_cyl`, `v` and `Q`. acados generates only
what it solves, which is five force rows and one pump row already divided by
their conditioning constants; `wiki/mpc.md` §5.3 requirement 4 wants the
residuals in physical units, so they are shipped too.

acados' own generated `Makefile`, `main_*.c`, `acados_sim_solver_*` and
`acados_solver.pxd` are pruned by the export. The `Makefile` is the only
generated file that carries an absolute path, and the `main_*.c` carry a `main`.

## There is no staleness guard, and that is a decision

The conditioning divisors and the smoothing widths come from
`config/hydraulic_limits.yaml` and `crane_model/config/hydraulics.yaml`; the
dynamics come from `pzs100.urdf` under `crane_model/test/description/`.
**Nothing in the build or the test suite fails when one of those files and this
tree disagree.**

`docs/features/cbs-ocp-python/grill.md` §4 records the alternative that was
considered and rejected -- hashing the inputs into the generated code and
comparing in a test, which is the pattern `crane_model`'s own fixture and
`test_contract.cpp` use -- and records that this is therefore a deliberate
divergence from both existing generator scripts in this repository. The failure
mode it accepts is a solver silently running the previous hydraulic constants or
the previous description.

One thing narrows it, and it does not close it: **`config/crane_mpc.yaml` is not
in the exposed surface at all.** `N`, `T_s`, the weights, the box limits, the
slack prices and the Levenberg-Marquardt term are every one of them set on the
generated solver at configure time -- `N` and `T_s` through acados' own
`_acados_create_with_discretization`, which grill D5 did not expect -- so moving
any of them needs no re-export. `crane_mpc_ocp_generated.h` carries the grid the
export shipped as a *default* and not as a contract.

What is left, and unguarded, is the **structure**: the hydraulic constants that
went into `h`, the conditioning divisors, and the description the dynamics were
built from. Those three files are the staleness surface.
"""


def generate(
    output: Path, descriptions: Path, parameters: dict, hydraulics: dict
) -> None:
    """Write the whole tree, from an empty directory."""
    output.mkdir(parents=True, exist_ok=True)
    ocp, scale, model = build_ocp(
        (descriptions / DESCRIPTION).read_text(), parameters, hydraulics
    )
    tree = ox.generate_solver(ocp, output)
    # `crane_symbolic`'s `z` over `(x, u, p)`, shipped beside the solver because
    # acados generates only the rows it solves and `wiki/mpc.md` §5.3
    # requirement 4 wants the residuals in physical units.
    ox.write_output_map(model, ocp.model.name, tree)

    write_header(output, parameters, scale, cs.load_constants())
    ox.finalise(output, README)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ox.add_arguments(parser, PACKAGE)
    arguments = parser.parse_args()

    parameters = ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    hydraulics = ox.read_ros_parameters(
        PACKAGE / "config" / "hydraulic_limits.yaml", "crane_mpc"
    )["hydraulics"]

    return ox.run(
        arguments.output,
        arguments.check,
        lambda output: generate(output, arguments.descriptions, parameters, hydraulics),
        "export_ocp.py",
    )


if __name__ == "__main__":
    sys.exit(main())
