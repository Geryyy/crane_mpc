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

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent

# Neither `crane_ocp`'s nor `crane_model`'s `scripts/` is installed (issue 070's
# notes), so both shared modules are imported by path out of the source tree.
sys.path.insert(0, str(PACKAGE.parent / "crane_ocp" / "scripts"))

import crane_ocp_export as ox  # noqa: E402

cs = ox.import_crane_symbolic(PACKAGE)

# The problem itself is in the installed package, because the node builds it at
# startup and `scripts/` is on no installed path. A from-scratch build runs this
# script before `crane_mpc` is installed, so the source tree is the fallback --
# `ox.import_crane_symbolic` does the same for `cs`.
try:
    from crane_mpc import problem  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PACKAGE))

from crane_mpc.problem import (  # noqa: E402
    DESCRIPTION,
    K_LAG_AXIS,
    K_PROGRESS_RATE_REFERENCE,
    NP,
    NY_TERMINAL,
    P_PROGRESS_NOMINAL,
    P_REFERENCE_FIRST,
    P_REFERENCE_POSITION,
    P_REFERENCE_SECOND,
    Y_ACTUATED_FORCE,
    Y_INPUT,
    Y_LAG,
    Y_PASSIVE_POSITION,
    Y_PASSIVE_VELOCITY,
    Y_PLANNED_POSITION,
    Y_PLANNED_VELOCITY,
    Y_PROGRESS_RATE,
    build_ocp,
    shooting_intervals,
)

# Where the description lives by default, so `scripts/mpc_a2b.py` can reach the
# same file without re-deriving the path.
DEFAULT_DESCRIPTIONS = ox.default_descriptions(PACKAGE)

# The header this script writes beside the solver, carrying the numbers the
# C++ would otherwise have to derive a second time.
GENERATED_HEADER = "crane_mpc_ocp_generated.h"


# ------------------------------------------------------------------ generation


def write_header(
    output: Path,
    parameters: dict,
    scale: np.ndarray,
    constants: cs.Constants,
    actuator: cs.ActuatorFit,
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
        f"#define CRANE_MPC_OCP_PARAMETER_DOF {NP}",
        f"#define CRANE_MPC_OCP_PARAMETER_TOOL_POSITION {cs.P_TOOL_POSITION}",
        f"#define CRANE_MPC_OCP_PARAMETER_PAYLOAD_MASS {cs.P_PAYLOAD_MASS}",
        f"#define CRANE_MPC_OCP_PARAMETER_PAYLOAD_COM {cs.P_PAYLOAD_COM}",
        f"#define CRANE_MPC_OCP_PARAMETER_PAYLOAD_INERTIA {cs.P_PAYLOAD_INERTIA}",
        "",
        "// The rest of `p` is this OCP's own and is written per stage before every",
        "// solve: the stage's nominal virtual time, then the second-order expansion",
        "// of its reference about it -- value, d/ds, d^2/ds^2, per planned axis. The",
        "// progress state is virtual time, so the cost compares against",
        "// `q_a,ref(s)`; a spline cannot be baked into a generated solver, and this",
        "// local model is what carries both derivatives into the gradient and the",
        "// Hessian. At `s = s_nom` it is today's time-indexed reference exactly.",
        f"#define CRANE_MPC_OCP_PARAMETER_PROGRESS_NOMINAL {P_PROGRESS_NOMINAL}",
        f"#define CRANE_MPC_OCP_PARAMETER_REFERENCE_POSITION {P_REFERENCE_POSITION}",
        f"#define CRANE_MPC_OCP_PARAMETER_REFERENCE_FIRST {P_REFERENCE_FIRST}",
        f"#define CRANE_MPC_OCP_PARAMETER_REFERENCE_SECOND {P_REFERENCE_SECOND}",
        "",
        "// The (row, column) of each of those six entries, in the order they are",
        "// packed. Theta_L is symmetric and about the payload's own centre of mass",
        "// with the axes of K8, which is the URDF `<inertial>` convention.",
        "#define CRANE_MPC_OCP_PARAMETER_INERTIA_ENTRIES {"
        + ", ".join(f"{{{row}, {column}}}" for row, column in cs.INERTIA_ENTRIES)
        + "}",
        "",
        "// The blocks of `x`. C3 (`wiki/hydraulic_actuator_model.md` §1) adds two",
        "// actuator blocks after the rigid-body state: the PT1 lagged command on",
        "// the axes whose fitted `tau_v` is positive -- the arm's is zero, which is",
        "// a pole at infinity, so it has no lag state and its `u_f` is `u` -- and",
        "// then the force state on every planned axis. Neither has a counterpart in",
        "// `crane_model::State`, so both are OCP-only rows.",
        f"#define CRANE_MPC_OCP_STATE_PLANNED_POSITION {cs.X_PLANNED_POSITION}",
        f"#define CRANE_MPC_OCP_STATE_PASSIVE_POSITION {cs.X_PASSIVE_POSITION}",
        f"#define CRANE_MPC_OCP_STATE_PLANNED_VELOCITY {cs.X_PLANNED_VELOCITY}",
        f"#define CRANE_MPC_OCP_STATE_PASSIVE_VELOCITY {cs.X_PASSIVE_VELOCITY}",
        f"#define CRANE_MPC_OCP_STATE_COMMAND_LAG {cs.X_COMMAND_LAG}",
        f"#define CRANE_MPC_OCP_STATE_COMMAND_LAG_DOF {cs.K_COMMAND_LAG_DOF}",
        "#define CRANE_MPC_OCP_STATE_COMMAND_LAG_AXES {"
        + ", ".join(str(axis) for axis in cs.K_LAG_AXES)
        + "}",
        "",
        "// The progress pair of `docs/features/mpc-full-authority/brief.md` §2.1:",
        "// `s`, virtual time in seconds of nominal plan, and `v_s`, how fast the",
        "// plan is being spent. `v_s = 1` is exactly time-indexed tracking. The",
        "// sixth input is the progress **acceleration**, one order above Marc's",
        "// `s_dot`-as-input, so the plan's speed cannot step between cycles. They",
        "// sit before the force state because `v_s >= 0` is a box and the force",
        "// states carry none: the boxed rows have to stay a contiguous prefix.",
        f"#define CRANE_MPC_OCP_STATE_PROGRESS {cs.X_PROGRESS}",
        f"#define CRANE_MPC_OCP_STATE_PROGRESS_RATE {cs.X_PROGRESS_RATE}",
        f"#define CRANE_MPC_OCP_INPUT_PLANNED_DOF {cs.NU}",
        f"#define CRANE_MPC_OCP_INPUT_PROGRESS_ACCEL {cs.U_PROGRESS_ACCEL}",
        "",
        f"#define CRANE_MPC_OCP_STATE_ACTUATED_FORCE {cs.X_ACTUATED_FORCE}",
        "",
        "// The boxed rows of `x`: the rigid-body state, the lagged command and the",
        "// progress pair, a contiguous prefix. The force states are left out on",
        "// purpose -- see `export_ocp.py`; constraint 6 is the nonlinear row and not",
        "// a box.",
        f"#define CRANE_MPC_OCP_BOXED_STATE_DOF {cs.NBX}",
        "",
        "// C3's fitted numbers as they were folded into the dynamics, per planned",
        "// axis, from `crane_model/config/c3_full_model.json`. Here so the C++ and",
        "// the node can *say* which fit is in the artifact rather than assume one.",
        "// The dead time is C3 block 1 and is **not** in the model: it belongs to",
        "// the node's predictor, and a second copy inside the horizon double-counts",
        "// it (`docs/features/mpc-full-authority/brief.md` §2.2).",
        "#define CRANE_MPC_OCP_ACTUATOR_STIFFNESS {"
        + ", ".join(f"{value!r}" for value in actuator.k)
        + "}",
        "#define CRANE_MPC_OCP_ACTUATOR_COMMAND_LAG_S {"
        + ", ".join(f"{value!r}" for value in actuator.tau_v)
        + "}",
        f"#define CRANE_MPC_OCP_ACTUATOR_DEAD_TIME_S {actuator.dead_time_s!r}",
        "",
        "// The blocks of the stage residual",
        "// `y = [q_a, dq_a, q_u, dq_u, lag, v_s, tau_a, u]`. The order is §2's --",
        "// tracking, sway, lag, progress, effort, smoothness -- and not the state's,",
        "// and everything that has to agree with `W` reads it from here. The",
        "// terminal residual is the **prefix** up to and including the progress row,",
        "// so one set of offsets addresses both.",
        "//",
        "// The tracking rows carry their own reference and their `yref` is zero:",
        "// `q_a,ref(s)` is a function of a decision variable and acados subtracts",
        "// `yref` as a constant. The progress row's `yref` is the one below.",
        f"#define CRANE_MPC_OCP_RESIDUAL_PLANNED_POSITION {Y_PLANNED_POSITION}",
        f"#define CRANE_MPC_OCP_RESIDUAL_PLANNED_VELOCITY {Y_PLANNED_VELOCITY}",
        f"#define CRANE_MPC_OCP_RESIDUAL_PASSIVE_POSITION {Y_PASSIVE_POSITION}",
        f"#define CRANE_MPC_OCP_RESIDUAL_PASSIVE_VELOCITY {Y_PASSIVE_VELOCITY}",
        f"#define CRANE_MPC_OCP_RESIDUAL_LAG {Y_LAG}",
        f"#define CRANE_MPC_OCP_RESIDUAL_PROGRESS_RATE {Y_PROGRESS_RATE}",
        f"#define CRANE_MPC_OCP_RESIDUAL_ACTUATED_FORCE {Y_ACTUATED_FORCE}",
        f"#define CRANE_MPC_OCP_RESIDUAL_INPUT {Y_INPUT}",
        f"#define CRANE_MPC_OCP_RESIDUAL_TERMINAL_DOF {NY_TERMINAL}",
        "",
        "// The axis the lag row is written on, and the progress row's reference.",
        "// The lag row is Marc's 1-D projection on the slewing joint reproduced and",
        "// not generalised: five planned coordinates carry four radians and one",
        "// metre, so the unit tangent a single scalar projection needs is a norm",
        "// over mixed units. `export_ocp.py` carries the argument.",
        f"#define CRANE_MPC_OCP_LAG_AXIS {K_LAG_AXIS}",
        "// One second of plan per second of wall clock. A **constant**: any other",
        "// value is a reference that is not the plan.",
        f"#define CRANE_MPC_OCP_PROGRESS_RATE_REFERENCE {K_PROGRESS_RATE_REFERENCE!r}",
        "",
        "// The rows of `h`: constraint 6 once per planned axis, then constraint 7.",
        "#define CRANE_MPC_OCP_CONSTRAINT_CYLINDER_FORCE 0",
        f"#define CRANE_MPC_OCP_CONSTRAINT_PUMP_FLOW {cs.K_PLANNED_DOF}",
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
machine's, and the PZS100 is the machine that has to run.

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

    write_header(output, parameters, scale, cs.load_constants(), model.actuator)
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
