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
`crane_mpc/solver.py` writes it onto every stage before each solve. A block picked
up mid-run therefore changes the model without a reconfigure and without a
regeneration, which is what a behaviour tree calling `crane_msgs/SetPayload` at
grasp and release needs.

Runtime-settable on the generated solver, and set by `crane_mpc/solver.py` at
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
generated expression alone. There is one derivation of it, in
`crane_mpc.problem`, which both the artifact and the node's own startup build
call; the header written beside the tree is that number *recorded*, not a second
derivation of it. Its reader was `ocp_solver.cpp` and issue 132 deleted that, so
the header is now part of what a reviewer diffs and nothing else.

## There is no staleness guard, by decision -- except on the fit

`config/hydraulics.yaml`, `config/hydraulic_limits.yaml` and the description
all require this script to be re-run and the workspace rebuilt. **Nothing fails
when one of them and the checked-in tree disagree.** grill §4 records that the
alternative -- hashing the inputs into the generated code and comparing in a test,
which is what `crane_model`'s own fixture does -- was considered and rejected, and
that this is therefore a deliberate divergence from both existing generator
scripts in this repository. `generated/README.md` says the same thing where a
reader of the tree will find it.

**`c3_full_model.json` is the exception**, because it is the one input that
exists twice and reaches the solver by two routes. Its digest is in the generated
header, so a refit fails `--check`; its copies are compared, so a drift between
them fails generation; and the description's damping is held against the fit's
own `d`, so a refit that moved `k` and left `d` behind is refused, not shipped.

## What is boilerplate lives in `crane_ocp`

The scratch-JSON code generation, the pruning, the whitespace normalisation and
the `--check` byte-compare are `crane_ocp/scripts/crane_ocp_export.py` -- the same
module `crane_planning/scripts/export_timing_ocp.py` uses, because that half of
the two exporters was the same code twice. What is left here is the problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# The header this script writes beside the solver, recording the numbers the
# export chose.
GENERATED_HEADER = "crane_mpc_ocp_generated.h"

# --- the C3 fit, and the places it lives --------------------------------------
#
# `load_actuator_fit` opens whichever copy `crane_model` carries -- the install
# prefix by preference -- so the path it returns is a machine's and cannot go in
# a generated header. This is the repository's name for the same file.
FIT_LABEL = "crane_model/config/c3_full_model.json"

# The fit's other homes. `wiki/` is not reachable from an installed package, so
# `crane_model` carries a byte copy: one identification in several files with
# nothing failing when they drift. This export is what fails now.
WIKI_FIT = (
    PACKAGE.parents[2]
    / "wiki"
    / "diagrams"
    / "hydraulic_calibration"
    / "c3_full_model.json"
)

# The copy that bites. `default_actuator_path` prefers the **install prefix**
# while the description comes off the source tree unconditionally, so editing the
# source fit and rebuilding only `crane_mpc` used to regenerate nothing and
# report "current": `k` from one revision, `d` from another, every guard green.
SOURCE_FIT = PACKAGE.parent / "crane_model" / "config" / "c3_full_model.json"

# How far the description's damping may sit from the fit's `d` and still be the
# same identification. A rounding allowance and nothing wider -- a refit moves
# these by tens of percent, not tenths. The description states `d` to four
# significant figures except for the rotator, which gets three (`484` for
# `483.671816`), so the budget is 2e-3: measured gaps, worst last, are sw 1.8e-5,
# sa 5.4e-5, ka 6.1e-5, ha 1.4e-4, ro 6.8e-4.
FIT_DAMPING_TOLERANCE = 2.0e-3


# --------------------------------------------------------------------- the fit


def read_fit(path: Path) -> dict:
    """Read one copy of the C3 fit as a document, not as numbers."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def fit_digest(fit: dict) -> str:
    """
    Return a content digest of the whole fit: which identification this is.

    Over the **parsed** document rather than the bytes, because the two copies
    differ by a trailing newline and are the same fit.

    Every number is in it, including the `d` that reaches the dynamics through
    the description and the diagnostics this export never reads. That is the
    point: a refit moves the digest, the digest is in the generated header, so
    `--check` fails until the tree is rewritten instead of the artifact shipping
    under the previous fit's identity.
    """
    canonical = json.dumps(fit, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def check_fit_is_one_artifact(fit: dict) -> None:
    """
    Refuse to generate while any two copies of the fit disagree.

    `fit` is the copy that was actually read; every other copy that exists is
    compared against it as a parsed document. A standalone checkout has no
    `wiki/` and a source-only workspace has no install copy -- a missing
    candidate is skipped, because it is not a drift.
    """
    for other in (WIKI_FIT, SOURCE_FIT):
        if not other.is_file() or read_fit(other) == fit:
            continue
        raise ValueError(
            f"{other} and {FIT_LABEL} are one identification in two files and "
            "they have drifted. Copy whichever the fitter wrote last over the "
            "other, rebuild `crane_model` so the install copy follows, and "
            "re-run this script."
        )


def check_fit(model: cs.CraneSymbolicModel, fit: dict) -> None:
    """
    Refuse to generate on a fit the solver must not be built from, by axis name.

    Two failures, both per axis and both invisible in every quantity except the
    machine's response:

    * **a non-finite `k`, `tau_v` or `d`.** `load_actuator_fit` refuses a missing
      axis and a non-positive `k`, but `inf > 0` is true, so an infinity walks
      through it into the dynamics. `d` has to be checked here or nowhere:
      nothing upstream reads it, because it reaches the dynamics through the
      description, and a NaN would defeat the comparison below rather than fail
      it -- `nan > nan` is False, so the axis would pass;
    * **a description whose damping is not this fit's `d`.** `d_i` and `k_i` are
      one identification and only pair with each other, and they arrive by two
      routes: `k` and `tau_v` out of the fit at full precision, `d` through the
      description's `<dynamics damping>` at four figures. A refit that moved one
      and not the other is a damping-ratio error, not a rounding one.
    """
    actuator = model.actuator
    for axis, key in enumerate(cs.K_AXIS_KEYS):
        for name, value in (("k", actuator.k[axis]), ("tau_v", actuator.tau_v[axis])):
            if not np.isfinite(value):
                raise ValueError(
                    f"{FIT_LABEL}: axis {key!r} has {name}={value}, which is not a "
                    "number a solver can be built on"
                )
        entry = fit["axes"][key]
        if "d" not in entry:
            raise ValueError(f"{FIT_LABEL}: axis {key!r} carries no d")
        fitted = float(entry["d"])
        if not np.isfinite(fitted):
            raise ValueError(
                f"{FIT_LABEL}: axis {key!r} has d={fitted}, which is not a number "
                "a solver can be built on"
            )
        damped = float(model.description.damping[cs.K_PLANNED_ROWS[axis]])
        if abs(damped - fitted) > FIT_DAMPING_TOLERANCE * abs(fitted):
            raise ValueError(
                f"axis {key!r}: the description damps it at {damped} and this "
                f"fit's d is {fitted}. `d` and `k` are one identification -- "
                f"regenerate the description from {FIT_LABEL} before exporting."
            )
    if not np.isfinite(actuator.dead_time_s):
        raise ValueError(
            f"{FIT_LABEL}: dead_time_common_ms is {actuator.dead_time_s * 1.0e3}"
        )


# ------------------------------------------------------------------ generation


def write_header(
    output: Path,
    parameters: dict,
    scale: np.ndarray,
    constants: cs.Constants,
    actuator: cs.ActuatorFit,
    digest: str,
) -> Path:
    """
    Record what the export chose, beside the solver it chose it for.

    The dimensions are acados' own, in `acados_solver_<name>.h`, and are not
    repeated here. What is here is the grid, the row offsets this script chose,
    and the conditioning divisors -- the three things that cannot be read off the
    generated solver. It was written for `ocp_solver.cpp`, which issue 132
    deleted; it is kept because it is the part of the artifact a reviewer can
    actually read.
    """
    path = output / GENERATED_HEADER
    guard = "CRANE_MPC_OCP_GENERATED_H_"
    scales = ", ".join(f"{value!r}" for value in scale.tolist())
    lines = [
        "// Generated by `scripts/export_ocp.py`. Do not edit; re-run the script.",
        "//",
        "// The OCP of `wiki/mpc.md` §1 as it was written, in the numbers the",
        "// export chose for it. Dimensions are acados' own,",
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
        "// **order**, so this is where the packing is written down rather than",
        "// guessed from the solver. Both halves are set on every stage before every",
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
        "// axis, from `crane_model/config/c3_full_model.json`. Here so a reader can",
        "// *say* which fit is in the artifact rather than assume one.",
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
        "// **Which** fit those came from. The path is the repository's name for the",
        "// file and not the one the reader opened -- `crane_model` is installed, so",
        "// that one is a machine's install prefix. The digest is sha256 over the",
        "// parsed document with sorted keys, so it is a property of the numbers and",
        "// not the formatting, and **every** number is in it: the `d` that reaches",
        "// the dynamics through the description as well as the `k` and `tau_v`",
        "// above. A refit therefore fails `export_ocp.py --check` rather than",
        "// shipping under the previous fit's identity.",
        f'#define CRANE_MPC_OCP_ACTUATOR_FIT_SOURCE "{FIT_LABEL}"',
        f'#define CRANE_MPC_OCP_ACTUATOR_FIT_DIGEST "{digest}"',
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
                                     conditioning divisors, as the export chose
                                     them
    crane_mpc_pzs100/                the PZS100 solver, constrained

**Nothing compiles this tree.** The node builds its own solver at startup, into
`CRANE_MPC_OCP_CACHE`, and never opens what is here; the C++ that did was deleted
by issue 132. The tree is kept as the reviewable form of the OCP -- a diff of it
is how a change to the problem is seen -- and `--check` is kept with it as a
build-time target, because a tree nobody checks is a tree nobody can read as
current.

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
comparing in a test, which is the pattern `crane_model`'s own fixture uses --
and records that this is therefore a deliberate
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
    # The same file `load_actuator_fit` reads, read once more as a document: the
    # checks below and the header's provenance are about the fit's *identity*,
    # which is more than the three numbers the dynamics take out of it.
    fit = read_fit(Path(cs.default_actuator_path()))
    check_fit_is_one_artifact(fit)
    ocp, scale, model = build_ocp(
        (descriptions / DESCRIPTION).read_text(), parameters, hydraulics
    )
    # Before the code generation, so a bad fit stops generation rather than
    # producing a tree and then complaining about it.
    check_fit(model, fit)
    tree = ox.generate_solver(ocp, output)
    # `crane_symbolic`'s `z` over `(x, u, p)`, shipped beside the solver because
    # acados generates only the rows it solves and `wiki/mpc.md` §5.3
    # requirement 4 wants the residuals in physical units.
    ox.write_output_map(model, ocp.model.name, tree)

    write_header(
        output, parameters, scale, cs.load_constants(), model.actuator, fit_digest(fit)
    )
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
