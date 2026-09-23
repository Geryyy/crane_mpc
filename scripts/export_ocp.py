#!/usr/bin/env python3
"""
Write the crane MPC optimal control problem as a generated acados solver.

`docs/features/cbs-ocp-python/grill.md` D1/D8 for `crane_mpc`: problem defined
here in Python over `crane_model/scripts/crane_symbolic.py`, shipped as
generated C under `generated/`.

    ./scripts/export_ocp.py                # rewrite `generated/`
    ./scripts/export_ocp.py --check        # regenerate into a scratch tree and diff

Algebra is imported, never restated: `crane_symbolic` carries dynamics,
transmission, output map; this file adds the shooting problem, the
nonlinear-least-squares residual, seven constraints, and the grid.

## Baked vs runtime-settable

Baked at code generation: constraint structure (boxed/soft rows, `h` row
order), each `h` row's conditioning divisor (`constraint_scale` below), and
the dynamics/description the solver was built from.

**Payload is not baked** (issue 072): mass, COM and the six entries of
`Theta_L` ride in `p`, written every stage by `crane_mpc/solver.py` -- lets a
block picked up mid-run change the model with no reconfigure/regeneration,
what `crane_msgs/SetPayload` at grasp/release needs.

Runtime-settable at configure time: `W`, `yref`, `lbx`/`ubx`, `lbu`/`ubu`,
`lh`/`uh`, slack prices, Levenberg-Marquardt, **and `N`/`T_s`** -- acados
generates `<name>_acados_create_with_discretization(capsule, N, steps)`
beside the fixed-`N` entry point, so the grid is a runtime argument (grill D5
expected regenerate-only; it isn't).

## `constraint_scale`: conditioning, not a limit

`h` rows are divided by a fixed number: in physical units they span eight
decades (newtons ~1e5, m^3/s ~1e-3) and HPIPM fails on that Jacobian.
`|F_cyl,i| <= F_i^max` goes in as `lh_i = -F_i^max / scale_i`, so a
relief-pressure change moves `lh`/`uh`, not the generated expression. One
derivation (`crane_mpc.problem`); the header just records it. Its C++ reader
(`ocp_solver.cpp`) was deleted by issue 132, so the header is read by no code --
it is what a reviewer diffs, and what carries the digests `--check` gates on.

## What `--check` guards

`test/test_generated_is_current.py` runs `--check`, so an un-re-exported
`hydraulics.yaml` or description edit fails the suite. (An older note here said
nothing failed -- true while this was a CMake target, issue 133 moved it.)

Only `REVIEWED` is compared file by file; the rest is digested into
`CRANE_MPC_OCP_GENERATED_DIGEST`, which is what keeps that guard -- a
description edit lands in the CasADi bodies and nothing else.

`c3_full_model.json` keeps its own digest: two copies, two routes, and its `d`
reaches the dynamics only through the description. Copies compared (drift fails
generation); description damping checked against `d` (moved `k` without `d` is
refused).

## Boilerplate lives in `crane_ocp`

Scratch-JSON generation, pruning, whitespace normalisation and `--check`
byte-compare are `crane_ocp/scripts/crane_ocp_export.py`, shared with
`crane_planning/scripts/export_timing_ocp.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent

# neither crane_ocp's nor crane_model's scripts/ is installed (issue 070); import by path
sys.path.insert(0, str(PACKAGE.parent / "crane_ocp" / "scripts"))

import crane_ocp_export as ox  # noqa: E402

cs = ox.import_crane_symbolic(PACKAGE)

from crane_model import hydraulic_limits  # noqa: E402

# problem lives in the installed package (node builds it at startup); a from-scratch
# build runs this before crane_mpc is installed, so source tree is the fallback --
# same as import_crane_symbolic does for cs
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
    P_PATH_CONTROL,
    P_PATH_ORIGIN,
    PATH_POINTS,
    SOLVER_PREFIX,
    TOOL,
    Y_ACTUATED_FORCE,
    Y_INPUT,
    Y_LAG,
    Y_PASSIVE_POSITION,
    Y_PASSIVE_VELOCITY,
    Y_PLANNED_POSITION,
    Y_PLANNED_VELOCITY,
    Y_PROGRESS,
    Y_PROGRESS_RATE,
    build_ocp,
    shooting_intervals,
)

# default description location, shared with scripts/mpc_a2b.py
DEFAULT_DESCRIPTIONS = ox.default_descriptions(PACKAGE)

# header written beside the solver, recording the numbers the export chose
GENERATED_HEADER = "crane_mpc_ocp_generated.h"

# What is committed, and so what `--check` compares file by file. The whole tree
# is still written; everything outside this is gitignored and reaches the
# comparison through `CRANE_MPC_OCP_GENERATED_DIGEST`. The CasADi bodies are 80 %
# of the tree and no review has ever read one.
SOLVER_TREE = f"{SOLVER_PREFIX}_{TOOL}"
REVIEWED = (
    "README.md",
    GENERATED_HEADER,
    f"{SOLVER_TREE}/acados_solver_{SOLVER_TREE}.h",
)

# --- the C3 fit, and the places it lives --------------------------------------
#
# `load_actuator_fit` prefers the install-prefix copy; its path is machine-local,
# can't go in a generated header -- this is the repo's name for the same file.
FIT_LABEL = "crane_model/config/c3_full_model.json"

# installed packages can't reach the source tree, so crane_model carries a byte
# copy -- one identification, several files, nothing failed on drift until this export
WIKI_FIT = (
    PACKAGE.parents[2]
    / "wiki"
    / "diagrams"
    / "hydraulic_calibration"
    / "c3_full_model.json"
)

# default_actuator_path prefers the install prefix, description comes off source
# tree unconditionally -- editing the source fit and rebuilding only crane_mpc used
# to report "current" with k from one revision, d from another
SOURCE_FIT = PACKAGE.parent / "crane_model" / "config" / "c3_full_model.json"

# rounding allowance on damping vs fit's d (refit moves these by tens of percent);
# d stated to 4 figures (3 for rotator: 484 for 483.671816) -> budget 2e-3;
# measured gaps sw 1.8e-5, sa 5.4e-5, ka 6.1e-5, ha 1.4e-4, ro 6.8e-4.
FIT_DAMPING_TOLERANCE = 2.0e-3


# --------------------------------------------------------------------- the fit


def read_fit(path: Path) -> dict:
    """Read one copy of the C3 fit as a document, not as numbers."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def fit_digest(fit: dict) -> str:
    """
    Return a content digest of the whole fit: which identification this is.

    Parsed document, not bytes (two copies differ by a trailing newline, same
    fit). Every number is in it, including `d`, which the description carries
    into the dynamics without this export reading it directly: a refit moves
    the digest, the digest is in the header, so `--check` fails until the tree
    is rewritten rather than shipping under the previous fit's identity.
    """
    canonical = json.dumps(fit, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def check_fit_is_one_artifact(fit: dict) -> None:
    """
    Refuse to generate while any two copies of the fit disagree.

    `fit` is the copy actually read; every other existing copy is compared
    against it as a parsed document. A standalone checkout or source-only
    workspace may lack one candidate -- skipped, not treated as drift.
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

    Two failures, per axis, invisible except in the machine's response:

    * **non-finite `k`, `tau_v` or `d`.** `load_actuator_fit` refuses a missing
      axis and non-positive `k`, but `inf > 0` is true, so infinity walks into
      the dynamics. `d` must be checked here: nothing upstream reads it (it
      reaches the dynamics only through the description), and `nan > nan` is
      False, so a NaN would pass rather than fail the comparison below;
    * **description damping not this fit's `d`.** `d_i`/`k_i` are one
      identification, arriving by two routes -- `k`/`tau_v` from the fit at
      full precision, `d` through the description at four figures. A refit
      that moved one and not the other is a damping-ratio error, not rounding.
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
    tree_digest: str,
) -> Path:
    """
    Record what the export chose, beside the solver it chose it for.

    Dimensions are acados' own (`acados_solver_<name>.h`), not repeated here.
    This carries the grid, row offsets and conditioning divisors -- what can't
    be read off the generated solver. Written for `ocp_solver.cpp` (issue 132
    deleted it); kept as the part of the artifact a reviewer can read.
    """
    path = output / GENERATED_HEADER
    guard = "CRANE_MPC_OCP_GENERATED_H_"
    scales = ", ".join(f"{value!r}" for value in scale.tolist())
    lines = [
        "// Generated by `scripts/export_ocp.py`. Do not edit; re-run the script.",
        "//",
        "// The OCP as it was written, in the numbers the",
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
        "// The rest of `p` is this OCP's own: the path, and the window it is read",
        "// over. One vector for the whole horizon, written once before every solve --",
        "// which stage a stage is, the progress state carries, not the parameters.",
        "// A clamped B-spline: the control points are numbers here and the basis",
        "// is in the generated code, which is how a spline bakes into a solver.",
        "// The progress state is the path parameter, so `ORIGIN` is where on the",
        "// path this cycle starts and `s` carries the rest of the horizon. The",
        "// second-order expansion this replaced was only the path near one point.",
        f"#define CRANE_MPC_OCP_PARAMETER_PATH_ORIGIN {P_PATH_ORIGIN}",
        f"#define CRANE_MPC_OCP_PARAMETER_PATH_CONTROL {P_PATH_CONTROL}",
        f"#define CRANE_MPC_OCP_PARAMETER_PATH_POINTS {PATH_POINTS}",
        "",
        "// The (row, column) of each of those six entries, in the order they are",
        "// packed. Theta_L is symmetric and about the payload's own centre of mass",
        "// with the axes of K8, which is the URDF `<inertial>` convention.",
        "#define CRANE_MPC_OCP_PARAMETER_INERTIA_ENTRIES {"
        + ", ".join(f"{{{row}, {column}}}" for row, column in cs.INERTIA_ENTRIES)
        + "}",
        "",
        "// The blocks of `x`. C3 adds two",
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
        f"#define CRANE_MPC_OCP_RESIDUAL_PROGRESS {Y_PROGRESS}",
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
        "// §3.1's two smoothing widths, as `crane_model/config/hydraulics.yaml` states",
        "// them and as they went into `Q`. Here so that a test can assert the identity",
        "// `A±(v) sqrt(v² + eps²)` from outside without restating either number --",
        "// `crane_model`'s own `hydraulics::Constants` is private to that package.",
        f"#define CRANE_MPC_OCP_SMOOTHING_EPS_ABS {constants.eps_abs!r}",
        f"#define CRANE_MPC_OCP_SMOOTHING_EPS_V {constants.eps_v!r}",
        "",
        "// The soft rows of §3.2 among the state box, in acados' own slack order.",
        f"#define CRANE_MPC_OCP_SOFT_PASSIVE_POSITION {cs.X_PASSIVE_POSITION}",
        f"#define CRANE_MPC_OCP_SOFT_PASSIVE_VELOCITY {cs.X_PASSIVE_VELOCITY}",
        "",
        "// The rest of the tree, uncommitted: sha256 over every generated file",
        "// outside `REVIEWED`, on normalised content. A description or hydraulics",
        "// edit lands in the CasADi bodies and nothing else here (measured: a link",
        "// mass moves `impl_dae_*.c` and `_output.c`, the numbers above hold), so",
        "// shipping those bodies was the whole guard on them. This replaces it.",
        f'#define CRANE_MPC_OCP_GENERATED_DIGEST "{tree_digest}"',
        "",
        f"#endif  // {guard}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


README = """\
# `crane_mpc/generated/` -- the reviewable form of the OCP

Machine output, do not edit. Change `scripts/export_ocp.py`,
`crane_model/scripts/crane_symbolic.py` or a config file, re-run
`./scripts/export_ocp.py`, commit. `--check` regenerates into a scratch tree
and diffs; `test/test_generated_is_current.py` runs it.

## Three files committed, the rest digested

    README.md
    crane_mpc_ocp_generated.h                          grid, residual offsets,
                                                       divisors, two digests
    crane_mpc_pzs100/acados_solver_crane_mpc_pzs100.h  dimensions

The export writes the whole solver (`--check` needs a tree to compare against);
the rest is gitignored. It was committed until this change: 4.0 MB, 176 kLOC,
141 kLOC of that four `impl_dae_*jac*.c` nobody has ever read. Nothing compiles
this tree -- the node builds its own solver at startup into
`CRANE_MPC_OCP_CACHE`, the C++ that read this went with issue 132.

Dropping the bodies would have dropped a real guard, so it did not:
`CRANE_MPC_OCP_GENERATED_DIGEST` is sha256 over every uncommitted generated
file. A description or hydraulics edit lands in those bodies and nowhere else
here -- measured, a link mass moves `impl_dae_*.c` and `_output.c` while every
number in the header holds still.

**One solver**: the description is baked in, and the PZS100 is what has to run.
`crane_mpc_pzs100_output.{c,h}` is not acados' -- acados generates only the
conditioned rows it solves, so the output map ships beside it for `tau_a`,
`F_cyl`, `v`, `Q` in physical units. `Makefile`, `main_*.c`,
`acados_sim_solver_*` and `acados_solver.pxd` are pruned (absolute path, and a
`main`).

## Not in the guarded surface

`config/crane_mpc.yaml`, deliberately. `N`, `T_s`, weights, boxes, slack prices
and Levenberg-Marquardt are all set at configure time -- `N`/`T_s` through
acados' `_acados_create_with_discretization` -- so none can make this tree
stale. The header's grid is a default, not a contract.
"""


def generate(
    output: Path, descriptions: Path, parameters: dict, hydraulics: dict
) -> None:
    """Write the whole tree, from an empty directory."""
    output.mkdir(parents=True, exist_ok=True)
    # same file load_actuator_fit reads, as a document: checks below need the fit's
    # identity, not just the three numbers the dynamics use
    fit = read_fit(Path(cs.default_actuator_path()))
    check_fit_is_one_artifact(fit)
    ocp, scale, model, _chamber = build_ocp(
        (descriptions / DESCRIPTION).read_text(), parameters, hydraulics
    )
    # before code generation, so a bad fit stops generation rather than complaining after
    check_fit(model, fit)
    tree = ox.generate_solver(ocp, output)
    # crane_symbolic's z over (x, u, p), shipped beside the solver: acados generates
    # only the rows it solves, but residuals in physical units are needed too
    ox.write_output_map(model, ocp.model.name, tree)

    write_header(
        output,
        parameters,
        scale,
        cs.load_constants(),
        model.actuator,
        fit_digest(fit),
        ox.tree_digest(output, REVIEWED),
    )
    ox.finalise(output, README)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ox.add_arguments(parser, PACKAGE)
    arguments = parser.parse_args()

    parameters = ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    # the constants live in crane_model; nothing here repeats them
    hydraulics = hydraulic_limits()

    return ox.run(
        arguments.output,
        arguments.check,
        lambda output: generate(output, arguments.descriptions, parameters, hydraulics),
        "export_ocp.py",
        reviewed=REVIEWED,
    )


if __name__ == "__main__":
    sys.exit(main())
