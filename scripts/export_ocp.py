#!/usr/bin/env python3
"""
Export the crane MPC optimal control problem: generate, compile, record.

`docs/features/cbs-ocp-python/grill.md` D1/D8 for `crane_mpc`: the problem is
defined in Python over `crane_model/scripts/crane_symbolic.py` and shipped as a
compiled acados solver.

    ./scripts/export_ocp.py          # or: ros2 run crane_mpc export_ocp

**Run this before the node or either harness.** Nothing compiles a solver at
startup any more: `crane_mpc.solver.Ocp` opens what this wrote and refuses
anything that is not this problem, naming what moved. That is the trade -- one
deliberate step, and a startup that can no longer build the wrong thing quietly.

Algebra is imported, never restated: `crane_symbolic` carries dynamics,
transmission and the output map; `crane_mpc/problem.py` adds the shooting
problem, the nonlinear-least-squares residual, seven constraints and the grid.

## What the export records

`crane_mpc.solver.export_key` -- the grid, every acados setting, the hydraulics
table and digests of the description, `problem.py`, `crane_symbolic.py` and the
C3 fit. Weights, boxes and slack prices are deliberately absent: they go in
through the runtime API every cycle, so retuning reuses this solver.

The integrators go in the same export: the cold-start stepper and one per
dead-time replay segment, from `solver.predictor_intervals`, so the set this
compiles and the set `Ocp` opens are one list.

## The C3 fit keeps its own checks

Two copies, two routes, and its `d` reaches the dynamics only through the
description. The copies are compared (drift fails the export) and the
description's damping is checked against `d`, so a moved `k` without `d` is
refused -- before code generation, not after.

## Boilerplate lives in `crane_ocp`

`crane_ocp_export` is the one export path in this stack, shared with
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

from crane_mpc import config as ocp_config  # noqa: E402
from crane_mpc import solver as ocp_runtime  # noqa: E402
from crane_mpc.problem import (  # noqa: E402
    DESCRIPTION,
    TOOL,
    build_ocp,
    shooting_intervals,  # noqa: F401  -- re-exported for scripts/mpc_a2b.py
)

# default description location, shared with scripts/mpc_a2b.py
DEFAULT_DESCRIPTIONS = ox.default_descriptions(PACKAGE)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ox.add_arguments(parser, PACKAGE, ocp_runtime.EXPORT_ENV, f"crane_mpc_{TOOL}")
    arguments = parser.parse_args()

    parameters = ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    # the constants live in crane_model; nothing here repeats them
    hydraulics = hydraulic_limits()
    # the box and u^+ are crane_model's, same as the node reads them
    parameters["limits"].update(ocp_config.machine_limits())

    description_xml = (arguments.descriptions / DESCRIPTION).read_text()
    # same file `load_actuator_fit` reads, as a document: the checks want the
    # fit's identity, not just the three numbers the dynamics use
    fit = read_fit(Path(cs.default_actuator_path()))
    check_fit_is_one_artifact(fit)
    ocp, _scale, model, _chamber = build_ocp(description_xml, parameters, hydraulics)
    # before code generation, so a bad fit stops the export rather than
    # complaining about a solver it already compiled
    check_fit(model, fit)

    root = ox.export(
        ocp,
        arguments.output,
        ocp_runtime.export_key(parameters, hydraulics, description_xml),
        sims=ocp_runtime.predictor_sims(ocp, parameters),
        verbose=arguments.verbose,
    )
    print(f"exported and compiled into {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
