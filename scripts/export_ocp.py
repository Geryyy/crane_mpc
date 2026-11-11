#!/usr/bin/env python3
"""
Export the crane MPC optimal control problem: generate, compile, record.

`docs/features/cbs-ocp-python/grill.md` D1/D8 for `crane_mpc`: the problem is
defined in Python over `crane_model/scripts/crane_symbolic.py` and shipped as a
compiled acados solver.

    ./scripts/export_ocp.py          # from the source checkout; not a `ros2 run` executable

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

The integrators go in the same export: one `Ts` step, from
`solver.predictor_sims`, which is the cold start, one horizon stage and the
dead time alike -- so the set this compiles and the set `Ocp` opens are one list.

## The C3 fit keeps its own checks

Two copies, two routes, and its `d` reaches the dynamics only through the
description. The copies are compared (drift fails the export) and the
description's damping is checked against `d`, so a moved `k` without `d` is
refused -- before code generation, not after.

`ocp_actuator_fit` poses the OCP on a **design** model instead (issue 311): a
scaled copy of the fit, read through `config.ocp_actuator_model` so the node
and this script compile one thing. The identification keeps
`check_fit_is_one_artifact` whichever is selected; the design file answers to
`check_design_fit` -- same schema, `tau_v` equal, `k` and `d` scaled together.

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

# `crane_ocp_export` is installed (crane_ocp declares it), but these scripts are
# meant to run from a source checkout without the overlay sourced, the same way
# they fall back to the source tree for `crane_mpc` and `crane_model`. The
# runtime imports it too -- `crane_mpc.solver` does -- so this has to go on the
# path *before* that module, not just before the exporter.
try:
    import crane_ocp_export  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PACKAGE.parent / "crane_ocp"))

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

# the installed package owns both: the node selects a C3 file through the same
# two functions, and `scripts/wire_chain/harness.py` reads them off this module
from crane_mpc.config import (  # noqa: E402
    ocp_actuator_model,
    read_fit,
    retuned_description,  # noqa: F401
)
from crane_mpc.problem import (  # noqa: E402
    DESCRIPTION,
    TOOL,
    build_ocp,
)

# default description location, shared with scripts/wire_chain/harness.py
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

# --- the OCP's design model (`ocp_actuator_fit`, issue 311) -------------------

#: What a design file says about itself, in its own top-level field.
DESIGN_FIELD = "design_margin"

#: The fields a design axis may differ in: the scaled pair, and the three that
#: follow from it (`f_n_hz` and `zeta` both go with the square root of the
#: factor). Every other field is provenance and is carried over unchanged.
DESIGN_SCALED_FIELDS = ("k", "d", "f_n_hz", "zeta", "k_within_5pc", "d_within_5pc")

#: Allowance on `d/k` against the fit's: the pair is written by scaling in
#: double, so this is round-off and not a margin.
DESIGN_RATIO_TOLERANCE = 1.0e-9

#: The top-level fields a design file may word for itself. Everything else above
#: `axes` is the fit's own value, not just its name -- `dead_time_common_ms`
#: reaches `ActuatorFit.dead_time_s` and `check_fit` only tests it for finiteness.
DESIGN_TOP_FIELDS = (DESIGN_FIELD, "identified_on")

#: Widest `k` the gate admits, x the identification. 307's step-0 grid: x2 and x4
#: are calm on plants x1..x12, x12 exits `past_clamp` on x1 with 0.139 rad error
#: and a 1.35/cycle ring -- too stiff a model is as bad as too soft.
DESIGN_MAX_FACTOR = 4.0


# --------------------------------------------------------------------- the fit


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


def check_design_fit(design: dict, fit: dict) -> None:
    """
    Refuse an OCP design model that is not the identification with scaled axes.

    `ocp_actuator_fit` selects a robustness margin on the OCP's own C3 (issue
    307: the shipped OCP rings on any plant stiffer than its model), never a
    second identification, and this is where that is said in code:

    * it declares itself in `design_margin` and carries the fit's own schema,
      so a copy found later cannot read as an identification;
    * every field but the scaled pair and what follows from it is the fit's
      own, **`tau_v` included**: `LAG_AXES` and `NX` are derived from the
      shipped fit at import, so a moved `tau_v` would pose a problem with a
      state the runtime does not have;
    * `k` and `d` move **together** per axis and never below the fit, so
      `tau = d/k` stays the machine's. `k` alone is a stiffer spring and a
      different plant -- the JTC rings on it by itself (issue 308);
    * and never above `DESIGN_MAX_FACTOR`, the band the gate measured. Only a
      planned axis may scale at all: `gr` is outside `AXIS_KEYS`, so a margin
      written there would be dropped by `retuned_description` and claim
      something the OCP never gets.
    """
    if not str(design.get(DESIGN_FIELD, "")).strip():
        raise ValueError(
            f"the OCP's actuator file carries no {DESIGN_FIELD!r}, so nothing in it "
            f"says it is a design margin on {FIT_LABEL} rather than a fit"
        )
    if set(design) != set(fit) | {DESIGN_FIELD}:
        differs = set(design) ^ (set(fit) | {DESIGN_FIELD})
        raise ValueError(
            f"the design model and {FIT_LABEL} differ in the fields "
            f"{sorted(differs)}; it is that file with scaled axes and keeps its schema"
        )
    for field, value in design.items():
        if field in DESIGN_TOP_FIELDS or field == "axes" or value == fit[field]:
            continue
        raise ValueError(
            f"the design model's {field} is {value} and {FIT_LABEL}'s is "
            f"{fit[field]}; outside the axes a design model words only "
            f"{DESIGN_TOP_FIELDS} for itself"
        )
    if set(design["axes"]) != set(fit["axes"]):
        raise ValueError(
            f"the design model's axes {sorted(design['axes'])} are not "
            f"{FIT_LABEL}'s {sorted(fit['axes'])}"
        )
    for key, entry in design["axes"].items():
        identified = fit["axes"][key]
        if key not in cs.AXIS_KEYS:
            if entry != identified:
                raise ValueError(
                    f"design axis {key!r} differs from {FIT_LABEL}'s, and it is not "
                    "a planned axis: the OCP would never see the margin"
                )
            continue
        if set(entry) != set(identified):
            raise ValueError(
                f"design axis {key!r} carries the fields {sorted(entry)} and "
                f"{FIT_LABEL}'s carries {sorted(identified)}"
            )
        for field, value in entry.items():
            if field in DESIGN_SCALED_FIELDS or value == identified[field]:
                continue
            raise ValueError(
                f"design axis {key!r}: {field} is {value} and {FIT_LABEL}'s is "
                f"{identified[field]}; a design model scales {DESIGN_SCALED_FIELDS} "
                "and nothing else"
            )
        factor = float(entry["k"]) / float(identified["k"])
        if factor < 1.0:
            raise ValueError(
                f"design axis {key!r}: k is x{factor:g} the identified one. A design "
                "model is a margin -- softer than the machine is not one, and the "
                "mismatch that rings is an OCP under its plant"
            )
        if factor > DESIGN_MAX_FACTOR:
            raise ValueError(
                f"design axis {key!r}: k is x{factor:g} the identified one, past the "
                f"x{DESIGN_MAX_FACTOR:g} the gate admits. Issue 307: an OCP x12 exits "
                "past_clamp on the nominal plant -- a margin this wide asks for a "
                "joint velocity the axis answers faster than the model believes"
            )
        # cross-multiplied: no division, so a fit with d = 0 is not a special case
        if not np.isclose(
            float(entry["d"]) * float(identified["k"]),
            float(identified["d"]) * float(entry["k"]),
            rtol=DESIGN_RATIO_TOLERANCE,
            atol=0.0,
        ):
            raise ValueError(
                f"design axis {key!r}: k is x{factor:g} the identified one and d is "
                f"{entry['d']} against {identified['d']}. `k` and `d` scale together "
                "so that tau = d/k stays the machine's; `k` alone is a different plant"
            )


def check_fit(model: cs.CraneSymbolicModel, fit: dict, label: str = FIT_LABEL) -> None:
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

    `label` is the file the refusal names: the identification, or the design
    model the OCP is posed on instead.
    """
    actuator = model.actuator
    for axis, key in enumerate(cs.AXIS_KEYS):
        for name, value in (("k", actuator.k[axis]), ("tau_v", actuator.tau_v[axis])):
            if not np.isfinite(value):
                raise ValueError(
                    f"{label}: axis {key!r} has {name}={value}, which is not a "
                    "number a solver can be built on"
                )
        entry = fit["axes"][key]
        if "d" not in entry:
            raise ValueError(f"{label}: axis {key!r} carries no d")
        fitted = float(entry["d"])
        if not np.isfinite(fitted):
            raise ValueError(
                f"{label}: axis {key!r} has d={fitted}, which is not a number "
                "a solver can be built on"
            )
        damped = float(model.description.damping[cs.PLANNED_INDICES[axis]])
        if abs(damped - fitted) > FIT_DAMPING_TOLERANCE * abs(fitted):
            raise ValueError(
                f"axis {key!r}: the description damps it at {damped} and this "
                f"fit's d is {fitted}. `d` and `k` are one identification -- "
                f"regenerate the description from {label} before exporting."
            )
    if not np.isfinite(actuator.dead_time_s):
        raise ValueError(
            f"{label}: dead_time_common_ms is {actuator.dead_time_s * 1.0e3}"
        )


# ------------------------------------------------------------------ generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ox.add_arguments(parser, PACKAGE, ocp_runtime.EXPORT_ENV, f"crane_mpc_{TOOL}")
    parser.add_argument(
        "--config-overlay",
        default="",
        metavar="NAME",
        help="parameter file in config/ read over crane_mpc.yaml, the same name "
        "`crane_mpc.launch.py config_overlay:=` takes -- so a lab-day variant "
        "(labday_candidate.yaml, labday_fallback_a.yaml, labday_fallback_b.yaml) "
        "is exported and launched off one file instead of an edit of the "
        "committed one. Only a variant that moves the export key needs it: the "
        "weights go in through the runtime API and are not in that key",
    )
    arguments = parser.parse_args()

    parameters = ox.read_ros_parameters(
        PACKAGE / "config" / "crane_mpc.yaml", "crane_mpc"
    )
    if arguments.config_overlay:
        overlay = ocp_config.overlay_path(arguments.config_overlay)
        ocp_config.apply_overlay(
            parameters, ox.read_ros_parameters(overlay, "crane_mpc"), str(overlay)
        )
        print(f"over the committed yaml: {overlay}")
    # the constants live in crane_model; nothing here repeats them
    hydraulics = hydraulic_limits()
    # the box and u^+ are crane_model's, same as the node reads them
    parameters["limits"].update(ocp_config.machine_limits())

    description_xml = (arguments.descriptions / DESCRIPTION).read_text()
    # same file `load_actuator_fit` reads, as a document: the checks want the
    # fit's identity, not just the three numbers the dynamics use
    fit = read_fit(Path(cs.default_actuator_path()))
    # the identification is guarded whichever model the OCP is posed on: the
    # machine is still described by this file and `crane_planning` inverts it
    check_fit_is_one_artifact(fit)
    # `ocp_actuator_fit`, the node's own parameter, so the two compile one thing
    description_xml, design = ocp_actuator_model(
        parameters["ocp_actuator_fit"], description_xml
    )
    posed_on = fit if design is None else read_fit(design)
    # the resolved path, not the name: `config/` exists twice (installed share
    # first) and the key is over the content of whichever copy was read
    label = FIT_LABEL if design is None else str(design)
    if design is not None:
        check_design_fit(posed_on, fit)
    ocp, _scale, model, _chamber = build_ocp(
        description_xml,
        parameters,
        hydraulics,
        None if design is None else cs.load_actuator_fit(design),
    )
    # before code generation, so a bad fit stops the export rather than
    # complaining about a solver it already compiled
    check_fit(model, posed_on, label)

    root = ox.export(
        ocp,
        arguments.output,
        ocp_runtime.export_key(parameters, hydraulics, description_xml, design),
        sims=ocp_runtime.predictor_sims(ocp, parameters),
        verbose=arguments.verbose,
    )
    print(f"exported and compiled into {root}")
    if design is not None:
        print(f"posed on {label}, a design margin and not the identification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
