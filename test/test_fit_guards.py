"""
The export's guards on the C3 fit, failure paths only: a bad fit is never
checked in, so `test_generated_is_current.py`'s `export_ocp.py --check` never
covers one.

`check_fit` gets a stub, not a `CraneSymbolicModel`: it reads two attributes;
the real model costs pinocchio, the URDF and a second of work for nothing extra.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from crane_model.conventions import canonical_joints

PACKAGE = Path(__file__).resolve().parent.parent


def _import_export_ocp():
    spec = importlib.util.spec_from_file_location(
        "export_ocp", PACKAGE / "scripts" / "export_ocp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eo = _import_export_ocp()
cs = eo.cs


@dataclasses.dataclass
class Description:
    damping: np.ndarray


@dataclasses.dataclass
class Model:
    actuator: cs.ActuatorFit
    description: Description


#: `ocp_actuator_fit`'s design model, the one file this package ships (issue 311).
DESIGN_FIT = PACKAGE / "config" / "c3_ocp_design.json"


def model_for(document: dict) -> Model:
    """Build the stub `check_fit` reads, with the description returned to `document`."""
    shipped = (eo.DEFAULT_DESCRIPTIONS / eo.DESCRIPTION).read_text()
    tree = ET.fromstring(eo.retuned_description(shipped, document))
    damped = {
        joint.get("name"): float(joint.find("dynamics").get("damping"))
        for joint in tree.iter("joint")
        if joint.find("dynamics") is not None
    }
    names = canonical_joints()
    damping = np.zeros(max(cs.PLANNED_INDICES) + 1)
    for row in cs.PLANNED_INDICES:
        damping[row] = damped[names[row]]
    axes = [document["axes"][key] for key in cs.AXIS_KEYS]
    return Model(
        actuator=cs.ActuatorFit(
            k=tuple(float(entry["k"]) for entry in axes),
            tau_v=tuple(float(entry["tau_v"]) for entry in axes),
            dead_time_s=float(document["dead_time_common_ms"]) * 1.0e-3,
        ),
        description=Description(damping=damping),
    )


@pytest.fixture
def fit() -> dict:
    return eo.read_fit(Path(cs.default_actuator_path()))


@pytest.fixture
def design() -> dict:
    return eo.read_fit(DESIGN_FIT)


@pytest.fixture
def model(fit) -> Model:
    """What the shipped fit builds: `k`, `tau_v`, and a description matching `d`."""
    axes = [fit["axes"][key] for key in cs.AXIS_KEYS]
    damping = np.zeros(max(cs.PLANNED_INDICES) + 1)
    for axis, row in enumerate(cs.PLANNED_INDICES):
        damping[row] = float(axes[axis]["d"])
    return Model(
        actuator=cs.ActuatorFit(
            k=tuple(float(entry["k"]) for entry in axes),
            tau_v=tuple(float(entry["tau_v"]) for entry in axes),
            dead_time_s=float(fit["dead_time_common_ms"]) * 1.0e-3,
        ),
        description=Description(damping=damping),
    )


@pytest.mark.parametrize("name", ["k", "tau_v"])
def test_non_finite_is_refused_by_axis_name(model, fit, name):
    """`load_actuator_fit` lets an infinity through: `inf > 0` is true."""
    axis = cs.AXIS_KEYS.index("sa")
    values = list(getattr(model.actuator, name))
    values[axis] = np.inf
    model.actuator = dataclasses.replace(model.actuator, **{name: tuple(values)})
    with pytest.raises(ValueError, match=rf"axis 'sa' has {name}=inf"):
        eo.check_fit(model, fit)


def test_damping_from_another_fit_is_refused(model, fit):
    """`d` and `k` are one identification; `d` arrives via the description."""
    model.description.damping[cs.PLANNED_INDICES[cs.AXIS_KEYS.index("ro")]] *= 2.0
    with pytest.raises(ValueError, match=r"axis 'ro': the description damps it at"):
        eo.check_fit(model, fit)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_non_finite_d_is_refused_by_axis_name(model, fit, value):
    """
    A NaN `d` used to pass: `nan > nan` is False.

    Nothing upstream covers `d` -- `load_actuator_fit` never reads it, it
    reaches the dynamics via the description -- so this guard is the only one.
    """
    fit["axes"]["ha"]["d"] = value
    with pytest.raises(ValueError, match=r"axis 'ha' has d="):
        eo.check_fit(model, fit)


def test_a_missing_d_is_refused_by_axis_name(model, fit):
    """Without the check this is a KeyError, not the message the guard writes."""
    del fit["axes"]["ka"]["d"]
    with pytest.raises(ValueError, match=r"axis 'ka' carries no d"):
        eo.check_fit(model, fit)


def test_a_stale_install_copy_stops_generation(fit, tmp_path, monkeypatch):
    """
    `load_actuator_fit` prefers the installed copy.

    Editing the source and rebuilding only `crane_mpc` leaves export reading
    the stale install, so `k`/`d` differ by revision with every other guard
    green. Only comparing copies catches it.
    """
    edited = copy.deepcopy(fit)
    edited["axes"]["sw"]["k"] += 1.0
    source = tmp_path / "c3_full_model.json"
    source.write_text(json.dumps(edited))
    monkeypatch.setattr(eo, "SOURCE_FIT", source)
    monkeypatch.setattr(eo, "WIKI_FIT", tmp_path / "absent.json")
    with pytest.raises(ValueError, match=r"one identification in two files"):
        eo.check_fit_is_one_artifact(fit)


def test_the_two_copies_drifting_stops_generation(fit, tmp_path, monkeypatch):
    drifted = copy.deepcopy(fit)
    drifted["axes"]["ro"]["d"] += 1.0
    other = tmp_path / "c3_full_model.json"
    other.write_text(json.dumps(drifted))
    monkeypatch.setattr(eo, "WIKI_FIT", other)
    # Pinned to a nonexistent path, so this asserts the other copy alone.
    monkeypatch.setattr(eo, "SOURCE_FIT", tmp_path / "absent.json")
    with pytest.raises(ValueError, match=r"one identification in two files"):
        eo.check_fit_is_one_artifact(fit)


def test_a_scaled_fit_keeps_d_with_k_through_the_description(fit):
    """
    The bench's stiffer OCP (issue 307): `k` and `d` have to move together.

    `d` reaches the dynamics only through the description, so a fit file with
    boom/arm `k, d` x2 exported against the *shipped* description is a
    damping-ratio error nothing else would notice. `check_fit` is the guard, so
    the rewrite has to satisfy it on all five planned axes, not the two that
    moved.
    """
    scaled = copy.deepcopy(fit)
    for key in ("ha", "ka"):
        scaled["axes"][key]["k"] *= 2.0
        scaled["axes"][key]["d"] *= 2.0
    model = model_for(scaled)
    eo.check_fit(model, scaled)
    # and it is a different plant, not a no-op rewrite
    boom = model.description.damping[cs.PLANNED_INDICES[cs.AXIS_KEYS.index("ha")]]
    assert boom == pytest.approx(2.0 * float(fit["axes"]["ha"]["d"]))


def test_a_planned_joint_it_cannot_retune_is_refused(fit):
    """
    The bench path never runs `check_fit`, so the rewrite guards itself.

    A planned joint this cannot reach keeps the *shipped* damping while the
    OCP takes the new `k`, and nothing downstream reads the attribute again.
    """
    shipped = (eo.DEFAULT_DESCRIPTIONS / eo.DESCRIPTION).read_text()
    arm = canonical_joints()[cs.PLANNED_INDICES[cs.AXIS_KEYS.index("ka")]]
    without = shipped.replace('<dynamics damping="4.700e+04" friction="0"/>', "", 1)
    assert f'name="{arm}"' in without and without != shipped
    with pytest.raises(ValueError, match=rf"joint '{arm}' carries no <dynamics>"):
        eo.retuned_description(without, fit)
    # and the infinity `load_actuator_fit` lets through
    infinite = copy.deepcopy(fit)
    infinite["axes"]["sa"]["k"] = float("inf")
    with pytest.raises(ValueError, match=r"axis 'sa' has k=inf"):
        eo.retuned_description(shipped, infinite)


def test_the_shipped_design_model_is_the_identification_with_scaled_axes(design, fit):
    """
    `ocp_actuator_fit`'s file (issue 311), against both guards it has to pass.

    Its own rule first -- it is `c3_full_model.json` with boom/arm `k, d` x2 and
    nothing else moved -- and then `check_fit` on the description returned to it,
    which is the pairing the export actually generates from.
    """
    eo.check_design_fit(design, fit)
    eo.check_fit(model_for(design), design, "config/c3_ocp_design.json")
    for key in cs.AXIS_KEYS:
        factor = 2.0 if key in ("ha", "ka") else 1.0
        for field in ("k", "d"):
            assert design["axes"][key][field] == pytest.approx(
                factor * float(fit["axes"][key][field])
            ), (key, field)
        # `tau_v` fixes `LAG_AXES` and `NX`, which come off the shipped fit
        assert design["axes"][key]["tau_v"] == fit["axes"][key]["tau_v"], key


def test_a_design_model_that_is_not_a_scaled_fit_is_refused(design, fit):
    """
    Each way a design file stops being a margin on the identification.

    `k` without `d` is the one that compiles cleanly and rings (issue 308): it
    is a stiffer spring rather than a velocity source with the machine's own
    `tau = d/k`, and nothing downstream reads either number again.
    """
    stiffer = copy.deepcopy(design)
    stiffer["axes"]["ha"]["k"] *= 2.0
    with pytest.raises(ValueError, match=r"design axis 'ha': k is x4"):
        eo.check_design_fit(stiffer, fit)

    lagged = copy.deepcopy(design)
    lagged["axes"]["sw"]["tau_v"] += 0.01
    with pytest.raises(ValueError, match=r"design axis 'sw': tau_v is"):
        eo.check_design_fit(lagged, fit)

    softer = copy.deepcopy(design)
    for field in ("k", "d"):
        softer["axes"]["ka"][field] *= 0.25
    with pytest.raises(ValueError, match=r"design axis 'ka': k is x0.5"):
        eo.check_design_fit(softer, fit)

    # scaled together, so `tau = d/k` holds, and still refused: 307's gate put
    # the admissible band at x2..x4 and an OCP x12 exits past_clamp on plant x1
    wide = copy.deepcopy(design)
    for field in ("k", "d"):
        wide["axes"]["ha"][field] *= 3.0
    with pytest.raises(ValueError, match=r"design axis 'ha': k is x6 .*past the x4"):
        eo.check_design_fit(wide, fit)

    # `dead_time_common_ms` reaches ActuatorFit and check_fit only tests finiteness
    slow = copy.deepcopy(design)
    slow["dead_time_common_ms"] = 600.0
    with pytest.raises(ValueError, match=r"dead_time_common_ms is 600"):
        eo.check_design_fit(slow, fit)

    # `gr` is outside AXIS_KEYS, so retuned_description drops its d: a margin
    # written there claims something the OCP never gets
    tool = copy.deepcopy(design)
    for field in ("k", "d"):
        tool["axes"]["gr"][field] *= 2.0
    with pytest.raises(ValueError, match=r"design axis 'gr' .*not a planned axis"):
        eo.check_design_fit(tool, fit)

    undeclared = copy.deepcopy(design)
    del undeclared[eo.DESIGN_FIELD]
    with pytest.raises(ValueError, match=r"carries no 'design_margin'"):
        eo.check_design_fit(undeclared, fit)
    # and the identification itself is not a design model: it says nothing
    with pytest.raises(ValueError, match=r"carries no 'design_margin'"):
        eo.check_design_fit(fit, fit)


def test_the_digest_is_over_the_numbers_and_not_the_bytes(fit):
    """The two copies differ by a trailing newline and are the same fit."""
    assert eo.fit_digest(fit) == eo.fit_digest(json.loads(json.dumps(fit, indent=4)))
    # `gr.d` is unread by the model but still moves the digest -- what makes
    # a refit fail `--check` rather than ship.
    moved = copy.deepcopy(fit)
    moved["axes"]["gr"]["d"] = 99.0
    assert eo.fit_digest(moved) != eo.fit_digest(fit)
