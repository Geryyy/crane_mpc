"""
The export's guards on the C3 fit, failure paths only.

The happy path is `test_generated_is_current.py`: it runs `export_ocp.py
--check`, so a fit that passes these guards and a tree that matches it are
checked there. What is *not* checked there is a bad fit, because a bad fit is
never checked in -- hence this file.

`check_fit` is given a stub rather than a `CraneSymbolicModel`: it reads two
attributes, and building the real model means pinocchio, the URDF and a second
of work for no extra coverage.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

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


@pytest.fixture
def fit() -> dict:
    return eo.read_fit(Path(cs.default_actuator_path()))


@pytest.fixture
def model(fit) -> Model:
    """What the shipped fit builds: `k`, `tau_v`, and a description matching `d`."""
    axes = [fit["axes"][key] for key in cs.K_AXIS_KEYS]
    damping = np.zeros(max(cs.K_PLANNED_ROWS) + 1)
    for axis, row in enumerate(cs.K_PLANNED_ROWS):
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
    axis = cs.K_AXIS_KEYS.index("sa")
    values = list(getattr(model.actuator, name))
    values[axis] = np.inf
    model.actuator = dataclasses.replace(model.actuator, **{name: tuple(values)})
    with pytest.raises(ValueError, match=rf"axis 'sa' has {name}=inf"):
        eo.check_fit(model, fit)


def test_damping_from_another_fit_is_refused(model, fit):
    """`d` and `k` are one identification; `d` arrives via the description."""
    model.description.damping[cs.K_PLANNED_ROWS[cs.K_AXIS_KEYS.index("ro")]] *= 2.0
    with pytest.raises(ValueError, match=r"axis 'ro': the description damps it at"):
        eo.check_fit(model, fit)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_non_finite_d_is_refused_by_axis_name(model, fit, value):
    """
    A NaN `d` used to *pass*: the tolerance comparison is `nan > nan`, i.e. False.

    Nothing upstream covers `d` -- `load_actuator_fit` never reads it, because it
    reaches the dynamics through the description -- so this guard is the only one.
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
    The copy `load_actuator_fit` prefers is the installed one.

    Editing the source fit and rebuilding only `crane_mpc` leaves the export
    reading the stale install, so `k` and `d` come from different revisions with
    every other guard green. Only comparing the copies catches it.
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
    # Pinned to a path that does not exist, so this asserts the wiki copy alone.
    monkeypatch.setattr(eo, "SOURCE_FIT", tmp_path / "absent.json")
    with pytest.raises(ValueError, match=r"one identification in two files"):
        eo.check_fit_is_one_artifact(fit)


def test_the_digest_is_over_the_numbers_and_not_the_bytes(fit):
    """The two copies differ by a trailing newline and are the same fit."""
    assert eo.fit_digest(fit) == eo.fit_digest(json.loads(json.dumps(fit, indent=4)))
    # `gr.d` is a number nothing in the model reads, and it still moves the
    # digest -- that is what makes a refit fail `--check` rather than ship.
    moved = copy.deepcopy(fit)
    moved["axes"]["gr"]["d"] = 99.0
    assert eo.fit_digest(moved) != eo.fit_digest(fit)
