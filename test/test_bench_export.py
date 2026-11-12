"""Where `wire_chain` exports: its own tree, never the one the node opens."""

import importlib
import sys
from pathlib import Path

import pytest
from crane_mpc import solver as ocp_runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wire_chain"))

# the harness pulls `crane_planning` and MuJoCo, which this package does not
# depend on; without them there is no bench to pin a tree for
harness = pytest.importorskip("harness")


def test_the_bench_never_exports_where_the_node_opens(monkeypatch):
    """
    Separation is the guard: `create_solver` exports on `check_settings` alone.

    No `check_fit_is_one_artifact`, no `check_design_fit`, no `check_fit`, and
    what it writes carries the node's own `export_key` -- so in one tree "the
    key matches an export" stops meaning "a fit guard read this model"
    (issue 315). Taken on the key that can actually collide: `ocp_fit()` with no
    `--ocp-fit` is the shipped `ocp_actuator_fit`, the node's own model.
    """
    monkeypatch.delenv(ocp_runtime.EXPORT_ENV, raising=False)
    node = ocp_runtime.export_base()
    importlib.reload(harness)  # importing the harness is what moves the lever
    bench = ocp_runtime.export_base()
    assert bench != node and node not in bench.parents

    asked = []
    monkeypatch.setattr(harness.ox, "manifest", lambda base, *_: asked.append(base))
    monkeypatch.setattr(harness.ocp_runtime, "Ocp", lambda *_, **__: None)
    fit = harness.ocp_fit()
    harness.create_solver(*harness.settings(actuator_fit=fit), actuator_fit=fit)
    assert asked == [bench]  # the writer, not just the constant

    importlib.reload(harness)  # a subprocess inherits it; no second leaf
    assert ocp_runtime.export_base() == bench
