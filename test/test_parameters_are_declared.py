"""
A shipped key nobody declared is dropped in silence.

`generate_parameter_library` reads the declaration and `rclpy` discards what does
not match it -- no warning, no log, and the default runs. Issue 137's divergence
is not that (a declared key with two values, which is sometimes correct and is
therefore not under test here); this is the cheaper half of the same gap, and it
is the one a typo produces.
"""

from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent.parent
DECLARATION = PACKAGE / "crane_mpc_parameters.yaml"
SHIPPED = ("crane_mpc.yaml", "hydraulic_limits.yaml")


def _declared(node, prefix=""):
    """Dotted names of the declaration's leaves. A leaf is what carries `type`."""
    for key, value in node.items():
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("type"), str):
            yield prefix + key
        else:
            yield from _declared(value, f"{prefix}{key}.")


def _written(node, prefix=""):
    """Dotted names a config file sets. Its leaves are scalars and lists."""
    for key, value in node.items():
        if isinstance(value, dict):
            yield from _written(value, f"{prefix}{key}.")
        else:
            yield prefix + key


def test_every_shipped_key_is_a_declared_parameter():
    declared = set(_declared(yaml.safe_load(DECLARATION.read_text())["crane_mpc"]))
    written = set()
    for name in SHIPPED:
        document = yaml.safe_load((PACKAGE / "config" / name).read_text())
        written |= set(_written(document["crane_mpc"]["ros__parameters"]))
    assert written - declared == set()


# --- the pump is one number ---------------------------------------------------
#
# `pump_flow_max` and `pump_flow_planning_factor` are also `crane_planning`'s,
# and both packages turn them into an actuator limit. They had been typed into
# each separately -- 1.4e-3 here, 0.0014 there -- with nothing comparing them.
# `crane_model/config/hydraulics.yaml` is the source; this pins both copies.
def test_the_pump_is_the_one_in_crane_model():
    from crane_model.conventions import default_hydraulics_path

    with open(default_hydraulics_path(), encoding="utf-8") as handle:
        pump = yaml.safe_load(handle)["pump"]
    declared = yaml.safe_load(DECLARATION.read_text())["crane_mpc"]["hydraulics"]
    deployed = yaml.safe_load(
        (PACKAGE / "config" / "hydraulic_limits.yaml").read_text()
    )
    deployed = deployed["crane_mpc"]["ros__parameters"]["hydraulics"]

    for here, there in (
        ("pump_flow_max", "flow_max"),
        ("pump_flow_planning_factor", "planning_factor"),
    ):
        assert declared[here]["default_value"] == pump[there], here
        assert deployed[here] == pump[there], here


# --- the control-safe box is one box ------------------------------------------
#
# The same rows bound `crane_planning`, which used to read its box off the
# description and so certified poses and speeds constraint 1 refuses. The box
# lives in crane_model now; this pins the declaration and the deployed config to
# it, in the actuated order of wiki/nomenclature.md 4.
ACTUATED = ("slewing", "boom", "arm", "telescope", "rotator", "tool")
BOX_ROWS = ("q_a_lower", "q_a_upper", "dq_a_max", "q_a_margin")


def test_the_control_safe_box_is_the_one_in_crane_model():
    from crane_model.conventions import control_safe_limits

    box = control_safe_limits()
    declared = yaml.safe_load(DECLARATION.read_text())["crane_mpc"]["limits"]
    deployed = yaml.safe_load((PACKAGE / "config" / "crane_mpc.yaml").read_text())
    deployed = deployed["crane_mpc"]["ros__parameters"]["limits"]

    for row in BOX_ROWS:
        expected = [box[row][axis] for axis in ACTUATED]
        assert declared[row]["default_value"] == expected, row
        assert deployed[row] == expected, row
