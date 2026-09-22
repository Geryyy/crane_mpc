"""
A shipped key nobody declared is dropped in silence.

`generate_parameter_library` reads the declaration; `rclpy` discards anything
that doesn't match it -- no warning, default runs. Issue 137's divergence
(a declared key with two values) is separate; this is the typo half.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

PACKAGE = Path(__file__).resolve().parent.parent
DECLARATION = PACKAGE / "crane_mpc_parameters.yaml"
SHIPPED = ("crane_mpc.yaml",)


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


# --- the hydraulic constants are crane_model's --------------------------------
# Declared here only so a deployment can override, at sentinel 0.0; the numbers
# live in crane_model/config/hydraulics.yaml and nothing shipped repeats one.
HYDRAULICS = ("pump_flow_max", "pump_flow_planning_factor", "system_pressure_pa")


def test_the_hydraulic_constants_are_the_ones_in_crane_model():
    from crane_model import hydraulic_limits
    from crane_mpc import config

    declared = yaml.safe_load(DECLARATION.read_text())["crane_mpc"]["hydraulics"]
    assert set(declared) == set(HYDRAULICS)
    for name in HYDRAULICS:
        assert declared[name]["default_value"] == 0.0, name
    for name in SHIPPED:
        shipped = yaml.safe_load((PACKAGE / "config" / name).read_text())
        assert "hydraulics" not in shipped["crane_mpc"]["ros__parameters"], name

    # The sentinels resolve to crane_model's numbers, which is the whole contract.
    values = SimpleNamespace(
        hydraulics=SimpleNamespace(
            **{name: declared[name]["default_value"] for name in HYDRAULICS}
        )
    )
    assert config.hydraulics_dict(values) == hydraulic_limits()


# --- the machine's own numbers are crane_model's ------------------------------
# Same rows bound `crane_planning`, which used to read its box off the
# description and certified poses/speeds constraint 1 refuses. Neither file may
# carry one: `config.machine_limits` is the only reader, and the ACTUATED order
# below is the order it returns them in.
ACTUATED = ("slewing", "boom", "arm", "telescope", "rotator", "tool")
BOX_ROWS = ("q_a_lower", "q_a_upper", "dq_a_max", "q_a_margin")


def test_the_control_safe_box_is_the_one_in_crane_model():
    from crane_model.conventions import control_safe_limits
    from crane_mpc import config

    box = control_safe_limits()
    resolved = config.machine_limits()
    for row in BOX_ROWS:
        assert resolved[row] == [box[row][axis] for axis in ACTUATED], row

    declared = yaml.safe_load(DECLARATION.read_text())["crane_mpc"]["limits"]
    for name in SHIPPED:
        shipped = yaml.safe_load((PACKAGE / "config" / name).read_text())
        written = shipped["crane_mpc"]["ros__parameters"]["limits"]
        for row in resolved:
            assert row not in declared, row
            assert row not in written, (name, row)


def test_constraint_5_is_the_smaller_of_psis_domain_and_the_speed_bound():
    """
    `u^+` is a product of two crane_model tables, so it is derived, not typed.

    The arm is the row that says so: 0.3059 is Psi's own edge, where the
    shipped number was a hand-rounded 0.305. Every other row is bound by
    `dq_a_max`, so a narrowing there now carries into constraint 5 by itself.
    """
    from crane_model.conventions import ACTUATED_INDICES, canonical_joints
    from crane_model.velocity_loop import load_velocity_loop
    from crane_mpc import config

    gains, _rate_hz = load_velocity_loop()
    joints = canonical_joints()
    resolved = config.machine_limits()
    for axis, index in enumerate(ACTUATED_INDICES):
        clamp = gains[joints[index]]
        assert resolved["u_max"][axis] == min(
            abs(clamp.u_clamp_min), clamp.u_clamp_max, resolved["dq_a_max"][axis]
        )


def test_the_transport_delay_is_the_fitted_dead_time():
    """
    Declared, not read: it is the deployed sensor-to-valve path, which the
    fit's own dead time only happens to equal. Pinned so the two stop agreeing
    loudly rather than quietly.
    """
    from crane_model.symbolic import K_ACTUATOR_FIT

    declared = yaml.safe_load(DECLARATION.read_text())["crane_mpc"]
    assert declared["sensor_to_valve_delay"]["default_value"] == float(
        K_ACTUATOR_FIT.dead_time_s
    )
