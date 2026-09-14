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
