"""
What the launch has to keep true, as files rather than as a running graph.

This is the oracle for the two things a sim run would not notice: a forgotten
`hydraulic_limits.yaml`, whose generated defaults equal the shipped values, and
a merge order that decides a value by accident.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent.parent
LAUNCH = PACKAGE / "launch" / "crane_mpc.launch.py"
MPC_CONFIG = PACKAGE / "config" / "crane_mpc.yaml"
HYDRAULIC_LIMITS = PACKAGE / "config" / "hydraulic_limits.yaml"
# Package root, not `src/`: the declaration moved there with the ament_python
# conversion.
DECLARATION = PACKAGE / "crane_mpc_parameters.yaml"
NODE_NAME = "crane_mpc"
SHADOW = "shadow"


def _source():
    return LAUNCH.read_text(encoding="utf-8")


def _node_call():
    """The one `Node` call of the launch. Two MPCs in one launch is issue 107's failure."""
    calls = [
        node
        for node in ast.walk(ast.parse(_source()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Node"
    ]
    assert len(calls) == 1, "one horizon producer, one launch"
    return calls[0]


def _keyword(call, name):
    return next(word.value for word in call.keywords if word.arg == name)


def _strings(node):
    """String literals under `node`. The launch writes this node's arguments out, not via names."""
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _flat(document, prefix=""):
    """Dotted parameter names, the form the merge happens in."""
    names = set()
    for key, value in document.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            names |= _flat(value, f"{name}.")
        else:
            names.add(name)
    return names


def _parameters(document):
    return document[NODE_NAME]["ros__parameters"]


def test_the_launch_starts_the_node_with_both_configuration_files():
    """Both files, in merge order, neither of them optional."""
    call = _node_call()
    assert _strings(_keyword(call, "package")) == [NODE_NAME]
    assert _strings(_keyword(call, "executable")) == ["crane_mpc_node"]
    # The node name is the key both files are written under, so a rename applies
    # neither of them.
    assert _strings(_keyword(call, "name")) == [NODE_NAME]
    assert yaml.safe_load(MPC_CONFIG.read_text(encoding="utf-8")).keys() == {NODE_NAME}
    assert yaml.safe_load(HYDRAULIC_LIMITS.read_text(encoding="utf-8")).keys() == {
        NODE_NAME
    }

    parameters = _keyword(call, "parameters")
    assert isinstance(parameters, ast.List)
    files = [
        [name for name in _strings(element) if name.endswith(".yaml")]
        for element in parameters.elts
    ]
    assert [name for group in files for name in group] == [
        MPC_CONFIG.name,
        HYDRAULIC_LIMITS.name,
    ], "the machine limits go last, so they win a key the other file ever grows"
    assert MPC_CONFIG.is_file() and HYDRAULIC_LIMITS.is_file()

    # Unconditional: nothing in this launch can leave either file out.
    assert "Condition" not in _source()


def test_neither_configuration_file_can_decide_the_others_values():
    """Disjoint, which is what makes the merge order above cost nothing."""
    mpc = _flat(_parameters(yaml.safe_load(MPC_CONFIG.read_text(encoding="utf-8"))))
    limits = _flat(
        _parameters(yaml.safe_load(HYDRAULIC_LIMITS.read_text(encoding="utf-8")))
    )
    assert not mpc & limits, sorted(mpc & limits)

    # And every declared hydraulic limit is carried by the machine file, so a new
    # one cannot ship on a generated default that nobody would notice.
    declared = yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))[NODE_NAME][
        "hydraulics"
    ]
    assert limits == {f"hydraulics.{name}" for name in declared}


def test_the_launch_leaves_the_node_in_shadow():
    """Turning the shadow off is a separate, deliberate act (issue 146)."""
    assert (
        _parameters(yaml.safe_load(MPC_CONFIG.read_text(encoding="utf-8")))["mode"]
        == SHADOW
    )
    source = _source()
    assert '"mode"' not in source and "'mode'" not in source
