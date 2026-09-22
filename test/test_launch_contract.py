"""
What the launch has to keep true, as files rather than as a running graph.

Oracle for what a sim run wouldn't notice: a config file left out or keyed under
a renamed node, and a default that quietly leaves shadow.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent.parent
LAUNCH = PACKAGE / "launch" / "crane_mpc.launch.py"
MPC_CONFIG = PACKAGE / "config" / "crane_mpc.yaml"
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


def _parameters(document):
    return document[NODE_NAME]["ros__parameters"]


def test_the_launch_starts_the_node_with_its_configuration_file():
    """One file, and it is not optional."""
    call = _node_call()
    assert _strings(_keyword(call, "package")) == [NODE_NAME]
    assert _strings(_keyword(call, "executable")) == ["crane_mpc_node"]
    # Node name is the key the file is written under; a rename applies none of it.
    assert _strings(_keyword(call, "name")) == [NODE_NAME]
    assert yaml.safe_load(MPC_CONFIG.read_text(encoding="utf-8")).keys() == {NODE_NAME}

    parameters = _keyword(call, "parameters")
    assert isinstance(parameters, ast.List)
    files = [
        [name for name in _strings(element) if name.endswith(".yaml")]
        for element in parameters.elts
    ]
    assert [name for group in files for name in group] == [MPC_CONFIG.name]
    assert MPC_CONFIG.is_file()

    # Unconditional: nothing in this launch can leave the file out.
    assert "Condition" not in _source()


def test_the_launch_leaves_the_node_in_shadow():
    """Turning the shadow off is a separate, deliberate act (issue 146).

    Asserts on the `mode` argument's default, not the word's absence: nothing
    an including profile forgets to pass may leave shadow.
    """
    assert (
        _parameters(yaml.safe_load(MPC_CONFIG.read_text(encoding="utf-8")))["mode"]
        == SHADOW
    )
    declaration = next(
        node
        for node in ast.walk(ast.parse(_source()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "DeclareLaunchArgument"
        and node.args
        and getattr(node.args[0], "value", None) == "mode"
    )
    assert _strings(_keyword(declaration, "default_value")) == [SHADOW]
    assert _strings(_keyword(declaration, "choices")) == [SHADOW, "active"]


def test_the_follower_stream_is_remappable_and_defaults_to_the_contract_name():
    """Shadow mode is judged on it, and a profile that renames it says nothing.

    Without it the node subscribes a name no bringup reusing the timber
    spawners publishes; comparison reports `follower.velocity_source: none`
    while the solver looks healthy.
    """
    source = _source()
    declaration = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "DeclareLaunchArgument"
        and node.args
        and getattr(node.args[0], "value", None) == "controller_state_topic"
    )
    assert _strings(_keyword(declaration, "default_value")) == [
        "/crane/controller_state"
    ]
    assert '("/crane/controller_state", controller_state_topic)' in source


def test_the_start_gate_is_declared_and_defaults_to_no_gate():
    """A gate that defaulted on would be a tree that plans and never moves.

    The behaviour tree sends no FollowJointTrajectory goal whenever the MPC
    drives (issue 148), so empty string is the only safe default, read as no
    gate at all.
    """
    source = _source()
    declaration = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "DeclareLaunchArgument"
        and node.args
        and getattr(node.args[0], "value", None) == "start_signal_action"
    )
    assert _strings(_keyword(declaration, "default_value")) == [""]
    assert '"start_signal_action": ParameterValue(' in source
