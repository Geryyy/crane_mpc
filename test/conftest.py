"""
Every test that opens a solver needs one exported for its own configuration.

Nothing builds a solver at startup any more, so a test that varies `Ts` or the
horizon is asking for an artifact that does not exist yet -- and `Ocp` refuses
rather than quietly compiling one, which is the property under test elsewhere.
`export_for` is how a fixture says "export this, once per session", keyed by the
same `export_key` the runtime checks, so two tests on one configuration compile
it once.

`shipped_init_args` is the other half: the declaration carries no defaults, so a
node gets its values the one way a deployment does -- the shipped config file.
"""

import os
from pathlib import Path

import crane_ocp_export as ox
import pytest
from ament_index_python.packages import get_package_share_directory
from crane_mpc import problem
from crane_mpc import solver as ocp_runtime

#: Configurations already exported this session, by key digest.
_DONE: set = set()


def shipped_config() -> Path:
    """Locate the installed `config/crane_mpc.yaml`, the only source of the node's values."""
    return Path(get_package_share_directory("crane_mpc")) / "config" / "crane_mpc.yaml"


def shipped_init_args() -> list:
    """
    `rclpy.init` arguments that hand every node in the context the shipped values.

    `--params-file`, the way the launch passes it, rather than
    `parameter_overrides`: the declaration has no defaults, so a test building a
    node without this gets the same `ParameterUninitializedException` a
    deployment that forgot the file would.
    """
    return ["--ros-args", "--params-file", str(shipped_config())]


@pytest.fixture(scope="session")
def export_base(tmp_path_factory) -> Path:
    """One base for the whole session; each problem gets its own directory in it."""
    base = tmp_path_factory.mktemp("exports")
    os.environ[ocp_runtime.EXPORT_ENV] = str(base)
    return base


def export_for(
    base: Path, parameters: dict, hydraulics: dict, description: str
) -> Path:
    """Export this problem unless the session already did."""
    os.environ[ocp_runtime.EXPORT_ENV] = str(base)
    key = ocp_runtime.export_key(parameters, hydraulics, description)
    digest = ox.key_digest(key)
    if digest not in _DONE:
        ocp = problem.build_ocp(description, parameters, hydraulics)[0]
        ox.export(
            ocp,
            ocp_runtime.export_base(),
            key,
            sims=ocp_runtime.predictor_sims(ocp, parameters),
        )
        _DONE.add(digest)
    return ox.solver_root(ocp_runtime.export_base(), key)
