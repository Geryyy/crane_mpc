import os
import sys
from glob import glob
from pathlib import Path

from setuptools import setup

PACKAGE = "crane_mpc"

# Parameter declaration as a Python module, generated from the one yaml so
# `node.py`'s `crane_mpc.parameters` import can't diverge from it (before
# issue 133: CMake did this).
#
# Not `generate_parameter_library_py.setup_helper`: it re-derives the
# workspace from `--build-directory`, wrong under `RALPH_ISOLATE_BUILD=1`,
# and silently generates nothing without `--build-directory`/`--build-base`.
# Destination here comes from `__file__` instead, so a broken generator raises.
#
# Goes into the source package (only importable location under
# `--symlink-install`), so node/`ros2 run`/pytest share one package. Gitignored.
_BUILD_COMMANDS = {
    "build",
    "build_py",
    "bdist_wheel",
    "develop",
    "editable_wheel",
    "install",
}
if _BUILD_COMMANDS.intersection(sys.argv[1:]):
    # Skipped for colcon's metadata dry run: no build command, and it reads
    # this file's stdout as the package manifest.
    from generate_parameter_library_py.generate_python_module import (
        run as generate_module,
    )

    # Resolves through colcon's build-space symlink back to the source.
    SOURCE = Path(__file__).resolve().parent
    MODULE = SOURCE / PACKAGE / "parameters.py"
    # Generated beside the module, renamed onto it: two colcon runs sharing
    # this checkout (`RALPH_ISOLATE_BUILD=1` isolates build/install, not src/)
    # write the same bytes; importing a half-written file fails the loser.
    scratch = MODULE.with_name(f"{MODULE.name}.{os.getpid()}")
    generate_module(str(scratch), str(SOURCE / f"{PACKAGE}_parameters.yaml"))
    scratch.replace(MODULE)

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    # colcon picks its pytest step off `tests_require`, not package.xml;
    # without this, `test/` never runs.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Architecture maintainers",
    maintainer_email="maintainers@example.invalid",
    description="The acados RTI MPC node: it solves the crane_model OCP "
    "and publishes `/crane/mpc/horizon` at 25 Hz.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            # Name unchanged from the C++ package, minus the `.py`
            # `install(PROGRAMS ...)` forced: a cross-node contract, not a detail.
            f"crane_mpc_node = {PACKAGE}.node:main",
        ],
    },
)
