import os
import sys
from glob import glob
from pathlib import Path

from setuptools import setup

PACKAGE = "crane_mpc"

# The parameter declaration as a Python module, generated from the one yaml so a
# parameter cannot be declared in one place and read in another. `node.py`
# imports it as `crane_mpc.parameters`; before issue 133 the same module came
# out of CMake's `generate_parameter_module`.
#
# Deliberately **not** `generate_parameter_library_py.setup_helper`. It locates
# its output by re-deriving the workspace from `--build-directory`, which is
# wrong whenever colcon builds into a base that is not `<ws>/build` -- ralph's
# `RALPH_ISOLATE_BUILD=1` does exactly that, and the write then lands in another
# run's install tree -- and when it finds neither `--build-directory` nor
# `--build-base` it generates *nothing at all* and the package ships without the
# module. Here the destination comes from `__file__`, so there is no path to
# guess and no silent no-op: a generator that cannot run raises.
#
# The module goes into the source package. Under `--symlink-install` that is the
# only importable location anyway (the installed package is a symlink to it), and
# one destination means the node, `ros2 run` and pytest -- which colcon runs with
# the source directory as its working directory -- all import the same complete
# package. It is gitignored.
_BUILD_COMMANDS = {
    "build",
    "build_py",
    "bdist_wheel",
    "develop",
    "editable_wheel",
    "install",
}
if _BUILD_COMMANDS.intersection(sys.argv[1:]):
    # Skipped for colcon's metadata dry run, which has no build command and reads
    # this file's stdout as the package manifest.
    from generate_parameter_library_py.generate_python_module import (
        run as generate_module,
    )

    # Resolves through colcon's build-space symlink back to the source.
    SOURCE = Path(__file__).resolve().parent
    MODULE = SOURCE / PACKAGE / "parameters.py"
    # Generated beside the module and renamed onto it. Two colcon runs sharing
    # this checkout -- which `RALPH_ISOLATE_BUILD=1` allows, since it isolates
    # `build/` and `install/` and not `src/` -- write the same bytes, but a
    # reader that imports a half-written file fails in whichever run did not
    # write it.
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
    # colcon picks its pytest step off `tests_require`, not off package.xml, so
    # without this line `test/` is never run.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Architecture maintainers",
    maintainer_email="maintainers@example.invalid",
    description="The acados RTI MPC node: it solves the crane_model OCP "
    "and publishes `/crane/mpc/horizon` at 25 Hz.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            # The name is unchanged from when the package was C++, minus the
            # `.py` that `install(PROGRAMS ...)` forced: it is a cross-node
            # contract (`wiki/implementation/ros2_interfaces.md` §8), not a
            # detail. `crane_planning` made the same move.
            f"crane_mpc_node = {PACKAGE}.node:main",
        ],
    },
)
