"""`generated/` is still what `scripts/export_ocp.py` writes."""

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent


def test_the_checked_in_solver_is_current():
    """
    The staleness gate, re-homed.

    It was a CMake `ALL` target until issue 133 and could not survive the
    `ament_python` conversion as one. Losing it was the alternative: the same
    check is documented as "what CI runs" in `crane_planning`, and nothing runs
    it there. `generated/` is compiled by nobody -- the node builds its own
    solver at startup -- so this test is the whole reason a diff of that tree is
    worth reviewing.

    A subprocess rather than an import: the exporter is a script, it puts
    `crane_ocp/scripts` on `sys.path` and it regenerates a solver, none of which
    belongs in the test process.
    """
    check = subprocess.run(
        [sys.executable, str(PACKAGE / "scripts" / "export_ocp.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
