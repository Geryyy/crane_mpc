"""`generated/` is still what `scripts/export_ocp.py` writes."""

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent


def test_the_checked_in_solver_is_current():
    """
    The staleness gate, re-homed.

    Was a CMake `ALL` target until issue 133; `generated/` is compiled by
    nobody (the node builds its own solver at startup), so this test is the
    only reason a diff of that tree gets reviewed.

    Subprocess, not import: the exporter puts `crane_ocp/scripts` on
    `sys.path` and regenerates a solver, none of which belongs in-process.
    """
    check = subprocess.run(
        [sys.executable, str(PACKAGE / "scripts" / "export_ocp.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
