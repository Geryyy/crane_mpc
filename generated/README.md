# `crane_mpc/generated/` -- the reviewable form of the OCP

Machine output, do not edit. Change `scripts/export_ocp.py`,
`crane_model/scripts/crane_symbolic.py` or a config file, re-run
`./scripts/export_ocp.py`, commit. `--check` regenerates into a scratch tree
and diffs; `test/test_generated_is_current.py` runs it.

## Three files committed, the rest digested

    README.md
    crane_mpc_ocp_generated.h                          grid, residual offsets,
                                                       divisors, two digests
    crane_mpc_pzs100/acados_solver_crane_mpc_pzs100.h  dimensions

The export writes the whole solver (`--check` needs a tree to compare against);
the rest is gitignored. It was committed until this change: 4.0 MB, 176 kLOC,
141 kLOC of that four `impl_dae_*jac*.c` nobody has ever read. Nothing compiles
this tree -- the node builds its own solver at startup into
`CRANE_MPC_OCP_CACHE`, the C++ that read this went with issue 132.

Dropping the bodies would have dropped a real guard, so it did not:
`CRANE_MPC_OCP_GENERATED_DIGEST` is sha256 over every uncommitted generated
file. A description or hydraulics edit lands in those bodies and nowhere else
here -- measured, a link mass moves `impl_dae_*.c` and `_output.c` while every
number in the header holds still.

**One solver**: the description is baked in, and the PZS100 is what has to run.
`crane_mpc_pzs100_output.{c,h}` is not acados' -- acados generates only the
conditioned rows it solves, so the output map ships beside it for `tau_a`,
`F_cyl`, `v`, `Q` in physical units. `Makefile`, `main_*.c`,
`acados_sim_solver_*` and `acados_solver.pxd` are pruned (absolute path, and a
`main`).

## Not in the guarded surface

`config/crane_mpc.yaml`, deliberately. `N`, `T_s`, weights, boxes, slack prices
and Levenberg-Marquardt are all set at configure time -- `N`/`T_s` through
acados' `_acados_create_with_discretization` -- so none can make this tree
stale. The header's grid is a default, not a contract.
