# `crane_mpc/generated/` -- the shipped solver

Machine output. **Do not edit any file here**: change
`scripts/export_ocp.py`, `crane_model/scripts/crane_symbolic.py` or one of the
config files below, re-run

    ./scripts/export_ocp.py

and commit what changes. `./scripts/export_ocp.py --check` regenerates into a
scratch tree and diffs, which is what says the tree still matches its inputs.

## What is in here

    crane_mpc_ocp_generated.h        the grid, the residual offsets and the
                                     conditioning divisors, as the export chose
                                     them
    crane_mpc_pzs100/                the PZS100 solver, constrained

**Nothing compiles this tree.** The node builds its own solver at startup, into
`CRANE_MPC_OCP_CACHE`, and never opens what is here; the C++ that did was deleted
by issue 132. The tree is kept as the reviewable form of the OCP -- a diff of it
is how a change to the problem is seen -- and `--check` is kept with it as a
build-time target, because a tree nobody checks is a tree nobody can read as
current.

**One solver.** The description is *baked in*, so a generated solver is one
machine's, and the PZS100 is the machine that has to run.

`docs/features/cbs-ocp-python/grill.md` D6's second artifact is retired too, and
the question it existed to settle is settled: dropping `wiki/mpc.md` §3's
nonlinear cylinder-force and pump-flow rows moves this OCP's solve from 19 QP
iterations and 7.15 ms to 20 and 7.02 ms. Those rows cost essentially nothing,
so there is no case for shipping a second artifact without them.

Inside the solver directory, `crane_mpc_pzs100_output.{c,h}` is not acados' --
it is the output map of `wiki/nomenclature.md` §10 code-generated beside the
solver, all six axes of `tau_a`, `F_cyl`, `v` and `Q`. acados generates only
what it solves, which is five force rows and one pump row already divided by
their conditioning constants; `wiki/mpc.md` §5.3 requirement 4 wants the
residuals in physical units, so they are shipped too.

acados' own generated `Makefile`, `main_*.c`, `acados_sim_solver_*` and
`acados_solver.pxd` are pruned by the export. The `Makefile` is the only
generated file that carries an absolute path, and the `main_*.c` carry a `main`.

## There is no staleness guard, and that is a decision

The conditioning divisors and the smoothing widths come from
`config/hydraulic_limits.yaml` and `crane_model/config/hydraulics.yaml`; the
dynamics come from `pzs100.urdf` under `crane_model/test/description/`.
**Nothing in the build or the test suite fails when one of those files and this
tree disagree.**

`docs/features/cbs-ocp-python/grill.md` §4 records the alternative that was
considered and rejected -- hashing the inputs into the generated code and
comparing in a test, which is the pattern `crane_model`'s own fixture uses --
and records that this is therefore a deliberate
divergence from both existing generator scripts in this repository. The failure
mode it accepts is a solver silently running the previous hydraulic constants or
the previous description.

One thing narrows it, and it does not close it: **`config/crane_mpc.yaml` is not
in the exposed surface at all.** `N`, `T_s`, the weights, the box limits, the
slack prices and the Levenberg-Marquardt term are every one of them set on the
generated solver at configure time -- `N` and `T_s` through acados' own
`_acados_create_with_discretization`, which grill D5 did not expect -- so moving
any of them needs no re-export. `crane_mpc_ocp_generated.h` carries the grid the
export shipped as a *default* and not as a contract.

What is left, and unguarded, is the **structure**: the hydraulic constants that
went into `h`, the conditioning divisors, and the description the dynamics were
built from. Those three files are the staleness surface.
