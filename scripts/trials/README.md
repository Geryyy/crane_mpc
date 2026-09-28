# Trial harness — issue 161

Scoring an MPC arm on final goal error does not work: identical configuration
gives 145.5, 1010.0 and 3288.3 mrad, because the end state of an exponentially
growing mode depends on *when* it saturates. Three "fixes" were accepted and
then withdrawn on that metric. Score on the growth rate instead.

- `wire_chain.py --goal out [--latency S]` — the node's `Cycle` closed through
  the published `JointTrajectory` and a `trajectory.cpp` playback (cubic pos/vel,
  linear effort, PI at the tick, feedforward a tick later) into MuJoCo (issue
  171); settings, solver, plan and plant are `harness.py`'s. Deterministic: a
  repeat matches to the digit except `solve_time_s`. Stable at 0/20/40 ms.
  Regression check, each must exit 1: `--wire 161 --set command_state=false --set weights.du=10` (old encoding on C3, hunts; under C4 it is nearly harmless) and `--cadence-phase 0.02`
  (once-anchored grid, diverges). ~10 s a run; a script, not a pytest, because
  the test session exports its own solver.
- `wire_chain.py --goal out --viewer [--realtime 0]` — the same run in MuJoCo's
  viewer; past the plan it keeps cycling, unscored, so the settling sway shows.
  Space ends that, closing the window ends the run. `--realtime 0` is as fast as
  it computes.
- `wire_chain.py --random [N] --viewer [--seed S]` — chain N random moves (no N:
  until the window closes), each from where the last ended; space releases the
  next. Headless, `--random N` runs the N moves back to back.
- `sweep_robust.py` — `wire_chain` over moves x plant samples x levers; the worst
  cell per setting.
- `trial.sh <outdir>` — one live arm: tear down by PID, launch headless,
  activate `trajectory_controller_a2b`, call `/a2b_movement`, capture.
  `CTRL=pid MOVE=<script>` swaps the arm. ~3 min.
- `capture.py` — records `/joint_states`, `/crane/mpc/horizon` knot 0 and
  `/crane/mpc/solver_health`. Joints by `name[]`, never by index.
- `growth.py <cap.json> <label> ...` — the score: fit `log|q_u - q_eq|` against
  sim time over the first rise. `<= 0` is stable. Two runs per arm, worst reported.
- `steptest.py <out.json>` — one open-loop velocity step per axis on a live
  sim, MPC killed, recording position, velocity and applied torque.
- `plant_step.py <out.json> [joint]` — the same step on every plant we have,
  side by side: Gazebo (from that capture), MuJoCo through `C3Actuator`, and the
  OCP's own integrator. They agreed to 0.01 through 500 ms once issue 161's
  three actuator faults were fixed; before that Gazebo was alone.
- `modelfit.py bag '<glob>' | capture <cap.json>` — one-cycle prediction error of
  the MPC's own model against what the crane actually did, per axis, with a
  skill score against "nothing changes". Works on the HydraulicCalib bags (real
  machine) and on a live capture. Score against the command the machine
  *received* (`controller_state.output`), not the one the MPC asked for — on the
  same run the latter reads 20x worse.
- `score.py` — cadence, command chatter, travel, arrival. Context, not the score.
- `pid_move.py` — the same plan with the MPC off: the stable reference arm.
