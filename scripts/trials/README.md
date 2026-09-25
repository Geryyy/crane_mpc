# Trial harness — issue 161

Scoring an MPC arm on final goal error does not work: identical configuration
gives 145.5, 1010.0 and 3288.3 mrad, because the end state of an exponentially
growing mode depends on *when* it saturates. Three "fixes" were accepted and
then withdrawn on that metric. Score on the growth rate instead.

- `closedloop.py` — the node's feedback loop offline, no ROS, no Gazebo, the
  OCP's own model as the plant. Seconds per arm, deterministic. Run this first:
  it separates a fault in the loop from a plant/model mismatch.
- `nulltest.py` — the OCP alone, at rest, asked to hold. Seed the passive pair
  from `passive_equilibrium`, or the solver is indicted for the seed's own
  1.57 rad error.
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
