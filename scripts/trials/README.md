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
- `score.py` — cadence, command chatter, travel, arrival. Context, not the score.
- `pid_move.py` — the same plan with the MPC off: the stable reference arm.
