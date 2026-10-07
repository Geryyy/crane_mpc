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
  cell per setting, `--by-plant` for one row per sample. The plant-mismatch map
  it was run for, and the settings that came out of it, are `robustness.md`
  (issue 190).
- `trial.sh <outdir>` — one live arm: tear down by PID, launch headless,
  activate `trajectory_controller_a2b`, call `/a2b_movement`, capture.
  `CTRL=pid MOVE=<script>` swaps the arm. ~3 min.
- `capture.py <seconds> <out.json> [--sim-time|--no-sim-time]` — records
  `/joint_states`, `/crane/mpc/horizon` knot 0, the two shadow topics
  (`~/shadow_horizon` knot 0 as `shz`, `~/shadow_comparison`'s key/value pairs
  as `cmp`) and `/crane/mpc/solver_health`. Joints by `name[]`, never by index.
  Each row carries three times: wall `t`, the header stamp `sim` and the
  receipt `rx` on the stamps' own clock. `sim - rx` is the only way to a
  latency — wall and sim are different clocks. **The clock is not assumed**:
  with no flag the `use_sim_time` parameter is read as given and printed, and
  the two flags pin it. `--sim-time` is the Gazebo arm below; on the machine it
  stays false.
- `looptiming.py <cap.json>` — of the four numbers issue 189 wants, the two a
  capture actually contains: delivery latency (`--latency`) and the
  controller-state publish rate, with the share of cycles under
  `Timing.jtc_output`'s quorum — taken against the **nominal** tick count, because
  against the measured rate a uniformly decimated stream reads as a healthy slow
  one. Offline; the recording is the live part. It refuses the other two and says
  why:
  - **cadence phase is not observable.** `node.py` stamps the horizon with
    `Timing.stamp_ns()`, which `Timing.anchor` makes identically the measurement
    stamp — so the horizon stamp *is* a `/joint_states` stamp and any offset
    against that set is 0 by construction, on a once-anchored loop as much as a
    healthy one. The live quantity is the age of the state at the node's timer
    tick, and no tick instant is recorded.
  - **dead time is not measurable from a capture**, and `sensor_to_valve_delay`
    is not a free lever — `config.check_settings` refuses any value but `Ts`.
    Only the *sum* of dead time and `tau_v` is identified, so the lever is
    `--lag-shift` (constant sum) and the instrument is `steptest.py`, open loop
    with the MPC out. A correlation of `out` against `/joint_states` `vel` would
    measure the closed loop's dominant phase — `out` carries the PI term, so it
    carries the measurement — plus `tau_v` plus the body's own lag.

  Two caveats on the latency it does report: arrival is the capture node's, not
  the JTC's (same transport, different callback queue, and the recorder's own
  `spin_once` backlog is inside the number), and `rx` is a held sim-clock value,
  so the resolution is one `/clock` period and a lagging capture clock can read
  negative.
- `growth.py <cap.json> <label> ...` — the score: fit `log|q_u - q_eq|` against
  sim time over the first rise. `<= 0` is stable. Two runs per arm, worst reported.
- `steptest.py <out.json> [--sim-time|--no-sim-time]` — one open-loop velocity
  step per axis on a live sim, MPC killed, recording position, velocity and
  applied torque. Same clock rule as `capture.py`. On the machine the steps are
  issued by a person through `FollowJointTrajectory` (commissioning §3), not by
  this script.
- `plant_step.py <out.json> [joint]` — the same step on every plant we have,
  side by side: Gazebo (from that capture), MuJoCo through `C3Actuator`, and the
  OCP's own integrator. They agreed to 0.01 through 500 ms once issue 161's
  three actuator faults were fixed; before that Gazebo was alone. `ocp_model`
  takes the *planned* row (`planned_row(joint)`) — before issue 212 it stepped
  row 0 whatever axis was asked for, so every non-slewing column was slewing's
  response under another axis' name.
- `labday_report.py <bag.mcap>` — the lab-day sections below, offline, off an
  mcap. Also the refactor target for `looptiming.py`'s latency and rate, which
  now take either a capture's streams or a bag's.
- `modelfit.py bag '<glob>' | capture <cap.json>` — one-cycle prediction error of
  the MPC's own model against what the crane actually did, per axis, with a
  skill score against "nothing changes". Works on the HydraulicCalib bags (real
  machine) and on a live capture. Score against the command the machine
  *received* (`controller_state.output`), not the one the MPC asked for — on the
  same run the latter reads 20x worse.
- `score.py` — cadence, command chatter, travel, arrival. Context, not the score.
- `pid_move.py` — the same plan with the MPC off: the stable reference arm.

## The lab-day shadow kit — issue 212

Hardware time is the scarce thing, so the lab day **records** and the numbers are
taken afterwards. Nothing in this section drives anything; every launch and
every step on the crane is human-run (PRD §14, `wiki/commissioning.md`).

    # on the crane, by a person, beside commissioning.md §6's shadow launch
    ros2 bag record -s mcap -o labday \
      $(grep -v '^#' scripts/trials/labday_topics.txt)

    # afterwards, anywhere
    ./scripts/trials/labday_report.py labday/labday_0.mcap

`labday_topics.txt` is the checked-in profile: the topic list, that command, and
the clock note. Three things in it are worth knowing before the session.

- **The `~/` topics are not under `/crane/mpc/`.** `~` resolves against the
  *node* name, and `crane_mpc.launch.py` sets `name="crane_mpc"`, so the two
  shadow topics are `/crane_mpc/shadow_horizon` and
  `/crane_mpc/shadow_comparison`. Rename or namespace the node and they move.
- **There is no tip/tilt broadcaster and no tip/tilt `status` topic.** Issues
  101–103 deleted `crane_control`; the passive pair reaches `/joint_states` from
  the plain `joint_state_broadcaster`. `/crane/sway_settled` is recorded instead.
- **Latency is a difference of two clocks** — the recorder's receipt against the
  publishing host's `header.stamp`. The MPC host and the ros2_control host have
  to be chrony-synced or the number is the offset between them. The report
  reports a negative or implausible value as a clock problem and refuses it as a
  result; both `control_recordings` bags read ≈ −23 ms, which is what that looks
  like.

`labday_report.py` reads the bag with `mcap_ros2` off the embedded schemas, so
it needs no sourced workspace and no built `crane_msgs` — except for the two
sections that evaluate the OCP's own integrator (model skill, steps), which
import `crane_model`/`crane_mpc` lazily and say so when the import fails. Six
sections; one whose topics are absent says "not recorded" and does not fail.

1. **Latency** — stamp to receipt per stream, then `looptiming.py`'s
   controller-state rate and `jtc_output` quorum share, against the shadow
   horizon's cadence as the cycle grid.
2. **Solve timing** — `solve_time` p50/p90/max against `solve_budget`; the late,
   non-converged, `used_slack` and `applied_previous_solution` shares;
   `FAULT_SOLVER` count; silence episodes. The stream is decimated by
   `solver_health_decimation`, so a gap is not a missed solve.
3. **Model skill** — `modelfit.py`'s one-cycle skill per axis, scored against
   `controller_state.output`, the command the machine actually received. Where
   `/cranedata` is in the bag, a second row with the force state seeded from the
   measured chamber pressures.
4. **Shadow vs drive** — per axis, `<joint>.difference` out of
   `~/shadow_comparison`: RMS, bias and the worst `--window` second mean, worst
   axis named. `u != dq_ref` by construction, so a bias is expected; there is no
   documented threshold and the checklist records the number rather than judging
   it.
5. **Steps** — single-axis episodes in the driven command that rise from a quiet
   machine and then hold: dead time, time to 63 %, and the C3 response from
   `plant_step.ocp_model` beside the measured one. `k` is *not* separately
   identifiable from a velocity step without `M_ii`, which is pose dependent, so
   what is compared is the response `k` and `tau_v` produce. Past ~300 ms the
   episode is not open loop.
6. **Checklist** — a markdown table of each quantity, its value, the threshold
   from `commissioning.md` §6 and a verdict, ready to paste back into §6.

The measurement is checked in `test/test_labday_report.py` against a synthetic
mcap with an injected latency and an injected multiplicative model error, both
recovered. The model arm takes its plant as an argument there: a skill number
off the real OCP is a measurement, and asserting one tests only that the model
has not changed.

## Matching `wire_chain` to the live Gazebo loop — issue 189

A robustness answer from standalone MuJoCo transfers only if its loop matches
the live one. `--latency` and `--cadence-phase` are free parameters and have
**never been fitted to Gazebo**; Gazebo has read ~3x faster than the identified
model before, and the wire has ~10 ms of margin, so an unmatched harness can
call a live instability stable. Until the table below is filled, treat a
`wire_chain` stability verdict as unvalidated against Gazebo.

The recording is human-run — it needs a live Gazebo, which the loop may not
start (PRD §14). The measurement after it is offline.

    ./scripts/trials/trial.sh /tmp/a2b-loaded     # one live A2B move, ~3 min
    ./scripts/trials/looptiming.py /tmp/a2b-loaded/cap.json
    ./scripts/trials/wire_chain.py --goal out --latency <L>
    ./scripts/trials/steptest.py /tmp/step.json     # open loop, for the lag
    ./scripts/trials/plant_step.py /tmp/step.json   # Gazebo / MuJoCo / OCP

Once loaded at 800 kg and once empty (`--payload-mass`/`--plant-payload` on the
replay side), then compare sway peak and path error against the recording.

| quantity                | Gazebo measured | `wire_chain` set       | how                |
| ----------------------- | --------------- | ---------------------- | ------------------ |
| tick-to-arrival latency | p50 26 / p90 30 / max 62 ms | `--latency 0.026` | `looptiming.py`    |
| controller-state rate   | 100 Hz, 6.0 ticks/cycle, 0 % starved | `harness` `jtc_period` | `looptiming.py`    |
| dead time + `tau_v`     | matches MuJoCo to 300 ms on all five axes | `--lag-shift 0.0` | `steptest.py` + `plant_step.py` |
| cadence phase           | **not observable from a capture** | `--cadence-phase 0.0` | — |

Measured 2026-10-01: one unloaded A2B move, `trial.sh`, headless, otherwise idle machine,
2 ms physics step, N=40, 30 deg sway box. Live sway peak 0.009 rad, final error 1e-4,
127/127 solves converged (p50 4.9 ms); `wire_chain --latency 0.026` on a similar move:
0.007 rad, 1e-4. `plant_step.py`: MuJoCo tracks Gazebo's step to <=0.02 over the first
200-300 ms (telescope 0.83 vs 0.71); the OCP model is ~3x slower there on boom, arm,
telescope and rotator, slewing agrees. Past ~300 ms the step test is not open loop (the
JTC's PI holds position), so later columns do not compare. Under the full interactive
stack (GUI, rviz, perception, load avg 20+) solves ran 75-173 ms: latency is a property of
the load, not of the loop. Still unmeasured: a loaded (800 kg) move.

Fill a cell only from the instrument named in the last column. A wrong number
here does not leave the harness unvalidated — it declares it validated, which is
worse than an empty cell. In particular a command-to-response correlation on a
`cap.json` is not the dead time (see `looptiming.py`), and `--cadence-phase`
stays at its default because the capture cannot say otherwise.

Known gaps in `wire_chain` that no argument closes, and whether they bear on
stability rather than tracking:

- **The solve takes no time.** `--latency` is delivery only and the budget is
  off, so a late solve never lands late. Live, `applied_previous_solution` is a
  different closed loop — stability, not tracking.
- **Measurement is exact at the cycle instant.** No encoder quantisation, no
  sensor age, and the passive state is read rather than estimated. The live
  passive state comes from a complementary filter whose lag is inside the sway
  loop — stability.
- **Arrival is measured at a second subscriber.** The JTC's own callback
  scheduling adds to it under load, and the issue's own note records
  controller-state starvation at load avg 20-28 — stability. `--latency` is one
  number, so it cannot carry the jitter either: the p90 and the p50 are set
  separately or not at all.
- **The cadence anchor is exact.** `wire_chain` measures at the cycle instant,
  so the state is never stale; live it is up to a `/joint_states` period old at
  the timer tick. This is the gap `--cadence-phase` was meant to stand for and
  the one the capture cannot measure — stability, and currently unbounded.
- **The tool axis is not driven**, and the JTC's tolerance checks, goal handling
  and angle wraparound are skipped — tracking, except where a tolerance abort
  would have stopped the move.
