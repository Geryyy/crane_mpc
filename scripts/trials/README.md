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
- `capture.py` — records `/joint_states`, `/crane/mpc/horizon` knot 0 and
  `/crane/mpc/solver_health`. Joints by `name[]`, never by index. Each row
  carries three times: wall `t`, the header stamp `sim` and the receipt `rx` on
  the stamps' own clock (`use_sim_time`, so sim time in sim). `sim - rx` is the
  only way to a latency — wall and sim are different clocks.
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
