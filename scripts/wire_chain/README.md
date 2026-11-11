# Offline chain — issues 161, 171

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
- `wire_chain.py --goal out --pj [PORT]` — stream every cycle as JSON over UDP
  to PlotJuggler (Streaming → UDP server, port 9870 default, JSON, timestamp
  field `t`). `plotjuggler.py`; layout `wire_chain_pj.xml` (tabs sway, axes, tracking, authority, cost, solver; authority = sway accel the joints can produce at `du_max`, B = -M_uu^-1 M_ua). Cost split every cycle under `--pj`.
- Joint readings carry the Gazebo description's noise and bias, read off its
  xacro; `--noise-scale 0` is the exact measurement the numbers above were taken on.
- `--drop P [--topic-timeout S]` — lose a fraction P of horizons on the wire (P = 0.84: ~1.3 of
  8.3 Hz arrive). At the JTC's `topic_timeout` 0.24 s one lost horizon already
  means a hold; at 1.0 s, P = 0.5 matches a lossless run.
- `tune_weights.py --trials N --storage sqlite:///tune.db` — Optuna search over the MPC weights
  (`dq_u`, `q_u`, `du`, `dq_a` luffing / slew+telescope, `tcp`, `terminal_scale`, log-uniform).
  Each trial: 4 runs of 4 random moves (seeds 1, 2 × empty, 800 kg) and 4 kick tests (tip, tilt ×
  near, far pose). Minimises kick decay to 10 % (worst), sway rate left after moves, TCP path error
  p95; any failed run makes the trial infeasible. First trial = the yaml. ~50 s a trial at
  `--jobs 4`; more workers: start it again on the same `--storage`. Browse: `optuna-dashboard
  sqlite:///tune.db`. Prints the Pareto front as `--set` lines.
- `--kick TIP TILT` — swing the load this far at the first move's start; the plan starts hanging.
- Loaded-slew instability (issue 304) — the 11-09 stacking slew, 880 kg, plant boom/arm stiffer than C3:

      ./scripts/wire_chain/wire_chain.py --seconds 20 --latency 0.06 \
        --start -0.2568 0.6704 0.5493 1.0676 0.3863 1.5644 5.0444 0.40 0 0 \
        --goal-pose 7.0276 2.2039 2.7452 1.5552 \
        --payload-mass 880 --payload-com 0 0 1.05 --plant-payload-com 0 0 1.12 \
        --k-scale 1 6 6 1 1 --d-scale 1 6 6 1 1

  Arm rings at ~1.3 Hz with command reversals; `--k-scale 1 1 1 1 1` is the calm baseline. Loaded on
  the machine the arm passes ~1.0 of u with a ~0.15 s pure delay and no mode in 0.3-2 Hz (instrument
  estimate, 11-09 880 kg PID bags); the fitted C3 passes ~0.35 at 1.1 Hz and has a resonance at
  0.5 Hz, so k and d scaled together -- a velocity source, tau = d/k -- is what matches the machine.
  `--jtc-only` plays the plan through the JTC alone, no MPC, on the same plant: calm on x1/x6/x12,
  which is what makes those plants admissible (loaded, the machine's PID is stable). k x12 with
  d x1 is not admissible -- the JTC alone holds a 1.65 Hz ring there (issue 308).
  Score it from **move start**, not after arrival: the machine never arrived, so the json's
  `ring_ptp` / `ring_growth_per_cycle` / `ring_hz` (arm, 0.8 s windows) are what compares --
  machine 0.017 .. 0.243 rad/s at 1.10 Hz, bench at plant x2 0.015 .. 0.248 at 0.95 Hz.
  Issue 312 reads it at the **control tick** (10 ms, `ring_step_s`) and on all seven axes:
  `ring_{hz,growth,ptp_max}_per_axis` over `sw ha ka sa ro tip tilt` plus `ring_worst_axis` --
  the fastest axis **growing** over 1.0 /cycle, `null` when none does, which is the gate: a
  decaying ring is not a worst anything (the calm x2/x2 cell scores a 0.90 /cycle rotator).
  The arm fields are the arm's, unchanged in name, and the 0.12 s cycle log scores the same
  (`records/analysis_312/table.py` reproduces 307's three cells to the digit) -- so the
  per-axis numbers and 307's grid are one metric. On 307's shipped/x2 cell all six other axes
  ring on the same 0.95 Hz mode and the tip swings 4x further than the arm turns, which the
  arm-only score could not see. Mind the floor when comparing axes: it is each axis' own
  planned peak and spans 45x in one run, so slew's 0.043 rad/s reads `—` while the rotator's
  0.089 is scored -- amplitudes compare within a row, not across them.
- `--ocp-fit PATH` — export the OCP from another C3 fit file instead of the shipped
  `c3_full_model.json`; the description's damping follows that file's `d`, so it is the whole
  model and not `k` alone. Minutes of code generation the first time per file, then cached by
  export key. Issue 307's gate ran it over boom/arm `k, d` x2/x4/x12 against plants x1..x12
  (`records/analysis_307/`): **the OCP's own C3 is the lever** -- x2 and x4 are calm on every
  plant, the shipped one rings on everything above x1, and x12 rings on x1 (too stiff a model
  is as bad as too soft). Solve time is unchanged, p50 5-7 ms, p95 6-18 ms.
- `--pi-scale-axes F*5` — the JTC's `p, i, d` per axis, on top of `--pi-scale`; `0` opens that
  axis' loop properly, which `--pi-rung no-integral` does not (it leaves `i`). Issue 310 swept
  boom/arm 1.0 / 0.5 / 0.3 / 0 and all five axes at 0.3 under the x2 OCP design model, 69 moves
  each (`records/analysis_310/`): **no scale rings** -- not one of 345 moves clears 1.05 /cycle on
  `sw ha ka tip tilt`, the only finite growth in the family is 1.04 at scale 1.0 on plant x12, and
  the shipped OCP on the same move rings at 1.37. So the PI is not the stability lever (313 found
  the same over 587 cells, and on its velocity-source plant a *lower* PI was worse than the
  deployed one). What the scale moves is the kick: on the nominal plant with a 0.1 rad tip kick,
  the tip's sway rate 1.5-2.5 s after arrival (`sway_rate_after_2s`) is 58 / 47 / 42 / 33 mrad/s
  at 1.0 / 0.5 / 0.3 / 0, and the residual at the end of a 20 s run 12.6 / 6.9 / 5.1 / 2.6; it
  costs final error monotonically (+0.025 / +0.06 / +0.13 m) on the one move that already rides
  the MPC's rate limit. Scaling the other three axes buys nothing. The JTC alone never rings at
  any scale (worst ptp 0.124-0.125 rad/s on plants x1..x12), and only scale 0 is visible there at
  all (6 mrad final error against 1).
