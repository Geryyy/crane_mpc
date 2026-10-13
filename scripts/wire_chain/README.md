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
