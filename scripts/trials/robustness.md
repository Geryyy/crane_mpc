# How much plant mismatch the MPC survives — issue 190

The first live run with the MPC driving went unstable carrying a block: the sim
block weighed 80 kg and the behaviour tree had declared 800 kg. That is one
point of a map nobody had drawn. This is the map: **which plant mismatch breaks
the loop, and which knob widens the region that holds.**

Everything here is offline, `trials/wire_chain.py` through `trials/sweep_robust.py`,
MuJoCo under C3 and the JTC's velocity loop. Nothing was launched.

## Read this before you use a number

- **The harness is still unvalidated against Gazebo.** Issue 189's live half was
  never run, so `--latency`, the cadence phase and the tick rate are unfitted —
  see `README.md`'s table, whose cells are still empty. Every run below uses
  `--latency 0.06`, the value the 190 issue text quotes from the live run, not a
  measured one. A stability verdict here is a verdict about `wire_chain`.
- **`growth_per_s` cannot be the score on its own.** `growth.py` fits the first
  rise through `2e-3 < dev < 0.2` and wants ten samples inside it, so a
  deviation that crosses 0.2 quickly — which is what a hard mismatch produces —
  returns `nan`. Over all 997 runs in this study the rule is one-directional and
  exact: every one of the 439 `nan`s has a sway peak above 0.2, and no run that
  stays under 0.2 ever returns one — but 117 runs cross 0.2 and still get a
  slope, so a finite `growth_per_s` is not evidence that a run was quiet.
  `sweep_robust`'s worst column takes `max(..., key=np.nan_to_num)`, so a single
  `nan` propagates to it. Read `exit`, `sway pk` and the clamp ratio with it
  (`dq/clp` in the tables below, `u/clp` on any sweep run after 209), never the
  slope alone.
- **`sway_late` is the discriminator the issue asked `growth.py` for.** It is
  the sway peak over the *last 3 s* (`wire_chain`'s `late`), so `late ≈ peak`
  means the largest swing of the whole run is still happening when the run
  ends, and `late << peak` means the move excited a swing and the loop damped
  it. It is a *strictly* more conservative reading than the failure count, not
  an agreeing one: `late/peak` is exactly 1.00 on five cells of §1 and
  0.10..0.91 on the other eighteen (not a clean gap — `k` x0.5 sits at 0.91),
  but only two of those five cells contain a failing run. It flags payload 80,
  `k` x2 and Psi x0.7 as unsettled where the exit column says `ok` three times
  over. Treat it as "did not settle", which is what it measures, and not as a
  prediction of the gate. It is still better than `growth_per_s`, which one cell
  of §3's payload
  family (`weights.du=10` at 400 kg) scores `+8.0/s` with `peak 0.215,
  late 0.015` and no failure — the "growth" there is the first swing rising out
  of the band at the move's start, fitted as if it were a divergence. **Rank on
  `late` and the failure count; keep the slope as a cross-check on the cells
  where `late ≈ peak`.**
- **The runs are deterministic**, so a `4/27` against an `8/27` is exact for
  those moves — and those moves are three (`--goal out` and two random seeds),
  which is a small sample of the move space, not of the noise.
- **`past_clamp` — which is what "failed" almost always means here — is scored
  on the wrong variable, and this limits every conclusion below.**
  The gate took `dq[PLANNED]`, the plant's *measured joint velocity*, and tested
  it against `chain.high`/`low`, which are `u_clamp_max`/`min`. Two things are
  wrong with that pairing. `u_clamp_*` is
  Psi's fitted domain on `u`, the *commanded* joint velocity, and `u != dq` —
  the command runs 5-8 % RMS ahead of the velocity. And in the loop that ships
  (`velocity_loop.py:9, 14, 22`) `u_clamp_*` bounds the **PI correction alone,
  never the total command**. So `past_clamp` is an overspeed test wearing a
  Psi-domain label, and the `dq/clp` column is a fraction of a bound that does
  not apply to the quantity in its numerator.
  This bites hardest on the Psi rows. `chain.psi.apply(command)` multiplies the
  command on its way into the plant, so a Psi gain of x1.3 raises the achieved
  joint velocity by roughly 30 % for the same command — and then trips a
  threshold on achieved joint velocity. §3's headline result about Psi is
  therefore not separable from the gate that measures it *in the tables below*.
  **Issue 209 changed the gate; every table in this document predates it.**
  `wire_chain` now logs the JTC's `forward + correction` per tick, scores
  `past_clamp` on that against `u_clamp_*`, and keeps the old `dq` test as the
  `overspeed` column. Re-running the Psi family alone (shipped, `--latency 0.06`,
  five moves) scores **both gates off one sweep: old 4/25, new 3/25.** One of the
  five Psi-x1.3 runs overspeeds the clamp without the command ever leaving Psi's
  domain; three leave it for real. So the gate inflated the Psi verdict but did
  not manufacture it, and Psi x1.3 is still the only failing cell of the family.
  **That old 4/25 is not §3's recorded 3/25, and the difference is drift, not the
  gate**: on §1's three-move subset the old gate now scores Psi x1.3 at 2/3 where
  §1 below records 1/3. The tables here do not reproduce on today's `crane_mpc`,
  so compare old against new only within one sweep. Every table below is the old
  gate's.
- **The ranking is the `past_clamp` count and nothing else.** Over all 997 runs:
  831 `ok`, 149 `past_clamp`, 17 `diverged`, and **`hunting` never fired once** —
  the u0-reversal gate contributed no information to any table, although `rev` is
  a column in all of them. `diverged`, the one unambiguous failure, runs 0-2 per
  setting and does not track the ranking: `weights.q_a=4` is the worst weight at
  10/27 with *zero* divergences. §3 splits it out.
- **The node goes silent on a third to a half of the cycles on exactly the
  mismatched runs, and no table reports it.** `Step.silence` is a pre-solve gate
  trip (clock skew / reference age / state age / `propagate`), after which
  nothing is published and the JTC flies the last horizon until it times out.
  Median 0 % on the nominal plant (p90 6 %, max 26 %); on mismatched plants p90
  is 46 % and the max 56 %, and 48 % of mismatched runs have a nonzero count —
  including runs that exit `ok`. So on the cells that decide the ranking, part
  of what is being scored is the stale-horizon/hold path, not the MPC law.
  This also hollows out the `max_consecutive_failures` row of §2:
  `wire_chain` sets `ocp.solve_budget_s = 10.0`, so the solver never times
  out, and a silent cycle does not count toward the consecutive-failure counter.
  That knob had almost nothing to act on.

## 1. One mismatch at a time, at the shipped settings

    ./scripts/trials/sweep_robust.py --latency 0.06 --payload-mass 800 \
        --samples 0 --jobs 7 --by-plant --band <family>=<values>

OCP payload 800 kg throughout; the band is what the *plant* got. Three moves per
cell, worst of the three.

`late/pk` is the stability column: 1.00 means the run ended at its own worst
moment — nothing settled. It does not say *why*, so read it with the two
columns beside it: `late/pk = 1.00` with a large `rev`/`dq/clp` is a swing the
loop is not damping, `late/pk = 1.00` with a large final error and `rev ≈ 0`
(Psi x0.7) is the machine still crawling toward a goal it never reaches. Bold
marks a cell that is one of the two.

| mismatch | value | sway pk | late | late/pk | final err | rev | dq/clp | growth /s | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *(none)* | nominal | 0.148 | 0.029 | 0.20 | 0.002 | 0.01 | 0.27 | −0.10 | 0/3 |
| payload mass | 80 kg (0.1x) | **0.692** | **0.692** | **1.00** | 0.217 | 0.15 | 0.92 | **+0.51** | 0/3 |
| | 400 kg (0.5x) | 0.195 | 0.020 | 0.10 | 0.002 | 0.01 | 0.31 | −0.11 | 0/3 |
| | 1200 kg (1.5x) | 0.128 | 0.052 | 0.41 | 0.022 | 0.01 | 0.27 | −0.03 | 0/3 |
| | 1600 kg (2x) | 0.120 | 0.069 | 0.58 | 0.026 | 0.01 | 0.28 | +0.02 | 0/3 |
| C3 `k`, `d` | x0.5 | 0.114 | 0.104 | 0.91 | 0.014 | 0.01 | 0.29 | 0.00 | 0/3 |
| | x0.7 | 0.120 | 0.064 | 0.53 | 0.005 | 0.01 | 0.27 | +0.01 | 0/3 |
| | x1.5 | 0.189 | 0.023 | 0.12 | 0.002 | 0.01 | 0.29 | −0.10 | 0/3 |
| | x2 | **0.426** | **0.426** | **1.00** | 0.089 | 0.24 | 0.74 | **+0.30** | 0/3 |
| C3 `tau_v` | x0.5 | 0.172 | 0.032 | 0.19 | 0.005 | 0.01 | 0.29 | −0.10 | 0/3 |
| | x0.7 | 0.160 | 0.024 | 0.15 | 0.003 | 0.01 | 0.28 | −0.10 | 0/3 |
| | x1.5 | 0.135 | 0.072 | 0.53 | 0.005 | 0.01 | 0.25 | −0.06 | 0/3 |
| | x2 | 0.124 | 0.099 | 0.80 | 0.030 | 0.01 | 0.25 | −0.03 | 0/3 |
| dead time | 30 ms (−30) | 0.173 | 0.024 | 0.14 | 0.002 | 0.01 | 0.28 | −0.11 | 0/3 |
| | 90 ms (+30) | 0.140 | 0.056 | 0.40 | 0.003 | 0.01 | 0.25 | −0.08 | 0/3 |
| Psi gain | x0.7 | 0.209 | **0.209** | **1.00** | **4.128** | 0.06 | 0.22 | +0.05 | 0/3 |
| | x0.85 | 0.122 | 0.080 | 0.66 | **2.376** | 0.01 | 0.20 | −0.04 | 0/3 |
| | x1.15 | 0.258 | 0.109 | 0.42 | 0.011 | 0.08 | 0.77 | **+0.27** | 0/3 |
| | x1.3 | **0.791** | **0.791** | **1.00** | 0.909 | 0.42 | 1.12 | +0.13 | 1/3 |
| payload COM | +0.25 m x | 0.210 | 0.072 | 0.34 | 0.194 | 0.11 | 0.81 | nan | 0/3 |
| | +0.50 m x | **0.389** | **0.389** | **1.00** | 0.843 | 0.16 | 1.35 | nan | 2/3 |
| | +0.25 m z | 0.150 | 0.033 | 0.22 | 0.004 | 0.01 | 0.27 | −0.06 | 0/3 |
| | +0.50 m z | 0.155 | 0.039 | 0.25 | 0.004 | 0.01 | 0.27 | −0.03 | 0/3 |

What that says:

- **Payload mismatch is one-sided.** Believing the load *heavier* than it is, is
  what destabilises: 0.1x grows at +0.51/s and swings 4.7x the nominal peak. The
  other direction — a plant twice as heavy as declared — costs 26 mm of final
  error and nothing else. The 2026-09-30 live failure is exactly the bad
  direction, and it is the direction a released-but-not-noticed grasp produces.
- **Command-path gain is the sharpest single lever**, and also one-sided, but
  the *other* way round: too much gain (x1.15, x1.3) destabilises, too little
  (x0.85, x0.7) is stable and simply never arrives — 2.4 and 4.1 rad of final
  error is the machine crawling, not swinging. This confirms the earlier "MPC
  weights buy no robustness to a command-path gain error" finding from its
  failure side.
- **`tau_v` and the dead time do not matter over x0.5..x2 and ±30 ms** — with a
  caveat that empties part of the `tau_v` row. `scale_tau_v` is a *multiplier*,
  and the shipped fit is `tau_v = [0.1, 0.025, 0.0, 0.05, 0.125]`: axis 2,
  `theta3_arm_joint`, is fitted at exactly zero, so **every `tau_v` row above
  left the arm's lag untouched.** The arm carries the payload moment arm, so the
  family covers four of the five axes and not the most interesting one. The
  multiplier was chosen over `perturb_fit`'s `lag_shift_s` precisely because a
  shift at `tau_v = 0` is refused (`mismatch.py`) — which is the same wall from
  the other side, not a way round it.
  On the four axes it does reach, nothing fails and no `late/pk` reaches 1, and
  both knobs move `late/pk` monotonically in the *slower*-plant direction
  (`tau_v` x2 -> 0.80, dead time 90 ms -> 0.40, `k` x0.5 -> 0.91), so margin is
  being spent, just not to exhaustion. Perturbing the arm's lag needs an
  *additive* offset; §5 makes that run (`--tau-v-offset`, ±30 ms on all five axes
  including the arm, no failure at any of the three settings it tried).
- **A payload COM error matters only in the swing plane.** 0.5 m along z (down
  the sling) is invisible; 0.5 m along x pushes two of three moves past the
  command clamp. Read these two rows knowing that `wire_chain` scores sway
  against the **plant's** equilibrium — it passes the plant's mass and COM to
  `passive_equilibrium`, so a COM offset moves the reference with it. The rows
  therefore measure *swinging*, not the static hang offset the OCP is wrong
  about; that offset is what the 0.19 and 0.84 rad of final error are.
- **C3 `k` is asymmetric too**: a *stiffer* plant than modelled (x2) is the bad
  side — sustained 0.43 rad of sway and a quarter of the cycles reversing. x0.5
  costs only a small residual (0.10 rad, not growing).

So the mismatch envelope that holds at the shipped settings is roughly

    payload  >= 0.5x declared      Psi gain  <= ~1.15x
    C3 k     <= ~1.5x              COM error <= ~0.25 m in the swing plane
    tau_v    x0.5 .. x2            dead time 60 +- 30 ms

## 2. Which knob widens it

Same three moves, against the eight corners of payload x{800, 80} kg,
C3 `k` x{1, 2} and Psi gain x{1, 1.3} — one at a time and paired — plus the
nominal plant. 27 runs per setting.

    ./scripts/trials/sweep_robust.py --latency 0.06 --payload-mass 800 \
        --samples 0 --jobs 7 --by-plant \
        --band payload=800,80 --band k=1,2 --band psi=1,1.3 --lever <knob>=<v>

| setting | failed | worst sway pk | worst late | worst rev | nominal sway pk | nominal err |
| --- | --- | --- | --- | --- | --- | --- |
| `horizon_length=20` | **4/27** | 1.232 | 1.232 | 0.29 | 0.269 | 0.0005 |
| `weights.du=10` | **4/27** | 1.383 | 1.383 | 0.26 | 0.168 | 0.0015 |
| `max_consecutive_failures=1` | 6/27 | 0.983 | 0.814 | 0.25 | 0.148 | 0.0017 |
| `horizon_length=60` | 7/27 | 1.241 | 1.241 | 0.26 | 0.175 | 0.0024 |
| shipped | 8/27 | 1.205 | 1.205 | 0.42 | 0.148 | 0.0017 |
| `weights.dq_u=256` | 8/27 | 1.254 | 1.254 | 0.43 | 0.143 | 0.0062 |
| `weights.q_u=20` | 8/27 | 1.267 | 1.267 | 0.40 | 0.149 | 0.0012 |
| `weights.u=10` | 8/27 | 1.212 | 1.212 | 0.39 | 0.148 | 0.0016 |
| `kappa=0.5` (planner) | 8/27 | 1.185 | 0.942 | 0.40 | 0.149 | 0.0014 |
| `kappa=0.65` (planner) | 8/27 | 1.293 | 1.293 | 0.29 | 0.146 | 0.0017 |
| `weights.dq_u=16` | 9/27 | 1.030 | 0.819 | 0.46 | 0.153 | 0.0018 |
| `weights.q_a=80` | 9/27 | 1.334 | 1.334 | 0.31 | 0.148 | 0.0019 |
| `max_consecutive_failures=20` | 9/27 | 1.204 | 1.204 | 0.33 | 0.148 | 0.0017 |
| `weights.q_a=4` | 10/27 | 0.998 | 0.828 | 0.26 | 0.148 | 0.0008 |
| `command_state=false`, `du=10` (C3) | 12/27 | 1.375 | 1.375 | 0.49 | 0.165 | 0.0015 |
| `dq_a_feedback=false` | **19/27** | 1.284 | 1.284 | 0.42 | 0.238 | 0.0907 |

- **The worst amplitude does not rank the settings; the failure count does.**
  Twelve of the sixteen rows have `worst late = worst pk`, i.e. their worst cell
  never settles; the other four (`max_consecutive_failures=1`, `dq_u=16`,
  `q_a=4`, `kappa=0.5`) come in strictly under, so at least there the peak and
  the residual are different cells. Either way the amplitude is a saturation
  level, not a margin — it moves 0.98..1.38 with no relation to the failure
  count, and `q_a=4` has the *smallest* worst peak with the *most* failures.
  What separates the settings is how many of the 27 runs end in `past_clamp` or
  `diverged` — which, per the caveats above, is in practice the `past_clamp`
  count, on a gate that is measuring the wrong variable.
- **No weight buys tolerance.** `q_a` flattened to 4 and to 80 (it ships 40 on
  the four arm rows, 10 on rotator and tool), `dq_u` x0.25 and x4, `q_u` from 0
  to 20, `u` x10 — every one of them lands between 8 and 10 failures of 27,
  against the shipped 8. The earlier sweep's conclusion holds under payload and
  Psi mismatch too: **cost weights move tracking, not the stable region.**
- **`weights.du` is the exception, and it is not really a weight** — it is the
  price of command *rate*, i.e. of the chatter the mismatch excites. 1 -> 10
  halves the failures (8 -> 4) and takes a third off the worst reversal
  fraction (0.42 -> 0.26), for 0.148 -> 0.168 of nominal sway peak and nothing
  else — nominal `late` and final error come out unchanged or better.
- **A shorter horizon is the other 4/27, but the trend in N is not monotone and
  the mechanism is not isolated.** `horizon_length=20` (1.2 s against 2.34 s)
  wins on the mismatched cells: at payload 80 kg the sway peak drops
  0.692 -> 0.371 and the growth +0.51 -> +0.19; at Psi x1.3 it drops
  0.791 -> 0.394 with no failure. But in N the failure count goes
  **20 -> 4/27, 40 -> 8/27, 60 -> 7/27** — N=60 ranks *above* shipped, so this
  is not "short good, long bad" and `horizon_length=60` is not the same
  statement from the other side. Three mechanisms are confounded and this study
  separates none of them:
  1. N=20 *damps the nominal plant less*, not more — nominal sway peak
     0.148 -> 0.269. A setting that swings more on the plant it was tuned for is
     not obviously widening a stable region.
  2. 1.2 s may be shorter than the dominant sway period, in which case the MPC
     cannot shape that mode and simply commands less. The `dq/clp` numbers say
     it does command less (payload family, worst 0.92 -> 0.70) — and commanding
     less is exactly what the `past_clamp` gate rewards, per the caveats above.
  3. `horizon_length` is in the solver's export key, so N=20 re-exports; the
     terminal-to-stage cost ratio moves with N and nothing here holds it fixed.

  So read this row as "N=20 scores best on this gate", not as a demonstration
  that less horizon buys model-mismatch tolerance. §3 runs it together with
  `weights.du=10`.
- **`dq_a_feedback=false` is the worst thing in the table**, by a wide margin
  (19/27, and it breaks the *nominal* plant: 0.09 rad of final error and a
  positive growth rate). The lever ships all-true and there is no headroom left
  in it; the earlier note that it "was the only lever" was about switching it
  on, and that has already been spent.
- **C3 command encoding is worse than C4** under mismatch: 12/27, the worst in
  the table bar `dq_a_feedback`, and every *mismatched* cell reverses on
  0.38-0.49 of its cycles against 0.01 on the nominal plant — i.e. C3's hunting
  is switched on by the mismatch, not present throughout. Same verdict as
  `README.md`'s `--wire 161` regression check, which catches it on the nominal
  plant only because it pairs C3 with the old knot encoding.
- **`max_consecutive_failures` and the planner's `kappa` are not robustness
  levers.** They move individual cells by more than the ranking does but not in
  one direction, and neither improves on shipped outside noise.

## 3. Does the recommendation hold

§2 ranks knobs on one hard grid. The two that won there — `horizon_length=20`
and `weights.du=10` — are taken back to §1's one-at-a-time families here, alone
and together, to see whether the win is a property of that grid or of the
mismatch. "both" is `--set weights.du=10 --lever horizon_length=20`.

    ./scripts/trials/sweep_robust.py --latency 0.06 --payload-mass 800 \
        --samples 0 --jobs 7 --by-plant --band <family>=<values> \
        --set weights.du=10 --lever horizon_length=20

Payload and Psi ran five moves (`--goal out` plus seeds 1-4), `k` and COM three
(seeds 1-2) — compare down a block, not across.

**These rows are not a cell.** Unlike §1's, which are grouped per plant,
`sweep_robust`'s worst line maxes *each column independently* over the whole
family, so one row can mix plants. The Psi/shipped row is the clearest case:
`sway pk`, `late`, `rev` and `dq/clp` all come from the Psi-x1.3 cell while
`final err 4.128` comes from Psi-**x0.7**, the crawling one. Likewise the
payload/shipped row's `rev 0.25` is the 1600 kg cell, not the 80 kg cell (0.15).
Do not read a row across.

| family | setting | sway pk | late | final err | rev | dq/clp | failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| payload | shipped | 0.692 | 0.692 | 0.358 | 0.25 | 0.92 | 0/25 |
| 0.1x..2x | `horizon_length=20` | 0.587 | 0.497 | 0.215 | 0.23 | 0.70 | 0/25 |
| | `weights.du=10` | **0.532** | **0.463** | 0.156 | **0.06** | 0.75 | 0/25 |
| | both | 0.804 | 0.804 | 0.514 | 0.21 | 1.00 | 0/25 |
| Psi gain | shipped | 0.791 | 0.791 | 4.128 | 0.42 | 1.50 | 3/25 |
| 0.7..1.3 | `horizon_length=20` | 0.583 | 0.583 | 3.939 | 0.13 | 1.06 | 1/25 |
| | `weights.du=10` | 0.865 | 0.865 | 3.668 | 0.19 | 1.33 | 2/25 |
| | both | **0.473** | **0.352** | 3.629 | **0.06** | **0.74** | **0/25** |
| C3 `k` | shipped | 0.426 | 0.426 | 0.089 | 0.24 | 0.74 | 0/15 |
| x0.5..x2 | `horizon_length=20` | 0.372 | 0.092 | 0.045 | 0.03 | 0.37 | 0/15 |
| | `weights.du=10` | **0.229** | **0.084** | 0.011 | 0.01 | 0.45 | 0/15 |
| | both | 0.362 | 0.096 | 0.039 | 0.01 | 0.37 | 0/15 |
| COM | shipped | 0.389 | 0.389 | 0.843 | 0.16 | 1.35 | 2/15 |
| 0.25/0.5 m | `horizon_length=20` | 0.534 | 0.217 | 0.441 | 0.22 | 1.31 | 1/15 |
| | `weights.du=10` | **0.290** | **0.173** | 0.163 | 0.03 | **0.55** | **0/15** |
| | both | 0.508 | 0.193 | 0.211 | 0.04 | 0.58 | **0/15** |

and back on §2's paired grid, where "both" had not been run — split by exit
reason, because the aggregate hides the one that matters:

| setting | failed | `past_clamp` | `diverged` | which cells |
| --- | --- | --- | --- | --- |
| shipped | 8/27 | 7 | 1 | all four Psi-1.3 cells, plus payload-80 / `k`-x2 |
| `horizon_length=20` | 4/27 | 3 | 1 | payload-80 x Psi-1.3, both `k` |
| `weights.du=10` | 4/27 | 3 | 1 | all four Psi-1.3 cells, one move each |
| both | **3/27** | 1 | **2** | payload-80 x Psi-1.3, both `k` |

What that says:

- **Psi gain x1.3 is in every failing cell of every tuned setting — but this is
  the finding the gate is least able to support.** All three tuned settings keep
  their whole Psi-x1 column clean (shipped is the only one that fails in it, at
  payload-80 x `k`-x2), and no setting clears the two payload-80 x Psi-x1.3
  cells. Taken at face value that says command-path gain is the one mismatch no
  tuning survives. Taken with the first caveat at the top of this document, it
  may say much less: the failures are `past_clamp`, `past_clamp` tests the
  achieved joint velocity, and a Psi gain of x1.3 raises the achieved joint
  velocity by ~30 % by construction. **The two are not separated here.** The
  test that would separate them is a one-line harness change plus a re-run
  (§4, first bullet): log the JTC's `forward + correction`, the signal
  `u_clamp` is actually the domain of, and re-score `past_clamp` on it. Until
  then, treat "Psi is the axis no tuning survives" as this study's leading
  *hypothesis*, not its result.
- **The two levers are nested, not complementary, and "both" is not clearly
  better than the horizon alone.** Both settings fail 4/27, but the *cells* are
  not different parts of the grid: `horizon_length=20` fails in
  {payload-80 x Psi-1.3, both `k`}, and `weights.du=10` fails in those two plus
  both payload-800 Psi-1.3 cells. The first set is a **strict subset** of the
  second — there is no cell `h20` fails and `du=10` passes. Adding `du=10` to
  `h20` removes one run of 27, which is inside the resolution this study sets
  for itself two bullets down. On the *severe* exit it goes the other way:
  "both" is the only setting in the study with two divergences.
  Across all four §3 families plus the grid (107 runs) the totals are `h20` 6
  failures, "both" 3 — a 2x gap, and the only real argument for the pair.
- **But "both" is not uniformly better, and the payload family is where it
  loses.** At payload 80 kg alone it is worse than *either* lever on its own —
  0.804 rad of unsettled sway against 0.463 and 0.497, and it is the only one of
  the four that puts a joint on its clamp (`dq/clp` 1.00). Pick it for the
  paired envelope, not because it dominates.
- **That reversal only shows up at five moves.** At three moves the same cell
  reads 0.363/0.332 for "both" — better than either lever, the opposite
  ranking. Three moves is enough to separate 8/27 from 3/27; it is not enough
  to rank two settings that are close. Treat any §1-§3 gap under ~2x as
  unresolved.
- **The nominal cost of "both" is a faster transient, not a worse one.** Nominal
  sway peak goes 0.148 -> 0.268 but nominal `late` goes 0.029 -> 0.015 and
  final error 0.0017 -> 0.0021. The shorter horizon buys its mismatch tolerance
  by committing harder early; it settles at least as well as shipped.

### The recommendation

    weights.du: 10            (ships 1)
    horizon_length: 20        (ships 40)

with `command_state` left **true**. `crane_mpc.yaml:28` reads "`du: 10` and
false is C3", which is the pair issue 161 fitted; `du: 10` on its own under C4
is a different setting and the better one — the C3 pair is 12/27 in §2, this is
4/27. Do not carry `command_state: false` along with the `du`.

**It is provisional.** The gate it wins on is measuring the wrong variable, its
margin over `horizon_length=20` alone is one run in 27, and it carries the
study's only pair of divergences. It is the best setting *this* study can
distinguish, which is not the same as the best setting.

Each of these was run one at a time, and no run ends in `past_clamp` or
`diverged` over

    payload   0.5x .. 2x declared       C3 k      x0.5 .. x2
    COM error <= 0.5 m, any axis        C3 tau_v  x0.5 .. x2 on 4 of 5 axes
    dead time 60 +- 30 ms               Psi gain  <= ~1.15x

Paired, only `payload x k x psi` was ever run as a grid, so the envelope closes
over **those three and no others**. `COM x k`, `COM x payload` and
`dead time x k` are untested — and COM +0.5 m alone is already 2/3 `past_clamp`
at shipped settings, which makes it the axis least safe to assume composes.
`tau_v` excludes `theta3_arm_joint`, whose shipped lag is 0 and which a
multiplier cannot move.

Two qualifications inside the lines above:

- **payload 0.1x is excluded on purpose**, though it does not fail. It does not
  settle either — 0.80 rad still swinging when the run ends, worst `dq/clp`
  0.996, i.e. right up against the gate without tripping it. On a boom of this
  length that is the block still traversing metres after the move is nominally
  over. §1 drew the line at 0.5x for the shipped setting and the same line
  belongs here. If a grasp can silently be a tenth of the declared mass,
  `weights.du=10` alone is the better setting there (0.46 rad) and the paired
  envelope is what you trade for it.
- **Psi x1.15 is the weakest line on the page**, because it is one step below
  the value whose failures may be a gate artifact. If `past_clamp` is re-scored
  on the commanded `u` and the Psi failures survive, this line is right; if they
  do not, the real Psi envelope is wider and unknown.

**No tuning in this study holds Psi gain x1.3**, alone or paired: 3 to 7 of the
twelve Psi-x1.3 runs end in `past_clamp` or `diverged` at every setting tried,
the shipped one included. Whether that is a control result or a gate artifact is
the open question above — but it points the same way either direction. If it is
real, the answer is not an MPC setting: it is `command_domain`, a Psi refit, or
feedback on the achieved joint velocity. If it is the gate, then the thing this
study has actually been ranking settings on is an overspeed test, and the
rankings need redoing before any of them is used.

The 2026-09-30 live failure was blamed on the payload axis, and the payload axis
is the tunable one: every setting in §3's first table gets through 0.1x without
a failure. The gain axis is the one that is not, and it had not been the
suspect. That inversion is the most useful thing in this document, and it is
also the thing most dependent on a gate that needs re-scoring. Do not spend a
machine day on it before doing that.

## 4. What none of this can see

Listed so a later reader does not mistake the map for the machine. The first
four are cheap and would change conclusions; they should be done before this
map is used for anything.

- **Every table re-run on the new `past_clamp`.** Issue 209 did the harness
  half: the gate now tests the JTC's `forward + correction` against `u_clamp_*`,
  `overspeed` carries the old `dq` test, and `sweep_robust` prints `u/clp`,
  `ovrspd`, `silent` and `handbk`. What it did *not* do is re-run the sweeps —
  only the Psi family, which went 4/25 -> 3/25. §1's, §2's and §3's tables are
  all still the old gate's, so their rankings are still rankings on an overspeed
  test. About 30 min at `--jobs 7` on 8 cores.
- **The `silence` fraction read off a re-run.** p90 46 % on mismatched plants,
  max 56 %, 0 % median on nominal, and in no table here — `sweep_robust` prints
  it as of 209, but only the Psi family has been re-run through it. A setting
  recommended for hardware ought to hold it low inside its own envelope; §5
  checks that and the answer is **no** — shipped, C3 and the recommendation all
  read p90 22-23 %, max 24-26 %, and the measured latency does not lower it.
- **The exit reason, always split.** `hunting` fired zero times in 997 runs, so
  the `rev` column informs nothing about pass/fail; `diverged` is 0-2 per
  setting and does not track the aggregate. §3's second table splits them; §2's
  does not.
- **Which gate produces the silences.** `Step.silence` carries a `why`;
  `wire_chain` counts and discards it. The histogram would say whether the node
  is tripping on state age, reference age or `propagate` — a supervisor problem
  rather than a control one.
- **The scoring equilibrium is a third model.** `sway` is measured against
  `planner.model.passive_equilibrium`, not against the OCP's model and not
  against MuJoCo's settled state. If those disagree for a given
  `(mass, com)`, every sway number carries a constant bias. Not checked.
- **§3's recommendation is not applied.** `config/crane_mpc.yaml` still ships
  `weights.du: 1`, `horizon_length: 40`. This issue was asked for the map, and
  changing the shipped settings on an unvalidated harness (first bullet below)
  would be changing them on the strength of a simulation nobody has checked
  against Gazebo. That is a separate decision with a separate gate.
- `horizon_length=20` is 1.2 s of horizon. Nothing here checks that 1.2 s is
  still long enough for the *longest* CBS move, only for the moves in the
  sample — a horizon shorter than the move's own settling time is a different
  failure and would not show as sway.
- The gaps `README.md` records for `wire_chain` — the solve takes no time,
  measurement is exact at the cycle instant, the cadence anchor is exact, the
  passive state is read rather than estimated. Three of the four are stability
  gaps, and the cadence one is currently unbounded.
- The mismatches tested are *static*. A `k` that varies with load pressure, a
  Psi gain that differs per axis and per direction (which is the shape
  `mismatch.py` says it really has), a payload swinging on a sling rather than
  rigid on the mount — none of them are in the map.
- Psi gain is applied here as one multiplier on all five axes and both
  directions. Per-axis draws are what `sweep_robust --samples` does, and they
  are not in this map; §5 runs eight of them over 0.85..1.15.
- Only `payload x k x psi` was ever run as a grid. `tau_v`, `dead_time` and
  `com` appear one at a time only, so no statement here covers them paired with
  anything.
- Nothing here touches the real hydraulics — counterbalance valves, deadband,
  spool overlap, bore/rod asymmetry, load-sensing flow. C3 lumps the whole
  chain as velocity-in/joint-force-out by design. The envelope above is an
  envelope on `wire_chain`, and will be read as one on the machine unless it
  says otherwise. It says otherwise here.
- Nothing here validates the path that caused the 2026-09-30 failure. That was a
  wrong *declaration* — 800 kg for an 80 kg block. This study measures how much
  wrongness the loop absorbs; it does not stop the declaration being wrong.

## 5. Is the tuning forgiving once the delay is compensated? — issue 210

190 ran every sweep at `--latency 0.06`, a number quoted from an issue text
rather than measured. Issue 189 has since measured the live Gazebo loop (delivery
p50 26 / p90 30 / max 62 ms) and 209 moved `past_clamp` onto the commanded `u`.
The hypothesis: with the delay identified and compensated, stability is far less
sensitive to the tuning than 190 found.

**It is not.** Over the same grid the failure count goes 171/459 at 60 ms to
162/459 at 30 ms — two percentage points — and the failing cells do not move. The
levers that widen the envelope are the pair 190 already recommended, and they
widen it by the same amount at both latencies. Latency was not what 190 was
measuring.

Offline throughout: `wire_chain`/`sweep_robust` on MuJoCo, 209's gate, OCP
payload 800 kg. Nothing launched.

### `--latency` resolves to 10 ms, so 189's p50 and p90 are one point

`wire_chain` hands a horizon to the JTC on the first tick at or after
`cycle instant + --latency`, and the tick is 10 ms, so the delay that acts is
`10 ms * ceil(latency / 10 ms)`. `--latency 0.021`, `0.026` and `0.030` produce
byte-identical runs; `0.031` is the next distinct value. The issue asked for
three columns — 0.026 (189's p50), 0.030 (p90), 0.060 (190's) — and the harness
can only hold two. The 459-run grid below was run at 0.026 *and* at 0.030: all
459 rows match on every column including `exit`, which is the check rather than a
coincidence. **"30 ms" below is both of 189's measured numbers.** The p50/p90
spread `README.md` wants set separately is under this harness' resolution, not
merely unset.

### 1. Latency x detuning

    ./scripts/trials/sweep_robust.py --latency <0.026|0.030|0.060> \
        --payload-mass 800 --samples 0 --jobs 7 --by-plant \
        --band payload=800,80 --band k=1,2 --band psi=1,1.3 \
        --lever "weights.q_a=[20.0,20.0,20.0,20.0,5.0,5.0],[80.0,...,20.0,20.0]" \
        --lever "weights.dq_a=[0.2,...,0.05,0.05],[0.8,...,0.2,0.2]" \
        --lever weights.dq_u=128.0,512.0 --lever weights.u=0.5,2.0 \
        --lever weights.du=0.5,2.0 --lever weights.lag=20.0,80.0 \
        --lever weights.tcp=500.0,2000.0 --lever horizon_length=20,60

190's §2 grid — the plant takes payload x{800, 80} kg, C3 `k` x{1, 2}, Psi gain
x{1, 1.3}, three moves, 27 runs per setting. `q_a` and `dq_a` ship non-uniform
rows and are scaled *as rows* here; 190's `q_a=4`/`80` flattened them to a scalar,
which is a different perturbation.

| weight group | scale | 30 ms `late` | 30 ms failed | 60 ms `late` | 60 ms failed | nominal `late` 30 / 60 |
| --- | --- | --- | --- | --- | --- | --- |
| *(shipped)* | -- | 1.320 | 14/27 | 1.298 | 12/27 | 0.034 / 0.034 |
| `q_a` | x0.5 | 1.259 | 11/27 | 1.286 | 8/27 | 0.034 / 0.034 |
|  | x2 | 1.375 | 8/27 | 0.868 | 9/27 | 0.034 / 0.034 |
| `dq_a` | x0.5 | 1.279 | 10/27 | 1.279 | 10/27 | 0.034 / 0.034 |
|  | x2 | 1.293 | 9/27 | 1.456 | 11/27 | 0.034 / 0.034 |
| `dq_u` | x0.5 | 1.374 | 8/27 | 1.319 | 12/27 | 0.028 / 0.027 |
|  | x2 | 1.312 | 10/27 | 0.894 | 12/27 | 0.043 / 0.043 |
| `u` | x0.5 | 1.446 | 10/27 | 1.372 | 11/27 | 0.034 / 0.034 |
|  | x2 | 1.243 | 10/27 | 1.386 | 13/27 | 0.034 / 0.034 |
| `du` | x0.5 | 1.282 | 11/27 | 1.261 | 13/27 | 0.034 / 0.033 |
|  | x2 | 1.254 | **6/27** | 1.240 | **4/27** | 0.034 / 0.033 |
| `lag` | x0.5 | 1.377 | 8/27 | 1.358 | 12/27 | 0.034 / 0.034 |
|  | x2 | 1.214 | 9/27 | 1.321 | 9/27 | 0.034 / 0.034 |
| `tcp` | x0.5 | 1.229 | 10/27 | 1.402 | 9/27 | 0.043 / 0.041 |
|  | x2 | 1.305 | 10/27 | 1.295 | 11/27 | 0.030 / 0.029 |
| `horizon_length` | 20 | 1.257 | **6/27** | 1.251 | **4/27** | 0.025 / 0.025 |
|  | 60 | 0.803 | 12/27 | 1.274 | 11/27 | 0.023 / 0.022 |

**On this grid the worst `late` column is a restatement of the divergence count,
and nothing else.** Over all 34 setting x latency groups, `late > 1.2` holds
exactly when the group contains a `diverged` run — 34 of 34, no exception. That
follows from the instrument: `DIVERGED_SWAY` is 1.2, a diverged run breaks the
moment |sway| crosses it, and the last 3 s of a log that was cut there contain
that crossing. Since `diverged` itself runs 0-2 per setting and tracks nothing
(as in 190), the worst `late` inherits that and ranks nothing either. The issue
asked to score on `sway_late` + failures; on this grid only the failure count and
the *nominal* `late` carry information, and `late` recovers its meaning in §2's
one-at-a-time families, where runs finish.

- **Latency is not the lever.** 162/459 against 171/459 is the whole effect of
  halving the delivery delay, and per plant cell it is noise in both directions:
  payload-80 x `k`-2 x Psi-1.3 goes 44/51 to 47/51, payload-800 x `k`-2 x Psi-1.3
  goes 18/51 to 15/51. Three cells — nominal, payload-800 x `k`-1 x Psi-1, and
  payload-800 x `k`-2 x Psi-1 — are 0/51 at *both* latencies across all 17
  settings, and `k` x2 on a correctly declared payload is one of them. The
  mismatch decides; the delay does not.
- **The answer per lever is "no more forgiving".** Each weight group x0.5 and x2
  lands between 6/27 and 14/27 at 30 ms and between 4/27 and 13/27 at 60 ms, i.e.
  the same broad band 190 reported, at the same width, with the same two
  exceptions. Nothing in the table is a lever that the compensated delay unlocks.
- **`weights.du` x2 and `horizon_length=20` are the exceptions, at both
  latencies** — 6/27 and 6/27 at 30 ms, 4/27 and 4/27 at 60 ms, against shipped's
  14 and 12. That reproduces 190's §2 finding (`du=10` and `h=20`, 4/27 each) on
  the new gate and at the measured latency, which is the one thing here that
  transfers. Both shrink the *set* of failing cells rather than thinning it:
  shipped fails in six cells at both latencies, `du` x2 in five (30 ms) / three
  (60 ms), `h=20` in four / three. Both still fail in payload-80 x Psi-1.3, which
  is 190's conclusion unchanged.
- **Shipped is not a local optimum of robustness.** At 30 ms every one of the
  sixteen detunings scores at or below shipped's 14/27; at 60 ms eleven of
  sixteen do and two are worse. With 27 runs and ~12 failures most of those gaps
  are inside noise and the ranking among them should not be used — but the
  direction is consistent across two latencies and the shipped row is at the
  pessimistic end of the band, not the middle. 190's "no weight buys tolerance"
  survives; "shipped is a reasonable place in that band" does not.
- **Detuning does not touch the plant the MPC was tuned for.** The nominal
  column is 0.022..0.043 rad of `late` over all 17 settings at both latencies,
  final error <= 2.9 mm, no failure in any of the 102 nominal runs. In that
  narrow sense the tuning *is* forgiving, and it was already forgiving at 60 ms.
  What no weight buys is mismatch tolerance.
- **The silence share did not improve either.** Over both 459-run grids,
  median 4 %, p90 25 %, max 26 % of cycles with nothing published, identical at
  the two latencies. Lower than the p90 46 % 190 measured, but still a quarter of
  the worst run's cycles flying a stale horizon, and 190's §4 question — whether a
  setting recommended for hardware holds it low — is answered **no** in §2 below:
  shipped, C3 and the recommendation all sit at p90 22-23 %, max 24-26 %.

Not comparable to 190's printed numbers: 209 established that those tables do not
reproduce on today's `crane_mpc`. **The 60 ms column here is the like-for-like
baseline**, measured in this same study, and the 30 ms column is read against it
and against nothing else.

### 2. The mismatch envelope at the measured latency, three variants

    ./scripts/trials/sweep_robust.py --latency 0.026 --payload-mass 800 \
        --jobs 7 --by-plant --samples 0 <variant> --band <family>=<values>
    # variants: (none) | --set command_state=false --set weights.du=10
    #                  | --set weights.du=10 --set horizon_length=20
    # families: payload=80,400,1200,1600  k=0.5,0.7,1.5,2  psi=0.7,0.85,1.15,1.3
    #           com=[0.25,0,0],[0.5,0,0],[0,0,0.25],[0,0,0.5]
    #           tau_v_offset=-0.03,0.03   dead_time=0.03,0.09
    #           psi=0.85,1.15 --samples 8 --seed 1      (per-axis draws)

One sweep per family so the rows stay one-at-a-time. Three moves per cell, worst
of the three; `late` in rad, then the cell's failure count.

| family | cell | shipped | C3 (`command_state=false`, `du=10`) | 190's rec (`du=10`, `h=20`) |
| --- | --- | --- | --- | --- |
| *(none)* | nominal | 0.034 / 0/3 | 0.031 / 0/3 | 0.011 / 0/3 |
| payload | 80 kg (0.1x) | 0.633 / **1/3** | 0.769 / **3/3** | 0.329 / 0/3 |
| | 400 kg | 0.032 / 0/3 | 0.038 / 0/3 | 0.033 / 0/3 |
| | 1200 kg | 0.057 / 0/3 | 0.055 / 0/3 | 0.038 / 0/3 |
| | 1600 kg | 0.074 / 0/3 | 0.071 / 0/3 | 0.061 / 0/3 |
| C3 `k`, `d` | x0.5 | 0.088 / 0/3 | 0.085 / 0/3 | 0.084 / 0/3 |
| | x0.7 | 0.068 / 0/3 | 0.066 / 0/3 | 0.051 / 0/3 |
| | x1.5 | 0.032 / 0/3 | 0.040 / 0/3 | 0.033 / 0/3 |
| | x2 | 0.126 / 0/3 | 0.428 / **1/3** | 0.090 / 0/3 |
| Psi gain | x0.7 | 0.111 / 0/3 | 0.154 / 0/3 | 0.003 / 0/3 |
| | x0.85 | 0.013 / 0/3 | 0.011 / 0/3 | 0.022 / 0/3 |
| | x1.15 | 0.106 / 0/3 | 0.787 / **2/3** | 0.050 / 0/3 |
| | x1.3 | 0.394 / **3/3** | 0.799 / **3/3** | 0.190 / 0/3 |
| payload COM | +0.25 m x | 0.042 / 0/3 | 0.066 / 0/3 | 0.022 / 0/3 |
| | +0.50 m x | 0.233 / **3/3** | 0.338 / **3/3** | 0.137 / 0/3 |
| | +0.25 m z | 0.038 / 0/3 | 0.035 / 0/3 | 0.016 / 0/3 |
| | +0.50 m z | 0.043 / 0/3 | 0.038 / 0/3 | 0.023 / 0/3 |
| `tau_v` offset | −30 ms | 0.015 / 0/3 | 0.012 / 0/3 | 0.014 / 0/3 |
| | +30 ms | 0.069 / 0/3 | 0.069 / 0/3 | 0.023 / 0/3 |
| dead time | 30 ms (−30) | 0.014 / 0/3 | 0.012 / 0/3 | 0.014 / 0/3 |
| | 90 ms (+30) | 0.070 / 0/3 | 0.070 / 0/3 | 0.022 / 0/3 |
| Psi per axis | 8 draws, 0.85..1.15 | 0.071 / 0/24 | 0.324 / **2/24** | 0.085 / 0/24 |
| **all, de-duplicated** | | **7/87** | **14/87** | **0/87** |

The totals count each distinct cell once: the Psi-x0.85 and x1.15 corners are in
two families and the nominal cell in seven, so the per-family counts above sum
higher than 87.

- **190's recommendation holds the widest envelope, and it is not close: 0/87.**
  `weights.du=10` with `horizon_length=20` and `command_state` left true clears
  every cell of every family — payload 0.1x, Psi x1.3, 0.5 m of COM error, ±30 ms
  of lag, and all eight per-axis Psi draws — where shipped fails 7 and C3 fails
  14. It also settles the nominal plant better (`late` 0.011 against shipped's
  0.034), at the faster transient §3 recorded (nominal sway peak 0.28 against
  0.15, from table 1's `horizon_length=20` row).
- **It does not clear the paired grid**, only the one-at-a-time families: table 1
  has `h=20` at 6/27 and `du` x2 at 6/27 on payload x `k` x Psi. So the envelope
  that closes here closes over each axis alone, exactly as 190's did, and the
  payload-80 x Psi-1.3 corner is still outside it for every setting tried.
- **C3 is worse than C4 under mismatch at the measured latency too** — 14/87
  against 7/87, and it is the only variant that fails on `k` x2, on Psi x1.15 and
  on a per-axis Psi draw. Same verdict as 190's §2 (12/27 against 8/27) and
  `README.md`'s `--wire 161` check, now at the measured delay.
- **The arm's lag is not a sensitive axis.** This is the run 190's §4 asked for
  and could not make: `scale_tau_v` is a multiplier and the arm is fitted at
  `tau_v = 0`, so every `tau_v` family in §1 left the axis carrying the payload
  moment arm untouched. `--tau-v-offset` is additive (new here, floored at zero,
  `wire_chain.offset_tau_v`) and ±30 ms on all five axes — the arm included — costs
  nothing: 0/9 for all three variants, worst `late` 0.069 rad. The `tau_v` and
  dead-time rows are also near-identical cell for cell (0.015/0.069 against
  0.014/0.070), which is what "only the sum is identified" looks like from the
  plant side.
- **`Psi gain <= ~1.15x` now rests on a measurement.** 190 called that the
  weakest line in its envelope, because x1.15 sat one step below an x1.3 whose
  failures might have been the old gate's artifact. On 209's gate x1.15 is clean
  at shipped settings (0/3, `late` 0.106, against 190's §1 reading of 0.109 with
  a +0.27/s slope), and so are eight per-axis draws spanning 0.85..1.15 — the
  per-axis shape `mismatch.py` says the error really has, and which was not in
  190's map at all. x1.3 remains the wall: 3/3 at shipped, 3/3 at C3, and the one
  family where the recommendation's margin is visibly spent (`late` 0.190 against
  0.003..0.137 everywhere else).

### What this still cannot see

The four `README.md` gaps that bear on stability rather than tracking are all
live here, and the verdict above should be read with them:

- **The solve takes no time.** `wire_chain` sets `ocp.solve_budget_s = 10.0` and
  `--latency` is delivery only, so a late solve never lands late and
  `applied_previous_solution` — a different closed loop — never happens. Live,
  under the full interactive stack (GUI, rviz, perception, load avg 20+), solves
  ran **75-173 ms against a 60 ms budget** (189). A study whose own conclusion is
  "latency is not the lever" is the wrong instrument for saying what that does.
- **Measurement is exact at the cycle instant.** No encoder quantisation, no
  sensor age, and the passive state is *read*, not estimated — live it comes from
  a complementary filter whose lag is inside the sway loop. Every sway number
  above is a lower bound on the phase the real loop closes.
- **The cadence anchor is exact**, and 189 established the phase is not
  observable from a capture, so this gap is still unbounded.
  `--cadence-phase 0.02` is a divergence on the nominal plant — the harness'
  largest single known sensitivity, and it was held at 0 for all 1 710 runs here.
- **The OCP model is ~3x slower than the MuJoCo plant** over the first 300 ms on
  boom, arm, telescope and rotator (189, `plant_step.py`; slewing agrees). So the
  "nominal" plant of every table above is already a mismatched one in the
  direction the study does not sweep, and the one-at-a-time `k` and `tau_v`
  families perturb around a point that is itself off.

Carried over from §4 and unchanged: `past_clamp` fires on a quantity
`velocity_loop` does not bound (209), the scoring equilibrium is a third model,
every mismatch here is static, and nothing touches the real hydraulics.

### For 211

The recommendation this section supports is **190's, unchanged**:

    weights.du: 10            (ships 1)
    horizon_length: 20        (ships 40)

with `command_state` left **true**. It is now measured on 209's gate, at 189's
measured latency, and it is 0/87 on the one-at-a-time envelope against shipped's
7/87 — a stronger result than 190 had, and no longer resting on an unfitted
0.06. No config change was made here; `config/crane_mpc.yaml` still ships
`du: 1`, `horizon_length: 40`.

Two things 211 should weigh against it, neither new:

- The paired corner payload-80 x Psi-1.3 fails at every setting tried, at both
  latencies. If a grasp can silently be a tenth of the declared mass *and* Psi
  can be 30 % high, no MPC setting in this study is an answer — `command_domain`,
  a Psi refit, or feedback on the achieved joint velocity is.
- `horizon_length=20` is 1.2 s, and nothing here checks that 1.2 s covers the
  longest CBS move. A horizon shorter than the move's own settling time fails in
  a way that does not show as sway.

## 6. The shipped settings moved — issue 223

`horizon_length` 40 -> 60, `weights.dq_u` 256 -> 1024, `hpipm_mode` BALANCE ->
SPEED_ABS, `qp_solver_iter_max` 50 -> 35. Those came out of a cost-and-damping
sweep, not out of this study, so the gate was §5.2's envelope re-run on both:

    ./scripts/trials/sweep_robust.py --latency 0.026 --payload-mass 800 \
        --samples 0 --jobs 7 --by-plant --band <family>=<values>
    # new: as shipped.  old: CRANE_MPC_OCP_OPTIONS='{"hpipm_mode":"BALANCE",
    #      "qp_solver_iter_max":50}' --set horizon_length=40 --set weights.dq_u=256.0

Same seven families, three moves a cell, 34 distinct cells = 102 runs each.

**New 0/102, old 0/102.** No cell fails at either setting, so the envelope is
unchanged and the question this section was asked is answered "no worse". Three
things to read with it:

- **Neither column is §5.2's.** §5.2 scored shipped at 7/87 and that does not
  reproduce: 209 had already established these tables do not hold on today's
  `crane_mpc`, and the PZS description has since moved again (issues 222, 224).
  The `old` column is the like-for-like baseline, measured in this same run.
- **`handbk` is where the new setting shows.** On the *nominal* plant the old
  setting hands back on 0.27 of its cycles and the new one on 0.00 — the
  `qp_solver_iter_max` change, which converts a cycle that would wander to
  `ACADOS_MINSTEP` and be refused into one that publishes its own truncated
  step. Most mismatched cells move the same way.
- **The unsettled cells are the same cells**, and about as unsettled: Psi x1.3
  `late` 0.352 against 0.363, COM +0.5 m 0.237 against 0.228, payload 80 kg
  0.172 against 0.199. Nothing here widens the envelope and nothing narrows it;
  §3's and §5's conclusions about *which* mismatch is survivable stand.

## 7. The sway centre became a table, and `weights.q_u` was paid — issue 229

`q_eq` was pinned at the propagated measurement and held across the horizon, so
the offset residual charged the hanging pose's *own* travel — 1:1 with
`-(q_boom + q_arm)`, ~0.9 rad over an A2B move against a 0.2 rad box — and
`weights.q_u` had to ship at zero, leaving the tilt damped by rate alone.
`equilibrium.py` gives the OCP one hanging pose per stage, so the residual is
the swing; this section is the sweep that then priced `q_u`.

### The sweep

226's eight cells (A2B, `--goal out`, two random moves, each empty and at
800 kg), `--latency 0.026`, worst cell per setting:

    ./scripts/trials/wire_chain.py <cell> --latency 0.026 \
        --set weights.q_u=<q> --set weights.dq_u=<dq>

| q_u | dq_u | late tip/tilt rate | peak sway | final err | ms p50/p90/max | capped | refused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1024 | 0.045/0.047 | 0.132 | 0.0048 | 7.2/15.9/22.4 | 0.4 % | 24.8 % |
| 20 | 1024 | 0.045/0.046 | 0.132 | 0.0047 | 8.0/16.3/54.6 | 1.3 % | 23.9 % |
| 100 | 1024 | 0.044/0.046 | 0.134 | 0.0047 | 8.1/15.0/23.7 | 0.9 % | 25.2 % |
| 400 | 1024 | 0.044/0.045 | 0.134 | 0.0046 | 10.0/19.3/35.1 | 0.4 % | 24.8 % |
| 1600 | 1024 | 0.045/0.040 | 0.133 | 0.0046 | 22.0/51.9/131.4 | 0.9 % | 24.3 % |
| 3200 | 1024 | 0.047/0.042 | 0.131 | 0.0047 | 7.3/13.8/29.3 | 1.7 % | 23.5 % |
| **6400** | **1024** | **0.048/0.029** | **0.129** | **0.0041** | **9.6/14.7/59.6** | **0.9 %** | **21.3 %** |
| 12800 | 1024 | 0.067/0.049 | 0.126 | 0.0085 | 7.7/14.7/18.5 | 0.4 % | 23.9 % |
| 25600 | 1024 | 0.055/0.055 | 0.124 | 0.0044 | 7.5/15.0/22.5 | 0.9 % | 21.7 % |
| 6400 | 512 | 0.042/0.042 | 0.138 | 0.0043 | 8.9/14.6/31.4 | 0.4 % | 24.3 % |
| 6400 | 2048 | 0.035/0.050 | 0.114 | 0.0047 | 7.7/15.6/59.3 | 1.7 % | 23.5 % |
| 6400 | 4096 | 0.080/0.056 | 0.098 | 0.0053 | 12.8/21.1/29.7 | 2.2 % | 23.0 % |

The worst column is the wrong place to read this from: `out p800` dominates
every row and it is the cell 231 is about — the arm starts 0.24-0.41 rad outside
the control-safe box, so a fifth of its cycles refuse whatever the weight is.
Per cell, late tip/tilt rate with the refused share in brackets:

| q_u | a2b empty | a2b p800 | out empty | out p800 | r1 empty | r1 p800 | r2 empty | r2 p800 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.027/0.024 (0) | 0.027/0.019 (0) | 0.028/0.041 (25) | 0.045/0.022 (23) | 0.020/0.022 (0) | 0.024/0.015 (0) | 0.011/0.047 (0) | 0.013/0.031 (0) |
| 6400 | 0.019/0.018 (0) | 0.024/0.016 (0) | 0.015/0.017 (2) | 0.048/0.029 (21) | 0.016/0.018 (0) | 0.020/0.014 (0) | 0.007/0.029 (0) | 0.010/0.025 (0) |
| 12800 | 0.017/0.016 (0) | 0.021/0.014 (0) | 0.029/0.049 (24) | 0.067/0.030 (6) | 0.014/0.017 (0) | 0.017/0.015 (0) | 0.007/0.023 (0) | 0.009/0.022 (0) |

Both rates fall with the weight on all six cells that do not refuse, and keep
falling past 6400 — 6400 is where the *cost* starts: the final joint error on a
clean cell goes 0.0002 -> 0.0005 rad at 6400 and 0.0002 -> 0.0012 at 12800,
and 12800's worst cell is 0.0085. `dq_u` was swept with it and stays at 1024;
512 and 2048 trade the two axes against each other and 4096 buys peak sway
(0.129 -> 0.098) by moving the residual onto the tip (0.048 -> 0.080).

### Live, and this is where it stops agreeing

Three interleaved Gazebo pairs, `gazebo_wall_assembly_pzs100.launch.py` with
`enable_livox_sim:=off gui:=False initial_pose:=2`, A2B to (6.06, 0.35, 2.0),
no payload, `solver_health_decimation 1`; `on` is the shipped 6400, `off` the
same tree with `q_u` patched to 0 in the *installed* yaml. Sway rate is the max
|velocity| on each passive row over arrival+2..5 s.

| run | load (1 min) | late tip/tilt | solve ms p50/p90/max | refused | over budget |
| --- | --- | --- | --- | --- | --- |
| on 1 | 3.14 | 0.028/0.045 | 19.1/23.4/55.6 | 0/465 | 0 |
| off 1 | 9.98 | 0.028/0.038 | 17.2/20.7/72.4 | 0/475 | 1 |
| on 2 | 11.49 | 0.029/0.025 | 14.1/18.2/36.7 | 0/466 | 0 |
| off 2 | 11.18 | 0.033/0.034 | 14.8/18.7/44.2 | 0/472 | 0 |
| on 3 | 10.96 | 0.037/0.043 | 13.5/16.1/34.5 | 0/480 | 0 |
| off 3 | 7.55 | 0.040/0.037 | 13.6/16.4/40.1 | 0/474 | 0 |

A second round, four more `on` runs after the centre's closed form replaced the
affine one (no change for these bags — they carry no payload, where the two
agree to 1e-12 — but the tree underneath had been rebuilt):

| run | load (1 min) | late tip/tilt | solve ms p50/p90/max | refused | over budget |
| --- | --- | --- | --- | --- | --- |
| on 4 | 16.33 | **0.019/0.014** | 14.0/16.7/49.7 | 0/463 | 0 |
| on 5 | 15.54 | 0.035/0.030 | 13.2/15.8/22.3 | 0/475 | 0 |
| on 6 | 4.69 | 0.036/0.041 | 13.0/16.2/32.5 | 0/481 | 0 |
| on 7 | 18.89 | 0.043/0.039 | 13.2/15.9/39.6 | 0/477 | 0 |
| off 6 | 6.68 | 0.040/0.034 | 13.6/16.6/31.8 | 0/475 | 0 |

**The weight does not separate live, and the target is not reached.** Over seven
`on` runs the late rate spans 0.014–0.045 rad/s and exactly one of them — on 4,
at the *highest* load of the set — lands under 0.02 on both rows. The spread
between nominally identical runs is wider than any `on`/`off` difference, and it
does not track load. So 0.02 rad/s is not met, and the sweep's offline ordering
does not reproduce here.

0.02 is nonetheless a measurable target, not a noise floor: both passive rows
read exactly 0.000 rad/s over the whole pre-move window of every bag, so
0.03–0.045 is real residual swing. What live does say is that 6400 **costs
nothing** — solve p50/p90, `qp_capped` and the refused share all sit on 227's,
0/3277 cycles refused across all seven runs, and the single over-budget cycle in
thirteen runs is an `off` run.

Why offline and live disagree is open, and these runs cannot close it. They
differ in more than the weight: they start at `initial_pose:=2` because
`crane_planning`'s concurrent branch refuses the pose 226 and 227 captured from,
`crane_model` and `crane_planning` were themselves being edited on another
branch in this checkout while rounds one and two ran (the same race put three
`crashed: libacados...so: file too short` into the envelope table below), the
box sat between load 3 and 19, and §5's warning that this harness is unvalidated
against Gazebo has not been retired. The honest reading is that the structural
change is right and measured, and the weight it unlocks is offline-only
evidence.

### The envelope, §6's gate re-run

Same seven families, shipped against the one lever that moved:

    ./scripts/trials/sweep_robust.py --latency 0.026 --payload-mass 800 \
        --samples 0 --jobs 7 --by-plant --lever weights.q_u=0.0 --band <family>=<values>

**New 0/102, old 0/102** — no cell fails at either setting, so the envelope is
unchanged and §3's and §5's conclusions about which mismatch is survivable
stand. (`tau_v_offset` logged 3/9 `crashed:` on the shipped column, all three
`t_renderer`/`libacados_ocp_solver...so: file too short` — `crane_planning`'s
OCP cache being regenerated under the run by a concurrent branch, not a control
failure. The lever column, run minutes later, is clean on the same cells.)

Two cells move enough to name, and they move opposite ways:

- **Payload 80 kg is where the weight earns its keep.** `late` 0.118 -> 0.040,
  final error 0.0214 -> 0.0064 rad. That is 190's original failure — the tree
  declaring 800 kg over an 80 kg block — and it is the most mismatched cell the
  weight helps rather than hurts.
- **An off-axis *plant* CoM is where it costs.** COM +0.5 m: `late` 0.111 ->
  0.185, peak 0.247 -> 0.253, `grow/s` 6.10 -> 7.39. The centre is exact at both
  settings here — the band moves the plant's CoM, never the OCP's — so this is
  the weight pricing an offset the plant will not hold, and paying for it in the
  one family §3 already called the least survivable. +0.25 m does not do it
  (0.035 -> 0.041) and neither does an offset along z.

## 8. Can the linearisation be made cheaper? — issue 230

226 left the latency as the thing that shifts the previous horizon and named
three candidates: fewer knots, a cheaper integrator, or the split behind a
predicate. The first two are measured here and **neither is taken**.

The A2B cell (`--goal-pose 6.06 0.35 2.0 1.5708 --latency 0.026`, empty,
`OMP_NUM_THREADS=1`), four repeats **interleaved** so the box's load cannot
order the settings — it moves the absolute scale by a factor of two inside one
sweep, which is how an earlier non-interleaved run read `newton_iter 1` as 30 %
cheaper:

| `CRANE_MPC_OCP_OPTIONS` | median of the four p50 | p90 | worst max |
| --- | --- | --- | --- |
| shipped | 12.13 ms | 15.23 | 23.72 |
| `sim_method_newton_iter: 2` | 11.40 | 14.06 | 23.69 |
| `sim_method_newton_iter: 1` | 11.05 | 13.75 | 22.17 |
| `horizon_length: 50` | 9.70 | 14.28 | 30.38 |

**`sim_method_newton_iter` is a null result, and acados' own timers say why.**
On twelve replayed ticks `time_lin` is 10.64 ms shipped and 10.55 ms at one
Newton iteration — unmoved. With `sim_method_jac_reuse: 1` and a warm iterate
the second and third iterations are a back-substitution on a residual that is
already small; the cost of the linearisation is the *sensitivity* propagation,
which runs once however many Newton steps precede it. The 1 ms the closed-loop
table shows is inside the run-to-run spread.

**`sim_method_num_stages: 1` is the only cheap integrator, and it breaks the C3
force rows.** `time_lin` 10.64 -> 5.94 ms, exactly the factor the stage count
predicts — and on the A2B cell the final error goes 0.0003 -> 0.0132 rad, the
late tilt rate 0.018 -> **0.39** rad/s and 4 % of QPs stop at their iteration
cap. Backward Euler is L-stable but first order, and at `|lambda| T_s = 8.5` on
the actuator lag that order is the model. `sim_method_num_stages: 3` costs 12 %
and changes no quality column.

**Fewer knots works and is not free.** `horizon_length: 50` buys 20 % of the
solve for a late tip rate of 0.0217 against 0.0194 and a late tilt rate of
0.0199 against 0.0176; 40 buys 48 % for 0.0236/0.0210. 223 bought those knots
for the damping and this does not undo that trade for 2.4 ms.

On the eight cells of 226 §3 (four moves x {empty, 800 kg}) `newton_iter 1` is
bit-identical to the shipped setting on every quality column, which is the
other half of calling it a null result.

**Nothing was taken, and `solve_budget` was not widened either.** The one lever
that costs no solve time looked like publishing a late horizon anyway --
`command_delay` is two `T_s` while the budget is one -- and it is wrong: the
horizon is stamped at the measurement and knot 0 sits one `T_s` into it, behind
`hz._command_message`'s in-flight prepend, so past `stamp + T_s` the JTC has
already played that prepend and the command steps instead of ramping. The
second `T_s` is the dead time to the joint, not slack for the message.
`horizon_length: 50` is the only honest remaining trade here and 2.4 ms does
not buy back what 223 paid those knots for.

## 9. The C3 force estimator, and why it is gone — issue 233

227 shipped `force_filter` with `enabled: false`: a per-axis steady-state Kalman
filter on `(dq_a, tau_a)` that held 5-79x of the velocity noise out of the force
rows, bought by trusting C3 and so paid for in the mismatch envelope. 227's own
reading of that trade was that the filter *rejected* the gravity error it should
have estimated, and named a force-bias state as the fix. **It was built,
measured, and it is not the fix. The filter and its parameters are deleted;
`git show 524faa5` is the whole of it if a future issue wants it back.**

The third state was a per-axis random walk `b` entering the force row,
`tau+ = A10 dq + A11 tau + b`, with `tau_a_bias` its process variance — zero
reproducing 227's two-row filter exactly, which is the control every number
below is read against.

**It does estimate a sustained error.** Share of a constant unmodelled force
(10 % of each axis' hold force) that has reached `tau_hat` by cycle 200, at the
resting pose, with `tau_a_bias = m * tau_a`:

| axis | `A[1,1]` | m = 0 | 1e-2 | 1e-1 | 1 |
| --- | --- | --- | --- | --- | --- |
| slewing | 0.98 | 95 % | 99 % | 100 % | 100 % |
| boom | 0.78 | 51 % | 84 % | 99 % | 100 % |
| arm | 0.61 | 21 % | 57 % | 89 % | 100 % |
| telescope | 0.18 | −2 % | 83 % | 100 % | 100 % |
| rotator | 0.51 | 17 % | 42 % | 73 % | 97 % |

At m = 0 a sustained force error is **permanently** rejected on three of five
rows, which is 227's mechanism as a number; the bias state removes that. Noise
rejection pays for it, the telescope first (its sensor variance is 1.8e-7): over
the three i226 bags with the measured sensor noise added, `m = [0, 1e-1, 5e-2,
0, 0]` holds 5.6-9.2x out of the boom and arm against the two-row filter's
11-79x, and is the largest `m` that keeps 14 of 15 axis-runs at or above the 5x
gate — the same 14 of 15 as shipped.

**And it buys nothing where it was supposed to.** `sweep_robust.py --latency
0.026 --payload-mass 800 --samples 0 --by-plant --band payload=80,400`, worst of
three moves, `late` in rad (load 0.9-7.1 over the sweeps):

| setting | payload 80 kg | payload 400 kg |
| --- | --- | --- |
| filter off | **0.040** | **0.028** |
| on, `tau_a_bias` 0 (= 227) | 0.396 | 0.219 |
| on, the candidate above | 0.390 | 0.229 |
| on, `m` = 1 / 1e1 / 1e2 uniform | 0.348 / 0.314 / 0.320 | — |

**No bias variance over four decades recovers the cell** — not even m = 1e2,
where the noise rejection is 0.26-0.29x, i.e. the filter makes the force rows
worse than the open-loop carry. What does recover it is `dq_a`, exactly as 227
said: `late` 0.461 (1e-7) / 0.107 (1e-6) / 0.047 (1e-5) / 0.041 (1e-4) against
0.040 off — and at 1e-5 and 1e-6 the rejection is 1.1-2.9x whatever the bias
does (measured with the bias at 0, at the candidate and at m = 1; the three
columns agree to 10 %).

So the conflict 227 measured is **not** the rejected gravity error. It is the
velocity row: `x_0`'s `dq_a` is what the MPC plans the next 3.5 s from, and a
`dq_a` the filter low-passes is a velocity the plant does not have. A bias state
cannot fix a state that is wrong rather than missing. The two acceptance
criteria remain in direct conflict and no third state reconciles them.

Two notes for whoever comes back to this:

- **The baseline moved.** 227 measured payload-80 `late` at 0.114 off; in this
  checkout it is 0.040 (231 moved `position_box`, 229 repriced `weights.q_u`).
  The filter's cost is therefore *larger* relative to today's baseline than 227
  reported, not smaller. Re-measure, do not read 227's column.
- **228's chamber pressure is the remaining route.** It makes `tau_a` a
  measurement instead of an estimate, and a measured force needs no trust in C3
  at all — which is the only thing here that was ever the problem.
