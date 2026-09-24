#!/usr/bin/env python3
"""
Sweep the chain over application delays, PI rungs and plant samples. One CSV row per run.

Issue 156's oracle. Drives `sim_chain.py` once per cell as a subprocess -- same
plant, same solver, same inner loop -- and reads the score JSON it writes.
Nothing here reimplements the chain, and nothing here chooses a band: the bands
arrive on the command line, because picking them is the follow-up.

The deliverable is the delay at which each PI rung goes marginal, so the default
grid is the delay axis alone at the nominal plant, from zero past the 0.5 s the
pump needs to build flow. `--k-band`/`--psi-band` add the plant mismatch on top.

Two seeds minimum, and the **worst** column is the one reported: a sampler
almost never lands on the joint worst case and that case is what breaks a crane.
So every band contributes its corners deterministically before a single random
draw is taken.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
HARNESS = PACKAGE / "scripts" / "sim_chain.py"
OUT_ROOT = PACKAGE / "build" / "sweep_mismatch"

#: `crane_model/config/c3_full_model.json`, in `K_AXIS_KEYS` order, read here
#: only to name the axes a row reports. Never written.
AXIS_NAMES = ("slew", "boom", "arm", "telescope", "rotator")

#: Where the machine sits: issue 155 measured a 27.8 ms median solve headless
#: and a 53-118 ms p90 active under Gazebo GUI + RViz, against an 80 ms budget.
#: The modelled 60 ms is therefore a factor-two understatement at the top end,
#: and the pump's own ~0.5 s flow build-up is an order past that -- so the grid
#: reaches it. Marks on the sweep, not a design value.
DEFAULT_DELAYS = (0.0, 0.03, 0.06, 0.09, 0.12, 0.25, 0.5)

#: The columns a row is judged on. `worst` means max over the cell's moves.
SCORE_KEYS = (
    "path_worst",
    "goal_final",
    "sway_peak",
    "sway_end",
    "reversals_worst",
    "dq_sustained_worst",
)


def samples(arguments, rng) -> list[dict]:
    """
    Draw one cell's plant samples: every band's corners, then the random draws.

    A band of zero contributes no knob at all, so the default grid is the
    nominal plant and the delay axis on its own.
    """
    knobs = {}
    if arguments.k_band > 0.0:
        knobs["k"] = (1.0 - arguments.k_band, 1.0 + arguments.k_band)
    if arguments.psi_band > 0.0:
        knobs["psi"] = (1.0 - arguments.psi_band, 1.0 + arguments.psi_band)
    if arguments.lag_shift != 0.0:
        knobs["lag"] = (arguments.lag_shift, 0.0)

    drawn = [dict(zip(knobs, corner)) for corner in itertools.product(*knobs.values())]
    for corner in drawn:
        corner["kind"] = "corner"
    for _ in range(arguments.samples):
        draw = {"kind": "random"}
        for knob, (low, high) in knobs.items():
            if knob == "lag":
                # To the millisecond, which divides the 0.5 ms plant step. A raw
                # uniform draw almost never leaves `60 ms - shift` a whole number
                # of steps, and `C3Actuator` refuses a fractional dead time -- so
                # every random lag cell would have died on arithmetic.
                draw[knob] = round(float(rng.uniform(low, high)), 3)
            else:
                # Per axis, and independently: the axes' fits were identified one
                # axis at a time, so there is nothing to correlate them with.
                draw[knob] = rng.uniform(low, high, len(AXIS_NAMES))
        drawn.append(draw)
    # No band at all still leaves one cell: the nominal plant, which is the
    # delay study's own baseline.
    return drawn


def cell_field(sample: dict, knob: str) -> str:
    """Render a knob's draw as one CSV cell; empty if the knob was not in the grid."""
    if knob not in sample:
        return ""
    axes = np.broadcast_to(np.asarray(sample[knob], float), len(AXIS_NAMES))
    return " ".join(f"{value:.6g}" for value in axes)


def cell_arguments(sample: dict) -> list[str]:
    """One sample as `sim_chain.py` flags. `k` moves `d` with it, never alone."""
    flags: list[str] = []
    if "k" in sample:
        scale = np.broadcast_to(np.asarray(sample["k"], float), len(AXIS_NAMES))
        # k and d are one identification; the harness refuses one without the
        # other, so the same multiplier goes on both rather than moving zeta.
        for name in ("--k-scale", "--d-scale"):
            flags += [name, *(f"{value:.6g}" for value in scale)]
    if "psi" in sample:
        gain = np.broadcast_to(np.asarray(sample["psi"], float), len(AXIS_NAMES))
        for name in ("--psi-gain-positive", "--psi-gain-negative"):
            flags += [name, *(f"{value:.6g}" for value in gain)]
    if sample.get("lag"):
        flags += ["--lag-shift", f"{sample['lag']:.6g}"]
    return flags


def run_cell(
    delay: float, rung: str, seed: int, sample: dict, extra: list[str]
) -> dict:
    """One `sim_chain.py` run. Returns the worst move in it, or why it failed."""
    stem = f"d{int(round(delay * 1000)):03d}_{rung}_s{seed}_{sample['kind']}"
    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    score_json = out_dir / "score.json"
    command = [
        sys.executable,
        str(HARNESS),
        "--apply-delay",
        str(delay),
        "--pi-rung",
        rung,
        "--random",
        str(max(1, sample.get("moves", 1))),
        "--seed",
        str(seed),
        "--score-json",
        str(score_json),
        *cell_arguments(sample),
        *extra,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (out_dir / "log.txt").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0 or not score_json.exists():
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        return {"failed": tail[-1] if tail else "no output"}

    scored = json.loads(score_json.read_text())
    moves = scored["moves"]
    if not moves:
        return {"failed": "the harness scored no move"}
    row = {key: max(move[key] for move in moves) for key in SCORE_KEYS}
    row["hunting"] = any(move["hunting"] for move in moves)
    row["hunting_axes"] = sorted(
        {axis for move in moves for axis in move["hunting_axes"]}
    )
    # Per axis, so the 1/p ordering is readable off the row rather than inferred.
    worst = np.max([move["e_pos_worst"] for move in moves], axis=0)
    row.update(
        {f"e_pos_{name}": float(value) for name, value in zip(AXIS_NAMES, worst)}
    )
    row["refused"] = sum(move["refused"] for move in moves)
    row["cycles"] = sum(move["cycles"] for move in moves)
    # Which branch of `adopt_solution` filled `positions`: on the curve the
    # optimizer picks the reference the integral action works against.
    row["reference_branch"] = moves[0]["reference_branch"]
    row["plant_tau_v"] = " ".join(f"{v:.4g}" for v in scored["sample"]["plant_tau_v"])
    # Off the move, not off the sample: `--dead-time` overrides what the shift
    # asked for, and this has to be the dead time C3 actually ran.
    row["plant_dead_time_s"] = moves[0]["plant_dead_time_s"]
    return row


def marginal(rows: list[dict], rung: str) -> str:
    """
    Find the smallest delay at which this rung hunts on any seed or sample.

    A cell that *failed* is not a quiet cell. A diverged plant and an infeasible
    sample both come back with no scores at all, and silently dropping them would
    read as "this rung never goes marginal" -- the one column this sweep exists
    to produce, reporting the best case because the worst one crashed.
    """
    mine = [row for row in rows if row["rung"] == rung]
    hunted = [row["delay_s"] for row in mine if row.get("hunting")]
    unknown = sorted({row["delay_s"] for row in mine if "failed" in row})
    caveat = (
        ""
        if not unknown
        else "; no verdict at " + " ".join(f"{delay:.3f}" for delay in unknown)
    )
    return (f"{min(hunted):.3f} s" if hunted else "not on this grid") + caveat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--delay", type=float, nargs="+", default=list(DEFAULT_DELAYS))
    parser.add_argument(
        "--rung", nargs="+", default=["full", "no-integral", "feedforward"]
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2],
        help="two minimum: one run cannot tell a marginal cell from a bad draw",
    )
    parser.add_argument("--moves", type=int, default=1, help="goals per run")
    parser.add_argument("--samples", type=int, default=0, help="random draws per cell")
    parser.add_argument(
        "--k-band",
        type=float,
        default=0.0,
        help="relative half-width on C3's k, with d following it. 0 keeps the "
        "shipped fit; choosing a band is not this issue's",
    )
    parser.add_argument(
        "--psi-band", type=float, default=0.0, help="relative half-width on Psi's gain"
    )
    parser.add_argument(
        "--lag-shift",
        type=float,
        default=0.0,
        help="s of lag/dead-time split to corner against, at constant sum. The "
        "shipped fit refuses every non-zero shift (the arm sits at tau_v = 0 "
        "and the dead time is common), so this needs a refit to mean anything",
    )
    parser.add_argument("--csv", type=Path, default=OUT_ROOT / "sweep.csv")
    arguments, passthrough = parser.parse_known_args()
    if len(arguments.seeds) < 2:
        print("error: --seeds needs at least two", file=sys.stderr)
        return 2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rung in arguments.rung:
        for delay in arguments.delay:
            for seed in arguments.seeds:
                rng = np.random.default_rng(seed)
                for index, sample in enumerate(samples(arguments, rng)):
                    sample["moves"] = arguments.moves
                    print(
                        f"delay={delay:.3f} rung={rung:<12} seed={seed} "
                        f"sample={index}/{sample['kind']} ...",
                        flush=True,
                    )
                    row = run_cell(delay, rung, seed, sample, passthrough)
                    row.update(
                        delay_s=delay,
                        rung=rung,
                        seed=seed,
                        sample=index,
                        kind=sample["kind"],
                        k_scale=cell_field(sample, "k"),
                        psi_gain=cell_field(sample, "psi"),
                        lag_shift_s=sample.get("lag", 0.0),
                    )
                    rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with arguments.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "| delay s | rung | runs | failed | worst goal mm | worst reversals | "
        "worst dq rad/s | hunting | worst e_pos |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for rung in arguments.rung:
        for delay in arguments.delay:
            whole = [
                row for row in rows if row["rung"] == rung and row["delay_s"] == delay
            ]
            cell = [row for row in whole if "failed" not in row]
            # A failed run gets its own column rather than being dropped: a cell
            # scored on its survivors reports the best case of a sample set whose
            # worst case is the reason it has a hole in it.
            broken = len(whole) - len(cell)
            if not cell:
                print(
                    f"| {delay:.3f} | {rung} | {len(whole)} | {broken} | "
                    f"all failed: {whole[0]['failed'] if whole else 'no runs'} |"
                    + " |"
                    * 5
                )
                continue
            axes = [f"e_pos_{name}" for name in AXIS_NAMES]
            print(
                f"| {delay:.3f} | {rung} | {len(whole)} | {broken} | "
                f"{max(row['goal_final'] for row in cell):.1f} | "
                f"{max(row['reversals_worst'] for row in cell):.0%} | "
                f"{max(row['dq_sustained_worst'] for row in cell):.3f} | "
                f"{'yes' if any(row['hunting'] for row in cell) else 'no'} | "
                + " ".join(f"{max(row[axis] for row in cell):.3f}" for axis in axes)
                + " |"
            )
    print()
    for rung in arguments.rung:
        print(f"{rung:<12} marginal at {marginal(rows, rung)}")
    print(f"Wrote: {arguments.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
