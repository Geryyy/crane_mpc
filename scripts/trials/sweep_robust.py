#!/usr/bin/env python3
"""
Sweep MPC levers over moves x plant samples on wire_chain; report the worst cell.

    ./scripts/trials/sweep_robust.py --band k=0.7,1.4 --band payload=0,800 \
        --lever weights.du=0,10,30 --lever pi-scale=1,2 --samples 4 --jobs 3

Plant samples: the nominal plant, every band corner, then `--samples` random
draws (`k`/`psi` per axis, independently). A lever key with a hyphen is a
wire_chain flag (`pi-scale`), else a `--set` on crane_mpc.yaml. Levers vary one
at a time against the shipped settings unless `--cartesian`. Flags not named
here go to every run (e.g. `--latency 0.03`); a band on the same flag wins.
The first run per setting goes alone: a compiled key re-exports, unlocked.
"""

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "build" / "sweep_robust"
AXES = 5
MOVES = ["--goal out", "--random 1 --seed 1", "--random 1 --seed 2"]
BANDS = {  # band -> wire_chain flags it sets
    "k": ["--k-scale", "--d-scale"],  # k and d are one identification
    "psi": ["--psi-gain-positive", "--psi-gain-negative"],
    "payload": ["--plant-payload"],
    "ocp_payload": ["--payload-mass"],
    "latency": ["--latency"],
}
PER_AXIS = ("k", "psi")
#: One run spreads over ~4 cores for a slower wall clock; one core each scales.
ONE_THREAD = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def split(text):
    """Split `a,b` on commas outside brackets, so `[1,2],[3,4]` is two values."""
    return re.split(r",(?![^\[]*\])", text)


def samples(bands, count, seed):
    corners = [dict(zip(bands, c)) for c in itertools.product(*bands.values())]
    rng = np.random.default_rng(seed)
    draws = [
        {
            b: rng.uniform(lo, hi, AXES if b in PER_AXIS else None)
            for b, (lo, hi) in bands.items()
        }
        for _ in range(count)
    ]
    return (
        [("nominal", {})]
        + [("corner", c) for c in corners]
        + [("random", d) for d in draws]
    )


def plant_flags(sample):
    flags = []
    for band, value in sample.items():
        size = AXES if band in PER_AXIS else 1
        text = [f"{v:.6g}" for v in np.broadcast_to(np.asarray(value, float), size)]
        for flag in BANDS[band]:
            flags += [flag, *text]
    return flags


def lever_flags(setting):
    flags = []
    for key, value in setting.items():
        flags += [f"--{key}", value] if "-" in key else ["--set", f"{key}={value}"]
    return flags


def run(job):
    stem, command = job
    out = OUT / stem
    out.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [*command, "--json", str(out / "score.json")],
        capture_output=True,
        text=True,
        env=ONE_THREAD,
    )
    (out / "log.txt").write_text(done.stdout + done.stderr)
    try:
        return json.loads((out / "score.json").read_text())
    except FileNotFoundError:
        tail = (done.stderr or done.stdout).strip().splitlines()
        return {"exit": "crashed: " + (tail[-1] if tail else "no output")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--move", action="append", help=f"default {MOVES}")
    parser.add_argument("--band", action="append", default=[], help="name=lo,hi")
    parser.add_argument("--lever", action="append", default=[], help="key=v1,v2")
    parser.add_argument("--cartesian", action="store_true")
    parser.add_argument("--samples", type=int, default=0, help="random plant draws")
    parser.add_argument("--seed", type=int, default=0, help="the draws' stream")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--csv", type=Path, default=OUT / "sweep.csv")
    args, passthrough = parser.parse_known_args()

    bands = {}
    for text in args.band:
        name, _, values = text.partition("=")
        if name not in BANDS:
            parser.error(f"--band {name}: one of {list(BANDS)}")
        bands[name] = tuple(float(v) for v in values.split(","))
    levers = {k: split(v) for k, _, v in (t.partition("=") for t in args.lever)}
    if args.cartesian:
        settings = [dict(zip(levers, c)) for c in itertools.product(*levers.values())]
    else:
        settings = [{}] + [{k: v} for k, vs in levers.items() for v in vs]
    plants = samples(bands, args.samples, args.seed)

    cells, jobs = [], []
    for s, setting in enumerate(settings):
        for p, (kind, sample) in enumerate(plants):
            for m, move in enumerate(args.move or MOVES):
                cells.append(
                    {
                        "setting": " ".join(f"{k}={v}" for k, v in setting.items())
                        or "shipped",
                        "plant": p,
                        "kind": kind,
                        **{b: np.round(v, 4).tolist() for b, v in sample.items()},
                        "move": move,
                    }
                )
                command = [sys.executable, str(HERE / "wire_chain.py"), *passthrough]
                command += shlex.split(move) + plant_flags(sample)
                jobs.append((f"s{s}_p{p}_m{m}", command + lever_flags(setting)))

    first = {}
    for index, cell in enumerate(cells):
        first.setdefault(cell["setting"], index)
    results = {i: run(jobs[i]) for i in first.values()}
    rest = [i for i in range(len(jobs)) if i not in results]
    print(
        f"{len(settings)} settings x {len(plants)} plants x "
        f"{len(args.move or MOVES)} moves",
        flush=True,
    )
    with ThreadPoolExecutor(args.jobs) as pool:
        results.update(zip(rest, pool.map(run, [jobs[i] for i in rest])))
    rows = [{**cells[i], **results[i]} for i in range(len(cells))]

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def summary(group):
        scored = [r for r in group if "final_error" in r]
        failed = [r["exit"] for r in group if r["exit"] != "ok"]
        if not scored:
            return f"{'-':>8} {'-':>7} {'-':>7} {'-':>6} {'-':>6} {'-':>6}  {failed}"
        worst = lambda key: max(np.max(r[key]) for r in scored)  # noqa: E731
        # growth.py's slope; nan where the sway never rose through its band
        growth = max((r["growth_per_s"] for r in scored), key=np.nan_to_num)
        return (
            f"{worst('final_error'):8.4f} {worst('sway_peak'):7.3f} "
            f"{worst('sway_late'):7.3f} {worst('reversals'):6.2f} "
            f"{worst('dq_clamp_ratio'):6.2f} {growth:6.2f}  {len(failed)}/{len(group)} failed"
            + (f" {sorted(set(failed))}" if failed else "")
        )

    print(
        f"\n{'setting':<28} {'cell':<8} {'err rad':>8} {'sway pk':>7} "
        f"{'late':>7} {'rev':>6} {'dq/clp':>6} {'grow/s':>6}"
    )
    for name in dict.fromkeys(r["setting"] for r in rows):
        mine = [r for r in rows if r["setting"] == name]
        print(
            f"{name:<28} {'nominal':<8} "
            + summary([r for r in mine if r["kind"] == "nominal"])
        )
        print(f"{'':<28} {'worst':<8} " + summary(mine))
    print(f"Wrote: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
