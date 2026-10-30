#!/usr/bin/env python3
"""
Optuna search over the MPC weights, every trial scored on `wire_chain` runs.

    ./scripts/wire_chain/tune_weights.py --trials 100 --storage sqlite:///tune.db
    ./scripts/wire_chain/tune_weights.py --trials 100 --storage sqlite:///tune.db  # 2nd worker
    optuna-dashboard sqlite:///tune.db

A trial runs, in parallel (--jobs): random moves (empty and loaded) and kick tests (tip
and tilt swung by --kick rad at poses near and far from the slew column). Objectives,
all minimised: kick decay to 10 % (s, worst case), sway rate left after the moves
(rad/s), TCP distance from the path (m, p95, sway included). A run that diverges,
hunts, leaves u_clamp or fails a solve makes the trial infeasible.

The first trial is the shipped crane_mpc.yaml. The Pareto front is printed with the
--set lines that reproduce each point; check a pick in Gazebo before it ships.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from crane_model.conventions import Tool  # noqa: E402
from crane_model.model import CraneModel  # noqa: E402

#: slew, boom, arm, telescope, rotator: near the column and at long reach
KICK_POSES = {
    "near": (0.03, 0.91, -0.08, 0.69, 6.25),
    "far": (0.03, 0.55, 0.60, 1.90, 6.25),
}
MOVE_SEEDS = (1, 2)
PAYLOADS = (0.0, 880.0)
MOVES_PER_RUN = 4
KICK_SECONDS = 30.0
SWAY_PERIOD_S = 2.34  # s, envelope window for the kick decay
SPACE = {  # log-uniform ranges
    "dq_u": (16.0, 4096.0),
    "q_u": (0.5, 400.0),
    "du": (1.0, 256.0),
    "dq_a_luff": (10.0, 3000.0),  # boom and arm
    "dq_a_other": (1.0, 3000.0),  # slew and telescope
    "tcp": (100.0, 10000.0),
    "terminal_scale": (1.0, 100.0),
}


def sets(params: dict, shipped: dict) -> list[str]:
    """Turn a parameter point into the `--set` arguments it stands for."""
    rotator, tool = shipped["dq_a"][4:6]
    luff, other = params["dq_a_luff"], params["dq_a_other"]
    out = [f"weights.dq_a=[{other},{luff},{luff},{other},{rotator},{tool}]"]
    out += [f"weights.{key}={params[key]}" for key in ("dq_u", "q_u", "du", "tcp")]
    out.append(f"weights.terminal_scale={params['terminal_scale']}")
    return out


def shipped_point(shipped: dict) -> dict:
    dq_a = shipped["dq_a"]
    return {
        "dq_u": shipped["dq_u"][0],
        "q_u": shipped["q_u"][0],
        "du": shipped["du"][0],
        "dq_a_luff": dq_a[1],
        "dq_a_other": dq_a[0],
        "tcp": shipped["tcp"],
        "terminal_scale": shipped["terminal_scale"],
    }


def kick_starts() -> dict:
    """--start vectors: the poses with the load hanging still."""
    model = CraneModel(harness.description_xml(), Tool.PZS100)
    starts = {}
    for name, (slew, boom, arm, tele, rot) in KICK_POSES.items():
        q_u = model.passive_equilibrium([slew, boom, arm, tele, rot, 0.0])
        starts[name] = [slew, boom, arm, tele, *q_u, rot, 0.0, 0.0, 0.0]
    return starts


def cases(kick: float, starts: dict) -> list[tuple[str, list[str]]]:
    out = []
    for seed in MOVE_SEEDS:
        for mass in PAYLOADS:
            out.append(
                (
                    f"move_s{seed}_{mass:.0f}kg",
                    ["--random", str(MOVES_PER_RUN), "--seed", str(seed)]
                    + ["--payload-mass", str(mass)],
                )
            )
    for name, start in starts.items():
        for axis, kicked in (("tip", (kick, 0.0)), ("tilt", (0.0, kick))):
            out.append(
                (
                    f"kick_{name}_{axis}",
                    ["--goal", "here", "--start", *map(str, start)]
                    + ["--kick", *map(str, kicked), "--seconds", str(KICK_SECONDS)],
                )
            )
    return out


def run(name, arguments, assignments, folder):
    """One wire_chain run; `(scores, csv rows)` or None when it wrote no scores."""
    stem = Path(folder) / name
    command = [sys.executable, str(HERE / "wire_chain.py"), *arguments]
    command += ["--json", f"{stem}.json", "--csv", f"{stem}.csv"]
    for assignment in assignments:
        command += ["--set", assignment]
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(
        command, capture_output=True, env=environment, timeout=900, check=False
    )
    if not Path(f"{stem}.json").exists():
        return None
    scores = json.loads(Path(f"{stem}.json").read_text())
    rows = list(csv.reader(open(f"{stem}.csv")))
    return (scores if isinstance(scores, list) else [scores]), rows


def decay_to_10(rows, axis: int, kick: float) -> float:
    """Seconds until the kicked sway envelope stays under 10 % of the kick."""
    header, data = rows[0], np.array(rows[1:], float)
    t, sway = data[:, header.index("t")], np.abs(data[:, header.index(f"sway{axis}")])
    width = max(1, int(round(SWAY_PERIOD_S / (t[1] - t[0]))))
    envelope = np.array(
        [sway[i : i + width].max() for i in range(len(sway) - width + 1)]
    )
    above = np.flatnonzero(envelope > 0.1 * kick)
    if len(above) == 0:
        return float(t[0])
    last = above[-1] + 1
    return float(t[last]) if last < len(envelope) else KICK_SECONDS * 2  # never settled


def objective_factory(options, shipped, starts):
    plan = cases(options.kick, starts)

    def objective(trial: optuna.Trial):
        params = {
            key: trial.suggest_float(key, low, high, log=True)
            for key, (low, high) in SPACE.items()
        }
        assignments = sets(params, shipped)
        with tempfile.TemporaryDirectory() as folder:
            with ThreadPoolExecutor(options.jobs) as pool:
                results = dict(
                    zip(
                        [name for name, _ in plan],
                        pool.map(lambda c: run(c[0], c[1], assignments, folder), plan),
                    )
                )
        failures = 0
        decays, late_rates, path_errors = [], [], []
        for name, result in results.items():
            if result is None:
                failures += 1
                continue
            scores, rows = result
            failures += sum(
                s["exit"] != "ok" or s["nonzero_status"] > 0 for s in scores
            )
            if name.startswith("kick"):
                decays.append(
                    decay_to_10(rows, 0 if name.endswith("tip") else 1, options.kick)
                )
            else:
                late_rates += [max(s["sway_rate_late"]) for s in scores]
                path_errors += [s["tcp_path_error_p95"] for s in scores]
        values = (
            max(decays) if decays else 1e3,
            float(np.mean(late_rates)) if late_rates else 1e3,
            float(np.mean(path_errors)) if path_errors else 1e3,
        )
        trial.set_user_attr("failures", failures)
        trial.set_user_attr("decays", decays)
        trial.set_user_attr("set", assignments)
        return values

    return objective


def constraints(trial) -> list[float]:
    """> 0 is infeasible: any failed run."""
    return [float(trial.user_attrs.get("failures", 1))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--trials", type=int, default=50, help="this worker's trials")
    parser.add_argument("--storage", default="sqlite:///tune_weights.db")
    parser.add_argument("--study", default="mpc_weights")
    parser.add_argument(
        "--jobs", type=int, default=4, help="wire_chain runs in parallel"
    )
    parser.add_argument("--kick", type=float, default=0.3, help="rad, kick test swing")
    parser.add_argument("--seed", type=int, default=0, help="sampler seed")
    options = parser.parse_args(argv)

    parameters, _ = harness.settings()
    shipped = parameters["weights"]
    sampler = optuna.samplers.TPESampler(
        seed=options.seed, constraints_func=constraints
    )
    study = optuna.create_study(
        study_name=options.study,
        storage=options.storage,
        load_if_exists=True,
        directions=["minimize"] * 3,
        sampler=sampler,
    )
    study.set_metric_names(["kick_decay_s", "late_sway_rate", "tcp_path_p95_m"])
    if len(study.trials) == 0:
        study.enqueue_trial(shipped_point(shipped))
    study.optimize(objective_factory(options, shipped, kick_starts()), options.trials)

    print("\nPareto front (kick decay s, late sway rate rad/s, TCP path p95 m):")
    for trial in sorted(study.best_trials, key=lambda t: t.values[0]):
        if trial.user_attrs.get("failures", 1):
            continue
        values = ", ".join(f"{v:.4g}" for v in trial.values)
        print(f"  #{trial.number}: {values}")
        print("    " + " ".join(f"--set {a}" for a in trial.user_attrs["set"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
