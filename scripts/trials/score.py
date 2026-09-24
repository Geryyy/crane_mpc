#!/usr/bin/env python3
"""Score one trial: solve cadence, command chatter, measured chatter, arrival."""

import json
import sys
from collections import Counter

import numpy as np

ACT = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
]
SHORT = dict(zip(ACT, ["sw", "ha", "ka", "sa", "ro", "gr"]))


def by_name(rows, field, names_key="names"):
    """Columns keyed by joint name. Index mapping is wrong across these stacks."""
    out = {j: [] for j in ACT}
    for r in rows:
        idx = {n: i for i, n in enumerate(r[names_key])}
        for j in ACT:
            v = r[field]
            out[j].append(v[idx[j]] if j in idx and idx[j] < len(v) else np.nan)
    return {j: np.asarray(v, float) for j, v in out.items()}


def reversals(x, dead):
    s = np.sign(x[np.abs(x) > dead])
    return 0.0 if len(s) < 2 else float(np.mean(s[1:] != s[:-1]))


def main(path, label):
    d = json.load(open(path))
    h, hz, js = d["health"], d["hz"], d["js"]
    s = np.array([x["solve"] for x in h])
    bud = h[0]["budget"]
    print(f"\n=== {label} ===")
    print(
        f"solves {len(h)}  median {np.median(s) * 1e3:5.1f} ms  p90 {np.percentile(s, 90):.4f}"
        f" -> {np.percentile(s, 90) * 1e3:5.1f} ms  max {s.max() * 1e3:5.1f} ms"
        f"  budget {bud * 1e3:.0f} ms  over {int((s > bud).sum())}"
    )
    print(
        f"outcome {dict(Counter(x['outcome'] for x in h))}  "
        f"applied_previous {sum(x['prev'] for x in h)}  horizons {len(hz)}"
    )
    print(
        f"stamps ref={d['ref']} path={d['path']} "
        f"paired={'YES' if d['ref'] and d['ref'] == d['path'] else 'NO'}"
    )

    cmd = by_name(hz, "vel")
    mea = by_name(js, "vel")
    pos = by_name(js, "pos")
    print(
        f"{'axis':>4} {'cmd rev%':>9} {'meas rev%':>10} {'|dq|max':>9} {'travel rad':>11}"
    )
    for j in ACT:
        c, m, p = cmd[j], mea[j], pos[j]
        c, m, p = c[~np.isnan(c)], m[~np.isnan(m)], p[~np.isnan(p)]
        travel = (p.max() - p.min()) if p.size else 0.0
        print(
            f"{SHORT[j]:>4} {reversals(c, 1e-3) * 100:8.1f}% {reversals(m, 5e-3) * 100:9.1f}%"
            f" {np.abs(m).max() if m.size else 0:9.4f} {travel:11.4f}"
        )
    # settling: RMS joint speed over the last 3 s of capture
    n = min(300, len(js))
    tail = np.vstack([mea[j][-n:] for j in ACT])
    tail = np.nan_to_num(tail)
    print(f"tail RMS |dq| over last {n} samples: {np.sqrt((tail**2).mean()):.4f} rad/s")


for i in range(1, len(sys.argv), 2):
    main(sys.argv[i], sys.argv[i + 1])
