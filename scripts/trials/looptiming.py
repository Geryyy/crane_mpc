#!/usr/bin/env python3
"""
What a `capture.py` recording says about the live loop timing, and what it cannot say.

    ./scripts/trials/looptiming.py cap.json

Issue 189 wants four numbers, to set `wire_chain` to the live Gazebo loop. Only
two of the four are in a capture:

- **delivery latency** -- measured here, `--latency`.
- **controller-state rate** -- measured here, against `Timing.jtc_output`'s quorum.
- **cadence phase** -- *not observable*. See `CADENCE_PHASE_IS_NOT_OBSERVABLE`.
- **dead time** -- not measurable from this recording, and `sensor_to_valve_delay`
  is not a free lever: `config.check_settings` refuses any value but `Ts`. The
  identified quantity is the *sum* of dead time and `tau_v`, so the harness lever
  is `--lag-shift` (constant sum) and the instrument is `steptest.py`, which steps
  one axis open loop with the MPC out of the loop. A command-to-response
  correlation on this capture would measure the closed loop's dominant phase --
  `out` contains the PI term, so it contains the measurement -- plus `tau_v` plus
  the body's own lag, and report it as a transport delay.

Times come from the header stamp `sim` and the receipt `rx`, both on the stamps'
own clock. A capture without them cannot answer this issue: the wall clock `t`
is a different clock, and at an RTF near 0.5 reading one for the other is out by
a factor of two (`modelfit.py` computes an RTF for exactly this reason).
"""

import argparse
import json
import sys

import numpy as np

#: `velocity_loop.yaml` rate_hz. The quorum has to be the *nominal* tick count:
#: normalised against the measured rate instead, a uniformly decimated stream
#: reads as a healthy slow one.
NOMINAL_JTC_HZ = 100.0

CADENCE_PHASE_IS_NOT_OBSERVABLE = """\
cadence phase is not in a capture. `node.py` stamps the horizon with
`Timing.stamp_ns()`, which is `next_first_knot_ns - command_delay_ns`, which
`Timing.anchor` makes identically the measurement stamp -- so the horizon stamp
*is* a `/joint_states` stamp and any offset measured against that set is 0 by
construction, on a healthy loop and a once-anchored one alike. The live quantity
is the age of the state at the node's timer tick, and the capture records no
tick instant. Recording one (the horizon carrying both its tick instant and the
js stamp it solved from) is what would make this measurable."""


def _times(rows, key):
    return np.asarray([r[key] for r in rows if r.get(key) is not None], float)


def horizon_rows(cap):
    """
    Pick the horizon stream: `/crane/mpc/horizon`, or in shadow `~/shadow_horizon`.

    In shadow the contract topic is silent by design, so a capture of a shadow
    run carries only `shz`. Same cadence and same `Timing.stamp_ns()`, so every
    number below reads the same off either.
    """
    return (cap.get("hz") or []) or (cap.get("shz") or [])


def stamped(cap):
    """Whether every stream carries usable `sim`/`rx`. Nothing here works without it."""
    for rows in (horizon_rows(cap), cap.get("ctrl") or [], cap.get("js") or []):
        if not rows:
            continue
        sim, rx = _times(rows, "sim"), _times(rows, "rx")
        if len(sim) < len(rows) or len(rx) < len(rows):
            return False
        if not np.any(sim > 0) or not np.any(rx > 0):
            return False
    return True


def period_of(stamps):
    """Median spacing of a stamp set, or None when there is no grid to read."""
    sorted_stamps = np.sort(np.asarray(stamps, float))
    return float(np.median(np.diff(sorted_stamps))) if len(sorted_stamps) >= 3 else None


def cycle_period(cap):
    """`Ts` from the horizon cadence, or None."""
    return period_of(_times(horizon_rows(cap), "sim"))


def latency_of(rows):
    """
    Stamp to arrival over one stream: `{p50, p90, max, negative, n}`.

    Sets `--latency` off the horizon; `labday_report.py` runs it per recorded
    stream with `rx` the recorder's receipt instead. Two caveats the number
    cannot carry: arrival is the *subscriber's*, so the JTC's own callback
    scheduling is not in it; and `rx` under `use_sim_time` is a held value
    between `/clock` messages, so the resolution is one `/clock` period and a
    subscriber whose clock lags the publisher's can read negative.
    """
    rows = [r for r in rows or [] if r.get("sim") and r.get("rx")]
    if not rows:
        return None
    lag = np.array([r["rx"] - r["sim"] for r in rows])
    p50, p90, worst = (float(x) for x in np.percentile(lag, [50, 90, 100]))
    return dict(p50=p50, p90=p90, max=worst, negative=int(np.sum(lag < 0)), n=len(lag))


def latency(cap):
    """`latency_of` on the capture's horizon stream."""
    return latency_of(horizon_rows(cap))


def rate_of(ctrl, cycle_stamps=None, nominal_hz=NOMINAL_JTC_HZ):
    """
    `/controller_state` publish rate and the share of cycles under quorum.

    `/trajectory_controllers/controller_state` is shared with the grasping
    controller, so the busiest `src` is taken alone -- merged, two publishers
    read as one at double the rate.

    This is the *publish* rate, a lower bound on the tick rate: the JTC publishes
    on a non-blocking `RealtimePublisher`, which drops under exactly the load
    that starves the loop. `starved` mirrors `Timing.jtc_output`'s test against
    the nominal tick count, so it answers "would the dead time have replayed the
    command that was sent". Without `cycle_stamps` there is no cycle grid to
    count against and only the rate is reported.
    """
    if not ctrl:
        return None
    by_source: dict = {}
    for r in ctrl:
        by_source.setdefault(r.get("src", ""), []).append(r)
    src, rows = max(by_source.items(), key=lambda kv: len(kv[1]))
    ct = np.sort(_times(rows, "sim"))
    step = period_of(ct)
    if step is None:
        return None
    out = dict(
        src=src, hz=1.0 / step if step > 0 else float("nan"), sources=len(by_source)
    )
    Ts = period_of(cycle_stamps) if cycle_stamps is not None else None
    if Ts is None:
        return out
    expected = Ts * nominal_hz
    counts = np.diff(np.searchsorted(ct, np.sort(np.asarray(cycle_stamps, float))))
    out.update(
        per_cycle=float(np.median(counts)) if len(counts) else None,
        expected=expected,
        starved=float(np.mean(counts < 0.5 * expected)) if len(counts) else None,
    )
    return out


def controller_rate(cap, nominal_hz=NOMINAL_JTC_HZ):
    """`rate_of` on the capture's controller stream, against its horizon grid."""
    return rate_of(cap.get("ctrl") or [], _times(horizon_rows(cap), "sim"), nominal_hz)


def report(cap, nominal_hz=NOMINAL_JTC_HZ):
    """Lay out what the capture measures, and refuse what it does not."""
    counts = (
        f"horizons={len(cap.get('hz') or [])} ctrl={len(cap.get('ctrl') or [])} "
        f"js={len(cap.get('js') or [])}"
    )
    if not stamped(cap):
        return (
            f"{counts}\nno usable sim/rx stamps -- re-record with a capture.py that "
            "writes them (issue 189). Wall `t` is a different clock from the header "
            "stamps and at RTF != 1 the two differ by more than the quantities here."
        )
    Ts = cycle_period(cap)
    lines = [counts + (f" Ts={1e3 * Ts:.1f} ms" if Ts else " (no cycle grid)"), ""]

    lag = latency(cap)
    if lag is None:
        lines.append("latency        -- no horizons; nothing arrived to time")
    else:
        note = (
            f"  [{lag['negative']}/{lag['n']} negative: /clock lag]"
            if lag["negative"]
            else ""
        )
        lines.append(
            f"latency        p50/p90/max {1e3 * lag['p50']:.1f}/{1e3 * lag['p90']:.1f}/"
            f"{1e3 * lag['max']:.1f} ms   -> --latency {max(lag['p50'], 0.0):.3f}{note}"
        )

    rate = controller_rate(cap, nominal_hz)
    if rate is None:
        lines.append("controller_state  -- not recorded")
    else:
        shown = (
            f"controller_state  {rate['hz']:.1f} Hz publish ({rate['src'] or 'unnamed'}"
        )
        shown += f", {rate['sources']} sources)" if rate["sources"] > 1 else ")"
        lines.append(shown)
        if rate.get("expected") is not None:
            lines.append(
                f"                  {rate['per_cycle']:.1f} ticks/cycle vs "
                f"{rate['expected']:.1f} nominal, "
                f"{1e2 * rate['starved']:.0f}% of cycles under jtc_output quorum"
            )
    lines += ["", "cadence phase  NOT MEASURABLE", CADENCE_PHASE_IS_NOT_OBSERVABLE]
    lines += [
        "",
        "dead time      NOT MEASURABLE from this capture. `steptest.py` (open loop, "
        "MPC out)\n               is the instrument; `--lag-shift` is the lever. "
        "`sensor_to_valve_delay`\n               is pinned to `Ts` by "
        "`config.check_settings` and is not free.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("capture", help="capture.py json")
    parser.add_argument(
        "--jtc-rate",
        type=float,
        default=NOMINAL_JTC_HZ,
        help="nominal velocity_loop.yaml rate_hz, for the quorum",
    )
    options = parser.parse_args(argv)
    with open(options.capture) as handle:
        cap = json.load(handle)
    print(report(cap, options.jtc_rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
