#!/usr/bin/env python3
"""
What a lab-day mcap says about the MPC running in shadow. Offline, after the run.

    ./scripts/trials/labday_report.py labday_0.mcap [--jtc-rate 100] [--window 1.0]

Hardware time is the scarce thing, so the lab day records and this reads the
recording back (`scripts/trials/labday_topics.txt` is the profile). Nothing
here drives anything: it opens a file.

The bag is read with `mcap_ros2`, which decodes off the schemas the recorder
embedded -- no sourced workspace and no built `crane_msgs`. The model and step
sections are the exception: they evaluate the OCP's own integrator, so they
import `crane_model`/`crane_mpc` lazily and say so when the import fails.
Joints are mapped by `name[]`, never by index: the two stacks publish different
`/joint_states` name sets in different orders.

Six sections. A section whose topics are not in the bag says "not recorded" and
does not fail -- that is the normal state of a bag from the wrong layer.

**Latency here is a difference of two clocks**: the recorder's receipt
(`log_time`) against the publishing host's `header.stamp`. Unsynced hosts make
it the offset between them. A negative or implausible value is reported as a
clock problem and never as a result.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages

sys.path.insert(0, str(Path(__file__).resolve().parent))

import looptiming  # noqa: E402

#: A stamp-to-receipt difference past this is a clock offset, not a delivery
#: delay: the whole chain budgets 60 ms and the loop runs at 100 Hz.
CLOCK_IMPLAUSIBLE = 1.0
#: `commissioning.md` 6: `SOLVE_CONVERGED` on the large majority of cycles.
CONVERGED_SHARE = 0.9
#: A gap this many nominal periods wide is a silence episode, not jitter.
SILENCE_FACTOR = 5.0
#: Just under `steptest.py`'s smallest amplitude (boom, 0.10 rad/s). An ordinary
#: single-axis move on the machine commands ~0.05, and at that size the section
#: reports a dozen candidates that are not step tests at all.
STEP_MIN = 0.09
#: A plateau shorter than this cannot carry a dead time plus a lag.
STEP_HOLD = 0.3
#: Past this the follower's command has collapsed under the position error, so
#: the response is the PI loop's and no dead time or lag can be read off it.
MAX_DROOP = 0.5
#: An episode ends when the command falls below this share of the step it began
#: with. Not `STEP_MIN`: a drooping command crosses that while still commanding
#: (telescope, `p = 5.14`, falls from 0.16 to 0.022 rad/s over a 0.6 s hold).
STEP_FLOOR = 0.1

SCHEMAS = {
    "control_msgs/msg/JointTrajectoryControllerState": "ctrl",
    "crane_msgs/msg/SolverHealth": "health",
    "epsilon_crane_msgs/msg/CraneData": "cranedata",
}
OUTCOME = {0: "unknown", 1: "converged", 2: "budget_exceeded", 3: "failed"}
#: `SupervisorStatus.FAULT_NONE`; anything else on this stream is FAULT_SOLVER.
FAULT_NONE = 0
#: Recorded, and the layer's abort condition holds -- not the same as absent.
NO_FOLLOWER = "no follower velocity on any axis"
#: A `/cranedata` sample further than this from a state stamp is not that state's.
PRESSURE_STALE = 0.06
AXES = ("sw", "ha", "ka", "sa", "ro")


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def role_of(topic, schema):
    """
    Which stream a channel is, by schema and -- where a schema is reused -- by name.

    `/joint_states` is matched by name on purpose: a bringup publishes several
    `JointState` topics (`*/js_desired` is the one that bites) and taking a
    desired state for the measured one scores the model against itself.
    """
    if topic.endswith("/shadow_comparison"):
        return "shadow_cmp"
    if topic.endswith("/shadow_horizon"):
        return "shadow_hz"
    if topic == "/crane/mpc/horizon":
        return "hz"
    if topic == "/joint_states" and schema == "sensor_msgs/msg/JointState":
        return "js"
    return SCHEMAS.get(schema)


def load(path):
    """
    One pass over the channels the report reads. `{role: rows}` plus `{topic: rows}`.

    Selected off the summary rather than decoding everything: a lab-day bag
    carries point clouds, and decoding those to time them costs minutes.
    """
    with open(path, "rb") as handle:
        summary = make_reader(handle).get_summary()
    wanted, roles = {}, {}
    for channel in summary.channels.values():
        schema = summary.schemas[channel.schema_id].name
        role = role_of(channel.topic, schema)
        if role is not None:
            wanted[channel.topic] = schema
            roles[channel.topic] = role
    bag = defaultdict(list)
    streams = defaultdict(list)
    for message in read_ros2_messages(path, topics=list(wanted)):
        topic, ros = message.channel.topic, message.ros_msg
        row = dict(rx=message.log_time_ns * 1e-9, sim=stamp(ros.header), topic=topic)
        streams[topic].append(row)
        bag[roles[topic]].append(dict(row, msg=ros, src=topic))
    return bag, streams


def by_name(message, field, joints):
    """Order one `field.velocities` row by `joints`; NaN where a joint is absent."""
    names = list(message.joint_names)
    values = getattr(message, field).velocities
    out = np.full(len(joints), np.nan)
    for axis, joint in enumerate(joints):
        if joint in names:
            row = names.index(joint)
            if row < len(values):
                out[axis] = values[row]
    return out


def busiest(rows):
    """Take the busiest `src` alone -- `/trajectory_controllers/...` is shared."""
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["src"]].append(row)
    return max(by_source.values(), key=len) if by_source else []


def percentiles(values):
    return (float(x) for x in np.percentile(np.asarray(values, float), [50, 90, 100]))


def missing(section, topics):
    return [f"{section}", f"  not recorded -- the bag carries none of {topics}"]


# --------------------------------------------------------------------------- 1


def latency(bag, streams, jtc_rate):
    """Stamp to receipt per stream, then the controller-state rate and quorum."""
    lines = ["1 LATENCY  (recorder receipt - header stamp, so two clocks)", ""]
    suspect = []
    for topic in sorted(streams):
        lag = looptiming.latency_of(streams[topic])
        if lag is None:
            lines.append(f"  {topic:<52} no stamps")
            continue
        if lag["negative"] or lag["p50"] > CLOCK_IMPLAUSIBLE:
            suspect.append(topic)
        lines.append(
            f"  {topic:<52} p50/p90/max "
            f"{1e3 * lag['p50']:8.1f}/{1e3 * lag['p90']:8.1f}/"
            f"{1e3 * lag['max']:8.1f} ms  n={lag['n']}"
        )
    if suspect:
        lines += [
            "",
            "  CLOCK PROBLEM, not a result: a negative or implausible stamp-to-receipt",
            "  difference means the publishing host and the recorder are not synced.",
            "  chrony the MPC and the ros2_control hosts and record again; until then",
            f"  read no latency off {', '.join(suspect)}.",
        ]

    grid = [r["sim"] for r in bag.get("shadow_hz") or bag.get("hz") or []]
    rate = looptiming.rate_of(bag.get("ctrl") or [], grid, jtc_rate)
    lines.append("")
    if rate is None:
        lines.append("  controller_state  not recorded")
    else:
        shown = f"  controller_state  {rate['hz']:.1f} Hz publish ({rate['src']}"
        lines.append(
            shown + (f", {rate['sources']} sources)" if rate["sources"] > 1 else ")")
        )
        if rate.get("expected") is not None:
            lines.append(
                f"                    {rate['per_cycle']:.1f} ticks/cycle vs "
                f"{rate['expected']:.1f} nominal, "
                f"{1e2 * rate['starved']:.0f}% of cycles under jtc_output quorum"
            )
        else:
            lines.append(
                "                    no horizon grid in the bag, so no quorum share"
            )
    return lines, rate, suspect


# --------------------------------------------------------------------------- 2


def solve_timing(bag):
    """
    Judge the budget by the node's own verdict, and `solve_time` only as context.

    `commissioning.md` 6 is explicit that the budget criterion is the
    `SOLVE_BUDGET_EXCEEDED` and `applied_previous_solution` shares and **not**
    `solve_time`: that field carries acados' `time_tot` alone while lateness is
    judged end to end from the `/joint_states` stamp, so a node overrunning
    every cycle still reports a p90 near 10 ms inside a 60 ms budget. The
    percentiles are printed because they say where the solve itself sits; the
    verdict is taken off `outcome`.
    """
    rows = bag.get("health") or []
    if not rows:
        return missing("2 SOLVE TIMING", "/crane/mpc/solver_health"), None
    solve = np.array([r["msg"].solve_time for r in rows], float)
    budget = np.array([r["msg"].solve_budget for r in rows], float)
    outcome = np.array([int(r["msg"].outcome) for r in rows])
    fallback = np.array([bool(r["msg"].applied_previous_solution) for r in rows])
    fault = np.array([int(r["msg"].fault) for r in rows])
    slack = np.array([bool(r["msg"].used_slack) for r in rows])
    p50, p90, worst = percentiles(solve)
    stats = dict(
        n=len(rows),
        p50=p50,
        p90=p90,
        max=worst,
        budget=float(np.median(budget)),
        exceeded=float(np.mean(outcome == 2)),
        converged=float(np.mean(outcome == 1)),
        fallback=float(np.mean(fallback)),
        slack=float(np.mean(slack)),
        fault=int(np.sum(fault != FAULT_NONE)),
    )
    lines = [
        "2 SOLVE TIMING",
        "",
        f"  SOLVE_BUDGET_EXCEEDED {1e2 * stats['exceeded']:.1f}%   "
        f"fallback (applied_previous_solution) {1e2 * stats['fallback']:.1f}%"
        "   <- the budget criterion",
        f"  converged {1e2 * stats['converged']:.1f}%   "
        f"used_slack {1e2 * stats['slack']:.1f}%",
        f"  acados solve_time p50/p90/max "
        f"{1e3 * p50:.1f}/{1e3 * p90:.1f}/{1e3 * worst:.1f} ms vs solve_budget "
        f"{1e3 * stats['budget']:.1f} ms -- context only, not the lateness test:",
        "    it is `time_tot`, while the budget runs end to end from the "
        "/joint_states stamp",
        "  outcome  "
        + "  ".join(
            f"{OUTCOME.get(code, code)} {int(np.sum(outcome == code))}"
            for code in sorted(set(outcome.tolist()))
        ),
        f"  FAULT_SOLVER on {stats['fault']} of {len(rows)} reports",
        "  every share above is of the *published* reports, which "
        "`reports.health_is_due` thins",
        "  by `solver_health_decimation` and then exempts a changed verdict from "
        "-- so a rare",
        "  fault is over-represented and the denominator is not the cycle count.",
    ]

    sims = np.sort(np.array([r["sim"] for r in rows], float))
    period = looptiming.period_of(sims)
    if period is None:
        lines.append("  too few reports for a cadence, so no silence episodes")
        return lines, stats
    gaps = np.diff(sims)
    silent = gaps > SILENCE_FACTOR * period
    stats.update(silences=int(np.sum(silent)), worst_gap=float(gaps.max()))
    lines.append(
        f"  {1.0 / period:.1f} Hz reported; {stats['silences']} silence episodes "
        f"over {SILENCE_FACTOR:.0f} periods, worst gap {1e3 * gaps.max():.0f} ms"
    )
    # The budget is end to end from the /joint_states stamp, so a publisher
    # that merely stopped reporting and one that stopped solving look the same
    # here. `solver_health_decimation` thins the stream before it is recorded.
    lines.append(
        "  (decimated by solver_health_decimation; a gap is not a missed solve)"
    )
    return lines, stats


# --------------------------------------------------------------------------- 3


def model_skill(bag, ocp=None):
    """
    `modelfit.py`'s one-cycle skill, against the command the machine received.

    `ocp` is the plant to predict with; `None` builds the shipped OCP. The test
    passes a one-line plant with a known error instead, because a skill number
    off the real OCP is a measurement and not an assertion.
    """
    head = "3 MODEL SKILL  (one cycle, against controller_state.output)"
    js, ctrl = bag.get("js") or [], busiest(bag.get("ctrl") or [])
    if not js or not ctrl:
        return missing(head, "/joint_states + a controller_state"), None
    try:
        import modelfit
    except Exception as error:  # acados, crane_model, a built crane_mpc
        return [head, f"  not available offline: {str(error)[:90]}"], None

    # `np.diff`, `np.interp` and the `ct <= when` lookup below all assume a
    # rising stamp; the reader yields log-time order, which is not the same thing.
    js, ctrl = sorted(js, key=lambda r: r["sim"]), sorted(ctrl, key=lambda r: r["sim"])
    canon = modelfit.CANON
    wanted = [canon[i] for i in modelfit.PLANNED_ROWS + modelfit.PASSIVE_ROWS + [7]]
    planned = [canon[i] for i in modelfit.PLANNED_ROWS]
    present = set(js[0]["msg"].name)
    if not set(wanted) <= present:
        return [
            head,
            f"  /joint_states is missing {sorted(set(wanted) - present)}",
        ], None

    t = np.array([r["sim"] for r in js])

    def pick(field):
        out = []
        for row in js:
            index = {n: i for i, n in enumerate(row["msg"].name)}
            values = getattr(row["msg"], field)
            out.append([values[index[n]] if n in index else np.nan for n in wanted])
        return np.asarray(out, float)

    ct = np.array([r["sim"] for r in ctrl])
    cu = np.array([by_name(r["msg"], "output", planned) for r in ctrl])

    def command(k, when):
        due = np.nonzero(ct <= when)[0]
        return cu[due[-1]] if len(due) else None

    if ocp is None:
        from crane_mpc import problem
        from crane_mpc.solver import Ocp

        parameters = modelfit.shipped()
        ocp = Ocp(
            problem.default_description().read_text(),
            parameters,
            parameters["hydraulics"],
        )
    q, dq = pick("position"), pick("velocity")
    # Both the bag's stamps and `Ts` are on the publishing host's clock, so no
    # RTF rescaling: that correction belongs to a `capture.py` wall-stamped run.
    arms = [("force row carried", None)]
    pressure = chamber_pressures(bag, t)
    measured = (
        modelfit.measured_force(q, pressure) if isinstance(pressure, dict) else None
    )
    if measured is not None:
        arms.append(("force row from measured pressure", measured))
    lines, scored = [head], {}
    for label, force in arms:
        # `force(k)` reads the boom four-bar and raises on a configuration it
        # cannot close, so one bad measured sample would otherwise take the
        # whole report down -- after the hardware time is spent.
        try:
            model, actual = modelfit.score(ocp, t, q, dq, command, force=force)
        except Exception as error:
            lines.append(f"  {label}: refused -- {str(error)[:80]}")
            continue
        if not len(model):
            lines.append(f"  {label}: no cycle integrated")
            continue
        lines.append(modelfit.report(f"  {label}", model, actual))
        scored[label] = modelfit.skill(model, actual)
    if pressure is None:
        lines.append("  /cranedata not recorded, so no pressure-measured force row")
    elif isinstance(pressure, int):
        lines.append(
            f"  /cranedata has no sample within {1e3 * PRESSURE_STALE:.0f} ms of "
            f"{pressure} state stamps,"
        )
        lines.append(
            "  so the pressure arm is refused rather than fed a held or "
            "backwards-held reading"
        )
    elif measured is None:
        lines.append("  /cranedata carries no finite chamber pressures on every sample")
    return lines, scored


def chamber_pressures(bag, t):
    """
    Resample `/cranedata`'s A/B pressures onto the `/joint_states` grid, or None.

    Refused outright when any state stamp has no `/cranedata` sample within
    `PRESSURE_STALE`. Held or backwards-extrapolated pressure is not a
    measurement, and this is the arm a reader believes: a late-starting or
    dropped-out publisher would otherwise hand the first or last sample to
    hundreds of cycles with nothing in the output saying so. `measured_force`'s
    only check is for NaN, which this never produces.
    """
    rows = bag.get("cranedata") or []
    if not rows:
        return None
    axes = ("sw", "ha", "ka", "sa", "ro", "gr")
    sims = np.array([r["sim"] for r in rows], float)
    order = np.argsort(sims)
    sims = sims[order]
    hydraulic = [rows[i]["msg"].hydraulic for i in order]
    p_a = np.array([[getattr(h, a).p_a for a in axes] for h in hydraulic], float)
    p_b = np.array([[getattr(h, a).p_b for a in axes] for h in hydraulic], float)
    # Nearest sample at or before each state stamp; the two streams are both
    # 100 Hz but neither is a subsample of the other.
    at = np.clip(np.searchsorted(sims, t, side="right") - 1, 0, len(sims) - 1)
    stale = int(np.sum(np.abs(t - sims[at]) > PRESSURE_STALE))
    if stale:
        return stale
    return dict(pressures_a=p_a[at], pressures_b=p_b[at])


# --------------------------------------------------------------------------- 4


def shadow_vs_drive(bag, window):
    """Per axis, the shadow command minus what drove: RMS, bias, worst window."""
    head = "4 SHADOW VS DRIVE  (shadow command - follower velocity, per axis)"
    rows = bag.get("shadow_cmp") or []
    if not rows:
        return missing(head, "~/shadow_comparison"), None
    rows = sorted(rows, key=lambda r: r["sim"])
    pairs = [
        {kv.key: kv.value for kv in s.values} for r in rows for s in r["msg"].status
    ]
    sims = np.array([r["sim"] for r in rows for _ in r["msg"].status], float)
    sources = defaultdict(int)
    for values in pairs:
        sources[values.get("follower.velocity_source", "absent")] += 1

    joints = sorted(
        {k.rsplit(".", 1)[0] for v in pairs for k in v if k.endswith(".difference")}
    )
    lines = [
        head,
        "",
        "  follower.velocity_source  "
        + "  ".join(f"{k} {n}" for k, n in sorted(sources.items())),
    ]
    if not joints:
        # `commissioning.md` 6's **Abort**, and the recorded-but-failed case is
        # not the not-recorded one: a wrong `controller_state_topic` is silent,
        # degrades to `follower.velocity_source: none`, and every number on the
        # layer is then measured against a command that was never applied.
        lines += [
            "",
            "  NO FOLLOWER VELOCITY on any axis -- commissioning.md 6 calls this an",
            "  abort, not a result. Either the node produced no shadow command at all,",
            "  or `controller_state_topic` does not name the topic the follower",
            "  publishes. Fix the remap before reading anything else in this report:",
            "  sections 3 and 5 score against the busiest controller_state in the bag,",
            "  which is not necessarily the one the node read.",
        ]
        return lines, NO_FOLLOWER
    period = looptiming.period_of(sims) or 1.0
    span = max(1, int(round(window / period)))
    stats = {}
    for joint in joints:
        series = np.array(
            [
                float(v[f"{joint}.difference"])
                for v in pairs
                if f"{joint}.difference" in v
            ],
            float,
        )
        kernel = np.ones(span) / span
        rolling = (
            np.convolve(series, kernel, mode="valid") if len(series) >= span else series
        )
        stats[joint] = dict(
            rms=float(np.sqrt(np.mean(series**2))),
            bias=float(np.mean(series)),
            worst=float(np.max(np.abs(rolling))),
            n=len(series),
        )
    lines += [
        "",
        f"  {'joint':<24} {'rms':>10} {'bias':>10} {'worst ' + f'{window:g}s':>12} {'n':>7}",
    ]
    for joint, row in sorted(stats.items(), key=lambda kv: -kv[1]["rms"]):
        lines.append(
            f"  {joint:<24} {row['rms']:10.4f} {row['bias']:10.4f} "
            f"{row['worst']:12.4f} {row['n']:7d}"
        )
    lines.append("  rad/s, m/s on `q4_big_telescope`.")
    loudest = max(stats.items(), key=lambda kv: kv[1]["rms"])
    lines += [
        "",
        f"  the MPC asks for most of what the PID does not on {loudest[0]}: "
        f"{loudest[1]['rms']:.4f} RMS, bias {loudest[1]['bias']:+.4f}.",
        "  `u != dq_ref` by construction, so a bias is expected; a bias that is a large",
        "  share of the axis' `u_clamp_*` is the MPC asking for a different move.",
    ]
    return lines, stats


# --------------------------------------------------------------------------- 5


def steps(bag):
    """
    Read a dead time plus a lag off each step candidate, with the C3 response beside it.

    A *candidate*, not a step test: nothing in the bag says a run was one, so
    what is found is a single-axis command off a quiet machine. The command
    droop is the tell -- `commissioning.md` 3 zeroes the gains on the axis under
    test, so a real step holds its command and an ordinary move's collapses.
    One candidate per axis, the largest, or the section buries the real runs.
    """
    head = "5 STEPS  (single-axis step candidates in the driven command)"
    js, ctrl = bag.get("js") or [], busiest(bag.get("ctrl") or [])
    if not js or not ctrl:
        return missing(head, "/joint_states + a controller_state"), None
    try:
        import plant_step
    except Exception as error:
        return [head, f"  not available offline: {str(error)[:90]}"], None

    canon = plant_step.CANON
    ctrl = sorted(ctrl, key=lambda r: r["sim"])
    js = sorted(js, key=lambda r: r["sim"])
    names = list(ctrl[0]["msg"].joint_names)
    axes = [j for j in canon if j in names]
    ct = np.array([r["sim"] for r in ctrl], float)
    cu = np.array([by_name(r["msg"], "output", axes) for r in ctrl])
    found = episodes(ct, np.nan_to_num(cu))
    if not found:
        return [
            head,
            "  no single-axis step candidate in this bag -- not recorded",
        ], None
    # Largest per axis: a long bag carries many slow single-axis moves and the
    # real runs would be buried among them.
    largest = {}
    for episode in found:
        best = largest.get(episode[0])
        if best is None or abs(episode[3]) > abs(best[3]):
            largest[episode[0]] = episode

    jt = np.array([r["sim"] for r in js], float)
    lines = [head, ""]
    stats = []
    for column, start, end, amplitude, droop in largest.values():
        joint = axes[column]
        row = response(js, jt, joint, start, end)
        if row is None:
            continue
        times, speed = row
        cells, dead, lag, settled = step_shape(
            times, speed, amplitude, end - start, plant_step.TIMES
        )
        stats.append(
            dict(joint=joint, amplitude=amplitude, dead=dead, lag=lag, droop=droop)
        )
        lines.append(
            f"  {joint}, step {amplitude:+.3f} at t+{start - jt[0]:.1f} s, "
            f"held {end - start:.2f} s, command droop {1e2 * droop:.0f}%"
        )
        if abs(droop) > MAX_DROOP:
            lines.append(
                f"    no dead time or lag: the command moved by "
                f"{1e2 * abs(droop):.0f}% over the hold, so the response is the"
            )
            lines.append(
                "    follower's loop and not the plant. Zero the axis' gains and "
                "step it again."
            )
        else:
            lines.append(
                f"    dead time {1e3 * dead:.0f} ms, lag to 63% of the settled "
                f"{settled:+.3f} is {1e3 * lag:.0f} ms, "
                f"sum {1e3 * (dead + lag):.0f} ms"
            )
        lines += [
            f"    {'':>26} "
            + " ".join(f"{int(f * 1000):>7}ms" for f in plant_step.TIMES),
            f"    {'measured, normalised':>26} " + " ".join(cells),
        ]
        model = c3_row(plant_step, js, jt, joint, start, amplitude)
        lines.append(f"    {'C3 / OCP model':>26} " + (model or "unavailable"))
    lines += [
        "",
        "  Velocities are rad/s, m/s on `q4_big_telescope`. `--` is past the recorded",
        "  window, not a settled value. A command droop means the follower's PI was live,",
        "  so the episode is not open loop and the lag carries the loop:",
        "  commissioning.md 3 zeroes the gains on the axis under test for this reason.",
        "  `k` is not separately identifiable from a velocity step without `M_ii`, which",
        "  is pose dependent; the row above is the response `k` and `tau_v` produce, which",
        "  is `plant_step.py`'s comparison. Past ~300 ms the episode is not open loop.",
    ]
    return lines, stats


def episodes(ct, cu):
    """
    Find each run where one axis alone is commanded off a quiet machine.

    `(column, start, end, amplitude, droop)`. The command is **not** required to
    be flat: `steptest.py` puts the step in `effort` and holds position, so with
    the follower's PI live `output.velocities` decays as the position error
    builds -- on the boom from 0.10 to 0.022 rad/s over a 0.6 s hold. Requiring
    flatness rejected every real steptest run and reported "not recorded".
    What is required instead is a rise from quiet, one axis, a held sign and no
    larger command inside the run; `amplitude` is the commanded step at the
    rise, which is what the C3 row is normalised by too, and `droop` says how
    far the command fell so a reader knows whether it was open loop.
    """
    live = np.abs(cu) > STEP_MIN
    busy = live.sum(axis=1)
    out = []
    k = 0
    while k < len(ct):
        if busy[k] != 1 or k == 0 or busy[k - 1] != 0:
            k += 1
            continue
        column = int(np.argmax(live[k]))
        first = cu[k, column]
        others = [c for c in range(cu.shape[1]) if c != column]
        j = k + 1
        while (
            j < len(ct)
            and not live[j, others].any()
            and cu[j, column] * first >= 0.0
            and STEP_FLOOR * abs(first) <= abs(cu[j, column]) <= 1.2 * abs(first)
        ):
            j += 1
        held = cu[k:j, column]
        if ct[j - 1] - ct[k] >= STEP_HOLD:
            out.append(
                (
                    column,
                    float(ct[k]),
                    float(ct[j - 1]),
                    float(first),
                    float(1.0 - abs(held[-1]) / abs(first)),
                )
            )
        k = max(j, k + 1)
    return out


def response(js, jt, joint, start, end):
    """`(t, dq)` of one joint over an episode, extended past it for the tail."""
    window = (jt >= start) & (jt <= end + 0.3)
    if window.sum() < 10 or joint not in js[0]["msg"].name:
        return None
    speed = []
    for row, keep in zip(js, window):
        if not keep:
            continue
        names = list(row["msg"].name)
        speed.append(
            row["msg"].velocity[names.index(joint)] if joint in names else np.nan
        )
    times = jt[window] - start
    return times, np.asarray(speed, float)


def step_shape(times, speed, amplitude, held, sample_at):
    """
    Normalise the response at `sample_at`, and read the dead time and lag off it.

    The reference is the **settled** velocity -- the median over the last third
    of the hold -- not the peak over the window. A boom or arm step through a
    load-holding valve overshoots, and against a peak 30 % above steady state
    the 63 % threshold becomes 82 % of the settled value, which a first-order
    response reaches at about 1.7 tau: the lag would read ~70 % long, and past
    about 60 % overshoot the threshold is never crossed and the lag reads NaN.

    Cells past `times[-1]` print `--`: `np.interp` clamps, so a 0.8 s column of
    a 0.65 s window would otherwise repeat the last sample in the same column
    as the model's genuine 0.8 s value and read as a settled plant.

    **Only the sum is meaningful**, as everywhere else in this stack: a true
    dead time of `d` followed by a first-order rise of `tau` crosses 10 % at
    `d + tau*ln(10/9)`, so the split moves ~0.1 `tau` out of the lag and into
    the dead time while `dead + lag` stays `d + tau`. That is the quantity
    `commissioning.md` 3 bounds per axis and the one `--lag-shift` holds.
    """
    # Inside the hold only: `response` keeps 0.3 s past it for the tail of the
    # response, and once the command stops so does the axis, so a median across
    # the boundary is half post-episode zeros.
    tail = speed[(times >= 2.0 * held / 3.0) & (times <= held)]
    settled = float(np.median(tail)) if len(tail) else float(speed[-1])
    if not np.isfinite(settled) or settled == 0.0:
        settled = float(speed[np.argmax(np.abs(speed))])
    crossed = np.nonzero(np.abs(speed) > 0.1 * abs(settled))[0]
    dead = float(times[crossed[0]]) if len(crossed) else float("nan")
    reached = np.nonzero(np.abs(speed) > 0.632 * abs(settled))[0]
    lag = float(times[reached[0]]) - dead if len(reached) else float("nan")
    cells = [
        f"{'--':>9}"
        if f > times[-1]
        else f"{float(np.interp(f, times, speed)) / amplitude:>9.3f}"
        for f in sample_at
    ]
    return cells, dead, lag, settled


def c3_row(plant_step, js, jt, joint, start, amplitude):
    """`plant_step.ocp_model` from the pose the episode started in, as a text row."""
    import copy

    at = int(np.argmin(np.abs(jt - start)))
    names = list(js[at]["msg"].name)
    row = plant_step.planned_row(joint)
    if row is None:
        return "not a planned axis, so the OCP carries no model of it"
    if not set(plant_step.CANON) <= set(names):
        return None
    hold = {
        j: (
            js[at]["msg"].position[names.index(j)],
            js[at]["msg"].velocity[names.index(j)],
        )
        for j in plant_step.CANON
    }
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory
        from crane_model import hydraulic_limits
        from crane_mpc.config import machine_limits

        share = get_package_share_directory("crane_mpc")
        with open(f"{share}/config/crane_mpc.yaml") as stream:
            parameters = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
        parameters.update({"hydraulics": hydraulic_limits()})
        parameters["limits"].update(machine_limits())
        model = plant_step.ocp_model(hold, amplitude, copy.deepcopy(parameters), row)
    except Exception as error:
        return f"unavailable: {str(error)[:60]}"
    return " ".join(f"{v:>9.3f}" for v in model)


# --------------------------------------------------------------------------- 6


def checklist(timing, rate, suspect, shadow, scored, jtc_rate):
    """Lay out the table `commissioning.md` 6 is waiting for, as markdown."""
    rows = []

    def check(quantity, value, threshold, verdict):
        rows.append((quantity, value, threshold, verdict))

    if timing is None:
        check("solve timing", "not recorded", "-", None)
    else:
        # The budget criterion `commissioning.md` 6 names, and not `solve_time`:
        # that is `time_tot`, so a node late on every cycle still reports a p90
        # well inside the budget.
        check(
            "`SOLVE_BUDGET_EXCEEDED` share",
            f"{1e2 * timing['exceeded']:.1f} %",
            "0 %",
            timing["exceeded"] == 0.0,
        )
        check(
            "`SOLVE_CONVERGED` share",
            f"{1e2 * timing['converged']:.1f} %",
            f">= {1e2 * CONVERGED_SHARE:.0f} % of *published* reports",
            timing["converged"] >= CONVERGED_SHARE,
        )
        check(
            "acados `solve_time` p90",
            f"{1e3 * timing['p90']:.1f} ms "
            f"(`solve_budget` {1e3 * timing['budget']:.1f} ms)",
            "context only -- `time_tot`, not end to end",
            None,
        )
        check(
            "`used_slack` share",
            f"{1e2 * timing['slack']:.1f} %",
            "0 % on ordinary moves",
            timing["slack"] == 0.0,
        )
        check(
            "`FAULT_SOLVER` reports",
            str(timing["fault"]),
            "0",
            timing["fault"] == 0,
        )
        check(
            "fallback (`applied_previous_solution`)",
            f"{1e2 * timing['fallback']:.1f} %",
            "0 % unless exercised deliberately",
            timing["fallback"] == 0.0,
        )
        if "silences" in timing:
            check(
                "silence episodes",
                f"{timing['silences']}, worst {1e3 * timing['worst_gap']:.0f} ms",
                "0",
                timing["silences"] == 0,
            )
    if rate is None:
        check("`controller_state` rate", "not recorded", f"{jtc_rate:.0f} Hz", None)
    else:
        check(
            "`controller_state` rate",
            f"{rate['hz']:.1f} Hz",
            f">= {jtc_rate:.0f} Hz",
            rate["hz"] >= 0.98 * jtc_rate,
        )
        if rate.get("starved") is not None:
            check(
                "cycles under `jtc_output` quorum",
                f"{1e2 * rate['starved']:.0f} %",
                "0 %",
                rate["starved"] == 0.0,
            )
    check(
        "host clocks",
        "not measured"
        if suspect is None
        else ("offset or unsynced" if suspect else "consistent"),
        "no negative stamp-to-receipt",
        None if suspect is None else not suspect,
    )
    if shadow is None:
        check("shadow vs drive", "not recorded", "-", None)
    elif isinstance(shadow, str):
        check("`follower.velocity_source`", shadow, "never `none` (6 **Abort**)", False)
    else:
        loudest = max(shadow.items(), key=lambda kv: kv[1]["rms"])
        check(
            f"shadow-vs-drive RMS, worst axis (`{loudest[0]}`)",
            f"{loudest[1]['rms']:.4f}",
            "record it; `u != dq_ref`, so no threshold",
            None,
        )
    if not scored:
        check("model skill", "not recorded", "> 0 per axis", None)
    else:
        for label, per_axis in scored.items():
            shown = {a: per_axis[a] for a in AXES if per_axis.get(a) is not None}
            if not shown:
                check(f"model skill, {label}", "no axis moved enough", "> 0", None)
                continue
            check(
                f"model skill, {label}",
                " ".join(f"{a} {v[1]:.2f}" for a, v in shown.items()),
                "> 0 on every moved axis",
                all(v[1] > 0 for v in shown.values()),
            )

    width = max(len(r[0]) for r in rows)
    mark = {True: "pass", False: "**FAIL**", None: "record"}
    lines = [
        "6 CHECKLIST  (paste into wiki/commissioning.md 6)",
        "",
        f"| {'quantity':<{width}} | value | pass threshold | verdict |",
        f"|{'-' * (width + 2)}|---|---|---|",
    ]
    for quantity, value, threshold, verdict in rows:
        lines.append(
            f"| {quantity:<{width}} | {value} | {threshold} | {mark[verdict]} |"
        )
    return lines


def guarded(section, call, *arguments):
    """
    Run one section, or say what it raised. Never lose the other five to it.

    The hardware time is already spent by the time this runs, so a section that
    meets something it did not expect has to degrade to a line of text. The
    blank result reads the same as "not recorded" downstream, which is correct:
    nothing was measured.
    """
    try:
        return call(*arguments)
    except Exception as error:
        kind = type(error).__name__
        return ([section, f"  refused: {kind}: {str(error)[:100]}"],) + (None,) * 2


def report(path, jtc_rate=looptiming.NOMINAL_JTC_HZ, window=1.0):
    bag, streams = load(path)
    out = [f"{path}", f"{len(streams)} recorded streams the report reads", ""]
    first, rate, suspect = guarded("1 LATENCY", latency, bag, streams, jtc_rate)
    second, timing = guarded("2 SOLVE TIMING", solve_timing, bag)[:2]
    third, scored = guarded("3 MODEL SKILL", model_skill, bag)[:2]
    fourth, shadow = guarded("4 SHADOW VS DRIVE", shadow_vs_drive, bag, window)[:2]
    fifth = guarded("5 STEPS", steps, bag)[0]
    sixth = checklist(timing, rate, suspect, shadow, scored, jtc_rate)
    for section in (first, second, third, fourth, fifth, sixth):
        out += section + [""]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("bag", help="one .mcap recorded with labday_topics.txt")
    parser.add_argument(
        "--jtc-rate",
        type=float,
        default=looptiming.NOMINAL_JTC_HZ,
        help="nominal velocity_loop.yaml rate_hz, for the quorum",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="seconds of the worst shadow-vs-drive window",
    )
    options = parser.parse_args(argv)
    print(report(options.bag, options.jtc_rate, options.window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
