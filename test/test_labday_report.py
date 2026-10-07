"""
`labday_report.py` has to read a lab day's numbers back out of an mcap -- and refuse the rest.

The recording half is human-run on the machine (PRD 14), so the measurement is
checked against a synthetic mcap whose delivery latency and model error are
injected and therefore known. The model arm uses a one-line plant rather than
the shipped OCP: a skill number off the real OCP is a measurement, and an
assertion on one tests nothing but that the model has not changed.

Written with `mcap_ros2`'s writer against the *installed* message definitions,
so the fixture carries the same embedded schemas a `ros2 bag record` writes.
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory
from mcap_ros2.writer import Writer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trials"))

import labday_report as lab  # noqa: E402
import modelfit  # noqa: E402

LATENCY = 0.018
TICK = 0.01
TS = 0.06
STRIDE = int(round(TS / TICK))
#: The model predicts this multiple of the command the machine received.
ERROR_GAIN = 1.3
#: Where a `step=` fixture puts its one held command, and for how long.
STEP_AT, STEP_FOR = 2.0, 1.0
BUDGET = 0.06
#: A plausible wall epoch: a bag's `log_time` is unsigned, so a stamp that runs
#: ahead of the receipt has to still land after it.
EPOCH = 1.787e9
HEADER = "=" * 80
PRIMITIVE = re.compile(
    r"^(bool|byte|char|float32|float64|int8|uint8|int16|uint16|int32|uint32"
    r"|int64|uint64|string|wstring)\b"
)
HDR = {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}
POINT = {
    "positions": [],
    "velocities": [],
    "accelerations": [],
    "effort": [],
    "time_from_start": {"sec": 0, "nanosec": 0},
}


def _msg_path(datatype):
    package, _, name = datatype.replace("/msg/", "/").partition("/")
    return Path(get_package_share_directory(package)) / "msg" / f"{name}.msg"


def _dependencies(datatype, text):
    """Non-primitive field types, qualified: a same-package reference has no slash."""
    package = datatype.split("/")[0]
    out = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or len(line.split()) < 2 or "=" in line.split()[1]:
            continue
        base = line.split()[0].split("[")[0]
        if PRIMITIVE.match(base):
            continue
        out.append(
            "std_msgs/Header"
            if base == "Header"
            else (base if "/" in base else f"{package}/{base}")
        )
    return out


def msgdef(datatype):
    """rosbag2's concatenated ROS 2 definition for a type, read off the install."""
    root = _msg_path(datatype).read_text()
    blocks, seen, queue = [], set(), _dependencies(datatype, root)
    while queue:
        dependency = queue.pop(0)
        if dependency in seen:
            continue
        seen.add(dependency)
        text = _msg_path(dependency).read_text()
        blocks.append(f"{HEADER}\nMSG: {dependency}\n{text}")
        queue += _dependencies(dependency, text)
    return root + ("\n" + "".join(blocks) if blocks else "")


def stamped(seconds):
    at = EPOCH + seconds
    return {
        "stamp": {"sec": int(at), "nanosec": int(round((at % 1.0) * 1e9))},
        "frame_id": "",
    }


class Plant:
    """One cycle: the next velocity is `ERROR_GAIN` times the command in flight."""

    Ts = TS
    command_state = False

    def __init__(self, gain=ERROR_GAIN):
        self.gain = gain
        self.nx = modelfit.cs.NX
        self.parameters = modelfit.shipped()
        self._stepper = None

    def pin_tool(self, value):
        pass

    def static_hold_force(self, x):
        return np.zeros(modelfit.PL)

    def _integrate(self, stepper, x, u):
        out = np.array(x, float)
        out[modelfit.VEL] = self.gain * np.asarray(u[: modelfit.PL], float)
        return out


def signal(samples):
    """A slewing-axis velocity that actually changes speed, so cycles get scored."""
    return 0.3 * np.sin(2.0 * np.pi * 0.25 * np.arange(samples) * TICK)


def traces(samples, step):
    """`(velocity, command)` per sample: the model-error sine, or one held step."""
    if step is None:
        dq = signal(samples + STRIDE)
        return dq[:samples], dq[STRIDE : samples + STRIDE]
    amplitude, dead, tau = step
    t = np.arange(samples) * TICK
    live = (t >= STEP_AT) & (t < STEP_AT + STEP_FOR)
    since = np.clip(t - STEP_AT - dead, 0.0, None)
    return (
        np.where(live, amplitude * (1.0 - np.exp(-since / tau)), 0.0),
        np.where(live, amplitude, 0.0),
    )


def build(
    path,
    latency=LATENCY,
    seconds=12.0,
    shadow=0.04,
    health=True,
    differences=True,
    step=None,
):
    """
    A bag delivered `latency` late, whose slewing velocity is `signal`.

    The command on the interface at sample `k` is the velocity that lands one
    cycle later, so a plant that answers `gain * u` has a model error of exactly
    `(gain - 1) * dq[k + STRIDE]`.
    """
    samples = int(seconds / TICK)
    dq, cmd = traces(samples, step)
    joints = [
        modelfit.CANON[i] for i in modelfit.PLANNED_ROWS + modelfit.PASSIVE_ROWS + [7]
    ]
    planned = [modelfit.CANON[i] for i in modelfit.PLANNED_ROWS]
    with open(path, "wb") as handle:
        writer = Writer(handle)
        schemas = {
            name: writer.register_msgdef(name, msgdef(name))
            for name in (
                "sensor_msgs/msg/JointState",
                "control_msgs/msg/JointTrajectoryControllerState",
                "diagnostic_msgs/msg/DiagnosticArray",
                "crane_msgs/msg/SolverHealth",
                "trajectory_msgs/msg/JointTrajectory",
            )
        }

        def put(topic, datatype, message, at):
            writer.write_message(
                topic,
                schemas[datatype],
                message,
                log_time=int(round((EPOCH + at + latency) * 1e9)),
            )

        for k in range(samples):
            at = k * TICK
            velocity = [0.0] * len(joints)
            velocity[0] = float(dq[k])
            put(
                "/joint_states",
                "sensor_msgs/msg/JointState",
                {
                    "header": stamped(at),
                    "name": joints,
                    "position": [0.0] * len(joints),
                    "velocity": velocity,
                    "effort": [],
                },
                at,
            )
            command = [0.0] * len(planned)
            command[0] = float(cmd[k])
            put(
                "/crane/controller_state",
                "control_msgs/msg/JointTrajectoryControllerState",
                {
                    "header": stamped(at),
                    "joint_names": planned,
                    "reference": POINT,
                    "feedback": POINT,
                    "error": POINT,
                    "output": dict(POINT, velocities=command),
                    "multi_dof_joint_names": [],
                    "multi_dof_reference": [],
                    "multi_dof_feedback": [],
                    "multi_dof_error": [],
                    "multi_dof_output": [],
                    "speed_scaling_factor": 1.0,
                },
                at,
            )
        for k in range(int(seconds / TS)):
            at = k * TS
            # The cycle grid the quorum share is counted against. In shadow the
            # contract topic is silent, so this is the only horizon in the bag.
            put(
                "/crane_mpc/shadow_horizon",
                "trajectory_msgs/msg/JointTrajectory",
                {
                    "header": stamped(at),
                    "joint_names": planned,
                    "points": [dict(POINT, velocities=[0.0] * len(planned))],
                },
                at,
            )
            if health:
                put(
                    "/crane/mpc/solver_health",
                    "crane_msgs/msg/SolverHealth",
                    {
                        "header": stamped(at),
                        "outcome": 1,
                        "fault": 0,
                        "status": 0,
                        "status_word": "SUCCESS",
                        "iterations": 2,
                        "qp_status": 0,
                        "qp_iterations": 7,
                        "solve_time": 0.5 * BUDGET,
                        "solve_budget": BUDGET,
                        "applied_previous_solution": False,
                        "used_slack": False,
                        "slack_penalty": 0.0,
                        "constraint_violation": [0.0] * 4,
                        "cost_term": [0.0] * 8,
                        "joint_names": planned,
                        "message": "shadow: ok",
                    },
                    at,
                )
            values = [
                SimpleNamespace(
                    key="follower.velocity_source",
                    value="output.velocities" if differences else "none",
                )
            ]
            if differences:
                values.append(
                    SimpleNamespace(
                        key=f"{planned[0]}.difference", value=f"{shadow:.17g}"
                    )
                )
            put(
                "/crane_mpc/shadow_comparison",
                "diagnostic_msgs/msg/DiagnosticArray",
                {
                    "header": stamped(at),
                    "status": [
                        SimpleNamespace(
                            level=0,
                            name="crane_mpc",
                            message="ok",
                            hardware_id="pzs100",
                            values=values,
                        )
                    ],
                },
                at,
            )
        writer.finish()
    return path


def test_recovers_the_injected_delivery_latency(tmp_path):
    bag, streams = lab.load(build(tmp_path / "late.mcap"))
    lines, rate, suspect = lab.latency(bag, streams, 100.0)
    assert not suspect
    for topic in ("/joint_states", "/crane/controller_state"):
        assert lab.looptiming.latency_of(streams[topic])["p50"] == pytest.approx(
            LATENCY, abs=1e-6
        )
    assert rate["hz"] == pytest.approx(1 / TICK, rel=1e-3)
    assert rate["per_cycle"] == pytest.approx(TS / TICK, rel=1e-3)
    assert rate["starved"] == 0.0


def test_a_stamp_ahead_of_the_receipt_is_a_clock_problem_not_a_result(tmp_path):
    """Unsynced hosts make the latency the offset between them; say so, don't report it."""
    bag, streams = lab.load(build(tmp_path / "skewed.mcap", latency=-0.004))
    lines, _, suspect = lab.latency(bag, streams, 100.0)
    assert "/joint_states" in suspect
    assert "CLOCK PROBLEM, not a result" in "\n".join(lines)
    assert "chrony" in "\n".join(lines)


def test_recovers_the_injected_model_error(tmp_path):
    """`(gain - 1) * u` is the one-cycle error; a stride or command slip breaks it."""
    bag, _ = lab.load(build(tmp_path / "model.mcap"))
    lines, scored = lab.model_skill(bag, ocp=Plant())
    per_axis = scored["force row carried"]

    dq = signal(int(12.0 / TICK) + STRIDE)
    k = np.arange(0, int(12.0 / TICK) - STRIDE, STRIDE)
    error, change = (ERROR_GAIN - 1.0) * dq[k + STRIDE], dq[k] - dq[k + STRIDE]
    moved = np.abs(change) > modelfit.MOVING
    rms = float(np.sqrt(np.mean(error[moved] ** 2)))

    assert per_axis["sw"][0] == pytest.approx(rms, rel=1e-6)
    assert per_axis["sw"][1] == pytest.approx(
        1.0 - rms / float(np.sqrt(np.mean(change[moved] ** 2))), rel=1e-6
    )
    # Only slewing was driven, so no other axis changed speed and none is scored.
    assert [a for a in modelfit.AX if per_axis[a] is not None] == ["sw"]
    # No /cranedata in the fixture, so the pressure arm says so instead of failing.
    assert "/cranedata not recorded" in "\n".join(lines)
    assert "force row from measured pressure" not in scored


def test_an_exact_model_scores_one(tmp_path):
    bag, _ = lab.load(build(tmp_path / "exact.mcap"))
    _, scored = lab.model_skill(bag, ocp=Plant(gain=1.0))
    assert scored["force row carried"]["sw"][1] == pytest.approx(1.0, abs=1e-9)


def test_reads_the_shadow_difference_and_the_solve_budget(tmp_path):
    bag, _ = lab.load(build(tmp_path / "shadow.mcap", shadow=0.04))
    lines, stats = lab.shadow_vs_drive(bag, 1.0)
    axis = modelfit.CANON[0]
    assert stats[axis]["rms"] == pytest.approx(0.04, abs=1e-9)
    assert stats[axis]["bias"] == pytest.approx(0.04, abs=1e-9)
    assert "output.velocities" in "\n".join(lines)

    lines, timing = lab.solve_timing(bag)
    assert timing["p90"] == pytest.approx(0.5 * BUDGET, abs=1e-9)
    assert timing["budget"] == pytest.approx(BUDGET)
    assert timing["exceeded"] == 0.0 and timing["converged"] == 1.0
    assert timing["silences"] == 0
    # The wiki's 6 is explicit that `solve_time` is not the lateness test.
    assert "not the lateness test" in "\n".join(lines)


def test_no_follower_velocity_is_an_abort_not_a_blank(tmp_path):
    """
    A wrong `controller_state_topic` is silent and section 6 of the wiki aborts on it.

    The node then emits no `<joint>.difference` at all, which must not render the
    same as a bag that never carried the topic.
    """
    bag, _ = lab.load(build(tmp_path / "nofollower.mcap", differences=False))
    lines, shadow = lab.shadow_vs_drive(bag, 1.0)
    assert shadow == lab.NO_FOLLOWER
    assert "NO FOLLOWER VELOCITY" in "\n".join(lines)
    table = "\n".join(lab.checklist(None, None, None, shadow, {}, 100.0))
    assert "**FAIL**" in table and "follower.velocity_source" in table


def test_recovers_a_steps_dead_time_and_lag(tmp_path):
    """
    The *sum* of the dead time and the lag is `d + tau`; only the sum is identified.

    The 10 % threshold moves `tau*ln(10/9)` out of the lag into the dead time,
    so the split is asserted against that and the sum against `d + tau`. The
    detector must also *fire*: it keys on a command held off a quiet machine,
    not on a flat one, because `steptest.py` puts its step in `effort` and the
    follower's command decays as the position error builds.
    """
    dead, tau = 0.05, 0.12
    bag, _ = lab.load(build(tmp_path / "step.mcap", step=(0.30, dead, tau)))
    lines, found = lab.steps(bag)
    assert len(found) == 1
    assert found[0]["joint"] == modelfit.CANON[0]
    assert found[0]["amplitude"] == pytest.approx(0.30)
    assert found[0]["droop"] == pytest.approx(0.0, abs=1e-9)
    assert found[0]["dead"] + found[0]["lag"] == pytest.approx(dead + tau, abs=TICK)
    assert found[0]["dead"] == pytest.approx(dead + tau * np.log(10 / 9), abs=TICK)
    # 0.8 s is inside the 1.0 s hold, so the response is settled and normalises to 1.
    assert "0.998" in "\n".join(lines)


def test_a_collapsed_command_is_refused_rather_than_timed(tmp_path):
    """A command that decays over the hold measures the follower's loop, not the plant."""
    bag, _ = lab.load(build(tmp_path / "droop.mcap", step=(0.30, 0.05, 0.12)))
    ctrl = lab.busiest(bag["ctrl"])
    for row in ctrl:  # collapse the second half of the hold
        if STEP_AT + 0.5 <= row["sim"] - EPOCH < STEP_AT + STEP_FOR:
            row["msg"].output.velocities[0] = 0.1 * 0.30
    lines, found = lab.steps(bag)
    assert found[0]["droop"] == pytest.approx(0.9, abs=1e-9)
    assert "no dead time or lag" in "\n".join(lines)


def test_a_missing_topic_says_not_recorded_and_does_not_fail(tmp_path):
    """A bag from the wrong layer is the normal case, not an error."""
    bag, _ = lab.load(build(tmp_path / "nohealth.mcap", health=False))
    lines, timing = lab.solve_timing(bag)
    assert timing is None and "not recorded" in "\n".join(lines)
    # No steptest episode either: the command never sits on one axis alone.
    lines, found = lab.steps(bag)
    assert found is None and "not recorded" in "\n".join(lines)


def test_the_checklist_carries_the_thresholds_and_a_verdict(tmp_path):
    bag, streams = lab.load(build(tmp_path / "full.mcap"))
    _, rate, suspect = lab.latency(bag, streams, 100.0)
    _, timing = lab.solve_timing(bag)
    _, shadow = lab.shadow_vs_drive(bag, 1.0)
    _, scored = lab.model_skill(bag, ocp=Plant())
    table = "\n".join(lab.checklist(timing, rate, suspect, shadow, scored, 100.0))
    assert "paste into wiki/commissioning.md 6" in table
    assert "`SOLVE_BUDGET_EXCEEDED` share" in table
    # `solve_time` is in the table, but it carries no verdict: it is `time_tot`.
    assert "context only -- `time_tot`, not end to end" in table
    assert "**FAIL**" in table  # the injected model error, which is real
    assert table.count("| pass |") >= 4
