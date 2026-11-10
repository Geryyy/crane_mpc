#!/usr/bin/env python3
"""
Issue 304's loaded-slew arm: the 11-09 stacking slew, in Gazebo, MPC driving.

    S=src/concrete_block_stack/crane_mpc/scripts/trials
    MOVE=$PWD/$S/slew_repro.py ./$S/trial.sh /tmp/304-mpc            # MPC driving
    CTRL=pid MOVE=$PWD/$S/slew_repro.py ./$S/trial.sh /tmp/304-pid   # the plan on the JTC alone
    ./$S/slew_loop.py /tmp/304-mpc/slew/slew_0.mcap

`trial.sh` owns the sim (lock, `sim_env.sh`, headless, `trajectory_controller_a2b`
activated); this owns the move. Start and goal are the bag's `/joint_states` at
75.6 s and 81.9 s (`records/2026-11-09_16-16-56-stacking_mpc`), as a park pose
and -- through `crane_model`'s FK, since `y_n` is the tip K5 in K0 -- an
`/a2b_movement` goal. 0.562 rad of slew at ~7.4 m reach, 880 kg.

The payload is physical: `block_0` is teleported into the jaws and welded with
the same `DetachableJoint` call `gazebo_grasp_weld.py` makes on a real grasp,
with physics paused in between,
because this launch spawns the blocks 6-8 m away and a scripted A2B grips
nothing. The hang length is measured afterwards and printed -- the pendulum is
the thing being reproduced, so a wrong one has to be visible.

Plant switches (valve lag, ...) are sim launch args, not set here.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from controller_manager_msgs.srv import SwitchController
from crane_model import CraneModel, Frame, Tool, canonical_joints
from crane_msgs.srv import SetPayload
from geometry_msgs.msg import Point, Vector3
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from timber_crane_planning_interfaces.srv import CalcMovement

# The bag's `/joint_states`, by name. Tip and tilt are the passive pair and are
# parked at what they hang at, not commanded.
START = {
    "theta1_slewing_joint": -0.2568,
    "theta2_boom_joint": 0.6704,
    "theta3_arm_joint": 0.5493,
    "q4_big_telescope": 1.0676,
    "theta6_tip_joint": 0.3863,
    "theta7_tilt_joint": 1.5644,
    "theta8_rotator_joint": 5.0444,
    # Jaws open (1.22 m > the block's 0.9): clamped, gazebo_grasp_weld adds its own weld,
    # then detaches ours on the next 0.01 of q9 and welds whatever block it finds next.
    "q9_left_rail_joint": 0.40,
}
GOAL = dict(
    START,
    theta1_slewing_joint=0.3039,
    theta2_boom_joint=0.7098,
    theta3_arm_joint=0.5554,
    q4_big_telescope=1.0502,
    theta6_tip_joint=0.3641,
    theta7_tilt_joint=1.5807,
    theta8_rotator_joint=5.0319,
)
PASSIVE = ("theta6_tip_joint", "theta7_tilt_joint")
HANG_M = 1.05  # what the tree hands the MPC and the planner: block_grip 0.10 + tcp 0.95
PLANT_HANG_M = (
    1.12  # the block in Gazebo: 11-09 lidar; at 1.05 it intersects the gripper body
)
BLOCK = "block_0"
WORLD = "default"
A2B = "trajectory_controller_a2b"
TOPICS = (
    "/joint_states",
    "/trajectory_controllers/controller_state",
    "/crane/mpc/horizon",
    "/crane/mpc/solver_health",
    "/crane/mpc/cost_terms",
    "/crane/sway_settled",
    "/crane/reference",
)


def gz(*args: str, timeout: float = 10.0) -> str:
    done = subprocess.run(
        ["gz", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    return done.stdout.strip()


def gz_service(
    service: str, reqtype: str, request: str, reptype="gz.msgs.Boolean"
) -> str:
    return gz(
        "service",
        "-s",
        service,
        "--reqtype",
        reqtype,
        "--reptype",
        reptype,
        "--timeout",
        "5000",
        "--req",
        request,
    )


def pause(on: bool) -> None:
    gz_service(
        f"/world/{WORLD}/control", "gz.msgs.WorldControl", f"pause: {str(on).lower()}"
    )


def _xyzw(q: dict) -> tuple[float, float, float, float]:
    """GZ JSON drops zero fields: an absent w is 0 unless the whole quaternion is absent."""
    x, y, z = (float(q.get(k, 0.0)) for k in "xyz")
    return x, y, z, float(q.get("w", 0.0 if (x or y or z) else 1.0))


def compose(a: dict, b: dict) -> dict:
    """Quaternion product a*b."""
    ax, ay, az, aw = _xyzw(a)
    bx, by, bz, bw = _xyzw(b)
    return {
        "x": aw * bx + ax * bw + ay * bz - az * by,
        "y": aw * by - ax * bz + ay * bw + az * bx,
        "z": aw * bz + ax * by - ay * bx + az * bw,
        "w": aw * bw - ax * bx - ay * by - az * bz,
    }


def rotate(quaternion: dict, vector: list[float]) -> list[float]:
    """`vector` in a frame at `quaternion`, in the parent frame."""
    x, y, z, w = _xyzw(quaternion)
    vx, vy, vz = vector
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def gz_poses() -> dict:
    """`/world/<w>/dynamic_pose/info`'s newest frame, by name -- the weld's own source."""
    raw = gz(
        "topic",
        "-e",
        "-t",
        f"/world/{WORLD}/dynamic_pose/info",
        "-n",
        "1",
        "--json-output",
        timeout=20.0,
    )
    return {entry["name"]: entry for entry in json.loads(raw)["pose"]}


class Repro(Node):
    def __init__(self) -> None:
        super().__init__("slew_repro")
        self.state: JointState | None = None
        self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self.jtc = ActionClient(
            self, FollowJointTrajectory, f"/{A2B}/follow_joint_trajectory"
        )
        self.a2b = self.create_client(CalcMovement, "/a2b_movement")
        self.payload = self.create_client(SetPayload, "/crane/mpc/set_payload")
        self.switcher = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )

    def _on_state(self, message: JointState) -> None:
        self.state = message

    def wait(self, future, timeout: float):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def measured(self) -> dict:
        while self.state is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        return dict(zip(self.state.name, self.state.position))

    def rate(self, joint: str) -> float:
        row = self.state.name.index(joint)
        return abs(self.state.velocity[row])

    def activate_jtc(self) -> None:
        """
        Toggle the JTC off, then on.

        The tree owns this switch live, and a scripted arm has
        to do it itself. The off edge is not a formality -- `/crane/reference`
        is latched and `crane_mpc` drops a held plan on active -> inactive, so
        without it an earlier run's plan would start the next move.
        """
        self.switcher.wait_for_service(timeout_sec=60.0)
        for activate, deactivate in (([], [A2B]), ([A2B], [])):
            request = SwitchController.Request()
            request.activate_controllers = activate
            request.deactivate_controllers = deactivate
            request.strictness = 1  # BEST_EFFORT: already-inactive is not a failure
            response = self.wait(self.switcher.call_async(request), 30.0)
            print(f"switch +{activate} -{deactivate} ok={response.ok}")
            self.spin(1.0)

    def park(self, pose: dict, seconds: float) -> bool:
        """
        Hold every commanded joint at `pose`; the passive pair where it hangs.

        Zero stamp (play now) and a tolerance on the five actuated joints only:
        q9 stalls against the clamped block and a tolerance on it aborts.
        """
        measured = self.measured()
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            name for name in canonical_joints() if name in measured
        ]
        point = JointTrajectoryPoint()
        point.positions = [
            measured[name] if name in PASSIVE else pose[name]
            for name in trajectory.joint_names
        ]
        point.velocities = [0.0] * len(trajectory.joint_names)
        point.time_from_start.sec = int(seconds)
        trajectory.points.append(point)
        goal = FollowJointTrajectory.Goal(trajectory=trajectory)
        goal.goal_tolerance = [
            JointTolerance(name=name, position=0.05)
            for name in trajectory.joint_names
            if name not in PASSIVE and name != "q9_left_rail_joint"
        ]
        self.jtc.wait_for_server(timeout_sec=60.0)
        handle = self.wait(self.jtc.send_goal_async(goal), 20.0)
        if handle is None or not handle.accepted:
            print("park goal was not accepted")
            return False
        result = self.wait(handle.get_result_async(), seconds + 90.0)
        print(f"park error_code={result.result.error_code}")
        return result.result.error_code == 0

    def settle(self, seconds: float) -> None:
        """
        Spin until the crane is calm.

        Sway under 0.05 rad/s and every actuated joint under
        0.02 (its rest deadband) for 2 s, or the planner refuses the start.
        """
        end = time.monotonic() + seconds
        calm_since = time.monotonic()
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            sway = max(self.rate(joint) for joint in PASSIVE)
            drive = max(
                self.rate(joint)
                for joint in START
                if joint not in (*PASSIVE, "q9_left_rail_joint")
            )
            if sway >= 0.05 or drive >= 0.02:
                calm_since = time.monotonic()
            elif (
                time.monotonic() - calm_since > 2.0
            ):  # a swing's turning point reads zero rate
                break
        print(f"settled at sway {sway:.4f}, drive {drive:.4f} rad/s")

    def set_payload(self, mass: float) -> None:
        self.payload.wait_for_service(timeout_sec=30.0)
        request = SetPayload.Request()
        request.payload.shape = 1  # SHAPE_BOX
        request.payload.dimensions = Vector3(x=0.6, y=0.6, z=0.9)
        request.payload.mass = mass
        request.payload.com = Point(x=0.0, y=0.0, z=HANG_M)
        response = self.wait(self.payload.call_async(request), 30.0)
        print(
            f"set_payload success={response.success} active_mass={response.active_mass}"
        )

    def plan(self, y_n, yaw: float, mass: float):
        self.a2b.wait_for_service(timeout_sec=60.0)
        request = CalcMovement.Request()
        request.y_n = Point(x=float(y_n[0]), y=float(y_n[1]), z=float(y_n[2]))
        request.phi_tool_n = yaw
        request.t_end = 0.0
        request.slow_down = 1.0
        request.carries_log = True
        request.m_log = mass
        # The tree's own collision body for the upright block, both fields, or
        # `a2b.py` refuses the pair.
        for shape in (request.log_carrying, request.coll_shape):
            shape.length = 0.60
            shape.radius_top = 0.541
            shape.radius_bottom = 0.541
        request.s_log_8 = Point(x=0.0, y=0.0, z=HANG_M)
        request.p_cyl_8 = Point(x=0.0, y=0.0, z=HANG_M)
        response = self.wait(self.a2b.call_async(request), 120.0)
        print(
            f"plan success={response.success} points={len(response.trajectory.points)}"
        )
        if (
            response.success and os.environ.get("CTRL") == "pid"
        ):  # no MPC: the plan itself, on the JTC
            response.trajectory.header.stamp.sec = 0
            response.trajectory.header.stamp.nanosec = 0
            goal = FollowJointTrajectory.Goal(trajectory=response.trajectory)
            self.wait(self.jtc.send_goal_async(goal), 20.0)
        return response.success


def tool_pose() -> tuple[list[float], dict]:
    """`moved_with_q8`'s position and orientation in world."""
    poses = gz_poses()
    crane, link = poses["crane_model"], poses["moved_with_q8"]
    position = lambda entry: [  # noqa: E731
        float(entry.get("position", {}).get(key, 0.0)) for key in "xyz"
    ]
    tool = [
        base + offset
        for base, offset in zip(
            position(crane), rotate(crane.get("orientation", {}), position(link))
        )
    ]
    orientation = compose(crane.get("orientation", {}), link.get("orientation", {}))
    return tool, orientation


def push(force: float | None) -> None:
    """
    Add a persistent vertical force on the block (gz sums them); None clears all.

    Never clear-then-set mid-ramp: two gz calls leave ~0.2 s unsupported, and an
    880 kg block dropped and caught at 2 Hz hammers boom and arm +-0.1 rad.
    """
    entity = f'name: "{BLOCK}::link", type: LINK'
    if force is None:
        gz(
            "topic",
            "-t",
            f"/world/{WORLD}/wrench/clear",
            "-m",
            "gz.msgs.Entity",
            "-p",
            entity,
        )
        return
    gz(
        "topic",
        "-t",
        f"/world/{WORLD}/wrench/persistent",
        "-m",
        "gz.msgs.EntityWrench",
        "-p",
        f"entity: {{{entity}}}, wrench: {{force: {{z: {force}}}}}",
    )


def weld_block(node: Repro, mass: float) -> float:
    """
    Teleport `block_0` into the jaws and weld it; return the hang length.

    The weld is `gazebo_grasp_weld.py`'s own `DetachableJoint` add, which welds
    at the relative pose of the moment -- so the block is placed first and the
    length is measured after, since it falls freely in between.
    """
    # Along the rotator axis and squared to it: straight down in world with yaw 0
    # put the block off the rotator axis and into the jaws (-218 N m held at rest).
    tool, orientation = tool_pose()
    down = rotate(
        orientation, [0.0, 0.0, PLANT_HANG_M]
    )  # the link's +z points down the rotator axis
    print(f"link +z in world {[round(v / PLANT_HANG_M, 3) for v in down]}")
    target = [t + d for t, d in zip(tool, down)]
    crane_id = gz_poses()["crane_model"]["id"]
    gz_service(
        f"/world/{WORLD}/entity/system/add",
        "gz.msgs.EntityPlugin_V",
        f'entity: {{name: "{WORLD}", type: WORLD}}, plugins: {{name: '
        '"gz::sim::systems::ApplyLinkWrench", filename: "gz-sim-apply-link-wrench-system"}',
    )
    pause(True)  # else the block free-falls ~1 m before the weld lands
    gz_service(
        f"/world/{WORLD}/set_pose",
        "gz.msgs.Pose",
        f'name: "{BLOCK}", position: {{x: {target[0]}, y: {target[1]}, z: {target[2]}}}, '
        "orientation: {" + ", ".join(f"{k}: {orientation[k]}" for k in "xyzw") + "}",
    )
    plugin = (
        f"<parent_link>moved_with_q8</parent_link><child_model>{BLOCK}</child_model>"
        "<child_link>link</child_link>"
        f"<attach_topic>/cbs_sim/{BLOCK}/attach</attach_topic>"
        f"<detach_topic>/cbs_sim/{BLOCK}/detach</detach_topic>"
        f"<output_topic>/cbs_sim/{BLOCK}/state</output_topic>"
    )
    reply = gz_service(
        f"/world/{WORLD}/entity/system/add",
        "gz.msgs.EntityPlugin_V",
        f"entity: {{id: {crane_id}}}, plugins: {{name: "
        '"gz::sim::systems::DetachableJoint", filename: '
        f'"gz-sim-detachable-joint-system", innerxml: {json.dumps(plugin)}}}',
    )
    print(f"weld reply {reply!r}")
    # A weld in the air is a step load: boom and arm sag 0.05/0.08 rad and the
    # undamped load swings +-0.4 rad for good. Hand the weight over in 10 s instead.
    weight = mass * 9.81
    push(weight)
    node.spin(1.0)
    pause(False)
    for _ in range(20):
        node.spin(0.5)
        push(-weight / 20)
    push(None)  # net zero by now; drops the 21 entries
    node.spin(2.0)
    poses = gz_poses()
    block = [float(poses[BLOCK].get("position", {}).get(key, 0.0)) for key in "xyz"]
    return tool[2] - block[2]


def record(out: str) -> subprocess.Popen:
    return subprocess.Popen(
        ["ros2", "bag", "record", "-s", "mcap", "-o", out, *TOPICS],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> int:
    out = os.environ.get("OUT", "/tmp/304-repro")
    mass = float(os.environ.get("PAYLOAD_MASS", "880.0"))
    watch = float(os.environ.get("WATCH_S", "22.0"))

    rclpy.init()
    node = Repro()
    node.measured()

    node.activate_jtc()
    model = CraneModel(_description(), Tool("pzs100"))
    if not node.park(START, 25.0):
        return 1
    hang = weld_block(node, mass)
    print(f"welded {BLOCK} at a hang of {hang:.3f} m (asked {PLANT_HANG_M})")
    if os.environ.get("CTRL") != "pid":  # the MPC's service
        node.set_payload(mass)
    node.settle(60.0)
    row = node.state.name.index("theta8_rotator_joint")
    print(
        f"rotator effort at rest {node.state.effort[row]:.0f}"
        if node.state.effort
        else ""
    )

    goal = [GOAL[name] for name in canonical_joints()]
    y_n = model.forward_kinematics(goal, Frame.MOUNTING_BASE, Frame.TIP).position_m
    tcp = model.forward_kinematics(goal, Frame.MOUNTING_BASE, Frame.TCP).as_matrix()
    yaw = math.atan2(
        tcp[1][1], tcp[0][1]
    )  # `phi_tool_n` is TCP y's heading; 0 turns the rotator 1.5 rad
    print(
        f"y_n (tip K5 in K0) {y_n.round(4)} reach {math.hypot(*y_n[:2]):.3f} m yaw {yaw:.4f}"
    )

    bag = record(f"{out}/slew")
    node.spin(3.0)
    for _ in range(
        8
    ):  # the loaded rotator chatters +-0.04 rad/s; the planner reads one sample
        ok = node.plan(y_n, yaw, mass)
        if ok:
            break
        node.spin(0.5)
    node.spin(watch)
    os.killpg(bag.pid, signal.SIGINT)
    bag.wait(timeout=60.0)
    print(f"bag written to {out}/slew")
    rclpy.shutdown()
    return 0 if ok else 1


def _description() -> str:
    """Return the description the solver was baked from -- one file, one compiled solver."""
    from crane_mpc import problem

    return problem.default_description().read_text()


if __name__ == "__main__":
    sys.exit(main())
