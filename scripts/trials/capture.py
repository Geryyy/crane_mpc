#!/usr/bin/env python3
"""Record 155 criteria 4/5/6 in one pass. Joints mapped by name, never by index."""

import json
import sys
import time

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from crane_msgs.msg import SolverHealth
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    from crane_msgs.msg import JointPath
except ImportError:
    JointPath = None


def stamp_ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


class Cap(Node):
    def __init__(self):
        super().__init__("cap")
        self.health, self.js, self.hz = [], [], []
        self.ref_stamps, self.path_stamps = [], []
        self.ref = {}
        self.ctrl = []
        self.create_subscription(
            SolverHealth, "/crane/mpc/solver_health", self.on_h, 50
        )
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.create_subscription(JointTrajectory, "/crane/mpc/horizon", self.on_hz, 50)
        self.create_subscription(JointTrajectory, "/crane/reference", self.on_ref, 10)
        # What the JTC actually put on the command interface. Without this the
        # MPC's `u` and the machine's response are two ends of an unobserved
        # link, and a gain or sign error in between reads as a plant mismatch.
        for topic in (
            "/trajectory_controllers/controller_state",
            "/crane/controller_state",
            "/trajectory_controller_a2b/controller_state",
        ):
            self.create_subscription(
                JointTrajectoryControllerState,
                topic,
                lambda m, s=topic: self.on_ctrl(m, s),
                50,
            )
        if JointPath is not None:
            self.create_subscription(JointPath, "/crane/joint_path", self.on_path, 10)

    def on_h(self, m):
        self.health.append(
            dict(
                t=time.time(),
                solve=m.solve_time,
                budget=m.solve_budget,
                outcome=int(m.outcome),
                prev=bool(m.applied_previous_solution),
                status=int(m.status),
                status_word=m.status_word,
                iters=int(m.iterations),
                qp_status=int(m.qp_status),
                qp_iters=int(m.qp_iterations),
                viol=list(m.constraint_violation),
                slack=bool(m.used_slack),
                penalty=float(m.slack_penalty),
                cost=list(m.cost_term),
                msg=m.message,
            )
        )

    def on_js(self, m):
        self.js.append(
            dict(
                t=time.time(),
                sim=stamp_ns(m.header) * 1e-9,
                names=list(m.name),
                pos=list(m.position),
                vel=list(m.velocity),
            )
        )

    def on_hz(self, m):
        if not m.points:
            return
        p = m.points[0]
        self.hz.append(
            dict(
                t=time.time(),
                names=list(m.joint_names),
                pos=list(p.positions),
                vel=list(p.velocities),
                eff=list(p.effort),
            )
        )

    def on_ctrl(self, m, source):
        if not m.joint_names:
            return
        self.ctrl.append(
            dict(
                t=time.time(),
                src=source,
                names=list(m.joint_names),
                out=list(m.output.velocities),
                ref_vel=list(m.reference.velocities),
                ref_pos=list(m.reference.positions),
                fb_vel=list(m.feedback.velocities),
                err_pos=list(m.error.positions),
            )
        )

    def on_ref(self, m):
        self.ref_stamps.append(stamp_ns(m.header))
        self.ref = dict(
            names=list(m.joint_names),
            n=len(m.points),
            dur=(
                m.points[-1].time_from_start.sec
                + m.points[-1].time_from_start.nanosec * 1e-9
            )
            if m.points
            else 0.0,
            first=list(m.points[0].positions) if m.points else [],
            last=list(m.points[-1].positions) if m.points else [],
        )

    def on_path(self, m):
        self.path_stamps.append(stamp_ns(m.header))


def main():
    dur, out = float(sys.argv[1]), sys.argv[2]
    rclpy.init()
    n = Cap()
    end = time.time() + dur
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.1)
    json.dump(
        dict(
            health=n.health,
            js=n.js,
            hz=n.hz,
            ref=n.ref_stamps,
            path=n.path_stamps,
            ctrl=n.ctrl,
            refmsg=n.ref,
        ),
        open(out, "w"),
    )
    print(
        f"health={len(n.health)} js={len(n.js)} hz={len(n.hz)} ctrl={len(n.ctrl)} "
        f"ref={n.ref_stamps} path={n.path_stamps}"
    )
    rclpy.shutdown()


main()
