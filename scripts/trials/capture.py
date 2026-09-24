#!/usr/bin/env python3
"""Record 155 criteria 4/5/6 in one pass. Joints mapped by name, never by index."""

import json
import sys
import time

import rclpy
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
        self.create_subscription(
            SolverHealth, "/crane/mpc/solver_health", self.on_h, 50
        )
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.create_subscription(JointTrajectory, "/crane/mpc/horizon", self.on_hz, 50)
        self.create_subscription(JointTrajectory, "/crane/reference", self.on_ref, 10)
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
            refmsg=n.ref,
        ),
        open(out, "w"),
    )
    print(
        f"health={len(n.health)} js={len(n.js)} hz={len(n.hz)} "
        f"ref={n.ref_stamps} path={n.path_stamps}"
    )
    rclpy.shutdown()


main()
