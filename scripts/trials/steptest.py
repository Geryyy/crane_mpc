#!/usr/bin/env python3
"""
Open-loop step on one axis at a time, MPC out of the loop.

Holds every position where it was found and puts a constant feed-forward
velocity in `effort`, which is the field the MPC's `u` travels in. Records the
whole joint state so the same step can be replayed against the OCP's own model
and the two compared.
"""

import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

CANON = [
    "theta1_slewing_joint",
    "theta2_boom_joint",
    "theta3_arm_joint",
    "q4_big_telescope",
    "theta6_tip_joint",
    "theta7_tilt_joint",
    "theta8_rotator_joint",
    "q9_left_rail_joint",
]
# planned five -> canonical row, and a step well inside each u_max
STEPS = [(0, 0.30), (1, 0.10), (2, 0.12), (3, 0.16), (6, 0.80)]
HOLD, SETTLE, TS = 0.6, 2.4, 0.06


class Step(Node):
    def __init__(self):
        super().__init__(
            "steptest",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.q = self.dq = self.names = None
        self.trace = []
        self.recording = False
        self.create_subscription(JointState, "/joint_states", self.on_js, 100)
        self.pub = self.create_publisher(
            JointTrajectory,
            "/crane/mpc/horizon",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

    def on_js(self, m):
        self.names, self.q, self.dq = list(m.name), list(m.position), list(m.velocity)
        if self.recording:
            self.trace.append(
                dict(
                    sim=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                    names=list(m.name),
                    pos=list(m.position),
                    vel=list(m.velocity),
                    eff=list(m.effort),
                )
            )

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def state(self):
        return {
            j: (self.q[self.names.index(j)], self.dq[self.names.index(j)])
            for j in CANON
            if j in self.names
        }


def trajectory(hold, row, amplitude):
    msg = JointTrajectory()
    msg.joint_names = CANON
    n = int((HOLD + SETTLE) / TS)
    for k in range(n + 1):
        t = k * TS
        p = JointTrajectoryPoint()
        p.positions = [hold.get(j, (0.0, 0.0))[0] for j in CANON]
        p.velocities = [0.0] * 8
        p.effort = [0.0] * 8
        if t < HOLD:
            p.effort[row] = amplitude
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        msg.points.append(p)
    return msg


def main():
    rclpy.init()
    n = Step()
    while n.q is None and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.1)
    runs = []
    for row, amplitude in STEPS:
        n.spin(2.0)
        hold = n.state()
        msg = trajectory(hold, row, amplitude)
        msg.header.stamp = n.get_clock().now().to_msg()
        n.trace = []
        n.recording = True
        n.pub.publish(msg)
        n.spin((HOLD + SETTLE) / 0.5)  # sim runs about half real time
        n.recording = False
        runs.append(
            dict(
                row=row,
                amplitude=amplitude,
                joint=CANON[row],
                hold={k: list(v) for k, v in hold.items()},
                trace=n.trace,
            )
        )
        print(f"{CANON[row]:>22} step {amplitude:+.2f}: {len(n.trace)} samples")
    json.dump(runs, open(sys.argv[1], "w"))
    rclpy.shutdown()


main()
