#!/usr/bin/env python3
"""
Record 155 criteria 4/5/6 in one pass. Joints mapped by name, never by index.

    capture.py <seconds> <out.json> [--sim-time|--no-sim-time] [--mpc-node NAME]

`rx` is only comparable with a header stamp on the same clock, so the clock has
to match the rig: sim stamps are sim time, the machine's are wall time. Neither
is assumed -- with no flag the `use_sim_time` parameter is read as given
(`--ros-args -p use_sim_time:=true`, or the default `false`), and the resolved
value is printed. `--sim-time` is the Gazebo arm of issue 189; the lab day
(issue 212) records on the machine and leaves it false.
"""

import argparse
import json
import time

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from crane_msgs.msg import SolverHealth
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    from crane_msgs.msg import JointPath
except ImportError:
    JointPath = None


def stamp_ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def sim_time_override(sim_time):
    """`None` reads the parameter as given; a bool pins it."""
    return [] if sim_time is None else [Parameter("use_sim_time", value=bool(sim_time))]


class Cap(Node):
    def __init__(self, sim_time=None, mpc_node="crane_mpc"):
        super().__init__("cap", parameter_overrides=sim_time_override(sim_time))
        self.health, self.js, self.hz = [], [], []
        self.shz, self.cmp = [], []
        self.ref_stamps, self.path_stamps = [], []
        self.ref = {}
        self.ctrl = []
        self.create_subscription(
            SolverHealth, "/crane/mpc/solver_health", self.on_h, 50
        )
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.create_subscription(JointTrajectory, "/crane/mpc/horizon", self.on_hz, 50)
        # Shadow mode leaves the contract topic silent, so a shadow run carries
        # nothing above. `~/` resolves against the *node* name, not the topic
        # namespace: `name="crane_mpc"` in the launch file makes these
        # `/crane_mpc/shadow_*` and not `/crane/mpc/shadow_*`.
        self.create_subscription(
            JointTrajectory, f"/{mpc_node}/shadow_horizon", self.on_shz, 50
        )
        self.create_subscription(
            DiagnosticArray, f"/{mpc_node}/shadow_comparison", self.on_cmp, 50
        )
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

    def rx(self):
        """Receipt on the stamps' own clock. `sim - rx` is the delivery latency."""
        return self.get_clock().now().nanoseconds * 1e-9

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
                rx=self.rx(),
                sim=stamp_ns(m.header) * 1e-9,
                names=list(m.name),
                pos=list(m.position),
                vel=list(m.velocity),
            )
        )

    def on_hz(self, m):
        row = self.knot_zero(m)
        if row is not None:
            self.hz.append(row)

    def on_shz(self, m):
        row = self.knot_zero(m)
        if row is not None:
            self.shz.append(row)

    def on_cmp(self, m):
        """`shadow_comparison` is a DiagnosticArray; keep the key/value pairs flat."""
        for status in m.status:
            self.cmp.append(
                dict(
                    t=time.time(),
                    rx=self.rx(),
                    sim=stamp_ns(m.header) * 1e-9,
                    level=int(status.level),
                    message=status.message,
                    values={kv.key: kv.value for kv in status.values},
                )
            )

    def knot_zero(self, m):
        if not m.points:
            return None
        p = m.points[0]
        return dict(
            t=time.time(),
            rx=self.rx(),
            # `Timing.stamp_ns`: the measurement instant knot 0 counts from.
            # Latency and cadence phase are both measured off this, so a
            # capture without it cannot answer issue 189 at all.
            sim=stamp_ns(m.header) * 1e-9,
            names=list(m.joint_names),
            pos=list(p.positions),
            vel=list(p.velocities),
            eff=list(p.effort),
        )

    def on_ctrl(self, m, source):
        if not m.joint_names:
            return
        self.ctrl.append(
            dict(
                t=time.time(),
                rx=self.rx(),
                sim=stamp_ns(m.header) * 1e-9,
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


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("seconds", type=float)
    parser.add_argument("out", help="json for looptiming.py / modelfit.py / growth.py")
    clock = parser.add_mutually_exclusive_group()
    clock.add_argument("--sim-time", dest="sim_time", action="store_true", default=None)
    clock.add_argument("--no-sim-time", dest="sim_time", action="store_false")
    parser.add_argument(
        "--mpc-node",
        default="crane_mpc",
        help="node name the `~/shadow_*` topics resolve against",
    )
    return parser.parse_known_args(argv)[0]


def main(argv=None):
    options = parse(argv)
    rclpy.init()
    n = Cap(options.sim_time, options.mpc_node)
    clock = bool(n.get_parameter("use_sim_time").value)
    print(
        f"use_sim_time={clock}" + ("" if options.sim_time is not None else " (as set)")
    )
    end = time.time() + options.seconds
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.1)
    json.dump(
        dict(
            health=n.health,
            js=n.js,
            hz=n.hz,
            shz=n.shz,
            cmp=n.cmp,
            ref=n.ref_stamps,
            path=n.path_stamps,
            ctrl=n.ctrl,
            refmsg=n.ref,
        ),
        open(options.out, "w"),
    )
    print(
        f"health={len(n.health)} js={len(n.js)} hz={len(n.hz)} shz={len(n.shz)} "
        f"cmp={len(n.cmp)} ctrl={len(n.ctrl)} ref={n.ref_stamps} path={n.path_stamps}"
    )
    # Under `use_sim_time` and no /clock: `rx` is 0 and no latency can be
    # measured. Say it now -- discovering it after the sim is torn down costs
    # the run. On wall time `rx` is never 0, so the check only applies in sim.
    horizons = n.hz or n.shz
    if clock and horizons and not any(r["rx"] for r in horizons):
        print("WARNING: no /clock, so rx=0 -- looptiming.py cannot measure latency")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
