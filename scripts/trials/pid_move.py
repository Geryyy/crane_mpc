#!/usr/bin/env python3
"""Plan through /a2b_movement, then execute it on the JTC action (no MPC)."""

import sys

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node

from timber_crane_planning_interfaces.srv import CalcMovement


def main():
    rclpy.init()
    n = Node("pid_move")
    cli = n.create_client(CalcMovement, "/a2b_movement")
    cli.wait_for_service(timeout_sec=30.0)
    req = CalcMovement.Request()
    req.y_n = Point(x=4.13, y=4.127, z=2.0)
    req.phi_tool_n = 0.0
    req.t_end = 0.0
    req.slow_down = 1.0
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=90.0)
    res = fut.result()
    print(f"plan success={res.success} points={len(res.trajectory.points)}")
    if not res.success:
        return 1
    ac = ActionClient(
        n, FollowJointTrajectory, "/trajectory_controller_a2b/follow_joint_trajectory"
    )
    ac.wait_for_server(timeout_sec=30.0)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = res.trajectory
    gf = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, gf, timeout_sec=30.0)
    handle = gf.result()
    print(f"goal accepted={handle.accepted}")
    rf = handle.get_result_async()
    rclpy.spin_until_future_complete(n, rf, timeout_sec=120.0)
    r = rf.result()
    print(f"result error_code={r.result.error_code} {r.result.error_string!r}")
    rclpy.shutdown()
    return 0


sys.exit(main())
