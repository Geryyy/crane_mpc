"""
Start the horizon producer, in shadow, with both of its configuration files.

Node-only: a component a profile includes, not a bringup. The including
profile supplies `/robot_description`, `/joint_states` and
`/crane/reference`; missing any of them the node publishes no horizon and
says why on `/crane/mpc/solver_health`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    controller_state_topic = LaunchConfiguration("controller_state_topic")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    mode = LaunchConfiguration("mode")
    start_signal_action = LaunchConfiguration("start_signal_action")
    use_sim_time = LaunchConfiguration("use_sim_time")
    share = FindPackageShare("crane_mpc")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            # Shadow by default; leaving it is deliberate. Passed as a node
            # parameter so it wins over both files.
            DeclareLaunchArgument(
                "mode",
                default_value="shadow",
                choices=["shadow", "active"],
                description=(
                    "Whether this node's horizon drives. 'shadow' publishes "
                    "~/shadow_horizon and leaves /crane/mpc/horizon silent; "
                    "'active' publishes the contract topic."
                ),
            ),
            # Empty is no gate, and no gate is what a deployment runs: the
            # behaviour tree plans when it wants motion, so the reference is
            # the go signal (the second RViz start button).
            DeclareLaunchArgument(
                "start_signal_action",
                default_value="",
                description=(
                    "FollowJointTrajectory action whose accepted goal releases "
                    "this node to drive. Empty (the default) drives as soon as "
                    "a reference arrives."
                ),
            ),
            # Absolute: a relative default stops being a no-op once included
            # under a pushed namespace, since remapping wouldn't move with it.
            DeclareLaunchArgument(
                "joint_states_topic",
                default_value="/joint_states",
                description=(
                    "Measured joint state to read. Must be the same corrected "
                    "topic every node that maps a joint state onto the "
                    "description reads."
                ),
            ),
            # Same reason as `joint_states_topic`; timber-bringup profiles
            # remap the JTC's state here. Wrong here is silent:
            # `follower.velocity_source: none`.
            DeclareLaunchArgument(
                "controller_state_topic",
                default_value="/crane/controller_state",
                description=(
                    "Trajectory-following controller's state, the velocity a "
                    "shadow command is judged against."
                ),
            ),
            Node(
                package="crane_mpc",
                executable="crane_mpc_node",
                # Key both config files are written under; renaming or
                # namespacing the node silently applies neither.
                name="crane_mpc",
                # Last file to declare a key wins, so machine limits go
                # second (disjoint, kept so by `test_launch_contract.py`).
                # Neither is optional: `hydraulics` defaults equal the
                # second file's values, so a forgotten file looks identical.
                parameters=[
                    PathJoinSubstitution([share, "config", "crane_mpc.yaml"]),
                    PathJoinSubstitution([share, "config", "hydraulic_limits.yaml"]),
                    {
                        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                        "mode": ParameterValue(mode, value_type=str),
                        "start_signal_action": ParameterValue(
                            start_signal_action, value_type=str
                        ),
                    },
                ],
                remappings=[
                    ("/joint_states", joint_states_topic),
                    ("/crane/controller_state", controller_state_topic),
                ],
                # acados parallelises shooting nodes via OpenMP; eight threads
                # cost 1.7x the median of one, p95 27.6 ms vs 30 ms budget.
                # `passive` stops idle threads spinning against gazebo.
                additional_env={
                    "OMP_NUM_THREADS": "1",
                    "OMP_WAIT_POLICY": "passive",
                },
                output="screen",
            ),
        ]
    )
