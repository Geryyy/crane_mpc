#!/bin/bash
source /opt/ros/${ROS_DISTRO}/setup.bash
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$WS/install/setup.bash"
exec ros2 launch concrete_block_behavior_tree gazebo_wall_assembly_pzs100.launch.py \
  controller:=${CTRL:-cbs_mpc_active} planner:=cbs enable_livox_sim:=off initial_pose:=2 \
  seed_file:=$WS/src/concrete_block_stack/concrete_block_world_model/config/world_model_seed_pick_place.yaml \
  gui:=False
