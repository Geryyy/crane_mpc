#!/bin/bash
# One trial: clean graph, launch headless, activate a2b, lift move, capture.
# Usage: trial.sh <outdir>
set +u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$S/../../../../.." && pwd)"
OUT=$1; mkdir -p "$OUT"

# Tear down whatever is running, by PID, never pkill -f (it kills this shell).
for p in $(ps -eo pid,cmd | grep -E "gz sim|ros2 launch|crane_planner_node|crane_mpc_node|spawner|robot_state_publisher|bt_action_server|lifecycle_manager|grip_traj_server|world_model_node|wall_plan_server|collision_body_handler|parameter_bridge|static_transform|best_accessability|unloading_loadbed|bundle_envelope|gripper_grasp" | grep -v grep | awk '{print $1}'); do
  [ "$p" = "$$" ] && continue
  kill "$p" 2>/dev/null
done
sleep 5
for p in $(ps -eo pid,cmd | grep -E "gz sim" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 2

# The sim wants concrete_block_behavior_tree's cyclonedds_sim.xml, and its
# env hook only sets the var if unset. A shell carrying the machine-local
# hardware file silently keeps it, multicast off.
unset CYCLONEDDS_URI
source /opt/ros/${ROS_DISTRO}/setup.bash
source "$WS/install/setup.bash"
ros2 daemon stop >/dev/null 2>&1; ros2 daemon start >/dev/null 2>&1

setsid nohup "$S/launch.sh" > "$OUT/launch.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do
  grep -q "crane_mpc configured on pzs100" "$OUT/launch.log" 2>/dev/null \
    && grep -q "planning for pzs100" "$OUT/launch.log" 2>/dev/null && break
  sleep 2
done
grep -q "planning for pzs100" "$OUT/launch.log" || { echo "SIM DID NOT COME UP"; exit 1; }
sleep 8   # let the passive pair settle

timeout 20 ros2 control switch_controllers --activate trajectory_controller_a2b 2>&1 | tail -1
sleep 2
python3 "${CAP:-$S/capture.py}" 45 "$OUT/cap.json" > "$OUT/cap.log" 2>&1 &
CAP=$!
sleep 2
if [ -n "${MOVE:-}" ]; then
  timeout 180 python3 "$MOVE"
else
  timeout 90 ros2 service call /a2b_movement timber_crane_planning_interfaces/srv/CalcMovement \
    "{y_n: {x: 4.13, y: 4.127, z: 2.0}, phi_tool_n: 0.0, t_end: 0.0, carries_log: false, slow_down: 1.0}" \
    2>&1 | grep -oE "success=(True|False)" | head -1
fi
wait $CAP
cat "$OUT/cap.log"
