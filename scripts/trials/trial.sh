#!/bin/bash
# One trial: clean graph, launch headless, activate a2b, lift move, capture.
# Usage: trial.sh <outdir>
set +u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$S/../../../../.." && pwd)"
OUT=$1; mkdir -p "$OUT"

# Isolated graph (own domain, Gazebo partition, lo-only DDS) and one sim at a time.
# Only what this script starts is torn down -- never by name, another session may run a sim.
source "$WS/ralph/sim_env.sh"
exec 9>/tmp/ralph-sim.lock; flock 9

setsid nohup "$S/launch.sh" > "$OUT/launch.log" 2>&1 < /dev/null &
LPID=$!
trap 'kill -TERM -- -$LPID 2>/dev/null; sleep 5; kill -9 -- -$LPID 2>/dev/null' EXIT
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
