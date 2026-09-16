#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELTEC_WS="/ds1/workspace/wheeltec_4wd"
CHECKPOINT="${1:-${PROJECT_DIR}/results/checkpoints/gnn_n10.pt}"

source /opt/ros/jazzy/setup.bash
source "${WHEELTEC_WS}/install/setup.bash"
source "${PROJECT_DIR}/ros2_ws_merged/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${PROJECT_DIR}/ros2_ws_merged/install:${AMENT_PREFIX_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}/ros2_ws_merged/install/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PATH="${PROJECT_DIR}/ros2_ws_merged/install/lib/multi_robot_gnn_ros:${PATH}"

ros2 launch wheeltec_4wd_bringup four_wheel_sim.launch.py use_rviz:=true &
SIM_PID=$!
trap 'kill "${SIM_PID}" 2>/dev/null || true' EXIT INT TERM
sleep 4
ros2 run multi_robot_gnn_ros gnn_controller --ros-args -p checkpoint:="${CHECKPOINT}" -p goal_x:=2.0 -p goal_y:=2.0
