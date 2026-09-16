#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELTEC_WS="/ds1/workspace/wheeltec_4wd"
ROS_WS="${PROJECT_DIR}/ros2_ws_merged"
ROBOT_COUNT="${1:-3}"
CHECKPOINT="${2:-${PROJECT_DIR}/results/checkpoints/gnn_n10.npz}"
USE_RVIZ="${3:-true}"

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/jazzy/setup.bash
source "${WHEELTEC_WS}/install/setup.bash"
source "${ROS_WS}/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${ROS_WS}/install:${AMENT_PREFIX_PATH:-}"
export PYTHONPATH="${ROS_WS}/install/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PATH="${ROS_WS}/install/lib/multi_robot_gnn_ros:${PATH}"

exec ros2 launch multi_robot_gnn_ros multi_robot_sim.launch.py \
  robot_count:="${ROBOT_COUNT}" \
  checkpoint:="${CHECKPOINT}" \
  use_rviz:="${USE_RVIZ}"
