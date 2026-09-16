#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="${PROJECT_DIR}/ros2_ws_merged"
mkdir -p "${ROS_WS}/src"
ln -sfn "${PROJECT_DIR}/ros2/multi_robot_gnn_ros" "${ROS_WS}/src/multi_robot_gnn_ros"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/jazzy/setup.bash
set -u
cd "${ROS_WS}"
# A regular install is intentional here: the active setuptools version can
# omit ament's package-index marker during editable (symlink) installation.
colcon build --merge-install --packages-select multi_robot_gnn_ros
