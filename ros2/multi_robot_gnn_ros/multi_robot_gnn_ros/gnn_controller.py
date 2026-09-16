"""ROS 2 bridge for one Wheeltec namespace.

The node uses the same 8-feature observation contract as the offline model.
It is intentionally conservative: command output is clipped and stopped when
the LiDAR front sector reports a near obstacle. Multi-robot deployment can run
one namespaced instance per robot after the corresponding Gazebo namespaces
are available.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / 'core' / 'environment.py').exists()
)
sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import LaserScan

from core.environment import COMM_RADIUS, MAX_SPEED, observation, safety_filter


class GNNController(Node):
    def __init__(self) -> None:
        super().__init__('gnn_controller')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)
        self.declare_parameter('control_hz', 10.0)
        self.goal = np.array([float(self.get_parameter('goal_x').value), float(self.get_parameter('goal_y').value)], dtype=np.float32)
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.scan = None
        self.policy = self._load_policy(str(self.get_parameter('checkpoint').value))
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        period = 1.0 / max(float(self.get_parameter('control_hz').value), 1.0)
        self.timer = self.create_timer(period, self.control_callback)
        self.get_logger().info('GNN controller ready; waiting for odom/scan')

    def _load_policy(self, checkpoint: str):
        if not checkpoint:
            self.get_logger().warning('No checkpoint supplied; using goal-seeking fallback')
            return None
        from core.numpy_policy import load_numpy_policy
        if not checkpoint.endswith('.npz'):
            self.get_logger().error('ROS inference expects exported .npz; run scripts/export_numpy_checkpoint.py')
            return None
        return load_numpy_policy(checkpoint)

    def odom_callback(self, message: Odometry) -> None:
        self.position[:] = [message.pose.pose.position.x, message.pose.pose.position.y]
        self.velocity[:] = [message.twist.twist.linear.x, message.twist.twist.linear.y]

    def scan_callback(self, message: LaserScan) -> None:
        self.scan = np.asarray(message.ranges, dtype=np.float32)

    def control_callback(self) -> None:
        direction = self.goal - self.position
        distance = float(np.linalg.norm(direction))
        if distance < 0.25:
            self.publish_stop()
            return
        if self.policy is None:
            action = 1.0 * direction / max(distance, 1e-6)
        else:
            obstacles = np.asarray([[0.0, 0.0, 0.01, 0.01]], dtype=np.float32)
            features, adjacency = observation(self.position[None], self.velocity[None], self.goal[None], obstacles, COMM_RADIUS)
            action = self.policy(features, adjacency)[0] * MAX_SPEED
        if self.scan is not None and len(self.scan):
            center = len(self.scan) // 2
            front = np.concatenate([self.scan[: max(1, len(self.scan)//18)], self.scan[-max(1, len(self.scan)//18):]])
            finite_front = front[np.isfinite(front)]
            if len(finite_front) and float(np.min(finite_front)) < 0.45:
                action = np.zeros(2, dtype=np.float32)
        action = safety_filter(self.position[None], np.asarray(action, dtype=np.float32)[None])[0]
        command = Twist()
        command.linear.x = float(np.clip(action[0], -MAX_SPEED, MAX_SPEED))
        command.angular.z = float(np.clip(math.atan2(float(action[1]), float(action[0])), -1.2, 1.2))
        self.cmd_pub.publish(command)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GNNController()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
