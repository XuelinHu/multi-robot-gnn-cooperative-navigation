"""Multi-robot ROS 2 controller using the shared graph policy."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'core' / 'environment.py').exists())
sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from core.environment import COMM_RADIUS, MAX_SPEED, observation, safety_filter
from core.numpy_policy import load_numpy_policy


class MultiGNNController(Node):
    def __init__(self) -> None:
        super().__init__('multi_gnn_controller')
        self.declare_parameter('robot_count', 5)
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('control_hz', 10.0)
        self.count = int(self.get_parameter('robot_count').value)
        self.positions = np.zeros((self.count, 2), dtype=np.float32)
        self.velocities = np.zeros((self.count, 2), dtype=np.float32)
        self.received = np.zeros(self.count, dtype=bool)
        angles = np.linspace(0, 2 * np.pi, self.count, endpoint=False)
        self.goals = np.column_stack((5.0 * np.cos(angles + np.pi), 4.0 * np.sin(angles + np.pi))).astype(np.float32)
        checkpoint = str(self.get_parameter('checkpoint').value)
        self.policy = load_numpy_policy(checkpoint) if checkpoint else None
        self.odom_subscriptions = []
        self.cmd_publishers = []
        for index in range(self.count):
            self.odom_subscriptions.append(self.create_subscription(
                Odometry, f'/robot_{index}/odom', lambda message, i=index: self.odom_callback(i, message), 10))
            self.cmd_publishers.append(self.create_publisher(Twist, f'/robot_{index}/cmd_vel', 10))
        self.timer = self.create_timer(1.0 / max(float(self.get_parameter('control_hz').value), 1.0), self.control_callback)
        self.get_logger().info(f'multi GNN controller ready for {self.count} robots')

    def odom_callback(self, index: int, message: Odometry) -> None:
        self.positions[index] = [message.pose.pose.position.x, message.pose.pose.position.y]
        self.velocities[index] = [message.twist.twist.linear.x, message.twist.twist.linear.y]
        self.received[index] = True

    def control_callback(self) -> None:
        if not self.received.all():
            return
        if self.policy is None:
            action = self.goals - self.positions
            action /= np.maximum(np.linalg.norm(action, axis=1, keepdims=True), 1e-6)
        else:
            features, adjacency = observation(self.positions, self.velocities, self.goals, np.zeros((0, 4), dtype=np.float32), COMM_RADIUS)
            action = self.policy(features, adjacency) * MAX_SPEED
        action = safety_filter(self.positions, action)
        for index, publisher in enumerate(self.cmd_publishers):
            command = Twist()
            command.linear.x = float(np.clip(action[index, 0], -MAX_SPEED, MAX_SPEED))
            command.angular.z = float(np.clip(math.atan2(float(action[index, 1]), float(action[index, 0])), -1.2, 1.2))
            publisher.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiGNNController()
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
