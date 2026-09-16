"""RViz2 visualization for the namespaced multi-robot experiment."""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / 'core' / 'environment.py').exists()
)
sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from core.environment import COMM_RADIUS


class MultiRobotMarkers(Node):
    def __init__(self) -> None:
        super().__init__('multi_robot_markers')
        self.declare_parameter('robot_count', 5)
        self.count = int(self.get_parameter('robot_count').value)
        self.positions = np.zeros((self.count, 2), dtype=np.float32)
        self.received = np.zeros(self.count, dtype=bool)
        angles = np.linspace(0, 2 * np.pi, self.count, endpoint=False)
        self.goals = np.column_stack((5.0 * np.cos(angles + np.pi), 4.0 * np.sin(angles + np.pi))).astype(np.float32)
        self.trails = [[] for _ in range(self.count)]
        self.publisher = self.create_publisher(MarkerArray, '/multi_robot/markers', 10)
        self.odom_subscriptions = [
            self.create_subscription(
                Odometry, f'/robot_{i}/odom',
                lambda msg, j=i: self.update(j, msg), 10
            )
            for i in range(self.count)
        ]
        self.timer = self.create_timer(0.1, self.publish_markers)

    def update(self, index: int, message: Odometry) -> None:
        point = [message.pose.pose.position.x, message.pose.pose.position.y]
        self.positions[index] = point
        self.received[index] = True
        self.trails[index].append(point)
        if len(self.trails[index]) > 100:
            self.trails[index].pop(0)

    def marker(self, marker_id: int, marker_type: int, scale: tuple[float, float, float], color: tuple[float, float, float, float]) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'multi_robot_gnn'
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def publish_markers(self) -> None:
        output = MarkerArray()
        for index in range(self.count):
            if not self.received[index]:
                continue
            robot = self.marker(index, Marker.SPHERE, (0.55, 0.55, 0.18), (0.1, 0.7, 0.9, 0.9))
            robot.pose.position.x, robot.pose.position.y = map(float, self.positions[index])
            output.markers.append(robot)
            goal = self.marker(100 + index, Marker.CYLINDER, (0.4, 0.4, 0.05), (0.3, 0.9, 0.4, 0.8))
            goal.pose.position.x, goal.pose.position.y = map(float, self.goals[index])
            output.markers.append(goal)
            trail = self.marker(200 + index, Marker.LINE_STRIP, (0.035, 0.0, 0.0), (0.25, 0.65, 0.8, 0.65))
            trail.points = []
            for x, y in self.trails[index]:
                point = Point()
                point.x, point.y = float(x), float(y)
                trail.points.append(point)
            output.markers.append(trail)
        edges = self.marker(1000, Marker.LINE_LIST, (0.025, 0.0, 0.0), (0.8, 0.8, 0.2, 0.7))
        for i in range(self.count):
            for j in range(i + 1, self.count):
                if self.received[i] and self.received[j] and np.linalg.norm(self.positions[i] - self.positions[j]) <= COMM_RADIUS:
                    for index in (i, j):
                        point = Point()
                        point.x, point.y = map(float, self.positions[index])
                        edges.points.append(point)
        output.markers.append(edges)
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiRobotMarkers()
    try:
        rclpy.spin(node)
    except (rclpy.executors.ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
