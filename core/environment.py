"""Small deterministic 2-D multi-robot benchmark used by every baseline."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np


WORLD_W = 16.0
WORLD_H = 12.0
ROBOT_RADIUS = 0.28
SAFE_DISTANCE = 0.72
COMM_RADIUS = 3.4
MAX_SPEED = 1.4
DT = 0.08


@dataclass
class Scenario:
    starts: np.ndarray
    goals: np.ndarray
    obstacles: np.ndarray


def default_obstacles() -> np.ndarray:
    return np.asarray([
        [-6.4, 2.7, 1.8, 0.28],
        [6.0, -2.0, 1.8, 0.28],
        [-1.0, 4.0, 2.6, 0.34],
        [1.3, -4.0, 2.7, 0.34],
        [0.0, 0.6, 0.9, 1.7],
    ], dtype=np.float32)


def obstacles_for_layout(layout: str) -> np.ndarray:
    """Return deterministic map geometry for the named benchmark layout."""
    if layout == "open":
        return np.zeros((0, 4), dtype=np.float32)
    if layout == "narrow":
        # Two wall sections leave a central passage and force interaction
        # between robots travelling in opposite directions.
        return np.asarray([
            [-2.6, 0.0, 5.2, 0.34],
            [2.6, 0.0, 5.2, 0.34],
        ], dtype=np.float32)
    return default_obstacles()


def make_scenario(n: int, seed: int = 0, layout: str = "crossing") -> Scenario:
    rng = np.random.default_rng(seed)
    obstacles = obstacles_for_layout(layout)
    phase = float(rng.uniform(0.0, 2 * np.pi))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    radius_x = float(rng.uniform(4.8, 5.3))
    radius_y = float(rng.uniform(3.7, 4.2))
    starts = np.column_stack((radius_x * np.cos(angles), radius_y * np.sin(angles))).astype(np.float32)
    if layout == "random":
        selected: list[np.ndarray] = []
        for _ in range(max(500, n * 100)):
            candidate = rng.uniform([-6.8, -4.8], [6.8, 4.8]).astype(np.float32)
            if any(point_in_obstacle(candidate, obstacle, ROBOT_RADIUS + 0.15) for obstacle in obstacles):
                continue
            if any(float(np.linalg.norm(candidate - other)) < 1.0 for other in selected):
                continue
            selected.append(candidate)
            if len(selected) == n:
                break
        if len(selected) != n:
            raise RuntimeError(f"could not place {n} robots in random layout with clearance")
        starts = np.asarray(selected, dtype=np.float32)
    goals = starts[::-1].copy()
    return Scenario(starts, goals, obstacles)


def point_in_obstacle(point: np.ndarray, obstacle: np.ndarray, margin: float = 0.0) -> bool:
    x, y = point
    ox, oy, ow, oh = obstacle
    return abs(x - ox) <= ow / 2 + margin and abs(y - oy) <= oh / 2 + margin


def build_graph(position: np.ndarray, radius: float = COMM_RADIUS) -> np.ndarray:
    delta = position[:, None, :] - position[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    adjacency = (distance <= radius) & (distance > 0)
    return adjacency.astype(np.float32)


def build_edge_features(position: np.ndarray, velocity: np.ndarray,
                        radius: float = COMM_RADIUS) -> np.ndarray:
    """Return directed relative edge features [dx, dy, dvx, dvy, closing]."""
    delta = position[:, None, :] - position[None, :, :]
    relative_velocity = velocity[:, None, :] - velocity[None, :, :]
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    unit = delta / np.maximum(distance, 1e-6)
    closing = -(relative_velocity * unit).sum(axis=-1, keepdims=True)
    edge = np.concatenate([delta / np.array([WORLD_W, WORLD_H]),
                           relative_velocity / MAX_SPEED,
                           closing / MAX_SPEED], axis=-1)
    edge *= build_graph(position, radius)[..., None]
    return edge.astype(np.float32)


def observation(position: np.ndarray, velocity: np.ndarray, goals: np.ndarray,
                obstacles: np.ndarray, radius: float = COMM_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    n = len(position)
    nearest = np.zeros((n, 2), dtype=np.float32)
    nearest_distance = np.full(n, 9.0, dtype=np.float32)
    for i, p in enumerate(position):
        for obstacle in obstacles:
            ox, oy, ow, oh = obstacle
            closest = np.array([np.clip(p[0], ox - ow / 2, ox + ow / 2),
                                np.clip(p[1], oy - oh / 2, oy + oh / 2)])
            vector = p - closest
            distance = float(np.linalg.norm(vector))
            if distance < nearest_distance[i]:
                nearest_distance[i] = distance
                nearest[i] = vector / max(distance, 1e-6)
    goal_delta = goals - position
    features = np.column_stack([
        goal_delta / np.array([WORLD_W, WORLD_H]),
        velocity / MAX_SPEED,
        nearest / 1.0,
        np.clip(nearest_distance / 4.0, 0.0, 1.0),
        np.ones(n, dtype=np.float32),
    ]).astype(np.float32)
    return features, build_graph(position, radius)


def clip_action(action: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(action, axis=-1, keepdims=True)
    return action * np.minimum(1.0, MAX_SPEED / np.maximum(norm, 1e-6))


def safety_filter(position: np.ndarray, action: np.ndarray,
                  safe_distance: float = SAFE_DISTANCE,
                  horizon: float = 0.35) -> np.ndarray:
    """Project pairwise closing velocities away from imminent collisions.

    This is a deterministic, differentiable-at-runtime safety wrapper rather
    than a replacement policy. It is shared by learned-policy evaluation and
    the ROS adapter, so the ablation can compare raw versus safety-filtered
    actions under identical trajectories.
    """
    filtered = np.asarray(action, dtype=np.float32).copy()
    for i in range(len(position)):
        for j in range(i + 1, len(position)):
            delta = position[i] - position[j]
            distance = float(np.linalg.norm(delta))
            if distance >= safe_distance + horizon:
                continue
            unit = delta / max(distance, 1e-6)
            relative = filtered[i] - filtered[j]
            closing = float(np.dot(relative, unit))
            if closing < 0.0:
                correction = (-closing + max(0.0, safe_distance + horizon - distance)) * 0.5
                filtered[i] += correction * unit
                filtered[j] -= correction * unit
    return clip_action(filtered)


def orca_style_action(position: np.ndarray, velocity: np.ndarray, goals: np.ndarray,
                      obstacles: np.ndarray, radius: float = COMM_RADIUS) -> np.ndarray:
    """Reciprocal-avoidance-style local controller used as a transparent baseline.

    This is intentionally a lightweight benchmark controller. It shares the
    same local information and action limits as the learned policies.
    """
    actions = np.zeros_like(position, dtype=np.float32)
    for i, p in enumerate(position):
        to_goal = goals[i] - p
        distance = np.linalg.norm(to_goal) or 1.0
        action = 1.15 * to_goal / distance
        for j, other in enumerate(position):
            if i == j:
                continue
            delta = p - other
            d = np.linalg.norm(delta) or 1e-3
            if d < 1.9:
                action += 1.15 * max(0.0, 1.9 - d) / 1.9 * delta / d
        for ox, oy, ow, oh in obstacles:
            closest = np.array([np.clip(p[0], ox - ow / 2, ox + ow / 2),
                                np.clip(p[1], oy - oh / 2, oy + oh / 2)])
            delta = p - closest
            d = np.linalg.norm(delta)
            if d < 1.6:
                action += 1.25 * (1.6 - d) / 1.6 * delta / max(d, 1e-3)
        actions[i] = action
    return clip_action(actions)


def rvo_style_action(position: np.ndarray, velocity: np.ndarray, goals: np.ndarray,
                     obstacles: np.ndarray, radius: float = COMM_RADIUS) -> np.ndarray:
    """Deterministic velocity-obstacle candidate controller.

    The implementation evaluates a finite velocity lattice against predicted
    pairwise clearances and inflated rectangular obstacles. It is an explicit
    RVO-style baseline, while remaining dependency-free and reproducible.
    """
    del radius  # The local candidate test uses the safety distance directly.
    directions = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    candidates = [np.zeros(2, dtype=np.float32)]
    for speed in (0.35, 0.70, 1.05, MAX_SPEED):
        candidates.extend(
            np.asarray([speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32)
            for angle in directions
        )
    actions = np.zeros_like(position, dtype=np.float32)
    horizon = 1.0
    for i, point in enumerate(position):
        delta = goals[i] - point
        distance = float(np.linalg.norm(delta))
        preferred = delta / max(distance, 1e-6) * min(MAX_SPEED, distance / DT)
        best_score = float("inf")
        best = preferred
        for candidate in candidates:
            score = 0.35 * float(np.linalg.norm(candidate - preferred))
            predicted = point + horizon * candidate
            for j, other in enumerate(position):
                if i == j:
                    continue
                other_predicted = other + horizon * velocity[j]
                clearance = float(np.linalg.norm(predicted - other_predicted))
                score += 120.0 * max(0.0, SAFE_DISTANCE - clearance) ** 2
            for obstacle in obstacles:
                if point_in_obstacle(predicted, obstacle, ROBOT_RADIUS + 0.08):
                    score += 80.0
            # Prefer goal progress when candidates have comparable clearance.
            score += 0.08 * float(np.linalg.norm(goals[i] - (point + 0.5 * candidate)))
            if score < best_score:
                best_score, best = score, candidate
        actions[i] = best
    return clip_action(actions)


def astar_path(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray,
               resolution: float = 0.4) -> np.ndarray:
    """Grid A* path with obstacle inflation; returns world-coordinate waypoints."""
    cols = int(WORLD_W / resolution)
    rows = int(WORLD_H / resolution)
    def to_grid(p: np.ndarray) -> tuple[int, int]:
        return int(np.clip((p[0] + WORLD_W / 2) / resolution, 0, cols - 1)), int(np.clip((p[1] + WORLD_H / 2) / resolution, 0, rows - 1))
    def to_world(cell: tuple[int, int]) -> np.ndarray:
        return np.array([(cell[0] + 0.5) * resolution - WORLD_W / 2,
                         (cell[1] + 0.5) * resolution - WORLD_H / 2], dtype=np.float32)
    source, target = to_grid(start), to_grid(goal)
    frontier = [(0.0, source)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
    cost = {source: 0.0}
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == target:
            break
        for dx, dy in moves:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < cols and 0 <= nxt[1] < rows):
                continue
            if any(point_in_obstacle(to_world(nxt), obstacle, ROBOT_RADIUS + 0.12) for obstacle in obstacles):
                continue
            new_cost = cost[current] + math.hypot(dx, dy)
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + np.linalg.norm(to_world(nxt) - goal) / resolution
                heapq.heappush(frontier, (float(priority), nxt))
                parent[nxt] = current
    if target not in parent:
        return np.vstack([start, goal]).astype(np.float32)
    cells = []
    current: tuple[int, int] | None = target
    while current is not None:
        cells.append(current)
        current = parent[current]
    cells.reverse()
    return np.vstack([start, *(to_world(cell) for cell in cells[1:-1]), goal]).astype(np.float32)


def astar_action(position: np.ndarray, goals: np.ndarray, obstacles: np.ndarray,
                 paths: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    if paths is None:
        paths = [astar_path(p, g, obstacles) for p, g in zip(position, goals)]
    actions = np.zeros_like(position, dtype=np.float32)
    for i, path in enumerate(paths):
        distances = np.linalg.norm(path - position[i], axis=1)
        waypoint = path[min(int(np.argmin(distances)) + 1, len(path) - 1)]
        direction = waypoint - position[i]
        actions[i] = MAX_SPEED * direction / max(float(np.linalg.norm(direction)), 1e-6)
    return actions, paths


def rollout(scenario: Scenario, controller: str = "orca", steps: int = 240,
            policy=None, radius: float = COMM_RADIUS,
            safety: bool = False) -> dict[str, np.ndarray | float | int]:
    position = scenario.starts.copy()
    velocity = np.zeros_like(position)
    paths = None
    trajectories = [position.copy()]
    collisions = 0
    collision_events = 0
    active_collision_pairs: set[tuple[int, int]] = set()
    physical_collision_events = 0
    active_physical_pairs: set[tuple[int, int]] = set()
    min_separation = 99.0
    path_length = 0.0
    control_time = 0.0
    for _ in range(steps):
        control_start = time.perf_counter()
        if controller == "astar":
            action, paths = astar_action(position, scenario.goals, scenario.obstacles, paths)
        elif controller == "policy":
            features, adjacency = observation(position, velocity, scenario.goals, scenario.obstacles, radius)
            edge_features = build_edge_features(position, velocity, radius)
            try:
                action = policy(features, adjacency, edge_features)
            except TypeError:
                action = policy(features, adjacency)
        elif controller == "rvo":
            action = rvo_style_action(position, velocity, scenario.goals, scenario.obstacles, radius)
        else:
            action = orca_style_action(position, velocity, scenario.goals, scenario.obstacles, radius)
        control_time += time.perf_counter() - control_start
        if safety and controller == "policy":
            action = safety_filter(position, action)
        old = position.copy()
        velocity = clip_action(np.asarray(action, dtype=np.float32))
        position = np.clip(position + DT * velocity, [-7.5, -5.5], [7.5, 5.5])
        separation = np.linalg.norm(position[:, None] - position[None, :], axis=-1)
        separation += np.eye(len(position)) * 100.0
        current_min = float(separation.min())
        min_separation = min(min_separation, current_min)
        current_pairs = {
            (i, j) for i in range(len(position)) for j in range(i + 1, len(position))
            if separation[i, j] < SAFE_DISTANCE
        }
        collisions += len(current_pairs)
        collision_events += len(current_pairs - active_collision_pairs)
        active_collision_pairs = current_pairs
        physical_pairs = {
            (i, j) for i in range(len(position)) for j in range(i + 1, len(position))
            if separation[i, j] < 2.0 * ROBOT_RADIUS
        }
        physical_collision_events += len(physical_pairs - active_physical_pairs)
        active_physical_pairs = physical_pairs
        path_length += float(np.linalg.norm(position - old, axis=1).sum())
        trajectories.append(position.copy())
        if float(np.max(np.linalg.norm(position - scenario.goals, axis=1))) < 0.35:
            break
    final_error = float(np.mean(np.linalg.norm(position - scenario.goals, axis=1)))
    return {
        "trajectory": np.asarray(trajectories, dtype=np.float32),
        "position": position,
        "collisions": collisions,
        "collision_events": collision_events,
        "collision_episode": int(collision_events > 0),
        "physical_collision_events": physical_collision_events,
        "physical_collision_episode": int(physical_collision_events > 0),
        "min_separation": min_separation,
        "path_length": path_length,
        "steps": len(trajectories) - 1,
        "success": int(final_error < 0.65),
        "strict_success": int(final_error < 0.65 and physical_collision_events == 0),
        "final_error": final_error,
        "control_time_ms": 1000.0 * control_time / max(len(trajectories) - 1, 1),
    }
