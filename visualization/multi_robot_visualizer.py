#!/usr/bin/env python3
"""Interactive 2D baseline for the multi-robot GNN research project.

This is intentionally dependency-light. It visualizes the graph structure and
provides a reproducible cooperative potential-field baseline. A later GNN
policy can implement the same `desired_velocity` interface.
"""

from __future__ import annotations

import math
import random
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.environment import astar_action, observation, orca_style_action, rvo_style_action  # noqa: E402
from core.models import IndependentMLP, MessagePassingGNN  # noqa: E402


WORLD_W = 16.0
WORLD_H = 12.0
ROBOT_RADIUS = 0.28
COMM_RADIUS = 3.4
SAFE_DISTANCE = 0.72
DT = 0.08


@dataclass
class Robot:
    index: int
    x: float
    y: float
    goal_x: float
    goal_y: float
    vx: float = 0.0
    vy: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)
    arrived: bool = False


class MultiRobotVisualizer:
    def __init__(self, root: tk.Tk, checkpoint: str | None = None) -> None:
        self.root = root
        self.root.title("Multi-Robot GNN Research Visualizer - Baseline")
        self.root.minsize(980, 650)
        self.running = False
        self.elapsed = 0.0
        self.collisions = 0
        self.total_path = 0.0
        self.seed = 7
        self.robot_count = tk.IntVar(value=8)
        self.edge_radius = tk.DoubleVar(value=COMM_RADIUS)
        self.status = tk.StringVar(value="Ready: cooperative potential-field baseline")
        self.method = tk.StringVar(value="ORCA")
        self.policy = None
        self.device = None
        if checkpoint:
            self._load_checkpoint(checkpoint)
        self.robots: list[Robot] = []
        self.obstacles = [
            (-6.4, 2.7, 1.8, 0.28),
            (6.0, -2.0, 1.8, 0.28),
            (-1.0, 4.0, 2.6, 0.34),
            (1.3, -4.0, 2.7, 0.34),
            (0.0, 0.6, 0.9, 1.7),
        ]
        self._build_ui()
        self.reset()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(toolbar, text="Start", command=self.start).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Pause", command=self.pause).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="New scenario", command=self.new_scenario).pack(side=tk.LEFT, padx=5)
        ttk.Label(toolbar, text="Robots").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Spinbox(toolbar, from_=3, to=20, width=4, textvariable=self.robot_count,
                   command=self.reset).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Graph radius").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Scale(toolbar, from_=1.8, to=5.5, variable=self.edge_radius,
                  orient=tk.HORIZONTAL, length=130, command=lambda _: self.draw()).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Policy").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Combobox(toolbar, textvariable=self.method, values=("ORCA", "RVO", "A*", "MLP", "GNN"),
                     state="readonly", width=7).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.RIGHT)

        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, background="#101820", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        panel = ttk.Frame(body, width=220, padding=(12, 0, 0, 0))
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(panel, text="Experiment monitor", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.metric_text = tk.StringVar()
        ttk.Label(panel, textvariable=self.metric_text, justify=tk.LEFT).pack(anchor="w", pady=(12, 0))
        ttk.Separator(panel).pack(fill=tk.X, pady=14)
        ttk.Label(panel, text="Legend", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, text="● robot node\n○ goal node\n— interaction edge\n· planned trail\n■ static obstacle", justify=tk.LEFT).pack(anchor="w", pady=8)
        ttk.Separator(panel).pack(fill=tk.X, pady=14)
        ttk.Label(panel, text="Research mapping", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Nodes: robots\nEdges: local communication\nPolicy: shared action head\nCurrent: classical baseline", justify=tk.LEFT).pack(anchor="w", pady=8)
        self.canvas.bind("<Configure>", lambda _: self.draw())

    def _load_checkpoint(self, checkpoint: str) -> None:
        try:
            import torch
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model_class = IndependentMLP if state["method"] == "mlp" else MessagePassingGNN
            self.policy = model_class(features=state["features"], hidden=state["hidden"])
            self.policy.load_state_dict(state["model"])
            self.policy.eval()
            self.device = torch
        except Exception as exc:
            self.status.set(f"Checkpoint unavailable: {exc}")

    def reset(self) -> None:
        self.pause()
        rng = random.Random(self.seed)
        n = max(3, min(20, int(self.robot_count.get())))
        self.robots = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            self.robots.append(Robot(
                i, 5.1 * math.cos(angle), 4.0 * math.sin(angle),
                -5.1 * math.cos(angle), -4.0 * math.sin(angle),
                trail=[(5.1 * math.cos(angle), 4.0 * math.sin(angle))],
            ))
        rng.shuffle(self.robots)
        for i, robot in enumerate(self.robots):
            robot.index = i
        self.elapsed = 0.0
        self.collisions = 0
        self.total_path = 0.0
        self.astar_paths = None
        self.status.set("Ready: cooperative potential-field baseline")
        self.draw()

    def new_scenario(self) -> None:
        self.seed += 1
        self.reset()

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.status.set("Running: cooperative potential-field baseline")
            self.tick()

    def pause(self) -> None:
        self.running = False
        if self.robots and not all(r.arrived for r in self.robots):
            self.status.set("Paused")

    def desired_velocity(self, robot: Robot) -> tuple[float, float]:
        gx, gy = robot.goal_x - robot.x, robot.goal_y - robot.y
        distance = math.hypot(gx, gy) or 1.0
        vx, vy = 1.15 * gx / distance, 1.15 * gy / distance
        for other in self.robots:
            if other is robot:
                continue
            dx, dy = robot.x - other.x, robot.y - other.y
            d = math.hypot(dx, dy) or 0.01
            if d < 1.7:
                strength = 0.9 * (1.7 - d) / 1.7
                vx += strength * dx / d
                vy += strength * dy / d
        for ox, oy, ow, oh in self.obstacles:
            dx = max(abs(robot.x - ox) - ow / 2, 0.01)
            dy = max(abs(robot.y - oy) - oh / 2, 0.01)
            d = math.hypot(dx, dy)
            if d < 1.4:
                sx = 1 if robot.x >= ox else -1
                sy = 1 if robot.y >= oy else -1
                strength = 1.0 * (1.4 - d) / 1.4
                vx += strength * sx * dx / d
                vy += strength * sy * dy / d
        speed = math.hypot(vx, vy) or 1.0
        return 1.4 * vx / speed, 1.4 * vy / speed

    def tick(self) -> None:
        if not self.running:
            return
        previous = [(r.x, r.y) for r in self.robots]
        positions = np.asarray([[r.x, r.y] for r in self.robots], dtype=np.float32)
        velocities = np.asarray([[r.vx, r.vy] for r in self.robots], dtype=np.float32)
        goals = np.asarray([[r.goal_x, r.goal_y] for r in self.robots], dtype=np.float32)
        obstacles = np.asarray(self.obstacles, dtype=np.float32)
        radius = float(self.edge_radius.get())
        selected_method = self.method.get()
        actions = None
        if selected_method == "ORCA":
            actions = orca_style_action(positions, velocities, goals, obstacles, radius)
        elif selected_method == "RVO":
            actions = rvo_style_action(positions, velocities, goals, obstacles, radius)
        elif selected_method == "A*":
            actions, self.astar_paths = astar_action(positions, goals, obstacles, self.astar_paths)
        elif self.policy is not None and selected_method in ("MLP", "GNN"):
            features, adjacency = observation(positions, velocities, goals, obstacles, radius)
            with self.device.no_grad():
                actions = self.policy(
                    self.device.from_numpy(features),
                    self.device.from_numpy(adjacency),
                ).numpy() * 1.4
        for robot in self.robots:
            if math.hypot(robot.goal_x - robot.x, robot.goal_y - robot.y) < 0.35:
                robot.arrived = True
                robot.vx = robot.vy = 0.0
                continue
            if actions is not None:
                robot.vx, robot.vy = actions[robot.index]
            else:
                robot.vx, robot.vy = self.desired_velocity(robot)
            robot.x += DT * robot.vx
            robot.y += DT * robot.vy
            robot.x = max(-7.55, min(7.55, robot.x))
            robot.y = max(-5.55, min(5.55, robot.y))
            robot.trail.append((robot.x, robot.y))
            if len(robot.trail) > 180:
                robot.trail.pop(0)
        self.elapsed += DT
        self.total_path += sum(math.hypot(r.x - p[0], r.y - p[1]) for r, p in zip(self.robots, previous))
        for i, robot in enumerate(self.robots):
            for other in self.robots[i + 1:]:
                if math.hypot(robot.x - other.x, robot.y - other.y) < SAFE_DISTANCE:
                    self.collisions += 1
        self.draw()
        if all(r.arrived for r in self.robots):
            self.running = False
            self.status.set("Finished: all robots reached goals")
        else:
            self.root.after(30, self.tick)

    def world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / WORLD_W, height / WORLD_H)
        return width / 2 + x * scale, height / 2 - y * scale

    def draw(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10:
            return
        scale = min(width / WORLD_W, height / WORLD_H)
        for x in range(-8, 9):
            px, _ = self.world_to_canvas(x, 0)
            self.canvas.create_line(px, 0, px, height, fill="#1b2b36")
        for y in range(-6, 7):
            _, py = self.world_to_canvas(0, y)
            self.canvas.create_line(0, py, width, py, fill="#1b2b36")
        for ox, oy, ow, oh in self.obstacles:
            x1, y1 = self.world_to_canvas(ox - ow / 2, oy + oh / 2)
            x2, y2 = self.world_to_canvas(ox + ow / 2, oy - oh / 2)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#b65f45", outline="#e08b68")
        radius = float(self.edge_radius.get())
        for i, robot in enumerate(self.robots):
            for other in self.robots[i + 1:]:
                if math.hypot(robot.x - other.x, robot.y - other.y) <= radius:
                    x1, y1 = self.world_to_canvas(robot.x, robot.y)
                    x2, y2 = self.world_to_canvas(other.x, other.y)
                    self.canvas.create_line(x1, y1, x2, y2, fill="#3a7080", width=1)
        for robot in self.robots:
            for a, b in zip(robot.trail, robot.trail[1:]):
                x1, y1 = self.world_to_canvas(*a)
                x2, y2 = self.world_to_canvas(*b)
                self.canvas.create_line(x1, y1, x2, y2, fill="#708d98", width=1)
            gx, gy = self.world_to_canvas(robot.goal_x, robot.goal_y)
            self.canvas.create_oval(gx - 7, gy - 7, gx + 7, gy + 7, outline="#8bd4a4", width=2)
            x, y = self.world_to_canvas(robot.x, robot.y)
            r = max(7, ROBOT_RADIUS * scale)
            color = "#8bd4a4" if robot.arrived else "#44b4d6"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="#d8f3fa", width=1)
            self.canvas.create_text(x, y, text=str(robot.index + 1), fill="#10222a", font=("TkDefaultFont", 8, "bold"))
        arrived = sum(r.arrived for r in self.robots)
        degrees = sum(sum(math.hypot(r.x - o.x, r.y - o.y) <= radius for o in self.robots if o is not r) for r in self.robots)
        avg_degree = degrees / len(self.robots) if self.robots else 0
        self.metric_text.set(
            f"Robots     {len(self.robots)}\n"
            f"Arrived    {arrived}/{len(self.robots)}\n"
            f"Collisions {self.collisions}\n"
            f"Path       {self.total_path:.1f} m\n"
            f"Time       {self.elapsed:.1f} s\n"
            f"Avg degree {avg_degree:.1f}\n"
            f"Seed       {self.seed}"
        )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Optional MLP/GNN .pt checkpoint")
    args = parser.parse_args()
    root = tk.Tk()
    MultiRobotVisualizer(root, args.checkpoint)
    root.mainloop()


if __name__ == "__main__":
    main()
