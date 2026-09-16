#!/usr/bin/env python3
"""Generate fixed-seed imitation data from the ORCA-style controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.environment import (COMM_RADIUS, DT, SAFE_DISTANCE, build_edge_features,
                               make_scenario, observation, orca_style_action,
                               rvo_style_action, safety_filter)


def generate(n: int, episodes: int, steps: int, seed: int, layout: str, safe_labels: bool = False, expert: str = "orca") -> dict[str, np.ndarray]:
    feature_rows, adjacency_rows, edge_rows, action_rows, position_rows = [], [], [], [], []
    episode_rows = []
    reward_rows = []
    first_obstacles = None
    for episode in range(episodes):
        scenario = make_scenario(n, seed + episode, layout)
        if first_obstacles is None:
            first_obstacles = scenario.obstacles.copy()
        position, velocity = scenario.starts.copy(), np.zeros_like(scenario.starts)
        for t in range(steps):
            features, adjacency = observation(position, velocity, scenario.goals, scenario.obstacles, COMM_RADIUS)
            edge_features = build_edge_features(position, velocity, COMM_RADIUS)
            if expert == "rvo" or (expert == "mixed" and episode % 3 == 1):
                action = rvo_style_action(position, velocity, scenario.goals, scenario.obstacles, COMM_RADIUS)
            else:
                action = orca_style_action(position, velocity, scenario.goals, scenario.obstacles, COMM_RADIUS)
            if safe_labels or (expert == "mixed" and episode % 3 == 2):
                action = safety_filter(position, action)
            feature_rows.append(features)
            adjacency_rows.append(adjacency)
            edge_rows.append(edge_features)
            action_rows.append(action / 1.4)
            position_rows.append(position.copy())
            distances = np.linalg.norm(position[:, None] - position[None, :], axis=-1)
            pair_collisions = np.triu(distances < SAFE_DISTANCE, k=1).sum()
            reward_rows.append(float(-np.linalg.norm(scenario.goals - position, axis=1).mean() - 2.0 * pair_collisions))
            position = np.clip(position + DT * action, [-7.5, -5.5], [7.5, 5.5])
            velocity = action
            episode_rows.append((episode, t))
    return {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "adjacency": np.asarray(adjacency_rows, dtype=np.float32),
        "edge_features": np.asarray(edge_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.float32),
        "positions": np.asarray(position_rows, dtype=np.float32),
        "episode_step": np.asarray(episode_rows, dtype=np.int32),
        "rewards": np.asarray(reward_rows, dtype=np.float32),
        "obstacles": np.asarray(first_obstacles, dtype=np.float32),
        "layout": np.asarray(layout),
        "expert": np.asarray(expert),
        "robot_count": np.asarray(n, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=["open", "random", "narrow", "crossing"], default="crossing")
    parser.add_argument("--expert", choices=["orca", "rvo", "mixed"], default="orca")
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--safe-labels", action="store_true", help="Apply the common safety projection before storing expert actions")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for n in args.robots:
        data = generate(n, args.episodes, args.steps, args.seed, args.layout, args.safe_labels, args.expert)
        prefix = f"safe_{args.expert}" if args.safe_labels else args.expert
        path = output / f"{prefix}_{args.layout}_n{n}.npz"
        np.savez_compressed(path, **data)
        print(f"wrote {path} samples={len(data['features'])} size={path.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
