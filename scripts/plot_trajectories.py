#!/usr/bin/env python3
"""Render comparable trajectories for one fixed paper figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.environment import make_scenario, rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--robots', type=int, default=10)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--gnn-checkpoint')
    parser.add_argument('--output', default='figures/generated/trajectories.png')
    args = parser.parse_args()
    scenario = make_scenario(args.robots, args.seed)
    methods = [('orca', 'ORCA'), ('astar', 'A*')]
    if args.gnn_checkpoint:
        import torch
        from evaluate_baselines import load_policy
        methods.append((load_policy(args.gnn_checkpoint, torch.device('cuda' if torch.cuda.is_available() else 'cpu')), 'GNN'))
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (controller, label) in zip(axes, methods):
        if isinstance(controller, str):
            result = rollout(scenario, controller, steps=240)
        else:
            result = rollout(scenario, 'policy', steps=240, policy=controller)
        trajectory = result['trajectory']
        for robot in range(args.robots):
            axis.plot(trajectory[:, robot, 0], trajectory[:, robot, 1], linewidth=0.9)
            axis.plot(scenario.starts[robot, 0], scenario.starts[robot, 1], 'o', markersize=3)
            axis.plot(scenario.goals[robot, 0], scenario.goals[robot, 1], 'x', markersize=4)
        for ox, oy, ow, oh in scenario.obstacles:
            axis.add_patch(plt.Rectangle((ox - ow / 2, oy - oh / 2), ow, oh, color='0.35'))
        axis.set_title(f'{label} | success={result["success"]}')
        axis.set_xlim(-8, 8); axis.set_ylim(-6, 6); axis.set_aspect('equal')
        axis.set_xlabel('x (m)'); axis.set_ylabel('y (m)')
        axis.grid(alpha=0.2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
