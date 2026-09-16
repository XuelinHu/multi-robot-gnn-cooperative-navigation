#!/usr/bin/env python3
"""Evaluate the same GNN under different communication graph radii."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.environment import COMM_RADIUS, build_graph, make_scenario, observation, rollout
from evaluate_baselines import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--robots', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10, 20)))
    parser.add_argument('--radii', type=float, nargs='+', default=[2.0, COMM_RADIUS, 5.5, 99.0])
    parser.add_argument('--output', default='results/ablation_radius.csv')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_policy(args.checkpoint, device)
    rows = []
    for radius in args.radii:
        for n in args.robots:
            for seed in args.seeds:
                scenario = make_scenario(n, seed)
                result = rollout(scenario, 'policy', 240, policy=policy, radius=radius)
                degrees = []
                for state in result['trajectory']:
                    degrees.append(float(build_graph(state, radius).sum(axis=1).mean()))
                rows.append({'radius': radius, 'robots': n, 'seed': seed,
                                 'success': result['success'], 'strict_success': result['strict_success'], 'physical_collision_events': result['physical_collision_events'], 'physical_collision_episode': result['physical_collision_episode'], 'collision_events': result['collision_events'],
                             'collision_episode': result['collision_episode'], 'min_separation': result['min_separation'],
                             'path_length': result['path_length'], 'final_error': result['final_error'],
                             'control_time_ms': result['control_time_ms'],
                             'average_degree': float(np.mean(degrees))})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'wrote {output} rows={len(rows)} device={device}')


if __name__ == '__main__':
    main()
