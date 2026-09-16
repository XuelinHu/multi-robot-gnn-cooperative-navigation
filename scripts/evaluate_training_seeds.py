#!/usr/bin/env python3
"""Evaluate independently trained checkpoints on identical episodes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.environment import COMM_RADIUS, make_scenario, rollout
from evaluate_baselines import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--robots', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10, 20)))
    parser.add_argument('--layouts', nargs='+', default=['crossing', 'open', 'random', 'narrow'])
    parser.add_argument('--output', default='results/training_seed_comparison.csv')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policies = [(Path(path).parent.name, load_policy(path, device)) for path in args.checkpoints]
    rows = []
    for layout in args.layouts:
        for n in args.robots:
            for seed in args.seeds:
                scenario = make_scenario(n, seed, layout)
                for checkpoint, policy in policies:
                    result = rollout(scenario, 'policy', 240, policy=policy, radius=COMM_RADIUS)
                    rows.append({'layout': layout, 'robots': n, 'scenario_seed': seed,
                                 'checkpoint': checkpoint,
                                 **{key: result[key] for key in ['success', 'collision_events',
                                 'collision_episode', 'min_separation', 'path_length',
                                 'final_error', 'control_time_ms']}})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {output} rows={len(rows)} device={device}')


if __name__ == '__main__':
    main()
