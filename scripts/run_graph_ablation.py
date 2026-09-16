#!/usr/bin/env python3
"""Evaluate whether local message passing contributes beyond node features."""

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
    parser.add_argument('--layouts', nargs='+', default=['crossing', 'open', 'random', 'narrow'])
    parser.add_argument('--output', default='results/ablation_graph_final.csv')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_policy(args.checkpoint, device)
    rows = []
    variants = [('no_edges', 0.0), ('local_graph', COMM_RADIUS), ('global_graph', 99.0)]
    for layout in args.layouts:
        for n in args.robots:
            for seed in args.seeds:
                scenario = make_scenario(n, seed, layout)
                for variant, radius in variants:
                    if variant == 'no_edges':
                        def position_policy(features, adjacency):
                            zeros = np.zeros_like(adjacency)
                            return policy(features, zeros)
                    elif variant == 'global_graph':
                        def position_policy(features, adjacency):
                            global_graph = np.ones_like(adjacency)
                            np.fill_diagonal(global_graph, 0.0)
                            return policy(features, global_graph)
                    else:
                        position_policy = policy
                    result = rollout(scenario, 'policy', 240, policy=position_policy,
                                     radius=radius if variant == 'local_graph' else COMM_RADIUS)
                    rows.append({'layout': layout, 'variant': variant, 'robots': n, 'seed': seed,
                                     **{key: result[key] for key in ['success', 'strict_success', 'physical_collision_events', 'physical_collision_episode', 'collision_events',
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
