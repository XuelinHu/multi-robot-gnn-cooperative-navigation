#!/usr/bin/env python3
"""Plot the contribution of graph connectivity variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/ablation_graph_final.csv')
    parser.add_argument('--output', default='figures/generated/ablation_graph_final.png')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    grouped = data.groupby(['variant', 'robots'], as_index=False).agg(
        success=('success', 'mean'), collisions=('collision_events', 'mean'),
        separation=('min_separation', 'mean'))
    labels = {'no_edges': 'No edges', 'local_graph': 'Local graph', 'global_graph': 'Global graph'}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for variant in grouped.variant.unique():
        part = grouped[grouped.variant == variant].sort_values('robots')
        label = labels.get(variant, variant)
        axes[0].plot(part.robots, part.success, marker='o', label=label)
        axes[1].plot(part.robots, part.collisions, marker='o', label=label)
        axes[2].plot(part.robots, part.separation, marker='o', label=label)
    axes[0].set_ylabel('Success rate')
    axes[1].set_ylabel('Collision events / episode')
    axes[2].set_ylabel('Minimum separation (m)')
    for axis in axes:
        axis.set_xlabel('Number of robots')
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
