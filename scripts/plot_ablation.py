#!/usr/bin/env python3
"""Plot raw versus safety-filtered learned-policy results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/ablation_safety.csv')
    parser.add_argument('--output', default='figures/generated/ablation_safety.png')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    grouped = df.groupby(['safety', 'robots'], as_index=False).agg(
        success=('success', 'mean'), collisions=('collision_events', 'mean'), separation=('min_separation', 'mean'))
    labels = {0: 'raw policy', 1: 'policy + safety'}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for safety in sorted(grouped.safety.unique()):
        part = grouped[grouped.safety == safety]
        axes[0].plot(part.robots, part.success, marker='o', label=labels[int(safety)])
        axes[1].plot(part.robots, part.collisions, marker='o', label=labels[int(safety)])
        axes[2].plot(part.robots, part.separation, marker='o', label=labels[int(safety)])
    axes[0].set_ylabel('Success rate')
    axes[1].set_ylabel('Collisions / episode')
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
