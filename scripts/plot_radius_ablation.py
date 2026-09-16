#!/usr/bin/env python3
"""Plot communication-radius ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/ablation_radius.csv')
    parser.add_argument('--output', default='figures/generated/ablation_radius.png')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    grouped = data.groupby(['radius', 'robots'], as_index=False).agg(
        success=('success', 'mean'), collisions=('collision_events', 'mean'), degree=('average_degree', 'mean'))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for radius in sorted(grouped.radius.unique()):
        part = grouped[grouped.radius == radius].sort_values('robots')
        label = 'global' if radius >= 50 else f'{radius:g} m'
        axes[0].plot(part.robots, part.success, marker='o', label=label)
        axes[1].plot(part.robots, part.collisions, marker='o', label=label)
        axes[2].plot(part.robots, part.degree, marker='o', label=label)
    axes[0].set_ylabel('Success rate'); axes[1].set_ylabel('Collision events / episode'); axes[2].set_ylabel('Average graph degree')
    for axis in axes:
        axis.set_xlabel('Number of robots'); axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=220)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()

