#!/usr/bin/env python3
"""Plot in-distribution versus random-layout generalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/baselines_summary.csv')
    parser.add_argument('--output', default='figures/generated/generalization.png')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    methods = list(df.method.unique())
    layouts = [layout for layout in ['crossing', 'open', 'random', 'narrow'] if layout in set(df.layout)]
    fig, axes = plt.subplots(len(layouts), 2, figsize=(8.2, 2.8 * len(layouts)), squeeze=False, constrained_layout=True)
    for column, metric, ylabel in [(0, 'success_mean', 'Success rate'), (1, 'collision_events_mean', 'Collision events / episode')]:
        for row, layout in enumerate(layouts):
            axis = axes[row, column]
            part = df[df.layout == layout]
            for method in methods:
                values = part[part.method == method].sort_values('robots')
                axis.plot(values.robots, values[metric], marker='o', label=method.upper())
            axis.set_title(layout)
            axis.set_xlabel('Number of robots')
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
