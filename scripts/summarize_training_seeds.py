#!/usr/bin/env python3
"""Summarize variation across independently trained policy checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/training_seed_comparison.csv')
    parser.add_argument('--output', default='results/training_seed_summary.csv')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    metrics = ['success', 'collision_events', 'collision_episode',
               'min_separation', 'path_length', 'final_error', 'control_time_ms']
    per_checkpoint = data.groupby(['checkpoint', 'layout', 'robots'])[metrics].mean().reset_index()
    summary = per_checkpoint.groupby(['layout', 'robots'])[metrics].agg(['mean', 'std']).reset_index()
    summary.columns = [
        '_'.join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f'wrote {output} rows={len(summary)}')


if __name__ == '__main__':
    main()
