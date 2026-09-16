#!/usr/bin/env python3
"""Create paper-ready mean/std/95%-CI tables from episode results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/baselines_full.csv')
    parser.add_argument('--output', default='results/baselines_summary.csv')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    preferred_metrics = ['success', 'strict_success', 'physical_collision_events', 'physical_collision_episode', 'collision_events', 'collision_episode', 'min_separation', 'path_length', 'final_error', 'control_time_ms']
    metrics = [metric for metric in preferred_metrics if metric in df.columns]
    grouped = df.groupby(['layout', 'method', 'robots'])
    rows = []
    for keys, part in grouped:
        row = dict(zip(['layout', 'method', 'robots'], keys))
        for metric in metrics:
            values = part[metric].astype(float)
            row[f'{metric}_mean'] = values.mean()
            row[f'{metric}_std'] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f'{metric}_ci95'] = 1.96 * row[f'{metric}_std'] / max(len(values) ** 0.5, 1.0)
        row['episodes'] = len(part)
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'wrote {output} rows={len(rows)}')


if __name__ == '__main__':
    main()
