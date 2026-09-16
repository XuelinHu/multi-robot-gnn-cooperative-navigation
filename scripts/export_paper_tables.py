#!/usr/bin/env python3
"""Export compact paper tables from the frozen experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mean_table(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = data.groupby(group_columns).agg(
        success=('success', 'mean'), collision_events=('collision_events', 'mean'),
        min_separation=('min_separation', 'mean'), path_length=('path_length', 'mean'),
        control_time_ms=('control_time_ms', 'mean')).reset_index()
    return grouped


def fmt(value: float) -> str:
    return f'{value:.3f}'


def markdown_table(data: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                values.append(value)
            elif column in {'robots', 'safety'}:
                values.append(str(int(value)))
            else:
                values.append(fmt(float(value)))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='manuscript/tables.md')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    main_data = pd.read_csv('results/baselines_final.csv')
    main_table = mean_table(main_data, ['method', 'robots'])
    safety = pd.read_csv('results/ablation_safety_final.csv')
    safety_table = safety.groupby(['safety', 'robots']).agg(
        success=('success', 'mean'), collision_events=('collision_events', 'mean'),
        min_separation=('min_separation', 'mean')).reset_index()
    graph = pd.read_csv('results/ablation_graph_final.csv')
    graph_table = graph.groupby(['variant', 'robots']).agg(
        success=('success', 'mean'), collision_events=('collision_events', 'mean'),
        min_separation=('min_separation', 'mean')).reset_index()
    text = '# Paper Tables\n\n'
    text += '## Main comparison\n\n' + markdown_table(
        main_table, ['method', 'robots', 'success', 'collision_events',
                     'min_separation', 'path_length', 'control_time_ms'],
        ['Method', 'Robots', 'Success', 'Collision events', 'Min separation (m)',
         'Path length (m)', 'Control time (ms)']) + '\n\n'
    text += '## Safety ablation\n\n' + markdown_table(
        safety_table, ['safety', 'robots', 'success', 'collision_events', 'min_separation'],
        ['Safety', 'Robots', 'Success', 'Collision events', 'Min separation (m)']) + '\n\n'
    text += '## Graph ablation\n\n' + markdown_table(
        graph_table, ['variant', 'robots', 'success', 'collision_events', 'min_separation'],
        ['Graph variant', 'Robots', 'Success', 'Collision events', 'Min separation (m)']) + '\n'
    output.write_text(text, encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
