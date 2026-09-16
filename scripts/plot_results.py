#!/usr/bin/env python3
"""Create compact paper figures from evaluate_baselines.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/baselines.csv")
    parser.add_argument("--output", default="figures/generated/baselines.png")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    grouped = df.groupby(["method", "robots"], as_index=False).agg(
        success=("success", "mean"), collision=("collision_events", "mean"), path=("path_length", "mean"))
    methods = list(grouped["method"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for method in methods:
        part = grouped[grouped.method == method]
        axes[0].plot(part.robots, part.success, marker="o", label=method.upper())
        axes[1].plot(part.robots, part.collision, marker="o", label=method.upper())
        axes[2].plot(part.robots, part.path, marker="o", label=method.upper())
    axes[0].set_ylabel("Success rate")
    axes[1].set_ylabel("Collisions / episode")
    axes[2].set_ylabel("Path length (m)")
    for axis in axes:
        axis.set_xlabel("Number of robots")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
