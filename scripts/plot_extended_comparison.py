#!/usr/bin/env python3
"""Publication-style summary of the extended model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "astar": "#0072B2", "orca": "#D55E00", "rvo": "#009E73",
    "mlp": "#CC79A7", "gnn": "#E69F00", "attention": "#56B4E9",
    "transformer": "#000000",
}
MARKERS = {"astar": "o", "orca": "s", "rvo": "^", "mlp": "D",
           "gnn": "P", "attention": "X", "transformer": "*"}
LABELS = {"astar": "A*", "orca": "ORCA-style", "rvo": "RVO-style",
          "mlp": "MLP", "gnn": "GNN", "attention": "Attention-GNN",
          "transformer": "Graph Transformer"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/extended_models.csv")
    parser.add_argument("--output", default="figures/generated/extended_models.png")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    grouped = df.groupby(["method", "robots"], as_index=False).agg(
        success=("success", "mean"), strict_success=("strict_success", "mean"),
        collision_events=("collision_events", "mean"),
        min_separation=("min_separation", "mean"),
        path_length=("path_length", "mean"),
        control_time_ms=("control_time_ms", "mean"))
    panels = [("success", "Goal success rate"),
              ("strict_success", "Collision-free success rate"),
              ("collision_events", "Collision events / episode"),
              ("min_separation", "Minimum separation (m)"),
              ("path_length", "Path length (m)"),
              ("control_time_ms", "Control time (ms/step)")]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.dpi": 320})
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for method in sorted(grouped["method"].unique(), key=lambda x: list(LABELS).index(x)):
            part = grouped[grouped.method == method].sort_values("robots")
            axis.plot(part.robots, part[metric], color=COLORS.get(method, "#444444"),
                      marker=MARKERS.get(method, "o"), linewidth=1.6, markersize=4.5,
                      label=LABELS.get(method, method))
        axis.set_xlabel("Number of robots")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted(grouped.robots.unique()))
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
