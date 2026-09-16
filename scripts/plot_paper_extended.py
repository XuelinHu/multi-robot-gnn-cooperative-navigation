"""Create additional publication figures from the recorded multiscale results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "multiscale_models_test_safety_summary.csv"
OUT = ROOT / "figures" / "generated"


def configure():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["method_label"] = df["method"].map({
        "gnn+safety": "GNN + safety",
        "attention+safety": "Attention-GNN + safety",
        "transformer+safety": "Graph Transformer + safety",
    })
    colors = {"GNN + safety": "#0072B2", "Attention-GNN + safety": "#D55E00",
              "Graph Transformer + safety": "#009E73"}
    markers = {"GNN + safety": "o", "Attention-GNN + safety": "s",
               "Graph Transformer + safety": "^"}

    # (a) Scaling curves: the strict metric is the primary safety result.
    grouped = df.groupby(["method_label", "robots"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharex=True)
    metrics = [("strict_success_mean", "Strict success"),
               ("min_separation_mean", "Minimum separation (m)"),
               ("control_time_ms_mean", "Control time (ms)")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for label in colors:
            part = grouped[grouped["method_label"] == label]
            ax.plot(part["robots"], part[metric], color=colors[label], marker=markers[label],
                    linewidth=1.7, markersize=4.5, label=label)
        ax.set_xlabel("Number of robots")
        ax.set_ylabel(ylabel)
        ax.set_xticks([5, 10, 20, 30, 40])
        ax.grid(axis="y", linewidth=0.45, color="#d9d9d9")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].legend(loc="lower left", frameon=False)
    save(fig, "multiscale_safety_scaling")

    # (b) Map-wise heatmap-like matrix with one cell per method and scale.
    pivot = df.pivot_table(index=["layout", "method_label"], columns="robots",
                           values="strict_success_mean", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    image = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{layout} / {method}" for layout, method in pivot.index])
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if pivot.iloc[row, col] > 0.42 else "white")
    ax.set_xlabel("Number of robots")
    ax.set_title("Strict no-collision success by map and model")
    fig.colorbar(image, ax=ax, label="Strict success")
    save(fig, "multiscale_map_heatmap")


if __name__ == "__main__":
    main()
