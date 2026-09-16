#!/usr/bin/env python3
"""Run classical and learned policies on identical deterministic episodes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.environment import COMM_RADIUS, make_scenario, rollout
from core.models import AttentionGNN, GraphTransformer, IndependentMLP, MessagePassingGNN


def load_policy(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model_classes = {
        "mlp": IndependentMLP,
        "gnn": MessagePassingGNN,
        "attention": AttentionGNN,
        "transformer": GraphTransformer,
    }
    model_class = model_classes[state["method"]]
    model = model_class(features=state["features"], hidden=state["hidden"],
                        layers=state.get("layers", 3)) if state["method"] != "mlp" else model_class(features=state["features"], hidden=state["hidden"])
    model = model.to(device)
    model.load_state_dict(state["model"], strict=bool(state.get("edge_features", 0)))
    model.eval()

    def policy(features: np.ndarray, adjacency: np.ndarray, edge_features: np.ndarray | None = None) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(features).to(device)
            a = torch.from_numpy(adjacency).to(device)
            if state.get("edge_features", 0) and edge_features is not None:
                e = torch.from_numpy(edge_features).to(device)
                action = model(x, a, e).cpu().numpy() * 1.4
            else:
                action = model(x, a).cpu().numpy() * 1.4
        return action
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--layouts", nargs="+", choices=["crossing", "open", "random", "narrow"], default=["crossing"])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--gnn-checkpoint")
    parser.add_argument("--attention-checkpoint")
    parser.add_argument("--transformer-checkpoint")
    parser.add_argument("--safety", action="store_true", help="Apply the common pairwise safety filter to learned policies")
    parser.add_argument("--output", default="results/baselines.csv")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--learned-only", action="store_true",
                        help="Skip classical controllers when collecting learned-model metrics")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = {} if args.learned_only else {"orca": None, "rvo": None, "astar": None}
    if args.mlp_checkpoint:
        policies["mlp"] = load_policy(args.mlp_checkpoint, device)
    if args.gnn_checkpoint:
        policies["gnn"] = load_policy(args.gnn_checkpoint, device)
    if args.attention_checkpoint:
        policies["attention"] = load_policy(args.attention_checkpoint, device)
    if args.transformer_checkpoint:
        policies["transformer"] = load_policy(args.transformer_checkpoint, device)
    rows = []
    for layout in args.layouts:
        for n in args.robots:
            for seed in args.seeds:
                scenario = make_scenario(n, seed, layout)
                for method, policy in policies.items():
                    result = rollout(scenario, "policy" if policy else method, args.steps, policy=policy, radius=COMM_RADIUS, safety=args.safety)
                    label = f"{method}+safety" if args.safety and policy else method
                    rows.append({"layout": layout, "method": label, "robots": n, "seed": seed, **{k: result[k] for k in ["success", "strict_success", "physical_collision_events", "physical_collision_episode", "collisions", "collision_events", "collision_episode", "min_separation", "path_length", "steps", "final_error", "control_time_ms"]}})
                    if not args.quiet:
                        print(layout, method, n, seed, rows[-1])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} rows={len(rows)} device={device}")


if __name__ == "__main__":
    main()
