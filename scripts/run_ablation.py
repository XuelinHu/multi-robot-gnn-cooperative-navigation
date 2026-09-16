#!/usr/bin/env python3
"""Run the safety-layer ablation for an already trained learned policy."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_baselines import load_policy
from core.environment import COMM_RADIUS, make_scenario, rollout
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["mlp", "gnn"], default="gnn")
    parser.add_argument("--robots", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    parser.add_argument("--output", default="results/ablation_safety.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device)
    rows = []
    for n in args.robots:
        for seed in args.seeds:
            scenario = make_scenario(n, seed)
            for enabled in (False, True):
                result = rollout(scenario, "policy", 240, policy=policy, radius=COMM_RADIUS, safety=enabled)
                rows.append({"method": args.method, "safety": int(enabled), "robots": n, "seed": seed,
                             **{key: result[key] for key in ["success", "strict_success", "physical_collision_events", "physical_collision_episode", "collisions", "collision_events", "collision_episode", "min_separation", "path_length", "steps", "final_error", "control_time_ms"]}})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} rows={len(rows)} device={device}")


if __name__ == "__main__":
    main()
