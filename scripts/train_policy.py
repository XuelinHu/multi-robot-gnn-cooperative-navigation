#!/usr/bin/env python3
"""Train a small MLP or message-passing GNN with bounded GPU memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import AttentionGNN, GraphTransformer, IndependentMLP, MessagePassingGNN


def build_model(method: str, hidden: int, layers: int = 3) -> nn.Module:
    models = {
        "mlp": IndependentMLP,
        "gnn": MessagePassingGNN,
        "attention": AttentionGNN,
        "transformer": GraphTransformer,
    }
    return models[method](hidden=hidden, layers=layers) if method != "mlp" else models[method](hidden=hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--method", choices=["mlp", "gnn", "attention", "transformer"], default="gnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="results/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision-weight", type=float, default=0.0,
                        help="Weight for one-step pairwise separation penalty")
    parser.add_argument("--progress-weight", type=float, default=0.0,
                        help="Weight for one-step goal-progress penalty")
    parser.add_argument("--smoothness-weight", type=float, default=0.0,
                        help="Weight for action change relative to current velocity")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_files = [np.load(path) for path in args.data]
    max_robots = max(int(data["robot_count"]) for data in data_files)
    padded = []
    for data in data_files:
        n = int(data["robot_count"])
        samples = len(data["features"])
        features = np.zeros((samples, max_robots, 8), dtype=np.float32)
        adjacency = np.zeros((samples, max_robots, max_robots), dtype=np.float32)
        edges = np.zeros((samples, max_robots, max_robots, 5), dtype=np.float32)
        actions = np.zeros((samples, max_robots, 2), dtype=np.float32)
        positions = np.zeros((samples, max_robots, 2), dtype=np.float32)
        features[:, :n] = data["features"]
        adjacency[:, :n, :n] = data["adjacency"]
        if "edge_features" in data:
            edges[:, :n, :n] = data["edge_features"]
        actions[:, :n] = data["actions"]
        positions[:, :n] = data["positions"]
        node_mask = np.zeros((samples, max_robots, 1), dtype=np.float32)
        node_mask[:, :n] = 1.0
        padded.append((features, adjacency, edges, actions, positions, node_mask))
    x = torch.from_numpy(np.concatenate([item[0] for item in padded]))
    adjacency = torch.from_numpy(np.concatenate([item[1] for item in padded]))
    edge_features = torch.from_numpy(np.concatenate([item[2] for item in padded]))
    y = torch.from_numpy(np.concatenate([item[3] for item in padded]))
    positions = torch.from_numpy(np.concatenate([item[4] for item in padded]))
    node_mask = torch.from_numpy(np.concatenate([item[5] for item in padded]))
    robot_count = max_robots
    dataset = TensorDataset(x, adjacency, edge_features, y, positions, node_mask)
    validation_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model = build_model(args.method, args.hidden, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, ab, eb, yb, pb, mb in train_loader:
            xb, ab, eb, yb, pb, mb = (xb.to(device, non_blocking=True), ab.to(device, non_blocking=True),
                                      eb.to(device, non_blocking=True), yb.to(device, non_blocking=True),
                                      pb.to(device, non_blocking=True), mb.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb, ab, eb)
            loss = (loss_fn(prediction, yb) * mb).sum() / mb.sum().clamp_min(1.0)
            if args.collision_weight > 0:
                next_position = pb + 0.08 * 1.4 * prediction
                distances = torch.linalg.vector_norm(next_position[:, :, None] - next_position[:, None, :], dim=-1)
                mask = (1.0 - torch.eye(distances.shape[-1], device=device).unsqueeze(0)) * (mb @ mb.transpose(1, 2))
                collision_penalty = torch.relu(0.72 - distances) ** 2 * mask
                loss = loss + args.collision_weight * collision_penalty.sum() / mask.sum().clamp_min(1.0)
            if args.progress_weight > 0:
                current_goal_distance = torch.linalg.vector_norm(xb[:, :, :2], dim=-1)
                world_scale = torch.tensor([16.0, 12.0], device=device).view(1, 1, 2)
                next_goal_delta = xb[:, :, :2] - (0.08 * prediction * 1.4) / world_scale
                next_goal_distance = torch.linalg.vector_norm(next_goal_delta, dim=-1)
                progress_penalty = (torch.relu(next_goal_distance - current_goal_distance) * mb.squeeze(-1)).sum() / mb.sum().clamp_min(1.0)
                loss = loss + args.progress_weight * progress_penalty
            if args.smoothness_weight > 0:
                smoothness_penalty = (((prediction - xb[:, :, 2:4]) ** 2) * mb).sum() / mb.sum().clamp_min(1.0)
                loss = loss + args.smoothness_weight * smoothness_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for xb, ab, eb, yb, pb, mb in validation_loader:
                prediction = model(xb.to(device), ab.to(device), eb.to(device))
                validation_loss += ((loss_fn(prediction, yb.to(device)) * mb.to(device)).sum() / mb.to(device).sum().clamp_min(1.0)).item() * len(xb)
        train_loss /= train_size
        validation_loss /= validation_size
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={validation_loss:.6f} device={device}")
        if validation_loss < best:
            best = validation_loss
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / f"{args.method}_n{robot_count}.pt"
            torch.save({"model": model.state_dict(), "method": args.method, "robot_count": robot_count, "features": 8, "hidden": args.hidden, "layers": args.layers, "heads": 4, "edge_features": 5}, checkpoint)
            checkpoint.with_suffix(".json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2), encoding="utf-8")
    print(f"best_validation={best:.6f}")


if __name__ == "__main__":
    main()
