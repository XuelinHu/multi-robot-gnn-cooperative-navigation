"""Dependency-light policies for multi-robot graph control."""

from __future__ import annotations

import torch
from torch import nn


class IndependentMLP(nn.Module):
    def __init__(self, features: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


class MessagePassingGNN(nn.Module):
    def __init__(self, features: int = 8, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.encoder = nn.Linear(features, hidden)
        self.edge_encoder = nn.Linear(5, hidden)
        self.message = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.update = nn.ModuleList(nn.Linear(hidden * 2, hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                edge_features: torch.Tensor | None = None) -> torch.Tensor:
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        h = torch.relu(self.encoder(x))
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        if edge_features is not None and edge_features.ndim == 3:
            edge_features = edge_features.unsqueeze(0)
        for message_layer, update_layer in zip(self.message, self.update):
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            aggregate = torch.bmm(adjacency, h) / degree
            if edge_features is not None:
                edge_aggregate = (adjacency.unsqueeze(-1) * edge_features).sum(dim=2) / degree
                aggregate = aggregate + self.edge_encoder(edge_aggregate)
            h = torch.relu(update_layer(torch.cat([h, torch.relu(message_layer(aggregate))], dim=-1)))
        output = self.head(h)
        return output.squeeze(0) if single else output


class AttentionGNN(nn.Module):
    """Local graph attention policy with shared multi-head message updates."""

    def __init__(self, features: int = 8, hidden: int = 128, layers: int = 3,
                 heads: int = 4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.encoder = nn.Linear(features, hidden)
        self.query = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.key = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.value = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.project = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.edge_bias = nn.Linear(5, heads)
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                edge_features: torch.Tensor | None = None) -> torch.Tensor:
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        if edge_features is not None and edge_features.ndim == 3:
            edge_features = edge_features.unsqueeze(0)
        h = torch.relu(self.encoder(x))
        graph_mask = adjacency.bool() | torch.eye(x.shape[1], device=x.device, dtype=torch.bool).unsqueeze(0)
        for q_layer, k_layer, v_layer, out_layer, norm_layer in zip(
                self.query, self.key, self.value, self.project, self.norm):
            batch, nodes, _ = h.shape
            q = q_layer(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
            k = k_layer(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
            v = v_layer(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if edge_features is not None:
                scores = scores + self.edge_bias(edge_features).permute(0, 3, 1, 2)
            scores = scores.masked_fill(~graph_mask[:, None, :, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            messages = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, nodes, self.hidden)
            h = norm_layer(h + out_layer(messages))
        output = self.head(h)
        return output.squeeze(0) if single else output


class GraphTransformer(nn.Module):
    """Graph Transformer policy using adjacency-masked self-attention."""

    def __init__(self, features: int = 8, hidden: int = 128, layers: int = 3,
                 heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.encoder = nn.Linear(features, hidden)
        self.query = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.key = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.value = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.project = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.edge_bias = nn.Linear(5, heads)
        self.norm1 = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden)
        ) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                edge_features: torch.Tensor | None = None) -> torch.Tensor:
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        if edge_features is not None and edge_features.ndim == 3:
            edge_features = edge_features.unsqueeze(0)
        h = torch.relu(self.encoder(x))
        graph_mask = adjacency.bool() | torch.eye(x.shape[1], device=x.device, dtype=torch.bool).unsqueeze(0)
        for q_layer, k_layer, v_layer, out_layer, norm1, norm2, ffn in zip(
                self.query, self.key, self.value, self.project, self.norm1, self.norm2, self.ffn):
            batch, nodes, _ = h.shape
            q = q_layer(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
            k = k_layer(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
            v = v_layer(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if edge_features is not None:
                scores = scores + self.edge_bias(edge_features).permute(0, 3, 1, 2)
            scores = scores.masked_fill(~graph_mask[:, None, :, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, nodes, self.hidden)
            attended = out_layer(attended)
            h = norm1(h + attended)
            h = norm2(h + ffn(h))
        output = self.head(h)
        return output.squeeze(0) if single else output
