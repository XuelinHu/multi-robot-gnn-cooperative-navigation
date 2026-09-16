"""NumPy inference for exported MLP/GNN checkpoints used by ROS Python."""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class NumpyMLP:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    def __call__(self, x: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        x = relu(x @ self.w['net.0.weight'].T + self.w['net.0.bias'])
        x = relu(x @ self.w['net.2.weight'].T + self.w['net.2.bias'])
        return x @ self.w['net.4.weight'].T + self.w['net.4.bias']


class NumpyGNN:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    def __call__(self, x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        h = relu(x @ self.w['encoder.weight'].T + self.w['encoder.bias'])
        layer_count = len([key for key in self.w if key.startswith('message.') and key.endswith('.weight')])
        for i in range(layer_count):
            aggregate = adjacency @ h / np.maximum(adjacency.sum(axis=-1, keepdims=True), 1.0)
            message = relu(aggregate @ self.w[f'message.{i}.weight'].T + self.w[f'message.{i}.bias'])
            h = relu(np.concatenate([h, message], axis=-1) @ self.w[f'update.{i}.weight'].T + self.w[f'update.{i}.bias'])
        h = relu(h @ self.w['head.0.weight'].T + self.w['head.0.bias'])
        return h @ self.w['head.2.weight'].T + self.w['head.2.bias']


def load_numpy_policy(path: str):
    data = np.load(path, allow_pickle=False)
    weights = {key: data[key] for key in data.files if key != 'method'}
    method = str(data['method'].item())
    return (NumpyMLP if method == 'mlp' else NumpyGNN)(weights)

