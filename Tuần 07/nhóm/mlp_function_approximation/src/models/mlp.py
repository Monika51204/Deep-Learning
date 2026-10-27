"""Định nghĩa MLP tùy biến cho bài toán hồi quy."""

from __future__ import annotations
import torch
from torch import nn


class MLP(nn.Module):
    """MLP nhận hai đặc trưng và dự đoán một giá trị Gaussian.

    Args:
        input_dim: số đặc trưng đầu vào, theo đề là 2.
        hidden_dims: số neuron của từng tầng ẩn; ví dụ (64, 64).
        output_dim: số đầu ra, theo đề là 1.
        activation: 'tanh' hoặc 'relu'.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, ...] = (64, 64),
                 output_dim: int = 1, activation: str = "tanh") -> None:
        super().__init__()
        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: list[nn.Module] = []
        in_features = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(in_features, width), activations[activation.lower()]()])
            in_features = width
        layers.append(nn.Linear(in_features, output_dim))  # output tuyến tính cho regression
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi; x shape (N, input_dim), trả (N, output_dim)."""
        return self.network(x)
