"""Smoke test shape cho MLP."""

import torch
from src.models.mlp import MLP


def test_mlp_output_shape():
    model = MLP(input_dim=2, hidden_dims=(8, 4), output_dim=1, activation="tanh")
    x = torch.randn(5, 2)
    y = model(x)
    assert y.shape == (5, 1)
