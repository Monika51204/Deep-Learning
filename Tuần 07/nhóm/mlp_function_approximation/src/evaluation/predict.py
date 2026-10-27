"""Tiện ích dự đoán và nạp checkpoint."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn


def predict(model: nn.Module, x, device: str = "cpu") -> np.ndarray:
    """Dự đoán cho mảng/tensor X shape (N,2), trả NumPy array shape (N,1)."""
    model = model.to(device)
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(xt)
    return output.detach().cpu().numpy()
