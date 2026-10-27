"""Metric hồi quy cho kết quả MLP."""

from __future__ import annotations
import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Trả về MSE, MAE và RMSE cho hai mảng cùng số phần tử."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}
