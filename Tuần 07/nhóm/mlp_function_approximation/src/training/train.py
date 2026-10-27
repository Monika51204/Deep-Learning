"""Vòng lặp train/validation PyTorch với checkpoint theo validation loss."""

from __future__ import annotations
import copy
import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_run(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  config: dict, checkpoint_path: str | Path) -> tuple[nn.Module, pd.DataFrame]:
    """Huấn luyện một cấu hình và trả về model tốt nhất cùng lịch sử epoch.

    config cần có: epochs, learning_rate, optimizer ('adam' hoặc 'sgd'),
    patience, device. DataLoader phải trả về cặp (X, y).
    """
    device = torch.device(config.get("device", "cpu"))
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer_name = config.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.get("learning_rate", 1e-3))
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    rows = []
    start = time.perf_counter()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config.get("epochs", 100)) + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(xb)
            train_count += len(xb)

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss = criterion(model(xb), yb)
                val_sum += val_loss.item() * len(xb)
                val_count += len(xb)
        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state, "config": config,
                        "best_val_loss": best_val, "epoch": epoch}, checkpoint_path)
            wait = 0
        else:
            wait += 1
        if wait >= int(config.get("patience", 30)):
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    history["elapsed_seconds"] = time.perf_counter() - start
    return model, history
