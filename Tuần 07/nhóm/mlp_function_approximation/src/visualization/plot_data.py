"""Tạo biểu đồ và thống kê mô tả cho dữ liệu train/validation/test."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.gaussian import gaussian_2d

ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILES = {
    "Train": "train_pool.csv",
    "Validation": "val.csv",
    "Test": "test.csv",
}
DATA_COLUMNS = ["x1", "x2", "y"]
SPLIT_STYLES = {
    "Train": {"color": "tab:blue", "marker": "o"},
    "Validation": {"color": "tab:orange", "marker": "^"},
    "Test": {"color": "tab:green", "marker": "s"},
}


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Đọc các split và kiểm tra schema đầu vào."""
    splits: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        data = pd.read_csv(data_dir / filename)
        missing = set(DATA_COLUMNS).difference(data.columns)
        if missing:
            raise ValueError(
                f"{filename} thiếu các cột bắt buộc: {', '.join(sorted(missing))}."
            )
        if data.empty:
            raise ValueError(f"{filename} không có mẫu dữ liệu.")
        splits[split] = data[DATA_COLUMNS]
    return splits


def create_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tính thống kê mô tả từng biến theo split."""
    rows: list[dict[str, float | int | str]] = []
    for split, data in splits.items():
        for variable in DATA_COLUMNS:
            values = data[variable]
            rows.append({
                "split": split.lower(),
                "variable": variable,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def _read_gaussian_parameters(data_dir: Path) -> tuple[list[float], list[list[float]]]:
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mean = metadata.get("mean")
    covariance = metadata.get("covariance")
    if mean is None or covariance is None:
        raise ValueError(
            f"{metadata_path} phải chứa các trường 'mean' và 'covariance'."
        )
    return mean, covariance


def plot_splits_3d(
    splits: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Vẽ scatter 3D của cả ba split với legend phân biệt."""
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    for split, data in splits.items():
        style = SPLIT_STYLES[split]
        axis.scatter(
            data["x1"],
            data["x2"],
            data["y"],
            label=f"{split} (n={len(data)})",
            color=style["color"],
            marker=style["marker"],
            alpha=0.65,
            s=18,
        )
    axis.set_title("Gaussian 2D samples by data split")
    axis.set_xlabel("x1 (coordinate units)")
    axis.set_ylabel("x2 (coordinate units)")
    axis.set_zlabel("y (probability density)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_true_surface(
    data_dir: Path,
    output_path: Path,
) -> None:
    """Vẽ bề mặt mật độ Gaussian thật theo miền trong metadata."""
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    mean, covariance = _read_gaussian_parameters(data_dir)
    x1_range = metadata.get("x1_range", [-2, 2])
    x2_range = metadata.get("x2_range", [-4, 4])
    x1 = np.linspace(x1_range[0], x1_range[1], 100)
    x2 = np.linspace(x2_range[0], x2_range[1], 100)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid = np.column_stack([grid_x1.ravel(), grid_x2.ravel()])
    values = gaussian_2d(
        grid,
        mean=np.asarray(mean, dtype=float),
        cov=np.asarray(covariance, dtype=float),
    ).reshape(grid_x1.shape)

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        grid_x1,
        grid_x2,
        values,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )
    axis.set_title("True Gaussian 2D probability density")
    axis.set_xlabel("x1 (coordinate units)")
    axis.set_ylabel("x2 (coordinate units)")
    axis.set_zlabel("y (probability density)")
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.1, label="Probability density")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_distributions(
    splits: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Vẽ phân phối x1, x2 và y theo split."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = {
        "x1": "x1 (coordinate units)",
        "x2": "x2 (coordinate units)",
        "y": "y (probability density)",
    }
    for axis, variable in zip(axes, DATA_COLUMNS):
        pooled = pd.concat([data[variable] for data in splits.values()])
        bins = np.histogram_bin_edges(pooled, bins=30)
        for split, data in splits.items():
            axis.hist(
                data[variable],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=f"{split} (n={len(data)})",
                color=SPLIT_STYLES[split]["color"],
            )
        axis.set_title(f"Distribution of {variable}")
        axis.set_xlabel(labels[variable])
        axis.set_ylabel("Probability density")
        axis.grid(True, alpha=0.25)
    axes[-1].legend()
    figure.suptitle("Input and target distributions by data split")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Tạo tất cả hình và bảng thống kê từ data/processed."""
    data_dir = ROOT / "data" / "processed"
    figures_dir = ROOT / "reports" / "figures"
    tables_dir = ROOT / "reports" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    splits = load_data(data_dir)
    plot_splits_3d(splits, figures_dir / "data_splits_3d.png")
    plot_true_surface(data_dir, figures_dir / "data_true_surface.png")
    plot_distributions(splits, figures_dir / "data_distributions.png")
    create_summary(splits).to_csv(tables_dir / "data_summary.csv", index=False)
    print("Saved EDA figures and data_summary.csv.")


if __name__ == "__main__":
    main()
