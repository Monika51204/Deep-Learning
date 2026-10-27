"""Tải Train/Validation/Test dạng mảng NumPy để huấn luyện và đánh giá mô hình."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DataSplit = Literal["train_pool", "val", "test"]
DATA_FILES: dict[DataSplit, str] = {
    "train_pool": "train_pool.csv",
    "val": "val.csv",
    "test": "test.csv",
}
DATA_COLUMNS = ["x1", "x2", "y"]
ROOT = Path(__file__).resolve().parents[2]


def load_data(
    split: DataSplit,
    n_samples: int | None = None,
    *,
    data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tải dữ liệu dưới dạng các features float32 kiểu ``(N, 2)`` và targets kiểu ``(N, 1)``.

    ``n_samples`` chọn một tiền tố (prefix) số lượng mẫu cố định và chỉ được hỗ trợ cho
    tập huấn luyện (training pool), theo yêu cầu của thí nghiệm A. Mặc định, các tệp CSV được đọc
    từ thư mục ``data/processed`` của dự án này, độc lập với thư mục
    làm việc hiện tại của tiến trình (process working directory).
    """
    if split not in DATA_FILES:
        raise ValueError(
            f"split phải là một trong: {', '.join(DATA_FILES)}; nhận được {split!r}."
        )
    if n_samples is not None:
        if split != "train_pool":
            raise ValueError("n_samples chỉ được dùng với split 'train_pool'.")
        if n_samples <= 0:
            raise ValueError("n_samples phải lớn hơn 0.")

    source_dir = ROOT / "data" / "processed" if data_dir is None else Path(data_dir)
    csv_path = source_dir / DATA_FILES[split]
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != DATA_COLUMNS:
        raise ValueError(
            f"{csv_path} phải có đúng các cột theo thứ tự: "
            f"{', '.join(DATA_COLUMNS)}."
        )
    if frame.empty:
        raise ValueError(f"{csv_path} không chứa mẫu dữ liệu.")
    if n_samples is not None:
        if n_samples > len(frame):
            raise ValueError(
                f"n_samples={n_samples} vượt quá train pool có {len(frame)} mẫu."
            )
        frame = frame.iloc[:n_samples]

    features = frame[["x1", "x2"]].to_numpy(dtype=np.float32)
    targets = frame[["y"]].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"{csv_path} chứa giá trị không hữu hạn.")
    return features, targets