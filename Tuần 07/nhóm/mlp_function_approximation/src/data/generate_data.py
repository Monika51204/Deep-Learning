"""Sinh, chia và lưu dữ liệu Gaussian 2D cho các thực nghiệm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.gaussian import gaussian_2d

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_A_SIZES = list(range(200, 2001, 200))
TEST_SIZE = 100


def sample_inputs(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sinh n điểm đều trong miền x1∈[-2,2], x2∈[-4,4]."""
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-4.0, 4.0, size=n)
    return np.column_stack([x1, x2])


def make_frame(n: int, seed: int) -> pd.DataFrame:
    """Tạo DataFrame gồm x1, x2, y với seed xác định."""
    if n <= 0:
        raise ValueError("Số lượng mẫu phải lớn hơn 0.")

    rng = np.random.default_rng(seed)
    x = sample_inputs(n, rng)
    y = gaussian_2d(x).ravel()
    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})


def check_no_duplicate_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Đảm bảo tọa độ đầu vào không trùng giữa các split."""
    splits = {
        "train_pool": train,
        "validation": validation,
        "test": test,
    }
    names = list(splits)

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            inputs_a = splits[name_a][["x1", "x2"]]
            inputs_b = splits[name_b][["x1", "x2"]]
            if not inputs_a.merge(inputs_b, on=["x1", "x2"], how="inner").empty:
                raise ValueError(
                    f"Phát hiện tọa độ trùng giữa {name_a} và {name_b}."
                )


def generate_and_save(
    seed: int = 42,
    train_pool_size: int = 2000,
    validation_size: int = 400,
    test_size: int = TEST_SIZE,
) -> None:
    """Tạo các split độc lập, kiểm tra trùng lặp và lưu CSV cùng metadata.

    Các cỡ mẫu của thực nghiệm A dùng prefix cố định của train_pool.
    """
    if train_pool_size < max(EXPERIMENT_A_SIZES):
        raise ValueError(
            "train_pool_size phải tối thiểu 2000 mẫu để đáp ứng thực nghiệm A."
        )
    if validation_size <= 0:
        raise ValueError("validation_size phải lớn hơn 0.")
    if test_size != TEST_SIZE:
        raise ValueError("Tập test phải có đúng 100 mẫu.")

    train = make_frame(train_pool_size, seed)
    validation = make_frame(validation_size, seed + 1)
    test = make_frame(test_size, seed + 2)
    check_no_duplicate_inputs(train, validation, test)

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train_pool.csv", index=False)
    validation.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    metadata = {
        "seed_train": seed,
        "seed_validation": seed + 1,
        "seed_test": seed + 2,
        "train_pool_size": train_pool_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "x1_range": [-2, 2],
        "x2_range": [-4, 4],
        "mean": [0, 0],
        "covariance": [[0.3, -0.5], [-0.5, 2.0]],
        "columns": ["x1", "x2", "y"],
        "experiment_A_train_sizes": EXPERIMENT_A_SIZES,
        "experiment_A_sampling": (
            "Use the first n rows of train_pool.csv for each training size n."
        ),
        "test_policy": (
            "Use test.csv only for final model evaluation; "
            "do not use it for hyperparameter selection."
        ),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Đã lưu dữ liệu tại: {out}")
    print(f"Train pool: {len(train)} mẫu")
    print(f"Validation: {len(validation)} mẫu")
    print(f"Test: {len(test)} mẫu")


if __name__ == "__main__":
    generate_and_save()
