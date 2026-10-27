import json

import numpy as np
import pandas as pd
import pytest

from src.visualization import plot_data


def test_create_summary_reports_statistics_by_split():
    splits = {
        "Train": pd.DataFrame({
            "x1": [1.0, 3.0],
            "x2": [2.0, 4.0],
            "y": [0.2, 0.4],
        })
    }

    summary = plot_data.create_summary(splits)

    assert list(summary.columns) == [
        "split", "variable", "count", "mean", "std",
        "min", "q25", "median", "q75", "max",
    ]
    x1_summary = summary.loc[summary["variable"] == "x1"].iloc[0]
    assert x1_summary["split"] == "train"
    assert x1_summary["count"] == 2
    assert x1_summary["mean"] == 2.0
    assert x1_summary["min"] == 1.0
    assert x1_summary["max"] == 3.0


def test_load_data_rejects_missing_schema(tmp_path):
    pd.DataFrame({"x1": [0.0], "x2": [0.0]}).to_csv(
        tmp_path / "train_pool.csv", index=False
    )

    with pytest.raises(ValueError, match="y"):
        plot_data.load_data(tmp_path)


def test_main_saves_figures_and_summary(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    for filename, offset, size in (
        ("train_pool.csv", 0.0, 12),
        ("val.csv", 0.1, 6),
        ("test.csv", 0.2, 4),
    ):
        x1 = np.linspace(-1.0, 1.0, size) + offset
        x2 = np.linspace(-2.0, 2.0, size) - offset
        pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "y": np.asarray(plot_data.gaussian_2d(np.column_stack([x1, x2]))).ravel(),
        }).to_csv(data_dir / filename, index=False)
    (data_dir / "metadata.json").write_text(
        json.dumps({
            "mean": [0, 0],
            "covariance": [[0.3, -0.5], [-0.5, 2.0]],
            "x1_range": [-2, 2],
            "x2_range": [-4, 4],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(plot_data, "ROOT", tmp_path)

    plot_data.main()

    figures_dir = tmp_path / "reports" / "figures"
    assert {
        "data_splits_3d.png",
        "data_true_surface.png",
        "data_distributions.png",
    } == {path.name for path in figures_dir.glob("data_*.png")}
    assert all(path.stat().st_size > 0 for path in figures_dir.glob("data_*.png"))
    summary = pd.read_csv(tmp_path / "reports" / "tables" / "data_summary.csv")
    assert len(summary) == 9
    assert set(summary["split"]) == {"train", "validation", "test"}
    assert set(summary["variable"]) == {"x1", "x2", "y"}
