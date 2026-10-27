import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_data


@pytest.fixture
def processed_data(tmp_path):
    rows = pd.DataFrame({
        "x1": [0.0, 1.0, 2.0],
        "x2": [3.0, 4.0, 5.0],
        "y": [0.1, 0.2, 0.3],
    })
    rows.to_csv(tmp_path / "train_pool.csv", index=False)
    rows.iloc[:2].to_csv(tmp_path / "val.csv", index=False)
    rows.iloc[:1].to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def test_load_data_returns_float32_arrays_and_fixed_prefix(processed_data):
    features, targets = load_data("train_pool", n_samples=2, data_dir=processed_data)

    assert features.shape == (2, 2)
    assert targets.shape == (2, 1)
    assert features.dtype == np.float32
    assert targets.dtype == np.float32
    np.testing.assert_array_equal(features, [[0, 3], [1, 4]])
    np.testing.assert_allclose(targets, [[0.1], [0.2]])


def test_load_data_rejects_invalid_split_and_subset(processed_data):
    with pytest.raises(ValueError, match="split"):
        load_data("validation", data_dir=processed_data)
    with pytest.raises(ValueError, match="chỉ"):
        load_data("val", n_samples=1, data_dir=processed_data)
    with pytest.raises(ValueError, match="lớn hơn 0"):
        load_data("train_pool", n_samples=0, data_dir=processed_data)
    with pytest.raises(ValueError, match="exceeds|vượt quá"):
        load_data("train_pool", n_samples=4, data_dir=processed_data)


def test_load_data_rejects_unexpected_csv_schema(tmp_path):
    pd.DataFrame({"x1": [0.0], "x2": [1.0], "y": [0.2], "index": [0]}).to_csv(
        tmp_path / "test.csv", index=False
    )

    with pytest.raises(ValueError, match="exactly|đúng"):
        load_data("test", data_dir=tmp_path)


def test_load_data_rejects_non_finite_values(tmp_path):
    pd.DataFrame({"x1": [np.nan], "x2": [1.0], "y": [0.2]}).to_csv(
        tmp_path / "test.csv", index=False
    )

    with pytest.raises(ValueError, match="finite|hữu hạn"):
        load_data("test", data_dir=tmp_path)
