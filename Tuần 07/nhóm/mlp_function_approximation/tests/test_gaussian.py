
"""Kiểm tra hàm Gaussian có shape đúng và giá trị hợp lệ."""

import numpy as np
import pytest

from src.data.gaussian import gaussian_2d


def test_gaussian_shape_and_values():
    """Kiểm tra kích thước, tính hữu hạn và tính không âm."""
    x = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    y = gaussian_2d(x)

    # Hai điểm đầu vào phải tạo ra hai giá trị đầu ra
    assert y.shape == (2, 1)

    # Các giá trị mật độ phải là số hữu hạn
    assert np.all(np.isfinite(y))

    # Mật độ xác suất không được âm
    assert np.all(y >= 0)


def test_gaussian_density_at_mean():
    """Kiểm tra mật độ tại điểm bằng vector trung bình."""
    x = np.array([[0.0, 0.0]])

    y = gaussian_2d(x)

    # Tại mean = [0, 0], số mũ bằng 0 nên exp(0) = 1
    expected = 1.0 / (2.0 * np.pi * np.sqrt(0.35))

    assert np.isclose(y[0, 0], expected)


def test_gaussian_invalid_input_shape():
    """Kiểm tra hàm báo lỗi khi đầu vào sai kích thước."""
    x = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        gaussian_2d(x)


def test_gaussian_invalid_covariance():
    """Kiểm tra hàm báo lỗi với ma trận hiệp phương sai không hợp lệ."""
    x = np.array([[0.0, 0.0]])

    # Ma trận này có định thức âm
    cov = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    with pytest.raises(ValueError):
        gaussian_2d(x, cov=cov)


def test_gaussian_non_positive_definite_covariance():
    """Kiểm tra trường hợp định thức dương nhưng ma trận không xác định dương."""
    x = np.array([[0.0, 0.0]])

    # Định thức bằng 1 nhưng ma trận không xác định dương
    cov = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
    ])

    with pytest.raises(ValueError):
        gaussian_2d(x, cov=cov)
