"""Hàm toán học tính mật độ Gaussian hai chiều."""

from __future__ import annotations

import numpy as np


def gaussian_2d(
    x: np.ndarray,
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
) -> np.ndarray:
    """Tính mật độ Gaussian 2D cho các điểm đầu vào.

    Args:
        x: Mảng shape (N, 2), mỗi hàng là một điểm (x1, x2).
        mean: Vector trung bình shape (2,), mặc định [0, 0].
        cov: Ma trận hiệp phương sai shape (2, 2),
             mặc định [[0.3, -0.5], [-0.5, 2.0]].

    Returns:
        Mảng mật độ xác suất shape (N, 1), kiểu float64.

    Raises:
        ValueError: Nếu kích thước đầu vào sai hoặc ma trận
                    hiệp phương sai không hợp lệ.
    """

    # Chuyển dữ liệu đầu vào sang float64
    x = np.asarray(x, dtype=np.float64)

    # Gán vector trung bình mặc định nếu không được cung cấp
    mean = np.asarray(
        [0.0, 0.0] if mean is None else mean,
        dtype=np.float64,
    )

    # Gán ma trận hiệp phương sai mặc định theo đề bài
    cov = np.asarray(
        [[0.3, -0.5], [-0.5, 2.0]] if cov is None else cov,
        dtype=np.float64,
    )

    # Kiểm tra đầu vào có đúng shape (N, 2) hay không
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must have shape (N, 2)")

    # Kiểm tra kích thước vector trung bình và ma trận hiệp phương sai
    if mean.shape != (2,) or cov.shape != (2, 2):
        raise ValueError("mean must be (2,), cov must be (2, 2)")

    # Kiểm tra các giá trị đầu vào có phải số hữu hạn hay không
    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(cov))
    ):
        raise ValueError("x, mean and cov must contain finite values")

    # Tính định thức của ma trận hiệp phương sai
    det = float(np.linalg.det(cov))

    # Ma trận hiệp phương sai phải đối xứng và xác định dương
        # Kiểm tra tính đối xứng và định thức dương
    if det <= 0 or not np.allclose(cov, cov.T):
        raise ValueError("cov must be symmetric positive definite")

        # Kiểm tra ma trận có xác định dương hay không
    try:
        np.linalg.cholesky(cov)
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "cov must be symmetric positive definite"
    ) from exc

    # Tính độ lệch của từng điểm so với vector trung bình
    # Shape kết quả: (N, 2)
    delta = x - mean

    # Tính dạng toàn phương:
    # (x - mean)^T * cov^(-1) * (x - mean)
    # Mỗi điểm tương ứng với một giá trị, shape (N,)
    quad = np.einsum(
        "ni,ij,nj->n",
        delta,
        inv_cov,
        delta,
    )

    # Tính mật độ Gaussian theo công thức:
    # f(x) = exp(-0.5 * quad) / (2 * pi * sqrt(det(cov)))
    values = np.exp(-0.5 * quad) / (
        2.0 * np.pi * np.sqrt(det)
    )

    # Chuyển kết quả từ shape (N,) sang (N, 1)
    # Mỗi hàng chứa mật độ xác suất của một điểm
    return values.reshape(-1, 1)
