"""Vẽ biểu đồ thực nghiệm từ các file log CSV."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]



def plot_error_vs_parameter(csv_path: str | Path, x_col: str, y_col: str,
                            title: str, output_path: str | Path) -> None:
    """Vẽ metric theo biến độc lập từ log; dùng cho A/B/C."""
    data = pd.read_csv(csv_path)
    ax = data.plot(x=x_col, y=y_col, marker="o", legend=False)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def main() -> None:
    """TODO: gọi plot_error_vs_parameter cho các log A/B/C hiện có."""
    print("Chưa có log thực nghiệm. Chạy runners trước rồi hoàn thiện phần tổng hợp hình.")


if __name__ == "__main__":
    main()
