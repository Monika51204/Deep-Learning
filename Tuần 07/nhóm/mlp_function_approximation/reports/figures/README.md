# reports/figures

Thư mục do nhóm quản lý theo phân công trong workbook. Ghi rõ mục đích, đầu vào, đầu ra, quy tắc đặt tên và lệnh tái lập tại đây. 

**Owner dự kiến:** TV4.

## Quy tắc chung

- Tên file sử dụng chữ thường, dấu gạch dưới `_` và định dạng `.png`.
- Lưu toàn bộ hình trong `reports/figures/`; giữ nguyên tên file khi cập nhật hình cùng mục đích.
- Script trực quan hóa phải tạo thư mục đầu ra nếu chưa tồn tại và ghi đè hình cũ khi chạy lại.
- Các hình phải được tạo từ dữ liệu, công thức và tham số thực tế của dự án.
- Chạy lệnh tái lập từ thư mục gốc của dự án để bảo đảm đường dẫn đầu vào và đầu ra nhất quán.

## 1. `data_splits_3d.png`

- **Mục đích:** Trực quan hóa và so sánh phân bố các điểm dữ liệu `(x1, x2, y)` giữa tập train, validation và test
- **Đầu vào:** `data/processed/train_pool.csv`, `data/processed/val.csv`, `data/processed/test.csv`.
- **Đầu ra:** File ảnh `reports/figures/data_splits_3d.png`, biểu diễn scatter 3D với các trục `x1`, `x2`, `y`; sử dụng màu sắc hoặc chú giải riêng cho từng split.
- **Lệnh tái lập:** Chạy từ thư mục gốc dự án sau khi dữ liệu đã được tạo:
  ```bash
  python -m src.visualization.plot_data
  ```

## 2. `data_true_surface.png`

- **Mục đích:** Biểu diễn bề mặt hàm Gaussian 2D làm hàm mục tiêu, hỗ trợ quan sát hình dạng và đặc điểm của hàm mà mô hình MLP cần xấp xỉ.
- **Đầu vào:** `data/processed/metadata.json` chứa các tham số Gaussian và miền khảo sát; công thức tính toán từ `src/data/gaussian.py`.
- **Đầu ra:** File ảnh `reports/figures/data_true_surface.png`, biểu diễn bề mặt Gaussian theo hai biến `x1`, `x2` và giá trị hàm `y`.
- **Lệnh tái lập:** Chạy từ thư mục gốc dự án:
  ```bash
  python -m src.visualization.plot_data
  ```

## 3. `data_distributions.png`

- **Mục đích:** Khảo sát và so sánh phân phối của các biến `x1`, `x2`, `y` giữa tập train, validation và test; hỗ trợ phát hiện bất thường hoặc khác biệt phân phối giữa các tập.
- **Đầu vào:** `data/processed/train_pool.csv`, `data/processed/val.csv`, `data/processed/test.csv`.
- **Đầu ra:** File ảnh `reports/figures/data_distributions.png`, thể hiện phân phối của ba biến theo từng split bằng biểu đồ phù hợp, có tiêu đề, nhãn trục và chú giải.
- **Lệnh tái lập:** Chạy từ thư mục gốc dự án:
  ```bash
  python -m src.visualization.plot_data
  ```


