# TV1 — Dữ liệu và EDA

## Phạm vi
Gaussian, sinh/chia dữ liệu, EDA, data contract.
TV1 phụ trách: 
- Xây dựng hàm Gaussian 2D
- Sinh và chia dữ liệu phục vụ thực nghiệm
- Khảo sát dữ liệu ban đầu (EDA) 
- Cập nhật data contract nhằm thống nhất giao diện dữ liệu giữa các thành viên trong nhóm.

## Nội dung cần điền
### 1. Phương pháp/code thực tế
- Triển khai hàm Gaussian 2D trong `src/data/gaussian.py`.
- Sinh và chia dữ liệu trong `src/data/generate_data.py`, gồm train pool (2.000 mẫu), validation (400 mẫu), test (100 mẫu); seed lần lượt 42, 43, 44.
- Thống nhất schema CSV `x1,x2,y`, kiểu dữ liệu sau khi đọc và interface giữa các thành viên. 
- Thực hiện EDA để khảo sát phân phối dữ liệu và trực quan hóa hàm Gaussian.

### 2. Kết quả/log minh chứng
Theo cấu hình mặc định, tổng dữ liệu dự kiến là 2.500 mẫu. Cần chạy `python -m src.data.generate_data` và kiểm tra các file CSV, `metadata.json` để xác nhận kết quả.

### 3. Hình/bảng cần chèn
- `data_splits_3d.png`: phân bố dữ liệu theo ba tập.
- `data_true_surface.png`: bề mặt Gaussian 2D.
- `data_distributions.png`: phân phối `x1`, `x2`, `y`.
- Bảng thống kê kích thước, seed và schema các tập dữ liệu.

### 4. Giới hạn/ghi chú
Dữ liệu được sinh nhân tạo, chưa phản ánh đầy đủ dữ liệu thực tế. Các thực nghiệm dùng prefix của cùng train pool, validation dùng chọn cấu hình, test 100 mẫu chỉ dùng đánh giá cuối.

### 5. Tài liệu đã sử dụng
- `src/data/gaussian.py`.
- `src/data/generate_data.py`.
- `docs/data_contract.md`.
- `data/processed/metadata.json`.
