# data/raw — TV1

## 1. Mô tả dữ liệu thô và nguồn dữ liệu

Dự án không sử dụng dataset bên ngoài. Dữ liệu được sinh tự động bằng script `src/data/generate_data.py` dựa trên công thức Gaussian 2D được định nghĩa trong `src/data/gaussian.py`.

Các điểm dữ liệu đầu vào gồm hai biến `x1` và `x2`, được lấy mẫu ngẫu nhiên độc lập theo phân phối đều trên các miền:

- `x1 ∈ [-2, 2]`.
- `x2 ∈ [-4, 4]`.

Giá trị mục tiêu `y` được tính từ hàm Gaussian 2D tương ứng. Quy trình sinh dữ liệu sử dụng seed xác định để có thể tái lập kết quả.

## 2. Nội dung thư mục

Thư mục `data/raw/` được giữ làm nơi lưu trữ dữ liệu nguồn hoặc các artefact gốc nếu phát sinh trong quá trình thực hiện dự án. Do dữ liệu hiện tại được sinh trực tiếp bằng script, thư mục này có thể để trống.

Các file CSV được tạo ra từ quy trình sinh dữ liệu được lưu trong `data/processed/`, không lưu trực tiếp trong `data/raw/`.

## 3. Quy tắc quản lý

- Không yêu cầu tải dữ liệu từ nguồn bên ngoài.
- Không lưu file cá nhân hoặc đường dẫn tuyệt đối trong thư mục.
- Nếu bổ sung dữ liệu nguồn hoặc artefact gốc, cần ghi rõ nguồn, định dạng và mục đích sử dụng.
- Giữ nguyên script sinh dữ liệu và seed đã thống nhất để bảo đảm khả năng tái lập các thực nghiệm.
