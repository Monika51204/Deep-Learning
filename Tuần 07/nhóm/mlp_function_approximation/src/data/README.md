# src/data — TV1

## Nhiệm vụ
- `gaussian.py`: cài đặt hàm mật độ Gaussian 2 chiều theo công thức; kiểm tra đầu vào và tham số cần thiết.
- `generate_data.py`: lấy mẫu x1, x2 trong miền đề bài, tính y, chia train/validation/test độc lập và lưu CSV.
- CSV thống nhất các cột `x1,x2,y`; metadata lưu seed, miền, μ, C và kích thước split.

## Dữ liệu 
- Các tệp được lưu trong data/processed/
    - `train_pool.csv` (2.000 mẫu): Tệp huấn luyện, thực nghiệm A dùng các prefix cố định của tập này 
    - `val.csv` (500 mẫu): Chọn cấu hình và theo dõi quá trình huấn luyện 
    - `test.csv` (100 mẫu): Đánh giá cuối cùng, không dùng để chọn cấu hình 
    - `metadata.json`: Lưu seed, miền lấy mẫu, tham số Gaussian và kích thước các tập 

## Cách tải dữ liệu

```python
from src.data.load_data import load_data

X_train, y_train = load_data("train_pool", n_samples=400)
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")
```

```python
print(X_train.shape, y_train.shape)  # (400, 2), (400, 1)
print(X_val.shape, y_val.shape)      # (500, 2), (500, 1)
print(X_test.shape, y_test.shape)    # (100, 2), (100, 1)
```