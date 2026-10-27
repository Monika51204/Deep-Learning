# data/processed — TV1
Sau khi chạy `python -m src.data.generate_data`, thư mục gồm:
- `train_pool.csv`: tối đa 2000 mẫu; các thực nghiệm A dùng prefix cố định.
- `val.csv`: validation độc lập.
- `test.csv`: 100 mẫu độc lập, chỉ đánh giá cuối.
- `metadata.json`: seed, miền, μ, C, kích thước và schema.
Schema mỗi CSV: `x1,x2,y`.
