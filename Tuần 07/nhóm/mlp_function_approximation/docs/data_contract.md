# Data contract (cả nhóm thống nhất)

- Schema mỗi CSV: `x1,x2,y`; không có index thừa.
- x1, x2: hai đặc trưng đầu vào, biểu diễn tọa độ của điểm trong miền lấy mẫu.
- y: giá trị Gaussian tương ứng với mỗi điểm đầu vào.
- Interface đọc dữ liệu: `load_data(split, n_samples=None, data_dir=None) -> (features, targets)`
- features: float32, shape `(N,2)`.
- targets: float32, shape `(N,1)`.
- MLP: `MLP(input_dim=2, hidden_dims=(...), output_dim=1, activation='tanh')`.
- `model(X)` trả tensor `(N,1)`.
- Metrics: dict `{'mse': float, 'mae': float, 'rmse': float}`.
- Train interface dự kiến: `train_one_run(model, train_loader, val_loader, config, checkpoint_path) -> (best_model, history_df)`.
- Test split cố định 100 mẫu; chỉ đánh giá cuối.