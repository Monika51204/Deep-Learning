# src/training — TV2

Huấn luyện phải dùng `model.train()` ở train phase; `model.eval()` và `torch.no_grad()` ở validation.
Theo dõi train/validation loss, lưu checkpoint tốt nhất theo validation loss, không chọn theo test.
