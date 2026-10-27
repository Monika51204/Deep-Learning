# outputs — TV3 tạo log; TV4 tạo hình
- `logs/`: kết quả từng lần chạy A/B/C, gồm tham số, seed, metrics, thời gian.
- `tables/`: bảng tổng hợp.
- `checkpoints/`: mô hình `.pt/.pth` tốt nhất theo validation.
- `predictions/`: y thật và y dự đoán cho biểu đồ.
Không chỉnh sửa thủ công số liệu trong log. Có thể không commit checkpoint lớn nếu ZIP vượt giới hạn, nhưng cần có lệnh tái lập.
