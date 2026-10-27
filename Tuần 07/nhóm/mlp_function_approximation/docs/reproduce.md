# Hướng dẫn tái lập (hoàn thiện khi tích hợp)
1. Cài môi trường bằng `pip install -r requirements.txt`.
2. Sinh dữ liệu: `python -m src.data.generate_data`.
3. Chạy A/B/C bằng các runner trong `src/experiments/`.
4. Tạo biểu đồ từ log bằng `python -m src.visualization.plot_results`.
5. Kiểm tra cấu hình, seed, log, checkpoint và phiên bản thư viện trước khi so sánh.

Lưu ý: runner A/B/C hiện là scaffold, chưa phải pipeline hoàn chỉnh; phải thay TODO/NotImplementedError trước khi dùng tạo kết quả báo cáo.
