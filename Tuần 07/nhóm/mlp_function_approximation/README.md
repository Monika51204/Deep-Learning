# MLP Function Approximation — PyTorch

## Mục tiêu
Xấp xỉ hàm mật độ Gaussian 2D bằng MLP tự triển khai với PyTorch; khảo sát có kiểm soát:
- A: số mẫu huấn luyện (200, 400, ..., 2000)
- B: số neuron của một tầng ẩn (1..10)
- C: số tầng ẩn (1..4)

## Chủ sở hữu theo workstream
- TV1: `src/data/`, `data/`, dữ liệu và EDA.
- TV2: `src/models/`, `src/training/`, `src/evaluation/`, `tests/`.
- TV3: `src/experiments/`, `configs/baseline.json`, `outputs/logs/`, bảng tổng hợp.
- TV4: tích hợp repo, `src/visualization/`, `reports/`, `slides/`, tài liệu hướng dẫn.

## Cài đặt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Thứ tự chạy dự kiến
```bash
python -m src.data.generate_data
python -m src.experiments.run_exp_a
python -m src.experiments.run_exp_b
python -m src.experiments.run_exp_c
python -m src.visualization.plot_results
```
Các runner đang là scaffold có TODO; cần hoàn thiện trước khi chạy thực nghiệm chính thức.

## Nguyên tắc khoa học
- Test độc lập đúng 100 mẫu; không dùng test để chọn siêu tham số.
- Chọn checkpoint/cấu hình theo validation loss.
- Trong mỗi nhóm A/B/C chỉ đổi biến độc lập đang khảo sát.
- Lưu seed, siêu tham số, metric và thời gian train vào CSV.
- Không đưa số liệu giả vào báo cáo; chỉ dùng log chạy thực tế.
