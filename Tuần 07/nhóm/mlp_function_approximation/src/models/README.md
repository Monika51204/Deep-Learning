# src/models — TV2

`mlp.py` chứa model PyTorch độc lập với quy trình train. Input shape `(batch,2)`, output `(batch,1)`.
Không áp activation sigmoid ở output vì đây là bài toán hồi quy; output tuyến tính cho phép xấp xỉ giá trị liên tục.
