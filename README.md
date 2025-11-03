# U-Net for Polyp Segmentation (Kvasir-SEG)
- Dự án này là một triển khai mô hình **U-Net** bằng **PyTorch** để thực hiện **semantic segmentation** (phân đoạn ngữ nghĩa) cho các polyp y tế. Mô hình được huấn luyện trên bộ dữ liệu [Kvasir-SEG](https://paperswithcode.com/dataset/kvasir-seg).
- **Mục tiêu là dự đoán một mặt nạ (mask) nhị phân, phân biệt đâu là vùng polyp và đâu là vùng nền (background)**.

## 📋 Cấu trúc thư mục

```
.
├── data/
│   └── Kvasir-SEG/
│       ├── images/       (Chứa ảnh gốc)
│       └── masks/        (Chứa ảnh mặt nạ)
├── trained_models/
│   └── best_unet.pt      (Checkpoint của model tốt nhất)
├── dataset.py            (Định nghĩa custom Dataset)
├── model.py              (Định nghĩa kiến trúc U-Net)
├── train.py              (Script để huấn luyện model)
├── test.py               (Script để kiểm thử và hiển thị dự đoán)
└── requirements.txt      (Tệp chứa các thư viện cần thiết)
```

## ✨ Tính năng chính

* **Kiến trúc U-Net:** Triển khai U-Net từ đầu bao gồm `ConvBlock`, `Encoder`, `Decoder` và các kết nối skip (skip connections).
* **Custom Dataset:** Sử dụng class `KvasirDatasetAugmented` để tải và xử lý ảnh.
* **Data Augmentation:** Tích hợp các phép tăng cường dữ liệu (elastic transform, flips) để tăng tính đa dạng của dữ liệu huấn luyện và cải thiện độ tổng quát của mô hình.
* **Training & Validation:** Vòng lặp huấn luyện đầy đủ với cả bước training và validation, tự động lưu lại mô hình có (validation loss) tốt nhất.
* **Inference:** Script `test.py` để tải checkpoint và hiển thị so sánh trực quan giữa "Ảnh gốc", "Đáp án đúng" (Ground Truth) và "Dự đoán" (Prediction).

## 🛠️ Cài đặt
1.  Clone repository này:
2.  Cài đặt các thư viện cần thiết:  :  pip install -r requirements.txt

3.  **Chuẩn bị dữ liệu:**
- Tải bộ dữ liệu Kvasir-SEG và đặt vào thư mục `./data/` theo cấu trúc đã mô tả ở trên.

## 🚀 Cách sử dụng
### 1. Huấn luyện (Training)
- Chạy script `train.py` để bắt đầu quá trình huấn luyện:
- Mô hình sẽ được huấn luyện với các tham số mặc định (ví dụ: 100 epochs, batch size 64). Checkpoint có validation loss tốt nhất sẽ được lưu tại `./trained_models/best_unet.pt`.
### 2. Kiểm thử (Inference)
- `test.py` để chạy dự đoán trên một ảnh và mặt nạ cụ thể.

## Ảnh để test trong file 
- `./demo/test`

## Demo
![Kết quả với ảnh 2.png](E./demo/result/2.png)
![Kết quả với ảnh 4.png](E./demo/result/4.png)