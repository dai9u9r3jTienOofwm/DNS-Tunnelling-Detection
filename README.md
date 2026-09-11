# Phát Hiện DNS Tunneling Bằng Machine Learning Và 1D-CNN

## 1. Cài đặt môi trường
Chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
2. Dữ liệu
Tải dữ liệu từ repository: https://github.com/aasthac67/DNS-Tunneling-Detection/tree/main/Tool và lưu vào thư mục data/raw/.

Chuẩn hóa dữ liệu thành định dạng CSV gồm 2 cột bắt buộc:

domain: Tên miền cần phân tích.

label: Nhãn phân loại (0 cho tên miền hợp lệ, 1 cho DNS Tunneling).

Lưu file đã xử lý vào đường dẫn: data/processed/train.csv.

3. Chạy chương trình
Huấn luyện mô hình
Bash
python run.py --data data/processed/train.csv
Dự đoán nhanh tên miền
Bash
python predict.py suspicious.example.com
Đánh giá mô hình
Bash
python -m src.evaluate --data data/processed/test.csv
