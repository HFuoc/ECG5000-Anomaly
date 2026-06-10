# Phát hiện bất thường trong tín hiệu Điện tâm đồ (ECG) bằng Variational Autoencoder (VAE)

Dự án Nghiên cứu  ứng dụng phương pháp Học sâu không giám sát (Unsupervised Deep Learning) để phát hiện các nhịp tim bất thường dựa trên cấu trúc mạng **Variational Autoencoder (VAE)** và kỹ thuật **Beta-Warmup**.

## 📊 Tổng quan Dự án

Dữ liệu y tế thực tế thường gặp phải tình trạng mất cân bằng nghiêm trọng (số ca bình thường chiếm đại đa số, ca bệnh lý rất khan hiếm). Dự án này tiếp cận theo hướng **Phát hiện bất thường (Anomaly Detection)**: Mô hình mạng VAE chỉ được huấn luyện trên các tín hiệu ECG khỏe mạnh (Normal) để học cách tái tạo cấu trúc của chúng. Khi gặp một tín hiệu biến dạng (Anomaly), mô hình không thể tái tạo tốt và sẽ trả về sai số tái tạo (Reconstruction Error) lớn, từ đó giúp phát hiện ra bất thường.

### 🚀 Kết quả nổi bật (Ablation Study)
Qua thực nghiệm phân tích thành phần đối với kích thước không gian tiềm ẩn (`LATENT_DIM`), dự án đạt hiệu năng tối ưu tại **`LATENT_DIM = 4`**.

## 📁 Cấu trúc Dự án (Project Tree)

```text
nckh-project/
├── data/
│   ├── raw/                    # Dữ liệu gốc (trước khi tiền xử lý)
│   └── processed/              # Dữ liệu đã qua tiền xử lý, chuẩn bị cho huấn luyện
├── demo/                       # Ứng dụng giao diện trực quan cho người dùng
│   ├── app.py                  # API Backend và kịch bản khởi chạy ứng dụng
│   ├── frontend/               # Mã nguồn giao diện Frontend (React)
│   ├── static_dist/            # Các tệp tĩnh đã build cho giao diện web
│   └── README.md               # Tài liệu hướng dẫn riêng cho phần Demo
├── docs/                       # Tài liệu thiết kế và kiến trúc hệ thống
│   └── ecg_education_design.md # Thiết kế chức năng giáo dục hỗ trợ phân tích ECG
├── notebooks/                  # Các file Jupyter Notebook thử nghiệm ban đầu
│   └── 01_eda.ipynb            # Phân tích dữ liệu thăm dò (Exploratory Data Analysis)
├── paper/                      # Chứa tài liệu, bài báo cáo khoa học và tài liệu tham khảo
│   ├── draft_v1.md             # Bản nháp chi tiết báo cáo kết quả nghiên cứu (V1)
│   ├── ieee_paper_draft.md     # Bản thảo định dạng IEEE
│   ├── paper_extended.md       # Phiên bản báo cáo mở rộng với đầy đủ chi tiết
│   ├── RELATED_WORK.md         # Phân tích các nghiên cứu liên quan
│   └── REFERENCES.md           # Danh sách các tài liệu tham khảo
├── results/                    # Lưu trữ kết quả huấn luyện và thực nghiệm
│   ├── baseline/               # Kết quả so sánh với các mô hình cơ sở
│   ├── checkpoints/            # Lưu trọng số mô hình tốt nhất (.pth)
│   ├── experiments/            # Cấu hình và kết quả thực nghiệm mở rộng
│   ├── figures/                # Các biểu đồ kết quả (evaluation, ablation)
│   └── logs/                   # Lịch sử huấn luyện mô hình
├── src/                        # Mã nguồn chính của dự án
│   ├── evaluate.py             # Đánh giá mô hình mạng VAE trên tập Test
│   ├── evaluate_baselines.py   # Kịch bản đánh giá cho các mô hình cơ sở
│   ├── model.py                # Kiến trúc mạng VAE và định nghĩa hàm Loss (Beta-VAE)
│   ├── plot_ablation.py        # Script vẽ biểu đồ so sánh hiệu năng Ablation Study
│   └── train.py                # Kịch bản huấn luyện mô hình với Beta-Warmup
├── crop_images.py              # Script hỗ trợ cắt xén/chuẩn hóa hình ảnh
├── download_data.py            # Script tự động tải dữ liệu ECG từ nguồn
├── ecg.rar                     # File nén chứa tập dữ liệu ECG (dự phòng)
└── README.md                   # Hướng dẫn chính của dự án (Tài liệu này)
```

## 🛠️ Hướng dẫn Cài đặt & Sử dụng

### 1. Kích hoạt Môi trường ảo
Dự án sử dụng bộ thư viện Python được cô lập trong môi trường ảo `nckh` (sử dụng venv). Mở terminal tại thư mục gốc dự án và chạy lệnh sau để kích hoạt:

**Trên Windows (PowerShell):**
```powershell
.\nckh\Scripts\Activate.ps1
```

**Trên Linux / MacOS:**
```bash
source nckh/bin/activate
```

### 2. Huấn luyện Mô hình (Training)
Để huấn luyện lại mô hình mạng VAE với cấu hình mong muốn, bạn chỉnh sửa biến `LATENT_DIM` trong file `src/train.py` và chạy lệnh:
```powershell
python src/train.py
```
Sau khi chạy xong, trọng số tốt nhất sẽ tự động được lưu tại `results/checkpoints/best_model.pth`.

### 3. Đánh giá Mô hình (Evaluation)
Để kiểm tra các chỉ số Precision, Recall, F1-score và vẽ biểu đồ phân phối lỗi/đường cong ROC, chỉnh sửa `LATENT_DIM` trong file `src/evaluate.py` trùng với mô hình đã train, sau đó chạy:
```powershell
python src/evaluate.py
```

### 4. Vẽ biểu đồ Ablation Study
Để trực quan hóa chuỗi thực nghiệm thay đổi số chiều từ 2 -> 4 -> 16 vào bài báo cáo khoa học, chạy kịch bản vẽ biểu đồ đường:
```powershell
python src/plot_ablation.py
```
Biểu đồ đường sắc nét sẽ được xuất ra thư mục `results/figures/ablation_latent_dim.png`.

### 5. Khởi chạy Ứng dụng Demo
Dự án tích hợp giao diện trực quan cho người dùng cuối. Bạn có thể sử dụng giao diện này để demo tương tác thời gian thực.
Tham khảo thêm tài liệu trong thư mục `demo/` và khởi chạy với:
```powershell
python demo/app.py
```
Sau khi dòng log hiển thị, mở trình duyệt web và truy cập địa chỉ mặc định.

## 📝 Nhật ký Phát triển & Tài liệu đi kèm

Toàn bộ nội dung khoa học, phương pháp luận chi tiết, kiến trúc toán học mạng VAE cùng phần thảo luận chuyên sâu đã được biên soạn đầy đủ trong các tệp tại thư mục `paper/` (đặc biệt là `ieee_paper_draft.md` và `paper_extended.md`).

Thiết kế kiến trúc ứng dụng giao diện tích hợp giáo dục y tế cũng đã được hoàn thiện tại `docs/ecg_education_design.md`. Các thư mục `results/baseline/` và `results/experiments/` hiện đang được giữ cấu trúc chuẩn hóa nhằm mục đích mở rộng so sánh đối chứng với các thuật toán truyền thống khác.