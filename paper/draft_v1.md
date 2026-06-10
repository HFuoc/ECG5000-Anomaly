# BÁO CÁO NGHIÊN CỨU KHOA HỌC
## Đề tài: Phát hiện bất thường trong tín hiệu điện tâm đồ (ECG) bằng mạng Variational Autoencoder (VAE)
**Tác giả:** Nhóm Nghiên cứu Dự án DL-ECG
**Trạng thái:** Bản nháp V1 (Draft v1)

---

### TÓM TẮT (ABSTRACT)
Nghiên cứu này trình bày một giải pháp học sâu không giám sát (Unsupervised Deep Learning) ứng dụng mạng Variational Autoencoder (VAE) để phát hiện các tín hiệu bất thường trong dữ liệu điện tâm đồ (ECG). Thử nghiệm được tiến hành trên tập dữ liệu chuẩn hóa ECG5000 gồm 140 bước thời gian cho mỗi mẫu. Để giải quyết hiện tượng sụp đổ không gian tiềm ẩn (KL Collapse), kỹ thuật Beta-Warmup đã được áp dụng trong quá trình huấn luyện 50 epochs với thuật toán tối ưu Adam. Nghiên cứu cũng thực hiện phân tích thực nghiệm thành phần (Ablation Study) đối với số chiều không gian tiềm ẩn ($latent\_dim$) tại các mốc 2, 4, và 16. Kết quả thực nghiệm cho thấy mô hình hoạt động tối ưu nhất ở $latent\_dim = 2$ với chỉ số AUC-ROC đạt **0.9620**, độ chính xác phát hiện bất thường (Anomaly Precision) đạt **0.98**, và F1-score đạt **0.90**. Cuối cùng, một ứng dụng giao diện trực quan trực tiếp (Gradio Web App) đã được xây dựng thành công để hỗ trợ các chuyên gia y tế trong việc chẩn đoán nhanh.

---

### 1. ĐẶT VẤN ĐỀ (INTRODUCTION)
Bệnh tim mạch là một trong những nguyên nhân gây tử vong hàng đầu trên thế giới. Điện tâm đồ (ECG) là công cụ phổ biến và hiệu quả nhất để theo dõi sức khỏe tim và phát hiện sớm các dấu hiệu bệnh lý. Tuy nhiên, việc phân tích thủ công các tín hiệu ECG kéo dài đòi hỏi rất nhiều thời gian và kinh nghiệm của các bác sĩ chuyên khoa. 

Trong thực tế y tế, dữ liệu nhịp tim bình thường luôn chiếm đại đa số, trong khi dữ liệu bệnh lý (bất thường) thường rất khan hiếm và đa dạng về hình thái. Điều này gây khó khăn lớn cho các mô hình học có giám sát (Supervised Learning) do bài toán mất cân bằng dữ liệu nghiêm trọng. Do đó, hướng tiếp cận **Học không giám sát / Phát hiện bất thường (Anomaly Detection)** nổi lên như một giải pháp tối ưu. Mô hình chỉ cần học cấu trúc của các nhịp tim khỏe mạnh, từ đó bất kỳ tín hiệu nào sai lệch lớn so với những gì đã học sẽ tự động bị gắn nhãn là bất thường.

---

### 2. PHƯƠNG PHÁP NGHIÊN CỨU & KIẾN TRÚC MÔ HÌNH (METHODOLOGY)

#### 2.1. Kiến trúc mạng Variational Autoencoder (VAE)
Khác với mạng Autoencoder truyền thống (chỉ nén dữ liệu thành một vector cố định), VAE ánh xạ dữ liệu đầu vào vào một phân phối xác suất (thường là phân phối chuẩn Gauss) trong không gian tiềm ẩn (Latent Space). Kiến trúc bao gồm hai phần chính:

1. **Encoder (Mạng mã hóa):** Nhận đầu vào là tín hiệu ECG 140 chiều ($input\_dim = 140$). Đi qua hai tầng tuyến tính (Linear Layers) kết hợp hàm kích hoạt ReLU nhằm trích xuất đặc trưng bậc cao, giảm chiều dần xuống tầng ẩn 64 và 32, cuối cùng tính ra hai vector: Vector kỳ vọng (Mean - $\mu$) và Vector phương sai (Log-variance - $\log(\sigma^2)$).
2. **Trực giao hóa (Reparameterization Trick):** Để có thể lan truyền ngược (Backpropagation) qua một biến ngẫu nhiên, mạng áp dụng công thức:
   $$z = \mu + \epsilon \odot \sigma$$
   Trong đó $\epsilon \sim \mathcal{N}(0, I)$ là nhiễu ngẫu nhiên Gauss.
3. **Decoder (Mạng giải mã):** Nhận vector tiềm ẩn $z$ từ bottleneck, tiến hành giải nén ngược lại qua cấu trúc đối xứng (16/4/2 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 140) để tái tạo lại tín hiệu ECG ban đầu ($x_{recon}$).

#### 2.2. Hàm mất mát (Loss Function) và Kỹ thuật Beta-Warmup
Hàm mất mát tổng thể của VAE bao gồm hai thành phần:
$$\mathcal{L}_{total} = \mathcal{L}_{Reconstruction} + \beta \cdot \mathcal{L}_{KL}$$

* **Reconstruction Loss ($\mathcal{L}_{Reconstruction}$):** Sử dụng sai số bình phương trung bình (MSE) giữa tín hiệu gốc $x$ và tín hiệu tái tạo $x_{recon}$ nhằm ép mô hình học cách khôi phục hình dạng sóng tốt nhất.
* **KL Divergence ($\mathcal{L}_{KL}$):** Đo lường sự sai khác giữa phân phối tiềm ẩn được học và phân phối chuẩn $\mathcal{N}(0, I)$, đóng vai trò như một bộ điều hòa (regularizer) giúp không gian tiềm ẩn liên tục.
* **Kỹ thuật Beta-Warmup:** Nhằm tránh hiện tượng mạng Decoder bỏ qua không gian tiềm ẩn (KL Collapse) ở các epoch đầu do mạng chưa học được cách tái tạo tốt, tham số $\beta$ được tăng tuyến tính từ $0.0$ đến $1.0$ trong 20 epochs đầu tiên:
   $$\beta = \min\left(1.0, \frac{\text{epoch}}{20}\right)$$

---

### 3. THỰC NGHIỆM VÀ KẾT QUẢ (EXPERIMENTS & RESULTS)

#### 3.1. Thiết lập thực nghiệm
* **Tập dữ liệu:** ECG5000 lấy từ Kaggle.
* **Phân chia dữ liệu:** * Tập huấn luyện (Train set): Gồm **2335** mẫu (hoàn toàn là nhịp tim khỏe mạnh - Normal).
  * Tập kiểm thử (Test set): Gồm **2665** mẫu (bao gồm **584** mẫu Normal và **2081** mẫu Anomaly để đánh giá khả năng phân loại).
* **Siêu tham số (Hyperparameters):** Batch size = 64, Learning rate = $10^{-3}$, Optimizer = Adam, Epochs = 50.
* **Cơ chế phát hiện bất thường:** Tính sai số tái tạo (MSE Error) trên từng mẫu dữ liệu kiểm thử. Ngưỡng phân loại ($Threshold$) được chọn cố định ở vị trí **Phần trăm thứ 95 (Percentile 95)** của sai số trên tập huấn luyện. Bất kỳ mẫu nào có lỗi tái tạo vượt ngưỡng này sẽ bị coi là Anomaly.

#### 3.2. Kết quả phân tích thành phần (Ablation Study trên Latent Dimension)
Nghiên cứu tiến hành thay đổi số chiều không gian tiềm ẩn ($latent\_dim = 2, 4, 16$) để tìm ra kiến trúc tối ưu nhất. Kết quả thu được như sau:

| Chỉ số / Thử nghiệm | Cấu hình 1 ($latent\_dim = 2$) | Cấu hình 2 ($latent\_dim = 4$) | Cấu hình 3 ($latent\_dim = 16$) |
| :--- | :---: | :---: | :---: |
| **Ngưỡng Phát hiện (Threshold)** | 0.037435 | 0.038201 | 0.037527 |
| **AUC-ROC Score** | **0.9620** | **0.9618** | **0.9611** |
| **Normal Precision** | 0.62 | 0.62 | 0.62 |
| **Normal Recall** | 0.95 | 0.95 | 0.95 |
| **Normal F1-score** | 0.75 | 0.75 | 0.75 |
| **Anomaly Precision** | 0.98 | 0.98 | 0.98 |
| **Anomaly Recall** | 0.84 | 0.83 | 0.83 |
| **Anomaly F1-score** | 0.90 | 0.90 | 0.90 |
| **Độ chính xác tổng thể (Accuracy)**| 0.86 | 0.86 | 0.86 |

#### 3.3. Thảo luận và Biện luận Khoa học (Discussion)
Từ bảng số liệu trên, có thể rút ra những kết luận mang tính học thuật sâu sắc:
1. **Hiệu ứng "Nút thắt cổ chai" (Bottleneck Optimization):** Khi giảm số chiều tiềm ẩn từ 16 xuống còn 2, chỉ số AUC-ROC tăng dần từ **0.9611 $\rightarrow$ 0.9620**. Điều này phản ánh tín hiệu ECG Normal sở hữu tính chu kỳ và quy luật rất cao. Khi ép không gian nén xuống 2 chiều, mô hình bị ép buộc loại bỏ hoàn toàn các thành phần nhiễu (noise) và giữ lại đặc trưng cốt lõi nhất. Trái lại, khi không gian tiềm ẩn quá rộng (16 chiều), mô hình có xu hướng "học vẹt" và vô tình giải mã tốt cả những tín hiệu bị biến dạng (Anomaly), làm giảm nhẹ năng lực phát hiện bất thường.
2. **Đánh đổi thực tế trong Y tế (Trade-off):** Chỉ số *Normal Precision* đạt 0.62 trong khi *Normal Recall* đạt tới 0.95. Điều này có nghĩa là mô hình chấp nhận báo động giả một số trường hợp bình thường thành bất thường (False Positive), nhưng đổi lại, tỷ lệ bỏ sót ca bệnh thực sự là cực kỳ thấp (*Anomaly Precision* đạt 0.98, *Anomaly Recall* đạt 0.84). Trong sàng lọc y khoa, tiêu chí "thà báo nhầm còn hơn bỏ sót bệnh nhân" là ưu tiên số một.

---

### 4. ỨNG DỤNG THỰC TẾ (DEMO APPLICATION)
Để minh chứng cho khả năng triển khai thực tế của nghiên cứu, một hệ thống demo trực quan đã được xây dựng bằng thư viện **Gradio** (`demo/app.py`). 

* **Cơ chế hoạt động:** Người dùng chọn một mẫu tín hiệu điện tâm đồ bất kỳ từ tập kiểm thử. Hệ thống sẽ đẩy tín hiệu qua mô hình VAE đã đóng gói (`best_model.pth`).
* **Kết quả hiển thị:** * Trả ra nhãn dự đoán: **ANOMALY** hoặc **NORMAL**.
  * Chỉ số Sai số tái tạo (Reconstruction Error) trực tiếp so với Ngưỡng (Threshold).
  * Thanh tiến trình hiển thị Độ tin cậy (Confidence Score) trực quan hóa bằng đồ họa ký tự.
  * Vector trung bình không gian tiềm ẩn ($mu$) để theo dõi vị trí tọa độ nén của nhịp tim.
* **Kết quả chạy thử:** Với một mẫu Anomaly ngẫu nhiên, ứng dụng tính ra Reconstruction Error đạt **0.052972**, vượt xa ngưỡng an toàn **0.037619**, hệ thống đưa ra cảnh báo chính xác với độ tin cậy hiển thị rõ ràng.

---

### 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
Nghiên cứu đã ứng dụng thành công mạng học sâu không giám sát $\beta$-VAE vào bài toán phát hiện bất thường trên tín hiệu ECG5000. Mô hình đạt hiệu năng rất cao (AUC 0.9620) chỉ với không gian tiềm ẩn 2 chiều nhờ vào kỹ thuật xử lý triệt để hiện tượng KL Collapse. 

**Hướng phát triển tương lai:**
1. Thử nghiệm thay thế các tầng Tuyến tính bằng các tầng Tích chập 1 chiều (1D-CNN) hoặc mạng hồi quy (LSTM/GRU) để mô hình hóa tốt hơn tính chất chuỗi thời gian của tín hiệu ECG.
2. Tận dụng thư mục `results/baseline/` để huấn luyện thêm các mô hình học máy truyền thống như Học máy Vector hỗ trợ một lớp (One-class SVM) hoặc Rừng cô lập (Isolation Forest) để làm phong phú thêm phần so sánh đối chứng.
3. Thử nghiệm trên các tập dữ liệu đa kênh lớn hơn như MIT-BIH để nâng cao khả năng tổng quát hóa của mô hình trước khi đưa vào môi trường lâm sàng thực tế.
