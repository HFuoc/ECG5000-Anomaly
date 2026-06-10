# CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

Nghiên cứu về phát hiện bất thường trên dữ liệu điện tâm đồ (ECG) đã thu hút nhiều sự quan tâm từ cộng đồng học máy và y tế. Việc phân tích tự động giúp giảm tải công việc cho bác sĩ và cải thiện độ chính xác trong chẩn đoán. Các phương pháp có thể được chia thành các nhóm chính như sau:

## 1. Phát hiện bất thường ECG (ECG Anomaly Detection)
Trước khi học sâu phổ biến, các kỹ thuật truyền thống như One-class SVM, Isolation Forest và PCA thường được sử dụng. Các kỹ thuật này đòi hỏi việc trích xuất đặc trưng thủ công (hand-crafted features) từ tín hiệu (như khoảng R-R, biên độ các sóng P-Q-R-S-T). Gần đây, các kỹ thuật học sâu sử dụng Mạng nơ-ron tích chập (1D-CNN) và Mạng bộ nhớ dài-ngắn (LSTM) đã được áp dụng để tự động hóa quá trình trích xuất đặc trưng. Các mô hình này có khả năng mô hình hóa sự phụ thuộc không gian và thời gian trong tín hiệu ECG, đem lại hiệu suất vượt trội so với các phương pháp truyền thống.

## 2. Autoencoder trong Phát hiện bất thường (Autoencoder for Anomaly Detection)
Mạng Autoencoder (AE) là một phương pháp học không giám sát kinh điển cho bài toán phát hiện bất thường chuỗi thời gian. Nguyên lý cơ bản là huấn luyện mô hình học cách tái tạo (reconstruct) dữ liệu bình thường. Khi gặp tín hiệu bất thường, mạng AE sẽ tạo ra sai số tái tạo (Reconstruction Error) lớn do chưa từng thấy mẫu hình biến dạng này. Các nghiên cứu đã chỉ ra rằng AE rất nhạy cảm với các biến dạng nhịp tim. Tuy nhiên, AE truyền thống thường có không gian tiềm ẩn (latent space) rời rạc, làm giảm khả năng tổng quát hóa trên dữ liệu hoàn toàn mới và đôi khi có xu hướng ghi nhớ (overfit) cả dữ liệu lỗi.

## 3. Variational Autoencoder cho ECG (VAE for ECG)
Nhằm khắc phục nhược điểm của AE, Variational Autoencoder (VAE) ánh xạ dữ liệu thành một phân phối xác suất liên tục trong không gian tiềm ẩn (thường là phân phối chuẩn Gauss). Trong bối cảnh xử lý tín hiệu ECG, VAE cho phép mô hình hóa độ không chắc chắn (uncertainty) và cung cấp các tính chất nội suy mượt mà hơn. Tuy nhiên, VAE thường gặp phải hiện tượng sụp đổ không gian tiềm ẩn (KL Collapse) ở các chu kỳ huấn luyện đầu, khi mà Decoder bỏ qua các đặc trưng từ Encoder. Việc sử dụng các kỹ thuật như $\beta$-Warmup (được đề xuất và áp dụng trong bản nháp *Draft v1* của dự án với $\beta$ tăng dần từ 0 đến 1 trong 20 epochs) là giải pháp tối ưu để ép mô hình tận dụng hiệu quả không gian tiềm ẩn, đặc biệt là khi nén xuống 2 chiều ($latent\_dim=2$).

## 4. Benchmark trên tập dữ liệu ECG5000
Tập dữ liệu ECG5000 (một phần của UCR Time Series Archive) là một trong những bộ dữ liệu chuẩn mực nhất để đánh giá các mô hình học máy trên tín hiệu ECG. Dữ liệu này chứa 5000 mẫu nhịp tim, trong đó mỗi mẫu được cắt đồng nhất với độ dài 140 bước thời gian (timesteps). Các nghiên cứu benchmark thường báo cáo các chỉ số về độ chính xác (Accuracy), độ nhạy (Recall), và AUC-ROC. Mặc dù các mô hình học có giám sát đạt hiệu năng cực cao trên ECG5000, bài toán học không giám sát (như trong nghiên cứu của chúng tôi với AUC-ROC đạt **0.9620**) luôn là một điểm nhấn mang tính ứng dụng thực tế cao hơn do giải quyết được sự khan hiếm của dữ liệu nhãn bệnh lý.

---

## BẢNG SO SÁNH CÁC PHƯƠNG PHÁP (COMPARISON TABLE)

Dưới đây là bảng so sánh tổng quan giữa các hướng tiếp cận chính trong phân tích và phát hiện bất thường ECG:

| Nhóm Phương pháp | Phương pháp tiêu biểu | Ưu điểm (Advantages) | Hạn chế (Limitations) |
|---|---|---|---|
| **Truyền thống (Traditional ML)** | One-Class SVM, Isolation Forest | Nhanh, dễ diễn giải, ít đòi hỏi tài nguyên tính toán | Cần trích xuất đặc trưng thủ công, khó nắm bắt phụ thuộc thời gian phức tạp |
| **Học có giám sát (Supervised)** | 1D-CNN, LSTM | Độ chính xác rất cao trên dữ liệu đã biết, dễ tối ưu | Cần dữ liệu gán nhãn lớn, thiên lệch khi mất cân bằng dữ liệu, kém hiệu quả với các ca bệnh hiếm |
| **Autoencoder Truyền thống (AE)** | Dense AE, Conv-AE | Học không giám sát, cấu trúc đơn giản, phát hiện bất thường trực tiếp qua độ lệch tái tạo | Không gian tiềm ẩn rời rạc, có thể "học vẹt" và khôi phục tốt cả dữ liệu bất thường |
| **Mạng VAE (Đề xuất / Proposed)** | $\beta$-VAE với Beta-Warmup | Không gian tiềm ẩn liên tục, tránh KL Collapse, đánh giá được độ không chắc chắn (Uncertainty) | Yêu cầu điều chỉnh siêu tham số phức tạp (như giá trị $\beta$, số chiều $latent\_dim$) |
