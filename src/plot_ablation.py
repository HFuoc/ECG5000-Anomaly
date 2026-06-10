import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đảm bảo thư mục lưu ảnh tồn tại
os.makedirs('results/figures', exist_ok=True)

# Dữ liệu từ thực nghiệm của bạn
latent_dims = ['Dim = 2', 'Dim = 4', 'Dim = 16']
aucs = [0.9620, 0.9618, 0.9611]

# Thiết lập style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Vẽ biểu đồ đường
sns.lineplot(x=latent_dims, y=aucs, marker='o', color='royalblue', linewidth=2.5, markersize=10)

# Thêm nhãn số liệu trên từng điểm
for i, v in enumerate(aucs):
    plt.text(i, v + 0.00015, f"{v:.4f}", ha='center', color='black', fontweight='bold', fontsize=11)

# Trang trí biểu đồ
plt.title("Ablation Study: Ảnh hưởng của Latent Dimension đến hiệu suất (AUC-ROC)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Kích thước không gian tiềm ẩn (Latent Dimension)", fontsize=11)
plt.ylabel("AUC-ROC Score", fontsize=11)
plt.ylim(0.90, 1.00) # Truc Y rong hon de tranh phong dai khac biet AUC rat nho

# Lưu và hiển thị
save_path = 'results/figures/ablation_latent_dim.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Đã lưu biểu đồ cực nét tại: {save_path}")
plt.show()
