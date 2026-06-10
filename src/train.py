import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ECG_VAE, vae_loss

# ── Config ────────────────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
LATENT_DIM  = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load dữ liệu ──────────────────────────────────────────────
train_np = np.load("data/processed/train_data.npy")
train_tensor = torch.FloatTensor(train_np).to(DEVICE)
loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

# ── Khởi tạo model & optimizer ────────────────────────────────
model = ECG_VAE(input_dim=140, hidden_dim=64, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Device: {DEVICE} | Train samples: {len(train_np)} | Batches/epoch: {len(loader)}")
print("-" * 55)

# ── Training loop ─────────────────────────────────────────────
best_loss = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total, recon_sum, kl_sum = 0, 0, 0

    # KL warmup: tăng dần beta từ 0 → 1 trong 20 epoch đầu
    beta = min(1.0, epoch / 20)

    for (batch,) in loader:
        optimizer.zero_grad()
        x_recon, mu, logvar = model(batch)
        loss, r, kl = vae_loss(x_recon, batch, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total     += loss.item()
        recon_sum += r.item()
        kl_sum    += kl.item()

    avg = total / len(loader)
    history.append(avg)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Beta: {beta:.2f} | Loss: {avg:.4f} "
              f"(Recon: {recon_sum/len(loader):.4f} | KL: {kl_sum/len(loader):.4f})")

    if avg < best_loss:
        best_loss = avg
        os.makedirs("results/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "results/checkpoints/best_model.pth")

print("-" * 55)
print(f"Training xong! Best loss: {best_loss:.4f}")

os.makedirs("results/logs", exist_ok=True)
np.save("results/logs/loss_history.npy", np.array(history))
print("Đã lưu loss history tại results/logs/loss_history.npy")