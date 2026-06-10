import torch
import torch.nn as nn

class ECG_VAE(nn.Module):
    """
    Variational Autoencoder cho bài toán Anomaly Detection trên tín hiệu ECG.
    
    Kiến trúc:
        Encoder: 140 → 64 → 32 → (mu, logvar) với dim=16
        Decoder: 16 → 32 → 64 → 140
    
    Args:
        input_dim  : Số chiều input (ECG5000 = 140 time steps)
        hidden_dim : Số neuron tầng ẩn
        latent_dim : Số chiều không gian tiềm ẩn (latent space)
    """
    def __init__(self, input_dim=140, hidden_dim=64, latent_dim=16):
        super(ECG_VAE, self).__init__()

        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   # ECG đã chuẩn hóa [0,1] bằng MinMaxScaler
        )

    def encode(self, x):
        """Trích xuất mu và logvar từ input."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + epsilon * std
        Cho phép backprop qua bước lấy mẫu ngẫu nhiên.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # Khi inference: dùng trực tiếp mu (deterministic)

    def decode(self, z):
        """Tái tạo tín hiệu từ latent vector z."""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def reconstruction_error(self, x):
        """
        Tính reconstruction error cho từng mẫu.
        Dùng khi inference để phát hiện Anomaly.
        Returns: error tensor shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            # MSE per sample (mean qua 140 time steps)
            error = torch.mean((x - x_recon) ** 2, dim=1)
        return error


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Loss function của VAE = Reconstruction Loss + β * KL Divergence.
    
    Args:
        x_recon : Output của decoder
        x       : Input gốc
        mu      : Mean của latent distribution
        logvar  : Log variance của latent distribution
        beta    : Trọng số KL (beta=1 là VAE chuẩn, beta>1 là β-VAE)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss: MSE (phù hợp vì dữ liệu liên tục, chuẩn hóa [0,1])
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

    # KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Smoke test — kiểm tra model hoạt động đúng
    model = ECG_VAE(input_dim=140, hidden_dim=64, latent_dim=16)
    print(model)

    dummy = torch.randn(32, 140)    # Batch 32 mẫu ECG
    x_recon, mu, logvar = model(dummy)

    loss, r, kl = vae_loss(x_recon, dummy, mu, logvar)
    print(f"\nForward pass OK:")
    print(f"  Input shape     : {dummy.shape}")
    print(f"  Recon shape     : {x_recon.shape}")
    print(f"  Latent mu shape : {mu.shape}")
    print(f"  Total loss      : {loss.item():.4f}")
    print(f"  Recon loss      : {r.item():.4f}  |  KL loss: {kl.item():.4f}")