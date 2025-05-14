import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),  # 42x42
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),           # 21x21
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=latent_dim, kernel_size=3, stride=1, padding=1)    # 21x21
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=64, kernel_size=4, stride=2, padding=1),  # 42x42
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),          # 84x84
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        # Reshape input
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat = x_perm.view(-1, self.embedding_dim) # Flatten to (BHW, C)

        # Distance computation
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.T
            + self.embeddings.weight.pow(2).sum(1)
        )

        # Vector quantization
        indices = torch.argmin(dist, dim=1)
        quantized = self.embeddings(indices).view(x_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]

        e_loss = F.mse_loss(quantized.detach(), x, reduction='sum') # e_loss = ∥ z_e(x) − sg[e] ∥^2 
        q_loss = F.mse_loss(quantized, x.detach(), reduction='sum') # q_loss = ∥ sg[z_e(x)] − e ∥^2

        loss = q_loss + self.commitment_cost * e_loss # q_loss + β * e_loss
        
        quantized = x + (quantized - x).detach() # straight-through estimator

        # quantized output z_q(x), quantization loss, indicies of chosen codebook vectors
        return quantized, loss, indices.view(x.shape[0], x.shape[2], x.shape[3]) # before reshaping

class EMAQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

    def reinitialize_unused_codes(self, min_usage=5):
        unused_codes = (self.ema_cluster_size < min_usage).nonzero(as_tuple=True)[0]
        if unused_codes.numel() > 0:
            rand_vecs = torch.randn(unused_codes.numel(), self.embedding_dim).to(self.embeddings.device)
            self.embeddings.data[unused_codes] = rand_vecs
            self.ema_w[unused_codes] = rand_vecs
            self.ema_cluster_size[unused_codes] = 1.0  # reset to small value to avoid divide-by-zero
            print(f"[INFO] Reinitialized {unused_codes.numel()} unused codes.")

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        flat = x_perm.view(-1, self.embedding_dim)  # [BHW, C]

        # Compute distances
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embeddings.t()
            + self.embeddings.pow(2).sum(1)
        )

        # Nearest embeddings
        indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).type(flat.dtype)

        # Quantize
        quantized = encodings @ self.embeddings  # [BHW, C]
        quantized = quantized.view(x_perm.shape).permute(0, 3, 1, 2).contiguous()

        # EMA update
        if self.training:
            encodings_sum = encodings.sum(0)
            dw = encodings.t() @ flat

            self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Normalize embeddings
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embeddings.data = self.ema_w / cluster_size.unsqueeze(1)

        # Loss
        e_loss = F.mse_loss(quantized.detach(), x, reduction='sum')
        loss = self.commitment_cost * e_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss, indices.view(x.shape[0], x.shape[2], x.shape[3])

class VQVAE(nn.Module):
    def __init__(self, latent_dim=32, num_embeddings=128, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        # self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.quantizer = EMAQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss