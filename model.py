
# Current implementation is straight up copy paste from asking chatGPT what an VQ-VAE is

import torch 
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 42x42
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),           # 21x21
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 3, 1, 1)    # 21x21
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1),  # 42x42
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),          # 84x84
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, 1, 1),
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
        # Flatten to (BHW, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat = x_perm.view(-1, self.embedding_dim)

        # Compute distances
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.T
            + self.embeddings.weight.pow(2).sum(1)
        )
        indices = torch.argmin(dist, dim=1)
        quantized = self.embeddings(indices).view(x_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]

        # Losses
        e_loss = F.mse_loss(quantized.detach(), x)
        q_loss = F.mse_loss(quantized, x.detach())
        loss = q_loss + self.commitment_cost * e_loss

        quantized = x + (quantized - x).detach()  # straight-through estimator
        return quantized, loss, indices

class VQVAE(nn.Module):
    def __init__(self, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss